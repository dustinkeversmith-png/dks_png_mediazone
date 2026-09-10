// Structural / syntactic pattern recognition over an Attributed Relational
// Graph, end to end.
//
// The architecture separates a slow, richly-supervised ingest pass from a fast,
// fully deterministic runtime pass:
//
//   INGEST (offline)                        RUNTIME (per frame)
//   ----------------                        -------------------
//   annotated instance masks                deterministic segmentation
//        -> vectorized contours                   -> vectorized contours
//        -> invariant descriptors                 -> invariant descriptors
//        -> glyph vocabulary  ------------->  top-k retrieval from the index
//        -> part-whole relations ---------->  ARG constraint validation
//        [compiled knowledge base]                -> explained classification
//
// Ingest stands in for the neural foundation stage. Rather than run SAM or
// DINOv2 here, it consumes ADE20K's human-annotated semantic index maps, which
// supply exactly what that stage would: instance masks with class identity.
// The grounding is therefore real supervision, and the runtime path never sees
// an annotation -- it must recover structure from pixels alone.
//
// Runtime contains no learned weights, no floating-point reductions whose order
// depends on threading, and no RNG that is not explicitly seeded. Running it
// twice on the same input is required to produce a bit-identical result, and
// that requirement is asserted rather than assumed.

#include "test_harness.hpp"
#include "math/contour_metrics.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"
#include "contour/rdp/rdp.hpp"
#include "descriptors/shape_invariants/shape_invariants.hpp"
#include "descriptors/texture_hash/texture_hash.hpp"
#include "segmentation/helpers/slic_rag/slic_rag.hpp"
#include "structures/glyph_index/glyph_index.hpp"
#include "structures/relational_dag/relational_dag.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pipeline {

using math::ImageBuffer;
using math::Vec2;

// ---------------------------------------------------------------------------
// Semantic vocabulary (ADE20K SceneParse150 index -> name)
// ---------------------------------------------------------------------------

inline const char* class_name(int index) {
    static const char* kNames[] = {
        "unlabeled", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
        "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table",
        "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf",
        "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock",
        "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
        "signboard", "chest", "counter", "sand", "sink", "skyscraper", "fireplace",
        "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pooltable",
        "pillow", "screendoor", "stairway", "river", "bridge", "bookcase", "blind", "coffee",
        "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm",
        "kitchenisland", "computer", "swivelchair", "boat", "bar", "arcade", "hovel", "bus",
        "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
        "television", "airplane", "dirt", "apparel", "pole", "land", "bannister",
        "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
        "fountain", "conveyer", "canopy", "washer", "plaything", "pool", "stool", "barrel",
        "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
        "step", "tank", "brand", "microwave", "pot", "animal", "bicycle", "lake",
        "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
        "trafficlight", "tray", "ashcan", "fan", "pier", "screen2", "plate", "monitor",
        "bulletinboard", "shower", "radiator", "glass", "clock", "flag"};
    constexpr int kCount = static_cast<int>(sizeof(kNames) / sizeof(kNames[0]));
    return index >= 0 && index < kCount ? kNames[index] : "class?";
}

// ---------------------------------------------------------------------------
// Primitives and scenes
// ---------------------------------------------------------------------------

// One vectorized, described region. This is the unit both phases operate on:
// ingest builds them from annotations, runtime from segmentation output, and
// everything downstream is identical in both cases.
struct Primitive {
    int id = 0;
    int semantic_class = -1;
    int area = 0;
    float area_fraction = 0.0f;
    int perimeter = 0;
    Vec2 centroid;
    float radius = 0.0f;  // equivalent-disc radius, used to normalize offsets
    math::Rect bbox;
    std::vector<Vec2> contour;
    std::vector<Vec2> polygon;
    descriptors::RegionMoments moments;
    descriptors::ShapeDescriptor shape;
    descriptors::TextureSignature texture;
    std::map<int, int> border_length;
    std::map<int, float> border_contrast_sum;
    int parent = -1;
    float enclosure = 0.0f;
    int glyph = -1;
};

struct Scene {
    std::string stem;
    int width = 0;
    int height = 0;
    std::vector<int> labels;  // primitive id per pixel, -1 where unassigned
    std::vector<Primitive> primitives;
    ImageBuffer rgb;
    ImageBuffer luma;
};

// Ramer-Douglas-Peucker at an epsilon proportional to the region's own scale.
// A fixed pixel epsilon would simplify a small region into a triangle while
// leaving a large one fully detailed, and the descriptors of the two would
// then differ for reasons that have nothing to do with their shapes.
inline std::vector<Vec2> simplify_scaled(const std::vector<Vec2>& contour, int area,
                                         float relative_epsilon) {
    if (contour.size() < 4) {
        return contour;
    }
    const float epsilon = std::max(0.6f, relative_epsilon * std::sqrt(static_cast<float>(area)));
    std::vector<Vec2> simplified = vision::RamerDouglasPeucker::simplify(contour, epsilon);
    if (simplified.size() >= 3) {
        return simplified;
    }
    return contour;
}

struct SceneOptions {
    float min_area_fraction = 0.003f;
    float relative_epsilon = 0.035f;
    // A neighbour must claim most of the outline *and* be substantially the
    // larger of the two before the pair counts as part-whole rather than as
    // two things that merely happen to touch.
    float enclosure_threshold = 0.50f;
    float parent_area_ratio = 2.0f;
};

// The joint retrieval vector: silhouette, surface and colour in one metric
// space. Shape alone cannot separate a chair from a car, and material alone
// cannot separate a door from the wall it is set into; the exemplar index
// needs both. Block weights set the exchange rate between them.
inline std::vector<float> joint_descriptor(const Primitive& primitive) {
    // Shape is deliberately the lightest block despite having the most
    // dimensions. A region's outline depends on where the segmenter happened
    // to cut, whereas its material and colour are properties of the surface
    // itself and survive that arbitrariness; weighting by dimension count
    // alone would let the least reliable evidence dominate the metric.
    constexpr float kShapeWeight = 0.25f;
    constexpr float kTextureWeight = 1.00f;
    constexpr float kColorWeight = 2.20f;
    constexpr float kScaleWeight = 1.00f;

    std::vector<float> vector;
    vector.reserve(descriptors::ShapeDescriptor::kDims + 18);
    for (float value : primitive.shape.flatten()) {
        vector.push_back(kShapeWeight * value);
    }
    const descriptors::TextureSignature& texture = primitive.texture;
    for (uint8_t bin : texture.lbp) {
        vector.push_back(kTextureWeight * static_cast<float>(bin) / 255.0f);
    }
    vector.push_back(kTextureWeight * texture.roughness);
    vector.push_back(kTextureWeight * texture.energy);
    vector.push_back(kTextureWeight * texture.homogeneity);
    vector.push_back(kTextureWeight * texture.anisotropy);
    vector.push_back(kTextureWeight * std::min(1.0f, texture.color_roughness / 30.0f));
    vector.push_back(kColorWeight * texture.mean_lab[0] / 100.0f);
    vector.push_back(kColorWeight * texture.mean_lab[1] / 100.0f);
    vector.push_back(kColorWeight * texture.mean_lab[2] / 100.0f);
    // Apparent size, square-rooted so it behaves like a length rather than an
    // area. A wall and a picture on it can share an outline but not a scale.
    vector.push_back(kScaleWeight * std::sqrt(primitive.area_fraction));
    return vector;
}

// Builds the full primitive set for a label map: contours, descriptors,
// adjacency with border contrast, and the part-whole forest.
inline Scene build_scene(const std::string& stem, const ImageBuffer& rgb, const ImageBuffer& luma,
                         const std::vector<int>& region_labels, int region_count,
                         const std::vector<int>& region_class, const SceneOptions& options) {
    Scene scene;
    scene.stem = stem;
    scene.width = rgb.width;
    scene.height = rgb.height;
    scene.rgb = rgb;
    scene.luma = luma;

    const int w = rgb.width;
    const int h = rgb.height;
    const int n = w * h;
    const int min_area = std::max(24, static_cast<int>(options.min_area_fraction * n));

    std::vector<int> area(static_cast<size_t>(region_count), 0);
    for (int label : region_labels) {
        if (label >= 0) {
            ++area[static_cast<size_t>(label)];
        }
    }
    // Regions below the minimum stay in the image but not in the graph: they
    // are too small for a Fourier series or a texture census to say anything
    // trustworthy about.
    std::vector<int> remap(static_cast<size_t>(region_count), -1);
    int kept = 0;
    for (int r = 0; r < region_count; ++r) {
        if (area[static_cast<size_t>(r)] >= min_area) {
            remap[static_cast<size_t>(r)] = kept++;
        }
    }
    scene.labels.assign(static_cast<size_t>(n), -1);
    for (int p = 0; p < n; ++p) {
        const int label = region_labels[static_cast<size_t>(p)];
        scene.labels[static_cast<size_t>(p)] =
            label >= 0 ? remap[static_cast<size_t>(label)] : -1;
    }
    if (kept == 0) {
        return scene;
    }

    scene.primitives.resize(static_cast<size_t>(kept));
    for (int r = 0; r < region_count; ++r) {
        if (remap[static_cast<size_t>(r)] < 0) {
            continue;
        }
        Primitive& primitive = scene.primitives[static_cast<size_t>(remap[static_cast<size_t>(r)])];
        primitive.id = remap[static_cast<size_t>(r)];
        primitive.semantic_class =
            r < static_cast<int>(region_class.size()) ? region_class[static_cast<size_t>(r)] : -1;
    }

    // Adjacency and perimeter in one walk over the 4-connected pixel pairs.
    // The Lab step across each pair is the boundary contrast the ARG later
    // uses to tell a printed marking from a physical edge.
    auto lab_step = [&rgb](int ax, int ay, int bx, int by) {
        const contour::Lab p = contour::LabColor::at(rgb, ax, ay);
        const contour::Lab q = contour::LabColor::at(rgb, bx, by);
        const float dl = p.L - q.L;
        const float da = p.a - q.a;
        const float db = p.b - q.b;
        return std::sqrt(dl * dl + da * da + db * db);
    };
    auto touch = [&scene](int a, int b, float contrast) {
        if (a < 0) {
            return;
        }
        Primitive& primitive = scene.primitives[static_cast<size_t>(a)];
        ++primitive.perimeter;
        if (b < 0) {
            return;
        }
        ++primitive.border_length[b];
        primitive.border_contrast_sum[b] += contrast;
    };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int p = y * w + x;
            const int a = scene.labels[static_cast<size_t>(p)];
            if (x + 1 < w) {
                const int b = scene.labels[static_cast<size_t>(p + 1)];
                if (a != b) {
                    const float contrast = lab_step(x, y, x + 1, y);
                    touch(a, b, contrast);
                    touch(b, a, contrast);
                }
            }
            if (y + 1 < h) {
                const int b = scene.labels[static_cast<size_t>(p + w)];
                if (a != b) {
                    const float contrast = lab_step(x, y, x, y + 1);
                    touch(a, b, contrast);
                    touch(b, a, contrast);
                }
            }
            // The image frame bounds a region just as a neighbour does.
            if (a >= 0) {
                Primitive& primitive = scene.primitives[static_cast<size_t>(a)];
                primitive.perimeter += static_cast<int>(x == 0) + static_cast<int>(x == w - 1) +
                                       static_cast<int>(y == 0) + static_cast<int>(y == h - 1);
            }
        }
    }

    ImageBuffer mask = math::make_gray(w, h, 0);
    for (Primitive& primitive : scene.primitives) {
        const descriptors::RegionView view =
            descriptors::make_region_view(scene.labels, w, h, primitive.id);
        primitive.area = view.area;
        primitive.area_fraction = static_cast<float>(view.area) / static_cast<float>(n);
        primitive.bbox = {static_cast<float>(view.min_x), static_cast<float>(view.min_y),
                          static_cast<float>(view.max_x - view.min_x + 1),
                          static_cast<float>(view.max_y - view.min_y + 1)};

        std::fill(mask.data.begin(), mask.data.end(), static_cast<uint8_t>(0));
        for (int p = 0; p < n; ++p) {
            if (scene.labels[static_cast<size_t>(p)] == primitive.id) {
                mask.data[static_cast<size_t>(p)] = 255;
            }
        }
        const vision::MooreNeighborTracer::Contour traced =
            vision::MooreNeighborTracer::trace(mask, 128);
        primitive.contour = traced.points;
        primitive.polygon =
            simplify_scaled(primitive.contour, primitive.area, options.relative_epsilon);

        primitive.moments =
            descriptors::HuMoments::accumulate(scene.labels, w, h, primitive.id);
        primitive.centroid = {static_cast<float>(primitive.moments.cx),
                              static_cast<float>(primitive.moments.cy)};
        primitive.radius = std::sqrt(static_cast<float>(primitive.area) / math::kPi);
        primitive.shape = descriptors::describe_shape(primitive.polygon, primitive.moments,
                                                      static_cast<int>(primitive.polygon.size()));
        primitive.texture = descriptors::analyze_texture(rgb, luma, view);
    }

    // Part-whole resolution. A region is a part of whichever neighbour claims
    // most of its outline, provided that neighbour is strictly larger and the
    // claim is decisive; anything less is mere adjacency, not containment.
    for (Primitive& primitive : scene.primitives) {
        int best = -1;
        int best_border = 0;
        for (const auto& entry : primitive.border_length) {
            if (entry.second > best_border) {
                best_border = entry.second;
                best = entry.first;
            }
        }
        if (best < 0 || primitive.perimeter <= 0) {
            continue;
        }
        const float enclosure =
            static_cast<float>(best_border) / static_cast<float>(primitive.perimeter);
        if (enclosure < options.enclosure_threshold ||
            scene.primitives[static_cast<size_t>(best)].area <
                options.parent_area_ratio * primitive.area) {
            continue;
        }
        primitive.parent = best;
        primitive.enclosure = enclosure;
    }
    return scene;
}

// The measured attributes of one parent-child pair, in the exact form the ARG
// stores and validates.
inline structures::RelationObservation observe_relation(const Scene& scene,
                                                        const Primitive& child) {
    structures::RelationObservation observation;
    if (child.parent < 0) {
        return observation;
    }
    const Primitive& parent = scene.primitives[static_cast<size_t>(child.parent)];
    observation.area_ratio =
        parent.area > 0 ? static_cast<float>(child.area) / static_cast<float>(parent.area) : 0.0f;
    observation.delta_e = descriptors::lab_delta_e(child.texture.mean_lab, parent.texture.mean_lab);
    observation.material_distance = descriptors::material_distance(child.texture, parent.texture);
    const float dx = child.centroid.x - parent.centroid.x;
    const float dy = child.centroid.y - parent.centroid.y;
    observation.radial_offset =
        parent.radius > 0.0f ? std::sqrt(dx * dx + dy * dy) / parent.radius : 0.0f;
    const auto it = child.border_contrast_sum.find(child.parent);
    const auto length = child.border_length.find(child.parent);
    if (it != child.border_contrast_sum.end() && length != child.border_length.end() &&
        length->second > 0) {
        observation.boundary_contrast = it->second / static_cast<float>(length->second);
    }
    observation.enclosure = child.enclosure;
    return observation;
}

// ---------------------------------------------------------------------------
// Label sources
// ---------------------------------------------------------------------------

// A dense region map, with a semantic class per region where one is known.
struct LabelField {
    std::vector<int> labels;
    std::vector<int> region_class;
    int count = 0;
};

// Runtime: the deterministic segmentation. No annotation is consulted.
inline LabelField regions_from_pixels(const ImageBuffer& rgb, int superpixels, int target_regions) {
    segmentation::slic_rag::Slic slic;
    slic.iterations = 10;
    slic.compactness = 12.0f;
    segmentation::slic_rag::RagHierarchicalMerger merger;
    merger.target_regions = target_regions;
    merger.max_cost = 0.50f;
    merger.min_region_fraction = 0.005f;

    const segmentation::slic_rag::SlicResult superpixel_map = slic.segment(rgb, superpixels);
    const segmentation::slic_rag::RagResult merged = merger.merge(superpixel_map, rgb);

    LabelField field;
    field.labels = merged.labels;
    field.count = merged.final_regions;
    field.region_class.assign(static_cast<size_t>(field.count), -1);
    return field;
}

// ---------------------------------------------------------------------------
// Compiled knowledge base
// ---------------------------------------------------------------------------

struct KnowledgeBase {
    // Shape archetypes. These are the ARG's node alphabet and what makes a
    // decision speakable ("a compact-quad inside an elongated-ragged").
    structures::GlyphVocabulary vocabulary;
    // Every ingested primitive, indexed on the joint shape+material vector.
    // Clustering discards precisely the detail that distinguishes a door from
    // a window, so retrieval keeps the exemplars themselves.
    structures::VectorIndex exemplars;
    std::vector<int> exemplar_class;
    structures::RelationalDAG relations;
    int ingest_scenes = 0;
    int ingest_primitives = 0;
    int ingest_relations = 0;
};

struct Classification {
    int primitive = 0;
    int glyph = -1;
    float retrieval_distance = 0.0f;
    int shape_only_class = -1;  // baseline: what the shape archetype alone says
    int knn_class = -1;         // exemplar retrieval on shape + material
    float knn_confidence = 0.0f;  // winning share of the retrieval vote
    int predicted_class = -1;   // after ARG arbitration
    bool dag_verified = false;
    bool dag_revised = false;
    float dag_score = 0.0f;
    int truth_class = -1;
    bool candidate_hit = false;
    std::string rationale;
};

// Two-stage resolution, exactly as the architecture prescribes: retrieval is
// local and shape-only, then the relational graph arbitrates between the
// candidates it returned. Retrieval alone cannot tell a button from a coin;
// only the context can.
inline int argmax(const std::map<int, float>& votes) {
    int best = -1;
    float best_weight = 0.0f;
    for (const auto& entry : votes) {
        if (entry.second > best_weight) {
            best_weight = entry.second;
            best = entry.first;
        }
    }
    return best;
}

// `truth_classes` is scoring-only and never influences a decision; it is
// carried through so the per-primitive report can be written in one pass.
inline std::vector<Classification> resolve(const Scene& scene, const KnowledgeBase& base,
                                           int top_k, int exemplar_k, float accept_threshold,
                                           float arg_gain, float parent_confidence,
                                           const std::vector<int>& truth_classes) {
    std::vector<Classification> results(scene.primitives.size());
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].truth_class = i < truth_classes.size() ? truth_classes[i] : -1;
    }

    // Stage 1: local candidate retrieval. Each region independently asks the
    // index what it resembles -- no context, no neighbours.
    std::vector<int> glyph_of(scene.primitives.size(), -1);
    std::vector<std::map<int, float>> votes(scene.primitives.size());
    std::vector<float> vote_mass(scene.primitives.size(), 0.0f);
    for (size_t i = 0; i < scene.primitives.size(); ++i) {
        const Primitive& primitive = scene.primitives[i];
        Classification& result = results[i];
        result.primitive = primitive.id;

        const auto archetypes = base.vocabulary.query(primitive.shape, 1);
        if (!archetypes.empty()) {
            glyph_of[i] = archetypes.front().id;
            result.glyph = archetypes.front().id;
            result.retrieval_distance = archetypes.front().distance;
            result.shape_only_class =
                base.vocabulary.glyphs()[static_cast<size_t>(result.glyph)].dominant_class;
        }

        // Nearer exemplars vote harder, which lets a single close match beat a
        // cluster of vaguely similar ones.
        for (const structures::VectorIndex::Neighbor& neighbor :
             base.exemplars.query(joint_descriptor(primitive), exemplar_k)) {
            const int semantic = base.exemplar_class[static_cast<size_t>(neighbor.id)];
            if (semantic <= 0) {
                continue;
            }
            const float weight = 1.0f / (1.0f + neighbor.distance);
            votes[i][semantic] += weight;
            vote_mass[i] += weight;
            if (semantic == result.truth_class) {
                result.candidate_hit = true;
            }
        }
        result.knn_class = argmax(votes[i]);
        result.predicted_class = result.knn_class;
        if (result.knn_class >= 0 && vote_mass[i] > 0.0f) {
            result.knn_confidence = votes[i][result.knn_class] / vote_mass[i];
        }
    }

    // Stage 2: structural parsing. Where a region is enclosed by another, the
    // ARG is asked whether that containment is one it has seen, and its answer
    // is folded into the same vote tally rather than replacing it -- context
    // should be able to break a tie, not overrule a confident match.
    for (size_t i = 0; i < scene.primitives.size(); ++i) {
        const Primitive& primitive = scene.primitives[i];
        Classification& result = results[i];
        if (primitive.parent < 0 || glyph_of[i] < 0 ||
            glyph_of[static_cast<size_t>(primitive.parent)] < 0) {
            result.rationale = "root region: no enclosing context to constrain against";
            continue;
        }
        // The whole is identified by what it *is* -- the class retrieval gave
        // it -- while the part is identified by its shape archetype. Keying
        // both ends on shape would ask "what is typically inside a broad
        // quad?", which has no answer; keying the parent semantically asks
        // "what is typically inside a building?", which does.
        const Classification& enclosing = results[static_cast<size_t>(primitive.parent)];
        const int parent_node = enclosing.knn_class;
        // Context is only worth consulting if the context itself is known.
        // A confidently-wrong parent turns the graph into a mechanism for
        // propagating one mistake into two.
        if (parent_node <= 0 || enclosing.knn_confidence < parent_confidence) {
            result.rationale = "enclosing region is itself too uncertain to reason from";
            continue;
        }
        const structures::RelationObservation observation = observe_relation(scene, primitive);
        const structures::ValidationResult validation =
            base.relations.validate(parent_node, glyph_of[i], observation);
        if (!validation.edge_known) {
            result.rationale = "enclosed, but this part-whole pairing was never observed";
            continue;
        }
        const float weighted = validation.score * validation.support_weight;
        if (weighted < accept_threshold || validation.predicted_child_class <= 0) {
            result.rationale = "enclosing relation attested but its constraints are not met";
            continue;
        }
        result.dag_verified = true;
        result.dag_score = weighted;

        // Re-rank every identity retrieval proposed by how often this exact
        // containment produced it. A candidate the relation has never yielded
        // keeps its retrieval score untouched rather than being penalized, so
        // context can only promote, never veto on absence of evidence.
        for (auto& vote : votes[i]) {
            vote.second *=
                1.0f + arg_gain * weighted *
                           base.relations.class_share(parent_node, glyph_of[i], vote.first);
        }
        // A class the relation strongly attests deserves consideration even if
        // retrieval missed it entirely.
        if (validation.class_purity > 0.5f) {
            votes[i][validation.predicted_child_class] +=
                std::max(vote_mass[i], 1e-3f) * arg_gain * weighted * validation.class_purity;
        }
        const int revised = argmax(votes[i]);
        result.dag_revised = revised != result.knn_class;
        result.predicted_class = revised;

        std::ostringstream why;
        why << base.vocabulary.glyphs()[static_cast<size_t>(glyph_of[i])].archetype
            << " enclosed by " << class_name(parent_node) << " -> "
            << class_name(validation.predicted_child_class) << "; " << validation.explain();
        result.rationale = why.str();
    }
    return results;
}

}  // namespace pipeline

namespace {

using math::Vec2;
using pipeline::Classification;
using pipeline::KnowledgeBase;
using pipeline::Primitive;
using pipeline::Scene;
using pipeline::SceneOptions;

struct Checks {
    int total = 0;
    int passed = 0;
    std::vector<std::string> failures;

    void expect(bool condition, const std::string& message) {
        ++total;
        if (condition) {
            ++passed;
        } else {
            failures.push_back(message);
        }
    }
};

// Order-independent digest of the runtime decision stream, used to assert that
// the pipeline really is reproducible rather than merely believed to be.
inline uint64_t digest(uint64_t seed, const std::string& text) {
    uint64_t hash = seed;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return hash;
}

inline std::string classification_record(const Classification& c) {
    std::ostringstream out;
    out << c.primitive << '|' << c.glyph << '|' << c.knn_class << '|' << c.dag_verified << '|'
        << std::fixed << std::setprecision(6) << c.dag_score << '|' << c.retrieval_distance << '|'
        << c.predicted_class;
    return out.str();
}

// ---------------------------------------------------------------------------
// Sample preparation
// ---------------------------------------------------------------------------

struct PreparedSample {
    std::string stem;
    math::ImageBuffer rgb;
    math::ImageBuffer luma;
    math::ImageBuffer annotation;  // ADE20K class indices, nearest-resampled
    bool usable = false;
};

inline PreparedSample prepare(const ProviderLoadedSample& provider, int max_side) {
    PreparedSample prepared;
    prepared.stem = stem_of(provider.row.file.empty() ? provider.sample.id : provider.row.file);
    if (provider.sample.rgb.empty() || provider.sample.rgb.channels < 3) {
        return prepared;
    }
    prepared.rgb = downscale_max_side(provider.sample.rgb, max_side);
    prepared.luma = vision::rgb_to_luma(prepared.rgb);
    if (provider.sample.mask.empty()) {
        return prepared;
    }
    // Nearest neighbour, never bilinear: the annotation stores class indices,
    // and interpolating them would invent classes that are not in the image.
    prepared.annotation =
        resize_nearest(provider.sample.mask, prepared.rgb.width, prepared.rgb.height);
    prepared.usable = prepared.annotation.width == prepared.rgb.width &&
                      prepared.annotation.height == prepared.rgb.height;
    return prepared;
}

// Majority annotation class inside each region, together with how cleanly the
// region sits inside that class. `min_purity` rejects regions that straddle a
// boundary: at ingest that keeps mislabelled exemplars out of the index, and
// at scoring time it means accuracy is measured only where the annotation
// actually commits to an answer.
inline std::vector<int> majority_classes(const Scene& scene, const math::ImageBuffer& annotation,
                                         float min_purity) {
    std::vector<std::map<int, int>> votes(scene.primitives.size());
    for (int y = 0; y < scene.height; ++y) {
        for (int x = 0; x < scene.width; ++x) {
            const int id = scene.labels[static_cast<size_t>(y * scene.width + x)];
            if (id < 0) {
                continue;
            }
            const int value = annotation.at(x, y);
            if (value > 0) {
                ++votes[static_cast<size_t>(id)][value];
            }
        }
    }
    std::vector<int> out(scene.primitives.size(), -1);
    for (size_t i = 0; i < votes.size(); ++i) {
        int best = -1;
        int best_count = 0;
        int total = 0;
        for (const auto& entry : votes[i]) {
            total += entry.second;
            if (entry.second > best_count) {
                best_count = entry.second;
                best = entry.first;
            }
        }
        const float purity =
            total > 0 ? static_cast<float>(best_count) / static_cast<float>(total) : 0.0f;
        out[i] = purity >= min_purity ? best : -1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Invariance harness
// ---------------------------------------------------------------------------
//
// The architecture's load-bearing claim is that a polygon matches its
// dictionary entry regardless of pose. That is testable in isolation: render
// known shapes at many rotations and scales, push them through the identical
// descriptor path, and check that the descriptor clusters by shape rather than
// by pose. If leave-one-out retrieval on this set is not near-perfect, nothing
// built on top of it can be trusted.

struct ShapeTemplate {
    const char* name;
    std::vector<Vec2> unit;
};

inline std::vector<ShapeTemplate> shape_templates() {
    auto regular = [](int sides, float phase) {
        std::vector<Vec2> points;
        for (int i = 0; i < sides; ++i) {
            const float t = phase + 2.0f * math::kPi * static_cast<float>(i) /
                                        static_cast<float>(sides);
            points.push_back({std::cos(t), std::sin(t)});
        }
        return points;
    };
    auto star = [](int points_count, float inner) {
        std::vector<Vec2> points;
        for (int i = 0; i < points_count * 2; ++i) {
            const float t = math::kPi * static_cast<float>(i) / static_cast<float>(points_count);
            const float r = (i % 2 == 0) ? 1.0f : inner;
            points.push_back({r * std::cos(t), r * std::sin(t)});
        }
        return points;
    };
    std::vector<ShapeTemplate> templates;
    templates.push_back({"square", regular(4, 0.25f * math::kPi)});
    templates.push_back({"disc", regular(48, 0.0f)});
    templates.push_back({"triangle", regular(3, 0.5f * math::kPi)});
    templates.push_back({"hexagon", regular(6, 0.0f)});
    std::vector<Vec2> ellipse = regular(48, 0.0f);
    for (Vec2& p : ellipse) {
        p.y *= 0.40f;
    }
    templates.push_back({"ellipse", ellipse});
    templates.push_back({"star5", star(5, 0.42f)});
    templates.push_back({"plus", {{-0.33f, -1.0f},
                                  {0.33f, -1.0f},
                                  {0.33f, -0.33f},
                                  {1.0f, -0.33f},
                                  {1.0f, 0.33f},
                                  {0.33f, 0.33f},
                                  {0.33f, 1.0f},
                                  {-0.33f, 1.0f},
                                  {-0.33f, 0.33f},
                                  {-1.0f, 0.33f},
                                  {-1.0f, -0.33f},
                                  {-0.33f, -0.33f}}});
    templates.push_back({"ell", {{-1.0f, -1.0f},
                                 {0.2f, -1.0f},
                                 {0.2f, -0.2f},
                                 {1.0f, -0.2f},
                                 {1.0f, 1.0f},
                                 {-1.0f, 1.0f}}});
    return templates;
}

inline bool point_in_polygon(const std::vector<Vec2>& polygon, float x, float y) {
    bool inside = false;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const Vec2& a = polygon[i];
        const Vec2& b = polygon[j];
        if ((a.y > y) != (b.y > y) &&
            x < (b.x - a.x) * (y - a.y) / (b.y - a.y + 1e-12f) + a.x) {
            inside = !inside;
        }
    }
    return inside;
}

// Renders one posed template and runs it through build_scene, so the measured
// invariance covers tracing and simplification too, not just the maths.
inline descriptors::ShapeDescriptor pose_descriptor(const ShapeTemplate& shape, float angle,
                                                    float scale, const SceneOptions& options) {
    constexpr int kCanvas = 200;
    const float radius = 0.34f * kCanvas * scale;
    const float cs = std::cos(angle);
    const float sn = std::sin(angle);
    std::vector<Vec2> posed;
    for (const Vec2& p : shape.unit) {
        posed.push_back({kCanvas * 0.5f + radius * (cs * p.x - sn * p.y),
                         kCanvas * 0.5f + radius * (sn * p.x + cs * p.y)});
    }
    std::vector<int> labels(static_cast<size_t>(kCanvas * kCanvas), -1);
    for (int y = 0; y < kCanvas; ++y) {
        for (int x = 0; x < kCanvas; ++x) {
            if (point_in_polygon(posed, static_cast<float>(x) + 0.5f,
                                 static_cast<float>(y) + 0.5f)) {
                labels[static_cast<size_t>(y * kCanvas + x)] = 0;
            }
        }
    }
    math::ImageBuffer rgb;
    rgb.width = kCanvas;
    rgb.height = kCanvas;
    rgb.channels = 3;
    rgb.data.assign(static_cast<size_t>(kCanvas * kCanvas * 3), 160);
    const math::ImageBuffer luma = math::make_gray(kCanvas, kCanvas, 160);
    const Scene scene =
        pipeline::build_scene("synthetic", rgb, luma, labels, 1, {-1}, options);
    return scene.primitives.empty() ? descriptors::ShapeDescriptor{}
                                    : scene.primitives.front().shape;
}

struct InvarianceReport {
    int poses = 0;
    int correct = 0;
    float intra_mean = 0.0f;
    float inter_mean = 0.0f;
    float separation = 0.0f;
    std::string table;
};

inline InvarianceReport measure_invariance(const SceneOptions& options) {
    const std::vector<ShapeTemplate> templates = shape_templates();
    const float angles[6] = {0.0f, 0.401f, 0.820f, 1.571f, 2.391f, 3.683f};
    const float scales[3] = {0.55f, 1.00f, 1.70f};

    std::vector<descriptors::ShapeDescriptor> descriptors_list;
    std::vector<int> shape_of;
    for (size_t s = 0; s < templates.size(); ++s) {
        for (float angle : angles) {
            for (float scale : scales) {
                descriptors::ShapeDescriptor descriptor =
                    pose_descriptor(templates[s], angle, scale, options);
                if (!descriptor.valid) {
                    continue;
                }
                descriptors_list.push_back(descriptor);
                shape_of.push_back(static_cast<int>(s));
            }
        }
    }

    InvarianceReport report;
    report.poses = static_cast<int>(descriptors_list.size());
    double intra_sum = 0.0;
    int intra_count = 0;
    double inter_sum = 0.0;
    int inter_count = 0;
    std::vector<int> per_shape_total(templates.size(), 0);
    std::vector<int> per_shape_correct(templates.size(), 0);
    std::vector<double> per_shape_intra(templates.size(), 0.0);
    std::vector<int> per_shape_intra_count(templates.size(), 0);

    for (size_t i = 0; i < descriptors_list.size(); ++i) {
        int nearest = -1;
        float nearest_distance = std::numeric_limits<float>::max();
        for (size_t j = 0; j < descriptors_list.size(); ++j) {
            if (i == j) {
                continue;
            }
            const float d =
                descriptors::ShapeDescriptor::distance(descriptors_list[i], descriptors_list[j]);
            if (shape_of[i] == shape_of[j]) {
                intra_sum += d;
                ++intra_count;
                per_shape_intra[static_cast<size_t>(shape_of[i])] += d;
                ++per_shape_intra_count[static_cast<size_t>(shape_of[i])];
            } else {
                inter_sum += d;
                ++inter_count;
            }
            if (d < nearest_distance) {
                nearest_distance = d;
                nearest = static_cast<int>(j);
            }
        }
        ++per_shape_total[static_cast<size_t>(shape_of[i])];
        if (nearest >= 0 && shape_of[static_cast<size_t>(nearest)] == shape_of[i]) {
            ++report.correct;
            ++per_shape_correct[static_cast<size_t>(shape_of[i])];
        }
    }
    report.intra_mean = intra_count > 0 ? static_cast<float>(intra_sum / intra_count) : 0.0f;
    report.inter_mean = inter_count > 0 ? static_cast<float>(inter_sum / inter_count) : 0.0f;
    report.separation = report.intra_mean > 1e-6f ? report.inter_mean / report.intra_mean : 0.0f;

    std::ostringstream table;
    table << "shape\tposes\tnn_correct\tintra_mean\n";
    for (size_t s = 0; s < templates.size(); ++s) {
        const double intra = per_shape_intra_count[s] > 0
                                 ? per_shape_intra[s] / per_shape_intra_count[s]
                                 : 0.0;
        table << templates[s].name << '\t' << per_shape_total[s] << '\t' << per_shape_correct[s]
              << '\t' << std::fixed << std::setprecision(4) << intra << '\n';
    }
    report.table = table.str();
    return report;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

class PipelineIntegration {
public:
    AtomDemoReport report{"structural_arg_pipeline"};
    Checks checks;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load annotated corpus");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr,
                                                  ingest_samples + runtime_samples, max_side);
        std::cout << "provider  : " << mission.provider_name << " ("
                  << mission.provider_samples.size() << " samples, max side " << max_side << ")\n";
        for (const ProviderLoadedSample& provider : mission.provider_samples) {
            PreparedSample prepared = prepare(provider, max_side);
            if (!prepared.usable) {
                continue;
            }
            if (static_cast<int>(train_.size()) < ingest_samples) {
                train_.push_back(std::move(prepared));
            } else if (static_cast<int>(test_.size()) < runtime_samples) {
                test_.push_back(std::move(prepared));
            }
        }
        report.n_inputs = static_cast<int>(train_.size() + test_.size());
        std::cout << "split     : " << train_.size() << " ingest / " << test_.size()
                  << " runtime (disjoint)\n";
        return !train_.empty() && !test_.empty();
    }

    // -----------------------------------------------------------------------
    // Phase 1: offline ingestion
    // -----------------------------------------------------------------------
    void ingest() {
        print_banner("phase 1: ingest annotated primitives, compile knowledge base");
        double elapsed = 0.0;
        {
            ScopedTimer timer(&elapsed);
            // Ingest vectorizes with the identical deterministic front end the
            // runtime uses, and consults the annotation only to name what it
            // found. Training on hand-drawn silhouettes and then querying with
            // segmenter output would put the two phases in different corners
            // of the descriptor space, and retrieval would be comparing things
            // that were never measured the same way.
            std::vector<structures::GlyphObservation> observations;
            for (const PreparedSample& sample : train_) {
                const pipeline::LabelField field =
                    pipeline::regions_from_pixels(sample.rgb, superpixels, target_regions);
                Scene scene = pipeline::build_scene(sample.stem, sample.rgb, sample.luma,
                                                    field.labels, field.count, field.region_class,
                                                    options_);
                const std::vector<int> labels =
                    majority_classes(scene, sample.annotation, label_purity);
                for (size_t i = 0; i < scene.primitives.size(); ++i) {
                    scene.primitives[i].semantic_class = labels[i];
                    if (scene.primitives[i].shape.valid && labels[i] > 0) {
                        observations.push_back({scene.primitives[i].shape, labels[i]});
                    }
                }
                ingest_scenes_.push_back(std::move(scene));
            }
            base_.ingest_scenes = static_cast<int>(ingest_scenes_.size());
            base_.ingest_primitives = static_cast<int>(observations.size());

            base_.vocabulary.target_glyphs = target_glyphs;
            base_.vocabulary.fit(observations);

            std::vector<structures::VectorIndex::Point> exemplars;
            for (const Scene& scene : ingest_scenes_) {
                for (const Primitive& primitive : scene.primitives) {
                    if (!primitive.shape.valid || primitive.semantic_class <= 0) {
                        continue;
                    }
                    exemplars.push_back(pipeline::joint_descriptor(primitive));
                    base_.exemplar_class.push_back(primitive.semantic_class);
                }
            }
            base_.exemplars.build(std::move(exemplars));

            // Second pass: now that every primitive can be named, record the
            // part-whole relations between the names.
            for (Scene& scene : ingest_scenes_) {
                for (Primitive& primitive : scene.primitives) {
                    const auto nearest = base_.vocabulary.query(primitive.shape, 1);
                    primitive.glyph = nearest.empty() ? -1 : nearest.front().id;
                }
                for (const Primitive& primitive : scene.primitives) {
                    if (primitive.parent < 0 || primitive.glyph < 0 ||
                        primitive.semantic_class <= 0) {
                        continue;
                    }
                    const Primitive& parent =
                        scene.primitives[static_cast<size_t>(primitive.parent)];
                    if (parent.semantic_class <= 0) {
                        continue;
                    }
                    base_.relations.observe(parent.semantic_class, primitive.glyph,
                                            pipeline::observe_relation(scene, primitive),
                                            primitive.semantic_class);
                    ++base_.ingest_relations;
                }
            }
            base_.relations.finalize();
        }
        ingest_ms_ = elapsed;

        std::cout << "  primitives ingested : " << base_.ingest_primitives << '\n';
        std::cout << "  glyph vocabulary    : " << base_.vocabulary.glyphs().size()
                  << " archetypes\n";
        std::cout << "  exemplar index      : " << base_.exemplars.size()
                  << " joint shape+material vectors\n";
        std::cout << "  ARG edges           : " << base_.relations.size() << " from "
                  << base_.ingest_relations << " containments\n";
        std::cout << "  compile time        : " << std::fixed << std::setprecision(1) << elapsed
                  << " ms\n";

        checks.expect(!base_.vocabulary.glyphs().empty(), "vocabulary is non-empty");
        checks.expect(base_.relations.size() > 0, "ARG learned at least one part-whole relation");
        bool supported = true;
        for (const structures::GlyphEntry& glyph : base_.vocabulary.glyphs()) {
            supported = supported && glyph.support > 0;
        }
        checks.expect(supported, "every compiled glyph has non-zero support");
    }

    // -----------------------------------------------------------------------
    // Phase 2: deterministic runtime
    // -----------------------------------------------------------------------
    void run_runtime(const std::string& artifact_dir) {
        print_banner("phase 2: runtime resolution on held-out frames (pixels only)");
        primitives_tsv_ << "file\tprimitive\tarea\tperimeter\tvertices\tcircularity\tconvexity"
                           "\telongation\trectangularity\tarchetype\tparent\tenclosure"
                           "\ttexture_hash\troughness\tanisotropy\tcolor_roughness\n";
        classify_tsv_ << "file\tprimitive\tglyph\tarchetype\tretrieval_d\tshape_only\tknn"
                         "\tdag_verified\tdag_revised\tdag_score\tpredicted\ttruth\tcorrect"
                         "\trationale\n";

        double elapsed = 0.0;
        {
            ScopedTimer timer(&elapsed);
            for (const PreparedSample& sample : test_) {
                const pipeline::LabelField field =
                    pipeline::regions_from_pixels(sample.rgb, superpixels, target_regions);
                Scene scene = pipeline::build_scene(sample.stem, sample.rgb, sample.luma,
                                                    field.labels, field.count, field.region_class,
                                                    options_);
                const std::vector<int> truth = majority_classes(scene, sample.annotation, label_purity);
                std::vector<Classification> results =
                    pipeline::resolve(scene, base_, top_k, exemplar_k, accept_threshold, arg_gain,
                                      parent_confidence, truth);
                score(scene, results, sample);
                write_scene_artifacts(artifact_dir, scene, results);
                runtime_scenes_.push_back(std::move(scene));
                runtime_results_.push_back(std::move(results));
            }
        }
        runtime_ms_ = elapsed;

        std::cout << "  frames              : " << test_.size() << '\n';
        std::cout << "  primitives resolved : " << stat_primitives_ << " (" << stat_contained_
                  << " contained)\n";
        std::cout << "  ARG verified        : " << stat_verified_ << "  revised by context: "
                  << stat_revised_ << " (" << stat_revision_fixed_ << " corrected, "
                  << stat_revision_broke_ << " broken)\n";
        std::cout << "  runtime             : " << std::fixed << std::setprecision(1) << elapsed
                  << " ms for " << test_.size() << " frames ("
                  << (test_.empty() ? 0.0 : elapsed / test_.size()) << " ms/frame)\n";

        const auto ratio = [&](int count) {
            return stat_scored_ > 0 ? static_cast<double>(count) / stat_scored_ : 0.0;
        };
        prior_accuracy_ = stat_scored_ > 0 ? best_prior() : 0.0;
        shape_accuracy_ = ratio(stat_shape_correct_);
        knn_accuracy_ = ratio(stat_knn_correct_);
        resolved_accuracy_ = ratio(stat_resolved_correct_);
        candidate_accuracy_ = ratio(stat_candidate_hit_);

        std::cout << std::setprecision(3);
        std::cout << "  class accuracy      : prior " << prior_accuracy_ << " | shape-archetype "
                  << shape_accuracy_ << " | +material retrieval " << knn_accuracy_
                  << " | +ARG context " << resolved_accuracy_ << "  (top-" << exemplar_k
                  << " reachable " << candidate_accuracy_ << ")\n";

        checks.expect(stat_primitives_ > 0, "runtime produced primitives");
        checks.expect(stat_unassigned_ == 0, "every runtime primitive retrieved a glyph");
        checks.expect(knn_accuracy_ > shape_accuracy_,
                      "joint shape+material retrieval beats the shape archetype alone");
        checks.expect(resolved_accuracy_ >= knn_accuracy_ - 1e-9,
                      "ARG arbitration does not degrade retrieval");
        checks.expect(resolved_accuracy_ > prior_accuracy_,
                      "resolved classification beats the majority-class prior");
        checks.expect(stat_revision_fixed_ > stat_revision_broke_,
                      "contextual revisions correct more labels than they break");
    }

    // -----------------------------------------------------------------------
    // Phase 3: architectural guarantees
    // -----------------------------------------------------------------------
    void validate() {
        print_banner("phase 3: determinism, index correctness, descriptor invariance");

        // Determinism. The same frames through the same code must produce a
        // bit-identical decision stream; anything else means a hidden
        // dependency on memory layout, iteration order or an unseeded RNG.
        uint64_t first = 14695981039346656037ull;
        for (const auto& results : runtime_results_) {
            for (const Classification& c : results) {
                first = digest(first, classification_record(c));
            }
        }
        uint64_t second = 14695981039346656037ull;
        for (const PreparedSample& sample : test_) {
            const pipeline::LabelField field =
                pipeline::regions_from_pixels(sample.rgb, superpixels, target_regions);
            const Scene scene = pipeline::build_scene(sample.stem, sample.rgb, sample.luma,
                                                      field.labels, field.count,
                                                      field.region_class, options_);
            const std::vector<int> truth = majority_classes(scene, sample.annotation, label_purity);
            for (const Classification& c :
                 pipeline::resolve(scene, base_, top_k, exemplar_k, accept_threshold, arg_gain,
                                      parent_confidence, truth)) {
                second = digest(second, classification_record(c));
            }
        }
        determinism_digest_ = first;
        std::cout << "  decision digest     : 0x" << std::hex << first << std::dec
                  << (first == second ? "  (reproduced)" : "  (MISMATCH)") << '\n';
        checks.expect(first == second, "re-running the runtime phase reproduces every decision");

        // Index correctness and the sublinear-retrieval claim.
        int mismatches = 0;
        long long visited = 0;
        int queries = 0;
        for (const Scene& scene : runtime_scenes_) {
            for (const Primitive& primitive : scene.primitives) {
                if (!primitive.shape.valid) {
                    continue;
                }
                const auto tree = base_.vocabulary.query(primitive.shape, top_k);
                visited += base_.vocabulary.index().visited_last_query();
                ++queries;
                const auto brute = brute_force(primitive.shape, top_k);
                if (tree.size() != brute.size()) {
                    ++mismatches;
                    continue;
                }
                for (size_t i = 0; i < tree.size(); ++i) {
                    if (tree[i].id != brute[i].id) {
                        ++mismatches;
                        break;
                    }
                }
            }
        }
        const double mean_visited = queries > 0 ? static_cast<double>(visited) / queries : 0.0;
        std::cout << "  VP-tree             : " << queries << " queries, " << mismatches
                  << " disagreements vs brute force, " << std::fixed << std::setprecision(1)
                  << mean_visited << " of " << base_.vocabulary.glyphs().size()
                  << " glyphs visited\n";
        vp_visited_ = mean_visited;
        checks.expect(mismatches == 0, "VP-tree top-k agrees with exhaustive search");
        checks.expect(mean_visited < static_cast<double>(base_.vocabulary.glyphs().size()),
                      "VP-tree pruning visits fewer glyphs than a linear scan");

        // Descriptor invariance.
        invariance_ = measure_invariance(options_);
        const double nn_accuracy =
            invariance_.poses > 0
                ? static_cast<double>(invariance_.correct) / invariance_.poses
                : 0.0;
        std::cout << "  shape invariance    : " << invariance_.correct << "/" << invariance_.poses
                  << " poses retrieve their own shape (" << std::setprecision(3) << nn_accuracy
                  << "), inter/intra separation " << invariance_.separation << "x\n";
        checks.expect(nn_accuracy >= 0.85,
                      "rotated and scaled poses retrieve their own shape class");
        checks.expect(invariance_.separation > 1.5f,
                      "between-shape distance exceeds within-shape distance");
        invariance_accuracy_ = nn_accuracy;

        // Texture illumination invariance. The relight is a gain and offset
        // chosen to stay inside [0, 255]: clipping is not an illumination
        // change but an irreversible loss of the very structure the signature
        // encodes, and no descriptor can be asked to survive it.
        //
        // The frequency hash is only asserted over surfaces that have a
        // frequency content to hash. On a near-uniform region the retained DCT
        // coefficients are quantization noise, the median threshold splits that
        // noise arbitrarily, and the resulting bits are meaningless -- which is
        // exactly what hash_confidence exists to report.
        int worst_hamming = 0;
        int worst_confident_hamming = 0;
        float worst_chi2 = 0.0f;
        int samples = 0;
        int confident = 0;
        for (const Scene& scene : runtime_scenes_) {
            math::ImageBuffer relit = scene.luma;
            for (uint8_t& value : relit.data) {
                value = static_cast<uint8_t>(
                    std::clamp(static_cast<int>(std::lround(value * 0.78 + 27.0)), 0, 255));
            }
            for (const Primitive& primitive : scene.primitives) {
                const descriptors::RegionView view = descriptors::make_region_view(
                    scene.labels, scene.width, scene.height, primitive.id);
                const descriptors::TextureSignature after =
                    descriptors::analyze_texture(scene.rgb, relit, view);
                const int hamming = descriptors::PerceptualHash::hamming(
                    primitive.texture.spatial_hash, after.spatial_hash);
                worst_hamming = std::max(worst_hamming, hamming);
                if (primitive.texture.hash_confidence > 0.5f) {
                    worst_confident_hamming = std::max(worst_confident_hamming, hamming);
                    ++confident;
                }
                worst_chi2 = std::max(worst_chi2, descriptors::LocalBinaryPattern::chi_square(
                                                      primitive.texture.lbp, after.lbp));
                ++samples;
            }
        }
        std::cout << "  texture relight     : worst LBP chi2 " << std::setprecision(4)
                  << worst_chi2 << ", worst pHash hamming " << worst_confident_hamming << "/64 over "
                  << confident << " textured regions (" << worst_hamming << "/64 including "
                  << (samples - confident) << " flat ones)\n";
        relight_hamming_ = worst_confident_hamming;
        relight_chi2_ = worst_chi2;
        checks.expect(confident == 0 || worst_confident_hamming <= 12,
                      "perceptual hash survives a relight on surfaces that have texture");
        checks.expect(worst_chi2 < 0.25f, "LBP census survives a gain-and-offset relight");

        // Negative control: the ARG must score attested pairings above
        // deliberately mismatched ones, otherwise its constraints carry no
        // information and the whole symbolic layer is decorative.
        double attested = 0.0;
        int attested_count = 0;
        double mismatched = 0.0;
        int mismatched_count = 0;
        const int glyph_count = static_cast<int>(base_.vocabulary.glyphs().size());
        for (const Scene& scene : ingest_scenes_) {
            for (const Primitive& primitive : scene.primitives) {
                if (primitive.parent < 0 || primitive.glyph < 0) {
                    continue;
                }
                const Primitive& parent = scene.primitives[static_cast<size_t>(primitive.parent)];
                if (parent.semantic_class <= 0) {
                    continue;
                }
                const auto observation = pipeline::observe_relation(scene, primitive);
                const auto real = base_.relations.validate(parent.semantic_class, primitive.glyph,
                                                           observation);
                if (real.edge_known) {
                    attested += real.score;
                    ++attested_count;
                }
                // Same measurements, wrong part: whatever score survives this
                // is score the constraints were never really checking.
                const int shifted = (primitive.glyph + glyph_count / 2) % std::max(1, glyph_count);
                const auto fake =
                    base_.relations.validate(parent.semantic_class, shifted, observation);
                if (fake.edge_known) {
                    mismatched += fake.score;
                    ++mismatched_count;
                }
            }
        }
        attested_score_ = attested_count > 0 ? attested / attested_count : 0.0;
        mismatched_score_ = mismatched_count > 0 ? mismatched / mismatched_count : 0.0;
        std::cout << "  ARG discrimination  : attested pairs score " << std::setprecision(3)
                  << attested_score_ << " vs " << mismatched_score_ << " for mismatched parents ("
                  << mismatched_count << " controls)\n";
        checks.expect(mismatched_count == 0 || attested_score_ > mismatched_score_,
                      "ARG scores attested part-whole pairs above mismatched ones");

        // Does the material fingerprint actually do the job the architecture
        // assigns it -- telling a part cut from the whole apart from a foreign
        // object sitting on it? Compare it against the plain colour difference
        // on the same containments, since colour is the cheaper alternative it
        // has to earn its place against.
        double same_material = 0.0;
        double same_delta_e = 0.0;
        int same_count = 0;
        double foreign_material = 0.0;
        double foreign_delta_e = 0.0;
        int foreign_count = 0;
        for (const structures::RelationEdge* edge : base_.relations.ordered_edges()) {
            const bool part_is_whole = edge->dominant_child_class == edge->parent_node;
            const double weight = edge->support;
            if (part_is_whole) {
                same_material += edge->material_distance.mean * weight;
                same_delta_e += edge->delta_e.mean * weight;
                same_count += edge->support;
            } else {
                foreign_material += edge->material_distance.mean * weight;
                foreign_delta_e += edge->delta_e.mean * weight;
                foreign_count += edge->support;
            }
        }
        if (same_count > 0 && foreign_count > 0) {
            same_material_ = same_material / same_count;
            foreign_material_ = foreign_material / foreign_count;
            same_delta_e_ = same_delta_e / same_count;
            foreign_delta_e_ = foreign_delta_e / foreign_count;
        }
        std::cout << "  material vs colour  : part-is-whole dE " << std::setprecision(1)
                  << same_delta_e_ << " / material " << std::setprecision(3) << same_material_
                  << "   foreign-part dE " << std::setprecision(1) << foreign_delta_e_
                  << " / material " << std::setprecision(3) << foreign_material_ << '\n';
    }

    void write(const std::string& artifact_dir) {
        std::ostringstream glyphs;
        glyphs << "glyph\tarchetype\tsupport\tmean_radius\tmax_radius\tdominant_class\tpurity"
                  "\tcircularity\tconvexity\telongation\trectangularity\textent\tcorner_density\n";
        for (const structures::GlyphEntry& glyph : base_.vocabulary.glyphs()) {
            glyphs << glyph.id << '\t' << glyph.archetype << '\t' << glyph.support << '\t'
                   << std::fixed << std::setprecision(4) << glyph.mean_radius << '\t'
                   << glyph.max_radius << '\t'
                   << (glyph.dominant_class >= 0 ? pipeline::class_name(glyph.dominant_class)
                                                 : "-")
                   << '\t' << glyph.purity << '\t' << glyph.traits.circularity << '\t'
                   << glyph.traits.convexity << '\t' << glyph.traits.elongation << '\t'
                   << glyph.traits.rectangularity << '\t' << glyph.traits.extent << '\t'
                   << glyph.traits.corner_density << '\n';
        }

        std::ostringstream relations;
        relations << "whole\tpart_glyph\tpart_archetype\tsupport\tarea_ratio\tdelta_e\tmaterial"
                     "\toffset\tcontrast\tenclosure\tpart_class\tclass_purity\n";
        for (const structures::RelationEdge* edge : base_.relations.ordered_edges()) {
            relations << pipeline::class_name(edge->parent_node) << '\t' << edge->child_node
                      << '\t'
                      << base_.vocabulary.glyphs()[static_cast<size_t>(edge->child_node)].archetype
                      << '\t' << edge->support << '\t' << std::fixed << std::setprecision(4)
                      << edge->area_ratio.mean << '\t' << edge->delta_e.mean << '\t'
                      << edge->material_distance.mean << '\t' << edge->radial_offset.mean << '\t'
                      << edge->boundary_contrast.mean << '\t' << edge->enclosure.mean << '\t'
                      << (edge->dominant_child_class >= 0
                              ? pipeline::class_name(edge->dominant_child_class)
                              : "-")
                      << '\t' << edge->child_class_purity << '\n';
        }

        std::ostringstream summary;
        summary << "metric\tvalue\n";
        summary << "ingest_frames\t" << train_.size() << '\n';
        summary << "runtime_frames\t" << test_.size() << '\n';
        summary << "ingest_primitives\t" << base_.ingest_primitives << '\n';
        summary << "glyph_vocabulary\t" << base_.vocabulary.glyphs().size() << '\n';
        summary << "arg_edges\t" << base_.relations.size() << '\n';
        summary << "arg_containments\t" << base_.ingest_relations << '\n';
        summary << "runtime_primitives\t" << stat_primitives_ << '\n';
        summary << "runtime_contained\t" << stat_contained_ << '\n';
        summary << "arg_verified\t" << stat_verified_ << '\n';
        summary << "arg_revised\t" << stat_revised_ << '\n';
        summary << "arg_revisions_corrected\t" << stat_revision_fixed_ << '\n';
        summary << "arg_revisions_broken\t" << stat_revision_broke_ << '\n';
        summary << std::fixed << std::setprecision(4);
        summary << "accuracy_prior\t" << prior_accuracy_ << '\n';
        summary << "accuracy_shape_archetype\t" << shape_accuracy_ << '\n';
        summary << "accuracy_joint_retrieval\t" << knn_accuracy_ << '\n';
        summary << "accuracy_arg_resolved\t" << resolved_accuracy_ << '\n';
        summary << "accuracy_topk_reachable\t" << candidate_accuracy_ << '\n';
        summary << "invariance_nn_accuracy\t" << invariance_accuracy_ << '\n';
        summary << "invariance_separation\t" << invariance_.separation << '\n';
        summary << "relight_worst_hamming\t" << relight_hamming_ << '\n';
        summary << "relight_worst_lbp_chi2\t" << relight_chi2_ << '\n';
        summary << "arg_attested_score\t" << attested_score_ << '\n';
        summary << "arg_mismatched_score\t" << mismatched_score_ << '\n';
        summary << "part_is_whole_delta_e\t" << same_delta_e_ << '\n';
        summary << "foreign_part_delta_e\t" << foreign_delta_e_ << '\n';
        summary << "part_is_whole_material\t" << same_material_ << '\n';
        summary << "foreign_part_material\t" << foreign_material_ << '\n';
        summary << "vp_tree_mean_visited\t" << vp_visited_ << '\n';
        summary << "decision_digest\t0x" << std::hex << determinism_digest_ << std::dec << '\n';
        summary << "ingest_ms\t" << ingest_ms_ << '\n';
        summary << "runtime_ms_per_frame\t"
                << (test_.empty() ? 0.0 : runtime_ms_ / test_.size()) << '\n';

        auto put = [&](const char* name, const std::string& text) {
            vision::write_text_file(vision::join_path(artifact_dir, name), text);
            written.insert(written.begin(), name);
        };
        put("kb_glyphs.tsv", glyphs.str());
        put("kb_relations.tsv", relations.str());
        put("runtime_primitives.tsv", primitives_tsv_.str());
        put("runtime_classifications.tsv", classify_tsv_.str());
        put("shape_invariance.tsv", invariance_.table);
        put("pipeline_summary.tsv", summary.str());

        std::ostringstream note;
        note << "ingest " << train_.size() << " annotated frames -> " << base_.ingest_primitives
             << " primitives -> " << base_.vocabulary.glyphs().size() << " glyphs, "
             << base_.relations.size() << " ARG edges";
        report.notes.push_back(note.str());
        std::ostringstream note2;
        note2 << std::fixed << std::setprecision(3) << "held-out class accuracy: prior "
              << prior_accuracy_ << ", shape-archetype " << shape_accuracy_
              << ", +material retrieval " << knn_accuracy_ << ", +ARG context "
              << resolved_accuracy_;
        report.notes.push_back(note2.str());
        report.n_outputs = static_cast<int>(written.size());

        write_atom_manifest(artifact_dir, report, written);
        report.print();
        std::cout << "invariants: " << checks.passed << " / " << checks.total << " passed\n";
        for (const std::string& failure : checks.failures) {
            std::cout << "  FAIL  " << failure << '\n';
        }
        std::cout << "artifacts -> " << artifact_dir << '\n';

        print_banner("worked example: an explained classification");
        print_example();
    }

    int status() const { return checks.failures.empty() ? 0 : 1; }

private:
    int max_side = 176;
    int ingest_samples = 260;
    int runtime_samples = 24;
    // The glyph alphabet is the ARG's node set, so it must stay coarse enough
    // that relations between glyphs accumulate real support. A fine alphabet
    // partitions the containments into singletons and every edge becomes an
    // anecdote; fine-grained identity is the exemplar index's job, not this
    // one's.
    int target_glyphs = 18;
    int top_k = 7;
    int exemplar_k = 12;
    int superpixels = 260;
    // Deliberately finer than a "one region per object" target: part-whole
    // structure only exists if parts are segmented separately from wholes.
    int target_regions = 34;
    float accept_threshold = 0.05f;
    float arg_gain = 2.0f;
    float parent_confidence = 0.40f;
    // A region must be this pure under the annotation before it is used, so
    // that a straddling region neither trains the index nor is scored as if
    // the ground truth had a single answer for it.
    float label_purity = 0.70f;
    SceneOptions options_;

    std::vector<PreparedSample> train_;
    std::vector<PreparedSample> test_;
    KnowledgeBase base_;
    std::vector<Scene> ingest_scenes_;
    std::vector<Scene> runtime_scenes_;
    std::vector<std::vector<Classification>> runtime_results_;
    std::ostringstream primitives_tsv_;
    std::ostringstream classify_tsv_;

    int stat_primitives_ = 0;
    int stat_contained_ = 0;
    int stat_verified_ = 0;
    int stat_revised_ = 0;
    int stat_unassigned_ = 0;
    int stat_scored_ = 0;
    int stat_shape_correct_ = 0;
    int stat_knn_correct_ = 0;
    int stat_resolved_correct_ = 0;
    int stat_revision_fixed_ = 0;
    int stat_revision_broke_ = 0;
    int stat_candidate_hit_ = 0;
    std::map<int, int> truth_histogram_;

    double ingest_ms_ = 0.0;
    double runtime_ms_ = 0.0;
    double shape_accuracy_ = 0.0;
    double knn_accuracy_ = 0.0;
    double resolved_accuracy_ = 0.0;
    double candidate_accuracy_ = 0.0;
    double prior_accuracy_ = 0.0;
    double invariance_accuracy_ = 0.0;
    double attested_score_ = 0.0;
    double mismatched_score_ = 0.0;
    double vp_visited_ = 0.0;
    double same_material_ = 0.0;
    double foreign_material_ = 0.0;
    double same_delta_e_ = 0.0;
    double foreign_delta_e_ = 0.0;
    int relight_hamming_ = 0;
    float relight_chi2_ = 0.0f;
    uint64_t determinism_digest_ = 0;
    InvarianceReport invariance_;

    // Accuracy of the trivial "always guess the commonest class" strategy, so
    // the reported numbers can be read against something.
    double best_prior() const {
        int best = 0;
        for (const auto& entry : truth_histogram_) {
            best = std::max(best, entry.second);
        }
        return stat_scored_ > 0 ? static_cast<double>(best) / stat_scored_ : 0.0;
    }

    std::vector<structures::VectorIndex::Neighbor> brute_force(
        const descriptors::ShapeDescriptor& shape, int k) const {
        const auto flat = shape.flatten();
        const structures::VectorIndex::Point query(flat.begin(), flat.end());
        std::vector<structures::VectorIndex::Neighbor> all;
        for (const structures::GlyphEntry& glyph : base_.vocabulary.glyphs()) {
            all.push_back({glyph.id, structures::VectorIndex::l2(query, glyph.centroid)});
        }
        std::sort(all.begin(), all.end(),
                  [](const structures::VectorIndex::Neighbor& a,
                     const structures::VectorIndex::Neighbor& b) {
                      return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
                  });
        if (static_cast<int>(all.size()) > k) {
            all.resize(static_cast<size_t>(k));
        }
        return all;
    }

    void score(const Scene& scene, const std::vector<Classification>& results,
               const PreparedSample& sample) {
        for (size_t i = 0; i < results.size(); ++i) {
            const Primitive& primitive = scene.primitives[i];
            const Classification& result = results[i];
            ++stat_primitives_;
            stat_contained_ += primitive.parent >= 0 ? 1 : 0;
            stat_verified_ += result.dag_verified ? 1 : 0;
            stat_revised_ += result.dag_revised ? 1 : 0;
            stat_unassigned_ += result.glyph < 0 ? 1 : 0;

            if (result.truth_class > 0) {
                ++stat_scored_;
                ++truth_histogram_[result.truth_class];
                stat_shape_correct_ += result.shape_only_class == result.truth_class ? 1 : 0;
                stat_knn_correct_ += result.knn_class == result.truth_class ? 1 : 0;
                stat_resolved_correct_ += result.predicted_class == result.truth_class ? 1 : 0;
                stat_candidate_hit_ += result.candidate_hit ? 1 : 0;
                if (result.dag_revised) {
                    const bool was = result.knn_class == result.truth_class;
                    const bool now = result.predicted_class == result.truth_class;
                    stat_revision_fixed_ += !was && now ? 1 : 0;
                    stat_revision_broke_ += was && !now ? 1 : 0;
                }
            }

            primitives_tsv_ << sample.stem << '\t' << primitive.id << '\t' << primitive.area
                            << '\t' << primitive.perimeter << '\t' << primitive.polygon.size()
                            << '\t' << std::fixed << std::setprecision(4)
                            << primitive.shape.traits.circularity << '\t'
                            << primitive.shape.traits.convexity << '\t'
                            << primitive.shape.traits.elongation << '\t'
                            << primitive.shape.traits.rectangularity << '\t'
                            << descriptors::name_archetype(primitive.shape.traits) << '\t'
                            << primitive.parent << '\t' << primitive.enclosure << '\t' << std::hex
                            << primitive.texture.spatial_hash << std::dec << '\t'
                            << primitive.texture.roughness << '\t'
                            << primitive.texture.anisotropy << '\t'
                            << primitive.texture.color_roughness << '\n';

            const auto named = [](int semantic) {
                return semantic > 0 ? pipeline::class_name(semantic) : "-";
            };
            classify_tsv_ << sample.stem << '\t' << result.primitive << '\t' << result.glyph
                          << '\t'
                          << (result.glyph >= 0
                                  ? base_.vocabulary.glyphs()[static_cast<size_t>(result.glyph)]
                                        .archetype
                                  : "-")
                          << '\t' << std::fixed << std::setprecision(4)
                          << result.retrieval_distance << '\t' << named(result.shape_only_class)
                          << '\t' << named(result.knn_class) << '\t'
                          << (result.dag_verified ? 1 : 0) << '\t' << (result.dag_revised ? 1 : 0)
                          << '\t' << result.dag_score << '\t' << named(result.predicted_class)
                          << '\t' << named(result.truth_class) << '\t'
                          << (result.predicted_class == result.truth_class && result.truth_class > 0
                                  ? 1
                                  : 0)
                          << '\t' << result.rationale << '\n';
        }
    }

    void save(const std::string& dir, const std::string& name, const vision::GrayImage& image) {
        vision::save_pgm(vision::join_path(dir, name), image);
        written.push_back(name);
    }

    void write_scene_artifacts(const std::string& dir, const Scene& scene,
                               const std::vector<Classification>& results) {
        const int w = scene.width;
        const int h = scene.height;
        save(dir, scene.stem + "_00_input.pgm", scene.luma);

        std::vector<int> shifted(scene.labels.size());
        for (size_t i = 0; i < scene.labels.size(); ++i) {
            shifted[i] = scene.labels[i] + 1;
        }
        save(dir, scene.stem + "_01_segmentation.pgm", colorize_labels(shifted, w, h));

        // Vectorized primitives: the simplified polygons that actually enter
        // the descriptor, not the raw pixel boundary.
        vision::GrayImage polygons = scene.luma;
        for (uint8_t& value : polygons.data) {
            value = static_cast<uint8_t>(value / 3);
        }
        for (const Primitive& primitive : scene.primitives) {
            for (size_t i = 0; i < primitive.polygon.size(); ++i) {
                const Vec2& a = primitive.polygon[i];
                const Vec2& b = primitive.polygon[(i + 1) % primitive.polygon.size()];
                plot_line(polygons, static_cast<int>(std::lround(a.x)),
                          static_cast<int>(std::lround(a.y)),
                          static_cast<int>(std::lround(b.x)),
                          static_cast<int>(std::lround(b.y)), 255);
            }
        }
        save(dir, scene.stem + "_02_vector_polygons.pgm", polygons);

        // Part-whole forest: each child linked to the parent that encloses it.
        vision::GrayImage dag = scene.luma;
        for (uint8_t& value : dag.data) {
            value = static_cast<uint8_t>(value / 3);
        }
        for (const Primitive& primitive : scene.primitives) {
            if (primitive.parent < 0) {
                continue;
            }
            const Primitive& parent = scene.primitives[static_cast<size_t>(primitive.parent)];
            plot_line(dag, static_cast<int>(std::lround(primitive.centroid.x)),
                      static_cast<int>(std::lround(primitive.centroid.y)),
                      static_cast<int>(std::lround(parent.centroid.x)),
                      static_cast<int>(std::lround(parent.centroid.y)), 200);
        }
        for (const Primitive& primitive : scene.primitives) {
            const int cx = static_cast<int>(std::lround(primitive.centroid.x));
            const int cy = static_cast<int>(std::lround(primitive.centroid.y));
            for (int d = -2; d <= 2; ++d) {
                if (cx + d >= 0 && cx + d < w) {
                    dag.at(cx + d, cy) = 255;
                }
                if (cy + d >= 0 && cy + d < h) {
                    dag.at(cx, cy + d) = 255;
                }
            }
        }
        save(dir, scene.stem + "_03_part_whole_dag.pgm", dag);

        // Predicted semantic class per pixel.
        std::vector<int> predicted(scene.labels.size(), 0);
        for (size_t p = 0; p < scene.labels.size(); ++p) {
            const int id = scene.labels[p];
            if (id >= 0 && results[static_cast<size_t>(id)].predicted_class > 0) {
                predicted[p] = results[static_cast<size_t>(id)].predicted_class;
            }
        }
        save(dir, scene.stem + "_04_predicted_classes.pgm", colorize_labels(predicted, w, h));

        // Which regions the ARG was able to verify.
        vision::GrayImage verified =
            overlay_mask(scene.luma, label_boundaries(shifted, w, h));
        for (size_t p = 0; p < scene.labels.size(); ++p) {
            const int id = scene.labels[p];
            if (id >= 0 && !results[static_cast<size_t>(id)].dag_verified) {
                verified.data[p] = static_cast<uint8_t>(verified.data[p] / 2);
            }
        }
        save(dir, scene.stem + "_05_arg_verified.pgm", verified);
    }

    void print_example() {
        for (size_t s = 0; s < runtime_scenes_.size(); ++s) {
            for (size_t i = 0; i < runtime_results_[s].size(); ++i) {
                const Classification& result = runtime_results_[s][i];
                if (!result.dag_verified || result.truth_class <= 0 ||
                    result.predicted_class != result.truth_class) {
                    continue;
                }
                const Primitive& primitive = runtime_scenes_[s].primitives[i];
                std::cout << "  frame " << runtime_scenes_[s].stem << ", region "
                          << primitive.id << " (" << primitive.area << " px)\n";
                std::cout << "    classified : " << pipeline::class_name(result.predicted_class)
                          << "   [annotation says " << pipeline::class_name(result.truth_class)
                          << "]\n";
                std::cout << "    because    : " << result.rationale << '\n';
                return;
            }
        }
        std::cout << "  (no ARG-verified region matched its annotation in this run)\n";
    }
};

}  // namespace

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "ade20k", [&](const AtomCli& cli) -> int {
        PipelineIntegration pipeline_run;
        if (!pipeline_run.load(cli, argc, argv)) {
            std::cerr << "structural pipeline needs annotated samples for both phases\n";
            return 1;
        }
        const std::string artifacts =
            make_artifact_dir(vision::join_path(cli.artifact_dir, "pipeline"));
        pipeline_run.ingest();
        pipeline_run.run_runtime(artifacts);
        pipeline_run.validate();
        pipeline_run.write(artifacts);
        return pipeline_run.status();
    });
}

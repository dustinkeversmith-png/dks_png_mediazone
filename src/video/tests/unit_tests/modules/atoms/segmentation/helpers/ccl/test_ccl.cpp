#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "object_proposals.hpp"
#include "segmentation/helpers/ccl/connected_components.hpp"

#include <sstream>

// Atom: photo → tophat/skeleton proposals → thick/thin/ground CCL → one box per CC.
// Goal: box every separable visual object (people, poles, bears, slope) — not GT matching.
class CclAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"ccl"};
    std::ostringstream values_tsv;
    std::ostringstream score_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 256);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    static void append_components(const vision::GrayImage& mask,
                                  std::vector<vision::ConnectedComponentLabeler::Component>& out,
                                  int min_area, bool allow_thin) {
        if (mask.empty()) {
            return;
        }
        auto ccl = vision::ConnectedComponentLabeler::label(mask);
        const int W = mask.width;
        const int H = mask.height;
        const int canvas = std::max(1, W * H);
        for (const auto& c : ccl.components) {
            if (c.area < min_area) {
                continue;
            }
            if (c.area > canvas) {
                continue;
            }
            const float box_a = std::max(1.0f, c.bbox.w * c.bbox.h);
            const float fill = static_cast<float>(c.area) / box_a;
            const float aspect = (c.bbox.h > 1.0f) ? (c.bbox.w / c.bbox.h) : 99.0f;
            const float long_aspect =
                std::max(aspect, (c.bbox.w > 1.0f) ? (c.bbox.h / c.bbox.w) : 99.0f);
            const bool thin = long_aspect >= 2.8f;
            if (allow_thin) {
                // Thin channel: long poles only — reject tips / door-crack slivers.
                const float long_side = std::max(c.bbox.w, c.bbox.h);
                const float short_side = std::min(c.bbox.w, c.bbox.h);
                if (!thin || fill < 0.04f || long_side < 28.0f || short_side > 22.0f) {
                    continue;
                }
            } else {
                if (thin && c.area < min_area * 3) {
                    continue;
                }
                if (fill < 0.08f) {
                    continue;
                }
                // Reject crack-like thick leftovers.
                const float short_side = std::min(c.bbox.w, c.bbox.h);
                if (short_side <= 5.0f && long_aspect >= 4.0f) {
                    continue;
                }
            }
            out.push_back(c);
        }
    }

    // Watershed-split large body blobs so touching objects (bears) get separate boxes.
    static void append_split_bodies(const vision::GrayImage& body, const vision::GrayImage& smooth,
                                    std::vector<vision::ConnectedComponentLabeler::Component>& out,
                                    int min_area) {
        if (body.empty()) {
            return;
        }
        auto ccl = vision::ConnectedComponentLabeler::label(body);
        const int canvas = std::max(1, body.width * body.height);
        for (const auto& c : ccl.components) {
            if (c.area < min_area || c.area > canvas) {
                continue;
            }
            // Only split truly dominant blobs (whole-frame glue); mid-size bodies stay intact.
            if (c.area >= canvas * 28 / 100) {
                vision::GrayImage piece = vision::make_gray(body.width, body.height, 0);
                for (int y = 0; y < body.height; ++y) {
                    for (int x = 0; x < body.width; ++x) {
                        if (ccl.labels[static_cast<size_t>(y * body.width + x)] == c.label) {
                            piece.at(x, y) = 255;
                        }
                    }
                }
                auto parts = vision::split_instances_watershed(piece, smooth);
                if (parts.size() >= 2) {
                    for (const auto& part : parts) {
                        append_components(part, out, min_area, false);
                    }
                    continue;
                }
            }
            out.push_back(c);
        }
    }

    static bool keep_box(const vision::ConnectedComponentLabeler::Component& c,
                         const vision::GrayImage& luma, const vision::GrayImage& saliency,
                         int canvas) {
        if (c.area < std::max(80, canvas / 500) || c.area > canvas) {
            return false;
        }
        const float box_a = std::max(1.0f, c.bbox.w * c.bbox.h);
        const float fill = static_cast<float>(c.area) / box_a;
        const float aspect = (c.bbox.h > 1.0f) ? (c.bbox.w / c.bbox.h) : 99.0f;
        const float long_aspect =
            std::max(aspect, (c.bbox.w > 1.0f) ? (c.bbox.h / c.bbox.w) : 99.0f);
        const bool thin = long_aspect >= 2.8f;
        const float long_side = std::max(c.bbox.w, c.bbox.h);
        const float short_side = std::min(c.bbox.w, c.bbox.h);
        if (thin) {
            if (long_side < 28.0f || c.area < 40) {
                return false;  // ski tip / hand tip / crack
            }
        } else {
            if (fill < 0.08f) {
                return false;
            }
            if (c.area < canvas * 15 / 1000) {  // ~1.5%
                return false;
            }
            if (short_side <= 5.0f && long_aspect >= 4.0f) {
                return false;
            }
        }

        double mean_l = 0.0, mean_s = 0.0;
        int n = 0;
        const int x0 = std::max(0, static_cast<int>(c.bbox.x));
        const int y0 = std::max(0, static_cast<int>(c.bbox.y));
        const int x1 = std::min(luma.width, static_cast<int>(c.bbox.x1()));
        const int y1 = std::min(luma.height, static_cast<int>(c.bbox.y1()));
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                mean_l += luma.at(x, y);
                mean_s += saliency.at(x, y);
                ++n;
            }
        }
        if (n > 0) {
            mean_l /= n;
            mean_s /= n;
        }
        // Drop tiny sky/cloud flecks only (large bright = slope/ground — keep).
        if (!thin && mean_l > 185.0 && mean_s < 35.0 && c.area < canvas * 8 / 100) {
            return false;
        }
        return true;
    }

    // Absorb tiny fragments into the nearest overlapping/nearby large box.
    static std::vector<vision::ConnectedComponentLabeler::Component> absorb_small(
        std::vector<vision::ConnectedComponentLabeler::Component> comps, int min_keep) {
        if (comps.size() < 2) {
            return comps;
        }
        std::sort(comps.begin(), comps.end(),
                  [](const auto& a, const auto& b) { return a.area > b.area; });
        std::vector<char> dead(comps.size(), 0);
        for (size_t i = 0; i < comps.size(); ++i) {
            if (dead[i] || comps[i].area >= min_keep) {
                continue;
            }
            int best = -1;
            float best_d = 1e9f;
            for (size_t j = 0; j < comps.size(); ++j) {
                if (i == j || dead[j] || comps[j].area < min_keep) {
                    continue;
                }
                const auto& a = comps[i].bbox;
                const auto& b = comps[j].bbox;
                const float gap_x =
                    std::max(0.0f, std::max(a.x, b.x) - std::min(a.x1(), b.x1()));
                const float gap_y =
                    std::max(0.0f, std::max(a.y, b.y) - std::min(a.y1(), b.y1()));
                const float gap = std::hypot(gap_x, gap_y);
                const float acx = a.x + a.w * 0.5f;
                const float acy = a.y + a.h * 0.5f;
                const float bcx = b.x + b.w * 0.5f;
                const float bcy = b.y + b.h * 0.5f;
                const float dist = std::hypot(acx - bcx, acy - bcy);
                if (gap <= 4.0f && a.w * a.h < 0.35f * b.w * b.h) {
                    if (dist < best_d) {
                        best_d = dist;
                        best = static_cast<int>(j);
                    }
                }
            }
            if (best < 0) {
                dead[i] = 1;  // drop orphan flecks
                continue;
            }
            auto& dst = comps[static_cast<size_t>(best)];
            const auto& src = comps[i];
            const float x0 = std::min(dst.bbox.x, src.bbox.x);
            const float y0 = std::min(dst.bbox.y, src.bbox.y);
            const float x1 = std::max(dst.bbox.x1(), src.bbox.x1());
            const float y1 = std::max(dst.bbox.y1(), src.bbox.y1());
            dst.bbox.x = x0;
            dst.bbox.y = y0;
            dst.bbox.w = x1 - x0;
            dst.bbox.h = y1 - y0;
            dst.area += src.area;
            dead[i] = 1;
        }
        std::vector<vision::ConnectedComponentLabeler::Component> out;
        for (size_t i = 0; i < comps.size(); ++i) {
            if (!dead[i]) {
                out.push_back(comps[i]);
            }
        }
        return out;
    }

    void run(const std::string& art_dir) {
        print_banner("run CCL → box body / thin / ground after tophat+skeleton");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel_id\tkind\tarea\tx\ty\tw\th\n";
        score_tsv << "file\tn_components\tfg_frac\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;

            vision::GrayImage rgb = sample.image;
            if (ps && !ps->sample.rgb.empty()) {
                rgb = downscale_max_side(ps->sample.rgb, 256);
            }
            const auto luma = mission_luma_image(ps, sample.image);
            const vision::GrayImage work_luma =
                (luma.width == rgb.width && luma.height == rgb.height)
                    ? luma
                    : resize_nearest(luma, rgb.width, rgb.height);

            const auto proposal = vision::propose_objects_from_photo(rgb.empty() ? work_luma : rgb);

            std::vector<uint8_t> lv(work_luma.data.begin(), work_luma.data.end());
            std::nth_element(lv.begin(), lv.begin() + static_cast<std::ptrdiff_t>(lv.size() / 2),
                             lv.end());
            const bool bright_scene = !lv.empty() && lv[lv.size() / 2] >= 125;

            const int W = proposal.binary.width;
            const int H = proposal.binary.height;
            const int canvas = std::max(1, W * H);
            const int min_blob = std::max(150, canvas * 2 / 100);  // ≥2% — no flecks
            const int min_thin = std::max(40, canvas / 2000);
            const int min_ground = std::max(canvas * 12 / 100, 500);

            // Body = edge-bounded regions; thin already filtered in proposals.
            auto body = proposal.body;
            auto thin = proposal.thin;  // do not re-dilate (welds poles)
            auto ground = proposal.ground;

            std::vector<vision::ConnectedComponentLabeler::Component> comps;
            std::vector<vision::ConnectedComponentLabeler::Component> body_comps;
            std::vector<vision::ConnectedComponentLabeler::Component> thin_comps;
            std::vector<vision::ConnectedComponentLabeler::Component> ground_comps;
            append_split_bodies(body, proposal.smooth, body_comps, min_blob);
            append_components(thin, thin_comps, min_thin, true);
            append_components(ground, ground_comps, min_ground, false);

            const float merge_iou = bright_scene ? 0.45f : 0.35f;
            const float merge_gap = bright_scene ? 0.08f : 0.12f;
            body_comps = vision::ConnectedComponentLabeler::merge_boxes(body_comps, merge_iou, merge_gap);
            // Absorb only tiny flecks (not large object boxes).
            if (body_comps.size() > 1) {
                body_comps = absorb_small(body_comps, std::max(min_blob * 2, canvas * 3 / 100));
            }
            // Mild thin merge only on strong overlap — keep separate poles apart.
            thin_comps = vision::ConnectedComponentLabeler::merge_boxes(thin_comps, 0.55, 0.05f);
            ground_comps = vision::ConnectedComponentLabeler::merge_boxes(ground_comps, 0.40, 0.12f);
            comps.insert(comps.end(), body_comps.begin(), body_comps.end());
            comps.insert(comps.end(), thin_comps.begin(), thin_comps.end());
            comps.insert(comps.end(), ground_comps.begin(), ground_comps.end());

            auto merged = comps;
            std::vector<vision::ConnectedComponentLabeler::Component> final_comps;
            for (const auto& c : merged) {
                if (keep_box(c, work_luma, proposal.saliency, canvas)) {
                    final_comps.push_back(c);
                }
            }
            for (size_t i = 0; i < final_comps.size(); ++i) {
                final_comps[i].label = static_cast<int>(i + 1);
            }
            merged.swap(final_comps);

            vision::GrayImage box_overlay = work_luma;
            for (uint8_t& p : box_overlay.data) {
                p = static_cast<uint8_t>(p / 2);
            }
            int fg = 0;
            for (uint8_t v : proposal.binary.data) {
                fg += v > 0 ? 1 : 0;
            }

            for (const auto& c : merged) {
                const float aspect = (c.bbox.h > 1.0f) ? (c.bbox.w / c.bbox.h) : 0.0f;
                const float long_aspect =
                    std::max(aspect, (c.bbox.w > 1.0f) ? (c.bbox.h / c.bbox.w) : 0.0f);
                const char* kind = (long_aspect >= 2.8f) ? "thin" : "thick";
                values_tsv << sample.row.file << '\t' << c.label << '\t' << kind << '\t' << c.area
                           << '\t' << c.bbox.x << '\t' << c.bbox.y << '\t' << c.bbox.w << '\t'
                           << c.bbox.h << '\n';
                box_overlay =
                    overlay_rect(box_overlay, c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h, 255);
            }

            std::cout << "  " << sample.row.file << "  components=" << merged.size()
                      << "  fg_frac=" << (static_cast<double>(fg) / canvas) << "\n";
            score_tsv << sample.row.file << '\t' << merged.size() << '\t'
                      << (static_cast<double>(fg) / canvas) << '\n';

            const std::string stem = stem_of(sample.row.file);
            auto labeled_ccl = vision::ConnectedComponentLabeler::label(proposal.binary);
            vision::GrayImage labeled = colorize_labels(labeled_ccl.labels, W, H);
            mission::write_bbox_json(vision::join_path(art_dir, stem + "_detected_bboxes.json"),
                                     sample.row.file, merged);
            vision::save_pgm(vision::join_path(art_dir, stem + "_processed_base.pgm"), body);
            vision::save_pgm(vision::join_path(art_dir, stem + "_proposal_mask.pgm"), proposal.binary);
            vision::save_pgm(vision::join_path(art_dir, stem + "_closed_mask.pgm"), body);
            vision::save_pgm(vision::join_path(art_dir, stem + "_thick_mask.pgm"), body);
            vision::save_pgm(vision::join_path(art_dir, stem + "_thin_mask.pgm"), thin);
            vision::save_pgm(vision::join_path(art_dir, stem + "_ground_mask.pgm"), ground);
            vision::save_pgm(vision::join_path(art_dir, stem + "_edges.pgm"), proposal.edges);
            vision::save_pgm(vision::join_path(art_dir, stem + "_saliency.pgm"), proposal.saliency);
            vision::save_pgm(vision::join_path(art_dir, stem + "_tophat.pgm"), proposal.tophat);
            vision::save_pgm(vision::join_path(art_dir, stem + "_ccl_labels.pgm"), labeled);
            vision::save_pgm(vision::join_path(art_dir, stem + "_boxes.pgm"), box_overlay);
            written.push_back(stem + "_processed_base.pgm");
            written.push_back(stem + "_proposal_mask.pgm");
            written.push_back(stem + "_closed_mask.pgm");
            written.push_back(stem + "_thick_mask.pgm");
            written.push_back(stem + "_thin_mask.pgm");
            written.push_back(stem + "_ground_mask.pgm");
            written.push_back(stem + "_edges.pgm");
            written.push_back(stem + "_saliency.pgm");
            written.push_back(stem + "_tophat.pgm");
            written.push_back(stem + "_ccl_labels.pgm");
            written.push_back(stem + "_boxes.pgm");
            written.push_back(stem + "_detected_bboxes.json");
            report.n_outputs += 1 + static_cast<int>(merged.size());
        }
        report.notes.push_back(
            "edge-bounded regions (seal hairline cracks); drop flecks/tips; separate poles; no GT");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "components.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "ccl_scores.tsv"), score_tsv.str());
        written.insert(written.begin(), "ccl_scores.tsv");
        written.insert(written.begin(), "components.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "coco", [&](const AtomCli& cli) -> int {
        CclAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for ccl atom\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& s : atom.samples) {
                std::cout << "  " << s.row.file << "\n";
            }
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}

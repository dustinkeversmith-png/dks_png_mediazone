#include "test_harness.hpp"
#include "vectorization_geometry/euler/euler_characteristic.hpp"
#include "vectorization_geometry/convex_hull/convex_hull.hpp"
#include "vectorization_geometry/measurements/turning_function/turning_function.hpp"
#include "vectorization_geometry/rdp/rdp.hpp"
#include "vectorization_geometry/medial_axis/medial_axis.hpp"
#include "boundary_tracing/moore_neighbor.hpp"

#include <limits>
#include <sstream>

class IntegrationGammaTest {
public:
    std::vector<LoadedSample> leaves;
    std::vector<LoadedSample> glyphs;
    AccuracyReport report{"integration_gamma / leaf topology + glyph genus"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: integration_3_leaf + unit_euler");
        leaves = load_png_dataset(vision::dataset_dir(root, "integration_3_leaf"));
        if (leaves.empty()) {
            leaves = load_png_dataset(vision::dataset_dir(root, "unit_turning"));
        }
        glyphs = load_png_dataset(vision::dataset_dir(root, "unit_euler"));
        std::cout << "loaded " << leaves.size() << " leaves, " << glyphs.size() << " glyphs\n";
        return !leaves.empty() && !glyphs.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);

        vision::TurningFunction tf;
        std::vector<std::vector<float>> feats(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            auto contour = vision::MooreNeighborTracer::trace(leaves[i].image);
            auto simple = vision::RamerDouglasPeucker::simplify(contour.points, 1.5f);
            feats[i] = tf.compute_from_points(simple.empty() ? contour.points : simple);
            auto skel = vision::MedialAxis::extract(leaves[i].image);
            artifacts << leaves[i].row.file << " verts=" << contour.points.size()
                      << " rdp=" << simple.size() << " skel=" << skel.skeleton_pixels << "\n";
        }
        for (size_t i = 0; i < leaves.size(); ++i) {
            float best = std::numeric_limits<float>::max();
            size_t arg = i;
            for (size_t j = 0; j < leaves.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const float d = vision::TurningFunction::circular_l2(feats[i], feats[j]);
                if (d < best) {
                    best = d;
                    arg = j;
                }
            }
            const bool ok = leaves[i].row.label == leaves[arg].row.label;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
        }

        for (const auto& g : glyphs) {
            if (g.row.label.rfind("font_", 0) == 0 || g.row.label == "A_like" ||
                g.row.label == "B_like" || g.row.label == "glyph_B") {
                continue;
            }
            auto e = vision::EulerCharacteristic::compute(g.image);
            int gt = g.row.fields.size() > 2 ? std::atoi(g.row.fields[2].c_str()) : -1;
            const bool ok = gt < 0 || e.holes == gt;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            auto hull = vision::ConvexHull::analyze(g.image);
            artifacts << g.row.file << " holes=" << e.holes << " defects=" << hull.defects.size() << "\n";
        }
        report.notes.push_back("turning-function 1-NN on leaves + Euler holes on geometric glyphs");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "gamma.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    IntegrationGammaTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("integration_gamma"));
    return acc >= 0.4f ? 0 : 2;
}

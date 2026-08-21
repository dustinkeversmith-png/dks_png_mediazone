#include "test_harness.hpp"
#include "vectorization_geometry/rdp/rdp.hpp"

#include <sstream>

class RdpUnitTest {
public:
    std::string folder;
    AccuracyReport report{"rdp / noisy contours"};
    std::ostringstream artifacts;
    std::vector<std::pair<std::string, std::vector<vision::Vec2>>> curves;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_rdp");
        folder = vision::dataset_dir(root, "unit_rdp");
        const char* names[] = {"noisy_ellipse.json", "noisy_bezier.json", "noisy_rdp_curve.json"};
        for (const char* name : names) {
            auto pts = vision::load_xy_json(vision::join_path(folder, name));
            if (!pts.empty()) {
                curves.push_back({name, pts});
            }
        }
        if (curves.empty()) {
            auto legacy = vision::load_xy_json(
                vision::join_path(vision::dataset_dir(root, "unit_synthetic"), "noisy_rdp_curve.json"));
            if (!legacy.empty()) {
                curves.push_back({"noisy_rdp_curve.json", legacy});
            }
        }
        std::cout << "loaded " << curves.size() << " polylines\n";
        return !curves.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        const float eps = 2.0f;
        for (const auto& curve : curves) {
            auto simplified = vision::RamerDouglasPeucker::simplify(curve.second, eps);
            const float err = vision::RamerDouglasPeucker::max_error(curve.second, simplified);
            const bool reduced = simplified.size() + 4 < curve.second.size();
            const bool bounded = err <= eps * 1.15f;
            const bool ok = reduced && bounded && simplified.size() >= 2;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            std::cout << "  " << curve.first << "  in=" << curve.second.size()
                      << "  out=" << simplified.size() << "  max_err=" << err
                      << (ok ? "  PASS\n" : "  FAIL\n");
            artifacts << curve.first << "\t" << curve.second.size() << "\t" << simplified.size()
                      << "\t" << err << "\n";
        }
        report.notes.push_back("RDP must drop vertices while keeping max perpendicular error <= epsilon");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "rdp.tsv"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    RdpUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("rdp"));
    return acc >= 0.66f ? 0 : 2;
}

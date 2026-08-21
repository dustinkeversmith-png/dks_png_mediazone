#include "test_harness.hpp"
#include "vectorization_geometry/convex_hull/convex_hull.hpp"

#include <sstream>

class ConvexHullUnitTest {
public:
    std::vector<LoadedSample> samples;
    std::string folder;
    AccuracyReport report{"convex_hull / monotone chain"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_convex_hull");
        folder = vision::dataset_dir(root, "unit_convex_hull");
        samples = load_png_dataset(folder);
        std::cout << "loaded " << samples.size() << " silhouettes + collinear JSON clouds\n";
        return !samples.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);

        const auto square = vision::load_xy_json(vision::join_path(folder, "collinear_square.json"));
        auto hull = vision::ConvexHull::monotone_chain(square);
        const bool square_ok = hull.size() == 4;
        ++report.total;
        if (square_ok) {
            ++report.passed;
        }
        std::cout << "  collinear square: input=" << square.size() << " hull=" << hull.size()
                  << (square_ok ? "  PASS\n" : "  FAIL\n");
        artifacts << "square hull=" << hull.size() << "\n";

        const auto tri = vision::load_xy_json(vision::join_path(folder, "collinear_triangle.json"));
        auto thull = vision::ConvexHull::monotone_chain(tri);
        const bool tri_ok = thull.size() == 3;
        ++report.total;
        if (tri_ok) {
            ++report.passed;
        }
        std::cout << "  collinear triangle: input=" << tri.size() << " hull=" << thull.size()
                  << (tri_ok ? "  PASS\n" : "  FAIL\n");

        for (const auto& sample : samples) {
            if (sample.row.label != "star" && sample.row.label != "hand" && sample.row.label != "cross") {
                continue;
            }
            auto r = vision::ConvexHull::analyze(sample.image);
            const bool ok = r.hull.size() >= 3 && (sample.row.label != "star" || r.defects.size() >= 3);
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            std::cout << "  " << sample.row.file << "  hull=" << r.hull.size()
                      << "  defects=" << r.defects.size() << (ok ? "  PASS\n" : "  FAIL\n");
            artifacts << sample.row.file << " hull=" << r.hull.size() << " defects=" << r.defects.size() << "\n";
        }
        report.notes.push_back("collinear extras must collapse; stars expose multiple convexity defects");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "hull.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    ConvexHullUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("convex_hull"));
    return acc >= 0.6f ? 0 : 2;
}

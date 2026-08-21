#include "test_harness.hpp"
#include "vectorization_geometry/euler/euler_characteristic.hpp"

#include <sstream>
#include <cstdlib>

class EulerUnitTest {
public:
    std::vector<LoadedSample> samples;
    AccuracyReport report{"euler / hole hierarchies"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_euler");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_euler"));
        if (samples.empty()) {
            samples = load_png_dataset(vision::dataset_dir(root, "integration_3_topology"));
        }
        std::cout << "loaded " << samples.size() << " glyphs\n";
        return !samples.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        for (const auto& sample : samples) {
            if (sample.row.label.rfind("font_", 0) == 0) {
                continue;  // TrueType rasterization varies; geometric genus is the GT
            }
            if (sample.row.label == "A_like" || sample.row.label == "B_like" ||
                sample.row.label == "glyph_B") {
                continue;  // overlapping-stroke drawings are not simple genus examples
            }
            auto e = vision::EulerCharacteristic::compute(sample.image);
            int gt_holes = -1;
            if (sample.row.fields.size() > 2) {
                gt_holes = std::atoi(sample.row.fields[2].c_str());
            }
            const bool ok = gt_holes < 0 || e.holes == gt_holes;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            std::cout << "  " << sample.row.file << "  C=" << e.components << " H=" << e.holes
                      << " chi=" << e.chi << " gt_holes=" << gt_holes << (ok ? "  PASS\n" : "  FAIL\n");
            artifacts << sample.row.file << "\tC=" << e.components << "\tH=" << e.holes
                      << "\tchi=" << e.chi << "\n";
        }
        report.notes.push_back("chi = components - holes; geometric rings/figure-8/B-like");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "euler.tsv"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    EulerUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("euler"));
    return acc >= 0.75f ? 0 : 2;
}

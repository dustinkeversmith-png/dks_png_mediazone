#include "test_harness.hpp"
#include "vectorization_geometry/measurements/turning_function/turning_function.hpp"

#include <limits>
#include <sstream>

class TurningFunctionUnitTest {
public:
    std::vector<LoadedSample> samples;
    std::vector<std::vector<float>> feats;
    AccuracyReport report{"turning_function / Swedish-leaf analogue"};
    std::ostringstream artifacts;
    vision::TurningFunction engine;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_turning");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_turning"));
        if (samples.size() < 16) {
            samples = load_png_dataset(vision::dataset_dir(root, "integration_3_leaf"));
        }
        std::cout << "loaded " << samples.size() << " leaf/kimia shapes\n";
        return samples.size() >= 8;
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        feats.resize(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            feats[i] = engine.compute(samples[i].image);
        }
        for (size_t i = 0; i < samples.size(); ++i) {
            if (samples[i].row.label.find('/') != std::string::npos) {
                continue;
            }
            float best = std::numeric_limits<float>::max();
            size_t arg = i;
            for (size_t j = 0; j < samples.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const float d = vision::TurningFunction::circular_l2(feats[i], feats[j]);
                if (d < best) {
                    best = d;
                    arg = j;
                }
            }
            const bool ok = samples[i].row.label == samples[arg].row.label;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            artifacts << samples[i].row.file << " pred=" << samples[arg].row.label
                      << " true=" << samples[i].row.label << " d=" << best << "\n";
        }
        report.notes.push_back("1-NN on circular-shifted turning functions");
        std::cout << "evaluated " << report.total << " turning-function queries\n";
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "turning_nn.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    TurningFunctionUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("turning_function"));
    return acc >= 0.35f ? 0 : 2;
}

#include "test_harness.hpp"
#include "featurizations/fourier/fourier_descriptors.hpp"

#include <limits>
#include <sstream>

class FourierDescriptorUnitTest {
public:
    std::vector<LoadedSample> samples;
    std::vector<std::vector<float>> descriptors;
    AccuracyReport report{"fourier_descriptors / MPEG-7-like 1-NN"};
    std::ostringstream artifacts;
    vision::FourierDescriptors engine;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_fourier (+ integration_1_shapes fallback)");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_fourier"));
        if (samples.size() < 8) {
            samples = load_png_dataset(vision::dataset_dir(root, "integration_1_shapes"));
        }
        std::cout << "loaded " << samples.size() << " silhouettes\n";
        return samples.size() >= 8;
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        descriptors.resize(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            descriptors[i] = engine.compute(samples[i].image);
        }
        int correct = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            float best = std::numeric_limits<float>::max();
            size_t arg = i;
            for (size_t j = 0; j < samples.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const float d = vision::FourierDescriptors::l2(descriptors[i], descriptors[j]);
                if (d < best) {
                    best = d;
                    arg = j;
                }
            }
            const bool ok = samples[i].row.label == samples[arg].row.label;
            if (ok) {
                ++correct;
            }
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            artifacts << samples[i].row.file << " pred=" << samples[arg].row.label
                      << " true=" << samples[i].row.label << " dist=" << best << "\n";
        }
        report.notes.push_back("leave-one-out 1-NN on rotation/scale-invariant Fourier magnitudes");
        std::cout << "1-NN matches: " << correct << " / " << samples.size() << "\n";
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "nn_pairs.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    FourierDescriptorUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        std::cerr << "dataset missing. Run: python data/vision/download_vision_datasets.py\n";
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("fourier_descriptors"));
    return acc >= 0.5f ? 0 : 2;
}

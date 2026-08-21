#include "test_harness.hpp"
#include "featurizations/fourier/fourier_descriptors.hpp"
#include "spatial_trees/lsh/fourier_lsh.hpp"

#include <algorithm>
#include <sstream>

class LshUnitTest {
public:
    std::vector<LoadedSample> samples;
    std::vector<std::vector<float>> feats;
    AccuracyReport report{"lsh / Fourier icon hashing"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_lsh");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_lsh"));
        if (samples.size() < 16) {
            samples = load_png_dataset(vision::dataset_dir(root, "unit_fourier"));
        }
        std::cout << "loaded " << samples.size() << " icon silhouettes\n";
        return samples.size() >= 16;
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        vision::FourierDescriptors fd;
        feats.resize(samples.size());
        vision::FourierLSH lsh(16, 16);
        for (size_t i = 0; i < samples.size(); ++i) {
            feats[i] = fd.compute(samples[i].image);
            lsh.insert(static_cast<int>(i), feats[i]);
        }
        const int k = 5;
        int hits = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            auto exact = vision::FourierLSH::exact_nn(feats, feats[i], k + 1);
            auto approx = lsh.query(feats[i], k + 1);
            // drop self
            exact.erase(std::remove(exact.begin(), exact.end(), static_cast<int>(i)), exact.end());
            approx.erase(std::remove(approx.begin(), approx.end(), static_cast<int>(i)), approx.end());
            bool found = false;
            if (!exact.empty()) {
                for (int id : approx) {
                    if (id == exact.front()) {
                        found = true;
                        break;
                    }
                }
            }
            ++report.total;
            if (found) {
                ++report.passed;
                ++hits;
            }
            artifacts << samples[i].row.file << " exact=" << (exact.empty() ? -1 : exact.front())
                      << " found=" << found << "\n";
        }
        std::cout << "recall@1 vs exact NN: " << hits << " / " << samples.size() << "\n";
        report.notes.push_back("LSH bucket + Hamming-1 probe should recover exact nearest neighbor");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "lsh_recall.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    LshUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("lsh"));
    return acc >= 0.3f ? 0 : 2;
}

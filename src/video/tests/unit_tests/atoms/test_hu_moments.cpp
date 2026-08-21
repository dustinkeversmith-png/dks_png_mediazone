#include "test_harness.hpp"
#include "featurizations/hu_moments/hu_moments.hpp"

#include <map>
#include <sstream>
#include <limits>

class HuMomentsUnitTest {
public:
    std::vector<LoadedSample> samples;
    std::vector<std::vector<float>> feats;
    AccuracyReport report{"hu_moments / RST invariance"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_hu_moments");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_hu_moments"));
        std::cout << "loaded " << samples.size() << " rotated/scaled glyphs\n";
        return samples.size() >= 8;
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        feats.resize(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            auto r = vision::HuMoments::compute(samples[i].image);
            feats[i] = vision::HuMoments::log_abs_vector(r);
            std::cout << "  " << samples[i].row.file << "  hu1=" << r.hu[0]
                      << " hu2=" << r.hu[1] << " area=" << r.area << "\n";
        }
        for (size_t i = 0; i < samples.size(); ++i) {
            float same = std::numeric_limits<float>::max();
            float other = std::numeric_limits<float>::max();
            for (size_t j = 0; j < samples.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const float d = vision::HuMoments::l2(feats[i], feats[j]);
                if (samples[i].row.label == samples[j].row.label) {
                    same = std::min(same, d);
                } else {
                    other = std::min(other, d);
                }
            }
            const bool ok = same < other;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            artifacts << samples[i].row.file << " same=" << same << " other=" << other
                      << (ok ? " PASS\n" : " FAIL\n");
        }
        report.notes.push_back("nearest same-glyph transform should beat nearest different glyph");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "invariance.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    HuMomentsUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("hu_moments"));
    return acc >= 0.6f ? 0 : 2;
}

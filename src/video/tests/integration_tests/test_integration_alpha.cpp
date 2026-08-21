#include "test_harness.hpp"
#include "featurizations/hu_moments/hu_moments.hpp"
#include "math/distance_transform.hpp"
#include "spatial_trees/vp_tree/vp_tree.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <vector>

class IntegrationAlphaTest {
public:
    std::vector<LoadedSample> samples;
    std::vector<std::vector<float>> feats;
    AccuracyReport report{"integration_alpha / Hu + SDF -> VP-tree"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: integration_1_shapes");
        samples = load_png_dataset(vision::dataset_dir(root, "integration_1_shapes"));
        std::cout << "loaded " << samples.size() << " MPEG-7/Kimia-style silhouettes\n";
        return samples.size() >= 16;
    }

    std::vector<float> profile(const vision::GrayImage& image) {
        auto hu = vision::HuMoments::log_abs_vector(vision::HuMoments::compute(image));
        auto sdf = vision::FelzenszwalbDistanceTransform::signed_distance(image);
        std::vector<float> hist(8, 0.0f);
        for (float v : sdf) {
            int b = static_cast<int>((v + 40.0f) / 10.0f);
            b = std::clamp(b, 0, 7);
            hist[static_cast<size_t>(b)] += 1.0f;
        }
        const float n = static_cast<float>(std::max<size_t>(1, sdf.size()));
        for (float& h : hist) {
            h /= n;
        }
        hu.insert(hu.end(), hist.begin(), hist.end());
        return hu;
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        feats.resize(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            feats[i] = profile(samples[i].image);
        }

        vision::VPTree<int> tree;
        std::vector<int> ids(samples.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<int>(i);
        }
        tree.build(ids, [&](const int& a, const int& b) {
            return vision::HuMoments::l2(feats[static_cast<size_t>(a)], feats[static_cast<size_t>(b)]);
        });

        double query_ms = 0.0;
        int queries = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            float best = std::numeric_limits<float>::max();
            int arg = -1;
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t j = 0; j < samples.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const float d = vision::HuMoments::l2(feats[i], feats[j]);
                if (d < best) {
                    best = d;
                    arg = static_cast<int>(j);
                }
            }
            float td = 0;
            tree.nearest(static_cast<int>(i), &td);
            const auto t1 = std::chrono::steady_clock::now();
            query_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++queries;
            const bool ok = arg >= 0 && samples[static_cast<size_t>(i)].row.label ==
                                            samples[static_cast<size_t>(arg)].row.label;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            artifacts << samples[i].row.file << " pred=" << (arg < 0 ? "?" : samples[static_cast<size_t>(arg)].row.label)
                      << " true=" << samples[i].row.label << "\n";
        }
        const double per = queries ? query_ms / queries : 0.0;
        report.notes.push_back("Hu(7)+SDF histogram into VP-tree; leave-one-out class 1-NN");
        std::cout << "mean query time: " << per << " ms  (target: sub-millisecond on this subset)\n";
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "alpha_nn.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    IntegrationAlphaTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("integration_alpha"));
    return acc >= 0.45f ? 0 : 2;
}

#include "test_harness.hpp"
#include "boundary_tracing/moore_neighbor.hpp"
#include "spatial_trees/vp_tree/vp_tree.hpp"

#include <sstream>
#include <limits>

class VpTreeUnitTest {
public:
    std::vector<LoadedSample> samples;
    std::vector<std::vector<vision::Vec2>> contours;
    AccuracyReport report{"vp_tree / Chamfer NN"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_vptree");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_vptree"));
        if (samples.size() < 10) {
            samples = load_png_dataset(vision::dataset_dir(root, "unit_fourier"));
        }
        std::cout << "loaded " << samples.size() << " binary contours\n";
        return samples.size() >= 8;
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        contours.resize(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            auto c = vision::MooreNeighborTracer::trace(samples[i].image);
            contours[i] = vision::MooreNeighborTracer::resample(c.points, 24);
        }

        vision::VPTree<int> tree;
        std::vector<int> ids(samples.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<int>(i);
        }
        tree.build(ids, [&](const int& a, const int& b) {
            return vision::VPTree<int>::chamfer(contours[static_cast<size_t>(a)],
                                                contours[static_cast<size_t>(b)]);
        });

        int nn_ok = 0;
        int class_ok = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            float brute_d = std::numeric_limits<float>::max();
            int brute = -1;
            for (size_t j = 0; j < samples.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const float d = vision::VPTree<int>::chamfer(contours[i], contours[j]);
                if (d < brute_d) {
                    brute_d = d;
                    brute = static_cast<int>(j);
                }
            }
            // query by temporarily using a copy metric against stored ids — nearest including self
            float td = 0;
            const int got = tree.nearest(static_cast<int>(i), &td);
            // self should be nearest; second check: brute class match
            const bool self_ok = got == static_cast<int>(i);
            const bool cls = brute >= 0 && samples[static_cast<size_t>(i)].row.label ==
                                               samples[static_cast<size_t>(brute)].row.label;
            ++report.total;
            if (self_ok) {
                ++report.passed;
                ++nn_ok;
            }
            if (cls) {
                ++class_ok;
            }
            artifacts << samples[i].row.file << " vp=" << got << " brute_nn=" << brute
                      << " class_match=" << cls << "\n";
        }
        std::cout << "VP-tree self-NN: " << nn_ok << " / " << samples.size()
                  << "  brute class-NN: " << class_ok << " / " << samples.size() << "\n";
        report.notes.push_back("VP-tree exact self retrieval under Chamfer; brute 1-NN class noted");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "vptree.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    VpTreeUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("vp_tree"));
    return acc >= 0.99f ? 0 : 2;
}

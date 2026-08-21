#include "test_harness.hpp"
#include "vectorization_geometry/medial_axis/medial_axis.hpp"

#include <sstream>

class MedialAxisUnitTest {
public:
    std::vector<LoadedSample> samples;
    AccuracyReport report{"medial_axis / FH EDT ridges"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_medial_axis");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_medial_axis"));
        std::cout << "loaded " << samples.size() << " articulated silhouettes\n";
        return !samples.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        int hand_skel = 0, blob_skel = 0, n_hand = 0, n_blob = 0;
        for (const auto& sample : samples) {
            auto m = vision::MedialAxis::extract(sample.image);
            const bool has_skel = m.skeleton_pixels > 0;
            ++report.total;
            if (has_skel) {
                ++report.passed;
            }
            if (sample.row.label == "hand") {
                hand_skel += m.skeleton_pixels;
                ++n_hand;
            } else if (sample.row.label == "disk" || sample.row.label == "ellipse" ||
                       sample.row.label == "square") {
                blob_skel += m.skeleton_pixels;
                ++n_blob;
            }
            std::cout << "  " << sample.row.file << "  skel=" << m.skeleton_pixels
                      << "  mean_r=" << m.mean_radius << "\n";
            artifacts << sample.row.file << "\t" << m.skeleton_pixels << "\t" << m.mean_radius << "\n";
        }
        const float hand_mean = n_hand ? static_cast<float>(hand_skel) / n_hand : 0.0f;
        const float blob_mean = n_blob ? static_cast<float>(blob_skel) / n_blob : 0.0f;
        report.notes.push_back("hands should retain more skeleton pixels than compact blobs");
        std::cout << "mean skeleton pixels  hands=" << hand_mean << " blobs=" << blob_mean << "\n";
        if (!samples.empty()) {
            auto m = vision::MedialAxis::extract(samples.front().image);
            last_skel_ = std::move(m.skeleton);
        }
    }

    vision::GrayImage last_skel_;

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "skeleton.tsv"), artifacts.str());
        if (!last_skel_.empty()) {
            vision::save_pgm(vision::join_path(dir, "skeleton.pgm"), last_skel_);
        }
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    MedialAxisUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("medial_axis"));
    return acc >= 0.7f ? 0 : 2;
}

#include "test_harness.hpp"
#include "featurizations/rbf/thin_plate_spline.hpp"
#include "boundary_tracing/moore_neighbor.hpp"

#include <sstream>

class RbfUnitTest {
public:
    std::vector<LoadedSample> samples;
    AccuracyReport report{"rbf / thin-plate spline hands"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_rbf");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_rbf"));
        std::cout << "loaded " << samples.size() << " articulated hand masks\n";
        return !samples.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        for (const auto& sample : samples) {
            auto contour = vision::MooreNeighborTracer::trace(sample.image);
            vision::ThinPlateSplineRBF rbf;
            const bool fitted = rbf.fit_identity(contour.points, 28);
            const float rmse = fitted ? rbf.reconstruction_rmse() : 1.0e9f;
            const bool ok = fitted && rmse < 8.0f;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            std::cout << "  " << sample.row.file << "  contour=" << contour.points.size()
                      << "  centers=" << rbf.centers.size() << "  rmse=" << rmse
                      << (ok ? "  PASS\n" : "  FAIL\n");
            artifacts << sample.row.file << "\trmse=" << rmse << "\n";
        }
        report.notes.push_back("TPS identity fit on subsampled contour; RMSE in pixels");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "rbf_rmse.tsv"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    RbfUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("rbf"));
    return acc >= 0.7f ? 0 : 2;
}

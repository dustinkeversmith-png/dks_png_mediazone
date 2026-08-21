#include "test_harness.hpp"
#include "boundary_tracing/moore_neighbor.hpp"
#include "vectorization_geometry/euler/euler_characteristic.hpp"

#include <sstream>

class BoundaryTracingUnitTest {
public:
    std::vector<LoadedSample> samples;
    AccuracyReport report{"boundary_tracing / Moore-neighbor"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_contour");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_contour"));
        std::cout << "loaded " << samples.size() << " raster grids from unit_contour\n";
        return !samples.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        for (const auto& sample : samples) {
            auto contour = vision::MooreNeighborTracer::trace(sample.image);
            auto euler = vision::EulerCharacteristic::compute(sample.image);
            const int pixel_area = sample.image.count_fg();
            const float area_err = pixel_area > 0
                                       ? std::fabs(contour.area - static_cast<float>(pixel_area)) /
                                             static_cast<float>(pixel_area)
                                       : 1.0f;

            bool ok = contour.closed && contour.points.size() >= 8;
            if (sample.row.label == "disk") {
                const float r = sample.row.fields.size() > 3 ? std::strtof(sample.row.fields[3].c_str(), nullptr) : 70.0f;
                const float expected = vision::kPi * r * r;
                const float disk_err = std::fabs(contour.area - expected) / expected;
                ok = ok && disk_err < 0.12f;
                artifacts << sample.row.file << " disk area_err=" << disk_err << " closed=" << contour.closed << "\n";
            } else if (sample.row.label == "donut" || sample.row.label == "nested") {
                ok = ok && euler.holes >= 1;
                artifacts << sample.row.file << " holes=" << euler.holes << " chi=" << euler.chi << "\n";
            } else {
                ok = ok && area_err < 0.20f;
                artifacts << sample.row.file << " verts=" << contour.points.size()
                          << " area_err=" << area_err << "\n";
            }
            std::cout << "  " << sample.row.file << "  closed=" << contour.closed
                      << "  verts=" << contour.points.size()
                      << "  area=" << contour.area
                      << "  perim=" << contour.perimeter
                      << "  holes=" << euler.holes
                      << (ok ? "  PASS\n" : "  FAIL\n");
            ++report.total;
            if (ok) {
                ++report.passed;
            }
        }
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "summary.txt"), artifacts.str());
        if (!samples.empty()) {
            auto c = vision::MooreNeighborTracer::trace(samples.front().image);
            vision::GrayImage vis = samples.front().image;
            for (const auto& p : c.points) {
                const int x = static_cast<int>(p.x);
                const int y = static_cast<int>(p.y);
                if (x >= 0 && y >= 0 && x < vis.width && y < vis.height) {
                    vis.at(x, y) = 128;
                }
            }
            vision::save_pgm(vision::join_path(dir, "contour_overlay.pgm"), vis);
        }
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    BoundaryTracingUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        std::cerr << "dataset missing. Run: python data/vision/download_vision_datasets.py\n";
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("boundary_tracing"));
    return acc >= 0.7f ? 0 : 2;
}

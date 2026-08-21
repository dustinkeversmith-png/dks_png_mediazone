#include "test_harness.hpp"
#include "screen_detection/CCL/connected_components.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <algorithm>

struct GtBox {
    std::string file;
    std::string type;
    vision::Rect box;
};

class CclUnitTest {
public:
    std::string folder;
    std::vector<LoadedSample> samples;
    std::vector<GtBox> boxes;
    AccuracyReport report{"CCL / SAUF document+UI layouts"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_ccl");
        folder = vision::dataset_dir(root, "unit_ccl");
        samples = load_png_dataset(folder);
        std::ifstream in(vision::join_path(folder, "boxes.tsv"));
        std::string line;
        bool header = true;
        while (std::getline(in, line)) {
            if (header) {
                header = false;
                continue;
            }
            std::istringstream ss(line);
            GtBox b;
            std::string xs, ys, ws, hs;
            if (!(ss >> b.file >> b.type >> b.box.x >> b.box.y >> b.box.w >> b.box.h)) {
                continue;
            }
            boxes.push_back(b);
        }
        std::cout << "loaded " << samples.size() << " screens, " << boxes.size() << " GT boxes\n";
        return !samples.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        for (const auto& sample : samples) {
            auto ccl = vision::ConnectedComponentLabeler::label(sample.image);
            std::vector<GtBox> gt;
            for (const auto& b : boxes) {
                if (b.file == sample.row.file) {
                    gt.push_back(b);
                }
            }
            int matched = 0;
            for (const auto& g : gt) {
                float best = 0.0f;
                for (const auto& c : ccl.components) {
                    best = std::max(best, c.bbox.iou(g.box));
                }
                const bool ok = best >= 0.45f;
                if (ok) {
                    ++matched;
                }
                ++report.total;
                if (ok) {
                    ++report.passed;
                }
            }
            std::cout << "  " << sample.row.file << "  components=" << ccl.components.size()
                      << "  gt=" << gt.size() << "  matched=" << matched << "\n";
            artifacts << sample.row.file << "\tcomp=" << ccl.components.size() << "\tmatched=" << matched << "\n";
        }
        report.notes.push_back("IoU>=0.45 of any CCL AABB vs GT xywh (buttons/cards/app_bar)");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "ccl_matches.tsv"), artifacts.str());
        if (!samples.empty()) {
            auto ccl = vision::ConnectedComponentLabeler::label(samples.front().image);
            vision::GrayImage vis = samples.front().image;
            for (const auto& c : ccl.components) {
                const int x0 = static_cast<int>(c.bbox.x);
                const int y0 = static_cast<int>(c.bbox.y);
                const int x1 = static_cast<int>(c.bbox.x1());
                const int y1 = static_cast<int>(c.bbox.y1());
                for (int x = x0; x < x1 && x < vis.width; ++x) {
                    if (y0 >= 0 && y0 < vis.height) vis.at(x, y0) = 128;
                    if (y1 - 1 >= 0 && y1 - 1 < vis.height) vis.at(x, y1 - 1) = 128;
                }
            }
            vision::save_pgm(vision::join_path(dir, "ccl_boxes.pgm"), vis);
        }
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    CclUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("ccl"));
    return acc >= 0.5f ? 0 : 2;
}

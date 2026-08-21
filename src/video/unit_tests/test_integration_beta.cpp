#include "test_harness.hpp"
#include "screen_detection/CCL/connected_components.hpp"
#include "screen_detection/LOOKUP/template_ncc.hpp"

#include <chrono>
#include <fstream>
#include <sstream>
#include <map>

struct GtBox {
    std::string file;
    std::string type;
    vision::Rect box;
};

class IntegrationBetaTest {
public:
    std::string folder;
    std::vector<LoadedSample> samples;
    std::vector<GtBox> boxes;
    AccuracyReport report{"integration_beta / RICO CCL + NCC"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: integration_2_rico_ui");
        folder = vision::dataset_dir(root, "integration_2_rico_ui");
        samples = load_png_dataset(folder);
        for (const auto& row : vision::load_tsv(vision::join_path(folder, "boxes.tsv"))) {
            if (row.fields.size() < 6) {
                continue;
            }
            GtBox b;
            b.file = row.fields[0];
            b.type = row.fields[1];
            b.box.x = std::strtof(row.fields[2].c_str(), nullptr);
            b.box.y = std::strtof(row.fields[3].c_str(), nullptr);
            b.box.w = std::strtof(row.fields[4].c_str(), nullptr);
            b.box.h = std::strtof(row.fields[5].c_str(), nullptr);
            boxes.push_back(b);
        }
        std::cout << "loaded " << samples.size() << " UI screens, " << boxes.size() << " GT boxes\n";
        return !samples.empty() && !boxes.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        int fp = 0;
        int controls = 0;
        for (const auto& sample : samples) {
            const auto t0 = std::chrono::steady_clock::now();
            auto ccl = vision::ConnectedComponentLabeler::label(sample.image);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::vector<GtBox> gt;
            for (const auto& b : boxes) {
                if (b.file == sample.row.file &&
                    (b.type == "button" || b.type == "checkbox" || b.type == "search_input" ||
                     b.type == "input")) {
                    gt.push_back(b);
                }
            }
            for (const auto& g : gt) {
                ++controls;
                float best = 0.0f;
                for (const auto& c : ccl.components) {
                    best = std::max(best, c.bbox.iou(g.box));
                }
                const float ncc = vision::TemplateNCC::ncc_at_box(sample.image, g.box);
                const bool ok = best >= 0.5f && ncc > 0.99f;
                ++report.total;
                if (ok) {
                    ++report.passed;
                } else {
                    ++fp;
                }
                artifacts << sample.row.file << "\t" << g.type << "\tiou=" << best << "\tncc=" << ncc << "\n";
            }
            std::cout << "  " << sample.row.file << "  ccl_ms=" << ms
                      << "  components=" << ccl.components.size() << "  controls=" << gt.size() << "\n";
        }
        report.notes.push_back("0% raster deformation: GT control boxes must NCC~1 and match a CCL AABB");
        std::cout << "control mismatches: " << fp << " / " << controls << "\n";
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "beta_controls.tsv"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    IntegrationBetaTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("integration_beta"));
    return acc >= 0.7f ? 0 : 2;
}

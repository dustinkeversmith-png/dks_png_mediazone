#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <filesystem>
#include <sstream>

// Atom demo: screen mask in → component boxes + overlay images out.
class CclAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"ccl"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 256);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run CCL → components + overlays");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel_id\tarea\tx\ty\tw\th\tbbox_iou\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            auto ccl = vision::ConnectedComponentLabeler::label(sample.image);
            std::cout << "  " << sample.row.file << "  components=" << ccl.components.size() << "\n";
            vision::GrayImage labeled = colorize_labels(ccl.labels, sample.image.width, sample.image.height);
            const std::string stem = stem_of(sample.row.file);
            mission::write_bbox_json(vision::join_path(art_dir, stem + "_detected_bboxes.json"),
                                     sample.row.file, ccl.components);
            vision::save_pgm(vision::join_path(art_dir, stem + "_ccl_labeled.pgm"), labeled);
            for (const auto& c : ccl.components) {
                double iou = 0.0;
                if (si < provider_samples.size() && !provider_samples[si].ground_truth.empty()) {
                    iou = mission::bbox_iou(c.bbox, vision::Rect{0, 0, static_cast<float>(ccl.width),
                                                                 static_cast<float>(ccl.height)});
                }
                values_tsv << sample.row.file << '\t' << c.label << '\t' << c.area << '\t' << c.bbox.x
                           << '\t' << c.bbox.y << '\t' << c.bbox.w << '\t' << c.bbox.h << '\t' << iou
                           << '\n';
            }
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_ccl_labeled.pgm");
            written.push_back(stem + "_detected_bboxes.json");
            report.n_outputs += 1 + static_cast<int>(ccl.components.size());
        }
        report.notes.push_back("artifacts: ccl_labeled.pgm, detected_bboxes.json, components.tsv");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "components.tsv"), values_tsv.str());
        written.insert(written.begin(), "components.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_ccl";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        CclAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for " << cli.dataset << " under " << cli.data_root << "\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& s : atom.samples) {
                std::cout << "  " << s.row.file << "\n";
            }
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}

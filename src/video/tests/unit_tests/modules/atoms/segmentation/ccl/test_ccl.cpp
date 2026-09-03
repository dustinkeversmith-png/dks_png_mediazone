#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <sstream>

// Atom: SBD multi-instance masks → 8-connected CCL label map + component stats.
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

    // In-test prep: any positive instance label → binary FG (preserve multi-blob).
    static vision::GrayImage to_multi_instance_binary(const vision::GrayImage& src) {
        vision::GrayImage out = src;
        for (uint8_t& p : out.data) {
            p = p > 0 ? 255 : 0;
        }
        return out;
    }

    void run(const std::string& art_dir) {
        print_banner("run CCL → components + overlays");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel_id\tarea\tcx\tcy\tx\ty\tw\th\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto mask_img = to_multi_instance_binary(mission_mask_image(ps, sample.image));
            const auto luma = mission_luma_image(ps, sample.image);
            auto ccl = vision::ConnectedComponentLabeler::label(mask_img);
            std::cout << "  " << sample.row.file << "  components=" << ccl.components.size() << "\n";
            vision::GrayImage labeled =
                colorize_labels(ccl.labels, sample.image.width, sample.image.height);
            // Prefer luma dims if mask was used as sample.image.
            if (labeled.width != mask_img.width || labeled.height != mask_img.height) {
                labeled = colorize_labels(ccl.labels, mask_img.width, mask_img.height);
            }
            const std::string stem = stem_of(sample.row.file);
            mission::write_bbox_json(vision::join_path(art_dir, stem + "_detected_bboxes.json"),
                                     sample.row.file, ccl.components);
            vision::save_pgm(vision::join_path(art_dir, stem + "_ccl_labeled.pgm"), labeled);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            for (const auto& c : ccl.components) {
                const float cx = c.bbox.x + 0.5f * c.bbox.w;
                const float cy = c.bbox.y + 0.5f * c.bbox.h;
                values_tsv << sample.row.file << '\t' << c.label << '\t' << c.area << '\t' << cx
                           << '\t' << cy << '\t' << c.bbox.x << '\t' << c.bbox.y << '\t' << c.bbox.w
                           << '\t' << c.bbox.h << '\n';
            }
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_ccl_labeled.pgm");
            written.push_back(stem + "_detected_bboxes.json");
            report.n_outputs += 1 + static_cast<int>(ccl.components.size());
        }
        report.notes.push_back("SBD: threshold>0 in test → multi-instance CCL");
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
    return run_atom_main(argc, argv, "sbd", [&](const AtomCli& cli) -> int {
        CclAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for ccl atom\n";
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

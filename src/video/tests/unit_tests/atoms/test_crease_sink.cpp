#include "test_harness.hpp"
#include "filters/crease_sink/crease_sink.hpp"

#include <sstream>

// Atom demo: image in → black top-hat / LoG valley / NMS crease spine out.
class CreaseSinkAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"crease_sink"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 160);
        }
        std::cout << "loaded " << samples.size() << " images (max side 160)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run crease/sink → tophat / valley / spine");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tspine_pixels\tmax_valley\n";
        contour::CreaseSink cs;
        for (const auto& sample : samples) {
            vision::GrayImage inv = sample.image;
            for (uint8_t& p : inv.pixels) {
                p = static_cast<uint8_t>(255 - p);
            }
            const auto src = to_contour(inv);
            const auto hat = cs.black_tophat(src);
            const auto valley = cs.valley_response(src);
            const auto spine = cs.detect(src);
            int nsp = 0;
            for (uint8_t p : spine.data) {
                nsp += p > 0 ? 1 : 0;
            }
            float vmax = 0;
            for (float v : valley.data) {
                vmax = std::max(vmax, v);
            }
            std::cout << "  " << sample.row.file << "  spine=" << nsp << "  max_valley=" << vmax << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << nsp << '\t' << vmax
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string inv_name = stem + "_inverted.pgm";
            const std::string hat_name = stem + "_tophat.pgm";
            const std::string val_name = stem + "_valley.pgm";
            const std::string sp_name = stem + "_spine.pgm";
            const std::string ov_name = stem + "_overlay.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, inv_name), inv);
            vision::save_pgm(vision::join_path(art_dir, hat_name), to_gray(hat));
            vision::save_pgm(vision::join_path(art_dir, val_name), field_to_gray(valley));
            vision::save_pgm(vision::join_path(art_dir, sp_name), to_gray(spine));
            vision::save_pgm(vision::join_path(art_dir, ov_name), overlay_mask(sample.image, to_gray(spine)));
            written.push_back(in_name);
            written.push_back(inv_name);
            written.push_back(hat_name);
            written.push_back(val_name);
            written.push_back(sp_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back(
            "stages: *_input.pgm, *_inverted.pgm, *_tophat.pgm, *_valley.pgm, *_spine.pgm, *_overlay.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "crease_sink.tsv"), values_tsv.str());
        written.insert(written.begin(), "crease_sink.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_crease_sink";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        CreaseSinkAtom atom;
        if (!atom.load(cli.data_root, cli.dataset, cli.sample_filter)) {
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

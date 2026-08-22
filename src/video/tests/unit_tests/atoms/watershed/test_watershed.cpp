#include "test_harness.hpp"
#include "filters/watershed/watershed.hpp"

#include <sstream>

// Atom demo: mask in → EDT markers → Meyer watershed basins / lines out.
class WatershedAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"watershed"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 128);
        }
        std::cout << "loaded " << samples.size() << " masks (max side 128)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run watershed → markers / basins / lines");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_basins\tn_watershed\n";
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const auto ws = contour::Watershed::from_mask(mask);
            std::cout << "  " << sample.row.file << "  basins=" << ws.n_basins
                      << "  lines=" << ws.n_watershed << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << ws.n_basins << '\t'
                       << ws.n_watershed << '\n';

            vision::GrayImage lines;
            lines.width = ws.width;
            lines.height = ws.height;
            lines.pixels.resize(static_cast<size_t>(ws.width * ws.height));
            for (size_t i = 0; i < lines.pixels.size(); ++i) {
                lines.pixels[i] = ws.labels[i] == 0 ? 255 : 0;
            }

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string mk_name = stem + "_markers.pgm";
            const std::string rf_name = stem + "_relief.pgm";
            const std::string lb_name = stem + "_basins.pgm";
            const std::string ln_name = stem + "_lines.pgm";
            const std::string ov_name = stem + "_overlay.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, mk_name),
                             colorize_labels(ws.markers, ws.width, ws.height));
            vision::save_pgm(vision::join_path(art_dir, rf_name), field_to_gray(ws.relief));
            vision::save_pgm(vision::join_path(art_dir, lb_name),
                             colorize_labels(ws.labels, ws.width, ws.height));
            vision::save_pgm(vision::join_path(art_dir, ln_name), lines);
            vision::save_pgm(vision::join_path(art_dir, ov_name), overlay_mask(sample.image, lines));
            written.push_back(in_name);
            written.push_back(mk_name);
            written.push_back(rf_name);
            written.push_back(lb_name);
            written.push_back(ln_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back(
            "stages: *_input.pgm, *_markers.pgm, *_relief.pgm, *_basins.pgm, *_lines.pgm, *_overlay.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "watershed.tsv"), values_tsv.str());
        written.insert(written.begin(), "watershed.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_watershed";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        WatershedAtom atom;
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

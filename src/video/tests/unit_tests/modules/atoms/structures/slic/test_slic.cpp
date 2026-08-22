#include "test_harness.hpp"
#include "graphs/slic/slic.hpp"

#include <sstream>

// Atom demo: image in → SLIC superpixels + Felzenszwalb grid-graph regions out.
class SlicAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"slic"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 128);
        }
        std::cout << "loaded " << samples.size() << " images (max side 128)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run SLIC + Felzenszwalb grid graph");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tslic_labels\tfh_labels\n";
        contour::SlicSuperpixels slic;
        slic.k = 48;
        slic.iterations = 6;
        contour::FelzenszwalbGridGraph fh;
        fh.k_scale = 60.0f;
        fh.min_size = 24.0f;
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const auto sl = slic.segment(im);
            const auto gs = fh.segment(im);
            std::cout << "  " << sample.row.file << "  slic=" << sl.n_labels << "  fh=" << gs.n_labels
                      << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << sl.n_labels << '\t'
                       << gs.n_labels << '\n';

            const auto slic_b = label_boundaries(sl.labels, sl.width, sl.height);
            const auto fh_b = label_boundaries(gs.labels, gs.width, gs.height);
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string sl_name = stem + "_slic.pgm";
            const std::string slb_name = stem + "_slic_bounds.pgm";
            const std::string fh_name = stem + "_felzenszwalb.pgm";
            const std::string fhb_name = stem + "_fh_bounds.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, sl_name),
                             colorize_labels(sl.labels, sl.width, sl.height));
            vision::save_pgm(vision::join_path(art_dir, slb_name), overlay_mask(sample.image, slic_b));
            vision::save_pgm(vision::join_path(art_dir, fh_name),
                             colorize_labels(gs.labels, gs.width, gs.height));
            vision::save_pgm(vision::join_path(art_dir, fhb_name), overlay_mask(sample.image, fh_b));
            written.push_back(in_name);
            written.push_back(sl_name);
            written.push_back(slb_name);
            written.push_back(fh_name);
            written.push_back(fhb_name);
            ++report.n_outputs;
        }
        report.notes.push_back(
            "stages: *_input.pgm, *_slic.pgm, *_slic_bounds.pgm, *_felzenszwalb.pgm, *_fh_bounds.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "slic.tsv"), values_tsv.str());
        written.insert(written.begin(), "slic.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_slic";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        SlicAtom atom;
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

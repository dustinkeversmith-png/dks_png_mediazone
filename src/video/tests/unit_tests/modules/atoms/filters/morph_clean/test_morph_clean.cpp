#include "test_harness.hpp"
#include "filters/morph_clean/morph_clean.hpp"

#include <sstream>

// Atom demo: mask in → morphological open/close/clean out.
class MorphCleanAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"morph_clean"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        std::cout << "loaded " << samples.size() << " masks\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    static int count_fg(const contour::ImageBuffer& im) {
        int n = 0;
        for (uint8_t p : im.data) {
            n += p > 127 ? 1 : 0;
        }
        return n;
    }

    void run(const std::string& art_dir) {
        print_banner("run MorphClean → open / close / clean");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tfg_in\tfg_open\tfg_close\tfg_clean\n";
        for (const auto& sample : samples) {
            const auto src = to_contour(sample.image);
            const auto opened = contour::MorphClean::opening(src);
            const auto closed = contour::MorphClean::closing(src);
            const auto cleaned = contour::MorphClean::clean(src);
            std::cout << "  " << sample.row.file << "  fg=" << count_fg(src)
                      << "  clean=" << count_fg(cleaned) << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << count_fg(src) << '\t'
                       << count_fg(opened) << '\t' << count_fg(closed) << '\t' << count_fg(cleaned)
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string o_name = stem + "_open.pgm";
            const std::string c_name = stem + "_close.pgm";
            const std::string k_name = stem + "_clean.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, o_name), to_gray(opened));
            vision::save_pgm(vision::join_path(art_dir, c_name), to_gray(closed));
            vision::save_pgm(vision::join_path(art_dir, k_name), to_gray(cleaned));
            written.push_back(in_name);
            written.push_back(o_name);
            written.push_back(c_name);
            written.push_back(k_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: morph_clean.tsv, *_input.pgm, *_open.pgm, *_close.pgm, *_clean.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "morph_clean.tsv"), values_tsv.str());
        written.insert(written.begin(), "morph_clean.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_morph_clean";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        MorphCleanAtom atom;
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

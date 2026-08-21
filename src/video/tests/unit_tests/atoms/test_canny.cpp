#include "test_harness.hpp"
#include "featurizations/edge/canny.hpp"

#include <sstream>

// Atom demo: gray in → Canny edge map out.
class CannyAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"canny"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        std::cout << "loaded " << samples.size() << " images\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Canny → edge PGMs");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tedge_pixels\tw\th\n";
        contour::Canny canny;
        for (const auto& sample : samples) {
            const auto edges = canny.detect(to_contour(sample.image));
            int n = 0;
            for (uint8_t p : edges.data) {
                n += p > 0 ? 1 : 0;
            }
            std::cout << "  " << sample.row.file << "  edges=" << n << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << n << '\t'
                       << sample.image.width << '\t' << sample.image.height << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string e_name = stem + "_edges.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, e_name), to_gray(edges));
            written.push_back(in_name);
            written.push_back(e_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: canny.tsv, *_input.pgm, *_edges.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "canny.tsv"), values_tsv.str());
        written.insert(written.begin(), "canny.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_canny";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        CannyAtom atom;
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

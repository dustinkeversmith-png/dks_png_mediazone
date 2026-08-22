#include "test_harness.hpp"
#include "geometrify/fourier_descriptors/fourier_descriptors.hpp"
#include "structures/lsh/fourier_lsh.hpp"

#include <iomanip>
#include <sstream>

// Atom demo: masks in → Fourier vectors + LSH bucket ids out.
class LshAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"lsh"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        if (samples.empty()) {
            samples = load_png_dataset(vision::dataset_dir(root, "unit_fourier"), sample_filter);
        }
        std::cout << "loaded " << samples.size() << " silhouettes\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Fourier LSH → codes + neighbor lists");
        ScopedTimer timer(&report.elapsed_ms);
        vision::FourierDescriptors fd;
        vision::FourierLSH lsh(16, 16);
        std::vector<std::vector<float>> feats(samples.size());
        values_tsv << "file\tlabel\tid\tn_neighbors\tneighbor_ids\n";
        for (size_t i = 0; i < samples.size(); ++i) {
            feats[i] = fd.compute(samples[i].image);
            lsh.insert(static_cast<int>(i), feats[i]);
        }
        for (size_t i = 0; i < samples.size(); ++i) {
            auto approx = lsh.query(feats[i], 6);
            std::cout << "  " << samples[i].row.file << "  neighbors=" << approx.size();
            if (!feats[i].empty()) {
                std::cout << "  f0=" << feats[i][0];
            }
            std::cout << "\n";
            values_tsv << samples[i].row.file << '\t' << samples[i].row.label << '\t' << i << '\t'
                       << approx.size() << '\t';
            for (size_t k = 0; k < approx.size(); ++k) {
                if (k) {
                    values_tsv << ',';
                }
                values_tsv << approx[k];
            }
            values_tsv << '\n';
            const std::string stem = stem_of(samples[i].row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), samples[i].image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: lsh_neighbors.tsv, *_input.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "lsh_neighbors.tsv"), values_tsv.str());
        written.insert(written.begin(), "lsh_neighbors.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_lsh";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        LshAtom atom;
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

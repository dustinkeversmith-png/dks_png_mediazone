#include "test_harness.hpp"
#include "featurizations/fourier/fourier_descriptors.hpp"

#include <iomanip>
#include <sstream>

// Atom demo: silhouette in → Fourier magnitude descriptor out.
class FourierAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"fourier_descriptors"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;
    vision::FourierDescriptors engine;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        std::cout << "loaded " << samples.size() << " silhouettes\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Fourier descriptors → vector TSV + masks");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tdim";
        // header filled after first compute
        bool header_dims = false;
        for (const auto& sample : samples) {
            auto desc = engine.compute(sample.image);
            if (!header_dims) {
                for (size_t k = 0; k < desc.size(); ++k) {
                    values_tsv << "\tf" << k;
                }
                values_tsv << "\n";
                header_dims = true;
            }
            std::cout << "  " << sample.row.file << "  dim=" << desc.size();
            if (!desc.empty()) {
                std::cout << "  f0=" << desc[0];
            }
            if (desc.size() > 1) {
                std::cout << "  f1=" << desc[1];
            }
            std::cout << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << desc.size();
            values_tsv << std::scientific << std::setprecision(6);
            for (float v : desc) {
                values_tsv << '\t' << v;
            }
            values_tsv << '\n';
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: fourier_descriptors.tsv, *_input.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "fourier_descriptors.tsv"), values_tsv.str());
        written.insert(written.begin(), "fourier_descriptors.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_fourier";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        FourierAtom atom;
        if (!atom.load(cli.data_root, cli.dataset, cli.sample_filter)) {
            std::cerr << "no inputs for " << cli.dataset << " under " << cli.data_root << "\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& s : atom.samples) {
                std::cout << "  " << s.row.file << "\t" << s.row.label << "\n";
            }
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}

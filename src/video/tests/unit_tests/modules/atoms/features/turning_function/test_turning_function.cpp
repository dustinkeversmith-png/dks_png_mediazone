#include "test_harness.hpp"
#include "vectorization_geometry/measurements/turning_function/turning_function.hpp"

#include <iomanip>
#include <sstream>

// Atom demo: mask in → turning-function samples out.
class TurningAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"turning_function"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;
    vision::TurningFunction engine;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        std::cout << "loaded " << samples.size() << " silhouettes\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run turning function → Θ(s) vectors");
        ScopedTimer timer(&report.elapsed_ms);
        bool header = false;
        for (const auto& sample : samples) {
            auto feat = engine.compute(sample.image);
            if (!header) {
                values_tsv << "file\tlabel\tdim";
                for (size_t k = 0; k < feat.size(); ++k) {
                    values_tsv << "\tt" << k;
                }
                values_tsv << "\n";
                header = true;
            }
            std::cout << "  " << sample.row.file << "  dim=" << feat.size();
            if (!feat.empty()) {
                std::cout << "  t0=" << feat[0];
            }
            std::cout << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << feat.size();
            values_tsv << std::fixed << std::setprecision(6);
            for (float v : feat) {
                values_tsv << '\t' << v;
            }
            values_tsv << '\n';
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: turning_function.tsv, *_input.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "turning_function.tsv"), values_tsv.str());
        written.insert(written.begin(), "turning_function.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_turning";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        TurningAtom atom;
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

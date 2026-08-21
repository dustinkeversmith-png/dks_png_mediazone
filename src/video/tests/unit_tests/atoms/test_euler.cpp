#include "test_harness.hpp"
#include "vectorization_geometry/euler/euler_characteristic.hpp"

#include <sstream>

// Atom demo: mask in → components / holes / chi out.
class EulerAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"euler"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        std::cout << "loaded " << samples.size() << " masks\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Euler characteristic → topology TSV + masks");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tcomponents\tholes\tchi\tw\th\n";
        for (const auto& sample : samples) {
            auto e = vision::EulerCharacteristic::compute(sample.image);
            std::cout << "  " << sample.row.file << "  C=" << e.components << "  H=" << e.holes
                      << "  chi=" << e.chi << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << e.components << '\t'
                       << e.holes << '\t' << e.chi << '\t' << sample.image.width << '\t'
                       << sample.image.height << '\n';
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: euler.tsv (C, H, chi), *_input.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "euler.tsv"), values_tsv.str());
        written.insert(written.begin(), "euler.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_euler";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        EulerAtom atom;
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

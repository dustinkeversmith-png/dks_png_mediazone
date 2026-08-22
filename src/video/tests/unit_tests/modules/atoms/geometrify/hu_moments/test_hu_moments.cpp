#include "test_harness.hpp"
#include "featurizations/hu_moments/hu_moments.hpp"

#include <iomanip>
#include <sstream>

// Atom demo: mask in → Hu / Flusser values + images out. No GT scoring.
class HuMomentsAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"hu_moments"};
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
        print_banner("run hu_moments → values + images");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tw\th\tarea\tcx\tcy\thu1\thu2\thu3\thu4\thu5\thu6\thu7"
                      "\tflusser1\tflusser2\tflusser3\tflusser4\n";
        values_tsv << std::scientific << std::setprecision(8);
        for (const auto& sample : samples) {
            const auto r = vision::HuMoments::compute(sample.image);
            std::cout << "  " << sample.row.file << "  area=" << r.area << "  cx=" << r.cx
                      << "  cy=" << r.cy << "\n";
            std::cout << "      hu=[" << r.hu[0] << ", " << r.hu[1] << ", " << r.hu[2] << ", "
                      << r.hu[3] << ", " << r.hu[4] << ", " << r.hu[5] << ", " << r.hu[6] << "]\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << sample.image.width
                       << '\t' << sample.image.height << '\t' << r.area << '\t' << r.cx << '\t'
                       << r.cy;
            for (double h : r.hu) {
                values_tsv << '\t' << h;
            }
            for (double f : r.flusser) {
                values_tsv << '\t' << f;
            }
            values_tsv << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: hu_moments.tsv (7 Hu + 4 Flusser), *_input.pgm masks");
        report.notes.push_back("GT / invariance scoring belongs in benchmarks, not this atom");
    }

    void write(const std::string& dir) {
        const std::string tsv = "hu_moments.tsv";
        vision::write_text_file(vision::join_path(dir, tsv), values_tsv.str());
        written.insert(written.begin(), tsv);
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_hu_moments";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        HuMomentsAtom atom;
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

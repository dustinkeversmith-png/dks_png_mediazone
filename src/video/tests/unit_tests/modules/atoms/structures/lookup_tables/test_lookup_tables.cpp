#include "test_harness.hpp"
#include "structures/lookup_tables/geometry_lut.hpp"

#include <sstream>

// Atom demo: mask in → aspect / compactness / area / perimeter out.
class LutAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"lookup_tables"};
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
        print_banner("run geometry LUT measure → stats TSV");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\taspect\tcompactness\tarea\tperimeter\n";
        for (const auto& sample : samples) {
            auto s = vision::GeometryLUT::measure(sample.image);
            std::cout << "  " << sample.row.file << "  aspect=" << s.aspect
                      << "  compact=" << s.compactness << "  area=" << s.area
                      << "  peri=" << s.perimeter << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << s.aspect << '\t'
                       << s.compactness << '\t' << s.area << '\t' << s.perimeter << '\n';
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: geometry_stats.tsv, *_input.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "geometry_stats.tsv"), values_tsv.str());
        written.insert(written.begin(), "geometry_stats.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_luts";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        LutAtom atom;
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

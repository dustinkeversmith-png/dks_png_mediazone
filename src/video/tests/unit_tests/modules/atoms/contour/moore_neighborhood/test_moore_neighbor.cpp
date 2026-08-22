#include "test_harness.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <sstream>

// Atom demo: mask in → Moore outer contour overlay + length/area stats out.
class MooreNeighborAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"moore_neighbor"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        std::cout << "loaded " << samples.size() << " masks\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Moore neighbor → contour overlay + stats");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_points\tclosed\tarea\tperimeter\n";
        for (const auto& sample : samples) {
            auto c = vision::MooreNeighborTracer::trace(sample.image);
            std::cout << "  " << sample.row.file << "  n=" << c.points.size()
                      << "  closed=" << (c.closed ? 1 : 0) << "  area=" << c.area << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << c.points.size() << '\t'
                       << (c.closed ? 1 : 0) << '\t' << c.area << '\t' << c.perimeter << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string ov_name = stem + "_contour.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, ov_name),
                             overlay_polyline(sample.image, c.points, c.closed));
            written.push_back(in_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: moore.tsv, *_input.pgm, *_contour.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "moore.tsv"), values_tsv.str());
        written.insert(written.begin(), "moore.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_moore_neighbor";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        MooreNeighborAtom atom;
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

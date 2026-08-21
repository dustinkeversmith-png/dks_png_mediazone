#include "test_harness.hpp"
#include "boundary_tracing/marching_squares.hpp"
#include "featurizations/sdf/chamfer.hpp"
#include "contour_kit/metrics.hpp"

#include <sstream>

// Atom demo: mask in → Chamfer SDF → marching-squares iso loops out.
class MarchingSquaresAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"marching_squares"};
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
        print_banner("run marching squares → iso polylines + SDF");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_loops\tn_points\tclosed\tarea\n";
        contour::MarchingSquares ms;
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const contour::Field sdf = contour::ChamferSDF::from_mask(mask);
            const auto loops = ms.extract(sdf, 0.0f);
            const contour::Polyline largest = ms.largest_closed(sdf, 0.0f);
            const float area = contour::shoelace(largest.points);
            std::cout << "  " << sample.row.file << "  loops=" << loops.size()
                      << "  n=" << largest.points.size() << "  area=" << area << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << loops.size() << '\t'
                       << largest.points.size() << '\t' << (largest.closed ? 1 : 0) << '\t' << area
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string sdf_name = stem + "_sdf.pgm";
            const std::string ov_name = stem + "_iso.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, sdf_name), field_to_gray(sdf));
            vision::save_pgm(vision::join_path(art_dir, ov_name),
                             overlay_polyline(sample.image, largest.points, largest.closed));
            written.push_back(in_name);
            written.push_back(sdf_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: marching_squares.tsv, *_input.pgm, *_sdf.pgm, *_iso.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "marching_squares.tsv"), values_tsv.str());
        written.insert(written.begin(), "marching_squares.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_marching_squares";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        MarchingSquaresAtom atom;
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

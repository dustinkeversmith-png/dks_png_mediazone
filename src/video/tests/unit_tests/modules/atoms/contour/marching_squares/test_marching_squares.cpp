#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/marching_squares/marching_squares.hpp"
#include "sdf/chamfer/chamfer.hpp"
#include "math/contour_metrics.hpp"

#include <sstream>

// Atom demo: mask in → Chamfer SDF → marching-squares iso loops out.
class MarchingSquaresAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"marching_squares"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 128);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run marching squares → iso polylines + SDF");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_loops\tn_points\tclosed\tarea\tchi\tboundary_f1\n";
        contour::MarchingSquares ms;
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const contour::Field sdf = contour::ChamferSDF::from_mask(mask);
            const auto loops = ms.extract(sdf, 0.0f);
            const contour::Polyline largest = ms.largest_closed(sdf, 0.0f);
            const float area = contour::shoelace(largest.points);
            const auto e = vision::EulerCharacteristic::compute(sample.image);
            double boundary_f1 = 0.0;
            if (!largest.points.empty()) {
                contour::Polyline lp = largest;
                boundary_f1 = contour::evaluate_polyline(lp, mask, 0.0).boundary_f1;
            }
            std::cout << "  " << sample.row.file << "  loops=" << loops.size()
                      << "  n=" << largest.points.size() << "  area=" << area << "  chi=" << e.chi
                      << "  f1=" << boundary_f1 << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << loops.size() << '\t'
                       << largest.points.size() << '\t' << (largest.closed ? 1 : 0) << '\t' << area
                       << '\t' << e.chi << '\t' << boundary_f1 << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string iso_name = stem + "_iso_field.pgm";
            const std::string svg_name = stem + "_contour_edges.svg";
            vision::save_pgm(vision::join_path(art_dir, iso_name), field_to_gray(sdf));
            mission::write_polyline_svg(vision::join_path(art_dir, svg_name), sample.image.width,
                                        sample.image.height, largest.points, largest.closed);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            vision::save_pgm(vision::join_path(art_dir, stem + "_iso.pgm"),
                             overlay_polyline(sample.image, largest.points, largest.closed));
            written.push_back(stem + "_input.pgm");
            written.push_back(iso_name);
            written.push_back(svg_name);
            written.push_back(stem + "_iso.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: iso_field.pgm, contour_edges.svg, marching_squares.tsv");
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
        if (!atom.load(cli, argc, argv)) {
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

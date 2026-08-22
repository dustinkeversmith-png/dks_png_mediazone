#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/dual_contouring/dual_contouring.hpp"
#include "sdf/8ssedt/8SSEDT.hpp"
#include "math/contour_metrics.hpp"

#include <sstream>

// Atom demo: mask in → exact SDF → dual-contouring Hermite polyline out.
class DualContouringAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"dual_contouring"};
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
        print_banner("run dual contouring → QEF vertices + loops");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_vertices\tn_edges\tn_loops\tn_points\tarea\tchi\tboundary_f1\n";
        contour::DualContouring2D dc;
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const contour::Field sdf = contour::ExactSDF::from_mask(mask);
            const contour::Polyline poly = dc.extract(sdf, 0.0f);
            const float area = contour::shoelace(poly.points);
            const auto e = vision::EulerCharacteristic::compute(sample.image);
            const double boundary_f1 =
                poly.points.empty() ? 0.0 : contour::evaluate_polyline(poly, mask, 0.0).boundary_f1;
            std::cout << "  " << sample.row.file << "  verts=" << dc.cell_vertices.size()
                      << "  loops=" << dc.last_loops.size() << "  n=" << poly.points.size()
                      << "  chi=" << e.chi << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << dc.cell_vertices.size()
                       << '\t' << dc.edges.size() << '\t' << dc.last_loops.size() << '\t'
                       << poly.points.size() << '\t' << area << '\t' << e.chi << '\t' << boundary_f1
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string mesh_name = stem + "_mesh_vertices.json";
            mission::write_mesh_vertices_json(vision::join_path(art_dir, mesh_name), dc.cell_vertices);
            vision::save_pgm(vision::join_path(art_dir, stem + "_iso_field.pgm"), field_to_gray(sdf));
            mission::write_polyline_svg(vision::join_path(art_dir, stem + "_contour_edges.svg"),
                                        sample.image.width, sample.image.height, poly.points, poly.closed);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            vision::save_pgm(vision::join_path(art_dir, stem + "_dc.pgm"),
                             overlay_polyline(sample.image, poly.points, poly.closed));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_iso_field.pgm");
            written.push_back(mesh_name);
            written.push_back(stem + "_contour_edges.svg");
            written.push_back(stem + "_dc.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: iso_field.pgm, mesh_vertices.json, contour_edges.svg");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "dual_contouring.tsv"), values_tsv.str());
        written.insert(written.begin(), "dual_contouring.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_dual_contouring";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        DualContouringAtom atom;
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

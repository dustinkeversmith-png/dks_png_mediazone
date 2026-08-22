#include "test_harness.hpp"
#include "boundary_tracing/dual_contouring.hpp"
#include "featurizations/sdf/8SSEDT.hpp"
#include "contour_kit/metrics.hpp"

#include <sstream>

// Atom demo: mask in → exact SDF → dual-contouring Hermite polyline out.
class DualContouringAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"dual_contouring"};
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
        print_banner("run dual contouring → QEF vertices + loops");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_vertices\tn_edges\tn_loops\tn_points\tarea\n";
        contour::DualContouring2D dc;
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const contour::Field sdf = contour::ExactSDF::from_mask(mask);
            const contour::Polyline poly = dc.extract(sdf, 0.0f);
            const float area = contour::shoelace(poly.points);
            std::cout << "  " << sample.row.file << "  verts=" << dc.cell_vertices.size()
                      << "  loops=" << dc.last_loops.size() << "  n=" << poly.points.size() << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << dc.cell_vertices.size()
                       << '\t' << dc.edges.size() << '\t' << dc.last_loops.size() << '\t'
                       << poly.points.size() << '\t' << area << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string sdf_name = stem + "_sdf.pgm";
            const std::string ov_name = stem + "_dc.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, sdf_name), field_to_gray(sdf));
            vision::save_pgm(vision::join_path(art_dir, ov_name),
                             overlay_polyline(sample.image, poly.points, poly.closed));
            written.push_back(in_name);
            written.push_back(sdf_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: dual_contouring.tsv, *_input.pgm, *_sdf.pgm, *_dc.pgm");
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

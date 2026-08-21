#include "test_harness.hpp"
#include "bbox/bbox_auto.hpp"
#include "boundary_tracing/level_set.hpp"
#include "contour_kit/metrics.hpp"

#include <sstream>

// Atom demo: image in → Chan-Vese level set polyline out.
class LevelSetAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"level_set"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 96);
        }
        std::cout << "loaded " << samples.size() << " images (max side 96)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Chan-Vese → zero-level overlay + phi");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_points\tclosed\tarea\n";
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const contour::Rect box = contour::BBoxAuto::from_mask(im);
            contour::ChanVeseLevelSet cv;
            cv.iterations = 30;
            const contour::Polyline poly = cv.segment(im, box);
            const float area = contour::shoelace(poly.points);
            std::cout << "  " << sample.row.file << "  n=" << poly.points.size() << "  area=" << area
                      << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << poly.points.size()
                       << '\t' << (poly.closed ? 1 : 0) << '\t' << area << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string phi_name = stem + "_phi.pgm";
            const std::string ov_name = stem + "_levelset.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, phi_name), field_to_gray(cv.phi));
            vision::save_pgm(vision::join_path(art_dir, ov_name),
                             overlay_polyline(sample.image, poly.points, poly.closed));
            written.push_back(in_name);
            written.push_back(phi_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: level_set.tsv, *_input.pgm, *_phi.pgm, *_levelset.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "level_set.tsv"), values_tsv.str());
        written.insert(written.begin(), "level_set.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_level_set";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        LevelSetAtom atom;
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

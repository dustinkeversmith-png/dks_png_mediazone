#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/bbox_auto/bbox_auto.hpp"
#include "contour/level_set/level_set.hpp"
#include "math/contour_metrics.hpp"

#include <sstream>

// Atom demo: image in → Chan-Vese level set polyline out.
class LevelSetAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"level_set"};
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
        print_banner("run Chan-Vese → zero-level overlay + phi");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_points\tclosed\tarea\tchamfer_px\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const auto im = to_contour(sample.image);
            const contour::Rect box = contour::BBoxAuto::from_mask(im);
            contour::ChanVeseLevelSet cv;
            cv.iterations = 30;
            const contour::Polyline poly = cv.segment(im, box);
            const float area = contour::shoelace(poly.points);
            vision::GrayImage gt;
            if (si < provider_samples.size()) {
                gt = !provider_samples[si].ground_truth.empty()
                         ? provider_samples[si].ground_truth
                         : provider_samples[si].sample.boundary;
            }
            const double chamfer = mission::chamfer_polyline(poly, gt);
            std::cout << "  " << sample.row.file << "  n=" << poly.points.size() << "  area=" << area
                      << "  chamfer=" << chamfer << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << poly.points.size()
                       << '\t' << (poly.closed ? 1 : 0) << '\t' << area << '\t' << chamfer << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_accumulated_graph.pgm"), field_to_gray(cv.phi));
            vision::save_pgm(vision::join_path(art_dir, stem + "_snake_evolution_final.pgm"),
                             overlay_polyline(sample.image, poly.points, poly.closed));
            mission::write_convergence_json(vision::join_path(art_dir, stem + "_convergence_log.json"),
                                            sample.row.file, cv.iterations, chamfer, area);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_accumulated_graph.pgm");
            written.push_back(stem + "_snake_evolution_final.pgm");
            written.push_back(stem + "_convergence_log.json");
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: accumulated_graph.pgm, snake_evolution_final.pgm, convergence_log.json");
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

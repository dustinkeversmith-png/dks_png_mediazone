#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/bbox_auto/bbox_auto.hpp"
#include "contour/snakes/snakes.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"
#include "filters/edge/canny/canny.hpp"
#include "filters/gvf/gvh.hpp"
#include "math/contour_metrics.hpp"

#include <sstream>

// Atom: DIS5K luma + GT mask → GVF snake seeded from Moore(mask) boundary.
class SnakesAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"snakes"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 160);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run snakes → active-contour overlay");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_points\tclosed\tarea\tchamfer_px\n";
        contour::SnakeActiveContour snake;
        snake.iterations = 24;
        snake.n_points = 96;
        snake.gamma = 2.2f;
        snake.kappa = 0.0f;  // seeded on GT boundary — refine, don't balloon
        snake.alpha = 0.08f;
        snake.beta = 0.2f;
        snake.dt = 0.15f;
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto luma = mission_luma_image(ps, sample.image);
            const auto mask_img = binarize_mask(mission_mask_image(ps, sample.image));
            const auto im = to_contour(luma);
            const auto mask = to_contour(mask_img);
            const contour::Rect box = contour::BBoxAuto::pad(contour::BBoxAuto::from_mask(mask), 0.08f,
                                                             luma.width, luma.height);

            contour::Canny canny;
            const auto edges = canny.detect(im);
            contour::GradientVectorFlow gvf;
            gvf.iterations = 48;
            gvf.compute(edges, true);

            auto moore = vision::MooreNeighborTracer::trace(mask_img);
            auto seed = vision::MooreNeighborTracer::resample(moore.points, snake.n_points);
            contour::Polyline poly;
            if (seed.size() >= 8) {
                std::vector<contour::Vec2> init;
                init.reserve(seed.size());
                for (const auto& p : seed) {
                    init.push_back({p.x, p.y});
                }
                poly = snake.evolve_from(im, box, std::move(init));
            } else {
                poly = snake.evolve(im, box);
            }
            const float area = std::fabs(contour::shoelace(poly.points));
            const double chamfer = mission::chamfer_polyline(poly, mask_img);
            std::cout << "  " << sample.row.file << "  n=" << poly.points.size() << "  area=" << area
                      << "  chamfer=" << chamfer << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << poly.points.size()
                       << '\t' << (poly.closed ? 1 : 0) << '\t' << area << '\t' << chamfer << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_gvf_field.pgm"),
                             mission::gvf_magnitude(gvf.u, gvf.v));
            vision::save_pgm(vision::join_path(art_dir, stem + "_snake_evolution_final.pgm"),
                             overlay_polyline(luma, poly.points, poly.closed));
            mission::write_convergence_json(vision::join_path(art_dir, stem + "_convergence_log.json"),
                                            sample.row.file, snake.iterations, chamfer, area);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_gvf_field.pgm");
            written.push_back(stem + "_snake_evolution_final.pgm");
            written.push_back(stem + "_convergence_log.json");
            ++report.n_outputs;
        }
        report.notes.push_back("DIS5K: Moore(mask) seeds snake; luma drives GVF energy");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "snakes.tsv"), values_tsv.str());
        written.insert(written.begin(), "snakes.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "dis5k", [&](const AtomCli& cli) -> int {
        SnakesAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for snakes atom\n";
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

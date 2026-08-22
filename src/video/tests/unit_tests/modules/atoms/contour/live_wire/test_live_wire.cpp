#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/bbox_auto/bbox_auto.hpp"
#include "contour/live_wire/live_wire.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <sstream>

// Atom demo: image in → livewire path through Moore/bbox seeds out.
class LiveWireAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"live_wire"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 96);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    static std::vector<contour::Vec2> make_seeds(const vision::GrayImage& image) {
        auto moore = vision::MooreNeighborTracer::trace(image);
        auto resampled = vision::MooreNeighborTracer::resample(moore.points, 12);
        std::vector<contour::Vec2> seeds;
        seeds.reserve(resampled.size());
        for (const auto& p : resampled) {
            seeds.push_back({p.x, p.y});
        }
        if (seeds.size() >= 3) {
            return seeds;
        }
        const auto box = contour::BBoxAuto::from_mask(to_contour(image));
        const float x0 = box.x + box.w * 0.15f;
        const float y0 = box.y + box.h * 0.15f;
        const float x1 = box.x + box.w * 0.85f;
        const float y1 = box.y + box.h * 0.85f;
        return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    }

    void run(const std::string& art_dir) {
        print_banner("run livewire → shortest-path contour");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_seeds\tn_points\tclosed\trmse_px\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const auto im = to_contour(sample.image);
            const auto seeds = make_seeds(sample.image);
            contour::Livewire lw;
            lw.build_cost(im);
            const contour::Polyline poly = lw.trace_waypoints(im, seeds, true);
            vision::GrayImage gt;
            if (si < provider_samples.size()) {
                gt = !provider_samples[si].ground_truth.empty() ? provider_samples[si].ground_truth
                                                                  : provider_samples[si].sample.mask;
            }
            const double rmse = mission::rmse_polyline(poly, gt);
            std::cout << "  " << sample.row.file << "  seeds=" << seeds.size()
                      << "  n=" << poly.points.size() << "  rmse=" << rmse << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << seeds.size() << '\t'
                       << poly.points.size() << '\t' << (poly.closed ? 1 : 0) << '\t' << rmse << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_cost_map.pgm"),
                             mission::cost_to_gray(lw.cost, lw.width, lw.height));
            mission::write_polyline_svg(vision::join_path(art_dir, stem + "_traced_path.svg"),
                                        sample.image.width, sample.image.height, poly.points, poly.closed);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            vision::save_pgm(vision::join_path(art_dir, stem + "_livewire.pgm"),
                             overlay_polyline(sample.image, poly.points, poly.closed));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_cost_map.pgm");
            written.push_back(stem + "_traced_path.svg");
            written.push_back(stem + "_livewire.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: cost_map.pgm, traced_path.svg, live_wire.tsv");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "live_wire.tsv"), values_tsv.str());
        written.insert(written.begin(), "live_wire.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_live_wire";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        LiveWireAtom atom;
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

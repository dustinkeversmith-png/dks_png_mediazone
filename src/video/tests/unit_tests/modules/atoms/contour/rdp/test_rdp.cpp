#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/rdp/rdp.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <sstream>

class RdpAtom {
public:
    struct Curve {
        std::string file;
        std::string label;
        vision::GrayImage luma;
        vision::GrayImage mask;
        std::vector<vision::Vec2> points;
        int width = 0;
        int height = 0;
    };

    std::vector<Curve> curves;
    AtomDemoReport report{"rdp"};
    std::ostringstream values_tsv;
    std::ostringstream simplified_tsv;
    std::vector<std::string> written;
    float eps = 1.5f;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 160);
        for (size_t i = 0; i < mission.samples.size(); ++i) {
            const auto& sample = mission.samples[i];
            const ProviderLoadedSample* ps =
                i < mission.provider_samples.size() ? &mission.provider_samples[i] : nullptr;
            Curve curve;
            curve.file = sample.row.file;
            curve.label = sample.row.label;
            curve.mask = binarize_mask(mission_mask_image(ps, sample.image));
            curve.luma = mission_luma_image(ps, sample.image);
            curve.width = curve.luma.width;
            curve.height = curve.luma.height;
            const auto traced = vision::MooreNeighborTracer::trace(curve.mask);
            // Keep native boundary density (cap only for huge silhouettes).
            const size_t n = traced.points.size();
            const size_t target = n > 512 ? 512 : n;
            curve.points = (target > 0 && target != n)
                               ? vision::MooreNeighborTracer::resample(traced.points, target)
                               : traced.points;
            if (curve.points.size() >= 3) {
                curves.push_back(std::move(curve));
            }
        }
        std::cout << "loaded " << curves.size() << " polylines via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(curves.size());
        return !curves.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run RDP → simplified polylines");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tn_in\tn_out\tmax_err\teps\treduction_pct\n";
        simplified_tsv << "file\ti\tx\ty\n";
        for (const auto& curve : curves) {
            auto simplified = vision::RamerDouglasPeucker::simplify(curve.points, eps);
            const float err = vision::RamerDouglasPeucker::max_error(curve.points, simplified);
            const double reduction =
                curve.points.empty()
                    ? 0.0
                    : (1.0 - static_cast<double>(simplified.size()) /
                                 static_cast<double>(curve.points.size())) *
                          100.0;
            std::cout << "  " << curve.file << "  in=" << curve.points.size()
                      << "  out=" << simplified.size() << "  reduction=" << reduction << "%\n";
            values_tsv << curve.file << '\t' << curve.points.size() << '\t' << simplified.size()
                       << '\t' << err << '\t' << eps << '\t' << reduction << '\n';
            const std::string stem = stem_of(curve.file);
            mission::write_polyline_svg(vision::join_path(art_dir, stem + "_rdp_simplified.svg"),
                                        curve.width, curve.height, simplified, true);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), curve.luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), curve.mask);
            vision::save_pgm(vision::join_path(art_dir, stem + "_rdp_overlay.pgm"),
                             overlay_polyline(curve.luma, simplified, true));
            written.push_back(stem + "_rdp_simplified.svg");
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_rdp_overlay.pgm");
            for (size_t i = 0; i < simplified.size(); ++i) {
                simplified_tsv << curve.file << '\t' << i << '\t' << simplified[i].x << '\t'
                               << simplified[i].y << '\n';
            }
            ++report.n_outputs;
        }
        report.notes.push_back("DIS5K mask boundary → RDP simplification");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "rdp_stats.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "rdp_simplified.tsv"), simplified_tsv.str());
        written.insert(written.begin(), "rdp_simplified.tsv");
        written.insert(written.begin(), "rdp_stats.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "dis5k", [&](const AtomCli& cli) -> int {
        RdpAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for rdp atom\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& c : atom.curves) {
                std::cout << "  " << c.file << "  n=" << c.points.size() << "\n";
            }
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}

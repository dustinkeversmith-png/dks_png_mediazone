#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/rdp/rdp.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <sstream>

class RdpAtom {
public:
    std::vector<std::pair<std::string, std::vector<vision::Vec2>>> curves;
    AtomDemoReport report{"rdp"};
    std::ostringstream values_tsv;
    std::ostringstream simplified_tsv;
    std::vector<std::string> written;
    float eps = 2.0f;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 128);
        for (const auto& sample : mission.samples) {
            const auto traced = vision::MooreNeighborTracer::trace(sample.image);
            auto pts = vision::MooreNeighborTracer::resample(traced.points, 96);
            if (pts.size() >= 3) {
                curves.push_back({sample.row.file, std::move(pts)});
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
            auto simplified = vision::RamerDouglasPeucker::simplify(curve.second, eps);
            const float err = vision::RamerDouglasPeucker::max_error(curve.second, simplified);
            const double reduction =
                curve.second.empty()
                    ? 0.0
                    : (1.0 - static_cast<double>(simplified.size()) / static_cast<double>(curve.second.size())) *
                          100.0;
            std::cout << "  " << curve.first << "  in=" << curve.second.size()
                      << "  out=" << simplified.size() << "  reduction=" << reduction << "%\n";
            values_tsv << curve.first << '\t' << curve.second.size() << '\t' << simplified.size()
                       << '\t' << err << '\t' << eps << '\t' << reduction << '\n';
            mission::write_polyline_svg(vision::join_path(art_dir, curve.first + "_rdp_simplified.svg"),
                                        128, 128, simplified, true);
            written.push_back(curve.first + "_rdp_simplified.svg");
            for (size_t i = 0; i < simplified.size(); ++i) {
                simplified_tsv << curve.first << '\t' << i << '\t' << simplified[i].x << '\t'
                               << simplified[i].y << '\n';
            }
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: rdp_simplified.svg, rdp_stats.tsv");
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
    return run_atom_main(argc, argv, "synthetic_silhouettes", [&](const AtomCli& cli) -> int {
        RdpAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for " << cli.dataset << " under " << cli.data_root << "\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& c : atom.curves) {
                std::cout << "  " << c.first << "  n=" << c.second.size() << "\n";
            }
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}

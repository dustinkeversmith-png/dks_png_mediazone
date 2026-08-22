#include "test_harness.hpp"
#include "vectorization_geometry/rdp/rdp.hpp"

#include <filesystem>
#include <sstream>

// Atom demo: polyline JSON in → simplified polyline + error stats out.
class RdpAtom {
public:
    std::string folder;
    std::vector<std::pair<std::string, std::vector<vision::Vec2>>> curves;
    AtomDemoReport report{"rdp"};
    std::ostringstream values_tsv;
    std::ostringstream simplified_tsv;
    std::vector<std::string> written;
    float eps = 2.0f;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        folder = vision::dataset_dir(root, dataset);
        const auto rows = vision::load_tsv(vision::join_path(folder, "index.tsv"));
        if (!rows.empty()) {
            for (const auto& row : rows) {
                if (!sample_filter.empty() && row.file.find(sample_filter) == std::string::npos) {
                    continue;
                }
                auto pts = vision::load_xy_json(vision::join_path(folder, row.file));
                if (!pts.empty()) {
                    curves.push_back({row.file, std::move(pts)});
                }
            }
        } else if (std::filesystem::exists(folder)) {
            for (const auto& entry : std::filesystem::directory_iterator(folder)) {
                if (!entry.is_regular_file() || entry.path().extension() != ".json") {
                    continue;
                }
                const std::string name = entry.path().filename().string();
                if (!sample_filter.empty() && name.find(sample_filter) == std::string::npos) {
                    continue;
                }
                auto pts = vision::load_xy_json(entry.path().string());
                if (!pts.empty()) {
                    curves.push_back({name, std::move(pts)});
                }
            }
        }
        std::cout << "loaded " << curves.size() << " polylines\n";
        report.n_inputs = static_cast<int>(curves.size());
        return !curves.empty();
    }

    void run(const std::string& /*art_dir*/) {
        print_banner("run RDP → simplified polylines");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tn_in\tn_out\tmax_err\teps\n";
        simplified_tsv << "file\ti\tx\ty\n";
        for (const auto& curve : curves) {
            auto simplified = vision::RamerDouglasPeucker::simplify(curve.second, eps);
            const float err = vision::RamerDouglasPeucker::max_error(curve.second, simplified);
            std::cout << "  " << curve.first << "  in=" << curve.second.size()
                      << "  out=" << simplified.size() << "  max_err=" << err << "\n";
            values_tsv << curve.first << '\t' << curve.second.size() << '\t' << simplified.size()
                       << '\t' << err << '\t' << eps << '\n';
            for (size_t i = 0; i < simplified.size(); ++i) {
                simplified_tsv << curve.first << '\t' << i << '\t' << simplified[i].x << '\t'
                               << simplified[i].y << '\n';
            }
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: rdp_stats.tsv, rdp_simplified.tsv");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "rdp_stats.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "rdp_simplified.tsv"), simplified_tsv.str());
        written = {"rdp_stats.tsv", "rdp_simplified.tsv"};
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_rdp";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        RdpAtom atom;
        if (!atom.load(cli.data_root, cli.dataset, cli.sample_filter)) {
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

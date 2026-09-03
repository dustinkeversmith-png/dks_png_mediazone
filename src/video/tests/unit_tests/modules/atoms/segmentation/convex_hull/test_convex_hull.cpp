#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/convex_hull/convex_hull.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <sstream>

// Atom: COCO mask contour → monotone-chain convex hull + convexity ratio.
class ConvexHullAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"convex_hull"};
    std::ostringstream values_tsv;
    std::ostringstream hull_tsv;
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

    static double mask_area(const vision::GrayImage& mask, uint8_t thr = 127) {
        double a = 0.0;
        for (uint8_t p : mask.data) {
            if (p > thr) {
                a += 1.0;
            }
        }
        return a;
    }

    void run(const std::string& art_dir) {
        print_banner("run convex hull → hull + convexity ratio");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\thull_n\tdefects\thull_area\tmask_area\tconvexity\n";
        hull_tsv << "file\ti\tx\ty\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            // In-test prep: binarize + largest CC → Moore vertices → hull.
            const auto mask_img = prepare_contour_mask(mission_mask_image(ps, sample.image));
            const auto luma = mission_luma_image(ps, sample.image);
            auto r = vision::ConvexHull::analyze(mask_img);
            const double m_area = mask_area(mask_img);
            const double convexity =
                (m_area > 1.0) ? static_cast<double>(std::fabs(r.hull_area)) / m_area : 0.0;
            std::cout << "  " << sample.row.file << "  hull=" << r.hull.size()
                      << "  defects=" << r.defects.size() << "  convexity=" << convexity << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << r.hull.size() << '\t'
                       << r.defects.size() << '\t' << r.hull_area << '\t' << m_area << '\t'
                       << convexity << '\n';
            for (size_t i = 0; i < r.hull.size(); ++i) {
                hull_tsv << sample.row.file << '\t' << i << '\t' << r.hull[i].x << '\t' << r.hull[i].y
                         << '\n';
            }
            const std::string stem = stem_of(sample.row.file);
            mission::write_polyline_svg(vision::join_path(art_dir, stem + "_convex_hull_defects.svg"),
                                        luma.width, luma.height, r.hull, true);
            std::ostringstream hull_json;
            hull_json << "{\n  \"file\": \"" << mission::json_escape(sample.row.file)
                      << "\",\n  \"hull_n\": " << r.hull.size() << ",\n  \"defects\": "
                      << r.defects.size() << ",\n  \"hull_area\": " << mission::json_num(r.hull_area, 2)
                      << ",\n  \"convexity\": " << mission::json_num(convexity, 4) << "\n}\n";
            vision::write_text_file(vision::join_path(art_dir, stem + "_hull_metrics.json"),
                                    hull_json.str());
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_hull_overlay.pgm"),
                             overlay_polyline(luma, r.hull, true));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_convex_hull_defects.svg");
            written.push_back(stem + "_hull_metrics.json");
            written.push_back(stem + "_hull_overlay.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("COCO: mask→Moore→hull; convexity=Area(hull)/Area(mask)");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "hull_stats.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "hull_points.tsv"), hull_tsv.str());
        written.insert(written.begin(), "hull_points.tsv");
        written.insert(written.begin(), "hull_stats.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "coco", [&](const AtomCli& cli) -> int {
        ConvexHullAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for convex_hull atom\n";
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

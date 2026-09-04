#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "object_proposals.hpp"
#include "segmentation/convex_hull/convex_hull.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <sstream>

// Atom: COCO instance masks → polarity fix → watershed split → per-instance convex hull.
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
        print_banner("run convex hull → per-instance hulls after watershed split");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tinstance\thull_n\tdefects\thull_area\tmask_area\tconvexity\n";
        hull_tsv << "file\tinstance\ti\tx\ty\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto luma = mission_luma_image(ps, sample.image);

            // Prefer per-instance COCO polygons; else GT mask + watershed split.
            std::vector<vision::GrayImage> instances;
            if (ps && !ps->sample.instance_masks.empty()) {
                for (const auto& im : ps->sample.instance_masks) {
                    auto m = downscale_max_side(im, 160);
                    for (uint8_t& p : m.data) {
                        p = p > 0 ? 255 : 0;
                    }
                    if (mask_area(m) >= 32) {
                        instances.push_back(std::move(m));
                    }
                }
            }
            vision::GrayImage mask_img;
            if (instances.empty()) {
                mask_img = binarize_mask(mission_mask_image(ps, sample.image));
                if (mask_img.empty() || mask_area(mask_img) < 16) {
                    vision::GrayImage rgb = luma;
                    if (ps && !ps->sample.rgb.empty()) {
                        rgb = downscale_max_side(ps->sample.rgb, 160);
                    }
                    mask_img = vision::propose_objects_from_photo(rgb).binary;
                }
                mask_img = vision::ensure_dark_object_polarity(luma, mask_img);
                instances = vision::split_instances_watershed(mask_img);
            } else {
                mask_img = vision::make_gray(luma.width, luma.height, 0);
                // Union for overview artifact.
                for (const auto& im : instances) {
                    for (size_t i = 0; i < mask_img.data.size() && i < im.data.size(); ++i) {
                        if (im.data[i]) {
                            mask_img.data[i] = 255;
                        }
                    }
                }
            }
            vision::GrayImage overlay = luma;
            for (uint8_t& p : overlay.data) {
                p = static_cast<uint8_t>(p / 2);
            }

            int inst_i = 0;
            for (const auto& inst : instances) {
                auto r = vision::ConvexHull::analyze(inst);
                if (r.hull.size() < 3) {
                    continue;
                }
                const double m_area = mask_area(inst);
                const double convexity =
                    (m_area > 1.0) ? static_cast<double>(std::fabs(r.hull_area)) / m_area : 0.0;
                values_tsv << sample.row.file << '\t' << inst_i << '\t' << r.hull.size() << '\t'
                           << r.defects.size() << '\t' << r.hull_area << '\t' << m_area << '\t'
                           << convexity << '\n';
                for (size_t i = 0; i < r.hull.size(); ++i) {
                    hull_tsv << sample.row.file << '\t' << inst_i << '\t' << i << '\t' << r.hull[i].x
                             << '\t' << r.hull[i].y << '\n';
                }
                overlay = overlay_polyline(overlay, r.hull, true, 255);
                ++inst_i;
            }

            std::cout << "  " << sample.row.file << "  instances=" << inst_i
                      << "  from_split=" << instances.size() << "\n";

            const std::string stem = stem_of(sample.row.file);
            mission::write_polyline_svg(vision::join_path(art_dir, stem + "_convex_hull_defects.svg"),
                                        luma.width, luma.height,
                                        instances.empty()
                                            ? std::vector<vision::Vec2>{}
                                            : vision::ConvexHull::analyze(instances.front()).hull,
                                        true);
            std::ostringstream hull_json;
            hull_json << "{\n  \"file\": \"" << mission::json_escape(sample.row.file)
                      << "\",\n  \"n_instances\": " << inst_i << "\n}\n";
            vision::write_text_file(vision::join_path(art_dir, stem + "_hull_metrics.json"),
                                    hull_json.str());
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_hull_overlay.pgm"), overlay);
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_convex_hull_defects.svg");
            written.push_back(stem + "_hull_metrics.json");
            written.push_back(stem + "_hull_overlay.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("COCO: dark polarity + watershed split → per-instance hulls");
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

#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/suzuki_abe_top_border/suz_top_border.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <sstream>

// Atom: SBD/COCO binary masks → Suzuki–Abe hierarchy (outer frames + hole children).
class SuzukiAbeAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"suz_top_border"};
    std::ostringstream values_tsv;
    std::ostringstream hierarchy_tsv;
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
        print_banner("run Suzuki–Abe → border hierarchy tree");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tn_borders\tn_outer\tn_holes\n";
        hierarchy_tsv << "file\tid\tis_hole\tparent\tchild\tnext\tprev\tn\tarea\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            // In-test prep: binarize, drop dust speckles, keep holes for hierarchy.
            auto mask_img = binarize_mask(mission_mask_image(ps, sample.image));
            for (uint8_t& p : mask_img.data) {
                p = p > 0 ? 255 : 0;
            }
            {
                auto ccl = vision::ConnectedComponentLabeler::label(mask_img);
                vision::GrayImage cleaned = mask_img;
                cleaned.data.assign(cleaned.data.size(), 0);
                for (const auto& c : ccl.components) {
                    if (c.area < 32) {
                        continue;
                    }
                    for (int y = 0; y < ccl.height; ++y) {
                        for (int x = 0; x < ccl.width; ++x) {
                            if (ccl.labels[static_cast<size_t>(y * ccl.width + x)] == c.label) {
                                cleaned.at(x, y) = 255;
                            }
                        }
                    }
                }
                // Also fill background speckles inside FG via inverse small-CC removal.
                mask_img = cleaned;
            }
            const auto luma = mission_luma_image(ps, sample.image);
            const auto result = vision::SuzukiAbeTopBorder::find(mask_img);

            std::cout << "  " << sample.row.file << "  borders=" << (result.borders.size() - 1)
                      << "  outer=" << result.n_outer << "  holes=" << result.n_holes << "\n";
            values_tsv << sample.row.file << '\t' << (result.borders.size() - 1) << '\t'
                       << result.n_outer << '\t' << result.n_holes << '\n';

            std::vector<contour::Polyline> loops;
            for (size_t bi = 1; bi < result.borders.size(); ++bi) {
                const auto& b = result.borders[bi];
                hierarchy_tsv << sample.row.file << '\t' << b.id << '\t' << (b.is_hole ? 1 : 0)
                              << '\t' << b.parent << '\t' << b.first_child << '\t' << b.next << '\t'
                              << b.prev << '\t' << b.points.size() << '\t' << b.area << '\n';
                contour::Polyline poly;
                poly.points = b.points;
                poly.closed = true;
                loops.push_back(std::move(poly));
            }

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_border_labels.pgm"), result.labeled);
            vision::save_pgm(vision::join_path(art_dir, stem + "_hierarchy_overlay.pgm"),
                             overlay_polylines(luma, loops));

            std::ostringstream tree_json;
            tree_json << "{\n  \"file\": \"" << mission::json_escape(sample.row.file)
                      << "\",\n  \"n_outer\": " << result.n_outer << ",\n  \"n_holes\": "
                      << result.n_holes << ",\n  \"borders\": [\n";
            for (size_t bi = 1; bi < result.borders.size(); ++bi) {
                const auto& b = result.borders[bi];
                if (bi > 1) {
                    tree_json << ",\n";
                }
                tree_json << "    {\"id\": " << b.id << ", \"is_hole\": " << (b.is_hole ? "true" : "false")
                          << ", \"parent\": " << b.parent << ", \"first_child\": " << b.first_child
                          << ", \"next\": " << b.next << ", \"prev\": " << b.prev
                          << ", \"n\": " << b.points.size() << ", \"area\": "
                          << mission::json_num(b.area, 2) << "}";
            }
            tree_json << "\n  ]\n}\n";
            vision::write_text_file(vision::join_path(art_dir, stem + "_hierarchy.json"),
                                    tree_json.str());

            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_border_labels.pgm");
            written.push_back(stem + "_hierarchy_overlay.pgm");
            written.push_back(stem + "_hierarchy.json");
            ++report.n_outputs;
        }
        report.notes.push_back("SBD: binary mask → Suzuki–Abe outer/hole hierarchy");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "suzuki_abe.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "hierarchy.tsv"), hierarchy_tsv.str());
        written.insert(written.begin(), "hierarchy.tsv");
        written.insert(written.begin(), "suzuki_abe.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "sbd", [&](const AtomCli& cli) -> int {
        SuzukiAbeAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for suz_top_border atom\n";
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

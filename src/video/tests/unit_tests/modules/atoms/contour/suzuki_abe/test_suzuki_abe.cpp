#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "object_proposals.hpp"
#include "contour/suzuki_abe/suzuki_abe.hpp"
#include "segmentation/helpers/ccl/connected_components.hpp"
#include "topology/euler/euler_characteristic.hpp"

#include <sstream>

namespace {

int mask_area(const vision::GrayImage& mask) {
    int area = 0;
    for (uint8_t p : mask.data) {
        area += p > 0 ? 1 : 0;
    }
    return area;
}

vision::GrayImage clean_instance_mask(const vision::GrayImage& input) {
    auto mask = binarize_mask(input);
    const auto ccl = vision::ConnectedComponentLabeler::label(mask);
    vision::GrayImage cleaned = vision::make_gray(mask.width, mask.height, 0);
    const int min_piece_area = std::max(2, (mask.width * mask.height) / 10000);
    for (const auto& component : ccl.components) {
        if (component.area < min_piece_area) {
            continue;
        }
        for (size_t i = 0; i < ccl.labels.size(); ++i) {
            if (ccl.labels[i] == component.label) {
                cleaned.data[i] = 255;
            }
        }
    }
    return cleaned;
}

struct InstanceTrace {
    int instance_id = 0;
    vision::GrayImage mask;
    vision::SuzukiAbeTopBorder::Result hierarchy;
    vision::EulerCharacteristic::Result euler;
};

bool verify_fixture(const char* name, const vision::GrayImage& mask, int outer, int holes) {
    const auto hierarchy = vision::SuzukiAbeTopBorder::find(mask);
    const auto euler = vision::EulerCharacteristic::compute(mask);
    const bool counts_ok = hierarchy.n_outer == outer && hierarchy.n_holes == holes;
    const bool chi_ok = hierarchy.n_outer - hierarchy.n_holes == euler.chi;
    if (!counts_ok || !chi_ok) {
        std::cerr << "Suzuki fixture failed: " << name << " outer=" << hierarchy.n_outer
                  << " holes=" << hierarchy.n_holes << " chi=" << euler.chi << "\n";
        return false;
    }
    return true;
}

bool run_regression_checks() {
    auto filled = vision::make_gray(24, 20, 0);
    for (int y = 4; y < 16; ++y) {
        for (int x = 5; x < 19; ++x) {
            filled.at(x, y) = 255;
        }
    }

    auto donut = filled;
    for (int y = 8; y < 12; ++y) {
        for (int x = 9; x < 15; ++x) {
            donut.at(x, y) = 0;
        }
    }

    auto separate = filled;
    for (int y = 4; y < 16; ++y) {
        for (int x = 11; x < 13; ++x) {
            separate.at(x, y) = 0;
        }
    }

    const auto donut_hierarchy = vision::SuzukiAbeTopBorder::find(donut);
    const bool hierarchy_ok =
        donut_hierarchy.borders.size() == 3 && donut_hierarchy.borders[2].is_hole &&
        donut_hierarchy.borders[2].parent == 1 &&
        donut_hierarchy.borders[1].first_child == 2;
    if (!hierarchy_ok) {
        std::cerr << "Suzuki fixture failed: donut hierarchy links\n";
    }
    return verify_fixture("filled", filled, 1, 0) &&
           verify_fixture("donut", donut, 1, 1) &&
           verify_fixture("separate", separate, 2, 0) && hierarchy_ok;
}

}  // namespace

// Atom: SBD/COCO binary masks → Suzuki–Abe hierarchy (outer frames + hole children).
class SuzukiAbeAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"suzuki_abe"};
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
        values_tsv
            << "file\tn_instances\tn_borders\tn_outer\tn_holes\tchi\ttopology_ok\n";
        hierarchy_tsv
            << "file\tinstance\tid\tis_hole\tparent\tchild\tnext\tprev\tn\tarea\tperimeter\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto luma = mission_luma_image(ps, sample.image);

            // Preserve dataset instance identity. A union mask destroys the
            // boundary between touching objects and cannot be separated by a
            // border follower after the fact.
            std::vector<vision::GrayImage> instance_masks;
            if (ps && !ps->sample.instance_masks.empty()) {
                for (const auto& im : ps->sample.instance_masks) {
                    auto mask = clean_instance_mask(
                        resize_nearest(im, luma.width, luma.height));
                    if (mask_area(mask) >= std::max(8, (luma.width * luma.height) / 4000)) {
                        instance_masks.push_back(std::move(mask));
                    }
                }
            }

            if (instance_masks.empty()) {
                auto merged = mission_mask_image(ps, sample.image);
                if (merged.width != luma.width || merged.height != luma.height) {
                    merged = resize_nearest(merged, luma.width, luma.height);
                }
                merged = clean_instance_mask(merged);
                instance_masks = vision::split_instances_watershed(merged);
            }

            std::vector<InstanceTrace> traces;
            for (size_t ii = 0; ii < instance_masks.size(); ++ii) {
                auto mask = clean_instance_mask(instance_masks[ii]);
                if (mask_area(mask) == 0) {
                    continue;
                }
                InstanceTrace trace;
                trace.instance_id = static_cast<int>(ii + 1);
                trace.mask = std::move(mask);
                trace.hierarchy = vision::SuzukiAbeTopBorder::find(trace.mask);
                trace.euler = vision::EulerCharacteristic::compute(trace.mask);
                traces.push_back(std::move(trace));
            }

            auto mask_img = vision::make_gray(luma.width, luma.height, 0);
            auto instance_labels = vision::make_gray(luma.width, luma.height, 0);
            auto border_labels = vision::make_gray(luma.width, luma.height, 0);
            std::vector<contour::Polyline> loops;
            int n_borders = 0;
            int n_outer = 0;
            int n_holes = 0;
            int chi = 0;
            bool topology_ok = true;
            for (const auto& trace : traces) {
                const uint8_t instance_value = static_cast<uint8_t>(
                    32 + ((trace.instance_id * 47) % 208));
                for (size_t i = 0; i < trace.mask.data.size(); ++i) {
                    if (trace.mask.data[i]) {
                        mask_img.data[i] = 255;
                        instance_labels.data[i] = instance_value;
                    }
                    if (trace.hierarchy.labeled.data[i]) {
                        border_labels.data[i] = instance_value;
                    }
                }
                n_borders += static_cast<int>(trace.hierarchy.borders.size()) - 1;
                n_outer += trace.hierarchy.n_outer;
                n_holes += trace.hierarchy.n_holes;
                chi += trace.euler.chi;
                topology_ok = topology_ok &&
                              trace.hierarchy.n_outer - trace.hierarchy.n_holes ==
                                  trace.euler.chi;

                for (size_t bi = 1; bi < trace.hierarchy.borders.size(); ++bi) {
                    const auto& b = trace.hierarchy.borders[bi];
                    hierarchy_tsv << sample.row.file << '\t' << trace.instance_id << '\t'
                                  << b.id << '\t' << (b.is_hole ? 1 : 0) << '\t'
                                  << b.parent << '\t' << b.first_child << '\t' << b.next
                                  << '\t' << b.prev << '\t' << b.points.size() << '\t'
                                  << b.area << '\t' << b.perimeter << '\n';
                    contour::Polyline poly;
                    poly.points = b.points;
                    poly.closed = true;
                    loops.push_back(std::move(poly));
                }
            }

            std::cout << "  " << sample.row.file << "  instances=" << traces.size()
                      << "  borders=" << n_borders << "  outer=" << n_outer
                      << "  holes=" << n_holes << "  topology="
                      << (topology_ok ? "ok" : "MISMATCH") << "\n";
            values_tsv << sample.row.file << '\t' << traces.size() << '\t' << n_borders
                       << '\t' << n_outer << '\t' << n_holes << '\t' << chi << '\t'
                       << (topology_ok ? 1 : 0) << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_processed_base.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_instance_labels.pgm"),
                             instance_labels);
            vision::save_pgm(vision::join_path(art_dir, stem + "_border_labels.pgm"),
                             border_labels);
            vision::save_pgm(vision::join_path(art_dir, stem + "_suzuki_abe_hierarchy.pgm"),
                             overlay_polylines(luma, loops));

            std::ostringstream tree_json;
            tree_json << "{\n  \"file\": \"" << mission::json_escape(sample.row.file)
                      << "\",\n  \"n_instances\": " << traces.size()
                      << ",\n  \"n_outer\": " << n_outer << ",\n  \"n_holes\": "
                      << n_holes << ",\n  \"topology_ok\": "
                      << (topology_ok ? "true" : "false") << ",\n  \"instances\": [";
            for (size_t ti = 0; ti < traces.size(); ++ti) {
                const auto& trace = traces[ti];
                if (ti > 0) {
                    tree_json << ",";
                }
                tree_json << "\n    {\"id\": " << trace.instance_id
                          << ", \"chi\": " << trace.euler.chi << ", \"borders\": [";
                for (size_t bi = 1; bi < trace.hierarchy.borders.size(); ++bi) {
                    const auto& b = trace.hierarchy.borders[bi];
                    if (bi > 1) {
                        tree_json << ", ";
                    }
                    tree_json << "{\"id\": " << b.id << ", \"is_hole\": "
                              << (b.is_hole ? "true" : "false")
                              << ", \"parent\": " << b.parent
                              << ", \"first_child\": " << b.first_child
                              << ", \"next\": " << b.next << ", \"prev\": " << b.prev
                              << ", \"n\": " << b.points.size() << ", \"area\": "
                              << mission::json_num(b.area, 2) << ", \"perimeter\": "
                              << mission::json_num(b.perimeter, 2) << "}";
                }
                tree_json << "]}";
            }
            tree_json << "\n  ]\n}\n";
            vision::write_text_file(vision::join_path(art_dir, stem + "_hierarchy.json"),
                                    tree_json.str());

            written.push_back(stem + "_processed_base.pgm");
            written.push_back(stem + "_instance_labels.pgm");
            written.push_back(stem + "_border_labels.pgm");
            written.push_back(stem + "_suzuki_abe_hierarchy.pgm");
            written.push_back(stem + "_hierarchy.json");
            ++report.n_outputs;
        }
        report.notes.push_back(
            "SBD instances kept separate; watershed fallback; Euler-checked hierarchy");
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
        if (!run_regression_checks()) {
            return 2;
        }
        SuzukiAbeAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for suzuki_abe atom\n";
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

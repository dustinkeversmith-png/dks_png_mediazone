#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "atom_config.hpp"
#include "segmentation/helpers/watershed/watershed.hpp"
#include "math/contour_metrics.hpp"
#include "filters/sobel/sobel.hpp"

#include <sstream>

// Atom: BSDS500 photo → Sobel |∇I| relief + marker minima → Meyer watershed.
class WatershedAtom {
public:
    vision::AtomConfig config;
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"watershed"};
    std::ostringstream values_tsv;
    std::ostringstream genus_tsv;
    std::vector<std::string> written;

    explicit WatershedAtom(int argc, char** argv) {
        config = vision::load_atom_config_near(argc > 0 ? argv[0] : nullptr);
    }

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
        print_banner("run watershed pipeline (gradient relief)");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_basins\tn_watershed\tboundary_f1\tiou\n";
        genus_tsv << "file\tlabel\tboundary_f1\tiou\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto luma = mission_luma_image(ps, sample.image);

            // Recipe: prefer clean binary → EDT markers → Meyer flood; else gradient relief.
            auto bin = mission_mask_image(ps, sample.image);
            contour::Watershed::Result ws;
            int bin_fg = 0;
            for (uint8_t p : bin.data) {
                bin_fg += p > 0 ? 1 : 0;
            }
            if (!bin.empty() && bin_fg > 32) {
                if (bin.width != luma.width || bin.height != luma.height) {
                    bin = resize_nearest(bin, luma.width, luma.height);
                }
                bin = binarize_mask(bin);
                ws = contour::Watershed::from_mask(to_contour(bin));
            } else {
                ws = contour::Watershed::from_gradient(to_contour(luma));
            }

            double boundary_f1 = 0.0;
            double iou = 0.0;
            if (ps != nullptr) {
                contour::ImageBuffer gt_b;
                if (!ps->sample.boundary.empty()) {
                    gt_b = to_contour(downscale_max_side(ps->sample.boundary, 160));
                } else if (!ps->ground_truth.empty()) {
                    gt_b = contour::boundary_pixels(
                        to_contour(downscale_max_side(ps->ground_truth, 160)));
                }
                if (!gt_b.empty()) {
                    if (gt_b.width != ws.width || gt_b.height != ws.height) {
                        // Nearest resize GT boundary to watershed canvas.
                        auto resized = contour::make_gray(ws.width, ws.height, 0);
                        for (int y = 0; y < ws.height; ++y) {
                            for (int x = 0; x < ws.width; ++x) {
                                const int sx = x * gt_b.width / std::max(1, ws.width);
                                const int sy = y * gt_b.height / std::max(1, ws.height);
                                resized.at(x, y) = gt_b.at(sx, sy);
                            }
                        }
                        gt_b = std::move(resized);
                    }
                    auto pred_b = contour::make_gray(ws.width, ws.height, 0);
                    for (size_t i = 0; i < pred_b.data.size() && i < ws.labels.size(); ++i) {
                        pred_b.data[i] = ws.labels[i] == 0 ? 255 : 0;
                    }
                    boundary_f1 = contour::boundary_f1(pred_b, gt_b, 2);
                }
                if (!ps->ground_truth.empty()) {
                    auto gt = downscale_max_side(ps->ground_truth, 160);
                    if (gt.width == ws.width && gt.height == ws.height) {
                        auto pred_fg = contour::make_gray(ws.width, ws.height, 0);
                        for (size_t i = 0; i < pred_fg.data.size() && i < ws.labels.size(); ++i) {
                            pred_fg.data[i] = ws.labels[i] > 0 ? 255 : 0;
                        }
                        iou = contour::mask_iou(pred_fg, to_contour(gt));
                    }
                }
            }

            std::cout << "  " << sample.row.file << "  basins=" << ws.n_basins
                      << "  lines=" << ws.n_watershed << "  f1=" << boundary_f1 << "  iou=" << iou
                      << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << ws.n_basins << '\t'
                       << ws.n_watershed << '\t' << boundary_f1 << '\t' << iou << '\n';
            genus_tsv << sample.row.file << '\t' << sample.row.label << '\t' << boundary_f1 << '\t'
                      << iou << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::GrayImage lines = vision::make_gray(ws.width, ws.height, 0);
            for (size_t i = 0; i < lines.data.size() && i < ws.labels.size(); ++i) {
                lines.data[i] = ws.labels[i] == 0 ? 255 : 0;
            }
            vision::save_pgm(vision::join_path(art_dir, stem + "_processed_base.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_marker_seeds.pgm"),
                             colorize_labels(ws.markers, ws.width, ws.height));
            vision::save_pgm(vision::join_path(art_dir, stem + "_edt_relief.pgm"),
                             field_to_gray(ws.relief));
            vision::save_pgm(vision::join_path(art_dir, stem + "_watershed_lines.pgm"), lines);
            vision::save_pgm(vision::join_path(art_dir, stem + "_segmented_labels.pgm"),
                             colorize_labels(ws.labels, ws.width, ws.height));
            written.push_back(stem + "_processed_base.pgm");
            written.push_back(stem + "_marker_seeds.pgm");
            written.push_back(stem + "_edt_relief.pgm");
            written.push_back(stem + "_watershed_lines.pgm");
            written.push_back(stem + "_segmented_labels.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("recipe: binary/EDT markers → Meyer; *_processed_base, *_watershed_lines");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "watershed.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "genus_summary.tsv"), genus_tsv.str());
        written.insert(written.begin(), "genus_summary.tsv");
        written.insert(written.begin(), "watershed.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "bsds500", [&](const AtomCli& cli) -> int {
        WatershedAtom atom(argc, argv);
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for watershed atom\n";
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

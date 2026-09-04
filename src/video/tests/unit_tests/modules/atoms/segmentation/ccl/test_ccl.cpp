#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "object_proposals.hpp"
#include "datasets/io/mat_io.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <sstream>

// Atom: GT instance masks (COCO/SBD .mat/json) → CCL; photo proposals for polarity demo.
class CclAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"ccl"};
    std::ostringstream values_tsv;
    std::ostringstream score_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 256);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    static double best_iou(const vision::Rect& pred, const std::vector<vision::Rect>& gts) {
        double best = 0.0;
        for (const auto& g : gts) {
            best = std::max(best, mission::bbox_iou(pred, g));
        }
        return best;
    }

    static vision::GrayImage scale_box_space(const vision::GrayImage& src, int max_side) {
        return downscale_max_side(src, max_side);
    }

    void run(const std::string& art_dir) {
        print_banner("run CCL → GT instances + photo proposals");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tsource\tlabel_id\tarea\tx\ty\tw\th\tbest_gt_iou\n";
        score_tsv << "file\tn_pred\tn_gt\tmean_best_iou\trecall_at_0.3\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;

            vision::GrayImage rgb = sample.image;
            if (ps && !ps->sample.rgb.empty()) {
                rgb = scale_box_space(ps->sample.rgb, 256);
            }
            const auto luma = mission_luma_image(ps, sample.image);

            // --- Primary: CCL on GT instance foreground (algorithm correctness) ---
            vision::GrayImage gt_mask;
            std::vector<vision::Rect> gt_boxes;
            if (ps) {
                // Build GT from per-instance polygon masks when available (authoritative).
                if (!ps->sample.instance_masks.empty()) {
                    gt_mask = vision::make_gray(luma.width, luma.height, 0);
                    for (const auto& im0 : ps->sample.instance_masks) {
                        auto im = downscale_max_side(im0, 256);
                        if (im.width != luma.width || im.height != luma.height) {
                            vision::GrayImage resized = vision::make_gray(luma.width, luma.height, 0);
                            for (int y = 0; y < luma.height; ++y) {
                                for (int x = 0; x < luma.width; ++x) {
                                    const int sx2 = x * im.width / std::max(1, luma.width);
                                    const int sy2 = y * im.height / std::max(1, luma.height);
                                    resized.at(x, y) = im.at(sx2, sy2) > 0 ? 255 : 0;
                                }
                            }
                            im = std::move(resized);
                        } else {
                            for (uint8_t& p : im.data) {
                                p = p > 0 ? 255 : 0;
                            }
                        }
                        int x0 = im.width, y0 = im.height, x1 = -1, y1 = -1;
                        for (int y = 0; y < im.height; ++y) {
                            for (int x = 0; x < im.width; ++x) {
                                if (im.at(x, y) == 0) {
                                    continue;
                                }
                                gt_mask.at(x, y) = 255;
                                x0 = std::min(x0, x);
                                y0 = std::min(y0, y);
                                x1 = std::max(x1, x);
                                y1 = std::max(y1, y);
                            }
                        }
                        if (x1 >= x0) {
                            gt_boxes.push_back({static_cast<float>(x0), static_cast<float>(y0),
                                                static_cast<float>(x1 - x0 + 1),
                                                static_cast<float>(y1 - y0 + 1)});
                        }
                    }
                }
                if (gt_mask.empty()) {
                    vision::GrayImage raw_gt;
                    if (!ps->ground_truth.empty()) {
                        raw_gt = ps->ground_truth;
                    } else if (!ps->sample.mask.empty()) {
                        raw_gt = ps->sample.mask;
                    }
                    if (!raw_gt.empty()) {
                        gt_mask = downscale_max_side(raw_gt, 256);
                        if (gt_mask.width != luma.width || gt_mask.height != luma.height) {
                            vision::GrayImage resized = vision::make_gray(luma.width, luma.height, 0);
                            for (int y = 0; y < luma.height; ++y) {
                                for (int x = 0; x < luma.width; ++x) {
                                    const int sx2 = x * gt_mask.width / std::max(1, luma.width);
                                    const int sy2 = y * gt_mask.height / std::max(1, luma.height);
                                    resized.at(x, y) = gt_mask.at(sx2, sy2) > 0 ? 255 : 0;
                                }
                            }
                            gt_mask = std::move(resized);
                        } else {
                            for (uint8_t& p : gt_mask.data) {
                                p = p > 0 ? 255 : 0;
                            }
                        }
                    }
                }
                if (gt_boxes.empty()) {
                    float sx = 1.0f, sy = 1.0f;
                    const int src_w = !ps->sample.rgb.empty() ? ps->sample.rgb.width
                                      : (!ps->sample.mask.empty() ? ps->sample.mask.width : luma.width);
                    const int src_h = !ps->sample.rgb.empty() ? ps->sample.rgb.height
                                      : (!ps->sample.mask.empty() ? ps->sample.mask.height : luma.height);
                    if (src_w > 0) {
                        sx = static_cast<float>(luma.width) / static_cast<float>(src_w);
                        sy = static_cast<float>(luma.height) / static_cast<float>(src_h);
                    }
                    for (const auto& b : ps->sample.boxes) {
                        gt_boxes.push_back({b.x * sx, b.y * sy, b.w * sx, b.h * sy});
                    }
                }
                if (gt_boxes.empty() && !gt_mask.empty()) {
                    gt_boxes = datasets::boxes_from_label_map(gt_mask);
                }
            }

            // Photo proposals (edge+chrominance) — polarity-safe demo path.
            const auto proposal = vision::propose_objects_from_photo(
                (!ps || ps->sample.rgb.empty()) ? luma : downscale_max_side(ps->sample.rgb, 256));

            const bool use_gt = !gt_mask.empty();
            vision::GrayImage ccl_input = use_gt ? gt_mask : proposal.binary;
            auto ccl = vision::ConnectedComponentLabeler::label(ccl_input);

            double sum_iou = 0.0;
            int hit = 0;
            vision::GrayImage box_overlay = luma;
            for (uint8_t& p : box_overlay.data) {
                p = static_cast<uint8_t>(p / 2);
            }
            for (const auto& c : ccl.components) {
                const double iou = best_iou(c.bbox, gt_boxes);
                sum_iou += iou;
                if (iou >= 0.3) {
                    ++hit;
                }
                values_tsv << sample.row.file << '\t' << (use_gt ? "gt_mask" : "proposal")
                           << '\t' << c.label << '\t' << c.area << '\t' << c.bbox.x << '\t' << c.bbox.y
                           << '\t' << c.bbox.w << '\t' << c.bbox.h << '\t' << iou << '\n';
                box_overlay = overlay_rect(box_overlay, c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h, 255);
            }
            for (const auto& g : gt_boxes) {
                box_overlay = overlay_rect(box_overlay, g.x, g.y, g.w, g.h, 140);
            }

            const double mean_iou =
                ccl.components.empty() ? 0.0 : sum_iou / static_cast<double>(ccl.components.size());
            // Recall: fraction of GT boxes matched by some pred at IoU>=0.3
            int gt_hit = 0;
            for (const auto& g : gt_boxes) {
                double best = 0.0;
                for (const auto& c : ccl.components) {
                    best = std::max(best, mission::bbox_iou(c.bbox, g));
                }
                if (best >= 0.3) {
                    ++gt_hit;
                }
            }
            const double recall =
                gt_boxes.empty() ? 0.0
                                 : static_cast<double>(gt_hit) / static_cast<double>(gt_boxes.size());
            std::cout << "  " << sample.row.file << "  source=" << (use_gt ? "gt_mask" : "proposal")
                      << "  components=" << ccl.components.size() << "  gt=" << gt_boxes.size()
                      << "  mean_iou=" << mean_iou << "  recall@0.3=" << recall << "\n";
            score_tsv << sample.row.file << '\t' << ccl.components.size() << '\t' << gt_boxes.size()
                      << '\t' << mean_iou << '\t' << recall << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::GrayImage labeled =
                colorize_labels(ccl.labels, ccl_input.width, ccl_input.height);
            mission::write_bbox_json(vision::join_path(art_dir, stem + "_detected_bboxes.json"),
                                     sample.row.file, ccl.components);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_gt_mask.pgm"),
                             gt_mask.empty() ? proposal.binary : gt_mask);
            vision::save_pgm(vision::join_path(art_dir, stem + "_proposal_mask.pgm"), proposal.binary);
            vision::save_pgm(vision::join_path(art_dir, stem + "_edges.pgm"), proposal.edges);
            vision::save_pgm(vision::join_path(art_dir, stem + "_saliency.pgm"), proposal.saliency);
            vision::save_pgm(vision::join_path(art_dir, stem + "_ccl_labeled.pgm"), labeled);
            vision::save_pgm(vision::join_path(art_dir, stem + "_boxes.pgm"), box_overlay);
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_gt_mask.pgm");
            written.push_back(stem + "_proposal_mask.pgm");
            written.push_back(stem + "_edges.pgm");
            written.push_back(stem + "_saliency.pgm");
            written.push_back(stem + "_ccl_labeled.pgm");
            written.push_back(stem + "_boxes.pgm");
            written.push_back(stem + "_detected_bboxes.json");
            report.n_outputs += 1 + static_cast<int>(ccl.components.size());
            (void)hit;
        }
        report.notes.push_back("CCL on GT masks (score vs boxes); photo proposals keep dark polarity");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "components.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "ccl_scores.tsv"), score_tsv.str());
        written.insert(written.begin(), "ccl_scores.tsv");
        written.insert(written.begin(), "components.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "coco", [&](const AtomCli& cli) -> int {
        CclAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for ccl atom\n";
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

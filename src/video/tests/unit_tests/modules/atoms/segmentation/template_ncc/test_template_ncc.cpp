#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/template_ncc/template_ncc.hpp"
#include "segmentation/bbox_auto/bbox_auto.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <cmath>
#include <sstream>
#include <vector>

// Atom: COCO luma + GT-mask crop as rigid template → NCC score field + PSR.
class TemplateNccAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"template_ncc"};
    std::ostringstream values_tsv;
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

    static vision::GrayImage crop_box(const vision::GrayImage& im, const vision::Rect& box,
                                      int max_side) {
        const int x = std::clamp(static_cast<int>(box.x), 0, std::max(0, im.width - 1));
        const int y = std::clamp(static_cast<int>(box.y), 0, std::max(0, im.height - 1));
        int w = std::max(8, static_cast<int>(box.w));
        int h = std::max(8, static_cast<int>(box.h));
        w = std::min(w, im.width - x);
        h = std::min(h, im.height - y);
        if (std::max(w, h) > max_side) {
            const float s = static_cast<float>(max_side) / static_cast<float>(std::max(w, h));
            w = std::max(8, static_cast<int>(w * s));
            h = std::max(8, static_cast<int>(h * s));
        }
        vision::GrayImage t;
        t.width = w;
        t.height = h;
        t.data.resize(static_cast<size_t>(w * h));
        for (int yy = 0; yy < h; ++yy) {
            for (int xx = 0; xx < w; ++xx) {
                t.at(xx, yy) = im.at(std::min(x + xx, im.width - 1), std::min(y + yy, im.height - 1));
            }
        }
        return t;
    }

    void run(const std::string& art_dir) {
        print_banner("run TemplateNCC → luma self-match + PSR");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tself_ncc\tpeak_ncc\tpeak_x\tpeak_y\tpsr\ttw\tth\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            // RAW_RGB path: template from GT mask bbox on luma (prep inside test).
            const auto luma = mission_luma_image(ps, sample.image);
            auto mask_img = binarize_mask(mission_mask_image(ps, sample.image));
            if (mask_img.width != luma.width || mask_img.height != luma.height) {
                mask_img = binarize_mask(luma);  // fallback: threshold luma
            }
            auto ccl = vision::ConnectedComponentLabeler::label(mask_img);
            vision::Rect box = contour::BBoxAuto::from_mask(to_contour(mask_img));
            if (!ccl.components.empty()) {
                // Prefer largest component crop.
                int best = 0;
                for (size_t i = 1; i < ccl.components.size(); ++i) {
                    if (ccl.components[i].area > ccl.components[static_cast<size_t>(best)].area) {
                        best = static_cast<int>(i);
                    }
                }
                box = ccl.components[static_cast<size_t>(best)].bbox;
            }
            const vision::GrayImage templ = crop_box(luma, box, 32);
            const int ox =
                std::clamp(static_cast<int>(box.x), 0, std::max(0, luma.width - templ.width));
            const int oy =
                std::clamp(static_cast<int>(box.y), 0, std::max(0, luma.height - templ.height));
            const float self = vision::TemplateNCC::ncc(luma, templ, ox, oy);

            const int step = 4;
            const int gw = std::max(1, (luma.width - templ.width) / step + 1);
            const int gh = std::max(1, (luma.height - templ.height) / step + 1);
            std::vector<float> scores(static_cast<size_t>(gw * gh), -1.0f);
            float peak = -2.0f;
            int px = ox, py = oy, pidx = 0;
            for (int gy = 0; gy < gh; ++gy) {
                for (int gx = 0; gx < gw; ++gx) {
                    const int x = gx * step;
                    const int y = gy * step;
                    if (x + templ.width > luma.width || y + templ.height > luma.height) {
                        continue;
                    }
                    const float s = vision::TemplateNCC::ncc(luma, templ, x, y);
                    const int idx = gy * gw + gx;
                    scores[static_cast<size_t>(idx)] = s;
                    if (s > peak) {
                        peak = s;
                        px = x;
                        py = y;
                        pidx = idx;
                    }
                }
            }
            // PSR over sidelobes excluding a neighborhood around the peak.
            double sum = 0.0, sum2 = 0.0;
            int n = 0;
            const int pgx = pidx % gw;
            const int pgy = pidx / gw;
            for (int gy = 0; gy < gh; ++gy) {
                for (int gx = 0; gx < gw; ++gx) {
                    if (std::abs(gx - pgx) <= 2 && std::abs(gy - pgy) <= 2) {
                        continue;
                    }
                    const float v = scores[static_cast<size_t>(gy * gw + gx)];
                    if (v < -1.5f) {
                        continue;
                    }
                    sum += v;
                    sum2 += static_cast<double>(v) * v;
                    ++n;
                }
            }
            float psr = 0.0f;
            if (n > 1) {
                const double mean = sum / n;
                const double var = std::max(1e-12, sum2 / n - mean * mean);
                psr = static_cast<float>((peak - mean) / std::sqrt(var));
            }

            std::cout << "  " << sample.row.file << "  self=" << self << "  peak=" << peak
                      << "  psr=" << psr << " @(" << px << "," << py << ")\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << self << '\t' << peak
                       << '\t' << px << '\t' << py << '\t' << psr << '\t' << templ.width << '\t'
                       << templ.height << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::GrayImage heat;
            heat.width = luma.width;
            heat.height = luma.height;
            heat.data.assign(static_cast<size_t>(heat.width * heat.height), 0);
            for (int gy = 0; gy < gh; ++gy) {
                for (int gx = 0; gx < gw; ++gx) {
                    const float s = scores[static_cast<size_t>(gy * gw + gx)];
                    if (s < -1.5f) {
                        continue;
                    }
                    const int x = gx * step;
                    const int y = gy * step;
                    const uint8_t v =
                        static_cast<uint8_t>(std::clamp((s + 1.0f) * 127.5f, 0.0f, 255.0f));
                    for (int yy = y; yy < y + templ.height && yy < heat.height; ++yy) {
                        for (int xx = x; xx < x + templ.width && xx < heat.width; ++xx) {
                            heat.at(xx, yy) = std::max(heat.at(xx, yy), v);
                        }
                    }
                }
            }
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_template.pgm"), templ);
            vision::save_pgm(vision::join_path(art_dir, stem + "_ncc_heatmap.pgm"), heat);
            vision::save_pgm(vision::join_path(art_dir, stem + "_peak.pgm"),
                             overlay_rect(luma, static_cast<float>(px), static_cast<float>(py),
                                          static_cast<float>(templ.width),
                                          static_cast<float>(templ.height)));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_template.pgm");
            written.push_back(stem + "_ncc_heatmap.pgm");
            written.push_back(stem + "_peak.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("COCO luma: GT-mask crop template → NCC field + PSR");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "ncc.tsv"), values_tsv.str());
        written.insert(written.begin(), "ncc.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "coco", [&](const AtomCli& cli) -> int {
        TemplateNccAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for template_ncc atom\n";
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

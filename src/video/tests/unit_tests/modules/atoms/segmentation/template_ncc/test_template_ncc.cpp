#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/template_ncc/template_ncc.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <sstream>

// Atom demo: screen image in → NCC of a cropped component template, scan peak out.
class TemplateNccAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"template_ncc"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 128);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    static vision::GrayImage crop_box(const vision::GrayImage& im, const vision::Rect& box, int max_side) {
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
                t.at(xx, yy) = im.at(x + xx, y + yy);
            }
        }
        return t;
    }

    void run(const std::string& art_dir) {
        print_banner("run TemplateNCC → self-match + coarse scan peak");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tself_ncc\tpeak_ncc\tpeak_x\tpeak_y\ttw\tth\n";
        for (const auto& sample : samples) {
            auto ccl = vision::ConnectedComponentLabeler::label(sample.image);
            vision::Rect box{0, 0, static_cast<float>(std::min(32, sample.image.width)),
                             static_cast<float>(std::min(32, sample.image.height))};
            if (!ccl.components.empty()) {
                const auto& c = ccl.components.front();
                box = c.bbox;
            }
            const vision::GrayImage templ = crop_box(sample.image, box, 32);
            const int ox = std::clamp(static_cast<int>(box.x), 0, std::max(0, sample.image.width - templ.width));
            const int oy = std::clamp(static_cast<int>(box.y), 0, std::max(0, sample.image.height - templ.height));
            const float self = vision::TemplateNCC::ncc(sample.image, templ, ox, oy);

            float peak = self;
            int px = ox, py = oy;
            const int step = 8;
            for (int y = 0; y + templ.height <= sample.image.height; y += step) {
                for (int x = 0; x + templ.width <= sample.image.width; x += step) {
                    const float s = vision::TemplateNCC::ncc(sample.image, templ, x, y);
                    if (s > peak) {
                        peak = s;
                        px = x;
                        py = y;
                    }
                }
            }
            std::cout << "  " << sample.row.file << "  self=" << self << "  peak=" << peak << " @("
                      << px << "," << py << ")\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << self << '\t' << peak
                       << '\t' << px << '\t' << py << '\t' << templ.width << '\t' << templ.height
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string t_name = stem + "_template.pgm";
            const std::string hit_name = stem + "_peak.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, t_name), templ);
            vision::save_pgm(vision::join_path(art_dir, hit_name),
                             overlay_rect(sample.image, static_cast<float>(px), static_cast<float>(py),
                                          static_cast<float>(templ.width), static_cast<float>(templ.height)));
            vision::GrayImage heat;
            heat.width = sample.image.width;
            heat.height = sample.image.height;
            heat.data.assign(static_cast<size_t>(heat.width * heat.height), 0);
            for (int y = 0; y + templ.height <= sample.image.height; y += step) {
                for (int x = 0; x + templ.width <= sample.image.width; x += step) {
                    const float s = vision::TemplateNCC::ncc(sample.image, templ, x, y);
                    const uint8_t v = static_cast<uint8_t>(std::clamp((s + 1.0f) * 127.5f, 0.0f, 255.0f));
                    for (int yy = y; yy < y + templ.height && yy < heat.height; ++yy) {
                        for (int xx = x; xx < x + templ.width && xx < heat.width; ++xx) {
                            heat.at(xx, yy) = std::max(heat.at(xx, yy), v);
                        }
                    }
                }
            }
            const std::string heat_name = stem + "_ncc_heatmap.pgm";
            vision::save_pgm(vision::join_path(art_dir, heat_name), heat);
            written.push_back(heat_name);
            written.push_back(in_name);
            written.push_back(t_name);
            written.push_back(hit_name);
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: ncc.tsv, ncc_heatmap.pgm, *_peak.pgm");
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
            std::cerr << "no inputs for " << cli.dataset << " under " << cli.data_root << "\n";
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

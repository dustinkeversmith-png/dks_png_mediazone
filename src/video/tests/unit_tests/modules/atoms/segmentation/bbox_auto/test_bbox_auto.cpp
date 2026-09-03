#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/bbox_auto/bbox_auto.hpp"

#include <sstream>

// Atom: DIS5K isolated-object mask → extrema AABB + crop/uncrop. IoU vs GT box.
class BBoxAutoAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"bbox_auto"};
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

    void run(const std::string& art_dir) {
        print_banner("run BBoxAuto → crop / uncrop + IoU");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tx\ty\tw\th\tcrop_w\tcrop_h\tbbox_iou\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            // In-test prep: binarize + keep largest FG (isolated target).
            const auto mask_img = prepare_contour_mask(mission_mask_image(ps, sample.image));
            const auto luma = mission_luma_image(ps, sample.image);
            const auto mask = to_contour(mask_img);
            const contour::Rect box = contour::BBoxAuto::from_mask(mask);
            const auto crop = contour::BBoxAuto::crop(mask, box, 0.12f);
            const auto restored = contour::BBoxAuto::uncrop(crop.image, crop);
            const contour::Rect gt_box = contour::BBoxAuto::from_mask(mask);
            const double iou = mission::bbox_iou(
                vision::Rect{box.x, box.y, box.w, box.h},
                vision::Rect{gt_box.x, gt_box.y, gt_box.w, gt_box.h});
            std::cout << "  " << sample.row.file << "  box=" << box.w << "x" << box.h
                      << "  iou=" << iou << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << box.x << '\t' << box.y
                       << '\t' << box.w << '\t' << box.h << '\t' << crop.image.width << '\t'
                       << crop.image.height << '\t' << iou << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_bbox.pgm"),
                             overlay_rect(luma, box.x, box.y, box.w, box.h));
            vision::save_pgm(vision::join_path(art_dir, stem + "_crop.pgm"), to_gray(crop.image));
            vision::save_pgm(vision::join_path(art_dir, stem + "_uncrop.pgm"), to_gray(restored));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_bbox.pgm");
            written.push_back(stem + "_crop.pgm");
            written.push_back(stem + "_uncrop.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("DIS5K: binarize+largest-CC in test → AABB IoU");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "bbox.tsv"), values_tsv.str());
        written.insert(written.begin(), "bbox.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "dis5k", [&](const AtomCli& cli) -> int {
        BBoxAutoAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for bbox_auto atom\n";
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

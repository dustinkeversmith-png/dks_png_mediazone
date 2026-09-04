#ifndef BSDS500_PROVIDER_HPP
#define BSDS500_PROVIDER_HPP

#include "../dataset_provider.hpp"
#include "../io/mat_io.hpp"

namespace datasets {

class BSDS500Provider : public DatasetProvider {
public:
    using DatasetProvider::DatasetProvider;

    bool initialize() override {
        entries_.clear();
        static const char* splits[] = {"BSR/BSDS500/data/images/test", "BSR/BSDS500/data/images/train",
                                       "BSR/BSDS500/data/images/val", "bench/data/images"};
        for (const char* split : splits) {
            const auto dir = vision::join_path(root_dir_.string(), split);
            auto files = vision::list_image_files(dir, false);
            for (const auto& f : files) {
                entries_.push_back(f);
            }
            if (!entries_.empty()) {
                images_root_ = dir;
                break;
            }
        }
        if (entries_.empty()) {
            entries_.push_back("synthetic:edge");
        }
        return !entries_.empty();
    }

    size_t size() const override { return entries_.size(); }

    VisionSample load_sample(size_t idx) const override {
        VisionSample s;
        if (idx >= entries_.size()) {
            return s;
        }
        const auto& entry = entries_[idx];
        if (entry.rfind("synthetic:", 0) == 0) {
            auto im = math::make_gray(128, 128, 30);
            for (int y = 20; y < 108; ++y) {
                im.at(64, y) = 255;
                im.at(65, y) = 255;
            }
            for (int x = 20; x < 108; ++x) {
                im.at(x, 64) = 255;
            }
            s = from_mask("bsds500_synthetic", im);
            s.boundary = im;
            s.luma = im;
            s.rgb = im;
            return s;
        }
        s.image_path = entry;
        s.id = stem_of(std::filesystem::path(entry));
        s.label = s.id;
        s.rgb = vision::load_rgb_png(entry);
        s.luma = vision::rgb_to_luma(s.rgb);

        // Prefer .mat groundTruth next to images (…/groundTruth/<split>/<id>.mat).
        std::filesystem::path img_path(entry);
        std::filesystem::path gt_mat =
            img_path.parent_path().parent_path().parent_path() / "groundTruth" /
            img_path.parent_path().filename() / (img_path.stem().string() + ".mat");
        if (!std::filesystem::exists(gt_mat)) {
            gt_mat = root_dir_ / "BSR/BSDS500/data/groundTruth" / img_path.parent_path().filename() /
                     (img_path.stem().string() + ".mat");
        }
        if (std::filesystem::exists(gt_mat)) {
            s.gt_path = gt_mat.string();
            const auto bound_png = gt_mat.parent_path() / (gt_mat.stem().string() + "_boundary.png");
            const auto seg_png = gt_mat.parent_path() / (gt_mat.stem().string() + "_seg.png");
            if (std::filesystem::exists(bound_png)) {
                s.boundary = vision::load_gray_png(bound_png.string());
            } else {
                s.boundary = load_mat_boundaries(gt_mat.string());
            }
            if (std::filesystem::exists(seg_png)) {
                s.mask = vision::load_gray_png(seg_png.string());
                s.boxes = boxes_from_label_map(s.mask);
            }
        }
        if (s.boundary.empty()) {
            const auto png_gt = vision::join_path(root_dir_.string(), "bench/data/png");
            if (auto gt = find_paired_file(std::filesystem::path(entry), std::filesystem::path(png_gt),
                                           {".png"})) {
                s.gt_path = gt->string();
                s.boundary = vision::load_gray_png(s.gt_path);
            }
        }
        return s;
    }

private:
    std::string images_root_;
};

}  // namespace datasets

#endif  // BSDS500_PROVIDER_HPP

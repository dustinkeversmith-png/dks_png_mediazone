#ifndef DIS5K_PROVIDER_HPP
#define DIS5K_PROVIDER_HPP

#include "../dataset_provider.hpp"

namespace datasets {

class DIS5KProvider : public DatasetProvider {
public:
    using DatasetProvider::DatasetProvider;

    bool initialize() override {
        entries_.clear();
        const auto images = vision::join_path(root_dir_.string(), "images");
        const auto masks = vision::join_path(root_dir_.string(), "masks");
        auto files = vision::list_image_files(images, false);
        if (files.empty()) {
            files = vision::list_image_files(root_dir_.string(), true);
        }
        for (const auto& f : files) {
            entries_.push_back(f);
        }
        if (entries_.empty()) {
            entries_.push_back("synthetic:donut");
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
            s = from_mask("dis5k_synthetic", make_synthetic_donut(128, 128));
            s.image_path = entry;
            return s;
        }
        s.image_path = entry;
        s.id = stem_of(std::filesystem::path(entry));
        s.label = s.id;
        s.rgb = vision::load_rgb_png(entry);
        s.luma = vision::rgb_to_luma(s.rgb);
        const auto masks = vision::join_path(root_dir_.string(), "masks");
        if (auto gt = find_paired_file(std::filesystem::path(entry), std::filesystem::path(masks), {".png"})) {
            s.gt_path = gt->string();
            s.mask = vision::load_gray_png(s.gt_path);
        }
        return s;
    }
};

}  // namespace datasets

#endif  // DIS5K_PROVIDER_HPP

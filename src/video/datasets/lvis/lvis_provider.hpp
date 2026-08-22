#ifndef LVIS_PROVIDER_HPP
#define LVIS_PROVIDER_HPP

#include "../dataset_provider.hpp"
#include "../coco/coco_index.hpp"

namespace datasets {

class LVISProvider : public DatasetProvider {
public:
    using DatasetProvider::DatasetProvider;

    bool initialize() override {
        entries_.clear();
        const auto images = vision::join_path(root_dir_.string(), "lvis_subset_images");
        auto files = vision::list_image_files(images, true);
        if (files.empty()) {
            files = vision::list_image_files(root_dir_.string(), true);
        }
        for (const auto& f : files) {
            entries_.push_back(f);
        }
        ann_path_ = vision::join_path(root_dir_.string(), "lvis_v1_val_500mb.json");
        if (!std::filesystem::exists(ann_path_)) {
            ann_path_ = vision::join_path(root_dir_.string(), "lvis_v1_val.json");
        }
        if (std::filesystem::exists(ann_path_)) {
            index_ = CocoAnnIndex::load(ann_path_);
        }
        if (entries_.empty()) {
            entries_.push_back("synthetic:lvis");
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
            s = from_mask("lvis_synthetic", make_synthetic_disk(128, 128, 70, 70, 25));
            s.rgb = s.mask;
            s.luma = s.mask;
            return s;
        }
        s.image_path = entry;
        s.id = stem_of(std::filesystem::path(entry));
        s.label = s.id;
        s.rgb = vision::load_rgb_png(entry);
        s.luma = vision::rgb_to_luma(s.rgb);
        s.mask = index_.mask_for_stem(s.id, s.rgb.width, s.rgb.height);
        return s;
    }

private:
    std::string ann_path_;
    CocoAnnIndex index_;
};

}  // namespace datasets

#endif  // LVIS_PROVIDER_HPP

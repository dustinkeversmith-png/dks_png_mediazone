#ifndef COCO_PROVIDER_HPP
#define COCO_PROVIDER_HPP

#include "../dataset_provider.hpp"
#include "coco_index.hpp"

namespace datasets {

class COCOProvider : public DatasetProvider {
public:
    using DatasetProvider::DatasetProvider;

    bool initialize() override {
        entries_.clear();
        const auto images = vision::join_path(root_dir_.string(), "val2017");
        auto files = vision::list_image_files(images, false);
        for (const auto& f : files) {
            entries_.push_back(f);
        }
        ann_path_ = vision::join_path(root_dir_.string(), "annotations/instances_val2017.json");
        if (!std::filesystem::exists(ann_path_)) {
            ann_path_ = vision::join_path(root_dir_.string(), "instances_val2017_subset.json");
        }
        if (std::filesystem::exists(ann_path_)) {
            index_ = CocoAnnIndex::load(ann_path_);
        }
        if (entries_.empty()) {
            entries_.push_back("synthetic:coco");
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
            s = from_mask("coco_synthetic", make_synthetic_disk(128, 128, 64, 64, 30));
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

#endif  // COCO_PROVIDER_HPP

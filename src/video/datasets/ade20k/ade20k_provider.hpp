#ifndef ADE20K_PROVIDER_HPP
#define ADE20K_PROVIDER_HPP

#include "../dataset_provider.hpp"

namespace datasets {

class ADE20KProvider : public DatasetProvider {
public:
    using DatasetProvider::DatasetProvider;

    bool initialize() override {
        entries_.clear();
        const auto images = vision::join_path(root_dir_.string(), "images");
        const auto ann = vision::join_path(root_dir_.string(), "annotations");
        auto files = vision::list_image_files(images, true);
        if (files.empty()) {
            files = vision::list_image_files(root_dir_.string(), true);
        }
        for (const auto& f : files) {
            entries_.push_back(f);
        }
        ann_dir_ = ann;
        if (entries_.empty()) {
            entries_.push_back("synthetic:instances");
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
            auto a = make_synthetic_disk(128, 128, 40, 64, 24);
            auto b = make_synthetic_disk(128, 128, 88, 64, 20);
            for (size_t i = 0; i < a.data.size(); ++i) {
                if (b.data[i] > 0) {
                    a.data[i] = 200;
                }
            }
            s = from_mask("ade20k_synthetic", a);
            s.rgb = vision::load_rgb_png(s.image_path);  // empty ok
            s.rgb = a;
            s.luma = a;
            return s;
        }
        s.image_path = entry;
        s.id = stem_of(std::filesystem::path(entry));
        s.label = s.id;
        s.rgb = vision::load_rgb_png(entry);
        s.luma = vision::rgb_to_luma(s.rgb);
        if (auto gt = find_paired_file(std::filesystem::path(entry), std::filesystem::path(ann_dir_), {".png"})) {
            s.gt_path = gt->string();
            s.mask = vision::load_gray_png(s.gt_path);
        }
        return s;
    }

private:
    std::string ann_dir_;
};

}  // namespace datasets

#endif  // ADE20K_PROVIDER_HPP

#ifndef SBD_PROVIDER_HPP
#define SBD_PROVIDER_HPP

#include "../dataset_provider.hpp"
#include "../../modules/math/geometry.hpp"

namespace datasets {

class SBDProvider : public DatasetProvider {
public:
    using DatasetProvider::DatasetProvider;

    bool initialize() override {
        entries_.clear();
        const auto img = vision::join_path(root_dir_.string(), "dataset/img");
        const auto inst = vision::join_path(root_dir_.string(), "dataset/inst");
        const auto sdb_img = vision::join_path(root_dir_.string(), "../SDB/dataset/img");
        auto files = vision::list_image_files(img, false);
        img_dir_ = img;
        inst_dir_ = inst;
        if (files.empty()) {
            files = vision::list_image_files(sdb_img, false);
            if (!files.empty()) {
                img_dir_ = sdb_img;
                inst_dir_ = vision::join_path(root_dir_.string(), "../SDB/dataset/inst");
            }
        }
        for (const auto& f : files) {
            entries_.push_back(f);
        }
        if (entries_.empty()) {
            entries_.push_back("synthetic:star");
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
            auto im = math::make_gray(128, 128, 0);
            const math::Vec2 c{64, 64};
            for (int i = 0; i < 5; ++i) {
                const float ang = static_cast<float>(i) * 2.0f * math::kPi / 5.0f;
                const math::Vec2 p{c.x + 40.0f * std::cos(ang), c.y + 40.0f * std::sin(ang)};
                auto blob = make_synthetic_disk(128, 128, static_cast<int>(p.x), static_cast<int>(p.y), 12);
                for (size_t j = 0; j < im.data.size(); ++j) {
                    if (blob.data[j] > 0) {
                        im.data[j] = 255;
                    }
                }
            }
            s = from_mask("sbd_synthetic", im);
            s.rgb = im;
            s.luma = im;
            return s;
        }
        s.image_path = entry;
        s.id = stem_of(std::filesystem::path(entry));
        s.label = s.id;
        s.rgb = vision::load_rgb_png(entry);
        s.luma = vision::rgb_to_luma(s.rgb);
        if (auto gt = find_paired_file(std::filesystem::path(entry), std::filesystem::path(inst_dir_), {".png", ".mat"})) {
            s.gt_path = gt->string();
            if (gt->extension() == ".png") {
                s.mask = vision::load_gray_png(s.gt_path);
            }
        }
        return s;
    }

private:
    std::string img_dir_;
    std::string inst_dir_;
};

}  // namespace datasets

#endif  // SBD_PROVIDER_HPP

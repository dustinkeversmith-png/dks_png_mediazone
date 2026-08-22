#pragma once

#include "../io/vision_io.hpp"
#include "../../modules/math/vision_types.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace datasets {

struct VisionSample {
    std::string id;
    std::string label;
    std::string image_path;
    std::string gt_path;
    math::ImageBuffer rgb;
    math::ImageBuffer luma;
    math::ImageBuffer mask;
    math::ImageBuffer boundary;
};

class DatasetProvider {
public:
    explicit DatasetProvider(std::filesystem::path root_dir) : root_dir_(std::move(root_dir)) {}
    virtual ~DatasetProvider() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;
    virtual VisionSample load_sample(size_t idx) const = 0;
    virtual math::ImageBuffer get_ground_truth(size_t idx) const {
        return load_sample(idx).mask;
    }

    const std::filesystem::path& root() const { return root_dir_; }

protected:
    std::filesystem::path root_dir_;
    mutable std::vector<std::string> entries_;

    static std::string stem_of(const std::filesystem::path& p) {
        return p.stem().string();
    }

    static math::ImageBuffer make_synthetic_disk(int w, int h, int cx, int cy, int r) {
        auto im = math::make_gray(w, h, 0);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int dx = x - cx;
                const int dy = y - cy;
                if (dx * dx + dy * dy <= r * r) {
                    im.at(x, y) = 255;
                }
            }
        }
        return im;
    }

    static math::ImageBuffer make_synthetic_donut(int w, int h) {
        auto outer = make_synthetic_disk(w, h, w / 2, h / 2, w / 3);
        auto inner = make_synthetic_disk(w, h, w / 2, h / 2, w / 6);
        for (size_t i = 0; i < outer.data.size(); ++i) {
            if (inner.data[i] > 0) {
                outer.data[i] = 0;
            }
        }
        return outer;
    }

    static VisionSample from_mask(const std::string& id, const math::ImageBuffer& mask) {
        VisionSample s;
        s.id = id;
        s.label = id;
        s.mask = mask;
        s.luma = mask;
        s.rgb = mask;
        s.rgb.channels = 1;
        return s;
    }
};

inline std::filesystem::path resolve_dataset_root(const std::filesystem::path& vision_root,
                                                  const std::initializer_list<const char*> names) {
    for (const char* n : names) {
        const auto p = vision_root / n;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    for (const char* n : names) {
        const std::string lower = n;
        for (const auto& entry : std::filesystem::directory_iterator(vision_root)) {
            if (!entry.is_directory()) {
                continue;
            }
            std::string name = entry.path().filename().string();
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (name == lower) {
                return entry.path();
            }
        }
    }
    return vision_root / *names.begin();
}

inline std::optional<std::filesystem::path> find_paired_file(const std::filesystem::path& image_path,
                                                             const std::filesystem::path& gt_dir,
                                                             const std::initializer_list<const char*> exts) {
    const auto stem = image_path.stem();
    for (const char* suffix : {"", "_mask", "_gt", "-mask", "_label"}) {
        for (const char* ext : exts) {
            const auto direct = gt_dir / (stem.string() + suffix + ext);
            if (std::filesystem::exists(direct)) {
                return direct;
            }
        }
    }
    if (!std::filesystem::exists(gt_dir)) {
        return std::nullopt;
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(gt_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto estem = entry.path().stem().string();
        if (estem == stem.string() || estem == stem.string() + "_mask" || estem.find(stem.string()) == 0) {
            return entry.path();
        }
    }
    return std::nullopt;
}

}  // namespace datasets

#pragma once

#include "ade20k/ade20k_provider.hpp"
#include "bsds500/bsds500_provider.hpp"
#include "coco/coco_provider.hpp"
#include "dis5k/dis5k_provider.hpp"
#include "lvis/lvis_provider.hpp"
#include "sbd/sbd_provider.hpp"
#include "synthetic/synthetic_provider.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace datasets {

inline bool is_synthetic_provider(const std::string& name) {
    return name.rfind("synthetic", 0) == 0 || name == "mpeg7" || name == "kimia99" ||
           name == "char74k" || name == "swedish_leaf";
}

inline std::unique_ptr<DatasetProvider> make_provider(const std::string& name,
                                                       const std::filesystem::path& vision_root) {
    if (is_synthetic_provider(name)) {
        auto p = std::make_unique<SyntheticProvider>(vision_root / "synthetic",
                                                     synthetic_kind_from_name(name));
        p->initialize();
        return p;
    }
    if (name == "ade20k" || name == "ADE20K") {
        auto p = std::make_unique<ADE20KProvider>(resolve_dataset_root(vision_root, {"ADE20K", "ade20k"}));
        p->initialize();
        return p;
    }
    if (name == "bsds500" || name == "BSDS500") {
        auto p = std::make_unique<BSDS500Provider>(resolve_dataset_root(vision_root, {"BSDS500", "bsds500"}));
        p->initialize();
        return p;
    }
    if (name == "coco" || name == "COCO") {
        auto p = std::make_unique<COCOProvider>(resolve_dataset_root(vision_root, {"COCO", "coco"}));
        p->initialize();
        return p;
    }
    if (name == "dis5k" || name == "DIS5K") {
        auto p = std::make_unique<DIS5KProvider>(resolve_dataset_root(vision_root, {"DIS5K", "dis5k"}));
        p->initialize();
        return p;
    }
    if (name == "lvis" || name == "LVIS") {
        auto p = std::make_unique<LVISProvider>(resolve_dataset_root(vision_root, {"LVIS", "lvis"}));
        p->initialize();
        return p;
    }
    if (name == "sbd" || name == "SBD" || name == "SDB") {
        auto p = std::make_unique<SBDProvider>(resolve_dataset_root(vision_root, {"SBD", "SDB", "sbd"}));
        p->initialize();
        return p;
    }
    throw std::runtime_error("unknown dataset provider: " + name);
}

}  // namespace datasets

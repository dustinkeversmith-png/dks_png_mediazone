#pragma once

#include "../../math/vision_types.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace vision {

// STR-packed 2D R-tree. Optional libspatialindex wrap is behind VISION_HAS_SPATIALINDEX.
class RTree2D {
public:
    struct Item {
        int id = 0;
        Rect box;
    };

    void insert(int id, const Rect& box) {
        items_.push_back({id, box});
        built_ = false;
    }

    void build() {
        nodes_.clear();
        root_ = -1;
        if (items_.empty()) {
            built_ = true;
            return;
        }
        std::vector<int> idx(items_.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = static_cast<int>(i);
        }
        root_ = build_node(idx, 0);
        built_ = true;
    }

    std::vector<int> query_intersects(const Rect& q) {
        if (!built_) {
            build();
        }
        std::vector<int> out;
        if (root_ >= 0) {
            query_node(root_, q, out);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    std::vector<int> brute_intersects(const Rect& q) const {
        std::vector<int> out;
        for (const auto& it : items_) {
            if (it.box.intersects(q)) {
                out.push_back(it.id);
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    size_t size() const { return items_.size(); }

private:
    static constexpr int kFanout = 8;

    struct Node {
        Rect box;
        int item_index = -1;
        std::vector<int> children;
        bool leaf = true;
    };

    std::vector<Item> items_;
    std::vector<Node> nodes_;
    int root_ = -1;
    bool built_ = false;

    static Rect union_rect(const Rect& a, const Rect& b) {
        const float x = std::min(a.x, b.x);
        const float y = std::min(a.y, b.y);
        const float x1 = std::max(a.x1(), b.x1());
        const float y1 = std::max(a.y1(), b.y1());
        return {x, y, x1 - x, y1 - y};
    }

    int build_node(std::vector<int>& idxs, int depth) {
        Node node;
        if (idxs.size() <= static_cast<size_t>(kFanout) || depth > 12) {
            node.leaf = true;
            node.box = items_[static_cast<size_t>(idxs[0])].box;
            for (int id : idxs) {
                node.children.push_back(id);
                node.box = union_rect(node.box, items_[static_cast<size_t>(id)].box);
            }
            nodes_.push_back(node);
            return static_cast<int>(nodes_.size() - 1);
        }

        const bool by_x = (depth % 2) == 0;
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            const Rect& ra = items_[static_cast<size_t>(a)].box;
            const Rect& rb = items_[static_cast<size_t>(b)].box;
            const float ca = by_x ? (ra.x + ra.w * 0.5f) : (ra.y + ra.h * 0.5f);
            const float cb = by_x ? (rb.x + rb.w * 0.5f) : (rb.y + rb.h * 0.5f);
            return ca < cb;
        });

        const int slices = static_cast<int>((idxs.size() + kFanout - 1) / kFanout);
        const int chunk = static_cast<int>((idxs.size() + slices - 1) / slices);
        node.leaf = false;
        bool first = true;
        for (int s = 0; s < static_cast<int>(idxs.size()); s += chunk) {
            const int e = std::min(static_cast<int>(idxs.size()), s + chunk);
            std::vector<int> part(idxs.begin() + s, idxs.begin() + e);
            const int child = build_node(part, depth + 1);
            node.children.push_back(child);
            if (first) {
                node.box = nodes_[static_cast<size_t>(child)].box;
                first = false;
            } else {
                node.box = union_rect(node.box, nodes_[static_cast<size_t>(child)].box);
            }
        }
        nodes_.push_back(node);
        return static_cast<int>(nodes_.size() - 1);
    }

    void query_node(int ni, const Rect& q, std::vector<int>& out) const {
        const Node& node = nodes_[static_cast<size_t>(ni)];
        if (!node.box.intersects(q)) {
            return;
        }
        if (node.leaf) {
            for (int item_i : node.children) {
                if (items_[static_cast<size_t>(item_i)].box.intersects(q)) {
                    out.push_back(items_[static_cast<size_t>(item_i)].id);
                }
            }
            return;
        }
        for (int c : node.children) {
            query_node(c, q, out);
        }
    }
};

}  // namespace vision

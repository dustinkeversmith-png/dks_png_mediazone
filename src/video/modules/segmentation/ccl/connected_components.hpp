#pragma once

// Two-pass SAUF-style 8-connected CCL with:
//  - 1px zero padding (no per-neighbor bounds checks)
//  - union-by-rank + path compression
//  - integer min/max extents → consistent bboxes (w/h = max-min+1)
//  - optional post-filters: area gate + IoU/distance box merge
//
// Full YACCLAB (https://github.com/prittt/YACCLAB) is an OpenCV benchmark harness;
// this header is the algorithm used by the unit tests.

#include "../../math/vision_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vision {

class ConnectedComponentLabeler {
public:
    struct Component {
        int label = 0;
        int area = 0;
        Rect bbox;
    };

    struct Result {
        int width = 0;
        int height = 0;
        std::vector<int> labels;  // original (unpadded) size: width*height
        std::vector<Component> components;
    };

    struct Extents {
        int min_x = 0x3fffffff;
        int min_y = 0x3fffffff;
        int max_x = -1;
        int max_y = -1;
    };

    // Label FG pixels (value > thr) with 8-connectivity.
    static Result label(const GrayImage& image, uint8_t thr = 128) {
        Result r;
        r.width = image.width;
        r.height = image.height;
        if (image.empty() || image.width <= 0 || image.height <= 0) {
            return r;
        }

        const int w = image.width;
        const int h = image.height;
        // Pad 1px with zeros so neighbor probes never need bounds checks.
        const int pw = w + 2;
        const int ph = h + 2;
        std::vector<uint8_t> pad(static_cast<size_t>(pw * ph), 0);
        std::vector<int> plab(static_cast<size_t>(pw * ph), 0);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                pad[static_cast<size_t>((y + 1) * pw + (x + 1))] =
                    image.fg(x, y, thr) ? 255 : 0;
            }
        }

        std::vector<int> parent(1, 0);
        std::vector<int> rank(1, 0);
        auto find = [&](int x) {
            int root = x;
            while (parent[static_cast<size_t>(root)] != root) {
                root = parent[static_cast<size_t>(root)];
            }
            while (parent[static_cast<size_t>(x)] != root) {
                const int next = parent[static_cast<size_t>(x)];
                parent[static_cast<size_t>(x)] = root;
                x = next;
            }
            return root;
        };
        auto unite = [&](int a, int b) {
            a = find(a);
            b = find(b);
            if (a == b) {
                return;
            }
            // Union-by-rank (tie → lower id root for stable compacting).
            if (rank[static_cast<size_t>(a)] < rank[static_cast<size_t>(b)]) {
                parent[static_cast<size_t>(a)] = b;
            } else if (rank[static_cast<size_t>(a)] > rank[static_cast<size_t>(b)]) {
                parent[static_cast<size_t>(b)] = a;
            } else if (a < b) {
                parent[static_cast<size_t>(b)] = a;
                ++rank[static_cast<size_t>(a)];
            } else {
                parent[static_cast<size_t>(a)] = b;
                ++rank[static_cast<size_t>(b)];
            }
        };
        auto new_label = [&]() {
            const int id = static_cast<int>(parent.size());
            parent.push_back(id);
            rank.push_back(0);
            return id;
        };

        // SAUF-style scan over padded interior. Neighbors already labeled:
        //   p q r
        //   s x
        // Decision tree: check s, then q, then p/r only when needed.
        for (int y = 1; y <= h; ++y) {
            for (int x = 1; x <= w; ++x) {
                const size_t i = static_cast<size_t>(y * pw + x);
                if (pad[i] == 0) {
                    continue;
                }
                const int s = plab[i - 1];            // left
                const int q = plab[i - static_cast<size_t>(pw)];  // up
                if (s != 0) {
                    plab[i] = s;
                    if (q != 0 && find(s) != find(q)) {
                        unite(s, q);
                    }
                    // up-right may still differ when left+up disagree on diagonals
                    const int rr = plab[i - static_cast<size_t>(pw) + 1];
                    if (rr != 0 && find(plab[i]) != find(rr)) {
                        unite(plab[i], rr);
                    }
                } else if (q != 0) {
                    plab[i] = q;
                    const int p = plab[i - static_cast<size_t>(pw) - 1];  // up-left
                    const int rr = plab[i - static_cast<size_t>(pw) + 1]; // up-right
                    if (p != 0 && find(q) != find(p)) {
                        unite(q, p);
                    }
                    if (rr != 0 && find(plab[i]) != find(rr)) {
                        unite(plab[i], rr);
                    }
                } else {
                    const int p = plab[i - static_cast<size_t>(pw) - 1];
                    const int rr = plab[i - static_cast<size_t>(pw) + 1];
                    if (p != 0) {
                        plab[i] = p;
                        if (rr != 0 && find(p) != find(rr)) {
                            unite(p, rr);
                        }
                    } else if (rr != 0) {
                        plab[i] = rr;
                    } else {
                        plab[i] = new_label();
                    }
                }
            }
        }

        // Compact roots → 1..K and gather integer extents.
        std::vector<int> remap(parent.size(), 0);
        int compacted = 0;
        for (int i = 1; i < static_cast<int>(parent.size()); ++i) {
            const int root = find(i);
            if (remap[static_cast<size_t>(root)] == 0) {
                remap[static_cast<size_t>(root)] = ++compacted;
            }
            remap[static_cast<size_t>(i)] = remap[static_cast<size_t>(root)];
        }

        r.labels.assign(static_cast<size_t>(w * h), 0);
        r.components.assign(static_cast<size_t>(compacted), Component{});
        std::vector<Extents> extents(static_cast<size_t>(compacted));
        for (int i = 0; i < compacted; ++i) {
            r.components[static_cast<size_t>(i)].label = i + 1;
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int raw = plab[static_cast<size_t>((y + 1) * pw + (x + 1))];
                if (raw == 0) {
                    continue;
                }
                const int lab = remap[static_cast<size_t>(raw)];
                r.labels[static_cast<size_t>(y * w + x)] = lab;
                auto& c = r.components[static_cast<size_t>(lab - 1)];
                ++c.area;
                auto& ext = extents[static_cast<size_t>(lab - 1)];
                ext.min_x = std::min(ext.min_x, x);
                ext.min_y = std::min(ext.min_y, y);
                ext.max_x = std::max(ext.max_x, x);
                ext.max_y = std::max(ext.max_y, y);
            }
        }

        for (int i = 0; i < compacted; ++i) {
            const auto& ext = extents[static_cast<size_t>(i)];
            auto& c = r.components[static_cast<size_t>(i)];
            if (ext.max_x < ext.min_x) {
                c.bbox = {0, 0, 0, 0};
                continue;
            }
            c.bbox = {static_cast<float>(ext.min_x), static_cast<float>(ext.min_y),
                      static_cast<float>(ext.max_x - ext.min_x + 1),
                      static_cast<float>(ext.max_y - ext.min_y + 1)};
        }
        return r;
    }

    // Drop speckles; keep components with area >= min_area.
    static Result filter_min_area(const Result& in, int min_area) {
        Result out;
        out.width = in.width;
        out.height = in.height;
        out.labels.assign(in.labels.size(), 0);
        if (min_area <= 1) {
            return in;
        }
        std::vector<int> keep_map(in.components.size() + 1, 0);
        int next = 0;
        for (const auto& c : in.components) {
            if (c.area >= min_area) {
                keep_map[static_cast<size_t>(c.label)] = ++next;
                Component nc = c;
                nc.label = next;
                out.components.push_back(nc);
            }
        }
        for (size_t i = 0; i < in.labels.size(); ++i) {
            const int lab = in.labels[i];
            out.labels[i] = lab > 0 ? keep_map[static_cast<size_t>(lab)] : 0;
        }
        return out;
    }

    // Merge overlapping / nearby boxes (IoU or center-distance), union extents + areas.
    // Conservative defaults: prefer keeping separable instances over glueing a whole scene.
    static std::vector<Component> merge_boxes(const std::vector<Component>& comps,
                                              double iou_thr = 0.40,
                                              float dist_scale = 0.22f) {
        if (comps.empty()) {
            return {};
        }
        struct Node {
            Rect box;
            int area = 0;
            bool alive = true;
        };
        std::vector<Node> nodes;
        nodes.reserve(comps.size());
        for (const auto& c : comps) {
            nodes.push_back({c.bbox, c.area, true});
        }

        auto iou = [](const Rect& a, const Rect& b) -> double {
            return static_cast<double>(a.iou(b));
        };
        auto should_merge = [&](const Node& a, const Node& b) {
            const double j = iou(a.box, b.box);
            if (j >= iou_thr) {
                return true;
            }
            // Strict containment with high overlap of the smaller box.
            const float aa = a.box.area();
            const float ba = b.box.area();
            if (aa <= 0.0f || ba <= 0.0f) {
                return false;
            }
            if (a.box.contains(b.box) && ba / aa >= 0.55f) {
                return true;
            }
            if (b.box.contains(a.box) && aa / ba >= 0.55f) {
                return true;
            }
            const float acx = a.box.x + a.box.w * 0.5f;
            const float acy = a.box.y + a.box.h * 0.5f;
            const float bcx = b.box.x + b.box.w * 0.5f;
            const float bcy = b.box.y + b.box.h * 0.5f;
            const float dx = acx - bcx;
            const float dy = acy - bcy;
            const float dist = std::sqrt(dx * dx + dy * dy);
            const float mean_side =
                0.25f * (a.box.w + a.box.h + b.box.w + b.box.h);
            // Only distance-merge similarly sized neighbors that nearly touch.
            const float size_ratio =
                (std::min(aa, ba) > 0.0f) ? (std::max(aa, ba) / std::min(aa, ba)) : 99.0f;
            if (size_ratio > 4.0f) {
                return false;
            }
            const float gap_x = std::max(0.0f, std::max(a.box.x, b.box.x) -
                                                   std::min(a.box.x1(), b.box.x1()));
            const float gap_y = std::max(0.0f, std::max(a.box.y, b.box.y) -
                                                   std::min(a.box.y1(), b.box.y1()));
            const float gap = std::hypot(gap_x, gap_y);
            return gap <= 2.0f && dist <= dist_scale * mean_side * 2.0f;
        };
        auto union_box = [](const Rect& a, const Rect& b) {
            const float x0 = std::min(a.x, b.x);
            const float y0 = std::min(a.y, b.y);
            const float x1 = std::max(a.x1(), b.x1());
            const float y1 = std::max(a.y1(), b.y1());
            return Rect{x0, y0, x1 - x0, y1 - y0};
        };

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (!nodes[i].alive) {
                    continue;
                }
                for (size_t j = i + 1; j < nodes.size(); ++j) {
                    if (!nodes[j].alive) {
                        continue;
                    }
                    if (!should_merge(nodes[i], nodes[j])) {
                        continue;
                    }
                    nodes[i].box = union_box(nodes[i].box, nodes[j].box);
                    nodes[i].area += nodes[j].area;
                    nodes[j].alive = false;
                    changed = true;
                }
            }
        }

        std::vector<Component> out;
        int lab = 0;
        for (const auto& n : nodes) {
            if (!n.alive) {
                continue;
            }
            Component c;
            c.label = ++lab;
            c.area = n.area;
            c.bbox = n.box;
            out.push_back(c);
        }
        return out;
    }
};

}  // namespace vision

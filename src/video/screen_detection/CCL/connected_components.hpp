#pragma once

// Two-pass SAUF-style 8-connected CCL (the classic YACCLAB CPU family).
// Full YACCLAB (https://github.com/prittt/YACCLAB) is an OpenCV benchmark harness;
// this header is the algorithm used by the unit tests.

#include "../../math/vision_types.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>

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
        std::vector<int> labels;
        std::vector<Component> components;
    };

    static Result label(const GrayImage& image, uint8_t thr = 128) {
        Result r;
        r.width = image.width;
        r.height = image.height;
        const int n = image.width * image.height;
        r.labels.assign(static_cast<size_t>(n), 0);

        std::vector<int> parent(1, 0);
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
            if (a != b) {
                parent[static_cast<size_t>(b)] = a;
            }
        };

        int next_label = 1;
        const int w = image.width;
        const int h = image.height;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (!image.fg(x, y, thr)) {
                    continue;
                }
                int neighbors[4];
                int nn = 0;
                auto take = [&](int xx, int yy) {
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        return;
                    }
                    const int lab = r.labels[static_cast<size_t>(yy * w + xx)];
                    if (lab != 0) {
                        neighbors[nn++] = lab;
                    }
                };
                take(x - 1, y);
                take(x, y - 1);
                take(x - 1, y - 1);
                take(x + 1, y - 1);
                if (nn == 0) {
                    parent.push_back(next_label);
                    r.labels[static_cast<size_t>(y * w + x)] = next_label;
                    ++next_label;
                } else {
                    int m = neighbors[0];
                    for (int i = 1; i < nn; ++i) {
                        m = std::min(m, neighbors[i]);
                    }
                    r.labels[static_cast<size_t>(y * w + x)] = m;
                    for (int i = 0; i < nn; ++i) {
                        unite(m, neighbors[i]);
                    }
                }
            }
        }

        std::vector<int> remap(parent.size(), 0);
        int compacted = 0;
        for (int i = 1; i < static_cast<int>(parent.size()); ++i) {
            const int root = find(i);
            if (remap[static_cast<size_t>(root)] == 0) {
                remap[static_cast<size_t>(root)] = ++compacted;
            }
            remap[static_cast<size_t>(i)] = remap[static_cast<size_t>(root)];
        }

        r.components.assign(static_cast<size_t>(compacted), Component{});
        for (int i = 0; i < compacted; ++i) {
            r.components[static_cast<size_t>(i)].label = i + 1;
            r.components[static_cast<size_t>(i)].bbox = {static_cast<float>(w), static_cast<float>(h), 0, 0};
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int lab = r.labels[static_cast<size_t>(y * w + x)];
                if (lab == 0) {
                    continue;
                }
                lab = remap[static_cast<size_t>(lab)];
                r.labels[static_cast<size_t>(y * w + x)] = lab;
                auto& c = r.components[static_cast<size_t>(lab - 1)];
                ++c.area;
                const float xf = static_cast<float>(x);
                const float yf = static_cast<float>(y);
                const float x1 = c.bbox.x1();
                const float y1 = c.bbox.y1();
                if (c.area == 1) {
                    c.bbox = {xf, yf, 1, 1};
                } else {
                    const float nx = std::min(c.bbox.x, xf);
                    const float ny = std::min(c.bbox.y, yf);
                    const float nx1 = std::max(x1, xf + 1.0f);
                    const float ny1 = std::max(y1, yf + 1.0f);
                    c.bbox = {nx, ny, nx1 - nx, ny1 - ny};
                }
            }
        }
        return r;
    }
};

}  // namespace vision

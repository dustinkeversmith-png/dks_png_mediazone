#pragma once

#include "math/contour_compat.hpp"
#include "sdf/edt/edt.hpp"
#include "filters/sobel/sobel.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

namespace contour {

// Meyer flooding watershed (Beucher / Meyer). Relief is flooded from markers;
// unlabeled pixels where basins meet become watershed lines (label 0).
class Watershed {
public:
    struct Result {
        int width = 0;
        int height = 0;
        int n_basins = 0;
        int n_watershed = 0;
        std::vector<int> labels;
        Field relief;
        std::vector<int> markers;
    };

    static Result flood(const Field& relief, const std::vector<int>& marker_labels) {
        Result r;
        r.width = relief.width;
        r.height = relief.height;
        r.relief = relief;
        r.markers = marker_labels;
        const int n = r.width * r.height;
        r.labels.assign(static_cast<size_t>(n), 0);
        if (n <= 0) {
            return r;
        }

        int max_lab = 0;
        for (int v : marker_labels) {
            max_lab = std::max(max_lab, v);
        }
        r.n_basins = max_lab;

        struct Node {
            float pri;
            int idx;
            bool operator<(const Node& o) const {
                if (pri != o.pri) {
                    return pri > o.pri;
                }
                return idx > o.idx;
            }
        };
        std::priority_queue<Node> pq;
        std::vector<char> queued(static_cast<size_t>(n), 0);
        static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

        auto push_nbrs = [&](int idx) {
            const int x = idx % r.width;
            const int y = idx / r.width;
            for (int k = 0; k < 8; ++k) {
                const int xx = x + dx[k];
                const int yy = y + dy[k];
                if (xx < 0 || yy < 0 || xx >= r.width || yy >= r.height) {
                    continue;
                }
                const int ni = yy * r.width + xx;
                if (r.labels[static_cast<size_t>(ni)] != 0 || queued[static_cast<size_t>(ni)]) {
                    continue;
                }
                queued[static_cast<size_t>(ni)] = 1;
                pq.push({relief.at(xx, yy), ni});
            }
        };

        for (int i = 0; i < n; ++i) {
            if (marker_labels[static_cast<size_t>(i)] > 0) {
                r.labels[static_cast<size_t>(i)] = marker_labels[static_cast<size_t>(i)];
                queued[static_cast<size_t>(i)] = 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (r.labels[static_cast<size_t>(i)] > 0) {
                push_nbrs(i);
            }
        }

        while (!pq.empty()) {
            const Node cur = pq.top();
            pq.pop();
            if (r.labels[static_cast<size_t>(cur.idx)] != 0) {
                continue;
            }
            const int x = cur.idx % r.width;
            const int y = cur.idx / r.width;
            int seen = 0;
            bool mixed = false;
            for (int k = 0; k < 8; ++k) {
                const int xx = x + dx[k];
                const int yy = y + dy[k];
                if (xx < 0 || yy < 0 || xx >= r.width || yy >= r.height) {
                    continue;
                }
                const int lab = r.labels[static_cast<size_t>(yy * r.width + xx)];
                if (lab <= 0) {
                    continue;
                }
                if (seen == 0) {
                    seen = lab;
                } else if (lab != seen) {
                    mixed = true;
                }
            }
            if (seen > 0 && !mixed) {
                r.labels[static_cast<size_t>(cur.idx)] = seen;
                push_nbrs(cur.idx);
            }
        }

        for (int v : r.labels) {
            if (v == 0) {
                ++r.n_watershed;
            }
        }
        return r;
    }

    static std::vector<int> markers_from_minima(const Field& relief, float min_depth = 0.0f) {
        const int w = relief.width;
        const int h = relief.height;
        std::vector<char> is_min(static_cast<size_t>(w * h), 0);
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const float v = relief.at(x, y);
                if (v < min_depth) {
                    continue;
                }
                bool ok = true;
                for (int dy = -1; dy <= 1 && ok; ++dy) {
                    for (int dx = -1; dx <= 1 && ok; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        if (relief.at(x + dx, y + dy) < v) {
                            ok = false;
                        }
                    }
                }
                is_min[static_cast<size_t>(y * w + x)] = ok ? 1 : 0;
            }
        }
        return label_components(is_min, w, h);
    }

    // Classic blob-split: flood -EDT from interior distance maxima.
    static Result from_mask(const ImageBuffer& mask, uint8_t thr = 127) {
        const Field depth = EuclideanDistanceTransform::to_background(mask, thr);
        Field relief = make_field(depth.width, depth.height, 0);
        for (size_t i = 0; i < depth.data.size(); ++i) {
            const bool fg = mask.data[i] > thr;
            relief.data[i] = fg ? -depth.data[i] : 1.0e6f;
        }
        std::vector<char> is_max(static_cast<size_t>(mask.width * mask.height), 0);
        for (int y = 1; y < mask.height - 1; ++y) {
            for (int x = 1; x < mask.width - 1; ++x) {
                if (mask.at(x, y) <= thr || depth.at(x, y) < 1.5f) {
                    continue;
                }
                const float v = depth.at(x, y);
                bool ok = true;
                for (int dy = -1; dy <= 1 && ok; ++dy) {
                    for (int dx = -1; dx <= 1 && ok; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        if (depth.at(x + dx, y + dy) > v) {
                            ok = false;
                        }
                    }
                }
                is_max[static_cast<size_t>(y * mask.width + x)] = ok ? 1 : 0;
            }
        }
        auto markers = label_components(is_max, mask.width, mask.height);
        bool any = false;
        for (int v : markers) {
            if (v > 0) {
                any = true;
                break;
            }
        }
        if (!any) {
            int best = -1;
            float best_d = -1.0f;
            for (int y = 0; y < mask.height; ++y) {
                for (int x = 0; x < mask.width; ++x) {
                    if (mask.at(x, y) > thr && depth.at(x, y) > best_d) {
                        best_d = depth.at(x, y);
                        best = y * mask.width + x;
                    }
                }
            }
            if (best >= 0) {
                markers[static_cast<size_t>(best)] = 1;
            }
        }
        return flood(relief, markers);
    }

    static Result from_gradient(const ImageBuffer& image) {
        SobelFilter sobel;
        sobel.compute(image);
        auto markers = markers_from_minima(sobel.mag, 0.0f);
        return flood(sobel.mag, markers);
    }

private:
    static std::vector<int> label_components(const std::vector<char>& seeds, int w, int h) {
        std::vector<int> lab(static_cast<size_t>(w * h), 0);
        int next = 1;
        static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
        std::vector<int> st;
        for (int i = 0; i < w * h; ++i) {
            if (!seeds[static_cast<size_t>(i)] || lab[static_cast<size_t>(i)] != 0) {
                continue;
            }
            st.clear();
            st.push_back(i);
            lab[static_cast<size_t>(i)] = next;
            while (!st.empty()) {
                const int cur = st.back();
                st.pop_back();
                const int x = cur % w;
                const int y = cur / w;
                for (int k = 0; k < 8; ++k) {
                    const int xx = x + dx[k];
                    const int yy = y + dy[k];
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        continue;
                    }
                    const int ni = yy * w + xx;
                    if (!seeds[static_cast<size_t>(ni)] || lab[static_cast<size_t>(ni)] != 0) {
                        continue;
                    }
                    lab[static_cast<size_t>(ni)] = next;
                    st.push_back(ni);
                }
            }
            ++next;
        }
        return lab;
    }
};

}  // namespace contour

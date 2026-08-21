#pragma once

#include "contour_kit/types.hpp"
#include "color/lab_color_space.hpp"
#include "filters/sobel.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>

namespace contour {

// Simple Linear Iterative Clustering (Achanta et al., PAMI 2012) in CIELAB+xy.
class SlicSuperpixels {
public:
    int k = 64;
    float compactness = 10.0f;
    int iterations = 8;

    struct Result {
        int width = 0;
        int height = 0;
        int n_labels = 0;
        std::vector<int> labels;
    };

    Result segment(const ImageBuffer& im) const {
        Result r;
        r.width = im.width;
        r.height = im.height;
        const int n = im.width * im.height;
        if (n <= 0) {
            return r;
        }
        const int K = std::max(1, std::min(k, n));
        const float S = std::sqrt(static_cast<float>(n) / static_cast<float>(K));
        std::vector<Lab> lab(static_cast<size_t>(n));
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                lab[static_cast<size_t>(y * im.width + x)] = LabColor::at(im, x, y);
            }
        }
        SobelFilter sobel;
        sobel.compute(im);

        struct Center {
            float L = 0, a = 0, b = 0, x = 0, y = 0;
        };
        std::vector<Center> C;
        const int step = std::max(1, static_cast<int>(std::lround(S)));
        for (int y = step / 2; y < im.height; y += step) {
            for (int x = step / 2; x < im.width; x += step) {
                int bx = x, by = y;
                float best = 1e30f;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = std::clamp(x + dx, 0, im.width - 1);
                        const int yy = std::clamp(y + dy, 0, im.height - 1);
                        const float g = sobel.mag.at(xx, yy);
                        if (g < best) {
                            best = g;
                            bx = xx;
                            by = yy;
                        }
                    }
                }
                const Lab p = lab[static_cast<size_t>(by * im.width + bx)];
                C.push_back({p.L, p.a, p.b, static_cast<float>(bx), static_cast<float>(by)});
            }
        }
        if (C.empty()) {
            const Lab p = lab[0];
            C.push_back({p.L, p.a, p.b, 0, 0});
        }

        r.labels.assign(static_cast<size_t>(n), -1);
        const float m = compactness;
        const int win = std::max(2, static_cast<int>(std::ceil(2.0f * S)));
        for (int it = 0; it < iterations; ++it) {
            std::vector<float> dist(static_cast<size_t>(n), 1e30f);
            std::fill(r.labels.begin(), r.labels.end(), -1);
            for (int ci = 0; ci < static_cast<int>(C.size()); ++ci) {
                const Center& c = C[static_cast<size_t>(ci)];
                const int x0 = std::max(0, static_cast<int>(c.x) - win);
                const int x1 = std::min(im.width - 1, static_cast<int>(c.x) + win);
                const int y0 = std::max(0, static_cast<int>(c.y) - win);
                const int y1 = std::min(im.height - 1, static_cast<int>(c.y) + win);
                for (int y = y0; y <= y1; ++y) {
                    for (int x = x0; x <= x1; ++x) {
                        const int i = y * im.width + x;
                        const Lab& p = lab[static_cast<size_t>(i)];
                        const float dc = std::sqrt((p.L - c.L) * (p.L - c.L) + (p.a - c.a) * (p.a - c.a) +
                                                   (p.b - c.b) * (p.b - c.b));
                        const float ds = std::hypot(static_cast<float>(x) - c.x, static_cast<float>(y) - c.y);
                        const float D = dc + (m / std::max(1.0f, S)) * ds;
                        if (D < dist[static_cast<size_t>(i)]) {
                            dist[static_cast<size_t>(i)] = D;
                            r.labels[static_cast<size_t>(i)] = ci;
                        }
                    }
                }
            }
            std::vector<Center> acc(C.size());
            std::vector<int> cnt(C.size(), 0);
            for (int i = 0; i < n; ++i) {
                const int ci = r.labels[static_cast<size_t>(i)];
                if (ci < 0) {
                    continue;
                }
                const Lab& p = lab[static_cast<size_t>(i)];
                acc[static_cast<size_t>(ci)].L += p.L;
                acc[static_cast<size_t>(ci)].a += p.a;
                acc[static_cast<size_t>(ci)].b += p.b;
                acc[static_cast<size_t>(ci)].x += static_cast<float>(i % im.width);
                acc[static_cast<size_t>(ci)].y += static_cast<float>(i / im.width);
                ++cnt[static_cast<size_t>(ci)];
            }
            for (size_t ci = 0; ci < C.size(); ++ci) {
                if (cnt[ci] <= 0) {
                    continue;
                }
                const float inv = 1.0f / static_cast<float>(cnt[ci]);
                C[ci].L = acc[ci].L * inv;
                C[ci].a = acc[ci].a * inv;
                C[ci].b = acc[ci].b * inv;
                C[ci].x = acc[ci].x * inv;
                C[ci].y = acc[ci].y * inv;
            }
        }

        enforce_connectivity(r, im.width, im.height);
        return r;
    }

private:
    static void enforce_connectivity(Result& r, int w, int h) {
        const int n = w * h;
        std::vector<int> out(static_cast<size_t>(n), -1);
        int next = 0;
        static const int dx[4] = {1, -1, 0, 0};
        static const int dy[4] = {0, 0, 1, -1};
        const int thresh = std::max(8, n / 400);
        std::vector<int> st;
        for (int i = 0; i < n; ++i) {
            if (out[static_cast<size_t>(i)] >= 0) {
                continue;
            }
            const int seed_lab = r.labels[static_cast<size_t>(i)];
            st.clear();
            st.push_back(i);
            out[static_cast<size_t>(i)] = next;
            int adj = -1;
            int count = 0;
            while (!st.empty()) {
                const int cur = st.back();
                st.pop_back();
                ++count;
                const int x = cur % w;
                const int y = cur / w;
                for (int k = 0; k < 4; ++k) {
                    const int xx = x + dx[k];
                    const int yy = y + dy[k];
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        continue;
                    }
                    const int ni = yy * w + xx;
                    if (out[static_cast<size_t>(ni)] >= 0) {
                        if (out[static_cast<size_t>(ni)] != next) {
                            adj = out[static_cast<size_t>(ni)];
                        }
                        continue;
                    }
                    if (r.labels[static_cast<size_t>(ni)] == seed_lab) {
                        out[static_cast<size_t>(ni)] = next;
                        st.push_back(ni);
                    } else if (out[static_cast<size_t>(ni)] >= 0) {
                        adj = out[static_cast<size_t>(ni)];
                    }
                }
            }
            if (count < thresh && adj >= 0) {
                for (int j = 0; j < n; ++j) {
                    if (out[static_cast<size_t>(j)] == next) {
                        out[static_cast<size_t>(j)] = adj;
                    }
                }
            } else {
                ++next;
            }
        }
        int mx = 0;
        for (int v : out) {
            mx = std::max(mx, v);
        }
        r.labels.swap(out);
        r.n_labels = mx + 1;
        for (int& v : r.labels) {
            if (v < 0) {
                v = 0;
            }
        }
    }
};

// Felzenszwalb–Huttenlocher 2004 graph-based segmentation on the 4-connected pixel grid.
class FelzenszwalbGridGraph {
public:
    float k_scale = 80.0f;
    float min_size = 32.0f;

    struct Result {
        int width = 0;
        int height = 0;
        int n_labels = 0;
        std::vector<int> labels;
    };

    Result segment(const ImageBuffer& im) const {
        Result r;
        r.width = im.width;
        r.height = im.height;
        const int n = im.width * im.height;
        if (n <= 0) {
            return r;
        }
        struct Edge {
            int a, b;
            float w;
        };
        std::vector<Edge> edges;
        edges.reserve(static_cast<size_t>(n * 2));
        auto weight = [&](int x0, int y0, int x1, int y1) {
            return std::sqrt(LabColor::delta2(LabColor::at(im, x0, y0), LabColor::at(im, x1, y1)));
        };
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                const int i = y * im.width + x;
                if (x + 1 < im.width) {
                    edges.push_back({i, i + 1, weight(x, y, x + 1, y)});
                }
                if (y + 1 < im.height) {
                    edges.push_back({i, i + im.width, weight(x, y, x, y + 1)});
                }
            }
        }
        std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.w < b.w; });

        std::vector<int> parent(static_cast<size_t>(n));
        std::iota(parent.begin(), parent.end(), 0);
        std::vector<int> rank(static_cast<size_t>(n), 0);
        std::vector<int> size(static_cast<size_t>(n), 1);
        std::vector<float> internal(static_cast<size_t>(n), 0);
        const auto find = [&](auto&& self, int x) -> int {
            if (parent[static_cast<size_t>(x)] != x) {
                parent[static_cast<size_t>(x)] = self(self, parent[static_cast<size_t>(x)]);
            }
            return parent[static_cast<size_t>(x)];
        };
        auto unite = [&](int a, int b, float w) {
            a = find(find, a);
            b = find(find, b);
            if (a == b) {
                return;
            }
            if (rank[static_cast<size_t>(a)] < rank[static_cast<size_t>(b)]) {
                std::swap(a, b);
            }
            parent[static_cast<size_t>(b)] = a;
            size[static_cast<size_t>(a)] += size[static_cast<size_t>(b)];
            internal[static_cast<size_t>(a)] = w;
            if (rank[static_cast<size_t>(a)] == rank[static_cast<size_t>(b)]) {
                ++rank[static_cast<size_t>(a)];
            }
        };

        for (const Edge& e : edges) {
            int a = find(find, e.a);
            int b = find(find, e.b);
            if (a == b) {
                continue;
            }
            const float ta = internal[static_cast<size_t>(a)] + k_scale / static_cast<float>(size[static_cast<size_t>(a)]);
            const float tb = internal[static_cast<size_t>(b)] + k_scale / static_cast<float>(size[static_cast<size_t>(b)]);
            if (e.w <= std::min(ta, tb)) {
                unite(e.a, e.b, e.w);
            }
        }
        for (const Edge& e : edges) {
            int a = find(find, e.a);
            int b = find(find, e.b);
            if (a == b) {
                continue;
            }
            if (size[static_cast<size_t>(a)] < min_size || size[static_cast<size_t>(b)] < min_size) {
                unite(e.a, e.b, e.w);
            }
        }

        r.labels.resize(static_cast<size_t>(n));
        std::vector<int> remap(static_cast<size_t>(n), -1);
        int next = 0;
        for (int i = 0; i < n; ++i) {
            const int root = find(find, i);
            if (remap[static_cast<size_t>(root)] < 0) {
                remap[static_cast<size_t>(root)] = next++;
            }
            r.labels[static_cast<size_t>(i)] = remap[static_cast<size_t>(root)];
        }
        r.n_labels = next;
        return r;
    }
};

}  // namespace contour

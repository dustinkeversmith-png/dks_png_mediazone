#pragma once

#include "../contour_kit/types.hpp"
#include "../color/lab_color_space.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace contour {

class GraphCutSegmenter {
public:
    float gamma_n = 28.0f;
    float lambda = 24.0f;
    int refine_iters = 1;
    int k_gmm = 4;
    bool ellipse_fg = true;  // COCO-style blob prior; disable for sparse DIS silhouettes

    ImageBuffer segment(const ImageBuffer& image, const Rect& bbox) const {
        const int w = image.width;
        const int h = image.height;
        const int Npix = w * h;
        if (Npix <= 0) {
            return make_gray(w, h, 0);
        }

        const Rect box = clamp_rect(bbox, w, h);
        std::vector<Lab> labs(static_cast<size_t>(Npix));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                labs[static_cast<size_t>(y * w + x)] = LabColor::at(image, x, y);
            }
        }

        enum : uint8_t { kUnknown = 0, kFG = 1, kBG = 2 };
        std::vector<uint8_t> seed(static_cast<size_t>(Npix), kUnknown);
        std::vector<Lab> bg_pts;
        bg_pts.reserve(static_cast<size_t>(Npix / 4));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                const bool in_box = x >= box.x && x < box.x1() && y >= box.y && y < box.y1();
                if (!in_box) {
                    seed[static_cast<size_t>(i)] = kBG;
                    bg_pts.push_back(labs[static_cast<size_t>(i)]);
                }
            }
        }
        if (static_cast<int>(bg_pts.size()) < std::max(32, Npix / 50)) {
            bg_pts.clear();
            const int border = std::max(6, std::min(w, h) / 20);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (x < border || y < border || x >= w - border || y >= h - border) {
                        const int i = y * w + x;
                        seed[static_cast<size_t>(i)] = kBG;
                        bg_pts.push_back(labs[static_cast<size_t>(i)]);
                    }
                }
            }
        }

        const std::vector<Lab> mu_bg = kmeans(bg_pts, k_gmm);
        if (ellipse_fg) {
            const float cx = box.x + box.w * 0.5f;
            const float cy = box.y + box.h * 0.5f;
            const float hx = std::max(1.0f, box.w * 0.5f);
            const float hy = std::max(1.0f, box.h * 0.5f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (seed[static_cast<size_t>(y * w + x)] == kBG) {
                        continue;
                    }
                    const float fx = (static_cast<float>(x) - cx) / hx;
                    const float fy = (static_cast<float>(y) - cy) / hy;
                    if (fx * fx + fy * fy < 0.50f) {
                        seed[static_cast<size_t>(y * w + x)] = kFG;
                    }
                }
            }
        }
        const float use_gamma = ellipse_fg ? 12.0f : 16.0f;
        const float use_lambda = ellipse_fg ? 14.0f : 28.0f;
        std::vector<float> dbg(static_cast<size_t>(Npix));
        std::vector<float> inside_dbg;
        inside_dbg.reserve(static_cast<size_t>(Npix));
        for (int i = 0; i < Npix; ++i) {
            dbg[static_cast<size_t>(i)] = min_d2(labs[static_cast<size_t>(i)], mu_bg);
            const int x = i % w;
            const int y = i / w;
            if (x >= box.x && x < box.x1() && y >= box.y && y < box.y1()) {
                inside_dbg.push_back(dbg[static_cast<size_t>(i)]);
            }
        }
        float thr = 0;
        if (!inside_dbg.empty()) {
            std::vector<float> ds = inside_dbg;
            const auto ip = std::min(ds.size() - 1, static_cast<size_t>(ds.size() * 0.40));
            std::nth_element(ds.begin(), ds.begin() + static_cast<std::ptrdiff_t>(ip), ds.end());
            thr = ds[ip];
        }

        double mean_d2 = 8.0;
        int nd = 0;
        double acc_d2 = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w - 1; ++x) {
                acc_d2 += LabColor::delta2(labs[static_cast<size_t>(y * w + x)],
                                           labs[static_cast<size_t>(y * w + x + 1)]);
                ++nd;
            }
        }
        if (nd > 0) {
            mean_d2 = std::max(250.0, acc_d2 / nd);
        }

        ImageBuffer mask = make_gray(w, h, 0);
        std::vector<Lab> mu_fg;
        for (int it = 0; it < std::max(1, refine_iters); ++it) {
            std::vector<Lab> fg_pts;
            fg_pts.reserve(static_cast<size_t>(Npix / 4));
            if (it == 0) {
                for (int i = 0; i < Npix; ++i) {
                    const int x = i % w;
                    const int y = i / w;
                    const bool in_box = x >= box.x && x < box.x1() && y >= box.y && y < box.y1();
                    if (in_box && dbg[static_cast<size_t>(i)] >= thr) {
                        fg_pts.push_back(labs[static_cast<size_t>(i)]);
                    }
                }
            } else {
                for (int i = 0; i < Npix; ++i) {
                    if (mask.data[static_cast<size_t>(i)] > 0 && seed[static_cast<size_t>(i)] != kBG) {
                        fg_pts.push_back(labs[static_cast<size_t>(i)]);
                    }
                }
            }
            if (fg_pts.size() < 16) {
                for (int i = 0; i < Npix; ++i) {
                    const int x = i % w;
                    const int y = i / w;
                    if (x >= box.x && x < box.x1() && y >= box.y && y < box.y1()) {
                        fg_pts.push_back(labs[static_cast<size_t>(i)]);
                    }
                }
            }
            mu_fg = kmeans(fg_pts, k_gmm);

            std::vector<uint8_t> tseed = seed;
            if (!ellipse_fg) {
                for (int i = 0; i < Npix; ++i) {
                    if (tseed[static_cast<size_t>(i)] != kUnknown) {
                        continue;
                    }
                    const int x = i % w;
                    const int y = i / w;
                    const bool in_box = x >= box.x && x < box.x1() && y >= box.y && y < box.y1();
                    if (!in_box) {
                        continue;
                    }
                    const float d_bg = dbg[static_cast<size_t>(i)];
                    const float d_fg = min_d2(labs[static_cast<size_t>(i)], mu_fg);
                    if (d_fg * 3.0f < d_bg && d_fg < static_cast<float>(mean_d2) * 4.0f) {
                        tseed[static_cast<size_t>(i)] = kFG;
                    } else if (d_bg * 3.0f < d_fg && d_bg < static_cast<float>(mean_d2) * 4.0f) {
                        tseed[static_cast<size_t>(i)] = kBG;
                    }
                }
            }
            mask = maxflow(w, h, labs, tseed, mu_fg, mu_bg, dbg, mean_d2, use_gamma, use_lambda);
        }
        return mask;
    }

private:
    static Rect clamp_rect(const Rect& b, int w, int h) {
        Rect r = b;
        r.x = std::clamp(r.x, 0.0f, static_cast<float>(std::max(0, w - 1)));
        r.y = std::clamp(r.y, 0.0f, static_cast<float>(std::max(0, h - 1)));
        r.w = std::clamp(r.w, 1.0f, static_cast<float>(w) - r.x);
        r.h = std::clamp(r.h, 1.0f, static_cast<float>(h) - r.y);
        return r;
    }

    static float min_d2(const Lab& p, const std::vector<Lab>& mu) {
        float m = 1.0e30f;
        for (const Lab& c : mu) {
            m = std::min(m, LabColor::delta2(p, c));
        }
        return m;
    }

    static std::vector<Lab> kmeans(const std::vector<Lab>& pts, int K) {
        if (pts.empty()) {
            return {{50.0f, 0.0f, 0.0f}};
        }
        K = std::max(1, std::min(K, static_cast<int>(pts.size())));
        std::vector<Lab> mu(static_cast<size_t>(K));
        mu[0] = pts[0];
        for (int k = 1; k < K; ++k) {
            float best = -1;
            size_t bi = 0;
            for (size_t i = 0; i < pts.size(); ++i) {
                float md = 1.0e30f;
                for (int j = 0; j < k; ++j) {
                    md = std::min(md, LabColor::delta2(pts[i], mu[static_cast<size_t>(j)]));
                }
                if (md > best) {
                    best = md;
                    bi = i;
                }
            }
            mu[static_cast<size_t>(k)] = pts[bi];
        }
        std::vector<int> asg(pts.size(), 0);
        for (int it = 0; it < 10; ++it) {
            for (size_t i = 0; i < pts.size(); ++i) {
                float best = 1.0e30f;
                int bk = 0;
                for (int k = 0; k < K; ++k) {
                    const float d = LabColor::delta2(pts[i], mu[static_cast<size_t>(k)]);
                    if (d < best) {
                        best = d;
                        bk = k;
                    }
                }
                asg[i] = bk;
            }
            std::vector<Lab> acc(static_cast<size_t>(K));
            std::vector<int> cnt(static_cast<size_t>(K), 0);
            for (size_t i = 0; i < pts.size(); ++i) {
                const int k = asg[i];
                acc[static_cast<size_t>(k)].L += pts[i].L;
                acc[static_cast<size_t>(k)].a += pts[i].a;
                acc[static_cast<size_t>(k)].b += pts[i].b;
                ++cnt[static_cast<size_t>(k)];
            }
            for (int k = 0; k < K; ++k) {
                if (cnt[static_cast<size_t>(k)] > 0) {
                    mu[static_cast<size_t>(k)].L = acc[static_cast<size_t>(k)].L / cnt[static_cast<size_t>(k)];
                    mu[static_cast<size_t>(k)].a = acc[static_cast<size_t>(k)].a / cnt[static_cast<size_t>(k)];
                    mu[static_cast<size_t>(k)].b = acc[static_cast<size_t>(k)].b / cnt[static_cast<size_t>(k)];
                }
            }
        }
        return mu;
    }

    ImageBuffer maxflow(int w, int h, const std::vector<Lab>& labs, const std::vector<uint8_t>& seed,
                        const std::vector<Lab>& mu_fg, const std::vector<Lab>& mu_bg,
                        const std::vector<float>& dbg, double mean_d2, float gamma, float lam) const {
        const int Npix = w * h;
        const int S = Npix;
        const int T = Npix + 1;
        struct Edge {
            int to;
            int rev;
            float cap;
        };
        std::vector<std::vector<Edge>> g(static_cast<size_t>(Npix + 2));
        auto add = [&](int a, int b, float c) {
            Edge e1{b, static_cast<int>(g[static_cast<size_t>(b)].size()), c};
            Edge e2{a, static_cast<int>(g[static_cast<size_t>(a)].size()), 0.0f};
            g[static_cast<size_t>(a)].push_back(e1);
            g[static_cast<size_t>(b)].push_back(e2);
        };

        static const int dxk[4] = {1, 0, 1, 1};
        static const int dyk[4] = {0, 1, 1, -1};
        static const float dw[4] = {1, 1, 1.414f, 1.414f};
        const float inv2s = 1.0f / (2.0f * static_cast<float>(mean_d2));
        const float hard = 1.0e6f;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                const Lab& p = labs[static_cast<size_t>(i)];
                for (int k = 0; k < 4; ++k) {
                    const int nx = x + dxk[k];
                    const int ny = y + dyk[k];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
                        continue;
                    }
                    const int j = ny * w + nx;
                    const float d2 = LabColor::delta2(p, labs[static_cast<size_t>(j)]);
                    const float B = (gamma / dw[k]) * std::exp(-d2 * inv2s);
                    add(i, j, B);
                    add(j, i, B);
                }
                if (seed[static_cast<size_t>(i)] == 1) {
                    add(S, i, hard);
                    add(i, T, 0.0f);
                } else if (seed[static_cast<size_t>(i)] == 2) {
                    add(S, i, 0.0f);
                    add(i, T, hard);
                } else {
                    const float dfg = min_d2(p, mu_fg);
                    const float db = dbg[static_cast<size_t>(i)];
                    add(S, i, lam * std::exp(-dfg * inv2s));
                    add(i, T, lam * std::exp(-db * inv2s));
                }
            }
        }

        std::vector<int> level(g.size()), it(g.size());
        auto bfs = [&]() {
            std::fill(level.begin(), level.end(), -1);
            std::queue<int> q;
            level[static_cast<size_t>(S)] = 0;
            q.push(S);
            while (!q.empty()) {
                const int v = q.front();
                q.pop();
                for (const auto& e : g[static_cast<size_t>(v)]) {
                    if (e.cap > 1e-8f && level[static_cast<size_t>(e.to)] < 0) {
                        level[static_cast<size_t>(e.to)] = level[static_cast<size_t>(v)] + 1;
                        q.push(e.to);
                    }
                }
            }
            return level[static_cast<size_t>(T)] >= 0;
        };
        const auto dfs = [&](auto&& self, int v, float f) -> float {
            if (v == T) {
                return f;
            }
            for (int& i = it[static_cast<size_t>(v)]; i < static_cast<int>(g[static_cast<size_t>(v)].size()); ++i) {
                Edge& e = g[static_cast<size_t>(v)][static_cast<size_t>(i)];
                if (e.cap > 1e-8f && level[static_cast<size_t>(e.to)] == level[static_cast<size_t>(v)] + 1) {
                    const float pushed = self(self, e.to, std::min(f, e.cap));
                    if (pushed > 1e-8f) {
                        e.cap -= pushed;
                        g[static_cast<size_t>(e.to)][static_cast<size_t>(e.rev)].cap += pushed;
                        return pushed;
                    }
                }
            }
            return 0.0f;
        };
        while (bfs()) {
            std::fill(it.begin(), it.end(), 0);
            while (dfs(dfs, S, 1.0e9f) > 1e-8f) {
            }
        }

        ImageBuffer mask = make_gray(w, h, 0);
        std::vector<char> vis(g.size(), 0);
        std::queue<int> q;
        q.push(S);
        vis[static_cast<size_t>(S)] = 1;
        while (!q.empty()) {
            const int v = q.front();
            q.pop();
            if (v < Npix) {
                mask.at(v % w, v / w) = 255;
            }
            for (const auto& e : g[static_cast<size_t>(v)]) {
                if (e.cap > 1e-8f && !vis[static_cast<size_t>(e.to)]) {
                    vis[static_cast<size_t>(e.to)] = 1;
                    q.push(e.to);
                }
            }
        }
        return mask;
    }
};

}  // namespace contour

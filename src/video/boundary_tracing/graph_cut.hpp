#pragma once

#include "../contour_kit/types.hpp"
#include "../color/lab_color_space.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>

namespace contour {

class GraphCutSegmenter {
public:
    float gamma_n = 8.0f;
    float lambda_bg = 10.0f;
    float lambda_fg = 10.0f;

    ImageBuffer segment(const ImageBuffer& image, const Rect& bbox) const {
        const int w = image.width;
        const int h = image.height;
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

        static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
        static const float dw[8] = {1, 1, 1, 1, 1.414f, 1.414f, 1.414f, 1.414f};

        double mean_d2 = 0;
        int nd = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const Lab p = LabColor::at(image, x, y);
                if (x + 1 < w) {
                    mean_d2 += LabColor::delta2(p, LabColor::at(image, x + 1, y));
                    ++nd;
                }
                if (y + 1 < h) {
                    mean_d2 += LabColor::delta2(p, LabColor::at(image, x, y + 1));
                    ++nd;
                }
            }
        }
        mean_d2 = std::max(8.0, mean_d2 / std::max(1, nd));

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                const Lab p = LabColor::at(image, x, y);
                for (int k : {0, 2, 4, 5}) {
                    const int nx = x + dx[k];
                    const int ny = y + dy[k];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
                        continue;
                    }
                    const int j = ny * w + nx;
                    const float d2 = LabColor::delta2(p, LabColor::at(image, nx, ny));
                    const float B = (gamma_n / dw[k]) *
                                    std::exp(-d2 / (2.0f * static_cast<float>(mean_d2)));
                    add(i, j, B);
                    add(j, i, B);
                }
                const bool in_box = x >= bbox.x && x < bbox.x1() && y >= bbox.y && y < bbox.y1();
                const float fx = (x - (bbox.x + bbox.w * 0.5f)) / std::max(1.0f, bbox.w * 0.5f);
                const float fy = (y - (bbox.y + bbox.h * 0.5f)) / std::max(1.0f, bbox.h * 0.5f);
                const float r2 = fx * fx + fy * fy;
                if (in_box && r2 < 0.55f) {
                    add(S, i, lambda_fg);
                    add(i, T, 0.2f);
                } else if (!in_box) {
                    add(S, i, 0.2f);
                    add(i, T, lambda_bg);
                } else {
                    add(S, i, 1.2f);
                    add(i, T, 1.2f);
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

#pragma once

#include "math/contour_compat.hpp"
#include "filters/sobel/sobel.hpp"
#include "contour/laplace_gaussian/laplace_gaussian.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace contour {

class Livewire {
public:
    float wz = 0.35f;
    float wg = 0.45f;
    float wd = 0.20f;

    struct Node {
        float dist;
        int idx;
        bool operator<(const Node& o) const { return dist > o.dist; }
    };

    std::vector<float> cost;
    Field gx, gy;
    int width = 0;
    int height = 0;

    void build_cost(const ImageBuffer& image) {
        width = image.width;
        height = image.height;
        SobelFilter sobel;
        sobel.compute(image);
        gx = sobel.gx;
        gy = sobel.gy;
        LaplaceGaussian log;
        const ImageBuffer zc = log.zero_crossings(image);
        float mmax = 1e-6f;
        for (float v : sobel.mag.data) {
            mmax = std::max(mmax, v);
        }
        cost.assign(static_cast<size_t>(width * height), 1.0f);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const float fg = 1.0f - sobel.mag.at(x, y) / mmax;
                const float fz = zc.at(x, y) > 0 ? 0.0f : 1.0f;
                cost[static_cast<size_t>(y * width + x)] = 0.02f + wg * fg + wz * fz;
            }
        }
    }

    std::vector<Vec2> shortest_path(int x0, int y0, int x1, int y1) const {
        const int n = width * height;
        const int s = std::clamp(y0, 0, height - 1) * width + std::clamp(x0, 0, width - 1);
        const int t = std::clamp(y1, 0, height - 1) * width + std::clamp(x1, 0, width - 1);
        const int xmn = std::max(0, std::min(x0, x1) - 48);
        const int xmx = std::min(width - 1, std::max(x0, x1) + 48);
        const int ymn = std::max(0, std::min(y0, y1) - 48);
        const int ymx = std::min(height - 1, std::max(y0, y1) + 48);

        std::vector<float> dist(static_cast<size_t>(n), std::numeric_limits<float>::max());
        std::vector<int> parent(static_cast<size_t>(n), -1);
        std::priority_queue<Node> pq;
        dist[static_cast<size_t>(s)] = 0;
        pq.push({0.0f, s});
        static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
        static const float dw[8] = {1, 1, 1, 1, 1.414f, 1.414f, 1.414f, 1.414f};
        while (!pq.empty()) {
            const Node cur = pq.top();
            pq.pop();
            if (cur.idx == t) {
                break;
            }
            if (cur.dist != dist[static_cast<size_t>(cur.idx)]) {
                continue;
            }
            const int x = cur.idx % width;
            const int y = cur.idx / width;
            const Vec2 np = normalize({gx.at(x, y), gy.at(x, y)});
            for (int k = 0; k < 8; ++k) {
                const int nx = x + dx[k];
                const int ny = y + dy[k];
                if (nx < xmn || ny < ymn || nx > xmx || ny > ymx) {
                    continue;
                }
                const int ni = ny * width + nx;
                const Vec2 dir = normalize({static_cast<float>(dx[k]), static_cast<float>(dy[k])});
                const float fd = 0.5f * (1.0f - std::fabs(dot(np, dir)));
                const float nd = cur.dist + (cost[static_cast<size_t>(ni)] + wd * fd) * dw[k];
                if (nd < dist[static_cast<size_t>(ni)]) {
                    dist[static_cast<size_t>(ni)] = nd;
                    parent[static_cast<size_t>(ni)] = cur.idx;
                    pq.push({nd, ni});
                }
            }
        }
        std::vector<Vec2> path;
        if (parent[static_cast<size_t>(t)] < 0 && t != s) {
            return path;
        }
        for (int i = t; i >= 0; i = parent[static_cast<size_t>(i)]) {
            path.push_back({static_cast<float>(i % width), static_cast<float>(i / width)});
            if (i == s) {
                break;
            }
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    Polyline trace_waypoints(const ImageBuffer& image, const std::vector<Vec2>& seeds, bool closed = true) {
        build_cost(image);
        Polyline p;
        p.closed = closed && seeds.size() >= 3;
        if (seeds.size() < 2) {
            return p;
        }
        const size_t nseg = p.closed ? seeds.size() : seeds.size() - 1;
        for (size_t i = 0; i < nseg; ++i) {
            const Vec2 a = seeds[i];
            const Vec2 b = seeds[(i + 1) % seeds.size()];
            auto seg = shortest_path(static_cast<int>(std::lround(a.x)), static_cast<int>(std::lround(a.y)),
                                     static_cast<int>(std::lround(b.x)), static_cast<int>(std::lround(b.y)));
            if (seg.size() > 1) {
                p.points.insert(p.points.end(), seg.begin(), seg.end() - 1);
            }
        }
        return p;
    }

    static std::vector<Vec2> longest_boundary_chain(const ImageBuffer& gt) {
        std::vector<Vec2> best;
        ImageBuffer used = make_gray(gt.width, gt.height, 0);
        static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        for (int y = 0; y < gt.height; ++y) {
            for (int x = 0; x < gt.width; ++x) {
                if (gt.at(x, y) == 0 || used.at(x, y)) {
                    continue;
                }
                std::vector<Vec2> chain;
                int cx = x, cy = y;
                int px = -999, py = -999;
                for (int guard = 0; guard < gt.width * gt.height; ++guard) {
                    chain.push_back({static_cast<float>(cx), static_cast<float>(cy)});
                    used.at(cx, cy) = 255;
                    int nx = -1, ny = -1;
                    for (int k = 0; k < 8; ++k) {
                        const int xx = cx + dx[k];
                        const int yy = cy + dy[k];
                        if (xx < 0 || yy < 0 || xx >= gt.width || yy >= gt.height) {
                            continue;
                        }
                        if (gt.at(xx, yy) && !used.at(xx, yy) && !(xx == px && yy == py)) {
                            nx = xx;
                            ny = yy;
                            break;
                        }
                    }
                    if (nx < 0) {
                        break;
                    }
                    px = cx;
                    py = cy;
                    cx = nx;
                    cy = ny;
                }
                if (chain.size() > best.size()) {
                    best = std::move(chain);
                }
            }
        }
        return best;
    }

    static std::vector<Vec2> seeds_every_n_px(const std::vector<Vec2>& chain, float spacing = 30.0f) {
        std::vector<Vec2> seeds;
        if (chain.size() < 2) {
            return seeds;
        }
        seeds.push_back(chain.front());
        float acc = 0;
        for (size_t i = 1; i < chain.size(); ++i) {
            acc += dist(chain[i - 1], chain[i]);
            if (acc >= spacing) {
                seeds.push_back(chain[i]);
                acc = 0;
            }
        }
        if (dist(seeds.back(), chain.back()) > 8.0f) {
            seeds.push_back(chain.back());
        }
        if (seeds.size() > 24) {
            std::vector<Vec2> thin;
            const float step = static_cast<float>(seeds.size()) / 24.0f;
            for (int i = 0; i < 24; ++i) {
                thin.push_back(seeds[static_cast<size_t>(i * step)]);
            }
            seeds.swap(thin);
        }
        return seeds;
    }
};

}  // namespace contour

#pragma once

#include "../contour_kit/types.hpp"
#include "../filters/sobel.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace contour {

class Livewire {
public:
    float laplacian_weight = 0.3f;

    struct Node {
        float dist;
        int idx;
        bool operator<(const Node& o) const { return dist > o.dist; }
    };

    std::vector<float> cost;
    int width = 0;
    int height = 0;

    void build_cost(const ImageBuffer& image) {
        width = image.width;
        height = image.height;
        SobelFilter sobel;
        sobel.compute(image);
        Field lap = sobel.laplacian(image);
        float mmax = 1e-6f;
        for (float v : sobel.mag.data) {
            mmax = std::max(mmax, v);
        }
        cost.assign(static_cast<size_t>(width * height), 1.0f);
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const float g = sobel.mag.at(x, y) / mmax;
                const bool zc = (lap.at(x, y) > 0) != (lap.at(x - 1, y) > 0) ||
                                (lap.at(x, y) > 0) != (lap.at(x, y - 1) > 0);
                float c = 1.0f - g;
                if (zc) {
                    c *= (1.0f - laplacian_weight);
                }
                cost[static_cast<size_t>(y * width + x)] = 0.02f + std::max(0.0f, c);
            }
        }
    }

    std::vector<Vec2> shortest_path(int x0, int y0, int x1, int y1) const {
        const int n = width * height;
        const int s = y0 * width + x0;
        const int t = y1 * width + x1;
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
            for (int k = 0; k < 8; ++k) {
                const int nx = x + dx[k];
                const int ny = y + dy[k];
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                    continue;
                }
                const int ni = ny * width + nx;
                const float nd = cur.dist + cost[static_cast<size_t>(ni)] * dw[k];
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

    Polyline trace_waypoints(const ImageBuffer& image, const std::vector<Vec2>& seeds) {
        build_cost(image);
        Polyline p;
        p.closed = true;
        if (seeds.size() < 2) {
            return p;
        }
        for (size_t i = 0; i < seeds.size(); ++i) {
            const Vec2 a = seeds[i];
            const Vec2 b = seeds[(i + 1) % seeds.size()];
            auto seg = shortest_path(static_cast<int>(a.x), static_cast<int>(a.y),
                                     static_cast<int>(b.x), static_cast<int>(b.y));
            if (seg.size() > 1) {
                p.points.insert(p.points.end(), seg.begin(), seg.end() - 1);
            }
        }
        return p;
    }

    static std::vector<Vec2> seeds_from_boundary(const ImageBuffer& gt, int stride = 24) {
        std::vector<Vec2> pts;
        for (int y = 0; y < gt.height; ++y) {
            for (int x = 0; x < gt.width; ++x) {
                if (gt.at(x, y) > 0) {
                    pts.push_back({static_cast<float>(x), static_cast<float>(y)});
                }
            }
        }
        if (pts.size() < 4) {
            return pts;
        }
        std::vector<Vec2> seeds;
        for (size_t i = 0; i < pts.size(); i += static_cast<size_t>(std::max(1, stride))) {
            seeds.push_back(pts[i]);
        }
        if (seeds.size() > 16) {
            seeds.resize(16);
        }
        return seeds;
    }
};

}  // namespace contour

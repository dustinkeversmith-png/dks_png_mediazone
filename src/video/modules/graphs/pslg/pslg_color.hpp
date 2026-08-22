#pragma once

#include "graphs/half-region-adj/region_adjacency.hpp"
#include <algorithm>
#include <vector>

namespace contour {

// Greedy (Welsh–Powell) coloring of a PSLG / region-adjacency graph.
// Planar maps are 4-colorable; degree-ordered greedy typically uses 4–5 colors.
class PslgColor {
public:
    struct Result {
        std::vector<int> face_color;
        int n_colors = 0;
        std::vector<int> colored_labels;
        int width = 0;
        int height = 0;
    };

    static Result color(const RegionAdjacency::Result& rag) {
        Result r;
        r.width = rag.width;
        r.height = rag.height;
        const int n = static_cast<int>(rag.faces.size());
        r.face_color.assign(static_cast<size_t>(n), -1);
        std::vector<std::vector<int>> nbr(static_cast<size_t>(n));
        for (const auto& e : rag.adj) {
            if (e.a < 0 || e.b < 0 || e.a >= n || e.b >= n) {
                continue;
            }
            nbr[static_cast<size_t>(e.a)].push_back(e.b);
            nbr[static_cast<size_t>(e.b)].push_back(e.a);
        }
        std::vector<int> order;
        order.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (rag.faces[static_cast<size_t>(i)].area > 0) {
                order.push_back(i);
            }
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return nbr[static_cast<size_t>(a)].size() > nbr[static_cast<size_t>(b)].size();
        });
        int used = 0;
        for (int v : order) {
            bool taken[8] = {};
            for (int u : nbr[static_cast<size_t>(v)]) {
                const int c = r.face_color[static_cast<size_t>(u)];
                if (c >= 0 && c < 8) {
                    taken[c] = true;
                }
            }
            int c = 0;
            while (c < 8 && taken[c]) {
                ++c;
            }
            r.face_color[static_cast<size_t>(v)] = c;
            used = std::max(used, c + 1);
        }
        r.n_colors = used;
        r.colored_labels.resize(rag.labels.size(), 0);
        for (size_t i = 0; i < rag.labels.size(); ++i) {
            const int f = rag.labels[i];
            if (f >= 0 && f < n) {
                r.colored_labels[i] = r.face_color[static_cast<size_t>(f)] + 1;
            }
        }
        return r;
    }
};

}  // namespace contour

#pragma once

#include "math/vision_types.hpp"
#include "math/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vision {

// Suzuki–Abe topological border following (1985): one raster pass that recovers
// outer frames and hole borders with a Next/Prev/Child/Parent hierarchy tree.
class SuzukiAbeTopBorder {
public:
    struct Node {
        int id = 0;
        bool is_hole = false;
        int parent = -1;
        int first_child = -1;
        int next = -1;
        int prev = -1;
        std::vector<Vec2> points;
        float area = 0.0f;
        float perimeter = 0.0f;
    };

    struct Result {
        int width = 0;
        int height = 0;
        std::vector<Node> borders;  // index 0 = frame; borders[i].id == i
        GrayImage labeled;
        int n_outer = 0;
        int n_holes = 0;
    };

    static Result find(const GrayImage& image_in, uint8_t thr = 128) {
        Result out;
        constexpr int pad = 1;
        const int W = image_in.width + 2 * pad;
        const int H = image_in.height + 2 * pad;
        std::vector<int> f(static_cast<size_t>(W * H), 0);
        for (int y = 0; y < image_in.height; ++y) {
            for (int x = 0; x < image_in.width; ++x) {
                f[static_cast<size_t>((y + pad) * W + (x + pad))] =
                    image_in.fg(x, y, thr) ? 1 : 0;
            }
        }

        out.width = image_in.width;
        out.height = image_in.height;
        out.labeled.width = image_in.width;
        out.labeled.height = image_in.height;
        out.labeled.data.assign(static_cast<size_t>(image_in.width * image_in.height), 0);

        out.borders.push_back(Node{});
        out.borders[0].id = 0;
        out.borders[0].is_hole = true;  // infinite outer frame

        auto at = [&](int x, int y) -> int& {
            return f[static_cast<size_t>(y * W + x)];
        };

        // Clockwise from E: E, SE, S, SW, W, NW, N, NE
        static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

        auto link_hierarchy = [&](int new_id, int parent_id) {
            if (new_id <= 0 || new_id >= static_cast<int>(out.borders.size())) {
                return;
            }
            auto& node = out.borders[static_cast<size_t>(new_id)];
            if (parent_id < 0 || parent_id >= static_cast<int>(out.borders.size())) {
                parent_id = 0;
            }
            node.parent = parent_id;
            auto& parent = out.borders[static_cast<size_t>(parent_id)];
            if (parent.first_child < 0) {
                parent.first_child = new_id;
            } else {
                int sib = parent.first_child;
                int guard = 0;
                while (out.borders[static_cast<size_t>(sib)].next >= 0 &&
                       guard++ < static_cast<int>(out.borders.size())) {
                    sib = out.borders[static_cast<size_t>(sib)].next;
                }
                out.borders[static_cast<size_t>(sib)].next = new_id;
                node.prev = sib;
            }
        };

        auto follow = [&](int y0, int x0, int y_from, int x_from, int border_id) {
            Node node;
            node.id = border_id;
            // Internal raster labels must stay distinct from the unvisited
            // foreground sentinel (1), including for the first border.
            const int raster_label = border_id + 1;

            auto direction_from = [&](int x, int y, int nx, int ny) {
                for (int k = 0; k < 8; ++k) {
                    if (x + dx[k] == nx && y + dy[k] == ny) {
                        return k;
                    }
                }
                return 0;
            };

            int x = x0;
            int y = y0;
            int bx = x_from;
            int by = y_from;
            int first_x = -1;
            int first_y = -1;
            const int max_steps = W * H + 8;
            for (int step = 0; step < max_steps; ++step) {
                node.points.push_back(
                    {static_cast<float>(x - pad), static_cast<float>(y - pad)});

                // Moore-neighborhood tracing scans clockwise from the backtrack
                // pixel. Keep that pixel explicitly: using the previous travel
                // direction here cuts through filled regions and retraces the
                // same component as another border.
                const int back_dir = direction_from(x, y, bx, by);
                int found = -1;
                for (int k = 1; k <= 8; ++k) {
                    const int nd = (back_dir + k) % 8;
                    const int nx = x + dx[nd];
                    const int ny = y + dy[nd];
                    if (nx >= 0 && ny >= 0 && nx < W && ny < H && at(nx, ny) != 0) {
                        found = nd;
                        break;
                    }
                }

                // Apply Suzuki's sign convention after the neighborhood scan.
                // A negative label prevents the right edge of an already traced
                // border from being rediscovered as a hole start.
                at(x, y) =
                    (x + 1 < W && at(x + 1, y) == 0) ? -raster_label : raster_label;
                if (x >= pad && y >= pad && x < W - pad && y < H - pad) {
                    const int lx = x - pad;
                    const int ly = y - pad;
                    out.labeled.at(lx, ly) =
                        static_cast<uint8_t>(std::min(255, 40 + (border_id * 17) % 200));
                }

                if (found < 0) {
                    break;
                }

                const int nx = x + dx[found];
                const int ny = y + dy[found];
                if (first_x < 0) {
                    first_x = nx;
                    first_y = ny;
                } else if (step > 0 && x == x0 && y == y0 &&
                           nx == first_x && ny == first_y) {
                    node.points.pop_back();  // repeated start pixel
                    break;
                }

                // The predecessor of the selected neighbor in the clockwise
                // scan is the next backtrack pixel (Jacob stopping criterion).
                const int previous_dir = (found + 7) % 8;
                bx = x + dx[previous_dir];
                by = y + dy[previous_dir];
                x = nx;
                y = ny;
            }

            node.area = std::fabs(shoelace(node.points));
            node.perimeter = 0.0f;
            for (size_t t = 1; t < node.points.size(); ++t) {
                node.perimeter += dist(node.points[t - 1], node.points[t]);
            }
            if (node.points.size() >= 2) {
                node.perimeter += dist(node.points.back(), node.points.front());
            }
            return node;
        };

        int lnbd = 0;
        for (int y = 1; y < H - 1; ++y) {
            lnbd = 0;
            for (int x = 1; x < W - 1; ++x) {
                const int v = at(x, y);
                const int av = std::abs(v);
                bool is_outer = false;
                bool is_hole = false;
                int x_from = x;
                int y_from = y;

                if (v == 1 && at(x - 1, y) == 0) {
                    is_outer = true;
                    x_from = x - 1;
                    y_from = y;
                } else if (av >= 1 && at(x + 1, y) == 0 && v > 0) {
                    // Hole start only on non-negated border/FG (Suzuki –NBD skip).
                    is_hole = true;
                    x_from = x + 1;
                    y_from = y;
                    if (av > 1) {
                        lnbd = av - 1;
                    }
                }

                if (!is_outer && !is_hole) {
                    if (av > 1) {
                        lnbd = av - 1;
                    }
                    continue;
                }

                const int border_id = static_cast<int>(out.borders.size());
                Node node = follow(y, x, y_from, x_from, border_id);
                node.is_hole = is_hole;

                int parent = 0;
                if (lnbd > 0 && lnbd < static_cast<int>(out.borders.size())) {
                    const auto& last = out.borders[static_cast<size_t>(lnbd)];
                    parent = (last.is_hole == is_hole) ? last.parent : lnbd;
                    if (parent < 0 || parent >= static_cast<int>(out.borders.size())) {
                        parent = 0;
                    }
                }

                out.borders.push_back(std::move(node));
                link_hierarchy(border_id, parent);
                if (is_hole) {
                    ++out.n_holes;
                } else {
                    ++out.n_outer;
                }
                lnbd = border_id;
            }
        }

        return out;
    }
};

}  // namespace vision

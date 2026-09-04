#pragma once

#include "../../segmentation/helpers/ccl/connected_components.hpp"

#include <algorithm>
#include <queue>
#include <vector>

namespace vision {

class EulerCharacteristic {
public:
    struct Result {
        int components = 0;
        int holes = 0;
        int chi = 0;
    };

    static Result compute(const GrayImage& image, uint8_t thr = 128) {
        Result r;
        auto fg = ConnectedComponentLabeler::label(image, thr);
        r.components = static_cast<int>(fg.components.size());

        int holes = 0;
        // Pair 8-connected foreground with 4-connected background. Using
        // 8-connectivity for both merges cavities that only touch diagonally
        // and disagrees with Suzuki-Abe's border topology.
        std::vector<uint8_t> seen(image.data.size(), 0);
        static constexpr int dx[4] = {1, -1, 0, 0};
        static constexpr int dy[4] = {0, 0, 1, -1};
        for (int sy = 0; sy < image.height; ++sy) {
            for (int sx = 0; sx < image.width; ++sx) {
                const size_t start = static_cast<size_t>(sy * image.width + sx);
                if (seen[start] || image.fg(sx, sy, thr)) {
                    continue;
                }
                bool touches_border = false;
                std::queue<int> pending;
                pending.push(static_cast<int>(start));
                seen[start] = 1;
                while (!pending.empty()) {
                    const int index = pending.front();
                    pending.pop();
                    const int x = index % image.width;
                    const int y = index / image.width;
                    touches_border = touches_border || x == 0 || y == 0 ||
                                     x == image.width - 1 || y == image.height - 1;
                    for (int k = 0; k < 4; ++k) {
                        const int nx = x + dx[k];
                        const int ny = y + dy[k];
                        if (nx < 0 || ny < 0 || nx >= image.width || ny >= image.height) {
                            continue;
                        }
                        const size_t ni = static_cast<size_t>(ny * image.width + nx);
                        if (!seen[ni] && !image.fg(nx, ny, thr)) {
                            seen[ni] = 1;
                            pending.push(static_cast<int>(ni));
                        }
                    }
                }
                if (!touches_border) {
                    ++holes;
                }
            }
        }
        r.holes = holes;
        r.chi = r.components - r.holes;
        return r;
    }
};

}  // namespace vision

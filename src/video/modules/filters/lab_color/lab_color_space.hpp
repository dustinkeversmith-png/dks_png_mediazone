#pragma once

#include "math/contour_compat.hpp"
#include <algorithm>
#include <cmath>
#include <array>

namespace contour {

struct Lab {
    float L = 0;
    float a = 0;
    float b = 0;
};

class LabColor {
public:
    static Lab from_rgb(float r, float g, float b) {
        auto lin = [](float u) {
            u /= 255.0f;
            return u <= 0.04045f ? u / 12.92f : std::pow((u + 0.055f) / 1.055f, 2.4f);
        };
        const float R = lin(r), G = lin(g), B = lin(b);
        const float x = 0.4124564f * R + 0.3575761f * G + 0.1804375f * B;
        const float y = 0.2126729f * R + 0.7151522f * G + 0.0721750f * B;
        const float z = 0.0193339f * R + 0.1191920f * G + 0.9503041f * B;
        auto f = [](float t) {
            return t > 0.008856f ? std::cbrt(t) : (7.787f * t + 16.0f / 116.0f);
        };
        const float fx = f(x / 0.95047f);
        const float fy = f(y);
        const float fz = f(z / 1.08883f);
        return {116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz)};
    }

    static Lab at(const ImageBuffer& im, int x, int y) {
        if (im.channels >= 3) {
            return from_rgb(im.at(x, y, 0), im.at(x, y, 1), im.at(x, y, 2));
        }
        const float g = im.gray(x, y);
        return from_rgb(g, g, g);
    }

    static float delta2(const Lab& p, const Lab& q) {
        const float dL = p.L - q.L;
        const float da = p.a - q.a;
        const float db = p.b - q.b;
        return dL * dL + da * da + db * db;
    }

    static Field lightness(const ImageBuffer& im) {
        Field L = make_field(im.width, im.height);
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                L.at(x, y) = at(im, x, y).L;
            }
        }
        return L;
    }
};

}  // namespace contour

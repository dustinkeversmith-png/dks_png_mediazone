#pragma once

#include "../math/vision_types.hpp"
#include "../math/geometry.hpp"

namespace contour {

using Vec2 = math::Vec2;
using Rect = math::Rect;
using ImageBuffer = math::ImageBuffer;
using Field = math::Field;
using Polyline = math::Polyline;

using math::make_gray;
using math::make_field;
using math::dist;
using math::dist2;
using math::length;
using math::dot;
using math::normalize;

} // namespace contour

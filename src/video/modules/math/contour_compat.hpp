#pragma once

// Backward-compatible contour namespace aliases over modular math types.
// Replaces deleted contour_kit/types.hpp.

#include "vision_types.hpp"
#include "geometry.hpp"

namespace contour {

using math::Vec2;
using math::ImageBuffer;
using math::Field;
using math::Polyline;
using math::Rect;
using math::make_gray;
using math::make_field;
using math::dot;
using math::dist;
using math::dist2;
using math::length;
using math::normalize;
using math::shoelace;

constexpr float kPi = math::kPi;

}  // namespace contour

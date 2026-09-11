#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace vocal::metrics {

using Frames = std::vector<std::vector<double>>;

struct PitchScore {
    double f0_rmse_hz{};
    double vuv_error_rate{};
    std::size_t jointly_voiced_frames{};
};

[[nodiscard]] double mcd_db(const Frames& reference, const Frames& synthesized,
                            bool use_dtw = true);
[[nodiscard]] PitchScore pitch(std::span<const double> reference_hz,
                               std::span<const double> synthesized_hz);
[[nodiscard]] double word_error_rate(std::string_view reference,
                                     std::string_view hypothesis);

}  // namespace vocal::metrics


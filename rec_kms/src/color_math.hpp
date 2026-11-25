#pragma once

#include "cli.hpp"
#include "types.hpp"

#include <array>

namespace vkcap {

// Standard CIE 1931 chromaticity primaries
inline constexpr ColorPrimaries kPrimariesBt709 = {
    0.6400, 0.3300,
    0.3000, 0.6000,
    0.1500, 0.0600,
    0.3127, 0.3290,
    true,
};

inline constexpr ColorPrimaries kPrimariesDisplayP3 = {
    0.6800, 0.3200,
    0.2650, 0.6900,
    0.1500, 0.0600,
    0.3127, 0.3290,
    true,
};

inline constexpr ColorPrimaries kPrimariesBt2020 = {
    0.7080, 0.2920,
    0.1700, 0.7970,
    0.1310, 0.0460,
    0.3127, 0.3290,
    true,
};

// Returns true if `p` is valid and meaningfully wider/different from sRGB/BT.709.
bool is_wide_gamut(const ColorPrimaries &p) noexcept;

// Computes row-major 3x3 linear RGB->RGB conversion matrix from `src` primaries
// to `dst` primaries with Bradford chromatic adaptation for white point differences.
std::array<float, 9> compute_gamut_matrix(const ColorPrimaries &src,
                                          const ColorPrimaries &dst) noexcept;

struct ResolvedGamut {
    bool enabled = false;
    std::array<float, 9> matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::string description;
};

ResolvedGamut resolve_input_gamut(const Config &cfg, const ColorPrimaries &edid_primaries);

}  // namespace vkcap

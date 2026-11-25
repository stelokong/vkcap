#include "color_math.hpp"

#include <cmath>
#include <cstdlib>
#include <format>

namespace vkcap {

namespace {

struct Mat3 {
    double m[3][3]{};
};

Mat3 mul(const Mat3 &a, const Mat3 &b) noexcept {
    Mat3 r{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i][j] = a.m[i][0] * b.m[0][j] +
                        a.m[i][1] * b.m[1][j] +
                        a.m[i][2] * b.m[2][j];
        }
    }
    return r;
}

bool invert(const Mat3 &a, Mat3 &inv) noexcept {
    const double c00 = a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1];
    const double c01 = a.m[1][2] * a.m[2][0] - a.m[1][0] * a.m[2][2];
    const double c02 = a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0];
    const double det = a.m[0][0] * c00 + a.m[0][1] * c01 + a.m[0][2] * c02;
    if (std::abs(det) < 1e-12) return false;

    const double idet = 1.0 / det;
    inv.m[0][0] = c00 * idet;
    inv.m[0][1] = (a.m[0][2] * a.m[2][1] - a.m[0][1] * a.m[2][2]) * idet;
    inv.m[0][2] = (a.m[0][1] * a.m[1][2] - a.m[0][2] * a.m[1][1]) * idet;
    inv.m[1][0] = c01 * idet;
    inv.m[1][1] = (a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0]) * idet;
    inv.m[1][2] = (a.m[0][2] * a.m[1][0] - a.m[0][0] * a.m[1][2]) * idet;
    inv.m[2][0] = c02 * idet;
    inv.m[2][1] = (a.m[0][1] * a.m[2][0] - a.m[0][0] * a.m[2][1]) * idet;
    inv.m[2][2] = (a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0]) * idet;
    return true;
}

std::array<double, 3> xy_to_XYZ(double x, double y) noexcept {
    if (y <= 1e-9) return {0.0, 0.0, 0.0};
    return {x / y, 1.0, (1.0 - x - y) / y};
}

bool rgb_to_xyz_matrix(const ColorPrimaries &p, Mat3 &out) noexcept {
    if (!p.valid || p.ry <= 1e-6 || p.gy <= 1e-6 || p.by <= 1e-6 || p.wy <= 1e-6) {
        return false;
    }
    const auto r = xy_to_XYZ(p.rx, p.ry);
    const auto g = xy_to_XYZ(p.gx, p.gy);
    const auto b = xy_to_XYZ(p.bx, p.by);
    const auto w = xy_to_XYZ(p.wx, p.wy);

    Mat3 prim{};
    for (int row = 0; row < 3; ++row) {
        prim.m[row][0] = r[row];
        prim.m[row][1] = g[row];
        prim.m[row][2] = b[row];
    }

    Mat3 prim_inv{};
    if (!invert(prim, prim_inv)) return false;

    double s[3]{};
    for (int i = 0; i < 3; ++i) {
        s[i] = prim_inv.m[i][0] * w[0] + prim_inv.m[i][1] * w[1] + prim_inv.m[i][2] * w[2];
    }
    for (int row = 0; row < 3; ++row) {
        out.m[row][0] = prim.m[row][0] * s[0];
        out.m[row][1] = prim.m[row][1] * s[1];
        out.m[row][2] = prim.m[row][2] * s[2];
    }
    return true;
}

Mat3 bradford_adapt(const std::array<double, 3> &src_w,
                    const std::array<double, 3> &dst_w) noexcept {
    constexpr Mat3 kBfd = {{
        { 0.8951000,  0.2664000, -0.1614000},
        {-0.7502000,  1.7135000,  0.0367000},
        { 0.0389000, -0.0685000,  1.0296000},
    }};
    constexpr Mat3 kBfdInv = {{
        { 0.9869929, -0.1470543,  0.1599627},
        { 0.4323053,  0.5183603,  0.0492912},
        {-0.0085287,  0.0400428,  0.9684867},
    }};

    double src_cone[3]{}, dst_cone[3]{};
    for (int i = 0; i < 3; ++i) {
        src_cone[i] = kBfd.m[i][0] * src_w[0] + kBfd.m[i][1] * src_w[1] + kBfd.m[i][2] * src_w[2];
        dst_cone[i] = kBfd.m[i][0] * dst_w[0] + kBfd.m[i][1] * dst_w[1] + kBfd.m[i][2] * dst_w[2];
    }

    Mat3 scale{};
    for (int i = 0; i < 3; ++i) {
        scale.m[i][i] = (std::abs(src_cone[i]) > 1e-9) ? (dst_cone[i] / src_cone[i]) : 1.0;
    }
    return mul(kBfdInv, mul(scale, kBfd));
}

bool primaries_close(const ColorPrimaries &a, const ColorPrimaries &b, double eps) noexcept {
    return std::abs(a.rx - b.rx) <= eps && std::abs(a.ry - b.ry) <= eps &&
           std::abs(a.gx - b.gx) <= eps && std::abs(a.gy - b.gy) <= eps &&
           std::abs(a.bx - b.bx) <= eps && std::abs(a.by - b.by) <= eps &&
           std::abs(a.wx - b.wx) <= eps && std::abs(a.wy - b.wy) <= eps;
}

}  // namespace

bool is_wide_gamut(const ColorPrimaries &p) noexcept {
    if (!p.valid) return false;
    return !primaries_close(p, kPrimariesBt709, 0.02);
}

std::array<float, 9> compute_gamut_matrix(const ColorPrimaries &src,
                                          const ColorPrimaries &dst) noexcept {
    constexpr std::array<float, 9> kIdent = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Mat3 m_src{}, m_dst{}, m_dst_inv{};
    if (!rgb_to_xyz_matrix(src, m_src) ||
        !rgb_to_xyz_matrix(dst, m_dst) ||
        !invert(m_dst, m_dst_inv)) {
        return kIdent;
    }
    const Mat3 adapt = bradford_adapt(xy_to_XYZ(src.wx, src.wy), xy_to_XYZ(dst.wx, dst.wy));
    const Mat3 total = mul(m_dst_inv, mul(adapt, m_src));

    std::array<float, 9> out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[static_cast<size_t>(r * 3 + c)] = static_cast<float>(total.m[r][c]);
        }
    }
    return out;
}

ResolvedGamut resolve_input_gamut(const Config &cfg, const ColorPrimaries &edid_primaries) {
    const ColorPrimaries &dst = (cfg.color_space == ColorSpace::Bt2020)
        ? kPrimariesBt2020
        : kPrimariesBt709;
    const char *dst_name = (cfg.color_space == ColorSpace::Bt2020) ? "BT.2020" : "BT.709";

    ColorPrimaries src{};
    std::string src_label;

    switch (cfg.input_gamut) {
        case InputGamut::Srgb:
            src = kPrimariesBt709;
            src_label = "sRGB/BT.709";
            break;
        case InputGamut::DisplayP3:
            src = kPrimariesDisplayP3;
            src_label = "Display P3";
            break;
        case InputGamut::Bt2020:
            src = kPrimariesBt2020;
            src_label = "BT.2020";
            break;
        case InputGamut::Edid:
            if (edid_primaries.valid) {
                src = edid_primaries;
                src_label = std::format("EDID (R={:.3f},{:.3f} G={:.3f},{:.3f} B={:.3f},{:.3f})",
                                        src.rx, src.ry, src.gx, src.gy, src.bx, src.by);
            } else {
                src = kPrimariesBt709;
                src_label = "sRGB/BT.709 (no EDID)";
            }
            break;
        case InputGamut::Auto: {
            const bool wayland = std::getenv("WAYLAND_DISPLAY") != nullptr;
            if (wayland && is_wide_gamut(edid_primaries)) {
                src = edid_primaries;
                src_label = std::format("EDID wide-gamut (R={:.3f},{:.3f} G={:.3f},{:.3f} B={:.3f},{:.3f})",
                                        src.rx, src.ry, src.gx, src.gy, src.bx, src.by);
            } else {
                src = kPrimariesBt709;
                src_label = "sRGB/BT.709";
            }
            break;
        }
    }

    ResolvedGamut res{};
    if (!primaries_close(src, dst, 0.005)) {
        res.enabled = true;
        res.matrix = compute_gamut_matrix(src, dst);
        res.description = std::format("{} -> {}", src_label, dst_name);
    } else {
        res.enabled = false;
        res.description = std::format("{} (passthrough)", dst_name);
    }
    return res;
}

}  // namespace vkcap

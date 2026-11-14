# =============================================================================
# The polarity gate — ONE sign convention for every detector of the package.
#
# ASIFT gates its pool by the sign of the DoG response (`asift_detect!`'s
# `polarity`); the structure-tensor detectors (`detect_sfop`, the seeded
# `detect_foerstner`) are sign-invariant (N = Σ∇g∇gᵀ), so they take the same
# vocabulary and the same sign: the scale-normalised Laplacian of Gaussian,
# which the DoG approximates, positive at a DARK blob (a minimum of the image),
# negative at a LIGHT one. `Detection(…; polarity)` and `is_dark`/`is_light`
# read that sign too (`laplacian_scale_score > 0` dark).
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const

# The sign a kept detection's Laplacian must have: 0 for `:none` (no gate), +1 `:dark`, −1 `:light`.
_polarity_sign(polarity::Symbol) =
    polarity === :none ? 0.0f0 : polarity === :dark ? 1.0f0 : polarity === :light ? -1.0f0 :
    throw(ArgumentError("polarity must be :none, :dark or :light, got :$polarity"))

# The scale-normalised Laplacian of Gaussian s²∇²(G_s ⋆ I) of the H × W image `img` (rows y, columns x) at each
# COLMAP point (`p[1, k]`, `p[2, k]`) and scale `p[3, k]` (pixels), and its variance under the pixel noise `v` (a
# variance for every pixel: a number, or an H × W matrix): L = Σ_q I(q)·w′(q) over the pixels within ⌈3.3s⌉ that lie in
# the image, w = G_s(d)(|d|²/s² − 2), d = q − p, made zero-mean on them, w′ = w − c·G_s, c = Σw/ΣG (a constant image
# gives 0 exactly, at the border too); Var L = Σ_q v(q) w′(q)². `out[1, k]` = L, `out[2, k]` = Var L. Every sum in
# Float64. L is positive at a dark blob.
@kernel function _blob_laplacian!(out, @Const(img), @Const(p), v)
    k = @index(Global)
    @inbounds begin
        H, W = size(img)
        px, py = Float64(p[1, k]) + 0.5, Float64(p[2, k]) + 0.5     # 1-based (column, row)
        sk = Float64(p[3, k])
        r = ceil(Int, 3.3 * sk)
        c0, r0 = round(Int, px), round(Int, py)
        sw = 0.0; sg = 0.0; siw = 0.0; sig = 0.0; vww = 0.0; vwg = 0.0; vgg = 0.0
        for j in max(1, c0 - r):min(W, c0 + r), i in max(1, r0 - r):min(H, r0 + r)
            d2 = ((j - px)^2 + (i - py)^2) / sk^2
            g = exp(-d2 / 2)
            w = g * (d2 - 2)
            x = Float64(img[i, j])
            vq = Float64(_variance_at(v, i, j))
            sw += w; sg += g; siw += x * w; sig += x * g
            vww += vq * w * w; vwg += vq * w * g; vgg += vq * g * g
        end
        c = sg > 0 ? sw / sg : 0.0
        out[1, k] = siw - c * sig
        out[2, k] = vww - 2c * vwg + c^2 * vgg
    end
end
@inline _variance_at(v::Real, i, j) = v
@inline _variance_at(v::AbstractMatrix, i, j) = @inbounds v[i, j]

# Which of the points (`points`, a host 3 × n matrix: COLMAP x, y and the scale s of each, columns) pass the polarity gate of sign `sign` on the device
# image `img` (H × W) under the pixel noise `variance` (`_blob_laplacian!`'s `v`, on the device): their Laplacian has
# that sign SIGNIFICANTLY, `sign·L > z·√Var L`, z the standard normal's `significance_level` quantile — a one-sided
# test, so a point whose sign the noise could have given (a window on flat background between blobs) is not kept for
# either polarity: a missed blob, never one of the wrong sign.
function _polarity_kept(device, img::AbstractMatrix, points::AbstractMatrix, sign::Real, variance, significance_level::Real)
    n = size(points, 2)
    n == 0 && return falses(0)
    L = KA.allocate(device, Float64, 2, n)
    _launch!(_blob_laplacian!, device, L, img, ka_upload(Float64, device, points), variance; ndrange = n)
    Lh = Array(L)
    z = quantile(Normal(), significance_level)
    return [sign * Lh[1, k] > z * sqrt(max(Lh[2, k], 0.0)) for k in 1:n]   # Var L: rounding can take a 0 below it
end

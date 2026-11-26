# =============================================================================
# The detectors on a RAW mosaic — the bilinear green image on the raw grid
# =============================================================================
#
# `detect_sfop(m::RawMosaic; sensor_model)` and `detect_foerstner(m::RawMosaic, seeds; sensor_model)`: the green
# photosites at the raw resolution, without the 2×2 cells' decimation (`LinearRaw{:green_cells}`, whose square lattice
# aliases the quincunx's diagonal detail). The first image is the BILINEAR green on the raw grid, the normalized
# convolution of the usable green photosites with the cross kernel (the pixel and its four 4-neighbours, weight 1
# each):
#
#     Ī(x) = Σ_k w(x − p_k) c_k I_k / Σ_k w(x − p_k) c_k,     Var Ī(x) = Σ_k w² c_k σ_k² / (Σ_k w c_k)²,
#
# c_k = 1 for a usable green photosite, 0 for every other: a green site is its own value, a red or blue site the mean
# of its usable green 4-neighbours (the weights renormalised). The owner's ruling (2026-11-14) on the two kinds of bad
# photosite:
#   - SATURATED = CENSORED: a normal sample, its value clipped at the white level (1) and its variance the noise law's
#     at 1 (Core's `value_variance`, never 0 under a law); the pixels that read one are flagged (`saturated`);
#   - DEFECTIVE (the sensor model's defect list) = MISSING: it drops out of the sums. A pixel with no usable green
#     under the cross (a defective green, whose 4-neighbours are red and blue) is the mean of the usable greens of its
#     3×3 neighbourhood (its four diagonal greens), weights renormalised; only a pixel with no usable green in the 3×3
#     (a defect cluster) has no value: NaN, so every window reading it fails its tests (SFOP's λ₂ test, Förstner's
#     estimate: an :unknown row).
# I_k is Core's one linearisation (`linearview`), σ_k² Core's one variance law (the sensor model's). The greens G1 and
# G2 are taken to respond alike (identity response): the stack's sensor model carries no per-channel response (owner,
# 2026-11-14), so a G1/G2 gain mismatch stays in the image as a checkerboard (a known limit; measure it with flat
# fields if needed). One KernelAbstractions kernel, Float64 sums.
#
# Then SFOP's scale space as implemented (its 2× upsampled first octave) gives the seeds, and the seeded Förstner
# estimates on this raw-grid image itself (its own doubling, ip_fop.m step 1), its noise propagated from the
# PHOTOSITES through the bilinear weights (foerstner.jl's `_BilinearGreenNoise`). Positions and covariances are raw
# COLMAP pixels (the mosaic's `raw_frame_map` frame), no map.
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const
using VisualGeometryToolkitCore: photosite_colors, saturationview, defect_mask, value_variance, bayer_channels, Adapt

# The bilinear green image's blur, raw pixels (per axis, by its second moment): the photosite's box integration,
# variance 1/12, plus the interpolation's — a green site is its own value (variance 0), a red or blue site the mean of
# its four green neighbours at ±1 px (½ per axis), half the sites each, ¼ on average — √(1/12 + 1/4) = 1/√3. SFOP's
# `input_blur` (input pixels; the first octave, upsampled 2×, starts at 2/√3 ≈ 1.155 of its pixels).
const _BILINEAR_GREEN_BLUR = 0.5773502691896258   # 1/√3 raw px

# The cross kernel's taps, (Δrow, Δcolumn): the pixel, then its four 4-neighbours; and the 3×3 support's other taps,
# the four diagonal neighbours (a green site's diagonal greens).
const _CROSS_TAPS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
const _DIAGONAL_TAPS = ((-1, -1), (1, -1), (-1, 1), (1, 1))

# A pixel's support (`_bilinear_green!`'s `support` plane): the count n of its usable greens, flagged by
# `_SUPPORT_3X3` when they are its 3×3 neighbourhood's (no usable green under its cross); 0 without a value.
const _SUPPORT_3X3 = 0x80
@inline _support_count(s::UInt8) = Int(s & ~_SUPPORT_3X3)
# Whether photosite k at offset (d1, d2) from pixel p is in p's support `s`: under its cross, or in its 3×3.
@inline _in_support(s::UInt8, d1, d2) = (s & _SUPPORT_3X3) != 0 ? max(abs(d1), abs(d2)) <= 1 : abs(d1) + abs(d2) <= 1

# Photosite (i, j) is a usable green: a green colour index (1 or 3, LibRaw's G and G2), not a defect (the defect mask is
# over the visible raw: the photosite at the mosaic's window offset `origin`). A saturated green is usable (censored).
@inline _usable_green(colors, defects, origin, i, j) =
    @inbounds ((colors[i, j] == 0x01) | (colors[i, j] == 0x03)) && !_defect_at(defects, i + origin[1], j + origin[2])
@inline _defect_at(::Nothing, r, c) = false
@inline _defect_at(mask::AbstractMatrix{Bool}, r, c) = @inbounds mask[r, c]

# A usable green's sample (value, variance, saturated): its linear value and the law's variance there, or, saturated
# (censored), 1 and the law's variance at 1 (`value_variance`; 0 without a law, as for a plain value).
@inline _linear_variance(x::Real) = 0.0
@inline _linear_variance(x) = Float64(var(x))
@inline function _green_sample(lin, sat, i, j)
    @inbounds sat[i, j] && return 1.0, value_variance(lin, i, j, 1.0), true
    @inbounds x = lin[i, j]
    return Float64(value_of(x)), _linear_variance(x), false
end

# Σ over the usable greens at `taps` from (i, j): their count, values' sum, variances' sum, whether one is censored.
@inline function _green_sums(taps, lin, sat, defects, colors, origin, i, j, H, W)
    n = 0; sx = 0.0; sv = 0.0; censored = false
    for (di, dj) in taps
        ii, jj = i + di, j + dj
        if (1 <= ii <= H) & (1 <= jj <= W) && _usable_green(colors, defects, origin, ii, jj)
            x, v, c = _green_sample(lin, sat, ii, jj)
            n += 1; sx += x; sv += v; censored |= c
        end
    end
    return n, sx, sv, censored
end

# The bilinear green at raw pixel (i, j): `value` (T; NaN without a usable green in the 3×3), `support` (the count n of
# the greens it averages, each of weight 1/n, flagged `_SUPPORT_3X3` for the 3×3 fallback; so Var Ī = density/(2n)),
# `density` (the noise's white-equivalent variance per raw pixel, Float32: the photosites' variance averaged over the
# support, times 2 — the green lattice holds one photosite per 2 raw pixels and the interpolation keeps the DC gain, so
# at the scales the detectors integrate over the noise is that of white noise of variance 2σ² per pixel), `photosite`
# (σ_k² of a usable green photosite, 0 at every other site) and `saturated` (a censored green in the support).
@kernel function _bilinear_green!(value, support, density, photosite, saturated, @Const(lin), @Const(sat), @Const(defects),
                                  colors, origin)
    i, j = @index(Global, NTuple)
    H, W = size(value)
    T = eltype(value)
    n, sx, sv, censored = _green_sums(_CROSS_TAPS, lin, sat, defects, colors, origin, i, j, H, W)
    fallback = n == 0
    fallback && ((n, sx, sv, censored) = _green_sums(_DIAGONAL_TAPS, lin, sat, defects, colors, origin, i, j, H, W))
    own = _usable_green(colors, defects, origin, i, j) ? Float32(_green_sample(lin, sat, i, j)[2]) : 0.0f0
    @inbounds begin
        value[i, j] = n > 0 ? T(sx / n) : T(NaN)
        support[i, j] = n == 0 ? 0x00 : UInt8(n) | (fallback ? _SUPPORT_3X3 : 0x00)
        density[i, j] = n > 0 ? Float32(2sv / n) : NaN32
        photosite[i, j] = own
        saturated[i, j] = censored
    end
end

# The image's derivations (the cross, its 1/√3 blur, the 2σ² density, the F-test's checkerboard) hold on a 2×2 Bayer
# pattern only, its two greens on a diagonal (the quincunx): Core's `bayer_channels` refuses any other pattern, and a cell
# whose greens share a row or a column is refused here.
function _check_quincunx(m::RawMosaic)
    g = findall(c -> c in (:G1, :G2), bayer_channels(m))   # cell sites, column-major: (0,0), (1,0), (0,1), (1,1)
    g in ([1, 4], [2, 3]) || throw(ArgumentError(
        "the bilinear green image needs the greens of each 2×2 cell on a diagonal (a Bayer quincunx), got the cell $(bayer_channels(m))"))
    return nothing
end

"""
    bilinear_green([T = Float32,] m::RawMosaic; sensor_model = SensorModel(), device = default_device())
        -> (; value, support, density, photosite, saturated)

The green image of a Bayer mosaic, the one kernel the mosaic detectors read (`detect_sfop(m)`, `detect_foerstner(m,
seeds)`): per pixel, a green photosite its own linear value and any other pixel the bilinear mean of its usable green
neighbours, under the frame's `sensor_model` (its black, its defects); values in `T`, computed on `device`. The planes
(H × W arrays on `device`): `value`; `support`, the count of usable greens the value read (0: none, no value);
`density` and `photosite`, what the detectors' noise model reads; `saturated`, a green read at saturation. The greens
of each 2×2 cell must lie on a diagonal (a Bayer quincunx): another pattern is an `ArgumentError`.
"""
bilinear_green(m::RawMosaic; kw...) = bilinear_green(Float32, m; kw...)
bilinear_green(::Type{T}, m::RawMosaic; sensor_model::SensorModel = SensorModel(), device = default_device()) where {T} =
    _bilinear_green(T, device, m, sensor_model)
# The planes on `device` (`_bilinear_green!`'s), positional for the detectors' calls.
function _bilinear_green(::Type{T}, device, m::RawMosaic, model::SensorModel) where {T}
    _check_quincunx(m)
    md = Adapt.adapt(device, m)                             # Core's device form of a mosaic: its photosites moved, the rest kept
    H, W = size(md)
    value = KA.allocate(device, T, H, W)
    density, photosite = KA.allocate(device, Float32, H, W), KA.allocate(device, Float32, H, W)
    support, saturated = KA.allocate(device, UInt8, H, W), KA.allocate(device, Bool, H, W)
    _launch!(_bilinear_green!, device, value, support, density, photosite, saturated, linearview(Float64, md; sensor_model = model),
             saturationview(md), defect_mask(model, md; backend = device), photosite_colors(md), md.window_origin; ndrange = (H, W))
    return (; value, support, density, photosite, saturated)
end

"""
    detect_sfop(m::RawMosaic; sensor_model::SensorModel = SensorModel(), device = default_device(), kwargs...)

SFOP on a RAW mosaic's green photosites at the raw resolution: the scale space of the BILINEAR green image on the raw
grid (a green photosite its own value, a red or blue one the mean of its usable green 4-neighbours). A saturated
photosite is censored: a normal sample at the white level (1) with the noise law's variance there. A defect of the
sensor model is missing: it drops out, and a pixel with no usable green under its cross (a defective green) is the mean
of the usable greens of its 3×3 neighbourhood; only a pixel with none in the 3×3 (a defect cluster) has no value, and
every window reading it fails the λ₂ test. With SFOP's 2× upsampled first octave and `input_blur` the image's stated blur,
`1/√3` raw pixels (the photosite's box ⊕ the interpolation). Rows as for a matrix, in raw COLMAP pixels (σ in raw
pixels): the seeds of [`detect_foerstner(m, seeds)`](@ref detect_foerstner).

Under a gain-carrying `sensor_model` (a Poisson–Gaussian, the frame's) the λ₂ and polarity tests read the image's
noise from the photosites' variance law (each pixel's white-equivalent variance: twice the green photosites' under the
cross, one green photosite per two raw pixels); without a gain `noise_sigma` as for a matrix. The greens G1 and G2 are
taken to respond alike (the sensor model carries no per-channel response): a G1/G2 mismatch remains as a checkerboard
of that amplitude (a known limit). `input_blur` and `upsample` are the image's and are refused; the other keywords are
[`detect_sfop`](@ref)'s.
"""
function detect_sfop(m::RawMosaic; sensor_model::SensorModel = SensorModel(), device::KA.Backend = default_device(), kw...)
    _refuse_raw_grid_keywords("detect_sfop", kw)
    g = bilinear_green(Float32, m; sensor_model, device)
    variance = _propagated_status(sensor_model.noise) === nothing ? nothing : g.density
    return _detect_sfop(g.value, variance; device, input_blur = _BILINEAR_GREEN_BLUR, kw...)
end

_refuse_raw_grid_keywords(f, kw) = for k in (:input_blur, :upsample, :seed_sigma)
    haskey(kw, k) && throw(ArgumentError(
        "$f(m::RawMosaic): `$k` is the bilinear green image's own (input_blur 1/√3 raw px, SFOP's 2× upsampling), not a keyword"))
end

"""
    detect_foerstner([T = Float32], m::RawMosaic, seeds; sensor_model::SensorModel = SensorModel(), device = default_device(),
                     kwargs...) -> StructArray{<:Detection}

The seeded Förstner window estimation on a RAW mosaic's bilinear green image on the raw grid (the image of
[`detect_sfop(m)`](@ref detect_sfop), its seeds; the detector's own 2× doubling, ip_fop.m step 1), in raw COLMAP
pixels. Under a gain-carrying `sensor_model` the covariance is propagated from the PHOTOSITES: each green photosite `k`
moves the estimate by `J_k = Σ_q J[:, q] W[q, k]`, `W` the bilinear weights (`1/n` at each pixel whose `n` averaged
greens include it: its own site, its red and blue 4-neighbours, and a defective green's 3×3 fallback), a censored
(saturated) photosite with the law's variance at the white level, and `Σ = Σ_k σ_k² J_k J_kᵀ`, `σ_k²` the variance law's; without a gain the
a-posteriori covariance. The F-test's effective degrees of freedom are those of the residual sums under that photosite
noise (white over the green photosites, through the interpolation), not white input pixels. A window reading a pixel
without a value (see `detect_sfop(m)`) is an `:unknown` row. The other keywords and the rows are those of
[`detect_foerstner(img, seeds)`](@ref detect_foerstner).
"""
detect_foerstner(m::RawMosaic, seeds::AbstractVector{<:Detection}; kw...) = detect_foerstner(Float32, m, seeds; kw...)
function detect_foerstner(::Type{T}, m::RawMosaic, seeds::AbstractVector{<:Detection}; sensor_model::SensorModel = SensorModel(),
                          device::KA.Backend = default_device(), kw...) where {T <: AbstractFloat}
    _check_precision(T); _check_seeds(seeds)                 # refused before the image is built
    g = bilinear_green(T, m; sensor_model, device)
    noise = _BilinearGreenNoise(g.support, g.photosite, g.density, _propagated_status(sensor_model.noise))
    return _detect_seeded(T, g.value, noise, seeds; device, kw...)
end

# The image-wide search on a mosaic is not offered (its windows are SFOP's): refused, never the photosite codes read as
# a grey image.
detect_foerstner(::RawMosaic; kw...) = _no_raw_search()
detect_foerstner(::Type{<:AbstractFloat}, ::RawMosaic; kw...) = _no_raw_search()
_no_raw_search() = throw(ArgumentError(
    "detect_foerstner(m::RawMosaic): the image-wide window search is not run on a mosaic; seed it: detect_foerstner(m, detect_sfop(m; sensor_model); sensor_model)"))

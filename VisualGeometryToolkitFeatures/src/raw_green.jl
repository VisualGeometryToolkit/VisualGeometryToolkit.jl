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
# c_k = 1 for a usable green photosite (not saturated, not a defect of the sensor model), 0 for every other: a green
# site is its own value, a red or blue site the mean of its usable green 4-neighbours (the weights renormalised, so a
# masked neighbour drops out); a site with no usable green under the cross (a saturated or defect green site, whose
# 4-neighbours are red and blue; a red or blue site with all four greens masked) has NO value: NaN, never filled, so
# every window that reads it fails its tests (SFOP's λ₂ test, Förstner's estimate: an :unknown row). I_k is Core's one
# linearisation (`linearview`), σ_k² Core's one variance law (the sensor model's). The greens G1 and G2 are taken to
# respond alike (identity response): the stack's sensor model carries no per-channel response (owner, 2026-11-14), so
# a G1/G2 gain mismatch stays in the image as a checkerboard (a known limit; measure it with flat fields if needed).
# One KernelAbstractions kernel, Float64 sums.
#
# Then SFOP's scale space as implemented (its 2× upsampled first octave) gives the seeds, and the seeded Förstner
# estimates on this raw-grid image itself (its own doubling, ip_fop.m step 1), its noise propagated from the
# PHOTOSITES through the bilinear weights (foerstner.jl's `_BilinearGreenNoise`). Positions and covariances are raw
# COLMAP pixels (the mosaic's `raw_frame_map` frame), no map.
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const
using VisualGeometryToolkitCore: photosite_colors, saturationview, defect_mask, Adapt
using VisualGeometryToolkitCore: _masked   # THE defect lookup of the visible raw (Core's kernels' own)

# The bilinear green image's blur, raw pixels (per axis, by its second moment): the photosite's box integration,
# variance 1/12, plus the interpolation's — a green site is its own value (variance 0), a red or blue site the mean of
# its four green neighbours at ±1 px (½ per axis), half the sites each, ¼ on average — √(1/12 + 1/4) = 1/√3. SFOP's
# `input_blur` (input pixels; the first octave, upsampled 2×, starts at 2/√3 ≈ 1.155 of its pixels).
const _BILINEAR_GREEN_BLUR = 0.5773502691896258   # 1/√3 raw px

# The cross kernel's taps, (Δrow, Δcolumn): the pixel, then its four 4-neighbours.
const _CROSS_TAPS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

# Photosite (i, j) is a usable green: a green colour index (1 or 3, LibRaw's G and G2), not at the white level, not a
# defect (the defect mask is over the visible raw: the photosite at the mosaic's window offset `origin`).
@inline _usable_green(colors, sat, defects, origin, i, j) =
    @inbounds (colors[i, j] == 0x01) | (colors[i, j] == 0x03) && !sat[i, j] && !_masked(defects, i + origin[1], j + origin[2])

# A linear photosite's variance: the variance law's, or 0 for a plain value (no law).
@inline _linear_variance(x::Real) = 0.0
@inline _linear_variance(x) = Float64(var(x))

# The bilinear green at raw pixel (i, j): `value` (T, NaN without a usable green under the cross), `count` (n, the
# usable greens under the cross: the bilinear weight of each is 1/n, so Var Ī = density/(2n)), `density` (the noise's
# white-equivalent variance per raw pixel, Float32: the photosites' variance averaged under the cross, times 2 — the
# green lattice holds one photosite per 2 raw pixels and the interpolation keeps the DC gain, so at the scales the
# detectors integrate over the noise is that of white noise of variance 2σ² per pixel) and `photosite` (σ_k² of a
# usable green photosite, 0 at every other site).
@kernel function _bilinear_green!(value, count, density, photosite, @Const(lin), @Const(sat), @Const(defects), colors, origin)
    i, j = @index(Global, NTuple)
    H, W = size(value)
    T = eltype(value)
    n = 0; sx = 0.0; sv = 0.0
    own = 0.0f0
    for (di, dj) in _CROSS_TAPS
        ii, jj = i + di, j + dj
        if (1 <= ii <= H) & (1 <= jj <= W) && _usable_green(colors, sat, defects, origin, ii, jj)
            @inbounds x = lin[ii, jj]
            v = _linear_variance(x)
            n += 1
            sx += Float64(value_of(x))
            sv += v
            (di == 0) & (dj == 0) && (own = Float32(v))
        end
    end
    @inbounds begin
        value[i, j] = n > 0 ? T(sx / n) : T(NaN)
        count[i, j] = UInt8(n)
        density[i, j] = n > 0 ? Float32(2sv / n) : NaN32
        photosite[i, j] = own
    end
end

# The bilinear green image of `m` on `device`, values in `T`, under the frame's `model`: `(; value, count, density,
# photosite)` (H × W device arrays, `_bilinear_green!`'s planes).
function _bilinear_green(::Type{T}, device, m::RawMosaic, model::SensorModel) where {T}
    md = Adapt.adapt(device, m)                             # Core's device form of a mosaic: its photosites moved, the rest kept
    H, W = size(md)
    value = KA.allocate(device, T, H, W)
    density, photosite = KA.allocate(device, Float32, H, W), KA.allocate(device, Float32, H, W)
    count = KA.allocate(device, UInt8, H, W)
    _launch!(_bilinear_green!, device, value, count, density, photosite, linearview(Float64, md; sensor_model = model),
             saturationview(md), defect_mask(model, md; backend = device), photosite_colors(md), md.window_origin; ndrange = (H, W))
    return (; value, count, density, photosite)
end

"""
    detect_sfop(m::RawMosaic; sensor_model::SensorModel = SensorModel(), device = default_device(), kwargs...)

SFOP on a RAW mosaic's green photosites at the raw resolution: the scale space of the BILINEAR green image on the raw
grid (a green photosite its own value, a red or blue one the mean of its usable green 4-neighbours; saturated and the
sensor model's defect photosites masked, a site with no usable green under the cross left without a value, and every
window reading it failing the λ₂ test), with SFOP's 2× upsampled first octave and `input_blur` the image's stated blur,
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
    g = _bilinear_green(Float32, device, m, sensor_model)
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
moves the estimate by `J_k = Σ_q J[:, q] W[q, k]`, `W` the bilinear weights (1 at its own site, `1/n` at each red or
blue 4-neighbour with `n` usable greens), and `Σ = Σ_k σ_k² J_k J_kᵀ`, `σ_k²` the variance law's; without a gain the
a-posteriori covariance. The F-test's effective degrees of freedom are those of the residual sums under that photosite
noise (white over the green photosites, through the interpolation), not white input pixels. A window reading a pixel
without a value (see `detect_sfop(m)`) is an `:unknown` row. The other keywords and the rows are those of
[`detect_foerstner(img, seeds)`](@ref detect_foerstner).
"""
detect_foerstner(m::RawMosaic, seeds::AbstractVector{<:Detection}; kw...) = detect_foerstner(Float32, m, seeds; kw...)
function detect_foerstner(::Type{T}, m::RawMosaic, seeds::AbstractVector{<:Detection}; sensor_model::SensorModel = SensorModel(),
                          device::KA.Backend = default_device(), kw...) where {T <: AbstractFloat}
    g = _bilinear_green(T, device, m, sensor_model)
    noise = _BilinearGreenNoise(g.count, g.photosite, g.density, _propagated_status(sensor_model.noise))
    return _detect_seeded(T, g.value, noise, seeds; device, kw...)
end

# The image-wide search on a mosaic is not offered (its windows are SFOP's): refused, never the photosite codes read as
# a grey image.
detect_foerstner(::RawMosaic; kw...) = _no_raw_search()
detect_foerstner(::Type{<:AbstractFloat}, ::RawMosaic; kw...) = _no_raw_search()
_no_raw_search() = throw(ArgumentError(
    "detect_foerstner(m::RawMosaic): the image-wide window search is not run on a mosaic; seed it: detect_foerstner(m, detect_sfop(m; sensor_model); sensor_model)"))

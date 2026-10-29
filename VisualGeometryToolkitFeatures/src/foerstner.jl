# =============================================================================
# The Förstner interest operator (Förstner & Gülch 1987, with Köthe 2003's
# doubled resolution and window criterion), after the reference
# implementation ip_fop.m (M. Luxen, University of Bonn).
#
# Each step follows ip_fop.m; the line numbers below are that file's. The
# image-wide steps are KernelAbstractions kernels (one code path: `KA.CPU()` is
# the CPU path), their per-element math plain generic functions. The
# deviations from the reference are listed in `detect_foerstner`'s docstring
# and on the docs page, with the evidence for each.
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const

# One row of `detect_foerstner`'s result: a point frame with its 2×2
# covariance, labelled, with its window centre and the classification statistic.
const _FoerstnerDetection = Attributed{
    Uncertain{PointFrame{Float64}, CovMat{2, Float64, 4, UnlabeledFrame{2}}},
    @NamedTuple{label::Symbol, window::Point2{Float64}, F::Float64},
}

"""
    detect_foerstner([T = Float32], img; device = default_device(),
                     method = :foerstner, noise_sigma = 2.0,
                     derivative = :gaussian2d, integration = :box,
                     derivative_sigma = 1.0, integration_sigma = 1.41 * derivative_sigma,
                     precision_threshold = 0.5, roundness_threshold = 0.3,
                     significance_level = 0.999) -> StructArray{<:Detection}

The Förstner interest operator [Foerstner1987](@cite): the junction (corner)
points and circular points of a grey image at sub-pixel precision, each with
its covariance, after the reference implementation `ip_fop.m`
[Luxen2005](@cite), with Köthe's doubled resolution and window criterion
[Koethe2003](@cite).

Returns one row per optimal window, a `StructArray` of
`Detection(Uncertain(PointFrame(x), Σ), (; label, window, F))` (VisualGeometryToolkitCore's
`PointFrame`: a position, no scale):

- `label`: `:corner` (a junction point), `:circular` (Förstner & Gülch's
  circular point: the centre of a rotationally symmetric pattern, with no size
  estimate) or `:unknown` (the significance test assigned neither);
- the frame: the corner or circular point's position and its 2×2 covariance
  (px², `(x, y)` order); for an `:unknown` row the window centre with a NaN
  covariance (the reference returns no estimate for it);
- `window`: the window centre, a `Point2`;
- `F`: the classification statistic `Ω_corner / Ω_circular`.

`filter(d -> d.label === :corner, points)` are the corners; `Ellipse(d; sigma = 1)`
is a point's 1σ error ellipse (`ip_errell.m`'s default).

`img` is any grey image the package's detectors take, read through
[`gray_f32`](@ref) as grey values in `[0, 1]` (an 8-bit image as `value/255`,
as MATLAB's `im2double`); it is not stretched to its own range. The image-wide
steps run on `device`, any KernelAbstractions backend, in precision `T`
(the reference computes in `Float64`; pass `Float64` for that).

# Keywords (defaults: the values the reference's CODE runs with)
- `method`: `:foerstner` — windows where the roundness `4 det/tr² > roundness_threshold`
  and the precision `det/tr > (σₙ / precision_threshold)²`, ranked by `det/tr`;
  `:koethe` — windows passing the precision test, ranked by the smaller eigenvalue.
- `noise_sigma`: the standard deviation of the (constant) image noise, in GREY
  LEVELS OF AN 8-BIT IMAGE (0–255), as the reference's for a `uint8` image;
  `σₙ = noise_sigma / 255` on the `[0, 1]` image.
- `derivative`: `:gaussian2d` (a Gaussian derivative smoothed across by a
  Gaussian of the same σ) or `:gaussian1d` (the derivative alone).
- `integration`: `:box` (a box whose standard deviation is `integration_sigma`)
  or `:gaussian`.
- `derivative_sigma`, `integration_sigma`: in input pixels.
- `precision_threshold`: the largest accepted point standard deviation, input px.
- `roundness_threshold`: the smallest accepted roundness (`:foerstner` only).
- `significance_level`: of the F-test that labels a point corner or circular.

# Algorithm (ip_fop.m, step by step)
1. The image is doubled bilinearly (the package's one 2× upsampling, ASIFT's)
   and every scale and threshold doubled with it (lines 170–180).
2. The gradient by correlation with the derivative taps (radius `⌈3.3σ⌉`, zero
   padding, as `imfilter`), its squares and product (the squared gradient),
   and their integration by the box (width `2⌈√(12σ² + 1)/2⌉ + 1`) or the
   Gaussian (radius `⌈3.3σ⌉`), zero padded (lines 184–207, 617–678).
3. The window measure (`det/tr` or the smaller eigenvalue) masked by the tests
   above; its regional maxima (8-connected plateaus higher than every
   neighbour, `imregionalmax`), thinned by a second non-maximum suppression
   over a square of radius `2⌈√(12σᵢ² + 1)⌉` doubled pixels, away from the
   border (lines 210–281).
4. In the `(2ρ + 1)²` window around each centre, `ρ = ⌈√(12σᵢ² + 1)/2⌉ + 1`
   (σᵢ doubled), the pixels weighed equally (the reference's choice, also for
   a Gaussian integration): the corner is the least-squares intersection of
   the lines through every pixel ACROSS its gradient (normal matrix
   `Σ ∇g∇gᵀ`), the circular point that of the lines ALONG it (`Σ ∇g⊥∇g⊥ᵀ`).
   Each has a weighted residual sum `Ω`, redundancy `(2ρ + 1)² − 2` and
   covariance `Ω/redundancy · N⁻¹` (lines 314–423, 440, 449).
5. `F = Ω_corner / Ω_circular` against the F distribution of (redundancy,
   redundancy) degrees of freedom, as the reference's
   `(T < F(α)) − (T > F(1 − α))` (line 429): a corner below the
   `1 − significance_level` quantile, a circular point above the
   `significance_level` quantile, unknown in between.

# Coordinates
COLMAP pixel centres of the input image, `(x, y) = (column, row)`, converted
once, at the output. The doubled image's pixel `(r′, c′)` (1-based) samples the
input at 1-based `(r′/2, c′/2)`, so a doubled-grid position is at COLMAP
`((c′ − 1)/2, (r′ − 1)/2)` in the input, and its covariance is the doubled one
`/4`, reordered to `(x, y)`.

# Differences from ip_fop.m
1. **Undoing the doubling — fixed.** The reference doubles with MATLAB's
   `imresize(g, 2, 'bilinear')` (line 172; doubled pixel `j` samples input
   `j/2 + 1/4`) but maps positions back with `transf_H`, `transf_v`
   (lines 340–350, applied at 436, 446, 457), the inverse of an align-corners
   grid: its points are off by `((n + 1)/2 − x)/(2n − 1)` (1-based, `n` the
   image size), 0 at the centre, up to 0.25 px toward the centre at the
   borders. Here the doubling is the package's upsampling (input `j/2`, a
   quarter pixel from `imresize`'s grid) undone by its own exact inverse.
2. **The circular point's residual sum — kept as the reference has it.** Its
   line 423 takes two of the three terms from the corner model's residuals;
   the circular model's own residual sum would take all three from its own.
   Monte Carlo coverage favours the reference's form, so it is kept.
3. **Defaults — the code's, not the header's.** The header states
   `SIGMA_N = 2.5` and `SIGMA_DIFF = 0.7` (lines 37, 50); the code runs `2.0`
   and `1.0` (lines 521, 524). A documentation conflict, not a fix.

The docs page gives the evidence for each. The reference warns that with a
Gaussian integration the window estimation still weighs the pixels equally;
so does this.

# Example
```jldoctest
julia> using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore

julia> img = [Float32((c - 20.5)^2 + (r - 24.5)^2 < 6.25) for r in 1:48, c in 1:40];   # a dot at pixel (24.5, 20.5)

julia> points = detect_foerstner(img);

julia> only(points).label
:circular

julia> round.(Tuple(only(points).center); digits = 1)   # COLMAP: (column − ½, row − ½)
(20.0, 24.0)
```
"""
detect_foerstner(img::AbstractMatrix; kw...) = detect_foerstner(Float32, img; kw...)

function detect_foerstner(
        ::Type{T}, img::AbstractMatrix;
        device::KA.Backend = default_device(),
        method::Symbol = :foerstner,
        noise_sigma::Real = 2.0,
        derivative::Symbol = :gaussian2d,
        integration::Symbol = :box,
        derivative_sigma::Real = 1.0,
        integration_sigma::Real = 1.41 * derivative_sigma,
        precision_threshold::Real = 0.5,
        roundness_threshold::Real = 0.3,
        significance_level::Real = 0.999
    ) where {T <: AbstractFloat}
    0 < significance_level < 1 ||
        throw(ArgumentError("detect_foerstner: significance_level must lie in (0, 1), got $significance_level"))
    g = _ka_upload(device, T, Float32.(gray.(gray_f32(img))))
    doubled = KA.allocate(device, T, 2 .* size(g)...)
    _launch!(_upsample_doubled!, device, doubled, reshape(g, size(g)..., 1); ndrange = size(doubled))
    found = _foerstner_windows(
        device, doubled; method, noise_sigma, derivative, integration, derivative_sigma,
        integration_sigma, precision_threshold, roundness_threshold
    )
    return _foerstner_points(found, significance_level, _undouble)
end

# A kernel launch, completed before the next: KernelAbstractions launches are
# asynchronous (on the CPU backend too).
function _launch!(kernel!, device, args...; ndrange)
    kernel!(device)(args...; ndrange)
    KA.synchronize(device)
    return nothing
end

# A 1-based doubled-grid (row, column) → COLMAP (x, y) of the input: the
# package's upsampling puts doubled pixel j at input j/2 (1-based).
_undouble(r::Real, c::Real) = Point2{Float64}((c - 1) / 2, (r - 1) / 2)

@kernel function _upsample_doubled!(dst, @Const(src))
    I, J = @index(Global, NTuple)
    @inbounds dst[I, J] = _upsample_sample(src, I, J, 1)
end

# The image-wide steps on the doubled image (ip_fop.m 127, 177–281) and the
# window estimates (314–423), positions on the doubled grid, 1-based (row,
# column). Returns the window centres, the estimates (a 14 × K host matrix, see
# `_window_estimate`), the window half-width ρ, its redundancy and the squared
# gradient on the host (`grr, gcc, grc` as a H × W × 3 array).
function _foerstner_windows(
        device, doubled::AbstractMatrix{T}; method::Symbol, noise_sigma::Real, derivative::Symbol,
        integration::Symbol, derivative_sigma::Real, integration_sigma::Real,
        precision_threshold::Real, roundness_threshold::Real
    ) where {T}
    method in (:foerstner, :koethe) ||
        throw(ArgumentError("detect_foerstner: method must be :foerstner or :koethe, got :$method"))
    derivative in (:gaussian2d, :gaussian1d) ||
        throw(ArgumentError("detect_foerstner: derivative must be :gaussian2d or :gaussian1d, got :$derivative"))
    integration in (:box, :gaussian) ||
        throw(ArgumentError("detect_foerstner: integration must be :box or :gaussian, got :$integration"))
    upscale = 2                                             # 170–180
    nonmax_radius = upscale * ceil(Int, sqrt(12 * integration_sigma^2 + 1))
    σd = upscale * derivative_sigma
    σi = upscale * integration_sigma
    precision_floor = T(((noise_sigma / 255) / (upscale * precision_threshold))^2)   # 213
    rows, cols = size(doubled)

    # 184–207: the squared gradient and its integration
    derivative_taps, across_taps = _derivative_taps(derivative, σd)
    integration_taps, integration_radius = _integration_taps(integration, σi)
    up(taps) = _ka_upload(device, T, taps)
    dr = KA.allocate(device, T, rows, cols, 2)
    tmp = similar(dr)
    _launch!(_correlate_rows!, device, tmp, reshape(doubled, rows, cols, 1), up([derivative_taps across_taps]); ndrange = size(tmp))
    _launch!(_correlate_columns!, device, dr, tmp, up([across_taps derivative_taps]); ndrange = size(dr))
    gamma = KA.allocate(device, T, rows, cols, 3)
    _launch!(_squared_gradient!, device, gamma, dr; ndrange = (rows, cols))
    mean_gamma = similar(gamma)
    tmp3 = similar(gamma)
    it = up(repeat(integration_taps, 1, 3))
    _launch!(_correlate_rows!, device, tmp3, gamma, it; ndrange = size(tmp3))
    _launch!(_correlate_columns!, device, mean_gamma, tmp3, it; ndrange = size(mean_gamma))

    # 210–248: the window measure and its regional maxima
    candidates = KA.allocate(device, T, rows, cols)
    ranking = similar(candidates)
    _launch!(
        _window_measures!, device,
        candidates, ranking, mean_gamma, method === :koethe, precision_floor, T(roundness_threshold);
        ndrange = (rows, cols)
    )
    is_max = KA.allocate(device, Bool, rows, cols)
    _launch!(_plateau_maxima_init!, device, is_max, candidates; ndrange = (rows, cols))
    changed = KA.allocate(device, Int32, 1)
    while true                                              # a plateau loses where an equal neighbour lost
        fill!(changed, Int32(0))
        _launch!(_plateau_maxima_spread!, device, is_max, changed, candidates; ndrange = (rows, cols))
        KA.synchronize(device)
        Array(changed)[1] == 0 && break
    end

    # 249–281: the second non-maximum suppression, away from the border
    border = max(integration_radius, nonmax_radius) + 1
    keep = KA.allocate(device, Bool, rows, cols)
    _launch!(_second_suppression!, device, keep, is_max, ranking, Int32(nonmax_radius), Int32(border); ndrange = (rows, cols))
    keep_host = Array(keep)
    count(Array(is_max) .& (Array(ranking) .> 0)) == rows * cols && fill!(keep_host, false)   # 255–258
    centres = findall(keep_host)                            # column-major, as MATLAB's find

    # 314–423: both point models in each window
    ρ = ceil(Int, sqrt(12 * σi^2 + 1) / 2) + 1
    estimates = zeros(Float64, 14, length(centres))
    if !isempty(centres)
        centre_list = _ka_upload(device, Int32, [Int32(Tuple(p)[k]) for k in 1:2, p in centres])
        out = KA.allocate(device, T, 14, length(centres))
        _launch!(_window_estimates!, device, out, gamma, centre_list, Int32(ρ); ndrange = length(centres))
        estimates .= Array(out)
    end
    KA.synchronize(device)
    return (; centres, estimates, ρ, redundancy = (2ρ + 1)^2 - 2, upscale, gamma = Array(gamma))
end

# 323–460: the F-test and one labelled row per window, positions mapped by `undouble`.
function _foerstner_points(found, significance_level, undouble)
    (; centres, estimates, redundancy, upscale) = found
    fdist = FDist(redundancy, redundancy)
    corner_critical = quantile(fdist, significance_level)
    circular_critical = quantile(fdist, 1 - significance_level)
    unknown_cov = CovMat(SMatrix{2, 2, Float64, 4}(NaN, NaN, NaN, NaN))
    rows = map(enumerate(centres)) do (k, p)
        e = view(estimates, :, k)
        statistic = e[7] / e[14]                                                # 427
        class = (statistic < corner_critical) - (statistic > circular_critical) # 429
        window = undouble(p[1], p[2])
        frame = if class == 1
            _uncertain_point(undouble, e[1], e[2], e[3:6] .* (e[7] / redundancy), upscale)
        elseif class == -1
            _uncertain_point(undouble, e[8], e[9], e[10:13] .* (e[14] / redundancy), upscale)
        else
            Uncertain(PointFrame(window), unknown_cov)   # the reference's noclass: no estimate
        end
        label = class == 1 ? :corner : class == -1 ? :circular : :unknown
        _FoerstnerDetection(frame, (; label, window, F = statistic))
    end
    return StructArray(_FoerstnerDetection[rows...]; unwrap = T -> T <: NamedTuple)
end

# A doubled-grid estimate and its (row, column) covariance `Σ` (column-major
# 2×2) → an uncertain COLMAP (x, y) point frame of the input image.
function _uncertain_point(undouble, r, c, Σ, upscale)
    Σxy = SMatrix{2, 2, Float64, 4}(Σ[4], Σ[3], Σ[2], Σ[1]) / upscale^2
    return Uncertain(PointFrame(undouble(r, c)), CovMat(Σxy))
end

# ----- per-element math (plain functions, called inside the kernels) -----------

# 210–245: the window measures at one pixel from the integrated squared
# gradient (mrr, mcc, mrc): the masked candidate value and the value its
# regional maxima are ranked by (`det/tr` for Förstner, the smaller eigenvalue
# for Köthe). `tiny` is MATLAB's `eps` (Float64's, as the reference).
@inline function _window_measure(mrr::T, mcc::T, mrc::T, koethe::Bool, precision_floor::T, roundness_threshold::T) where {T}
    tiny = T(eps(Float64))
    trace = mrr + mcc
    det = mrr * mcc - mrc^2
    w = det / (trace + tiny)
    if koethe
        μ2 = (trace - sqrt((mrr - mcc)^2 + 4 * mrc^2)) / 2
        return ifelse(w > precision_floor, μ2, zero(T)), μ2
    end
    roundness = 4 * det / (trace^2 + tiny)
    return ifelse((roundness > roundness_threshold) & (w > precision_floor), w, zero(T)), w
end

# 352–423 at one window centre (rw, cw), half-width ρ, on the squared gradient
# `gamma` (H × W × 3: grr, gcc, grc), pixels weighed equally. The sums run in
# coordinates relative to the centre (the estimates are shift-equivariant;
# relative sums keep Float32 accurate). Returns the 14 numbers
# (corner r, c, vec(N⁻¹), Ω, dot r, c, vec(N⁻¹), Ω), positions 1-based.
@inline function _window_estimate(gamma, rw::Integer, cw::Integer, ρ::Integer)
    T = eltype(gamma)
    srr = scc = src = zero(T)
    srr_r = srr_c = scc_r = scc_c = src_r = src_c = zero(T)
    for dc in (-ρ):ρ, dr in (-ρ):ρ
        @inbounds a, b, d = gamma[rw + dr, cw + dc, 1], gamma[rw + dr, cw + dc, 2], gamma[rw + dr, cw + dc, 3]
        r, c = T(dr), T(dc)
        srr += a; scc += b; src += d
        srr_r += a * r; srr_c += a * c
        scc_r += b * r; scc_c += b * c
        src_r += d * r; src_c += d * c
    end
    corner_inverse = inv(SMatrix{2, 2, T, 4}(srr, src, src, scc))
    corner = corner_inverse * SVector(srr_r + src_c, src_r + scc_c)
    dot_inverse = inv(SMatrix{2, 2, T, 4}(scc, -src, -src, srr))
    dot = dot_inverse * SVector(scc_r - src_c, -src_r + srr_c)
    corner_residual = dot_residual = zero(T)
    for dc in (-ρ):ρ, dr in (-ρ):ρ
        @inbounds a, b, d = gamma[rw + dr, cw + dc, 1], gamma[rw + dr, cw + dc, 2], gamma[rw + dr, cw + dc, 3]
        er, ec = T(dr) - corner[1], T(dc) - corner[2]
        corner_residual += a * er^2 + 2 * d * er * ec + b * ec^2
        # line 423 as the reference has it: the rr term from the blob model's
        # residual, the rc and cc terms from the corner model's (see the docs)
        dot_residual += b * (T(dr) - dot[1])^2 - 2 * d * er * ec + a * ec^2
    end
    return (
        corner[1] + rw, corner[2] + cw, corner_inverse[1], corner_inverse[2], corner_inverse[3],
        corner_inverse[4], corner_residual, dot[1] + rw, dot[2] + cw, dot_inverse[1], dot_inverse[2],
        dot_inverse[3], dot_inverse[4], dot_residual,
    )
end

# ----- kernels ------------------------------------------------------------------

# MATLAB's imfilter: correlation, 'same' size, zero padding — along the rows
# (index 1) or the columns (index 2) of each slice k, with that slice's taps
# column `taps[:, k]` (odd length, centred).
@kernel function _correlate_rows!(out, @Const(a), @Const(taps))
    i, j, k = @index(Global, NTuple)
    H = size(a, 1)
    r = size(taps, 1) ÷ 2
    s = zero(eltype(out))
    for t in max(-r, 1 - i):min(r, H - i)
        @inbounds s += taps[t + r + 1, k] * a[i + t, j, min(k, size(a, 3))]
    end
    @inbounds out[i, j, k] = s
end

@kernel function _correlate_columns!(out, @Const(a), @Const(taps))
    i, j, k = @index(Global, NTuple)
    W = size(a, 2)
    r = size(taps, 1) ÷ 2
    s = zero(eltype(out))
    for t in max(-r, 1 - j):min(r, W - j)
        @inbounds s += taps[t + r + 1, k] * a[i, j + t, k]
    end
    @inbounds out[i, j, k] = s
end

# 196–199: (∂r g)², (∂c g)², ∂r g ∂c g from the two derivative slices.
@kernel function _squared_gradient!(gamma, @Const(d))
    i, j = @index(Global, NTuple)
    @inbounds begin
        gr, gc = d[i, j, 1], d[i, j, 2]
        gamma[i, j, 1] = gr * gr
        gamma[i, j, 2] = gc * gc
        gamma[i, j, 3] = gr * gc
    end
end

@kernel function _window_measures!(candidates, ranking, @Const(m), koethe::Bool, precision_floor, roundness_threshold)
    i, j = @index(Global, NTuple)
    @inbounds candidates[i, j], ranking[i, j] =
        _window_measure(m[i, j, 1], m[i, j, 2], m[i, j, 3], koethe, precision_floor, roundness_threshold)
end

# imregionalmax, 8-connected, for the positive values (a zero plateau is a
# regional maximum only when it is the whole image, which the reference
# discards, lines 255–258): a pixel starts as a maximum when no neighbour is
# greater, and `_plateau_maxima_spread!` takes it off while an equal neighbour
# is off, until nothing changes.
@kernel function _plateau_maxima_init!(is_max, @Const(a))
    i, j = @index(Global, NTuple)
    H, W = size(a)
    @inbounds v = a[i, j]
    m = v > zero(v)
    for dj in -1:1, di in -1:1
        ii, jj = i + di, j + dj
        @inbounds if 1 <= ii <= H && 1 <= jj <= W && a[ii, jj] > v
            m = false
        end
    end
    @inbounds is_max[i, j] = m
end

@kernel function _plateau_maxima_spread!(is_max, changed, @Const(a))
    i, j = @index(Global, NTuple)
    H, W = size(a)
    @inbounds if is_max[i, j]
        v = a[i, j]
        for dj in -1:1, di in -1:1
            ii, jj = i + di, j + dj
            if 1 <= ii <= H && 1 <= jj <= W && a[ii, jj] == v && !is_max[ii, jj]
                is_max[i, j] = false
                changed[1] = Int32(1)
            end
        end
    end
end

# 228, 243, 268–281: a regional maximum of positive rank, inside the border,
# not below any rank of the regional maxima within `radius`.
@kernel function _second_suppression!(keep, @Const(is_max), @Const(ranking), radius::Int32, border::Int32)
    i, j = @index(Global, NTuple)
    H, W = size(ranking)
    @inbounds v = ifelse(is_max[i, j], ranking[i, j], zero(eltype(ranking)))
    k = (v > zero(v)) & (border < i < H - border) & (border < j < W - border)
    if k
        for dj in (-radius):radius, di in (-radius):radius
            @inbounds if is_max[i + di, j + dj] && ranking[i + di, j + dj] > v
                k = false
            end
        end
    end
    @inbounds keep[i, j] = k
end

@kernel function _window_estimates!(out, @Const(gamma), @Const(centres), ρ::Int32)
    k = @index(Global)
    @inbounds e = _window_estimate(gamma, centres[1, k], centres[2, k], ρ)
    for n in 1:14
        @inbounds out[n, k] = e[n]
    end
end

# ----- taps -----------------------------------------------------------------------

# 617–650: the derivative taps x·exp(−x²/2σ²)/(√(2π)σ³) on radius ⌈3.3σ⌉ (the
# package's one Gaussian taps builder, at truncation 3.3) and the taps across
# the derivative: for `:gaussian2d` the unit-sum Gaussian of the same σ and
# radius (fspecial's), for `:gaussian1d` the identity (zero padded to the same
# length, so the two share one taps matrix).
function _derivative_taps(derivative::Symbol, σ::Real)
    full = _symmetric_profile(Float64(σ), 3.3)
    r = length(full) ÷ 2
    derivative_taps = [x * full[x + r + 1] / (sqrt(2π) * σ^3) for x in -r:r]
    across = derivative === :gaussian2d ? full ./ sum(full) : [x == 0 ? 1.0 : 0.0 for x in -r:r]
    return derivative_taps, across
end

# 656–678: the integration taps per axis (the 2D filter is their outer product:
# fspecial's 'average' box, or its normalized 'gaussian' on radius ⌈3.3σ⌉) and
# their radius.
function _integration_taps(integration::Symbol, σ::Real)
    if integration === :gaussian
        full = _symmetric_profile(Float64(σ), 3.3)
        return full ./ sum(full), length(full) ÷ 2
    end
    r = ceil(Int, sqrt(12σ^2 + 1) / 2)
    return fill(1 / (2r + 1), 2r + 1), r
end

_symmetric_profile(σ, truncation) = (g = _gaussian_profile(σ, truncation); [reverse(g[2:end]); g])

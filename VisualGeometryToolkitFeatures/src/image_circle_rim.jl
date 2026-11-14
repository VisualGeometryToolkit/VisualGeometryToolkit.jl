# =============================================================================
# The image circle's RIM: log-polar profiles of the frames' maximum, the rim's log-brightness drops, the coverage fit
# =============================================================================
#
# detect_image_circle (image_circle.jl) measures the lens coverage here. The frames' per-pixel maximum is resampled by
# Core's log-polar warp (`logpolar_map` → `image_warp` → `image_warp!`, ONE KernelAbstractions kernel on every backend)
# around an initial model (the image centre by default) at ≤ 1 px in radius and in arc; the samples are grouped into
# angle bins and each (angle bin, radius) is reduced to its MEDIAN by this file's kernel (same backend). Per angle every
# edge of the profile's outer envelope, at its steepest drop in LOG brightness, is a rim CANDIDATE (a log drop is a
# ratio: independent of the exposure); the RANSAC ellipse (or circle) on one candidate per angle, twice (each angle's
# outermost, then each angle's nearest that model; `_fit_image_circle`, the VisualGeometryToolkitEstimation extension),
# is the LENS COVERAGE, and per angle its inlier is the rim's edge there.
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: norm
using Statistics: median
using Distributions: Normal, pdf, cdf
using VisualGeometryToolkitCore: logpolar_map, transform_deriv, mad_scale, ambient_coordinates, scale, retract_jacobian, EllipseManifold

"""
    SensorModelNoise()

A [`RimProfile`](@ref)'s noise statistics (its `statistics`) when the frames are `Uncertain` cells (a RAW read under a
sensor model with a variance law): each bin's noise is the model's.
"""
struct SensorModelNoise end

"""
    EmpiricalNoise()

A [`RimProfile`](@ref)'s noise statistics (its `statistics`) when the frames carry no variances: each angle's noise is
the data's own, Core's `mad_scale` of the profile's drops across 2 px, each over its noise per unit pixel noise (the
bins' pixels).
"""
struct EmpiricalNoise end

"""
    RimProfile

The measurement behind [`detect_image_circle`](@ref)'s coverage (its `rim` field): the log-polar profiles of the
frames' per-pixel maximum around the initial model, every angle's candidates and its chosen edge.

Fields (radii and centre in the MEASURED plane's pixels: a RAW frame's green cells):
- `center::Point2{Float64}`, `initial` — the log-polar origin and the initial model it is the centre of (the measured
  plane's pixels);
- `angles::Vector{Float64}` — each angle bin's centre, radians, `atan(y − cy, x − cx)` (COLMAP pixels, y down);
- `radii::Vector{Float64}` — the samples' radii from `center`, ascending, `Δ(ln r) ≤ 1 / r_max` (≤ 1 px at r_max);
- `bin_samples::Vector{Int}` — each angle bin's sample rows (≤ 1 px of arc apart at r_max);
- `profile::Matrix{Float64}` — `profile[a, j]`, the median of angle bin `a` at radius `j`, `NaN` where the bin is skipped;
- `bin_reason::Matrix{Symbol}` — `:ok`, `:off_frame` (a sample's bilinear taps leave the frame) or `:saturated` (more
  than half the bin's samples saturated);
- `statistics` — [`SensorModelNoise`](@ref)`()` or [`EmpiricalNoise`](@ref)`()`, chosen by the frames' type (the
  provenance); `profile_scale::Matrix{Float64}` each bin's noise under the sensor model (`NaN` without one): the median
  over the bin's m_eff distinct pixels of the maximum of the N frames' noise, standard error
  `√v̄ / (2·N·φ(q)·Φ(q)^(N−1)·√m_eff)`, `q = Φ⁻¹(2^(−1/N))`, v̄ the model's variance there (one frame: `√(π/2·v̄/m_eff)`);
  m_eff the bin's rows × its cells' arc, from the log-polar map's differential (Core's `transform_deriv`): its pixels (a
  column is at most a pixel thick, sharing its pixels with its neighbours);
- `noise::Vector{Float64}` — each angle's noise PER PIXEL σ₁ (a bin of m_eff pixels has σ₁/√m_eff): the model's (the
  median over the angle of its bin noise × √m_eff) or the data's;
  `model_scatter::Vector{Float64}` — with a sensor model, the empirical noise over the model's per angle (≈ 1 when the
  model holds; `NaN` without one);
- `candidates::Vector{Vector{Float64}}` — per angle the radii of its candidates;
- `edge`, `edge_scale::Vector{Float64}` — per angle the chosen candidate (`NaN` where none is an inlier of the coverage)
  and its standard uncertainty, the drop's noise through the profile's slope;
- `reason::Vector{Symbol}` — per angle `:ok`, `:no_candidate` (no significant drop) or `:no_inlier`.
"""
struct RimProfile{St, I}
    center::Point2{Float64}
    initial::I
    angles::Vector{Float64}
    radii::Vector{Float64}
    bin_samples::Vector{Int}
    profile::Matrix{Float64}
    bin_reason::Matrix{Symbol}
    statistics::St
    profile_scale::Matrix{Float64}
    noise::Vector{Float64}
    model_scatter::Vector{Float64}
    candidates::Vector{Vector{Float64}}
    edge::Vector{Float64}
    edge_scale::Vector{Float64}
    reason::Vector{Symbol}
end

Base.show(io::IO, r::RimProfile) = print(io, "RimProfile(", length(r.angles), " angles, ", count(==(:ok), r.reason),
    " with an edge; ", length(r.radii), " radii ", round(first(r.radii); digits = 1), "…", round(last(r.radii); digits = 1),
    " px; ", nameof(typeof(r.statistics)), ")")

# The robust noise scale of scalar deviations: Core's mad_scale (median |d| / median of χ₁), the stack's one MAD rule.
_scalar_scale(d) = mad_scale([SVector{1, Float64}(x) for x in d])

# The radius of `model` along the unit direction `u` from its centre (the model's AffineMap carries the unit circle onto
# it: the radius is 1 / ‖A⁻¹u‖ for its linear part A).
_model_radius(model, u) = 1 / norm(AffineMap(model).linear \ u)

# =============================================================================
# The bins: Core's log-polar warp and the bin medians, on the device
# =============================================================================

# The median of each (angle bin, radius) bin of the log-polar samples: angle bin `a` holds the sample rows
# `((a − 1)·rows) ÷ n + 1 … (a·rows) ÷ n` of column `j`; `mask` the same layout of the saturation mask's bilinear samples.
# With the sensor model's variances `vars` (the same layout) it also writes each bin's mean variance to `meanvar`.
# ONE KernelAbstractions kernel, one work item per bin: the median by rank counting (the order statistics ⌊(m+1)/2⌋ and
# ⌈(m+1)/2⌉ of the bin's m samples, O(m²) with no scratch), `NaN` and code 1 when a sample is off the frame (`NaN`),
# code 2 when more than half the bin is saturated. The variance sum accumulates in Float64.
@kernel function _rim_bin_median_kernel!(med, code, meanvar, @Const(samples), @Const(mask), @Const(vars), nbins::Int32, rows::Int32)
    a, j = @index(Global, NTuple)
    T = eltype(med)
    lo = ((Int32(a) - Int32(1)) * rows) ÷ nbins + Int32(1)
    hi = (Int32(a) * rows) ÷ nbins
    m = hi - lo + Int32(1)
    off = false
    nsat = Int32(0)
    @inbounds for k in lo:hi
        off |= isnan(samples[k, j])
        nsat += mask[k, j] > T(0.5) ? Int32(1) : Int32(0)
    end
    m1 = (m + Int32(1)) ÷ Int32(2)
    m2 = (m + Int32(2)) ÷ Int32(2)
    s1 = T(NaN)
    s2 = T(NaN)
    if !off
        @inbounds for k in lo:hi
            v = samples[k, j]
            lt = Int32(0)
            le = Int32(0)
            for l in lo:hi
                w = samples[l, j]
                lt += w < v ? Int32(1) : Int32(0)
                le += w <= v ? Int32(1) : Int32(0)
            end
            (lt < m1 <= le) && (s1 = v)
            (lt < m2 <= le) && (s2 = v)
        end
    end
    @inbounds med[a, j] = off ? T(NaN) : (s1 + s2) / 2
    @inbounds code[a, j] = off ? UInt8(1) : (2 * nsat > m ? UInt8(2) : UInt8(0))
    _rim_bin_meanvar!(meanvar, vars, a, j, lo, hi, m, off)
end
@inline _rim_bin_meanvar!(::Nothing, ::Nothing, a, j, lo, hi, m, off) = nothing
@inline function _rim_bin_meanvar!(meanvar, vars, a, j, lo, hi, m, off)
    v = 0.0
    @inbounds for k in lo:hi
        v += Float64(vars[k, j])
    end
    @inbounds meanvar[a, j] = off ? eltype(meanvar)(NaN) : eltype(meanvar)(v / m)
    return nothing
end

# The log-polar samples of `upper` (its saturation mask `sat`, its variances or `nothing`) around `center`, radii
# r_min … r_max, on `device`: Core's logpolar_map of the circle (center, r_max) with inner radius r_min / r_max,
# image_warp, image_warp!. Radial spacing Δ(ln r) = ln(r_max / r_min) / nr ≤ 1 / r_max and angular Δθ = 2π / rows ≤
# 1 / r_max: ≤ 1 px at r_max both ways, every pixel of the band reached. The grid's geometry is the map's own (`_rim_grid`).
function _rim_samples(upper::AbstractMatrix{<:Real}, sat::AbstractMatrix{Bool}, variance, nframes, center, r_min, r_max; device)
    nr = ceil(Int, log(r_max / r_min) * r_max)
    rows = ceil(Int, 2π * r_max)
    math = logpolar_map(Circle(center, r_max), Rect(0, 0, nr, rows), r_min / r_max)
    bm = image_warp(Float32, math, Rect(0, 0, nr, rows))
    src = on_device(Float32, device, upper)
    samples = similar(src, Float32, rows, nr)
    masks = similar(src, Float32, rows, nr)
    image_warp!(samples, bm, src; fillvalue = NaN32)
    image_warp!(masks, bm, on_device(Float32, device, Matrix{Float32}(sat)); fillvalue = 0.0f0)
    vars = variance === nothing ? nothing : similar(src, Float32, rows, nr)   # the model's variances, bilinear like the values
    vars === nothing || image_warp!(vars, bm, on_device(Float32, device, variance); fillvalue = NaN32)
    θ(row) = (p = math(Point2(nr / 2, row)); atan(p[2] - center[2], p[1] - center[1]))
    return (; samples, masks, vars, nframes, rows, θ, _rim_grid(math, center, nr)...)
end

# The drop's span: 2 px, the bilinear taps' support (the sharpest edge, sampled, spreads over it).
const _RIM_SPAN = 2.0

# The log-polar grid's radial geometry, every quantity through Core's map, its differential and its inverse (the grid is
# a CIRCLE's, `_rim_samples`: nothing depends on the row). Column j holds the grid coordinate u_j = j − 0.5 (the column
# index is x = u + 0.5); with J = ∂(image)/∂(u, v) there (`transform_deriv`):
# - `radii[j]`, its radius from the centre (the forward map);
# - `drdu[j]` = ‖J[:, 1]‖, the radial spacing (px per column);
# - `pixels[j]`, the distinct pixels of one row's cell: its footprint |det J| = ‖J[:, 1]‖·‖J[:, 2]‖ px², but no fewer than
#   one pixel thick radially (adjacent columns closer than a pixel share their pixels), and the grid's radial spacing is
#   ≤ 1 px everywhere (`_rim_samples`), so the arc ‖J[:, 2]‖. A bin of k rows has k·pixels[j];
# - `stencil[j]`, the ends of the drop's stencil, r_j ∓ `_RIM_SPAN`/2 through the inverse map, as interpolation weights
#   (i₋, f₋, i₊, f₊) at the column indices i + f: exactly `_RIM_SPAN` px wide at every radius (i₋ = 0 when an end leaves
#   the band);
# - `radius_at(x)`, the radius at a fractional column index x (the forward map: a sub-sample edge to pixels).
function _rim_grid(math, center, nr)
    radius_at(x) = norm(math(Point2(x - 0.5, 0.5)) - center)
    radii = [radius_at(j) for j in 1:nr]
    J = [transform_deriv(math, Point2(j - 0.5, 0.5)) for j in 1:nr]
    drdu = [norm(Jj[:, 1]) for Jj in J]
    pixels = [norm(Jj[:, 2]) for Jj in J]                                # drdu ≤ 1 px on this grid (nr's construction)
    inv_math = inv(math)
    weights(x) = (i = min(floor(Int, x), nr - 1); (i, x - i))
    half = _RIM_SPAN / 2
    stencil = map(radii) do r
        x₋, x₊ = (inv_math(center + Point2(r + d, 0.0))[1] + 0.5 for d in (-half, half))
        (r > half && 1 <= x₋ && x₊ <= nr) ? (weights(x₋)..., weights(x₊)...) : (0, 0.0, 0, 0.0)
    end
    return (; radii, drdu, pixels, stencil, radius_at)
end

# `f` of the vector `v` at the two ends of column j's stencil, `v` interpolated between its columns; `NaN` without one.
# An interpolated noise counts as fully correlated with its columns' (adjacent columns share pixels).
function _stencil_ends(f, v, s, j)
    i₋, f₋, i₊, f₊ = s.stencil[j]
    i₋ == 0 && return NaN
    return f(_column_value(v, i₋, f₋), _column_value(v, i₊, f₊))
end
# `v` between columns i and i + 1 at the fraction f, a column's own value at f = 0 or 1 (a neighbour's NaN, weighted 0,
# leaves it alone)
_column_value(v, i, f) = f == 0 ? v[i] : f == 1 ? v[i + 1] : (1 - f) * v[i] + f * v[i + 1]

# The `nbins` angle bins' medians of the samples: the host profile, the bins' reasons, centres, sample rows and (under a
# sensor model) each bin's noise, the median's standard error of the maximum of the N frames' noise over its
# rows × `pixels[j]` pixels (RimProfile).
function _rim_bins(s, nbins::Int; device)
    nr = length(s.radii)::Int
    med = similar(s.samples, Float32, nbins, nr)
    code = similar(s.samples, UInt8, nbins, nr)
    meanvar = s.vars === nothing ? nothing : similar(s.samples, Float32, nbins, nr)
    _launch!(_rim_bin_median_kernel!, device, med, code, meanvar, s.samples, s.masks, s.vars, Int32(nbins), Int32(s.rows);
             ndrange = (nbins, nr))   # through the package's launch barrier (JET sees no abstract-backend launch)
    lo(a) = ((a - 1) * s.rows) ÷ nbins
    angles = [s.θ((lo(a) + lo(a + 1)) / 2) for a in 1:nbins]
    counts = [lo(a + 1) - lo(a) for a in 1:nbins]
    reason = map(c -> c == 0x00 ? :ok : c == 0x01 ? :off_frame : :saturated, Array(code))
    pscale = meanvar === nothing ? fill(NaN, nbins, nr) : let N = s.nframes, q = quantile(Normal(), 0.5^(1 / N))
        f = N * pdf(Normal(), q) * cdf(Normal(), q)^(N - 1)
        sqrt.(Float64.(Array(meanvar)) ./ (counts .* s.pixels')) ./ 2f
    end
    return Float64.(Array(med)), reason, angles, counts, pscale
end

# =============================================================================
# The candidates: each angle's significant log-brightness drops
# =============================================================================

# The noise of angle `a`'s bins along the radius, from its noise per PIXEL `σ₁` (a bin of m pixels: σ₁/√m): the sensor
# model's own (`profile_scale`), or σ₁ over the bins' pixels (rows × `pixels[j]`).
_bin_noise(::SensorModelNoise, pscale, counts, s, a, σ₁) = view(pscale, a, :)
_bin_noise(::EmpiricalNoise, pscale, counts, s, a, σ₁) = σ₁ ./ sqrt.(counts[a] .* s.pixels)

# Each angle's noise per pixel σ₁: the sensor model's (the median over the angle of its bins' noise times √(their
# pixels)), or the data's (the profile's drops across the stencil, each over its noise for σ₁ = 1: mad_scale of those,
# the rim's few drops leaving the median alone).
function _rim_noise(stat::SensorModelNoise, profile, pscale, counts, s, a)
    v = filter(isfinite, view(pscale, a, :) .* sqrt.(counts[a] .* s.pixels))
    return isempty(v) ? NaN : median(v)
end
function _rim_noise(stat::EmpiricalNoise, profile, pscale, counts, s, a)
    p, σ = view(profile, a, :), _bin_noise(stat, pscale, counts, s, a, 1.0)
    d = [x for x in (_stencil_ends(-, p, s, j) / _stencil_ends(hypot, σ, s, j) for j in eachindex(p)) if isfinite(x)]
    return isempty(d) ? NaN : _scalar_scale(d)
end

# The candidates of one angle: every EDGE of the profile's OUTER ENVELOPE (the brightest the profile gets at or beyond
# each radius: where the light ends it does not come back, while a scene edge or a dark object inside the disc is
# followed by light again and leaves the envelope alone). Every radius is in PIXELS through the grid's map (`_rim_grid`):
# the drop at column j is across its stencil, exactly `_RIM_SPAN` px about r_j (its ends through the inverse map, the
# profile interpolated between columns), so the drop and its log slope are continuous in r, with no step where the
# grid's spacing changes the sample count of a span. A drop is significant when it exceeds its noise
# σ_D = √(σ(r − 1)² + σ(r + 1)²) (each bin's noise, `σ`, at its pixels) by z = Φ⁻¹(1 − α/m), α the family-wise level over
# the angle's m stencils: the floor's noise alone makes an edge in an angle with probability at most α. An edge is a
# maximal run of consecutive significant drops; its candidate is the steepest LOG slope, per pixel (a ratio:
# independent of the exposure; a blurred sharp stop's is at the stop, a linear ramp's at its foot), searched from the
# run's start up to the next edge's (a soft ramp's significant run ends short of its foot, where its drops sink below the
# noise), the log floored at the angle's darkest level or its noise, whichever is larger. A rim that falls in two steps
# (lit, a dim band, the floor) gives two candidates, both for the RANSAC (owner ruling 2026-11-14). The candidate's
# radius is the log slope's sub-sample peak in the grid coordinate (`_slope_peak`), taken to pixels by the forward map;
# its uncertainty σ_D over the profile's slope (the drop over the stencil's `_RIM_SPAN` px); its strength
# the log slope.
function _rim_candidates(profile, s, σ, α)
    p = _outer_envelope(profile)
    D = [_stencil_ends(-, p, s, j) for j in eachindex(p)]
    σD = [_stencil_ends(hypot, σ, s, j) for j in eachindex(p)]
    valid = isfinite.(D) .& isfinite.(σD)
    m = count(valid)
    m == 0 && return (Float64[], Float64[], Float64[])
    z = quantile(Normal(), 1 - α / m)
    sig = valid .& (D .> z .* σD)
    # the log's floor: the darkest the angle gets (its envelope's last value) or its noise, whichever is larger
    lowest = minimum(x for x in p if !isnan(x); init = Inf)
    σv = filter(isfinite, σ)
    ε = max(isempty(σv) ? 0.0 : median(σv), isfinite(lowest) ? lowest : 0.0, floatmin(Float64))
    logdrop(x, y) = log(max(x, ε)) - log(max(y, ε))                   # of the interpolated profile (NaN stays NaN)
    g = [valid[j] ? _stencil_ends(logdrop, p, s, j) / _RIM_SPAN : -Inf for j in eachindex(p)]
    starts = [j for j in eachindex(sig) if sig[j] && (j == 1 || !sig[j - 1])]
    rad, unc, strength = Float64[], Float64[], Float64[]
    for (n, j0) in enumerate(starts)
        j1 = n < length(starts) ? starts[n + 1] - 1 : length(p)
        best = argmax(j -> g[j], j0:j1)
        push!(rad, _slope_peak(g, s, best))
        push!(unc, σD[best] * _RIM_SPAN / max(D[best], eps()))
        push!(strength, g[best])
    end
    return (rad, unc, strength)
end

# The log slope's sub-sample peak about column i, in the grid coordinate: the parabola through the slopes of columns
# i − 1, i, i + 1 (uniformly spaced in u), within half a column (0 when a neighbour is missing or the three are not
# concave), taken to pixels by the forward map. Candidates on the radial grid would otherwise tie exactly, and a model
# centred on the log-polar origin would fit them with zero residual (so would the maximum of the interpolated slope,
# whose kinks sit where a stencil end crosses a column).
function _slope_peak(g, s, i)
    δ = 0.0
    if 1 < i < length(g) && isfinite(g[i - 1]) && isfinite(g[i + 1])
        curv = g[i - 1] - 2g[i] + g[i + 1]
        curv < 0 && (δ = clamp(0.5 * (g[i - 1] - g[i + 1]) / curv, -0.5, 0.5))
    end
    return s.radius_at(i + δ)
end

# The outer envelope of a profile: at each valid sample the maximum of the valid samples at or beyond it (`NaN` kept).
function _outer_envelope(p)
    q = fill(NaN, length(p))
    m = -Inf
    for j in reverse(eachindex(p))
        isnan(p[j]) && continue
        m = max(m, p[j])
        q[j] = m
    end
    return q
end

# The rim around `initial` (measured plane's pixels) of the maximum `upper`: the samples, the angle bins (`nangles`, or
# chosen as below), the candidates, their RANSAC fit (`T`, at `outlier_halfwidth`) and each angle's edge. Returns
# (rim, fit, candidate points): the fit `nothing` when no model is found.
function _rim_measure(rng, ::Type{T}, upper, sat, variance, nframes, initial; nangles, band, significance,
                      outlier_halfwidth, device) where {T}
    c = Point2{Float64}(initial.center[1], initial.center[2])
    axes = initial isa Circle ? (initial.r, initial.r) : (initial.a, initial.b)
    s = _rim_samples(upper, sat, variance, nframes, c, (1 - band[1]) * minimum(axes), (1 + band[2]) * maximum(axes); device)
    stat = variance === nothing ? EmpiricalNoise() : SensorModelNoise()
    measure(nbins) = _rim_at(rng, T, s, stat, c, initial, nbins; significance, outlier_halfwidth, device)
    nangles isa Integer && return measure(Int(nangles))
    # The angle bins from the image: the binning whose coverage is the most PRECISE, the smallest standard error of its
    # mean radius, the inliers' rms over √(the wedges with an inlier), the rms no finer than the edges' own uncertainty
    # (their median: a scatter below it is a degenerate fit, not a precise one) nor the radial grid's quantization (its
    # spacing over √12). Finer bins give more wedges but noisier medians and,
    # in a textured scene, the scene's edges among each wedge's candidates (the rms grows); coarser bins, fewer wedges.
    # The ladder doubles from 45 bins (8°) to the finest with 3 sample rows per bin (a median with a breakdown: one bad
    # pixel of three); a binning counts with at least 20 wedges with an inlier (four times the ellipse's minimal sample).
    δ = maximum(s.drdu)                                               # the radial grid's spacing (≤ 1 px), ∂r/∂u
    best, best_se = nothing, Inf
    nbins = 45
    while nbins <= max(45, s.rows ÷ 3)
        m = measure(nbins)
        rim, fit = m[1], m[2]
        support = count(==(:ok), rim.reason)
        if fit !== nothing && support >= 20
            noise = median(filter(isfinite, rim.edge_scale))           # the edges' own uncertainty: no fit is more precise
            se = sqrt(max(fit.rms, isfinite(noise) ? noise : 0.0)^2 + δ^2 / 12) / sqrt(support)
            se < best_se && ((best, best_se) = (m, se))
        end
        best === nothing && nbins * 2 > s.rows ÷ 3 && (best = m)       # no binning fits: the finest, its reasons recorded
        nbins *= 2
    end
    return something(best, measure(45))
end

# The rim at `nbins` angle bins of the samples `s`: the bins, each angle's noise and candidates, the RANSAC fit and each
# angle's edge (the inlier candidate nearest the model).
function _rim_at(rng, ::Type{T}, s, stat, c, initial, nbins; significance, outlier_halfwidth, device) where {T}
    profile, bin_reason, angles, counts, pscale = _rim_bins(s, nbins; device)
    noise = [_rim_noise(stat, profile, pscale, counts, s, a) for a in 1:nbins]
    scatter = _model_scatter(stat, profile, pscale, counts, s, noise)
    cand = [_rim_candidates(view(profile, a, :), s, _bin_noise(stat, pscale, counts, s, a, noise[a]), significance) for a in 1:nbins]
    ray(a, r) = c + r * Point2(cos(angles[a]), sin(angles[a]))
    # ONE point per wedge in each fit, so the support counts wedges and a wedge with many edges does not dominate:
    # (1) the RANSAC on each wedge's OUTERMOST edge (where its light ends); (2) every wedge's edge NEAREST that model
    # (any of its edges: an outermost one that is stray light, or a step off the model, gives way to
    # the wedge's consistent edge); (3) the RANSAC on those, the coverage, and each wedge's edge an inlier of it.
    function fit_on(pick)
        as = [a for a in 1:nbins if pick[a] > 0]
        pts = [ray(a, first(cand[a])[pick[a]]) for a in as]
        trial = length(pts) >= 5 ? _fit_image_circle(T, pts; outlier_halfwidth, rng = copy(rng)) : nothing
        return (trial === nothing || !found_model(trial)) ? (nothing, as, pts) : (trial, as, pts)
    end
    outer = [isempty(first(cand[a])) ? 0 : argmax(first(cand[a])) for a in 1:nbins]
    fit, as, pts = fit_on(outer)
    edge, edge_scale = fill(NaN, nbins), fill(NaN, nbins)
    reason = [isempty(first(cand[a])) ? :no_candidate : :no_inlier for a in 1:nbins]
    if fit !== nothing
        m = estimate(fit)
        off(q) = (d = q - m.center; abs(norm(d) - _model_radius(m, d / norm(d))))
        nearest = [isempty(first(cand[a])) ? 0 : argmin(r -> off(ray(a, r)), first(cand[a])) |> r -> findfirst(==(r), first(cand[a])) for a in 1:nbins]
        fit2, as2, pts2 = fit_on(nearest)
        if fit2 !== nothing
            fit, as, pts = fit2, as2, pts2
            inl = inlier_mask(fit)
            for (i, a) in enumerate(as)
                inl[i] || continue
                edge[a], edge_scale[a], reason[a] = first(cand[a])[nearest[a]], cand[a][2][nearest[a]], :ok
            end
        end
    end
    fit === nothing || any(==(:ok), reason) || (fit = nothing)       # the second fit failed: no coverage from this binning
    rim = RimProfile(c, initial, angles, s.radii, counts, profile, bin_reason, stat, pscale, noise, scatter,
                     [first(x) for x in cand], edge, edge_scale, reason)
    return rim, fit, pts
end

# The sensor model against the data: each angle's empirical noise over the model's (`NaN`s without a model).
_model_scatter(::EmpiricalNoise, profile, pscale, counts, s, noise) = fill(NaN, length(noise))
_model_scatter(::SensorModelNoise, profile, pscale, counts, s, noise) =
    [_rim_noise(EmpiricalNoise(), profile, pscale, counts, s, a) / noise[a] for a in eachindex(noise)]

# The coverage's uncertainty: the covariance of the model `m` in Core's `ambient_coordinates` (an ellipse: centre and the
# form Q's (Q₁₁, Q₁₂, Q₂₂); a circle: centre and radius) from the inliers `pts`, s²·solver_covariance (the
# VisualGeometryToolkitEstimation extension: the unit-noise covariance of the points' constraint rows), s² the inliers'
# squared distances to `m` along their rays (the orthogonal distance to first order on a near-circle) over n − dof; and
# the standard deviations of the centre and the semi-axes propagated from it by Core's ellipse chart: the covariance on
# the log-shape chart (cx, cy, S₁₁, S₁₂, S₂₂) through the inverse of `retract_jacobian(EllipseManifold(m))` (its
# ∂ambient/∂chart at `m`), and log a, log b the eigenvalues of S along the axes u = (cos θ, sin θ) and its normal:
# d(log a) = uᵀ dS u (the symmetric eigenvalue's derivative), σ_a = a·√(wᵀ Σ_S w).
function _coverage_uncertainty(m, pts)
    n, dof = length(pts), length(ambient_coordinates(m))
    Σ̃ = n > dof ? _coverage_covariance(m, pts) : nothing
    Σ̃ === nothing && return (; covariance = nothing, center_scale = (NaN, NaN), axes_scale = (NaN, NaN))
    d = [norm(q - m.center) - _model_radius(m, (q - m.center) / norm(q - m.center)) for q in pts]   # along the rays: ≈ orthogonal on a near-circle
    Σ = sum(abs2, d) / (n - dof) * Σ̃
    return (; covariance = Σ, center_scale = (sqrt(Σ[1, 1]), sqrt(Σ[2, 2])), axes_scale = _axes_scale(m, Σ))
end
_axes_scale(m::Circle, Σ) = (sqrt(Σ[3, 3]), sqrt(Σ[3, 3]))
function _axes_scale(m::Ellipse, Σ)
    J = retract_jacobian(EllipseManifold(m))                          # ∂(cx, cy, Q₁₁, Q₁₂, Q₂₂)/∂(cx, cy, S₁₁, S₁₂, S₂₂) at m
    ΣS = J \ Σ / J'
    s, c = sincos(m.θ)
    wa = SVector(0.0, 0.0, c^2, 2s * c, s^2)                          # uᵀ dS u along the a axis, u = (c, s)
    wb = SVector(0.0, 0.0, s^2, -2s * c, c^2)                         # along the b axis, (−s, c)
    return (m.a * sqrt(wa' * ΣS * wa), m.b * sqrt(wb' * ΣS * wb))
end

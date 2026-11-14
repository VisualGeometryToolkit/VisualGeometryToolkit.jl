# =============================================================================
# The image circle's RIM: log-polar profiles of the frames' maximum, the rim's log-brightness drops, the coverage fit
# =============================================================================
#
# detect_image_circle (image_circle.jl) measures the lens coverage here. The frames' per-pixel maximum is resampled by
# Core's log-polar warp (`logpolar_map` → `image_warp` → `image_warp!`, ONE KernelAbstractions kernel on every backend)
# around an initial model (the image centre by default) at ≤ 1 px in radius and in arc; the samples are grouped into
# angle bins and each (angle bin, radius) is reduced to its MEDIAN by this file's kernel (same backend). Per angle the
# profile's largest significant drops in LOG brightness are the rim's CANDIDATES (a log drop is a ratio: independent of
# the exposure); the RANSAC ellipse (or circle) through every angle's candidates (`_fit_image_circle`, the
# VisualGeometryToolkitEstimation extension) is the LENS COVERAGE, and per angle the inlier candidate nearest it is the
# rim's edge there.
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: norm
using Statistics: median
using Distributions: Normal, pdf, cdf
using VisualGeometryToolkitCore: logpolar_map, mad_scale, ambient_coordinates, scale, retract_jacobian, EllipseManifold

"""
    SensorModelNoise()

A [`RimProfile`](@ref)'s noise statistics (its `statistics`) when the frames are `Uncertain` cells (a RAW read under a
sensor model with a variance law): each bin's noise is the model's.
"""
struct SensorModelNoise end

"""
    EmpiricalNoise()

A [`RimProfile`](@ref)'s noise statistics (its `statistics`) when the frames carry no variances: each angle's noise is
the data's own, Core's `mad_scale` of the profile's differences between samples 2 px apart, over √2.
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
  over the bin's m_eff distinct pixels (its rows × r / r_max) of the maximum of the N frames' noise, standard error
  `√v̄ / (2·N·φ(q)·Φ(q)^(N−1)·√m_eff)`, `q = Φ⁻¹(2^(−1/N))`, v̄ the model's variance there (one frame: `√(π/2·v̄/m_eff)`);
- `noise::Vector{Float64}` — each angle's noise scale (the model's median bin noise, or the empirical one);
  `model_scatter::Vector{Float64}` — with a sensor model, the empirical noise over the model's per angle (≈ 1 when the
  model holds; `NaN` without one);
- `candidates::Vector{Vector{Float64}}` — per angle the radii of its candidates;
- `edge`, `edge_scale::Vector{Float64}` — per angle the chosen candidate (`NaN` where none is an inlier of the coverage)
  and its standard uncertainty, the noise through the drop's slope;
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
# 1 / r_max: ≤ 1 px at r_max both ways, every pixel of the band reached. The radii are read from the warp's own map.
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
    radii = [norm(math(Point2(j - 0.5, 0.5)) - center) for j in 1:nr]
    θ(row) = (p = math(Point2(nr / 2, row)); atan(p[2] - center[2], p[1] - center[1]))
    return (; samples, masks, vars, nframes, radii, rows, θ)
end

# The `nbins` angle bins' medians of the samples: the host profile, the bins' reasons, centres, sample rows and (under a
# sensor model) each bin's noise, the median's standard error of the maximum of the N frames' noise (RimProfile).
function _rim_bins(s, nbins::Int; device)
    nr = length(s.radii)::Int
    med = similar(s.samples, Float32, nbins, nr)
    code = similar(s.samples, UInt8, nbins, nr)
    meanvar = s.vars === nothing ? nothing : similar(s.samples, Float32, nbins, nr)
    OnBackend(device)(_rim_bin_median_kernel!)(med, code, meanvar, s.samples, s.masks, s.vars, Int32(nbins), Int32(s.rows);
                                               ndrange = (nbins, nr))
    KA.synchronize(device)
    lo(a) = ((a - 1) * s.rows) ÷ nbins
    angles = [s.θ((lo(a) + lo(a + 1)) / 2) for a in 1:nbins]
    counts = [lo(a + 1) - lo(a) for a in 1:nbins]
    reason = map(c -> c == 0x00 ? :ok : c == 0x01 ? :off_frame : :saturated, Array(code))
    pscale = fill(NaN, nbins, nr)
    if meanvar !== nothing
        q = quantile(Normal(), 0.5^(1 / s.nframes))
        f = s.nframes * pdf(Normal(), q) * cdf(Normal(), q)^(s.nframes - 1)
        meff = counts .* (s.radii' ./ last(s.radii))
        pscale = sqrt.(Float64.(Array(meanvar))) ./ (2f .* sqrt.(meff))
    end
    return Float64.(Array(med)), reason, angles, counts, pscale
end

# =============================================================================
# The candidates: each angle's significant log-brightness drops
# =============================================================================

# The drop's span: 2 px, the bilinear taps' support (the sharpest edge, sampled, spreads over it).
const _RIM_SPAN = 2.0

# The (j, k) sample pairs `_RIM_SPAN` apart (k the first sample at least the span outward of j), both valid.
_rim_spans(p, radii) = [(j, k) for j in eachindex(p) for k in (searchsortedfirst(radii, radii[j] + _RIM_SPAN),)
                        if k <= length(p) && !isnan(p[j]) && !isnan(p[k])]

# Each angle's noise scale: the sensor model's (the median bin noise over the angle), or the data's (the profile's
# differences across the span, whose scale is √2 σ: mad_scale over √2, the rim's few drops leaving the median alone).
function _rim_noise(::SensorModelNoise, profile, pscale, radii, a)
    v = filter(isfinite, view(pscale, a, :))
    return isempty(v) ? NaN : median(v)
end
function _rim_noise(::EmpiricalNoise, profile, pscale, radii, a)
    d = [profile[a, j] - profile[a, k] for (j, k) in _rim_spans(view(profile, a, :), radii)]
    return isempty(d) ? NaN : _scalar_scale(d) / sqrt(2)
end

# The candidates of one angle: every EDGE of the profile's OUTER ENVELOPE (the brightest the profile gets at or beyond
# each radius: where the light ends it does not come back, while a scene edge or a dark object inside the disc is
# followed by light again and leaves the envelope alone). An edge is a maximal run of consecutive significant drops
# across the span; its candidate is the run's steepest drop in LOG brightness (a ratio: independent of the exposure;
# a blurred sharp stop's steepest log drop is at the stop, a linear ramp's at its foot), so a rim that falls in two
# steps (lit, a dim band, the floor) gives two candidates, both for the RANSAC (owner ruling 2026-11-14). A drop is
# significant when it exceeds its noise √2·σ (σ the angle's noise) by z = Φ⁻¹(1 − α/m), α the family-wise level over
# the angle's m spans: the floor's noise alone makes an edge in an angle with probability at most α. The log is floored
# at the angle's darkest level or its noise, whichever is larger, and an edge's steepest drop is searched up to the next
# edge: a soft ramp's significant run ends short of its foot (where its drops sink below the noise), and the floored log
# drop is steepest at the foot, where the ramp meets the floor. The candidate's radius is its span's middle (refined to
# the log drop's sub-sample peak), its uncertainty
# √2·σ over the span's slope; its strength the log drop.
function _rim_candidates(profile, radii, σ, α)
    (isfinite(σ) && σ >= 0) || return (Float64[], Float64[], Float64[])
    p = _outer_envelope(profile)
    spans = _rim_spans(p, radii)
    isempty(spans) && return (Float64[], Float64[], Float64[])
    z = quantile(Normal(), 1 - α / length(spans))
    sig = [p[j] - p[k] > z * sqrt(2) * σ for (j, k) in spans]
    # the log's floor: the darkest the angle gets (its envelope's last value) or its noise, whichever is larger
    lowest = minimum(x for x in p if !isnan(x); init = Inf)
    ε = max(σ, isfinite(lowest) ? lowest : 0.0, floatmin(Float64))
    # the log drop per pixel: a span holds the samples to at least 2 px (on the log grid its length jumps between 2 and
    # 2 + Δr, Δr the sample spacing; the jump at Δr = 2/3 px made a ring at a fixed radius about any origin)
    logdrop(t) = (log(max(p[spans[t][1]], ε)) - log(max(p[spans[t][2]], ε))) / (radii[spans[t][2]] - radii[spans[t][1]])
    # the edges: maximal runs of consecutive significant spans; each searched for its steepest floored log drop from
    # its start to the next edge's start (a soft ramp's run ends where its 2-px drops sink below the noise, short of the
    # foot where it meets the floor: the floored log drop is steepest there, a level of the profile, not of the noise)
    starts = [t for t in eachindex(spans) if sig[t] && (t == 1 || !sig[t - 1] || spans[t - 1][1] != spans[t][1] - 1)]
    rad, unc, strength = Float64[], Float64[], Float64[]
    for (n, t0) in enumerate(starts)
        t1 = n < length(starts) ? starts[n + 1] - 1 : length(spans)
        best = argmax(t -> logdrop(t), t0:t1)
        jb, kb = spans[best]
        push!(rad, _subsample_middle(p, radii, spans, best, ε))
        push!(unc, sqrt(2) * σ * (radii[kb] - radii[jb]) / max(p[jb] - p[kb], eps()))
        push!(strength, logdrop(best))
    end
    return (rad, unc, strength)
end

# The middle of span `i`, refined between its neighbours' middles by the parabola through the three spans' log drops
# (the peak's sub-sample position, within half a span step): candidates on the radial grid would otherwise tie exactly,
# and a model centred on the log-polar origin would fit them with zero residual.
function _subsample_middle(p, radii, spans, i, ε)
    mid(t) = (radii[spans[t][1]] + radii[spans[t][2]]) / 2
    drop(t) = (log(max(p[spans[t][1]], ε)) - log(max(p[spans[t][2]], ε))) / (radii[spans[t][2]] - radii[spans[t][1]])   # per pixel
    (1 < i < length(spans) && spans[i - 1][1] == spans[i][1] - 1 && spans[i + 1][1] == spans[i][1] + 1) || return mid(i)
    d₋, d₀, d₊ = drop(i - 1), drop(i), drop(i + 1)
    curv = d₋ - 2d₀ + d₊
    curv < 0 || return mid(i)
    δ = clamp(0.5 * (d₋ - d₊) / curv, -0.5, 0.5)
    return mid(i) + δ * (δ >= 0 ? mid(i + 1) - mid(i) : mid(i) - mid(i - 1))
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
    δ = maximum(diff(s.radii))                                        # the radial grid's spacing (≤ 1 px)
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
    noise = [_rim_noise(stat, profile, pscale, s.radii, a) for a in 1:nbins]
    scatter = _model_scatter(stat, profile, pscale, s.radii, noise)
    cand = [_rim_candidates(view(profile, a, :), s.radii, noise[a], significance) for a in 1:nbins]
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
_model_scatter(::EmpiricalNoise, profile, pscale, radii, noise) = fill(NaN, length(noise))
_model_scatter(::SensorModelNoise, profile, pscale, radii, noise) =
    [_rim_noise(EmpiricalNoise(), profile, pscale, radii, a) / noise[a] for a in eachindex(noise)]

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

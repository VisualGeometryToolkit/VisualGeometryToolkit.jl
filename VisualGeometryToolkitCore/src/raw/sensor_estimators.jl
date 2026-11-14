# =============================================================================
# The sensor estimators — a characterisation's components from frames, offline (D1)
# =============================================================================
#
#   photon_transfer   the ONE estimator of the gain K: the photon-transfer series (frames at equal exposure: per
#                     green photosite their temporal variance, the intercept the darks' at each exposure), a
#                     measured PoissonGaussian in the stack's FitResult; and, scheduled for removal in the next
#                     breaking wave, one frame's green pairs (an estimated PoissonGaussian); one block kernel, one fit
#   master_dark       the ONE dark-frame estimator: a Dark (its ChannelBlack, the
#                     per-photosite offset and variance), the ReadNoise, the defects
#   defect_onsets     per-exposure defect masks into ONE DefectList (each site's
#                     shortest exposure at which it is flagged)
#
# The resolver never runs them (sensor_model is a lookup); a characterisation's
# entry is built from what they return.
# =============================================================================

# ------------------------------------------------------------------------------
# The estimators
# ------------------------------------------------------------------------------

"""
    NoiseEstimateFailed(reason)

The exception [`photon_transfer`](@ref) and [`master_dark`](@ref) throw when
the data cannot give the estimate (too few usable samples, a degenerate fit),
`reason` a sentence. The resolver [`sensor_model`](@ref) never runs an
estimator (a lookup): a characterisation without the estimate lacks the
component, and a frame resolves to its absent type.
"""
struct NoiseEstimateFailed <: Exception
    reason::String
end
Base.showerror(io::IO, e::NoiseEstimateFailed) = print(io, "NoiseEstimateFailed: ", e.reason)

# The fit's points: each the median of at least this many whole blocks (an exposure of fewer gives no point).
const _PTC_BLOCKS_PER_POINT = 8
# The weighted line's cap on its reweighting iterations (`_ptc_line`).
const _PTC_LINE_MAX_ITER = 100
# The default family-wise significance of the μ² and rollover tests (configs/default.toml, [photon_transfer]).
const _PTC_SIGNIFICANCE = toml_default(_CORE_DEFAULTS, "photon_transfer", "significance")::Float64

# Per block of `B × B` 2×2 cells, one work item each: the count of usable samples and the sums Σm, Σd and Σq the fit
# reads, from the cell rule `rule` (`_cell_moments`):
# - `_GreenPair` (one frame): per cell its two greens, m = their black-subtracted mean, d = their difference and q = d²
#   (a cell with a green at the white level, or masked, gives nothing);
# - `_FrameStack` (N frames at one exposure, each scaled to the group's mean): per cell each green photosite, m = its
#   mean over the frames, d = 0 and q = Σₖ(yₖ − m)², its sum of squares about that mean (a photosite at the white level
#   in any frame, or masked, gives nothing).
# The photosites are read from the held arrays by the channel view's index rule (the greens' cell sites `s1`, `s2`,
# `_QUAD_SITES` entries), as LinearRaw's cell kernel reads them; the blacks are the greens' (a dark map's offsets per
# photosite when given, `_black_at`). A cell that gives nothing adds zeros, which leaves every sum bit for bit as is.
@kernel function _difference_blocks_kernel!(n, sm, sd, sd2, rule, dark, mask, black, white::Float64, B::Int, origin)
    bi, bj = @index(Global, NTuple)
    H2, W2 = _cell_extent(rule)
    c = 0; t1 = 0.0; t2 = 0.0; t3 = 0.0
    for jj in 1:B, ii in 1:B
        i = (bi - 1) * B + ii; j = (bj - 1) * B + jj
        (i <= H2 && j <= W2) || continue
        k, a, b, q = _cell_moments(rule, i, j, dark, mask, black, white, origin)
        c += k; t1 += a; t2 += b; t3 += q
    end
    @inbounds n[bi, bj] = c; @inbounds sm[bi, bj] = t1; @inbounds sd[bi, bj] = t2; @inbounds sd2[bi, bj] = t3
end

# One frame's green pairs: the held array and the two greens' cell sites.
struct _GreenPair{A, S}
    raw::A
    s1::S
    s2::S
end
Adapt.adapt_structure(to, r::_GreenPair) = _GreenPair(Adapt.adapt(to, r.raw), r.s1, r.s2)
@inline _cell_extent(r::_GreenPair) = (size(r.raw, 1) ÷ 2, size(r.raw, 2) ÷ 2)
@inline function _cell_moments(r::_GreenPair, i, j, dark, mask, black, white, origin)
    r1 = 2i - 1 + r.s1[1]; c1 = 2j - 1 + r.s1[2]
    r2 = 2i - 1 + r.s2[1]; c2 = 2j - 1 + r.s2[2]
    @inbounds x1 = r.raw[r1, c1]; @inbounds x2 = r.raw[r2, c2]
    (_at_white(x1, white) || _at_white(x2, white)) && return (0, 0.0, 0.0, 0.0)
    (_masked(mask, r1 + origin[1], c1 + origin[2]) || _masked(mask, r2 + origin[1], c2 + origin[2])) && return (0, 0.0, 0.0, 0.0)
    y1 = Float64(x1) - _black_at(dark, black[1], r1 + origin[1], c1 + origin[2])
    y2 = Float64(x2) - _black_at(dark, black[2], r2 + origin[1], c2 + origin[2])
    return (1, (y1 + y2) / 2, y1 - y2, (y1 - y2)^2)
end

# N frames at one exposure, stacked (H × W × N), each frame's black-subtracted values times its `scale` (the group's
# mean over the frame's); the greens' cell sites.
struct _FrameStack{A, V, S}
    stack::A
    scale::V
    s1::S
    s2::S
end
Adapt.adapt_structure(to, r::_FrameStack) = _FrameStack(Adapt.adapt(to, r.stack), Adapt.adapt(to, r.scale), r.s1, r.s2)
@inline _cell_extent(r::_FrameStack) = (size(r.stack, 1) ÷ 2, size(r.stack, 2) ÷ 2)
@inline function _cell_moments(r::_FrameStack, i, j, dark, mask, black, white, origin)
    k1, m1, q1 = _site_moments(r, 2i - 1 + r.s1[1], 2j - 1 + r.s1[2], dark, mask, black[1], white, origin)
    k2, m2, q2 = _site_moments(r, 2i - 1 + r.s2[1], 2j - 1 + r.s2[2], dark, mask, black[2], white, origin)
    return (k1 + k2, m1 + m2, 0.0, q1 + q2)
end
# One photosite over the frames: (1, its mean, its sum of squares about the mean), or nothing when any frame is at the
# white level there or the photosite is masked.
@inline function _site_moments(r::_FrameStack, row, col, dark, mask, black, white, origin)
    _masked(mask, row + origin[1], col + origin[2]) && return (0, 0.0, 0.0)
    ok, μ, q = _temporal_moments(r.stack, row, col, _black_at(dark, black, row + origin[1], col + origin[2]), r.scale, white)
    return ok ? (1, μ, q) : (0, 0.0, 0.0)
end

# THE temporal moments of one photosite of stacked frames (H × W × n), read by master_dark's kernel and the series' cell
# rule alike: the mean over the frames of its codes less `b`, each frame's times its scale (`nothing`: 1), and the sum
# of squares about that mean; `(false, 0, 0)` when a frame's code is at the white level `white` (`nothing`: none is),
# tested in the same read of the frames as the sum.
@inline _frame_scale(::Nothing, k) = 1.0
@inline _frame_scale(scale, k) = @inbounds scale[k]
@inline _at_white(code, ::Nothing) = false
@inline function _temporal_moments(stack, row, col, b::Float64, scale, white)
    nf = size(stack, 3)
    s = 0.0
    for k in 1:nf
        @inbounds x = stack[row, col, k]
        _at_white(x, white) && return (false, 0.0, 0.0)
        s += (Float64(x) - b) * _frame_scale(scale, k)
    end
    μ = s / nf
    q = 0.0
    for k in 1:nf
        @inbounds q += ((Float64(stack[row, col, k]) - b) * _frame_scale(scale, k) - μ)^2
    end
    return (true, μ, q)
end

# THE stacking of frames: an H × W × n buffer on the frames' device (`_frame_stack`), filled frame k into slice k
# (`_stack_frames!`, reused group by group by the series).
_frame_stack(frames, n::Integer) = similar(parent(first(frames)), eltype(parent(first(frames))), size(first(frames))..., n)
function _stack_frames!(stack, frames)
    for (k, f) in enumerate(frames)
        copyto!(view(stack, :, :, k), parent(f))
    end
    return stack
end

# The blocks every estimate reads: the whole blocks (the most usable samples any block has, `full`), at least 16 of at
# least 8 samples.
function _whole_blocks(n)
    full = maximum(n)
    keep = findall(==(full), vec(n))                     # whole blocks only: one χ² law, one median correction
    full >= 8 && length(keep) >= 16 || throw(NoiseEstimateFailed(
        "too few usable blocks: $(length(keep)) unclipped blocks of $(full) samples (at least 16 of at least 8 samples)"
    ))
    return full, keep
end

"""
    PhotonTransferMetadata

The [`FitResult`](@ref) metadata of [`photon_transfer`](@ref)`(groups, darks)`
(its estimate a [`PoissonGaussian`](@ref)), read through the result's forwarded
properties:

- `level_range::NTuple{2, Float64}` — the fit's lowest and highest point (block
  means, codes above black); `level_fraction` the same as fractions of the white
  level above black (the greens' span);
- `intercept::Float64` — σ₀² (codes²), the variance at zero signal, fixed from
  the darks at each group's own exposure (the median over the fitted points);
- `curvature`, `curvature_t` — the μ² coefficient `c` of `v = σ₀² + μ/K + cμ²`
  over the range and its `t` (`c` over its standard error);
- `tests`, `test_level` — the μ² tests the lower end's search can perform (`m`,
  the candidate lower ends) and each test's two-sided level, `significance / m`
  (Bonferroni: the search's family-wise rate is `significance`);
  `rollover_tests`, `rollover_level` the same for the upper end's rollover tests
  (one-sided);
- `birge_ratio` — R = √(χ²/dof) of the fitted points about the line (each in its
  median's standard deviation from the counts; dof the points less the one
  slope): the scatter the counts explain is R ≤ 1; when R > 1, K's standard
  error is multiplied by R (the Birge ratio: never shrunk);
- `points_offered` — the points before the range search (the result's `n` is the
  points fitted).
"""
struct PhotonTransferMetadata
    level_range::NTuple{2, Float64}
    level_fraction::NTuple{2, Float64}
    intercept::Float64
    curvature::Float64
    curvature_t::Float64
    tests::Int
    test_level::Float64
    rollover_tests::Int
    rollover_level::Float64
    birge_ratio::Float64
    points_offered::Int
end

"""
    photon_transfer(groups::AbstractDict{<:Quantity, <:AbstractVector{<:RawMosaic}}, darks::Darks;
                    defects::DefectComponent = NoDefects(), block = 8, normalize = true, significance = 0.001)
        -> FitResult{PoissonGaussian, PhotonTransferMetadata}
    photon_transfer(groups::AbstractDict{<:Quantity, <:AbstractVector{<:RawMosaic}},
                    darks::AbstractDict{<:Quantity, <:ReadNoise}; sensor_model::SensorModel = SensorModel(),
                    block = 8, normalize = true, significance = 0.001) -> FitResult{PoissonGaussian, PhotonTransferMetadata}
    photon_transfer(mosaic::RawMosaic; sensor_model::SensorModel = SensorModel(), block = 8, significance = 0.001)
        -> PoissonGaussian

THE estimator of a sensor's Poisson–Gaussian noise for a characterisation's
entry, built offline (owner ruling D1: the resolver [`sensor_model`](@ref)
never estimates): the conversion gain K (electrons per code), by a
photon-transfer fit of variance against mean, `v = σ₀² + μ/K`.

**The photon-transfer series** (`groups`, exposure `=>` the frames at that
exposure, at least 2 each, of one body at one ISO), a MEASURED
[`PoissonGaussian`](@ref) (`SensorSource.measured`, a dedicated capture) in a
[`FitResult`](@ref) (its [`estimate`](@ref)), its metadata a
[`PhotonTransferMetadata`](@ref): per green photosite its TEMPORAL variance over
the group's frames, which cancels the scene and every fixed pattern, about its
mean μ. Pairs are the case of 2 frames. With `normalize` (the default) each frame
is first scaled to its group's mean (its greens' black-subtracted mean over the
group's: a source that flickers between frames adds no μ² term). Per block, the
photosites' means and their pooled variance give a point, `v̂/v` following
`χ²_{p(N−1)}/(p(N − 1))` over a whole block of `p` photosites and `N` frames;
every group gives its first `N` frames, `N` the smallest group's, so every block
follows that one law. The darks come in two forms:

- a [`Darks`](@ref) (Core 0.10.4; [`master_dark`](@ref)'s Darks, one per
  exposure band): each group is served by THE Dark whose band holds its exposure
  (the Darks' own rule, as [`sensor_model`](@ref) resolves a frame: an exposure
  of any number type, `10 * ms` and `0.01 s` alike). Each group is linearised
  with its own Dark — its offset map per photosite and its channel black — and
  its intercept `σ₀²(t)` is that Dark's mean green variance over the photosites
  `defects` leaves (cut to the group's exposure: a site is masked from its
  onset); σ_r's standard error is carried from the Dark's frame and photosite
  counts, `√(2/((n − 1)N))·σ₀²` (master_dark's rule). A group whose exposure no
  band serves is an `ArgumentError`: never the shared black, never another
  band's Dark;
- a `Dict` of [`ReadNoise`](@ref) (exposure `=>` master_dark's ReadNoise over
  the darks of that exposure; scheduled for removal in the next breaking wave,
  use the `Darks` form): every group linearised with `sensor_model`'s black and
  defects (ONE model for every exposure), its intercept the greens'
  `σ[c + 1]²` of `darks[t]`. A group whose exposure has no entry in `darks`
  (EQUAL in seconds: `1//100 s` is not the `Float64` `0.01 s`) is an
  `ArgumentError`: never interpolated, never substituted. σ_r's standard error
  is the ReadNoise's (`0` when it states none).

- THE INTERCEPT is fixed at each group's own exposure t; the fit is the slope
  alone, `v − σ₀²(t) = μ/K`.
- THE RANGE is derived from the data, at least 3 points: the upper end below the
  variance's rollover (the top point dropped while it lies significantly below
  the line fitted to the points under it, one-sided), then the lower end, the
  lowest point from which the μ² test on the line succeeds (the lowest point dropped
  until it does). Each search runs its tests at `significance / m`, `m` the tests
  it can perform (Bonferroni: the search's family-wise false-alarm rate is
  `significance`); the range (in codes and as fractions of the white level), `m`
  and the levels are in the result's metadata.
- THE STANDARD ERROR of K is the fitted line's from the counts, multiplied by
  the Birge ratio R = √(χ²/dof) of the fitted points when R > 1 (the points
  scatter more than the counts explain); when R ≤ 1 it is unchanged, never
  shrunk. R is in the result's metadata.
- The result's σ_r is `√σ₀²` of the fitted points' darks (their median), its
  standard error the darks' at that median (the mean of the two middle points'
  when it falls between two exposures), by first order; a characterisation
  states σ_r per colour from `master_dark`'s `ReadNoise`.
- The result's return code and stop rule are the weighted line's: `Success` and
  `IncrementSize` when its weights reached their fixed point (`iterations` the
  reweightings), `MaxIters` and `IterationCap` when they did not.

**One frame** (`mosaic`), in the line of Foi et al. (2008), an ESTIMATED
`PoissonGaussian` (`SensorSource.estimated`): unreliable on flats, scheduled for
removal in the next breaking wave; use the multi-frame method. A single flat
frame cannot certify its own K (its brightness range gives the μ² test little
power); for flat fields use the multi-frame photon-transfer method. Each 2×2
cell's two greens see the same light, so their difference `d` carries no scene
where the scene is flat across the cell, and `Var d = 2(μ/K + σ_r²)` about their
black-subtracted mean μ. Per block, the cells' mean and the difference's sample
variance give a point `(μ̂, v̂ = s²/2)`, `v̂/v` following `χ²_{n−1}/(n − 1)` over a
whole block of `n` usable samples (cells); the line's intercept σ_r² is fitted,
over every point, with its standard error (σ_r stated for every colour index, the
greens'). A fixed pattern between the two greens (their gain or response
mismatch) does NOT cancel in `d`: it adds a term in μ², which the μ² test
REFUSES; a frame that gives fewer than 4 points, on which the μ² test cannot
run, is refused too (its K is never returned untested).

Both: per block of `block × block` cells (one KernelAbstractions work item each,
on the frames' device), a sample with a photosite at the white level, or under the
model's defect mask, is left out (CLIPPED, never modelled), and only whole blocks
(the most usable samples any block has) enter the fit. The blocks are sorted by
μ̂ into points of equal count (`min(32, blocks ÷ 8)`; the series': per exposure,
so a point never mixes two exposures' intercepts, an exposure of fewer than 8
whole blocks left out); each point's median μ̂ and median `v̂ / q` (`q` that law's
median, so the ratio is median-unbiased for v).
The line is fitted by weighted least squares, each point weighted by the inverse
of its median's variance from the counts, `v²/(4 m f(q)² q²)` over a point of `m`
blocks (`f` the law's density: the sample median's asymptotic variance), with v
the line's own prediction (iterated to a fixed point from a start weighted at
each point's own value, at most 100 reweightings). The median is robust to the
blocks a scene edge raises. The standard errors are the fitted line's covariance
from those counts; K's by first order.

THE μ² TEST: the points fitted again with a μ² term, `v = … + μ/K + cμ²`, at the
same weights, and `c` significant when `|ĉ / se(ĉ)|` exceeds the two-sided normal
quantile at the test's level (one frame: `significance`, 3.29 at the default
0.001, over every point; the series: the lower end's search). It never returns a
K biased by a μ² term it detects. The default `significance` is
`configs/default.toml`'s `[photon_transfer] significance`.

`sensor_model` gives what is subtracted and left out, as it does for
[`LinearRaw`](@ref): its black slot's per-colour black (a [`Dark`](@ref)'s
offset per photosite in its place) and its defect slot's photosites; its noise
slot is not read. Throws [`NoiseEstimateFailed`](@ref) with the reason when
fewer than 16 whole blocks of at least 8 usable samples remain (a saturated
frame), the points hold one level only (no slope), one frame gives fewer than 4
points, one frame's μ² term is significant, one frame's weighted line does not
converge, the series' μ² test fails on every contiguous range of at least 3
points, or the fit is not physical (K, or one frame's σ_r², not positive). A
group of fewer than 2 frames, frames of another size, pattern or white level, or
a group's exposure without darks, is an `ArgumentError`; dark maps that do not
cover the frames a `DimensionMismatch`.
"""
function photon_transfer(m::RawMosaic; sensor_model::SensorModel = SensorModel(), block::Integer = 8,
                         significance::Real = _PTC_SIGNIFICANCE)
    _check_significance(significance)
    raw = parent(m)
    backend = KA.get_backend(raw)
    g = _green_sites(m, block, "photon_transfer")
    black = _green_black(g, _channel_black(sensor_model.black, m))
    n, sm, sd, sd2 = _block_moments(_GreenPair(raw, g.s1, g.s2), m, _offset_map(sensor_model.black, m, backend),
                                    defect_mask(sensor_model.defects, m; backend), black, g, block)
    full, keep = _whole_blocks(n)
    μ = [sm[k] / full for k in keep]
    v = [(sd2[k] - sd[k]^2 / full) / (full - 1) / 2 for k in keep]   # the difference's sample variance, halved
    p = _ptc_check(_ptc_points(μ, v, zeros(length(μ)), full - 1))
    np = length(p.xs)
    np >= 4 || throw(NoiseEstimateFailed(
        "the frame gives $(np) points ($(length(keep)) whole blocks, at least $(_PTC_BLOCKS_PER_POINT) per point): the μ² test " *
        "needs at least 4 (the line's slope and intercept, the μ² term and one degree of freedom), and one frame's K is never " *
        "returned untested: a larger frame, a smaller block, or photon_transfer(groups, darks) on frames at equal exposure"
    ))
    A = hcat(p.xs, ones(np))
    l = _ptc_line(A, p.ys, p.os, p.ms, p.rel)
    (; β, C) = l
    z = Distributions.quantile(Distributions.Normal(), 1 - significance / 2)
    c, t = _ptc_curvature(A, p, A * β .+ p.os)
    abs(t) > z && throw(NoiseEstimateFailed(
        "the photon-transfer points carry a μ² term (c = $(c) per code, t = $(round(t; sigdigits = 4)), |t| > $(round(z; sigdigits = 4)) " *
        "at significance $(significance)): a fixed pattern the single-frame green difference does not cancel (a bright flat field): " *
        "a single flat frame cannot certify its own K; use photon_transfer(groups, darks) on frames at equal exposure"
    ))
    slope, intercept = β
    (slope > 0 && intercept > 0) || throw(NoiseEstimateFailed(
        "the photon-transfer line is not physical: slope $(slope) (1/K), intercept $(intercept) (σ_r²)"
    ))
    l.converged || throw(NoiseEstimateFailed(
        "the weighted photon-transfer line did not converge: its weights reached no fixed point in $(l.iterations) iterations"
    ))
    σr = sqrt(intercept)
    return PoissonGaussian(1 / slope, ntuple(_ -> σr, Val(4)),
                           (gain = sqrt(C[1, 1]) / slope^2, σ = ntuple(_ -> sqrt(C[2, 2]) / (2σr), Val(4)));
                           source = SensorSource.estimated)
end

function photon_transfer(groups::AbstractDict{<:Quantity, <:AbstractVector{<:RawMosaic}}, darks::Darks;
                         defects::DefectComponent = NoDefects(), block::Integer = 8, normalize::Bool = true,
                         significance::Real = _PTC_SIGNIFICANCE)
    _check_significance(significance)
    exposures, frames_of, m = _ptc_groups(groups)
    g = _green_sites(m, block, "photon_transfer")
    backend = KA.get_backend(parent(m))
    # each group's Dark: the one whose band serves its exposure (the Darks' own rule), never the shared black
    times = [Cameras._positive_quantity(t, _SECOND, :exposure) for t in exposures]
    which = map(times) do t
        k = _serving_dark(darks, t)
        k === nothing && throw(ArgumentError(
            "photon_transfer: no Dark of the darks serves the exposure $(ustrip(_SECOND, t)) s (their bands: " *
            "$(join((map(b -> ustrip(_SECOND, b), d.exposure_band) for d in darks.darks), ", ")) s): every group is " *
            "linearised with, and its intercept is, the Dark of its own exposure, never the shared black or another band's"))
        k
    end
    sz = size(first(darks.darks).offset)
    _check_covers(sz, m, "photon_transfer: the darks' maps")
    # the maps, uploaded ONCE each into one stack on the frames' device: the used Darks' offsets, the groups' masks
    used = unique(which)
    slot = Dict(k => s for (s, k) in enumerate(used))
    offsets = on_device(Float32, backend, _host_stack(Float32, sz, (darks.darks[k].offset for k in used)))
    variances = Dict(k => Array(darks.darks[k].variance) for k in used)   # each used Dark's variance on the host once
    host_masks = [defect_mask(_frame_defects(defects, t), m; backend = KA.CPU()) for t in times]
    masked = findall(!isnothing, host_masks)
    mask_slot = Dict(i => s for (s, i) in enumerate(masked))
    masks = isempty(masked) ? nothing :
            on_device(Bool, backend, _host_stack(Bool, size(host_masks[first(masked)]), (host_masks[i] for i in masked)))
    levels = map(eachindex(exposures)) do i
        k = which[i]; d = darks.darks[k]
        mask = haskey(mask_slot, i) ? view(masks, :, :, mask_slot[i]) : nothing
        σ0, σ0_se = _dark_intercept(variances[k], d.frames, host_masks[i], m, g)
        (; black = _green_black(g, _channel_black(d, m)), dark = view(offsets, :, :, slot[k]), mask, σ0, σ0_se)
    end
    return _ptc_series(frames_of, levels, m, g, block, normalize, significance)
end

function photon_transfer(groups::AbstractDict{<:Quantity, <:AbstractVector{<:RawMosaic}}, darks::AbstractDict{<:Quantity, <:ReadNoise};
                         sensor_model::SensorModel = SensorModel(), block::Integer = 8, normalize::Bool = true,
                         significance::Real = _PTC_SIGNIFICANCE)
    _check_significance(significance)
    exposures, frames_of, m = _ptc_groups(groups)
    g = _green_sites(m, block, "photon_transfer")
    backend = KA.get_backend(parent(m))
    # ONE model for every group: its black, offset map and mask uploaded once
    black = _green_black(g, _channel_black(sensor_model.black, m))
    dark = _offset_map(sensor_model.black, m, backend)
    mask = defect_mask(sensor_model.defects, m; backend)
    # σ₀²(t) of each group: the greens' read noise² of the darks at that exposure, never another exposure's
    dark_at = Dict(ustrip(_SECOND, t) => rn for (t, rn) in darks)
    levels = map(exposures) do t
        rn = get(dark_at, ustrip(_SECOND, t), nothing)
        rn === nothing && throw(ArgumentError(
            "photon_transfer: no darks at the exposure $(ustrip(_SECOND, t)) s (darks at $(sort!(collect(keys(dark_at)))) s): " *
            "every group's intercept is its own exposure's dark noise, never interpolated or substituted; a Darks matches by band"))
        σ0, σ0_se = _read_noise_intercept(rn, g)
        (; black, dark, mask, σ0, σ0_se)
    end
    return _ptc_series(frames_of, levels, m, g, block, normalize, significance)
end

# The series' groups in increasing exposure, their frames, and the first frame (every frame of one sensor).
function _ptc_groups(groups)
    isempty(groups) && throw(ArgumentError("photon_transfer: no group of frames"))
    exposures = sort!(collect(keys(groups)); by = t -> ustrip(_SECOND, t))
    frames_of = [groups[t] for t in exposures]
    all(gr -> length(gr) >= 2, frames_of) ||
        throw(ArgumentError("photon_transfer: a group's temporal variance needs at least 2 frames, got $(map(length, frames_of))"))
    m = first(first(frames_of))
    for gr in frames_of, f in gr
        (size(f) == size(m) && f.pattern == m.pattern && f.white == m.white && f.window_origin == m.window_origin) ||
            throw(ArgumentError("photon_transfer: every frame is of one sensor (size, pattern, white level and window)"))
    end
    return exposures, frames_of, m
end

# A host stack (sz..., n) of eltype T of the matrices `maps` (a device map copied to the host first), uploaded by its
# caller once.
function _host_stack(::Type{T}, sz, maps) where {T}
    ms = collect(maps)
    out = Array{T}(undef, sz..., length(ms))
    for (k, a) in enumerate(ms)
        out[:, :, k] .= a isa Array ? a : Array(a)
    end
    return out
end

# σ₀² of a ReadNoise (the greens' mean σ²) and its standard error: the ReadNoise's by first order (2σ·se per green
# channel; the two greens independent channels, or one channel twice), 0 when it states none.
function _read_noise_intercept(rn::ReadNoise, g)
    σ0 = sum(rn.σ[c + 1]^2 for c in g.colors) / length(g.colors)
    rn.std_error === nothing && return σ0, 0.0
    d = map(c -> 2 * rn.σ[c + 1] * rn.std_error[c + 1], g.colors)
    return σ0, g.colors[1] == g.colors[2] ? (d[1] + d[2]) / 2 : sqrt(d[1]^2 + d[2]^2) / 2
end

# σ₀² of a Dark (its mean variance `v`, on the host, over the mosaic's green photosites that `mask`, on the host, leaves)
# and its standard error by first order from its frame count `frames` and photosite count, √(2/((n − 1)N))·σ₀²
# (master_dark's rule).
function _dark_intercept(v::Array, frames::Integer, mask, m::RawMosaic, g)
    o = m.window_origin
    H2, W2 = size(m, 1) ÷ 2, size(m, 2) ÷ 2
    s = 0.0; N = 0
    for site in (g.s1, g.s2), j in 1:W2, i in 1:H2
        r = 2i - 1 + site[1] + o[1]; c = 2j - 1 + site[2] + o[2]
        _masked(mask, r, c) && continue
        s += Float64(v[r, c]); N += 1
    end
    σ0 = s / N
    return σ0, sqrt(2 / ((frames - 1) * N)) * σ0
end

# THE series fit, of both forms of the darks: per group (in increasing exposure) its frames and its `levels` — its
# greens' blacks, its dark offset map and defect mask on the frames' device (`nothing` for none), its intercept σ₀²
# and that intercept's standard error. The frames are stacked into ONE buffer, reused group by group.
function _ptc_series(frames_of, levels, m::RawMosaic, g, block::Integer, normalize::Bool, significance)
    N = minimum(length, frames_of)                       # one χ² law: every group gives its first N frames
    stack = _frame_stack(first(frames_of), N)            # no splat of a data-length set
    stats = map(frames_of, levels) do gr, lv
        frames = gr[1:N]
        _stack_frames!(stack, frames)
        scale = normalize ? _flicker_scale(frames, g, lv.black) : ones(N)
        _block_moments(_FrameStack(stack, Adapt.adapt(KA.get_backend(stack), scale), g.s1, g.s2), m, lv.dark, lv.mask,
                       lv.black, g, block)
    end
    n = reduce(vcat, (vec(st[1]) for st in stats))
    sm = reduce(vcat, (vec(st[2]) for st in stats)); sq = reduce(vcat, (vec(st[4]) for st in stats))
    gid = reduce(vcat, (fill(i, length(st[1])) for (i, st) in enumerate(stats)))   # each block's exposure
    full, keep = _whole_blocks(n)
    # the points of each exposure apart (its blocks sorted by μ into points of equal count): a point never mixes two
    # exposures' intercepts; then every exposure's points, by μ
    parts = map(eachindex(stats)) do i
        ks = filter(k -> gid[k] == i, keep)
        length(ks) >= _PTC_BLOCKS_PER_POINT || return nothing
        pt = _ptc_points([sm[k] / full for k in ks], [sq[k] / (full * (N - 1)) for k in ks], fill(levels[i].σ0, length(ks)),
                         full * (N - 1))
        (; pt..., gs = fill(i, length(pt.xs)))
    end
    held = filter(!isnothing, parts)
    isempty(held) && throw(NoiseEstimateFailed("no exposure holds $(_PTC_BLOCKS_PER_POINT) whole blocks"))
    order = sortperm(reduce(vcat, (pt.xs for pt in held)))
    cat_sorted(f) = reduce(vcat, (getfield(pt, f) for pt in held))[order]
    p = _ptc_check((; xs = cat_sorted(:xs), ys = cat_sorted(:ys), ms = cat_sorted(:ms), os = cat_sorted(:os),
                    gs = cat_sorted(:gs), rel = held[1].rel))
    r = _ptc_range(p, significance)
    q = _ptc_subset(p, r.lo:r.hi)
    f = _series_slope(q)
    σ0f = Statistics.median(q.os)
    σr = sqrt(σ0f)
    σr_se = _median_std_error(q.os, [levels[i].σ0_se for i in q.gs]) / (2σr)
    noise = PoissonGaussian(f.gain, ntuple(_ -> σr, Val(4)), (gain = f.gain_se, σ = ntuple(_ -> σr_se, Val(4)));
                            source = SensorSource.measured)
    span = m.white - sum(levels[1].black) / 2
    meta = PhotonTransferMetadata((q.xs[1], q.xs[end]), (q.xs[1] / span, q.xs[end] / span), σ0f, f.c, f.t,
                                  r.tests, r.level, r.rollover_tests, r.rollover_level, f.R, length(p.xs))
    return FitResult(noise, f.retcode, f.rule, f.iterations, f.rms, length(q.xs), meta)
end

# The standard error of the median of `os` from each value's own `ses`: the middle value's, or the mean of the two
# middle values' (the median their mean).
function _median_std_error(os, ses)
    order = sortperm(os)
    n = length(order)
    return isodd(n) ? ses[order[(n + 1) ÷ 2]] : (ses[order[n ÷ 2]] + ses[order[n ÷ 2 + 1]]) / 2
end

# The series' slope over its range `q`: K = 1/slope with its standard error from the counts times the Birge ratio R when
# R > 1 (never shrunk), the μ² statistic over the range, R and the residuals' rms, and the weighted line's convergence
# (its return code, stop rule and reweightings).
function _series_slope(q; max_iter::Integer = _PTC_LINE_MAX_ITER)
    A = reshape(q.xs, :, 1)
    l = _ptc_line(A, q.ys, q.os, q.ms, q.rel; max_iter)
    (; β, C) = l
    β[1] > 0 || throw(NoiseEstimateFailed("the photon-transfer line is not physical: slope $(β[1]) (1/K) over the range"))
    pred = A * β .+ q.os
    c, t = _ptc_curvature(A, q, pred)
    R = _birge_ratio(pred, q, 1)
    retcode, rule = l.converged ? (ReturnCode.Success, IncrementSize) : (ReturnCode.MaxIters, IterationCap)
    return (; gain = 1 / β[1], gain_se = sqrt(C[1, 1]) / β[1]^2 * max(R, 1.0), c, t, R, rms = _birge_ratio(pred, q, 0),
            retcode, rule, iterations = l.iterations)
end

_check_significance(α) = 0 < α < 1 || throw(ArgumentError("photon_transfer: significance is in (0, 1), got $α"))

# The greens' cell sites and colour indices of a mosaic (its greens' LibRaw indices, computed once here), and the block
# count.
function _green_sites(m::RawMosaic, block::Integer, who)
    block >= 2 || throw(ArgumentError("$who: a block holds at least 2 × 2 cells, got $block"))
    cell = _cell_colors(m, who)
    gks = Tuple(k for k in 1:4 if cell[k] in (0x01, 0x03))
    H2, W2 = size(m, 1) ÷ 2, size(m, 2) ÷ 2
    H2 >= 1 && W2 >= 1 || throw(NoiseEstimateFailed("a $(join(size(m), '×')) mosaic holds no 2×2 cell"))
    return (s1 = _QUAD_SITES[gks[1]], s2 = _QUAD_SITES[gks[2]], colors = (cell[gks[1]], cell[gks[2]]),
            blocks = (cld(H2, block), cld(W2, block)))
end
# The two greens' blacks of a per-colour-index black.
_green_black(g, black) = (Float64(black[g.colors[1] + 1]), Float64(black[g.colors[2] + 1]))

# The blocks' count and sums under a cell rule, one launch on the frames' device (the offset map `dark` and the `mask`
# there already, or `nothing`), read back to the host.
function _block_moments(rule, m::RawMosaic, dark, mask, black, g, block::Integer)
    backend = KA.get_backend(parent(m))
    nb, mb = g.blocks
    n = KA.allocate(backend, Int, nb, mb); sm = KA.allocate(backend, Float64, nb, mb)
    sd = KA.allocate(backend, Float64, nb, mb); sd2 = KA.allocate(backend, Float64, nb, mb)
    OnBackend(backend)(_difference_blocks_kernel!)(n, sm, sd, sd2, rule, dark, mask, black, m.white, Int(block),
                                                   m.window_origin; ndrange = (nb, mb))
    KA.synchronize(backend)
    return Array(n), Array(sm), Array(sd), Array(sd2)
end

# Each frame's scale to its group's mean: the group's mean over the frame's, of the greens' black-subtracted codes (the
# greens' channel blacks `black`; a clipped photosite included: the scale is a ratio of whole-frame means).
function _flicker_scale(frames, g, black)
    means = map(frames) do f
        raw = parent(f)
        a = view(raw, (1 + g.s1[1]):2:size(raw, 1), (1 + g.s1[2]):2:size(raw, 2))
        b = view(raw, (1 + g.s2[1]):2:size(raw, 1), (1 + g.s2[2]):2:size(raw, 2))
        (sum(Float64, a) - black[1] * length(a) + sum(Float64, b) - black[2] * length(b)) / (length(a) + length(b))
    end
    all(>(0), means) || throw(NoiseEstimateFailed("a frame of a group holds no light above the black level: no scale to its group"))
    return Statistics.mean(means) ./ means
end


# The fit's points (host; a few thousand blocks): the whole blocks' means `μ`, variances `v` (each `v̂/v ~ χ²_dof/dof`)
# and intercepts `o` (zeros when fitted; one exposure's, the series'), sorted by μ into
# `min(32, blocks ÷ _PTC_BLOCKS_PER_POINT)` points of equal count — each point's median μ, median-unbiased variance,
# block count and median intercept — with `rel`, the counts' relative variance of a point's median.
function _ptc_points(μ, v, o, dof::Integer)
    groups = min(32, length(μ) ÷ _PTC_BLOCKS_PER_POINT)
    order = sortperm(μ)
    G = Distributions.Gamma(dof / 2, 2 / dof)                         # the law of v̂/v: χ²_dof/dof
    q = Statistics.median(G)
    rel = 1 / (4 * Distributions.pdf(G, q)^2 * q^2)                  # m·Var(median v̂ / q) / v², the counts' own
    xs = zeros(groups); ys = zeros(groups); ms = zeros(groups); os = zeros(groups)
    for g in 1:groups
        idx = order[(div((g - 1) * length(order), groups) + 1):div(g * length(order), groups)]
        xs[g] = Statistics.median(view(μ, idx)); ys[g] = Statistics.median(view(v, idx)) / q; ms[g] = length(idx)
        os[g] = Statistics.median(view(o, idx))
    end
    return (; xs, ys, ms, os, rel)
end
# The points every fit needs: more than one level, and a variance at each.
function _ptc_check(p)
    maximum(p.xs) > minimum(p.xs) || throw(NoiseEstimateFailed("the usable blocks hold one level only: no slope"))
    all(>(0), p.ys) || throw(NoiseEstimateFailed("a level of the usable blocks has no variance: a noise-free frame"))
    return p
end
# The points `r` of `p` (every per-point column; `rel` kept).
_ptc_subset(p, r) = map(x -> x isa AbstractVector ? x[r] : x, p)

# THE weighted least squares of `ys − os` on the columns of `A` (the slope and, for one frame, the intercept), each
# point weighted by 1 / the variance of its median at the model's prediction, iterated to a fixed point from a start
# weighted at each point's own value: the one line fit of both inputs.
# Returns `(; β, C, iterations, converged)`: `converged` when the weights reached their fixed point within `max_iter`
# reweightings (false at the cap, or when a coefficient left the positive orthant, where no weight is defined).
function _ptc_line(A, ys, os, ms, rel; max_iter::Integer = _PTC_LINE_MAX_ITER)
    fit(w) = (C = inv(Symmetric(A' * Diagonal(w) * A)); (C * (A' * (w .* (ys .- os))), C))
    β, C = fit(ms ./ (rel .* ys .^ 2))                   # the start: each point weighted at its own variance
    for iter in 1:max_iter
        all(>(0), β) || return (; β, C, iterations = iter - 1, converged = false)
        β′, C = fit(ms ./ (rel .* (A * β .+ os) .^ 2))    # 1 / Var of each point at the model's prediction
        done = isapprox(β′, β; rtol = sqrt(eps()))
        β = β′
        done && return (; β, C, iterations = iter, converged = true)
    end
    return (; β, C, iterations = Int(max_iter), converged = false)
end

# THE μ² statistic: the points fitted again with a μ² column beside `A`, each weighted at `prediction` (at its own value
# where the line predicts none): `(c, c / se(c))`.
function _ptc_curvature(A, p, prediction)
    vw = ifelse.(prediction .> 0, prediction, p.ys)
    w = p.ms ./ (p.rel .* vw .^ 2)
    Q = hcat(A, p.xs .^ 2)
    Cq = inv(Symmetric(Q' * Diagonal(w) * Q))
    c = (Cq * (Q' * (w .* (p.ys .- p.os))))[end]
    return c, c / sqrt(Cq[end, end])
end

# THE Birge ratio of the points about the fitted line, √(χ²/dof), each residual in its median's standard deviation from
# the counts at the line's prediction, dof the points less the line's `nparams` parameters; with `nparams = 0` the
# residuals' root mean square (≈ 1 when the counts' law holds).
_birge_ratio(prediction, p, nparams) =
    sqrt(sum(((p.ys .- prediction) ./ sqrt.(p.rel .* prediction .^ 2 ./ p.ms)) .^ 2) / (length(p.ys) - nparams))

# THE range of the series' fit, derived from the data (owner rulings 1–2), at least 3 points (the slope and a μ² term
# with one degree of freedom left): the UPPER end below the rollover — the top point dropped while it lies significantly
# BELOW the line fitted to the points under it (one-sided, its median's variance plus the line's at it) — then the
# LOWER end, the lowest point dropped until the μ² test on the line succeeds. Each search runs its tests at
# `significance / m`, `m` the tests it can perform (Bonferroni: its family-wise false-alarm rate is `significance`).
function _ptc_range(p, significance)
    np = length(p.xs)
    np >= 3 || throw(NoiseEstimateFailed("the photon-transfer series gives $(np) points: a range needs at least 3"))
    lo, hi = 1, np
    m_up = np - 3                                        # the candidate drops of the top point
    level_up = m_up > 0 ? significance / m_up : 0.0
    while hi - lo + 1 > 3
        q = _ptc_subset(p, lo:(hi - 1))
        β, C = _ptc_line(reshape(q.xs, :, 1), q.ys, q.os, q.ms, q.rel)
        pred = p.xs[hi] * β[1] + p.os[hi]
        sd = sqrt(p.rel * pred^2 / p.ms[hi] + p.xs[hi]^2 * C[1, 1])
        (p.ys[hi] - pred) / sd < -Distributions.quantile(Distributions.Normal(), 1 - level_up) || break
        hi -= 1
    end
    m_lo = hi - 2                                        # the candidate lower ends: 1 … hi − 2
    level_lo = significance / m_lo
    z = Distributions.quantile(Distributions.Normal(), 1 - level_lo / 2)
    while true
        q = _ptc_subset(p, lo:hi)
        A = reshape(q.xs, :, 1)
        β, _ = _ptc_line(A, q.ys, q.os, q.ms, q.rel)
        _, t = _ptc_curvature(A, q, A * β .+ q.os)
        abs(t) <= z && break
        lo += 1
        hi - lo + 1 >= 3 || throw(NoiseEstimateFailed(
            "no contiguous range of at least 3 levels is consistent with one line v = σ₀²(t) + μ/K: every lower end fails the μ² test " *
            "at $(level_lo) per test ($(m_lo) tests, family-wise $(significance)): the groups' temporal variance is not linear in the " *
            "mean: frames of a group at unequal light (normalize = true scales them to the group's mean) or a nonlinear response"))
    end
    return (; lo, hi, tests = m_lo, level = level_lo, rollover_tests = m_up, rollover_level = level_up)
end

# Per photosite of the stacked frames (H × W × n), the temporal mean and sample variance, one work item each.
@kernel function _dark_moments_kernel!(offset, variance, @Const(stack))
    i, j = @index(Global, NTuple)
    _, μ, q = _temporal_moments(stack, i, j, 0.0, nothing, nothing)
    @inbounds offset[i, j] = Float32(μ)
    @inbounds variance[i, j] = Float32(q / (size(stack, 3) - 1))
end

"""
    master_dark(frames::AbstractVector{<:RawMosaic}; exposure_band, false_defects = 0.01)
        -> (; dark::Dark, read_noise::ReadNoise, defects::BitMatrix)

THE dark-frame estimator: the lens-capped frames of one body at one ISO and one
exposure band (whole mosaics of one size, at least 2) into the components of a
characterisation's entry, all `SensorSource.measured`:

- `dark` — the [`Dark`](@ref): per photosite the temporal mean (the offset:
  black, DSNU and dark current) and sample variance, one KernelAbstractions work
  item each, on the frames' device; its channel black (`dark.black`) a
  [`ChannelBlack`](@ref), per colour index (LibRaw's `0`–`3`) the mean offset of
  its photosites (defects left out), with its standard error from their spread
  and count;
- `read_noise` — a [`ReadNoise`](@ref): per colour index, the root mean temporal
  variance (the read noise with the rounding, and at a long exposure the dark
  current's shot noise), its standard error from the `χ²` spread of the frame
  count and the photosite count (`√(2/(n − 1)/N)·σ²` for the mean variance, by
  first order);
- `defects` — a `BitMatrix`, the photosites whose offset leaves their
  channel's median by more than `z` robust standard deviations (the median
  absolute deviation over the normal's, [`mad_scale`](@ref)'s rule for
  codimension 1, about the channel's median), `z` the normal quantile at which
  `false_defects` defect-free photosites are expected to be flagged over the
  whole sensor of `N` photosites (`1 − false_defects / (2N)`, two-sided, for
  offsets that are Gaussian about their channel's black). The masks of several
  exposures make one [`DefectList`](@ref) by [`defect_onsets`](@ref).

A colour index the pattern lacks (a pattern with one green index) has black
and read noise `0` (no photosite reads it).

`exposure_band` is `(shortest, longest)`, time quantities, the exposures the
dark serves. A frame of another size, or fewer than 2, is an
`ArgumentError`; a channel without a usable photosite throws
[`NoiseEstimateFailed`](@ref).
"""
function master_dark(frames::AbstractVector{<:RawMosaic}; exposure_band, false_defects::Real = 0.01)
    length(frames) >= 2 || throw(ArgumentError("master_dark: a temporal variance needs at least 2 frames, got $(length(frames))"))
    sz = size(first(frames))
    all(f -> size(f) == sz && f.window_origin == (0, 0), frames) ||
        throw(ArgumentError("master_dark: the frames are whole mosaics of one size ($(sz))"))
    0 < false_defects < 1 || throw(ArgumentError("master_dark: false_defects is in (0, 1), got $false_defects"))
    raw1 = parent(first(frames))
    backend = KA.get_backend(raw1)
    stack = _stack_frames!(_frame_stack(frames, length(frames)), frames)
    offset = similar(raw1, Float32, sz...); variance = similar(raw1, Float32, sz...)
    OnBackend(backend)(_dark_moments_kernel!)(offset, variance, stack; ndrange = sz)
    KA.synchronize(backend)
    # per colour index, on the host: a few numbers per channel over the maps
    col = collect(photosite_colors(first(frames)))
    off = Array(offset); var = Array(variance)
    N = length(off)
    z = Distributions.quantile(Distributions.Normal(), 1 - false_defects / (2N))
    defects = falses(sz)
    nf = length(frames)
    black = zeros(4); black_se = zeros(4); σr = zeros(4); σr_se = zeros(4)
    for c in 0x00:0x03
        sel = findall(==(c), col)
        isempty(sel) && continue
        o = Float64.(off[sel])
        med = Statistics.median(o)
        s = _mad_scale_norms(; norms = abs.(o .- med), codimension = 1)   # mad_scale's rule, about the channel's median
        bad = s > 0 ? abs.(o .- med) .> z * s : falses(length(o))
        defects[sel[bad]] .= true
        good = sel[.!bad]
        isempty(good) && throw(NoiseEstimateFailed("master_dark: every photosite of colour index $(c) is a defect"))
        og = Float64.(off[good]); vg = Float64.(var[good])
        black[c + 1] = Statistics.mean(og)
        black_se[c + 1] = length(og) > 1 ? Statistics.std(og) / sqrt(length(og)) : 0.0
        σ² = Statistics.mean(vg)
        σr[c + 1] = sqrt(σ²)
        σr_se[c + 1] = sqrt(2 / (nf - 1) / length(vg)) * σ² / (2 * max(σr[c + 1], eps()))
    end
    source = SensorSource.measured
    dark = Dark(ChannelBlack(black, black_se; source), offset, variance, exposure_band, nf; source)
    return (; dark, read_noise = ReadNoise(σr, σr_se; source), defects)
end

"""
    defect_onsets(masks; source::SensorSource.T = SensorSource.measured) -> DefectList

ONE [`DefectList`](@ref) from per-exposure defect masks: `masks` is a vector of
`exposure => mask` (a time quantity and a `Bool` matrix over the visible raw,
[`master_dark`](@ref)'s `defects` with the shortest exposure of its band, say),
all masks of one size. Each photosite flagged in any mask is listed once, its
onset the shortest exposure whose mask flags it (the resolver masks it at
every exposure from there on: a site flagged at a short exposure and missed at
a longer one is a defect at both). The sites are in column-major order of the
visible raw. An empty vector or masks of different sizes are an `ArgumentError`.
"""
function defect_onsets(masks::AbstractVector{<:Pair{<:Quantity, <:AbstractMatrix{Bool}}};
                       source::SensorSource.T = SensorSource.measured)
    isempty(masks) && throw(ArgumentError("defect_onsets: no masks"))
    sz = size(last(first(masks)))
    all(p -> size(last(p)) == sz, masks) || throw(ArgumentError("defect_onsets: the masks are of one size ($(sz))"))
    onset = fill(Inf, sz)                                           # seconds; Inf where no mask flags the site
    for (t, mask) in masks
        ts = ustrip(_SECOND, Cameras._positive_quantity(t, _SECOND, :exposure))
        host = Array(mask)
        @inbounds for i in eachindex(onset, host)
            host[i] && (onset[i] = min(onset[i], ts))
        end
    end
    sites = findall(isfinite, onset)
    return DefectList(getindex.(sites, 2), getindex.(sites, 1), onset[sites] .* _SECOND; size = sz, source)
end

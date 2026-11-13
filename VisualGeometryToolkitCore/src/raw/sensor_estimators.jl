# =============================================================================
# The sensor estimators — a characterisation's components from frames, offline (D1)
# =============================================================================
#
#   photon_transfer   the ONE estimator of the gain K and σ_r: from one frame (its green pairs, an estimated
#                     PoissonGaussian) or from groups of frames at equal exposure (the per-photosite temporal
#                     variance, a measured PoissonGaussian); one block kernel, one fit
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
    nf = size(r.stack, 3)
    b = _black_at(dark, black, row + origin[1], col + origin[2])
    s = 0.0
    for k in 1:nf
        @inbounds x = r.stack[row, col, k]
        _at_white(x, white) && return (0, 0.0, 0.0)
        @inbounds s += (Float64(x) - b) * r.scale[k]
    end
    μ = s / nf
    q = 0.0
    for k in 1:nf
        @inbounds q += ((Float64(r.stack[row, col, k]) - b) * r.scale[k] - μ)^2
    end
    return (1, μ, q)
end

# The blocks every estimate reads: the whole blocks (the most usable samples any block has, `full`), at least 16 of at
# least 8 samples.
function _whole_blocks(n)
    full = maximum(n)
    keep = findall(==(full), vec(n))                     # whole blocks only: one χ² law, one median correction
    full >= 8 && length(keep) >= 16 || throw(NoiseEstimateFailed(
        "too few usable blocks: $(length(keep)) unclipped blocks of $(full) cells (at least 16 of at least 8 cells)"
    ))
    return full, keep
end

"""
    photon_transfer(mosaic::RawMosaic; sensor_model::SensorModel = SensorModel(), block = 8, significance = 0.001)
        -> PoissonGaussian
    photon_transfer(groups::AbstractVector{<:AbstractVector{<:RawMosaic}}; sensor_model::SensorModel = SensorModel(),
                    block = 8, normalize = true, significance = 0.001) -> PoissonGaussian

THE estimator of a sensor's Poisson–Gaussian noise for a characterisation's
entry, built offline (owner ruling D1: the resolver [`sensor_model`](@ref)
never estimates): the conversion gain K (electrons per code) and the read noise
σ_r (codes, the rounding included: a Gaussian read noise σ plus the code's
rounding is `√(σ² + 1/12)`), by a photon-transfer fit of variance against mean,
`v = μ/K + σ_r²`. Two inputs, one fit:

- **one frame** (`mosaic`), in the line of Foi et al. (2008), an ESTIMATED
  [`PoissonGaussian`](@ref) (`SensorSource.estimated`): each 2×2 cell's two
  greens see the same light, so their difference `d` carries no scene where the
  scene is flat across the cell, and `Var d = 2(μ/K + σ_r²)` about their
  black-subtracted mean μ. Per block, the cells' mean and the difference's sample
  variance give a point `(μ̂, v̂ = s²/2)`, `v̂/v` following `χ²_{n−1}/(n − 1)` over
  a whole block of `n` cells. A fixed pattern between the two greens (their gain
  or response mismatch) does NOT cancel in `d`: it adds a term in μ², negligible
  in a dim natural scene, dominant on a bright flat; the fit refuses it (below).
- **groups of frames** (`groups`, the photon-transfer series: one group per
  exposure, at least 2 frames each, of one body at one ISO), a MEASURED
  `PoissonGaussian` (`SensorSource.measured`, a dedicated capture): per green
  photosite its TEMPORAL variance over the group's frames, which cancels the
  scene and every fixed pattern, `v = μ/K + σ_r²` about its mean μ. Pairs are the
  case of 2 frames. With `normalize` (the default) each frame is first scaled to
  its group's mean (its greens' black-subtracted mean over the group's: a source
  that flickers between frames adds no μ² term). Per block, the photosites' means
  and their pooled variance give a point, `v̂/v` following
  `χ²_{p(N−1)}/(p(N − 1))` over a whole block of `p` photosites and `N` frames;
  every group gives its first `N` frames, `N` the smallest group's, so every
  block follows that one law.

Per block of `block × block` cells (one KernelAbstractions work item each, on
the frames' device), a sample with a photosite at the white level, or under the
model's defect mask, is left out (CLIPPED, never modelled), and only whole blocks
(the most usable samples any block has) enter the fit, every group's alike. The
blocks are sorted by μ̂ into groups of equal count (`min(32, blocks ÷ 8)`); each
group's median μ̂ and median `v̂ / q` (`q` that law's median, so the ratio is
median-unbiased for v) are a point of the line. The line is fitted by weighted
least squares, each point weighted by the inverse of its median's variance from
the counts, `v²/(4 m f(q)² q²)` over a group of `m` blocks (`f` the law's
density: the sample median's asymptotic variance), with v the line's own
prediction (iterated to a fixed point from a start weighted at each point's own
value). The median is robust to the blocks a scene edge raises. The standard
errors are the fitted line's covariance from those counts; K's and σ_r's by
first order. The read noise is the greens': the result states it for every
colour index (`σ[c + 1]`, the same four numbers); a characterisation that holds
σ_r per colour takes it from [`master_dark`](@ref)'s `ReadNoise`.

THE REFUSAL: with at least 4 points, the same points are fitted again with a μ²
term, `v = a + μ/K + cμ²`, at the same weights, and the estimate is refused when
`c` is significant, `|ĉ / se(ĉ)|` above the two-sided normal quantile at
`significance` (3.29 at the default 0.001): a fixed pattern the input does not
cancel, or a nonlinear response. It never returns a K biased by it.

`sensor_model` gives what is subtracted and left out, as it does for
[`LinearRaw`](@ref): its black slot's per-colour black (a [`Dark`](@ref)'s
offset per photosite in its place) and its defect slot's photosites; its noise
slot is not read. Throws [`NoiseEstimateFailed`](@ref) with the reason when
fewer than 16 whole blocks of at least 8 usable samples remain (a saturated
frame), the points hold one level only (no slope), the μ² term is significant,
or the fit is not physical (K or σ_r² not positive). A group of fewer than 2
frames, or frames of another size, pattern or white level, is an `ArgumentError`.
"""
function photon_transfer(m::RawMosaic; sensor_model::SensorModel = SensorModel(), block::Integer = 8,
                         significance::Real = 0.001)
    raw = parent(m)
    g = _green_sites(m, sensor_model, block, "photon_transfer")
    n, sm, sd, sd2 = _block_moments(_GreenPair(raw, g.s1, g.s2), m, sensor_model, g, block)
    full, keep = _whole_blocks(n)
    μ = [sm[k] / full for k in keep]
    v = [(sd2[k] - sd[k]^2 / full) / (full - 1) / 2 for k in keep]   # the difference's sample variance, halved
    e = _photon_transfer_fit(μ, v, full - 1, significance,
        "a fixed pattern the single-frame green difference does not cancel (a bright flat field): " *
        "use photon_transfer(groups) on frames at equal exposure")
    return PoissonGaussian(e.gain, ntuple(_ -> e.σ, Val(4)), (gain = e.gain_std_error, σ = ntuple(_ -> e.σ_std_error, Val(4)));
                           source = SensorSource.estimated)
end

function photon_transfer(groups::AbstractVector{<:AbstractVector{<:RawMosaic}}; sensor_model::SensorModel = SensorModel(),
                         block::Integer = 8, normalize::Bool = true, significance::Real = 0.001)
    isempty(groups) && throw(ArgumentError("photon_transfer: no group of frames"))
    all(gr -> length(gr) >= 2, groups) ||
        throw(ArgumentError("photon_transfer: a group's temporal variance needs at least 2 frames, got $(map(length, groups))"))
    m = first(first(groups))
    for gr in groups, f in gr
        (size(f) == size(m) && f.pattern == m.pattern && f.white == m.white && f.window_origin == m.window_origin) ||
            throw(ArgumentError("photon_transfer: every frame is of one sensor (size, pattern, white level and window)"))
    end
    N = minimum(length, groups)                          # one χ² law: every group gives its first N frames
    g = _green_sites(m, sensor_model, block, "photon_transfer")
    stats = map(groups) do gr
        frames = gr[1:N]
        stack = cat((parent(f) for f in frames)...; dims = 3)
        scale = normalize ? _flicker_scale(frames, g) : ones(N)
        _block_moments(_FrameStack(stack, Adapt.adapt(KA.get_backend(stack), scale), g.s1, g.s2), m, sensor_model, g, block)
    end
    n = reduce(vcat, (vec(st[1]) for st in stats))
    sm = reduce(vcat, (vec(st[2]) for st in stats)); sq = reduce(vcat, (vec(st[4]) for st in stats))
    full, keep = _whole_blocks(n)
    μ = [sm[k] / full for k in keep]
    v = [sq[k] / (full * (N - 1)) for k in keep]         # the photosites' pooled temporal variance
    e = _photon_transfer_fit(μ, v, full * (N - 1), significance,
        "the groups' temporal variance is not linear in the mean: frames of a group at unequal light " *
        "(normalize = true scales them to the group's mean) or a nonlinear response")
    return PoissonGaussian(e.gain, ntuple(_ -> e.σ, Val(4)), (gain = e.gain_std_error, σ = ntuple(_ -> e.σ_std_error, Val(4)));
                           source = SensorSource.measured)
end

# The greens' cell sites and their blacks (the model's black slot, per colour index) of a mosaic, and the block count.
function _green_sites(m::RawMosaic, sensor_model::SensorModel, block::Integer, who)
    block >= 2 || throw(ArgumentError("$who: a block holds at least 2 × 2 cells, got $block"))
    cell = _cell_colors(m, who)
    gks = Tuple(k for k in 1:4 if cell[k] in (0x01, 0x03))
    black = _channel_black(sensor_model.black, m)
    H2, W2 = size(m, 1) ÷ 2, size(m, 2) ÷ 2
    H2 >= 1 && W2 >= 1 || throw(NoiseEstimateFailed("a $(join(size(m), '×')) mosaic holds no 2×2 cell"))
    return (s1 = _QUAD_SITES[gks[1]], s2 = _QUAD_SITES[gks[2]], black = (black[cell[gks[1]] + 1], black[cell[gks[2]] + 1]),
            blocks = (cld(H2, block), cld(W2, block)))
end

# The blocks' count and sums under a cell rule, one launch on the frames' device, read back to the host.
function _block_moments(rule, m::RawMosaic, sensor_model::SensorModel, g, block::Integer)
    backend = KA.get_backend(parent(m))
    nb, mb = g.blocks
    dark_d = _offset_map(sensor_model.black, m, backend)
    mask_d = _defect_mask(sensor_model.defects, m, backend)
    n = KA.allocate(backend, Int, nb, mb); sm = KA.allocate(backend, Float64, nb, mb)
    sd = KA.allocate(backend, Float64, nb, mb); sd2 = KA.allocate(backend, Float64, nb, mb)
    OnBackend(backend)(_difference_blocks_kernel!)(n, sm, sd, sd2, rule, dark_d, mask_d, g.black, m.white, Int(block),
                                                   m.window_origin; ndrange = (nb, mb))
    KA.synchronize(backend)
    return Array(n), Array(sm), Array(sd), Array(sd2)
end

# Each frame's scale to its group's mean: the group's mean over the frame's, of the greens' black-subtracted codes (the
# channel blacks; a clipped photosite included: the scale is a ratio of whole-frame means).
function _flicker_scale(frames, g)
    means = map(frames) do f
        raw = parent(f)
        a = view(raw, (1 + g.s1[1]):2:size(raw, 1), (1 + g.s1[2]):2:size(raw, 2))
        b = view(raw, (1 + g.s2[1]):2:size(raw, 1), (1 + g.s2[2]):2:size(raw, 2))
        (sum(Float64, a) - g.black[1] * length(a) + sum(Float64, b) - g.black[2] * length(b)) / (length(a) + length(b))
    end
    all(>(0), means) || throw(NoiseEstimateFailed("a frame of a group holds no light above the black level: no scale to its group"))
    return Statistics.mean(means) ./ means
end

# The fit over the blocks' points (host; a few thousand numbers): `μ`, `v` the whole blocks' means and variances,
# each `v̂/v ~ χ²_dof/dof`; the refusal of a significant μ² term at `significance`, its message ending with `remedy`.
function _photon_transfer_fit(μ, v, dof::Integer, significance::Real, remedy::AbstractString)
    0 < significance < 1 || throw(ArgumentError("photon_transfer: significance is in (0, 1), got $significance"))
    groups = min(32, length(μ) ÷ 8)
    order = sortperm(μ)
    G = Distributions.Gamma(dof / 2, 2 / dof)                         # the law of v̂/v: χ²_dof/dof
    q = Statistics.median(G)
    rel = 1 / (4 * Distributions.pdf(G, q)^2 * q^2)                  # m·Var(median v̂ / q) / v², the counts' own
    xs = zeros(groups); ys = zeros(groups); ms = zeros(groups)
    for g in 1:groups
        idx = order[(div((g - 1) * length(order), groups) + 1):div(g * length(order), groups)]
        xs[g] = Statistics.median(view(μ, idx)); ys[g] = Statistics.median(view(v, idx)) / q; ms[g] = length(idx)
    end
    maximum(xs) > minimum(xs) || throw(NoiseEstimateFailed("the usable blocks hold one level only: no slope"))
    all(>(0), ys) || throw(NoiseEstimateFailed("a level of the usable blocks has no variance: a noise-free frame"))
    A = hcat(xs, ones(groups))
    fit(w) = (C = inv(Symmetric(A' * Diagonal(w) * A)); (C * (A' * (w .* ys)), C))
    β, C = fit(ms ./ (rel .* ys .^ 2))                   # the start: each point weighted at its own variance
    for _ in 1:100
        (β[1] > 0 && β[2] > 0) || break
        β′, C = fit(ms ./ (rel .* (A * β) .^ 2))          # 1 / Var of each point at the line's prediction
        done = isapprox(β′, β; rtol = sqrt(eps()))
        β = β′
        done && break
    end
    _refuse_curvature(xs, ys, ms, rel, A * β, significance, remedy)
    slope, intercept = β
    (slope > 0 && intercept > 0) || throw(NoiseEstimateFailed(
        "the photon-transfer line is not physical: slope $(slope) (1/K), intercept $(intercept) (σ_r²)"
    ))
    σr = sqrt(intercept)
    return (; gain = 1 / slope, σ = σr, gain_std_error = sqrt(C[1, 1]) / slope^2, σ_std_error = sqrt(C[2, 2]) / (2σr))
end

# THE refusal of a μ² term: the points fitted again as v = a + μ/K + cμ², each weighted at the line's prediction (at
# its own value where the line predicts none), and `c` significant at `significance` (two-sided, by its own standard
# error from the counts). Needs 4 points (one degree of freedom beyond the quadratic).
function _refuse_curvature(xs, ys, ms, rel, prediction, significance, remedy)
    length(xs) >= 4 || return nothing
    vw = ifelse.(prediction .> 0, prediction, ys)
    w = ms ./ (rel .* vw .^ 2)
    Q = hcat(xs, ones(length(xs)), xs .^ 2)
    Cq = inv(Symmetric(Q' * Diagonal(w) * Q))
    c = (Cq * (Q' * (w .* ys)))[3]
    t = c / sqrt(Cq[3, 3])
    z = Distributions.quantile(Distributions.Normal(), 1 - significance / 2)
    abs(t) > z && throw(NoiseEstimateFailed(
        "the photon-transfer points carry a μ² term (c = $(c) per code, t = $(round(t; sigdigits = 4)), |t| > $(round(z; sigdigits = 4)) " *
        "at significance $(significance)): $(remedy)"
    ))
    return nothing
end

# Per photosite of the stacked frames (H × W × n), the temporal mean and sample variance, one work item each.
@kernel function _dark_moments_kernel!(offset, variance, @Const(stack))
    i, j = @index(Global, NTuple)
    nf = size(stack, 3)
    s = 0.0
    for k in 1:nf
        @inbounds s += Float64(stack[i, j, k])
    end
    μ = s / nf
    q = 0.0
    for k in 1:nf
        @inbounds q += (Float64(stack[i, j, k]) - μ)^2
    end
    @inbounds offset[i, j] = Float32(μ)
    @inbounds variance[i, j] = Float32(q / (nf - 1))
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
    stack = similar(raw1, UInt16, sz..., length(frames))
    for (k, f) in enumerate(frames)
        copyto!(view(stack, :, :, k), parent(f))
    end
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

# =============================================================================
# The sensor estimators — a characterisation's components from frames, offline (D1)
# =============================================================================
#
#   photon_transfer   the ONE estimator of the gain K and σ_r from a frame: an
#                     estimated PoissonGaussian
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

# Per block of `B × B` 2×2 cells, the green pairs' statistics, one work item per block: the count of usable cells
# (neither green at the white level, or masked), Σm, Σd and Σd² of m = the pair's black-subtracted mean and d = its
# difference (both in codes). The photosites are read from the held array by the channel view's index rule (the two
# greens' cell sites `s1`, `s2`, `_QUAD_SITES` entries), as LinearRaw's cell kernel reads them; the blacks are the
# greens' (a dark map's offsets per photosite when given, `_black_at`).
@kernel function _green_pair_blocks_kernel!(n, sm, sd, sd2, @Const(raw), dark, mask, s1, s2, black, white::Float64,
                                             B::Int, origin)
    bi, bj = @index(Global, NTuple)
    H2, W2 = size(raw, 1) ÷ 2, size(raw, 2) ÷ 2
    c = 0; t1 = 0.0; t2 = 0.0; t3 = 0.0
    for jj in 1:B, ii in 1:B
        i = (bi - 1) * B + ii; j = (bj - 1) * B + jj
        (i <= H2 && j <= W2) || continue
        r1 = 2i - 1 + s1[1]; c1 = 2j - 1 + s1[2]
        r2 = 2i - 1 + s2[1]; c2 = 2j - 1 + s2[2]
        @inbounds x1 = raw[r1, c1]; @inbounds x2 = raw[r2, c2]
        (_at_white(x1, white) || _at_white(x2, white)) && continue
        (_masked(mask, r1 + origin[1], c1 + origin[2]) || _masked(mask, r2 + origin[1], c2 + origin[2])) && continue
        y1 = Float64(x1) - _black_at(dark, black[1], r1 + origin[1], c1 + origin[2])
        y2 = Float64(x2) - _black_at(dark, black[2], r2 + origin[1], c2 + origin[2])
        c += 1; t1 += (y1 + y2) / 2; t2 += y1 - y2; t3 += (y1 - y2)^2
    end
    @inbounds n[bi, bj] = c; @inbounds sm[bi, bj] = t1; @inbounds sd[bi, bj] = t2; @inbounds sd2[bi, bj] = t3
end

"""
    photon_transfer(mosaic::RawMosaic; sensor_model::SensorModel = SensorModel(), block = 8) -> PoissonGaussian

THE estimator of a frame's Poisson–Gaussian noise, an estimated
[`PoissonGaussian`](@ref) (`SensorSource.estimated`, with its standard errors)
for a characterisation's entry, built offline (owner ruling D1: the resolver
[`sensor_model`](@ref) never estimates): the conversion gain K (electrons per code) and the read
noise σ_r (codes, the rounding included: a Gaussian read noise σ plus the code's
rounding is `√(σ² + 1/12)`), from the frame itself, by a photon-transfer fit of
local variance against mean, in the line of Foi et al. (2008).

Each 2×2 cell's two greens see the same light, so their difference `d` carries
no scene where the scene is flat across the cell, and `Var d = 2(μ/K + σ_r²)`
about their black-subtracted mean μ. Per block of `block × block` cells (one
KernelAbstractions work item each, on the mosaic's device) the cells' mean and
the difference's sample variance give a point `(μ̂, v̂ = s²/2)`; a cell with a
green at the white level, or under `mask`, is left out (CLIPPED, never
modelled), and only whole blocks (the most usable cells any block has, `n`)
enter the fit, so every `v̂/v` follows one law, `χ²_{n−1}/(n − 1)` for a
Gaussian difference. The blocks are sorted by μ̂ into groups of equal count
(`min(32, blocks ÷ 8)` groups); each group's median μ̂ and median `v̂ / q`
(`q` that law's median, so the ratio is median-unbiased for v) are a point of
the line `v = μ/K + σ_r²`. The line is fitted by weighted least squares, each
point weighted by the inverse of its median's variance from the counts,
`v²/(4 m f(q)² q²)` over a group of `m` blocks (`f` the law's density: the
sample median's asymptotic variance), with v the line's own prediction
(iterated to a fixed point from a start weighted at each point's own value, so
the final weights do not read their own point's noise). The median is robust to the blocks a scene edge
raises. The standard errors are the fitted line's covariance from those counts;
K's and σ_r's by first order. The read noise is the greens': the result states
it for every colour index (`σ[c + 1]`, the same four numbers).

`sensor_model` gives what is subtracted and left out, as it does for
[`LinearRaw`](@ref): its black slot's per-colour black (a [`Dark`](@ref)'s
offset per photosite in its place) and its defect slot's photosites (left out
like clipped cells); its noise slot is not read. Throws
[`NoiseEstimateFailed`](@ref) with the reason when fewer than 16 whole blocks
of at least 8 usable cells remain (a saturated frame), the groups hold one
level only (no slope), or the fit is not physical (K or σ_r² not positive).
"""
function photon_transfer(m::RawMosaic; sensor_model::SensorModel = SensorModel(), block::Integer = 8)
    block >= 2 || throw(ArgumentError("photon_transfer: a block holds at least 2 × 2 cells, got $block"))
    cell = _cell_colors(m, "photon_transfer")
    gks = Tuple(k for k in 1:4 if cell[k] in (0x01, 0x03))
    black = _channel_black(sensor_model.black, m)
    gblack = (black[cell[gks[1]] + 1], black[cell[gks[2]] + 1])
    raw = parent(m)
    H2, W2 = size(raw, 1) ÷ 2, size(raw, 2) ÷ 2
    H2 >= 1 && W2 >= 1 || throw(NoiseEstimateFailed("a $(join(size(m), '×')) mosaic holds no 2×2 cell"))
    nb, mb = cld(H2, block), cld(W2, block)
    backend = KA.get_backend(raw)
    dark_d = _offset_map(sensor_model.black, m, backend)
    mask_d = _defect_mask(sensor_model.defects, m, backend)
    n = similar(raw, Int, nb, mb); sm = similar(raw, Float64, nb, mb); sd = similar(raw, Float64, nb, mb); sd2 = similar(raw, Float64, nb, mb)
    OnBackend(backend)(_green_pair_blocks_kernel!)(n, sm, sd, sd2, raw, dark_d, mask_d, _QUAD_SITES[gks[1]], _QUAD_SITES[gks[2]],
                                                   gblack, m.white, Int(block), m.window_origin; ndrange = (nb, mb))
    KA.synchronize(backend)
    e = _photon_transfer_fit(Array(n), Array(sm), Array(sd), Array(sd2))
    return PoissonGaussian(e.gain, ntuple(_ -> e.σ, Val(4)), (gain = e.gain_std_error, σ = ntuple(_ -> e.σ_std_error, Val(4)));
                           source = SensorSource.estimated)
end

# The fit over the blocks' sums (host; a few thousand numbers).
function _photon_transfer_fit(n, sm, sd, sd2)
    full = maximum(n)
    keep = findall(==(full), vec(n))                     # whole blocks only: one χ² law, one median correction
    full >= 8 && length(keep) >= 16 || throw(NoiseEstimateFailed(
        "too few usable blocks: $(length(keep)) unclipped blocks of $(full) cells (at least 16 of at least 8 cells)"
    ))
    μ = [sm[k] / full for k in keep]
    v = [(sd2[k] - sd[k]^2 / full) / (full - 1) / 2 for k in keep]   # the difference's sample variance, halved
    groups = min(32, length(keep) ÷ 8)
    order = sortperm(μ)
    G = Distributions.Gamma((full - 1) / 2, 2 / (full - 1))          # the law of v̂/v: χ²_{n−1}/(n − 1)
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
    slope, intercept = β
    (slope > 0 && intercept > 0) || throw(NoiseEstimateFailed(
        "the photon-transfer line is not physical: slope $(slope) (1/K), intercept $(intercept) (σ_r²)"
    ))
    σr = sqrt(intercept)
    return (; gain = 1 / slope, σ = σr, gain_std_error = sqrt(C[1, 1]) / slope^2, σ_std_error = sqrt(C[2, 2]) / (2σr))
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

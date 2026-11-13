# =============================================================================
# Sensor noise — the Poisson–Gaussian model of a body at an ISO, with provenance
# =============================================================================
#
# A RAW photosite's code is `black + electrons / K + read noise` (K electrons per
# code, the read noise σ_r codes, Gaussian, the rounding included), so its
# variance about its black-subtracted mean μ is `μ / K + σ_r²`. Lens-cap (dark)
# frames add the per-photosite offset (DSNU and dark current) and the defects.
#
#   SensorNoise          the values of one body at one ISO, each field `nothing`
#                        or a value with its SOURCE: :measured (a record of dark
#                        frames, flats or a photon-transfer capture), :estimated
#                        (fitted from the frame being processed) or :file (the
#                        RAW's own metadata, LibRaw's black level)
#   DarkFrame            a master dark: the per-photosite offset and temporal
#                        variance of an exposure-time band
#   sensor_noise         THE resolver, the ONE place a missing value is decided
#   photon_transfer      the ONE estimator of K and σ_r from a frame (case 3)
#   master_dark          the ONE dark-frame estimator (the record's black, σ_r,
#                        offset map and defects)
#
# The black subtraction itself is LinearRaw's one kernel, which takes the
# resolved noise (`LinearRaw(mosaic; mode, noise)`, linear_raw.jl). Hardware
# identity lives beside the values (a DeviceIdentity key), never in a camera.
# =============================================================================

"""
    NOISE_SOURCES = (:measured, :estimated, :file)

Where a [`SensorNoise`](@ref) field's value came from: `:measured` (a record of
record: dark frames, flats or a photon-transfer capture), `:estimated` (fitted
from the frames being processed, with its standard error) or `:file` (the RAW
file's own metadata: LibRaw's black level).
"""
const NOISE_SOURCES = (:measured, :estimated, :file)

"""
    NoiseField{V} = Union{Nothing, Attributed{V, @NamedTuple{source::Symbol, std_error::Union{Nothing, V}}}}

One field of a [`SensorNoise`](@ref): `nothing` where it is not known, or the
value with its `source` (one of [`NOISE_SOURCES`](@ref)) and its standard
error (`nothing` where none is stated; an estimate always states one). Built
by [`noise_value`](@ref).
"""
const NoiseField{V} = Union{Nothing, Attributed{V, @NamedTuple{source::Symbol, std_error::Union{Nothing, V}}}}

"""
    noise_value(value, source::Symbol; std_error = nothing) -> Attributed

A [`SensorNoise`](@ref) field: `value` with its `source` (one of
[`NOISE_SOURCES`](@ref), `ArgumentError` otherwise) and its standard error in
the value's units and shape (`nothing` where none is stated).
"""
function noise_value(value::V, source::Symbol; std_error::Union{Nothing, V} = nothing) where {V}
    source in NOISE_SOURCES || throw(ArgumentError("a sensor-noise value's source is one of $(NOISE_SOURCES), got :$source"))
    # the attributes' type states `Union{Nothing, V}` whatever is given, so every field of one kind is one type
    return Attributed(value, NamedTuple{(:source, :std_error), Tuple{Symbol, Union{Nothing, V}}}((source, std_error)))
end

"""
    DarkFrame(offset, variance, exposure_band, frames)

A master dark of one body at one ISO, for one exposure-time band: per
photosite of the visible raw (the size of the decoded mosaic, `raw[row,
column]`), `offset` the mean code of the lens-capped frames (the black level,
the DSNU and the dark current together) and `variance` its temporal variance
(codes²; the read noise and the dark current's shot noise); `exposure_band =
(shortest, longest)`, time quantities, the exposure times it serves (stored in
seconds); `frames`, the number of frames averaged. Made by
[`master_dark`](@ref); [`LinearRaw`](@ref)'s kernel subtracts `offset` per
photosite when a resolved [`SensorNoise`](@ref) carries it.
"""
struct DarkFrame{M <: AbstractMatrix{Float32}}
    offset::M
    variance::M
    exposure_band::NTuple{2, BaseQuantity}
    frames::Int
    function DarkFrame(offset::M, variance::M, exposure_band, frames::Integer) where {M <: AbstractMatrix{Float32}}
        size(offset) == size(variance) ||
            throw(DimensionMismatch("DarkFrame: the offset is $(size(offset)), the variance $(size(variance))"))
        lo, hi = map(t -> Cameras._positive_quantity(t, _SECOND, :exposure_band), exposure_band)
        lo <= hi || throw(ArgumentError("DarkFrame: the exposure band runs from its shortest to its longest time, got ($lo, $hi)"))
        frames >= 2 || throw(ArgumentError("DarkFrame: a temporal variance needs at least 2 frames, got $frames"))
        return new{M}(offset, variance, (lo, hi), Int(frames))
    end
end
Base.:(==)(a::DarkFrame, b::DarkFrame) =
    a.offset == b.offset && a.variance == b.variance && a.exposure_band == b.exposure_band && a.frames == b.frames
Adapt.adapt_structure(to, d::DarkFrame) =
    DarkFrame(Adapt.adapt(to, d.offset), Adapt.adapt(to, d.variance), d.exposure_band, d.frames)

# Whether the band serves an exposure time.
_serves(d::DarkFrame, t) = d.exposure_band[1] <= t <= d.exposure_band[2]

"""
    SensorNoise(; body, iso, electrons_per_dn = nothing, read_noise = nothing, black = nothing,
                dark = nothing, defects = nothing, reason = nothing)

The Poisson–Gaussian noise of one camera body at one ISO, every field `nothing`
or a value with its source ([`noise_value`](@ref), [`NOISE_SOURCES`](@ref)):

- `body::DeviceIdentity` — the key: which body (serial if known, else make and
  model). Held beside the camera, like [`CaptureSettings`](@ref), never in a
  camera or lens model.
- `iso::Float64` — the ISO the values hold at (positive).
- `electrons_per_dn` — K, the conversion gain, electrons per raw code (positive).
- `read_noise` — σ_r per CFA colour index (LibRaw's `0` R, `1` G, `2` B, `3` the
  second G: `read_noise[c + 1]`), in codes, the rounding included (nonnegative).
- `black` — the black level per colour index, in codes (as `RawMosaic.black`).
- `dark` — a [`DarkFrame`](@ref), the per-photosite offset and variance of the
  exposure band the frame's exposure falls in.
- `defects` — a hot/dead-photosite mask over the visible raw (`true` = defective:
  masked, never interpolated).
- `reason` — why the noise (`electrons_per_dn` and `read_noise`, both) is
  `nothing`, a sentence: the resolver's case 4. A record states none, a partial
  one included (a field it does not hold is simply `nothing`).

The resolver [`sensor_noise`](@ref) makes one per frame and is the ONLY place a
missing value is decided: no consumer constructs one or substitutes a constant
(`ArgumentError`s here guard the values, not their absence). Which case a
resolved value came from is its fields' sources: all `:measured` (a record),
mixed (a partial record, the rest estimated), `:estimated`/`:file` (no
record), or a `nothing` noise with its `reason` (the estimate failed).
"""
# Every field's type is a parameter, so a value is stored as it is built (no conversion into the field's Union).
struct SensorNoise{K <: NoiseField{Float64}, R <: NoiseField{NTuple{4, Float64}}, B <: NoiseField{NTuple{4, Float64}},
                   D <: Union{Nothing, Attributed{<:DarkFrame}}, M <: NoiseField{<:AbstractMatrix{Bool}}}
    body::DeviceIdentity
    iso::Float64
    electrons_per_dn::K
    read_noise::R
    black::B
    dark::D
    defects::M
    reason::Union{Nothing, String}
end
function SensorNoise(; body::DeviceIdentity, iso::Real, electrons_per_dn::NoiseField{Float64} = nothing,
                     read_noise::NoiseField{NTuple{4, Float64}} = nothing, black::NoiseField{NTuple{4, Float64}} = nothing,
                     dark::Union{Nothing, Attributed{<:DarkFrame}} = nothing, defects = nothing,
                     reason::Union{Nothing, AbstractString} = nothing)
    isfinite(iso) && iso > 0 || throw(ArgumentError("SensorNoise: the ISO is positive and finite, got $iso"))
    for (name, f) in ((:electrons_per_dn, electrons_per_dn), (:read_noise, read_noise), (:black, black), (:dark, dark), (:defects, defects))
        f === nothing || getfield(f, :attributes).source in NOISE_SOURCES ||
            throw(ArgumentError("SensorNoise: `$name`'s source is one of $(NOISE_SOURCES), got :$(getfield(f, :attributes).source)"))
    end
    electrons_per_dn === nothing || (k = value_of(electrons_per_dn); isfinite(k) && k > 0) ||
        throw(ArgumentError("SensorNoise: electrons_per_dn is positive and finite, got $(value_of(electrons_per_dn))"))
    read_noise === nothing || all(s -> isfinite(s) && s >= 0, value_of(read_noise)) ||
        throw(ArgumentError("SensorNoise: the read noise is nonnegative and finite per channel, got $(value_of(read_noise))"))
    black === nothing || all(isfinite, value_of(black)) || throw(ArgumentError("SensorNoise: the black level is finite, got $(value_of(black))"))
    defects === nothing || value_of(defects) isa AbstractMatrix{Bool} ||
        throw(ArgumentError("SensorNoise: `defects` is a Bool mask over the visible raw, got $(typeof(value_of(defects)))"))
    reason === nothing || (electrons_per_dn === nothing && read_noise === nothing) || throw(ArgumentError(
        "SensorNoise: a `reason` states why the noise (electrons_per_dn and read_noise) is missing, and only then"
    ))
    return SensorNoise(body, Float64(iso), electrons_per_dn, read_noise, black, dark, defects,
                       reason === nothing ? nothing : String(reason))
end
Base.:(==)(a::SensorNoise, b::SensorNoise) = all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(SensorNoise))

# ------------------------------------------------------------------------------
# The resolver
# ------------------------------------------------------------------------------

# Whether a record is of this body: by serial when both state one (make and model agreeing where both state them),
# else by make and model, both stated.
function _same_body(a::DeviceIdentity, b::DeviceIdentity)
    agree(x, y) = x === nothing || y === nothing || x == y
    a.serial !== nothing && b.serial !== nothing && return a.serial == b.serial && agree(a.make, b.make) && agree(a.model, b.model)
    return a.make !== nothing && a.model !== nothing && a.make == b.make && a.model == b.model
end

"""
    sensor_noise(capture::CaptureSettings, frame::Pair{<:Integer, <:RawMosaic}; record = nothing) -> SensorNoise

THE resolver of a frame's sensor noise: frame `t => mosaic` of `capture` (its
body `capture.body`, its ISO and exposure time `capture[t].exposure`), with the
measured `record` (a [`SensorNoise`](@ref) or a vector of them, a body's records
of record at its ISOs; `nothing` when none exists). The ONLY place a missing
value is decided, in this order, the result's fields stating which case each
came from:

1. A record of this body (serial if both state one, else make and model) at
   this ISO: its values, as recorded (`:measured`); its dark frame only when the
   frame's exposure lies in the dark's band.
2. A partial record (dark frames but no gain, say): what it holds, and the rest
   estimated as in 3 — sources per field.
3. No record: K and σ_r estimated from the frame itself by
   [`photon_transfer`](@ref) (`:estimated`, with standard errors), σ_r stated
   for every channel; the black level the file's per-channel `black` (`:file`,
   today's behaviour); no dark map and no defect mask.
4. The estimate fails (too few flat blocks, a saturated or degenerate frame): the
   noise is `nothing` with its `reason` recorded. No constant is invented.

The black level and the dark offset the estimate subtracts are the resolved
ones. Never a silent default: every field states its source.
"""
function sensor_noise(capture::CaptureSettings, frame::Pair{<:Integer, <:RawMosaic}; record = nothing)
    t, mosaic = frame
    haskey(capture, t) || throw(ArgumentError("sensor_noise: the capture has no settings for frame $t"))
    ex = capture[t].exposure
    rec = _matching_record(record, capture.body, ex.iso)
    file_black = noise_value(mosaic.black, :file)
    black = rec === nothing || rec.black === nothing ? file_black : rec.black
    dark = rec === nothing || rec.dark === nothing || !_serves(value_of(rec.dark), ex.exposure_time) ? nothing : rec.dark
    defects = rec === nothing ? nothing : rec.defects
    K = rec === nothing ? nothing : rec.electrons_per_dn
    σ = rec === nothing ? nothing : rec.read_noise
    reason = nothing
    if K === nothing || σ === nothing
        est = try
            photon_transfer(mosaic; black = value_of(black), dark = dark === nothing ? nothing : value_of(dark).offset,
                            mask = defects === nothing ? nothing : value_of(defects))
        catch e
            e isa NoiseEstimateFailed || rethrow()
            reason = "sensor_noise: no record of this body at ISO $(ex.iso) gives the noise, and the frame's photon-transfer estimate failed: $(e.reason)"
            nothing
        end
        if est === nothing
            K = σ = nothing
        else
            K === nothing && (K = noise_value(est.electrons_per_dn, :estimated; std_error = est.electrons_per_dn_std_error))
            σ === nothing && (σ = noise_value(ntuple(_ -> est.read_noise, Val(4)), :estimated;
                                              std_error = ntuple(_ -> est.read_noise_std_error, Val(4))))
        end
    end
    return SensorNoise(; body = capture.body, iso = ex.iso, electrons_per_dn = reason === nothing ? K : nothing,
                       read_noise = reason === nothing ? σ : nothing, black, dark, defects, reason)
end

_matching_record(::Nothing, body::DeviceIdentity, iso::Float64) = nothing
_matching_record(r::SensorNoise, body::DeviceIdentity, iso::Float64) = _matching_record((r,), body, iso)
function _matching_record(records, body::DeviceIdentity, iso::Float64)
    found = nothing
    for r in records
        r isa SensorNoise || throw(ArgumentError("sensor_noise: a record is a SensorNoise, got $(typeof(r))"))
        (_same_body(r.body, body) && r.iso == iso) || continue
        found === nothing || throw(ArgumentError("sensor_noise: two records of body $(body) at ISO $(iso) — a body has one record per ISO"))
        found = r
    end
    return found
end

# ------------------------------------------------------------------------------
# The estimators
# ------------------------------------------------------------------------------

"""
    NoiseEstimateFailed(reason)

The exception [`photon_transfer`](@ref) and [`master_dark`](@ref) throw when
the data cannot give the estimate (too few usable samples, a degenerate fit),
`reason` a sentence. [`sensor_noise`](@ref) turns it into a `nothing` noise
with that reason (case 4).
"""
struct NoiseEstimateFailed <: Exception
    reason::String
end
Base.showerror(io::IO, e::NoiseEstimateFailed) = print(io, "NoiseEstimateFailed: ", e.reason)

# The black a photosite's code is taken from: a dark map's offset there, else its channel's constant — THE rule, read
# by LinearRaw's cell kernel, linearview and the estimator alike. `r, c` index the visible raw (a window's photosite
# plus its `window_origin`).
@inline _black_at(::Nothing, black::Float64, r, c) = black
@inline _black_at(dark::AbstractMatrix, black::Float64, r, c) = @inbounds Float64(dark[r, c])
# Whether a photosite of the visible raw is masked (a defect).
@inline _masked(::Nothing, r, c) = false
@inline _masked(mask::AbstractMatrix{Bool}, r, c) = @inbounds mask[r, c]

# A per-photosite map of the visible raw (a dark offset, a defect mask) as a kernel reads it: on the mosaic's device as
# eltype `T` (the map itself when it is there already), checked to cover the mosaic's photosites at its window offset.
# `nothing` stays `nothing`. A packed BitMatrix is unpacked to Bools beforehand, on every backend (a device copy reads
# dense memory; the kernel then reads one layout everywhere).
_photosite_map(::Nothing, m, backend, ::Type, what) = nothing
function _photosite_map(A::AbstractMatrix, m::RawMosaic, backend, ::Type{T}, what) where {T}
    all(m.window_origin .+ size(m) .<= size(A)) || throw(DimensionMismatch(
        "$what is $(join(size(A), '×')) over the visible raw; the mosaic's photosites reach $(join(m.window_origin .+ size(m), '×'))"
    ))
    return on_device(T, backend, A isa BitArray ? Array{Bool}(A) : A)
end

# The levels a resolved noise (or `nothing`) gives the linearisation of `m`: the four channels' black and span
# (`_bayer_levels`, with the noise's per-colour-index black when it holds one, else the mosaic's) and, on the
# backend, the dark offset map and the defect mask (`nothing` where absent). THE resolution LinearRaw and
# linearview share; with `nothing` it is the mosaic's own levels and no maps, bit for bit today's.
_noise_levels(m::RawMosaic, ::Nothing, backend) = (_bayer_levels(m)..., nothing, nothing)
function _noise_levels(m::RawMosaic, noise::SensorNoise, backend)
    black, span = _bayer_levels(m, noise.black === nothing ? m.black : value_of(noise.black))
    dark = noise.dark === nothing ? nothing : _photosite_map(value_of(noise.dark).offset, m, backend, Float32, "the dark offset")
    defects = noise.defects === nothing ? nothing : _photosite_map(value_of(noise.defects), m, backend, Bool, "the defect mask")
    return black, span, dark, defects
end
_noise_levels(::RawMosaic, noise, backend) =
    throw(ArgumentError("the linearisation's noise is a resolved SensorNoise (sensor_noise) or nothing, got $(typeof(noise))"))

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
    photon_transfer(mosaic::RawMosaic; black = mosaic.black, dark = nothing, mask = nothing, block = 8)
        -> (; electrons_per_dn, read_noise, electrons_per_dn_std_error, read_noise_std_error, blocks)

THE estimator of a frame's Poisson–Gaussian noise (case 3 of
[`sensor_noise`](@ref)): the conversion gain K (electrons per code) and the read
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
K's and σ_r's by first order.

`black` is the per-colour-index black level (`RawMosaic.black`'s layout), `dark`
an offset map over the visible raw that replaces it per photosite (THE rule of
[`LinearRaw`](@ref)'s kernel), `mask` the photosites to leave out; both maps
cover the visible raw (a window reads them at its `window_origin`). Throws
[`NoiseEstimateFailed`](@ref) with the reason when fewer than 16 whole blocks
of at least 8 usable cells remain (a saturated frame), the groups hold one
level only (no slope), or the fit is not physical (K or σ_r² not positive).
"""
function photon_transfer(m::RawMosaic; black::NTuple{4, <:Real} = m.black, dark::Union{Nothing, AbstractMatrix} = nothing,
                         mask::Union{Nothing, AbstractMatrix{Bool}} = nothing, block::Integer = 8)
    block >= 2 || throw(ArgumentError("photon_transfer: a block holds at least 2 × 2 cells, got $block"))
    cell = _cell_colors(m, "photon_transfer")
    gks = Tuple(k for k in 1:4 if cell[k] in (0x01, 0x03))
    gblack = (Float64(black[cell[gks[1]] + 1]), Float64(black[cell[gks[2]] + 1]))
    raw = parent(m)
    H2, W2 = size(raw, 1) ÷ 2, size(raw, 2) ÷ 2
    H2 >= 1 && W2 >= 1 || throw(NoiseEstimateFailed("a $(join(size(m), '×')) mosaic holds no 2×2 cell"))
    nb, mb = cld(H2, block), cld(W2, block)
    backend = KA.get_backend(raw)
    dark_d = _photosite_map(dark, m, backend, Float32, "photon_transfer: the dark offset")
    mask_d = _photosite_map(mask, m, backend, Bool, "photon_transfer: the mask")
    n = similar(raw, Int, nb, mb); sm = similar(raw, Float64, nb, mb); sd = similar(raw, Float64, nb, mb); sd2 = similar(raw, Float64, nb, mb)
    OnBackend(backend)(_green_pair_blocks_kernel!)(n, sm, sd, sd2, raw, dark_d, mask_d, _QUAD_SITES[gks[1]], _QUAD_SITES[gks[2]],
                                                   gblack, m.white, Int(block), m.window_origin; ndrange = (nb, mb))
    KA.synchronize(backend)
    return _photon_transfer_fit(Array(n), Array(sm), Array(sd), Array(sd2))
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
    return (; electrons_per_dn = 1 / slope, read_noise = σr, electrons_per_dn_std_error = sqrt(C[1, 1]) / slope^2,
            read_noise_std_error = sqrt(C[2, 2]) / (2σr), blocks = length(keep))
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
        -> (; dark::DarkFrame, black, read_noise, black_std_error, read_noise_std_error, defects)

THE dark-frame estimator: the lens-capped frames of one body at one ISO and one
exposure band (whole mosaics of one size, at least 2) into the values of its
record ([`SensorNoise`](@ref) with source `:measured`):

- `dark` — the [`DarkFrame`](@ref): per photosite the temporal mean (the offset:
  black, DSNU and dark current) and sample variance, one KernelAbstractions work
  item each, on the frames' device;
- `black` — per colour index (LibRaw's `0`–`3`), the mean offset of its
  photosites (defects left out), and `black_std_error` its standard error from
  their spread and count;
- `read_noise` — per colour index, the root mean temporal variance (the read
  noise with the rounding, and at a long exposure the dark current's shot noise),
  and `read_noise_std_error` from the `χ²` spread of the frame count and the
  photosite count (`√(2/(n − 1)/N)·σ²` for the mean variance, by first order);
- `defects` — a `BitMatrix`, the photosites whose offset leaves their
  channel's median by more than `z` robust standard deviations (the median
  absolute deviation over the normal's, [`mad_scale`](@ref)'s rule for
  codimension 1, about the channel's median), `z` the normal quantile at which
  `false_defects` defect-free photosites are expected to be flagged over the
  whole sensor of `N` photosites (`1 − false_defects / (2N)`, two-sided, for
  offsets that are Gaussian about their channel's black).

A colour index the pattern lacks (a pattern with one green index) has black
and read noise `0` (no photosite reads it).

`exposure_band` is `(shortest, longest)`, time quantities, the exposures the
record serves. A frame of another size, or fewer than 2, is an
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
    dark = DarkFrame(offset, variance, exposure_band, length(frames))
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
    return (; dark, black = Tuple(black), read_noise = Tuple(σr), black_std_error = Tuple(black_se),
            read_noise_std_error = Tuple(σr_se), defects)
end

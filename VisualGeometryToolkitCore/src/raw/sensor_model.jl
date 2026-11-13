# =============================================================================
# The sensor model — what a body's sensor adds to its codes, by component, with provenance
# =============================================================================
#
# A RAW photosite's code is `black + electrons / K + read noise` (K electrons per
# code, the read noise σ_r codes, Gaussian, the rounding included), so its
# variance about its black-subtracted mean μ is `μ / K + σ_r²`. Lens-cap (dark)
# frames add the per-photosite offset and variance, and the defects.
#
# Three levels (spec 2026-11-13, Phase 1 and the owner's rulings D1–D13):
#
#   Sensor                   geometry only (cameras/sensors.jl), part of BoundedCamera
#   SensorCharacterisation   host-side: ONE physical body (its DeviceIdentity, held
#                            beside the camera) and, per ISO, a component per slot
#   SensorModel{B,N,D}       per frame: the black, noise and defect components
#                            [`sensor_model`](@ref) picked for its ISO and exposure,
#                            and why (a `ModelReason`)
#
# The slots and their components (one type each, every present one with its
# `source`, a `SensorSource.T`):
#
#   black     FileBlack (the RAW's own per-channel black: the absent default),
#             ChannelBlack (a per-channel black), Dark (a master dark: its channel
#             black plus the per-photosite offset and variance, D2)
#   noise     NoNoise (absent), ReadNoise (σ_r), PoissonGaussian (the gain and σ_r),
#             RelativeNoise (a variance (σ_rel·x)² of the value x)
#   defects   NoDefects (absent), DefectList (photosites with their onset exposures)
#
# Absence is a TYPE: a consumer dispatches on the component, never checks for
# `nothing`. The absent model `SensorModel()` linearises bit for bit as the RAW
# alone does. Consumers never construct a component (check_duplication's
# `:sensor_component` pattern); tests build them through Core's one fixture
# helper, `sensor_fixture`.
# =============================================================================

"""
    SensorSource

Where a sensor component's values came from, `SensorSource.T` (an isbits
`@enum`): `SensorSource.measured` (a record of dark frames, flats or a
photon-transfer capture), `SensorSource.estimated` (fitted by an estimator,
[`photon_transfer`](@ref), with its standard error), `SensorSource.specified`
(stated by a person: a datasheet, a config, TOML literals) or
`SensorSource.file` (the RAW file's own metadata: [`FileBlack`](@ref)'s).
`string(s)` is its tag (`"measured"`, …). Each present component holds one in
its `source` field. A specified component wins over a measured one at the same
slot ([`sensor_model`](@ref)).
"""
module SensorSource
@enum T::UInt8 measured estimated specified file
end

"""
    ModelReason

Why [`sensor_model`](@ref) gave a frame the model it did (an isbits `@enum`,
`SensorModel.reason`):

- `Matched`: a characterisation of this body holds an entry at the frame's ISO;
  its components are the model's (an absent slot of the entry is the absent type);
- `NoCharacterisation`: none was given (also the absent default `SensorModel()`);
- `UnmatchedISO`: characterisations of this body exist, none at this ISO (no
  extrapolation across ISO);
- `OtherBody`: the characterisations given are of other bodies.

Every reason but `Matched` gives the absent model.
"""
@enum ModelReason::UInt8 Matched NoCharacterisation UnmatchedISO OtherBody

@doc "`Matched`: a [`ModelReason`](@ref) — an entry of this body at the frame's ISO gave the model." Matched
@doc "`NoCharacterisation`: a [`ModelReason`](@ref) — no characterisation was given; the absent model." NoCharacterisation
@doc "`UnmatchedISO`: a [`ModelReason`](@ref) — this body is characterised, but not at the frame's ISO; the absent model." UnmatchedISO
@doc "`OtherBody`: a [`ModelReason`](@ref) — the characterisations given are of other bodies; the absent model." OtherBody

"""
    BlackComponent, NoiseComponent, DefectComponent

The three slots of a [`SensorModel`](@ref) and of a
[`SensorCharacterisation`](@ref) entry: what a code's black level is
([`FileBlack`](@ref), [`ChannelBlack`](@ref), [`Dark`](@ref)), its noise
([`NoNoise`](@ref), [`ReadNoise`](@ref), [`PoissonGaussian`](@ref),
[`RelativeNoise`](@ref)) and which photosites are defects
([`NoDefects`](@ref), [`DefectList`](@ref)).
"""
abstract type BlackComponent end
@doc (@doc BlackComponent) abstract type NoiseComponent end
@doc (@doc BlackComponent) abstract type DefectComponent end

# A present component's source: one of the four, and `file` only where the RAW's metadata is what it holds.
function _component_source(source::SensorSource.T, what)
    source === SensorSource.file &&
        throw(ArgumentError("$what: `SensorSource.file` is the RAW's own black, FileBlack(); a $what is measured, estimated or specified"))
    return source
end

_finite_tuple(v, what) = (t = NTuple{4, Float64}(Tuple(v)); all(isfinite, t) || throw(ArgumentError("$what is finite per channel, got $t")); t)
_nonnegative_tuple(v, what) =
    (t = _finite_tuple(v, what); all(>=(0), t) || throw(ArgumentError("$what is nonnegative per channel, got $t")); t)

# ------------------------------------------------------------------------------
# The black slot
# ------------------------------------------------------------------------------

"""
    FileBlack()

The black slot's absent default: the RAW file's own per-channel black level
(`RawMosaic.black`, LibRaw's), today's behaviour bit for bit. Its `source` is
`SensorSource.file`.
"""
struct FileBlack <: BlackComponent
    source::SensorSource.T
    FileBlack() = new(SensorSource.file)
end

"""
    ChannelBlack(values, std_error = nothing; source::SensorSource.T)

A per-channel black level: `values` per CFA colour index (LibRaw's `0` R, `1` G,
`2` B, `3` the second G: `values[c + 1]`, `RawMosaic.black`'s layout), in codes,
finite; `std_error` its standard error per channel (nonnegative), or `nothing`
where none is stated (a specified value). `source` is measured, estimated or
specified (the RAW's own is [`FileBlack`](@ref)). Every value is below the white
level where it is used (`ArgumentError` there).
"""
struct ChannelBlack{E <: Union{Nothing, NTuple{4, Float64}}} <: BlackComponent
    values::NTuple{4, Float64}
    std_error::E
    source::SensorSource.T
end
ChannelBlack(values, std_error = nothing; source::SensorSource.T) =
    ChannelBlack(_finite_tuple(values, "ChannelBlack: the black level"),
                 std_error === nothing ? nothing : _nonnegative_tuple(std_error, "ChannelBlack: the standard error"),
                 _component_source(source, "ChannelBlack"))

"""
    Dark(black, offset, variance, exposure_band, frames; source::SensorSource.T)

A master dark of one body at one ISO, for one exposure-time band — the black
slot's per-photosite form (D2: all dark information in ONE component):

- `black` — its per-channel black ([`ChannelBlack`](@ref) or [`FileBlack`](@ref)):
  the span `white − black[c]` of every photosite of the channel, and the black
  a frame whose exposure the band does not serve is linearised with;
- `offset` — per photosite of the visible raw (`raw[row, column]`, the size of the
  decoded mosaic), the mean code of the lens-capped frames (the black, the DSNU
  and the dark current together): [`LinearRaw`](@ref)'s kernel subtracts it in
  place of the channel's black;
- `variance` — its temporal variance (codes²: the read noise with the rounding,
  and the dark current's shot noise), the Gaussian part of a
  [`PoissonGaussian`](@ref) model's variance at that photosite;
- `exposure_band = (shortest, longest)`, time quantities, the exposures it serves
  (stored in seconds);
- `frames`, the number of frames averaged (at least 2).

Made by [`master_dark`](@ref) (`SensorSource.measured`). Moved to a device by
`Adapt.adapt`, as [`LinearRawView`](@ref) is.
"""
struct Dark{B <: Union{FileBlack, ChannelBlack}, M <: AbstractMatrix{Float32}} <: BlackComponent
    black::B
    offset::M
    variance::M
    exposure_band::NTuple{2, BaseQuantity}
    frames::Int
    source::SensorSource.T
    function Dark(black::B, offset::M, variance::M, exposure_band, frames::Integer; source::SensorSource.T) where {B <: Union{FileBlack, ChannelBlack}, M <: AbstractMatrix{Float32}}
        size(offset) == size(variance) ||
            throw(DimensionMismatch("Dark: the offset is $(size(offset)), the variance $(size(variance))"))
        lo, hi = map(t -> Cameras._positive_quantity(t, _SECOND, :exposure_band), exposure_band)
        lo <= hi || throw(ArgumentError("Dark: the exposure band runs from its shortest to its longest time, got ($lo, $hi)"))
        frames >= 2 || throw(ArgumentError("Dark: a temporal variance needs at least 2 frames, got $frames"))
        return new{B, M}(black, offset, variance, (lo, hi), Int(frames), _component_source(source, "Dark"))
    end
end
Adapt.adapt_structure(to, d::Dark) =
    Dark(d.black, Adapt.adapt(to, d.offset), Adapt.adapt(to, d.variance), d.exposure_band, d.frames; source = d.source)

# Whether a dark's band serves an exposure time.
_serves(d::Dark, t) = d.exposure_band[1] <= t <= d.exposure_band[2]

# ------------------------------------------------------------------------------
# The noise slot
# ------------------------------------------------------------------------------

"""
    NoNoise()

The noise slot's absent default: no noise is stated. A [`LinearRaw`](@ref)
image is plain, and each detector applies its own documented policy.
"""
struct NoNoise <: NoiseComponent end

"""
    ReadNoise(σ, std_error = nothing; source::SensorSource.T)

The read noise alone: `σ` per CFA colour index (`σ[c + 1]`), in codes, the
rounding included (nonnegative, finite), and its standard error per channel or
`nothing`. Stored and recorded, but it defines no pixel variance: a read-noise
variance alone badly underestimates the noise of a well-exposed pixel, so a
[`LinearRaw`](@ref) image under it is plain, as under [`NoNoise`](@ref)
(ruling A).
"""
struct ReadNoise{E <: Union{Nothing, NTuple{4, Float64}}} <: NoiseComponent
    σ::NTuple{4, Float64}
    std_error::E
    source::SensorSource.T
end
ReadNoise(σ, std_error = nothing; source::SensorSource.T) =
    ReadNoise(_nonnegative_tuple(σ, "ReadNoise: the read noise"),
              std_error === nothing ? nothing : _nonnegative_tuple(std_error, "ReadNoise: the standard error"),
              _component_source(source, "ReadNoise"))

"""
    PoissonGaussian(gain, σ, std_error = nothing; source::SensorSource.T)

The Poisson–Gaussian noise: `gain` K, electrons per code (positive, finite), and
the read noise `σ` per CFA colour index, in codes, the rounding included; a
photosite's variance about its black-subtracted mean μ (codes) is `μ/K + σ[c+1]²`.
`std_error` is `(; gain, σ)` (a number and a 4-tuple) or `nothing`. `source` is
measured, estimated ([`photon_transfer`](@ref)) or specified. Under it a
[`LinearRaw`](@ref) image carries each value's variance (an `Uncertain` image).
"""
struct PoissonGaussian{E <: Union{Nothing, @NamedTuple{gain::Float64, σ::NTuple{4, Float64}}}} <: NoiseComponent
    gain::Float64
    σ::NTuple{4, Float64}
    std_error::E
    source::SensorSource.T
end
function PoissonGaussian(gain::Real, σ, std_error = nothing; source::SensorSource.T)
    isfinite(gain) && gain > 0 || throw(ArgumentError("PoissonGaussian: the gain (electrons per code) is positive and finite, got $gain"))
    std_error === nothing || isfinite(std_error.gain) && std_error.gain >= 0 ||
        throw(ArgumentError("PoissonGaussian: the gain's standard error is nonnegative and finite, got $(std_error.gain)"))
    se = std_error === nothing ? nothing :
        (gain = Float64(std_error.gain), σ = _nonnegative_tuple(std_error.σ, "PoissonGaussian: the read noise's standard error"))
    return PoissonGaussian(Float64(gain), _nonnegative_tuple(σ, "PoissonGaussian: the read noise"), se,
                           _component_source(source, "PoissonGaussian"))
end

"""
    RelativeNoise(σ_rel; source::SensorSource.T)

A noise proportional to the value: a value `x` has variance `(σ_rel·x)²`
(`σ_rel` positive, finite; typically `SensorSource.specified`). The model of a
processed (tone-mapped) intensity, not of the photosites' physics; a
[`LinearRaw`](@ref) image under it carries each value's variance (an
`Uncertain` image), with no read-noise or dark term.
"""
struct RelativeNoise <: NoiseComponent
    σ_rel::Float64
    source::SensorSource.T
    function RelativeNoise(σ_rel::Real; source::SensorSource.T)
        isfinite(σ_rel) && σ_rel > 0 || throw(ArgumentError("RelativeNoise: σ_rel is positive and finite, got $σ_rel"))
        return new(Float64(σ_rel), _component_source(source, "RelativeNoise"))
    end
end

# ------------------------------------------------------------------------------
# The defect slot
# ------------------------------------------------------------------------------

"""
    NoDefects()

The defect slot's absent default: no photosite is masked beyond the saturated ones.
"""
struct NoDefects <: DefectComponent end

"""
    DefectList(x, y, onset; size, source::SensorSource.T)

The defective photosites of a sensor: photosite `(x[i], y[i])` — `x` the column
and `y` the row of the visible raw, 1-based (`raw[y, x]`) — is a defect at every
exposure of at least `onset[i]` (a time quantity, stored in seconds: the
shortest exposure at which it is hot). `size = (rows, columns)` is the visible
raw's (every site inside it, each listed once). Made from per-exposure masks by
[`defect_onsets`](@ref). [`sensor_model`](@ref) keeps the sites whose onset is
at most the frame's exposure, and [`LinearRaw`](@ref) masks them (masked, never
interpolated). Moved to a device by `Adapt.adapt`.
"""
struct DefectList{I <: AbstractVector{Int}, O <: AbstractVector{BaseQuantity}} <: DefectComponent
    x::I
    y::I
    onset::O
    size::NTuple{2, Int}
    source::SensorSource.T
    # the adapted list: the sites as checked on the host
    DefectList{I, O}(x::I, y::I, onset::O, size::NTuple{2, Int}, source::SensorSource.T) where {I, O} =
        new{I, O}(x, y, onset, size, source)
end
function DefectList(x::AbstractVector{<:Integer}, y::AbstractVector{<:Integer}, onset::AbstractVector; size::NTuple{2, Integer},
                    source::SensorSource.T)
    length(x) == length(y) == length(onset) ||
        throw(DimensionMismatch("DefectList: x, y and onset are one per site, got $(length(x)), $(length(y)), $(length(onset))"))
    H, W = Int.(size)
    xs = collect(Int, x); ys = collect(Int, y)
    all(i -> 1 <= xs[i] <= W && 1 <= ys[i] <= H, eachindex(xs)) ||
        throw(ArgumentError("DefectList: every site (x, y) lies in the visible raw, 1 ≤ x ≤ $W, 1 ≤ y ≤ $H"))
    allunique(zip(xs, ys)) || throw(ArgumentError("DefectList: a photosite is listed once"))
    ts = BaseQuantity[Cameras._positive_quantity(t, _SECOND, :onset) for t in onset]
    return DefectList{Vector{Int}, Vector{BaseQuantity}}(xs, ys, ts, (H, W), _component_source(source, "DefectList"))
end
Adapt.adapt_structure(to, d::DefectList) =
    (x = Adapt.adapt(to, d.x); y = Adapt.adapt(to, d.y); o = Adapt.adapt(to, d.onset);
     DefectList{typeof(x), typeof(o)}(x, y, o, d.size, d.source))

# ------------------------------------------------------------------------------
# Equality and hashing, by every field, named (no reflection: a field added is a line added here)
# ------------------------------------------------------------------------------

_sensor_fields(c::FileBlack) = (c.source,)
_sensor_fields(c::ChannelBlack) = (c.values, c.std_error, c.source)
_sensor_fields(c::Dark) = (c.black, c.offset, c.variance, c.exposure_band, c.frames, c.source)
_sensor_fields(::NoNoise) = ()
_sensor_fields(c::ReadNoise) = (c.σ, c.std_error, c.source)
_sensor_fields(c::PoissonGaussian) = (c.gain, c.σ, c.std_error, c.source)
_sensor_fields(c::RelativeNoise) = (c.σ_rel, c.source)
_sensor_fields(::NoDefects) = ()
_sensor_fields(c::DefectList) = (c.x, c.y, c.onset, c.size, c.source)
const _SensorComponent = Union{BlackComponent, NoiseComponent, DefectComponent}
Base.:(==)(a::_SensorComponent, b::_SensorComponent) = typeof(a).name === typeof(b).name && _sensor_fields(a) == _sensor_fields(b)
Base.isequal(a::_SensorComponent, b::_SensorComponent) = typeof(a).name === typeof(b).name && isequal(_sensor_fields(a), _sensor_fields(b))
Base.hash(c::_SensorComponent, h::UInt) = hash(_sensor_fields(c), hash(nameof(typeof(c)), h))

# ------------------------------------------------------------------------------
# The characterisation and the model
# ------------------------------------------------------------------------------

const _ENTRY_SLOTS = (:black, :noise, :defects)
"""
    SensorEntry = @NamedTuple{black::BlackComponent, noise::NoiseComponent, defects::DefectComponent}

One ISO's entry of a [`SensorCharacterisation`](@ref): its three slots, each a
component (an absent slot its absent type).
"""
const SensorEntry = @NamedTuple{black::BlackComponent, noise::NoiseComponent, defects::DefectComponent}

"""
    SensorCharacterisation(body::DeviceIdentity; entries)

The description of ONE physical sensor, host-side: `body`, the key (serial if
known, else make and model), held beside the camera like
[`CaptureSettings`](@ref), never in a camera or lens model; and `entries`, per
ISO its components — an iterable of `iso => (; black, noise, defects)`, each
slot optional (an absent slot IS the absent type: [`FileBlack`](@ref),
[`NoNoise`](@ref), [`NoDefects`](@ref)), so a partial characterisation is a mix
of components. `c.entries` is a `Dict{Float64, SensorEntry}` (ISO → the three
slots). An ISO listed twice, a slot not among `black`, `noise` and `defects`
(a dark map is the black slot's [`Dark`](@ref)), or a non-positive ISO is an
`ArgumentError`.

[`sensor_model`](@ref) resolves a frame's model from it. Written and read by
VisualGeometryToolkitIO's `vgio/sensor_characterisation` kind.
"""
struct SensorCharacterisation
    body::DeviceIdentity
    entries::Dict{Float64, SensorEntry}
end
function SensorCharacterisation(body::DeviceIdentity; entries)
    d = Dict{Float64, SensorEntry}()
    for (iso, slots) in entries
        isfinite(iso) && iso > 0 || throw(ArgumentError("SensorCharacterisation: an ISO is positive and finite, got $iso"))
        haskey(d, Float64(iso)) && throw(ArgumentError("SensorCharacterisation: ISO $iso is listed twice — a body has one entry per ISO"))
        extra = setdiff(keys(slots), _ENTRY_SLOTS)
        isempty(extra) || throw(ArgumentError(
            "SensorCharacterisation: an entry's slots are $(_ENTRY_SLOTS), got $(Tuple(extra)) (a dark map is the black slot's Dark)"))
        d[Float64(iso)] = SensorEntry((get(slots, :black, FileBlack()), get(slots, :noise, NoNoise()), get(slots, :defects, NoDefects())))
    end
    return SensorCharacterisation(body, d)
end
Base.:(==)(a::SensorCharacterisation, b::SensorCharacterisation) = a.body == b.body && a.entries == b.entries
Base.isequal(a::SensorCharacterisation, b::SensorCharacterisation) = isequal(a.body, b.body) && isequal(a.entries, b.entries)
Base.hash(c::SensorCharacterisation, h::UInt) = hash(c.entries, hash(c.body, hash(:SensorCharacterisation, h)))

"""
    SensorModel(black::BlackComponent, noise::NoiseComponent, defects::DefectComponent;
                reason::ModelReason = Matched, specified_override::Bool = false)
    SensorModel()

The sensor model of ONE frame: the `black`, `noise` and `defects` components
[`sensor_model`](@ref) picked for its ISO and exposure (a [`Dark`](@ref) only
when its band serves the exposure, a [`DefectList`](@ref) holding only the
sites whose onset is at most the exposure), `reason` why (a
[`ModelReason`](@ref)) and `specified_override`, whether a specified component
displaced a measured one (the run record notes the override).

`SensorModel()` is the absent model, `SensorModel(FileBlack(), NoNoise(),
NoDefects(); reason = NoCharacterisation)`: [`LinearRaw`](@ref) and
[`linearview`](@ref) under it are bit for bit the RAW's own linearisation. A
model of isbits components is isbits; one holding a dark map or a defect list
moves to a device with `Adapt.adapt`. [`content_digest`](@ref) is its identity.
"""
struct SensorModel{B <: BlackComponent, N <: NoiseComponent, D <: DefectComponent}
    black::B
    noise::N
    defects::D
    reason::ModelReason
    specified_override::Bool
end
SensorModel(black::BlackComponent, noise::NoiseComponent, defects::DefectComponent; reason::ModelReason = Matched,
            specified_override::Bool = false) = SensorModel(black, noise, defects, reason, specified_override)
SensorModel() = SensorModel(FileBlack(), NoNoise(), NoDefects(); reason = NoCharacterisation)
_sensor_fields(m::SensorModel) = (m.black, m.noise, m.defects, m.reason, m.specified_override)
Base.:(==)(a::SensorModel, b::SensorModel) = _sensor_fields(a) == _sensor_fields(b)
Base.isequal(a::SensorModel, b::SensorModel) = isequal(_sensor_fields(a), _sensor_fields(b))
Base.hash(m::SensorModel, h::UInt) = hash(_sensor_fields(m), hash(:SensorModel, h))
Adapt.adapt_structure(to, m::SensorModel) =
    SensorModel(Adapt.adapt(to, m.black), m.noise, Adapt.adapt(to, m.defects), m.reason, m.specified_override)

# ------------------------------------------------------------------------------
# The identity of a model (D9): the stack's one content-identity scheme, sha256_hex(canonical_json(tree))
# ------------------------------------------------------------------------------

_digest_tree(s::SensorSource.T) = string(s)
_digest_tree(t::NTuple{4, Float64}) = collect(t)
_digest_tree(t::BaseQuantity) = ustrip(_SECOND, t)                          # a time, in seconds
_digest_tree(A::AbstractArray{<:Number}) = sha256_hex(reinterpret(UInt8, vec(Array(A))))   # a map by the bytes of its values
_with_std_error(d, ::Nothing) = d
_with_std_error(d, se::NTuple{4, Float64}) = (d["std_error"] = _digest_tree(se); d)
_with_std_error(d, se::NamedTuple) = (d["std_error"] = Dict{String, Any}("gain" => se.gain, "sigma" => _digest_tree(se.σ)); d)
_digest_tree(c::FileBlack) = Dict{String, Any}("type" => "file", "source" => _digest_tree(c.source))
_digest_tree(c::ChannelBlack) = _with_std_error(Dict{String, Any}("type" => "channel", "value" => _digest_tree(c.values), "source" => _digest_tree(c.source)), c.std_error)
_digest_tree(c::Dark) = Dict{String, Any}("type" => "dark", "black" => _digest_tree(c.black), "size" => collect(Int64, size(c.offset)),
                                   "offset" => _digest_tree(c.offset), "variance" => _digest_tree(c.variance),
                                   "exposure_band" => collect(map(_digest_tree, c.exposure_band)), "frames" => Int64(c.frames),
                                   "source" => _digest_tree(c.source))
_digest_tree(::NoNoise) = Dict{String, Any}("type" => "none")
_digest_tree(c::ReadNoise) = _with_std_error(Dict{String, Any}("type" => "read_noise", "sigma" => _digest_tree(c.σ), "source" => _digest_tree(c.source)), c.std_error)
_digest_tree(c::PoissonGaussian) = _with_std_error(Dict{String, Any}("type" => "poisson_gaussian", "gain" => c.gain, "sigma" => _digest_tree(c.σ),
                                                              "source" => _digest_tree(c.source)), c.std_error)
_digest_tree(c::RelativeNoise) = Dict{String, Any}("type" => "relative", "sigma_rel" => c.σ_rel, "source" => _digest_tree(c.source))
_digest_tree(::NoDefects) = Dict{String, Any}("type" => "none")
_digest_tree(c::DefectList) = Dict{String, Any}("type" => "list", "size" => collect(Int64, c.size), "x" => collect(Int64, Array(c.x)),
                                         "y" => collect(Int64, Array(c.y)), "onset" => map(_digest_tree, Array(c.onset)),
                                         "source" => _digest_tree(c.source))

"""
    content_digest(model::SensorModel) -> String

The identity of a frame's sensor model: the SHA-256 (lower-case hex) of the
canonical JSON ([`canonical_json`](@ref), the stack's one content-identity
encoding) of a tree of its components — each component's type, values, standard
errors and source, a map (a [`Dark`](@ref)'s offset and variance) by the
SHA-256 of its Float32 values' bytes and its size, a [`DefectList`](@ref) by its
sites — its `reason` and `specified_override`. Equal models have equal digests,
on any device (a device map is read to the host); a cache keyed by a model
(VisualGeometryToolkitFeatures' decode cache) keys on it.
"""
content_digest(m::SensorModel) =
    sha256_hex(canonical_json(Dict{String, Any}("black" => _digest_tree(m.black), "noise" => _digest_tree(m.noise), "defects" => _digest_tree(m.defects),
                                                 "reason" => string(m.reason), "specified_override" => m.specified_override)))

# ------------------------------------------------------------------------------
# THE resolver (D1: a lookup, no mosaic; it never estimates)
# ------------------------------------------------------------------------------

# Whether a characterisation is of this body: by serial when both state one (make and model agreeing where both state
# them), else by make and model, both stated.
function _same_body(a::DeviceIdentity, b::DeviceIdentity)
    agree(x, y) = x === nothing || y === nothing || x == y
    a.serial !== nothing && b.serial !== nothing && return a.serial == b.serial && agree(a.make, b.make) && agree(a.model, b.model)
    return a.make !== nothing && a.model !== nothing && a.make == b.make && a.model == b.model
end

# Whether a slot holds something: the absent types hold nothing (FileBlack is the black slot's absence).
_present(::Union{FileBlack, NoNoise, NoDefects}) = false
_present(::_SensorComponent) = true

# The precedence of sources at one slot (D5): specified over measured over estimated.
_precedence(s::SensorSource.T) = s === SensorSource.specified ? 3 : s === SensorSource.measured ? 2 : 1

# The component of one slot among the matched entries': the absent type when none holds one, else the one of the
# highest precedence (two of it are ambiguous), and whether a specified one displaced a measured one.
function _pick(candidates, absent, slot, iso)
    present = filter(_present, candidates)
    isempty(present) && return absent, false
    best = maximum(c -> _precedence(c.source), present)
    top = filter(c -> _precedence(c.source) == best, present)
    length(top) == 1 || throw(ArgumentError(
        "sensor_model: two $(string(top[1].source)) components fill the $slot slot at ISO $iso — a body has one per source"))
    return only(top), best == 3 && any(c -> c.source === SensorSource.measured, present)
end

# A frame's black: a Dark only when its band serves the exposure, else its channel black.
_frame_black(b::BlackComponent, t) = b
_frame_black(d::Dark, t) = _serves(d, t) ? d : d.black
# A frame's defects: the sites whose onset is at most the exposure (none: NoDefects).
_frame_defects(d::DefectComponent, t) = d
function _frame_defects(d::DefectList, t)
    keep = findall(<=(t), d.onset)
    isempty(keep) && return NoDefects()
    return DefectList{typeof(d.x), typeof(d.onset)}(d.x[keep], d.y[keep], d.onset[keep], d.size, d.source)
end

"""
    sensor_model(characterisation, capture::CaptureSettings, t::Integer) -> SensorModel

THE resolver of frame `t`'s [`SensorModel`](@ref): a LOOKUP (owner ruling D1;
it never estimates — [`photon_transfer`](@ref) makes an estimated entry
offline). `characterisation` is a [`SensorCharacterisation`](@ref), a vector or
tuple of them, or `nothing`; the frame's body is `capture.body`, its ISO and
exposure `capture[t].exposure`.

1. The characterisations of this body (serial if both state one, else make and
   model; [`DeviceIdentity`](@ref)), and their entries at exactly the frame's ISO
   (no extrapolation across ISO).
2. Per slot, independently, the component those entries hold; where several
   hold one, the specified one over the measured over the estimated (D5), and
   `specified_override` records a specified one displacing a measured one; two
   of the same source are an `ArgumentError`, as are two different bodies
   matching a capture that states no serial.
3. A [`Dark`](@ref) only when its exposure band serves the frame's exposure
   (otherwise its channel black); a [`DefectList`](@ref) cut to the sites whose
   onset is at most the exposure (none left: [`NoDefects`](@ref)).

With no characterisation (`NoCharacterisation`), none of this body
(`OtherBody`) or none at this ISO (`UnmatchedISO`) the result is the absent
model `SensorModel(FileBlack(), NoNoise(), NoDefects())` with that
[`ModelReason`](@ref): today's linearisation, bit for bit. Otherwise its
`reason` is `Matched`. A frame the capture does not hold is an `ArgumentError`.
An explicit model a caller holds is used as it is: this resolver is where a
characterisation becomes one, never a substitute for one given.
"""
function sensor_model(c::Union{Nothing, SensorCharacterisation}, capture::CaptureSettings, t::Integer)
    return sensor_model(c === nothing ? () : (c,), capture, t)
end
function sensor_model(chars::Union{Tuple, AbstractVector}, capture::CaptureSettings, t::Integer)
    haskey(capture, t) || throw(ArgumentError("sensor_model: the capture has no settings for frame $t"))
    ex = capture[t].exposure
    absent(reason) = SensorModel(FileBlack(), NoNoise(), NoDefects(); reason)
    isempty(chars) && return absent(NoCharacterisation)
    foreach(c -> c isa SensorCharacterisation || throw(ArgumentError(
        "sensor_model: a characterisation is a SensorCharacterisation, got $(typeof(c))")), chars)
    mine = [c for c in chars if _same_body(c.body, capture.body)]
    isempty(mine) && return absent(OtherBody)
    allequal(c -> c.body, mine) || throw(ArgumentError(
        "sensor_model: characterisations of $(length(unique(c -> c.body, mine))) bodies match $(capture.body) — state the body's serial"))
    entries = [c.entries[ex.iso] for c in mine if haskey(c.entries, ex.iso)]
    isempty(entries) && return absent(UnmatchedISO)
    black, ob = _pick([e.black for e in entries], FileBlack(), :black, ex.iso)
    noise, on = _pick([e.noise for e in entries], NoNoise(), :noise, ex.iso)
    defects, od = _pick([e.defects for e in entries], NoDefects(), :defects, ex.iso)
    return SensorModel(_frame_black(black, ex.exposure_time), noise, _frame_defects(defects, ex.exposure_time);
                       reason = Matched, specified_override = ob || on || od)
end

# ------------------------------------------------------------------------------
# What a model gives the one linearisation kernel (LinearRaw, linearview, photon_transfer)
# ------------------------------------------------------------------------------

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


# The per-colour-index black of the black slot: the RAW's own, a ChannelBlack's, a Dark's channel black.
_channel_black(::FileBlack, m::RawMosaic) = m.black
_channel_black(b::ChannelBlack, ::RawMosaic) = b.values
_channel_black(d::Dark, m::RawMosaic) = _channel_black(d.black, m)
# The per-photosite offset map that replaces the channel black (a Dark's), on the backend; `nothing` otherwise.
_offset_map(::BlackComponent, m, backend) = nothing
_offset_map(d::Dark, m::RawMosaic, backend) = _photosite_map(d.offset, m, backend, Float32, "the dark offset")

# The levels a model gives the linearisation of `m`: the four channels' black and span (`_bayer_levels` of the black
# slot's per-colour black) and, on the backend, the offset map (`nothing` without a Dark). With the absent model it is
# the mosaic's own levels and no map, bit for bit the RAW's linearisation.
_model_levels(m::RawMosaic, model::SensorModel, backend) =
    (_bayer_levels(m, _channel_black(model.black, m))..., _offset_map(model.black, m, backend))

# The defect slot as the kernel's mask over the visible raw, on the backend: `nothing` for NoDefects, a DefectList's
# sites scattered into a Bool matrix of its `size` (one work item per site), checked to cover the mosaic.
@kernel function _defect_scatter_kernel!(mask, @Const(x), @Const(y))
    i = @index(Global)
    @inbounds mask[y[i], x[i]] = true
end
_defect_mask(::NoDefects, m, backend) = nothing
function _defect_mask(d::DefectList, m::RawMosaic, backend)
    mask = KA.zeros(backend, Bool, d.size...)
    isempty(d.x) || OnBackend(backend)(_defect_scatter_kernel!)(mask, on_device(Int, backend, d.x), on_device(Int, backend, d.y);
                                                                ndrange = length(d.x))
    KA.synchronize(backend)
    return _photosite_map(mask, m, backend, Bool, "the defect list's sensor")
end

# A pixel value's variance law under the noise slot, as the kernel reads it per cell site k (D11): a value x of a plane
# of m photosites (spans d_k), times the gain g, has variance
#     g·max(x, 0)/(m²K)·Σ 1/d_k  +  g²/m²·Σ s_k/d_k²  +  (σ_rel·x)²
# with s_k the photosite's Gaussian variance: σ_r[c]² of its colour, or, under a Dark, its dark variance times
# (1 + 1/frames) (the read noise and dark current at that photosite, plus the error of the offset subtracted, a mean
# of `frames`). The Poisson term of a value below the black level is 0 (a mean count is never negative).
# PoissonGaussian fills the first two terms, RelativeNoise the third; NoNoise and ReadNoise define no variance (nothing:
# a plain image).
struct _VarianceLaw{D}
    inv_gain::Float64                  # 1/K, electrons per code inverted; 0 without a Poisson term
    σ²::NTuple{4, Float64}             # per cell site k, its colour's read-noise variance (codes²)
    dark_variance::D                   # a Dark's variance map on the backend, or nothing
    dark_scale::Float64                # 1 + 1/frames
    rel²::Float64                      # σ_rel², 0 without a relative term
end
Adapt.adapt_structure(to, l::_VarianceLaw) = _VarianceLaw(l.inv_gain, l.σ², Adapt.adapt(to, l.dark_variance), l.dark_scale, l.rel²)
_variance_law(::Union{NoNoise, ReadNoise}, black, m, backend) = nothing
_variance_law(n::PoissonGaussian, black::BlackComponent, m::RawMosaic, backend) =
    (cell = _cell_colors(m, "the variance law"); _VarianceLaw(1 / n.gain, ntuple(k -> n.σ[cell[k] + 1]^2, Val(4)),
                                                             _dark_variance(black, m, backend), _dark_scale(black), 0.0))
_variance_law(n::RelativeNoise, black, m, backend) = _VarianceLaw(0.0, (0.0, 0.0, 0.0, 0.0), nothing, 1.0, n.σ_rel^2)
_dark_variance(::BlackComponent, m, backend) = nothing
_dark_variance(d::Dark, m::RawMosaic, backend) = _photosite_map(d.variance, m, backend, Float32, "the dark variance")
_dark_scale(::BlackComponent) = 1.0
_dark_scale(d::Dark) = 1 + 1 / d.frames

# The Gaussian variance of cell site k's photosite (visible-raw index r, c).
@inline _site_variance(l::_VarianceLaw{Nothing}, k, r, c) = l.σ²[k]
@inline _site_variance(l::_VarianceLaw, k, r, c) = @inbounds Float64(l.dark_variance[r, c]) * l.dark_scale

# The element type of an uncertain image's covariance plane: a value's variance as a 1×1 isotropic covariance.
const _ValueCovariance = typeof(CovMat(ScaledIdentity{1}(0.0)))

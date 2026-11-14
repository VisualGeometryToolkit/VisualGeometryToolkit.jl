# =============================================================================
# vgio/sensor_characterisation/v1.1 ↔ VGC `SensorCharacterisation` — one sensor, per ISO its components
# =============================================================================
#
# {"schema": "vgio/sensor_characterisation/v1.1", <envelope>,
#  "device": {"body": {"make": "Sony", "model": "ILME-FX3", "serial": "5012345"}},
#  "entries": [
#    {"iso": 100,
#     "black": {"type": "channel", "values": [b0, b1, b2, b3], "std_error": [s0, s1, s2, s3], "source": "measured"}
#            | {"type": "dark", "black": {channel body}, "exposure_band": [0.001, 1.0], "frames": 16,
#               "size": {"width": W, "height": H},
#               "offset": {"file": "fx3.iso100.dark_offset.tiff", "sha256": "<hex>"},
#               "variance": {"file": "fx3.iso100.dark_variance.tiff", "sha256": "<hex>"}, "source": "measured"}
#            | {"type": "darks", "black": {channel body}, "source": "measured",
#               "darks": [{dark body, its rasters "fx3.iso100.band1.dark_offset.tiff", …}, …]},   (v1.1)
#     "noise": {"type": "read_noise", "sigma": [r0, r1, r2, r3], "std_error": [s0, …], "source": "specified"}
#            | {"type": "poisson_gaussian", "gain": 2.1, "sigma": [r0, …], "std_error": {"gain": 0.01, "sigma": [s0, …]},
#               "source": "estimated"}
#            | {"type": "relative", "sigma_rel": 0.01, "source": "specified"},
#     "defects": {"size": {"width": W, "height": H}, "sites": [[x, y, onset], …], "source": "measured"}},
#    …]}
#
# One SensorCharacterisation (VisualGeometryToolkitCore raw/sensor_model.jl): the
# key is the body's identity, `device.body` (the ONE device-identity encoding, a
# capture's and a calibration record's; a sensor's description has no lens), and
# per ISO (each once) its three slots. A slot holds ONE component body, its kind
# by `type` (the black slot's ChannelBlack "channel", Dark "dark" or Darks "darks", the noise
# slot's ReadNoise "read_noise", PoissonGaussian "poisson_gaussian" or
# RelativeNoise "relative"; the defect slot has one kind, DefectList), each with
# its `source` (measured | estimated | specified) and, where stated, its standard
# error in the value's units and shape. An ABSENT slot is absent from the entry
# and loads as its absent type (FileBlack, NoNoise, NoDefects): the RAW's own
# black, no noise stated, no defect. A Dark's own `black` is its channel black,
# absent when it is the RAW's. A Darks (v1.1: the darks of several exposures at one
# ISO) holds its shared channel black (absent when it is the RAW's) and its Darks in
# increasing band, each a dark body with its own rasters (`<stem>.band<k>.…`, k the
# band's place); the bands are disjoint.
#
# VERSIONS: v1.1 (IO 0.10.2) added the "darks" body; a v1.0 file (IO 0.10.1's, at
# most one Dark per ISO) is a v1.1 file as it stands, read as written.
#
# Per CFA colour index 0–3 (LibRaw's): `values[c + 1]`, codes. Times — a dark's
# exposure band, a defect's onset (the shortest exposure at which it is hot) —
# are in `units.time` (seconds). A defect site is `[x, y, onset]`: x the COLUMN
# and y the ROW of the visible raw, 1-based (`raw[y, x]`, `conventions.index_base`),
# inside `size`, each photosite once.
#
# THE ARTIFACTS: a dark's per-photosite offset and variance are rasters beside
# the record, the stack's one bulk-array binding, the raster PAYLOAD BODY
# `{file, sha256}` + a float32 TIFF of the raster profile (raster_schema.jl),
# over the visible raw (`size`), row 1 on top. The reader refuses a file whose
# digest is not the record's, before it is decoded.

# A present component's source: one of Core's `SensorSource` values but `file` (the RAW's own black's).
const _SENSOR_SOURCES = [string(s) for s in instances(VgtkCore.SensorSource.T) if s !== VgtkCore.SensorSource.file]
const _SENSOR_SOURCE = _Named(in(_SENSOR_SOURCES), "a present component's source ($(join(_SENSOR_SOURCES, ", ")))")
const _FOUR = _Vec(_Real(), 4)

# A slot's component body, its kind by `type`: one of `alts` (tag => the body's table, its `type` included).
struct _ByType <: _Spec
    what::String
    alts::Vector{Pair{String, _Spec}}
end
function _validate(s::_ByType, x, file, at)
    tags = join(first.(s.alts), ", ")
    x isa AbstractDict && haskey(x, :type) || _refuse(file, at, "$(s.what) is an object with a `type` ($(tags))")
    i = findfirst(a -> a.first == x[:type], s.alts)
    i === nothing && _refuse(file, _at(at, :type), "$(repr(x[:type])) is not $(s.what)'s type ($(tags))")
    return _validate(s.alts[i].second, x, file, at)
end

const _CHANNEL_BLACK = _Obj(Pair{Symbol, _Spec}[:type => _Tok("channel"), :values => _FOUR, :source => _SENSOR_SOURCE],
                            Pair{Symbol, _Spec}[:std_error => _FOUR])
const _DARK_BODY = _Obj(Pair{Symbol, _Spec}[:type => _Tok("dark"), :exposure_band => _Vec(_Real(), 2), :frames => _PosInt(),
                                             :size => _SIZE_INT, :offset => _RASTER_PAYLOAD, :variance => _RASTER_PAYLOAD,
                                             :source => _SENSOR_SOURCE],
                         Pair{Symbol, _Spec}[:black => _CHANNEL_BLACK])
const _BLACK_COMPONENT = _ByType("a black component", Pair{String, _Spec}[
    "channel" => _CHANNEL_BLACK,
    "dark" => _DARK_BODY,
    "darks" => _Obj(Pair{Symbol, _Spec}[:type => _Tok("darks"), :darks => _Vec(_DARK_BODY), :source => _SENSOR_SOURCE],
                    Pair{Symbol, _Spec}[:black => _CHANNEL_BLACK]),
])
const _NOISE_COMPONENT = _ByType("a noise component", Pair{String, _Spec}[
    "read_noise" => _Obj(Pair{Symbol, _Spec}[:type => _Tok("read_noise"), :sigma => _FOUR, :source => _SENSOR_SOURCE],
                         Pair{Symbol, _Spec}[:std_error => _FOUR]),
    "poisson_gaussian" => _Obj(Pair{Symbol, _Spec}[:type => _Tok("poisson_gaussian"), :gain => _Real(), :sigma => _FOUR,
                                                   :source => _SENSOR_SOURCE],
                               Pair{Symbol, _Spec}[:std_error => _Obj(:gain => _Real(), :sigma => _FOUR)]),
    "relative" => _Obj(:type => _Tok("relative"), :sigma_rel => _Real(), :source => _SENSOR_SOURCE),
])
const _DEFECT_COMPONENT = _Obj(:size => _SIZE_INT, :sites => _Vec(_Vec(_Real(), 3)), :source => _SENSOR_SOURCE)

# The rules the tables cannot state: a dark's band and frames; a defect site inside the raw, once, its onset positive;
# a gain and σ_rel positive; each ISO positive and once.
function _dark_problem(b)
    b[:type] == "darks" && return _darks_problem(b[:darks])
    b[:type] == "dark" || return nothing
    lo, hi = b[:exposure_band]
    0 < lo <= hi || return "a dark's exposure band runs from its shortest to its longest time, both positive"
    b[:frames] >= 2 && return nothing
    return "a dark's temporal variance needs at least 2 frames"
end
function _darks_problem(ds)
    isempty(ds) && return "a darks body holds at least one dark"
    for d in ds
        p = _dark_problem(d); p === nothing || return p
    end
    bands = sort!([(Float64(d[:exposure_band][1]), Float64(d[:exposure_band][2])) for d in ds])
    all(k -> bands[k][1] > bands[k - 1][2], 2:length(bands)) || return "a darks body's exposure bands are disjoint"
    return nothing
end
function _noise_problem(n)
    n[:type] == "poisson_gaussian" && !(n[:gain] > 0) && return "a Poisson–Gaussian gain (electrons per code) is positive"
    n[:type] == "relative" && !(n[:sigma_rel] > 0) && return "a relative noise's sigma_rel is positive"
    return nothing
end
function _defects_problem(d)
    W, H = d[:size][:width], d[:size][:height]
    seen = Set{Tuple{Int, Int}}()
    for (x, y, t) in d[:sites]
        isinteger(x) && isinteger(y) && 1 <= x <= W && 1 <= y <= H ||
            return "a defect site [x, y, onset] is a photosite of the visible raw: 1 ≤ x ≤ $(W), 1 ≤ y ≤ $(H) (1-based, x the column)"
        t > 0 || return "a defect's onset (the shortest exposure at which it is hot) is positive"
        (Int(x), Int(y)) in seen && return "a defect site is listed once"
        push!(seen, (Int(x), Int(y)))
    end
    return nothing
end
function _entry_problem(e)
    e[:iso] > 0 || return "an entry's ISO is positive"
    haskey(e, :black) && (p = _dark_problem(e[:black]); p === nothing || return p)
    haskey(e, :noise) && (p = _noise_problem(e[:noise]); p === nothing || return p)
    haskey(e, :defects) && (p = _defects_problem(e[:defects]); p === nothing || return p)
    return nothing
end
_entries_problem(x) = allunique(Float64(e[:iso]) for e in x[:entries]) ? nothing : "an ISO has one entry"

const _SENSOR_ENTRY = _CheckedSpec(
    _Obj(Pair{Symbol, _Spec}[:iso => _Real()],
         Pair{Symbol, _Spec}[:black => _BLACK_COMPONENT, :noise => _NOISE_COMPONENT, :defects => _DEFECT_COMPONENT]),
    _entry_problem;
    rule = "the ISO is positive; a dark's exposure band runs from its shortest to its longest time, both positive, over at " *
           "least 2 frames, and a darks body's bands (at least one) are disjoint; a gain and a sigma_rel are positive; a defect site is a photosite of the visible raw (1-based, " *
           "x the column), listed once, its onset positive",
)
const _SENSOR_CHARACTERISATION_SPEC = _kind_spec(
    "sensor_characterisation",
    Pair{Symbol, _Spec}[:device => _Obj(:body => _DEVICE_IDENTITY), :entries => _Vec(_SENSOR_ENTRY)];
    check = _entries_problem,
    rule = "each ISO has one entry",
)

# ----------------------------------------------------------------------------- writing

_four(t::NTuple{4, Float64}) = collect(t)
_std_error_json(r::NamedTuple, ::Nothing) = r
_std_error_json(r::NamedTuple, se::NTuple{4, Float64}) = merge(r, (std_error = _four(se),))
_std_error_json(r::NamedTuple, se::NamedTuple) = merge(r, (std_error = (gain = se.gain, sigma = _four(se.σ)),))
_seconds(t) = ustrip(parse_unit(_TIME_UNIT), t)
_iso_name(iso::Float64) = isinteger(iso) ? string(Int(iso)) : string(iso)

# A slot's body (`nothing`: the absent type, absent from the entry); `files` names a dark's rasters.
_component_json(::Union{VgtkCore.FileBlack, VgtkCore.NoNoise, VgtkCore.NoDefects}, files) = nothing
_component_json(c::VgtkCore.ChannelBlack, files) =
    _std_error_json((type = "channel", values = _four(c.values), source = string(c.source)), c.std_error)
function _component_json(d::VgtkCore.Dark, files)
    r = (type = "dark", exposure_band = Float64[_seconds(d.exposure_band[1]), _seconds(d.exposure_band[2])], frames = d.frames,
         size = (width = size(d.offset, 2), height = size(d.offset, 1)),
         offset = _raster_payload_json(files.dir, "$(files.stem).dark_offset.tiff", Array(d.offset)),
         variance = _raster_payload_json(files.dir, "$(files.stem).dark_variance.tiff", Array(d.variance)), source = string(d.source))
    b = _component_json(d.black, files)
    return b === nothing ? r : merge((type = r.type, black = b), r)
end
function _component_json(d::VgtkCore.Darks, files)
    r = (type = "darks", darks = [_component_json(x, (dir = files.dir, stem = "$(files.stem).band$(k)")) for (k, x) in enumerate(d.darks)],
         source = string(d.source))
    b = _component_json(d.black, files)
    return b === nothing ? r : merge((type = r.type, black = b), r)
end
_component_json(c::VgtkCore.ReadNoise, files) = _std_error_json((type = "read_noise", sigma = _four(c.σ), source = string(c.source)), c.std_error)
_component_json(c::VgtkCore.PoissonGaussian, files) =
    _std_error_json((type = "poisson_gaussian", gain = c.gain, sigma = _four(c.σ), source = string(c.source)), c.std_error)
_component_json(c::VgtkCore.RelativeNoise, files) = (type = "relative", sigma_rel = c.σ_rel, source = string(c.source))
_component_json(d::VgtkCore.DefectList, files) =
    (size = (width = d.size[2], height = d.size[1]),
     sites = [Any[x, y, _seconds(t)] for (x, y, t) in zip(Array(d.x), Array(d.y), Array(d.onset))], source = string(d.source))

"""
    save(path, characterisation::SensorCharacterisation; provenance = (;))

Write a VGC `SensorCharacterisation` as a `vgio/sensor_characterisation/v1.1`
file: the body's identity (`device.body`) and one entry per ISO (in increasing
ISO), each present slot as its component's body with its source and standard
error, an absent slot (`FileBlack`, `NoNoise`, `NoDefects`) absent. A `Dark`'s
offset and variance maps are written beside `path` as rasters, the raster payload
body (`<stem>.iso<ISO>.dark_offset.tiff`, `<stem>.iso<ISO>.dark_variance.tiff`,
float32 TIFFs of the raster profile, row 1 on top) bound by their SHA-256; a
`Darks`' maps one pair per band, `<stem>.iso<ISO>.band<k>.dark_offset.tiff` and
`….dark_variance.tiff`, `k` the band's place in increasing exposure.
"""
function save(path::AbstractString, c::SensorCharacterisation; provenance::NamedTuple = (;))
    dir, stem = dirname(path), first(splitext(basename(path)))
    entries = map(sort!(collect(keys(c.entries)))) do iso
        e = c.entries[iso]
        files = (dir = dir, stem = "$(stem).iso$(_iso_name(iso))")
        slots = ((k, _component_json(getfield(e, k), files)) for k in (:black, :noise, :defects))
        merge((iso = iso,), NamedTuple(k => b for (k, b) in slots if b !== nothing))
    end
    _write_kind(path, "sensor_characterisation", (device = (body = _identity_json(c.body),), entries = entries), provenance;
                style = VgioStyle())
    return nothing
end

# ----------------------------------------------------------------------------- reading

_four_from(x) = ntuple(i -> Float64(x[i]::Real), 4)
_source_from(x) = only(s for s in instances(VgtkCore.SensorSource.T) if string(s) == x::String)
_std_error_from(b, f) = haskey(b, :std_error) ? f(b.std_error) : nothing
_channel_black_from(b) = VgtkCore.ChannelBlack(_four_from(b.values), _std_error_from(b, _four_from); source = _source_from(b.source))
function _black_from(path, b)
    b.type == "channel" && return _channel_black_from(b)
    b.type == "darks" && return VgtkCore.Darks(haskey(b, :black) ? _channel_black_from(b.black) : VgtkCore.FileBlack(),
                                               [_black_from(path, x) for x in b.darks]; source = _source_from(b.source))
    sec = parse_unit(_TIME_UNIT)
    W, H = Int(b.size.width::Integer), Int(b.size.height::Integer)
    return VgtkCore.Dark(haskey(b, :black) ? _channel_black_from(b.black) : VgtkCore.FileBlack(),
                         _raster_payload_from_json(path, b.offset, Float32, W, H), _raster_payload_from_json(path, b.variance, Float32, W, H),
                         (Float64(b.exposure_band[1]::Real) * sec, Float64(b.exposure_band[2]::Real) * sec), Int(b.frames::Integer);
                         source = _source_from(b.source))
end
function _noise_from(n)
    s = _source_from(n.source)
    n.type == "read_noise" && return VgtkCore.ReadNoise(_four_from(n.sigma), _std_error_from(n, _four_from); source = s)
    n.type == "relative" && return VgtkCore.RelativeNoise(Float64(n.sigma_rel::Real); source = s)
    se = _std_error_from(n, e -> (gain = Float64(e.gain::Real), σ = _four_from(e.sigma)))
    return VgtkCore.PoissonGaussian(Float64(n.gain::Real), _four_from(n.sigma), se; source = s)
end
function _defects_from(d)
    sec = parse_unit(_TIME_UNIT)
    sites = d.sites::Vector{Any}
    return VgtkCore.DefectList(Int[Int(s[1]::Real) for s in sites], Int[Int(s[2]::Real) for s in sites],
                               [Float64(s[3]::Real) * sec for s in sites];
                               size = (Int(d.size.height::Integer), Int(d.size.width::Integer)), source = _source_from(d.source))
end

"""
    load(SensorCharacterisation, path) -> SensorCharacterisation

Read a `vgio/sensor_characterisation/v1.x` file ([`save`](@ref)'s; a v1.0 file as written): the
`SensorCharacterisation` exactly as written — per ISO each present slot's
component with its source and standard error, an absent slot its absent type
(`FileBlack`, `NoNoise`, `NoDefects`); a `Dark`'s maps (`Matrix{Float32}`, each of a `Darks`' too) read
from the rasters beside it, each checked against its SHA-256, the raster profile
and its size.
"""
function load(::Type{SensorCharacterisation}, path::AbstractString)
    d = _read_store(path, "sensor_characterisation", _SENSOR_CHARACTERISATION_SPEC)
    entries = map(d.entries::Vector{Any}) do e
        slots = (haskey(e, :black) ? (black = _black_from(path, e.black),) : (;),
                 haskey(e, :noise) ? (noise = _noise_from(e.noise),) : (;),
                 haskey(e, :defects) ? (defects = _defects_from(e.defects),) : (;))
        Float64(e.iso::Real) => merge(slots[1], slots[2], slots[3])
    end
    return SensorCharacterisation(_identity_from_json(d.device.body); entries)
end

# =============================================================================
# vgio/sensor_characterization/v3.0 ↔ VGC `SensorCharacterization`
# =============================================================================
#
# {"schema": "vgio/sensor_characterization/v3.0", <envelope>,
#  "device": {"body": {"make": "Sony", "model": "ILME-FX3", "serial": "5012345"}},
#  "entries": [
#    {"iso": 100,
#     "black": {"type": "channel", "values": [b0, b1, b2, b3], "std_error": [s0, s1, s2, s3], "source": "measured"}
#            | {"type": "dark", "black": {channel body}, "exposure_band": [0.001, 1.0], "frames": 16,
#               "size": {"width": W, "height": H},
#               "offset": {"file": "fx3.iso100.dark_offset.tiff", "sha256": "<hex>"},
#               "variance": {"file": "fx3.iso100.dark_variance.tiff", "sha256": "<hex>"}, "source": "measured"}
#            | {"type": "darks", "black": {channel body}, "source": "measured",
#               "darks": [{dark body, its rasters "fx3.iso100.band1.dark_offset.tiff", …}, …]},   (v2)
#     "noise": {"type": "read_noise", "sigma": [r0, r1, r2, r3], "std_error": [s0, …], "source": "specified"}
#            | {"type": "poisson_gaussian", "gain": 2.1, "sigma": [r0, …], "std_error": {"gain": 0.01, "sigma": [s0, …]},
#               "source": "estimated"}
#            | {"type": "relative", "sigma_rel": 0.01, "source": "specified"},
#     "defects": {"size": {"width": W, "height": H}, "sites": [[x, y, onset], …], "source": "measured"}},
#    …]}
#
# One SensorCharacterization (VisualGeometryToolkitCore raw/sensor_model.jl): the
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
# absent when it is the RAW's. A Darks (v2: the darks of several exposures at one
# ISO) holds its shared channel black (absent when it is the RAW's) and its Darks in
# increasing band, each a dark body with its own rasters (`<stem>.band<k>.…`, k the
# band's place); the bands are disjoint.
#
# VERSIONS: v3.0 (IO 0.12) is the kind's American name, `sensor_characterization`
# (owner decision 11, 2026-11-23: American in identifiers); its body is v2.0's as it
# stands. The kind's British name (v1.0 and v2.0) is read through its steps: v1.0
# (IO 0.10.1) → v2.0, which only renames the schema (v2.0, IO 0.10.2, added the
# "darks" body, a new value of the black slot's `type`, so a major version), and
# v2.0 → v3.0, which renames the kind. Only v3.0 is written: the earlier writer's
# v1.0 for a record without a Darks (owner ruling (a), IO 0.10.2) ends with the
# rename, as no reader of the former kind reads the new one. `vgio_migrate`
# rewrites a former-kind file without a dark map (`_sensor_characterization_file_form`); one that
# holds a dark map is read in place (its rasters are named by its record, and the
# staging directory a migration writes into is removed).
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

const _CHANNEL_BLACK = _Obj(Pair{Symbol, _Spec}[:type => _Tok("channel"), :values => _FOUR, :source => _SENSOR_SOURCE],
                            Pair{Symbol, _Spec}[:std_error => _FOUR])
const _DARK_BODY = _Obj(Pair{Symbol, _Spec}[:type => _Tok("dark"), :exposure_band => _Vec(_Real(), 2), :frames => _PosInt(),
                                             :size => _SIZE_INT, :offset => _RASTER_PAYLOAD, :variance => _RASTER_PAYLOAD,
                                             :source => _SENSOR_SOURCE],
                         Pair{Symbol, _Spec}[:black => _CHANNEL_BLACK])
const _BLACK_COMPONENT = _ByType("a black component", Pair{String, _Spec}[
    "channel" => _CHANNEL_BLACK, "dark" => _DARK_BODY,
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
    b[:type] == "darks" && (all(d -> d[:source] == b[:source], b[:darks]) || return "a darks body's darks are of its source")
    b[:type] == "darks" && return _darks_problem(b[:darks])
    b[:type] == "dark" || return nothing
    lo, hi = b[:exposure_band]
    0 < lo <= hi || return "a dark's exposure band runs from its shortest to its longest time, both positive"
    b[:frames] >= 2 && return nothing
    return "a dark's temporal variance needs at least 2 frames"
end
# (the rule is VisualGeometryToolkitCore's `Darks` constructor's, stated here so `validate` refuses the file without one)
function _darks_problem(ds)
    isempty(ds) && return "a darks body holds at least one dark"
    for d in ds
        p = _dark_problem(d); p === nothing || return p
    end
    allequal(d -> (d[:size][:width], d[:size][:height]), ds) || return "a darks body's darks are of one size"
    # listed in increasing band, as the writer writes them (Core's `Darks` stores them so): each band after the last
    bands = [(Float64(d[:exposure_band][1]), Float64(d[:exposure_band][2])) for d in ds]
    all(k -> bands[k][1] > bands[k - 1][2], 2:length(bands)) ||
        return "a darks body's exposure bands are disjoint and listed in increasing exposure, got $(join(bands, ", "))"
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

# An entry: its ISO and its present slots.
const _SENSOR_ENTRY = _CheckedSpec(
    _Obj(Pair{Symbol, _Spec}[:iso => _Real()],
         Pair{Symbol, _Spec}[:black => _BLACK_COMPONENT, :noise => _NOISE_COMPONENT, :defects => _DEFECT_COMPONENT]),
    _entry_problem;
    rule = "the ISO is positive; a dark's exposure band runs from its shortest to its longest time, both positive, over at " *
           "least 2 frames, and a darks body's darks (at least one) of its source and of one size, their bands disjoint, in " *
           "increasing exposure; a gain and a sigma_rel are positive; a defect site is a photosite of the visible raw (1-based, " *
           "x the column), listed once, its onset positive",
)
const _SENSOR_CHARACTERIZATION_SPEC = _kind_spec(
    "sensor_characterization",
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
    return _with_black(r, d.black, files)
end
_component_json(d::VgtkCore.Darks, files) =
    _with_black((type = "darks", darks = [_component_json(x, (dir = files.dir, stem = "$(files.stem).band$(k)")) for (k, x) in enumerate(d.darks)],
                 source = string(d.source)), d.black, files)
# A dark's or a darks' body with its channel black after its type (absent when it is the RAW's).
_with_black(r, black, files) = (b = _component_json(black, files); b === nothing ? r : merge((type = r.type, black = b), r))
_component_json(c::VgtkCore.ReadNoise, files) = _std_error_json((type = "read_noise", sigma = _four(c.σ), source = string(c.source)), c.std_error)
_component_json(c::VgtkCore.PoissonGaussian, files) =
    _std_error_json((type = "poisson_gaussian", gain = c.gain, sigma = _four(c.σ), source = string(c.source)), c.std_error)
_component_json(c::VgtkCore.RelativeNoise, files) = (type = "relative", sigma_rel = c.σ_rel, source = string(c.source))
_component_json(d::VgtkCore.DefectList, files) =
    (size = (width = d.size[2], height = d.size[1]),
     sites = [Any[x, y, _seconds(t)] for (x, y, t) in zip(Array(d.x), Array(d.y), Array(d.onset))], source = string(d.source))

"""
    save(path, characterization::SensorCharacterization; provenance = (;))

Write a VGC `SensorCharacterization` as a `vgio/sensor_characterization/v3.0` file: the
body's identity (`device.body`) and one entry per ISO (in increasing ISO), each present slot as its component's body with its source and standard
error, an absent slot (`FileBlack`, `NoNoise`, `NoDefects`) absent. A `Dark`'s
offset and variance maps are written beside `path` as rasters, the raster payload
body (`<stem>.iso<ISO>.dark_offset.tiff`, `<stem>.iso<ISO>.dark_variance.tiff`,
float32 TIFFs of the raster profile, row 1 on top) bound by their SHA-256; a
`Darks`' maps one pair per band, `<stem>.iso<ISO>.band<k>.dark_offset.tiff` and
`….dark_variance.tiff`, `k` the band's place in increasing exposure.
"""
function save(path::AbstractString, c::SensorCharacterization; provenance::NamedTuple = (;))
    dir, stem = dirname(path), first(splitext(basename(path)))
    entries = map(sort!(collect(keys(c.entries)))) do iso
        e = c.entries[iso]
        files = (dir = dir, stem = "$(stem).iso$(_iso_name(iso))")
        slots = ((k, _component_json(getfield(e, k), files)) for k in (:black, :noise, :defects))
        merge((iso = iso,), NamedTuple(k => b for (k, b) in slots if b !== nothing))
    end
    _write_kind(path, "sensor_characterization", (device = (body = _identity_json(c.body),), entries = entries), provenance;
                style = VgioStyle())
    return nothing
end

# ----------------------------------------------------------------------------- reading

# Every component and the characterization are built by keyword through Core's one table builder,
# `construct_from_table`, never by a constructor call here (check_duplication's `:sensor_component`): a body's keys are
# its fields' TOML spellings (`sigma` the field `σ`), the children built first (a dark's or a darks' channel black, a
# darks' Darks, an entry's slots), an absent black omitted (the keyword form's default is the RAW's own). A body keeps
# only the keys its spec declares: a newer minor's key, which the validator tolerates, is not a keyword.

_four_from(x) = ntuple(i -> Float64(x[i]::Real), 4)
_source_from(x) = only(s for s in instances(VgtkCore.SensorSource.T) if string(s) == x::String)
_declared(spec, b) = Dict{String, Any}(String(k) => v for (k, v) in b if _declares(spec, Symbol(k)))
_alt(s::_ByType, type) = last(only(a for a in s.alts if first(a) == type))

# One value of a body in its field's form (`construct_from_table`'s `from_toml`); `path` locates a dark's rasters, `b`
# is the body (a dark's `size`).
_sensor_value(path, b, ::Type, ::Val{:source}, v) = _source_from(v)
_sensor_value(path, b, ::Type, ::Union{Val{:values}, Val{:σ}, Val{:std_error}}, v) = _four_from(v)
_sensor_value(path, b, ::Type{<:VgtkCore.PoissonGaussian}, ::Val{:std_error}, v) = (gain = Float64(v.gain::Real), σ = _four_from(v.sigma))
_sensor_value(path, b, ::Type, ::Union{Val{:gain}, Val{:σ_rel}}, v) = Float64(v::Real)
_sensor_value(path, b, ::Type, ::Val{:frames}, v) = Int(v::Integer)
_sensor_value(path, b, ::Type, ::Val{:exposure_band}, v) = (sec = parse_unit(_TIME_UNIT); (Float64(v[1]::Real) * sec, Float64(v[2]::Real) * sec))
_sensor_value(path, b, ::Type, ::Union{Val{:offset}, Val{:variance}}, v) =
    _raster_payload_from_json(path, v, Float32, Int(b.size.width::Integer), Int(b.size.height::Integer))
_sensor_value(path, b, ::Type, ::Val{:black}, v) = _black_from(path, v)
_sensor_value(path, b, ::Type, ::Val{:darks}, v) = [_black_from(path, x) for x in v]
_sensor_value(path, b, ::Type, ::Union{Val{:x}, Val{:y}}, v) = Int[Int(s::Real) for s in v]
_sensor_value(path, b, ::Type, ::Val{:onset}, v) = (sec = parse_unit(_TIME_UNIT); [Float64(s::Real) * sec for s in v])
_sensor_value(path, b, ::Type, ::Val{:size}, v) = (Int(v.height::Integer), Int(v.width::Integer))
_sensor_value(path, b, ::Type, ::Val{:body}, v) = _identity_from_json(v)
_sensor_value(path, b, ::Type, ::Val{:entries}, v) = [_entry_from(path, e) for e in v::Vector{Any}]

# A component of type `T` from its body `b` (its spec `spec`), the keys in `skip` the reader's alone.
_component_from(T, path, spec, b; skip = ("type",)) =
    VgtkCore.construct_from_table(T, _declared(spec, b); skip, section = "$(path): $(nameof(T))",
                                  from_toml = (T, f, v) -> _sensor_value(path, b, T, f, v))
function _black_from(path, b)
    b.type == "channel" && return _component_from(VgtkCore.ChannelBlack, path, _CHANNEL_BLACK, b)
    b.type == "darks" && return _component_from(VgtkCore.Darks, path, _alt(_BLACK_COMPONENT, "darks"), b)
    return _component_from(VgtkCore.Dark, path, _DARK_BODY, b; skip = ("type", "size"))   # `size`: the rasters' (`b`'s)
end
const _NOISE_TYPES = Dict("read_noise" => VgtkCore.ReadNoise, "poisson_gaussian" => VgtkCore.PoissonGaussian,
                          "relative" => VgtkCore.RelativeNoise)
_noise_from(path, n) = _component_from(_NOISE_TYPES[n.type], path, _alt(_NOISE_COMPONENT, n.type), n)
# A defect body's sites `[x, y, onset]` are the list's three fields, column by column.
function _defects_from(path, d)
    sites = d.sites::Vector{Any}
    t = Dict{String, Any}("x" => Any[s[1] for s in sites], "y" => Any[s[2] for s in sites], "onset" => Any[s[3] for s in sites],
                          "size" => d.size, "source" => d.source)
    return VgtkCore.construct_from_table(VgtkCore.DefectList, t; section = "$(path): DefectList",
                                         from_toml = (T, f, v) -> _sensor_value(path, d, T, f, v))
end
# An entry: its ISO and its present slots (an absent slot is the characterization's absent type).
function _entry_from(path, e)
    slots = (haskey(e, :black) ? (black = _black_from(path, e.black),) : (;),
             haskey(e, :noise) ? (noise = _noise_from(path, e.noise),) : (;),
             haskey(e, :defects) ? (defects = _defects_from(path, e.defects),) : (;))
    return Float64(e.iso::Real) => merge(slots[1], slots[2], slots[3])
end

"""
    load(SensorCharacterization, path) -> SensorCharacterization

Read a `vgio/sensor_characterization/v3.0` file ([`save`](@ref)'s; a file of
the kind's British name, v1.0 or v2.0, through its steps): the
`SensorCharacterization` exactly as written — per ISO each present slot's
component with its source and standard error, an absent slot its absent type
(`FileBlack`, `NoNoise`, `NoDefects`); a `Dark`'s maps (`Matrix{Float32}`, each of a `Darks`' too) read
from the rasters beside it, each checked against its SHA-256, the raster profile
and its size. Each component, and the characterization, is built by keyword with
VisualGeometryToolkitCore's `construct_from_table` (the children first).
"""
function load(::Type{SensorCharacterization}, path::AbstractString)
    d = _read_store(path, "sensor_characterization", _SENSOR_CHARACTERIZATION_SPEC)
    t = Dict{String, Any}("body" => d.device.body, "entries" => d.entries)
    return VgtkCore.construct_from_table(SensorCharacterization, t; section = path,
                                         from_toml = (T, f, v) -> _sensor_value(path, d, T, f, v))
end

# ----------------------------------------------------------------------------- the steps of the former kind

# v1.0 (IO 0.10.1) → v2.0, on read: the body is unchanged (v2 only added the "darks" body), so the step renames the
# schema and states the current generator.
function _sensor_characterization_v1_to_v2(t::AbstractDict)
    for e in get(t, "entries", ())
        e isa AbstractDict && haskey(e, "black") || continue
        b = e["black"]
        b isa AbstractDict || throw(ArgumentError("a v1.0 entry's black is an object with a `type` (channel or dark), got $(repr(b))"))
        # v1.0 has no darks body: a file of that version holding one is no v1.0 file (an IO 0.10.1 reader refuses it)
        get(b, "type", nothing) == "darks" && throw(ArgumentError("a v1.0 entry's black is a channel black or a dark: a darks body is v2.0's"))
    end
    n = Dict{String, Any}(String(k) => v for (k, v) in t)
    n["schema"] = _schema("sensor_characterisation", (2, 0))   # v2.0, not the current version: a later step runs after it
    n["generator"] = _lower(_generator_json())
    return n
end
# The step of the major, keyed for every v1 minor read: "v1.0" (IO 0.10.1's) and "v1", the same version by
# `_split_schema` (a schema without a minor is minor 0).
for s in ("vgio/sensor_characterisation/v1", "vgio/sensor_characterisation/v1.0")
    @eval migrate_schema(::Val{Symbol($s)}, t::AbstractDict) = _sensor_characterization_v1_to_v2(t)
end

# v2.0 (IO 0.10.2) → v3.0, on read: the kind's American name (decision 11, IO 0.12); the body is unchanged.
function _sensor_characterization_v2_to_v3(t::AbstractDict)
    n = Dict{String, Any}(String(k) => v for (k, v) in t)
    n["schema"] = _schema("sensor_characterization")
    n["generator"] = _lower(_generator_json())
    return n
end
for s in ("vgio/sensor_characterisation/v2", "vgio/sensor_characterisation/v2.0")
    @eval migrate_schema(::Val{Symbol($s)}, t::AbstractDict) = _sensor_characterization_v2_to_v3(t)
end

# The file form (`vgio_migrate`) of a former-kind file: read through the steps, written by the current writer, read
# back and compared. A record that holds a dark map is refused: its rasters are named by the record and would be
# written into the staging directory, which is removed once the file is renamed into place (IO 0.10.3). Such a file
# reads as it stands; `save` writes it anew, with its rasters, where its caller puts it.
function _sensor_characterization_file_form(src::AbstractString, dst::AbstractString)
    c = load(SensorCharacterization, src)
    any(e -> e.black isa Union{VgtkCore.Dark, VgtkCore.Darks}, values(c.entries)) && throw(ArgumentError(
        "$(src): a sensor characterisation holding a dark map is not rewritten by vgio_migrate (its rasters are named " *
        "by the record); it reads as it stands, and save(path, load(SensorCharacterization, src)) writes it anew with its rasters"))
    save(dst, c)
    load(SensorCharacterization, dst) == c || throw(ArgumentError("$(src): the migrated characterisation does not read back as the original"))
    return (; to = _schema("sensor_characterization"))
end
for s in ("vgio/sensor_characterisation/v1", "vgio/sensor_characterisation/v1.0", "vgio/sensor_characterisation/v2", "vgio/sensor_characterisation/v2.0")
    @eval migrate_schema(::Val{Symbol($s)}, src::AbstractString, dst::AbstractString) = _sensor_characterization_file_form(src, dst)
end

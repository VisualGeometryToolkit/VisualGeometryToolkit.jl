# =============================================================================
# vgio/sensor_noise/v1.0 ↔ VGC `SensorNoise` — a body's Poisson–Gaussian noise at one ISO, per-field provenance
# =============================================================================
#
# {"schema": "vgio/sensor_noise/v1.0", <envelope>,
#  "device": {"body": {"make": "Sony", "model": "ILME-FX3", "serial": "5012345"}},
#  "iso": 100.0,
#  "electrons_per_dn": {"value": 2.1, "source": "measured", "std_error": 0.01},        (each field optional)
#  "read_noise": {"value": [r0, r1, r2, r3], "source": "estimated", "std_error": [s0, s1, s2, s3]},
#  "black": {"value": [b0, b1, b2, b3], "source": "file"},
#  "dark": {"source": "measured", "exposure_band": [0.001, 1.0], "frames": 16, "size": {"width": W, "height": H},
#           "offset": {"file": "noise.dark_offset.tiff", "sha256": "<hex>"},
#           "variance": {"file": "noise.dark_variance.tiff", "sha256": "<hex>"}},
#  "defects": {"source": "measured", "size": {"width": W, "height": H}, "mask": {"file": "noise.defects.tiff", "sha256": "<hex>"}},
#  "reason": "…"}                                                                       (iff the noise is unknown)
#
# One SensorNoise (VisualGeometryToolkitCore raw/sensor_noise.jl): the key is the
# body's identity, `device.body` — the ONE device-identity encoding (a capture's
# and a calibration record's `device`; a sensor's noise has no lens). Every field
# is a value with its SOURCE (measured | estimated | file) and, where stated, its
# standard error in the value's units and shape, or absent (`nothing`: not
# known); `reason` says why the noise is unknown, and only then. Per CFA colour
# index 0–3 (LibRaw's), codes.
#
# THE ARTIFACTS: a master dark's per-photosite offset and variance and a defect
# mask are rasters beside the record, each the stack's one bulk-array binding,
# the raster PAYLOAD BODY `{file, sha256}` + a TIFF of the raster profile
# (raster_schema.jl): over the visible raw (`size`), row 1 on top; the maps
# float32, the mask uint8 (1 a defect, 0 not). The reader refuses a file whose
# digest is not the record's, before it is decoded.
# Exposure times are in `units.time` (seconds).

const _SOURCE = _Named(in(String.(VgtkCore.NOISE_SOURCES)), "a value's source ($(join(VgtkCore.NOISE_SOURCES, ", ")))")
_noise_field_spec(value::_Spec) = _Obj(Pair{Symbol, _Spec}[:value => value, :source => _SOURCE], Pair{Symbol, _Spec}[:std_error => value])
const _FOUR = _Vec(_Real(), 4)

function _sensor_noise_check(x)
    known = haskey(x, :electrons_per_dn) && haskey(x, :read_noise)
    known == !haskey(x, :reason) || return "`reason` states why the noise (electrons_per_dn, read_noise) is unknown, and only then"
    for k in (:electrons_per_dn,)
        haskey(x, k) && !(x[k][:value] > 0) && return "electrons_per_dn is positive"
    end
    if haskey(x, :dark)
        b = x[:dark][:exposure_band]
        0 < b[1] <= b[2] || return "dark: its exposure band runs from its shortest to its longest time, both positive"
        x[:dark][:frames] >= 2 || return "dark: a temporal variance needs at least 2 frames"
    end
    return nothing
end

const _SENSOR_NOISE_SPEC = _kind_spec(
    "sensor_noise",
    Pair{Symbol, _Spec}[:device => _Obj(:body => _DEVICE_IDENTITY), :iso => _Real()],
    Pair{Symbol, _Spec}[
        :electrons_per_dn => _noise_field_spec(_Real()), :read_noise => _noise_field_spec(_FOUR), :black => _noise_field_spec(_FOUR),
        :dark => _Obj(:source => _SOURCE, :exposure_band => _Vec(_Real(), 2), :frames => _PosInt(), :size => _SIZE_INT,
                      :offset => _RASTER_PAYLOAD, :variance => _RASTER_PAYLOAD),
        :defects => _Obj(:source => _SOURCE, :size => _SIZE_INT, :mask => _RASTER_PAYLOAD),
        :reason => _NonEmpty(),
    ];
    check = _sensor_noise_check,
    rule = "`reason` is stated exactly when electrons_per_dn or read_noise is not; electrons_per_dn is positive; a dark's " *
           "exposure band runs from its shortest to its longest time, both positive, over at least 2 frames",
    schema = [Dict("if" => Dict("required" => ["electrons_per_dn", "read_noise"]),
                   "then" => Dict("not" => Dict("required" => ["reason"])), "else" => Dict("required" => ["reason"])),
              Dict("properties" => Dict("dark" => Dict("properties" => Dict("frames" => Dict("minimum" => 2)))))],
)

# A field on the wire: its value (a tuple as an array), its source and its standard error where stated.
_noise_scalar_json(v::Real) = Float64(v)
_noise_scalar_json(v::Tuple) = Float64[v...]
function _noise_field_json(f)
    a = getfield(f, :attributes)
    r = (value = _noise_scalar_json(value_of(f)), source = String(a.source))
    return a.std_error === nothing ? r : merge(r, (std_error = _noise_scalar_json(a.std_error),))
end
_noise_from(::Type{Float64}, x) = Float64(x::Real)
_noise_from(::Type{NTuple{4, Float64}}, x) = ntuple(i -> Float64(x[i]::Real), 4)
_noise_field_from(::Type{V}, x) where {V} = VgtkCore.noise_value(_noise_from(V, x.value), Symbol(x.source::String);
    std_error = haskey(x, :std_error) ? _noise_from(V, x.std_error) : nothing)

# A defect mask as its uint8 raster (1 a defect), and back (a raster of 0s and 1s).
_mask_raster(M::AbstractMatrix{Bool}) = UInt8.(Array(M))
function _mask_from_raster(A::AbstractMatrix{UInt8}, where_)
    all(b -> b <= 0x01, A) || throw(ArgumentError("$(where_): a defect mask holds 0 or 1 per photosite"))
    return A .== 0x01
end
_map_size(M) = (width = size(M, 2), height = size(M, 1))

"""
    save(path, noise::SensorNoise; provenance = (;))

Write a VGC `SensorNoise` (the resolver's value, or a record of record from
`master_dark` and a photon-transfer fit) as a `vgio/sensor_noise/v1.0` file:
the body's identity (`device.body`), the ISO, each known field with its source
and standard error (an unknown one omitted), `reason` when the noise is
unknown. A master dark's offset and variance maps and the defect mask are
written beside `path` as rasters, the raster payload body (`<stem>.dark_offset.tiff`,
`<stem>.dark_variance.tiff` float32, `<stem>.defects.tiff` uint8; TIFFs of the
raster profile, row 1 on top) bound by their SHA-256.
"""
function save(path::AbstractString, n::SensorNoise; provenance::NamedTuple = (;))
    dir, stem = dirname(path), first(splitext(basename(path)))
    sec = parse_unit(_TIME_UNIT)
    fields = NamedTuple(k => _noise_field_json(getfield(n, k)) for k in (:electrons_per_dn, :read_noise, :black) if getfield(n, k) !== nothing)
    dark = if n.dark === nothing
        (;)
    else
        D = value_of(n.dark)
        (dark = (source = String(getfield(n.dark, :attributes).source),
                 exposure_band = Float64[ustrip(sec, D.exposure_band[1]), ustrip(sec, D.exposure_band[2])], frames = D.frames,
                 size = _map_size(D.offset), offset = _raster_payload_json(dir, "$(stem).dark_offset.tiff", D.offset),
                 variance = _raster_payload_json(dir, "$(stem).dark_variance.tiff", D.variance)),)
    end
    defects = n.defects === nothing ? (;) :
        (defects = (source = String(getfield(n.defects, :attributes).source), size = _map_size(value_of(n.defects)),
                    mask = _raster_payload_json(dir, "$(stem).defects.tiff", _mask_raster(value_of(n.defects)))),)
    body = merge((device = (body = _identity_json(n.body),), iso = n.iso), fields, dark, defects,
                 n.reason === nothing ? (;) : (reason = n.reason,))
    _write_kind(path, "sensor_noise", body, provenance; style = VgioStyle())
    return nothing
end

"""
    load(SensorNoise, path) -> SensorNoise

Read a `vgio/sensor_noise/v1.0` file ([`save`](@ref)'s): the `SensorNoise`
exactly as written, each field with its source and standard error, `nothing`
where the file states none; the dark maps (`Matrix{Float32}`) and the defect
mask (`BitMatrix`) read from the rasters beside it, each checked against its
SHA-256, the raster profile and its size.
"""
function load(::Type{SensorNoise}, path::AbstractString)
    d = _read_store(path, "sensor_noise", _SENSOR_NOISE_SPEC)
    sec = parse_unit(_TIME_UNIT)
    field(k, V) = haskey(d, k) ? _noise_field_from(V, d[k]) : nothing
    dark = if haskey(d, :dark)
        x = d.dark
        W, H = Int(x.size.width::Integer), Int(x.size.height::Integer)
        D = VgtkCore.DarkFrame(_raster_payload_from_json(path, x.offset, Float32, W, H), _raster_payload_from_json(path, x.variance, Float32, W, H),
                               (Float64(x.exposure_band[1]::Real) * sec, Float64(x.exposure_band[2]::Real) * sec), Int(x.frames::Integer))
        VgtkCore.noise_value(D, Symbol(x.source::String))
    end
    defects = if haskey(d, :defects)
        A = _raster_payload_from_json(path, d.defects.mask, UInt8, Int(d.defects.size.width::Integer), Int(d.defects.size.height::Integer))
        VgtkCore.noise_value(_mask_from_raster(A, path), Symbol(d.defects.source::String))
    end
    return SensorNoise(; body = _identity_from_json(d.device.body), iso = Float64(d.iso::Real),
                       electrons_per_dn = field(:electrons_per_dn, Float64), read_noise = field(:read_noise, NTuple{4, Float64}),
                       black = field(:black, NTuple{4, Float64}), dark, defects,
                       reason = haskey(d, :reason) ? String(d.reason::String) : nothing)
end

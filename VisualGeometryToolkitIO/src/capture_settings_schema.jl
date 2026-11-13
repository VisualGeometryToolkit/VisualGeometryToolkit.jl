# =============================================================================
# vgio/capture_settings/v2.0 ↔ VGC `CaptureSettings` — a capture's camera settings
# =============================================================================
#
# {"schema": "vgio/capture_settings/v2.0", <envelope>,
#  "sensor": "cam1",                                                 (optional) the camera these settings are of
#  "device": {"body": {"make": "Sony", "model": "ILME-FX3"},
#             "lens": {"make": "SIGMA", "model": "14mm F1.8 DG HSM | Art"}},
#  "identity_sources": {"body": "exif", "lens": "config"},
#  "exif_lens": {"model": "DT 14mm F2.8 SAM"},                       (optional; beside a config-stated lens only)
#  "frames": [{"frame_index": 1, "exposure_time": 0.016666667535901070, "f_number": 4.0, "iso": 100.0,
#              "focal_length": 14.0, "focus_distance": 1200.0}]}   (focus_distance: optional)
#
# Per frame index (unique; the join key of the capture's other kinds, with
# `sensor` naming the camera when a capture holds several), the exposure and the
# NOMINAL lens settings — `exposure_time` in `units.time` (seconds),
# `focal_length` and `focus_distance` in `units.length` through the
# decimal-exact pair — and, once per file, the identities of the body and the
# lens: `device`, the ONE device-identity encoding (a camera's
# `calibration_record.device` is the same object), each a VGC `DeviceIdentity`
# (`make`, `model`, `serial`, each stated or omitted). Values are math and
# identity is an attribute: the camera store holds no names, so which hardware
# took a capture is recorded here.
#
# PROVENANCE OF AN IDENTITY: `identity_sources` states, for each, `"exif"` (what
# the files said) or `"config"` (what the user stated). EXIF is unreliable for an
# adapted lens (an FX3 with Sigma EF lenses names a Sony "DT" lens), so a
# config-stated lens may keep the claim it overrode, `exif_lens`. In the value
# the sources, that claim and the camera's name are attributes:
#
#     Attributed(cs::CaptureSettings, (; identity_sources = (body = "exif", lens = "config"),
#                                        exif_lens = DeviceIdentity(…), sensor = "cam1"))    (exif_lens, sensor optional)

const _IDENTITY_SOURCES = ("exif", "config")
const _IDENTITY_SOURCE = _Named(s -> s in _IDENTITY_SOURCES, "an identity source (exif, config)")
const _DEVICE = _Obj(:body => _DEVICE_IDENTITY, :lens => _DEVICE_IDENTITY)

const _CAPTURE_SETTINGS_SPEC = _kind_spec(
    "capture_settings",
    Pair{Symbol, _Spec}[
        :device => _DEVICE,
        :identity_sources => _Obj(:body => _IDENTITY_SOURCE, :lens => _IDENTITY_SOURCE),
        :frames => _Vec(
            _Obj(
                Pair{Symbol, _Spec}[:frame_index => _Int(), :exposure_time => _Real(), :f_number => _Real(), :iso => _Real(), :focal_length => _Real()],
                Pair{Symbol, _Spec}[:focus_distance => _Real()],
            )
        ),
    ],
    Pair{Symbol, _Spec}[:sensor => _NonEmpty(), :exif_lens => _DEVICE_IDENTITY];
    check = x -> something(
        (haskey(x, :exif_lens) && x[:identity_sources][:lens] != "config") ?
            "`exif_lens` is the claim a config-stated lens overrode; this lens's source is $(repr(x[:identity_sources][:lens]))" : nothing,
        _unique_check([f[:frame_index] for f in x[:frames]], "frame indices `frame_index`"),
        Some(nothing),
    ),
)

const _CaptureSettingsRecord = Attributed{CaptureSettings, <:NamedTuple}

# The attributes' shape; the values (each source one of exif/config, `exif_lens`
# only beside a config-stated lens) are the spec's, checked on the bytes written.
const _CAPTURE_ATTRIBUTES = ((:identity_sources,), (:identity_sources, :exif_lens), (:identity_sources, :sensor), (:identity_sources, :exif_lens, :sensor))
function _identity_sources(a::NamedTuple)
    (keys(a) in _CAPTURE_ATTRIBUTES && a.identity_sources isa NamedTuple{(:body, :lens)} &&
        get(a, :exif_lens, DeviceIdentity()) isa DeviceIdentity && get(a, :sensor, "") isa AbstractString) || throw(
        ArgumentError("capture settings: the attributes are `(; identity_sources = (body = …, lens = …)[, exif_lens::DeviceIdentity][, sensor::String])`, got $(a)")
    )
    return a.identity_sources
end

"""
    save(path, x::Attributed{CaptureSettings}; length_unit = mm, provenance = (;))

Write a VGC `CaptureSettings` as a `vgio/capture_settings/v2.0` file: per frame
index (sorted) its exposure time (seconds), f-number and ISO, its nominal
focal length and its focus distance where known, in `length_unit`; and the
body and lens identities (`device`). The value is `Attributed` with where each
identity came from, `(; identity_sources = (body = "exif" | "config", lens =
…))`, optionally `exif_lens`, the EXIF claim a config-stated lens overrode, and
optionally `sensor`, the camera these settings are of. A bare `CaptureSettings`
is an `ArgumentError`: an identity's source is not guessed.
"""
function save(path::AbstractString, x::_CaptureSettingsRecord; length_unit::Units = mm, provenance::NamedTuple = (;))
    a = getfield(x, :attributes)
    src = _identity_sources(a)
    cs = getfield(x, :value)
    sec = parse_unit(_TIME_UNIT)
    frames = map(sort!(collect(keys(cs)))) do t
        e, l = cs[t].exposure, cs[t].lens_settings
        f = (
            frame_index = t, exposure_time = Float64(ustrip(sec, e.exposure_time)), f_number = e.f_number, iso = e.iso,
            focal_length = write_length(l.focal_length, length_unit),
        )
        l.focus_distance === nothing ? f : merge(f, (focus_distance = write_length(l.focus_distance, length_unit),))
    end
    body = merge(
        haskey(a, :sensor) ? (sensor = String(a.sensor),) : (;),
        (device = _device_json((body = cs.body, lens = cs.lens)), identity_sources = (body = src.body, lens = src.lens)),
        haskey(a, :exif_lens) ? (exif_lens = _identity_json(a.exif_lens),) : (;),
        (frames = frames,),
    )
    _write_kind(path, "capture_settings", body, provenance; style = VgioStyle(; length_unit))
    return nothing
end
save(::AbstractString, ::CaptureSettings; kwargs...) = throw(
    ArgumentError(
        "save: a CaptureSettings file states where each identity came from; " *
            "save `Attributed(cs, (; identity_sources = (body = \"exif\", lens = \"exif\")))`"
    )
)

"""
    load(CaptureSettings, path) -> Attributed{CaptureSettings}

Read a `vgio/capture_settings/v2.0` file: the VGC `CaptureSettings` (exposure
times in seconds, lengths in metres), `Attributed` with its
`identity_sources` and, where the file states them, `exif_lens` and `sensor` —
exactly the value [`save`](@ref) wrote. A v1 file is read through its version
step.
"""
function load(::Type{CaptureSettings}, path::AbstractString)
    d = _read_store(path, "capture_settings")
    u, sec = _file_length_unit(d), parse_unit(_TIME_UNIT)
    frames = Dict{Int, FrameSettings}()
    for f in d.frames::Vector{Any}
        fd = haskey(f, :focus_distance) ? read_length(f.focus_distance::Real, u) : nothing
        frames[Int(f.frame_index::Integer)] = (
            exposure = ExposureSettings(Float64(f.exposure_time::Real) * sec, f.f_number::Real, f.iso::Real),
            lens_settings = LensSettings(read_length(f.focal_length::Real, u), fd),
        )
    end
    dev = _device_from_json(d.device)
    cs = CaptureSettings(frames; body = dev.body, lens = dev.lens)
    a = (identity_sources = (body = String(d.identity_sources.body::String), lens = String(d.identity_sources.lens::String)),)
    haskey(d, :exif_lens) && (a = merge(a, (exif_lens = _identity_from_json(d.exif_lens),)))
    haskey(d, :sensor) && (a = merge(a, (sensor = String(d.sensor::String),)))
    return Attributed(cs, a)
end

# =============================================================================
# vgio/capture_settings/v1 ↔ VGC `CaptureSettings` — a capture's camera settings
# =============================================================================
#
# {"schema": "vgio/capture_settings/v1", "length_unit": "mm", "time_unit": "s",
#  "body": {"make": "Sony", "model": "ILME-FX3", "source": "exif"},
#  "lens": {"make": "SIGMA", "model": "14mm F1.8 DG HSM | Art", "source": "config",
#           "exif": {"model": "DT 14mm F2.8 SAM"}},                 (exif: optional, config only)
#  "frames": [{"t": 1, "exposure_time": 0.016666667535901070, "f_number": 4.0, "iso": 100.0,
#              "focal_length": 14.0, "focus_distance": 1200.0}]}   (focus_distance: optional)
#
# Per frame index `t` (unique; the join key of the capture's other kinds), the
# exposure and the NOMINAL lens settings — `exposure_time` in `time_unit`
# (seconds), `focal_length` and `focus_distance` in `length_unit` through the
# decimal-exact pair — and, once per file, the identities of the body and the
# lens. Values are math and identity is an attribute: the camera store holds
# no names, so which hardware took a capture is recorded here.
#
# PROVENANCE OF AN IDENTITY: each states its `source`, `"exif"` (what the
# files said) or `"config"` (what the user stated). EXIF is unreliable for an
# adapted lens (an FX3 with Sigma EF lenses names a Sony "DT" lens), so a
# config-stated lens may keep the claim it overrode, `exif`, beside it. In the
# value the sources are an attribute:
#
#     Attributed(cs::CaptureSettings, (; identity_sources = (body = "exif", lens = "config"),
#                                        exif_lens = DeviceIdentity(…)))       (exif_lens optional)

const _TIME_UNIT = "s"   # the one time unit a file states (exposure times)
const _IDENTITY_SOURCES = ("exif", "config")
const _IDENTITY_SOURCE = _Named(s -> s in _IDENTITY_SOURCES, "an identity source (exif, config)")

const _CAPTURE_SETTINGS_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("capture_settings"), :length_unit => _LENGTH_UNIT, :time_unit => _Tok(_TIME_UNIT),
            :body => _Obj(Pair{Symbol, _Spec}[:source => _IDENTITY_SOURCE], _DEVICE_IDENTITY_FIELDS),
            :lens => _Checked(
                _Obj(Pair{Symbol, _Spec}[:source => _IDENTITY_SOURCE], Pair{Symbol, _Spec}[_DEVICE_IDENTITY_FIELDS..., :exif => _DEVICE_IDENTITY]),
                x -> (haskey(x, :exif) && x[:source] != "config") ? "`exif` is the claim a config-stated lens overrode; this lens's source is $(repr(x[:source]))" : nothing,
            ),
            :frames => _Vec(
                _Obj(
                    Pair{Symbol, _Spec}[:t => _Int(), :exposure_time => _Real(), :f_number => _Real(), :iso => _Real(), :focal_length => _Real()],
                    Pair{Symbol, _Spec}[:focus_distance => _Real()],
                )
            ),
        ],
        Pair{Symbol, _Spec}[_PROVENANCE...],
    ),
    x -> _unique_check([f[:t] for f in x[:frames]], "frame indices `t`"),
)

const _CaptureSettingsRecord = Attributed{CaptureSettings, <:NamedTuple}

# The attributes' shape; the values (each source one of exif/config, `exif_lens`
# only beside a config-stated lens) are the spec's, checked on the bytes written.
function _identity_sources(a::NamedTuple)
    (keys(a) in ((:identity_sources,), (:identity_sources, :exif_lens)) && a.identity_sources isa NamedTuple{(:body, :lens)} &&
        get(a, :exif_lens, DeviceIdentity()) isa DeviceIdentity) || throw(
        ArgumentError("capture settings: the attributes are `(; identity_sources = (body = …, lens = …)[, exif_lens::DeviceIdentity])`, got $(a)")
    )
    return a.identity_sources
end

"""
    save(path, x::Attributed{CaptureSettings}; length_unit = mm, provenance = (;))

Write a VGC `CaptureSettings` as a `vgio/capture_settings/v1` file: per frame
index (sorted) its exposure time (seconds), f-number and ISO, its nominal
focal length and its focus distance where known, in `length_unit`; and the
body and lens identities. The value is `Attributed` with where each identity
came from, `(; identity_sources = (body = "exif" | "config", lens = …))`, and
optionally `exif_lens`, the EXIF claim a config-stated lens overrode. A bare
`CaptureSettings` is an `ArgumentError`: an identity's source is not guessed.
"""
function save(path::AbstractString, x::_CaptureSettingsRecord; length_unit::Units = mm, provenance::NamedTuple = (;))
    a = getfield(x, :attributes)
    src = _identity_sources(a)
    cs = getfield(x, :value)
    sec = parse_unit(_TIME_UNIT)
    frames = map(sort!(collect(keys(cs)))) do t
        e, l = cs[t].exposure, cs[t].lens_settings
        f = (
            t = t, exposure_time = Float64(ustrip(sec, e.exposure_time)), f_number = e.f_number, iso = e.iso,
            focal_length = write_length(l.focal_length, length_unit),
        )
        l.focus_distance === nothing ? f : merge(f, (focus_distance = write_length(l.focus_distance, length_unit),))
    end
    lens = merge(_identity_json(cs.lens), (source = src.lens,), haskey(a, :exif_lens) ? (exif = _identity_json(a.exif_lens),) : (;))
    body = (
        schema = _schema("capture_settings"), length_unit = unit_string(length_unit), time_unit = _TIME_UNIT,
        body = merge(_identity_json(cs.body), (source = src.body,)), lens = lens, frames = frames,
    )
    _write_store(path, _CAPTURE_SETTINGS_SPEC, _with_provenance(body, provenance); style = VgioStyle(; length_unit))
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

Read a `vgio/capture_settings/v1` file: the VGC `CaptureSettings` (exposure
times in seconds, lengths in metres), `Attributed` with its
`identity_sources` and, where the file keeps it, `exif_lens` — exactly the
value [`save`](@ref) wrote.
"""
function load(::Type{CaptureSettings}, path::AbstractString)
    d = _read_store(path, "capture_settings", _CAPTURE_SETTINGS_SPEC)
    u, sec = parse_unit(d.length_unit::String), parse_unit(_TIME_UNIT)
    frames = Dict{Int, FrameSettings}()
    for f in d.frames::Vector{Any}
        fd = haskey(f, :focus_distance) ? read_length(f.focus_distance::Real, u) : nothing
        frames[Int(f.t::Integer)] = (
            exposure = ExposureSettings(Float64(f.exposure_time::Real) * sec, f.f_number::Real, f.iso::Real),
            lens_settings = LensSettings(read_length(f.focal_length::Real, u), fd),
        )
    end
    cs = CaptureSettings(frames; body = _identity_from_json(d.body), lens = _identity_from_json(d.lens))
    src = (identity_sources = (body = String(d.body.source::String), lens = String(d.lens.source::String)),)
    return Attributed(cs, haskey(d.lens, :exif) ? merge(src, (exif_lens = _identity_from_json(d.lens.exif),)) : src)
end

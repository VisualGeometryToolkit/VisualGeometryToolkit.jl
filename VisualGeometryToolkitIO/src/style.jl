# =============================================================================
# VgioStyle — the serializer every record's VGC values go through
# =============================================================================
#
# A vgio record embeds VGC values: a pose, a camera, a target, its layout, a
# domain ellipse. Each is written by ONE `lower` method and read by ONE `lift`
# method (a target's calibration, by the target's: a `lower` on VGC's
# `TargetCalibration` union would claim every planar map left in a record —
# rig_schema.jl) — StructUtils' functions, reached by JSON.jl's public
# names `JSON.lower` and `JSON.lift` — both on IO's own style, `VgioStyle`, so
# no method lands on them for a style IO does not own (no piracy, no global
# `JSON.lower(x)`), and the file's declarations a value needs — its length unit
# and its pixel frame — travel with the style rather than as arguments of each
# encoder:
#
#   JSON.lower(VgioStyle(; length_unit = mm), pose)         {"rotation", "translation"[, "covariance"]}
#   JSON.lift(VgioStyle(; length_unit = mm), EuclideanMap{3}, record)
#   JSON.json(x; style = VgioStyle(…))                      the writers (`_write_store`, the file's own style)
#
# The load pipeline is parse → `migrate_record` (an earlier version's steps) →
# validate (the kind's spec) → `lift`: a lift reads a record the validator has
# already checked, so it narrows each field at its use and checks nothing the
# spec checks. A lift takes the parsed tree (`_Record`: a `JSON.Object` or a
# NamedTuple, read by property access; an array for an ellipse), never JSON.jl's
# lazy values.
#
# THE PARAMETRIC RECORD. A parametric model is written by its tag and its
# parameters, and nothing else unless its structure is more than its type:
#
#   {"type": <tag>, ["nodes": […]], ["parameters": [θ₁, …]]}
#
# the tag from the kind's `TagRegistry` (the tag rule: snake_case of the type's
# name), the parameters Core's parameter protocol — `parameter_vector` in
# `parameter_names` order, a length (by `length_mask`) in the style's unit
# through `write_length`'s decimal shift, any other number as it is — and
# `parameters` omitted for a kind that has none. It is read back by
# `with_parameters` on the kind's prototype: the parameters are the value. A
# kind whose structure the type does not fix (a thin-plate kind's nodes) states
# it beside its parameters and is rebuilt by its own constructor.
#
# MINIMAL FORMS: a pose is its quaternion (w ≥ 0) and translation, a K its
# five numbers `fx, fy, cx, cy, skew`, an ellipse `[cx, cy, a, b, θ]`; a dense
# matrix is never written for a geometric value (only a foreign export whose
# spec demands one writes one).

"""
    VgioStyle(; length_unit = mm, pixel_frame = PixelFrame{:colmap}) <: JSON.JSONStyle

VisualGeometryToolkitIO's serialization style: every VGC value a `vgio/<kind>`
record embeds is written by a `JSON.lower(::VgioStyle, x)` method and read by
a `JSON.lift(::VgioStyle, T, record)` method (StructUtils' `lower` and `lift`,
the `JSON.JSONStyle` hooks), on this style
only. It carries what the enclosing file declares for the value — the
`length_unit` its lengths are written in (decimal-exact, through
[`write_length`](@ref) and [`read_length`](@ref)) and the `pixel_frame` its
principal points are written in — and is handed to the writer as
`JSON.json(x; style = VgioStyle(…))`.

A package serializing a kind of its own embeds VGC values the same way (a
pose, a `CameraModel`, a target with its calibration, a layout, an
`Ellipse`); a lift reads a parsed `JSON.Object` or a `NamedTuple`:

```jldoctest
julia> import JSON

julia> s = VgtkIO.VgioStyle(; length_unit = mm);

julia> e = Ellipse(Point2(1.0, 2.0), 3.0, 1.0, 0.5);

julia> JSON.json((domain = e,); style = s)
"{\\"domain\\":[1.0,2.0,3.0,1.0,0.5]}"

julia> JSON.lift(s, Ellipse, JSON.parse(JSON.json(e; style = s))) == e
true
```
"""
struct VgioStyle{U, F <: PixelFrame} <: JSON.JSONStyle
    length_unit::U          # Units, or a private leaf form (`_PERSISTED`, the identity's; `_CHART`, a placed target's)
end
VgioStyle(; length_unit = mm, pixel_frame::Type{<:PixelFrame} = CANONICAL_PIXEL_FRAME) =
    VgioStyle{typeof(length_unit), pixel_frame}(length_unit)
_pixel_frame(::VgioStyle{U, F}) where {U, F} = F
# The same style with another length form (a placed target's calibration: chart numbers).
_with_length(s::VgioStyle{U, F}, u) where {U, F} = VgioStyle{typeof(u), F}(u)
# What a lift reads: a record whose fields are read by property access — a
# parsed JSON object (JSON.jl's `Object`, String keys as every reader hands it,
# or Symbol keys) or a NamedTuple (a record built in memory,
# `camera_from_json(camera_json(…))`). A `Dict` has no property access.
const _Record = Union{JSON.Object, NamedTuple}

# `CANONICAL_PIXEL_FRAME` is the single thing VGIo must state: which frame a
# `CameraModel`'s stored K is in. VGC guarantees it — every `CameraModel`
# constructor converts its `image_origin` argument to this frame on the way in
# — but exposes no accessor for it, so it is named here ONCE rather than as a
# `:colmap` literal in every writer and reader.
const CANONICAL_PIXEL_FRAME = PixelFrame{:colmap}

# -----------------------------------------------------------------------------
# The parametric record
# -----------------------------------------------------------------------------

# A parameter vector on the wire: lengths (by `mask`) in the style's unit, the
# rest as they are; and back, plain (lengths in metres, a placed target's chart
# numbers as stored).
_parameters_json(θ, mask, u) = Float64[mask[i] ? _metres_json(θ[i], u) : Float64(θ[i]) for i in eachindex(θ)]
function _parameters_from_json(p, mask, u, kind::AbstractString)
    length(p) == length(mask) || throw(
        ArgumentError("a \"$kind\" record takes $(length(mask)) parameters (its kind's parameter_vector); got $(length(p))")
    )
    return Float64[mask[i] ? _metres(p[i], u) : Float64(p[i]::Real) for i in eachindex(mask)]
end

# The record of a parametric value `x` of tag `tag`: its structure (`structure`,
# a NamedTuple; empty when the type fixes it) and its parameters.
function _parametric_record(tag::AbstractString, x, u; structure::NamedTuple = (;))
    p = _parameters_json(parameter_vector(x), length_mask(x), u)
    return merge((type = tag,), structure, isempty(p) ? (;) : (parameters = p,))
end
# …and back: the prototype's `with_parameters` of the record's parameters, as
# many as the kind has (more or fewer is an `ArgumentError` naming the tag).
_from_parametric_record(proto, x, u) =
    with_parameters(proto, _parameters_from_json(get(x, :parameters, ()), length_mask(proto), u, x.type::String))

# -----------------------------------------------------------------------------
# The minimal forms: a K, an ellipse
# -----------------------------------------------------------------------------

# A K's five numbers, its principal point moved from the canonical frame (a
# stored K's, VGC's) into `frame`, the pixel frame the file declares; and back,
# the K as the record states it, in the file's frame (`CameraModel(K;
# image_origin)` moves it into the canonical one).
#
# The principal point always goes through the frame conversion, the canonical
# frame's identity included: its `+ 0.0` writes a stored `-0.0` as `0.0`, as
# the camera bodies always have.
function _k_record(K::CameraCalibrationMat, frame::Type{<:PixelFrame})
    c = EuclideanMap(CANONICAL_PIXEL_FRAME, frame)(SVector{2, Float64}(K[1, 3], K[2, 3]))
    return _k_five(K, c[1], c[2])
end
_k_five(K, cx, cy) = (fx = Float64(K[1, 1]), fy = Float64(K[2, 2]), cx = Float64(cx), cy = Float64(cy), skew = Float64(K[1, 2]))
_k_from_record(k) = CameraCalibrationMat(
    SMatrix{3, 3, Float64}(
        Float64(k.fx::Real), 0, 0, Float64(k.skew::Real), Float64(k.fy::Real), 0, Float64(k.cx::Real), Float64(k.cy::Real), 1,
    )
)
# the five numbers as an array, as stored (a generic model's nominal pinhole,
# the identity's intrinsics: no frame conversion, so a `-0.0` stays), and back
_k_values(K::CameraCalibrationMat) = collect(Float64, _k_five(K, K[1, 3], K[2, 3]))
_k_from_values(v) = _k_from_record(NamedTuple{(:fx, :fy, :cx, :cy, :skew)}(Tuple(v)))

# An ellipse: its centre, its semi-axes (the major first, as VGC's `Ellipse`
# holds them) and the major axis's angle, in the value's own units.
JSON.lower(::VgioStyle, e::Ellipse) = Float64[e.center[1], e.center[2], e.a, e.b, e.θ]
JSON.lift(::VgioStyle, ::Type{Ellipse}, v::AbstractVector{Any}) =
    (e = Float64[Float64(x::Real) for x in v]; Ellipse(Point2{Float64}(e[1], e[2]), e[3], e[4], e[5]))

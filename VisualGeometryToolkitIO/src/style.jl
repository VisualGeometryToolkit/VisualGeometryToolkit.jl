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
# spec checks. A lift takes the parsed tree (`Record`: a `JSON.Object` or a
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
# five numbers `fx, fy, cx, cy, skew`, an ellipse `[cx, cy, a, b, θ]`, a plane
# homography the parametric record of its nine entries under a DECLARED
# normalization (`{"type": "homography_map", "parameters": [h11 … h33]}`, unit
# Frobenius norm and h33 ≥ 0: 8 degrees of freedom have no global 8-number
# chart, so the ninth number and the rule that fixes it are stated instead); a
# dense matrix is never written for a geometric value (only a foreign export
# whose spec demands one writes one).

"""
    VgioStyle(; length_unit = mm, pixel_frame = PixelFrame{:colmap}) <: JSON.JSONStyle

VisualGeometryToolkitIO's serialization style: every VGC value a `vgio/<kind>`
record embeds is written by a `JSON.lower(::VgioStyle, x)` method and read by
a `JSON.lift(::VgioStyle, T, record)` method (StructUtils' `lower` and `lift`,
the `JSON.JSONStyle` hooks), on this style
only. It carries what the enclosing file declares for the value — the
`length_unit` its lengths are written in (decimal-exact, through
[`write_length`](@ref) and [`read_length`](@ref)) and the `pixel_frame` its
principal points are written in — and the writer takes it as
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
"""
    Record

What a lift reads (`JSON.lift(::VgioStyle, T, record)`): a record whose fields
are read by property access — a parsed JSON object (JSON.jl's `Object`, String
keys as every reader hands it, or Symbol keys) or a NamedTuple (a record built
in memory, `camera_from_json(camera_json(…))`). A `Dict` has no property
access. Another package's lifts of its own kinds (BlobBoards') take it too.
"""
const Record = Union{JSON.Object, NamedTuple}

# `CANONICAL_PIXEL_FRAME` is the single thing VGIo must state: which frame a
# `CameraModel`'s stored K is in. VGC guarantees it — every `CameraModel`
# constructor converts its `pixel_frame` argument to this frame on the way in
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
    return Float64[mask[i] ? metres(p[i], u) : Float64(p[i]::Real) for i in eachindex(mask)]
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
# the K as the record states it, in the file's frame, moved into the canonical
# one. Both moves are Core's one pixel-frame change of a K
# (`reframed_calibration`, which `CameraModel(K; pixel_frame)` makes too).
#
# The principal point always goes through the frame conversion, the canonical
# frame's identity included: its `+ 0.0` writes a stored `-0.0` as `0.0`, as
# the camera bodies always have.
_k_record(K::CameraCalibrationMat, frame::Type{<:PixelFrame}) = _k_five(VgtkCore.Cameras.reframed_calibration(CANONICAL_PIXEL_FRAME, frame, K))
_k_five(K) = (fx = Float64(K[1, 1]), fy = Float64(K[2, 2]), cx = Float64(K[1, 3]), cy = Float64(K[2, 3]), skew = Float64(K[1, 2]))
_k_from_record(k) = CameraCalibrationMat(
    SMatrix{3, 3, Float64}(
        Float64(k.fx::Real), 0, 0, Float64(k.skew::Real), Float64(k.fy::Real), 0, Float64(k.cx::Real), Float64(k.cy::Real), 1,
    )
)
# …and the K a record states in `frame`, in the canonical frame.
_k_from_record(k, frame::Type{<:PixelFrame}) = VgtkCore.Cameras.reframed_calibration(frame, CANONICAL_PIXEL_FRAME, _k_from_record(k))
# the five numbers as an array, as stored (an old grid record's nominal pinhole,
# the identity's intrinsics: no frame conversion, so a `-0.0` stays), and back
_k_values(K::CameraCalibrationMat) = collect(Float64, _k_five(K))
_k_from_values(v) = _k_from_record(NamedTuple{(:fx, :fy, :cx, :cy, :skew)}(Tuple(v)))

# An ellipse: its centre, its semi-axes (the major first, as VGC's `Ellipse`
# holds them) and the major axis's angle, in the value's own units.
JSON.lower(::VgioStyle, e::Ellipse) = Float64[e.center[1], e.center[2], e.a, e.b, e.θ]
JSON.lift(::VgioStyle, ::Type{Ellipse}, v::AbstractVector{Any}) =
    (e = Float64[Float64(x::Real) for x in v]; Ellipse(Point2{Float64}(e[1], e[2]), e[3], e[4], e[5]))
# A circle: its centre and its radius, in the value's own units.
JSON.lower(::VgioStyle, c::Circle) = Float64[c.center[1], c.center[2], c.r]
JSON.lift(::VgioStyle, ::Type{Circle}, v::AbstractVector{Any}) =
    (c = Float64[Float64(x::Real) for x in v]; Circle(Point2{Float64}(c[1], c[2]), c[3]))
# A rectangle: its origin and its widths, in the value's own units.
JSON.lower(::VgioStyle, r::Rect2) = Float64[r.origin[1], r.origin[2], r.widths[1], r.widths[2]]
JSON.lift(::VgioStyle, ::Type{Rect2}, v::AbstractVector{Any}) =
    (r = Float64[Float64(x::Real) for x in v]; Rect2{Float64}(r[1], r[2], r[3], r[4]))

# A spline record's lattice (both spline lens schemas): its `grid_size` `(gw, gh)`, its
# present nodes from the stored row runs (`nothing`: every node), its `area`
# `[x, y, w, h]` — the mask `P` the controls are read on, the area, and the lens's
# `present` (`nothing` for every node).
function _lattice_from_json(grid_size, runs, a)
    gw, gh = Int(grid_size[1]::Integer), Int(grid_size[2]::Integer)
    P = runs === nothing ? trues(gh, gw) : _present_from_runs(runs, gw, gh)
    return (; P, area = JSON.lift(VgioStyle(), Rect2, Any[a...]), present = runs === nothing ? nothing : P)
end

# A stored ellipse's invariants, as VGC's `Ellipse` holds it — semi-axes positive, the major first — so it reads back as
# the numbers the file holds; the message names `what`, or `nothing` when they hold. (A spline lens camera's domain, a
# camera store's image ellipse.)
function _ellipse_record_problem(e, what)
    (e[3] > 0 && e[4] > 0) || return "$(what)'s semi-axes are positive; got a = $(e[3]), b = $(e[4])"
    e[3] >= e[4] || return "$(what) is stored major semi-axis first (a ≥ b, θ the angle of a); got a = $(e[3]) < b = $(e[4])"
    return nothing
end

# -----------------------------------------------------------------------------
# THE LATTICE RECORD: a uniform cubic B-spline lattice over a rectangle
# -----------------------------------------------------------------------------

"""
    lattice_json(spline::ChartSpline; present = nothing, length_unit = nothing) -> NamedTuple

A uniform cubic B-spline lattice — VisualGeometryToolkitCore's `ChartSpline`,
its `K`-vector controls over a rectangle of a chart — as the lattice record,
the spline lens record's (a `SplineLens`'s backward and forward splines) and
every spline a package stores over one (RadiometricCamera's aperture spline):

    {"area": [x, y, w, h], "grid_size": [gw, gh],
     "present": [[row, [[col_lo, col_hi], …]], …],     (optional) the held nodes; absent = every node
     "controls": [[c₁, …, c_K], …]}                    the held nodes' controls, row by row

`area` is the spline's rectangle, `grid_size` its lattice (columns, rows).
`present` (a `Bool` matrix of the lattice's size, `[gy, gx]`, Core's presence
mask; `nothing`: every node) is written as runs per lattice row, 0-based and
inclusive, and only the held nodes' controls are written. The controls are
written as they are (JSON's shortest round-trip floats, so a `Float64` spline
reads back bit for bit), or, with `length_unit`, as lengths: plain metres in
memory, the file's numbers in that unit through [`write_length`](@ref)'s
decimal shift. [`lattice_from_json`](@ref) reads it back.

```jldoctest
julia> rec = (area = [0.0, 0.0, 1.0, 1.0], grid_size = [4, 4], controls = fill([2.0], 16));   # radii, in mm

julia> L = VgtkIO.lattice_from_json(rec, Val(1); length_unit = mm); L.spline.controls[1, 1][1]   # metres
0.002

julia> VgtkIO.lattice_json(L.spline; length_unit = mm) == rec
true
```
"""
function lattice_json(cs::ChartSpline; present::Union{Nothing, AbstractMatrix{Bool}} = nothing, length_unit::Union{Nothing, Units} = nothing)
    present === nothing || size(present) == size(cs.controls) || throw(
        ArgumentError("lattice_json: `present` is $(size(present, 1)) × $(size(present, 2)), the lattice $(size(cs.controls, 1)) × $(size(cs.controls, 2))")
    )
    P = _held_mask(present, size(cs.controls))
    f = length_unit === nothing ? identity : x -> _metres_json(x, length_unit)
    return merge(
        (area = JSON.lower(VgioStyle(), cs.area), grid_size = [size(cs.controls, 2), size(cs.controls, 1)]),
        all(P) ? (;) : (present = _present_runs(P),), (controls = _controls_json(cs.controls, P, f),),
    )
end

"""
    lattice_from_json(record, ::Val{K}; length_unit = nothing) -> (; spline::ChartSpline{K}, present)

The lattice a lattice record states (the inverse of [`lattice_json`](@ref)):
`spline` the `ChartSpline` of `K`-vector controls over its `area`, NaN in the
nodes the record does not hold (no evaluation reads one), and `present` its
presence mask (`[gy, gx]`), or `nothing` when the record holds every node.
With `length_unit` the controls are lengths in that unit, read as plain
metres, decimal-exactly. `record` is a [`Record`](@ref) or a parsed
`AbstractDict`; it is validated first — exactly its keys, a finite area of
positive widths, at least 4 × 4 nodes, the runs inside the lattice and apart,
one `K`-vector control per held node — each problem an `ArgumentError`
naming it.
"""
lattice_from_json(record, ::Val{K}; length_unit::Union{Nothing, Units} = nothing) where {K} =
    _lattice_from_record(record isa NamedTuple ? _lower(record) : record, Val(K), length_unit, ("present",), "a lattice record")
# …and the lens record's: its optional keys `optional` (`present`, the held nodes, among them; a
# forward spline holds every node), named `what` in a message.
function _lattice_from_record(s::AbstractDict, ::Val{K}, length_unit, optional, what) where {K}
    spec = _Obj(
        Pair{Symbol, _Spec}[:area => _Vec(_Real(), 4), :grid_size => _Vec(_PosInt(), 2), :controls => _Vec(_Vec(_Real(), K))],
        Pair{Symbol, _Spec}[Symbol(k) => (k == "present" ? _RowRuns() : _Free()) for k in optional],
    )
    _validate(spec, s, what, "")
    msg = _lattice_problem(s, (:controls,))
    msg === nothing || throw(ArgumentError("$(what): $(msg)"))
    L = _lattice_from_json(_getkey(s, :grid_size), _haskey(s, :present) ? _getkey(s, :present) : nothing, _getkey(s, :area))
    f = length_unit === nothing ? (x -> Float64(x)) : (x -> _read_power(x, length_unit, 1))
    return (; spline = ChartSpline(L.area, _controls_from_json(_getkey(s, :controls), L.P, Val(K), f)), L.present)
end

# -----------------------------------------------------------------------------
# A spline lattice's held nodes: the presence runs, and the controls over them
# -----------------------------------------------------------------------------
#
# A spline lattice (a SplineLens's backward spline, a non-central model's two)
# holds the nodes its region reaches. A record lists them as runs per lattice
# row — `[[row, [[col_lo, col_hi], …]], …]`, 0-based and inclusive, every row
# that holds one, rows increasing, its runs increasing and apart (any shape:
# convex, cut by the frame, with holes) — and its controls as the held nodes'
# values, row by row. A record without runs holds every node. The ONE codec of
# the lens record (camera_schema.jl) and the non-central record
# (noncentral_spline_schema.jl), and of the old grid bodies their migrations read.

# The stored type names of the old grid models — the `type` of Schöps et al.'s
# intrinsics0.yaml and of the old vgio grid records (generic_camera v2/v3, a rig
# v6–v8 sensor's body), which schoeps.jl and the migrations read — and the old
# grid records' schemas, read and never rewritten (migrate.jl).
const _OLD_CENTRAL_TYPE = "CentralGenericModel"
const _OLD_NONCENTRAL_TYPE = "NoncentralGenericModel"
const _OLD_GRID_SCHEMAS = ("vgio/generic_camera/v2", "vgio/generic_camera/v3")

# The present nodes of a `[gy, gx]` mask as runs per row, 0-based and inclusive.
function _present_runs(P::AbstractMatrix{Bool})
    rows = Any[]
    for gy in 1:size(P, 1)
        runs = Any[]
        gx = 1
        while gx <= size(P, 2)
            if P[gy, gx]
                lo = gx
                while gx < size(P, 2) && P[gy, gx + 1]
                    gx += 1
                end
                push!(runs, [lo - 1, gx - 1])
            end
            gx += 1
        end
        isempty(runs) || push!(rows, Any[gy - 1, runs])
    end
    return rows
end
# The runs back to a `[gy, gx]` mask of a `gh × gw` lattice.
function _present_from_runs(runs, gw::Int, gh::Int)
    P = falses(gh, gw)
    for e in runs, r in e[2]::Vector{Any}
        P[(e[1]::Integer) + 1, ((r[1]::Integer) + 1):((r[2]::Integer) + 1)] .= true
    end
    return P
end
# The number of nodes runs hold.
_runs_count(runs) = sum(r[2] - r[1] + 1 for e in runs for r in e[2]; init = 0)
# A spline's presence mask (`nothing`: every node) as a `Bool` matrix of its grid's size.
_held_mask(::Nothing, sz) = trues(sz)
_held_mask(P::AbstractMatrix{Bool}, _) = P

# The held nodes' controls (`[gy, gx]` row-major over the mask `P`), each coordinate
# through `f` (a length through its unit; any other number as it is).
_controls_json(C::AbstractMatrix{<:StaticVector}, P::AbstractMatrix{Bool}, f = identity) =
    [Float64[f(Float64(x)) for x in C[gy, gx]] for gy in 1:size(C, 1) for gx in 1:size(C, 2) if P[gy, gx]]
# …and back: `K`-vectors in the held nodes' slots, each coordinate through `f`, NaN in
# the absent ones (no value: no evaluation reads one). The controls as stored — no
# re-normalization, which would change the spline.
function _controls_from_json(a, P::AbstractMatrix{Bool}, ::Val{K}, f = x -> Float64(x)) where {K}
    C = fill(SVector{K, Float64}(ntuple(_ -> NaN, Val(K))), size(P))
    for (I, v) in zip(findall(permutedims(P)), a)   # the held nodes in row-major order
        C[I[2], I[1]] = SVector{K, Float64}(ntuple(k -> Float64(f(v[k]::Real)), Val(K)))
    end
    return C
end

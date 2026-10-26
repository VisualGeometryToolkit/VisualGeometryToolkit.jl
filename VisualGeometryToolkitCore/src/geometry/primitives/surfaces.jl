# =============================================================================
# The surface a ray meets: Quad (a bounded plane) and HeightSurface (a height
# field over one)
# =============================================================================
#
# Two isbits primitives behind one protocol, every method generic in the
# element type (Float32, Float64, ForwardDiff.Dual), allocation-free and
# usable inside a KernelAbstractions kernel:
#
#   intersection(s, o, d) -> (t, x)   the ray o + t d meets s at x; NaN if it misses
#   normal(s, x)                      the unit normal of s at its point x
#   chart(s, x) -> (u, v)             the chart coordinates of the point x of s
#   bounds(s) -> Rect                 the chart rectangle s covers
#
# A ray is an origin and a direction, two 3-vectors, as `ray_plane_intersection`
# takes it; there is no ray type.
#
# The chart is a 3-D frame: an origin o and an orthonormal in-plane basis
# (e_u, e_v), the normal n = e_u × e_v. The chart coordinates of x are
# ((x − o)·e_u, (x − o)·e_v), in the units of the frame (a posed target's
# chart is its TargetFrame: metres when physical). A HeightSurface is the
# points o + u e_u + v e_v + h(u, v) n of its quad's chart, h the thin-plate
# spline (numerics/thin_plate_spline.jl) through its nodal heights.
#
# A ray misses when it is parallel to the plane or when its hit's chart
# coordinates fall outside `bounds` (closed): t and every coordinate of x are
# then NaN. The rule is the same for both primitives; a HeightSurface tests
# the surface point Newton ends on.
#
# Kept here, not in the Primitives submodule, because the height field
# evaluates the thin-plate spline (included after Primitives), as bbox.jl is
# kept for the frames it needs. `surface(::PosedTarget)` builds either
# primitive from a posed target (geometry/cameras/target_surfaces.jl).
# =============================================================================

"""
    intersection(s, o, d) -> (t, x)

Where the ray `o + t d` (origin `o`, direction `d`, both 3-vectors; `d` need
not be unit) meets the surface `s`: the ray parameter `t` and the point `x`.
A ray that misses — parallel to a plane, or hitting outside
[`bounds`](@ref)`(s)` — returns `t` and every coordinate of `x` as `NaN`, never
an error. Methods: [`Quad`](@ref) (closed form, the arithmetic of
[`ray_plane_intersection`](@ref)) and [`HeightSurface`](@ref) (Newton).
Allocation-free and generic in the element type, so it runs on the GPU path
and under `ForwardDiff`.
"""
function intersection end

"""
    chart(s, x) -> SVector{2}

The chart coordinates `(u, v)` of the point `x` of the surface `s`: the
projection of `x − o` on the in-plane basis of `s` (the component along the
normal is ignored, so for a [`HeightSurface`](@ref) they are the coordinates
under the surface point).
"""
function chart end

"""
    bounds(s) -> Rect2

The rectangle of chart coordinates the surface `s` covers ([`chart`](@ref)).
"""
function bounds end

"""
    surface(pt::PosedTarget{<:PlanarTarget}) -> Quad | HeightSurface

The surface a ray meets on the posed planar target `pt`, in the pose's
destination frame: the frame of the [`Quad`](@ref) is the target's
TargetFrame placed by `pose(pt)` (origin its translation, `e_u`, `e_v` and the
normal the rotation's columns), its chart coordinates are TargetFrame `(x, y)`,
and its rectangle is the bounding box of the CALIBRATED design's in-plane
coordinates ([`points`](@ref)`(model(pt))`).

- A planar calibration ([`is_planar`](@ref): `Rigid`, `Affine`,
  `ThinPlateSpline` and chains of them) gives a `Quad`.
- A [`HeightField`](@ref), alone or last in a [`CalibrationChain`](@ref) after
  in-plane members, gives a [`HeightSurface`](@ref) over that `Quad` with the
  field's nodes and heights and the weights the field formed once, when it was
  built; nothing is solved per ray.

A physical target (metres) gives a primitive in metres, plain numbers. An
uncertain pose or calibration gives its value. A `PointTarget` has no surface
(a `MethodError`).
"""
function surface end   # the method: geometry/cameras/target_surfaces.jl

"""
    Quad(origin, e_u, e_v, rect::Rect2)
    Quad{T}(q::Quad)

A bounded plane: the rectangle `rect` of the chart carried by the 3-D frame
`origin`, `(e_u, e_v)` — `e_u`, `e_v` an orthonormal in-plane basis, the normal
`e_u × e_v`. A point with chart coordinates `(u, v)` is `origin + u e_u + v e_v`.
`Quad{T}(q)` converts the element type (`Float32` for a device).

isbits. [`intersection`](@ref) is [`ray_plane_intersection`](@ref)'s
arithmetic, bit for bit, `NaN` outside `rect`; [`normal`](@ref) is the plane's;
[`chart`](@ref) and [`bounds`](@ref) as their docstrings. A rigid or in-plane
calibrated target's [`surface`](@ref) is a `Quad`.
"""
struct Quad{T}
    plane::Plane3D{T}
    origin::SVector{3, T}
    e_u::SVector{3, T}
    e_v::SVector{3, T}
    rect::Rect2{T}
end
function Quad(origin::StaticVector{3}, e_u::StaticVector{3}, e_v::StaticVector{3}, rect::Rect2)
    T = float(promote_type(eltype(origin), eltype(e_u), eltype(e_v), eltype(rect.origin)))
    o, u, v = SVector{3, T}(origin), SVector{3, T}(e_u), SVector{3, T}(e_v)
    tol = sqrt(eps(T))
    (abs(dot(u, u) - 1) <= tol && abs(dot(v, v) - 1) <= tol && abs(dot(u, v)) <= tol) ||
        throw(ArgumentError("Quad: the in-plane basis (e_u, e_v) must be orthonormal"))
    plane = Plane3D(Point3{T}(o...), Vec3{T}(cross(u, v)...))
    return Quad{T}(plane, o, u, v, Rect2{T}(rect))
end
Quad{T}(q::Quad{T}) where {T} = q
Quad{T}(q::Quad) where {T} = Quad{T}(Plane3D{T}(q.plane.coeffs), q.origin, q.e_u, q.e_v, Rect2{T}(q.rect))
Base.:(==)(a::Quad, b::Quad) = a.plane.coeffs == b.plane.coeffs && a.origin == b.origin && a.e_u == b.e_u && a.e_v == b.e_v && a.rect == b.rect

normal(q::Quad) = normal(q.plane)
normal(q::Quad, x::StaticVector{3}) = normal(q.plane)
bounds(q::Quad) = q.rect
@inline function chart(q::Quad, x::StaticVector{3})
    δ = x - q.origin
    return SVector(dot(δ, q.e_u), dot(δ, q.e_v))
end

# The unbounded plane hit, as `ray_plane_intersection` computes it (its
# shared-origin form, on scalars).
@inline function _plane_hit(q::Quad, o::StaticVector{3}, d::StaticVector{3})
    x, y, z, t = ray_plane_intersection(q.plane, o, d[1], d[2], d[3])
    return t, SVector(x, y, z)
end

# A closed rectangle test that a NaN fails; `&`, not `&&`: no branch.
@inline function _in_rect(r::Rect2, c::StaticVector{2})
    lo, w = r.origin, r.widths
    return (lo[1] <= c[1]) & (c[1] <= lo[1] + w[1]) & (lo[2] <= c[2]) & (c[2] <= lo[2] + w[2])
end

# (t, x) when x's chart coordinates lie in the rectangle, NaN otherwise.
@inline function _within(q::Quad, t, x::StaticVector{3})
    inside = _in_rect(q.rect, chart(q, x))
    nan = oftype(t, NaN)
    return ifelse(inside, t, nan), ifelse(inside, x, SVector(nan, nan, nan))
end

@inline function intersection(q::Quad, o::StaticVector{3}, d::StaticVector{3})
    t, x = _plane_hit(q, o, d)
    return _within(q, t, x)
end

"""
    HeightSurface(quad, nodes, heights, w, a)
    HeightSurface{K, T}(s::HeightSurface)

A height field over a [`Quad`](@ref): the points `o + u e_u + v e_v + h(u, v) n`
for `(u, v)` in the quad's chart, `h` the thin-plate spline through the `K`
nodal `heights` at the chart points `nodes`, with its weights `w` and affine
part `a` ([`tps_coefficients`](@ref); formed once, by whoever builds the
surface — [`surface`](@ref) takes them from the [`HeightField`](@ref)).
`HeightSurface{K, T}(s)` converts the element type.

isbits. [`intersection`](@ref) is Newton's method on
`g(t) = (o_ray + t d − o)·n − h(chart(o_ray + t d))`, started at the plane hit
and run for exactly four iterations (RadiometricCamera's count: the slope of
a print's height field is small, and the convergence quadratic); the hit is
`NaN` when the surface point it ends on lies outside [`bounds`](@ref).
[`normal`](@ref)`(s, x)` is the unit normal `(n − ∂h/∂u e_u − ∂h/∂v e_v) / ‖·‖`
from the spline's analytic gradient ([`tps_gradient`](@ref)), on the side of
the quad's normal.
"""
struct HeightSurface{K, T}
    quad::Quad{T}
    nodes::SVector{K, SVector{2, T}}
    heights::SVector{K, T}
    w::SVector{K, T}
    a::SVector{3, T}
end
function HeightSurface(quad::Quad, nodes::StaticVector{K, <:StaticVector{2}}, heights::StaticVector{K},
                       w::StaticVector{K}, a::StaticVector{3}) where {K}
    T = promote_type(eltype(eltype(nodes)), eltype(heights), eltype(w), eltype(a), eltype(quad.origin))
    return HeightSurface{K, T}(Quad{T}(quad), SVector{K, SVector{2, T}}(nodes), SVector{K, T}(heights),
                               SVector{K, T}(w), SVector{3, T}(a))
end
HeightSurface{K, T}(s::HeightSurface{K, T}) where {K, T} = s
HeightSurface{K, T}(s::HeightSurface{K}) where {K, T} =
    HeightSurface{K, T}(Quad{T}(s.quad), SVector{K, SVector{2, T}}(s.nodes), SVector{K, T}(s.heights), SVector{K, T}(s.w), SVector{3, T}(s.a))
Base.:(==)(a::HeightSurface, b::HeightSurface) =
    a.quad == b.quad && a.nodes == b.nodes && a.heights == b.heights && a.w == b.w && a.a == b.a

bounds(s::HeightSurface) = bounds(s.quad)
chart(s::HeightSurface, x::StaticVector{3}) = chart(s.quad, x)
@inline function normal(s::HeightSurface, x::StaticVector{3})
    q = s.quad
    g = tps_gradient(s.nodes, s.w, s.a, chart(q, x))
    return LinearAlgebra.normalize(normal(q) - g[1] * q.e_u - g[2] * q.e_v)
end

# RadiometricCamera's count (src/board/warping.jl `intersect_board`)
const _HEIGHT_NEWTON_ITERATIONS = 4

@inline function intersection(s::HeightSurface, o::StaticVector{3}, d::StaticVector{3})
    q = s.quad
    n = normal(q)
    den = dot(d, n)                       # g′ without the height's slope
    s0 = dot(q.origin - o, n)             # g(t) = t den − s0 − h(chart(x))
    t, x = _plane_hit(q, o, d)            # the seed
    cd = SVector(dot(d, q.e_u), dot(d, q.e_v))   # the chart velocity of the ray
    for _ in 1:_HEIGHT_NEWTON_ITERATIONS
        c = chart(q, x)
        g = t * den - s0 - tps_evaluate(s.nodes, s.w, s.a, c)
        gp = den - dot(tps_gradient(s.nodes, s.w, s.a, c), cd)
        t -= g / gp
        x = o + t * d
    end
    return _within(q, t, x)
end

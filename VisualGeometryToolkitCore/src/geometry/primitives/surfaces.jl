# =============================================================================
# The surface a ray meets: Quad (a bounded plane) and HeightSurface (a height
# field over one)
# =============================================================================
#
# Two isbits primitives behind one protocol, every method generic in the
# element type (Float32, Float64, ForwardDiff.Dual), allocation-free and
# usable inside a KernelAbstractions kernel:
#
#   intersection(s, o, d) -> (t, x)   the ray o + t d (t ≥ 0) meets s at x; NaN if it misses
#   normal(s, x)                      the unit normal of s at its point x
#   chart(s, x) -> (u, v)             the chart coordinates of the point x of s
#   surface_point(s, uv)              the point of s at chart coordinates uv (chart's inverse)
#   bounds(s) -> Rect                 the chart rectangle s covers
#
# A ray is an origin and a direction, two 3-vectors, as `ray_plane_intersection`
# takes it; there is no ray type. It is in the surface's frame (the posed
# target's destination frame F) and units (metres when physical).
#
# Two planar coordinate systems. The METRIC plane coordinates of x are
# m = ((x − o)·e_u, (x − o)·e_v), (e_u, e_v, n) a right-handed orthonormal
# frame (a posed target's TargetFrame: metres when physical). The CHART is
# the design's: m = A c + b, an in-plane affine fixed at construction
# (identity for a rigid target, the print's Affine otherwise), so
# chart(s, x) = A⁻¹ (m − b) — a pattern lookup uses nominal coordinates and
# the print's calibration is corrected, not ignored. The bounds are a
# rectangle in the chart. A HeightSurface's height is a function of m (a
# HeightField acts on the calibrated plane point): the points
# o + m₁ e_u + m₂ e_v + h(m) n.
#
# A ray misses when it is parallel to the plane, when the hit is behind its
# origin (t < 0: a ray is a half-line), when its hit's chart coordinates fall
# outside `bounds` (closed), or (HeightSurface) when Newton does not
# converge: t and every coordinate of x are then NaN.
#
# Kept here, not in the Primitives submodule, because the height field
# evaluates the thin-plate spline (included after Primitives), as bbox.jl is
# kept for the frames it needs. `surface(::PosedTarget)` builds either
# primitive from a posed target (geometry/cameras/target_surfaces.jl).
# =============================================================================

"""
    intersection(s, o, d) -> (t, x)

Where the ray `o + t d`, `t ≥ 0` (origin `o`, direction `d`, both 3-vectors;
`d` need not be unit) meets the surface `s`: the ray parameter `t` and the
point `x`. The ray and `x` are in the surface's frame — for
[`surface`](@ref)`(pt)` the pose's destination frame `F` — and its units
(metres when the target is physical). A ray that misses — parallel to a
plane, hitting behind its origin (`t < 0`), hitting outside
[`bounds`](@ref)`(s)`, or (a [`HeightSurface`](@ref)) not converging —
returns `t` and every coordinate of `x` as `NaN`, never an error. Methods: [`Quad`](@ref) (closed form, the arithmetic of
[`ray_plane_intersection`](@ref)) and [`HeightSurface`](@ref) (Newton).
Allocation-free and generic in the element type, so it runs on the GPU path
and under `ForwardDiff`.
"""
function intersection end

"""
    chart(s, x) -> SVector{2}

The chart coordinates `(u, v)` of the point `x` of the surface `s` — the
DESIGN chart (for [`surface`](@ref)`(pt)`, the coordinates of the target's
nominal points): the projection of `x − o` on the in-plane basis of `s`, then
the inverse of the surface's in-plane affine (the print's calibration). The
component along the normal is ignored, so for a [`HeightSurface`](@ref) they
are the coordinates under the surface point. `x` is in the surface's frame
and units, as for [`intersection`](@ref); [`surface_point`](@ref) is the
inverse.
"""
function chart end

"""
    surface_point(s, uv) -> SVector{3}

The point of the surface `s` at chart coordinates `uv` (the inverse of
[`chart`](@ref)): the in-plane affine applied, placed in the frame, lifted by
the height for a [`HeightSurface`](@ref). In the surface's frame and units.
"""
function surface_point end

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
normal the rotation's columns: the normal is TargetFrame +z, heights are
positive along it, and nothing is turned towards a viewer). Its CHART is the
design chart: the target's in-plane calibration (`Rigid` or `Affine`, or a
chain of them) is the quad's affine, so [`chart`](@ref) of a hit is the
nominal coordinates it shows. Its rectangle, in that chart, is
`Rect(`[`board`](@ref)`(t))` when the target has a [`Board`](@ref), else the
design bounding box grown by 16 eps (of its coordinates), so a ray at a design
point on the box's edge is a hit.

- A planar calibration (`Rigid`, `Affine` and chains of them) gives a `Quad`.
- A [`HeightField`](@ref), alone or last in a [`CalibrationChain`](@ref) after
  those, gives a [`HeightSurface`](@ref) over that `Quad` with the field's
  nodes and heights (in the calibrated plane coordinates, TargetFrame
  `(x, y)`) and the weights the field formed once, when it was built; nothing
  is solved per ray.
- An in-plane [`ThinPlateSpline`](@ref) member is not supported by the ray
  primitives yet: an `ArgumentError` naming it.

A physical target (metres) gives a primitive in metres, plain numbers. An
uncertain pose or calibration gives its value. A `PointTarget` has no surface
(a `MethodError`). The kind is chosen by the calibration's type, so a rigid
target's ray caster never compiles the height field's Newton loop.
"""
function surface end   # the method: geometry/cameras/target_surfaces.jl

"""
    Quad(origin, frame, rect::Rect2, A = I, b = 0)
    Quad{T}(q::Quad)

A bounded plane. `frame` is a right-handed orthonormal 3 × 3 matrix (a
rotation) whose columns are the in-plane basis `e_u`, `e_v` and the normal
`n = e_u × e_v`; a point with metric plane coordinates `m` is
`origin + m₁ e_u + m₂ e_v`. The CHART is `c` with `m = A c + b` (`A` an
invertible 2 × 2 matrix, inverted once here), and `rect` is a rectangle in
the chart. A left-handed or non-orthonormal `frame` (to `64 eps` of the
frame's element type: 1.4e-14 in `Float64`, 7.6e-6 in `Float32`), a
singular `A` or a negative width is an `ArgumentError`. `Quad{T}(q)` converts
the element type (`Float32` for a device).

isbits. [`intersection`](@ref) is [`ray_plane_intersection`](@ref)'s
arithmetic, bit for bit, `NaN` when `t < 0` or the hit's chart is outside
`rect`; [`chart`](@ref)`(q, x) = A⁻¹ (m − b)`; [`surface_point`](@ref) its
inverse; [`bounds`](@ref) is `rect`. [`normal`](@ref) is the GEOMETRIC normal
`n` on either side: a consumer that wants the normal facing the camera flips
it against the ray (shading, not geometry). A rigid or affine-calibrated
target's [`surface`](@ref) is a `Quad`.

Under ChainRules (reverse mode) the plane is a constant of
`ray_plane_intersection`, so a `Quad`'s intersection carries no gradient to
its pose; ForwardDiff does.
"""
struct Quad{T}
    plane::Plane3D{T}
    origin::SVector{3, T}
    e_u::SVector{3, T}
    e_v::SVector{3, T}
    A::SMatrix{2, 2, T, 4}        # chart → metric plane coordinates, m = A c + b
    b::SVector{2, T}
    A_inv::SMatrix{2, 2, T, 4}    # formed once, here
    rect::Rect2{T}                # in the chart
end
# the frame's orthonormality, in eps of its own element type (a Float32 pose
# is orthonormal to Float32 rounding, whatever the Quad's type)
_frame_tolerance(::Type{T}) where {T} = 64 * eps(float(T))
# no affine: the identity, in the type of the frame data (a Float32 frame gives a Float32 Quad)
function Quad(origin::StaticVector{3}, frame::StaticMatrix{3, 3}, rect::Rect2)
    T = float(promote_type(eltype(origin), eltype(frame), eltype(rect.origin)))
    return Quad(origin, frame, rect, SMatrix{2, 2, T}(1, 0, 0, 1), zero(SVector{2, T}))
end
function Quad(origin::StaticVector{3}, frame::StaticMatrix{3, 3}, rect::Rect2, A::StaticMatrix{2, 2},
              b::StaticVector{2} = zero(SVector{2, float(eltype(A))}))
    T = float(promote_type(eltype(origin), eltype(frame), eltype(rect.origin), eltype(A), eltype(b)))
    R = SMatrix{3, 3, T}(frame)
    tol = _frame_tolerance(eltype(frame))
    maximum(abs, R' * R - I) <= tol ||
        throw(ArgumentError("Quad: the frame (e_u, e_v, n) must be orthonormal to $(tol)"))
    det(R) > 0 || throw(ArgumentError("Quad: the frame (e_u, e_v, n) is left-handed; the normal is e_u × e_v"))
    all(>=(0), rect.widths) || throw(ArgumentError("Quad: the rectangle has a negative width, $(rect.widths)"))
    Aₜ = SMatrix{2, 2, T}(A)
    dA = det(Aₜ)
    (isfinite(dA) && !iszero(dA)) || throw(ArgumentError("Quad: the chart's affine A is singular"))
    o, u, v = SVector{3, T}(origin), R[:, 1], R[:, 2]
    plane = Plane3D(Point3{T}(o...), Vec3{T}(cross(u, v)...))
    return Quad{T}(plane, o, u, v, Aₜ, SVector{2, T}(b), inv(Aₜ), Rect2{T}(rect))
end
Quad{T}(q::Quad{T}) where {T} = q
Quad{T}(q::Quad) where {T} = Quad{T}(Plane3D{T}(q.plane.coeffs), q.origin, q.e_u, q.e_v, q.A, q.b, q.A_inv, Rect2{T}(q.rect))
_quad_fields(q::Quad) = (q.plane.coeffs, q.origin, q.e_u, q.e_v, q.A, q.b, q.A_inv, q.rect)
Base.:(==)(a::Quad, b::Quad) = _quad_fields(a) == _quad_fields(b)
Base.hash(q::Quad, h::UInt) = hash(_quad_fields(q), hash(:Quad, h))

normal(q::Quad) = normal(q.plane)
normal(q::Quad, x::StaticVector{3}) = normal(q.plane)
bounds(q::Quad) = q.rect
# the metric plane coordinates m of x
@inline function _plane_coordinates(q::Quad, x::StaticVector{3})
    δ = x - q.origin
    return SVector(dot(δ, q.e_u), dot(δ, q.e_v))
end
@inline chart(q::Quad, x::StaticVector{3}) = q.A_inv * (_plane_coordinates(q, x) - q.b)
@inline function surface_point(q::Quad, uv::StaticVector{2})
    m = q.A * uv + q.b
    return q.origin + m[1] * q.e_u + m[2] * q.e_v
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

# (t, x) for a hit in front of the origin whose chart coordinates lie in the
# rectangle, NaN otherwise.
@inline function _within(q::Quad, t, x::StaticVector{3}, ok = true)
    inside = ok & (t >= 0) & _in_rect(q.rect, chart(q, x))
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

A height field over a [`Quad`](@ref): the points `o + m₁ e_u + m₂ e_v + h(m) n`
for `m` in the quad's metric plane coordinates, `h` the thin-plate spline
through the `K` nodal `heights` at the plane points `nodes` (metric plane
coordinates: a target's calibrated TargetFrame `(x, y)`, which is the design
chart when the quad's affine is the identity), with its weights `w` and
affine part `a` ([`tps_coefficients`](@ref); formed once, by whoever builds
the surface — [`surface`](@ref) takes them from the [`HeightField`](@ref)).
Heights are positive along the quad's normal. `HeightSurface{K, T}(s)`
converts the element type.

isbits. [`intersection`](@ref) is Newton's method on
`g(t) = (o_ray + t d − o)·n − h(m(o_ray + t d))`, started at the plane hit:
four iterations unconditionally (RadiometricCamera's count, so a ray that
has converged by then is bit-identical to it), then up to sixteen more while
the update is above rounding. The update `Δt` counts as rounding when
`|Δt|·‖d‖ ≤ 64 eps(T) · (L + |t|‖d‖)` (it is then not applied), `L` the
diagonal of [`bounds`](@ref) in metric plane coordinates (the quad's affine
applied to its widths) — a length scale of the surface; the iteration
also counts as converged if its last applied update was below
`√eps(T) · (L + |t|‖d‖)` (rounding amplified at grazing incidence).
Otherwise the ray is a miss (`NaN`), as is one whose final surface point is
behind the origin or outside `bounds`. The four iterations alone suffice
while the height's slope × tan(incidence) is well below 1 (a print's warp);
the continuation covers steep or grazing cases.

[`normal`](@ref)`(s, x)` is the unit normal `(n − ∂h/∂m₁ e_u − ∂h/∂m₂ e_v) / ‖·‖`
from the spline's analytic gradient ([`tps_gradient`](@ref)), on the side of
the quad's normal (geometric; flip it against the ray for a camera-facing
one). [`chart`](@ref) and [`surface_point`](@ref) are the quad's, the latter
lifted by the height.
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
_height_surface_fields(s::HeightSurface) = (s.quad, s.nodes, s.heights, s.w, s.a)
Base.:(==)(a::HeightSurface, b::HeightSurface) = _height_surface_fields(a) == _height_surface_fields(b)
Base.hash(s::HeightSurface, h::UInt) = hash(_height_surface_fields(s), hash(:HeightSurface, h))

bounds(s::HeightSurface) = bounds(s.quad)
chart(s::HeightSurface, x::StaticVector{3}) = chart(s.quad, x)
@inline _height(s::HeightSurface, m::StaticVector{2}) = tps_evaluate(s.nodes, s.w, s.a, m)
# The height and its gradient at `m` computed together, one r² and one log per
# node: `tps_evaluate`'s value and `tps_gradient`'s gradient term for
# term, in their operation order (the nodes converted to the gradient's element
# type, as `tps_gradient` converts them: the same r² as `tps_evaluate`'s for the
# surface's own element type), so Newton's iterates keep their bits.
@inline function _height_and_gradient(s::HeightSurface{K}, m::StaticVector{2}) where {K}
    w, a = s.w, s.a
    S = promote_type(eltype(w), eltype(m))
    h = a[1] + a[2] * m[1] + a[3] * m[2]
    g1 = zero(S); g2 = zero(S)
    @inbounds for i in 1:K
        n = s.nodes[i]
        d1 = m[1] - S(n[1]); d2 = m[2] - S(n[2])
        r2 = d1 * d1 + d2 * d2
        if r2 > 0
            l = log(r2)
            h += w[i] * (r2 * l / 2)
            f = w[i] * (l + 1)
        else
            h += w[i] * zero(r2)
            f = zero(S)
        end
        g1 += f * d1; g2 += f * d2
    end
    return h, SVector(a[2] + g1, a[3] + g2)
end
@inline function surface_point(s::HeightSurface, uv::StaticVector{2})
    q = s.quad
    m = q.A * uv + q.b
    return q.origin + m[1] * q.e_u + m[2] * q.e_v + _height(s, m) * normal(q)
end
@inline function normal(s::HeightSurface, x::StaticVector{3})
    q = s.quad
    g = tps_gradient(s.nodes, s.w, s.a, _plane_coordinates(q, x))
    return LinearAlgebra.normalize(normal(q) - g[1] * q.e_u - g[2] * q.e_v)
end

# RadiometricCamera's count (src/board/warping.jl `intersect_board`), then the
# continuation's cap.
const _HEIGHT_NEWTON_ITERATIONS = 4
const _HEIGHT_NEWTON_MAX_ITER = 20

# Newton's g(t) = t den − s0 − h(m(x)) and g′(t) = den − ∇h·cd at the ray point
# x = o + t d, the height and its gradient computed together.
@inline function _newton_terms(s::HeightSurface, den, s0, cd, t, x)
    h, ∇h = _height_and_gradient(s, _plane_coordinates(s.quad, x))
    return t * den - s0 - h, den - dot(∇h, cd)
end

@inline function intersection(s::HeightSurface, o::StaticVector{3}, d::StaticVector{3})
    q = s.quad
    n = normal(q)
    den = dot(d, n)                       # g′ without the height's slope
    s0 = dot(q.origin - o, n)             # g(t) = t den − s0 − h(m(x))
    t, x = _plane_hit(q, o, d)            # the seed
    cd = SVector(dot(d, q.e_u), dot(d, q.e_v))   # the plane velocity of the ray
    for _ in 1:_HEIGHT_NEWTON_ITERATIONS
        g, gp = _newton_terms(s, den, s0, cd, t, x)
        t -= g / gp
        x = o + t * d
    end
    # the continuation: stop at a rounding-level update, which is not applied
    # (a converged ray keeps RadiometricCamera's four-iteration bits)
    nd = norm(d)
    w = q.A * q.rect.widths               # the bounds' widths in metric plane coordinates
    L = hypot(w[1], w[2])
    ε = eps(typeof(t))                   # a Dual's is its value type's (ForwardDiff)
    converged = false
    last = oftype(abs(t), Inf)
    for _ in (_HEIGHT_NEWTON_ITERATIONS + 1):_HEIGHT_NEWTON_MAX_ITER
        g, gp = _newton_terms(s, den, s0, cd, t, x)
        Δ = g / gp
        scale = L + abs(t) * nd
        if abs(Δ) * nd <= 64 * ε * scale
            converged = true
            break
        end
        last = abs(Δ) * nd / scale
        t -= Δ
        x = o + t * d
    end
    converged |= last <= sqrt(ε)
    return _within(q, t, x, converged)
end

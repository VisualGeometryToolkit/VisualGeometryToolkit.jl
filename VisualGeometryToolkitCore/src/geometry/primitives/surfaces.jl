# =============================================================================
# The surface a ray meets: the protocol, and Quad (a bounded plane)
# =============================================================================
#
# One protocol, every method generic in the element type (Float32, Float64,
# ForwardDiff.Dual), allocation-free and usable inside a KernelAbstractions
# kernel:
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
# Two primitives: the closed-form Quad here, and the MappedSurface of a chart
# map (geometry/cameras/mapped_surface.jl: a deformed target, solved beside
# the spline models' LM). For a Quad, the METRIC plane coordinates of x are
# m = ((x − o)·e_u, (x − o)·e_v), (e_u, e_v, n) a right-handed orthonormal
# frame (a posed target's TargetFrame: metres when physical), and its CHART is
# the design's: m = A c + b, an in-plane affine fixed at construction
# (identity for a rigid target, the print's affine factors otherwise), so
# chart(s, x) = A⁻¹ (m − b). The bounds are a rectangle in the chart.
#
# A ray misses when it is parallel to the plane, when the hit is behind its
# origin (t < 0: a ray is a half-line), when its hit's chart coordinates fall
# outside `bounds` (closed), or (MappedSurface) when its root does not
# converge: t and every coordinate of x are then NaN.
# =============================================================================

"""
    intersection(s, o, d) -> (t, x)

Where the ray `o + t d`, `t ≥ 0` (origin `o`, direction `d`, both 3-vectors;
`d` need not be unit) meets the surface `s`: the ray parameter `t` and the
point `x`. The ray and `x` are in the surface's frame — for
[`surface`](@ref)`(pt)` the pose's destination frame `F` — and its units
(metres when the target is physical). A ray that misses — parallel to a
plane, hitting behind its origin (`t < 0`), hitting outside
[`bounds`](@ref)`(s)`, or (a [`MappedSurface`](@ref)) not converging —
returns `t` and every coordinate of `x` as `NaN`, never an error. Methods:
[`Quad`](@ref) (closed form, the arithmetic of
[`ray_plane_intersection`](@ref)) and [`MappedSurface`](@ref) (a 2 × 2 root in
its chart). Allocation-free and generic in the element type, so it runs on the
GPU path and under `ForwardDiff` (a MappedSurface's root by the implicit
function theorem).
"""
function intersection end

"""
    chart(s, x) -> SVector{2}

The chart coordinates `(u, v)` of the point `x` of the surface `s` — the
DESIGN chart (for [`surface`](@ref)`(pt)`, the coordinates of the target's
nominal points). For a [`Quad`](@ref) the projection of `x − o` on its
in-plane basis, then the inverse of its in-plane affine (the print's
calibration; the component along the normal is ignored); for a
[`MappedSurface`](@ref) the chart point whose image is `x`. `x` is in the
surface's frame and units, as for [`intersection`](@ref);
[`surface_point`](@ref) is the inverse.
"""
function chart end

"""
    surface_point(s, uv) -> SVector{3}

The point of the surface `s` at chart coordinates `uv` (the inverse of
[`chart`](@ref)): a [`Quad`](@ref)'s in-plane affine applied and placed in its
frame, a [`MappedSurface`](@ref)'s chart map. In the surface's frame and units.
"""
function surface_point end

"""
    bounds(s) -> Rect2

The rectangle of chart coordinates the surface `s` covers ([`chart`](@ref)).
"""
function bounds end

"""
    surface(pt::PosedTarget{<:PlanarTarget}) -> Quad | MappedSurface

The surface a ray meets on the posed planar target `pt`, in the pose's
destination frame, its CHART the design chart (so [`chart`](@ref) of a hit is
the nominal coordinates it shows). Its rectangle, in that chart, is
`Rect(`[`board`](@ref)`(t))` when the target has a [`Board`](@ref), else the
design bounding box — either grown by 16 eps of its coordinates, the margin
within which `PlanarTarget` accepts a point on its board, so a ray at a point
on the edge is a hit (up to the chart's own rounding).

- A planar affine calibration (`Rigid`, the factors `EuclideanMap ∘ Scale ∘
  Aspect ∘ Skew` or any of them) gives a [`Quad`](@ref): the target's
  TargetFrame placed by `pose(pt)` (origin its translation, `e_u`, `e_v` and
  the normal the rotation's columns), the calibration its affine (placed:
  `A p + b`, `b = s β − (A − I) o`).
- Any other — a [`HeightField`](@ref), an in-plane [`ThinPlateSpline`](@ref),
  either after the affine factors (`HeightField ∘ ThinPlateSpline ∘ Aspect ∘
  Skew`) — gives the [`MappedSurface`](@ref) of the composed chart map
  `pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹` (the pose, the [`placement`](@ref) `S₃` and
  its in-plane part `S₂`, the calibration `C`, the lift `(c, 0)` of a planar
  one), built by `MappedSurface(pose, calibration, placement, rect)`. Heights
  are along TargetFrame +z.

A physical target (metres) gives a primitive in metres, plain numbers. An
uncertain pose or calibration gives its value. A `PointTarget` has no surface
(a `MethodError`). The kind is chosen by the calibration's type, so a rigid
target's ray caster never compiles the root.
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
# The quad of an orthonormal frame with the identity chart affine, unchecked —
# the checked constructor's numbers exactly, without its checks' messages: a
# device builder's (a MappedSurface's frame, geometry/cameras/target_surfaces.jl).
@inline function _frame_quad(origin::StaticVector{3}, frame::StaticMatrix{3, 3}, rect::Rect2)
    T = float(promote_type(eltype(origin), eltype(frame), eltype(rect.origin)))
    R = SMatrix{3, 3, T}(frame)
    o, u, v = SVector{3, T}(origin), R[:, 1], R[:, 2]
    I2 = SMatrix{2, 2, T}(1, 0, 0, 1)
    return Quad{T}(Plane3D(Point3{T}(o...), Vec3{T}(cross(u, v)...)), o, u, v, I2, zero(SVector{2, T}), I2, Rect2{T}(rect))
end
Quad{T}(q::Quad) where {T} = _map_numbers(x -> convert(T, x), q, T)
# Every number of q mapped by f (a conversion, or a Dual's value), as a Quad{T}:
# the requested T (a conversion's, concrete or not), else the mapped numbers' type.
@inline _map_numbers(f::F, q::Quad) where {F} = (c = map(f, q.plane.coeffs); _map_numbers(f, q, eltype(c), c))
@inline function _map_numbers(f::F, q::Quad, ::Type{T}, c = map(f, q.plane.coeffs)) where {F, T}
    return Quad{T}(Plane3D{T}(c), map(f, q.origin), map(f, q.e_u), map(f, q.e_v), map(f, q.A), map(f, q.b), map(f, q.A_inv),
                   Rect2{T}(map(f, q.rect.origin), map(f, q.rect.widths)))
end
_eltype(::Quad{T}) where {T} = T
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
@inline _within(q::Quad, t, x::StaticVector{3}, ok = true) = _masked(ok & (t >= 0) & _in_rect(q.rect, chart(q, x)), t, x)
# (t, x) if ok, else NaN (a Dual's with zero partials); branch-free.
@inline function _masked(ok::Bool, t, x::StaticVector{3})
    nan = oftype(t, NaN)
    return ifelse(ok, t, nan), ifelse(ok, x, SVector(nan, nan, nan))
end

@inline function intersection(q::Quad, o::StaticVector{3}, d::StaticVector{3})
    t, x = _plane_hit(q, o, d)
    return _within(q, t, x)
end

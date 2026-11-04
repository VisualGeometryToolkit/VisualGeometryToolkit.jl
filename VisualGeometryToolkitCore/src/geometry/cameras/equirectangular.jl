# =============================================================================
# Panorama projections — the cylindrical family: Equirectangular, Cylindrical,
# CylindricalEqualArea, Mercator — and Backprojection, a back-projection as a map
# =============================================================================
#
# The maps of the sphere of directions onto a (longitude, vertical) rectangle —
# the projection of a panorama raster, a sky map, an environment map. They are
# not lenses: they have no radial profile, no distortion and no camera behind
# them, so they are projections of their own beside the lens families
# (lenses.jl) rather than `AbstractProjection`s, whose contract is a radially
# symmetric ray → frame map. All four share the longitude and differ in the
# vertical coordinate `v` of the latitude `φ`: `φ` (equirectangular), `tan φ`
# (central cylindrical), `sin φ` (Lambert's cylindrical equal-area) and
# `asinh(tan φ)` (Mercator).
#
# `project`/`backproject` are their interface; each is a `Transformation`
# (its call is `project`, with `transform_deriv`) whose `inv` is its
# [`Backprojection`](@ref), so a panorama raster's pixel → ray map composes into
# an `image_warp` chain: `camera ∘ LinearMap(R') ∘ inv(projection) ∘ raster`.
# `Backprojection(camera)` does the same for a camera model, the virtual camera
# of an azimuthal (rectilinear, fisheye, stereographic) view.
#
# Axes are the camera frame's (x right, y down, z forward): longitude is the
# azimuth about the up axis −y, zero forward and +π/2 right; the vertical
# coordinate grows upwards (towards −y). A raster in pixels is an affine map of
# (longitude, vertical) chosen by the caller.
# =============================================================================

"""
    AbstractPanoramaProjection <: CoordinateTransformations.Transformation

A projection of directions onto a (longitude, vertical) rectangle, the
cylindrical family: [`Equirectangular`](@ref), [`Cylindrical`](@ref),
[`CylindricalEqualArea`](@ref), [`Mercator`](@ref). The call `p(X)` is
[`project`](@ref)`(p, X)` (its `transform_deriv(p, X)` the `2 × 3` Jacobian),
and `inv(p)` is its [`Backprojection`](@ref) (`inv(p)(q) == backproject(p, q)`,
`inv(inv(p)) === p`). All are kernel-lawful (singletons, no allocation, no
throw), so a chain built from them runs in `image_warp!` and
[`pointwise!`](@ref) on any backend.
"""
abstract type AbstractPanoramaProjection <: CoordinateTransformations.Transformation end

"""
    Backprojection(m) <: CoordinateTransformations.Transformation

The back-projection of `m` as a map: its call is [`backproject`](@ref)`(m, q)`
and `transform_deriv(b, q)` its `3 × 2` Jacobian (a camera model's own,
`last(backproject_with_jacobian(m, q))`; a panorama projection's by
ForwardDiff through its `backproject`). `m` is a
panorama projection (`inv(p)` builds it, `inv(Backprojection(p)) === p`) or a
camera model, a pixel → ray map: `Backprojection(model)` of a
`CameraModel(...; lens = Lens(Perspective()))` is a rectilinear virtual camera,
of `Lens(Equidistant())` or `Lens(Stereographic())` a fisheye one, and
`LinearMap(R) ∘ Backprojection(model)` aims it. Kernel-lawful when `m`'s
`backproject` is.
"""
struct Backprojection{M} <: CoordinateTransformations.Transformation
    model::M
end

@inline (p::AbstractPanoramaProjection)(X::StaticVector{3}) = project(p, X)
@inline (b::Backprojection)(q::StaticVector{2}) = backproject(b.model, q)
Base.inv(p::AbstractPanoramaProjection) = Backprojection(p)
Base.inv(b::Backprojection{<:AbstractPanoramaProjection}) = b.model
Base.show(io::IO, b::Backprojection) = print(io, "Backprojection(", b.model, ")")
# A camera model's own back-projection Jacobian (tangent to the sphere); a
# panorama projection's by ForwardDiff through its `backproject`.
transform_deriv(b::Backprojection, q::StaticVector{2}) = last(backproject_with_jacobian(b.model, q))
transform_deriv(b::Backprojection{<:AbstractPanoramaProjection}, q::StaticVector{2}) =
    last(_value_and_jacobian(p -> SVector{3}(Tuple(backproject(b.model, p))), _float_static(q)))

# The shared parts of the family: the longitude, its gradient, and the guard
# (a zero, non-finite or — where `pole_ok` is false — polar vector has no
# coordinates).
@inline function _panorama_coordinates(X::StaticVector{3}, vertical::F, pole_ok::Bool) where {F}
    x, y, z = float(X[1]), float(X[2]), float(X[3])
    ρ = hypot(x, z)
    T = typeof(ρ)
    (isfinite(ρ) && isfinite(y) && (ρ > 0 || (pole_ok && y != 0))) || return SVector{2, T}(T(NaN), T(NaN))
    return SVector{2, T}(atan(x, z), vertical(x, y, z, ρ))
end
# ∂λ/∂X and the vertical row (as a function of x, y, z, ρ, r = ‖X‖) stacked.
@inline function _panorama_jacobian(X::StaticVector{3}, vertical_row::F) where {F}
    x, y, z = float(X[1]), float(X[2]), float(X[3])
    ρ2 = x * x + z * z; ρ = sqrt(ρ2); r = sqrt(ρ2 + y * y)
    gv = vertical_row(x, y, z, ρ, r)
    return SMatrix{2, 3}(z / ρ2, gv[1], zero(ρ), gv[2], -x / ρ2, gv[3])
end

"""
    Equirectangular()

The equirectangular (latitude/longitude) projection of directions, in the
camera frame's axes (x right, y down, z forward, up = −y).

- [`project`](@ref)`(Equirectangular(), X)` — the direction `X` (any nonzero
  3-vector; its length does not enter) to `SVector(λ, φ)` in radians: the
  longitude `λ = atan(X₁, X₃) ∈ (−π, π]` (0 forward, +π/2 right) and the
  latitude `φ = atan(−X₂, √(X₁² + X₃²)) ∈ [−π/2, π/2]` (+π/2 up). NaN for a
  zero or non-finite vector. At the poles the longitude is `atan`'s value of
  the remaining components (0 for an exact pole).
- [`backproject`](@ref)`(Equirectangular(), p)` — `p = (λ, φ)` to the unit
  direction `(cos φ sin λ, −sin φ, cos φ cos λ)`, defined for every `p`
  (periodic in `λ`).

A panorama raster is an affine map of `(λ, φ)` to pixels, and a rotation `R`
with `d_world = R·d` puts the projection in another frame:
`project(Equirectangular(), R' * d_world)`. An [`AbstractPanoramaProjection`](@ref).
"""
struct Equirectangular <: AbstractPanoramaProjection end

project(::Equirectangular, X::StaticVector{3}) = _panorama_coordinates(X, (x, y, z, ρ) -> atan(-y, ρ), true)

function backproject(::Equirectangular, p::StaticVector{2})
    λ, φ = float(p[1]), float(p[2])
    sλ, cλ = sincos(λ)
    sφ, cφ = sincos(φ)
    return Vec3(cφ * sλ, -sφ, cφ * cλ)
end

transform_deriv(::Equirectangular, X::StaticVector{3}) =
    _panorama_jacobian(X, (x, y, z, ρ, r) -> SVector(y * x / (ρ * r^2), -ρ / r^2, y * z / (ρ * r^2)))

"""
    Cylindrical()

The central cylindrical projection of directions onto the unit cylinder about
the up axis, in the camera frame's axes (x right, y down, z forward, up = −y):
the longitude about the axis, and the height `h = tan φ` at which the ray
through the origin meets the cylinder of radius 1 (`φ` the latitude). A
cylindrical panorama raster is an affine map of `(λ, h)`: its columns are the
longitude, its rows the height, so verticals stay vertical and straight.

- [`project`](@ref)`(Cylindrical(), X)` — the direction `X` (any 3-vector off
  the up axis; its length does not enter) to `SVector(λ, h)`: the longitude
  `λ = atan(X₁, X₃) ∈ (−π, π]` (0 forward, +π/2 right) and `h = −X₂ / √(X₁² + X₃²)`
  (+ up). NaN for a zero, non-finite or polar (`X₁ = X₃ = 0`) vector: the
  poles have no height.
- [`backproject`](@ref)`(Cylindrical(), p)` — `p = (λ, h)` to the unit
  direction `(sin λ, −h, cos λ) / √(1 + h²)`, defined for every finite `p`
  (periodic in `λ`).

Its longitude is [`Equirectangular`](@ref)'s, and `h = tan φ` of its latitude.
An [`AbstractPanoramaProjection`](@ref).
"""
struct Cylindrical <: AbstractPanoramaProjection end

project(::Cylindrical, X::StaticVector{3}) = _panorama_coordinates(X, (x, y, z, ρ) -> -y / ρ, false)

function backproject(::Cylindrical, p::StaticVector{2})
    λ, h = float(p[1]), float(p[2])
    sλ, cλ = sincos(λ)
    n = inv(sqrt(1 + h * h))
    return Vec3(sλ * n, -h * n, cλ * n)
end

transform_deriv(::Cylindrical, X::StaticVector{3}) =
    _panorama_jacobian(X, (x, y, z, ρ, r) -> SVector(y * x / ρ^3, -1 / ρ, y * z / ρ^3))

"""
    CylindricalEqualArea()

Lambert's cylindrical equal-area projection of directions, in the camera
frame's axes (x right, y down, z forward, up = −y): the longitude and
`v = sin φ` of the latitude — the height at which the ray's parallel meets the
unit cylinder when projected ALONG the planes of constant height (orthogonally
onto the axis' cylinder), so equal areas of the raster are equal solid angles
(`dΩ = cos φ dφ dλ = dv dλ`): a uniform raster's pixels all subtend the same
solid angle.

- [`project`](@ref)`(CylindricalEqualArea(), X)` — `X` (any nonzero 3-vector;
  its length does not enter) to `SVector(λ, v)`, `λ = atan(X₁, X₃)` and
  `v = −X₂ / ‖X‖ ∈ [−1, 1]`. NaN for a zero or non-finite vector; the poles
  are `v = ±1` with `atan`'s longitude.
- [`backproject`](@ref)`(CylindricalEqualArea(), p)` — `p = (λ, v)` to the
  unit direction `(√(1 − v²) sin λ, −v, √(1 − v²) cos λ)`; NaN for `|v| > 1`.

An [`AbstractPanoramaProjection`](@ref).
"""
struct CylindricalEqualArea <: AbstractPanoramaProjection end

project(::CylindricalEqualArea, X::StaticVector{3}) = _panorama_coordinates(X, (x, y, z, ρ) -> -y / hypot(ρ, y), true)

function backproject(::CylindricalEqualArea, p::StaticVector{2})
    λ, v = float(p[1]), float(p[2])
    T = typeof(v)
    abs(v) <= 1 || return Vec3(T(NaN), T(NaN), T(NaN))
    sλ, cλ = sincos(λ)
    c = sqrt(1 - v * v)
    return Vec3(c * sλ, -v, c * cλ)
end

transform_deriv(::CylindricalEqualArea, X::StaticVector{3}) =
    _panorama_jacobian(X, (x, y, z, ρ, r) -> SVector(y * x / r^3, -ρ^2 / r^3, y * z / r^3))

"""
    Mercator()

The Mercator projection of directions, in the camera frame's axes (x right,
y down, z forward, up = −y): the longitude and `v = asinh(tan φ)` of the
latitude — the conformal member of the cylindrical family (angles are kept
locally; the scale grows as `1/cos φ`).

- [`project`](@ref)`(Mercator(), X)` — `X` (any 3-vector off the up axis; its
  length does not enter) to `SVector(λ, v)`, `λ = atan(X₁, X₃)` and
  `v = asinh(−X₂ / √(X₁² + X₃²))`. NaN for a zero, non-finite or polar vector.
- [`backproject`](@ref)`(Mercator(), p)` — `p = (λ, v)` to the unit direction
  `(sin λ / cosh v, −tanh v, cos λ / cosh v)`, defined for every finite `p`.

An [`AbstractPanoramaProjection`](@ref).
"""
struct Mercator <: AbstractPanoramaProjection end

project(::Mercator, X::StaticVector{3}) = _panorama_coordinates(X, (x, y, z, ρ) -> asinh(-y / ρ), false)

function backproject(::Mercator, p::StaticVector{2})
    λ, v = float(p[1]), float(p[2])
    sλ, cλ = sincos(λ)
    c = inv(cosh(v))
    return Vec3(sλ * c, -tanh(v), cλ * c)
end

transform_deriv(::Mercator, X::StaticVector{3}) =
    _panorama_jacobian(X, (x, y, z, ρ, r) -> SVector(y * x / (ρ^2 * r), -1 / r, y * z / (ρ^2 * r)))

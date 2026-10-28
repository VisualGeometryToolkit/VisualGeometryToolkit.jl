# =============================================================================
# Equirectangular — the latitude/longitude projection of directions
# =============================================================================
#
# The map of the whole sphere of directions onto the (longitude, latitude)
# rectangle — the projection of a panorama raster, a sky map, an environment
# map. It is not a lens: it has no radial profile, no distortion and no
# camera behind it, so it is a projection of its own beside the lens families
# (lenses.jl) rather than an `AbstractProjection`, whose contract is a radially
# symmetric ray → frame map. `project`/`backproject` are its whole interface.
#
# Axes are the camera frame's (x right, y down, z forward): longitude is the
# azimuth about the up axis −y, zero forward and +π/2 right; latitude is the
# elevation above the x–z plane, positive up (towards −y). A raster in pixels is
# an affine map of (longitude, latitude) chosen by the caller.
# =============================================================================

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
`project(Equirectangular(), R' * d_world)`.
"""
struct Equirectangular end

function project(::Equirectangular, X::StaticVector{3})
    x, y, z = float(X[1]), float(X[2]), float(X[3])
    ρ = hypot(x, z)
    T = typeof(ρ)
    (isfinite(ρ) && isfinite(y) && (ρ > 0 || y != 0)) || return SVector{2, T}(T(NaN), T(NaN))
    return SVector{2, T}(atan(x, z), atan(-y, ρ))
end

function backproject(::Equirectangular, p::StaticVector{2})
    λ, φ = float(p[1]), float(p[2])
    sλ, cλ = sincos(λ)
    sφ, cφ = sincos(φ)
    return Vec3(cφ * sλ, -sφ, cφ * cλ)
end

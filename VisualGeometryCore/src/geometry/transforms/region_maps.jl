# =============================================================================
# Log-Polar Transform
# =============================================================================

"""
    LogPolarToCartesian{T<:Real} <: CoordinateTransformations.Transformation

Transformation from log-polar to Cartesian coordinates in normalized unit
space. Subtypes `CoordinateTransformations.Transformation` so it composes
into a `ComposedTransformation` (not `Base.ComposedFunction`) — required
for the `BoundedMap(::Transformation, ::Rect)` constructor used by
`image_warp`.
"""
struct LogPolarToCartesian{T <: Real} <: CoordinateTransformations.Transformation
    log_inner_radius::T
end

@inline function (t::LogPolarToCartesian)(p)
    θ = π * (p[2] + 1)                  # Maps [-1,1] → [0, 2π]
    normalized_log_r = (p[1] + 1) / 2   # Maps [-1,1] → [0,1]
    log_r = t.log_inner_radius + (-t.log_inner_radius) * normalized_log_r
    r = exp(log_r)
    return r * Point2(cos(θ), sin(θ))
end

"""
    logpolar_to_cartesian(inner_radius=0.01) -> LogPolarToCartesian

Construct a `LogPolarToCartesian` transformation. Maps points in unit
space [-1,1]² interpreted as (log_r, θ) to Cartesian (x, y) in the unit
circle.

# Coordinate Convention
- x-axis (horizontal/columns) = scale (radius)
- y-axis (vertical/rows) = rotation (angle)

# Mapping
- Input x-coordinate in [-1,1] → radius from inner_radius to 1.0 (logarithmically)
- Input y-coordinate in [-1,1] → angle from 0 to 2π
- Output: Cartesian (x, y) in unit circle (radius ≤ 1.0)

# Arguments
- `inner_radius::Real`: Minimum radius (default: 0.01, avoids singularity at origin)

# Example
```julia
transform = logpolar_to_cartesian()
transform(Point2(0.0, 0.0))   # mid-radius, π angle
transform(Point2(-1.0, -1.0)) # min radius, 0
transform(Point2(1.0, 1.0))   # max radius (1.0), 2π
```
"""
logpolar_to_cartesian(inner_radius::Real = 0.01) =
    LogPolarToCartesian(log(Float64(inner_radius)))

"""
    logpolar_map(geometry::Union{Circle, Ellipse}, logpolar_rect::HyperRectangle{2}, inner_radius::Real) -> Function

Create a composed transformation that maps from log-polar patch coordinates to image coordinates
via a circle or ellipse.

This is a convenience function that composes three transformations:
1. `logpolar_rect` → `CANONICAL_SQUARE` (normalize patch coordinates)
2. `CANONICAL_SQUARE` → `UNIT_CIRCLE` via log-polar transform
3. `UNIT_CIRCLE` → `geometry` (map to target circle/ellipse in image)

The resulting transformation can be used with `warp` to extract log-polar patches.

# Arguments
- `geometry::Union{Circle, Ellipse}`: Target geometry in image coordinates
- `logpolar_rect::HyperRectangle{2}`: Rectangle defining the log-polar patch domain
- `inner_radius::Real`: Minimum radius for log-polar transform (avoids singularity at origin)

# Returns
- `Function`: Composed transformation `logpolar_rect → geometry`

# Broadcasting
This function is designed to be broadcastable across multiple geometries:
```julia
# Extract patches from multiple circles
transforms = logpolar_map.(circles, Ref(logpolar_rect), Ref(inner_radius))
```

# Example
```julia
using VisualGeometryCore
using GeometryBasics: Circle, Point2

# Define target circle and log-polar patch size
circle = Circle(Point2(100.0, 200.0), 50.0)
ix, iy = 1:512, 1:256  # x-axis (scale), y-axis (rotation)
logpolar_rect = Rect((ix..., iy...))
inner_radius = 0.01

# Create transformation and resample through the fused warp
transform = logpolar_map(circle, logpolar_rect, inner_radius)
bm = image_warp(transform, Rect(0, 0, 512, 256))
patch = Matrix{Float32}(undef, 256, 512)
image_warp!(patch, bm, image; fillvalue=0.5f0)
```

# See Also
- [`AffineMap`](@ref): canonical → primitive coordinate mapping (constructor)
- [`logpolar_to_cartesian`](@ref): Log-polar to Cartesian transformation
"""
function logpolar_map(geometry::Union{Circle, Ellipse}, logpolar_rect::HyperRectangle{2}, inner_radius::Real)
    # Source image and output patch share the image convention (y-down): no flip.
    return AffineMap(geometry) ∘
        logpolar_to_cartesian(inner_radius) ∘
        inv(AffineMap(logpolar_rect))
end


function canonical_map(
        geometry::Union{Circle, Ellipse}, canonical_rect::HyperRectangle{2};
        bound::Real = 1.0
    )
    # Scale the geometry by bound to map CANONICAL_SQUARE [-1,1]² to [-bound, bound]²
    scaled_geometry = _scale_geometry(geometry, bound)
    # Source image and output patch share the image convention (y-down): no flip.
    return AffineMap(scaled_geometry) ∘
        inv(AffineMap(canonical_rect))
end

# Helper to scale geometry by bound factor
_scale_geometry(c::Circle, bound::Real) = Circle(GeometryBasics.origin(c), radius(c) * bound)
_scale_geometry(e::Ellipse, bound::Real) = Ellipse(GeometryBasics.origin(e), e.a * bound, e.b * bound, e.θ)

# =============================================================================
# Size-form overloads — build the patch-domain rect internally
# =============================================================================
#
# The math builders above take an explicit `HyperRectangle{2}` patch domain
# for advanced callers (custom rect, non-square, etc.).  For the common case
# of a square `size × size` patch in COLMAP math, the two size-form
# overloads below construct the rect inline.
#
# `_patch_domain_rect(size; align_corners)` is the shared helper:
#   - `align_corners=false`: rect spans `[0, size]` (pixel centers
#     `0.5, 1.5, …, size−0.5` inset half a pixel from the boundary)
#   - `align_corners=true`:  rect spans `[0.5, size−0.5]` (outermost pixel
#     centers ARE the rect corners; canonical `±1` corresponds to them)

@inline function _patch_domain_rect(size::Integer; align_corners::Bool = false)
    ix = iy = Primitives.range_interval((1:size) .- 0.5; align_corners = align_corners)
    return Rect((ix, iy))
end

"""
    canonical_map(geom, size::Integer; bound=1.0, align_corners=false) -> AffineMap

Size-form convenience: builds the COLMAP-math patch-domain rect for a
`size × size` patch internally, then delegates to
`canonical_map(geom, rect; bound)`.  See the rect-form for full details.
"""
canonical_map(
    geom::Union{Circle, Ellipse}, size::Integer;
    bound::Real = 1.0, align_corners::Bool = false
) =
    canonical_map(geom, _patch_domain_rect(size; align_corners); bound)

"""
    logpolar_map(geom, size::Integer, inner_radius; align_corners=false) -> ComposedFunction

Size-form convenience: builds the COLMAP-math patch-domain rect for a
`size × size` log-polar patch internally, then delegates to
`logpolar_map(geom, rect, inner_radius)`.
"""
logpolar_map(
    geom::Union{Circle, Ellipse}, size::Integer, inner_radius::Real;
    align_corners::Bool = false
) =
    logpolar_map(geom, _patch_domain_rect(size; align_corners), inner_radius)

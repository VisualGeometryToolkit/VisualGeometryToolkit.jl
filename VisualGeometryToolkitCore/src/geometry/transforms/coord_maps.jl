# =============================================================================
# Constructors: geometric primitives → CT.AffineMap (canonical → image-space)
# =============================================================================
#
# Each constructor produces the AffineMap that takes a canonical [-1,1]²
# point and returns the corresponding point in the target shape's coordinate
# frame.  Canonical space is implicit on the LHS — callers compose:
#
#   inv(AffineMap(target))                          # image → canonical
#   AffineMap(target) ∘ inv(AffineMap(source))      # source → target

"""
    AffineMap(rect::HyperRectangle{2}) -> AffineMap

AffineMap from canonical [-1,1]² to a 2D rectangle's coordinate frame.
"""
function CoordinateTransformations.AffineMap(target::HyperRectangle{2})
    c = centroid(target)
    w = GeometryBasics.widths(target)
    A = SMatrix{2, 2}(w[1] / 2, 0, 0, w[2] / 2)
    return CoordinateTransformations.AffineMap(A, c)
end

"""
    AffineMap(ellipse::Ellipse) -> AffineMap

AffineMap from canonical [-1,1]² to an ellipse's coordinate frame
(rotation + anisotropic scaling, then translation to centre).
"""
function CoordinateTransformations.AffineMap(target::Ellipse)
    # Pure geometry; the region maps compose it with image-convention (y-down)
    # patches on both sides, so no flip is applied. The linear part is the
    # ellipse's own shape matrix R(θ)·diag(a, b).
    return CoordinateTransformations.AffineMap(shape_matrix(target), GeometryBasics.origin(target))
end

"""
    AffineMap(circle::Circle) -> AffineMap

AffineMap from canonical [-1,1]² to a circle's coordinate frame
(isotropic scaling by radius + translation to centre).
"""
function CoordinateTransformations.AffineMap(target::Circle)
    c = GeometryBasics.origin(target)
    r = radius(target)
    return CoordinateTransformations.AffineMap(SMatrix{2, 2}(r, 0, 0, r), c)
end

# =============================================================================
# Constructor: PixelFrame types → labeled EuclideanMap (origin conversion)
# =============================================================================

# Map pixel-origin conventions to their corner offsets (in units of pixels).
# Corner offset is the position of the top-left corner of the image relative
# to (0,0). Only three canonical conventions are supported:
# - :opencv - Top-left pixel center at (0, 0)
# - :colmap - Top-left pixel center at (0.5, 0.5)
# - :julia  - Top-left pixel center at (1, 1)
# This is the value table behind the `PixelFrame{Origin}` tags (frame-level
# data, so it lives with the frame conversion — the Cameras submodule imports
# and re-exports it).
const INTRINSICS_COORDINATE_OFFSET = Base.ImmutableDict(
    # OpenCV: pixel center [0,0] at (0,0) → corner at (-0.5, -0.5)
    :opencv => Vec2(-0.5, -0.5),

    # COLMAP: pixel center [0,0] at (0.5,0.5) → corner at (0,0)
    :colmap => Vec2(0.0, 0.0),

    # Julia: pixel center [1,1] at (1,1) → corner at (0.5, 0.5)
    :julia => Vec2(0.5, 0.5)
)

"""
    EuclideanMap(::Type{PixelFrame{From}}, ::Type{PixelFrame{To}}) -> EuclideanMap{2}

Origin-convention conversion between two pixel frames, as a LABELED rigid
map: identity rotation, constant translation,
`Src = PixelFrame{From}, Dst = PixelFrame{To}` (monogram `X_{To,From}`).
Registered origins: `:opencv`, `:colmap`, `:julia` — see the table in the
[`PixelFrame`](@ref) docstring. This constructor-as-factory on the frame
TYPES is the only origin-conversion API (the 0.6-era `ImageOrigin` value
surface is removed in 0.7).

The returned map is UNITLESS (`Vec2{Float64}` translation): origin
conversion is a compute-side operation and compute runs on stripped values
(units are enforced at construction and stripped at the `region` gate).
Apply it to plain-`Float64` points; strip united data first
(`ustrip.(px, ...)`).

For identical frames the translation is zero (≡ identity). `inv` swaps the
frames; composition against other labeled 2D maps is frame-checked, and the
map fuses in `image_warp` chains like any `EuclideanMap{2}`.

# Examples
```julia
T = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
T.t == Vec2(0.5, 0.5)              # true
pt_julia = T(pt_colmap)            # pt_colmap unitless

inv(T) isa EuclideanMap{2, <:Any, <:Any, <:Any, PixelFrame{:julia}, PixelFrame{:colmap}}
```
"""
function EuclideanMap(
        ::Type{PixelFrame{From}}, ::Type{PixelFrame{To}}
    ) where {From, To}
    haskey(INTRINSICS_COORDINATE_OFFSET, From) || _origin_error(From)
    haskey(INTRINSICS_COORDINATE_OFFSET, To)   || _origin_error(To)
    offset = INTRINSICS_COORDINATE_OFFSET[To] - INTRINSICS_COORDINATE_OFFSET[From]
    return relabel(
        PixelFrame{From}, PixelFrame{To},
        EuclideanMap(one(RotMatrix{2, Float64}), Vec2{Float64}(offset))
    )
end

@noinline function _origin_error(s::Symbol)
    valid = join(sort(collect(keys(INTRINSICS_COORDINATE_OFFSET))), ", :")
    error("Unknown image-origin convention :$s. Valid options: :$valid")
end

# Conic transformations (Circle/Ellipse under a homography) are handled by
# `HomographyMap` in `homography_map.jl` — use `h = HomographyMap(matrix); Q′ = h(Q)`.

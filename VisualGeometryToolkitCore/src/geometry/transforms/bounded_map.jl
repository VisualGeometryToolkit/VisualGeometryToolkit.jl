# =============================================================================
# BoundedMap — wrap any Transformation with an output rect (codomain extent)
# =============================================================================
#
# Mirrors the BoundedCamera/PosedCamera split: `BoundedMap` decorates an inner
# transform with a `Rect{2,Int}` describing where the transform's output
# is intended to land (e.g., a warp's destination patch, or a homography's
# valid image region).  Bounds are extrinsic — the inner map's mathematical
# behaviour is unchanged.
#
# Trait opt-in: `bounds_trait(::BoundedMap) = HasBounds()`, with `extent`
# returning the stored rect.  Subtypes the umbrella `CT.Transformation`
# (not `AbstractMatrixMap`) so it works for non-matrix inners too
# (e.g., `LogPolarToCartesian`).
#
# When the inner map has a homogeneous matrix form, `to_homogeneous`
# transparently forwards through the bound.

"""
    BoundedMap{M<:CT.Transformation, R<:Rect{2,Int}} <: CT.Transformation

Decorator that pairs a `CT.Transformation` with an output `Rect{2,Int}`.
The rect is the codomain extent — typically the destination region of a
warp or the valid pixel area of a homography.

# Fields
- `map::M`  — the inner transform (the math)
- `rect::R` — the output region (decoration)

# Trait
`bounds_trait(::BoundedMap) = HasBounds()`. `extent(bm)` returns `bm.rect`.

# Example
```julia
H  = HomographyMap(K * [r1 r2 t])
bm = BoundedMap(H, Rect(0, 0, 1920, 1080))
extent(bm)               # Rect(0, 0, 1920, 1080)
to_homogeneous(bm) === H.forward   # forwards through
```
"""
struct BoundedMap{M <: CoordinateTransformations.Transformation, R <: GeometryBasics.Rect{2, Int}} <: CoordinateTransformations.Transformation
    map::M
    rect::R
end

# Promote Rect{2} → Rect{2,Int} at the boundary so callers can supply either.
BoundedMap(m::CoordinateTransformations.Transformation, r::GeometryBasics.Rect{2}) =
    BoundedMap(m, GeometryBasics.Rect{2, Int}(r))

# Idempotent re-wrap: don't nest BoundedMaps; replace the rect.
BoundedMap(bm::BoundedMap, r::GeometryBasics.Rect{2}) = BoundedMap(bm.map, r)
BoundedMap(bm::BoundedMap, r::GeometryBasics.Rect{2, Int}) = BoundedMap(bm.map, r)   # vs the default ctor

# Trait opt-in + extent accessor (matches BoundedCamera).
bounds_trait(::BoundedMap) = HasBounds()
extent(::HasBounds, bm::BoundedMap) = bm.rect

# Callable forwards to the inner map.
@inline (bm::BoundedMap)(p) = bm.map(p)

# Matrix-form forwarding for matrix-backed inners.
to_homogeneous(bm::BoundedMap{<:AbstractMatrixMap}) = to_homogeneous(bm.map)


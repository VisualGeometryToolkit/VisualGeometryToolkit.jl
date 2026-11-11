# Coordinate Transforms

VisualGeometryToolkitCore provides 2D transforms as maps, coordinate mappings, and image warping utilities.

## Maps Carry the Transform Class

A transform is a MAP, and its type is its class: Euclidean ⊂ similarity ⊂
affine ⊂ projective. Each map acts on a primitive exactly as its class allows,
and composes and inverts within it.

| class | maps | `Circle` → | `Ellipse` → | feature frames |
|---|---|---|---|---|
| Euclidean | `EuclideanMap{2}`, `Translation` | `Circle` | `Ellipse` | same type |
| similarity | `SimilarityMap{2}`, `LinearMap` of a rotation or a `UniformScaling` | `Circle` | `Ellipse` | same type |
| affine | `AffineMap`, any other `LinearMap` | `HomEllipseMat` | `HomEllipseMat` | — |
| projective | `HomographyMap` | `Conic` | `Conic` | `MethodError` |

```julia
using VisualGeometryToolkitCore
using CoordinateTransformations: Translation, LinearMap, AffineMap
using LinearAlgebra: Diagonal, I
using Rotations: RotMatrix
using StaticArrays: SVector

E = EuclideanMap(RotMatrix{2}(π/4), SVector(10.0, 20.0))    # rotation + translation
S = SimilarityMap(2.0, RotMatrix{2}(π/4), SVector(10.0, 20.0))  # x ↦ s·R·x + t
L = LinearMap(Diagonal(SVector(2.0, 1.5)))                  # 2× in x, 1.5× in y

circle = Circle(Point2(0.0, 0.0), 10.0)
S(circle)          # a Circle: centre mapped, radius 20
L(circle)          # a HomEllipseMat (an affine image of a circle is an ellipse)
S ∘ E              # a SimilarityMap, from the parts (s, R, t) — exact
inv(S)             # a SimilarityMap: (1/s, Rᵀ, −Rᵀt/s)
```

### Homographies

`HomographyMap` is projective in meaning: it maps a circle or an ellipse to a
`Conic`, whatever its matrix (possibly a hyperbola: the circle may cross the
line at infinity). Any structured map lifts to one — `HomographyMap(S)` stores
`to_homogeneous(S)` and, as its inverse, `to_homogeneous(inv(S))`, never a 3×3
inversion — and a mixed composition (`E ∘ H`, `H ∘ AffineMap(…)`) is a
`HomographyMap`. To keep an ellipse an ellipse under an affinity, apply the
affine map itself rather than its lift.

```julia
H_proj = HomographyMap(HomographyMat(camera))
Q = H_proj(circle)                                     # a Conic
ellipse = classify_conic(Q) isa EllipseTrait ? Ellipse(Q) : nothing

p = Point2(100.0, 150.0)
p_mapped = (H_proj ∘ E)(p)                             # homogeneous Point3
```

## Coordinate Representations

### Homogeneous ↔ Affine ↔ Euclidean

`to_homogeneous(map)` is a plain `SMatrix` (`[A t; 0 1]`, the last row the
constant `0 … 0 1`) of every map: for algebra and interop. It carries no class —
the map does — so geometric code takes the map and converts inside.

```julia
mat_hom = to_homogeneous(E)          # 3×3 SMatrix
p_hom = to_affine(p)                 # [x, y, 1]
p_transformed = to_euclidean(mat_hom * p_hom)
```

## Coordinate Mappings

Map between geometric regions via constructors on `AffineMap`. Each
primitive's `AffineMap` is the map from canonical [-1, 1]² to that
primitive's frame; compose / invert as needed.

### Rectangle to Rectangle

```julia
# Source and target rectangles
source = Rect(Point2(0.0, 0.0),  Vec2(100.0, 50.0))
target = Rect(Point2(10.0, 20.0), Vec2(200.0, 100.0))

# Compose source → canonical → target
map = AffineMap(target) ∘ inv(AffineMap(source))

# Apply to point
p_source = Point2(50.0, 25.0)   # Center of source
p_target = map(p_source)         # Center of target
```

### Circle/Ellipse to Canonical Square

```julia
# Map circle to canonical square [-1, 1]² is the inverse of canonical → circle
circle = Circle(Point2(100.0, 150.0), 25.0)
map = inv(AffineMap(circle))

# Point on circle boundary → point on square boundary
p_circle = Point2(125.0, 150.0)  # Right edge of circle
p_canonical = map(p_circle)
```

## Image Warping API

VisualGeometryToolkitCore exposes a **single adapter** [`image_warp`](@ref) that
wraps a math-space `Transformation` into a [`BoundedMap`](@ref) callable
from Julia `(row, col)` array indices.  VGC math is always COLMAP
convention (matching camera-intrinsics math); `image_warp` bakes the fixed
COLMAP↔Julia bridges + row/col swap via composition.  No kwargs.

The user composes the math chain themselves with the two-step pattern:
build the math transform → wrap with `image_warp`.

- [`canonical_map(geom, n; bound, align_corners)`](@ref) — Cartesian-patch math.
- [`logpolar_map(geom, n, inner_radius; align_corners)`](@ref) — log-polar math.
- [`image_warp(math, n)`](@ref) — final wrap; no kwargs.

For advanced use (non-square or custom patch-domain), both math builders
also accept an explicit `HyperRectangle{2}` instead of an integer size, and
`image_warp` accepts a `Rect{2}` for the output region.

### Basic Image Warping

[`image_warp!`](@ref)`(dst, src, bm; fillvalue)` resamples `src` into `dst`
through the warp: for each `(row, col)` of `dst` it bilinearly samples `src`
at `bm((row, col))`, running one KernelAbstractions kernel on the backend
that owns `dst` (supply `CuArray`s for GPU execution).

```julia
using VisualGeometryToolkitCore

# img :: Matrix{Float32}

# Create a warp from geometry
math = canonical_map(ellipse, 256; bound=1.0)
bm    = image_warp(math, 256)

# Resample into a pre-allocated patch
patch = Matrix{Float32}(undef, 256, 256)
image_warp!(patch, bm, img)
```

### Reusing a warp

The returned `BoundedMap` is a plain value — keep it and apply it to many images:

```julia
math = canonical_map(ellipse, 256; bound=1.0)
bm    = image_warp(math, 256)
image_warp!(patch1, bm, img1)
image_warp!(patch2, bm, img2)
```

[`image_warp_batch_to!`](@ref) samples one image through a vector of warps
into a 3D stack. A multi-plane image whose planes are sampled at different
points (a Bayer image's R, G and B) is warped plane by plane, each through its
own warp: `image_warp!(dst, warps, src)` with `dst` and `src` `H × W × P` and
`warps` a tuple of `P` warps, each composing its plane's frame map,
`image_warp(T, inv(plane_map) ∘ math, rect)`.

### Fillvalue Conventions

Supply `fillvalue` (the out-of-bounds tap value, in the float accumulation
domain) as a keyword argument:
```julia
image_warp!(patch, bm, img; fillvalue=1.0f0)  # white out of bounds
```

## Patch Extraction

### Canonical Patches

Extract patches in canonical (normalized) coordinates:

```julia
# From an ellipse - maps ellipse interior to square patch
ellipse = Ellipse(Point2(100.0, 150.0), 30.0, 20.0, π/4)
math = canonical_map(ellipse, 64; bound=1.0)
bm    = image_warp(math, 64)
patch = image_warp!(Matrix{Float32}(undef, 64, 64), bm, img)

# From geometric primitives
math = canonical_map(circle, 64; bound=1.0)
bm    = image_warp(math, 64)
```

### Feature Frame Patches

Extract patches from feature frames (Disc, EllipticFrame, AffineFrame). The
math builders take a `Circle` or `Ellipse`, so convert the frame first:

```julia
# From a Disc (isotropic feature)
disc = Disc(Point2(100.0, 200.0), 5.0)
math = canonical_map(Circle(disc), 31; bound=1.0)
bm    = image_warp(math, 31)
patch = image_warp!(Matrix{Float32}(undef, 31, 31), bm, img)

# From an EllipticFrame (anisotropic feature)
frame = EllipticFrame(center, S)  # S is the shape factor (shape matrix S²)
math = canonical_map(Ellipse(frame), 31; bound=1.0)
bm    = image_warp(math, 31)
```

### Log-Polar Patches

Extract rotation-invariant patches using log-polar sampling:

```julia
lp_patch = Matrix{Float32}(undef, 256, 256)

# Log-polar patch from a disc
math = logpolar_map(Circle(disc), 256, 0.01)
bm    = image_warp(math, 256)
image_warp!(lp_patch, bm, img)

# Custom inner radius (avoids singularity at center)
math = logpolar_map(Circle(disc), 256, 0.05)
bm    = image_warp(math, 256)
image_warp!(lp_patch, bm, img)

# From ellipse
math = logpolar_map(ellipse, 256, 0.01)
bm    = image_warp(math, 256)
image_warp!(lp_patch, bm, img)
```

Log-polar sampling maps:
- x-axis (columns) → radius (logarithmically scaled)
- y-axis (rows) → angle (0 to 2π)

This makes the patch invariant to rotation (becomes translation in y) and scale (becomes translation in x).

### Math-side patch composition

When you need to compose further math on the bare patch transform (e.g.,
detector-side compositions that chain `patch → input → level`), bypass the
image-warp builders and use the math builders directly:

```julia
# Bare math: patch coords → input image coords
patch_to_input  = canonical_map(geometry, patch_size; bound=1.0, align_corners=true)

# Compose with additional math
scale = 1.0 / step
input_to_level  = AffineMap(SMatrix{2,2}(scale, 0.0, 0.0, scale),
                            SVector(1.0 - scale, 1.0 - scale))
patch_to_level  = input_to_level ∘ patch_to_input    # still bare math

# Bake conventions at the boundary when you're ready to warp:
bm = image_warp(patch_to_level, patch_size)
patch = image_warp!(Matrix{Float32}(undef, patch_size, patch_size), bm, level_img)
```

## HomographyMap and Perspective Division

When the inner transform is a `HomographyMap` (whose output is in
homogeneous coordinates), `image_warp` automatically composes a
`PerspectiveMap` into the chain so the warp sees inhomogeneous `(x, y)`
output ready for sampling:

```julia
# HomographyMap output needs perspective division (handled internally)
H  = HomographyMap(HomographyMat(matrix))
bm = image_warp(H, 256)
patch = image_warp!(Matrix{Float32}(undef, 256, 256), bm, img)
```

## Basis Vectors

Canonical basis for coordinate systems:

```julia
# 2D canonical basis
e1, e2 = canonical_basis(SVector{2,Float64})  # ([1.0, 0.0], [0.0, 1.0])

# A single basis vector (internal, not exported)
e1 = VisualGeometryToolkitCore.basis_vector(SVector{2,Float64}, 1)  # [1.0, 0.0]
```

## Camera Homographies

Project between world plane and image:

```julia
using Rotations

# Create camera
camera = PosedCamera(model, EuclideanMap(R, t))

# Homography from world plane (z=0) to image
H_mat = HomographyMat(camera)
H = HomographyMap(H_mat)  # Wrap for callable syntax

# Map world point to image
p_world = Point2(100.0, 50.0)  # On z=0 plane
p_image = H(p_world)

# Transform circles (a Conic under a general homography)
circle_world = Circle(Point2(100.0, 50.0), 25.0)
Q_image = H(circle_world)  # a Conic; Ellipse(Q_image) once classify_conic(Q_image) isa EllipseTrait
```

## Full Image Warping Pipeline

```julia
# Create camera viewing a planar surface
camera = PosedCamera(model, pose)

# Get homography and wrap in HomographyMap
H = HomographyMap(HomographyMat(camera))

# Get output size from sensor
image_rect = Rect(sensor)
w_size = widths(image_rect)

# Create warp (handles coordinate conventions and perspective division)
bm = image_warp(inv(H), image_rect)

# Resample the board image (a Matrix{Float32}) into the sensor frame
warped_img = Matrix{Float32}(undef, Int(w_size[2]), Int(w_size[1]))  # (height, width)
image_warp!(warped_img, bm, board_image; fillvalue=1.0f0)
```

## Units in a Pose

A physical [`EuclideanMap`](@ref), such as a camera pose, a target pose or a rig
member, holds its translation as lengths in metres
(`UnitfulEuclideanMap`). An inner loop works on a plain map, whose
translation is plain numbers (`PlainEuclideanMap`). The two forms meet
only at an estimator's entry and exit, the same place where it normalizes its
data (see the units layering on the [Conventions](conventions.md) page).

```julia
using StaticArrays, Rotations
import VisualGeometryToolkitCore as VgtkCore
X = EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.1, 0.0, 1.5) .* m)  # a pose, in metres
VgtkCore.has_units(X)         # true
Xp = ustrip(mm, X)            # the plain map, translation in millimetres
Xm = EuclideanMap(Xp.R, Xp.t .* mm)   # a unit attached by construction
X(SVector(0.0, 0.0, 1.0) .* m)                           # a unitful map acts on lengths
```

Strip the whole map with `ustrip(u, X)` rather than rebuilding it from a
stripped translation. A unitful map that meets plain numbers (applied to a
plain point, or composed with a plain map) raises FlexUnits' `DimensionError`
instead of promoting. Where a plain-only object meets a unitful map — a
homography (`HomographyMap(E)`, `E ∘ H`, `E ∘` a model matrix, `to_homogeneous(E)`),
a `SimilarityMap` (`SimilarityMap(E)`, `E ∘ S`) or a planar primitive's action
(`E(circle)`) — the error is an `ArgumentError` naming the fix, `ustrip(m, E)`.

## See Also

- [Transforms API Reference](../api/transforms.md)
- [Coordinate Frames](frames.md) - The `Src`/`Dst` frame tags carried by
  `EuclideanMap` and `HomographyMap`, and the checked composition rules
- [Batched Maps](batched.md) - The structure-of-arrays transform protocol
- [Primitives Guide](primitives.md) - Transforming geometric shapes
- [Cameras Guide](cameras.md) - Camera-based transformations

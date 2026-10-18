# Coordinate Transforms

VisualGeometryToolkitCore provides homogeneous 2D transforms, coordinate mappings, and image warping utilities.

## Homogeneous 2D Transforms

### Basic Transforms

```julia
using VisualGeometryToolkitCore
using LinearAlgebra: Diagonal
using Rotations: RotMatrix
using StaticArrays: SVector

# Rotation (around origin)
R = HomRotationMat(RotMatrix{2}(π/4))  # 45° counter-clockwise (from a Rotations.jl rotation)

# Translation
T = HomTranslationMat(Vec2(10.0, 20.0))

# Uniform scaling (matches Julia's UniformScaling)
S = HomUniformScalingMat{2}(2.0)  # 2× scaling

# Diagonal scaling (non-uniform)
S_diag = HomScalingMat(Diagonal(SVector(2.0, 1.5, 1.0)))  # 2× in x, 1.5× in y
```

### Composite Transforms

The composite matrix types follow the projective-group hierarchy
`AbstractHomMat ⊃ AbstractAffineMat ⊃ AbstractSimilarityMat ⊃
AbstractEuclideanMat`:

```julia
# Euclidean (rotation + translation), similarity, affine, projective —
# each wraps a 3×3 matrix
euclidean  = HomEuclideanMat(matrix)
similarity = HomSimilarityMat(matrix)
affine     = HomAffineMat(matrix)
homography = HomographyMat(matrix)
```

The type records the tightest transform class, which drives dispatch —
e.g. a similarity maps a `Circle` to a `Circle`, an affinity maps it to a
`HomEllipseMat`, and a general homography to a `Conic` (an ellipse only while the
circle stays clear of the line at infinity — test with `classify_conic`).

### Applying Transforms

Use `HomographyMap` to make matrices callable:

```julia
# Wrap matrix in HomographyMap for callable syntax
H_rot = HomographyMap(R)
H_trans = HomographyMap(T)

# Transform a point
p = Point2(100.0, 150.0)
p_rotated = H_rot(p)

# Compose maps
H_combined = H_trans ∘ H_rot  # First rotate, then translate
p_final = H_combined(p)

# Transform geometric primitives. The result type follows the map's matrix
# type: a Circle under a similarity, a HomEllipseMat under an affinity, a
# Conic under a general homography (possibly a hyperbola: the circle may cross
# the line at infinity).
circle = Circle(Point2(0.0, 0.0), 10.0)
H_proj = HomographyMap(HomographyMat(camera))
Q = H_proj(circle)                                     # a Conic
ellipse = classify_conic(Q) isa EllipseTrait ? Ellipse(Q) : nothing
```

For matrix multiplication without wrapping:

```julia
# Direct matrix operations
p_hom = to_affine(p)  # [x, y, 1]
p_transformed = to_euclidean(SMatrix{3,3}(R) * p_hom)

# Compose matrices
combined_mat = SMatrix{3,3}(T) * SMatrix{3,3}(R)
```

## Coordinate Representations

### Homogeneous ↔ Affine ↔ Euclidean

```julia
# Get underlying matrix
mat = to_affine(R)      # 2×3 matrix [A | t]
mat_hom = to_homogeneous(R)  # 3×3 matrix

# Extract Euclidean components (rotation + translation only)
euclidean_part = to_euclidean(affine)
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
into a 3D stack.

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
frame = EllipticFrame(center, S)  # S is shape matrix
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
camera = Camera(model, EuclideanMap(R, t))

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
camera = Camera(model, pose)

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

## See Also

- [Transforms API Reference](../api/transforms.md)
- [Coordinate Frames](frames.md) - The `Src`/`Dst` frame tags carried by
  `EuclideanMap` and `HomographyMap`, and the checked composition rules
- [Batched Maps](batched.md) - The structure-of-arrays transform protocol
- [Primitives Guide](primitives.md) - Transforming geometric shapes
- [Cameras Guide](cameras.md) - Camera-based transformations

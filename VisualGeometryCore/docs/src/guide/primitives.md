# Geometric Primitives

VisualGeometryCore provides 2D geometric primitives with homogeneous matrix representations.

## Points and Rectangles

### Points

```julia
using GeometryBasics

# 2D points (from GeometryBasics, re-exported)
p = Point2(100.0, 150.0)
p = Point2f(100.0, 150.0)  # Float32 version

# 3D points
p3d = Point3(100.0, 50.0, 0.0)
```

### Rectangles

```julia
# Create rectangle from origin and size
origin = Point2(10.0, 20.0)
size = Vec2(100.0, 50.0)
rect = Rect(origin, size)

# Or from intervals
using IntervalSets
x_range = ClosedInterval(10.0, 110.0)
y_range = ClosedInterval(20.0, 70.0)
rect = Rect(x_range, y_range)

# Utility functions
c = centroid(rect)         # Get center point
```

## Circles

```julia
# Create circle
center = Point2(100.0, 150.0)
radius = 25.0
circle = Circle(center, radius)

# Scale (dilate) circle
larger_circle = dilate(circle, 2.0)  # 2× larger radius, same center
```

### Homogeneous Circle Matrix

Circles can be represented as 3×3 symmetric matrices:

```julia
# Convert circle to matrix form
Q = HomCircleMat(circle)

# Transform circle with HomographyMap
H = HomographyMap(HomographyMat(camera))
Q′ = H(circle)       # a Conic; Ellipse(Q′) once classify_conic(Q′) isa EllipseTrait

# Or work with conic matrices directly via matrix multiplication
H_mat = SMatrix{3,3}(HomographyMat(camera))
Q_transformed = inv(H_mat)' * SMatrix{3,3}(Q) * inv(H_mat)
```

## Ellipses

```julia
# Create ellipse
center = Point2(100.0, 150.0)
a = 30.0  # Semi-major axis
b = 20.0  # Semi-minor axis
θ = π/4   # Rotation angle
ellipse = Ellipse(center, a, b, θ)

# From homogeneous matrix
Q = HomEllipseMat(...)
if is_ellipse(Q)
    ellipse = Ellipse(Q)
end

# Dilate ellipse
larger_ellipse = dilate(ellipse, 2.0)  # 2× larger, same center
```

### Ellipse Properties

```julia
c = centroid(ellipse)      # Center point
gradient(ellipse, point)   # Gradient at a point on the ellipse
```

## Geometric Queries

### Containment and Intersection

```julia
# Check if point is inside a shape
point ∈ circle    # Returns true/false
point in ellipse  # Same using 'in' operator

# Check intersection between shapes
intersects(circle1, circle2)
intersects(ellipse, rect)
intersects(rect1, rect2)
```

### Conic Classification

```julia
# Check if a conic matrix represents an ellipse
Q = HomEllipseMat(...)
is_ellipse(Q)  # a real, non-degenerate ellipse?

# Classify the conic
trait = classify_conic(Q)
# For a conic matrix: EllipseTrait(), ParabolaTrait(), HyperbolaTrait() or
# DegenerateTrait(); CircleTrait() comes from classify_conic(::Circle)
```

## Transformations

### Homogeneous Representations

All primitives can be converted to homogeneous matrix form:

```julia
# Circles → 3×3 symmetric matrix
circle = Circle(Point2(0.0, 0.0), 10.0)
Q_circle = HomCircleMat(circle)

# Ellipses → 3×3 symmetric matrix
ellipse = Ellipse(Point2(0.0, 0.0), 20.0, 10.0, 0.0)
Q_ellipse = HomEllipseMat(ellipse)
```

### Transforming Shapes

Use `HomographyMap` to transform shapes:

```julia
# Create homography and wrap for callable syntax
H_mat = HomographyMat(camera)
H = HomographyMap(H_mat)

# Transform circles. A general homography returns a `Conic` — an ellipse
# for a circle in view, a hyperbola for one crossing the line at infinity.
Q = H(circle)
if classify_conic(Q) isa EllipseTrait
    ellipse = Ellipse(Q)  # valid ellipse in view
end
```

### Affine Transforms

```julia
using Rotations: RotMatrix

# Create transform matrices
R = HomRotationMat(RotMatrix{2}(π/4))    # 45° rotation
T = HomTranslationMat(Vec2(10.0, 20.0))  # Translation
S = HomUniformScalingMat{2}(2.0)         # Uniform 2× scaling

# Compose into a single matrix
combined = HomSimilarityMat(SMatrix{3,3}(T) * SMatrix{3,3}(R) * SMatrix{3,3}(S))

# Wrap for callable syntax
H = HomographyMap(combined)

# Apply to shapes - similarity preserves circles
transformed_circle = H(circle)  # Returns Circle (similarity preserves shape)
```

## Working with Units

Unit constants (`mm`, `μm`, `px`, `dpi`, …) and the quantity machinery are
exported from the main module (FlexUnits-based; see
[Getting Started](getting-started.md#Physical-Units)). Geometric storage is
unitless (lengths in mm, pixels in px); units are enforced at API
boundaries (e.g. sensor pitches, focal lengths, metric tolerances) by
stripping in a named unit (`ustrip(mm, q)`) or comparing
`dimension(q) == dimension(px)`.

## Feature Frames

The transformation-based feature primitives — [`Disc`](@ref) (center + σ),
`OrientedDisc` (+ orientation), [`EllipticFrame`](@ref) (+ aspect/shape),
`AffineFrame` — represent detections geometrically; each defines
`AffineMap(frame)` mapping canonical coordinates to its image-space
support. [`Paraboloid3D`](@ref) is the quadratic surface used for
refinement fits. The detectors that produce them live in
VisualGeometryFeatures.

## See Also

- [Primitives API Reference](../api/primitives.md)
- [Transforms Guide](transforms.md) - Transforming primitives
- [Cameras Guide](cameras.md) - Projecting primitives with cameras

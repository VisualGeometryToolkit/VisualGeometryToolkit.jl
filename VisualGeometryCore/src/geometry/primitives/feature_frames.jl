# =============================================================================
# Feature Frames - Geometric Coordinate Frames
# =============================================================================
#
# Feature frames define transformations from canonical space to image space.
# They represent regions (discs, elliptic regions) not boundaries (circles, ellipses).
#
# Mathematical distinction:
#   - Disc = filled region {x : |x| ≤ r}
#   - Circle = boundary curve {x : |x| = r}
#
# Frame hierarchy (by degrees of freedom):
#   - Disc (3 DOF): translation + isotropic scale  [defined in blob.jl]
#   - OrientedDisc (4 DOF): translation + isotropic scale + orientation
#   - EllipticFrame (5 DOF): translation + anisotropic scale (symmetric)
#   - AffineFrame (6 DOF): translation + general linear transform
#
# Key operations:
#   - AffineMap(frame) → AffineMap from canonical to image space
#   - region(frame, k) → unitless Circle/Ellipse at the k·σ contour
#
# NOTE: FeatureFrame and Disc are defined in blob_types.jl (loaded earlier).
# This file adds: OrientedDisc, EllipticFrame, AffineFrame, and all methods.
# =============================================================================

# =============================================================================
# OrientedDisc (4 DOF) - Isotropic Frame with Orientation
# =============================================================================

"""
    OrientedDisc{T} <: FeatureFrame{T}

Isotropic feature frame with orientation: 4 degrees of freedom.

Defines the transformation: `x = σ·R(θ)·u + center`

where R(θ) is a 2D rotation matrix. This maps the unit circle with a marked
orientation to an oriented circle in image space. Invariant to full similarity
transformations.

# Fields
- `center::Point2{T}`: Center position in image coordinates
- `σ::T`: Characteristic scale
- `θ::T`: Dominant orientation in radians (measured from +x axis)

# Example
```julia
disc = OrientedDisc(Point2(100.0, 200.0), 5.0, π/4)
T = AffineMap(disc)  # AffineMap with rotation
```

See also: [`Disc`](@ref), [`EllipticFrame`](@ref)
"""
struct OrientedDisc{T} <: FeatureFrame{T}
    center::Point2{T}
    σ::T
    θ::T  # orientation in radians

    function OrientedDisc{T}(center::Point2{T}, σ::T, θ::T) where {T}
        σ > 0 || throw(ArgumentError("σ must be positive, got $σ"))
        return new{T}(center, σ, θ)
    end
end

# Convenience constructors
function OrientedDisc(center::Point2{S}, σ::Real, θ::Real) where {S}
    T = promote_type(S, typeof(σ), typeof(θ))
    return OrientedDisc{T}(Point2{T}(center), T(σ), T(θ))
end
OrientedDisc(x::Real, y::Real, σ::Real, θ::Real) = OrientedDisc(Point2(x, y), σ, θ)
OrientedDisc(center::AbstractVector, σ::Real, θ::Real) = OrientedDisc(Point2(center...), σ, θ)
OrientedDisc(o::OrientedDisc) = o

# Construct from Disc by adding orientation
OrientedDisc(d::Disc{T}, θ::Real) where {T} = OrientedDisc{T}(d.center, d.σ, T(θ))

# Properties
scale(d::OrientedDisc) = d.σ

# Strip orientation
Disc(d::OrientedDisc) = Disc(d.center, d.σ)

# =============================================================================
# EllipticFrame (5 DOF) - Anisotropic Frame
# =============================================================================

"""
    EllipticFrame{T} <: FeatureFrame{T}

Anisotropic feature frame with 5 degrees of freedom: translation (2) + symmetric shape (3).

Defines the transformation: `x = S·u + center`

where S is a symmetric positive-definite 2×2 matrix (the shape matrix).
The moment matrix is M = S·S' = S² (since S is symmetric).

This maps the unit circle to an ellipse in image space. Invariant to affine
transformations (up to in-plane rotation).

# Fields
- `center::Point2{T}`: Center position in image coordinates
- `S::SMatrix{2,2,T,4}`: Shape matrix (symmetric positive-definite)

# Relationship to Moment Matrix
- Shape matrix: S (what we store)
- Moment matrix: M = S² = S·S'
- To get S from M: S = √M (matrix square root)

# Example
```julia
S = @SMatrix [2.0 0.5; 0.5 1.0]  # symmetric shape matrix
frame = EllipticFrame(Point2(100.0, 200.0), S)
T = AffineMap(frame)  # AffineMap: u → S*u + center
ellipse = region(frame, 2.0)  # Ellipse at 2σ contour
```

See also: [`AffineFrame`](@ref), `Distributions.MvNormal`
"""
struct EllipticFrame{T} <: FeatureFrame{T}
    center::Point2{T}
    S::SMatrix{2, 2, T, 4}  # symmetric shape matrix
end

const DEFAULT_MAX_ASPECT_RATIO = 10.0

"""
    is_valid_shape(S; max_aspect_ratio=DEFAULT_MAX_ASPECT_RATIO) -> Bool

Check that a 2×2 shape matrix is symmetric, positive-definite, and has
axis ratio ≤ `max_aspect_ratio`.
"""
function is_valid_shape(
        S::AbstractMatrix;
        max_aspect_ratio::Float64 = DEFAULT_MAX_ASPECT_RATIO
    )
    is_positive_definite(S) || return false
    f = EllipticFrame(Point2(0.0, 0.0), S)
    return aspect_ratio(f) ≤ max_aspect_ratio
end

# =============================================================================
# Paraboloid3D - Quadratic Surface over 2D domain
# =============================================================================

"""
    Paraboloid3D{T,P}

Quadratic surface `z = a₀ + b'x + x'Cx` over a 2D domain.

The type parameter `P` encodes the number of free parameters:
- `P=6`: Full anisotropic — `b` and `C` are free
- `P=4`: Isotropic with center — `C = cI`, `b` is free
- `P=2`: Fixed-center isotropic — `C = cI`, `b = 0`

Storage is always the full `(a₀, b, C)` so the geometric interface
(`moment_matrix`, `is_peak`, `EllipticFrame`) works uniformly.

# Fields
- `a₀::T`: Intercept (peak value)
- `b::SVector{2,T}`: Linear gradient
- `C::SMatrix{2,2,T,4}`: Quadratic Hessian (neg-def for valid peaks)

# Constructors
- `Paraboloid3D(θ::SVector{6})`: Full anisotropic `{T,6}`
- `Paraboloid3D(θ::SVector{4})`: Isotropic with center `{T,4}`
- `Paraboloid3D(θ::SVector{2})`: Fixed-center isotropic `{T,2}`
- `Paraboloid3D(d::Disc, a₀)`: From detection `{T,2}`

See also: [`moment_matrix`](@ref), [`is_peak`](@ref), [`coeffs`](@ref), [`EllipticFrame`](@ref)
"""
struct Paraboloid3D{T, P}
    a₀::T
    b::SVector{2, T}
    C::SMatrix{2, 2, T, 4}
end

# Full anisotropic: θ = [a₀, bx, by, Cxx, Cxy, Cyy]
function Paraboloid3D(θ::SVector{6, T}) where {T}
    return Paraboloid3D{T, 6}(
        θ[1], SVector{2, T}(θ[2], θ[3]),
        SMatrix{2, 2, T}(θ[4], θ[5], θ[5], θ[6])
    )
end

# Isotropic with center: θ = [a₀, bx, by, c]
function Paraboloid3D(θ::SVector{4, T}) where {T}
    return Paraboloid3D{T, 4}(
        θ[1], SVector{2, T}(θ[2], θ[3]),
        SMatrix{2, 2, T}(θ[4], zero(T), zero(T), θ[4])
    )
end

# Fixed-center isotropic: θ = [a₀, c]
function Paraboloid3D(θ::SVector{2, T}) where {T}
    return Paraboloid3D{T, 2}(
        θ[1], SVector{2, T}(zero(T), zero(T)),
        SMatrix{2, 2, T}(θ[2], zero(T), zero(T), θ[2])
    )
end
Paraboloid3D(p::Paraboloid3D) = p

"""
    moment_matrix(p::Paraboloid3D) -> SMatrix{2,2}

Covariance of the equivalent Gaussian: `Σ = -½C⁻¹`.
"""
function moment_matrix(p::Paraboloid3D{T}) where {T}
    C = p.C
    d = det(C)
    inv_det = T(-0.5) / d
    return SMatrix{2, 2, T}(
        C[2, 2] * inv_det, -C[2, 1] * inv_det,
        -C[1, 2] * inv_det, C[1, 1] * inv_det
    )
end

"""
    is_positive_definite(M::AbstractMatrix) -> Bool

2×2 positive-definiteness test via Sylvester's criterion: M[1,1] > 0 and det(M) > 0.
"""
is_positive_definite(M::AbstractMatrix) = M[1, 1] > 0 && M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1] > 0

"""
    is_negative_definite(M::AbstractMatrix) -> Bool

2×2 negative-definiteness test: M[1,1] < 0 and det(M) > 0.
"""
is_negative_definite(M::AbstractMatrix) = M[1, 1] < 0 && M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1] > 0

"""
    is_peak(p::Paraboloid3D) -> Bool

Whether the paraboloid opens downward (its quadratic form is negative
definite) — i.e. the fitted surface has a maximum, as expected for a
bright-blob response peak. See [`Paraboloid3D`](@ref).
"""
is_peak(p::Paraboloid3D) = is_negative_definite(p.C)

"""
    Paraboloid3D(d::Disc, a₀::Real) -> Paraboloid3D{T,2}

Construct a fixed-center isotropic paraboloid from a `Disc` detection.

The disc's scale `σ` maps to the quadratic coefficient `c = -1/(2σ²)`,
with zero center offset (`b = 0`). The intercept `a₀` is typically
the peak log-intensity `z_max = log(max inverted intensity)`.
"""
function Paraboloid3D(d::Disc, a₀::Real)
    T = promote_type(typeof(d.σ), typeof(a₀))
    c = T(-1) / (2 * T(d.σ)^2)
    return Paraboloid3D{T, 2}(
        T(a₀), SVector{2, T}(zero(T), zero(T)),
        SMatrix{2, 2, T}(c, zero(T), zero(T), c)
    )
end

"""
    coeffs(p::Paraboloid3D{T,P}) -> SVector{P,T}

Return the free paraboloid coefficients. The size is determined by `P`:
- `P=6`: `[a₀, bx, by, Cxx, Cxy, Cyy]` (full anisotropic)
- `P=4`: `[a₀, bx, by, c]` (isotropic with center)
- `P=2`: `[a₀, c]` (fixed-center isotropic)
"""
coeffs(p::Paraboloid3D{T, 6}) where {T} =
    SVector{6, T}(p.a₀, p.b[1], p.b[2], p.C[1, 1], p.C[1, 2], p.C[2, 2])
coeffs(p::Paraboloid3D{T, 4}) where {T} = SVector{4, T}(p.a₀, p.b[1], p.b[2], p.C[1, 1])
coeffs(p::Paraboloid3D{T, 2}) where {T} = SVector{2, T}(p.a₀, p.C[1, 1])

"""
    Paraboloid3D{T,Q}(p::Paraboloid3D{T,P}) -> Paraboloid3D{T,Q}

Promote (or demote) a paraboloid to a different parameter count `Q`.
The underlying storage `(a₀, b, C)` is unchanged — only the type tag changes.
"""
Paraboloid3D{T, Q}(p::Paraboloid3D{T}) where {T, Q} =
    Paraboloid3D{T, Q}(p.a₀, p.b, p.C)

# Convenience constructors
function EllipticFrame(center::Point2{C}, S::AbstractMatrix) where {C}
    T = promote_type(C, eltype(S))
    return EllipticFrame{T}(Point2{T}(center), SMatrix{2, 2, T}(S))
end

EllipticFrame(center::AbstractVector, S::AbstractMatrix) = EllipticFrame(Point2(center...), S)
EllipticFrame(e::EllipticFrame) = e

# The symmetric square root S = Σ^½ of a 2×2 moment matrix (M = S·S): the ONE
# spelling for every frame constructor. An eigenvalue negative only to rounding
# (within 8·eps of the largest) clamps to zero, so a PSD matrix that rounds a
# hair negative is not NaN; a genuinely indefinite Σ — a paraboloid fit that is
# not a peak — is refused, as the stdlib's `sqrt(::Symmetric)` refuses it by
# going complex. The result is exactly symmetric (the lower triangle mirrors the
# upper).
function _sym_sqrt(Σ::AbstractMatrix)
    E = eigen(Symmetric(SMatrix{2, 2}(Σ)))
    λ = E.values
    tol = 8 * eps(float(eltype(λ))) * maximum(abs, λ)
    all(l -> l >= -tol, λ) ||
        throw(ArgumentError("moment matrix is not positive semidefinite (eigenvalues $(Tuple(λ)))"))
    S = E.vectors * Diagonal(sqrt.(max.(λ, zero(eltype(λ))))) * E.vectors'
    return SMatrix{2, 2}(S[1, 1], S[1, 2], S[1, 2], S[2, 2])
end

# Construct from moment matrix (M → S via matrix sqrt)
EllipticFrame(center::Point2{T}, Σ::AbstractMatrix, ::Val{:moment_matrix}) where {T} =
    EllipticFrame{T}(Point2{T}(center), SMatrix{2, 2, T}(_sym_sqrt(Σ)))

# Construct from Paraboloid3D fit: Σ = -½C⁻¹, center shifted by Σ*b
function EllipticFrame(center::Point2{T}, p::Paraboloid3D{T, P}) where {T, P}
    Σ = moment_matrix(p)
    offset = Σ * p.b
    new_center = Point2{T}(center[1] + offset[1], center[2] + offset[2])
    return EllipticFrame{T}(new_center, SMatrix{2, 2, T}(_sym_sqrt(Σ)))
end

# Construct isotropic (from σ) - accepts any Number including Unitful
function EllipticFrame(center::Point2{C}, σ::S) where {C, S <: Number}
    T = promote_type(C, S)
    σ_conv = convert(T, σ)
    shape = SMatrix{2, 2, T}(σ_conv, zero(T), zero(T), σ_conv)
    return EllipticFrame{T}(Point2{T}(center), shape)
end

# Construct from Disc (isotropic)
EllipticFrame(d::Disc{T}) where {T} = EllipticFrame(d.center, d.σ)

# Construct from Ellipse (compute symmetric shape matrix from moment matrix)
function EllipticFrame(e::Ellipse{T}) where {T}
    # EllipticFrame uses symmetric S where M = S²
    # Ellipse has M = A*A' where A = R(θ)*diag(a,b) is non-symmetric
    # Compute S = M^(1/2) via eigendecomposition
    return EllipticFrame{T}(e.center, SMatrix{2, 2, T}(_sym_sqrt(moment_matrix(e))))
end

# Properties
shape_matrix(f::EllipticFrame) = f.S
moment_matrix(f::EllipticFrame) = f.S * f.S'  # M = S²

# =============================================================================
# AffineFrame (6 DOF) - Full Affine Frame
# =============================================================================

"""
    AffineFrame{T} <: FeatureFrame{T}

Full affine feature frame with 6 degrees of freedom: translation (2) + general linear (4).

Defines the transformation: `x = A·u + center`

where A is a general invertible 2×2 matrix. This maps the unit circle to an
oriented ellipse in image space. Invariant to full affine transformations.

The extra degree of freedom compared to `EllipticFrame` is the in-plane rotation
that a symmetric matrix cannot represent.

# Fields
- `center::Point2{T}`: Center position in image coordinates
- `A::SMatrix{2,2,T,4}`: Shape matrix (general, not necessarily symmetric)

# Decomposition
Any A can be decomposed as A = R(φ)·S where:
- S is symmetric positive-definite (the "pure shape")
- R(φ) is an in-plane rotation (the extra DOF)

# Example
```julia
A = @SMatrix [2.0 0.5; -0.5 1.0]  # general shape matrix
frame = AffineFrame(Point2(100.0, 200.0), A)
T = AffineMap(frame)  # AffineMap: u → A*u + center
```

See also: [`EllipticFrame`](@ref)
"""
struct AffineFrame{T} <: FeatureFrame{T}
    center::Point2{T}
    A::SMatrix{2, 2, T, 4}  # general shape matrix

    function AffineFrame{T}(center::Point2{T}, A::SMatrix{2, 2, T, 4}) where {T}
        det(A) > 0 || throw(ArgumentError("Shape matrix A must have positive determinant"))
        return new{T}(center, A)
    end
end

# Convenience constructors
function AffineFrame(center::Point2{C}, A::AbstractMatrix) where {C}
    T = promote_type(C, eltype(A))
    return AffineFrame{T}(Point2{T}(center), SMatrix{2, 2, T}(A))
end

AffineFrame(center::AbstractVector, A::AbstractMatrix) = AffineFrame(Point2(center...), A)
AffineFrame(a::AffineFrame) = a

# Construct from EllipticFrame + rotation
function AffineFrame(f::EllipticFrame{T}, φ::Real) where {T}
    R = SMatrix{2, 2, T}(cos(φ), sin(φ), -sin(φ), cos(φ))
    return AffineFrame{T}(f.center, R * f.S)
end

# Construct from EllipticFrame (no extra rotation)
AffineFrame(f::EllipticFrame{T}) where {T} = AffineFrame{T}(f.center, f.S)

# Construct from Disc + rotation
function AffineFrame(d::Disc{T}, φ::Real) where {T}
    R = SMatrix{2, 2, T}(cos(φ), sin(φ), -sin(φ), cos(φ))
    return AffineFrame{T}(d.center, d.σ * R)
end

# Properties
shape_matrix(f::AffineFrame) = f.A
moment_matrix(f::AffineFrame) = f.A * f.A'  # M = A·A'

# Extract symmetric part (EllipticFrame)
function EllipticFrame(f::AffineFrame{T}) where {T}
    # Polar decomposition: A = R·S where S = √(A'A) is symmetric
    return EllipticFrame{T}(f.center, SMatrix{2, 2, T}(_sym_sqrt(f.A' * f.A)))
end

# Trait-dispatched scale/aspect. The IsotropicScale arms were missing from
# the trait refactor (3dd507e), so characteristic_scale(::Disc) and
# aspect_ratio(::Disc) threw MethodError — Disc/OrientedDisc carry a single
# σ, which IS the characteristic scale, with aspect ratio 1 by definition.
characteristic_scale(f::FeatureFrame) = _characteristic_scale(frame_scale(f), f)
_characteristic_scale(::ShapeMatrixScale, f) = sqrt(abs(det(shape_matrix(f))))
_characteristic_scale(::IsotropicScale, f) = f.σ

aspect_ratio(f::FeatureFrame) = _aspect_ratio(frame_scale(f), f)
function _aspect_ratio(::ShapeMatrixScale, f)
    svs = svdvals(shape_matrix(f))
    return sqrt(maximum(svs) / minimum(svs))
end
_aspect_ratio(::IsotropicScale, f) = one(f.σ)

# =============================================================================
# Coordinate Maps
# =============================================================================

"""
    AffineMap(frame::FeatureFrame) -> AffineMap

Affine transformation from canonical (unit-circle/square) coordinates to
image coordinates.  Maps the canonical region to the feature region in
image space.

# Example
```julia
disc = Disc(Point2(100.0, 200.0), 5.0)
T = AffineMap(disc)
image_point = T(canonical_point)
```
"""
function CoordinateTransformations.AffineMap(d::Disc{T}) where {T}
    M = SMatrix{2, 2, T}(d.σ, 0, 0, d.σ)
    return AffineMap(M, SVector{2, T}(d.center...))
end

function CoordinateTransformations.AffineMap(d::OrientedDisc{T}) where {T}
    c, s = cos(d.θ), sin(d.θ)
    R = SMatrix{2, 2, T}(c, s, -s, c)
    M = d.σ * R
    return AffineMap(M, SVector{2, T}(d.center...))
end

CoordinateTransformations.AffineMap(f::FeatureFrame) = _affine_map_frame(frame_scale(f), f)
_affine_map_frame(::ShapeMatrixScale, f) = AffineMap(shape_matrix(f), SVector{2}(f.center...))

# =============================================================================
# Boundary Sampling (HasBoundary trait implementation)
# =============================================================================

"""
    boundary_point(frame::FeatureFrame, θ::Real) -> Point2

Return a point on the frame's boundary at parameter θ.

Uses the coordinate map to transform a unit circle point to image space:
    p(θ) = AffineMap(frame)(cos(θ), sin(θ))
"""
function boundary_point(f::FeatureFrame{T}, θ::Real) where {T}
    m = CoordinateTransformations.AffineMap(f)
    return Point2{T}(m(SVector{2, T}(cos(θ), sin(θ))))
end

# `boundary_points` comes from the generic in interface_functions.jl, which
# walks this `boundary_point`. It is NOT an alias for
# `GeometryBasics.coordinates`: `coordinates` meshes a FILLED shape (its ring
# closes, first == last, and for a Rect its `n` is an area grid), while
# `boundary_points` returns `n` DISTINCT outline samples.

# =============================================================================
# Boundary Extraction (for visualization)
# =============================================================================

"""
    region(frame::FeatureFrame, k::Real=1.0) -> Circle or Ellipse

THE detection → primitive gate: the image region supported by a detection at
the `k`·σ contour, as a plain unitless primitive.

This is also the unit-strip boundary of the geometry stack: frames may carry
units (px-typed detections, mm-typed world blobs); the returned primitive is
always unitless, ready for transforms and the predicate algebra
(`overlap_area`, `iou`, `coverage`, `intersects`).

Defers to the appropriate geometric constructor on the stripped frame:
- `Circle(frame, k)` for isotropic frames (`Disc`, `OrientedDisc`)
- `Ellipse(frame, k)` for anisotropic frames (`EllipticFrame`, `AffineFrame`)

# Arguments
- `frame`: The feature frame
- `k`: Sigma multiplier (default 1.0 = 1σ contour)

# Example
```julia
disc = Disc(Point2(100.0px, 200.0px), 5.0px)
circ = region(disc, 3.0)  # unitless Circle at 3σ, radius 15.0

frame = EllipticFrame(Point2(100.0, 200.0), S)
ell = region(frame, 2.0)  # Ellipse at 2σ
```
"""
region(f::FeatureFrame, k::Real = 1.0) = _frame_region(frame_scale(f), _strip_frame(f), k)
_frame_region(::IsotropicScale, d, k) = Circle(d, k)
_frame_region(::ShapeMatrixScale, f, k) = Ellipse(f, k)

"""
    boundary(center, Σ::AbstractMatrix, n_sigma::Real=1.0) -> Ellipse

Construct an n-sigma confidence ellipse from a 2×2 covariance matrix.

Defers to `Ellipse(center, Symmetric(n_sigma² * Σ))`.

```julia
Σ = SA[0.04 0.01; 0.01 0.09]
e = boundary(Point2(0.0, 0.0), Σ, 3.0)  # 3σ confidence ellipse
```
"""
boundary(center, Σ::AbstractMatrix, n_sigma::Real = 1.0) =
    Ellipse(center, Symmetric(n_sigma^2 * Σ))

# =============================================================================
# Circle Construction from FeatureFrame
# =============================================================================

"""
    GeometryBasics.Circle(frame::FeatureFrame, r::Real=1.0) -> Circle

Construct a Circle from a FeatureFrame at the `r`-σ contour.

For isotropic frames (Disc, OrientedDisc), radius = `r * σ`.
For anisotropic frames (EllipticFrame, AffineFrame), radius = `r * characteristic_scale`
(circumscribed circle).

# Example
```julia
disc = Disc(Point2(100.0, 200.0), 5.0)
circle = Circle(disc, 3.0)  # Circle with radius 15.0 (3σ)
```
"""
Circle(f::FeatureFrame, r::Real = 1.0) = _frame_circle(frame_scale(f), f, r)
# through the coordinate compute boundary, like `Circle(::AbstractDetection)`
function _frame_circle(::IsotropicScale, d, r)
    cx, cy, σ = _strip_coords((d.center[1], d.center[2], d.σ))
    return Circle(Point2(cx, cy), r * σ)
end
_frame_circle(::ShapeMatrixScale, f, r) = Circle(f.center, r * characteristic_scale(f))

# =============================================================================
# Ellipse Construction from FeatureFrame
# =============================================================================

"""
    Ellipse(frame::FeatureFrame, r::Real=1.0) -> Ellipse

Construct an Ellipse from a FeatureFrame at the `r`-σ contour.

For isotropic frames, returns a circle-equivalent ellipse (a == b).
For anisotropic frames, computes semi-axes via SVD of the scaled shape matrix.

# Example
```julia
frame = EllipticFrame(Point2(100.0, 200.0), S)
ell = Ellipse(frame, 2.0)  # Ellipse at 2σ contour
```
"""
Ellipse(f::FeatureFrame, r::Real = 1.0) = _frame_ellipse(frame_scale(f), f, r)
function _frame_ellipse(::IsotropicScale, d, r)
    radius = r * d.σ
    T = typeof(radius)
    return Ellipse(d.center, radius, radius, zero(T))
end
function _frame_ellipse(::ShapeMatrixScale, f, r)
    M = r * shape_matrix(f)
    U, σs, _ = svd(M)
    return Ellipse(f.center, σs[1], σs[2], atan(U[2, 1], U[1, 1]))
end

# =============================================================================
# Frame Inversion
# =============================================================================

"""
    inv(frame::FeatureFrame) -> FeatureFrame

Return the frame with inverted transformation.

For a frame with transformation T: canonical → image,
the inverse frame has transformation T⁻¹: image → canonical.
"""
function Base.inv(d::Disc{T}) where {T}
    # Inverse: x = σ·u + c  →  u = (x - c)/σ
    # New frame: center at origin (in canonical), σ = 1/σ
    # Actually, inversion changes the meaning... need to think about this
    return Disc{T}(Point2{T}(-d.center[1] / d.σ, -d.center[2] / d.σ), one(T) / d.σ)
end

function Base.inv(f::EllipticFrame{T}) where {T}
    S_inv = inv(f.S)
    new_center = Point2{T}(-(S_inv * SVector{2, T}(f.center...))...)
    return EllipticFrame{T}(new_center, SMatrix{2, 2, T}(S_inv))
end

function Base.inv(f::AffineFrame{T}) where {T}
    A_inv = inv(f.A)
    new_center = Point2{T}(-(A_inv * SVector{2, T}(f.center...))...)
    return AffineFrame{T}(new_center, SMatrix{2, 2, T}(A_inv))
end

# =============================================================================
# Conversions between MvNormal and Frames
# =============================================================================

# Note: MvNormal stores covariance Σ, EllipticFrame stores shape matrix S
# Relationship: M = S·S' (S²) for symmetric S; MvNormal uses M as its covariance
# EllipticFrame(d::AbstractMvNormal) is defined in mvnormal_geometry.jl

"""
    MvNormal(f::EllipticFrame) -> MvNormal
    MvNormal(f::AffineFrame) -> MvNormal

Convert a shape-matrix frame to an MvNormal (statistical).

Computes the moment matrix M = shape·shape' and uses it as MvNormal covariance.
Note: For AffineFrame, this loses the in-plane rotation since Σ = A·A' is symmetric.
"""
MvNormal(f::FeatureFrame) = _frame_mvnormal(frame_scale(f), f)
function _frame_mvnormal(::ShapeMatrixScale, f)
    T = eltype(f.center)
    Σ = SMatrix{2, 2, T}(moment_matrix(f))
    return MvNormal(SVector{2, T}(f.center...), Σ)
end

# =============================================================================
# GeometryBasics Integration
# =============================================================================

"""
    GeometryBasics.origin(frame::FeatureFrame) -> Point2

Get the center of a frame as its origin, making it compatible with generic
GeometryBasics primitives for coordinate convention conversion.

# Example
```julia
disc = Disc(Point2(10.0pd, 20.0pd), 2.0pd)
pos = origin(disc)  # Returns Point2(10.0pd, 20.0pd)
```
"""
GeometryBasics.origin(f::FeatureFrame) = f.center

# =============================================================================
# Centroid
# =============================================================================

"""
    centroid(frame::FeatureFrame) -> Point2

Return the centroid of a feature frame.

# Example
```julia
disc = Disc(Point2(100.0, 200.0), 5.0)
c = centroid(disc)  # Returns Point2(100.0, 200.0)
```
"""
centroid(f::FeatureFrame) = f.center

# =============================================================================
# Unit Conversion
# =============================================================================

"""
    ustrip(u, frame::FeatureFrame)

The frame with its coordinates and scale stripped in unit `u` (`px`, `mm`, …):
`ustrip(px, Disc(Point2(100.0px, 200.0px), 5.0px)) == Disc(100.0, 200.0, 5.0)`.
FlexUnits checks each component's dimension against `u`. Enables
`ustrip.(px, frames)` on a vector of frames.
"""
FlexUnits.ustrip(u::Units, d::Disc) =
    Disc(Point2(ustrip(u, d.center[1]), ustrip(u, d.center[2])), ustrip(u, d.σ))

FlexUnits.ustrip(u::Units, d::OrientedDisc) =   # θ is radians, a dimensionless quantity or a plain number
    OrientedDisc(Point2(ustrip(u, d.center[1]), ustrip(u, d.center[2])), ustrip(u, d.σ), ustrip(d.θ))

FlexUnits.ustrip(u::Units, f::EllipticFrame) =
    EllipticFrame(Point2(ustrip(u, f.center[1]), ustrip(u, f.center[2])), ustrip.(u, f.S))

FlexUnits.ustrip(u::Units, f::AffineFrame) =
    AffineFrame(Point2(ustrip(u, f.center[1]), ustrip(u, f.center[2])), ustrip.(u, f.A))

# The unit-free frame a region is drawn from: through the compute boundary
# (pixels in px, lengths in mm; plain numbers unchanged).
function _strip_frame(d::Disc)
    cx, cy, σ = _strip_coords((d.center[1], d.center[2], d.σ))
    return Disc(Point2(cx, cy), σ)
end
_strip_frame(f::Union{OrientedDisc, EllipticFrame, AffineFrame}) =
    f.center[1] isa Real ? f : ustrip(_coord_unit(f.center[1]), f)

# A frame as plain numbers in `u` (`strip_or_bare`, base/units.jl): stripped
# when its coordinates are quantities, as it is when they are plain numbers.
strip_or_bare(u::Units, f::FeatureFrame) = f.center[1] isa Real ? f : ustrip(u, f)

# =============================================================================
# Arithmetic Operations
# =============================================================================

"""
    +(frame::FeatureFrame, offset)

Translate a frame by adding an offset to its center position.

# Arguments
- `frame::FeatureFrame`: The frame to translate
- `offset`: A tuple or vector representing the translation (x, y)

# Returns
- New frame of the same type with translated center

# Example
```julia
disc = Disc(Point2(10.0mm, 20.0mm), 2.0mm)
translated = disc + (5.0mm, 5.0mm)  # Center at (15.0mm, 25.0mm)
```
"""
Base.:+(f::FeatureFrame, offset::Union{AbstractVector, Tuple}) = @set f.center = f.center .+ offset

"""
    -(frame::FeatureFrame, offset)

Translate a frame by subtracting an offset from its center position.

# Example
```julia
disc = Disc(Point2(10.0mm, 20.0mm), 2.0mm)
translated = disc - (5.0mm, 5.0mm)  # Center at (5.0mm, 15.0mm)
```
"""
Base.:-(f::FeatureFrame, offset::Union{AbstractVector, Tuple}) = @set f.center = f.center .- offset

"""
    *(disc::Disc, scale)

Scale a Disc's center coordinates and σ by a scalar factor.

# Example
```julia
disc = Disc(Point2(10.0mm, 20.0mm), 2.0mm)
scaled = disc * 300dpi  # Scale to logical units
```
"""
function Base.:*(d::Disc, scale::Number)
    return Disc(d.center .* scale, d.σ * scale)
end

"""
    /(disc::Disc, scale)

Divide a Disc's center coordinates and σ by a scalar factor.

# Example
```julia
disc = Disc(Point2(300.0px, 600.0px), 15.0px)
scaled = disc / 300dpi  # Scale to physical units
```
"""
function Base.:/(d::Disc, scale::Number)
    return Disc(d.center ./ scale, d.σ / scale)
end

# OrientedDisc scaling
function Base.:*(d::OrientedDisc, scale::Number)
    return OrientedDisc(d.center .* scale, d.σ * scale, d.θ)
end

function Base.:/(d::OrientedDisc, scale::Number)
    return OrientedDisc(d.center ./ scale, d.σ / scale, d.θ)
end

# EllipticFrame scaling
function Base.:*(f::EllipticFrame, scale::Number)
    return EllipticFrame(f.center .* scale, f.S .* scale)
end

function Base.:/(f::EllipticFrame, scale::Number)
    return EllipticFrame(f.center ./ scale, f.S ./ scale)
end

# AffineFrame scaling
function Base.:*(f::AffineFrame, scale::Number)
    return AffineFrame(f.center .* scale, f.A .* scale)
end

function Base.:/(f::AffineFrame, scale::Number)
    return AffineFrame(f.center ./ scale, f.A ./ scale)
end

# =============================================================================
# Degrees of Freedom
# =============================================================================


# JSON serialization moved to VisualGeometryFeatures/serialization.jl

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

# Strip orientation
Disc(d::OrientedDisc) = Disc(d.center, d.σ)

# =============================================================================
# EllipticFrame (5 DOF) - Anisotropic Frame
# =============================================================================

"""
    EllipticFrame{T} <: FeatureFrame{T}

Anisotropic feature frame with 5 degrees of freedom: translation (2) + symmetric shape (3).

Defines the transformation: `x = S·u + center`

where S is a symmetric positive-definite 2×2 matrix (the shape factor,
[`shape_factor`](@ref)). The shape matrix is M = S·S' = S² (since S is
symmetric, [`shape_matrix`](@ref)): the region is `(x-c)ᵀ M⁻¹ (x-c) ≤ 1`.

This maps the unit circle to an ellipse in image space. Invariant to affine
transformations (up to in-plane rotation).

# Fields
- `center::Point2{T}`: Center position in image coordinates
- `S::SMatrix{2,2,T,4}`: Shape factor (symmetric positive-definite)

# Relationship to the shape matrix
- Shape factor: S (what we store)
- Shape matrix: M = S² = S·S'
- To get S from M: S = √M (matrix square root)

# Example
```julia
S = @SMatrix [2.0 0.5; 0.5 1.0]  # symmetric shape factor
frame = EllipticFrame(Point2(100.0, 200.0), S)
T = AffineMap(frame)  # AffineMap: u → S*u + center
ellipse = region(frame, 2.0)  # Ellipse at 2σ contour
```

See also: [`AffineFrame`](@ref), `Distributions.MvNormal`
"""
struct EllipticFrame{T} <: FeatureFrame{T}
    center::Point2{T}
    S::SMatrix{2, 2, T, 4}  # symmetric shape factor (M = S²)
end

const DEFAULT_MAX_ASPECT_RATIO = 10.0

"""
    is_valid_shape(S; max_aspect_ratio=DEFAULT_MAX_ASPECT_RATIO) -> Bool

Check that a 2×2 shape factor is positive-definite and has
[`aspect_ratio`](@ref) `√(σ_max/σ_min) ≤ max_aspect_ratio` (the frame's, of
`EllipticFrame(c, S)`). Closed form, without building the frame or a
decomposition: `σ_max·σ_min = |det S|` and `σ_max² + σ_min² = ‖S‖²_F`, so
with `r = σ_max/σ_min ≥ 1` and `k = max_aspect_ratio` the test `√r ≤ k` is
`r + 1/r ≤ k² + 1/k²`, i.e. `‖S‖²_F ≤ (k² + k⁻²)·|det S|`. Allocation-free and
branch-light in the element type: callable inside a kernel on a static 2×2
(a fitted peak's negated quadratic form `-C` after [`is_peak`](@ref)).
"""
function is_valid_shape(
        S::AbstractMatrix;
        max_aspect_ratio::Real = DEFAULT_MAX_ASPECT_RATIO
    )
    is_positive_definite(S) || return false
    T = float(eltype(S))
    k2 = T(max_aspect_ratio)^2
    d = abs(S[1, 1] * S[2, 2] - S[1, 2] * S[2, 1])
    return S[1, 1]^2 + S[1, 2]^2 + S[2, 1]^2 + S[2, 2]^2 ≤ (k2 + inv(k2)) * d
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
(`shape_matrix`, `is_peak`, `EllipticFrame`) works uniformly.

# Fields
- `a₀::T`: Intercept (peak value)
- `b::SVector{2,T}`: Linear gradient
- `C::SMatrix{2,2,T,4}`: Quadratic Hessian (neg-def for valid peaks)

# Constructors
- `Paraboloid3D(θ::SVector{6})`: Full anisotropic `{T,6}`
- `Paraboloid3D(θ::SVector{4})`: Isotropic with center `{T,4}`
- `Paraboloid3D(θ::SVector{2})`: Fixed-center isotropic `{T,2}`
- `Paraboloid3D(d::Disc, a₀)`: From detection `{T,2}`

See also: [`shape_matrix`](@ref), [`is_peak`](@ref), [`coeffs`](@ref), [`EllipticFrame`](@ref)
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
    shape_matrix(p::Paraboloid3D) -> SMatrix{2,2}

The covariance of the equivalent Gaussian, `Σ = -½C⁻¹`, as the matrix of its
1σ contour (`M = Σ`).
"""
function shape_matrix(p::Paraboloid3D{T}) where {T}
    C = p.C
    d = det(C)
    inv_det = T(-0.5) / d
    return SMatrix{2, 2, T}(
        C[2, 2] * inv_det, -C[2, 1] * inv_det,
        -C[1, 2] * inv_det, C[1, 1] * inv_det
    )
end

"""
    is_positive_definite(M::StaticMatrix{N, N}) -> Bool
    is_positive_definite(M::AbstractMatrix) -> Bool

Positive-definiteness by Sylvester's criterion: every leading principal minor of the
symmetric `M` is positive. A static `N × N` matrix tests all `N` minors, unrolled in
plain arithmetic (`N ≤ 3`: the cofactor expansions; larger `N`: StaticArrays' `det`
of each leading block); any other matrix tests its leading 2×2 block. At `N = 2` the
expression is `M[1,1] > 0 && M[1,1]·M[2,2] − M[1,2]·M[2,1] > 0`, as before.
Allocation-free and kernel-callable (no factorization). The test behind
[`is_valid_shape`](@ref) (and, negated, behind [`is_peak`](@ref)); `λ_min(M) > c` is
`is_positive_definite(M − c·I)`. Public, not exported.
"""
is_positive_definite(M::AbstractMatrix) = M[1, 1] > 0 && M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1] > 0
@generated function is_positive_definite(M::StaticMatrix{N, N}) where {N}
    N == 1 && return :(@inbounds M[1, 1] > 0)
    tests = Any[:(M[1, 1] > 0), :(M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1] > 0)]
    N >= 3 && push!(tests, :(
        M[1, 1] * (M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]) - M[1, 2] * (M[2, 1] * M[3, 3] - M[2, 3] * M[3, 1]) +
            M[1, 3] * (M[2, 1] * M[3, 2] - M[2, 2] * M[3, 1]) > 0
    ))
    for k in 4:N
        push!(tests, :(LinearAlgebra.det(SMatrix{$k, $k}($([:(M[$i, $j]) for j in 1:k for i in 1:k]...))) > 0))
    end
    return :(@inbounds $(foldr((a, b) -> :($a && $b), tests)))
end

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
    is_peak(C::StaticMatrix{2, 2}) -> Bool

Whether the quadratic form `C` of a paraboloid `z = a₀ + bᵀx + xᵀCx` opens
downward (negative definite: `C₁₁ < 0`, `det C > 0`): the paraboloid's peak
test on its 2×2 coefficients, closed form and kernel-callable — a device fit
that holds `(C₁₁, C₁₂, C₂₂)` tests `is_peak(C)` and then
[`is_valid_shape`](@ref)`(-C; max_aspect_ratio)` without building a
`Paraboloid3D`.
"""
is_peak(C::StaticMatrix{2, 2}) = is_negative_definite(C)

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

# The symmetric square root S = M^½ of a 2×2 shape matrix (M = S·S): the ONE
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
        throw(ArgumentError("shape matrix is not positive semidefinite (eigenvalues $(Tuple(λ)))"))
    S = E.vectors * Diagonal(sqrt.(max.(λ, zero(eltype(λ))))) * E.vectors'
    return SMatrix{2, 2}(S[1, 1], S[1, 2], S[1, 2], S[2, 2])
end

# Construct from the shape matrix (M → S via matrix sqrt); from a Gaussian's covariance, its 1σ contour (M = Σ)
EllipticFrame(center::Point2{T}, Σ::AbstractMatrix, ::Val{:shape_matrix}) where {T} =
    EllipticFrame{T}(Point2{T}(center), SMatrix{2, 2, T}(_sym_sqrt(Σ)))

# Construct from Paraboloid3D fit: Σ = -½C⁻¹, center shifted by Σ*b
function EllipticFrame(center::Point2{T}, p::Paraboloid3D{T, P}) where {T, P}
    Σ = shape_matrix(p)
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

# Construct from Ellipse (the symmetric shape factor from the shape matrix)
function EllipticFrame(e::Ellipse{T}) where {T}
    # EllipticFrame uses symmetric S where M = S²
    # Ellipse has M = A*A' where its shape factor A = R(θ)*diag(a,b) is non-symmetric
    # Compute S = M^(1/2) via eigendecomposition
    return EllipticFrame{T}(e.center, SMatrix{2, 2, T}(_sym_sqrt(shape_matrix(e))))
end

# Properties (the shape factor: the per-type primitive the hoisted verbs read)
shape_factor(f::EllipticFrame) = f.S

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
- `A::SMatrix{2,2,T,4}`: Shape factor (general, not necessarily symmetric; shape matrix M = A·Aᵀ)

# Decomposition
Any A can be decomposed as A = R(φ)·S where:
- S is symmetric positive-definite (the "pure shape")
- R(φ) is an in-plane rotation (the extra DOF)

# Example
```julia
A = @SMatrix [2.0 0.5; -0.5 1.0]  # general shape factor
frame = AffineFrame(Point2(100.0, 200.0), A)
T = AffineMap(frame)  # AffineMap: u → A*u + center
```

See also: [`EllipticFrame`](@ref)
"""
struct AffineFrame{T} <: FeatureFrame{T}
    center::Point2{T}
    A::SMatrix{2, 2, T, 4}  # general shape factor (M = A·Aᵀ)

    function AffineFrame{T}(center::Point2{T}, A::SMatrix{2, 2, T, 4}) where {T}
        det(A) > 0 || throw(ArgumentError("the shape factor A must have positive determinant"))
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
shape_factor(f::AffineFrame) = f.A
# A disc's shape factor: `σ I`; an oriented disc's `σ R(θ)` (its canonical frame turned by θ).
shape_factor(d::Disc{T}) where {T} = SMatrix{2, 2, T}(d.σ, 0, 0, d.σ)
function shape_factor(d::OrientedDisc{T}) where {T}
    c, s = cos(d.θ), sin(d.θ)
    return d.σ * SMatrix{2, 2, T}(c, s, -s, c)
end

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
_characteristic_scale(::ShapeFactorScale, f) = sqrt(abs(det(shape_factor(f))))
_characteristic_scale(::IsotropicScale, f) = f.σ

"""
    scale(f::FeatureFrame) -> σ

The single scale σ of an isotropic frame (`Disc`, `OrientedDisc`): its
[`characteristic_scale`](@ref). A shape-factor frame has no single σ (read
`characteristic_scale` or `shape_factor`), and a `PointFrame` none at all.
"""
scale(f::FeatureFrame) = _isotropic_scale(frame_scale(f), f)
_isotropic_scale(::IsotropicScale, f) = f.σ

"""
    shape_matrix(f::FeatureFrame) -> SMatrix{2, 2}

The frame's shape matrix `M = F·Fᵀ` (`F` its [`shape_factor`](@ref)): the region
is `(x − c)ᵀ M⁻¹ (x − c) ≤ 1`. An isotropic frame's is exactly `σ² I`, oriented
or not (the orientation turns nothing round).
"""
shape_matrix(f::FeatureFrame) = _shape_matrix(frame_scale(f), f)
_shape_matrix(::IsotropicScale, d) = (s2 = d.σ * d.σ; z = zero(s2); SMatrix{2, 2, typeof(s2), 4}(s2, z, z, s2))
_shape_matrix(::ShapeFactorScale, f) = (F = shape_factor(f); F * F')

aspect_ratio(f::FeatureFrame) = _aspect_ratio(frame_scale(f), f)
function _aspect_ratio(::ShapeFactorScale, f)
    svs = svdvals(shape_factor(f))
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
CoordinateTransformations.AffineMap(f::FeatureFrame) = _affine_map_frame(frame_scale(f), f)
_affine_map_frame(::Union{IsotropicScale, ShapeFactorScale}, f::FeatureFrame{T}) where {T} =
    AffineMap(shape_factor(f), SVector{2, T}(f.center...))

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
_frame_region(::ShapeFactorScale, f, k) = Ellipse(f, k)

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
# through the coordinate compute boundary (a detection's is its frame's: `Circle(::AbstractDetection)`)
function _frame_circle(::Union{IsotropicScale, ShapeFactorScale}, f, r)
    cx, cy, σ = _strip_coords((f.center[1], f.center[2], characteristic_scale(f)))
    return Circle(Point2(float(cx), float(cy)), r * float(σ))
end

# =============================================================================
# Ellipse Construction from FeatureFrame
# =============================================================================

"""
    Ellipse(frame::FeatureFrame, r::Real=1.0) -> Ellipse

Construct an Ellipse from a FeatureFrame at the `r`-σ contour.

For isotropic frames, returns a circle-equivalent ellipse (a == b).
For anisotropic frames, computes semi-axes via SVD of the scaled shape factor.

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
function _frame_ellipse(::ShapeFactorScale, f, r)
    M = r * shape_factor(f)
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
Base.inv(f::FeatureFrame) = _inv_frame(frame_scale(f), f)
# A disc's canonical frame x = σ·u + c inverts to u = x/σ - c/σ: centre -c/σ, σ ↦ 1/σ.
_inv_frame(::IsotropicScale, d::Disc{T}) where {T} = Disc{T}(Point2{T}(-d.center[1] / d.σ, -d.center[2] / d.σ), one(T) / d.σ)
# A shape-factor frame x = F·u + c inverts to u = F⁻¹x - F⁻¹c, in its own class (F⁻¹ of a symmetric S is symmetric).
function _inv_frame(::ShapeFactorScale, f::FeatureFrame{T}) where {T}
    Fi = SMatrix{2, 2, T}(inv(shape_factor(f)))
    return _with_shape(f, Point2{T}(-(Fi * SVector{2, T}(f.center...))...), Fi)
end
# The frame of `f`'s class at centre `c` with shape factor `F`: the per-type constructor the hoisted shape verbs use.
_with_shape(::EllipticFrame{T}, c, F) where {T} = EllipticFrame{T}(c, F)
_with_shape(::AffineFrame{T}, c, F) where {T} = AffineFrame{T}(c, F)

# =============================================================================
# Conversions between MvNormal and Frames
# =============================================================================

# Note: MvNormal stores covariance Σ, EllipticFrame stores the shape factor S
# Relationship: M = S·S' (S²) for symmetric S, the shape matrix; the frame is the 1σ contour, Σ = M
# EllipticFrame(d::AbstractMvNormal) is defined in mvnormal_geometry.jl

"""
    MvNormal(f::FeatureFrame) -> MvNormal

Convert a scaled frame (Disc, OrientedDisc, EllipticFrame, AffineFrame) to an
MvNormal (statistical).

The frame is read as the Gaussian's 1σ contour: its shape matrix
M = F·Fᵀ ([`shape_matrix`](@ref), F the shape factor; `σ²I` for a disc) is the
covariance. Note: For OrientedDisc and AffineFrame, this loses the in-plane
rotation since Σ = F·Fᵀ is symmetric. A `PointFrame` has none (absent).
"""
MvNormal(f::FeatureFrame) = _frame_mvnormal(frame_scale(f), f)
function _frame_mvnormal(::Union{IsotropicScale, ShapeFactorScale}, f)
    T = eltype(f.center)
    Σ = SMatrix{2, 2, T}(shape_matrix(f))
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
FlexUnits.ustrip(u::Units, f::FeatureFrame) = _map_lengths(x -> ustrip(u, x), f; angle = ustrip)

# The unit-free frame a region is drawn from: through the compute boundary
# (pixels in px, lengths in mm; plain numbers unchanged).
_strip_frame(f::FeatureFrame) = f.center[1] isa Real ? f : ustrip(_coord_unit(f.center[1]), f)

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
    *(frame::FeatureFrame, scale)

The frame with its centre and every length (σ, a shape factor) multiplied by a
scalar factor, its angles kept: a change of unit (`frame * 300dpi`) or of length
scale; `frame / scale` divides.

# Example
```julia
disc = Disc(Point2(10.0mm, 20.0mm), 2.0mm)
scaled = disc * 300dpi  # Scale to logical units
```
"""
Base.:*(f::FeatureFrame, scale::Number) = _map_lengths(x -> x * scale, f)
"""
    /(frame::FeatureFrame, scale)

The frame with its centre and every length divided by a scalar factor, its angles
kept (`frame / 300dpi`): the inverse of `frame * scale`.
"""
Base.:/(f::FeatureFrame, scale::Number) = _map_lengths(x -> x / scale, f)

# =============================================================================
# The FeatureFrame interface (conventions/feature-frames.md)
# =============================================================================
#
# Every verb is implemented ONCE, on FeatureFrame or on its DOF trait
# (`frame_scale`: PointShape, IsotropicScale, ShapeFactorScale), from the
# per-type primitives: the centre (`center`), the local shape where a frame has
# one (`shape_factor`; `σ` for the isotropic frames), and the two constructors a
# type states once — its frame from mapped lengths (`_map_lengths`) and its
# similarity image (`_similar_frame`, ct_contravariant.jl). A `PointFrame` has
# only the centre, so it reaches every verb a point has through the same code.

"""
    frame_dof(::Type{<:FeatureFrame}) -> Int
    frame_dof(f::FeatureFrame) -> Int

The frame's degrees of freedom, the size of its parameter vector and of an
uncertain frame's covariance: `PointFrame` 2, `Disc` 3, `OrientedDisc` 4,
`EllipticFrame` 5, `AffineFrame` 6. The covariance's coordinates are the centre
`(x, y)` first, then the shape's in the producer's documented chart (a `Disc`'s
`log σ`), so its first 2×2 block is always the centre's ([`center`](@ref)).
"""
frame_dof(f::FeatureFrame) = frame_dof(typeof(f))
frame_dof(::Type{<:PointFrame}) = 2
frame_dof(::Type{<:Disc}) = 3
frame_dof(::Type{<:OrientedDisc}) = 4
frame_dof(::Type{<:EllipticFrame}) = 5
frame_dof(::Type{<:AffineFrame}) = 6

# The frame with every length mapped by `g` (the centre's coordinates, σ, a
# shape factor's entries) and every angle by `angle`: the one primitive of the
# unit and scaling verbs (`ustrip`, `*`, `/`), one line per type.
_map_lengths(g::G, f::PointFrame; angle::A = identity) where {G, A} = PointFrame(Point2(g(f.center[1]), g(f.center[2])))
_map_lengths(g::G, d::Disc; angle::A = identity) where {G, A} = Disc(Point2(g(d.center[1]), g(d.center[2])), g(d.σ))
_map_lengths(g::G, d::OrientedDisc; angle::A = identity) where {G, A} =
    OrientedDisc(Point2(g(d.center[1]), g(d.center[2])), g(d.σ), angle(d.θ))   # θ in radians: a dimensionless quantity or a number
_map_lengths(g::G, f::EllipticFrame; angle::A = identity) where {G, A} = EllipticFrame(Point2(g(f.center[1]), g(f.center[2])), map(g, f.S))
_map_lengths(g::G, f::AffineFrame; angle::A = identity) where {G, A} = AffineFrame(Point2(g(f.center[1]), g(f.center[2])), map(g, f.A))

"""
    center(u::Uncertain{<:FeatureFrame}) -> Uncertain{Point2{T}}
    center(d::Attributed{<:Uncertain{<:FeatureFrame}}) -> Uncertain{Point2{T}}

THE detection → point observation conversion, for every frame type: the
frame's centre with the centre's block of its uncertainty — the first 2×2 block
of the frame's covariance, whose coordinates start with `(x, y)`
([`frame_dof`](@ref)). A covariance gives its block, in its storage (full,
diagonal or isotropic), labeled by the frame's label when the frame is 2-DOF
and unlabeled otherwise. A precision gives the precision of the centre's
marginal (`inv` of the covariance block; for a `PointFrame` its own precision,
unchanged). A detection (an `Attributed` uncertain frame with its metadata)
gives its frame's. Allocation-free, and for an `Uncertain{PointFrame}` the
identity on the numbers: no estimator, codec or consumer takes a centre and a
covariance block apart by hand.
"""
@inline center(u::Uncertain{<:FeatureFrame{T}, <:Union{CovMat, PrecMat}}) where {T} =
    Uncertain(Point2{T}(center(getfield(u, :value))), _center_block(getfield(u, :attributes)))
@inline center(d::Attributed{<:Uncertain{<:FeatureFrame}}) = center(getfield(d, :value))

# The centre's block: a 2-DOF form is the block (its frame kept); a larger covariance its leading 2×2 in its
# storage; a larger precision the marginal's precision, the inverse of the covariance's block.
@inline _center_block(Σ::Union{CovMat{2}, PrecMat{2}}) = Σ
@inline _center_block(Σ::CovMat{N, T, <:SMatrix}) where {N, T} = CovMat{2, T}(SMatrix{2, 2, T, 4}(Σ[1, 1], Σ[2, 1], Σ[1, 2], Σ[2, 2]))
@inline _center_block(Σ::CovMat{N, T, <:LinearAlgebra.Diagonal}) where {N, T} =
    CovMat(LinearAlgebra.Diagonal(SVector{2, T}(parent(Σ).diag[1], parent(Σ).diag[2])))
@inline _center_block(Σ::CovMat{N, T, <:ScaledIdentity}) where {N, T} = CovMat(ScaledIdentity{2, T}(parent(Σ).λ))
@inline _center_block(Λ::PrecMat) = inv(_center_block(inv(Λ)))

"""
    Ellipse(u::Uncertain{<:FeatureFrame}; sigma = 3) -> Ellipse

The `sigma`-σ error ellipse of an uncertain frame's centre, for every frame:
`Ellipse(center(u); sigma)`, the error ellipse of its point observation
([`center`](@ref)). (A scaled frame's REGION is `region(f, k)`, a different
verb.)
"""
Ellipse(u::Uncertain{<:FeatureFrame, <:Union{CovMat, PrecMat}}; sigma::Real = 3) = Ellipse(center(u); sigma)

"""
    extrema(f::FeatureFrame) -> (Point2, Point2)

The frame's axis-aligned bounding box at its 1σ contour, `(min, max)`: a
point's is the point twice, an isotropic frame's `center ∓ σ`, a shape-factor
frame's its region's (the ellipse's exact box). `Rect2(f)` and `Rect2{Int}(f)`
read it.
"""
Base.extrema(f::FeatureFrame) = _frame_extrema(frame_scale(f), f)
_frame_extrema(::PointShape, f::FeatureFrame) = (f.center, f.center)
function _frame_extrema(::IsotropicScale, d::FeatureFrame{T}) where {T}
    c, σ = d.center, d.σ
    return (Point2{T}(c[1] - σ, c[2] - σ), Point2{T}(c[1] + σ, c[2] + σ))
end
_frame_extrema(::ShapeFactorScale, f::FeatureFrame) = extrema(Ellipse(f))

# The scaled frames' verbs a PointFrame does not have: a MethodError with its hint (`__init__`).
const _SCALED_FRAME_VERBS = (
    :scale, :_isotropic_scale, :shape_factor, :shape_matrix, :_shape_matrix, :AffineMap, :_affine_map_frame,
    :region, :_frame_region, :Circle, :_frame_circle, :Ellipse, :_frame_ellipse, :characteristic_scale,
    :_characteristic_scale, :aspect_ratio, :_aspect_ratio, :boundary_point, :MvNormal, :_frame_mvnormal, :intersects, :inv,
    :_inv_frame,
)
function _point_frame_hint(io::IO, e::MethodError, argtypes, kwargs)
    f = e.f
    (f isa Function || f isa Type) && nameof(f) in _SCALED_FRAME_VERBS || return nothing
    any(T -> T <: PointFrame || T <: PointShape, argtypes) || return nothing
    print(io, "\nA PointFrame is the 2-DOF point feature: it has no scale and no shape, so `", nameof(f),
          "` (a verb of the scaled frames only: scale, shape_factor, shape_matrix, AffineMap, region, Circle, ",
          "Ellipse(frame, r), characteristic_scale, aspect_ratio, boundary_point, MvNormal, intersects, inv) ",
          "is absent by design. Its centre is `center(f)`, an uncertain one's point observation `center(u)` and its ",
          "error ellipse `Ellipse(u; sigma)` (VisualGeometryToolkitCore's FeatureFrame interface table).")
    return nothing
end

# =============================================================================
# Degrees of Freedom
# =============================================================================


# JSON serialization moved to VisualGeometryToolkitFeatures/serialization.jl

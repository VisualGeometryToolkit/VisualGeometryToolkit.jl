# =============================================================================
# Conic Sections: Ellipse and Circle Types
# =============================================================================
#
# This file provides:
# - Ellipse type with semantic representation (center, a, b, θ)
# - Circle extensions for GeometryBasics.Circle
# - Conic matrix conversions (HomEllipseMat, HomCircleMat)
# - Conic classification and validation
# - Gradient computation for implicit curves
#
# =============================================================================

# =============================================================================
# Ellipse Type
# =============================================================================

"""
    Ellipse{T}

Semantic ellipse representation with center, semi-major axis `a`, semi-minor axis `b`,
and orientation angle `θ` in radians.

The constructor automatically ensures `a ≥ b` by swapping axes and adjusting the angle if needed.

# Fields
- `center::Point2{T}`: Center coordinates (cx, cy)
- `a::T`: Semi-major axis length (always ≥ b)
- `b::T`: Semi-minor axis length (always ≤ a)
- `θ::T`: Orientation angle in radians [0, π)

# Constructors
```julia
# Basic constructor with Point2 center
ellipse = Ellipse(Point2(1.0, 2.0), 3.0, 2.0, π/4)

# Constructor with vector center (automatically converted to Point2)
ellipse = Ellipse([1.0, 2.0], 3.0, 2.0, π/4)

# Mixed numeric types are automatically promoted
ellipse = Ellipse([1, 2], 3.0f0, 2, π/4)
```

# Integration with GeometryBasics
```julia
# Generate boundary points using GeometryBasics.coordinates
points = GeometryBasics.coordinates(ellipse, 64)

# Access geometric properties
center = GeometryBasics.coordinates(ellipse)  # Returns ellipse.center
major_radius = GeometryBasics.radius(ellipse)  # Returns max(a, b)
```
"""
struct Ellipse{T}
    center::Point2{T}  # (cx, cy)
    a::T               # semimajor (should be >= b)
    b::T               # semiminor (should be <= a)
    θ::T               # radians

    # Inner constructor to ensure a >= b (swap if needed)
    function Ellipse{T}(center::Point2{T}, a::T, b::T, θ::T) where {T}
        return if a >= b
            new{T}(center, a, b, θ)
        else
            # Swap axes and rotate by π/2
            new{T}(center, b, a, θ + T(π / 2))
        end
    end
end

# Outer constructors for convenience
Ellipse(center::Point2{T}, a::T, b::T, θ::T) where {T} = Ellipse{T}(center, a, b, θ)
Ellipse(e::Ellipse) = e

# Accept Point2 with mixed numeric types and promote
Ellipse(center::Point2{T}, a::Real, b::Real, θ::Real) where {T} = begin
    S = promote_type(T, typeof(a), typeof(b), typeof(θ))
    Ellipse{S}(Point2{S}(center), S(a), S(b), S(θ))
end

# Accept AbstractVector (but not Point2) and convert to Point2
Ellipse(center::AbstractVector{T}, a::T, b::T, θ::T) where {T} = begin
    length(center) == 2 || throw(ArgumentError("Center must be a 2D vector, got length $(length(center))"))
    Ellipse{T}(Point2(center[1], center[2]), a, b, θ)
end

# Accept AbstractVector with mixed types and promote
Ellipse(center::AbstractVector, a::Real, b::Real, θ::Real) = begin
    length(center) == 2 || throw(ArgumentError("Center must be a 2D vector, got length $(length(center))"))
    T = promote_type(eltype(center), typeof(a), typeof(b), typeof(θ))
    Ellipse(Point2{T}(center[1], center[2]), T(a), T(b), T(θ))
end

"""
    Ellipse(center, A::AbstractMatrix)

Refused: a plain 2×2 matrix is ambiguous — it could be a moment matrix Σ
(1σ contour `(x-c)ᵀ Σ⁻¹ (x-c) = 1`) or an affine "square root" A with
Σ = A·A', and for symmetric input the two interpretations silently differ
by a squaring. Say which one you mean:

- moment matrix: `Ellipse(center, Symmetric(Σ))`
- affine square root (e.g. Baumberg adaptation): `Ellipse(center, Symmetric(A * A'))`
"""
function Ellipse(center, A::AbstractMatrix)
    throw(
        ArgumentError(
            "Ellipse(center, A::AbstractMatrix) is ambiguous: supply the moment " *
                "matrix explicitly as Ellipse(center, Symmetric(Σ)), or for an " *
                "affine square root A use Ellipse(center, Symmetric(A * A'))."
        )
    )
end

"""
    Ellipse(center, Σ::Symmetric) -> Ellipse

Construct an ellipse from a 2×2 moment matrix (symmetric PD). The boundary is the
1σ ellipse: `{x : (x-c)ᵀ Σ⁻¹ (x-c) = 1}`.

Semi-axes are `√λᵢ` where `λᵢ` are eigenvalues of `Σ`.

```julia
Σ = Symmetric([4.0 1.0; 1.0 9.0])
e = Ellipse(Point2(100.0, 200.0), Σ)
```
"""
function Ellipse(center, Σ::Symmetric{<:Real})
    size(Σ) == (2, 2) || throw(ArgumentError("Σ must be 2×2, got size $(size(Σ))"))
    eig = eigen(Σ)
    all(v -> v > 0, eig.values) || throw(ArgumentError("Σ must be positive definite, got eigenvalues $(eig.values)"))
    a = sqrt(eig.values[2])  # semi-major
    b = sqrt(eig.values[1])  # semi-minor
    v = eig.vectors[:, 2]
    θ = atan(v[2], v[1])
    return Ellipse(center, a, b, θ)
end

"""
    Ellipse(d::Uncertain{<:AbstractVector, <:Union{CovMat{2}, PrecMat{2}}}; sigma=3) -> Ellipse

The `sigma`-σ confidence ellipse of an uncertain 2D point: `Ellipse(center,
Σ::Symmetric)` at its covariance [`cov`](@ref)`(d)` (a stored precision is
inverted once), semi-axes scaled by `sigma`.

```julia
Σ = CovMat{2,Float64}(4.0, 1.0, 1.0, 9.0)
e = Ellipse(Uncertain(Point2(123.4, 56.7), Σ); sigma=3)
Λ = PrecMat{2,Float32}(2.0f0, 0.1f0, 0.1f0, 3.0f0)
e = Ellipse(Uncertain(Point2(123.4, 56.7), Λ); sigma=3)
```
"""
function Ellipse(
        d::Uncertain{<:AbstractVector, <:Union{CovMat{2}, PrecMat{2}}};
        sigma::Real = 3
    )
    Σ = Symmetric(SMatrix{2, 2, Float64}(cov(d)))
    e1σ = Ellipse(getfield(d, :value), Σ)
    return Ellipse(e1σ.center, sigma * e1σ.a, sigma * e1σ.b, e1σ.θ)
end

# Unit circle conic: x² + y² - 1 = 0
const CIRCLE = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -1.0]

"""
    Ellipse(c::Circle) -> Ellipse

Convert a Circle to an Ellipse with equal semi-axes.

# Example
```julia
circle = Circle(Point2(100.0, 200.0), 5.0)
ellipse = Ellipse(circle)  # Ellipse with a=b=5.0, θ=0
```
"""
function Ellipse(c::GeometryBasics.Circle{T}) where {T}
    r = GeometryBasics.radius(c)
    return Ellipse{T}(c.center, r, r, zero(T))
end

# =============================================================================
# Ellipse GeometryBasics Interface
# =============================================================================

"""
    GeometryBasics.radius(e::Ellipse)

Return the major axis radius of the ellipse as required by GeometryBasics interface.
"""
GeometryBasics.radius(e::Ellipse) = max(e.a, e.b)  # Major axis radius

"""
    GeometryBasics.origin(e::Ellipse) -> Point2

Get the center of an ellipse as its origin, making it compatible with generic
GeometryBasics primitives for coordinate convention conversion and transformations.
"""
GeometryBasics.origin(e::Ellipse) = e.center

"""
    centroid(e::Ellipse) -> Point2

Return the centroid of an ellipse.
"""
centroid(e::Ellipse) = e.center

"""
    area(e::Ellipse) -> T

Return the area of an ellipse: π * a * b.
"""
area(e::Ellipse) = π * e.a * e.b

"""
    shape_matrix(e::Ellipse) -> SMatrix{2,2}

Return the 2×2 shape matrix A where the ellipse boundary satisfies ‖A⁻¹(x - c)‖ = 1.

The shape matrix is A = R(θ) * diag(a, b) where R is the rotation matrix.
"""
function shape_matrix(e::Ellipse{T}) where {T}
    cosθ, sinθ = cos(e.θ), sin(e.θ)
    # R(θ) * diag(a, b) = [cosθ -sinθ; sinθ cosθ] * [a 0; 0 b]. `float(T)`: an
    # integer-typed ellipse has a non-integer shape matrix (it threw an
    # InexactError here); float(T) is the identity for every float and Dual T.
    return SMatrix{2, 2, float(T)}(
        e.a * cosθ, e.a * sinθ,
        -e.b * sinθ, e.b * cosθ
    )
end

"""
    moment_matrix(e::Ellipse) -> SMatrix{2,2}

Return the moment matrix M = A * A' where A is the shape matrix.

This represents the ellipse as a 1-σ contour: `{x : (x-c)ᵀ M⁻¹ (x-c) = 1}`.
"""
function moment_matrix(e::Ellipse{T}) where {T}
    A = shape_matrix(e)
    return A * A'
end

"""
    aspect_ratio(e::Ellipse) -> T

Return the aspect ratio of an ellipse: a/b (semi-major / semi-minor).

A value of 1.0 indicates a circle. Higher values indicate more elongated ellipses.
"""
aspect_ratio(e::Ellipse) = e.a / e.b

# =============================================================================
# Ellipse Boundary Sampling
# =============================================================================

"""
    boundary_point(e::Ellipse, θ::Real) -> Point2

Return a point on the ellipse boundary at parameter θ.

The parameterization is:
    p(θ) = center + [a*cos(θ)*cos(φ) - b*sin(θ)*sin(φ),
                     a*cos(θ)*sin(φ) + b*sin(θ)*cos(φ)]
where φ is the ellipse orientation angle.
"""
function boundary_point(e::Ellipse{T}, θ::Real) where {T}
    cos_θ, sin_θ = cos(θ), sin(θ)
    cos_φ, sin_φ = cos(e.θ), sin(e.θ)
    x = e.center[1] + e.a * cos_θ * cos_φ - e.b * sin_θ * sin_φ
    y = e.center[2] + e.a * cos_θ * sin_φ + e.b * sin_θ * cos_φ
    return Point2{T}(x, y)
end

# GeometryBasics' vertex interface for an ellipse — the standard-interface
# FACADE over `boundary_point`, not a second sampler: the θ → point map has
# one spelling, `boundary_point(::Ellipse, θ)`. `Ellipse3D` delegates the
# same way (see its `coordinates`).
#
# CLOSED RING, exactly as `GeometryBasics.coordinates(::Circle, n)`: `n` points
# of which the last repeats the first, so `n - 1` are distinct. An ellipse is
# the circle's analogue, and a caller who swaps one for the other must not get
# a curve that no longer closes — `lines(coordinates(e))` would show the gap.
# This is a MESHING contract and is not `boundary_points`, which returns `n`
# DISTINCT samples and is what algorithmic callers want.
function GeometryBasics.coordinates(e::Ellipse, nvertices::Int = 32)
    nvertices < 2 && return boundary_points(e, max(nvertices, 0))
    # Exact size at once: the θ grid INCLUDES 2π, whose sample is the first
    # point again up to rounding, so overwrite it to close EXACTLY. Sampling
    # `nvertices - 1` and `push!`ing the closing vertex reallocated the whole
    # array on every call.
    ps = [boundary_point(e, θ) for θ in range(0, 2π; length = nvertices)]
    @inbounds ps[end] = ps[1]
    return ps
end

"""
    Base.extrema(e::Ellipse) -> (Point2, Point2)

Compute the axis-aligned bounding box of the ellipse analytically.

Returns `(min_point, max_point)` where min_point contains the minimum x and y coordinates,
and max_point contains the maximum x and y coordinates.
"""
function Base.extrema(e::Ellipse{T}) where {T}
    cos_θ = cos(e.θ)
    sin_θ = sin(e.θ)

    rx = sqrt((e.a * cos_θ)^2 + (e.b * sin_θ)^2)
    ry = sqrt((e.a * sin_θ)^2 + (e.b * cos_θ)^2)

    min_pt = Point2{T}(e.center[1] - rx, e.center[2] - ry)
    max_pt = Point2{T}(e.center[1] + rx, e.center[2] + ry)

    return (min_pt, max_pt)
end

# =============================================================================
# Circle Interface Methods
# =============================================================================

"""
    GeometryBasics.Circle(det::AbstractDetection, cutoff::Real=1.0) -> Circle

Construct a unitless GeometryBasics.Circle from a detection with radius
`cutoff * σ` (σ the frame's `characteristic_scale`), through the checked unit
strip. The geometry is the detection's [`feature_frame`](@ref), so a detection
of an `Uncertain` frame gives the circle of its frame.
"""
function GeometryBasics.Circle(det::AbstractDetection, cutoff::Real = 1.0)
    f = feature_frame(det)
    cx, cy, σ = _strip_coords((f.center[1], f.center[2], characteristic_scale(f)))
    return GeometryBasics.Circle(Point2(float(cx), float(cy)), cutoff * float(σ))
end

"""
    aspect_ratio(c::Circle) -> T

Return the aspect ratio of a circle: always 1.0.
"""
aspect_ratio(c::GeometryBasics.Circle) = one(typeof(c.r))

"""
    centroid(c::Circle) -> Point2

Return the centroid of a circle.
"""
centroid(c::GeometryBasics.Circle) = c.center

"""
    area(c::Circle) -> T

Return the area of a circle: π * r².
"""
area(c::GeometryBasics.Circle) = π * GeometryBasics.radius(c)^2

"""
    boundary_point(c::Circle, θ::Real) -> Point2

Return a point on the circle boundary at parameter θ.
"""
function boundary_point(c::GeometryBasics.Circle{T}, θ::Real) where {T}
    r = GeometryBasics.radius(c)
    return Point2{T}(c.center[1] + r * cos(θ), c.center[2] + r * sin(θ))
end

# =============================================================================
# Conic Matrix Construction
# =============================================================================

"""
    HomEllipseMat(e::Ellipse{T}) -> HomEllipseMat{T}

Construct conic matrix Q from Ellipse via Q = A^{-T} * CIRCLE * A^{-1},
where A maps unit circle → ellipse.
"""
function HomEllipseMat(e::Ellipse{T}) where {T}
    L = shape_matrix(e)   # R(θ)·diag(a, b): unit circle → ellipse, linear part
    A = @SMatrix [
        L[1, 1]   L[1, 2]   e.center[1];
        L[2, 1]   L[2, 2]   e.center[2];
        T(0)      T(0)      T(1)
    ]

    Q = propagate(transpose(inv(A)), CIRCLE)   # A^{-T}·CIRCLE·A^{-1}

    return HomEllipseMat{T}(Tuple(Q))
end

"""
    HomCircleMat(c::Circle{T}) -> HomCircleMat

Construct conic matrix Q from Circle via Q = A^{-T} * CIRCLE * A^{-1}.
Units are stripped from the circle to create a unitless conic matrix.
"""
function HomCircleMat(c::Circle{T}) where {T}
    cx, cy, r = map(float, _strip_coords((c.center[1], c.center[2], c.r)))
    U = typeof(r)

    A = @SMatrix [
        r       U(0)    cx;
        U(0)    r       cy;
        U(0)    U(0)    U(1)
    ]

    Q = propagate(transpose(inv(A)), CIRCLE)   # A^{-T}·CIRCLE·A^{-1}

    return HomCircleMat{U}(Tuple(Q))
end

# =============================================================================
# Circle → HomCircleMat delegation
# =============================================================================

# Delegate HomEllipseMat(::Circle) to HomCircleMat for type correctness
HomEllipseMat(c::Circle) = HomCircleMat(c)

# =============================================================================
# Conic Classification and Validation
# =============================================================================

# Symmetrize + unit-Frobenius normalization shared by the conic classifiers.
# Returns `nothing` for a numerically null matrix (degenerate).
@inline function _normalized_symmetric_conic(Q::StaticMatrix{3, 3, T}) where {T}
    C_raw = SMatrix{3, 3, T}(Q)
    C_sym = (C_raw + transpose(C_raw)) / T(2)
    n = norm(C_sym)
    (iszero(n) || !isfinite(n)) && return nothing   # relative only: a homogeneous conic has no absolute scale
    return C_sym / n
end

"""
    classify_conic(Q::Union{HomEllipseMat, HomCircleMat}) -> ConicTrait

Classify a conic matrix and return the appropriate Holy Trait.

Returns one of:
- `EllipseTrait()`: Valid ellipse (bounded, non-degenerate)
- `HyperbolaTrait()`: Hyperbola (indefinite quadratic form)
- `ParabolaTrait()`: Parabola (singular quadratic form)
- `DegenerateTrait()`: a point ellipse, an imaginary ellipse, or a conic with
  no quadratic part (a line). Intersecting line pairs classify as
  `HyperbolaTrait` and parallel pairs as `ParabolaTrait` (their quadratic forms).

Invariant to the conic's scale and sign (a homogeneous conic is defined only
up to a nonzero multiplier). A point ellipse is told from a real one by γ, the
constant term at the centre, against the rounding of the terms that produce it
(`|c| + 2|b||centre| + |λ|·|centre|²`) magnified by the quadratic block's
condition number κ — never against an absolute threshold, which called large or
distant ellipses degenerate. The bound was measured: on
20,000 elongated point ellipses (κ up to 1e6) the computed |γ| stays below
0.55·κ·eps of those terms, and the test allows 2·(1 + κ)·eps. An ellipse whose γ
is within that noise — thin and far along its minor axis, or small in Float32 —
reads degenerate, because the arithmetic cannot tell it from a point.
"""
function classify_conic(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}) where {T}
    C = _normalized_symmetric_conic(Q)
    C === nothing && return DegenerateTrait()

    A = SMatrix{2, 2, T}(C[1, 1], C[2, 1], C[1, 2], C[2, 2])
    b = SVector{2, T}(C[1, 3], C[2, 3])
    c = C[3, 3]

    # the quadratic block's eigenvalues ordered by MAGNITUDE: `extrema` orders
    # them by value, so a negated conic read its ratio upside down
    λ1, λ2 = eigen(Symmetric(A)).values
    small, large = abs(λ1) <= abs(λ2) ? (λ1, λ2) : (λ2, λ1)
    iszero(large) && return DegenerateTrait()          # no quadratic part: a line

    abs(small / large) < eps(T)^(T(3) / T(4)) && return ParabolaTrait()
    small * large < zero(T) && return HyperbolaTrait()

    # definite: flip a negated conic so the block is positive definite
    if large < zero(T)
        A, b, c = -A, -b, -c
    end
    center = -(A \ b)
    γ = c + dot(b, center)
    # γ < 0 is a real ellipse, unless it does not rise above the rounding of the
    # terms that produce it (a point ellipse is γ = 0 up to that rounding)
    κ = abs(large / small)
    terms = abs(c) + 2 * norm(b) * norm(center) + abs(large) * dot(center, center)
    return γ < -2 * (1 + κ) * eps(T) * terms ? EllipseTrait() : DegenerateTrait()
end

# Trivial classification for geometric primitives
classify_conic(::Ellipse) = EllipseTrait()
classify_conic(::Circle) = CircleTrait()

"""
    is_ellipse(Q) -> Bool

Whether `Q` — a conic matrix, an `Ellipse` or a `Circle` — is a real,
non-degenerate ellipse: [`classify_conic`](@ref) answering `EllipseTrait` or
`CircleTrait`, so the two can never disagree.
"""
is_ellipse(Q) = classify_conic(Q) isa Union{EllipseTrait, CircleTrait}

# =============================================================================
# Ellipse Extraction from Conic Matrix
# =============================================================================

"""
    Ellipse(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}) -> Ellipse{T}

Extract Ellipse from conic matrix Q using robust eigenvalue decomposition.
"""
function Ellipse(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}) where {T}
    C = SMatrix{3, 3, T}(Q)

    A = SMatrix{2, 2, T}(C[1, 1], C[2, 1], C[1, 2], C[2, 2])
    b = SVector{2, T}(C[1, 3], C[2, 3])

    center_vec = -(A \ b)
    γ = C[3, 3] + dot(b, center_vec)

    A_sym = (A + transpose(A)) / T(2)
    eigen_result = eigen(Symmetric(A_sym))
    eigenvals = eigen_result.values
    eigenvecs = eigen_result.vectors

    # Clamp to avoid DomainError from numerically negative arguments
    a1 = sqrt(max(zero(T), -γ / eigenvals[1]))
    a2 = sqrt(max(zero(T), -γ / eigenvals[2]))

    if a1 >= a2
        a, b_axis = a1, a2
        major_axis_vec = Vec2{T}(eigenvecs[1, 1], eigenvecs[2, 1])
    else
        a, b_axis = a2, a1
        major_axis_vec = Vec2{T}(eigenvecs[1, 2], eigenvecs[2, 2])
    end

    θ = atan(major_axis_vec[2], major_axis_vec[1])
    θ = mod(θ, T(π))

    return Ellipse{T}(Point2{T}(center_vec), a, b_axis, θ)
end

"""
    Circle(Q::Union{HomEllipseMat, HomCircleMat}) -> Circle

Extract circle parameters from conic matrix.
Throws an error if the conic doesn't represent a circle (a and b differ significantly).
"""
function GeometryBasics.Circle(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}; atol = 1.0e-6) where {T}
    ellipse = Ellipse(Q)

    if !isapprox(ellipse.a, ellipse.b; atol = atol)
        error("Conic does not represent a circle: a=$(ellipse.a), b=$(ellipse.b)")
    end

    r = (ellipse.a + ellipse.b) / 2
    return GeometryBasics.Circle(ellipse.center, r)
end

# =============================================================================
# Gradient Computation
# =============================================================================

"""
    gradient(Q::HomEllipseMat{T}, p::Point2{T}) -> Vec2{T}

Compute the gradient of the conic implicit function at point p.

For conic equation x^T Q x = 0, the gradient is ∇f(x) = 2Qx (homogeneous form).
Returns the 2D Euclidean gradient [∂f/∂x, ∂f/∂y] as a `Vec2` — it's a
direction (steepest-ascent), not a position.
"""
function gradient(Q::HomEllipseMat{T}, p::Point2{T}) where {T}
    Q_mat = SMatrix{3, 3, T}(Q)
    grad_h = 2 * Q_mat * to_affine(p)
    return Vec2{T}(grad_h[1], grad_h[2])
end

# Allow mixed types with automatic promotion
function gradient(Q::HomEllipseMat{T1}, p::Point2{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return gradient(HomEllipseMat{T}(Q.data), Point2{T}(p))
end

# Allow AbstractVector as input
function gradient(Q::HomEllipseMat{T}, p::AbstractVector) where {T}
    length(p) == 2 || throw(ArgumentError("Point must be 2D, got length $(length(p))"))
    return gradient(Q, Point2{T}(p[1], p[2]))
end

# =============================================================================
# Conic geometric primitive — delegates to matrix types
# =============================================================================

is_ellipse(c::Conic) = is_ellipse(HomEllipseMat(c))
classify_conic(c::Conic) = classify_conic(HomEllipseMat(c))
Ellipse(c::Conic) = Ellipse(HomEllipseMat(c))
GeometryBasics.Circle(c::Conic; kw...) = Circle(HomEllipseMat(c); kw...)
gradient(c::Conic, p) = gradient(HomEllipseMat(c), p)

# Identity constructors — idempotent conversion
GeometryBasics.Circle(c::Circle) = c
Conic(c::Conic) = c

# Construct Conic from geometric primitives
Conic(e::Ellipse) = Conic(HomEllipseMat(e))
Conic(c::Circle) = Conic(HomCircleMat(c))

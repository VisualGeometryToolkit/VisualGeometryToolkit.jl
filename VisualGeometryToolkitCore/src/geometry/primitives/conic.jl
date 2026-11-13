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

Refused: a plain 2×2 matrix is ambiguous — it could be a shape matrix M
(`(x-c)ᵀ M⁻¹ (x-c) = 1`, the 1σ contour of a covariance M) or a shape factor
A with M = A·A', and for symmetric input the two interpretations silently
differ by a squaring. Say which one you mean:

- shape matrix: `Ellipse(center, Symmetric(M))`
- shape factor (e.g. Baumberg adaptation): `Ellipse(center, Symmetric(A * A'))`
"""
function Ellipse(center, A::AbstractMatrix)
    throw(
        ArgumentError(
            "Ellipse(center, A::AbstractMatrix) is ambiguous: supply the shape " *
                "matrix explicitly as Ellipse(center, Symmetric(M)), or for a " *
                "shape factor A use Ellipse(center, Symmetric(A * A'))."
        )
    )
end

"""
    Ellipse(center, Σ::Symmetric) -> Ellipse

Construct an ellipse from its 2×2 shape matrix (symmetric PD; a covariance's
1σ contour). The boundary is `{x : (x-c)ᵀ Σ⁻¹ (x-c) = 1}`.

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

# The zero conic — the point ellipse/circle at the origin. It is the stack's `zero(M)` placeholder of a model type, which
# fills an empty minimal-solver slot (`FixedModels`) and the estimate of a RANSAC that finds no model (`:no_model`); it is
# never read as a model.
Base.zero(::Type{Ellipse{T}}) where {T} = Ellipse{T}(zero(Point2{T}), zero(T), zero(T), zero(T))
Base.zero(::Type{GeometryBasics.Circle{T}}) where {T} = GeometryBasics.Circle{T}(zero(Point2{T}), zero(T))

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
    shape_factor(e::Ellipse) -> SMatrix{2,2}

The 2×2 factor `A = R(θ)·diag(a, b)` of the ellipse's [`shape_matrix`](@ref)
`M = A·Aᵀ` (`R` the rotation by θ): the boundary is `‖A⁻¹(x - c)‖ = 1`, the
unit circle mapped by `x = c + A u`.
"""
function shape_factor(e::Ellipse{T}) where {T}
    cosθ, sinθ = cos(e.θ), sin(e.θ)
    # R(θ) * diag(a, b) = [cosθ -sinθ; sinθ cosθ] * [a 0; 0 b]. `float(T)`: an
    # integer-typed ellipse has a non-integer shape factor (it threw an
    # InexactError here); float(T) is the identity for every float and Dual T.
    return SMatrix{2, 2, float(T)}(
        e.a * cosθ, e.a * sinθ,
        -e.b * sinθ, e.b * cosθ
    )
end

"""
    shape_matrix(e::Ellipse) -> SMatrix{2,2}

The ellipse's shape matrix `M = A·Aᵀ`, `A` its [`shape_factor`](@ref): the
boundary is `{x : (x-c)ᵀ M⁻¹ (x-c) = 1}` (the 1σ contour of `N(c, M)`).
"""
function shape_matrix(e::Ellipse{T}) where {T}
    A = shape_factor(e)
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
    L = shape_factor(e)   # R(θ)·diag(a, b): unit circle → ellipse, linear part
    A = @SMatrix [
        L[1, 1]   L[1, 2]   e.center[1];
        L[2, 1]   L[2, 2]   e.center[2];
        T(0)      T(0)      T(1)
    ]

    Q = propagate(transpose(inv(A)), CIRCLE)   # A^{-T}·CIRCLE·A^{-1}

    return HomEllipseMat{T}(_averaged_coefficients(T, Q), _Unchecked())   # an ellipse by construction
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

    return HomCircleMat{U}(_averaged_coefficients(U, Q), _Unchecked())   # a circle by construction
end

# =============================================================================
# Circle → HomCircleMat delegation
# =============================================================================

# Delegate HomEllipseMat(::Circle) to HomCircleMat for type correctness
HomEllipseMat(c::Circle) = HomCircleMat(c)

# =============================================================================
# Conic Classification and Validation
# =============================================================================

# (classify_conic of a conic matrix — the six coefficients — is in base/types.jl,
# beside the class checks of the conic constructors, which run the same test.)

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
    # of its class: no check here. Read in the precision the class was decided
    # in (classify_conic): a Float32 conic at pixel coordinates loses γ to
    # cancellation in Float32 arithmetic.
    q = getfield(Q, :coeffs)
    q11, q12, q22, q13, q23, q33 = map(_conic_work_type(T), q)
    W = typeof(q11)

    A = SMatrix{2, 2, W}(q11, q12, q12, q22)
    b = SVector{2, W}(q13, q23)

    center_vec = -(A \ b)
    γ = q33 + dot(b, center_vec)

    eigen_result = eigen(Symmetric(A))
    eigenvals = eigen_result.values
    eigenvecs = eigen_result.vectors

    # Clamp to avoid DomainError from numerically negative arguments
    a1 = sqrt(max(zero(W), -γ / eigenvals[1]))
    a2 = sqrt(max(zero(W), -γ / eigenvals[2]))

    if a1 >= a2
        a, b_axis = a1, a2
        major_axis_vec = Vec2{W}(eigenvecs[1, 1], eigenvecs[2, 1])
    else
        a, b_axis = a2, a1
        major_axis_vec = Vec2{W}(eigenvecs[1, 2], eigenvecs[2, 2])
    end

    θ = atan(major_axis_vec[2], major_axis_vec[1])
    θ = mod(θ, W(π))

    R = float(T)   # an exact-typed conic (Int) still has an irrational ellipse
    return Ellipse{R}(Point2{R}(center_vec), R(a), R(b_axis), R(θ))
end

"""
    Circle(Q::Union{HomCircleMat, HomEllipseMat, Conic}) -> Circle

The circle of a conic matrix. A `HomCircleMat` is one by its class; an ellipse's
or a general conic's matrix is checked by THE circle predicate, the
`HomCircleMat` class check (isotropy within `√eps · ‖A‖`, relative, so a large
circle is read as a small one is), with its `ArgumentError`.
"""
function GeometryBasics.Circle(Q::HomCircleMat)
    ellipse = Ellipse(Q)
    r = (ellipse.a + ellipse.b) / 2
    return GeometryBasics.Circle(ellipse.center, r)
end
GeometryBasics.Circle(Q::HomEllipseMat) = Circle(HomCircleMat(Q))

# =============================================================================
# Gradient Computation
# =============================================================================

"""
    gradient(Q::HomEllipseMat{T}, p::Point2{T}) -> Vec2{T}

Compute the gradient of the conic implicit function at point p.

For conic equation x^T Q x = 0, the gradient is ∇f(x) = 2Qx (homogeneous form).
Returns the 2D Euclidean gradient [∂f/∂x, ∂f/∂y] as a `Vec2` — it's a
direction (steepest-ascent), not a position: the point-on-conic constraint's
observation gradient (`constraint_with_jacobians(::Conic, p)`'s `G_x`).
"""
gradient(Q::HomEllipseMat, p::AbstractVector) = gradient(Conic(Q), p)

# =============================================================================
# Conic geometric primitive — the class matrices for the ellipse forms
# =============================================================================

# Any conic's gradient (the constraint's observation gradient), mixed eltypes promoted
function gradient(c::Conic{T1}, p::AbstractVector) where {T1}
    length(p) == 2 || throw(ArgumentError("Point must be 2D, got length $(length(p))"))
    T = promote_type(T1, eltype(p))
    return Vec2{T}(constraint_with_jacobians(c, SVector{2, T}(p[1], p[2]))[2])
end
# an ellipse is extracted from a conic of that class: `HomEllipseMat` checks it, once
Ellipse(c::Conic) = Ellipse(HomEllipseMat(c))
GeometryBasics.Circle(c::Conic) = Circle(HomCircleMat(c))

# Identity constructors — idempotent conversion
GeometryBasics.Circle(c::Circle) = c

# Construct Conic from geometric primitives
Conic(e::Ellipse) = Conic(HomEllipseMat(e))
Conic(c::Circle) = Conic(HomCircleMat(c))

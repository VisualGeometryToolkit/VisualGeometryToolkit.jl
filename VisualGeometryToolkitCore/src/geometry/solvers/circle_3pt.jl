# =============================================================================
# Circle3pt — the circle through three 2D points (the minimal solver of the
# circle; `solve(solver, points, idx)`)
# =============================================================================

"""
    Circle3pt()

Minimal 3-point circle solver: the circle through three 2D points (the
circumcircle; `solve(Circle3pt(), points, idx)`), `sample_size` 3,
`max_solutions` 1. Its data are POINTS on the curve, an `AbstractVector` of
`StaticVector{2}` — a circle has no second view, so there is no correspondence
to form, as for [`tls_line`](@ref) and [`direct_ellipse`](@ref). The solve is
closed form and frame-covariant (pixels in, a circle in pixels out), computed
about the sample's first point so it needs no normalization.

A (near-)collinear sample is the sample gate's to refuse, before the solve
([`test_sample`](@ref)`(Circle, points, idx)`, the homography's collinearity
ratio). The solver keeps only the last line of defence: exactly collinear
points (a zero determinant) or a non-finite circle give `count == 0`.

Deliberately NOT entered in the [`minimal_solvers`](@ref) registry: that
registry answers correspondence element-type pairs.
"""
struct Circle3pt <: MinimalSolver{Circle{Float64}} end
sample_size(::Circle3pt) = 3
max_solutions(::Circle3pt) = 1

function solve(::Circle3pt, points::AbstractVector, idx::AbstractVector{<:Integer})
    M = Circle{Float64}
    p₁ = SVector{2, Float64}(@inbounds points[idx[1]])
    a = SVector{2, Float64}(@inbounds points[idx[2]]) - p₁
    b = SVector{2, Float64}(@inbounds points[idx[3]]) - p₁
    a² = a[1]^2 + a[2]^2; b² = b[1]^2 + b[2]^2
    D = 2 * (a[1] * b[2] - a[2] * b[1])           # four times the triangle's signed area
    o = SVector((b[2] * a² - a[2] * b²) / D, (a[1] * b² - b[1] * a²) / D)   # the centre about p₁
    c = p₁ + o
    (!iszero(D) && all(isfinite, c)) || return FixedModels{1, M}(0, (M(Point2(0.0, 0.0), 1.0),))
    return FixedModels{1, M}(1, (M(Point2{Float64}(c), norm(o)),))
end

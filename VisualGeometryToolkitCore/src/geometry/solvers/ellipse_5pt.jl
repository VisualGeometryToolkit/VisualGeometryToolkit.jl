# =============================================================================
# Ellipse5pt — the ellipse through five 2D points (the minimal solver of the
# ellipse; `solve(solver, points, idx)`)
# =============================================================================

"""
    Ellipse5pt()

Minimal 5-point ellipse solver: the conic through five 2D points
(`solve(Ellipse5pt(), points, idx)`), `sample_size` 5, `max_solutions` 1,
returned only when it is a real, non-degenerate ELLIPSE
([`is_ellipse`](@ref): a hyperbola, a parabola, a line pair or a point/imaginary
ellipse gives `count == 0`). Its data are POINTS on the curve, an
`AbstractVector` of `StaticVector{2}`, as for [`Circle3pt`](@ref) and
[`direct_ellipse`](@ref). The conic is the null vector of the 4×5 design matrix
of the other four points taken about the sample's first point (through which
the conic has no constant term), by cofactors: closed form, allocation-free and
frame-covariant (pixels in, an ellipse in pixels out), with no normalization.

A sample with three (near-)collinear points — whose conic is a line pair — is
the sample gate's to refuse, before the solve ([`test_sample`](@ref)`(Ellipse,
points, idx)`, the homography's collinearity ratio). The solver keeps only the
conic classification and a non-finite guard.

Deliberately NOT entered in the [`minimal_solvers`](@ref) registry: that
registry answers correspondence element-type pairs.
"""
struct Ellipse5pt <: MinimalSolver{Ellipse{Float64}} end
sample_size(::Ellipse5pt) = 5
max_solutions(::Ellipse5pt) = 1

function solve(::Ellipse5pt, points::AbstractVector, idx::AbstractVector{<:Integer})
    M = Ellipse{Float64}
    none = FixedModels{1, M}(0, (M(Point2(0.0, 0.0), 1.0, 1.0, 0.0),))
    p₁ = SVector{2, Float64}(@inbounds points[idx[1]])
    q = ntuple(k -> SVector{2, Float64}(@inbounds points[idx[k + 1]]) - p₁, Val(4))
    # the rows (x², xy, y², x, y) of the four other points about p₁
    D = SMatrix{4, 5, Float64}(ntuple(j -> (c = (j - 1) ÷ 4 + 1; x = q[(j - 1) % 4 + 1];
        c == 1 ? x[1]^2 : c == 2 ? x[1] * x[2] : c == 3 ? x[2]^2 : c == 4 ? x[1] : x[2]), Val(20)))
    # its null vector by cofactors: vⱼ = (−1)^(j+1)·det(D without column j)
    v = SVector(ntuple(j -> (isodd(j) ? 1 : -1) * det(D[:, SVector{4}(ntuple(k -> k < j ? k : k + 1, Val(4)))]), Val(5)))
    A, B, C, Dx, Ey = v
    Q = HomEllipseMat{Float64}((A, B / 2, Dx / 2, B / 2, C, Ey / 2, Dx / 2, Ey / 2, 0.0))
    (all(isfinite, v) && is_ellipse(Q)) || return none
    e = Ellipse(Q)
    c = p₁ + SVector(e.center[1], e.center[2])
    (all(isfinite, c) && isfinite(e.a) && isfinite(e.b) && e.b > 0) || return none
    return FixedModels{1, M}(1, (M(Point2{Float64}(c), e.a, e.b, e.θ),))
end

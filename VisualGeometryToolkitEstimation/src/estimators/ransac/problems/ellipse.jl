# =============================================================================
# RANSAC Ellipse Problem — a 2D ellipse through one image's points
# =============================================================================
#
# `EllipseProblem` is the `MinimalSolveProblem{Ellipse}` alias: the ellipse
# (x − c)ᵀ Q (x − c) = 1 through 2D points (a contour, edge points), as the
# circle's `CircleProblem` (circle.jl). The ellipse supplies only what a model
# supplies, all of it VisualGeometryToolkitCore's:
#
#   the minimal solver   `Ellipse5pt` — the conic through five points, kept
#                        only when it is a real ellipse, behind VGC's ellipse
#                        sample gate (`test_sample(Ellipse, …)`);
#   the constraint       `constraint_with_jacobians(e::Ellipse, x)`:
#                        g = (x − c)ᵀ Q (x − c) − 1, G_x = 2Q(x − c), G_θ over
#                        (c, Q₁₁, Q₁₂, Q₂₂);
#   the chart            `EllipseManifold` (`minimal_manifold(e)`): the centre
#                        and the log shape, 5 dof.
#
# Its construction, sampler and RANSAC wiring are the family's (cspond.jl), as
# the circle's. The residual is VGC's default `residuals!` — the Sampson
# distance g/‖G_x‖ through VGC's `whiten`: the geometric distance to the ellipse
# to first order, in PIXELS. The refit is the family's chart LM on the rows
# below, and every local optimization of the family runs on them:
# `find_ellipse`'s robustified LM (`Triggs`), and the IRLS and the lifts alike.
#
# DATA: one point of one image, stored as the canonical points
# (`Vector{Point2{Float64}}`) as the circle's. No normalization: the minimal
# solve is computed about the sample's first point and the chart is
# translation-covariant.
# =============================================================================

"""
    EllipseProblem

The RANSAC problem of a 2D ellipse through points: the `MinimalSolveProblem`
whose model is an `Ellipse{Float64}` (VisualGeometryToolkitCore's). Minimal
sample 5 (VisualGeometryToolkitCore's `Ellipse5pt`), codimension 1 (the Sampson
distance of `(x − c)ᵀ Q (x − c) − 1`, in the points' units), a 5-dof chart
(VisualGeometryToolkitCore's `EllipseManifold`: the centre and the log shape).
Build it with `EllipseProblem(points)`; the public estimator is
[`find_ellipse`](@ref).
"""
const EllipseProblem = MinimalSolveProblem{VgtkCore.Ellipse{Float64}, 5, Float64}

"""
    EllipseProblem(points; sampler = nothing)

An `EllipseProblem` from 2D points (`StaticVector{2}`, pixels or any plain unit:
the residual and the threshold are in that unit), canonicalized to
`Point2{Float64}`. Uniform sampling unless a `sampler` is supplied.
"""
function EllipseProblem(points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}};
                        sampler::Union{Nothing, AbstractSampler} = nothing)
    smplr = _problem_sampler(points, 5, sampler)
    pts = Point2{Float64}[Point2{Float64}(p[1], p[2]) for p in points]
    return MinimalSolveProblem{VgtkCore.Ellipse{Float64}, 5}(pts, nothing, pts, nothing, VgtkCore.Ellipse5pt(), nothing, smplr)
end

codimension(::Type{<:MinimalSolveProblem{<:VgtkCore.Ellipse}}) = 1   # one point-on-ellipse equation per point
chart_dof(::Type{<:EllipseProblem}) = 5

# The sample gate: VGC's ellipse gate, the homography's collinearity ratio on
# every triple of the five points (scale-free, so in the points' own frame).
test_sample(p::EllipseProblem, idx::AbstractVector{Int}) = test_sample(model_type(p), p.cs, idx)

# The scoring constraint — VGC's point-on-ellipse constraint. `residuals!` is
# VGC's default through it (the Sampson distance, VGC's one degeneracy rule).
@inline constraint_with_jacobians(p::EllipseProblem, e::VgtkCore.Ellipse, i::Int) =
    @inbounds VgtkCore.constraint_with_jacobians(e, p.cs[i])

# =============================================================================
# The chart rows — VGC's EllipseManifold, the default `minimal_manifold(e)`
# =============================================================================

# √w-weighted signed Sampson distances at chart coordinates `a`: the same
# constraint and whitening as the scoring (VGC `whitened_rows`). A point at the
# centre (G_x = 0, ±Inf under the rule) contributes a zero row.
function residual_rows!(f, lm::ChartLm{<:VgtkCore.EllipseManifold, <:EllipseProblem}, a)
    pts = lm.problem.cs; sqrtw = lm.sqrtw
    e = retract(lm.ch, SVector{5}(a[1], a[2], a[3], a[4], a[5]))
    @inbounds for i in eachindex(pts)
        g, G_x, G_θ = VgtkCore.constraint_with_jacobians(e, pts[i])
        d = first(whitened_rows(SampsonWhitening(), g, G_x, G_θ))
        f[i] = isinf(d) ? 0.0 : sqrtw[i] * d
    end
    return f
end

# ∂d/∂a: VGC's signed-Sampson derivative (`signed_sampson_jacobian`) along the
# chart directions A = ∂(c, Q)/∂a (`retract_jacobian`). G_x = 2Q(x − c) is
# bilinear in the ambient coordinates and x − c: ∂G_x/∂c = −2Q,
# ∂G_x/∂(Q₁₁, Q₁₂, Q₂₂) = 2·[d₁ d₂ 0; 0 d₁ d₂], d = x − c.
function residual_jacobian_rows!(J, lm::ChartLm{<:VgtkCore.EllipseManifold, <:EllipseProblem}, a)
    pts = lm.problem.cs; sqrtw = lm.sqrtw
    av = SVector{5}(a[1], a[2], a[3], a[4], a[5])
    e = retract(lm.ch, av)
    A = retract_jacobian(lm.ch, av)
    θ = VgtkCore.ambient_coordinates(e)
    c₁, c₂, q₁₁, q₁₂, q₂₂ = θ
    @inbounds for i in eachindex(pts)
        g, G_x, G_θ = VgtkCore.constraint_with_jacobians(e, pts[i])
        d₁ = pts[i][1] - c₁; d₂ = pts[i][2] - c₂
        ∂G_x = SMatrix{2, 5, Float64, 10}(-2q₁₁, -2q₁₂, -2q₁₂, -2q₂₂, 2d₁, 0.0, 2d₂, 2d₁, 0.0, 2d₂) * A
        row = signed_sampson_jacobian(g, G_x, A' * G_θ, ∂G_x)
        for q in 1:5
            J[i, q] = sqrtw[i] * row[q]
        end
    end
    return J
end

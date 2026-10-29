# =============================================================================
# RANSAC Circle Problem — a 2D circle through one image's points
# =============================================================================
#
# `CircleProblem` is the `MinimalSolveProblem{Circle}` alias: the circle
# ‖x − c‖ = r through 2D points (a contour, edge points). The circle supplies
# only what a model supplies, all of it VisualGeometryToolkitCore's:
#
#   the minimal solver   `Circle3pt` — the circumcircle of three points, behind
#                        VGC's circle sample gate (`test_sample(Circle, …)`);
#   the constraint       `constraint_with_jacobians(c::Circle, x)`:
#                        g = ‖x − c‖² − r², G_x = 2(x − c), G_θ over (c, r);
#   the chart            `CircleManifold` (`minimal_manifold(c)`): the centre
#                        and log r, 3 dof.
#
# Its construction, sampler and RANSAC wiring are the family's (cspond.jl). The
# residual is VGC's default `residuals!` — the Sampson distance g/‖G_x‖ of the
# constraint through VGC's `whiten`, (d² − r²)/(2d) for a point at distance d
# from the centre: the geometric distance d − r to first order, in PIXELS. The
# refit is the family's chart LM (`refine!` → `_refine!(::NonlinLsq, …)`) on the
# rows below — the same constraint, whitened the same way — and every local
# optimization of the family runs on them: `find_circle`'s, the robustified LM
# (`RobustLm`), and the IRLS and the lifts alike.
#
# DATA: the circle's datum is one point of one image; there is no second view
# and so no correspondence. The problem stores the canonical points
# (`Vector{Point2{Float64}}`) as its data `cs`, as `tls_line(points)` and
# `direct_ellipse(points)` take them, and the minimal solver reads the same
# vector (`solve(Circle3pt(), points, idx)`). The only shared change this
# needed is the problem scalar read off a point vector (`_problem_scalar`,
# cspond.jl). The points are plain (unit isotropic noise); no normalization —
# the minimal solve is computed about the sample's first point and the chart is
# translation-covariant.
# =============================================================================

"""
    CircleProblem

The RANSAC problem of a 2D circle through points: the `MinimalSolveProblem`
whose model is a `Circle{Float64}` (GeometryBasics', re-exported by
VisualGeometryToolkitCore). Minimal sample 3 (VisualGeometryToolkitCore's
`Circle3pt`), codimension 1 (the Sampson distance of `‖x − c‖² − r²`, in the
points' units), a 3-dof chart (VisualGeometryToolkitCore's `CircleManifold`: the
centre and log r). Build it with `CircleProblem(points)`; the public estimator
is [`find_circle`](@ref).
"""
const CircleProblem = MinimalSolveProblem{VgtkCore.Circle{Float64}, 3, Float64}

"""
    CircleProblem(points; sampler = nothing)

A `CircleProblem` from 2D points (`StaticVector{2}`, pixels or any plain unit:
the residual and the threshold are in that unit), canonicalized to
`Point2{Float64}`. Uniform sampling unless a `sampler` is supplied.
"""
function CircleProblem(points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}};
                       sampler::Union{Nothing, AbstractSampler} = nothing)
    smplr = _problem_sampler(points, 3, sampler)
    pts = Point2{Float64}[Point2{Float64}(p[1], p[2]) for p in points]
    return MinimalSolveProblem{VgtkCore.Circle{Float64}, 3}(pts, nothing, pts, nothing, VgtkCore.Circle3pt(), nothing, smplr)
end

codimension(::Type{<:MinimalSolveProblem{<:VgtkCore.Circle}}) = 1   # one point-on-circle equation per point
chart_dof(::Type{<:CircleProblem}) = 3

# The sample gate: VGC's circle gate, the homography's collinearity ratio on the
# three points (scale-free, so in the points' own frame).
test_sample(p::CircleProblem, idx::AbstractVector{Int}) = test_sample(model_type(p), p.cs, idx)

# The scoring constraint — VGC's point-on-circle constraint. `residuals!` is
# VGC's default through it (the Sampson distance, VGC's one degeneracy rule).
@inline constraint_with_jacobians(p::CircleProblem, c::VgtkCore.Circle, i::Int) =
    @inbounds VgtkCore.constraint_with_jacobians(c, p.cs[i])

# =============================================================================
# The chart rows — VGC's CircleManifold, the default `minimal_manifold(c)`
# =============================================================================

# √w-weighted signed Sampson distances at chart coordinates `a`: the same
# constraint and whitening as the scoring (VGC `whitened_rows`). A point at the
# centre (G_x = 0, ±Inf under the rule) contributes a zero row.
function residual_rows!(f, lm::ChartLm{<:VgtkCore.CircleManifold, <:CircleProblem}, a)
    pts = lm.problem.cs; sqrtw = lm.sqrtw
    c = retract(lm.ch, SVector{3}(a[1], a[2], a[3]))
    @inbounds for i in eachindex(pts)
        g, G_x, G_θ = VgtkCore.constraint_with_jacobians(c, pts[i])
        d = first(whitened_rows(SampsonWhitening(), g, G_x, G_θ))
        f[i] = isinf(d) ? 0.0 : sqrtw[i] * d
    end
    return f
end

# ∂d/∂a: VGC's signed-Sampson derivative (`signed_sampson_jacobian`) along the
# chart directions A = ∂(c, r)/∂a (`retract_jacobian`). The constraint's second
# derivative is constant: G_x = 2(x − c), so ∂G_x/∂(c, r) = [−2I 0].
const _CIRCLE_∂G_x = SMatrix{2, 3, Float64, 6}(-2.0, 0.0, 0.0, -2.0, 0.0, 0.0)
function residual_jacobian_rows!(J, lm::ChartLm{<:VgtkCore.CircleManifold, <:CircleProblem}, a)
    pts = lm.problem.cs; sqrtw = lm.sqrtw
    av = SVector{3}(a[1], a[2], a[3])
    c = retract(lm.ch, av)
    A = retract_jacobian(lm.ch, av)
    ∂G_x = _CIRCLE_∂G_x * A
    @inbounds for i in eachindex(pts)
        g, G_x, G_θ = VgtkCore.constraint_with_jacobians(c, pts[i])
        row = signed_sampson_jacobian(g, G_x, A' * G_θ, ∂G_x)
        for q in 1:3
            J[i, q] = sqrtw[i] * row[q]
        end
    end
    return J
end

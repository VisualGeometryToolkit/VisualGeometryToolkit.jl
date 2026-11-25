# =============================================================================
# VisualGeometryToolkitEstimation backend of detect_image_circle — the fit
# =============================================================================
#
# The RANSAC of the image circle's model on the rim's candidates (src/image_circle.jl, step 4), and its covariance
# (`_coverage_covariance`): Estimation's `Ransac` on a `CircleProblem` or an `EllipseProblem`
# (`solve(EstimationProblem(CircleProblem(contour)), Ransac(; scoring); rng)`), scored by VisualGeometryToolkitCore's
# Student-t + uniform marginal (the inlier scale marginalized; the outliers uniform over ±`outlier_halfwidth`, in the
# contour's pixels), Estimation's `default_student_t_uniform_scoring`: its priors are the stack's default (Estimation's
# configs/default.toml [scoring.student_t_uniform]).
# =============================================================================

module VisualGeometryToolkitFeaturesEstimationExt

import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: Circle, Ellipse, EstimationProblem, solve
using VisualGeometryToolkitEstimation: CircleProblem, EllipseProblem, Ransac
using VisualGeometryToolkitFeatures: VisualGeometryToolkitFeatures

# The family problem of a model type.
_problem(::Type{<:Circle}) = CircleProblem
_problem(::Type{<:Ellipse}) = EllipseProblem

VisualGeometryToolkitFeatures._fit_image_circle(::Type{T}, contour; outlier_halfwidth, rng) where {T <: Union{Circle, Ellipse}} =
    solve(EstimationProblem(_problem(T)(contour)),
          Ransac(; scoring = VgtkEstimation.default_student_t_uniform_scoring(_problem(T), outlier_halfwidth)); rng)

# The coverage's UNIT-noise covariance in Core's ambient coordinates: solver_covariance of the problem's constraint rows
# on the inliers `pts` at `m` (Core's implicit covariance). `nothing` for a singular system, too few points, or a
# problem that provides none (the circle's).
function VisualGeometryToolkitFeatures._coverage_covariance(m::Union{Circle, Ellipse}, pts)
    P = _problem(typeof(m))(pts)
    VgtkCore.provides_solver_covariance(typeof(P)) || return nothing   # the circle problem has no constraint rows (yet)
    return VgtkCore.solver_covariance(P, m, collect(eachindex(pts)))
end

end

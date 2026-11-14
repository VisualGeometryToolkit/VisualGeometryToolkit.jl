# =============================================================================
# VisualGeometryToolkitEstimation backend of detect_image_circle — the fit
# =============================================================================
#
# The RANSAC of the image circle's model on its contour (src/image_circle.jl, step 6): Estimation's `find_circle` /
# `find_ellipse`, scored by VisualGeometryToolkitCore's Student-t + uniform marginal (the inlier scale marginalized; the
# outliers uniform over ±`outlier_halfwidth`, in the contour's pixels), built through Estimation's scorer table so its
# priors are the stack's default (Estimation's configs/default.toml [scoring.student_t_uniform]).
# =============================================================================

module VisualGeometryToolkitFeaturesEstimationExt

import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: Circle, Ellipse
using VisualGeometryToolkitFeatures: VisualGeometryToolkitFeatures

VisualGeometryToolkitFeatures._fit_image_circle(::Type{<:Circle}, contour; outlier_halfwidth, rng) =
    VgtkEstimation.find_circle(contour; scoring = _image_circle_scoring(VgtkEstimation.CircleProblem, outlier_halfwidth), rng)
VisualGeometryToolkitFeatures._fit_image_circle(::Type{<:Ellipse}, contour; outlier_halfwidth, rng) =
    VgtkEstimation.find_ellipse(contour; scoring = _image_circle_scoring(VgtkEstimation.EllipseProblem, outlier_halfwidth), rng)

# The coverage's UNIT-noise covariance in Core's ambient coordinates: solver_covariance of the problem's constraint rows
# on the inliers `pts` at `m` (Core's implicit covariance). `nothing` for a singular system, too few points, or a
# problem that provides none (the circle's).
function VisualGeometryToolkitFeatures._coverage_covariance(m::Union{Circle, Ellipse}, pts)
    P = m isa Circle ? VgtkEstimation.CircleProblem(pts) : VgtkEstimation.EllipseProblem(pts)
    VgtkCore.provides_solver_covariance(typeof(P)) || return nothing   # the circle problem has no constraint rows (yet)
    return VgtkCore.solver_covariance(P, m, collect(eachindex(pts)))
end

# The Student-t + uniform scorer at the half-width, through the stack's scorer table.
_image_circle_scoring(P, outlier_halfwidth) = VgtkEstimation.scoring_from_toml(
    Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => Float64(outlier_halfwidth)); problem = P)

end

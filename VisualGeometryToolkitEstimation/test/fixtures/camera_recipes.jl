# The two-solve recipes of the camera regressions and the camera-and-poses refine, shared by the suite
# (`@testsetup module CameraRecipes`, camera_regression_testsetups.jl) and the golden replay (ci/golden/replay/est_camera_fits.jl):
#
# - `fit_then_refine(L, source, region; settings…, stops)`: a camera of lens family `L` from no guess — the linear fit
#   (`CameraRegression`) and its exact pixel least squares from it (`LevenbergMarquardt` on the problem without a
#   target); a family with no linear form, and a linear fit that failed, start from the family's anchor (its zero lens
#   at the source's K). A bounded source is refined as its model over the part of the region it images, the region the
#   fit ran over.
# - `poses_stops(; max_iter, ftol, xtol, max_time)`: the camera-and-poses refine's stops of record
#   ([camera_poses_problem]) with the given ones replaced.
using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, BoundedCamera, CameraModel,
    Division, model, successful_retcode, default_stops
using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, CameraPosesProblem
import VisualGeometryToolkitEstimation as VgtkEstimation

function fit_then_refine(::Type{L}, source, region; stops = nothing, settings...) where {L}
    L === Division && return fit_then_refine(Division{1}, source, region; stops, settings...)
    camera, part = source isa BoundedCamera ? (model(source), VgtkEstimation._imaged_part(source, region)) : (source, region)
    start = if VgtkEstimation._linear_form(L) isa VgtkEstimation._NoLinearForm
        source isa BoundedCamera &&
            (part = VgtkEstimation._bounded_part(source, region, get(settings, :support, VgtkEstimation.VgtkCore.valid_domain(source)), "CameraRegression"))
        VgtkEstimation._anchor_camera(L, camera)
    else
        lin = solve(EstimationProblem(CameraRegressionProblem(L, source, region; settings...)), CameraRegression())
        part = lin.metadata.region
        successful_retcode(lin) ? lin.u : VgtkEstimation._anchor_camera(L, camera)
    end
    refine = source isa BoundedCamera ? (; (k => v for (k, v) in pairs(settings) if k !== :support)..., support = part) : settings
    f = CameraRegressionProblem(camera, part; refine...)
    return stops === nothing ? solve(RefineProblem(f, start), LevenbergMarquardt()) : solve(RefineProblem(f, start), LevenbergMarquardt(); stops)
end

poses_stops(; kw...) = (d = default_stops(CameraPosesProblem(nothing)); LmStops(; merge((; max_iter = d.max_iter, ftol = d.ftol, xtol = d.xtol), values(kw))...))

# The camera-and-poses refine's stops of record, shared by the suite
# (`@testsetup module CameraRecipes`, camera_regression_testsetups.jl) and the golden replay (ci/golden/replay/est_camera_fits.jl):
#
# - `poses_stops(; max_iter, ftol, xtol, max_time)`: the camera-and-poses refine's stops of record
#   ([camera_poses_problem]) with the given ones replaced.
using VisualGeometryToolkitCore: LmStops, default_stops
using VisualGeometryToolkitEstimation: CameraPosesProblem

poses_stops(; kw...) = (d = default_stops(CameraPosesProblem(nothing)); LmStops(; merge((; max_iter = d.max_iter, ftol = d.ftol, xtol = d.xtol), values(kw))...))

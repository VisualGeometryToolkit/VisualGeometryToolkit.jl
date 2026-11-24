# The one spelling of a PoseLib refine that the suite and the golden replay (`ci/golden/replay/poselib.jl`) share: the
# strategy is `LevenbergMarquardt(PoseLibSolver())`'s field, `max_iter` the stops' (Core's `LmStops`), the context `p`
# the problem's; the scale input and anything else go to the solve as given.
import VisualGeometryToolkitCore as VgtkCore
poselib_refine(data, θ, p = VgtkCore.NullParameters(); max_iter, optimization_strategy = VgtkCore.ClassDefault(), kw...) =
    VgtkCore.solve(VgtkCore.RefineProblem(data, θ, p), VgtkCore.LevenbergMarquardt(VgtkCore.PoseLibSolver(); optimization_strategy);
                   stops = VgtkCore.LmStops(; max_iter), kw...)

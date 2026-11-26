# =============================================================================
# PoseLibEstimationExt — PoseLib's backend refines of VisualGeometryToolkitEstimation's
# two-view family problems, so a RANSAC's local optimization can be Core's
# `BackendRefit(LevenbergMarquardt(PoseLibSolver(); optimization_strategy), stops)`:
#
# - H and F: `lm_backend_refine` on a `HomographyProblem` / `FundMatProblem` forwards to
#   PoseLib's raw-data form on the problem's own correspondences (`problem.cs`, the data
#   frame the problem's models are in). A homography with observation maps (a model in
#   another frame than its data) is refused.
# - E: the refine of an essential hypothesis needs the two-view frame's calibrations
#   (`external_workspace`: `frame.Kc1`, `frame.Kc2`, `frame.cs`) and the hypothesis's
#   pose: the seed is decomposed by VGC's fast cheirality vote on a strided subsample of
#   the problem's unit bearings (about `lo.cheirality_vote_points` points), then refined
#   camera-aware on the frame's pixel correspondences. The ONE copy of that body, which
#   LocalOptimizationDoneRight's truncated arms run.
# =============================================================================
module PoseLibEstimationExt

using StaticArrays: SMatrix
using VisualGeometryToolkitCore: BackendRefit, EssentialMat, FundMat, HomographyMat, LevenbergMarquardt,
    LossWidth, NullParameters, PoseLibSolver, RefineProblem, cheiral_pose_fast, loss_width, solve
import VisualGeometryToolkitCore: external_workspace, external_refit!, lm_backend_refine
using VisualGeometryToolkitEstimation: EssentialProblem, FundMatProblem, HomographyProblem

# The family problems' refines: PoseLib's raw-data form on the problem's correspondences.
function lm_backend_refine(b::PoseLibSolver, G::HomographyMat, f::HomographyProblem, ::NullParameters,
                           alg::LevenbergMarquardt; kw...)
    f.mapped && throw(ArgumentError("PoseLib refines a homography on its data frame: a HomographyProblem with " *
                                    "observation maps (a model in another frame) has no PoseLib refine"))
    return lm_backend_refine(b, G, f.cs, NullParameters(), alg; kw...)
end
lm_backend_refine(b::PoseLibSolver, G::FundMat, f::FundMatProblem, ::NullParameters, alg::LevenbergMarquardt; kw...) =
    lm_backend_refine(b, G, f.cs, NullParameters(), alg; kw...)

# The essential route of a PoseLib BackendRefit.
const _PoseLibRefit = BackendRefit{<:LevenbergMarquardt{PoseLibSolver}}

# The frame's calibrations (as PoseLib's refines read them, 3×3 Float64) and its pixel correspondences, held as the
# frame carries them.
external_workspace(::_PoseLibRefit, ::EssentialProblem, frame, _scoring) =
    (; Kc1 = SMatrix{3, 3, Float64, 9}(frame.Kc1), Kc2 = SMatrix{3, 3, Float64, 9}(frame.Kc2), cs = frame.cs)

function external_refit!(ws, lo::_PoseLibRefit, problem::EssentialProblem, seed_E, w::LossWidth, _scoring)
    pose = cheiral_pose_fast(seed_E, problem.cs; vote_points = lo.cheirality_vote_points)
    u = solve(RefineProblem(ws.cs, pose, (; K1 = ws.Kc1, K2 = ws.Kc2)), lo.alg; stops = lo.stops, width = loss_width(w)).u
    return EssentialMat(u.R, u.t)
end

end # module PoseLibEstimationExt

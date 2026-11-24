# =============================================================================
# The absolute pose: the solve's units and frames at the boundary
# =============================================================================
#
#     solve(EstimationProblem(AbsolutePoseProblem(camera, X .=> u), (; extent)), Ransac(…); rng)
#     solve(RefineProblem(AbsolutePoseProblem(camera, X .=> u), pose), LevenbergMarquardt(…); scale | width)
#     solve(RefineProblem(BearingPoseProblem(X .=> f), pose), LevenbergMarquardt(…); scale | width)
#
# The problems are the inner loop's (plain metres or the caller's unit, unlabelled `Extrinsics3`). What the caller
# handed in is restored once, at the solve's exit: the guess's frame labels (`solution_model`) and, for a physical
# problem (lengths: the constructor stripped them in metres and recorded it), the metre — a find's pose through
# `_with_units` (units.jl), a refine's through `solution_model`. A refine's guess is physical exactly when its
# problem is (a mix is a `MethodError`).
# =============================================================================

const _PoseProblem = Union{AbsolutePoseProblem, BearingPoseProblem}
_problem_unit(p::_PoseProblem) = p.context.unit

# The guess as the inner loop's model: unlabelled, plain (stripped in metres when the problem is physical) and typed
# `Extrinsics3` (`convert` labels it). The bearing form refuses a guess at which a ray origin coincides with its point —
# a guard of the LM's rows, so not on fewer data than a minimal sample, where no LM runs and the refine hands the guess
# back (`InitialFailure`).
guess_model(p::_PoseProblem, pose::EuclideanMap{3}) = _pose_guess(p, _plain_guess(_problem_unit(p), pose))
_pose_guess(p::AbsolutePoseProblem, pose) = convert(model_type(p), unlabel(pose))
function _pose_guess(p::BearingPoseProblem, pose)
    θ₀ = unlabel(pose)
    _holds_minimal_sample(p, data_size(p)) || return convert(model_type(p), θ₀)
    R0 = rotation(θ₀); t0 = translation(θ₀); o = p.context.origins
    for i in 1:data_size(p)
        iszero(norm(R0 * sources(p.cs)[i] + t0 - o[i])) &&
            throw(ArgumentError("BearingPoseProblem: point $i coincides with its ray's origin at the guess"))
    end
    return convert(model_type(p), θ₀)
end

# The refined pose under the guess's frame labels (its translation an `SVector{3, Float64}`), in metres when the
# problem is physical.
solution_model(p::_PoseProblem, E::EuclideanMap{3}, pose::EuclideanMap{3}) =
    _pose_out(_problem_unit(p), relabel(VgtkCore.map_frames(pose)...,
        EuclideanMap(RotMatrix{3, Float64}(rotation(E)), SVector{3, Float64}(translation(E)))))
_pose_out(::Nothing, E) = E
_pose_out(::_Metres, E) = _attach_metres(E)

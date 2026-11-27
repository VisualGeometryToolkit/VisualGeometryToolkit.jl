# The relative-pose bundle's cases of group est_epipolar (record/est_epipolar.jl, `relpose_bundle/*`), included by
# replay/est_epipolar.jl: the LevenbergMarquardt refine of a `RelativePoseProblem` replaces the deleted pose-or-nothing
# function. Its `u` is the old return value (the refined pose, or `nothing` when the bundle cannot run); every other field
# of the result is new (`""` in `new_only` names the root path only: the new root, whose `u` takes the old root's path).
using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, LmStops, solve
using VisualGeometryToolkitEstimation: RelativePoseProblem


# The old `max_iter` keyword: the bundle's stops of record with that cap.
_relpose_cap(f, k) = LmStops(; max_iter = k, ftol = VgtkCore.default_stops(f).ftol)
relpose(pose, K1, K2, cs; max_iter = nothing) = (f = RelativePoseProblem(K1, K2, cs);
    max_iter === nothing ? solve(RefineProblem(f, pose), LevenbergMarquardt()) :
                           solve(RefineProblem(f, pose), LevenbergMarquardt(); stops = _relpose_cap(f, max_iter)))

function relpose_bundle_replays!(cases)
    for (name, fx) in ep_relpose_fixtures()
        # (a physical seed: a MethodError on both sides, its message naming the new function)
        replay!(cases, "relpose_bundle/$name/default", fx, ((pose, K1, K2, cs),) -> relpose(pose, K1, K2, cs);
                map = name == "physical_seed_10" ? REFUSAL_MAP : BARE_MAP)
        name == "exact_30" && replay!(cases, "relpose_bundle/$name/max_iter200", fx,
            ((pose, K1, K2, cs),) -> relpose(pose, K1, K2, cs; max_iter = 200); map = BARE_MAP)
        name == "harness_100" && replay!(cases, "relpose_bundle/$name/max_iter1", fx,
            ((pose, K1, K2, cs),) -> relpose(pose, K1, K2, cs; max_iter = 1); map = BARE_MAP)
    end
    return cases
end

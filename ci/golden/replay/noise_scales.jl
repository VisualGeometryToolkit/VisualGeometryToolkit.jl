# The noise-scale estimators on Core 0.14: `mad_scale(rs)` → `solve(EstimationProblem(NoiseScale, rs), MedianNormScale())`
# (a bare scale → a FitResult, `u` the scale); `pooled_noise_scale(results)` / `pooled_noise_scale(s, ν)` →
# `solve(EstimationProblem(NoiseScale, results | s => ν), PooledScale())`, the find results of the fixtures rebuilt with
# the new find (`Ransac()`, the class defaults: the old `find_homography`'s).
using VisualGeometryToolkitCore: EstimationProblem, NoiseScale, MedianNormScale, solve, dof
using VisualGeometryToolkitEstimation: HomographyProblem, Ransac, PooledScale

# A closed form that returned a bare value: `u` is it, every other field new.
const _NS_BARE = FieldMap(; renames = ["u" => ""],
    new_only = ["", "resid", "retcode", "stats", "prob", "alg", "original", "stop_rule", "rms", "n", "metadata"])
const _NS_BARE_REFUSAL = FieldMap(; ignored = ["exception.msg"])   # the message named `mad_scale`
# A pooled scale (a FitResult before and after): SciML's names, the new fields; the result type moved.
const _NS_POOLED = FieldMap(; renames = ["u" => "estimate", "stats.nsteps" => "iterations"],
    new_only = ["prob", "alg", "original", "stats", "resid"], retyped = [""])
const _NS_POOLED_REFUSAL = FieldMap(; ignored = ["exception.msg"])   # the message named `pooled_noise_scale`

# The old leaf sequence of a RANSAC result (its digest inside an array fixture): the old field order of
# FitResult{P, RansacMetadata} — estimate, retcode, stop_rule, iterations, rms, n, metadata(residuals, weights, quality,
# scale, scale_source, dof, sample_acceptance_rate, lo_retcode, lo_stop_rule, nonfinite_scale) — read from the new one.
function _ns_old_leaves(r)
    md = r.metadata
    return (r.u, r.retcode, r.stop_rule, r.stats.nsteps, r.rms, r.n,
            (r.resid, md.weights, md.quality, md.scale, md.scale_source, md.dof, md.sample_acceptance_rate, md.lo_retcode,
             md.lo_stop_rule, md.nonfinite_scale))
end
_ns_results_view(rs) = map(_ns_old_leaves, rs)
const _NS_RESULTS_FIXTURE = FieldMap(; retyped = [""])   # the array's type changes; its leaves' digest is compared

_ns_find(cs, seed) = solve(EstimationProblem(HomographyProblem(cs)), Ransac(); rng = StableRNG(seed))
_ns_pooled(x) = solve(EstimationProblem(NoiseScale, x), PooledScale())

function replay_noise_scales(cases)
    mns(rs) = solve(EstimationProblem(NoiseScale, rs), MedianNormScale())
    for seed in SEEDS, (name, rs) in ns_residuals(seed)
        replay!(cases, "median_norm_scale/$name/seed$seed", rs, mns; map = _NS_BARE)
    end
    for (name, rs) in ns_exact_residuals()
        replay!(cases, "median_norm_scale/$name", rs, mns; map = _NS_BARE)
    end
    replay!(cases, "median_norm_scale/refused/plain_vector", [1.0, 2.0, 3.0], mns; map = _NS_BARE_REFUSAL)
    for seed in SEEDS
        rs = [_ns_find(cs, seed) for cs in ns_views(seed)]
        failed = _ns_find(homography_too_few(), seed)
        for (id, x) in ("results4" => rs, "results1" => rs[1:1], "results_with_failed" => [rs; failed], "only_failed" => [failed])
            replay!(cases, "pooled_scale/$id/seed$seed", x, _ns_pooled; map = _NS_POOLED,
                    fixture_map = _NS_RESULTS_FIXTURE, fixture_view = _ns_results_view)
        end
        s = [VgtkCore.noise_scale(r) for r in rs]
        ν = [dof(r) for r in rs]
        replay!(cases, "pooled_scale/numbers/seed$seed", (s, ν), ((s, ν),) -> _ns_pooled(s => ν); map = _NS_POOLED)
        replay!(cases, "pooled_scale/numbers_left_out/seed$seed", ([s; NaN; 0.7], [ν; 12; 0]), ((s, ν),) -> _ns_pooled(s => ν); map = _NS_POOLED)
        replay!(cases, "pooled_scale/refused/length_mismatch/seed$seed", (s, ν[1:2]), ((s, ν),) -> _ns_pooled(s => ν); map = _NS_POOLED_REFUSAL)
    end
    replay!(cases, "pooled_scale/numbers_none", ([NaN], [5]), ((s, ν),) -> _ns_pooled(s => ν); map = _NS_POOLED)
    return cases
end

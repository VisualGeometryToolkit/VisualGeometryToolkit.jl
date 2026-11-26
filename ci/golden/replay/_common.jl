# The replay groups' shared pieces (not a group: replay.jl runs the groups that have records): the field mappings of the
# Core 0.14 results onto the recorded ones, and the translation of an old find_* / refine_* keyword set into the new
# solve (the algorithm's fields, the problem's context `p`, the solve's run settings).
using VisualGeometryToolkitCore: EstimationProblem, RefineProblem, MinimalProblem, LevenbergMarquardt, RansacScoring,
    MsacScoring, solve, init, solve!
using VisualGeometryToolkitEstimation: Ransac, LocalOptimization, RansacConfig, RansacWorkspace

# A FitResult replacing a FitResult: SciML's names for the estimate and the counter, `resid` for the per-datum residuals
# where the old metadata held them, and the new fields.
const _NEW_FIELDS = ["prob", "alg", "original", "stats"]
const FIND_MAP = FieldMap(; renames = ["u" => "estimate", "stats.nsteps" => "iterations", "resid" => "metadata.residuals"],
                          new_only = _NEW_FIELDS, retyped = [""])
# A refine: its metadata type moved to Core (VisualGeometryToolkitCore.RefineMetadata).
const REFINE_MAP = FieldMap(; renames = ["u" => "estimate", "stats.nsteps" => "iterations", "resid" => "metadata.residuals"],
                            new_only = _NEW_FIELDS, retyped = ["", "metadata"])
# A result whose residuals were never in its metadata (a fit, a pooled scale): `resid` is new.
const RESULT_MAP = FieldMap(; renames = ["u" => "estimate", "stats.nsteps" => "iterations"],
                            new_only = [_NEW_FIELDS; "resid"], retyped = [""])
# A refusal whose message named the old entry or keywords: the exception's type is compared, its message not.
const REFUSAL_MAP = FieldMap(; ignored = ["exception.msg"])

"""
    prefixed(m::FieldMap, pre) -> FieldMap

`m` applied below the path `pre` (a fixture's tuple slot holding an upstream result: `prefixed(FIND_MAP, "3")`).
"""
function prefixed(m::FieldMap, pre::AbstractString)
    j(p) = isempty(p) ? pre : string(pre, ".", p)
    return FieldMap([j(a) => j(b) for (a, b) in m.renames], j.(m.new_only), [""; j.(m.retyped)], m.ignored, m.aliases,
                    [j(a) => b for (a, b) in m.retagged])   # the tuple holding it is retyped too
end

# The old find_* keywords as the new solve: the algorithm's configuration (`Ransac`'s fields; `max_trials`/`min_trials`
# into its `RansacConfig`, the old pipeline's one rule; a workspace's storage precision), the context (`extent`), the
# run setting (`rng`). Any other keyword goes to `Ransac` as given (a refusal: it refuses or has no such keyword).
function ransac_solve(f; rng = nothing, extent = missing, max_trials = nothing, min_trials = nothing, config = nothing,
                      workspace = nothing, kw...)
    if config === nothing && (max_trials !== nothing || min_trials !== nothing)
        config = RansacConfig(; (max_trials === nothing ? (;) : (; max_trials))..., (min_trials === nothing ? (;) : (; min_trials))...)
    end
    alg = Ransac(workspace === nothing ? Float32 : eltype(workspace.best_residuals); (config === nothing ? (;) : (; config))..., kw...)
    prob = extent === missing ? EstimationProblem(f) : EstimationProblem(f, (; extent))   # a given extent, even `nothing`, is the context
    return rng === nothing ? solve(prob, alg) : solve(prob, alg; rng)
end

# The old refine_* keywords as the new solve: the strategy is the LevenbergMarquardt's field, the scale input and the
# stops are solve keywords; a workspace is no keyword any more (the chart LM's task-local workspace is the same
# buffers); any other keyword goes to the solve as given (a refusal: no such keyword).
function lm_solve(f, θ; optimization_strategy = VgtkCore.ClassDefault(), workspace = nothing, kw...)
    alg = LevenbergMarquardt(; optimization_strategy)
    return solve(RefineProblem(f, θ), alg; kw...)
end

# --- the case matrices of fixtures/shared_cases.jl, replayed -----------------------------------------------------
# `find(fx; kw...)` and `refine(fx, θ; kw...)` are the NEW solves on the old keyword sets (`ransac_solve`, `lm_solve`).

# The refusals every find and refine shares: each keeps its recorded exception type (or is an intended change).
function refusal_replays!(cases, family, fx, find, refine, θ, P)
    if find !== nothing
        replay!(cases, "$family/refused/find_threshold_kw", fx, fx -> find(fx; threshold = 3.0); map = REFUSAL_MAP)
        replay!(cases, "$family/refused/find_outlier_halfwidth_kw", fx, fx -> find(fx; outlier_halfwidth = 20.0); map = REFUSAL_MAP)
        replay!(cases, "$family/refused/find_retired_local_optimization", fx, fx -> find(fx; local_optimization = Irls{TLSShape}()); map = REFUSAL_MAP)
        replay!(cases, "$family/refused/find_unknown_kw", fx, fx -> find(fx; no_such_keyword = 1); map = REFUSAL_MAP)
    end
    for (lbl, kw) in (
            "no_scale" => (;), "scale_and_width" => (; scale = 1.0, width = 1.0), "negative_scale" => (; scale = -1.0),
            "nan_scale" => (; scale = NaN), "zero_width" => (; width = 0.0), "ls_with_scale" => (; optimization_strategy = nothing, scale = 1.0),
            "tls_scale_no_constant" => (; optimization_strategy = Triggs{TLSShape}(), scale = 1.0),
            "constant_with_width" => (; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5), width = 1.0),
            "irls_huber" => (; optimization_strategy = Irls{HuberShape}(), scale = 1.0),
            "irls_gaussian" => (; optimization_strategy = Irls{GaussianShape}(), scale = 1.0),
            "terminal_lo" => (; optimization_strategy = TerminalLO(Irls{TLSShape}()), scale = 1.0),
            "scoring_kw" => (; scale = 1.0, scoring = MsacScoring(P; threshold = 3.0)),
            "rng_kw" => (; scale = 1.0, rng = StableRNG(1)), "max_trials_kw" => (; scale = 1.0, max_trials = 10),
            "extent_kw" => (; scale = 1.0, extent = nothing), "unknown_kw" => (; scale = 1.0, nonsense = 1))
        replay!(cases, "$family/refused/refine_$lbl", (fx, θ), ((fx, θ),) -> refine(fx, θ; kw...); map = REFUSAL_MAP)
    end
    return cases
end

# The refines from each seed's find (its model and its search's scale) and from the raw incumbent. A fixture that holds
# the find's result maps it as a find result (`prefixed(FIND_MAP, "3")`).
function refine_replays!(cases, family, name, fx, find, refine; full = true, workspace = false)
    fmap = prefixed(FIND_MAP, "3")
    for seed in SEEDS
        r = find(fx; rng = StableRNG(seed))
        θ = r.u
        θ === nothing && continue
        replay!(cases, "$family/$name/scale_upstream/seed$seed", (fx, θ, r), ((fx, θ, r),) -> refine(fx, θ; scale = r); map = REFINE_MAP, fixture_map = fmap)
        replay!(cases, "$family/$name/ls/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; optimization_strategy = nothing); map = REFINE_MAP)
        replay!(cases, "$family/$name/scale1/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; scale = 1.0); map = REFINE_MAP)
        full || continue
        replay!(cases, "$family/$name/width2/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; width = 2.0); map = REFINE_MAP)
        replay!(cases, "$family/$name/triggs_cauchy_ec2.5_upstream/seed$seed", (fx, θ, r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r);
            map = REFINE_MAP, fixture_map = fmap)
        replay!(cases, "$family/$name/triggs_tls_width3/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{TLSShape}(), width = 3.0); map = REFINE_MAP)
        replay!(cases, "$family/$name/triggs_huber_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{HuberShape}(), scale = 1.0); map = REFINE_MAP)
        # the old workspace keyword: the refine's chart LM runs on the same (task-local) buffers without it
        workspace && replay!(cases, "$family/$name/workspace_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; scale = 1.0); map = REFINE_MAP)
        raw_r = find(fx; local_optimization_strategy = nothing, rng = StableRNG(seed))
        raw = raw_r.u
        raw === nothing && continue
        replay!(cases, "$family/$name/from_raw_upstream/seed$seed", (fx, raw, raw_r), ((fx, θ, r),) -> refine(fx, θ; scale = r);
            map = REFINE_MAP, fixture_map = fmap)
        replay!(cases, "$family/$name/from_raw_cauchy_ec2.5/seed$seed", (fx, raw, raw_r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r);
            map = REFINE_MAP, fixture_map = fmap)
    end
    return cases
end

"""
    message_free(m::FieldMap) -> FieldMap

`m` for a case that throws, whose message names a type or entry the migration renamed (the exception's type is still
compared).
"""
message_free(m::FieldMap) = FieldMap(m.renames, m.new_only, m.retyped, ["exception.msg"], m.aliases, m.retagged)

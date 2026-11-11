# =============================================================================
# VisualGeometryToolkitEvalTestExt — the evaluation engine's shared test fixtures
# =============================================================================
#
# ONE definition of every fixture more than one engine test builds, here and in
# the paper projects that test the engine on their own components
# (LocalOptimizationDoneRight's accv2026 test_engine_integration.jl copied them
# from this package's test/evaluation_engine_tests.jl): `engine_fixtures()`
# returns the module `EngineFixtures`, whose exports are the fixtures. Every
# caller is a test, so they load with the Test standard library.
# =============================================================================
module VisualGeometryToolkitEvalTestExt

using Test: Test
using VisualGeometryToolkitEval: VisualGeometryToolkitEval

module EngineFixtures
    using VisualGeometryToolkitCore.Pools: PoolPair, num_models
    using VisualGeometryToolkitEval.Pools: PoolResiduals, ScoringPool, synthetic_e_pool_pair, GT_CONVENTION
    using VisualGeometryToolkitEval.Tuning: tuning_spec, GridSpec, SpacedGrid, LinearCount, grid, save_tuning_artifact,
        tuning_table, EfficiencyConstantWindow, PairUnit
    using VisualGeometryToolkitCore: width_axis_name

    export EPAIR_GT_VARIANCE, epair, starts_pool, subpair, engine_scene, roster_tuning, write_roster_sidecar

    "The noise variance of [`epair`](@ref)'s pairs (σ̂² of the pool frame)."
    const EPAIR_GT_VARIANCE = 4.0e-6

    """
        epair(rng) -> (pair, residuals)

    THE synthetic E pool pair of the engine tests: 30 noisy correspondences, 12
    single-model minimal samples, σ̂² = `EPAIR_GT_VARIANCE` (Pools'
    `synthetic_e_pool_pair`), with its residual matrix.
    """
    epair(rng) = synthetic_e_pool_pair(rng; sibling_counts = fill(1, 12), noise = 0.002, δ_rot = 0.2,
                                       t_jitter = 0.1, gt_variance = EPAIR_GT_VARIANCE)

    """
        starts_pool(pairs, total) -> ScoringPool

    A one-scene E `ScoringPool` of `pairs` stamped with `total` minimal samples:
    what `sample_starts` certifies its boundaries against.
    """
    starts_pool(pairs, total) = ScoringPool(pairs, :E, "synthetic"; scene_names = fill("synthetic", length(pairs)),
        stamps = (gt_convention = GT_CONVENTION, pool_rng = "LehmerRNG", julia_version = string(VERSION),
                  total_samples = total, total_solutions = sum(num_models, pairs), mean_solutions_per_sample = 1.0,
                  provenance = Dict{String,Any}()))

    """
        subpair(pair, cols) -> PoolPair

    The pool pair restricted to its stored models `cols` (their sample indices and
    errors with them); the correspondences, cameras, ground truth and maps as they are.
    """
    subpair(pair::PoolPair{M}, cols) where {M} = PoolPair{M}(;
        csponds = pair.csponds, K1 = pair.K1, K2 = pair.K2, R_gt = pair.R_gt, t_gt = pair.t_gt,
        models = pair.models[cols], sample_indices = pair.sample_indices[:, cols],
        errors_r = pair.errors_r[cols], errors_t = pair.errors_t[cols],
        gt_inliers = pair.gt_inliers, gt_variance = pair.gt_variance,
        source_map = pair.source_map, destination_map = pair.destination_map, snn = pair.snn)

    """
        engine_scene(pairs, residuals, r) -> (scene, PoolResiduals)

    The scene the selection stream loads: the pairs `r` and their residual matrices
    (`:tangent_sampson`, key `"k"`).
    """
    engine_scene(pairs, residuals, r) = ((pairs = pairs[r],), PoolResiduals("k", :tangent_sampson, residuals[r]))

    """
        roster_tuning() -> (; spec, grids, values, window)

    The tuning record the roster readers are tested on: a windowed-bootstrap spec on
    the registered phototourism split, five-point grids of the three width axes
    (`grids`, and their `values`) and the efficiency-constant window `[1, 3.3]`.
    """
    function roster_tuning()
        spec = tuning_spec(Dict{String,Any}("tuning" => Dict{String,Any}("t" => Dict{String,Any}(
            "type" => "windowed_bootstrap", "efficiency_constant_window" => Dict("lo" => 1.0, "hi" => 3.3),
            "cross_validation" => Dict{String,Any}("scene_split" => Dict{String,Any}(
                "type" => "registered", "name" => "phototourism"))))), "t")
        grids = Dict{Symbol,GridSpec}(:outlier_halfwidth => SpacedGrid(0.01, 0.1, LinearCount(5)),
                                      :loss_width => SpacedGrid(0.001, 0.01, LinearCount(5)),
                                      :efficiency_constant => SpacedGrid(1.0, 3.3, LinearCount(5)))
        values = Dict(k => grid(v) for (k, v) in grids)
        return (; spec, grids, values, window = EfficiencyConstantWindow(1.0, 3.3))
    end

    """
        write_roster_sidecar(path, tuning, arms; code, mode, hits = no hits)

    A tuning sidecar of `arms` (composed or solo specs) for `tuning`
    ([`roster_tuning`](@ref)): every arm tuned at the third grid value of its
    selection and refit axes, objective 0, the window recorded; `hits` its
    boundary hits per arm; `code` the modules whose commits the record carries.
    """
    function write_roster_sidecar(path, tuning, arms; code, mode, hits = Dict(a.base => String[] for a in arms))
        g = tuning.values
        save_tuning_artifact(path, tuning.spec; grids = tuning.grids, unit = PairUnit(), code, mode,
            problem = "E", d_g = 1, s0 = 0.001, arms = [a.base for a in arms],
            tuned_selection = Dict(a.base => g[width_axis_name(a.selection)][3] for a in arms),
            tuned_width = Dict(a.base => g[width_axis_name(a.kind)][3] for a in arms),
            tuned_objective = Dict(a.base => 0.0 for a in arms), boundary_hits = hits,
            windows = Dict(a.base => tuning_table(tuning.window) for a in arms))
    end
end # module EngineFixtures

VisualGeometryToolkitEval.engine_fixtures() = EngineFixtures

end # module VisualGeometryToolkitEvalTestExt

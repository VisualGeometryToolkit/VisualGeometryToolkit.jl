# The case matrices record/est_epipolar.jl and record/est_pose.jl share (they call the old pipelines handed to them).
# They live beside the fixtures because every file of record/ is a group. The conic RANSAC's matrices
# (record/est_conics_ransac.jl, `cr_refine_cases!`/`cr_refusal_cases!`) are their own: other scales and widths
# (0.5, 1.5) and another refusal set, as the conic tests have them.

# The refusals every find_* and refine_* shares (core_defaults, refine_contract, ransac_no_model). `find === nothing`:
# the class has no find_*; `P`, the class's problem type, names the scorer of the refused `scoring` keyword.
function refusal_cases!(cases, family, fx, find, refine, θ, P)
    if find !== nothing
        case!(cases, "$family/refused/find_threshold_kw", fx, fx -> find(fx; threshold = 3.0))
        case!(cases, "$family/refused/find_outlier_halfwidth_kw", fx, fx -> find(fx; outlier_halfwidth = 20.0))
        case!(cases, "$family/refused/find_retired_local_optimization", fx, fx -> find(fx; local_optimization = Irls{TLSShape}()))
        case!(cases, "$family/refused/find_unknown_kw", fx, fx -> find(fx; no_such_keyword = 1))
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
        case!(cases, "$family/refused/refine_$lbl", (fx, θ), ((fx, θ),) -> refine(fx, θ; kw...))
    end
    return cases
end

# The refine_* settings from each seed's find (its model and its search's scale), and from the raw incumbent.
# `refine(fx, θ; kw...)` is the old refine on the find's fixture; `workspace(fx)` the class's RANSAC workspace for the
# workspace case, a closure `fx -> RansacWorkspace(…)` (`nothing`: no such case).
function refine_cases!(cases, family, name, fx, find, refine; full = true, workspace = nothing)
    for seed in SEEDS
        r = find(fx; rng = StableRNG(seed))
        θ = estimate(r)
        θ === nothing && continue
        case!(cases, "$family/$name/scale_upstream/seed$seed", (fx, θ, r), ((fx, θ, r),) -> refine(fx, θ; scale = r))
        case!(cases, "$family/$name/ls/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; optimization_strategy = nothing))
        case!(cases, "$family/$name/scale1/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; scale = 1.0))
        full || continue
        case!(cases, "$family/$name/width2/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; width = 2.0))
        case!(cases, "$family/$name/triggs_cauchy_ec2.5_upstream/seed$seed", (fx, θ, r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r))
        case!(cases, "$family/$name/triggs_tls_width3/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{TLSShape}(), width = 3.0))
        case!(cases, "$family/$name/triggs_huber_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{HuberShape}(), scale = 1.0))
        workspace === nothing || case!(cases, "$family/$name/workspace_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; scale = 1.0, workspace = workspace(fx)))
        # from the raw RANSAC incumbent (no local optimization) at its search's scale (core_defaults)
        raw_r = find(fx; local_optimization_strategy = nothing, rng = StableRNG(seed))
        raw = estimate(raw_r)
        raw === nothing && continue
        case!(cases, "$family/$name/from_raw_upstream/seed$seed", (fx, raw, raw_r), ((fx, θ, r),) -> refine(fx, θ; scale = r))
        case!(cases, "$family/$name/from_raw_cauchy_ec2.5/seed$seed", (fx, raw, raw_r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r))
    end
    return cases
end

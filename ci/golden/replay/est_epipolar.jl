# The epipolar estimators on the new API (record/est_epipolar.jl, case for case): the RANSAC find and the
# LevenbergMarquardt refine of the fundamental, essential and rotation problems; the relative-pose bundle's cases are
# its block problem's (replay/_relpose_bundle_part.jl).
using VisualGeometryToolkitCore: MsacScoring, RansacScoring, StudentTUniformScoring, Irls, Triggs, TLSShape, WelschShape,
    CauchyShape, HuberShape, GaussianShape
using VisualGeometryToolkitEstimation: FundMatProblem, EssentialProblem, RotationProblem, RansacConfig, RansacWorkspace,
    TerminalLO, scoring_from_toml
Base.include(@__MODULE__, joinpath(@__DIR__, "_common.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "_relpose_bundle_part.jl"))

const EP_LOS = [
    "classdefault" => (;),
    "none" => (; local_optimization_strategy = nothing),
    "irls_tls" => (; local_optimization_strategy = Irls{TLSShape}()),
    "irls_welsch_ec1.5" => (; local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5)),
    "triggs_welsch" => (; local_optimization_strategy = Triggs{WelschShape}()),
    "terminal_irls_tls" => (; local_optimization_strategy = TerminalLO(Irls{TLSShape}())),
]

# Every find setting of one class on one scene; the old `workspace` keyword is the storage precision of the new
# `Ransac(T)` (its MSAC workspace is the one `init` builds).
function ep_find_replays!(cases, family, name, fx, find, problem, thr, P, n; full = true)
    for (lbl, kw) in (full ? EP_LOS : EP_LOS[1:2]), seed in SEEDS
        replay!(cases, "$family/$name/lo_$lbl/seed$seed", fx, fx -> find(fx; rng = StableRNG(seed), kw...); map = FIND_MAP)
    end
    full || return cases
    for seed in SEEDS
        replay!(cases, "$family/$name/msac_class_threshold/seed$seed", fx,
            fx -> find(fx; scoring = MsacScoring(problem(fx); threshold = thr), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/ransac_scoring/seed$seed", fx,
            fx -> find(fx; scoring = RansacScoring(problem(fx); threshold = 2thr), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/student_t_uniform_7.25/seed$seed", fx,
            fx -> find(fx; scoring = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 7.25);
                                                       problem = typeof(problem(fx))), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/capped_2_2/seed$seed", fx,
            fx -> find(fx; config = RansacConfig(max_trials = 2, min_trials = 2), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/cfg5000_500_irls_welsch/seed$seed", fx,
            fx -> find(fx; config = RansacConfig(max_trials = 5000, min_trials = 500),
                       local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/workspace/seed$seed", fx,
            fx -> find(fx; workspace = RansacWorkspace(P, n), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/workspace_f64/seed$seed", fx,
            fx -> find(fx; workspace = RansacWorkspace(Float64, P, n), rng = StableRNG(seed)); map = FIND_MAP)
    end
    return cases
end

function replay_est_epipolar(cases)
    I3 = SMatrix{3, 3, Float64}(I)
    X3 = SMatrix{3, 3, Float64}(0, 0, 0, 0, 0, -1, 0, 1, 0)

    # ---------------------------------------------------------------- fundamental matrix
    ffind(cs; kw...) = ransac_solve(FundMatProblem(cs); kw...)
    frefine(cs, θ; kw...) = lm_solve(FundMatProblem(cs), θ; kw...)
    for (k, (name, cs)) in enumerate(ep_fundmat_scenes())
        full = k == 1 || k == 2
        ep_find_replays!(cases, "fundmat_find", name, cs, ffind, FundMatProblem, 3.0, FundMatProblem{Float64}, length(cs); full)
        refine_replays!(cases, "fundmat_refine", name, cs, ffind, frefine; full = k == 1, workspace = true)
    end
    few = few_correspondences(6)
    for seed in SEEDS
        replay!(cases, "fundmat_find/too_few_6/seed$seed", few, cs -> ffind(cs; rng = StableRNG(seed)); map = FIND_MAP)
    end
    F0 = VgtkCore.FundMat{Float64}(Tuple(X3))
    replay!(cases, "fundmat_refine/too_few_6/scale1", (few, F0), ((cs, F),) -> frefine(cs, F; scale = 1.0); map = REFINE_MAP)
    replay!(cases, "fundmat_refine/too_few_6/ls", (few, F0), ((cs, F),) -> frefine(cs, F; optimization_strategy = nothing); map = REFINE_MAP)
    cs1 = last(first(ep_fundmat_scenes()))
    refusal_replays!(cases, "fundmat", cs1, ffind, frefine, ffind(cs1; rng = StableRNG(1)).u, FundMatProblem)
    replay!(cases, "fundmat/refused/find_zero_threshold", cs1, cs -> ffind(cs; scoring = MsacScoring(FundMatProblem(cs); threshold = 0.0));
        map = REFUSAL_MAP)

    # ---------------------------------------------------------------- essential matrix
    efind((c1, c2, cs); kw...) = ransac_solve(EssentialProblem(c1, c2, cs); kw...)
    erefine((c1, c2, cs), θ; kw...) = lm_solve(EssentialProblem(c1, c2, cs), θ; kw...)
    eprob((c1, c2, cs)) = EssentialProblem(c1, c2, cs)
    for (k, (name, fx)) in enumerate(ep_essential_scenes())
        full = k <= 2
        n = length(fx[3])
        ep_find_replays!(cases, "essential_find", name, fx, efind, eprob, 1.0, EssentialProblem{Float64}, n; full)
        refine_replays!(cases, "essential_refine", name, fx, efind, erefine; full = k == 1, workspace = true)
        if name == "calib_ecam_120+50"
            for seed in SEEDS
                replay!(cases, "essential_find/$name/msac5/seed$seed", fx,
                    fx -> efind(fx; scoring = MsacScoring(EssentialProblem; threshold = 5.0), rng = StableRNG(seed)); map = FIND_MAP)
            end
        end
    end
    fx4 = (EP_ECAM, EP_ECAM, few_correspondences(4))
    for seed in SEEDS
        replay!(cases, "essential_find/too_few_4/seed$seed", fx4, fx -> efind(fx; rng = StableRNG(seed)); map = FIND_MAP)
    end
    E0 = VgtkCore.EssentialMat{Float64}(Tuple(X3))
    replay!(cases, "essential_refine/too_few_4/scale1", (fx4, E0), ((fx, E),) -> erefine(fx, E; scale = 1.0); map = REFINE_MAP)
    replay!(cases, "essential_refine/too_few_4/ls", (fx4, E0), ((fx, E),) -> erefine(fx, E; optimization_strategy = nothing); map = REFINE_MAP)
    efx = last(ep_essential_scenes()[2])
    refusal_replays!(cases, "essential", efx, efind, erefine, efind(efx; rng = StableRNG(1)).u, FundMatProblem)
    for thr in (0.02, 0.05)
        replay!(cases, "essential/refused/find_msac_threshold$thr", efx, fx -> efind(fx; scoring = MsacScoring(eprob(fx); threshold = thr));
            map = REFUSAL_MAP)
    end
    replay!(cases, "essential/refused/find_msac_type_threshold0.02", efx,
        fx -> efind(fx; scoring = MsacScoring(EssentialProblem; threshold = 0.02)); map = REFUSAL_MAP)
    for seed in SEEDS
        replay!(cases, "essential_find/epix_cam800_150+60/msac_floor0.06/seed$seed", efx,
            fx -> efind(fx; scoring = MsacScoring(eprob(fx); threshold = 0.06), rng = StableRNG(seed)); map = FIND_MAP)
    end

    # ---------------------------------------------------------------- pure rotation (camera and bearing forms)
    rfind((c1, c2, cs); kw...) = ransac_solve(RotationProblem(c1, c2, cs); kw...)
    rrefine((c1, c2, cs), θ; kw...) = lm_solve(RotationProblem(c1, c2, cs), θ; kw...)
    rprob((c1, c2, cs)) = RotationProblem(c1, c2, cs)
    bfind(b; kw...) = ransac_solve(RotationProblem(b); kw...)
    brefine(b, θ; kw...) = lm_solve(RotationProblem(b), θ; kw...)
    for (k, (name, fx)) in enumerate(ep_rotation_scenes())
        n = length(fx[3])
        ep_find_replays!(cases, "rotation_find", name, fx, rfind, rprob, 1.0, RotationProblem, n; full = k == 1)
        refine_replays!(cases, "rotation_refine", name, fx, rfind, rrefine; full = k == 1, workspace = true)
        b = ep_bearings(fx...)
        ep_find_replays!(cases, "rotation_find", "$(name)_bearings", b, bfind, RotationProblem, 1.0, RotationProblem, n; full = k == 1)
        refine_replays!(cases, "rotation_refine", "$(name)_bearings", b, bfind, brefine; full = false, workspace = true)
    end
    for (name, fx) in ("pcam_1" => (PCAM, PCAM, few_correspondences(1)),
                       "fisheye_1" => (FISHEYE, FISHEYE, VgtkCore.Correspondences([SA[600.0, 600.0]], [SA[600.0, 600.0]])))
        for seed in SEEDS
            replay!(cases, "rotation_find/too_few_$name/seed$seed", fx, fx -> rfind(fx; rng = StableRNG(seed)); map = FIND_MAP)
        end
        R1 = one(RotMatrix{3, Float64})
        replay!(cases, "rotation_refine/too_few_$name/scale1", (fx, R1), ((fx, R),) -> rrefine(fx, R; scale = 1.0); map = REFINE_MAP)
    end
    b1 = VgtkCore.Correspondences([SA[0.0, 0.0, 1.0]], [SA[0.0, 0.0, 1.0]])
    for seed in SEEDS
        replay!(cases, "rotation_find/too_few_bearings_1/seed$seed", b1, b -> bfind(b; rng = StableRNG(seed)); map = FIND_MAP)
    end
    replay!(cases, "rotation_refine/too_few_bearings_1/scale1", (b1, one(RotMatrix{3, Float64})), ((b, R),) -> brefine(b, R; scale = 1.0);
        map = REFINE_MAP)
    replay!(cases, "rotation_refine/too_few_bearings_1/ls", (b1, one(RotMatrix{3, Float64})),
        ((b, R),) -> brefine(b, R; optimization_strategy = nothing); map = REFINE_MAP)
    rfx = last(first(ep_rotation_scenes()))
    refusal_replays!(cases, "rotation", rfx, rfind, rrefine, rfind(rfx; rng = StableRNG(1)).u, FundMatProblem)
    replay!(cases, "rotation/refused/find_msac_threshold0.01", rfx,
        fx -> rfind(fx; scoring = MsacScoring(RotationProblem; threshold = 0.01)); map = REFUSAL_MAP)
    # a pair vector carries no attribute columns: no bearing form takes one (a MethodError, as before)
    replay!(cases, "rotation/refused/bearings_pairs_vector", b1,
        b -> bfind(collect(VgtkCore.sources(b)) .=> collect(VgtkCore.targets(b)); rng = StableRNG(1)); map = REFUSAL_MAP)

    # ---------------------------------------------------------------- the relative-pose bundle (its block problem)
    relpose_bundle_replays!(cases)
    return cases
end

# The homography estimators on Core 0.14: DLT, the 4-point minimal solve, the RANSAC find and the LevenbergMarquardt
# refine of a HomographyProblem (record/homography.jl, case for case).
using VisualGeometryToolkitCore: DLT, Homography4pt, MsacScoring, RansacScoring, Irls, Triggs, TLSShape, WelschShape,
    CauchyShape, HuberShape, GaussianShape, withattrs, Correspondences, HomographyMat
using VisualGeometryToolkitEstimation: HomographyProblem, RansacConfig, TerminalLO
Base.include(@__MODULE__, joinpath(@__DIR__, "_common.jl"))

const HOMOGRAPHY_LOS = [
    "classdefault" => (;),
    "none" => (; local_optimization_strategy = nothing),
    "irls_tls" => (; local_optimization_strategy = Irls{TLSShape}()),
    "triggs_welsch" => (; local_optimization_strategy = Triggs{WelschShape}()),
    "terminal_irls_tls" => (; local_optimization_strategy = TerminalLO(Irls{TLSShape}())),
]

# A closed form that returned a bare model now returns a FitResult: its `u` is the old value.
const DLT_MAP = FieldMap(; renames = ["u" => ""], new_only = ["", "resid", "retcode", "stats", "prob", "alg", "original", "stop_rule",
                                                             "rms", "n", "metadata"])
dlt(cs) = solve(EstimationProblem(HomographyMat, cs), DLT())

function replay_homography(cases)
    hfind(cs; kw...) = ransac_solve(HomographyProblem(cs); kw...)
    hrefine(cs, θ; kw...) = lm_solve(HomographyProblem(cs), θ; kw...)
    for (name, cs) in homography_scenes()
        n = length(cs)
        attributed = eltype(VgtkCore.sources(cs)) <: VgtkCore.Attributed
        if !attributed
            inl = collect(1:min(n, 50))
            mask = falses(n); mask[inl] .= true
            replay!(cases, "dlt/$name/plain", cs, dlt; map = DLT_MAP)
            # a mask is data selection (spec §9): the masked correspondences
            replay!(cases, "dlt/$name/index_mask", (cs, inl), ((cs, m),) -> dlt(cs[m]); map = DLT_MAP)
            replay!(cases, "dlt/$name/bool_mask", (cs, mask), ((cs, m),) -> dlt(cs[m]); map = DLT_MAP)
            # `svd_ws` left the public form (a workspace keyword): the same solve
            replay!(cases, "dlt/$name/svd_ws", cs, dlt; map = DLT_MAP)
            w = collect(range(0.2, 1.0; length = n))
            csw = withattrs(Correspondences(VgtkCore.sources(cs), VgtkCore.targets(cs)); weights = w)
            replay!(cases, "dlt/$name/weights", csw, dlt; map = DLT_MAP)
            rng = StableRNG(1)
            idxs = [randperm(rng, n)[1:4] for _ in 1:300]
            replay!(cases, "minimal4pt/$name/300samples", (cs, idxs),
                ((cs, idxs),) -> [solve(MinimalProblem(HomographyMat, cs, idx), Homography4pt()) for idx in idxs])
        end
        for (lbl, kw) in HOMOGRAPHY_LOS, seed in SEEDS
            replay!(cases, "find/$name/lo_$lbl/seed$seed", cs, cs -> hfind(cs; rng = StableRNG(seed), kw...); map = FIND_MAP)
        end
        for seed in SEEDS
            replay!(cases, "find/$name/msac2_cfg300_20/seed$seed", cs,
                cs -> hfind(cs; scoring = MsacScoring(HomographyProblem; threshold = 2.0),
                    config = RansacConfig(max_trials = 300, min_trials = 20), rng = StableRNG(seed)); map = FIND_MAP)
            replay!(cases, "find/$name/ransac3/seed$seed", cs,
                cs -> hfind(cs; scoring = RansacScoring(HomographyProblem(cs); threshold = 3.0), rng = StableRNG(seed)); map = FIND_MAP)
        end
        fmap = prefixed(FIND_MAP, "3")
        for seed in SEEDS
            r = hfind(cs; rng = StableRNG(seed))
            H0 = r.u
            H0 === nothing && continue
            fx = (cs, H0, r)
            replay!(cases, "refine/$name/classdefault_scale_r/seed$seed", fx, ((cs, H0, r),) -> hrefine(cs, H0; scale = r);
                map = REFINE_MAP, fixture_map = fmap)
            replay!(cases, "refine/$name/ls/seed$seed", (cs, H0), ((cs, H0),) -> hrefine(cs, H0; optimization_strategy = nothing); map = REFINE_MAP)
            replay!(cases, "refine/$name/triggs_tls_width3/seed$seed", (cs, H0),
                ((cs, H0),) -> hrefine(cs, H0; optimization_strategy = Triggs{TLSShape}(), width = 3.0); map = REFINE_MAP)
            replay!(cases, "refine/$name/triggs_cauchy_scale0.7/seed$seed", (cs, H0),
                ((cs, H0),) -> hrefine(cs, H0; optimization_strategy = Triggs{CauchyShape}(), scale = 0.7); map = REFINE_MAP)
        end
    end
    few = homography_too_few()
    H = _HS.H_TRUE
    for seed in SEEDS
        replay!(cases, "find/too_few_3/seed$seed", few, cs -> hfind(cs; rng = StableRNG(seed)); map = FIND_MAP)
    end
    replay!(cases, "refine/too_few_3/scale1", (few, H), ((cs, H),) -> hrefine(cs, H; scale = 1.0); map = REFINE_MAP)
    replay!(cases, "dlt/too_few_3", few, dlt; map = DLT_MAP)
    cs = last(first(homography_scenes()))
    replay!(cases, "refine/refused/no_scale", (cs, H), ((cs, H),) -> hrefine(cs, H); map = REFUSAL_MAP)
    replay!(cases, "refine/refused/scale_and_width", (cs, H), ((cs, H),) -> hrefine(cs, H; scale = 1.0, width = 1.0); map = REFUSAL_MAP)
    replay!(cases, "refine/refused/negative_scale", (cs, H), ((cs, H),) -> hrefine(cs, H; scale = -1.0); map = REFUSAL_MAP)
    return cases
end

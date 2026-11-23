# The old homography estimators at 3d8a06da: homography_dlt, solve(Homography4pt(), cs, idx), find_homography,
# refine_homography (and ransac/ransac! on HomographyProblem, the preallocated form).
using VisualGeometryToolkitCore: homography_dlt, estimate, Homography4pt, MsacScoring, RansacScoring, Irls, Triggs,
    TLSShape, WelschShape, CauchyShape, withattrs, SvdWorkspace, Correspondences
using VisualGeometryToolkitEstimation: find_homography, refine_homography, HomographyProblem, RansacConfig, TerminalLO

const HOMOGRAPHY_LOS = [
    "classdefault" => (;),
    "none" => (; local_optimization_strategy = nothing),
    "irls_tls" => (; local_optimization_strategy = Irls{TLSShape}()),
    "triggs_welsch" => (; local_optimization_strategy = Triggs{WelschShape}()),
    "terminal_irls_tls" => (; local_optimization_strategy = TerminalLO(Irls{TLSShape}())),
]

function record_homography(cases)
    for (name, cs) in homography_scenes()
        n = length(cs)
        attributed = eltype(VgtkCore.sources(cs)) <: VgtkCore.Attributed   # the DLT and the 4-point kernel take plain points
        if !attributed
        # --- closed form: DLT -------------------------------------------------------------------------------
        inl = collect(1:min(n, 50))
        mask = falses(n); mask[inl] .= true
        case!(cases, "dlt/$name/plain", cs, cs -> homography_dlt(cs))
        case!(cases, "dlt/$name/index_mask", (cs, inl), ((cs, m),) -> homography_dlt(cs; mask = m))
        case!(cases, "dlt/$name/bool_mask", (cs, mask), ((cs, m),) -> homography_dlt(cs; mask = m))
        case!(cases, "dlt/$name/svd_ws", cs, cs -> homography_dlt(cs; svd_ws = SvdWorkspace{Float64}(2 * length(cs), 9)))
        w = collect(range(0.2, 1.0; length = n))
        csw = withattrs(Correspondences(VgtkCore.sources(cs), VgtkCore.targets(cs)); weights = w)
        case!(cases, "dlt/$name/weights", csw, cs -> homography_dlt(cs))
        # --- minimal: 300 random 4-samples, every root ------------------------------------------------------
        rng = StableRNG(1)
        idxs = [randperm(rng, n)[1:4] for _ in 1:300]
        case!(cases, "minimal4pt/$name/300samples", (cs, idxs), ((cs, idxs),) -> [solve(Homography4pt(), cs, idx) for idx in idxs])
        end
        # --- find: every LO strategy × seeds ------------------------------------------------------------
        for (lbl, kw) in HOMOGRAPHY_LOS, seed in SEEDS
            case!(cases, "find/$name/lo_$lbl/seed$seed", cs, cs -> find_homography(cs; rng = StableRNG(seed), kw...))
        end
        for seed in SEEDS
            case!(cases, "find/$name/msac2_cfg300_20/seed$seed", cs,
                cs -> find_homography(cs; scoring = MsacScoring(HomographyProblem; threshold = 2.0),
                    config = RansacConfig(max_trials = 300, min_trials = 20), rng = StableRNG(seed)))
            case!(cases, "find/$name/ransac3/seed$seed", cs,
                cs -> find_homography(cs; scoring = RansacScoring(HomographyProblem(cs); threshold = 3.0), rng = StableRNG(seed)))
        end
        # --- refine from each seed's find ---------------------------------------------------------------
        for seed in SEEDS
            r = find_homography(cs; rng = StableRNG(seed))
            H0 = estimate(r)
            H0 === nothing && continue
            fx = (cs, H0, r)
            case!(cases, "refine/$name/classdefault_scale_r/seed$seed", fx, ((cs, H0, r),) -> refine_homography(H0, cs; scale = r))
            case!(cases, "refine/$name/ls/seed$seed", (cs, H0), ((cs, H0),) -> refine_homography(H0, cs; optimization_strategy = nothing))
            case!(cases, "refine/$name/triggs_tls_width3/seed$seed", (cs, H0),
                ((cs, H0),) -> refine_homography(H0, cs; optimization_strategy = Triggs{TLSShape}(), width = 3.0))
            case!(cases, "refine/$name/triggs_cauchy_scale0.7/seed$seed", (cs, H0),
                ((cs, H0),) -> refine_homography(H0, cs; optimization_strategy = Triggs{CauchyShape}(), scale = 0.7))
        end
    end
    # --- too few data, and the refusals ---------------------------------------------------------------------
    few = homography_too_few()
    H = _HS.H_TRUE
    for seed in SEEDS
        case!(cases, "find/too_few_3/seed$seed", few, cs -> find_homography(cs; rng = StableRNG(seed)))
    end
    case!(cases, "refine/too_few_3/scale1", (few, H), ((cs, H),) -> refine_homography(H, cs; scale = 1.0))
    case!(cases, "dlt/too_few_3", few, cs -> homography_dlt(cs))
    cs = last(first(homography_scenes()))
    case!(cases, "refine/refused/no_scale", (cs, H), ((cs, H),) -> refine_homography(H, cs))
    case!(cases, "refine/refused/scale_and_width", (cs, H), ((cs, H),) -> refine_homography(H, cs; scale = 1.0, width = 1.0))
    case!(cases, "refine/refused/negative_scale", (cs, H), ((cs, H),) -> refine_homography(H, cs; scale = -1.0))
    return cases
end

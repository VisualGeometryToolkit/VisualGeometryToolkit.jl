# TriggsLM — the robustified chart LM (one NLLSsolver LM under a loss with
# Triggs' second-order correction, width scale_multiple·ŝ): its strategy
# identity and configuration, and that it serves every chart problem — the
# circle (its pipeline default) and the homography.

@testitem "TriggsLM: strategy identity and configuration" begin
    using Test
    using VisualGeometryToolkitCore: CauchyShape, WelschShape, ScaleMultiple, LossWidth, lo_width, with_width,
        optimization_path, scale_interface, ConsensusAnchored, lo_label
    using VisualGeometryToolkitEstimation: TriggsLM, LO_TAGS, local_optimization_from_toml, config_dict
    lo = TriggsLM{CauchyShape}(scale_multiple = 2.5)
    @test lo_width(lo) == ScaleMultiple(2.5) && optimization_path(lo) == :irls && scale_interface(lo) == ConsensusAnchored()
    @test LO_TAGS["triggs_lm"] === TriggsLM
    @test local_optimization_from_toml(Dict{String, Any}("type" => "triggs_lm", "loss" => "cauchy", "scale_multiple" => 2.5);
                                       class = :circle) == lo
    @test local_optimization_from_toml(config_dict(lo); class = :circle) == lo
    @test with_width(lo, ScaleMultiple(3.0)) == TriggsLM{CauchyShape}(scale_multiple = 3.0)
    # the multiple is required, and the width is always a scale multiple
    @test_throws UndefKeywordError TriggsLM{CauchyShape}()
    @test_throws ArgumentError TriggsLM{CauchyShape}(loss_width = 2.0)
end

@testitem "TriggsLM: the homography and the circle refit with it" setup = [HomographyScenes] begin
    using Test, StableRNGs, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: CauchyShape, WelschShape, MsacScoring, estimate, converged, csponds, Point2
    using VisualGeometryToolkitCore.Synthetic: projective_distance
    using VisualGeometryToolkitEstimation: TriggsLM, HomographyProblem, CircleProblem, ransac, locally_optimize,
        find_homography, RansacConfig
    import VisualGeometryToolkitEstimation as VgtkEstimation
    src, dst = homography_pairs(StableRNG(41), H_PX; n_in = 150, n_out = 60, σ = 0.5, source = gaussian_source)
    pairs = src .=> dst
    lo = TriggsLM{CauchyShape}(scale_multiple = 2.5)
    res = find_homography(pairs; local_optimization = lo, rng = StableRNG(42))
    irls = find_homography(pairs; rng = StableRNG(42))                    # the class default, Irls{TLSShape}
    err(H) = projective_distance(SMatrix{3, 3}(estimate(H)), SMatrix{3, 3}(H_PX))
    @test converged(res)
    @test err(res) < 1.0e-2 && err(res) < 2 * err(irls) + 1.0e-3          # at the default's accuracy (0.5 px noise)
    # one refit from a perturbed hypothesis lowers the robust cost and lands near the truth
    p = HomographyProblem(pairs)
    sc = MsacScoring(p; threshold = 3.0)
    H0 = estimate(find_homography(pairs; rng = StableRNG(43)))
    r = locally_optimize(p, H0; local_optimization = lo, scoring = sc)
    @test converged(r)
    @test err(r) < 1.0e-2
    # the Welsch loss runs through the same strategy (any shape the robustifier takes)
    @test converged(find_homography(pairs; local_optimization = TriggsLM{WelschShape}(scale_multiple = 2.0), rng = StableRNG(44)))
    # a codimension-1 problem through the same method: the circle. The Cauchy
    # loss's influence decays as loss_width²/r, never to zero, so far outliers
    # pull the M-estimate by ~loss_width²·n_out/(n_in·r_out): measured 0.006 px
    # here (noise 0.3 px, a straight run 20–200 px off the circle).
    rng = StableRNG(46)
    pts = [Point2{Float64}(100 + 50cos(t) + 0.3randn(rng), 80 + 50sin(t) + 0.3randn(rng)) for t in range(0, 2π, length = 61)[1:60]]
    append!(pts, [Point2{Float64}(10.0i, 3.0i) for i in 1:20])
    pc = CircleProblem(pts)
    rc = ransac(pc, MsacScoring(pc; threshold = 1.0); local_optimization = lo, config = RansacConfig(max_trials = 200),
                rng = StableRNG(45))
    c = estimate(rc)
    @test converged(rc) && norm(SVector(c.center...) - SVector(100.0, 80.0)) < 0.2 && abs(c.r - 50) < 0.2   # 0.3 px on 60 points: σ/√30 ≈ 0.05 px
end

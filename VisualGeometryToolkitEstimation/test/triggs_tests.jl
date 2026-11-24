# Triggs — the robustified chart LM (one NLLSsolver LM under a loss, width
# efficiency_constant·ŝ): its TOML reader and writer (its identity is Core's), that it serves every chart problem — the circle
# (its pipeline default, Welsch) and the homography — and that it admits exactly
# the loss shapes Irls admits.

@testitem "Triggs: the TOML reader and writer" begin
    # The strategy's identity, width modes and fixed_width's freeze are Core's (VisualGeometryToolkitCore's
    # strategies_tests); this item holds what this package owns: its registry and its TOML reader and writer.
    using Test
    using VisualGeometryToolkitCore: CauchyShape
    using VisualGeometryToolkitEstimation: OPTIMIZATION_STRATEGY_TAGS, optimization_strategy_from_toml, config_dict
    using VisualGeometryToolkitCore: Triggs
    lo = Triggs{CauchyShape}(efficiency_constant = 2.5)
    @test OPTIMIZATION_STRATEGY_TAGS["triggs"] === Triggs
    @test optimization_strategy_from_toml(Dict{String, Any}("type" => "triggs", "loss" => "cauchy", "efficiency_constant" => 2.5);
                                       class = :circle) == lo
    @test optimization_strategy_from_toml(config_dict(lo); class = :circle) == lo
    # a table with no constant reads the strategy with none (a refine's efficiency_constant(K, d))
    @test optimization_strategy_from_toml(Dict{String, Any}("type" => "triggs", "loss" => "cauchy"); class = :circle) ==
        Triggs{CauchyShape}()
    @test_throws ArgumentError optimization_strategy_from_toml(Dict{String, Any}("type" => "triggs", "loss" => "cauchy", "loss_width" => 2.0);
                                                            class = :circle)
end

@testitem "Triggs: the homography and the circle refit with it" setup = [HomographyScenes] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac, LocalOptimization
    using Test, StableRNGs, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: CauchyShape, WelschShape, MsacScoring, correspondences, Point2
    using VisualGeometryToolkitCore.Synthetic: projective_distance
    using VisualGeometryToolkitEstimation: HomographyProblem, CircleProblem, RansacConfig
    using VisualGeometryToolkitCore: Triggs
    import VisualGeometryToolkitEstimation as VgtkEstimation
    src, dst = homography_pairs(StableRNG(41), H_PX; n_in = 150, n_out = 60, σ = 0.5, source = gaussian_source)
    pairs = src .=> dst
    lo = Triggs{CauchyShape}(efficiency_constant = 2.5)
    res = solve(EstimationProblem(HomographyProblem(pairs)), Ransac(; local_optimization_strategy = lo); rng = StableRNG(42))
    irls = solve(EstimationProblem(HomographyProblem(pairs)), Ransac(); rng = StableRNG(42))                    # the class default, Irls{TLSShape}
    err(H) = projective_distance(SMatrix{3, 3}(H.u), SMatrix{3, 3}(H_PX))
    @test successful_retcode(res)
    @test err(res) < 1.0e-2 && err(res) < 2 * err(irls) + 1.0e-3          # at the default's accuracy (0.5 px noise)
    # one refit from a perturbed hypothesis lowers the robust cost and lands near the truth
    p = HomographyProblem(pairs)
    sc = MsacScoring(p; threshold = 3.0)
    H0 = solve(EstimationProblem(HomographyProblem(pairs)), Ransac(); rng = StableRNG(43)).u
    r = solve(RefineProblem(p, H0), LocalOptimization(lo; scoring = sc))
    @test successful_retcode(r)
    @test err(r) < 1.0e-2
    # the Welsch loss runs through the same strategy (any shape the robustifier takes)
    @test successful_retcode(solve(EstimationProblem(HomographyProblem(pairs)), Ransac(; local_optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0)); rng = StableRNG(44)))
    # a codimension-1 problem through the same method: the circle. The Cauchy
    # loss's influence decays as loss_width²/r, never to zero, so far outliers
    # pull the M-estimate by ~loss_width²·n_out/(n_in·r_out): measured 0.006 px
    # here (noise 0.3 px, a straight run 20–200 px off the circle).
    rng = StableRNG(46)
    pts = [Point2{Float64}(100 + 50cos(t) + 0.3randn(rng), 80 + 50sin(t) + 0.3randn(rng)) for t in range(0, 2π, length = 61)[1:60]]
    append!(pts, [Point2{Float64}(10.0i, 3.0i) for i in 1:20])
    pc = CircleProblem(pts)
    rc = solve(EstimationProblem(pc), Ransac(; scoring = MsacScoring(pc; threshold = 1.0), local_optimization_strategy = lo, config = RansacConfig(max_trials = 200)); rng = StableRNG(45))
    c = rc.u
    @test successful_retcode(rc) && norm(SVector(c.center...) - SVector(100.0, 80.0)) < 0.2 && abs(c.r - 50) < 0.2   # 0.3 px on 60 points: σ/√30 ≈ 0.05 px
end

@testitem "Triggs: admits exactly the shapes Irls admits, each refines a circle" begin
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using VisualGeometryToolkitEstimation: CircleProblem
    using Test, StableRNGs, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: GaussianShape, Point2, Circle
    using VisualGeometryToolkitEstimation: optimization_strategy_from_toml, config_dict
    using VisualGeometryToolkitCore: Triggs, Irls
    import VisualGeometryToolkitCore as VgtkCore
    admits(f) = try
        f(); true
    catch e
        e isa ArgumentError || rethrow()
        false
    end
    rng = StableRNG(51)
    C₀ = SVector(200.0, 150.0); R₀ = 80.0
    pts = [Point2{Float64}(C₀ + R₀ * SVector(cos(t), sin(t)) + 0.3 * randn(rng, SVector{2})) for t in 2π .* rand(rng, 200)]
    g = Circle(Point2(C₀ + SVector(1.0, -0.8)), R₀ + 1.2)
    n_admitted = 0
    for (K, tag) in VgtkCore._LOSS_TABLE   # the loss table is Core's (A2)
        # the same set as Irls: no narrower, no wider (Barron, whose shape carries its α, is refused by both)
        width = K === GaussianShape ? (;) : (; efficiency_constant = 2.5)
        @test admits(() -> Irls{K}()) == admits(() -> Triggs{K}(; width...))
        admits(() -> Irls{K}()) || (@test_throws "$(nameof(K))" Triggs{K}(; width...); continue)
        n_admitted += 1
        lo = Triggs{K}(; width...)
        toml = Dict{String, Any}("type" => "triggs", "loss" => tag, (K === GaussianShape ? () : ("efficiency_constant" => 2.5,))...)
        @test optimization_strategy_from_toml(toml; class = :circle) == lo
        @test optimization_strategy_from_toml(config_dict(lo); class = :circle) == lo
        # outlier-free points at σ = 0.3 px: to the noise level (σ/√100 ≈ 0.03 px)
        r = K === GaussianShape ? nothing : solve(RefineProblem(CircleProblem(pts), g), LevenbergMarquardt(; optimization_strategy = lo); scale = 0.3)
        r === nothing && continue          # (least squares is spelled `nothing` on a refine: refine_contract_tests.jl)
        c = r.u
        @test successful_retcode(r)
        @test norm(SVector(c.center...) - C₀) < 0.1 && abs(c.r - R₀) < 0.1
    end
    @test n_admitted == 7
    # the plain LM has no width
    @test_throws ArgumentError Triggs{GaussianShape}(efficiency_constant = 2.5)
    @test_throws ArgumentError Triggs{GaussianShape}(loss_width = 2.0)
end

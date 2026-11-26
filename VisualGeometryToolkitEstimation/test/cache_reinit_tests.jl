# reinit! with a new algorithm or rng (owner ruling 2026-11-26, SciML's cache pattern): an algorithm of the cache's
# concrete type re-resolves its scorer, strategy and config against the kept workspace — the same run as a fresh init
# with it —, an rng of the cache's type replaces the find's, another type is refused before anything changes, and a
# result aliases the cache's buffers until the next solve!.

@testitem "reinit!: a new algorithm or rng on a Ransac and a LocalOptimization cache" begin
    using Test, StaticArrays, StableRNGs, Random
    using VisualGeometryToolkitCore: solve, init, solve!, reinit!, EstimationProblem, RefineProblem, HomographyMat,
        MsacScoring, Irls, TLSShape, DLT, weights
    using VisualGeometryToolkitEstimation: Ransac, LocalOptimization, HomographyProblem, RansacConfig

    rng = StableRNG(5)
    H = @SMatrix [1.1 0.02 3.0; -0.01 0.95 -2.0; 1.0e-4 2.0e-5 1.0]
    src = [SVector(640rand(rng), 480rand(rng)) for _ in 1:60]
    dst = [(q = H * SVector(p[1], p[2], 1.0); SVector(q[1] / q[3], q[2] / q[3]) .+ 0.3 .* randn(rng, SVector{2})) for p in src]
    dst[1:15] .= [SVector(640rand(rng), 480rand(rng)) for _ in 1:15]
    f = HomographyProblem(src .=> dst)
    prob = EstimationProblem(f)
    find(t) = Ransac(; scoring = MsacScoring(f; threshold = t), local_optimization_strategy = Irls{TLSShape}())
    same(a, b) = a.u == b.u && a.resid == b.resid && weights(a) == weights(b) && a.stats == b.stats &&
                 a.retcode === b.retcode && a.stop_rule === b.stop_rule

    @testset "Ransac: an algorithm swap and an rng swap are a fresh init's runs" begin
        c = init(prob, find(2.0); rng = StableRNG(1))
        solve!(c)
        ws = c.cache.ws
        reinit!(c; alg = find(4.0), rng = StableRNG(9))
        r = solve!(c)
        @test c.cache.ws === ws                                            # the workspace is kept
        @test same(r, solve!(init(prob, find(4.0); rng = StableRNG(9)))) && r.alg === find(4.0) && c.alg === find(4.0)
        reinit!(c; rng = StableRNG(13))                                    # the rng alone
        @test same(solve!(c), solve(prob, find(4.0); rng = StableRNG(13)))
        cfg = Ransac(; scoring = MsacScoring(f; threshold = 4.0), local_optimization_strategy = Irls{TLSShape}(),
                     config = RansacConfig(; max_trials = 50))
        reinit!(c; alg = cfg, rng = StableRNG(2))                          # another config, the same type
        @test same(solve!(c), solve(prob, cfg; rng = StableRNG(2)))
    end

    @testset "refusals come before anything changes" begin
        c = init(prob, find(2.0); rng = StableRNG(1))
        alg0, rng0, prob0 = c.alg, c.cache.rng, c.prob
        @test_throws ArgumentError reinit!(c; alg = Ransac(; scoring = MsacScoring(f; threshold = 2.0)))   # another type
        @test_throws ArgumentError reinit!(c; alg = Ransac(Float64; scoring = MsacScoring(f; threshold = 2.0),
                                                           local_optimization_strategy = Irls{TLSShape}()))  # another storage type
        @test_throws ArgumentError reinit!(c; rng = Xoshiro(1))            # an rng of another type
        @test c.alg === alg0 && c.cache.alg === alg0 && c.cache.rng === rng0 && c.prob === prob0
        H0 = solve(EstimationProblem(HomographyMat, src[16:end] .=> dst[16:end]), DLT()).u
        lc = init(RefineProblem(f, H0), LocalOptimization(Irls{TLSShape}(); scoring = MsacScoring(f; threshold = 2.0)))
        @test_throws ArgumentError reinit!(lc; rng = StableRNG(1))         # a local optimization draws nothing
    end

    @testset "LocalOptimization: a new seed and a new scorer width per call" begin
        H0 = solve(EstimationProblem(HomographyMat, src[16:end] .=> dst[16:end]), DLT()).u
        H1 = solve(EstimationProblem(HomographyMat, src[30:end] .=> dst[30:end]), DLT()).u
        lo(t) = LocalOptimization(Irls{TLSShape}(); scoring = MsacScoring(f; threshold = t))
        c = init(RefineProblem(f, H0), lo(2.0))
        solve!(c)
        ws = c.cache.ws
        for (t, θ) in ((3.0, H0), (1.5, H1), (6.0, H0))
            reinit!(c, θ; alg = lo(t))
            @test same(solve!(c), solve(RefineProblem(f, θ), lo(t))) && c.cache.ws === ws
        end
    end

    @testset "a result aliases the cache's buffers until the next solve!" begin
        c = init(prob, find(2.0); rng = StableRNG(1))
        r1 = solve!(c)
        kept = copy(r1.resid)                                              # what a caller that keeps a result does
        reinit!(c; f = HomographyProblem(reverse(src) .=> reverse(dst)), alg = find(6.0), rng = StableRNG(4))
        r2 = solve!(c)
        @test r1.resid === r2.resid                                        # the same buffer, overwritten
        @test r1.resid != kept && kept == solve(prob, find(2.0); rng = StableRNG(1)).resid
    end
end

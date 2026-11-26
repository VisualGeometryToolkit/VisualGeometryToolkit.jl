@testitem "ScoreSelection: the best-scoring stored model of a pool, one-shot and streamed over a budget" begin
    using Test, StableRNGs
    using VisualGeometryToolkitCore: solve, init, solve!, reinit!, EstimationProblem, MsacScoring, ReturnCode, ClosedForm,
        NoRule, ClosedFormKind, estimator_kind
    using VisualGeometryToolkitEstimation: ScoreSelection, PoolSelection
    import VisualGeometryToolkitEstimation as VgtkEstimation

    rng = StableRNG(11)
    n, M = 60, 40
    D = abs.(randn(rng, n, M)) .* (1.0 .+ 4.0 .* rand(rng, 1, M))        # columns of widely different spread
    sc = MsacScoring{1, Float64}(1.5)
    @test estimator_kind(ScoreSelection(sc)) === ClosedFormKind()

    r = solve(EstimationProblem(PoolSelection, D), ScoreSelection(sc))
    @test r.retcode === ReturnCode.Success && r.stop_rule === ClosedForm && r.n == n && r.stats.nsteps == M
    @test r.u === VgtkEstimation._msac_best_over(D, 1:n, 1.5)              # SimFitPP's own argmax: one rule
    rows = 1:2:n
    @test solve(EstimationProblem(PoolSelection, D[rows, :]), ScoreSelection(sc)).u === VgtkEstimation._msac_best_over(D, rows, 1.5)

    # streamed: each budget's selection is the one-shot selection over its first models, scoring only the new ones
    c = init(EstimationProblem(PoolSelection, D, (; budget = 0)), ScoreSelection(sc))
    for k in (1, 5, 17, 17, 33, M)
        rk = solve!(reinit!(c; p = (; budget = k)))
        @test rk.u === solve(EstimationProblem(PoolSelection, D[:, 1:k]), ScoreSelection(sc)).u
        @test rk.stats.nsteps == k
    end
    # a smaller budget starts over
    @test solve!(reinit!(c; p = (; budget = 3))).u === solve(EstimationProblem(PoolSelection, D[:, 1:3]), ScoreSelection(sc)).u
    # the continuation allocates nothing once warm
    allocs(c, M) = @allocations solve!(reinit!(c; p = (; budget = M)))   # a function barrier: no global in the call
    allocs(c, M)
    @test allocs(c, M) == 0

    # no model scores above −Inf (here: no model): index 0, a failure
    rf = solve(EstimationProblem(PoolSelection, zeros(n, 0)), ScoreSelection(sc))
    @test rf.u == 0 && rf.retcode === ReturnCode.Failure && rf.stop_rule === NoRule
end

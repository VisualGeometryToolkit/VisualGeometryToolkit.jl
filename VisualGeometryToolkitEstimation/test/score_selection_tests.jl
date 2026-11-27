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

# The batched grid form of ScoreSelection (Eval V3): a grid family of another package selects, the ensemble records
@testitem "ScoreSelection over a SelectionGrid: the (pool, width) selections of a staged batch, one ensemble solve" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, EnsembleProblem, EnsembleKernel, ReturnCode, ClosedForm, NoRule, default_device
    using VisualGeometryToolkitEstimation: ScoreSelection, SelectionGrid, HomographyProblem
    cpu = default_device()
    # a grid family of another package: its own select_on_grid!, recording what it was handed
    struct _GridFamily
        seen::Vector{Any}
    end
    function VgtkCore.select_on_grid!(idxs::AbstractMatrix{Int32}, P::Type, sc::_GridFamily, grid::AbstractVector, staged; kw...)
        push!(sc.seen, (P, collect(grid), staged, values(kw)))
        for t in axes(idxs, 2), p in axes(idxs, 1)
            idxs[p, t] = isodd(p + t) ? Int32(0) : Int32(10p + t)
        end
        return idxs
    end
    fam = _GridFamily(Any[])
    staged = (; pairs = 1:3, residuals = nothing, backend = cpu)
    grid = [0.5, 1.0, 2.0, 4.0]
    rng_of(fingerprint, t) = (fingerprint, t)
    r = solve(EnsembleProblem(SelectionGrid(HomographyProblem, grid, staged; rng_of), 3 * 4), ScoreSelection(fam),
              EnsembleKernel(cpu))
    idxs = reshape(r.u, 3, 4)
    @test idxs == [isodd(p + t) ? 0 : 10p + t for p in 1:3, t in 1:4] && eltype(r.u) === Int32
    @test all(r.retcode[k] === (r.u[k] == 0 ? ReturnCode.Failure : ReturnCode.Success) for k in eachindex(r.u))
    @test all(r.stop_rule[k] === (r.u[k] == 0 ? NoRule : ClosedForm) for k in eachindex(r.u))
    @test all(==(0), r.nsteps) && all(isnan, r.rms)
    P, g, st, kw = only(fam.seen)
    @test P === HomographyProblem && g == grid && st === staged && kw === (; rng_of)   # the family's own keyword forwarded
    # without `rng_of` none is forwarded; the cells are whole pools; the ensemble runs on the backend of the staging
    solve(EnsembleProblem(SelectionGrid(HomographyProblem, grid, staged), 12), ScoreSelection(fam), EnsembleKernel(cpu))
    @test last(fam.seen)[4] === (;)
    @test_throws DimensionMismatch solve(EnsembleProblem(SelectionGrid(HomographyProblem, grid, staged), 10), ScoreSelection(fam),
                                         EnsembleKernel(cpu))
    @test_throws ArgumentError solve(EnsembleProblem(SelectionGrid(HomographyProblem, grid, staged), 12), ScoreSelection(fam),
                                     EnsembleKernel(:another_device))
end

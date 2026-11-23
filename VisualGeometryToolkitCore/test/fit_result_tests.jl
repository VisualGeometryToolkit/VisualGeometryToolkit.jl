# =============================================================================
# FitResult — the solution of every estimation problem (Core 0.14): read by field, no forwarding
# =============================================================================

@testitem "fit_result" begin
    using VisualGeometryToolkitCore: FitResult, SolveStats, BundleMetadata, weights, dof, successful_retcode,
        retcode_symbol, found_model, with_problem, ReturnCode, CostDecrease, IterationCap, NoRule
    import VisualGeometryToolkitCore as VgtkCore

    # An IRLS-style metadata (a spread, ν, per-point weights): its own fields are read as `r.metadata.<field>`.
    struct _IrlsMeta
        spread::Float64
        ν::Float64
        weights::Vector{Float64}
    end

    @testset "the fields, by field" begin
        lean = FitResult(3.14, ReturnCode.Success, CostDecrease, SolveStats(; nsteps = 7), 0.21, 100, nothing)
        @test fieldnames(FitResult) ==
              (:u, :resid, :retcode, :stats, :prob, :alg, :original, :stop_rule, :rms, :n, :metadata)
        @test lean.u === 3.14 && lean.retcode === ReturnCode.Success && lean.stop_rule === CostDecrease
        @test lean.stats === SolveStats(7, 0, 0, 0, 0) && lean.stats.nsteps == 7
        @test lean.rms === 0.21 && lean.n === 100 && lean.metadata === nothing
        # SciML's fields a solver does not set: nothing until the front end records them
        @test lean.resid === nothing && lean.prob === nothing && lean.alg === nothing && lean.original === nothing
        @test @inferred(successful_retcode(lean))
        @test @inferred(retcode_symbol(lean)) === :converged
        @test found_model(lean)
        # A lean result has no weights — an ERROR, never a silent `nothing`.
        @test_throws Exception weights(lean)

        ir = FitResult([1.0, 2.0], ReturnCode.MaxIters, IterationCap, SolveStats(; nsteps = 50), 0.5, 3,
                       _IrlsMeta(1.2, 8.0, [0.9, 1.0, 0.1]); resid = [0.1, 0.2, 3.0])
        @test !successful_retcode(ir) && retcode_symbol(ir) === :max_iterations
        @test @inferred(weights(ir)) == [0.9, 1.0, 0.1]
        @test ir.metadata.spread == 1.2 && ir.metadata.ν == 8.0
        @test ir.resid == [0.1, 0.2, 3.0]

        bu = FitResult(
            :cal, ReturnCode.Success, CostDecrease, SolveStats(; nsteps = 12), 0.2, 500, BundleMetadata(
                [1.0], [0.4], [1, 3], 488, 1.1,
                (rms = 0.31, median = 0.25, p90 = 0.62), (rms = 0.2, median = 0.18, p90 = 0.4);
                scale = 0.7, scale_source = VgtkCore.ScaleGiven
            )
        )
        # the bundle's scale record: read by noise_scale/scale_source, required (no default)
        @test VgtkCore.noise_scale(bu) === 0.7 && VgtkCore.scale_source(bu) === VgtkCore.ScaleGiven
        @test_throws UndefKeywordError BundleMetadata([1.0], [0.4], [1], 1, NaN, bu.metadata.before, bu.metadata.after)
        @test @inferred(dof(bu)) == 488
        @test bu.metadata.after.median == 0.18 && bu.metadata.before.p90 == 0.62
        @test bu.rms == bu.metadata.after.rms == 0.2
        # one residual per weight — one per observation
        @test_throws ArgumentError BundleMetadata([1.0], [0.4, 0.5], [1], 1, NaN, bu.metadata.before, bu.metadata.after;
                                                  scale = NaN, scale_source = VgtkCore.ScaleNone)
        # concrete types all the way down — no Union fields anywhere
        @test isconcretetype(typeof(ir.metadata)) && isconcretetype(typeof(bu.metadata))
        @test isconcretetype(typeof(ir)) && isconcretetype(typeof(lean))
        # type stability of a field read
        _u(r) = r.u; _m(r) = r.metadata.ν
        @test @inferred(_u(ir)) == [1.0, 2.0] && @inferred(_m(ir)) == 8.0
    end

    @testset "no property is forwarded into the metadata" begin
        ir = FitResult([1.0], ReturnCode.Success, CostDecrease, SolveStats(), 0.5, 3, _IrlsMeta(1.2, 8.0, [1.0, 1.0, 1.0]))
        @test propertynames(ir) == fieldnames(FitResult)
        # a metadata field is not a property of the result: one way per quantity
        @test_throws Exception ir.weights
        @test_throws Exception ir.spread
        @test_throws Exception ir.ν
        # nor are the retired spellings
        @test_throws Exception ir.estimate
        @test_throws Exception ir.iterations
        @test_throws Exception ir.converged
        # a metadata field named like a result field is no clash: it is read through `r.metadata`
        struct _NamedLikeCore
            n::Int
        end
        r = FitResult(1.0, ReturnCode.Success, CostDecrease, SolveStats(), 0.0, 10, _NamedLikeCore(99))
        @test r.n == 10 && r.metadata.n == 99
    end

    @testset "SolveStats: SciML's NLStats names, immutable and isbits" begin
        st = SolveStats(; nsteps = 3, nf = 4, njacs = 5, nfactors = 6, nsolve = 7)
        @test fieldnames(SolveStats) == (:nsteps, :nf, :njacs, :nfactors, :nsolve)
        @test (st.nsteps, st.nf, st.njacs, st.nfactors, st.nsolve) == (3, 4, 5, 6, 7)
        @test SolveStats() === SolveStats(0, 0, 0, 0, 0)
        @test isbitstype(SolveStats) && !ismutabletype(SolveStats)
    end

    @testset "with_problem: the problem and the algorithm recorded, every other field as it was" begin
        r = FitResult(2.0, ReturnCode.Success, CostDecrease, SolveStats(; nsteps = 4), 0.3, 5, (; a = 1); resid = [0.1])
        prob = VgtkCore.EstimationProblem(Float64, [1.0, 2.0])
        alg = VgtkCore.ChordalMean()
        w = with_problem(r, prob, alg)
        @test w.prob === prob && w.alg === alg
        @test w.u === r.u && w.resid === r.resid && w.retcode === r.retcode && w.stats === r.stats &&
              w.original === r.original && w.stop_rule === r.stop_rule && w.rms === r.rms && w.n === r.n &&
              w.metadata === r.metadata
        @test w isa FitResult{Float64, @NamedTuple{a::Int}}
    end

    @testset "an isbits result: a device solve returns it" begin
        r = FitResult(1.0f0, ReturnCode.Success, CostDecrease, SolveStats(; nsteps = 2), 0.0, 2, (; J = 1.0f0))
        @test isbits(r)
        @test isbits(with_problem(r, VgtkCore.RefineProblem(abs, 1.0f0), VgtkCore.StaticLevenbergMarquardt()))
    end
end

# The one RMS computation (base/fit_result.jl): over n points, a point's stacked rows its squared norm.
@testitem "rms(residuals, n): the fits' root-mean-square residual" begin
    using Test
    using VisualGeometryToolkitCore: rms
    e = [3.0, 4.0, 12.0]
    @test rms(e) === sqrt((9.0 + 16.0 + 144.0) / 3)
    rows = [3.0, 4.0, 6.0, 8.0]                     # two points' (x, y) rows: norms 5 and 10
    @test rms(rows, 2) === sqrt(sum(abs2, rows) / 2) && rms(rows, 2) ≈ sqrt((25 + 100) / 2)
end

# A RANSAC result's `u` is `Union{M, Nothing}`: `map_estimate` infers `f`'s image of `M` (not of the union, which
# a generic `f` with no `nothing` method widens to `Any`), present or absent.
@testitem "map_estimate: the union's present type mapped, inferred" begin
    using Test
    using VisualGeometryToolkitCore: FitResult, SolveStats, map_estimate, ReturnCode, NoRule, CostDecrease
    double(x::Float64) = 2x
    double(x) = Base.inferencebarrier(x)                   # a generic fallback, uninferable: what `nothing` would reach
    U = Union{Float64, Nothing}
    present = FitResult{U, Nothing, Nothing, Nothing, Nothing, Nothing}(1.5, nothing, ReturnCode.Success, SolveStats(; nsteps = 3),
        nothing, nothing, nothing, CostDecrease, 0.1, 10, nothing)
    absent = FitResult{U, Nothing, Nothing, Nothing, Nothing, Nothing}(nothing, nothing, ReturnCode.Failure, SolveStats(),
        nothing, nothing, nothing, NoRule, NaN, 10, nothing)
    @test map_estimate(double, present) isa FitResult{Union{Float64, Nothing}, Nothing}
    @test map_estimate(double, present).u == 3.0
    @test map_estimate(double, present).stats === present.stats
    @test map_estimate(double, absent).u === nothing && map_estimate(double, absent) isa FitResult{Union{Float64, Nothing}}
    @test map_estimate(string, present) isa FitResult{Union{String, Nothing}, Nothing}
    # a concrete `u` is mapped once and stays concrete
    calls = Ref(0)
    once(x) = (calls[] += 1; 2x)
    c = map_estimate(once, FitResult(1.0, ReturnCode.Success, CostDecrease, SolveStats(), 0.0, 1, nothing))
    @test c.u === 2.0 && calls[] == 1 && c isa FitResult{Float64, Nothing}
end

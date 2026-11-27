# The generic block-coordinate-descent loop had no test in this package — its only
# callers were VisualGeometryToolkitEstimation's problems. A weighted mean as a one-parameter IRLS
# pins the block order (model → residuals → nuisance → weights → check) and Julia Base's argument
# order: the mutated inputs first (state, then likelihood), then the inputs the loop
# only reads (method, problem, seed). The old method-first order is refused. Users solve through
# `BlockCoordinateDescent`; these tests reach the internal loop for a supplied (recorded) state.
@testitem "block_coordinate_descent" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: AbstractBlockCoordinateMethod, AbstractWeightPolicy, _block_coordinate_descent!, init_state!,
        StopAfterIteration

    struct MeanIRLS <: AbstractBlockCoordinateMethod end
    mutable struct CountingPolicy <: AbstractWeightPolicy
        nuisance_updates::Int
        weight_updates::Int
    end
    # The method's own criterion: the mean stopped moving.
    struct StopWhenMeanSettled <: VgtkCore.StoppingCriterion end
    VgtkCore.stop_reason(::StopWhenMeanSettled, state, ::MeanIRLS, data, θ, θ_old, iter) =
        abs(θ - θ_old) < 1.0e-12 ? (VgtkCore.ReturnCode.Success, VgtkCore.ParameterChange) : VgtkCore.CONTINUE
    VgtkCore.update_nuisance!(p::CountingPolicy) = (p.nuisance_updates += 1; p)
    VgtkCore.update_weights!(p::CountingPolicy) = (p.weight_updates += 1; p)
    VgtkCore.init_state!(::MeanIRLS, data, θ₀) = (; r = similar(data))
    VgtkCore.update_model!(state, ::MeanIRLS, data, θ) = sum(data) / length(data)
    VgtkCore.update_residuals!(state, ::MeanIRLS, data, θ) = (state.r .= data .- θ; state)
    VgtkCore.stopping_criterion(::MeanIRLS) = StopWhenMeanSettled()
    VgtkCore.block_coordinate_result(state, ::MeanIRLS, data, θ, code, iter, rule) = (; θ, code, iter, rule)

    data = [1.0, 2.0, 6.0]
    policy = CountingPolicy(0, 0)
    res = _block_coordinate_descent!(init_state!(MeanIRLS(), data, 0.0), policy, MeanIRLS(), data, 0.0, StopAfterIteration(10))
    @test res.θ ≈ 3.0
    @test res.code === VgtkCore.ReturnCode.Success && res.rule === VgtkCore.ParameterChange
    @test policy.nuisance_updates == res.iter == policy.weight_updates

    state = init_state!(MeanIRLS(), data, 0.0)
    @test _block_coordinate_descent!(state, CountingPolicy(0, 0), MeanIRLS(), data, 0.0, StopAfterIteration(10)).θ ≈ 3.0
    @test state.r ≈ data .- 3.0

    @test_throws MethodError _block_coordinate_descent!(MeanIRLS(), data, CountingPolicy(0, 0), state, 0.0, StopAfterIteration(10))
    # the front end: the same run, its state built by `init_state!`
    @test VgtkCore.solve(VgtkCore.RefineProblem(data, 0.0), VgtkCore.BlockCoordinateDescent(MeanIRLS(); likelihood = CountingPolicy(0, 0));
                         stops = StopAfterIteration(10)).θ ≈ 3.0

    # The nuisance half of the pair is named for its ROLE, not for ŝ²: across the
    # implementing policies it moves nothing, ŝ², (ν, ŝ²), (ŝ², π), or the Barron
    # shape α alone. `update_scale!` was accurate on one of those, and is GONE as
    # a policy verb — a hard rename, with no deprecation shim to keep the false
    # name alive. The name now belongs to the posterior-weight state's ŝ² block
    # only (robust/posterior.jl), which does exactly what it says.
    @test isdefined(VisualGeometryToolkitCore, :update_nuisance!)
    @test all(m -> m.sig.parameters[2] === VgtkCore.PosteriorWeightState,
              methods(VgtkCore.update_scale!))
end

# The stopping criteria (Manopt.jl's names, immutable, composed with `|`) and
# the order the driver reads them in: a failed model block first, then the
# method's own criterion, then the budget — so on the last iteration a method
# record wins over (MaxIters, IterationCap), and a non-positive budget runs nothing.
# A halving iteration θ ← θ/2 that fails at `fail_at` and converges at `conv_at`,
# its result the bare `(; θ, reason, iter)` or, with `fit = true`, a FitResult.
@testsetup module HalvingIteration
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: AbstractBlockCoordinateMethod, AbstractWeightPolicy, StoppingCriterion, FitResult
    export Halve, HalveState, NoPolicy, StopAtIteration

    struct Halve{Fit} <: AbstractBlockCoordinateMethod
        fail_at::Int
        conv_at::Int
    end
    Halve(fail_at, conv_at; fit = false) = Halve{fit}(fail_at, conv_at)
    mutable struct HalveState
        iter::Int
        cost::Float64
    end
    struct NoPolicy <: AbstractWeightPolicy end
    VgtkCore.update_nuisance!(p::NoPolicy) = p
    VgtkCore.update_weights!(p::NoPolicy) = p
    VgtkCore.current_s2(::NoPolicy) = 4.0
    VgtkCore.update_model!(s::HalveState, m::Halve, prob, θ) = s.iter + 1 == m.fail_at ? nothing : θ / 2
    VgtkCore.update_residuals!(s::HalveState, ::Halve, prob, θ) = (s.cost = θ^2; s)
    VgtkCore.post_iteration!(s::HalveState, ::Halve, prob, θ, iter) = (s.iter = iter; nothing)
    VgtkCore.current_cost(s::HalveState) = s.cost
    struct StopAtIteration <: StoppingCriterion
        k::Int
    end
    VgtkCore.stop_reason(c::StopAtIteration, s::HalveState, m, prob, θ, θ_old, iter) =
        s.iter >= c.k ? (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease) : VgtkCore.CONTINUE
    VgtkCore.stopping_criterion(m::Halve) = StopAtIteration(m.conv_at)
    VgtkCore.block_coordinate_result(s::HalveState, ::Halve{false}, prob, θ, code, iter, rule) = (; θ, code, iter, rule)
    VgtkCore.block_coordinate_result(s::HalveState, ::Halve{true}, prob, θ, code, iter, rule) =
        FitResult(θ, code, rule, VgtkCore.SolveStats(; nsteps = iter), sqrt(s.cost), 1, (;))
end

@testitem "block_coordinate_descent: stopping criteria and the recorder" setup = [HalvingIteration] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: AbstractBlockCoordinateMethod, _block_coordinate_descent!,
        StopAfterIteration, StopWhenSolveFailed, StopNever, StopWhenAny, stop_reason,
        stopping_criterion, Recorded, current_cost, current_s2

    run(m, max_iter) = _block_coordinate_descent!(HalveState(0, NaN), NoPolicy(), m, nothing, 1.0, StopAfterIteration(max_iter))
    # the record of the member that fired reaches the result (StopAtIteration's: Success, CostDecrease, the fixture's)
    RC = VgtkCore.ReturnCode
    @test run(Halve(0, 100), 5) == (; θ = 1 / 32, code = RC.MaxIters, iter = 5, rule = VgtkCore.IterationCap)
    @test run(Halve(0, 3), 5) == (; θ = 1 / 8, code = RC.Success, iter = 3, rule = VgtkCore.CostDecrease)
    @test run(Halve(0, 5), 5) == (; θ = 1 / 32, code = RC.Success, iter = 5, rule = VgtkCore.CostDecrease)   # the method wins at the budget
    @test run(Halve(3, 100), 5) == (; θ = 1 / 4, code = RC.Failure, iter = 3, rule = VgtkCore.NonFinite)  # the last good model
    @test run(Halve(5, 5), 5).code === RC.Failure                                    # a failed solve wins
    @test run(Halve(1, 1), 0) == (; θ = 1.0, code = RC.MaxIters, iter = 0, rule = VgtkCore.IterationCap)
    @test run(Halve(1, 1), -2) == (; θ = 1.0, code = RC.MaxIters, iter = 0, rule = VgtkCore.IterationCap)
    @test stop_reason(StopNever(), HalveState(0, NaN), Halve(0, 0), nothing, 1.0, 1.0, 1) === VgtkCore.CONTINUE   # the default

    # composition: flat, typed by its tuple, first reason wins
    c = StopWhenSolveFailed() | StopNever() | StopAfterIteration(3)
    @test c isa StopWhenAny{Tuple{StopWhenSolveFailed, StopNever, StopAfterIteration}}
    @test (c | c).criteria == (c.criteria..., c.criteria...)
    @test isbits(c)
    s = HalveState(0, NaN)
    @test stop_reason(c, s, Halve(0, 0), nothing, nothing, 1.0, 1) === (RC.Failure, VgtkCore.NonFinite)
    @test stop_reason(c, s, Halve(0, 0), nothing, 0.5, 1.0, 1) === VgtkCore.CONTINUE
    @test stop_reason(c, s, Halve(0, 0), nothing, 0.5, 1.0, 3) === (RC.MaxIters, VgtkCore.IterationCap)
    @test stopping_criterion(Halve(0, 7)) === StopAtIteration(7)
    struct Plain <: AbstractBlockCoordinateMethod end
    @test stopping_criterion(Plain()) === StopNever()                # the default: budget and failed solve only

    # the recorder: the same run, iterate for iterate, plus the trace
    for (m, mi) in ((Halve(0, 100), 5), (Halve(0, 3), 5), (Halve(3, 100), 5), (Halve(1, 1), 0))
        bare = run(m, mi)
        rec = Recorded(HalveState(0, NaN))
        @test _block_coordinate_descent!(rec, NoPolicy(), m, nothing, 1.0, StopAfterIteration(mi)) == bare
        k = bare.code === RC.Failure ? bare.iter - 1 : bare.iter
        @test rec.iterates == [0.5^i for i in 1:k]
        @test rec.cost == [0.25^i for i in 1:k]
        @test rec.s2 == fill(4.0, k)
        @test rec.reason[] === (bare.code, bare.rule)
    end
    @test current_cost(:anything) === NaN && current_s2(:anything) === NaN

    # the bare path allocates nothing per iteration
    m = Halve(0, 100); st = HalveState(0, NaN)
    _block_coordinate_descent!(st, NoPolicy(), m, nothing, 1.0, StopAfterIteration(50))
    alloc(st, m, k) = @allocated _block_coordinate_descent!(st, NoPolicy(), m, nothing, 1.0, StopAfterIteration(k))
    alloc(st, m, 50)
    @test alloc(st, m, 50) == 0
end

# The stack's one check that a `Recorded` run IS the bare run (LocalOptimizationDoneRight's
# recorded-BCD test and VisualGeometryToolkitEstimation's restated it as `check_recorded`).
@testitem "block_coordinate_descent: test_recorded" setup = [HalvingIteration] begin
    using Test
    using VisualGeometryToolkitCore: test_recorded
    make(fail_at, conv_at, max_iter) = () -> (HalveState(0, NaN), NoPolicy(), Halve(fail_at, conv_at; fit = true), nothing, 1.0, max_iter)
    @test test_recorded(make(0, 100, 5)) == 5          # the budget
    @test test_recorded(make(0, 3, 5)) == 3            # converged
    @test test_recorded(make(3, 100, 5)) == 2          # a failed solve records the good iterates only
end

@testitem "StopWhenIncrementLess: the increment relative to the new model, strict, (Success, IncrementSize); isbits, in a kernel" begin
    using Test, StaticArrays
    using Random: Xoshiro, randn
    import KernelAbstractions as KA
    using KernelAbstractions: @kernel, @index, @Const
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: StopWhenIncrementLess, StoppingCriterion, stop_reason, CONTINUE, ReturnCode, IncrementSize,
        StopAfterIteration, IterationCap
    c = StopWhenIncrementLess(1.0e-6)
    @test c isa StoppingCriterion && isbits(c) && isbits(StopWhenIncrementLess(1.0f-6))
    @test Base.ispublic(VisualGeometryToolkitCore, :StopWhenIncrementLess)
    reason(c, θ, θ_old) = stop_reason(c, nothing, nothing, nothing, θ, θ_old, 1)
    # the reweighted fits' rule, written out: the increment's norm over the NEW model's norm (+1e-300), strictly below rtol
    function reference(θ, θ_old, rtol)
        T = eltype(θ); diff_sq = zero(T); norm_sq = zero(T)
        for j in 1:length(θ)
            d = θ[j] - θ_old[j]
            diff_sq += d * d
            norm_sq += θ[j] * θ[j]
        end
        return sqrt(diff_sq) / (sqrt(norm_sq) + T(1.0e-300)) < rtol
    end
    rng = Xoshiro(21)
    for T in (Float64, Float32), _ in 1:200
        θ_old = SVector{6, T}(randn(rng, T, 6))
        θ = θ_old .+ T(10.0^(-rand(rng, 2:9))) .* SVector{6, T}(randn(rng, T, 6))
        rtol = T(10.0^(-rand(rng, 2:9)))
        @test reason(StopWhenIncrementLess(rtol), θ, θ_old) === (reference(θ, θ_old, rtol) ? (ReturnCode.Success, IncrementSize) : CONTINUE)
    end
    # relative to the new model, not the old: the same increment converges against a large θ, not a small one
    @test reason(c, SVector(1.0e3, 0.0), SVector(1.0e3 - 1.0e-4, 0.0)) === (ReturnCode.Success, IncrementSize)
    @test reason(c, SVector(1.0e-3, 0.0), SVector(1.0e-3 - 1.0e-4, 0.0)) === CONTINUE
    # strict: a ratio equal to rtol continues; no move at all converges, the zero model included (the 1e-300 guard)
    @test reason(StopWhenIncrementLess(0.5), SVector(2.0, 0.0), SVector(1.0, 0.0)) === CONTINUE
    @test reason(c, SVector(0.0, 0.0), SVector(0.0, 0.0)) === (ReturnCode.Success, IncrementSize)
    # a failed model block is StopWhenSolveFailed's; a composition reads it in order
    @test reason(c, nothing, SVector(1.0, 2.0)) === CONTINUE
    @test stop_reason(StopAfterIteration(1) | c, nothing, nothing, nothing, SVector(1.0), SVector(0.0), 1) === (ReturnCode.MaxIters, IterationCap)
    @test stop_reason(c | StopAfterIteration(1), nothing, nothing, nothing, SVector(1.0), SVector(1.0), 1) === (ReturnCode.Success, IncrementSize)

    # device-evaluable: a KernelAbstractions kernel judges each column of a parameter matrix, as a per-blob IRLS does
    @kernel function judged!(out, @Const(θ), @Const(θ_old), c, m)
        k = @index(Global)
        rr = stop_reason(c, nothing, nothing, nothing, view(θ, 1:m, k), view(θ_old, 1:m, k), 1)
        @inbounds out[k] = first(rr) === ReturnCode.Success
    end
    n = 257
    θ_old = randn(rng, Float32, 6, n)
    θ = θ_old .+ Float32.(10.0 .^ (-rand(rng, 2:9, 1, n))) .* randn(rng, Float32, 6, n)
    cf = StopWhenIncrementLess(1.0f-5)
    out = zeros(Bool, n)
    judged!(KA.CPU())(out, θ, θ_old, cf, 5; ndrange = n); KA.synchronize(KA.CPU())
    @test out == [reference(view(θ, 1:5, k), view(θ_old, 1:5, k), 1.0f-5) for k in 1:n]
    @test 0 < count(out) < n
end

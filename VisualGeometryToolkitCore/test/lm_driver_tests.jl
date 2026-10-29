@testitem "lm_driver" begin
# The chart LM's driver, NLLSsolver (src/robust/lm_core.jl): the fork's
# additions are detected, and the upstream path a registered NLLSsolver takes
# gives the fast path's results (the fallback test); NLLSsolver's termination
# bits decode to the chart LM's stop codes; a NaN trial is a rejected step.
import VisualGeometryToolkitCore as VgtkCore
using Test, LinearAlgebra, StaticArrays
using VisualGeometryToolkitCore: ManifoldLmWorkspace, lmsolve_chart!, lm_retcode, nlls_features, nlls_options
import NLLSsolver

# A 4-parameter nonlinear fit with a closed-form Jacobian: y = a·exp(b·s) + c·s + d.
struct _ExpFitLm
    s::Vector{Float64}
    y::Vector{Float64}
    p0::SVector{4, Float64}
end
function VgtkCore.residual_rows!(r, lm::_ExpFitLm, x)
    a, b, c, d = lm.p0 + x
    for i in eachindex(lm.s)
        r[i] = a * exp(b * lm.s[i]) + c * lm.s[i] + d - lm.y[i]
    end
    return r
end
function VgtkCore.residual_jacobian_rows!(J, lm::_ExpFitLm, x)
    a, b, c, d = lm.p0 + x
    for i in eachindex(lm.s)
        e = exp(b * lm.s[i])
        J[i, 1] = e; J[i, 2] = a * lm.s[i] * e; J[i, 3] = lm.s[i]; J[i, 4] = 1.0
    end
    return J
end

# The same rows behind a NaN wall at x[3] > 0.25.
struct _NaNBeyond{L}
    lm::L
end
VgtkCore.residual_rows!(r, w::_NaNBeyond, x) = (VgtkCore.residual_rows!(r, w.lm, x); x[3] > 0.25 ? fill!(r, NaN) : r)
VgtkCore.residual_jacobian_rows!(J, w::_NaNBeyond, x) = VgtkCore.residual_jacobian_rows!(J, w.lm, x)

@testset "the fork's additions, detected" begin
    f = nlls_features()
    # the shipped rule: the TYPED cache constructor, not the name alone (an
    # older fork with only the untyped `OptimizeCache()` takes the path
    # without a cache)
    @test f.cache === (isdefined(NLLSsolver, :OptimizeCache) &&
        hasmethod(NLLSsolver.OptimizeCache, Tuple{NLLSsolver.NLLSProblem, NLLSsolver.NLLSOptions, Any}))
    @test f.schur === (isdefined(NLLSsolver, :MultiVariateLSschur) && isdefined(NLLSsolver, :iseliminable))
    # NLLSsolver 5.3 has the cache (#62) but not the Schur system (#60, open);
    # our fork has both. Each is detected on its own; the comparisons below run
    # whichever paths the loaded NLLSsolver has.
end

@testset "the upstream path gives the fast path's results (fallback)" begin
    # `cache = false` is the path a registered NLLSsolver takes (no
    # OptimizeCache: the solver's internal data rebuilt per call). Same
    # iterates: measured 2026-10-27 on the fork, bit-identical x, F and
    # iteration counts on every fit below. On a registered NLLSsolver both
    # calls take that path (`cache` is then inert) and agree trivially.
    # ONE workspace serves the three problems in sequence, as the LO's does:
    # the second and third fits reuse the run (and its typed cache) the first
    # built, so nothing the cache carries from one fit may reach the next;
    # each is compared against a fresh workspace without a cache.
    s = collect(range(0.0, 2.0; length = 40))
    ws = ManifoldLmWorkspace(length(s), 4)
    run₁ = nothing
    for (k, ptrue) in enumerate((SVector(1.5, -0.8, 0.3, 0.2), SVector(0.4, 1.1, -0.5, 1.0), SVector(2.0, 0.2, 0.0, -1.0)))
        y = [ptrue[1] * exp(ptrue[2] * si) + ptrue[3] * si + ptrue[4] + 1.0e-3 * sin(7k * si) for si in s]
        lm = _ExpFitLm(s, y, SVector(1.0, 0.0, 0.0, 0.0))
        x1, F1, c1, it1 = lmsolve_chart!(ws, lm; stationarity_tol = 0.05, ftol = 1.0e-10)
        k == 1 ? (run₁ = ws.run; @test(run₁ !== nothing)) : @test(ws.run === run₁)   # the run is reused
        x2, F2, c2, it2 = lmsolve_chart!(ManifoldLmWorkspace(length(s), 4), lm; stationarity_tol = 0.05, ftol = 1.0e-10, cache = false)
        @test lm_retcode(c1) === :converged && lm_retcode(c2) === :converged
        @test isapprox(x1, x2; rtol = 1.0e-10, atol = 1.0e-12)
        @test isapprox(F1, F2; rtol = 1.0e-10)
        @test it1 == it2
        @test x1 == x2 && F1 == F2           # the measured bit identity (the same algorithm and data)
    end
end

# A function barrier: the testitem's locals are globals.
_refit!(ws, lm) = lmsolve_chart!(ws, lm; stationarity_tol = 0.05, ftol = 1.0e-10)
_refit_allocations(ws, lm) = @allocations _refit!(ws, lm)

@testset "a refit through the typed cache allocates nothing" begin
    # The fork's typed `OptimizeCache(problem, options, unfixed)` (#62 on
    # `stack`): measured 2026-10-27, 0 allocations per refit (the untyped cache
    # allocated 5). On a registered NLLSsolver the path without a cache
    # rebuilds the solver's internal data per refit: not asserted.
    s = collect(range(0.0, 2.0; length = 40))
    lm = _ExpFitLm(s, [1.5 * exp(-0.8 * si) + 0.3 * si + 0.2 for si in s], SVector(1.0, 0.0, 0.0, 0.0))
    ws = ManifoldLmWorkspace(length(s), 4)
    _refit!(ws, lm); _refit!(ws, lm)
    if nlls_features().cache
        @test _refit_allocations(ws, lm) == 0
    else
        @test_skip _refit_allocations(ws, lm) == 0
    end
end

@testset "the stop codes: one decoder, one order" begin
    # NLLSsolver's termination bits → `nlls_stop_code`, the one decoder the chart
    # LM and `nlls_retcode` read: failure without a step stop −3; step stop
    # (6) 2; cost-decrease stop (2, 3) 1; maxfails (7) 4; time cap −4;
    # iteration cap −1; anything else 4.
    code = VgtkCore.nlls_stop_code
    @test code(1 << 2) == 1 && code(1 << 3) == 1
    @test code(1 << 6) == 2
    @test code(1 << 8) == -1 && code(1 << 9) == -4 && code((1 << 8) | (1 << 9)) == -4
    @test code(1 << 7) == 4 && code(1 << 16) == 4
    for bit in (0, 1, 4, 5, 10)
        @test code(1 << bit) == -3
    end
    # the step stop is checked before the cost tolerance
    @test code((1 << 2) | (1 << 6)) == 2
    @test code((1 << 3) | (1 << 6)) == 2
    @test code((1 << 6) | (1 << 7)) == 2
    # maxfails outranks the caps (it is a stop of the run, the caps are budgets)
    @test code((1 << 7) | (1 << 8)) == 4 && code((1 << 7) | (1 << 9)) == 4
    # a tolerance outranks a cap; a non-finite trial at a step stop is a step stop
    @test code((1 << 8) | (1 << 2)) == 1 && code((1 << 8) | (1 << 6)) == 2
    @test code((1 << 0) | (1 << 6)) == 2 && code((1 << 1) | (1 << 6)) == 2
    @test code((1 << 1) | (1 << 8) | (1 << 9)) == -3
    @test lm_retcode(4) === :converged               # once the postcondition holds
    @test lm_retcode(-3) === :solve_failed && lm_retcode(-4) === :max_time
    # public, with the bits named
    @test Base.ispublic(VgtkCore, :nlls_stop_code) && Base.ispublic(VgtkCore, :nlls_stop_reasons)
    @test VgtkCore.nlls_stop_reasons(0) == Symbol[]
    @test VgtkCore.nlls_stop_reasons((1 << 2) | (1 << 8)) == [:relative_cost_decrease, :max_iterations]
    @test VgtkCore.nlls_stop_reasons((1 << 0) | (1 << 6) | (1 << 16)) == [:cost_infinite, :delta_size, :callback]
    @test length(VgtkCore.nlls_stop_reasons(typemax(Int32))) == 12                # every named bit (0–10, 16)
end

@testset "a NaN wall at the production cost tolerance" begin
    # The LO's ftol (1e-10, configs [chart_lm]): near the wall the last steps'
    # cost decrease and step size shrink together; the step stop is decoded
    # first, so the stall postcondition runs and the stop is :stalled.
    lm_wall = _NaNBeyond(_ExpFitLm([0.0, 1.0, 2.0, 3.0, 4.0], [1.0 + 2si for si in [0.0, 1.0, 2.0, 3.0, 4.0]], SVector(0.0, 0.0, 1.0, 1.0)))
    x, F, c, _ = lmsolve_chart!(ManifoldLmWorkspace(5, 4), lm_wall; stationarity_tol = 0.05, ftol = 1.0e-10)
    @test isfinite(F) && x[3] <= 0.25
    @test lm_retcode(c) === :stalled
end

@testset "the knob map" begin
    o = nlls_options(; max_iter = 7, ftol = 1.0e-9, xtol = 1.0e-11)
    @test o.maxiters == 7 && o.reldcost == 1.0e-9 && o.dstep == 1.0e-11 && o.absdcost == 0.0 && o.maxfails == 3
    @test NLLSsolver.dynamic(o.iterator) === :levenbergmarquardt_dtd
    @test NLLSsolver.dynamic(nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, iterator = :levenbergmarquardt).iterator) === :levenbergmarquardt
    @test o.maxtime == UInt64(1.0e6 * 1.0e9)                                     # no time cap by default
    @test nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, max_time = 2.5).maxtime == UInt64(2.5e9)
    @test typeof(nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, max_time = 2.5)) === typeof(o)
    # Inf is no cap, as `nothing` is; a negative or NaN cap is refused
    @test nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, max_time = Inf).maxtime == nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0).maxtime
    @test nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, max_time = 0).maxtime == 0
    @test_throws ArgumentError nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, max_time = -1.0)
    @test_throws ArgumentError nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, max_time = NaN)
end

@testset "a NaN trial is a rejected step" begin
    # NLLSsolver accepts a trial whose cost is not greater than the best, which
    # NaN is not; the adapter reads a non-finite cost as +Inf, so a NaN trial is
    # rejected.
    lm2 = _ExpFitLm([0.0, 1.0, 2.0, 3.0, 4.0], [1.0 + 2si for si in [0.0, 1.0, 2.0, 3.0, 4.0]], SVector(0.0, 0.0, 1.0, 1.0))   # optimum at c = 2: x[3] = 1, behind the wall
    x, F, c, _ = lmsolve_chart!(ManifoldLmWorkspace(5, 4), _NaNBeyond(lm2); stationarity_tol = 0.05)
    @test isfinite(F) && x[3] <= 0.25
end
end

@testitem "robust lmsolve_chart!: F is Σρ, the workspace keeps the rows at x" begin
    # The robustified chart LM (Triggs): its cost is Σᵢ ρ(‖rᵢ‖²) (NLLSsolver's
    # ½ρ doubled), and neither the solve nor its stall postcondition leaves the
    # workspace's rows scaled by √ρ′ — they are the rows at the returned x. A
    # redescending loss with every datum past its cutoff is a plateau, not a fit.
    import VisualGeometryToolkitCore as VgtkCore
    using Test, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: ManifoldLmWorkspace, lmsolve_chart!, loss_robustifier, WelschShape, TLSShape
    import NLLSsolver

    struct _RobustCircleLm
        pts::Vector{SVector{2, Float64}}
    end
    function VgtkCore.residual_rows!(f, lm::_RobustCircleLm, x)
        c = SVector(1.0 + x[1], -2.0 + x[2]); ρ = 3.0 + x[3]
        for (i, p) in enumerate(lm.pts)
            f[i] = norm(p - c) - ρ
        end
        return f
    end
    pts = [SVector(1.5 + 2.5cos(θ), -1.8 + 2.5sin(θ)) for θ in range(0, 2π; length = 13)[1:12]]
    append!(pts, [SVector(9.0, 9.0), SVector(-8.0, 7.0), SVector(6.0, -9.0)])   # three gross outliers
    lm = _RobustCircleLm(pts)
    n = length(pts)
    rows_at(x) = VgtkCore.residual_rows!(zeros(n), lm, x)
    k = loss_robustifier(WelschShape(), 0.1)
    ws = ManifoldLmWorkspace(n, 3)
    x, F, _, _ = lmsolve_chart!(ws, lm, k, Val(1); stationarity_tol = 0.05, ftol = 1.0e-12)
    @test x ≈ SVector(0.5, 0.2, -0.5) atol = 1.0e-6
    @test ws.r == rows_at(x)
    @test F ≈ sum(r -> NLLSsolver.robustify(k, r^2), rows_at(x)) rtol = 1.0e-10
    # the stall postcondition, taken directly: stationary, rows untouched
    S = VgtkCore._RobustChartState{3, typeof(lm), typeof(k)}
    state = S(lm, k, zero(SVector{3, Float64}), false, false, ws.r, ws.J)
    @test VgtkCore._robust_stationary(ws, state, x, Val(1), 0.05, 1.0e-12)
    @test ws.r == rows_at(x)
    @test !VgtkCore._robust_stationary(ws, state, x + SVector(0.05, 0.0, 0.0), Val(1), 0.05, 1.0e-12)
    @test ws.r == rows_at(x + SVector(0.05, 0.0, 0.0))
    # every datum past a TLS cutoff: ρ′ = 0 throughout, the exact-fit test reads the rows themselves
    kt = loss_robustifier(TLSShape(), 1.0e-3)
    St = VgtkCore._RobustChartState{3, typeof(lm), typeof(kt)}
    tstate = St(lm, kt, zero(SVector{3, Float64}), false, false, ws.r, ws.J)
    @test !VgtkCore._robust_stationary(ws, tstate, SVector(5.0, 5.0, 0.0), Val(1), 0.05, 1.0e-12)
end

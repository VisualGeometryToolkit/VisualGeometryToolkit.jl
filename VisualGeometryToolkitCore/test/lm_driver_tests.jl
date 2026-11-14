@testitem "lm_driver" begin
# The one LM adapter, NLLSsolver (src/robust/lm_core.jl): the open upstream
# addition (#60) is detected, a reused run gives a fresh workspace's results,
# NLLSsolver's termination bits decode to the chart LM's stop codes, the stop
# settings have one owner, a stop is a success only at a stationary point, and a
# NaN trial is a rejected step. (One of the adapter's own tests: the LM gate,
# check_duplication's `:lm_primitive`, lets this file reach its internals.)
import VisualGeometryToolkitCore as VgtkCore
using Test, LinearAlgebra, StaticArrays
using VisualGeometryToolkitCore: ManifoldLmWorkspace, lmsolve_chart!, retcode_symbol, nlls_features, LmStops
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

@testset "the open upstream addition, detected" begin
    # the Schur-complement system (upstream #60, open): detected by name; the
    # bundle's comparison runs whichever path the loaded NLLSsolver has
    @test nlls_features() === (; schur = isdefined(NLLSsolver, :MultiVariateLSschur) && isdefined(NLLSsolver, :iseliminable))
end

@testset "a reused run gives a fresh workspace's results" begin
    # ONE workspace serves the three problems in sequence, as the LO's does:
    # the second and third fits reuse the run (and its typed OptimizeCache)
    # the first built, so nothing the cache carries from one fit may reach the
    # next; each is compared against a fresh workspace (a new run and cache).
    s = collect(range(0.0, 2.0; length = 40))
    ws = ManifoldLmWorkspace(length(s), 4)
    run₁ = nothing
    for (k, ptrue) in enumerate((SVector(1.5, -0.8, 0.3, 0.2), SVector(0.4, 1.1, -0.5, 1.0), SVector(2.0, 0.2, 0.0, -1.0)))
        y = [ptrue[1] * exp(ptrue[2] * si) + ptrue[3] * si + ptrue[4] + 1.0e-3 * sin(7k * si) for si in s]
        lm = _ExpFitLm(s, y, SVector(1.0, 0.0, 0.0, 0.0))
        x1, F1, c1, it1, r1 = lmsolve_chart!(ws, lm)
        k == 1 ? (run₁ = ws.run; @test(run₁ !== nothing)) : @test(ws.run === run₁)   # the run is reused
        x2, F2, c2, it2, r2 = lmsolve_chart!(ManifoldLmWorkspace(length(s), 4), lm)
        @test retcode_symbol(r1...) === :converged && retcode_symbol(r2...) === :converged
        @test isapprox(x1, x2; rtol = 1.0e-10, atol = 1.0e-12)
        @test isapprox(F1, F2; rtol = 1.0e-10)
        @test it1 == it2
        @test x1 == x2 && F1 == F2           # the measured bit identity (the same algorithm and data)
    end
end

# A function barrier: the testitem's locals are globals.
_refit!(ws, lm) = lmsolve_chart!(ws, lm)
_refit_allocations(ws, lm) = @allocations _refit!(ws, lm)

@testset "a refit through the typed cache allocates nothing" begin
    # NLLSsolver's typed `OptimizeCache(problem, options, unfixed)` (upstream
    # #62, NLLSsolver ≥ 5.3): 0 allocations per refit (the untyped cache
    # allocated 5, measured 2026-10-27).
    s = collect(range(0.0, 2.0; length = 40))
    lm = _ExpFitLm(s, [1.5 * exp(-0.8 * si) + 0.3 * si + 0.2 for si in s], SVector(1.0, 0.0, 0.0, 0.0))
    ws = ManifoldLmWorkspace(length(s), 4)
    _refit!(ws, lm); _refit!(ws, lm)
    @test _refit_allocations(ws, lm) == 0
end

@testset "the stop codes: one decoder, one order" begin
    # NLLSsolver's termination bits → `nlls_stop_code`, the one decoder the chart
    # LM and `nlls_termination` read: failure without a step stop −3; step stop
    # (6) 2; cost-decrease stop (2, 3) 1; maxfails (7) 4; time cap −4;
    # iteration cap −1; anything else 4.
    code = VgtkCore.nlls_stop_code
    @test code(1 << 2) == 1 && code(1 << 3) == 1
    @test code(1 << 6) == 2
    @test code(1 << 8) == -1 && code(1 << 9) == -4 && code((1 << 8) | (1 << 9)) == -4
    @test code(1 << 7) == 4 && code(1 << 16) == 4
    for bit in (0, 1, 4, 5, 10, 11)
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
    term = VgtkCore.nlls_termination
    @test retcode_symbol(term(1 << 7, true)...) === :converged               # once the postcondition holds
    @test retcode_symbol(term(1 << 0, true)...) === :solve_failed && retcode_symbol(term(1 << 9, true)...) === :max_time
    # THE success rule: a cost-decrease stop is a success only where the postcondition holds
    @test term(1 << 2, true) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    @test term(1 << 2, false) === (VgtkCore.ReturnCode.Stalled, VgtkCore.CostDecrease)
    calls = Ref(0); probe = () -> (calls[] += 1; true)
    @test term(1 << 2, probe) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease) && calls[] == 1
    @test term(1 << 8, probe) === (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap) && calls[] == 1   # a cap needs none
    # the bits named in public; the decode is the adapter's own
    @test !Base.ispublic(VgtkCore, :nlls_stop_code) && !Base.ispublic(VgtkCore, :nlls_termination) && Base.ispublic(VgtkCore, :nlls_stop_reasons)
    @test VgtkCore.nlls_stop_reasons(0) == Symbol[]
    @test VgtkCore.nlls_stop_reasons((1 << 2) | (1 << 8)) == [:relative_cost_decrease, :max_iterations]
    @test VgtkCore.nlls_stop_reasons((1 << 0) | (1 << 6) | (1 << 16)) == [:cost_infinite, :delta_size, :callback]
    @test length(VgtkCore.nlls_stop_reasons(typemax(Int32))) == 14                # every named bit (0–12, 16)
end

@testset "a NaN wall at the production cost tolerance" begin
    # The stack's ftol (1e-10, Core's configs [lm]): near the wall the last steps'
    # cost decrease and step size shrink together; the step stop is decoded
    # first, so the stall postcondition runs and the stop is :stalled.
    lm_wall = _NaNBeyond(_ExpFitLm([0.0, 1.0, 2.0, 3.0, 4.0], [1.0 + 2si for si in [0.0, 1.0, 2.0, 3.0, 4.0]], SVector(0.0, 0.0, 1.0, 1.0)))
    x, F, c, _, term = lmsolve_chart!(ManifoldLmWorkspace(5, 4), lm_wall)
    @test isfinite(F) && x[3] <= 0.25
    @test retcode_symbol(term...) === :stalled
    # the termination record agrees: a step-size stop that fails the postcondition
    @test term === (VgtkCore.ReturnCode.Stalled, VgtkCore.IncrementSize)
end

@testset "the stop settings: one owner, mapped once" begin
    # every default is Core's configuration of record, configs/default.toml [lm]
    lm_table = VgtkCore._CORE_DEFAULTS["lm"]
    d = LmStops()
    @test (d.max_iter, d.ftol, d.xtol, d.stationarity_tol) == (lm_table["max_iter"], lm_table["ftol"], lm_table["xtol"], lm_table["stationarity_tol"])
    @test VgtkCore.StaticLmStops(SVector(1.0, 1.0); acceptance = 1.0).max_iter == lm_table["device_max_iter"]
    @test VgtkCore.StaticLmStops(SVector(1.0f0); acceptance = 1).stationarity_tol === Float32(lm_table["stationarity_tol"])
    @test isbits(VgtkCore.StaticLmStops(SVector(1.0, 1.0); acceptance = 1.0))
    @test VgtkCore.StaticLmStops(SVector(1.0, 1.0)).acceptance == lm_table["device_acceptance"] == Inf   # stationarity alone
    o = VgtkCore._nlls_options(LmStops(; max_iter = 7, ftol = 1.0e-9, xtol = 1.0e-11))
    @test o.maxiters == 7 && o.reldcost == 1.0e-9 && o.dstep == 1.0e-11 && o.absdcost == 0.0 && o.maxfails == 3
    @test NLLSsolver.dynamic(o.iterator) === :levenbergmarquardt_dtd                  # Marquardt's damping, not a knob
    @test o.maxtime == UInt64(1.0e6 * 1.0e9)                                     # no time cap by default
    @test VgtkCore._nlls_options(LmStops(; max_time = 2.5)).maxtime == UInt64(2.5e9)
    @test typeof(VgtkCore._nlls_options(LmStops(; max_time = 2.5))) === typeof(o)
    # Inf is no cap, as `nothing` is; a negative or NaN cap is refused, as is a cosine outside (0, 1]
    @test LmStops(; max_time = Inf).max_time == LmStops().max_time
    @test LmStops(; max_time = 0).max_time == 0
    @test_throws ArgumentError LmStops(; max_time = -1.0)
    @test_throws ArgumentError LmStops(; max_time = NaN)
    @test_throws ArgumentError LmStops(; stationarity_tol = 0.0)
    @test_throws ArgumentError LmStops(; ftol = -1.0)
    @test Base.ispublic(VgtkCore, :LmStops) && Base.ispublic(VgtkCore, :LmProblem) && Base.ispublic(VgtkCore, :StaticLmStops)
end

@testset "a cost-decrease stop short of stationarity is :stalled" begin
    # A relative cost tolerance of 1 − 1e-6 stops at the first accepted step that does not lower the cost by 99.9999%:
    # far from the minimizer (the start's slope b = 0 against the data's −0.8, the residuals of order 100 — far above
    # the exact-fit floor, which this ftol also sets), where the gradient is not orthogonal to the rows. The old rule
    # called a cost-decrease stop converged unchecked; THE success rule tests it.
    s = collect(range(0.0, 2.0; length = 40))
    lm = _ExpFitLm(s, [100 * (1.5 * exp(-0.8 * si) + 0.3 * si + 0.2 + 0.05 * sin(7si)) for si in s], SVector(1.0, 0.0, 0.0, 0.0))
    x, F, c, _, term = lmsolve_chart!(ManifoldLmWorkspace(length(s), 4), lm, LmStops(; ftol = 1 - 1.0e-6))
    @test term === (VgtkCore.ReturnCode.Stalled, VgtkCore.CostDecrease) && c == -2
    # the same fit at the stack's settings is a success at a stationary point
    _, _, c0, _, term0 = lmsolve_chart!(ManifoldLmWorkspace(length(s), 4), lm)
    @test retcode_symbol(term0...) === :converged && c0 in (1, 2)
end

@testset "a NaN trial is a rejected step" begin
    # NLLSsolver rejects a NaN trial (#64) but accepts any trial at a step below
    # `xtol`, and its per-iteration `cost == cost_` check is false for NaN; the
    # adapter reads a non-finite cost as +Inf, so a NaN trial is rejected and a
    # step at the wall does not throw.
    lm2 = _ExpFitLm([0.0, 1.0, 2.0, 3.0, 4.0], [1.0 + 2si for si in [0.0, 1.0, 2.0, 3.0, 4.0]], SVector(0.0, 0.0, 1.0, 1.0))   # optimum at c = 2: x[3] = 1, behind the wall
    x, F, c, _ = lmsolve_chart!(ManifoldLmWorkspace(5, 4), _NaNBeyond(lm2))
    @test isfinite(F) && x[3] <= 0.25
end
end

@testitem "robust lmsolve_chart!: F is Σρ, the workspace keeps the rows at x" setup = [CircleLmScene] begin
    # The robustified chart LM (Triggs): its cost is Σᵢ ρ(‖rᵢ‖²) (NLLSsolver's
    # ½ρ doubled), and neither the solve nor its stall postcondition leaves the
    # workspace's rows scaled by √ρ′ — they are the rows at the returned x. A
    # redescending loss with every datum past its cutoff is a plateau, not a fit.
    import VisualGeometryToolkitCore as VgtkCore
    using Test, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: ManifoldLmWorkspace, lmsolve_chart!, loss_robustifier, WelschShape, TLSShape, LmStops
    import NLLSsolver

    pts = circle_points()
    append!(pts, [SVector(9.0, 9.0), SVector(-8.0, 7.0), SVector(6.0, -9.0)])   # three gross outliers
    lm = CircleLm(pts)
    n = length(pts)
    rows_at(x) = VgtkCore.residual_rows!(zeros(n), lm, x)
    k = loss_robustifier(WelschShape(), 0.1)
    ws = ManifoldLmWorkspace(n, 3)
    x, F, _, _ = lmsolve_chart!(ws, lm, k, Val(1))
    @test x ≈ SVector(0.5, 0.2, -0.5) atol = 1.0e-6
    @test ws.r == rows_at(x)
    @test F ≈ sum(r -> NLLSsolver.robustify(k, r^2), rows_at(x)) rtol = 1.0e-10
    # the run (problem, cache, data blocks) is kept in the workspace, as the plain chart LM's is: a refit reuses it
    run = ws.run
    @test run isa VgtkCore.LmProblem && length(run.blocks) == n
    x2, F2, _, _ = lmsolve_chart!(ws, lm, k, Val(1))
    @test ws.run === run && x2 == x && F2 == F
    # the stall postcondition, taken directly on the run's data blocks: stationary, rows untouched
    tol, ftol = LmStops().stationarity_tol, LmStops().ftol
    stationary(blocks, x) = VgtkCore._blocks_stationary!(ws.g, ws.w, (0,), blocks, [x], (true,), tol, ftol)
    @test stationary(run.blocks, x)
    @test ws.r == rows_at(x)
    @test !stationary(run.blocks, x + SVector(0.05, 0.0, 0.0))
    @test ws.r == rows_at(x + SVector(0.05, 0.0, 0.0))
    # ... the same test as the public multi-block form's
    @test VgtkCore.is_stationary(run.blocks, [x], [true]; tol, ftol)
    # every datum past a TLS cutoff: ρ′ = 0 throughout, the exact-fit test reads the rows themselves
    kt = loss_robustifier(TLSShape(), 1.0e-3)
    St = VgtkCore._RobustChartState{3, typeof(lm), typeof(kt)}
    tstate = St(lm, kt, zero(SVector{3, Float64}), false, false, ws.r, ws.J)
    @test !stationary([VgtkCore._DatumRows{3, 1, St}(tstate, i) for i in 1:n], SVector(5.0, 5.0, 0.0))
end

# NLLSsolver 5.3.2's damping loop spun on a non-finite linearization at an accepted point (every damped step NaN,
# λ overflowing to Inf); 5.3.3 (#68, the stack's floor) stops there itself: termination bit 12, the accepted point kept,
# a failed solve by the stack's one decoder. No block wraps its linearization.
@testitem "NLLSsolver ≥ 5.3.3: a NaN linearization at an accepted point stops the LM (bit 12)" begin
    using StaticArrays: SVector, SMatrix
    using VisualGeometryToolkitCore: LmProblem, LmStops, lmsolve_chart!, NLLSsolver, ReturnCode, NonFinite
    # r(x) = [acos(min(x, 1)), x − 2]: finite everywhere, its derivative NaN for x ≥ 1 (acos′(1)·0); from x = 0 the first
    # step crosses x = 1 and lowers the cost, so it is accepted, and the linearization there is NaN
    struct NaNPastOne <: NLLSsolver.AbstractResidual end
    NLLSsolver.ndeps(::NaNPastOne) = NLLSsolver.static(1)
    NLLSsolver.nres(::NaNPastOne) = NLLSsolver.static(2)
    NLLSsolver.varindices(::NaNPastOne) = SVector(1)
    NLLSsolver.getvars(::NaNPastOne, vars::Vector) = (vars[1]::Float64,)
    Base.eltype(::NaNPastOne) = Float64
    NLLSsolver.computeresidual(::NaNPastOne, x) = SVector(acos(min(x, 1.0)), x - 2.0)
    function NLLSsolver.computecostgradhess(varflags, r::NaNPastOne, x)
        res = NLLSsolver.computeresidual(r, x)
        J = SMatrix{2, 1, Float64}(x < 1 ? -1 / sqrt(1 - x^2) : NaN, 1.0)
        return 0.5 * sum(abs2, res), J' * res, J' * J
    end
    p = LmProblem([0.0], [NaNPastOne()])
    start_cost = sum(abs2, NLLSsolver.computeresidual(NaNPastOne(), 0.0))
    sol = lmsolve_chart!(p)                                     # returns, rather than spinning
    @test (sol.code, sol.rule) === (ReturnCode.Failure, NonFinite)   # a failed solve (bit 12: the decoder's −3)
    @test sol.iterations <= 3
    @test isfinite(sol.cost) && sol.cost < start_cost
    @test p.variables[1] >= 1                                    # the accepted step is kept
end

# THE multi-block form: heterogeneous variables and blocks, a block's robustifier, a kept problem, a callback, and an
# indefinite normal system read as a failed solve.
@testitem "lmsolve_chart!(::LmProblem): multi-block, robust, reused, the success rule" begin
    using Test, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: LmProblem, LmStops, lmsolve_chart!, NLLSsolver, ReturnCode, loss_robustifier, WelschShape
    # a line y = a + b·s (variable 1, an SVector{2}) and an offset c (variable 2, an SVector{1}) seen as c − a = z
    struct _LineRow <: NLLSsolver.AbstractResidual
        s::Float64; y::Float64
    end
    NLLSsolver.ndeps(::_LineRow) = NLLSsolver.static(1)
    NLLSsolver.nres(::_LineRow) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_LineRow) = SVector(1)
    NLLSsolver.getvars(::_LineRow, vars::Vector) = (vars[1]::SVector{2, Float64},)
    Base.eltype(::_LineRow) = Float64
    NLLSsolver.computeresidual(r::_LineRow, ab) = SVector(ab[1] + ab[2] * r.s - r.y)
    struct _OffsetRow{K} <: NLLSsolver.AbstractResidual
        z::Float64; kernel::K
    end
    NLLSsolver.ndeps(::_OffsetRow) = NLLSsolver.static(2)
    NLLSsolver.nres(::_OffsetRow) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_OffsetRow) = SVector(1, 2)
    NLLSsolver.getvars(::_OffsetRow, vars::Vector) = (vars[1]::SVector{2, Float64}, vars[2]::SVector{1, Float64})
    Base.eltype(::_OffsetRow) = Float64
    NLLSsolver.robustkernel(r::_OffsetRow) = r.kernel
    NLLSsolver.computeresidual(r::_OffsetRow, ab, c) = SVector(c[1] - ab[1] - r.z)
    s = collect(range(-1.0, 1.0; length = 21)); y = [0.5 + 2.0 * si + 0.01 * sin(5si) for si in s]
    z = [1.0 + 0.01 * cos(3k) for k in 1:7]
    vars() = Union{SVector{2, Float64}, SVector{1, Float64}}[SVector(0.0, 0.0), SVector(0.0)]
    blocks = [[_LineRow(si, yi) for (si, yi) in zip(s, y)]; [_OffsetRow(zk, NLLSsolver.NoRobust()) for zk in z]]
    p = LmProblem(vars(), blocks)
    @test p.variables isa Vector{Union{SVector{2, Float64}, SVector{1, Float64}}}
    sol = lmsolve_chart!(p)
    # the linear least-squares answer
    A = [ones(21) s zeros(21); -ones(7) zeros(7) ones(7)]; θ = A \ [y; z]
    @test sol.code === ReturnCode.Success
    @test p.variables[1] ≈ θ[1:2] atol = 1.0e-9
    @test p.variables[2][1] ≈ θ[3] atol = 1.0e-9
    @test sol.cost ≈ sum(abs2, A * θ - [y; z]) rtol = 1.0e-8
    # reused: the kept problem and cache, restarted in place, give the same bits; a callback sees every iteration
    x1 = copy(p.variables)
    p.variables[1] = SVector(0.0, 0.0); p.variables[2] = SVector(0.0)
    seen = Ref(0)
    sol2 = lmsolve_chart!(p; callback = (cost, args...) -> (seen[] += 1; (cost, 0)))
    @test p.variables == x1 && sol2 == sol && seen[] >= 1
    # robust: a Welsch loss on the offset rows (width 3: the start sits 1 off, the outlier 49), one of them a gross outlier, is down-weighted
    zo = copy(z); zo[3] = 50.0
    pr = LmProblem(vars(), [[_LineRow(si, yi) for (si, yi) in zip(s, y)]; [_OffsetRow(zk, loss_robustifier(WelschShape(), 3.0)) for zk in zo]])
    solr = lmsolve_chart!(pr)
    @test solr.code === ReturnCode.Success && abs(pr.variables[2][1] - pr.variables[1][1] - 1.0) < 0.02
    # a cap is a cap, not a success
    ps = LmProblem(vars(), [[_LineRow(si, 100yi) for (si, yi) in zip(s, y)]; [_OffsetRow(zk, NLLSsolver.NoRobust()) for zk in z]])
    @test lmsolve_chart!(ps, LmStops(; max_iter = 1)).code === ReturnCode.MaxIters
    # an indefinite normal system thrown inside the solver is a failed solve; any other exception propagates
    struct _Throws{E} <: NLLSsolver.AbstractResidual
        e::E
    end
    NLLSsolver.ndeps(::_Throws) = NLLSsolver.static(1)
    NLLSsolver.nres(::_Throws) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_Throws) = SVector(1)
    NLLSsolver.getvars(::_Throws, vars::Vector) = (vars[1]::SVector{2, Float64},)
    Base.eltype(::_Throws) = Float64
    NLLSsolver.computeresidual(t::_Throws, ab) = (ab[1] == 0 ? SVector(ab[1] - 1.0) : throw(t.e))
    pf = LmProblem([SVector(0.0, 0.0)], [_Throws(PosDefException(1))])
    @test (lmsolve_chart!(pf).code, lmsolve_chart!(pf).rule) == (ReturnCode.Failure, VgtkCore.NonFinite)
    pe = LmProblem([SVector(0.0, 0.0)], [_Throws(ErrorException("a bug"))])
    @test_throws ErrorException lmsolve_chart!(pe)
    # `fixed` (0.10.1): the default holds nothing — the problem as without the keyword, bit for bit
    @test LmProblem(vars(), blocks).unfixed === nothing && LmProblem(vars(), blocks; fixed = nothing).unfixed === nothing
    # the offset c held at c₀: the line is the least-squares answer with c a constant (the reduced problem, by hand)
    c₀ = 0.8
    v0 = vars(); v0[2] = SVector(c₀)
    ph = LmProblem(v0, blocks; fixed = 2)
    @test ph.free == [true, false] && lmsolve_chart!(ph).code === ReturnCode.Success
    Ar = [ones(21) s; -ones(7) zeros(7)]; θr = Ar \ [y; z .- c₀]
    @test ph.variables[2] == SVector(c₀)
    @test ph.variables[1] ≈ θr atol = 1.0e-9
    @test LmProblem(v0, blocks; fixed = [2]).free == ph.free
    # the postcondition's columns of the free variable are the full problem's, the held one's dropped
    pf = LmProblem(copy(ph.variables), blocks)
    VgtkCore._lm_problem_stationary!(pf, 0.05, 1.0e-10); VgtkCore._lm_problem_stationary!(ph, 0.05, 1.0e-10)
    @test length(ph.g) == 2 && ph.g == pf.g[1:2] && ph.w == pf.w[1:2]
    # an index outside the variables, or every variable held, is refused
    @test_throws ArgumentError LmProblem(vars(), blocks; fixed = 3)
    @test_throws ArgumentError LmProblem(vars(), blocks; fixed = [0])
    @test_throws ArgumentError LmProblem(vars(), blocks; fixed = :c)
    @test_throws ArgumentError LmProblem(vars(), blocks; fixed = [1, 2])
end

# A kept problem's repeated solve allocates O(1): the success rule's postcondition reads the problem's own scratch
# (`p.offsets`, `p.g`, `p.w`, `p.free`, built once with it, as a chart run reads its workspace's), so neither it nor
# the solve's count grows with the number of variables (VisualGeometryToolkitEstimation's relpose allocation gate).
# Core 0.10.2: `LmStops.ftol` is a RELATIVE cost-decrease tolerance; the success rule's exact-fit test has its own,
# absolute threshold. A low-cost but non-stationary stop must not report Success: the Rosenbrock valley scaled by
# 1e-3 (cost ≈ 2.4e-5 at the start, far below a relative ftol of 0.5) stops on the cost-decrease rule (a step that
# lowers the cost by less than half) after its first steps, well short of the minimum, where the gradient is not
# orthogonal to the residual.
@testitem "lmsolve_chart!(::LmProblem): a low-cost stop short of stationarity is not a Success" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore: LmProblem, LmStops, lmsolve_chart!, NLLSsolver, ReturnCode
    struct _ScaledRosenbrock <: NLLSsolver.AbstractResidual
        s::Float64
    end
    NLLSsolver.ndeps(::_ScaledRosenbrock) = NLLSsolver.static(1)
    NLLSsolver.nres(::_ScaledRosenbrock) = NLLSsolver.static(2)
    NLLSsolver.varindices(::_ScaledRosenbrock) = SVector(1)
    NLLSsolver.getvars(::_ScaledRosenbrock, vars::Vector) = (vars[1]::SVector{2, Float64},)
    Base.eltype(::_ScaledRosenbrock) = Float64
    NLLSsolver.computeresidual(r::_ScaledRosenbrock, x) = r.s .* SVector(10 * (x[2] - x[1]^2), 1 - x[1])
    p = LmProblem([SVector(-1.2, 1.0)], [_ScaledRosenbrock(1.0e-3)])
    sol = lmsolve_chart!(p, LmStops(; ftol = 0.5))
    @test sol.cost < 0.5                                       # a cost below the relative tolerance's value …
    @test p.variables[1][1] < 0.9                              # … far from the minimum at (1, 1)
    @test sol.code !== ReturnCode.Success                      # so not a Success: the exact-fit test is not ftol
    # the same problem to its minimum is a Success (an exact fit there: the cost at its rounding)
    q = LmProblem([SVector(-1.2, 1.0)], [_ScaledRosenbrock(1.0e-3)])
    full = lmsolve_chart!(q)
    @test full.code === ReturnCode.Success
    @test q.variables[1] ≈ SVector(1.0, 1.0) atol = 1.0e-6
end

@testitem "lmsolve_chart!(::LmProblem): a repeated solve allocates O(1) in the number of variables" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: LmProblem, lmsolve_chart!, NLLSsolver, ReturnCode
    # a line y = a + b·s (variable 1) and n offsets cₖ (variables 2…n+1), each seen as cₖ − a = zₖ: ONE concrete block
    # type (a line row reads its line only: its offset column is zero), as a production bundle's blocks are
    struct _AllocRow <: NLLSsolver.AbstractResidual
        line::Bool; k::Int; s::Float64; y::Float64
    end
    NLLSsolver.ndeps(::_AllocRow) = NLLSsolver.static(2)
    NLLSsolver.nres(::_AllocRow) = NLLSsolver.static(1)
    NLLSsolver.varindices(r::_AllocRow) = SVector(1, r.k)
    NLLSsolver.getvars(r::_AllocRow, vars::Vector) = (vars[1]::SVector{2, Float64}, vars[r.k]::SVector{1, Float64})
    Base.eltype(::_AllocRow) = Float64
    NLLSsolver.computeresidual(r::_AllocRow, ab, c) = SVector(r.line ? ab[1] + ab[2] * r.s - r.y : c[1] - ab[1] - r.y)
    function problem(n)
        vars = Union{SVector{2, Float64}, SVector{1, Float64}}[SVector(0.0, 0.0)]; append!(vars, (SVector(0.0) for _ in 1:n))
        blocks = [[_AllocRow(true, 2, s, 0.5 + 2s + 0.01sin(5s)) for s in range(-1.0, 1.0; length = 21)];
            [_AllocRow(false, k + 1, 0.0, 1.0 + 0.01cos(3k)) for k in 1:n]]
        return LmProblem(vars, blocks)
    end
    # the same rows as TWO block types (the line's and the offsets'): the problem's blocks a Union, the postcondition
    # one compiled loop per type
    struct _AllocLine <: NLLSsolver.AbstractResidual
        s::Float64; y::Float64
    end
    NLLSsolver.ndeps(::_AllocLine) = NLLSsolver.static(1)
    NLLSsolver.nres(::_AllocLine) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_AllocLine) = SVector(1)
    NLLSsolver.getvars(::_AllocLine, vars::Vector) = (vars[1]::SVector{2, Float64},)
    Base.eltype(::_AllocLine) = Float64
    NLLSsolver.computeresidual(r::_AllocLine, ab) = SVector(ab[1] + ab[2] * r.s - r.y)
    function two_types(n)
        p = problem(n)
        lines = [_AllocLine(b.s, b.y) for b in p.blocks if b.line]
        return LmProblem(copy(p.variables), Union{_AllocLine, _AllocRow}[lines; filter(b -> !b.line, p.blocks)])
    end
    restart!(p) = (p.variables[1] = SVector(0.0, 0.0); for k in 2:length(p.variables); p.variables[k] = SVector(0.0); end; p)
    postcondition(p) = VgtkCore._lm_problem_stationary!(p, 0.05, 1.0e-10)
    function counts(n; build = problem)
        p = build(n)
        lmsolve_chart!(restart!(p)); lmsolve_chart!(restart!(p))     # compiled, the kept cache warm
        restart!(p)
        solve = @allocations lmsolve_chart!(p)
        postcondition(p)
        return (; solve, postcondition = (@allocations postcondition(p)), code = lmsolve_chart!(restart!(p)).code)
    end
    c = [counts(n) for n in (40, 400, 4000)]
    @test all(x -> x.code === ReturnCode.Success, c)
    @test all(x -> x.postcondition == 0, c)                  # the postcondition allocates nothing: its scratch is the problem's
    # and a solve's count does not grow with n. NLLSsolver's own count is not monotone at small n (job 14072: 9, 12, 3, 3
    # at n = 4, 40, 400, 4000, the postcondition 0 throughout), so the sizes compared are the two past it.
    @test c[2].solve == c[3].solve
    u = [counts(n; build = two_types) for n in (40, 400, 4000)]
    @test all(x -> x.code === ReturnCode.Success && x.postcondition == 0, u)
    # and the two-type solve does not grow from n = 400 to 4000 beyond NLLSsolver's own constant. Measured in job 14076:
    # 35 and 37 allocations, with the postcondition at 0 in both. So the whole difference of 2 is NLLSsolver's, and
    # NLLSSOLVER_GROWTH = 2 is that difference. A per-block or per-variable allocation in our code adds at least
    # 3600 here (n grows by 3600).
    NLLSSOLVER_GROWTH = 2
    @test u[3].solve - u[2].solve <= NLLSSOLVER_GROWTH
end

# A device state's two reductions from its per-work-item partial sums: the cost, Jᵀr and JᵀJ of every row, the same
# bits as the unpack VisualGeometryToolkitEstimation's lens regression wrote by hand until 0.10.1 (refine_camera.jl).
@testitem "reduce_normal_equations and reduce_cost: a device state's reductions from its work items' partial sums" begin
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm
    using StableRNGs: StableRNG
    using VisualGeometryToolkitCore: reduce_normal_equations, reduce_cost
    rng = StableRNG(20261113)
    N, n, W = 3, 101, 7                                   # 101 two-row samples dealt to 7 work items, as a kernel strides
    r = [randn(rng, SVector{2, Float64}) for _ in 1:n]; J = [randn(rng, SMatrix{2, N, Float64}) for _ in 1:n]
    partials = zeros(1 + N + N * N, W)
    for w in 1:W
        s = 0.0; g = zero(SVector{N, Float64}); H = zero(SMatrix{N, N, Float64})
        for i in w:W:n
            s += r[i][1] * r[i][1] + r[i][2] * r[i][2]; g += J[i]' * r[i]; H += J[i]' * J[i]
        end
        partials[1, w] = s; partials[2:(1 + N), w] = g; partials[(2 + N):end, w] = vec(H)
    end
    c, g, H = reduce_normal_equations(partials, Val(N))
    @test g isa SVector{N, Float64} && H isa SMatrix{N, N, Float64} && isbits((c, g, H))
    # every row's sum, to rounding (the order differs)
    @test c ≈ 0.5 * sum(x -> sum(abs2, x), r) && g ≈ sum(i -> J[i]' * r[i], 1:n) && H ≈ sum(i -> J[i]' * J[i], 1:n)
    # the hand unpack it replaces, bit for bit
    t = Array(sum(partials; dims = 2))
    @test (c, g, H) === (0.5 * t[1], SVector{N, Float64}(ntuple(a -> t[1 + a], Val(N))), SMatrix{N, N, Float64}(ntuple(k -> t[1 + N + k], Val(N * N))))
    @test reduce_cost(partials) === 0.5 * sum(view(partials, 1, :))
    @test reduce_cost(partials[1:1, :]) === reduce_cost(partials) && reduce_cost(partials) ≈ c
end

# NLLSsolver's termination bits decoded into the termination record (`nlls_termination`, the adapter's own decode).
@testitem "nlls_termination: NLLSsolver's bits, in nlls_stop_code's order" begin
    import NLLSsolver
    import VisualGeometryToolkitCore
    using Test
    using VisualGeometryToolkitCore: ReturnCode, CostDecrease, IncrementSize, RejectionCap, IterationCap, TimeCap, NonFinite, Callback
    using VisualGeometryToolkitCore: nlls_termination
    # a synthetic NLLSsolver result whose only meaningful field is its termination bits
    res(bits) = NLLSsolver.NLLSResult(map(fieldnames(NLLSsolver.NLLSResult), fieldtypes(NLLSsolver.NLLSResult)) do f, F
        f === :termination ? F(bits) : F <: Enum ? first(instances(F)) : zero(F)
    end...)
    t(bits, p = true) = @inferred nlls_termination(res(bits), p)
    for b in (0, 1, 4, 5, 10, 12)
        @test t(1 << b) === (ReturnCode.Failure, NonFinite)
    end
    @test t(1 << 11) === (ReturnCode.InitialFailure, NonFinite)
    @test t((1 << 1) | (1 << 11)) === (ReturnCode.InitialFailure, NonFinite)
    @test t(1 << 2) === (ReturnCode.Success, CostDecrease) && t(1 << 3) === (ReturnCode.Success, CostDecrease)
    @test t(1 << 2, false) === (ReturnCode.Stalled, CostDecrease)               # THE success rule: a cost stop too
    @test t(1 << 6) === (ReturnCode.Success, IncrementSize)
    @test t(1 << 6, false) === (ReturnCode.Stalled, IncrementSize)
    @test t((1 << 2) | (1 << 6), false) === (ReturnCode.Stalled, IncrementSize)      # the step stop first
    @test t((1 << 1) | (1 << 6), false) === (ReturnCode.Stalled, IncrementSize)      # a NaN trial at a step stop
    @test t(1 << 7) === (ReturnCode.Success, RejectionCap)
    @test t(1 << 7, false) === (ReturnCode.Stalled, RejectionCap)
    @test t(1 << 16, false) === (ReturnCode.Terminated, Callback)
    @test t(1 << 9) === (ReturnCode.MaxTime, TimeCap)
    @test t((1 << 8) | (1 << 9)) === (ReturnCode.MaxTime, TimeCap)              # time outranks iterations
    @test t(1 << 8) === (ReturnCode.MaxIters, IterationCap)
    @test t((1 << 8) | (1 << 2)) === (ReturnCode.Success, CostDecrease)         # a tolerance outranks the cap
    @test t((1 << 8) | (1 << 2), false) === (ReturnCode.Stalled, CostDecrease)
    @test t((1 << 1) | (1 << 8) | (1 << 9)) === (ReturnCode.Failure, NonFinite)  # failure outranks the caps
    # the postcondition runs only where it decides
    calls = Ref(0)
    probe = () -> (calls[] += 1; false)
    @test nlls_termination(1 << 8, probe) === (ReturnCode.MaxIters, IterationCap) && calls[] == 0
    @test nlls_termination(1 << 2, probe) === (ReturnCode.Stalled, CostDecrease) && calls[] == 1
    @test nlls_termination(1 << 6, probe) === (ReturnCode.Stalled, IncrementSize) && calls[] == 2
    # with the real postcondition (`is_stationary`), at a least-squares optimum and off it: THE success rule reads it at
    # every stop that is not a failure or a cap — a cost-decrease stop off the optimum is :stalled (until the one
    # success rule, 2026-11-13, it was :converged unchecked)
    J = [1.0 0.0; 0.0 2.0; 1.0 1.0]; y = [1.0, 2.0, 4.0]; x̂ = J \ y
    r_opt = J * x̂ - y; r_off = J * (x̂ .+ 0.1) - y
    sym(bits, r) = VisualGeometryToolkitCore.retcode_symbol(nlls_termination(bits, () -> VisualGeometryToolkitCore.is_stationary(J, r; tol = 1.0e-6, ftol = eps()))...)
    @test sym(1 << 2, r_opt) === :converged && sym(1 << 2, r_off) === :stalled
    @test sym((1 << 8) | (1 << 2), r_opt) === :converged && sym((1 << 8) | (1 << 2), r_off) === :stalled
    @test sym(1 << 6, r_opt) === :converged && sym((1 << 2) | (1 << 6), r_off) === :stalled
    @test sym(1 << 7, r_opt) === :converged && sym(1 << 7, r_off) === :stalled
    @test sym(1 << 8, r_off) === :max_iterations
end

# The launch-and-reduce forms (0.9.7): a row functor over the work items, one kernel on the scratch's backend, then the
# reduction — the same bits as the launch (`_launch_rows!`) and the reduction called apart, the device states' copy
# (Estimation's lens regression, RadiometricCamera's radiometry LM) folded here.
@testitem "reduce_normal_equations and reduce_cost of a row functor: one launch, then the reduction" begin
    using StaticArrays: SVector, SMatrix
    using StableRNGs: StableRNG
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: reduce_normal_equations, reduce_cost
    rng = StableRNG(20261113)
    N, n, W = 3, 101, 7
    r = [randn(rng, SVector{2, Float64}) for _ in 1:n]; J = [randn(rng, SMatrix{2, N, Float64}) for _ in 1:n]
    # work item w's share over samples w, w + W, …: Σ‖rᵢ‖², Σ Jᵢᵀrᵢ, Σ JᵢᵀJᵢ (column-major) into column w
    function normal_row!(out, w, r, J)
        s = 0.0; g = zero(SVector{3, Float64}); H = zero(SMatrix{3, 3, Float64})
        for i in w:size(out, 2):length(r)
            s += r[i][1] * r[i][1] + r[i][2] * r[i][2]; g += J[i]' * r[i]; H += J[i]' * J[i]
        end
        out[1, w] = s
        for a in 1:3; out[1 + a, w] = g[a]; end
        for k in 1:9; out[4 + k, w] = H[k]; end
        return nothing
    end
    function cost_row!(out, w, r)
        s = 0.0
        for i in w:size(out, 2):length(r); s += r[i][1] * r[i][1] + r[i][2] * r[i][2]; end
        out[1, w] = s
        return nothing
    end
    partials = zeros(1 + N + N * N, W)
    got = reduce_normal_equations(normal_row!, Val(N), r, J; partials)
    apart = zeros(1 + N + N * N, W)
    VgtkCore._launch_rows!(normal_row!, apart, W, r, J)
    @test got === reduce_normal_equations(apart, Val(N)) && partials == apart
    @test reduce_cost(cost_row!, r; partials) === reduce_cost(apart) && got[1] ≈ reduce_cost(apart)
    @test got[1] ≈ 0.5 * sum(x -> sum(abs2, x), r) && got[2] ≈ sum(i -> J[i]' * r[i], 1:n)
end

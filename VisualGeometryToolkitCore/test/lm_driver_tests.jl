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
    @test f.cache === isdefined(NLLSsolver, :OptimizeCache)
    @test f.schur === (isdefined(NLLSsolver, :MultiVariateLSschur) && isdefined(NLLSsolver, :iseliminable))
    # a checkout pins the fork (the workspace's [sources]): both are there
    @test f.cache && f.schur
end

@testset "the upstream path gives the fast path's results (fallback)" begin
    # `cache = false` is the path a registered NLLSsolver takes (no
    # OptimizeCache: the solver's internal data rebuilt per call). Same
    # iterates: measured 2026-10-27, bit-identical x, F and iteration counts
    # on every fit below; gated at the LM's tolerance.
    s = collect(range(0.0, 2.0; length = 40))
    for (k, ptrue) in enumerate((SVector(1.5, -0.8, 0.3, 0.2), SVector(0.4, 1.1, -0.5, 1.0), SVector(2.0, 0.2, 0.0, -1.0)))
        y = [ptrue[1] * exp(ptrue[2] * si) + ptrue[3] * si + ptrue[4] + 1.0e-3 * sin(7k * si) for si in s]
        lm = _ExpFitLm(s, y, SVector(1.0, 0.0, 0.0, 0.0))
        ws = ManifoldLmWorkspace(length(s), 4)
        x1, F1, c1, it1 = lmsolve_chart!(ws, lm; stationarity_tol = 0.05, ftol = 1.0e-10)
        x2, F2, c2, it2 = lmsolve_chart!(ws, lm; stationarity_tol = 0.05, ftol = 1.0e-10, cache = false)
        @test lm_retcode(c1) === :converged && lm_retcode(c2) === :converged
        @test isapprox(x1, x2; rtol = 1.0e-10, atol = 1.0e-12)
        @test isapprox(F1, F2; rtol = 1.0e-10)
        @test it1 == it2
        @test x1 == x2 && F1 == F2           # the measured bit identity (the same algorithm and data)
    end
end

@testset "the stop codes" begin
    # NLLSsolver's termination bits → the chart LM's codes (FastLevenbergMarquardt's
    # numbering, which `lm_retcode` reads): a cost-decrease stop is 1, a step
    # stop 2, the caps −1, any other stop 4; a tolerance outranks a cap.
    code = VgtkCore._lm_code
    @test code(1 << 2) == 1 && code(1 << 3) == 1
    @test code(1 << 6) == 2
    @test code(1 << 8) == -1 && code(1 << 9) == -1
    @test code((1 << 8) | (1 << 2)) == 1 && code((1 << 8) | (1 << 6)) == 2
    for bit in (0, 1, 4, 5, 7, 10, 16)
        @test code(1 << bit) == 4
    end
    @test code((1 << 0) | (1 << 6)) == 2             # a non-finite trial at a step-size stop
    @test lm_retcode(4) === :converged               # once the postcondition holds
end

@testset "the knob map" begin
    o = nlls_options(; max_iter = 7, ftol = 1.0e-9, xtol = 1.0e-11)
    @test o.maxiters == 7 && o.reldcost == 1.0e-9 && o.dstep == 1.0e-11 && o.absdcost == 0.0 && o.maxfails == 3
    @test NLLSsolver.dynamic(o.iterator) === :levenbergmarquardt_dtd
    @test NLLSsolver.dynamic(nlls_options(; max_iter = 1, ftol = 0.0, xtol = 0.0, iterator = :levenbergmarquardt).iterator) === :levenbergmarquardt
end

@testset "a NaN trial is a rejected step" begin
    # FastLevenbergMarquardt compared F(x + p) < F(x), so a NaN trial was
    # rejected; NLLSsolver accepts a trial whose cost is not greater than the
    # best, which NaN is not. The adapter reads a non-finite cost as +Inf.
    lm2 = _ExpFitLm([0.0, 1.0, 2.0, 3.0, 4.0], [1.0 + 2si for si in [0.0, 1.0, 2.0, 3.0, 4.0]], SVector(0.0, 0.0, 1.0, 1.0))   # optimum at c = 2: x[3] = 1, behind the wall
    x, F, c, _ = lmsolve_chart!(ManifoldLmWorkspace(5, 4), _NaNBeyond(lm2); stationarity_tol = 0.05)
    @test isfinite(F) && x[3] <= 0.25
end
end

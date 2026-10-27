@testitem "p3p_lm" setup = [P3PScenes, Cameras] begin
    import VisualGeometryToolkitEstimation as VgtkEstimation
    import VisualGeometryToolkitCore as VgtkCore
# =============================================================================
# p3p_lm_tests.jl — the P3P NonlinLsq refit: the FAMILY workspace mechanism
# (ManifoldLmWorkspace{6}, threaded like the essential/fundmat/homography
# refits) solved by the stack's chart LM, VGC's `lmsolve_chart!` — NLLSsolver
# with the adapter's static Gram (`computecostgradhess`). Gated across a size
# sweep:
#   (1) RESULTS AGREE WITH NLLSsolver's OWN LAYOUT — a fresh NLLSsolver problem
#       on the same rows through `computeresjac`, NLLSsolver forming Jᵀr and
#       JᵀJ itself (`jac' * jac`), no cache: the same algorithm, an
#       independent Gram, computed in the same process. The two sum the Gram
#       in different orders, so the agreement is a TOLERANCE: measured
#       2026-10-27 over this sweep (Julia 1.13), ‖Δx‖∞ = 1.2e-11 / 8.2e-12 / 5.3e-16 /
#       1.3e-12 (n = 8 / 20 / 100 / 400; 4 / 5 / 4 / 4 iterations both)
#       against ‖x‖∞ = 9.4e-3 / 1.3e-2 / 4.2e-3 / 2.8e-3, gated at atol 1e-9, rtol 1e-9 (the gate FastLevenbergMarquardt's
#       static `lmsolve` reference had until it left the stack, 2026-10-27).
#       The problems are deliberately WELL-CONDITIONED (convergence in tens of
#       iterations, guarded below): a fit that crawls amplifies benign ulp
#       differences and would test the compiler.
#   (2) ALLOCATION FLAT IN n — the 2n residual/Jacobian buffers, the static
#       iterate and Gram and the NLLSsolver problem are the workspace's (the
#       ~926 MB/frame hotspot this guards). FastLevenbergMarquardt's run
#       allocated nothing; NLLSsolver's `optimize!` through its `OptimizeCache`
#       allocates a few small objects per call (the cache holds its internal
#       data untyped): measured 2026-10-27, 5 allocations at every n of the
#       sweep on Julia 1.13, 9 / 10 / 9 / 9 on 1.12. Gated at the measured
#       maximum, ALLOCS_BOUND = 10, at every n.
# `sqrtw` uses the family per-ROW convention (length 2n, a point's weight
# duplicated across its two rows).
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays
using StableRNGs
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: ManifoldLmWorkspace, lmsolve_chart!
using VisualGeometryToolkitCore: sqrt_weights   # VGC accessor (public)
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: LeftRigidManifold6   # the PnP refit chart (public)
using VisualGeometryToolkitEstimation: _lm_state, residual_rows!, residual_jacobian_rows!
import NLLSsolver

# The reference layout: one residual block of all the rows, its Jacobian
# through `computeresjac` (NLLSsolver forms Jᵀr, JᵀJ).
struct _RefRows{L} <: NLLSsolver.AbstractResidual
    lm::L
    r::Vector{Float64}
    J::Matrix{Float64}
end
NLLSsolver.ndeps(::_RefRows) = NLLSsolver.static(1)
NLLSsolver.nres(c::_RefRows) = length(c.r)
NLLSsolver.varindices(::_RefRows) = SVector(1)
NLLSsolver.getvars(::_RefRows, vars::Vector) = (vars[1]::SVector{6, Float64},)
Base.eltype(::_RefRows) = Float64
NLLSsolver.computeresidual(c::_RefRows, x) = residual_rows!(c.r, c.lm, x)
NLLSsolver.computeresjac(_vf, c::_RefRows, x) = (residual_rows!(c.r, c.lm, x), residual_jacobian_rows!(c.J, c.lm, x))
function _reference_run(lm, n, stationarity_tol)
    problem = NLLSsolver.NLLSProblem(SVector{6, Float64}, _RefRows{typeof(lm)})
    NLLSsolver.addvariable!(problem, zero(SVector{6, Float64}))
    NLLSsolver.addcost!(problem, _RefRows(lm, zeros(2n), zeros(2n, 6)))
    result = NLLSsolver.optimize!(problem, VgtkCore.nlls_options(; max_iter = 1000, ftol = eps(), xtol = 1.0e-10), 1)
    return problem.variables[1], result.niterations
end

@testset "P3P LM refit — family workspace, NLLSsolver chart LM" begin
    stationarity_tol = VgtkEstimation._CHART_LM_STATIONARITY_TOL   # configs/default.toml [chart_lm]
    ALLOCS_BOUND = 10   # the measured maximum: 10 on Julia 1.12 (9 / 10 / 9 / 9), 5 on 1.13
    allocs = Int[]
    for (n, seed) in ((8, 1), (20, 13), (100, 7), (400, 99))
        rng = StableRNG(seed)
        R₀, t₀, prob, sw = p3p_scene(rng, IDCAM, n; σ = 0.001)
        ch = LeftRigidManifold6(R₀, t₀)

        # what `_refine!` runs: the family workspace + VGC's chart LM
        ews = ManifoldLmWorkspace(2n, 6)
        copyto!(sqrt_weights(ews), sw)
        lm = _lm_state(prob, ch, sqrt_weights(ews))
        x_b, _, _, iters = lmsolve_chart!(ews, lm; stationarity_tol)

        # reference: NLLSsolver's own layout on a fresh problem
        x_a, iters_a = _reference_run(_lm_state(prob, ch, copy(sw)), n, stationarity_tol)

        # conditioning guard: a crawling fit would amplify compiler-level ulp
        # variation into visible drift and gate nothing meaningful
        @test iters_a ≤ 60 && iters ≤ 60

        # (1) agreement with the reference layout at the measured level (see header)
        @test isapprox(x_b, x_a; rtol = 1.0e-9, atol = 1.0e-9)

        # (2) the run's allocations do not grow with n
        lmsolve_chart!(ews, lm; stationarity_tol)                 # warm
        push!(allocs, @allocations lmsolve_chart!(ews, lm; stationarity_tol))
    end
    # The bound is the fork's cache path (a checkout); on a registered NLLSsolver the
    # solver's internal data are rebuilt per refit, unmeasured here: no assertion.
    if VgtkCore.nlls_features().cache
        @test all(<=(ALLOCS_BOUND), allocs)    # one bound over a 50× range of n: flat
    else
        @test_skip all(<=(ALLOCS_BOUND), allocs)
    end
end
end

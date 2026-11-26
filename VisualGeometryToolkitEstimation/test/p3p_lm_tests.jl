# R22: contract test — the PnP chart rows (`residual_rows!`, `residual_jacobian_rows!`) against a reference LM
@testitem "p3p_lm" setup = [P3PScenes, Cameras] begin
    import VisualGeometryToolkitEstimation as VgtkEstimation
    import VisualGeometryToolkitCore as VgtkCore
# =============================================================================
# p3p_lm_tests.jl — the P3P NonlinLsq refit: the FAMILY workspace mechanism
# (the summary-free refine cache, threaded like the essential/fundmat/homography
# refits) solved by the stack's chart LM, Core's `LevenbergMarquardt` refine — NLLSsolver
# with the adapter's static Gram (`computecostgradhess`). Gated across a size
# sweep:
#   (1) RESULTS AGREE WITH NLLSsolver's OWN LAYOUT — a fresh NLLSsolver problem
#       on the same rows through `computeresjac`, NLLSsolver forming Jᵀr and
#       JᵀJ itself (`jac' * jac`), no cache: the same algorithm, an
#       independent Gram, computed in the same process. The two sum the Gram
#       in different orders, so the agreement is a TOLERANCE: measured
#       2026-10-27 over this sweep (Julia 1.13), ‖Δx‖∞ = 1.2e-11 / 8.2e-12 / 5.3e-16 /
#       1.3e-12 (n = 8 / 20 / 100 / 400; 4 / 5 / 4 / 4 iterations both)
#       against ‖x‖∞ = 9.4e-3 / 1.3e-2 / 4.2e-3 / 2.8e-3, gated at atol 1e-8, rtol 1e-8
#       (1e-9 until 2026-11-12: some of CI's runners, Julia 1.12.7, missed it by ~6e-10;
#       the two sums' rounding moves with the CPU and the compiler).
#       The problems are deliberately WELL-CONDITIONED (convergence in tens of
#       iterations, guarded below): a fit that crawls amplifies benign ulp
#       differences and would test the compiler.
#   (2) NO ALLOCATION — the 2n residual/Jacobian buffers, the static iterate
#       and Gram, the NLLSsolver problem and its TYPED OptimizeCache (upstream
#       #62, NLLSsolver ≥ 5.3) are the workspace's (the ~926 MB/frame hotspot
#       this guards): measured 2026-10-27, 0 allocations at every n (the
#       untyped cache allocated 5 per refit on 1.13, 9–10 on 1.12).
# `sqrtw` uses the family per-ROW convention (length 2n, a point's weight
# duplicated across its two rows).
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays
using StableRNGs
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: LmStops, solve, init, solve!, reinit!, RefineProblem, LevenbergMarquardt
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
# NLLSsolver's own layout through Core's multi-block LM (a one-block bundle), at the same stop settings as the chart LM
struct _RefBundle{B} <: VisualGeometryToolkitCore.AbstractBlockProblem
    block::B
end
VisualGeometryToolkitCore.block_variables(::_RefBundle, _u0) = [zero(SVector{6, Float64})]
VisualGeometryToolkitCore.residual_blocks(f::_RefBundle, _u0) = [f.block]
VisualGeometryToolkitCore.residual_summary(::_RefBundle) = nothing
VisualGeometryToolkitCore.block_result(::_RefBundle, _u0, r, _summary) = r
function _reference_run(lm, n)
    r = solve(RefineProblem(_RefBundle(_RefRows(lm, zeros(2n), zeros(2n, 6))), nothing), LevenbergMarquardt(; optimization_strategy = nothing);
              stops = LmStops())
    return r.u[1], r.stats.nsteps
end
# The family's chart rows as a row problem of Core's LevenbergMarquardt refine (what `_refine!` runs: the summary-free
# chart LM on the problem's rows).
struct _FamilyRows{L} <: VisualGeometryToolkitCore.AbstractRowProblem{6}
    lm::L
    rows::Int
end
VisualGeometryToolkitCore.chart_size(f::_FamilyRows) = (f.rows, Val(6))
VisualGeometryToolkitCore.residual_rows!(r, f::_FamilyRows, x) = residual_rows!(r, f.lm, x)
VisualGeometryToolkitCore.residual_jacobian_rows!(J, f::_FamilyRows, x) = residual_jacobian_rows!(J, f.lm, x)

@testset "P3P LM refit — family workspace, NLLSsolver chart LM" begin
    ALLOCS_BOUND = 0    # NLLSsolver's TYPED OptimizeCache (upstream #62): 0 at every n, Julia 1.13 and 1.12
    allocs = Int[]
    for (n, seed) in ((8, 1), (20, 13), (100, 7), (400, 99))
        rng = StableRNG(seed)
        R₀, t₀, prob, sw = p3p_scene(rng, IDCAM, n; σ = 0.001)
        ch = LeftRigidManifold6(R₀, t₀)

        # the chart LM on the family's rows (Core's [lm] stop settings, as the LO refit's)
        lm = _lm_state(prob, ch, copy(sw))
        c = init(RefineProblem(_FamilyRows(lm, 2n), zero(SVector{6, Float64})), LevenbergMarquardt(; optimization_strategy = nothing))
        rb = solve!(c)
        x_b, iters = rb.u, rb.stats.nsteps

        # reference: NLLSsolver's own layout on a fresh problem
        x_a, iters_a = _reference_run(_lm_state(prob, ch, copy(sw)), n)

        # conditioning guard: a crawling fit would amplify compiler-level ulp
        # variation into visible drift and gate nothing meaningful
        @test iters_a ≤ 60 && iters ≤ 60

        # (1) agreement with the reference layout at the measured level (see header)
        @test isapprox(x_b, x_a; rtol = 1.0e-8, atol = 1.0e-8)

        # (2) the run's allocations do not grow with n
        solve!(reinit!(c))                      # warm
        push!(allocs, @allocations solve!(reinit!(c)))
    end
    @test all(<=(ALLOCS_BOUND), allocs)    # one bound over a 50× range of n
    @test maximum(allocs) - minimum(allocs) <= 4   # flat in n
end
end

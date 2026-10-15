@testitem "p3p_lm" setup = [P3PScenes, Cameras] begin
# =============================================================================
# p3p_lm_tests.jl — the P3P NonlinLsq refit: the FAMILY workspace mechanism
# (ManifoldLmWorkspace{6}, threaded like the essential/fundmat/homography
# refits) solved by the stack's chart LM, VGC's `lmsolve_chart!` —
# FastLevenbergMarquardt's in-place `lmsolve!`. Gated across a size sweep:
#   (1) RESULTS AGREE WITH FLM's STATIC `lmsolve` — the same algorithm on
#       value-returning residuals, computed in the same process. The in-place
#       run forms g = J'f and J'J by `mul!` into static buffers and D by a
#       column sum where the static run forms them by value, so the agreement
#       is a TOLERANCE: measured 2026-10-01 over this sweep, ‖Δx‖ ≤ 1.87e-10
#       (n = 20, 30 vs 28 iterations; ≤ 2e-15 at the other sizes) against
#       ‖x‖ = 9.4e-3. Re-measured 2026-10-09 on the lens-family projections
#       (K+ merged with feat/lens-families; last-ulp shifts in the camera's
#       projection): ‖Δx‖ = 6.6e-10 at n = 20 against ‖x‖ = 1.3e-2, so gated
#       at atol 1e-9, rtol 1e-9. (The retired hand-rolled
#       core agreed to 7.5e-13 under rtol 1e-9, atol 1e-12.) Under --check-bounds both
#       paths' compiled results shift by last-ulp amounts, so the gate loosens
#       there. The problems are deliberately WELL-CONDITIONED (convergence in
#       tens of iterations, guarded below): a fit that crawls for ~100
#       iterations amplifies benign ulp differences and would test the compiler.
#   (2) NO ALLOCATION — the 2n residual/Jacobian buffers, the static iterate
#       and the static solver are the workspace's (the ~926 MB/frame hotspot).
# `sqrtw` uses the family per-ROW convention (length 2n, a point's weight
# duplicated across its two rows).
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays
using VisualGeometryCore
using VisualGeometryCore: ManifoldLmWorkspace, lmsolve_chart!
using VisualGeometryCore: sqrt_weights   # VGC accessor (public)
using RobustVisualGeometry
using VisualGeometryCore: LeftRigidManifold6   # the PnP refit chart (public)
using RobustVisualGeometry: _lm_state, residual_rows!, residual_jacobian_rows!
using RobustVisualGeometry.FastLevenbergMarquardt: lmsolve

@testset "P3P LM refit — family workspace, FLM chart LM" begin
    x0 = zero(SVector{6, Float64})
    stationarity_tol = RobustVisualGeometry._CHART_LM_STATIONARITY_TOL   # configs/default.toml [chart_lm]
    bounds_checked = Base.JLOptions().check_bounds == 1
    for (n, seed) in ((8, 1), (20, 13), (100, 7), (400, 99))
        rng = MersenneTwister(seed)
        R₀, t₀, prob, sw = p3p_scene(rng, IDCAM, n; σ = 0.001)
        ch = LeftRigidManifold6(R₀, t₀)
        data = _lm_state(prob, ch, sw)

        # reference: FLM's static `lmsolve`, fresh residual Vector per evaluation
        res(x, d) = residual_rows!(Vector{Float64}(undef, 2n), d, x)
        Ja = Matrix{Float64}(undef, 2n, 6)
        jaca(x, d) = residual_jacobian_rows!(Ja, d, x)
        out_a = lmsolve(res, jaca, x0, data)

        # conditioning guard: a crawling fit would amplify compiler-level ulp
        # variation into visible drift and gate nothing meaningful
        @test out_a[4] ≤ 60

        # what `_refine!` runs: the family workspace + VGC's chart LM
        ews = ManifoldLmWorkspace(2n, 6)
        copyto!(sqrt_weights(ews), sw)
        lm = _lm_state(prob, ch, sqrt_weights(ews))
        x_b, _, _ = lmsolve_chart!(ews, lm; stationarity_tol)

        # (1) agreement with the static FLM run at the measured level (see header)
        if bounds_checked
            @test isapprox(x_b, out_a[1]; rtol = 1.0e-6, atol = 1.0e-10)
        else
            @test isapprox(x_b, out_a[1]; rtol = 1.0e-9, atol = 1.0e-9)
        end

        # (2) the run allocates nothing at any size
        lmsolve_chart!(ews, lm; stationarity_tol)                 # warm
        @test (@allocated lmsolve_chart!(ews, lm; stationarity_tol)) == 0
    end
end
end

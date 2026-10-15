# The DLT DESIGN matrix's null-space gap: the measurement, the gate over it, and
# the boundary they share with RobustVisualGeometry's refusal. Tested at the
# boundary rather than in the middle — a mid-range sample agrees by construction
# and proves nothing about a discrete outcome.
@testitem "dlt_conditioning" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryCore
    using VisualGeometryCore: SvdWorkspace, dlt_conditioning, test_dlt_conditioning,
        CspondNormalization, HartleyNormalization, DiagonalNormalization,
        homography_dlt, homography_dlt!

    @testset "the ratio is RVG's quantity: S[1] / max(S[8], eps)" begin
        ws = SvdWorkspace{Float64}(64, 9)
        # Write the singular values directly: this pins the FORMULA, including
        # the index and the floor, without depending on a solve.
        ws.S .= [100.0, 9, 8, 7, 6, 5, 4, 1.0, 1.0e-30]
        @test dlt_conditioning(ws) == 100.0 / 1.0          # S[1]/S[8]; S[9] is the null direction, skipped
        @test dlt_conditioning(ws; null_dim = 2) == 100.0 / 4.0   # S[1]/S[7]

        # The eps floor: an exactly rank-deficient design matrix must not divide
        # by zero, it must report an enormous ratio and so refuse.
        ws.S .= [1.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0]
        @test dlt_conditioning(ws) == 1.0 / eps(Float64)
        @test isfinite(dlt_conditioning(ws))
        @test !test_dlt_conditioning(ws)
    end

    @testset "the boundary: exactly max_cond succeeds, one ulp above fails" begin
        ws = SvdWorkspace{Float64}(64, 9)
        # RVG refuses on a STRICT `>`, so a ratio of exactly the threshold is
        # accepted on both sides. One ulp either way must flip cleanly.
        ws.S .= [100.0, 9, 8, 7, 6, 5, 4, 1.0, 1.0e-30]
        @test dlt_conditioning(ws) == 100.0
        @test test_dlt_conditioning(ws; max_cond = 100.0)              # equal → succeeds
        @test !test_dlt_conditioning(ws; max_cond = prevfloat(100.0))  # one ulp below → fails

        ws.S[1] = nextfloat(100.0)
        @test !test_dlt_conditioning(ws; max_cond = 100.0)             # one ulp above → fails
        ws.S[1] = prevfloat(100.0)
        @test test_dlt_conditioning(ws; max_cond = 100.0)              # one ulp under → succeeds

        # max_cond = Inf is RVG's generic default: never refuse.
        ws.S .= [1.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0]
        @test test_dlt_conditioning(ws; max_cond = Inf)
    end

    @testset "against a directly computed svd of the same design matrix" begin
        rng = Xoshiro(5)
        H = HomographyMat(SMatrix{3, 3, Float64}(1.2, 0.03, 2.0e-4, -0.05, 0.9, 1.0e-4, 30.0, -12.0, 1.0))
        n = 24
        u₁ = [SVector(50 + 400rand(rng), 50 + 400rand(rng)) for _ in 1:n]
        u₂ = [PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)) for u in u₁]
        cs = u₁ .=> u₂

        A = Matrix{Float64}(undef, 2n, 9)
        ws = SvdWorkspace{Float64}(2n, 9)
        N = CspondNormalization(HomographyMat, cs)
        homography_dlt!(A, N, cs; svd_ws = ws)
        S = ws.S
        @test dlt_conditioning(ws) ≈ S[1] / S[8] rtol = 1.0e-15
        # a well-conditioned set is far from the threshold, both ways round
        @test dlt_conditioning(ws) < 100.0
        @test test_dlt_conditioning(ws)
    end

    @testset "a collinear set is refused; the healthy set is not" begin
        rng = Xoshiro(11)
        H = HomographyMat(SMatrix{3, 3, Float64}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, -3.0, 1.0))
        n = 16
        # Every source point on one line: the DLT null space is not isolated.
        line = [SVector(100.0 + 20i, 200.0 + 10i) for i in 1:n]
        collinear = line .=> [PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)) for u in line]

        A = Matrix{Float64}(undef, 2n, 9)
        ws = SvdWorkspace{Float64}(2n, 9)
        homography_dlt!(A, CspondNormalization(HomographyMat, collinear), collinear; svd_ws = ws)
        @test dlt_conditioning(ws) > 100.0
        @test !test_dlt_conditioning(ws)

        general = [SVector(50 + 400rand(rng), 50 + 400rand(rng)) for _ in 1:n]
        healthy = general .=> [PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)) for u in general]
        homography_dlt!(A, CspondNormalization(HomographyMat, healthy), healthy; svd_ws = ws)
        @test test_dlt_conditioning(ws)
    end

    @testset "the number lives in the solve's own coordinates" begin
        # The docstring's claim, pinned: the ratio is a property of the design
        # matrix AS BUILT, so a different normalization gives a different number
        # and a threshold calibrated for one does not transfer to the other.
        # Comparability with RVG's 100.0 exists BECAUSE both build Hartley.
        rng = Xoshiro(7)
        H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
        n = 20
        u₁ = [SVector(50 + 1500rand(rng), 50 + 1000rand(rng)) for _ in 1:n]
        u₂ = [PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)) for u in u₁]
        cs = u₁ .=> u₂

        A = Matrix{Float64}(undef, 2n, 9)
        ws_h = SvdWorkspace{Float64}(2n, 9)
        homography_dlt!(A, CspondNormalization(HomographyMat, cs), cs; svd_ws = ws_h)
        hartley = dlt_conditioning(ws_h)

        ws_d = SvdWorkspace{Float64}(2n, 9)
        d = AffineMap(DiagonalNormalization((1600, 1200)))
        Nd = CspondNormalization(d, d)
        homography_dlt!(A, Nd, cs; svd_ws = ws_d)
        diagonal = dlt_conditioning(ws_d)

        @test isfinite(hartley) && isfinite(diagonal)
        @test hartley != diagonal          # same data, same solver, different frame → different number
    end
end

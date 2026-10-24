# The measurement-noise rule in the correspondence problems (VisualGeometryToolkitCore
# conventions/measurement-noise.md): a plain point is unit isotropic noise, an
# `Uncertain` point carries its covariance, an `exact` point none — through
# scoring, the chart rows of the refit, the normalization and the maps.
@testitem "measurement_noise" setup = [HomographyScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using StableRNGs
    using Rotations: RotMatrix, RotX
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: csponds, exact, Uncertain, CovMat, pair_covariance, transfer_whitening!,
        whiten, constraint_rounding_scale, source_map, target_map, sampson_distance, residuals!,
        converged, estimate, MsacScoring, TLSShape
    using VisualGeometryToolkitEstimation
    using VisualGeometryToolkitEstimation: Irls, locally_optimize, RadialHomographyProblem

    rng = StableRNG(5)
    H = HomographyMat(Camera(CameraModel(; focal = 800.0, pp = (320.0, 240.0)), EuclideanMap(RotMatrix(RotX(0.25)), SA[-0.025, -0.025, 0.3])))
    warp = homography_warp(H)
    board = [SA[0.05 * i / 7, 0.05 * j / 7] for i in 0:7 for j in 0:7]          # metres
    pix = [warp(b) + SA[0.2 * randn(rng), 0.2 * randn(rng)] for b in board]
    n = length(board)
    transfer = [norm(warp(board[i]) - pix[i]) for i in 1:n]

    @testset "plain points: no noise columns, the isotropic constraint" begin
        p = HomographyProblem(csponds(board, pix))
        @test isempty(keys(p.cs.attrs))
        @test all(pair_covariance(p.cs, i) === I for i in 1:n)
        r = residuals!(zeros(n), p, H)
        # the unchanged isotropic whitening, bit for bit
        ref = map(1:n) do i
            g, Gx, Gθ = VgtkCore.constraint_with_jacobians(H, board[i], pix[i])
            norm(first(whiten(g, Gx, Gθ, I, constraint_rounding_scale(H, board[i], pix[i]))))
        end
        @test r == ref
        @test r == [sampson_distance(H, board[i], pix[i]) for i in 1:n]
        @test maximum(r) < 1.0e-3                     # metres: the unit the rule removes
    end

    @testset "exact source: the transfer error in target pixels" begin
        p = HomographyProblem(csponds(exact.(board), pix))
        @test keys(p.cs.attrs) == (:Σ₁,)
        @test all(iszero, p.cs.Σ₁)
        r = residuals!(zeros(n), p, H)
        @test maximum(abs.(r .- transfer) ./ transfer) < 1.0e-10
        @test [sampson_distance(H, exact(board[i]), pix[i]) for i in 1:n] ≈ r rtol = 1.0e-12
        # analytically: G₂ = −c·I, so Σ̃ = c²I and r = g/|c| — and the transfer
        # form's frozen whitening is the identity
        W = Vector{SMatrix{2, 2, Float64, 4}}(undef, n)
        transfer_whitening!(W, H, board, p.cs.Σ₁, nothing)
        @test all(w == SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0) for w in W)
        @test (@allocated transfer_whitening!(W, H, board, p.cs.Σ₁, nothing)) == 0
        @test (@allocated residuals!(r, p, H)) == 0
        # the chart-LM refit and its lift descend the same pixel residual
        Hs = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(H) .* (1 .+ 1.0e-3 .* SMatrix{3, 3, Float64, 9}(1, -2, 1, 2, -1, 1, -1, 1, 0))))
        lr = locally_optimize(p, Hs; scoring = MsacScoring(p; threshold = 3.0), local_optimization = Irls{TLSShape}())
        @test max_transfer_error(estimate(lr), H, [SA[0.01, 0.01], SA[0.04, 0.02], SA[0.025, 0.045]]) < 0.3
        # the solver covariance takes the singular (exact-side) noise shape
        C = VgtkEstimation.solver_covariance(p, H, collect(1:n))
        @test C !== nothing && all(isfinite, C)
        @test VgtkEstimation.solver_covariance(p, H, [1, 8, 57, 64]) !== nothing
    end

    @testset "explicit covariances whiten by G₁Σ₁G₁ᵀ + G₂Σ₂G₂ᵀ" begin
        Σ₁ = [SMatrix{2, 2}(1.0e-6 * (3 + 0.1i), 5.0e-7, 5.0e-7, 2.0e-6) for i in 1:n]
        Σ₂ = SMatrix{2, 2}(2.0, 0.3, 0.3, 1.0)
        src = [Uncertain(board[i], CovMat(Σ₁[i])) for i in 1:n]
        dst = [Uncertain(pix[i], CovMat(Σ₂)) for i in 1:n]
        p = HomographyProblem(csponds(src, dst))
        r = residuals!(zeros(n), p, H)
        manual = map(1:n) do i
            g, Gx, _ = VgtkCore.constraint_with_jacobians(H, board[i], pix[i])
            G₁ = Gx[:, SVector(1, 2)]; G₂ = Gx[:, SVector(3, 4)]
            sqrt(dot(g, (G₁ * Σ₁[i] * G₁' + G₂ * Σ₂ * G₂') \ g))
        end
        @test r ≈ manual rtol = 1.0e-10
        # the private normalization moves the noise with the points (Σ′ = AΣAᵀ),
        # so the Mahalanobis residual is the same in the normalized frame
        N = p.normalization
        A₁ = source_map(N).linear; A₂ = target_map(N).linear
        @test all(p.cs_hartley.Σ₁[i] ≈ A₁ * Σ₁[i] * A₁' for i in 1:n)
        @test all(p.cs_hartley.Σ₂[i] ≈ A₂ * Σ₂ * A₂' for i in 1:n)
        Hn = HomographyMat(Tuple(SMatrix{3, 3}((target_map(N) ∘ HomographyMap(H) ∘ inv(source_map(N))).forward)))
        cn = p.cs_hartley
        rn = [sampson_distance(Hn, Uncertain(sources(cn)[i], CovMat(cn.Σ₁[i])), Uncertain(targets(cn)[i], CovMat(cn.Σ₂[i]))) for i in 1:n]
        @test rn ≈ r rtol = 1.0e-9
    end

    @testset "refusals" begin
        @test_throws ArgumentError HomographyProblem(csponds(exact.(board), exact.(pix)))
        @test_throws ArgumentError FundMatProblem(csponds(exact.(board), pix))
        # a centred model's source is exact by construction: `exact` is accepted
        # and changes nothing; any other covariance is refused
        c = SA[320.0, 240.0]
        pr = RadialHomographyProblem(csponds(exact.(board), pix); center = c)
        pr0 = RadialHomographyProblem(csponds(board, pix); center = c)
        @test typeof(pr) == typeof(pr0)
        @test_throws ArgumentError RadialHomographyProblem(csponds(board, [Uncertain(q, CovMat{2}(σ = 1.0)) for q in pix]); center = c)
    end

    @testset "metre board: MSAC's pixel threshold means pixels" begin
        bad = copy(pix)
        for k in 1:8
            bad[k] = bad[k] + SA[40.0 * randn(rng), 40.0 * randn(rng)]
        end
        res = find_homography(csponds(exact.(board), bad); rng = StableRNG(1))
        @test converged(res)
        @test !any(res.inlier_mask[1:8])
        @test max_transfer_error(estimate(res), H, [SA[0.01, 0.01], SA[0.04, 0.02], SA[0.025, 0.045]]) < 1.0
    end
end

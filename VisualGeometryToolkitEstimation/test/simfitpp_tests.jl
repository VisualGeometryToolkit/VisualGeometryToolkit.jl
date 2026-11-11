# SIMFIT++ port (src/estimators/ransac/simfitpp.jl; Edstedt, CVPRW 2025,
# github.com/Parskatt/simfitpp) — behaviour pinned against the reference's
# algorithm on synthetic residuals; no dataset, CPU only. Moved with the
# estimator from RANSACScoringDoneRight (test/simfitpp_tests.jl), unchanged but
# for the pool fixture, built here.

@testitem "simfitpp" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, Random, Statistics
    using Distributions: Chisq, quantile, cdf
    using StaticArrays, LinearAlgebra
    using VisualGeometryToolkitEstimation
    using VisualGeometryToolkitEstimation: SimFitPP, unbiased_median_variance, simfitpp_loss_width,
                                           simfitpp_select, with_residual_unit,
                                           _msac_best_over
    using VisualGeometryToolkitCore.Pools: PoolPair, pool_normalization, pool_frame_calibrations,
                                    calibration_matrices, pixel_residuals, pool_pixel_scale
    using VisualGeometryToolkitCore

    # The light record of a pool of identity models over zero correspondences,
    # the shape of the residual matrix `D` it is scored with; identity maps, so
    # the pool frame IS the pixel frame.
    I3 = SMatrix{3,3,Float64,9}(I)
    KC = SMatrix{3,3,Float64,9}(400.0, 0.0, 0.0, 0.0, 400.0, 0.0, 320.0, 240.0, 1.0)
    identity_pool(D; sample_indices) = PoolPair{typeof(I3)}(;
        csponds = [SVector(0.0, 0.0) => SVector(0.0, 0.0) for _ in 1:size(D, 1)], K1 = KC, K2 = KC,
        R_gt = I3, t_gt = SVector(1.0, 0.0, 0.0), models = [I3 for _ in 1:size(D, 2)], sample_indices,
        errors_r = fill(NaN, size(D, 2)), errors_t = fill(NaN, size(D, 2)),
        gt_inliers = falses(size(D, 1)), gt_variance = NaN, source_map = I3, destination_map = I3)

    @testset "unbiased_median_variance (UnbiasedMed)" begin
        rng = MersenneTwister(7)
        σ = 1.3; n = 200_000
        r2 = (σ .* randn(rng, n)) .^ 2                       # χ²₁ scaled by σ²
        # Wide loss width: the truncation correction is negligible and the estimate
        # is the naive median/ppf(1/2).
        wide = unbiased_median_variance(r2, (20σ)^2)
        @test isapprox(wide, σ^2; rtol = 0.02)
        # Truncation at 2σ (ratio ≈ 8.8 > min_ratio 5): the naive estimate is biased
        # low, the corrected one is closer to σ².
        sq_loss_width = (2σ)^2
        kept = r2[r2 .< sq_loss_width]
        naive = median(kept) / quantile(Chisq(1), 0.5)
        corrected = unbiased_median_variance(r2, sq_loss_width)
        @test naive < σ^2
        @test abs(corrected - σ^2) < abs(naive - σ^2)
        @test isapprox(corrected, σ^2; rtol = 0.05)
        # Below the reference's min_ratio for χ²₁ (ratio < 5) the naive estimate is
        # returned unchanged (truncation at 0.8σ leaves a ratio near 4.7).
        th_sq_low = (0.8σ)^2
        kept_low = r2[r2 .< th_sq_low]
        @test th_sq_low / median(kept_low) < 5
        @test unbiased_median_variance(r2, th_sq_low) == median(kept_low) / quantile(Chisq(1), 0.5)
        # No residual below the threshold → NaN (the reference's median of an empty array).
        @test isnan(unbiased_median_variance(r2, 0.0))
        # The fixed point is the quantile the truncated median corresponds to:
        # q solves q = F(ratio·F⁻¹(q))/2, so m / F⁻¹(q) with m the truncated median.
        m = median(kept); ratio = sq_loss_width / m; q = 0.5
        for _ in 1:5; q = cdf(Chisq(1), ratio * quantile(Chisq(1), q)) / 2; end
        @test corrected == m / quantile(Chisq(1), q)
    end

    # A synthetic pool: one true model whose residuals are σ·N(0,1) on a 60%
    # inlier set and uniform on the rest, plus decoy models with uniform residuals.
    function _synthetic_pool(rng; n = 4000, σ = 1.0, inlier_frac = 0.6, n_decoys = 30, outlier_span = 40.0)
        n_in = round(Int, inlier_frac * n)
        true_d = vcat(σ .* abs.(randn(rng, n_in)), outlier_span .* rand(rng, n - n_in))
        D = Matrix{Float32}(undef, n, n_decoys + 1)
        D[:, 1] = true_d
        for j in 2:n_decoys + 1
            D[:, j] = outlier_span .* rand(rng, n)
        end
        return D
    end

    @testset "simfitpp_loss_width on a synthetic estimator" begin
        rng = MersenneTwister(11)
        σ = 1.0
        D = _synthetic_pool(rng; σ)
        n = size(D, 1)
        fit(loss_width, train) = _msac_best_over(D, train, loss_width)
        residuals_sq(j, val) = Float64[Float64(D[i, j])^2 for i in val]
        cfg = SimFitPP()                                          # alpha 0.99, bounds 0.25..8
        loss_width, ok = simfitpp_loss_width(fit, residuals_sq, n, 1.0, cfg; rng = MersenneTwister(1))
        @test ok
        # The loss width targets loss_width² = ppf_{χ²₁}(0.99)·σ² ≈ 6.63σ², i.e. ≈ 2.58σ.
        @test isapprox(loss_width, sqrt(quantile(Chisq(1), 0.99)) * σ; rtol = 0.15)
        # Deterministic under a fixed rng.
        repeat_loss_width, _ = simfitpp_loss_width(fit, residuals_sq, n, 1.0, cfg; rng = MersenneTwister(1))
        @test repeat_loss_width == loss_width
        # Cross-pair filtering: with p earlier estimates behind the initial loss
        # width, the result is the log-space weighted mean of the raw estimate and it.
        raw_loss_width, _ = simfitpp_loss_width(fit, residuals_sq, n, 1.0, cfg; rng = MersenneTwister(1))
        filtered_loss_width, ok_f = simfitpp_loss_width(fit, residuals_sq, n, 1.0, cfg; num_previous_est = 3, rng = MersenneTwister(1))
        @test ok_f
        @test isapprox(filtered_loss_width, exp(log(raw_loss_width) / 4 + 3 / 4 * log(1.0)); rtol = 1e-12)
        # Out of bounds at every round (σ = 0.02 puts every proposal near 0.05,
        # below min_loss_width = 0.25): the guess is returned and the estimate flagged.
        Dsm = _synthetic_pool(MersenneTwister(3); σ = 0.02)
        fitsm(loss_width, train) = _msac_best_over(Dsm, train, loss_width)
        ressm(j, val) = Float64[Float64(Dsm[i, j])^2 for i in val]
        small_loss_width, oksm = simfitpp_loss_width(fitsm, ressm, size(Dsm, 1), 1.0, cfg; rng = MersenneTwister(1))
        @test !oksm
        @test small_loss_width == 1.0
        # An estimator that fails (nothing) never produces a loss width.
        failed_loss_width, okf = simfitpp_loss_width((loss_width, tr) -> nothing, (m, v) -> nothing, n, 1.0, cfg)
        @test !okf && failed_loss_width == 1.0
        # with_residual_unit scales only the bounds.
        cfg_unit = with_residual_unit(cfg, 0.01)
        @test cfg_unit.min_loss_width == cfg.min_loss_width * 0.01 &&
              cfg_unit.max_loss_width == cfg.max_loss_width * 0.01 && cfg_unit.alpha == cfg.alpha
    end

    @testset "simfitpp_select on a pool's residual matrix" begin
        rng = MersenneTwister(5)
        D = _synthetic_pool(rng; σ = 0.8)
        n, K = size(D)
        pair = identity_pool(D; sample_indices = zeros(Int, 5, K))
        @test pool_pixel_scale(pair) == 1.0
        # identity maps: the pool frame IS the pixel frame, so the matrix method
        # on the stored residuals runs at the reference's pixel bounds
        j, loss_width, ok = simfitpp_select(D, 1.0; rng = MersenneTwister(2))
        @test ok
        @test j == 1                                             # the true model wins at the estimated loss width
        @test isapprox(loss_width, sqrt(quantile(Chisq(1), 0.99)) * 0.8; rtol = 0.15)
    end

    @testset "pixel_residuals: the pool models' residuals in the reference's frame" begin
        using VisualGeometryToolkitCore: sampson_distance, FundMat
        rng = MersenneTwister(21)
        n, K = 300, 12
        # A pool in its pool frame: correspondences in [-1,1]², random fundamental
        # models, pool-frame Sampson residuals computed exactly as the pool generator does.
        sim(s, tx, ty) = SMatrix{3,3,Float64,9}(s, 0.0, 0.0, 0.0, s, 0.0, tx, ty, 1.0)   # column-major: [s 0 tx; 0 s ty; 0 0 1]
        Kc = SMatrix{3,3,Float64,9}(400.0, 0.0, 0.0, 0.0, 400.0, 0.0, 320.0, 240.0, 1.0)
        I3s = SMatrix{3,3,Float64,9}(I)
        csponds = [SVector(2rand(rng) - 1, 2rand(rng) - 1) => SVector(2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:n]
        models = [FundMat(Tuple(SMatrix{3,3,Float64,9}(randn(rng, 9)...))) for _ in 1:K]
        Dc = Float32[sampson_distance(models[j], csponds[i].first, csponds[i].second) for i in 1:n, j in 1:K]
        mk(T1, T2; ms = models) = PoolPair(; csponds, K1 = Kc, K2 = Kc, R_gt = I3s, t_gt = SVector(1.0, 0.0, 0.0),
                                           models = ms, sample_indices = zeros(Int, 7, length(ms)),
                                           errors_r = fill(NaN, length(ms)), errors_t = fill(NaN, length(ms)),
                                           gt_inliers = falses(n), gt_variance = NaN, source_map = T1, destination_map = T2)
        # identity maps: the pixel frame IS the pool frame — the image-plane
        # Sampson of the models, recomputed (the stored matrix is never read)
        @test pixel_residuals(mk(I3s, I3s)) == Dc
        # equal similarity scales: exactly the pool-frame residual divided by the scale
        s = 1 / 650.0
        Deq = pixel_residuals(mk(sim(s, -0.9, -0.7), sim(s, -0.95, -0.65)))
        @test isapprox(Deq, Dc ./ Float32(s); rtol = 2e-5)
        # unequal scales: per-point ratios lie strictly between 1/s1 and 1/s2 — no
        # single scalar converts the frames, which is why SIMFIT++ runs on these
        s1, s2 = 1 / 650.0, 1 / 340.0
        Dne = pixel_residuals(mk(sim(s1, -0.9, -0.7), sim(s2, -0.95, -0.65)))
        ratio = [Dne[i, j] / Dc[i, j] for i in 1:n, j in 1:K if Dc[i, j] > 1e-4]
        @test minimum(ratio) > min(1 / s1, 1 / s2) - 1 && maximum(ratio) < max(1 / s1, 1 / s2) + 1
        @test maximum(ratio) / minimum(ratio) > 1.2
        # the pair method is pixels in, pixels out: SIMFIT++ with the reference's
        # pixel bounds on the pixel residuals — not on the pool's half-diagonal-unit
        # residual matrix, which the default pixel bounds do not fit
        pair = mk(sim(s, -0.9, -0.7), sim(s, -0.95, -0.65))
        @test simfitpp_select(pair, 3.0; rng = MersenneTwister(3)) ==
              simfitpp_select(pixel_residuals(pair), 3.0; rng = MersenneTwister(3))
        # the pool frame's unit stays available through the matrix method
        c = with_residual_unit(SimFitPP(), pool_pixel_scale(pair))
        @test first(simfitpp_select(Dc, 0.01, c; rng = MersenneTwister(3))) in 0:K
        @test pool_pixel_scale(pair) == s
        # the pool's normalization and pool-frame calibrations (VisualGeometryToolkitCore's)
        T1c, T2c = sim(s, -0.9, -0.7), sim(s, -0.95, -0.65)
        pc = mk(T1c, T2c)
        nz = pool_normalization(pc)
        Sn, Dn = VgtkCore.source_map(nz), VgtkCore.destination_map(nz)
        @test (to_homogeneous(Sn), to_homogeneous(Dn)) == (T1c, T2c)
        K1, K2 = calibration_matrices(pc)
        @test pool_frame_calibrations(pc) == (T1c * K1, T2c * K2)
        # a model type without a pixel mapping is refused: a bare matrix is
        # neither an H nor an F, so it has no map to compose
        @test_throws MethodError pixel_residuals(mk(T1c, T2c; ms = [I3s]))
    end
end

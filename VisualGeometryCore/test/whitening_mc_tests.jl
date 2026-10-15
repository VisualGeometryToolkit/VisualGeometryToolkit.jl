# The whitened residual currency against Monte Carlo: at the TRUE model, a
# correspondence perturbed by its measurement covariance has a whitened residual
# distributed N(0, I_{d_g}) to first order — `whiten` (vector constraint, a known
# anisotropic per-point Σx), `whitened_rows(SampsonWhitening(), …)` (scalar,
# isotropic), the frozen transfer whitening of the homography rows, and the
# heteroscedastic (per-point precision) Sampson distance.
@testitem "whitening_mc" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryCore
    using VisualGeometryCore: whiten, whitened_rows, SampsonWhitening, constraint_with_jacobians,
        transfer_whitening!, transfer_rows!, sampson_distance, Uncertain, PrecMat
    const CHar = VisualGeometryCore.Synthetic

    # the noisy correspondence i of `cs` (true pairs), its 4×4 shape scaled by σ²
    noisy(rng, c, σ, shape) = CHar.perturb_pairs(rng, [c], σ, [shape])[1]
    sv(p) = SVector{2, Float64}(p[1], p[2])

    @testset "whiten: H's algebraic rows with anisotropic per-point Σx → N(0, I₂)" begin
        σ = 0.5
        trial(rng) = begin
            H, cs = CHar.sample_homography_scene(rng; npts = 1)
            shape = CHar.make_point_covs(rng, 1; aniso = true, img2_scale = 2.0)[1]
            c = noisy(rng, cs[1], σ, shape)
            g, G_x, G_θ = constraint_with_jacobians(H, sv(c.first), sv(c.second))
            r, _, _ = whiten(g, G_x, G_θ, σ^2 * shape)
            (Vector(r), Matrix(1.0I, 2, 2))
        end
        @test CHar.is_calibrated(CHar.run_calibration(trial, MersenneTwister(91), 4000, 2))
    end

    @testset "whitened_rows(SampsonWhitening()): the epipolar row, isotropic → N(0, σ²)" begin
        σ = 0.5
        trial(rng) = begin
            _, _, F, _, cs, _ = CHar.sample_twoview_scene(rng, 1)
            c = noisy(rng, cs[1], σ, SMatrix{4, 4, Float64}(I))
            g, G_x, G_θ = constraint_with_jacobians(F, sv(c.first), sv(c.second))
            r, _ = whitened_rows(SampsonWhitening(), g, G_x, G_θ)
            ([r / σ], ones(1, 1))
        end
        @test CHar.is_calibrated(CHar.run_calibration(trial, MersenneTwister(92), 4000, 1))
    end

    @testset "transfer rows: the frozen transfer whitening at the true H → N(0, σ²I₂)" begin
        σ = 0.5
        trial(rng) = begin
            H, cs = CHar.sample_homography_scene(rng; npts = 1)
            c = noisy(rng, cs[1], σ, SMatrix{4, 4, Float64}(I))
            u₁ = [sv(c.first)]; u₂ = [sv(c.second)]
            W = Vector{SMatrix{2, 2, Float64, 4}}(undef, 1)
            transfer_whitening!(W, H, u₁)
            r = zeros(2)
            transfer_rows!(r, H, W, u₁, u₂, ones(2))
            (r / σ, Matrix(1.0I, 2, 2))
        end
        @test CHar.is_calibrated(CHar.run_calibration(trial, MersenneTwister(93), 4000, 2))
    end

    @testset "sampson_distance with per-point precisions: d² ~ χ²₁ at the true F" begin
        σ = 0.5
        trial(rng) = begin
            _, _, F, _, cs, _ = CHar.sample_twoview_scene(rng, 1)
            Σ₁ = σ^2 * CHar.make_query_cov(rng; aniso = true)
            Σ₂ = 4σ^2 * CHar.make_query_cov(rng; aniso = true)
            shape = SMatrix{4, 4, Float64}([Matrix(Σ₁) zeros(2, 2); zeros(2, 2) Matrix(Σ₂)])
            c = noisy(rng, cs[1], 1.0, shape)
            d = sampson_distance(F, Uncertain(c.first, PrecMat(inv(Σ₁))), Uncertain(c.second, PrecMat(inv(Σ₂))))
            ([d], ones(1, 1))
        end
        @test CHar.is_calibrated(CHar.run_calibration(trial, MersenneTwister(94), 4000, 1))
    end
end

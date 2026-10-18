# Chart covariances beyond the minimal two-view cells, against Monte Carlo:
# a pose covariance TRANSPORTED to another rigid chart and measured there
# (`transform_covariance` = pull ∘ push), a rigid chart's covariance PUSHED to its
# ambient space, the whitened-constraint sandwich with ANISOTROPIC per-point
# noise, and the radial camera's covariance on its sphere chart.
@testitem "chart_covariance_mc" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: implicit_covariance, transform_covariance, EuclideanManifold,
        LeftRigidManifold6, RigidExpManifold6, RigidManifold6, local_coordinates, minimal_manifold,
        RadialCamera
    const CHar = VgtkCore.Synthetic

    # a PnP20 scene, its implicit covariance in the left chart and the estimate
    function pnp_parts(rng; npts = 20, σ_px = 0.5, f = 600.0)
        pose, cs, _ = CHar.sample_pose_scene(rng, npts; f)
        shapes = [CHar.make_query_cov(rng) for _ in 1:npts]
        csn = CHar.perturb_pairs(rng, cs, σ_px / f, shapes)
        chL = LeftRigidManifold6(pose)
        Σ̃ = implicit_covariance(chL, cs; Σx = shapes)
        est = CHar._trial_estimate(nothing, chL, cs, csn)
        return pose, chL, Σ̃, est, (σ_px / f)^2
    end

    @testset "transform_covariance: PnP20's covariance in the MRP and decoupled charts" begin
        for (tochart, seed) in ((p -> RigidManifold6(RotMatrix{3, Float64}(p.R), SVector{3, Float64}(p.t)), 101),
                                (p -> RigidExpManifold6(p), 102))
            trial(rng) = begin
                pose, chL, Σ̃, est, s2 = pnp_parts(rng)
                (Σ̃ === nothing || est === nothing) && return nothing
                ch = tochart(pose)
                (Vector(local_coordinates(ch, est[1])), s2 .* Matrix(transform_covariance(chL, ch, Σ̃)))
            end
            @test CHar.is_calibrated(CHar.run_calibration(trial, MersenneTwister(seed), 400, 6))
        end
    end

    @testset "transform_covariance to the ambient chart: a rigid chart's covariance in [vec R; t]" begin
        # rank 6 in 12 dimensions: the NEES reads the pushed covariance's own
        # subspace (cond_cap), the displacement's second-order normal part aside
        trial(rng) = begin
            pose, chL, Σ̃, est, s2 = pnp_parts(rng)
            (Σ̃ === nothing || est === nothing) && return nothing
            amb = VgtkCore._ambient_coordinates          # [vec(R); t]
            (Vector(amb(est[1]) - amb(pose)), s2 .* Matrix(transform_covariance(chL, EuclideanManifold(chL), Σ̃)))
        end
        r = CHar.run_calibration(trial, MersenneTwister(103), 400, 12; cond_cap = 1.0e8)
        @test r.dof ≈ 6
        @test CHar.is_calibrated(r)
    end

    @testset "the sandwich with anisotropic per-point noise (overdetermined)" begin
        @test CHar.is_calibrated(CHar.run_calibration(CHar.homography_trial(σ = 0.5, npts = 30, aniso = true), MersenneTwister(104), 400, 8))
        # E: calibrated at 0.1 px (measured 5.04 / 5, 1200 trials); at 0.5 px the
        # anisotropy (major axis up to 1.3σ) reaches the nonlinear range — 5.3–5.6 / 5,
        # median ratio 1.05–1.16, vanishing with the noise: second order, pinned
        @test CHar.is_calibrated(CHar.run_calibration(CHar.essential_trial(σ_px = 0.1, npts = 30, aniso = true), MersenneTwister(205), 1200, 5; cond_cap = 1.0e8))
        rE = CHar.run_calibration(CHar.essential_trial(σ_px = 0.5, npts = 30, aniso = true), MersenneTwister(105), 400, 5; cond_cap = 1.0e8)
        @test rE.nees_mean < 6.5 && rE.nees_median_ratio <= 1.25
        @test CHar.is_calibrated(CHar.run_calibration(CHar.fundmat_trial(σ = 0.1, npts = 30, aniso = true, error = :ambient), MersenneTwister(106), 400, 7; cond_cap = 1.0e8))
        @test CHar.is_calibrated(CHar.run_calibration(CHar.pnp_trial(σ_px = 0.5, npts = 20, aniso = true), MersenneTwister(107), 400, 6))
    end

    @testset "the radial camera on its sphere chart (S⁵): minimal and overdetermined" begin
        # plane points → pixels by a homography with perspective; the radial camera
        # is the recentred H's first two rows. The pixel is the measurement.
        function radial_trial(npts, σ)
            return rng -> begin
                c = Point2(640.0, 480.0)
                θ = 0.4 * randn(rng)
                A = (3.0 + rand(rng)) * SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ)) + 0.3 * SMatrix{2, 2}(randn(rng, 4)...)
                H = SMatrix{3, 3}(A[1, 1], A[2, 1], 2.0e-4 * randn(rng), A[1, 2], A[2, 2], 2.0e-4 * randn(rng), c[1] + 20randn(rng), c[2] + 20randn(rng), 1.0)
                cs = map(1:npts) do i
                    X = Point2(-100 + 200rand(rng), -100 + 200rand(rng))
                    w = H * SVector(X[1], X[2], 1.0)
                    X => Point2(PerspectiveMap()(w))
                end
                P = SMatrix{2, 3}(H[1, 1] - c[1] * H[3, 1], H[2, 1] - c[2] * H[3, 1],
                    H[1, 2] - c[1] * H[3, 2], H[2, 2] - c[2] * H[3, 2],
                    H[1, 3] - c[1] * H[3, 3], H[2, 3] - c[2] * H[3, 3])
                ch = minimal_manifold(RadialCamera(P, c))
                shapes = [CHar.make_query_cov(rng; aniso = true) for _ in 1:npts]
                csn = CHar.perturb_pairs(rng, cs, σ, shapes)
                Σ̃ = implicit_covariance(ch, cs; Σx = shapes)
                Σ̃ === nothing && return nothing
                est = CHar._trial_estimate(nothing, ch, cs, csn)
                est === nothing && return nothing
                (Vector(est[2]), σ^2 .* Matrix(Symmetric(Σ̃)))
            end
        end
        # minimal (5 points): the bulk calibrated, the tail the minimal problem's own
        # curvature, as for the other minimal cells (measured: NEES 8.2 / 5 at 0.5 px,
        # 5.4 at 0.1 px, median ratio 1.02) — pinned; overdetermined: calibrated
        rmin = CHar.run_calibration(radial_trial(5, 0.5), MersenneTwister(108), 400, 5)
        @test !CHar.is_calibrated(rmin)
        @test 0.8 <= rmin.nees_median_ratio <= 1.25
        @test CHar.is_calibrated(CHar.run_calibration(radial_trial(30, 0.5), MersenneTwister(109), 400, 5))
    end
end

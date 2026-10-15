# The essential matrix on UNIT BEARINGS: the tangent constraint, its Jacobians
# against automatic differentiation through the real backprojection, and the
# property that motivates the whole form — it is defined where the z = 1 chart
# is not.
@testitem "tangent_sampson" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using FiniteDiff: FiniteDiff
    using VisualGeometryCore
    using VisualGeometryCore: EssentialMat, CameraModel, LogicalIntrinsics,
        CameraCalibrationMat, Pinhole, backproject,
        backproject_with_jacobian, constraint_with_jacobians, sampson_distance

    R = RotMatrix(RotXYZ(0.04, -0.07, 0.02)); Rm = SMatrix{3, 3}(R)
    t = normalize(SVector(1.0, 0.2, -0.15))
    E = EssentialMat(Rm, t)
    f = 800.0; c = 320.0
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(f, 0, 0, 0, f, 0, c, c, 1))
    cam = CameraModel(LogicalIntrinsics(K), Pinhole())

    rng = Xoshiro(17)
    pix1 = [SVector(120 + 400rand(rng), 120 + 400rand(rng)) for _ in 1:12]
    X = [backproject(cam, u) * 3.0 for u in pix1]
    pix2 = [SVector(Tuple(project(cam, Rm * x + t))...) for x in X]

    @testset "the constraint and G_θ are the bearing algebra" begin
        # PERTURBED on purpose: on an exact correspondence g is ~1e-17, and
        # comparing two routes to a quantity that is zero with a RELATIVE
        # tolerance tests nothing and fails on rounding. Displacing view 2 by a
        # few pixels makes g genuinely non-zero, so the equality has content.
        for i in 1:12
            f₁, M₁ = backproject_with_jacobian(cam, pix1[i])
            f₂, M₂ = backproject_with_jacobian(cam, pix2[i] + SVector(2.5, -1.75))
            g, G_x, G_θ = constraint_with_jacobians(E, f₁, f₂; M₁, M₂)
            @test abs(g) > 1.0e-6                       # the perturbation really did break the constraint
            @test g ≈ dot(f₂, SMatrix{3, 3}(E) * f₁) rtol = 1.0e-13
            # G_θ = vec(f₂ f₁ᵀ), column-major
            @test G_θ ≈ SVector{9}(vec(SVector{3}(f₂) * SVector{3}(f₁)')) rtol = 1.0e-13
            # M is tangent: this is what lets consumers skip a tangent projection
            @test abs(dot(f₁, M₁[:, 1])) < 1.0e-12
            @test abs(dot(f₁, M₁[:, 2])) < 1.0e-12
        end
    end

    @testset "G_x against central differences of the real backprojection" begin
        # The claim is that G_x is the gradient of g with respect to PIXELS,
        # which holds only if M really is ∂f/∂u. Differentiate the whole chain
        # numerically rather than checking the formula against itself.
        # (Central differences, not ForwardDiff: `backproject` is typed to
        # `<:AbstractFloat`, so a Dual cannot flow through it — several lenses
        # invert by an iterative solve. `backproject_with_jacobian` therefore
        # differentiates the FORWARD map, which is AD-safe, and inverts the
        # 2×2 restriction; this test is what pins that inversion.)
        gof(z) = dot(backproject(cam, SVector(z[3], z[4])),
            SMatrix{3, 3}(E) * backproject(cam, SVector(z[1], z[2])))
        for i in (1, 5, 9)
            z0 = SVector(pix1[i][1], pix1[i][2], pix2[i][1], pix2[i][2])
            fd = FiniteDiff.finite_difference_gradient(gof, Vector(z0))
            f₁, M₁ = backproject_with_jacobian(cam, pix1[i])
            f₂, M₂ = backproject_with_jacobian(cam, pix2[i])
            _, G_x, _ = constraint_with_jacobians(E, f₁, f₂; M₁, M₂)
            @test G_x ≈ fd rtol = 1.0e-6
        end
    end

    @testset "exact correspondences give a zero residual" begin
        for i in 1:12
            f₁, M₁ = backproject_with_jacobian(cam, pix1[i])
            f₂, M₂ = backproject_with_jacobian(cam, pix2[i])
            @test sampson_distance(E, f₁, f₂; M₁, M₂) < 1.0e-9
        end
    end

    @testset "the residual is in PIXELS, as a JOINT-space distance" begin
        # Sampson is the orthogonal distance to the constraint surface in the
        # JOINT 4-D measurement space (u₁, u₂), so it shares the correction
        # between the two views. Displacing ONE endpoint by δ px therefore reads
        # back δ·‖G_x^(2)‖/‖G_x‖, NOT δ — for a symmetric pair that factor is
        # ≈ 1/√2, which is exactly what this measured before the expectation was
        # corrected. The pixel claim is that the SCALE is pixels, and this pins
        # it against the closed form rather than against a guess.
        i = 3
        f₁, M₁ = backproject_with_jacobian(cam, pix1[i])
        f₂₀, M₂₀ = backproject_with_jacobian(cam, pix2[i])
        _, G_x0, _ = constraint_with_jacobians(E, f₁, f₂₀; M₁, M₂ = M₂₀)
        dir = normalize(SVector(G_x0[3], G_x0[4]))
        share = norm(SVector(G_x0[3], G_x0[4])) / norm(G_x0)
        @test 0.1 < share < 0.95                    # both views genuinely contribute
        for δ in (0.25, 1.0, 4.0)
            f₂d, M₂d = backproject_with_jacobian(cam, pix2[i] + δ * dir)
            d = sampson_distance(E, f₁, f₂d; M₁, M₂ = M₂d)
            @test isapprox(d, δ * share; rtol = 0.05)
        end
        # And the scale IS linear in pixels: doubling the displacement doubles it.
        f_a, M_a = backproject_with_jacobian(cam, pix2[i] + 1.0 * dir)
        f_b, M_b = backproject_with_jacobian(cam, pix2[i] + 2.0 * dir)
        d_a = sampson_distance(E, f₁, f_a; M₁, M₂ = M_a)
        d_b = sampson_distance(E, f₁, f_b; M₁, M₂ = M_b)
        @test isapprox(d_b / d_a, 2.0; rtol = 0.02)
    end

    @testset "defined past 90°, where the z = 1 chart is not" begin
        # The reason the tangent form exists. A fisheye ray at more than 90°
        # from the optical axis has NO gnomonic chart coordinate at all; the
        # bearing constraint is finite and ordinary there.
        fish = CameraModel(LogicalIntrinsics(K), Lens(Equidistant()))
        u_wide = SVector(c + 1.45 * f, c)          # θ ≈ 83°… past the chart's reach when lifted
        b = backproject(fish, u_wide)
        @test all(isfinite, b)
        f₁, M₁ = backproject_with_jacobian(fish, u_wide)
        f₂, M₂ = backproject_with_jacobian(fish, SVector(c + 30.0, c + 12.0))
        g, G_x, _ = constraint_with_jacobians(E, f₁, f₂; M₁, M₂)
        @test isfinite(g) && all(isfinite, G_x)
        @test isfinite(sampson_distance(E, f₁, f₂; M₁, M₂))
    end

    @testset "caller-supplied Σ scales the residual as a covariance should" begin
        i = 7
        f₁, M₁ = backproject_with_jacobian(cam, pix1[i])
        u2 = pix2[i] + SVector(0.8, -0.3)
        f₂, M₂ = backproject_with_jacobian(cam, u2)
        d1 = sampson_distance(E, f₁, f₂; M₁, M₂)
        d4 = sampson_distance(E, f₁, f₂; M₁, M₂, Σ = SMatrix{4, 4, Float64}(4I))
        @test d4 ≈ d1 / 2 rtol = 1.0e-12      # σ doubled → distance halved
    end
end

# Ray-domain triangulation and cheirality. The point of the ray form is that it
# is defined where the z = 1 chart is not, and that cheirality is a statement
# about ray depths rather than about a z coordinate the ray domain does not
# privilege.
@testitem "bearing_triangulation" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EuclideanMap, CameraModel, LogicalIntrinsics,
        CameraCalibrationMat, Pinhole, backproject,
        triangulate, test_cheirality, cheiral_pose_fast

    R = RotMatrix(RotXYZ(0.03, -0.05, 0.02)); Rm = SMatrix{3, 3}(R)
    t = SVector(0.8, 0.05, -0.02)
    pose = EuclideanMap(RotMatrix{3, Float64}(Rm), t)

    @testset "exact recovery of a known point, and both depths" begin
        rng = Xoshiro(3)
        for _ in 1:200
            X = SVector(2.0 * randn(rng), 2.0 * randn(rng), 4 + 2rand(rng))
            X2 = Rm * X + t
            X2[3] > 0.2 || continue
            f₁ = normalize(X)
            f₂ = normalize(X2)
            Xh, λ₁, λ₂ = triangulate(pose, f₁, f₂)
            @test norm(SVector(Xh[1], Xh[2], Xh[3]) - X) / norm(X) < 1.0e-12
            # the depths are the ray parameters, so they are the distances
            @test λ₁ ≈ norm(X) rtol = 1.0e-12
            @test λ₂ ≈ norm(X2) rtol = 1.0e-12
            @test test_cheirality(pose, f₁, f₂)
        end
    end

    @testset "cheirality is a RAY test, not a z test" begin
        # A point BEHIND camera 1 has a bearing pointing away from it. The ray
        # depth goes negative, which is the honest signal; a z-coordinate test
        # on the triangulated point would be testing the wrong thing for any
        # camera that sees past 90°.
        X = SVector(0.3, 0.1, -3.0)            # behind view 1
        X2 = Rm * X + t
        f₁ = normalize(X); f₂ = normalize(X2)
        _, λ₁, _ = triangulate(pose, f₁, f₂)
        @test λ₁ > 0                            # the BEARING points at it, so its own depth is positive
        # …but the bearing of the opposite ray must fail cheirality:
        @test !test_cheirality(pose, -f₁, f₂)
    end

    @testset "parallel rays are NaN, not an exception or a silent number" begin
        # Zero parallax: the rays never meet. The honest answer is "nothing was
        # triangulated", and the caller decides — not a throw, and not a
        # plausible-looking point.
        f = normalize(SVector(0.1, -0.05, 1.0))
        pure_translation_along_ray = EuclideanMap(RotMatrix{3, Float64}(SMatrix{3, 3, Float64}(I)), 0.5 * f)
        Xh, λ₁, λ₂ = triangulate(pure_translation_along_ray, f, f)
        @test isnan(λ₁) && isnan(λ₂)
        @test all(isnan, (Xh[1], Xh[2], Xh[3]))
        @test !test_cheirality(pure_translation_along_ray, f, f)
    end

    @testset "works past 90°, where a z = 1 chart has no coordinate" begin
        # The reason the ray form exists: a fisheye ray at more than 90° from
        # the optical axis cannot be written in the gnomonic chart at all, but
        # it triangulates here like any other.
        f = 800.0; c = 320.0
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(f, 0, 0, 0, f, 0, c, c, 1))
        fish = CameraModel(LogicalIntrinsics(K), Lens(Equidistant()))
        # an equidistant lens has r = θ, so the pixel offset IS f·θ: 1.75 rad
        # is 100°, genuinely past the chart's reach. (1.5 rad is 85.9° — under
        # 90°, and an earlier version of this test used it and proved nothing.)
        u_wide = SVector(c + 1.75 * f, c)
        f₁ = backproject(fish, u_wide)
        @test all(isfinite, f₁)
        @test f₁[3] < 0                                    # behind the z = 1 plane: no chart coordinate exists
        # place a real point on that ray and see it recovered
        X = 5.0 * SVector(f₁[1], f₁[2], f₁[3])
        X2 = Rm * X + t
        f₂ = normalize(X2)
        Xh, λ₁, λ₂ = triangulate(pose, SVector(f₁[1], f₁[2], f₁[3]), f₂)
        @test norm(SVector(Xh[1], Xh[2], Xh[3]) - X) / norm(X) < 1.0e-10
        @test λ₁ > 0 && λ₂ > 0
        @test test_cheirality(pose, SVector(f₁[1], f₁[2], f₁[3]), f₂)
    end

    @testset "cheiral_pose_fast: the vote recovers the true motion" begin
        rng = Xoshiro(9)
        tu = normalize(t)
        E = EssentialMat{Float64}(Tuple(skew(tu) * Rm))
        f₁ = SVector{3, Float64}[]; f₂ = SVector{3, Float64}[]
        while length(f₁) < 40
            X = SVector(2.0 * randn(rng), 2.0 * randn(rng), 4 + 2rand(rng))
            X2 = Rm * X + t
            X2[3] > 0.2 || continue
            push!(f₁, normalize(X)); push!(f₂, normalize(X2))
        end
        Rv, tv = cheiral_pose_fast(E, f₁, f₂)
        @test Rv isa SMatrix{3, 3, Float64, 9}
        @test norm(Rv - Rm) < 1.0e-9
        @test norm(tv - tu) < 1.0e-9
        # every pair is in front under the winner
        @test all(test_cheirality(EuclideanMap(Rv, tv), a, b) for (a, b) in zip(f₁, f₂))
        @test_throws DimensionMismatch cheiral_pose_fast(E, f₁, f₂[1:5])
    end
end

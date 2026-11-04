# RadialHomography{T, N}: a plane homography and a projective N-coefficient
# division camera — the N = 2 round trip, the N = 1 equivalence of the camera
# constructor with the scalar one, the frame law and the sl(3) ⊕ lens chart
# for N = 2, the refused scalar accessors, Float32, and the camera bridge.
@testitem "radial_homography_n" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using CoordinateTransformations: AffineMap
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: RadialHomography, radial_transfer, radial_ray, radial_transfer_rows!,
        radial_transfer_jacobian_rows!, retract, retract_jacobian, tangent_size, minimal_manifold,
        local_coordinates, ambient_coordinates, project, backproject, division_coeff, division_coeffs,
        homography, center, convert_eltype, rescale_radial_coeff, retract_directions, HomographyManifold8

    rng = Xoshiro(20261104)
    c = Point2(640.0, 480.0)
    λ = (-3.0e-7, 2.0e-13)
    # a seeded, well-conditioned pixel homography: near a similarity, small perspective
    A = SMatrix{3, 3, Float64}(
        0.9 + 0.05randn(rng), 0.02randn(rng), 1.0e-5randn(rng),
        0.02randn(rng), 0.9 + 0.05randn(rng), 1.0e-5randn(rng),
        100.0 + 10randn(rng), 60.0 + 10randn(rng), 1.0
    )
    H = HomographyMat(A)
    cam2 = CameraModel(ProjectiveIntrinsics(c), Division(λ...))
    m = RadialHomography(H, cam2)
    board = [SVector(1000rand(rng), 800rand(rng)) for _ in 1:50]

    @testset "N = 2: the type, the camera, the accessors" begin
        @test m isa RadialHomography{Float64, 2} && m isa RadialHomography{Float64}
        @test isbitstype(typeof(m))
        @test homography(m) == H && center(m) == c && division_coeffs(m) === λ
        @test CameraModel(m) == CameraModel(ProjectiveIntrinsics(c), Division(λ...))
        @test CameraModel(m) isa CameraModel{<:ProjectiveIntrinsics}
        @test RadialHomography(H, CameraModel(m)) == m
        # Review Focus 5: no silent λ₁
        @test_throws ArgumentError division_coeff(m)
        @test_throws ArgumentError m.k
        @test m.center == c
        @test propertynames(m) == (:H, :λ, :center)
        @test :k in propertynames(RadialHomography(H, -3.0e-7, c))
    end

    @testset "N = 2 round trip: backproject ∘ radial_transfer ∥ T₋c·H·s̃" begin
        for s in board
            d = radial_transfer(m, s)
            @test d !== nothing
            ℓ = backproject(m, d)
            r = radial_ray(m, s)
            @test norm(cross(SVector{3, Float64}(ℓ), r)) / (norm(ℓ) * norm(r)) <= 1.0e-12
        end
        # the lift is 1 + λ₁ρ² + λ₂ρ⁴
        d = SVector(900.0, 700.0)
        ρ² = (d[1] - c[1])^2 + (d[2] - c[2])^2
        @test backproject(m, d)[3] ≈ 1 + λ[1] * ρ² + λ[2] * ρ²^2 rtol = 1.0e-15
    end

    @testset "N = 2 refusals: a back ray and a ray past the fold" begin
        # λ₂ > 0 keeps the lift's third component positive (1 − 3e-7ρ² + 2e-13ρ⁴ ≥ 0.89),
        # so this lens images no ray behind 90° and its view angle folds before 90°.
        z0 = Point2(0.0, 0.0)
        back = RadialHomography{Float64}(HomographyMat(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, -1)), λ, z0)
        grazing = RadialHomography{Float64}(HomographyMat(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1.0e-4)), λ, z0)
        s = SVector(1000.0, 300.0)
        for mr in (back, grazing)
            @test project(mr, radial_ray(mr, s)) === nothing
            @test radial_transfer(mr, s) === nothing
            @test !VgtkCore.radial_in_front(mr, s)
            @test all(isnan, transform_deriv(mr, radial_ray(mr, s)))
        end
        @test radial_ray(back, s)[3] < 0
        # an in-field front ray: both accept it
        @test VgtkCore.radial_in_front(m, board[1]) && radial_transfer(m, board[1]) !== nothing
    end

    @testset "N = 1: the camera constructor is the scalar one" begin
        k = -3.0e-7
        m1 = RadialHomography(H, k, c)
        m1c = RadialHomography(H, CameraModel(ProjectiveIntrinsics(c), Division(k)))
        @test m1 isa RadialHomography{Float64, 1} && m1c isa RadialHomography{Float64, 1}
        @test m1.k == k == division_coeff(m1) && division_coeffs(m1) == (k,)
        for s in board
            ray = radial_ray(m1, s)
            d = radial_transfer(m1, s)
            @test d == radial_transfer(m1c, s)
            @test project(m1, ray) == project(m1c, ray)
            @test backproject(m1, d) == backproject(m1c, d)
            @test transform_deriv(m1, ray) == transform_deriv(m1c, ray)
            @test transform_deriv_params(m1, ray) == transform_deriv_params(m1c, ray)
        end
    end

    @testset "N = 2 derivatives: ray, coefficients, LM rows" begin
        for s in board[1:10]
            ray = radial_ray(m, s)
            @test transform_deriv(m, ray) ≈ ForwardDiff.jacobian(r -> SVector(project(m, r)...), ray) rtol = 1.0e-8
            Jλ = transform_deriv_params(m, ray)
            @test Jλ isa SMatrix{2, 2, Float64}
            fd = ForwardDiff.jacobian(SVector(λ...)) do q
                D = eltype(q)
                SVector(project(RadialHomography{D}(HomographyMat{D}(Tuple(H)), Tuple(q), Point2{D}(c)), ray)...)
            end
            @test Jλ ≈ fd rtol = 1.0e-8
        end
        n = length(board)
        u₂ = [radial_transfer(m, s) + 0.5 * SVector(randn(rng), randn(rng)) for s in board]
        sw = ones(2n)
        ch = HomographyManifold8(SMatrix{3, 3}(H))
        v = SVector(1.0e-3, -2.0e-3, 5.0e-4, 1.0e-3, -1.0e-4, 2.0e-4, 1.0e-6, -2.0e-6)
        dH = retract_directions(ch, v)
        mv = RadialHomography{Float64}(retract(ch, v), λ, c)
        J = zeros(2n, 10)
        radial_transfer_jacobian_rows!(J, mv, dH, board, sw)
        rv(w) = (out = zeros(2n); radial_transfer_rows!(out, RadialHomography{Float64}(retract(ch, w[SOneTo(8)]), (λ[1] + w[9], λ[2] + w[10]), c), board, u₂, sw); out)
        S = SVector(ntuple(q -> q == 9 ? abs(λ[1]) : q == 10 ? abs(λ[2]) : 1.0, 10))
        fd = FiniteDiff.finite_difference_jacobian(z -> rv(S .* z), vcat(v, 0.0, 0.0) ./ S, Val(:central)) ./ transpose(S)
        @test J ≈ fd rtol = 1.0e-5 atol = 1.0e-6 * maximum(abs, J)
    end

    @testset "N = 2 frame law: D ∘ m ∘ inv(S)" begin
        σ = 2 / hypot(1280.0, 960.0)
        D = AffineMap(SMatrix{2, 2}(σ, 0.0, 0.0, σ), SVector(-σ * c[1], -σ * c[2]))
        S = AffineMap(SMatrix{2, 2}(0.01, 0.002, -0.002, 0.01), SVector(-5.0, -4.0))
        mn = D ∘ m ∘ inv(S)
        @test mn isa RadialHomography{Float64, 2}
        @test division_coeffs(mn) == rescale_radial_coeff.(λ, inv(σ), (1, 2))
        for s in board
            @test norm(radial_transfer(mn, S(s)) - D(radial_transfer(m, s))) / σ <= 1.0e-9
        end
    end

    @testset "minimal_manifold: sl(3) ⊕ the lens chart" begin
        ch = minimal_manifold(m)
        @test tangent_size(ch) == 10
        @test retract(ch, zeros(10)) == m
        w = SVector{10}(1.0e-3 * randn(rng, 8)..., 1.0e-8, 1.0e-14)
        @test retract(ch, w) isa RadialHomography{Float64, 2}
        @test local_coordinates(ch, retract(ch, w)) ≈ w rtol = 1.0e-8
        Jfd = ForwardDiff.jacobian(z -> ambient_coordinates(retract(ch, z)), w)
        @test maximum(abs, retract_jacobian(ch, w) - Jfd) <= 1.0e-8
    end

    @testset "Float32 (Review Focus 1)" begin
        H32 = HomographyMat{Float32}(Tuple(A))
        m32 = RadialHomography(H32, -3.0f-7, Point2f(640, 480))
        @test m32 isa RadialHomography{Float32, 1}
        d = radial_transfer(m32, SVector(500.0f0, 400.0f0))
        @test d isa Point2{Float32} && all(isfinite, d)
        m64 = RadialHomography{Float64}(HomographyMat{Float64}(Tuple(H32)), Float64(-3.0f-7), Point2{Float64}(640, 480))
        @test convert_eltype(Float64, m32) == m64
    end

    @testset "extents of a projective camera (R10)" begin
        sz = (1280, 960)
        outer = undistorted_rect(m, sz)
        inner = valid_rect(m, sz)
        @test all(isfinite, outer.widths) && all(>(0), inner.widths)
        @test all(inner.origin .>= outer.origin .- 1.0e-6)
        @test outer == undistorted_rect(CameraModel(m), sz)
        @test division_bounded(m, sz) == division_bounded(CameraModel(m), sz)
        @test_throws ArgumentError undistorted_rect(m, sz; max_fov = deg2rad(70.0))
        @test_throws ArgumentError undistorted_rect(RadialHomography(H, -3.0e-7, c), sz; max_fov = deg2rad(70.0))
    end
end

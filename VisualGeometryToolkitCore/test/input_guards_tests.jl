# Inputs at the edge of each function's domain: the minimum correspondence
# count of the DLTs, over-long index masks, non-finite sample positions, dense
# covariances, 3-vector bearings handed to 2-vector forms, the tangent Sampson
# degeneracy rule under rescaling, and non-finite rays in the grid solver.

@testitem "input_guards" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: HomographyMat, EssentialMat, SvdWorkspace, homography_dlt,
        fundamental_dlt, dlt_conditioning, whiten, whitening_factor, signed_sampson_distance,
        sampson_rss, sampson_distance, CentralGenericModel, CameraModel,
        Division
    using StaticArrays, LinearAlgebra, Random

    rng = MersenneTwister(3)
    homog(x) = SVector(x[1], x[2], 1.0)

    H0 = SMatrix{3, 3}(1.1, 0.05, 1.0e-4, -0.1, 0.9, 2.0e-4, 30.0, -20.0, 1.0)
    xs = [SVector(640rand(rng), 480rand(rng)) for _ in 1:9]
    hcs = [x => PerspectiveMap()(H0 * homog(x)) for x in xs]
    herr(H, cs) = maximum(norm(PerspectiveMap()(SMatrix{3, 3}(H) * homog(x)) - y) for (x, y) in cs)

    K = SMatrix{3, 3}(500.0, 0, 0, 0, 500.0, 0, 320, 240, 1)
    c, s = cos(0.1), sin(0.1)
    R = SMatrix{3, 3}(c, 0, -s, 0, 1, 0, s, 0, c)
    t = SVector(1.0, 0.2, 0.1)
    Xs = [SVector(randn(rng), randn(rng), 5 + rand(rng)) for _ in 1:9]
    fcs = [PerspectiveMap()(K * X) => PerspectiveMap()(K * (R * X + t)) for X in Xs]
    ferr(F, cs) = maximum(
        abs(homog(y)' * SMatrix{3, 3}(F) * homog(x)) / norm(SMatrix{3, 3}(F)) for (x, y) in cs
    )

    @testset "DLT at the minimum count is exact" begin
        ws = SvdWorkspace{Float64}(100, 9)
        @test herr(homography_dlt(hcs[1:4]), hcs[1:4]) < 1.0e-8
        @test herr(homography_dlt(hcs[1:4]; svd_ws = ws), hcs[1:4]) < 1.0e-8
        @test isfinite(dlt_conditioning(ws))
        # a reused workspace: a 5-point solve, then a 4-point one, must not
        # hand back the first solve's null vector
        homography_dlt(hcs[5:9]; svd_ws = ws)
        @test herr(homography_dlt(hcs[1:4]; svd_ws = ws), hcs[1:4]) < 1.0e-8
        @test ferr(fundamental_dlt(fcs[1:8]), fcs[1:8]) < 1.0e-10
        @test ferr(fundamental_dlt(fcs[1:8]; svd_ws = ws), fcs[1:8]) < 1.0e-10
    end

    @testset "SvdWorkspace is Float64; other eltypes take the allocating SVD" begin
        # the workspace calls dgesdd, which is double precision only
        @test_throws MethodError SvdWorkspace{Float32}(100, 9)
        ws = SvdWorkspace{Float64}(100, 9)
        hcs32 = [SVector{2, Float32}(x) => SVector{2, Float32}(y) for (x, y) in hcs[1:6]]
        H32 = homography_dlt(hcs32; svd_ws = ws)
        @test eltype(H32) == Float32
        @test herr(H32, hcs[1:6]) < 1.0e-1
    end

    @testset "an index mask longer than the data is refused" begin
        @test_throws DimensionMismatch homography_dlt(hcs[1:5]; mask = [1, 2, 3, 4, 5, 1, 2, 3])
        @test_throws DimensionMismatch fundamental_dlt(fcs[1:8]; mask = [1:8; 1:3])
    end

    @testset "non-finite sample positions" begin
        img = rand(rng, UInt8, 4, 4)
        sample = VgtkCore._bilinear_sample
        @test sample(img, NaN, 2.0, 7.0f0, 4, 4) === 7.0f0
        @test sample(img, 2.0, Inf, 7.0f0, 4, 4) === 7.0f0
        @test sample(img, 1.0e30, 2.0, 7.0f0, 4, 4) === 7.0f0
        @test sample(img, 2.0, 3.0, 7.0f0, 4, 4) == Float32(img[2, 3])

        depth_at = VgtkCore.Cameras._depth_at
        D = reshape(collect(1.0:9.0), 3, 3)
        @test isnan(depth_at(D, SVector(NaN, 1.0)))
        @test isnan(depth_at(D, SVector(1.0e30, 1.0)))
        @test isnan(depth_at(D, SVector(3.0, 0.0)))
        @test depth_at(D, SVector(2.9, 0.0)) == D[1, 3]
    end

    @testset "whiten accepts a dense measurement covariance" begin
        g = SVector(1.0, 2.0)
        Gx = SMatrix{2, 4}(1, 0.2, -0.3, 1.1, 0.5, 0.4, 0, 0.9)
        Gθ = SMatrix{2, 3}(1, 0, 0.5, 2, -1, 0.3)
        Σ = Diagonal([1, 2, 0.5, 1.5])
        r = whiten(g, Gx, Gθ, SMatrix{4, 4}(Σ))[1]
        @test whiten(g, Gx, Gθ, Σ)[1] ≈ r
        @test whiten(g, Gx, Gθ, Matrix(Σ))[1] ≈ r
        @test whitening_factor(Matrix(Gx)) ≈ whitening_factor(Gx)
    end

    @testset "2-vector epipolar forms refuse a 3-vector bearing" begin
        E = EssentialMat(R, t)
        X = Xs[1]
        f₁ = normalize(X)
        f₂ = normalize(R * X + t)
        # 3-vectors reach the bearing form, which needs the pixel Jacobians M₁, M₂
        @test_throws UndefKeywordError signed_sampson_distance(E, f₁, f₂)
        @test_throws MethodError sampson_rss(E, [f₁], [f₂])
        u₁, u₂ = PerspectiveMap()(X), PerspectiveMap()(R * X + t)
        @test abs(signed_sampson_distance(E, u₁, u₂)) < 1.0e-12
        # a Float32 model scores Float64 points
        H32 = HomographyMat{Float32}(Tuple(Float32.(H0)))
        @test sampson_rss(H32, first.(hcs), last.(hcs)) < 1.0e-3
    end

    @testset "tangent Sampson degeneracy is scale-free" begin
        E = EssentialMat(R, t)
        M = SMatrix{3, 2}(1 / 500, 0, 0, 0, 1 / 500, 0)
        X = Xs[2]
        f₁ = normalize(X)
        f₂ = normalize(R * X + t + SVector(0.01, -0.02, 0.0))   # off the constraint
        E7 = EssentialMat{Float64}(Tuple(1.0e-7 * SMatrix{3, 3}(E)))
        d = sampson_distance(E, f₁, f₂; M₁ = M, M₂ = M)
        @test isfinite(d) && d > 1
        # the old absolute eps called every pair of a rescaled E degenerate (Inf)
        @test sampson_distance(E7, f₁, f₂; M₁ = M, M₂ = M) ≈ d
    end

    @testset "grid model: non-finite rays and bounded-domain lenses" begin
        mk(f, lens) = CameraModel(; focal = f, pp = (640, 480), lens = lens)
        ge = CentralGenericModel(mk(250.0, Lens(Equidistant())), (1280, 960))
        @test all(v -> all(isfinite, v), ge.grid)
        @test VgtkCore.project(ge, SVector(0.0, 0.0, 1.0)) ≈ SVector(640.0, 480.0) atol = 1.0e-6

        gd = CentralGenericModel(mk(300.0, Division(-0.3)), (1280, 960))
        d = normalize(SVector(0.1, 0.05, 1.0))
        p = VgtkCore.project(gd, d)
        @test p !== nothing
        @test VgtkCore.project(gd, 1.0e-170 * d) ≈ p
        @test VgtkCore.project(gd, 1.0e160 * d) ≈ p
        @test VgtkCore.project(gd, SVector(NaN, 0.0, 1.0)) === nothing
    end
end

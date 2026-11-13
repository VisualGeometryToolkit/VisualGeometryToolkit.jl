@testitem "target_homographies" begin
    using VisualGeometryToolkitCore: thrown_message
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: radial_homography, homography, geom, relabel, unlabel
    using LinearAlgebra: norm
    using StaticArrays

    # TargetHomographies (captures.jl): the projective TargetPoses — per frame,
    # each target's plane → pixel HomographyMap, stored unlabeled.

    mk(a) = HomographyMap(HomographyMat(SMatrix{3, 3, Float64}(a, 0, 0, 0, a, 0, 1, 2, 1)))
    Hu = typeof(mk(1.0))
    H1, H2, H3 = mk(1.0), mk(2.0), mk(3.0)
    X = EuclideanMap(one(SMatrix{3, 3, Float64}), SVector(1.0, 2.0, 3.0))
    frames = Dict(1 => Dict(:b1 => H1, :b2 => H2), 3 => Dict(:b1 => H3))
    kw = (images = Dict(1 => "a.png", 3 => "c.png"), sensors = "cam")

    tg = TargetHomographies{PixelFrame{:colmap}}(frames; kw...)
    tp = TargetPoses{CameraFrame}(Dict(1 => Dict(:b1 => X, :b2 => X), 3 => Dict(:b1 => X)); kw...)

    @testset "Dict surface mirrors TargetPoses" begin
        @test tg isa AbstractDict{Int, Dict{Symbol, Hu}}
        @test length(tg) == length(tp) == 2
        @test sort!(collect(keys(tg))) == sort!(collect(keys(tp))) == [1, 3]
        @test tg[1][:b1] == H1 && tg[1][:b2] == H2 && tg[3][:b1] == H3
        @test Set(first.(collect(tg))) == Set(first.(collect(tp)))
        @test haskey(tg, 3) && !haskey(tg, 2)
        @test tg.images == tp.images && tg.sensors == tp.sensors == Dict(1 => "cam", 3 => "cam")
        @test tg == TargetHomographies{PixelFrame{:colmap}}(frames; kw...)
        @test tg != TargetHomographies{PixelFrame{:colmap}}(frames; images = Dict(1 => "z.png"), sensors = "cam")
    end

    @testset "pixel frame and targets are part of the value" begin
        @test tg isa TargetHomographies{PixelFrame{:colmap}}
        @test tg.targets == String[]
        ids = [repeat("a", 64), repeat("b", 64)]
        cv = TargetHomographies{PixelFrame{:opencv}}(frames; kw..., targets = ids)
        @test cv isa TargetHomographies{PixelFrame{:opencv}, Hu} && cv.targets == ids
        @test cv != tg                                              # another frame, another observation
        @test cv == TargetHomographies{PixelFrame{:opencv}}(frames; kw..., targets = ids)
        @test cv != TargetHomographies{PixelFrame{:opencv}}(frames; kw...)   # other targets
        @test_throws ArgumentError TargetHomographies{PixelFrame{:opencv}}(frames; kw..., targets = [ids[1], ids[1]])
        @test_throws MethodError TargetHomographies(frames; kw...)         # the pixel frame is always stated
    end

    @testset "labels: stored unlabeled, minted by homography(th, t, id), as TargetPoses" begin
        F = PixelFrame{:colmap}
        h = homography(tg, 1, :b1)
        @test h isa HomographyMap{3, Float64, HomographyMat{Float64}, TargetPlaneFrame, F} && to_homogeneous(h) == to_homogeneous(H1)
        labeled = TargetHomographies{F}(Dict(1 => Dict(:b1 => h)); sensors = "cam")       # its own labels: accepted, stored unlabeled
        @test labeled[1][:b1] === H1
        @test_throws ArgumentError TargetHomographies{F}(Dict(1 => Dict(:b2 => relabel(PixelFrame{:opencv}, F, unlabel(h)))))   # other labels
        @test_throws ArgumentError TargetHomographies{PixelFrame{:opencv}}(Dict(1 => Dict(:b1 => h)))   # another pixel frame
        @test_throws ArgumentError TargetHomographies{F}(Dict(1 => Dict(:b1 => to_homogeneous(H1))))     # a matrix: lift it
        Σ = CovMat{8}(σ = 0.01)
        u = TargetHomographies{F}(Dict(1 => Dict(:b1 => Uncertain(h, Σ))))
        @test geom(u[1][:b1]) === H1 && homography(u, 1, :b1) isa Uncertain{<:HomographyMap{3, Float64, HomographyMat{Float64}, TargetPlaneFrame, F}}
    end

    @testset "the parameter protocol: nine canonical entries" begin
        P = VisualGeometryToolkitCore
        H = mk(2.0)
        θ = P.parameter_vector(H)
        M = to_homogeneous(H)
        @test θ ≈ vec(permutedims(M)) ./ norm(M) && abs(norm(θ) - 1) < 1.0e-15 && θ[9] > 0
        @test P.parameter_names(H) == (:h11, :h12, :h13, :h21, :h22, :h23, :h31, :h32, :h33)
        back = P.with_parameters(H, θ)
        @test P.parameter_vector(back) == θ                                # canonical: returned as held, bit for bit
        @test to_homogeneous(back) == SMatrix{3, 3}(θ[1], θ[4], θ[7], θ[2], θ[5], θ[8], θ[3], θ[6], θ[9])
        @test P.parameter_vector(HomographyMap(HomographyMat(-to_homogeneous(H)))) ≈ θ   # the sign rule: H[3,3] ≥ 0
        @test !any(P.length_mask(H)) && length(P.length_mask(H)) == 9
        lab = P.with_parameters(homography(tg, 1, :b1), θ)                  # the frames are kept
        @test lab isa HomographyMap{3, Float64, HomographyMat{Float64}, TargetPlaneFrame, PixelFrame{:colmap}}
        @test_throws ArgumentError P.with_parameters(H, θ[1:8])
        @test_throws ArgumentError P.with_parameters(H, zeros(9))           # singular
        @test_throws ArgumentError P.parameter_vector(HomographyMap(SMatrix{3, 3, Float64}(fill(NaN, 9)...)))
        @test VisualGeometryToolkitCore.tag_of(VisualGeometryToolkitCore.PLANAR_MAP_TAGS, H) == "homography_map"
        @test VisualGeometryToolkitCore.PLANAR_MAP_TAGS["homography_map"] === HomographyMap
    end

    @testset "radial_homography" begin
        cam = CameraModel(ProjectiveIntrinsics(SVector(10.0, 20.0)), Division(-1.0e-5))
        @test radial_homography(cam, tg[1][:b1]) == RadialHomography(HomographyMat(to_homogeneous(H1)), cam)
        @test radial_homography(cam, homography(tg, 1, :b1)) == radial_homography(cam, tg[1][:b1])
    end

    @testset "radial_homography of a metric camera" begin
        using Random, LinearAlgebra
        f, c = 2455.46, SVector(3000.0, 2000.0)
        K = SMatrix{3, 3, Float64}(f, 0, 0, 0, f, 0, c[1], c[2], 1)
        cam = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K)), Division(-0.183, 0.0411))
        rng = MersenneTwister(7)
        R = SMatrix{3, 3, Float64}(Matrix(qr(randn(rng, 3, 3)).Q))
        R = det(R) < 0 ? -R : R
        t = SVector(30.0, -20.0, 1500.0)
        H = HomographyMat(K * hcat(R[:, 1], R[:, 2], t))
        m = radial_homography(cam, H)
        @test m isa RadialHomography{Float64, 2}
        @test collect(VisualGeometryToolkitCore.division_coeffs(m)) ≈ [-0.183 / f^2, 0.0411 / f^4]
        for _ in 1:20
            X = 100 .* (rand(rng, 2) .- 0.5)
            u = project(cam, R * SVector(X[1], X[2], 0.0) + t)
            @test norm(radial_transfer(m, SVector(X[1], X[2])) - SVector(u[1], u[2])) < 1.0e-9
        end
        skewed = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(f, 0, 0, 1.0, f, 0, c[1], c[2], 1))), Division(-0.1))
        aniso = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(f, 0, 0, 0, 1.1f, 0, c[1], c[2], 1))), Division(-0.1))
        @test_throws ArgumentError radial_homography(skewed, H)
        @test_throws ArgumentError radial_homography(aniso, H)
    end

    @testset "validation refuses what TargetPoses refuses, with its messages" begin
        for (a, b) in (
                (() -> TargetHomographies{PixelFrame{:colmap}}(frames; images = Dict(2 => "b.png")), () -> TargetPoses{CameraFrame}(Dict(1 => Dict(:b => X)); images = Dict(2 => "b.png"))),
                (() -> TargetHomographies{PixelFrame{:colmap}}(frames; times = Dict(1 => 0.0)), () -> TargetPoses{CameraFrame}(Dict(1 => Dict(:b => X)); times = Dict(1 => 0.0))),
                (() -> TargetHomographies{PixelFrame{:colmap}}(frames; sensors = Dict(7 => "cam")), () -> TargetPoses{CameraFrame}(Dict(1 => Dict(:b => X)); sensors = Dict(7 => "cam"))),
            )
            @test !isempty(thrown_message(a)) && thrown_message(a) == thrown_message(b)
        end
        @test TargetHomographies{PixelFrame{:colmap}}(frames; times = Dict(1 => 0.5), time_origin = "2026-05-27T00:00:00").times == Dict(1 => 0.5)
        @test isempty(TargetHomographies{PixelFrame{:colmap}}(Dict{Int, Dict{Symbol, Hu}}()))
    end
end

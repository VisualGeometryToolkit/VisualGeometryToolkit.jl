@testitem "target_homographies" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: radial_homography
    using StaticArrays

    # TargetHomographies (captures.jl): the projective TargetPoses — per frame,
    # each target's plane → pixel HomographyMat.

    mk(a) = HomographyMat(SMatrix{3, 3, Float64}(a, 0, 0, 0, a, 0, 1, 2, 1))
    H1, H2, H3 = mk(1.0), mk(2.0), mk(3.0)
    X = EuclideanMap(one(SMatrix{3, 3, Float64}), SVector(1.0, 2.0, 3.0))
    frames = Dict(1 => Dict(:b1 => H1, :b2 => H2), 3 => Dict(:b1 => H3))
    kw = (images = Dict(1 => "a.png", 3 => "c.png"), sensors = "cam")

    tg = TargetHomographies(frames; kw...)
    tp = TargetPoses{CameraFrame}(Dict(1 => Dict(:b1 => X, :b2 => X), 3 => Dict(:b1 => X)); kw...)

    @testset "Dict surface mirrors TargetPoses" begin
        @test tg isa AbstractDict{Int, Dict{Symbol, HomographyMat{Float64}}}
        @test length(tg) == length(tp) == 2
        @test sort!(collect(keys(tg))) == sort!(collect(keys(tp))) == [1, 3]
        @test tg[1][:b1] == H1 && tg[1][:b2] == H2 && tg[3][:b1] == H3
        @test Set(first.(collect(tg))) == Set(first.(collect(tp)))
        @test haskey(tg, 3) && !haskey(tg, 2)
        @test tg.images == tp.images && tg.sensors == tp.sensors == Dict(1 => "cam", 3 => "cam")
        @test tg == TargetHomographies(frames; kw...)
        @test tg != TargetHomographies(frames; images = Dict(1 => "z.png"), sensors = "cam")
    end

    @testset "pixel frame and targets are part of the value" begin
        @test tg isa TargetHomographies{PixelFrame{:colmap}}
        @test tg.targets == String[]
        ids = [repeat("a", 64), repeat("b", 64)]
        cv = TargetHomographies{PixelFrame{:opencv}}(frames; kw..., targets = ids)
        @test cv isa TargetHomographies{PixelFrame{:opencv}, HomographyMat{Float64}} && cv.targets == ids
        @test cv != tg                                              # another frame, another measurement
        @test cv == TargetHomographies{PixelFrame{:opencv}}(frames; kw..., targets = ids)
        @test cv != TargetHomographies{PixelFrame{:opencv}}(frames; kw...)   # other targets
        @test_throws ArgumentError TargetHomographies{PixelFrame{:opencv}}(frames; kw..., targets = [ids[1], ids[1]])
    end

    @testset "radial_homography" begin
        cam = CameraModel(ProjectiveIntrinsics(SVector(10.0, 20.0)), Division(-1.0e-5))
        @test radial_homography(cam, tg[1][:b1]) == RadialHomography(H1, cam)
    end

    @testset "validation refuses what TargetPoses refuses, with its messages" begin
        msg(f) = try; f(); nothing; catch e; e isa ArgumentError ? e.msg : rethrow(); end
        for (a, b) in (
                (() -> TargetHomographies(frames; images = Dict(2 => "b.png")), () -> TargetPoses{CameraFrame}(Dict(1 => Dict(:b => X)); images = Dict(2 => "b.png"))),
                (() -> TargetHomographies(frames; times = Dict(1 => 0.0)), () -> TargetPoses{CameraFrame}(Dict(1 => Dict(:b => X)); times = Dict(1 => 0.0))),
                (() -> TargetHomographies(frames; sensors = Dict(7 => "cam")), () -> TargetPoses{CameraFrame}(Dict(1 => Dict(:b => X)); sensors = Dict(7 => "cam"))),
            )
            @test msg(a) !== nothing && msg(a) == msg(b)
        end
        @test TargetHomographies(frames; times = Dict(1 => 0.5), time_origin = "2026-05-27T00:00:00").times == Dict(1 => 0.5)
        @test isempty(TargetHomographies(Dict{Int, Dict{Symbol, HomographyMat{Float64}}}()))
    end
end

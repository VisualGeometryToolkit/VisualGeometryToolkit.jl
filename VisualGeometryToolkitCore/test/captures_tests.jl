@testitem "captures" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: pose, model
    using StaticArrays
    using LinearAlgebra: I
    using Statistics: cov

    # Per-frame values of a capture (captures.jl): a body's Trajectory, the
    # TargetPoses of a detector, TargetPoints2D / TargetPoints3D of a sensor;
    # the frame index is the join key, absent means unobserved, Uncertain
    # members keep their covariance; convert_frame moves pixels with the frame.

    X = EuclideanMap(one(SMatrix{3, 3, Float64}), SVector(1.0, 2.0, 3.0))
    Σ6 = CovMat(SMatrix{6, 6, Float64}(2.0I))

    @testset "Trajectory" begin
        tr = Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => X, 3 => Uncertain(X, Σ6));
            images = Dict(1 => "a.png"), times = Dict(3 => 0.5), time_origin = "2026-05-27T00:00:00", recovered = [3])
        @test length(tr) == 2 && sort!(collect(keys(tr))) == [1, 3]
        @test tr[1] isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:rig}, WorldFrame}
        @test tr[3] isa Uncertain && cov(tr[3]) == Σ6                     # the covariance rides along
        @test geom(tr[3]) isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:rig}, WorldFrame}
        @test tr == Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => X, 3 => Uncertain(X, Σ6));
            images = Dict(1 => "a.png"), times = Dict(3 => 0.5), time_origin = "2026-05-27T00:00:00", recovered = [3])
        @test_throws ArgumentError Trajectory(UnlabeledFrame{3}, WorldFrame, Dict(1 => X))
        @test_throws ArgumentError Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => relabel(BodyFrame{:other}, WorldFrame, X)))
        @test_throws ArgumentError Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => X); images = Dict(2 => "b.png"))
        @test_throws ArgumentError Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => X); times = Dict(1 => 0.0))
        @test_throws ArgumentError Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => X); recovered = [2])
    end

    @testset "TargetPoses" begin
        tp = TargetPoses{CameraFrame}(Dict(1 => Dict(:b => X, :c => Uncertain(X, Σ6))); sensors = "cam", images = Dict(1 => "a.png"))
        @test tp.sensors == Dict(1 => "cam")
        @test pose(tp, 1, :b) isa EuclideanMap{3, Float64, <:Any, <:Any, TargetFrame{:b}, CameraFrame}
        @test cov(pose(tp, 1, :c)) == Σ6
        @test tp[1][:b] === unlabel(X)                                    # stored unlabeled, minted on access
        @test_throws ArgumentError TargetPoses{CameraFrame}(Dict(1 => Dict(:b => relabel(TargetFrame{:x}, CameraFrame, X))))
        @test_throws ArgumentError TargetPoses{UnlabeledFrame{3}}(Dict(1 => Dict(:b => X)))
    end

    @testset "TargetPoints2D / TargetPoints3D, convert_frame" begin
        Σ2 = CovMat(SMatrix{2, 2, Float64}(0.25I))
        o = TargetPoints2D{PixelFrame{:opencv}}(
            Dict(1 => [(target = :b, index = 1, xy = SVector(1.0, 2.0), weight = 1.0), (target = :b, index = 2, xy = Uncertain(SVector(3.0, 4.0), Σ2), weight = 0.5)]);
            sensors = "cam", images = Dict(1 => "a.png"), targets = ["ab"],
        )
        c = convert_frame(PixelFrame{:colmap}, o)
        @test c isa TargetPoints2D{PixelFrame{:colmap}}
        @test c[1][1].xy == SVector(1.5, 2.5)                             # opencv → colmap: +½ px
        @test cov(c[1][2].xy) == Σ2                                       # a translation leaves the covariance
        @test convert_frame(PixelFrame{:opencv}, c) == o
        @test c != o                                                      # equal numbers, different frames: different
        @test (c.images, c.sensors, c.targets) == (o.images, o.sensors, o.targets)
        @test_throws ArgumentError TargetPoints2D{PixelFrame{:colmap}}(Dict(1 => [(target = :b, index = 1, xy = SVector(1.0, 2.0))]))
        p3 = TargetPoints3D{WorldFrame}(Dict(7 => [(target = :body, index = 3, xyz = SVector(1.0, 2.0, 3.0))]); sensors = "mocap")
        @test p3[7][1].xyz == SVector(1.0, 2.0, 3.0) && p3.sensors[7] == "mocap"
        @test_throws ArgumentError TargetPoints3D{UnlabeledFrame{3}}(Dict(7 => []))
    end

    @testset "an unposed BoundedCamera" begin
        cm = CameraModel(SMatrix{3, 3, Float64}(800, 0, 0, 0, 800, 0, 320, 240, 1))
        bc = BoundedCamera(cm, Size2(640, 480))
        @test model(bc) === cm && extent(bc) == Size2(640, 480)
        @test intrinsics(bc) == intrinsics(cm)
    end

    @testset "relabel / unlabel an Uncertain map" begin
        u = relabel(BodyFrame{:a}, WorldFrame, Uncertain(X, Σ6))
        @test geom(u) isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:a}, WorldFrame} && cov(u) == Σ6
        @test geom(unlabel(u)) === unlabel(X)
    end
end

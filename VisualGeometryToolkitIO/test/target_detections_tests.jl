# vgio/target_detections/v1 ↔ VGC TargetDetections: round trips (pixels plain
# and Uncertain with weights; 3D points plain and Uncertain, in mm and in m),
# the old measurement kinds read as TargetDetections, and the refusals.

@testitem "target detections round trip (vgio/target_detections/v1)" setup = [VGIoFixtures] begin
    using LinearAlgebra: I
    using Statistics: cov
    using VisualGeometryToolkitCore: TargetDetections, TargetPoints2D, TargetPoints3D, WorldFrame, Uncertain, CovMat,
        geom, csponds, withattrs
    h = repeat("ab", 32)
    Σ2 = CovMat(SMatrix{2, 2, Float64}(0.25I))
    Σ3 = CovMat(SMatrix{3, 3, Float64}(2.5e-7I))                 # m²
    mktempdir() do dir
        rt(x; kw...) = (p = joinpath(dir, "td.json"); save(p, x; kw...); load(TargetDetections, p))
        # pixels, plain
        p2 = TargetDetections{PixelFrame{:colmap}}(
            Dict(1 => Dict(:b => csponds([1, 4], [SVector(1.0, 2.0), SVector(3.0, 4.5)])), 2 => Dict(:c => csponds([2], [SVector(5.0, 6.0)])));
            sensors = "cam1", images = Dict(1 => "a.png", 2 => "b.png"), times = Dict(1 => 0.5), time_origin = "2026-11-05T00:00:00", targets = [h],
        )
        @test rt(p2) == p2
        d = parse_json(read(joinpath(dir, "td.json"), String))
        @test d.schema == "vgio/target_detections/v1" && d.pixel_frame == "colmap" && d.point_index_base == 1
        @test d.frames[1].detections[1].index == [1, 4] && !haskey(d.frames[1].detections[1], :covariance)
        # pixels, Uncertain, with weights: px², never converted
        u2 = TargetDetections{PixelFrame{:opencv}}(
            Dict(3 => Dict(:b => withattrs(csponds([2, 7], [Uncertain(SVector(1.0, 2.0), Σ2), Uncertain(SVector(3.0, 4.0), Σ2)]); weights = [1.0, 0.5])));
            sensors = "cam1", images = Dict(3 => "c.png"), targets = [h],
        )
        b = rt(u2)
        @test b == u2 && cov(b[3][:b][2].second) == Σ2 && b[3][:b].weights == [1.0, 0.5]
        @test parse_json(read(joinpath(dir, "td.json"), String)).frames[1].detections[1].covariance[1] == [[0.25, 0.0], [0.0, 0.25]]
        # 3D, physical: plain and Uncertain, written in mm and in m, the same values back
        p3 = TargetDetections{WorldFrame}(Dict(7 => Dict(:body => csponds([3, 1], [mm_lengths(SVector(1140.24, 327.55, 1614.39)), mm_lengths(SVector(62.8, 1.3, 2.0))]))); sensors = "mocap")
        u3 = TargetDetections{WorldFrame}(Dict(7 => Dict(:body => csponds([3], [Uncertain(mm_lengths(SVector(1.0, 2.0, 3.0)), Σ3)]))); sensors = "mocap")
        for x in (p3, u3), u in (mm, m)
            @test rt(x; length_unit = u) == x
        end
        @test cov(rt(u3; length_unit = mm)[7][:body][1].second) == Σ3
        save(joinpath(dir, "td.json"), u3)                          # mm²
        d3 = parse_json(read(joinpath(dir, "td.json"), String))
        @test d3.length_unit == "mm" && d3.frame == "world" && !haskey(d3, :targets) && !haskey(d3, :pixel_frame)
        @test d3.frames[1].detections[1].points[1] == [1.0, 2.0, 3.0] && d3.frames[1].detections[1].covariance[1][1][1] ≈ 0.25
        @test VisualGeometryToolkitIO.validate(joinpath(dir, "td.json")) == "target_detections"
    end
end

@testitem "target detections: the old measurement kinds read, converted" setup = [VGIoFixtures] begin
    using LinearAlgebra: I
    using VisualGeometryToolkitCore: TargetDetections, TargetPoints2D, TargetPoints3D, WorldFrame, Uncertain, CovMat
    h = repeat("cd", 32)
    mktempdir() do dir
        o2 = TargetPoints2D{PixelFrame{:colmap}}(
            Dict(1 => [(target = :b, index = 1, xy = SVector(1.0, 2.0), weight = 1.0), (target = :c, index = 3, xy = SVector(3.0, 4.0), weight = 0.5), (target = :b, index = 2, xy = SVector(5.0, 6.0), weight = 0.25)]);
            sensors = "cam1", images = Dict(1 => "a.png"), targets = [h],
        )
        p = joinpath(dir, "target_points2d.json")
        save(p, o2)
        td = load(TargetDetections, p)
        @test td == TargetDetections(o2) && td[1][:b].weights == [1.0, 0.25]
        @test load(TargetPoints2D, p) == o2                             # the old reader is unchanged
        o3 = TargetPoints3D{WorldFrame}(
            Dict(2 => [(target = :rig, index = 2, xyz = Uncertain(mm_lengths(SVector(1.0, 2.0, 3.0)), CovMat(SMatrix{3, 3, Float64}(2.5e-7I))))]);
            sensors = "mocap", times = Dict(2 => 10.5), time_origin = "2026-05-27T00:00:00",
        )
        p3 = joinpath(dir, "target_points3d.json")
        save(p3, o3)
        @test load(TargetDetections, p3) == TargetDetections(o3)
        # resaved, it is the current kind
        q = joinpath(dir, "target_detections.json")
        save(q, load(TargetDetections, p3))
        @test parse_json(read(q, String)).schema == "vgio/target_detections/v1" && load(TargetDetections, q) == TargetDetections(o3)
    end
end

@testitem "target detections: re-pointed at a migrated rig, every index resolved" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: TargetDetections, WorldFrame, BodyFrame, Rig, PosedTarget, PointTarget, csponds
    using VisualGeometryToolkitIO: content_id
    mktempdir() do dir
        ident = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
        rig = Rig{BodyFrame{:board_rig}}(Dict(:body => PosedTarget(PointTarget([mm_lengths(SVector(0.0, 0.0, 0.0)), mm_lengths(SVector(1.0, 0.0, 0.0)), mm_lengths(SVector(0.0, 1.0, 0.0))]), ident)))
        rp = joinpath(dir, "targets_board_rig.json")
        save(rp, rig)
        old = repeat("b", 64)
        td(i) = TargetDetections{WorldFrame}(Dict(7 => Dict(:body => csponds(i, [mm_lengths(SVector(1.0, 2.0, 3.0)) for _ in i]))); sensors = "mocap", targets = [old])
        p = joinpath(dir, "target_detections_mocap.json")
        save(p, td([3, 1]))
        r = VisualGeometryToolkitIO._repoint_targets(p, Dict(old => content_id(rp)), [rp])
        @test r.repointed == 1 && load(TargetDetections, r.out).targets == [content_id(rp)]
        q = joinpath(dir, "target_detections_bad.json")
        save(q, td([4]))                                               # the body has 3 markers
        @test_throws ArgumentError VisualGeometryToolkitIO._repoint_targets(q, Dict(old => content_id(rp)), [rp])
    end
end

@testitem "target detections: refusals" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: TargetDetections, WorldFrame, csponds, withattrs
    h = repeat("ef", 32)
    mktempdir() do dir
        p = joinpath(dir, "td.json")
        # plain physical 3D points: attach the unit
        plain = TargetDetections{WorldFrame}(Dict(1 => Dict(:a => csponds([1], [SVector(1.0, 2.0, 3.0)]))); sensors = "mocap")
        @test_throws ArgumentError save(p, plain)
        @test !isfile(p)
        # what the file requires: pixels name their image, sensor and targets; 3D their sensor
        px2 = Dict(1 => Dict(:b => csponds([1], [SVector(1.0, 2.0)])))
        @test_throws ArgumentError save(p, TargetDetections{PixelFrame{:colmap}}(px2; sensors = "cam", images = Dict(1 => "a.png")))
        @test_throws ArgumentError save(p, TargetDetections{PixelFrame{:colmap}}(px2; sensors = "cam", targets = [h]))
        @test_throws ArgumentError save(p, TargetDetections{WorldFrame}(Dict(1 => Dict(:a => csponds([1], [mm_lengths(SVector(1.0, 2.0, 3.0))])))))
        # a column the kind does not hold
        @test_throws ArgumentError save(p, TargetDetections{PixelFrame{:colmap}}(
            Dict(1 => Dict(:b => withattrs(csponds([1], [SVector(1.0, 2.0)]); score = [0.9]))); sensors = "cam", images = Dict(1 => "a.png"), targets = [h]))
        # the reserved payload, and malformed records, refused on read
        good = TargetDetections{PixelFrame{:colmap}}(px2; sensors = "cam", images = Dict(1 => "a.png"), targets = [h])
        save(p, good)
        j = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        refused(f) = (q = joinpath(dir, "bad.json"); t = deepcopy(j); f(t); write(q, JSON.json(t)); try load(TargetDetections, q); false catch e; e isa ArgumentError end)
        @test refused(t -> t["payload"] = Dict("file" => "x.arrow", "sha256" => "0"))
        @test refused(t -> t["frames"][1]["detections"][1]["index"] = [1, 1])                     # unique
        @test refused(t -> t["frames"][1]["detections"][1]["index"] = [1, 2])                     # one per point
        @test refused(t -> t["frames"][1]["detections"][1]["points"] = [[1.0, 2.0, 3.0]])         # a pixel has 2
        @test refused(t -> t["frames"][1]["detections"][1]["covariance"] = [[[1.0, 2.0], [0.0, 1.0]]])   # symmetric
        @test refused(t -> t["frame"] = "world")                                                  # one frame declaration
        @test refused(t -> push!(t["frames"][1]["detections"], deepcopy(t["frames"][1]["detections"][1])))   # a target once per frame
        @test !refused(identity)
    end
end

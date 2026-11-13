# vgio/observations/v3.0 ↔ VGC TargetObservations: round trips (pixels plain
# and Uncertain in every covariance storage, with weights; 3D points plain and
# Uncertain, in mm and in m), the IO 0.9 kinds (target_points2d/3d,
# target_detections) read as TargetObservations, and the refusals.

@testitem "observations round trip (vgio/observations/v3.0)" setup = [VGIoFixtures] begin
    using LinearAlgebra: I
    using StaticArrays: SDiagonal
    using Statistics: cov
    using VisualGeometryToolkitCore: TargetObservations, WorldFrame, Uncertain, CovMat, ScaledIdentity,
        geom, correspondences, withattrs
    h = repeat("ab", 32)
    Σ2 = CovMat(SMatrix{2, 2, Float64}(0.25I))
    Σ3 = CovMat(SMatrix{3, 3, Float64}(2.5e-7I))                 # m²
    mktempdir() do dir
        rt(x; kw...) = (p = joinpath(dir, "obs.json"); save(p, x; kw...); load(TargetObservations, p))
        doc() = parse_json(read(joinpath(dir, "obs.json"), String))
        # pixels, plain
        p2 = TargetObservations{PixelFrame{:colmap}}(
            Dict(1 => Dict(:b => correspondences([1, 4], [SVector(1.0, 2.0), SVector(3.0, 4.5)])), 2 => Dict(:c => correspondences([2], [SVector(5.0, 6.0)])));
            sensors = "cam1", images = Dict(1 => "a.png", 2 => "b.png"), times = Dict(1 => 0.5), time_origin = "2026-11-05T00:00:00Z", targets = [h],
        )
        @test rt(p2) == p2
        d = doc()
        @test d.schema == "vgio/observations/v3.0" && d.entity == "target_point" && d.units.pixel == "colmap"
        @test d.conventions.index_base == 1 && d.time_origin == "2026-11-05T00:00:00Z"
        @test [f.frame_index for f in d.frames] == [1, 2] && d.frames[1].time == 0.5
        o = d.frames[1].observations[1]
        @test o.target == "b" && o.index == [1, 4] && o.points[1].xy == [1.0, 2.0] && !haskey(o.points[1], :covariance) && !haskey(o, :weights)
        # pixels, Uncertain, with weights: px², never converted, in the storage held (full, diagonal, isotropic)
        for (Σ, wire) in ((Σ2, [[0.25, 0.0], [0.0, 0.25]]), (CovMat(SDiagonal(0.25, 0.5)), Dict(:diagonal => [0.25, 0.5])),
                          (CovMat(ScaledIdentity{2}(0.25)), Dict(:isotropic => 0.25)))
            u2 = TargetObservations{PixelFrame{:opencv}}(
                Dict(3 => Dict(:b => withattrs(correspondences([2, 7], [Uncertain(SVector(1.0, 2.0), Σ), Uncertain(SVector(3.0, 4.0), Σ)]); weights = [1.0, 0.5])));
                sensors = "cam1", images = Dict(3 => "c.png"), targets = [h],
            )
            b = rt(u2)
            @test b == u2 && typeof(b) === typeof(u2) && cov(b[3][:b][2].second) === cov(u2[3][:b][2].second) && b[3][:b].weights == [1.0, 0.5]
            o = doc().frames[1].observations[1]
            @test doc().units.pixel == "opencv" && o.weights == [1.0, 0.5] && o.points[1].covariance == wire
        end
        # 3D, physical: plain and Uncertain, written in mm and in m, the same values back
        p3 = TargetObservations{WorldFrame}(Dict(7 => Dict(:body => correspondences([3, 1], [mm_lengths(SVector(1140.24, 327.55, 1614.39)), mm_lengths(SVector(62.8, 1.3, 2.0))]))); sensors = "mocap")
        u3 = TargetObservations{WorldFrame}(Dict(7 => Dict(:body => correspondences([3], [Uncertain(mm_lengths(SVector(1.0, 2.0, 3.0)), Σ3)]))); sensors = "mocap")
        for x in (p3, u3), u in (mm, m)
            @test rt(x; length_unit = u) == x
        end
        @test cov(rt(u3; length_unit = mm)[7][:body][1].second) == Σ3
        # 3D, every covariance storage (m² in memory, units.length² on disk), with weights
        for Σ in (Σ3, CovMat(SDiagonal(1.0e-8, 4.0e-8, 9.0e-8)), CovMat(ScaledIdentity{3}(2.5e-7)))
            w3 = TargetObservations{WorldFrame}(Dict(7 => Dict(:body => withattrs(correspondences([3, 1], [Uncertain(mm_lengths(SVector(1.0, 2.0, 3.0)), Σ), Uncertain(mm_lengths(SVector(-4.0, 0.5, 6.0)), Σ)]); weights = [0.25, 1.0]))); sensors = "mocap")
            b3 = rt(w3)
            @test b3 == w3 && typeof(b3) === typeof(w3) && typeof(parent(cov(b3[7][:body][1].second))) == typeof(parent(Σ)) && b3[7][:body].weights == [0.25, 1.0]
        end
        save(joinpath(dir, "obs.json"), u3)                          # mm²
        d3 = doc()
        @test d3.units.length == "mm" && d3.frame == "world" && !haskey(d3, :targets)
        o3 = d3.frames[1].observations[1]
        @test o3.points[1].xyz == [1.0, 2.0, 3.0] && o3.points[1].covariance[1][1] ≈ 0.25
        @test VisualGeometryToolkitIO.validate(joinpath(dir, "obs.json")) == "observations"
    end
end

@testitem "observations: the IO 0.9 kinds read, converted" setup = [VGIoFixtures] begin
    using LinearAlgebra: I
    using Statistics: cov
    using VisualGeometryToolkitCore: TargetObservations, WorldFrame, Uncertain, CovMat, correspondences, withattrs
    h = repeat("cd", 32)
    mktempdir() do dir
        # an IO 0.9 target_points2d/v2 record (one flat list of points per frame)
        p = joinpath(dir, "target_points2d.json")
        write(p, JSON.json(Dict(
            "schema" => "vgio/target_points2d/v2", "pixel_frame" => "colmap", "point_index_base" => 1, "targets" => [h],
            "frames" => [Dict("t" => 1, "image" => "a.png", "sensor" => "cam1", "points" => [
                Dict("target" => "b", "index" => 1, "xy" => [1.0, 2.0], "weight" => 1.0),
                Dict("target" => "c", "index" => 3, "xy" => [3.0, 4.0], "weight" => 0.5),
                Dict("target" => "b", "index" => 2, "xy" => [5.0, 6.0], "weight" => 0.25)])])))
        td = load(TargetObservations, p)
        @test td == TargetObservations{PixelFrame{:colmap}}(
            Dict(1 => Dict(:b => withattrs(correspondences([1, 2], [SVector(1.0, 2.0), SVector(5.0, 6.0)]); weights = [1.0, 0.25]),
                           :c => withattrs(correspondences([3], [SVector(3.0, 4.0)]); weights = [0.5])));
            sensors = "cam1", images = Dict(1 => "a.png"), targets = [h])
        # an IO 0.9 target_points3d/v2 record, Uncertain, its time origin stated with an offset
        p3 = joinpath(dir, "target_points3d.json")
        write(p3, JSON.json(Dict(
            "schema" => "vgio/target_points3d/v2", "length_unit" => "mm", "frame" => "world", "point_index_base" => 1,
            "time_origin" => "2026-05-27T00:00:00Z",
            "frames" => [Dict("t" => 2, "sensor" => "mocap", "time" => 10.5, "points" => [
                Dict("target" => "rig", "index" => 2, "xyz" => [1.0, 2.0, 3.0], "covariance" => [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]])])])))
        o3 = load(TargetObservations, p3)
        want = TargetObservations{WorldFrame}(
            Dict(2 => Dict(:rig => correspondences([2], [Uncertain(mm_lengths(SVector(1.0, 2.0, 3.0)), CovMat(SMatrix{3, 3, Float64}(2.5e-7I)))])));
            sensors = "mocap", times = Dict(2 => 10.5), time_origin = "2026-05-27T00:00:00Z")
        @test o3 == want
        # resaved, it is the current kind
        q = joinpath(dir, "observations.json")
        save(q, o3)
        @test parse_json(read(q, String)).schema == "vgio/observations/v3.0" && load(TargetObservations, q) == want
    end
end

@testitem "observations: an IO 0.9 file with a duplicate or a form mix is refused by name" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: thrown_message, TargetObservations
    h = repeat("cd", 32)
    mktempdir() do dir
        # a (target, index) twice in one frame: the 0.9 target_points2d kind allowed it
        p = joinpath(dir, "dup_points2d.json")
        write(p, JSON.json(Dict(
            "schema" => "vgio/target_points2d/v2", "pixel_frame" => "colmap", "point_index_base" => 1, "targets" => [h],
            "frames" => [Dict("t" => 3, "image" => "a.png", "sensor" => "cam1", "points" => [
                Dict("target" => "b", "index" => 2, "xy" => [1.0, 2.0], "weight" => 1.0),
                Dict("target" => "b", "index" => 2, "xy" => [1.5, 2.5], "weight" => 1.0)])])))
        s = thrown_message(() -> load(TargetObservations, p))
        @test occursin(p, s) && occursin("frame 3 observes point 2 of target b twice", s)
        # plain and Uncertain points in one file: the 0.9 target_points3d kind allowed it
        p3 = joinpath(dir, "mixed_points3d.json")
        write(p3, JSON.json(Dict(
            "schema" => "vgio/target_points3d/v2", "length_unit" => "mm", "frame" => "world", "point_index_base" => 1,
            "frames" => [Dict("t" => 1, "sensor" => "mocap", "points" => [Dict("target" => "rig", "index" => 1, "xyz" => [1.0, 2.0, 3.0])]),
                         Dict("t" => 2, "sensor" => "mocap", "points" => [Dict("target" => "rig", "index" => 2, "xyz" => [1.0, 2.0, 3.0],
                                                                               "covariance" => [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]])])])))
        s = thrown_message(() -> load(TargetObservations, p3))
        @test occursin(p3, s) && occursin("Uncertain", s) && occursin("plain", s)
    end
end

@testitem "observations: refusals" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: TargetObservations, WorldFrame, correspondences, withattrs
    h = repeat("ef", 32)
    mktempdir() do dir
        p = joinpath(dir, "obs.json")
        # plain physical 3D points: attach the unit
        plain = TargetObservations{WorldFrame}(Dict(1 => Dict(:a => correspondences([1], [SVector(1.0, 2.0, 3.0)]))); sensors = "mocap")
        @test_throws ArgumentError save(p, plain)
        @test !isfile(p)
        # what the file requires: pixels name their image, sensor and targets; 3D their sensor
        px2 = Dict(1 => Dict(:b => correspondences([1], [SVector(1.0, 2.0)])))
        @test_throws ArgumentError save(p, TargetObservations{PixelFrame{:colmap}}(px2; sensors = "cam", images = Dict(1 => "a.png")))
        @test_throws ArgumentError save(p, TargetObservations{PixelFrame{:colmap}}(px2; sensors = "cam", targets = [h]))
        @test_throws ArgumentError save(p, TargetObservations{WorldFrame}(Dict(1 => Dict(:a => correspondences([1], [mm_lengths(SVector(1.0, 2.0, 3.0))])))))
        # a time origin without its UTC offset
        @test_throws ArgumentError save(p, TargetObservations{PixelFrame{:colmap}}(px2; sensors = "cam", images = Dict(1 => "a.png"), targets = [h],
                                                                                   times = Dict(1 => 0.5), time_origin = "2026-05-27T00:00:00"))
        # a column the kind does not hold
        @test_throws ArgumentError save(p, TargetObservations{PixelFrame{:colmap}}(
            Dict(1 => Dict(:b => withattrs(correspondences([1], [SVector(1.0, 2.0)]); score = [0.9]))); sensors = "cam", images = Dict(1 => "a.png"), targets = [h]))
        @test !isfile(p)
        # the reserved payload, and malformed records, refused on read
        good = TargetObservations{PixelFrame{:colmap}}(px2; sensors = "cam", images = Dict(1 => "a.png"), targets = [h])
        save(p, good)
        j = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        refused(f) = (q = joinpath(dir, "bad.json"); t = deepcopy(j); f(t); write(q, JSON.json(t)); try load(TargetObservations, q); false catch e; e isa ArgumentError end)
        set1(t) = t["frames"][1]["observations"][1]
        @test refused(t -> t["payload"] = Dict("file" => "x.arrow", "sha256" => "0"))
        @test refused(t -> set1(t)["index"] = [1, 1])                                          # unique
        @test refused(t -> set1(t)["index"] = [1, 2])                                          # one per point
        @test refused(t -> set1(t)["index"] = [0])                                             # 1-based
        @test refused(t -> set1(t)["points"] = [Dict("xy" => [1.0, 2.0, 3.0])])                # a pixel has 2
        @test refused(t -> set1(t)["points"] = [Dict("xyz" => [1.0, 2.0, 3.0])])               # pixels are `xy`
        @test refused(t -> set1(t)["points"][1]["covariance"] = [[1.0, 2.0], [0.0, 1.0]])      # symmetric
        @test refused(t -> set1(t)["weights"] = [1.0, 2.0])                                    # one per point
        @test refused(t -> t["frame"] = "world")                                               # one frame declaration
        @test refused(t -> t["entity"] = "keyframe")                                           # an entity of the kind's
        @test refused(t -> delete!(set1(t), "target"))                                         # a target point names its target
        @test refused(t -> push!(t["frames"][1]["observations"], deepcopy(set1(t))))          # a target once per frame
        @test refused(t -> t["frames"][1]["frame_index"] = 1.5)                                # an integer join key
        @test refused(t -> (t["frames"][1]["time"] = 1.0))                                     # a time needs its origin
        @test refused(t -> (t["frames"][1]["time"] = 1.0; t["time_origin"] = "2026-05-27T00:00:00"))   # with its UTC offset
        @test !refused(t -> (t["frames"][1]["time"] = 1.0; t["time_origin"] = "2026-05-27T00:00:00+02:00"))
        @test !refused(identity)
        # landmark observations: a valid file, but no VGC container holds them
        lm = deepcopy(j); lm["entity"] = "landmark"; delete!(lm["frames"][1]["observations"][1], "target")
        q = joinpath(dir, "landmarks.json"); write(q, JSON.json(lm))
        @test VisualGeometryToolkitIO.validate(q) == "observations"
        @test_throws ArgumentError load(TargetObservations, q)
    end
end

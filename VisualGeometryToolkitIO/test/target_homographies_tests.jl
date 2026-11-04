# =============================================================================
# vgio/target_homographies/v1 ↔ TargetHomographies: per frame, each target's
# plane → pixel homography, canonical (unit Frobenius norm, H[3,3] ≥ 0)
# =============================================================================

@testsetup module TargetHomographiesFixtures
    using VisualGeometryToolkitCore: TargetHomographies, HomographyMat, PixelFrame, sign_normalize
    using StaticArrays: SMatrix
    using JSON: JSON
    using VisualGeometryToolkitIO: load, save

    export colmap, ids, canonical, sample, written, rewritten, load, save

    const colmap = PixelFrame{:colmap}
    const ids = [repeat('a', 64), repeat('b', 64)]     # (the content ids of a rig of boards, of a single target)

    # a plane → pixel homography, as a placement would give it, in canonical form
    canonical(rows...) = sign_normalize(HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(permutedims(reduce(hcat, rows))))))

    function sample()
        a = canonical([812.3, -41.7, 1210.25], [37.9, 805.1, 640.5], [0.00012, -0.00007, 0.6])
        b = canonical([-799.0, 12.5, 1511.0], [-9.25, -803.75, 480.125], [0.0001, 0.0002, 0.7])
        c = canonical([790.5, 0.0, 300.0], [0.0, 790.5, 200.0], [0.0, 0.0, 0.4])
        return TargetHomographies{colmap}(
            Dict(1 => Dict(:a => a, :b => b), 4 => Dict(:a => c));
            images = Dict(1 => "img_001.png", 4 => "img_004.png"), sensors = "cam",
            times = Dict(1 => 0.5, 4 => 12.25), time_origin = "2026-11-05T00:00:00", targets = ids,
        )
    end

    written(path, th; kw...) = (save(path, th; kw...); JSON.parse(read(path, String)))
    rewritten(path, d) = (write(path, JSON.json(d)); path)
end

@testitem "target_homographies/v1: exact round trip" setup = [TargetHomographiesFixtures] begin
    using VisualGeometryToolkitCore: TargetHomographies, PixelFrame
    using JSON: JSON
    import VisualGeometryToolkitIO as VgtkIO
    th = sample()
    mktempdir() do dir
        p = joinpath(dir, "th.json")
        save(p, th; provenance = (; tool = "test"))
        back = load(TargetHomographies, p)
        @test back == th
        for t in keys(th), id in keys(th[t])
            @test collect(back[t][id]) == collect(th[t][id])      # bit for bit
        end
        @test back.images == th.images && back.sensors == th.sensors && back.times == th.times
        @test back.time_origin == th.time_origin
        @test back isa TargetHomographies{colmap} && back.targets == ids
        @test load(TargetHomographies{colmap}, p) == th
        @test_throws ArgumentError load(TargetHomographies{PixelFrame{:opencv}}, p)
        # another pixel frame is another type, written under its own label
        cv = TargetHomographies{PixelFrame{:opencv}}(th.frames; images = th.images, sensors = th.sensors, targets = ids)
        save(joinpath(dir, "cv.json"), cv)
        @test load(TargetHomographies, joinpath(dir, "cv.json")) isa TargetHomographies{PixelFrame{:opencv}}
        @test load(TargetHomographies, joinpath(dir, "cv.json")) != th
        @test JSON.parse(read(p, String))["targets"] == ids
        d = JSON.parse(read(p, String))
        @test d["schema"] == "vgio/target_homographies/v1" && d["pixel_frame"] == "colmap"
        @test d["provenance"] == Dict("tool" => "test")
        @test d["frames"][1]["homographies"][1]["H"] isa Vector && length(d["frames"][1]["homographies"][1]["H"][1]) == 3
        @test VgtkIO.validate(p) == "target_homographies"
        # the same placement always writes the same bytes
        save(joinpath(dir, "again.json"), th; provenance = (; tool = "test"))
        @test read(p, String) == read(joinpath(dir, "again.json"), String)
    end
end

@testitem "target_homographies/v1: the writer stores the canonical H" setup = [TargetHomographiesFixtures] begin
    using VisualGeometryToolkitCore: TargetHomographies, HomographyMat, sign_normalize
    using StaticArrays: SMatrix
    using LinearAlgebra: norm
    raw = HomographyMat{Float64}(Tuple(-37.5 .* SMatrix{3, 3, Float64}(3, 1, 0.2, 0.5, 2, 0.1, 100, 80, 1)))   # scaled, wrong sign
    th = TargetHomographies{colmap}(Dict(1 => Dict(:a => raw)); images = Dict(1 => "i.png"), sensors = "cam", targets = ids)
    mktempdir() do dir
        p = joinpath(dir, "th.json")
        save(p, th)
        H = load(TargetHomographies, p)[1][:a]
        @test collect(H) == collect(sign_normalize(raw))
        @test abs(norm(H) - 1) <= 4eps() && H[3, 3] >= 0
    end
    degenerate = TargetHomographies{colmap}(Dict(1 => Dict(:a => HomographyMat{Float64}(ntuple(_ -> 0.0, 9)))); images = Dict(1 => "i.png"), sensors = "cam", targets = ids)
    mktempdir() do dir
        p = joinpath(dir, "th.json")
        @test_throws ArgumentError save(p, degenerate)
        @test !isfile(p)
    end
end

@testitem "target_homographies/v1: refusals name the path" setup = [TargetHomographiesFixtures] begin
    using VisualGeometryToolkitCore: TargetHomographies
    th = sample()
    mktempdir() do dir
        base = joinpath(dir, "base.json")
        good = written(base, th)
        function refusal(mutate, name)
            d = deepcopy(good)
            mutate(d)
            p = rewritten(joinpath(dir, name * ".json"), d)
            err = try
                load(TargetHomographies, p)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test err isa ArgumentError && occursin(p, err.msg)
            return err isa ArgumentError ? err.msg : ""
        end
        @test occursin("bogus", refusal(d -> d["bogus"] = 1, "unknown_top"))
        @test occursin("bogus", refusal(d -> d["frames"][1]["bogus"] = 1, "unknown_frame"))
        @test occursin("pixel_frame", refusal(d -> delete!(d, "pixel_frame"), "no_pixel_frame"))
        @test occursin("pixel_frame", refusal(d -> d["pixel_frame"] = "nonsense", "bad_pixel_frame"))
        @test occursin("targets", refusal(d -> delete!(d, "targets"), "no_targets"))
        @test occursin("targets", refusal(d -> d["targets"] = String[], "empty_targets"))
        @test occursin("targets", refusal(d -> d["targets"] = [ids[1], ids[1]], "repeated_targets"))
        @test occursin("targets", refusal(d -> d["targets"] = ["not-a-digest"], "bad_targets"))
        @test occursin("H", refusal(d -> d["frames"][1]["homographies"][1]["H"] = [[1.0, 0.0], [0.0, 1.0]], "not_3x3"))
        @test occursin("H", refusal(d -> d["frames"][1]["homographies"][1]["H"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "two_rows"))
        @test occursin("singular", refusal(d -> d["frames"][1]["homographies"][1]["H"] = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]], "singular"))
        @test occursin("canonical", refusal(d -> d["frames"][1]["homographies"][1]["H"] = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], "not_unit"))
        @test occursin("canonical", refusal(d -> d["frames"][1]["homographies"][1]["H"] = [[-0.6, 0.0, 0.0], [0.0, -0.6, 0.0], [0.0, 0.0, -0.8]], "wrong_sign"))
        @test occursin("`t`", refusal(d -> d["frames"][2]["t"] = d["frames"][1]["t"], "duplicate_t"))
        @test occursin("targets", refusal(d -> push!(d["frames"][1]["homographies"], deepcopy(d["frames"][1]["homographies"][1])), "duplicate_target"))
        @test occursin("sensor", refusal(d -> delete!(d["frames"][1], "sensor"), "no_sensor"))
        @test occursin("image", refusal(d -> delete!(d["frames"][1], "image"), "no_image"))
        @test occursin("time_origin", refusal(d -> delete!(d, "time_origin"), "no_time_origin"))
        @test occursin("target_homographies/v1", refusal(d -> d["schema"] = "vgio/target_poses/v1", "wrong_kind"))
    end
    # a writer refuses a value missing what the kind requires, without leaving a file
    noimg = TargetHomographies{colmap}(sample().frames; sensors = "cam", targets = ids)
    notargets = TargetHomographies{colmap}(sample().frames; images = sample().images, sensors = "cam")
    mktempdir() do dir
        @test_throws ArgumentError save(joinpath(dir, "x.json"), noimg)
        @test_throws ArgumentError save(joinpath(dir, "y.json"), notargets)
        @test !isfile(joinpath(dir, "y.json"))
    end
end

@testitem "target_homographies/v1: version machinery" setup = [TargetHomographiesFixtures] begin
    using VisualGeometryToolkitCore: TargetHomographies
    import VisualGeometryToolkitIO as VgtkIO
    th = sample()
    mktempdir() do dir
        p = joinpath(dir, "th.json")
        tree = written(p, th)
        @test VgtkIO.migrate_record(tree) == tree                    # a current record comes back as it is
        @test VgtkIO.migrate_record(tree)["schema"] == "vgio/target_homographies/v1"
        @test !VgtkIO._has_record_migration("vgio/target_homographies/v1")   # no step until a v2 exists
        @test occursin("target_homographies/v1", VgtkIO._schema("target_homographies"))
    end
end

@testitem "content_id: a single target's identity is stable across write, read, write" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: PointTarget, PosedTarget, AbstractTarget, Attributed, EuclideanMap
    using VisualGeometryToolkitIO: content_id
    pts = [mm_lengths(SVector(0.0, 0.0, 0.0)), mm_lengths(SVector(10.0, 0.0, 0.0)), mm_lengths(SVector(0.0, 20.5, 0.0))]
    t = PointTarget(pts)
    mktempdir() do dir
        a, b = joinpath(dir, "a.json"), joinpath(dir, "b.json")
        save(a, t)
        id = content_id(t)
        @test length(id) == 64 && all(c -> c in "0123456789abcdef", id)
        @test content_id(a) == id                                   # the file, by its schema
        back = load(AbstractTarget, a)
        @test content_id(back) == id
        save(b, back)
        @test content_id(b) == id && content_id(load(AbstractTarget, b)) == id
        save(joinpath(dir, "cm.json"), t; length_unit = VisualGeometryToolkitCore.cm)    # any decimal unit
        @test content_id(joinpath(dir, "cm.json")) == id
        @test content_id(Attributed(t, (; calibration_record = (; tool = "t")))) == id
        @test content_id(PointTarget(pts[1:2])) != id
        ident = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
        @test_throws ArgumentError content_id(PosedTarget(t, ident))
    end
end

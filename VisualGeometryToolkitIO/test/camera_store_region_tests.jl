# =============================================================================
# vgio/cameras/v5 — a stored camera's lens coverage (its image circle)
# =============================================================================
# A camera's lens coverage (VisualGeometryToolkitCore's `lens_coverage`) is stored as an optional per-camera
# `lens_coverage` ({"circle": [cx, cy, r]} or {"ellipse": [cx, cy, a, b, θ]}, pixels); absent, the whole sensor is lit.
# A v4 store, which has no such key, reads with the whole sensor lit.

@testitem "camera store v5: the lens coverage" begin
    using Test
    using JSON
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: BoundedCamera, CameraModel, KannalaBrandt, Sensor, Size2, Point2, Circle, Ellipse,
        Rect2, Attributed, lens_coverage, with_lens_coverage, extent, sensor, px, μm
    using VisualGeometryToolkitCore: PosedCamera, EuclideanMap, StereoRig, mm
    using Rotations: RotMatrix
    using StaticArrays: SVector

    m = CameraModel(1754.7px, VgtkCore.SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))
    W, H = 4256, 2848
    regions = (nothing, Circle(Point2(2131.1, 1407.9), 1680.1), Ellipse(Point2(2133.9, 1423.7), 1611.8, 1607.8, 1.5777))
    bounds = (Size2(W, H), Sensor(Size2(W, H), 5.95μm))
    dir = mktempdir()
    whole = Rect2{Float64}(0, 0, W, H)

    for bound in bounds, region in regions
        bc = BoundedCamera(m, bound; lens_coverage = region)
        p = joinpath(dir, "cam.json")
        VgtkIO.save(p, Dict(:cam => bc))
        j = JSON.parsefile(p)
        @test j["schema"] == "vgio/cameras/v5"
        @test haskey(only(j["cameras"]), "lens_coverage") == (region !== nothing)
        bc2 = VgtkIO.load(BoundedCamera, p)
        @test lens_coverage(bc2) == lens_coverage(bc)                    # the kind and the numbers, Float64 printed round-trippably
        @test extent(bc2) == extent(bc) && (sensor(bc2) === nothing) == (sensor(bc) === nothing)
        # an Attributed camera (its calibration block) keeps its region too
        a = Attributed(bc, (; calibration = (; tool = "test", reprojection_rms = 0.2, n_images = 3)))
        VgtkIO.save(p, Dict(:cam => a))
        @test lens_coverage(getfield(VgtkIO.load(Dict{Symbol, BoundedCamera}, p)[:cam], :value)) == lens_coverage(bc)
        # the COLMAP export is the camera's alone: the region is not COLMAP's
        row(c) = (r = first(first(VgtkIO.colmap_cameras(Dict(:cam => c)))); ntuple(i -> getfield(r, i), fieldcount(typeof(r))))
        @test row(bc) == row(BoundedCamera(m, bound))             # field by field: a ColmapCamera holds a Vector
    end

    @testset "a v4 store: the whole sensor lit, the bound as before" begin
        for bound in bounds
            p = joinpath(dir, "v4.json")
            VgtkIO.save(p, Dict(:cam => BoundedCamera(m, bound)))
            write(p, replace(read(p, String), "vgio/cameras/v5" => "vgio/cameras/v4"))   # v4 is v5 without lens_coverage
            bc4 = VgtkIO.load(BoundedCamera, p)
            @test lens_coverage(bc4) == whole && extent(bc4) == Size2(W, H) && (sensor(bc4) === nothing) == !(bound isa Sensor)
        end
    end

    @testset "a Rect coverage (a letterboxed image) round-trips; the whole sensor writes nothing" begin
        bc = with_lens_coverage(BoundedCamera(m, Size2(W, H)), Rect2(10.0, 20.0, 4000.0, 2800.0))
        p = joinpath(dir, "rect.json")
        VgtkIO.save(p, Dict(:cam => bc))
        @test only(JSON.parsefile(p)["cameras"])["lens_coverage"] == Dict("rect" => [10.0, 20.0, 4000.0, 2800.0])
        @test lens_coverage(VgtkIO.load(BoundedCamera, p)) == Rect2{Float64}(10.0, 20.0, 4000.0, 2800.0)
        VgtkIO.save(p, Dict(:cam => with_lens_coverage(bc, Rect2(0.0, 0.0, W, H))))
        @test !haskey(only(JSON.parsefile(p)["cameras"]), "lens_coverage")
    end

    @testset "a store in a foreign pixel frame: the coverage moves as K's principal point does" begin
        p = joinpath(dir, "frame.json")
        VgtkIO.save(p, Dict(:cam => BoundedCamera(m, Size2(W, H); lens_coverage = Circle(Point2(2131.1, 1407.9), 1680.1))))
        q = joinpath(dir, "frame_opencv.json")
        write(q, replace(read(p, String), "\"pixel_frame\": \"colmap\"" => "\"pixel_frame\": \"opencv\""))
        a, b = VgtkIO.load(BoundedCamera, p), VgtkIO.load(BoundedCamera, q)
        Δk = VgtkCore.model(b).intrinsics.K[1:2, 3] - VgtkCore.model(a).intrinsics.K[1:2, 3]
        @test !iszero(Δk)
        @test lens_coverage(b).center - lens_coverage(a).center ≈ Δk && lens_coverage(b).r == lens_coverage(a).r
    end

    @testset "a v4 store that holds a lens_coverage was not written by v4" begin
        p = joinpath(dir, "v4lc.json")
        VgtkIO.save(p, Dict(:cam => BoundedCamera(m, Size2(W, H); lens_coverage = Circle(Point2(2131.1, 1407.9), 1680.1))))
        write(p, replace(read(p, String), "vgio/cameras/v5" => "vgio/cameras/v4"))
        @test_throws ArgumentError VgtkIO.load(BoundedCamera, p)
    end

    @testset "a rig stores each sensor's lens coverage (rig/v9)" begin
        pose(x) = EuclideanMap(one(RotMatrix{3, Float64}), SVector(x, 0.0, 0.0) .* mm)
        circle = Circle(Point2(2131.1, 1407.9), 1680.1)
        covered = BoundedCamera(PosedCamera(m, pose(0.0)), Size2(W, H); lens_coverage = circle)
        other = BoundedCamera(PosedCamera(m, pose(100.0)), Size2(W, H))
        rig = StereoRig(covered, other; ids = (:left, :right))
        p = joinpath(dir, "rig.json")
        VgtkIO.save(p, rig)
        j = JSON.parsefile(p)
        @test j["schema"] == "vgio/rig/v9"
        @test [haskey(s["camera"], "lens_coverage") for s in j["sensors"]] == [true, false]   # the whole sensor writes nothing
        back = VgtkIO.load(VgtkCore.Rig, p)
        @test lens_coverage(back[1]) == circle && lens_coverage(back[2]) == whole
        # the coverage is part of the camera's identity; the whole sensor's identity is the one it always had
        bare = StereoRig(with_lens_coverage(covered, nothing), other; ids = (:left, :right))
        @test VgtkIO.content_id(back) == VgtkIO.content_id(rig) == VgtkIO.content_id(p) != VgtkIO.content_id(bare)
        @test !haskey(only(x for x in VgtkIO._identity_tree(bare)["sensors"] if x["id"] == "left")["camera"], "lens_coverage")
        # a v8 rig holds no coverage: one that does was not written by v8
        write(p, replace(read(p, String), "vgio/rig/v9" => "vgio/rig/v8"))
        @test_throws ArgumentError VgtkIO.load(VgtkCore.Rig, p)
    end
end

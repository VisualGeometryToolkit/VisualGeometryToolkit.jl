# =============================================================================
# vgio/cameras/v5 — a stored camera's lit region (its image circle)
# =============================================================================
# A camera's lit region (VisualGeometryToolkitCore's `lit_region`) is stored as an optional per-camera
# `image_circle` ({"circle": [cx, cy, r]} or {"ellipse": [cx, cy, a, b, θ]}, pixels); absent, the whole sensor is lit.
# A v4 store, which has no such key, reads with the whole sensor lit.

@testitem "camera store v5: the lit region" begin
    using Test
    using JSON
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: BoundedCamera, CameraModel, KannalaBrandt, Sensor, Size2, Point2, Circle, Ellipse,
        Rect2, Attributed, lit_region, with_lit_region, extent, sensor, px, μm

    m = CameraModel(1754.7px, VgtkCore.SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))
    W, H = 4256, 2848
    regions = (nothing, Circle(Point2(2131.1, 1407.9), 1680.1), Ellipse(Point2(2133.9, 1423.7), 1611.8, 1607.8, 1.5777))
    bounds = (Size2(W, H), Sensor(Size2(W, H), 5.95μm))
    dir = mktempdir()
    whole = Rect2{Float64}(0, 0, W, H)

    for bound in bounds, region in regions
        bc = BoundedCamera(m, bound; region)
        p = joinpath(dir, "cam.json")
        VgtkIO.save(p, Dict(:cam => bc))
        j = JSON.parsefile(p)
        @test j["schema"] == "vgio/cameras/v5"
        @test haskey(only(j["cameras"]), "image_circle") == (region !== nothing)
        bc2 = VgtkIO.load(BoundedCamera, p)
        @test lit_region(bc2) == lit_region(bc)                    # the kind and the numbers, Float64 printed round-trippably
        @test extent(bc2) == extent(bc) && (sensor(bc2) === nothing) == (sensor(bc) === nothing)
        # an Attributed camera (its calibration block) keeps its region too
        a = Attributed(bc, (; calibration = (; tool = "test", reprojection_rms = 0.2, n_images = 3)))
        VgtkIO.save(p, Dict(:cam => a))
        @test lit_region(getfield(VgtkIO.load(Dict{Symbol, BoundedCamera}, p)[:cam], :value)) == lit_region(bc)
        # the COLMAP export is the camera's alone: the region is not COLMAP's
        row(c) = (r = first(first(VgtkIO.colmap_cameras(Dict(:cam => c)))); ntuple(i -> getfield(r, i), fieldcount(typeof(r))))
        @test row(bc) == row(BoundedCamera(m, bound))             # field by field: a ColmapCamera holds a Vector
    end

    @testset "a v4 store: the whole sensor lit, the bound as before" begin
        for bound in bounds
            p = joinpath(dir, "v4.json")
            VgtkIO.save(p, Dict(:cam => BoundedCamera(m, bound)))
            write(p, replace(read(p, String), "vgio/cameras/v5" => "vgio/cameras/v4"))   # v4 is v5 without image_circle
            bc4 = VgtkIO.load(BoundedCamera, p)
            @test lit_region(bc4) == whole && extent(bc4) == Size2(W, H) && (sensor(bc4) === nothing) == !(bound isa Sensor)
        end
    end

    @testset "only a Circle or an Ellipse is stored as an image circle" begin
        bc = with_lit_region(BoundedCamera(m, Size2(W, H)), Rect2(10.0, 10.0, 100.0, 100.0))
        @test_throws ArgumentError VgtkIO.save(joinpath(dir, "rect.json"), Dict(:cam => bc))
    end
end

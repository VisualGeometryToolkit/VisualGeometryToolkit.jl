# =============================================================================
# refusals the happy path never reaches
# =============================================================================

@testitem "refusals the happy path never reaches" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    # Every branch here is a REFUSAL, and refusing correctly is most of what
    # this package promises — so each one is asserted rather than left to the
    # first caller to discover. Named by a coverage audit, not by taste.
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 800.0, 0, 320.5, 240.5, 1))
    cam = CameraModel(K; lens = Pinhole(), pixel_frame = :colmap)
    model = ColmapModel(
        [colmap_camera(cam, 1, 640, 480)], ColmapImage[],
        ColmapPoint3D[], MM_PX
    )

    mktempdir() do dir
        @test_throws ArgumentError save(dir, model; format = :yaml)
        # `save` mkpath'd the directory before rejecting the format, so this
        # is a real "declared units, but no model here" case
        @test_throws ArgumentError load(ColmapModel, dir; contract = MM_PX)
    end

    # A caller contract that CONTRADICTS the sidecar is refused, never
    # preferred: the sidecar is the file's own declaration of its units, and
    # silently overriding it is the whole bug class this package exists for.
    mktempdir() do dir
        save(dir, model)
        @test load(ColmapModel, dir; contract = MM_PX).contract == MM_PX
        @test_throws ArgumentError load(
            ColmapModel, dir;
            contract = UnitContract(world = m)
        )
    end

    # The sidecar declares the world unit, once, and nothing else. A v1
    # sidecar is retired: refused by `load`, by name, and read by nothing.
    mktempdir() do dir
        save(dir, model)
        sidecar = joinpath(dir, VgtkIO.UNITS_SIDECAR)
        d = parse_json(read(sidecar, String))
        @test d.schema == "vgio/units/v3.0" && collect(keys(d)) == [:schema, :generator, :units, :conventions]   # the envelope alone
        @test d.units.length == "mm" && d.units.pixel == "colmap" && d.units.density == "dpi" && isempty(d.units.length_roles)
        @test d.generator.name == "VisualGeometryToolkitIO" && d.conventions.index_base == 1
        good = read(sidecar, String)
        # the envelope's declarations are its own: a pixel frame other than COLMAP's, a length role, an unknown key
        for (k, v) in ((:pixel, "opencv"), (:length_roles, Dict(:pattern => "pt")))
            write(sidecar, JSON.json(merge(d, Dict(:units => merge(d.units, Dict(k => v))))))
            @test_throws ArgumentError load(ColmapModel, dir)
        end
        write(sidecar, JSON.json(merge(d, Dict(:world => "mm"))))
        @test_throws ArgumentError load(ColmapModel, dir)
        write(sidecar, good)
        @test load(ColmapModel, dir).contract == MM_PX
        write(sidecar, JSON.json((schema = "vgio/units/v1", world = "m", image = "px")))
        err = try
            load(ColmapModel, dir); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("retired vgio/units/v1", err.msg)
        # a v2 sidecar (IO 0.9) reads through its step, and holds exactly its declaration
        write(sidecar, JSON.json((schema = "vgio/units/v2", world = "mm")))
        @test load(ColmapModel, dir).contract == MM_PX
        write(sidecar, JSON.json((schema = "vgio/units/v2", world = "mm", image = "px")))
        @test_throws ArgumentError load(ColmapModel, dir)
        write(sidecar, JSON.json((schema = "vgio/units/v2", world = "px")))
        @test_throws ArgumentError load(ColmapModel, dir)
    end

    # A unit that is not one of the registered units has no serialized
    # spelling — mm/px is a unit, but no registry symbol names it.
    @test_throws ArgumentError unit_string(mm / VgtkCore.px)   # no registered spelling

    # A quaternion with w < 0 is refused on read, never flipped silently.
    @test_throws ArgumentError VgtkIO.pose_from_json(parse_json("""{"rotation": [-0.5, 0.5, 0.5, 0.5], "translation": [0, 0, 0]}"""), "mm")
    # A retired kind is refused by name, with its successor.
    mktempdir() do dir
        for (s, successor) in (("vgio/point_observations/v1", "vgio/observations/v3.0"), ("vgio/session/v3", "session kind"),
                               ("vgio/correspondences/v2", "export/correspondences/v2"), ("vgio/observations/v2", "vgio/observations/v3.0"))
            p = joinpath(dir, "retired.json")
            write(p, JSON.json((schema = s,)))
            e = try VgtkIO.validate(p); nothing catch e; e end
            @test e isa ArgumentError && occursin("retired $(s)", e.msg) && occursin(successor, e.msg)
        end
        # a later major than this reader's: refused, naming the update
        p = joinpath(dir, "later.json")
        write(p, JSON.json((schema = "vgio/transforms/v99.0",)))
        e = try load(Vector{EuclideanMap}, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("later major", e.msg)
    end

    # An unknown generic-camera type is a corrupt record, not a default.
    mktempdir() do dir
        path = joinpath(dir, "g.json")
        write(
            path, JSON.json(
                (
                    schema = "vgio/generic_camera/v2", pixel_frame = "colmap", type = "Fisheye9000",
                    width = 4, height = 4,
                    calibration_min_x = 0, calibration_min_y = 0,
                    calibration_max_x = 3, calibration_max_y = 3,
                    grid_width = 1, grid_height = 1, grid_layout = "row_major",
                    direction_grid = [[0.0, 0.0, 1.0]],
                )
            )
        )
        e = try load(BoundedCamera, path); nothing catch e; e end
        @test e isa ArgumentError && occursin("Fisheye9000", e.msg)
    end
end

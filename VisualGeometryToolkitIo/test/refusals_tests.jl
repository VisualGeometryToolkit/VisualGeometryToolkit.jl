# =============================================================================
# refusals the happy path never reaches
# =============================================================================

@testitem "refusals the happy path never reaches" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIo as VgtkIo
    # Every branch here is a REFUSAL, and refusing correctly is most of what
    # this package promises — so each one is asserted rather than left to the
    # first caller to discover. Named by a coverage audit, not by taste.
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 800.0, 0, 320.5, 240.5, 1))
    cam = CameraModel(K; lens = Pinhole(), image_origin = :colmap)
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
        sidecar = joinpath(dir, VgtkIo.UNITS_SIDECAR)
        @test JSON3.read(read(sidecar, String)) == Dict(:schema => "vgio/units/v2", :world => "mm")
        write(sidecar, JSON3.write((schema = "vgio/units/v1", world = "m", image = "px")))
        err = try
            load(ColmapModel, dir); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("retired vgio/units/v1", err.msg)
        # a v2 sidecar holds exactly its declaration
        write(sidecar, JSON3.write((schema = "vgio/units/v2", world = "mm", image = "px")))
        @test_throws ArgumentError load(ColmapModel, dir)
        write(sidecar, JSON3.write((schema = "vgio/units/v2", world = "px")))
        @test_throws ArgumentError load(ColmapModel, dir)
    end

    # A unit that is not one of the registered units has no serialized
    # spelling — mm/px is a unit, but no registry symbol names it.
    @test_throws ArgumentError unit_string(mm / VgtkCore.px)   # no registered spelling

    # An unknown generic-camera type is a corrupt record, not a default.
    mktempdir() do dir
        path = joinpath(dir, "g.json")
        write(
            path, JSON3.write(
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
        @test_throws ArgumentError load(CentralGenericModel, path)
    end
end

# Generator of the current-version fixtures that the spline lens (VisualGeometryToolkitCore 0.9: generic_camera v4,
# rig v9) changed, from their old-version inputs, THROUGH THE MIGRATIONS — never by hand:
#   units/rig.json, rig_targets.json, rig_pitch.json   vgio/rig/v8 → v9 by `vgio_migrate` in place (verified)
#   units/generic_camera_v4.json                       units/generic_camera.json (generic_camera/v3, kept as a
#                                                      migration input) read through its v3 → v4 step, saved
#   units/spline_camera_masked.json                    units/generic_camera_masked.json (a central v3 grid, kept
#                                                      as a migration input) read as the camera store of its camera;
#                                                      every 4 × 4 window of its lattice holds an absent node, so it
#                                                      has no scale to gauge K by: its nominal pinhole is supplied
#                                                      (fx = fy = 640, the area's centre (32, 24): the K this fixture
#                                                      has held since 0.9's first generation), on a copy
# Run it once in the IO test env:  include(".../VisualGeometryToolkitIO/test/fixtures/make_spline_fixtures.jl")
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: BoundedCamera

let units = joinpath(@__DIR__, "units")
    for f in ("rig.json", "rig_targets.json", "rig_pitch.json")
        p = joinpath(units, f)
        VgtkIO._peek_schema(p) == "vgio/rig/v8" && VgtkIO.vgio_migrate(p; out = p)
    end
    src = joinpath(units, "generic_camera.json")
    VgtkIO.save(joinpath(units, "generic_camera_v4.json"), VgtkIO.load(BoundedCamera, src);
                VgtkIO._resave_kwargs(VgtkIO.migrate_record(VgtkIO._raw_tree(src)))...)
    mktempdir() do dir
        masked = joinpath(dir, "generic_camera_masked.json")   # the input's own name: the camera's name
        t = VgtkIO._raw_tree(joinpath(units, "generic_camera_masked.json"))
        t["nominal_intrinsics"] = Any[640.0, 640.0, 32.0, 24.0, 0.0]
        write(masked, VgtkIO.JSON.json(t))
        VgtkIO.save(joinpath(units, "spline_camera_masked.json"), VgtkIO.load(Dict{Symbol, BoundedCamera}, masked))
    end
end

# The current-version fixtures (IO 0.10) of units/, identity/ and capture_settings/, written by the 0.10 writer.
#
# IO 0.10 moved every kind to a new major (the envelope: `generator`, `units`, `conventions`). The fixtures were
# carried over from their IO 0.9 files (git history at 2f994d2b; the 0.9 files themselves are kept as the migration
# inputs under fixtures/old/) by the shape changes alone — the envelope, `t` → `frame_index`, `camera_optical` →
# `camera`, the `{width, height}` sizes, 1-based lattice runs, the lattice record's `controls`, the target body tags,
# the point vocabulary `{"xy" | "xyz", "covariance"}`, the one-camera store of the non-central record — with three
# changes of content a 0.10 file requires:
#   - every `time_origin` states its UTC offset (`Z`);
#   - target_points2d/3d became observations2d/3d (`vgio/observations/v3.0`), one point form per file: a 0.9
#     point without a covariance got one (px²: 0.25·I; 3D: zero, an exactly known point), and observations2d names
#     the 0.10 `content_id` of rig_targets.json;
#   - noncentral_spline.json became cameras_noncentral.json, the camera store of its one camera.
# This script then reads every one through its reader and writes it back through its writer, so the bytes are the
# writer's, and writes the correspondences export from the fixtures it is made of. Run it in the IO test env:
#   include(".../VisualGeometryToolkitIO/test/fixtures/make_current_fixtures.jl")
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: Rig, AbstractTarget, Trajectory, TargetPoses, TargetObservations, BoundedCamera,
    CaptureSettings, EuclideanMap, PixelFrame, parse_unit
using JSON: JSON

let dir = @__DIR__, units = joinpath(dir, "units")
    kinds = [
        "units/transforms.json" => Vector{EuclideanMap}, "units/rig.json" => Rig, "units/rig_targets.json" => Rig,
        "units/rig_pitch.json" => Rig, "units/target_points.json" => AbstractTarget, "units/target_planar.json" => AbstractTarget,
        "units/trajectory.json" => Trajectory, "units/target_poses.json" => TargetPoses,
        "units/observations3d.json" => TargetObservations, "units/cameras_noncentral.json" => Dict{Symbol, BoundedCamera},
        "units/spline_camera_masked.json" => Dict{Symbol, BoundedCamera}, "identity/rig_every_kind.json" => Rig,
        "capture_settings/capture_settings.json" => CaptureSettings,
    ]
    for (f, T) in kinds
        p = joinpath(dir, f)
        u = parse_unit(JSON.parsefile(p)["units"]["length"])
        x = VgtkIO.load(T, p)
        VgtkIO.save(p, x; length_unit = u)
    end
    # the pixels name the 0.10 identity of the rig they resolve in
    boards = joinpath(units, "rig_targets.json")
    p2 = joinpath(units, "observations2d.json")
    o = VgtkIO.load(TargetObservations, p2)
    VgtkIO.save(p2, TargetObservations{PixelFrame{:julia}}(getfield(o, :frames); stamps = o.stamps, targets = [VgtkIO.content_id(boards)]))
    VgtkIO.save_correspondences(PixelFrame{:opencv}, joinpath(units, "correspondences.json"), VgtkIO.load(TargetObservations, p2), VgtkIO.load(Rig, boards))
end

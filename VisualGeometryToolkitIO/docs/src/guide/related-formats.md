# Related Formats and Exporters

## How the kinds compare with community formats

The geometry kinds were designed against the formats the community already
uses. They keep the spirit of those formats but follow this package's rules:
one JSON schema per kind, declared conventions, typed frames, and `Uncertain`
values stored with a covariance. The exact foreign syntax is produced only by
the exporters. The references below were read from shallow clones taken on
2026-10-09.

| Kind | Related formats | Adopted | Not adopted |
|---|---|---|---|
| `transforms/v3` | ROS `geometry_msgs/TransformStamped` (`common_msgs/geometry_msgs/msg/TransformStamped.msg:1-2,8-10`: `header.frame_id` → `child_frame_id`); Kalibr camchain `T_cn_cnm1`, `T_cam_imu` (`kalibr_common/ConfigReader.py:678-717`, "LastCamToHere", "ImuToCam") | both ends of a transform named in each record; what the transform does to a point, stated | the direction encoded in a variable name (`T_to_from`); ROS's x,y,z,w quaternion order (`Quaternion.msg:3-6`), where ours is w,x,y,z and declared |
| `trajectory/v3` | TUM RGB-D `t tx ty tz qx qy qz qw` (evo `file_interface.py:121-174`); EuRoC/ASL `state_groundtruth_estimate0/data.csv` (`:224-243`, ns, w first, `p_RS_R`); KITTI odometry 3×4 rows without timestamps (`:177-220`); evo stores w,x,y,z (`core/trajectory.py:64,71`) | an explicit time per pose (`time`, in seconds since a declared `time_origin`); the body → world direction; an untracked frame absent rather than NaN | a positional join as in KITTI (frames carry `t` and the `image` name); the unit and the frames are declared, not implied |
| `target_poses/v1` | BOP `scene_gt.json` (`bop_toolkit_lib/inout.py:267-300`; `docs/bop_datasets_format.md:108-112`: per image `[{cam_R_m2c, cam_t_m2c, obj_id}]`) and `scene_camera.json` (`:79-90`, `inout.py:221-260`) | the per-image object list; the model → camera direction; an undetected object absent | an integer `obj_id` (targets keep their identity); a K per image (a frame names its camera record through `sensor`) |
| `target_points3d/v1` | OpenSim TRC (`OpenSim/Common/TRCFileAdapter.cpp:236-376`, reader `:38-233`: `Frame# Time` plus X/Y/Z per marker, empty means missing); C3D (`ezc3d/README.md:526-612`: POINT data, a residual < 0 means occluded) | a time per frame; a unit; named markers | NaN or empty fields (an unseen marker is absent); the scalar residual (a covariance is stored when it is known) |
| pose covariance (every kind) | ROS `PoseWithCovariance` (`PoseWithCovariance.msg:5-9`: row-major 6×6, x y z, then rotation about X Y Z); GTSAM Pose3 tangent `[ω; v]` (`gtsam/geometry/Pose3.cpp:100`, right retraction `T·Exp(ξ)`); Sophus SE3 `[υ; ω]` (`sophus/se3.hpp:840-850`) | GTSAM's order, rotation first | GTSAM's chart. Ours perturbs the rotation on the left, in the destination frame (VGC's `RigidExpManifold6`), the chart VisualGeometryToolkitEstimation reports in. The ROS layout is the same chart with the blocks swapped (`ros_pose_covariance`). |
| `cameras/v3`, COLMAP | COLMAP `images.txt` (`doc/format.rst:143,153-163`: world → camera, Hamilton w,x,y,z); pixel centres at 0.5 (`doc/cameras.rst:52-57`) | Hamilton w,x,y,z; the COLMAP pixel frame as the canonical one | — |

## Exporters

The exporters are write-only. The tests verify each by reading its text back
(`test/rig_schema_tests.jl`).

- `save_tum_trajectory(path, tr; stamp = :time)` writes TUM: metres,
  x,y,z,w, the pose of `from` in `to`, with a `# ` comment header that evo
  skips.
- `save_euroc_trajectory(path, tr; stamp = :time)` writes EuRoC ground truth:
  nanosecond timestamps, metres, w first. evo reads 8 or more columns.
- `save_bop_scene(to_frame, dir, tp, cameras; obj_ids, undistorted)` writes
  `scene_gt.json` and `scene_camera.json` laid out as `bop_toolkit`'s
  `save_json` writes them, with K moved into `to_frame`. It returns the
  obj_id and image maps.
- `save_trc(path, obs; data_rate, stamp)` writes OpenSim TRC: tab-separated,
  in millimetres, with missing markers as empty fields.

There is no C3D writer. A maintained Python writer exists (ezc3d, on
conda-forge), but a Python dependency for one binary format is out of
proportion for this package, and TRC carries the same marker trajectories.

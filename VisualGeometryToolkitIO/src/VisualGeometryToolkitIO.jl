#=
VisualGeometryToolkitIO — serialization for transformations and cameras.

Three layers:

1. Wire types (`wire.jl`, `colmap_text.jl`, `colmap_binary.jl`) — unit-free,
   format-faithful COLMAP records plus the vgio JSON transform schema. Always
   Float64 on disk (COLMAP binary is double; JSON numbers are double).
2. Unit contract (`contract.jl`) — the checked boundary between file magnitudes
   and FlexUnits quantities. Every dimensioned number crosses through
   FlexUnits' `quantity(x, u)` (read) and `ustrip(u, q)` (write), unit names
   through VisualGeometryToolkitCore's `parse_unit`/`unit_string`; a single-argument
   `ustrip` is never used in this package.
3. Assembly (`assemble.jl`) — typed conversion to/from VisualGeometryToolkitCore's
   `CameraModel` and rigid `EuclideanMap`s, quantity-first so VGC's construction-boundary
   unit enforcement is unavoidable for file data.

EVERY KIND IS THE DIRECT SERIALIZATION OF A VisualGeometryToolkitCore TYPE: `load`
returns the VGC value, `save` takes it, and the round trip is exact by type —
no IO-side record structs. Foreign formats (COLMAP, OpenCV/MATLAB exports)
keep adapter types of their own (`ColmapModel`, `ExternalCalibration`,
`UnitContract`) that convert to and from VGC types; the exporters (TUM, EuRoC,
BOP, TRC, correspondences) take VGC values and are write-only.

METADATA: envelope for the FILE, attributes for the VALUE. The envelope holds
what is about the file — the `schema` tag, the declared conventions, the
reserved `payload`, the free-form `provenance` (a `save` keyword, never read
back) — and none of it is part of the value. A fact about the value that must
round-trip is part of the VGC value: a field (a trajectory's per-frame images
and times) or an `Attributed` attribute (a camera's or a target's
`calibration` block; a covariance, as `Uncertain`).

Every vgio file is a versioned schema (`vgio/<kind>/v<n>`: rig v4; cameras,
transforms and trajectory v3; target, target_poses, target_points2d and
target_points3d v1; the rest v2), and a file is sufficient on its own and
minimal:
- every convention a reader needs is DECLARED in the file, once — the pixel
  frame, the length unit, the rotation encoding (Hamilton quaternions, w-x-y-z,
  w ≥ 0 — a raw 3×3 never appears) and what a transform does to a point — and
  a reader refuses a file missing any of them; nothing defaults;
- every value has ONE authority: a pose's frames are given by where it sits,
  and a value derivable from another stored one is not stored beside it;
- one validator per kind (`validate.jl`) checks a file on read AND before it
  is written; `provenance` is the only free-form object, and VGIo never reads
  it;
- frame names go through one grammar (`frame_names.jl`), so a file's frames
  read back as types;
- uncertainty mirrors the type (VisualGeometryToolkitCore
  conventions/persisted-geometry.md): a `covariance` is written iff the value
  is `Uncertain`, and read back as `Uncertain`;
- `payload` is reserved for an Arrow sidecar of bulk arrays (not implemented;
  refused until a kind implements it);
- a rotation is read as the stored quaternion itself (a `QuatRotation`), so
  file → value → file is exact; any other rotation is converted on write.
Retired kinds and versions are refused by name, with their successor; nothing
reads them (their one-shot migrations ran in 2026-10 and live in the git
history). COLMAP model directories keep COLMAP's own format (world → camera
poses, pixel centres at 0.5) and get a `vgio_units.json` sidecar declaring
their world unit.
=#
"""
    VisualGeometryToolkitIO

File formats for the VisualGeometryToolkit stack. Each `vgio/<kind>` JSON file
is the serialization of one VisualGeometryToolkitCore type: `load(T, path)`
returns a value of type `T`, and `save(path, value)` writes one. Both verbs
are public but not exported, so call them qualified
(`import VisualGeometryToolkitIO as VgtkIO; VgtkIO.load(...)`).

The package also reads and writes COLMAP models, reads COLMAP databases and
OpenCV/MATLAB calibration exports, and exports to TUM, EuRoC, BOP and TRC.
"""
module VisualGeometryToolkitIO

import VisualGeometryToolkitCore as VgtkCore

using StaticArrays: SMatrix, SVector, StaticMatrix
using LinearAlgebra: norm
using Statistics: cov
using Rotations: QuatRotation, RotMatrix, RotationVec, RotZ, params
using JSON: JSON
using SQLite: SQLite
using FlexUnits: Units, ustrip, dimension, quantity
using VisualGeometryToolkitCore: parse_unit, unit_string, mm, m,
    CameraModel, CameraCalibrationMat, AbstractLensModel, Lens, Pinhole, Division, KannalaBrandt, Scaramuzza,
    Perspective, Equidistant, DoubleSphere, Unified, ExtendedUnified,
    Radial, Tangential, ThinPrism, Tilt, FieldOfView, Additive, Chain, Even,
    lens_record, lens_from_record,
    EuclideanMap, Size2, Vec3, PixelFrame,
    Rig, Mounted, BoundedCamera, Camera, Attributed, sensor_rig, target_rig,
    AbstractTarget, PointTarget, PlanarTarget, Rigid, is_resolved,
    Trajectory, TargetPoses, TargetPoints2D, TargetPoints3D, TargetPoint2D, TargetPoint3D,
    CentralGenericModel, NoncentralGenericModel,
    Frame, WorldFrame, CameraFrame, ViewFrame, TargetFrame, BodyFrame, UnlabeledFrame,
    Uncertain, CovMat, geom,
    relabel, ids, pose, extrinsics, reference_frame, model, extent, points
import VisualGeometryToolkitCore

# =============================================================================
# The I/O surface: TWO verbs, dispatched on the VGC type.
# =============================================================================
#
# DELIBERATELY UNEXPORTED. FileIO and downstream packages (BlobBoards) export
# `load`; qualify as `VisualGeometryToolkitIO.load` (or alias the module:
# `import VisualGeometryToolkitIO as VGIo; VGIo.load(...)`).
#
# The typed load is an ASSERTION OF INTENT: `load(Trajectory, path)` on a
# transforms file errors instead of guessing, and the return type is what you
# asked for. `save` needs no type argument; the value already carries it.
#
#   load(Dict{Symbol, BoundedCamera}, path)      vgio/cameras/v3     the camera store
#   load(BoundedCamera | CameraModel, path; name = nothing)          one of its cameras (multi refused unnamed)
#   load(Vector{EuclideanMap}, path)             vgio/transforms/v3  labeled poses
#   load(Rig, path; resolve = nothing)           vgio/rig/v4         sensors, targets, or both
#   load(AbstractTarget, path; resolve = nothing) vgio/target/v1     one target (+ its calibration block)
#   load(Trajectory, path)                       vgio/trajectory/v3  a body's pose over time
#   load(TargetPoses, path)                      vgio/target_poses/v1
#   load(TargetPoints2D, path)                   vgio/target_points2d/v1
#   load(TargetPoints3D, path)                   vgio/target_points3d/v1
#   load(CentralGenericModel | NoncentralGenericModel, path)   the Schöps grid models
#   load(ColmapModel, dir; contract = nothing)   COLMAP model directory (foreign)
#   ColmapDatabase(dbpath)                       COLMAP sqlite database (lazy handle, read-only)
#
#   save(path, value; provenance = (;))          every kind above, dispatched on the value
#   save(dir, model::ColmapModel; format = :binary)
#
#   save_correspondences, save_tum_trajectory, save_euroc_trajectory, save_bop_scene, save_trc
#                                                exporters: VGC values in, a foreign shape out
#   pose_json / pose_from_json, camera_json / camera_from_json, target_json / target_from_json,
#   pose_declarations                            building blocks for a package's own kind
#
# PIXEL DATA CARRIES ITS FRAME IN ITS TYPE (`TargetPoints2D{F}`): the producer
# attaches it, `save` writes it as the file's declaration without converting a
# number, `load` returns the file's frame as the type, and VGC's
# `convert_frame(G, x)` — which moves the numbers — is the only way to change
# it. The foreign formats' readers state the FILE's frame (`file_frame`), the
# exporters their TARGET frame (`to_frame`).
#
# docs/src/guide/kinds.md tables every format with its reader and writer.
function load end
function save end

include("json.jl")
include("validate.jl")    # the v2 spec combinators + read/write boundaries (before every schema: specs are consts)
include("contract.jl")
include("wire.jl")
include("colmap_text.jl")
include("colmap_binary.jl")
include("colmap_database.jl")
include("frame_names.jl")     # the frame grammar: frame types ↔ names (every pose-carrying kind)
include("transform_schema.jl")
include("camera_schema.jl")
include("generic_camera_schema.jl")  # the Schöps grid models (must precede rig_schema: sensors embed it)
include("rig_schema.jl")
include("captures_schema.jl")  # trajectory/v3, target_poses/v1, target_points2d/v1, target_points3d/v1: VGC's captures
include("correspondences.jl")  # export: measurements + targets → per-(frame, target) point arrays for external calibrators
include("assemble.jl")
include("conventions.jl")  # OpenCV's and Kalibr's lens models as VGC lenses, exactly
include("external_calibration.jl")  # OpenCV / MATLAB calibration exports (load-only)
include("exporters.jl")    # TUM, EuRoC, BOP, TRC: our kinds in the community formats (write-only)
include("jsonl.jl")
include("canonical.jl")   # canonical JSON + sha256_hex — content identities (hoisted from BlobBoards)
include("migrate.jl")     # migrate_opencv / migrate_matlab + the bin/vgio_migrate.jl command

# Layer 2 — unit contract (the COLMAP adapter's world-unit declaration)
export UnitContract, MM_PX
export UnknownScale                   # the contract of an SfM model whose world has no unit (explicit, never a default)

# Layer 1 — foreign adapters + the frame grammar
export ColmapCamera, ColmapImage, ColmapPoint3D
export ColmapDatabase
export colmap_cameras
export frame_name, frame_from_name
export content_id                     # the identity a measurement names its target rig by
export save_correspondences           # export: measurements + targets → external calibrators' point arrays
export save_tum_trajectory, save_euroc_trajectory, save_bop_scene, save_trc   # exporters to the community formats
export ros_pose_covariance, pose_covariance_from_ros   # the pose covariance in ROS PoseWithCovariance order
export migrate_opencv, migrate_matlab # a third-party OpenCV / MATLAB calibration → native v3 camera + image-named poses
export append_jsonl
export canonical_json, sha256_hex

# Layer 3 — assembly to/from VGC types
export camera_model, colmap_camera, colmap_image, rigid_transform
export colmap_model, colmap_params
export opencv_lens, opencv_coefficients, kalibr_lens   # the OpenCV and Kalibr lens conventions

# Third-party calibration exports (load-only): OpenCV and MATLAB files
export ExternalCalibration, load_opencv_calibration, load_matlab_calibration
export save_opencv_calibration        # the inverse of load_opencv_calibration: our camera in the foreign shape

# Public but UNEXPORTED — reached qualified (`VisualGeometryToolkitIO.load`, or
# `import VisualGeometryToolkitIO as VGIo`), because the names are generic enough to
# collide with FileIO's, BlobBoards' or a caller's own.
public load, save                     # the two-verb I/O surface (see the top of this file)
public ColmapModel                    # a COLMAP model directory's contents + its unit contract
public validate                       # check any vgio file against its kind's table
public pixel_frame_name, pixel_frame_from_name   # a pixel frame's serialized name, and back (VGC decides which exist)
public pose_json, pose_from_json, pose_declarations, camera_json, camera_from_json,
    target_json, target_from_json     # building blocks for a package's own kind (a calibration pipeline's session)
public n_images, image_names, image_id, image_name, # ColmapDatabase accessors
    keypoints, descriptors, matches, matched_pairs, verified_matches,
    pair_id, pair_image_ids

end # module

#=
VisualGeometryToolkitIO — serialization for transformations and cameras.

Three layers:

1. Wire types (`wire.jl`, `colmap_text.jl`, `colmap_binary.jl`) — unit-free,
   format-faithful COLMAP records plus the vgio JSON transform schema. Always
   Float64 on disk (COLMAP binary is double; JSON numbers are double).
2. Unit contract (`lengths.jl`, `contract.jl`) — the checked boundary between
   file magnitudes and FlexUnits quantities. Memory holds lengths as base-form
   quantities in metres, a file in its declared unit; every file length
   crosses through `read_length(x, u)` and `write_length(q, u)` (decimal-exact
   for a power-of-ten unit), a covariance entry through the same shift raised
   to its power, and a writer refuses plain numbers where a length is meant.
   Unit names go through VisualGeometryToolkitCore's `parse_unit`/`unit_string`.
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
what is about the file — the `schema` tag (`vgio/<kind>/v<major>.<minor>`),
the `generator` (this package and its version, and the calling package), the
ONE `units` object (length, time, angle, pixel frame, density, radiance and the
length roles), the ONE `conventions` object (rotation encoding, transform
convention, matrix layout, grid layout, index base 1), the reserved `payload`,
the free-form `provenance` (a `save` keyword, never read back) — and none of it
is part of the value. A fact about the value that must round-trip is part of
the VGC value: a field (a trajectory's per-frame images and times) or an
`Attributed` attribute (a camera's and a target's `calibration_record`; a
covariance, as `Uncertain`).

Every vgio file is a versioned schema; the current versions are the table
`_SCHEMA_VERSION` (validate.jl) and `schema_versions()` lists, per kind, every
version read (derived from the version steps). A file is sufficient on its own
and minimal:
- every convention a reader needs is DECLARED in the file, once, in its
  envelope — the pixel frame, the length unit, the rotation encoding (Hamilton
  quaternions, w-x-y-z, w ≥ 0 — a raw 3×3 never appears; a stored w < 0 is
  refused), what a transform does to a point, the index base — and a reader
  refuses a file missing any of them; nothing defaults;
- every value has ONE authority: a pose's frames are given by where it sits,
  and a value derivable from another stored one is not stored beside it;
- one validator per kind (`validate.jl`) checks a file on read AND before it
  is written; `provenance` is the only free-form object, and VgtkIO never reads
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
EVERY EARLIER VERSION A PUBLISHED FILE HOLDS IS READ, and only the current one
is written. A schema change must not force the published data (Zenodo, Hugging
Face; multi-GB archives) to be uploaded again, so a published file keeps
loading: a reader that meets an earlier version runs the version steps in
memory (`migrate_record`: one `migrate_schema(::Val{schema}, tree)` step per
version, migrate.jl and migrate_v010.jl), silently but for an `@debug`, and
reads the current record they give. A reader of a version's major reads it at
any minor (a higher minor's added keys tolerated). `vgio_migrate` rewrites a
file once with the same steps; a schema's owner adds the steps of its own kinds
(`migrate_schema`) and calls `migrate_record` in its reader. A version no
published file held and no step reads is refused by name, with its successor.
COLMAP model directories keep COLMAP's own format (world → camera poses, pixel
centres at 0.5) and get a `vgio_units.json` sidecar declaring their world unit.
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

using StaticArrays: SMatrix, SVector, StaticMatrix, StaticVector
using LinearAlgebra: I, norm, normalize, Diagonal
using VisualGeometryToolkitCore: ScaledIdentity   # an isotropic covariance's storage (λ·I): its wire form {"isotropic": λ}
using VisualGeometryToolkitCore: AbstractVignetting, TensorProductVignetting, RadialVignetting, LogVignetting, knot_vectors,
    spline_degree, control_points, field_radius, log_spline, with_control_points, VIGNETTING_TAGS, RADIAL_CHARTS, radial_chart,
    EnvironmentLight, SphericalHarmonicMap, coefficients, max_degree, GaussianBlur, with_unit, ImageFrame   # the radiometric kinds (radiometry_schema.jl)
using Statistics: cov
using Printf: @sprintf
using Base.ScopedValues: ScopedValue, with
using Rotations: QuatRotation, RotMatrix, RotationVec, RotZ, Angle2d, params
using JSON: JSON
using SQLite: SQLite
using Libtiff_jll: libtiff   # the stack's one TIFF codec (tiff.jl)
using TOML: TOML
using FlexUnits: Units, Quantity, ustrip, dimension, uscale, quantity
using CoordinateTransformations: ComposedTransformation   # a target calibration's composition
using VisualGeometryToolkitCore: BaseQuantity, parse_unit, unit_string, mm, m, px, dpi, has_units,
    CameraModel, CameraCalibrationMat, AbstractLensModel, Lens, LiftLens, Pinhole, Division, KannalaBrandt, Scaramuzza,
    Perspective, Equidistant, DoubleSphere, Unified, ExtendedUnified,
    Radial, Tangential, ThinPrism, Tilt, FieldOfView, Additive, Even,
    lens_record, lens_from_record, sha256_hex,
    EuclideanMap, Size2, Vec3, PixelFrame,
    Rig, PosedTarget, BoundedCamera, PosedCamera, Attributed, value_of, sensor_rig, target_rig, Sensor, sensor,
    AbstractTarget, PointTarget, PlanarTarget, TargetCalibration, calibration, reference, is_resolved, Board, board,
    Canvas, CanvasTexture, canvas, placement, SimilarityMap, scale, translation,
    AbstractTargetLayout, AprilGrid, SchoepsPattern, layout, design, TARGET_LAYOUT_TAGS, tag_of,
    Rigid, ThinPlateSpline, HeightField, Scale, Aspect, Skew, TARGET_CALIBRATION_TAGS, calibration_tag, parameter_vector, with_parameters, length_mask,
    TARGET_CALIBRATION_ORDER, planar_kind, members, is_pure_bending, ThinPlateSplineSystem, iwasawa_parameters_with_jacobian, strip_or_bare,
    Trajectory, TargetPoses, TargetHomographies, HomographyMap, HomographyMat, PLANAR_MAP_TAGS,
    TargetObservations, Correspondences, correspondences, withattrs, sources, targets,
    SplineLens, NoncentralSplineModel, ChartSpline, LogicalIntrinsics, Ellipse, Circle, Rect2, RegionIntersection, WholePlane, lens_coverage,
    Frame, WorldFrame, CameraFrame, ViewFrame, TargetFrame, BodyFrame, UnlabeledFrame,
    Uncertain, CovMat, PrecMat, geom, frame_dof,
    Point2, FeatureFrame, PointFrame, Disc, OrientedDisc, EllipticFrame, AffineFrame, AbstractDetection,
    relabel, ids, pose, extrinsics, reference_frame, model, extent, points
import VisualGeometryToolkitCore
using VisualGeometryToolkitCore: FitResult   # a fit's termination record (validate.jl, `_FIT_RECORD`)
using VisualGeometryToolkitCore: DeviceIdentity, ExposureSettings, LensSettings, FrameSettings, CaptureSettings   # capture_settings/v1, a calibration's device
using VisualGeometryToolkitCore: SensorNoise, BinnedRadialProfile   # sensor_noise/v1.0, radial_profile/v1.0

# =============================================================================
# The I/O surface: TWO verbs, dispatched on the VGC type.
# =============================================================================
#
# DELIBERATELY UNEXPORTED. FileIO and downstream packages (BlobBoards) export
# `load`; qualify as `VisualGeometryToolkitIO.load` (or alias the module:
# `import VisualGeometryToolkitIO as VgtkIO; VgtkIO.load(...)`).
#
# The typed load is an ASSERTION OF INTENT: `load(Trajectory, path)` on a
# transforms file errors instead of guessing, and the return type is what you
# asked for. `save` needs no type argument; the value already carries it.
#
#   load(Dict{Symbol, BoundedCamera}, path)      vgio/cameras/v6.0   the camera store (any bounded camera, central or not)
#   load(BoundedCamera | CameraModel, path; name = nothing)          one of its cameras (multi refused unnamed); the
#                                                grid cameras of Schöps et al. too (old grid files, a vgio/noncentral_spline/v1
#                                                record, intrinsics0.yaml)
#   load(Vector{EuclideanMap}, path)             vgio/transforms/v5.0  labeled poses
#   load(Rig, path; resolve = nothing)           vgio/rig/v10.0      sensors, targets, or both
#   load(AbstractTarget, path; resolve = nothing) vgio/target/v6.0   one target (+ its calibration record)
#   load(PlanarTarget, path)                     a planar vgio/target, or a Schöps pattern YAML or a Kalibr aprilgrid YAML
#   load(Trajectory, path)                       vgio/trajectory/v5.0  a body's pose over time
#   load(TargetPoses, path)                      vgio/target_poses/v3.0
#   load(TargetHomographies, path)               vgio/target_homographies/v2.0 (and `load(TargetHomographies{F}, path)`)
#   load(TargetObservations, path)               vgio/observations/v3.0 (IO 0.9's target_points2d/3d and target_detections
#                                                through their steps; a Schöps dataset.bin with `target`)
#   load(CaptureSettings, path)                  vgio/capture_settings/v2.0 (Attributed with its identities' sources)
#   load(AbstractVignetting | EnvironmentLight | Dict{Symbol, GaussianBlur}, path)   vignetting, environment_light, blur v2.0
#   load_panorama(path) / save_panorama(path, camera, rotations, vignetting, gains)
#                                                vgio/panorama/v2.0  a panorama's results (a NamedTuple: no VGC type)
#   load(Dict{String, <:AbstractVector{<:AbstractDetection}}, path)  vgio/detection/v1.0  detections per image (the one frame codec)
#   load(BinnedRadialProfile, path)              vgio/radial_profile/v1.0  a measured radial profile (a flat field by field angle)
#   load(SensorNoise, path)                      vgio/sensor_noise/v1.0  a body's noise at an ISO, its dark maps as hashed artifacts
#   load_raster(path) / save_raster(path, values; frame, nan)
#                                                vgio/raster/v2.0    a Float32 or UInt8 raster: a JSON header + its TIFF payload
#   load(ColmapModel, dir; contract = nothing)   COLMAP model directory (foreign)
#   ColmapDatabase(dbpath)                       COLMAP sqlite database (lazy handle, read-only)
#
#   save(path, value; provenance = (;))          every kind above, dispatched on the value
#   save(dir, model::ColmapModel; format = :binary)
#
#   save_correspondences, save_tum_trajectory, save_euroc_trajectory, save_bop_scene, save_trc
#                                                exporters: VGC values in, a foreign shape out
#   VgioStyle                                    the style every embedded VGC value is lowered/lifted on (style.jl)
#   pose_json / pose_from_json, camera_json / camera_from_json, target_json / target_from_json,
#   target_calibration_json / target_calibration_from_json, lattice_json / lattice_from_json
#                                                building blocks for a package's own kind
#   write_record / read_record, VgioStyle(record), migrate_envelope, migrate_units, with_caller
#                                                a package's own kind in the vgio envelope, and its version steps
#
# PIXEL DATA CARRIES ITS FRAME IN ITS TYPE (`TargetObservations{F}`): the producer
# attaches it, `save` writes it as the file's declaration without converting a
# number, `load` returns the file's frame as the type, and VGC's
# `convert_frame(G, x)` — which moves the numbers — is the only way to change
# it. The foreign formats' readers state the FILE's frame (`file_frame`), the
# exporters their TARGET frame (`to_frame`).
#
# docs/src/guide/kinds.md tables every format with its reader and writer.
function load end
function save end

include("lengths.jl")     # read_length / write_length: decimal-exact file lengths (before every schema)
include("json.jl")
include("contract.jl")    # UnitContract: a file's length units (its roles) and density
include("style.jl")       # VgioStyle: the lower/lift of every embedded VGC value, the parametric record, the minimal forms
include("validate.jl")    # the spec combinators, the envelope, the read/write boundaries (before every schema: specs are consts)
include("units_sidecar.jl")  # vgio/units: a COLMAP model directory's units sidecar
include("core_json.jl")     # quantities, Size2 and THE feature-frame codec on the style (JSON.lower / JSON.lift)
include("wire.jl")
include("colmap_text.jl")
include("colmap_binary.jl")
include("colmap_database.jl")
include("frame_names.jl")     # the frame grammar: frame types ↔ names (every pose-carrying kind)
include("transform_schema.jl")
include("camera_schema.jl")
include("rig_schema.jl")
include("captures_schema.jl")  # trajectory, target_poses, target_homographies, observations: VGC's captures
include("capture_settings_schema.jl")  # capture_settings: the camera's settings per frame, the device identities
include("radiometry_schema.jl")  # vignetting, environment_light, blur: a radiometric camera's fields, light and blurs
include("panorama_schema.jl")    # vgio/panorama/v2.0: a panorama's results, composed of the camera, trajectory and vignetting bodies
include("detection_schema.jl")   # vgio/detection/v1.0: detections per image through the one frame codec (core_json.jl)
include("radial_profile_schema.jl")  # vgio/radial_profile/v1.0: a measured radial profile, binned (Core's BinnedRadialProfile)
include("calibration_manifest.jl")  # calibration.json: vgio/calibration, the hash-bound set of a result's records, any owner's (after the kinds it checks)
include("tiff.jl")         # the stack's one TIFF codec (libtiff): write_tiff / read_tiff
include("raster_schema.jl")  # vgio/raster/v2.0: a raster, a JSON header and its TIFF payload (the one bulk-array binding; after the manifest: its path rule)
include("sensor_noise_schema.jl")  # vgio/sensor_noise/v1.0: a body's SensorNoise, its dark maps and defect mask as hashed artifacts
include("json_schema.jl")   # the published JSON Schemas (draft 2020-12), generated from the validator's tables
include("schoeps.jl")      # Schöps et al.'s camera_calibration files (intrinsics0.yaml, dataset.bin), read through `load`
include("correspondences.jl")  # export/correspondences/v2: observations + targets → per-(frame, target) point arrays for external calibrators
include("assemble.jl")
include("conventions.jl")  # OpenCV's and Kalibr's lens models as VGC lenses, exactly
include("external_calibration.jl")  # OpenCV / MATLAB calibration exports (load-only)
include("exporters.jl")    # TUM, EuRoC, BOP, TRC: our kinds in the community formats (write-only)
include("jsonl.jl")
include("canonical.jl")   # RFC 8785 (JCS) canonical JSON — content identities, hashed by Core's sha256_hex
include("migrate.jl")     # the version steps, vgio_migrate, vgio_migrate_tree + the bin/vgio_migrate.jl command
include("migrate_v010.jl")   # the IO 0.10 steps: every kind's previous version → its 0.10 major

# Layer 2 — unit contract (the COLMAP adapter's world-unit declaration)
export UnitContract, MM_PX
export UnknownScale                   # the contract of an SfM model whose world has no unit (explicit, never a default)

# Layer 1 — foreign adapters + the frame grammar
export ColmapCamera, ColmapImage, ColmapPoint3D
export ColmapDatabase
export colmap_cameras
export frame_name, frame_from_name
export content_id                     # the identity an observation names its target rig by
export save_correspondences           # export: observations + targets → external calibrators' point arrays
export save_tum_trajectory, save_euroc_trajectory, save_bop_scene, save_trc   # exporters to the community formats
export ros_pose_covariance, pose_covariance_from_ros   # the pose covariance in ROS PoseWithCovariance order
export convert_opencv_calibration, convert_matlab_calibration   # a third-party OpenCV / MATLAB calibration → a camera store + image-named poses ("migrate" is schema migration only)
export vgio_migrate, vgio_migrate_tree   # an earlier schema version → the current one, once; a tree's observations re-pointed at its migrated rigs
export append_jsonl
export canonical_json   # its hash is Core's `sha256_hex(canonical_json(tree))`

# Layer 3 — assembly to/from VGC types
export camera_model, colmap_camera, colmap_image, rigid_transform
export colmap_model, colmap_params
export opencv_lens, opencv_coefficients, kalibr_lens   # the OpenCV and Kalibr lens conventions
export opencv_camera, opencv_parameters                 # OpenCV's camera matrix (its pixel frame) ↔ a camera in the COLMAP frame

# Third-party calibration exports (load-only): OpenCV and MATLAB files
export ExternalCalibration, load_opencv_calibration, load_matlab_calibration, load_kalibr_camchain
export save_opencv_calibration        # the inverse of load_opencv_calibration: our camera in the foreign shape

# Public but UNEXPORTED — reached qualified (`VisualGeometryToolkitIO.load`, or
# `import VisualGeometryToolkitIO as VgtkIO`), because the names are generic enough to
# collide with FileIO's, BlobBoards' or a caller's own.
public load, save                     # the two-verb I/O surface (see the top of this file)
public VgioStyle                      # the serialization style every embedded VGC value is lowered and lifted on (style.jl)
public Record, snap_integer, MATRIX_LAYOUT   # what a lift reads; a file number as a record writes it; the matrix layout a record declares
public write_record, read_record      # the vgio file boundary for a record of another package's kind (validate.jl)
public with_provenance, metres        # a record body with the writer's provenance; a file length as plain metres (a kind of another package's)
public SCHOEPS_PIXEL_FRAME            # the pixel frame of Schöps' camera_calibration files (COLMAP's)
public ColmapModel                    # a COLMAP model directory's contents + its unit contract
public read_length, write_length   # a file length <-> metres, decimal-exact (power-of-ten units shift the exponent)
public role_unit, migrate_units   # a contract's length roles; the 0.10 units object of a record in the retired UnitContract layout (core_json.jl)
public save_manifest, load_manifest     # vgio/calibration: the manifest binding a result's record files (any owner's) by SHA-256
public save_panorama, load_panorama     # vgio/panorama/v2.0: a panorama's camera, rotations, vignetting and gains
public save_raster, load_raster       # vgio/raster/v2.0: a raster, its JSON header and its TIFF payload
public write_tiff, read_tiff          # the stack's one TIFF codec (libtiff): a grey UInt8 or Float32 image
public validate                       # check any vgio file against its kind's table
public json_schema                    # a kind's JSON Schema (draft 2020-12), generated from that table
public migrate_schema, validate_record, CONTRACT_GENERICS   # the per-schema migration and the record check a schema owner extends (its declared extension points)
public schema_versions, with_caller   # the versions each kind reads (from its steps); the calling package a file's generator names
public migrate_envelope               # a pre-0.10 record's envelope as the 0.10 one: an owner's version step
public migrate_record                 # an earlier version's record as the current one, in memory: the version steps a reader runs
public same_value                     # two typed records the same, a length within the codec's ulps (a migration's check)
public write_staged                   # files written through a staging directory beside them, renamed into place once verified
public pixel_frame_name, pixel_frame_from_name   # a pixel frame's serialized name, and back (VGC decides which exist)
public pose_json, pose_from_json, camera_json, camera_from_json,
    target_json, target_from_json,    # building blocks for a package's own kind (a calibration pipeline's session)
    target_calibration_json, target_calibration_from_json   # a target's calibration record alone (an in-plane print error)
public lattice_json, lattice_from_json   # THE lattice record: a uniform cubic B-spline over a rectangle (a spline lens's, an aperture spline's)
public n_images, image_names, image_id, image_name, # ColmapDatabase accessors
    detections, descriptors, raw_matches, matched_pairs, verified_matches,
    pair_id, pair_image_ids

"""
    CONTRACT_GENERICS

VisualGeometryToolkitIO's declared extension points: the functions another
stack package may add methods to (VisualGeometryToolkitCore's
`contract_generic_violations` checks it). [`migrate_schema`](@ref): the owner
of a schema adds the step of each of its old versions, so its reader reads
them ([`migrate_record`](@ref)), and the file form, so [`vgio_migrate`](@ref)
rewrites its files. [`load`](@ref) and [`save`](@ref), the two verbs: a
package whose own kinds are files (BlobBoards' boards and patterns) adds
`load(::Type{<:ItsType}, path)` and `save(path, x::ItsType)` methods, so a
reader reaches every kind of the stack through the same two verbs.
"""
const CONTRACT_GENERICS = (migrate_schema, validate_record, load, save)

end # module

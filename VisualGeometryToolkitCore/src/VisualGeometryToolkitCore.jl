"""
    VisualGeometryToolkitCore

Geometry, camera models and the robust-estimation core of the visual-geometry
stack. VisualGeometryToolkitFeatures, VisualGeometryToolkitIO, VisualGeometryToolkitEstimation and
BlobBoards build on it.

# Submodules
- `Primitives`: conics and feature frames (`Ellipse`, `Circle`, `Disc`,
  `OrientedDisc`, `EllipticFrame`, `AffineFrame`), hyperplanes (`Line2D`,
  `Plane3D`), 3-D conics (`Ellipse3D`, `Circle3D`) and `Paraboloid3D`.
- `Cameras`: sensors (`CMOS_SENSORS`), intrinsics, lens models (`Lens`,
  `LiftLens`, `Pinhole`, `KannalaBrandt`, `Division`, …), `PosedCamera`, `CameraModel`,
  `Rig`, projection, calibration fits and P3P.
- `Synthetic`: ground-truth generators and metrics for benchmarking estimators.

# Key types in the parent module
- Transform matrices: `AbstractHomMat ⊃ AbstractAffineMat ⊃ AbstractSimilarityMat
  ⊃ AbstractEuclideanMat`, plus `FundMat`, `EssentialMat` and `CameraCalibrationMat`.
- Frame-tagged maps: `EuclideanMap` and `HomographyMap` carry `Src`/`Dst` tags from
  the open `Frame` hierarchy (`ViewFrame`, `PixelFrame`, `GnomonicFrame`, …);
  composition and `inv` are frame-checked.
- Minimal solvers (`homography_4pt`, `p3p`, the 5-point essential and the
  fundamental-matrix solvers) and the manifolds their models are refined on.
- The robust-loss family (`AbstractLossShape`, `deploy`), the block-coordinate-descent/LO interface
  shared with VisualGeometryToolkitEstimation, and `FitResult`, the one fit return type.
- Numerics: the B-spline kernels and the thin-plate spline (`ThinPlateSplineSystem`,
  `tps_coefficients`, `tps_evaluate`, `tps_gradient`), public, not exported.
- Units: FlexUnits as shipped, plus the registered `px`, `pd`, `dpi` and `pt` (`mm`, `px`, `dpi`, `Size2`).

Plotting loads as a package extension with any Makie backend.

See `ARCHITECTURE.md` (module graph, include order, extension points) and
`LOSSES.md` (the robust-loss family) at the repository root.
"""
module VisualGeometryToolkitCore

# Geometry and coordinate systems
using GeometryBasics:
    GeometryBasics, Vec, Vec2, Vec3, HyperRectangle, Point2f,
    Rect2, radius
import GeometryBasics: Point, Point2, Point3, Rect, Circle
using StaticArrays:
    StaticArrays, @SMatrix, FieldVector, SVector, SMatrix, MVector, MMatrix, SOneTo, StaticArray, StaticVector, StaticMatrix, similar_type
using Statistics: Statistics
using Distributions: Distributions
using LinearAlgebra: LinearAlgebra, Diagonal, UniformScaling, Symmetric, Adjoint, Transpose,
    cholesky, lu, issuccess, cross, dot, norm, det, I, eigen
using CoordinateTransformations:
    CoordinateTransformations,
    PerspectiveMap,
    AffineMap,
    LinearMap,
    Translation,
    transform_deriv,
    transform_deriv_params
using Rotations: Rotations, RotMatrix, RotationVec, Rotation, QuatRotation
using LogExpFunctions: xlogx, logit, log1pexp   # the Welsch conjugate's w·log(w); the mixture's log-odds and softplus
using SpecialFunctions: loggamma, digamma, trigamma   # the Student-t normaliser and the ν M-step (robust/mixture.jl)
using QuadGK: quadgk   # a loss shape's truncated normaliser (robust/normaliser.jl)
using Random: Random
import NLLSsolver   # the stack's Levenberg–Marquardt: the chart LM's driver, the variable and robustifier protocols (robust/lm_core.jl)


# Units and physical quantities — FlexUnits as shipped (see base/units.jl)
import FlexUnits
using FlexUnits: Quantity, Units, UnitRegistry, ustrip, uconvert, dimension, ubase,
    DimensionError, ConversionError
using FlexUnits.UnitRegistry: @ud_str, uparse, qparse

# Data structures
# JSON3/StructTypes serialization moved to VisualGeometryToolkitFeatures

# Functional updates
using Accessors: Accessors

# Colors (still needed for image types, independent of plotting)
using Colors: Colors, Gray
using FixedPointNumbers: FixedPointNumbers, N0f8, N0f16

# Scale space and image processing functionality
using IntervalSets: IntervalSets, ClosedInterval
using StructArrays: StructArrays

# =============================================================================
# EXPORTS - Organized by category
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GEOMETRIC PRIMITIVES - Points, shapes, and coordinate types
# -----------------------------------------------------------------------------
export Point2, Point3, Point2f, Vec2, Vec3, Rect, Rect2, HyperRectangle
export Circle, Ellipse, AbstractConic3D, Ellipse3D, Circle3D
public RegionIntersection, WholePlane   # `a ∩ b` of two 2-D regions, held lazily, and the identity of ∩ (primitives/regions.jl)
export boundary_point, boundary_points
export centroid, area
export moment_matrix, characteristic_scale
export iou, overlap_area, coverage, best_iou_match
public overlap_method, geom_rank

# Feature frames (transformation-based geometric primitives)
export FeatureFrame, Disc, OrientedDisc, EllipticFrame, AffineFrame
export Paraboloid3D, coeffs
export is_valid_shape
public DEFAULT_MAX_ASPECT_RATIO   # the default `is_valid_shape` bound, reached qualified
export is_peak
export boundary, region, scale, shape_matrix
export bbox

# Compute backend dispatch tags (shared across the package family)
export ClosedInterval  # Re-export from IntervalSets
export default_device, reset_default_device!   # the ONE KA backend resolver (base/device.jl)

# Uncertainty types and accessors
export Uncertain, PrecMat, CovMat
export geom, prec   # `cov` is reachable via `Statistics.cov` (extended for Uncertain)
export exact        # a zero-covariance measurement (the measurement-noise rule)
public pair_covariance, noise_columns   # a correspondence's 4×4 covariance; Uncertain endpoints → Σ₁/Σ₂ columns

# Hyperplanes (lines in 2D, planes in 3D)
export Hyperplane, Line2D, Plane3D
export normal, offset, distance, closest_point, project, side
export slope, intercept, clip_to_rect
# Batched SoA ray-plane intersection (0.14; hoisted from RayTracedCalibration):
# broadcasted hit points + ray parameters of SoA ray bundles against a Plane3D.
public ray_plane_intersection
# The surface a ray meets (geometry/primitives/surfaces.jl): one protocol,
# isbits and allocation-free. The verbs are generic words, so they stay public
# (qualified access), as `model` does.
export Quad, HeightSurface
public intersection, chart, bounds, surface, surface_point

# -----------------------------------------------------------------------------
# 2. TRANSFORMS - Maps for transforming geometry
# -----------------------------------------------------------------------------
# Primary API: Maps operate on primitives (a conic maps to a conic; see HomographyMap)
export AbstractMatrixMap         # parent for matrix-backed Maps (BoundedMap is a sibling)
export HomographyMap, EuclideanMap, CameraPose, CameraExtrinsics, Extrinsics3
public PlainEuclideanMap, UnitfulEuclideanMap   # the inner loop's map and the physical one
public rotation, translation   # EuclideanMap accessors (x ↦ R·x + t); qualified or imported by name
# Coordinate-frame tags (EuclideanMap Src/Dst parameters) + label escape hatches
export Frame, UnlabeledFrame, WorldFrame, CameraFrame, TargetPlaneFrame, PixelFrame
# View-identity tags (0.8): Symbol-parametric instance identity — ViewFrame{:left}
# is a different frame than ViewFrame{:right}; ImageFrame{id} is view id's 2D
# image frame (origin-convention conversion happens BEFORE the lift).
# GnomonicFrame{id} (0.9) is view id's CALIBRATED frame (x̂ = π(K⁻¹ũ)) — the
# frame on which EssentialMat-backed forms are honest; ImageFrame stays pixel.
# EquidistantFrame{id}/DistortedFrame{id} (0.10) complete the imaging
# vocabulary: the equidistant ray frame on S² (the omni models' defining
# frame) and the post-lens pre-K frame (see the table at DistortedFrame).
# The reserved symbol :_ is the anonymous view id (chain junctions check by
# frame KIND).
export ViewFrame, ImageFrame, TargetFrame, BodyFrame, GnomonicFrame
# NormalizedImageFrame{id}: view id's image frame centred and divided by the
# half-diagonal (sensor-independent pixel constants); its map is the factory
# FramedAffine(ImageFrame{id} or PixelFrame{O}, NormalizedImageFrame{id}, sz).
export NormalizedImageFrame
public RetinalFrame, EquidistantFrame, DistortedFrame
# The remaining retinal frames (0.13): the classical fisheye-projection
# family complete — stereographic (conformal, θ ∈ [0, π)), equisolid
# (equal-area, the whole sphere), orthographic (sine-law, θ ∈ [0, π/2]
# ONLY — the frame folds at the equator).
public StereographicFrame, EquisolidFrame, OrthographicFrame
# FrameDependence (0.14): what pins a frame tag down — the retina's
# geometry alone (every RetinalFrame kind, ImageFrame, NormalizedImageFrame,
# PixelFrame, TargetPlaneFrame) or a model's continuous parameters (DistortedFrame). The
# trait carries the commensurability rule; frame_transition and the sphere
# sections dispatch on it.
public FrameDependence, GeometryDefined, ModelDefined, frame_dependence
export unlabel, relabel
export BoundedMap                # Map decorator pairing inner Map with output Rect
# Image-warp adapter — wraps a math-space Transformation into a fused
# BoundedMap callable from Julia (row, col) indices.  VGC math is COLMAP
# throughout; image_warp bakes the fixed COLMAP↔Julia bridges via
# composition and `fuse` flattens matrix-representable runs.
export image_warp, fuse
public is_projective
export FramedImage        # canvas + its pixel-frame window (warp-layer return currency)
# The KernelAbstractions resampler: one bilinear kernel, every backend
# (CPU and CUDA share the implementation; backend from the output array).
export image_warp!, image_warp_batch_to!
export canonical_map, logpolar_map, logpolar_to_cartesian
export dilate, intersects
# Pointwise maps over arrays: one broadcast driver for every backend, with the
# host batch's threading as an execution policy.
export pointwise!
public ExecutionPolicy, Auto, Serial, Threaded, THREADED_MIN_LENGTH
# Iterative Cox–de Boor B-spline kernels (0.14; hoisted from
# RayTracedCalibration): broadcast-fused basis/surface/4D evaluation with
# derivative surfaces; the rrules live in the ChainRulesCore extension.
public bspline_basis_iterative, evaluate_bspline_surface, bspline_surface_partial
public evaluate_bspline_4d, bspline_4d_partial, contract_basis_4d_weighted
# Thin-plate spline (hoisted from RadiometricCamera, arithmetic unchanged): the
# host-side solver object and the isbits, allocation-free evaluator and gradient.
public ThinPlateSplineSystem, tps_radial_basis, grid_nodes, tps_coefficients
public tps_evaluate, tps_gradient, nodal_weights, bending_energy, bending_factor

# Coordinate conversions
export to_homogeneous, to_affine, to_euclidean
public saturation_bounds
export FitResult, BundleMetadata
export estimate, retcode, converged, iterations, rms, metadata, weights, dof
public nested_metadata   # contract generic: a metadata type lists its nested metadata fields
public homogeneous_w, maps_within


# -----------------------------------------------------------------------------
# 3. CAMERAS - Camera models, sensors, and projection
# -----------------------------------------------------------------------------
export AbstractCamera, PosedCamera, BoundedCamera, CameraModel, Sensor, sensor, CMOS_SENSORS, Board
public is_posed   # whether a camera or target carries a pose (the rig members' one placement trait)
export extent
public BoundsTrait, HasBounds, NoBounds, bounds_trait
export LogicalIntrinsics, AbstractIntrinsics
export Pinhole, Division, KannalaBrandt, Scaramuzza, AbstractLensModel, Lens, LiftLens
export AbstractProjection, Perspective, Equidistant, Stereographic, Equisolid, Orthographic
export DoubleSphere, Unified, ExtendedUnified
export AbstractDistortionTerm, Radial, Tangential, ThinPrism, Tilt, FieldOfView, Additive, Chain
public Even, Full, isdelta, projection_frame
export radial, parameters, lens_record, lens_from_record
export division_bounded, undistorted_rect, valid_rect, horizon_radius, fov_radius
public DEFAULT_MAX_FOV
export zhang_intrinsics, RadialCamera5pt, RadialCamera  # calibration.jl
public radial_camera_5pt   # the positional minimal kernel under RadialCamera5pt (public, not exported)
public radial_rows   # RadialCamera's rows accessor
public WhiteningPolicy, NoWhitening, SampsonWhitening, whitened_rows   # residual-currency strategy (base/propagate.jl)
# The double-sphere model (0.14): central scalar lens, closed-form inverse,
# served in the framed chain by the fused model-carrying projection (its
# ξ-parameterized defining frame family is ModelDefined — no frame split).
public FramedLensProjection
export imaging_transform, pinhole, undistort_normalized, InverseLens, transform_deriv, transform_deriv_params
public project_with_jacobians
public reprojection_error   # project(camera, pose(X)) − u (cameras/camera_models.jl)
export has_distortion
public monotone_radius
export distort_normalized
export view_angle, projected_radius
public radial_ratio
# Framed imaging chain (0.10): the typed composite {ViewFrame{id} →
# ImageFrame{id}} with every junction checked (imaging_chain), its stage
# family, and the defining-frame trait. imaging_transform is the frame-free
# AffineMap(K) ∘ lens.
# Public, not exported (2026-09-15 audit): the framed imaging chain and sphere
# maps have no consumer outside this package's tests.
public FramedStage, FramedProjection, FramedPerspective, FramedEquidistantProjection
public FramedDistort, FramedUndistort, FramedAffine, FramedChain
public imaging_chain, distort_stage, undistort_stage, calibration_stage
public similarity_scale   # a framed similarity's length ratio: pixel thresholds through the NormalizedImageFrame map
export distortion_frame
# The remaining retinal-frame projections + frame transitions (0.13).
public FramedStereographicProjection, FramedEquisolidProjection
public FramedOrthographicProjection
public FrameTransition, frame_transition
# Sphere self-map (0.11): ray_map(lens) is the lens's ray re-orientation
# D: S² → S² in no 2D frame — framed {ViewFrame{id} → ViewFrame{id}}, with
# an honest inv (the undistort conjugation).
public RayMap, InverseRayMap, ray_map
# A lens over a batch of points: pointwise! of the per-point maps.
export project!, backproject!, distort_normalized!, undistort_normalized!
export undistort, undistort!, distortion_map
export CameraCalibrationMat
export project, backproject, unproject, transfer
export focal_length, calibration_matrix  # Unit accessors; `focal_length(sensor, θ)` the focal a field of view implies
export fov   # the angle of view a bounded camera's extent subtends (its inverse)
export intrinsics, lens, model  # Camera component accessors
export aspect_ratio
# `depth_transfer` (0.8): the renamed unit-depth transfer closure formerly
# exported as `epipolar_map`; that name now refers ONLY to the EpipolarForm
# action (section 8 below). `view_frames`/`image_frames` expose the rig's
# minted per-view ViewFrame/ImageFrame tags.
export lookat, lookat_pose, pose, extrinsics, is_in_front, StereoRig, depth_transfer
export Rig, PosedTarget, source, destination, target_frames  # one Rig for sensors and targets: (model, pose) members; the frame is the gauge
export sensor_rig, target_rig             # the two halves of a rig carrying both
export AbstractTarget, PointTarget, PlanarTarget, resolve, is_resolved   # the target as a model (targets.jl)
export TargetCalibration, Rigid, Affine, ThinPlateSpline, HeightField, CalibrationChain   # a planar target's calibration kinds
export calibration_jacobian, calibration, is_planar
public is_affine   # a calibration's in-plane part is an affine of the design (the surface chart's)
export regularizer   # the calibratable-target protocol
public design, reference, board   # a planar target's design, its reference and its board (not exported: common words)
public PlanarCalibration, gauge_names, gauge_directions   # the shipped planar kinds as one type; a calibration's gauges
public TargetManifold                     # a calibratable target's Manifold (minimal_manifold(target))
export Trajectory, TargetPoses, TargetPoints2D, TargetPoints3D   # per-frame values of a capture (captures.jl)
export TargetPoint2D, TargetPoint3D, UncertainPixel, UncertainPoint3
export convert_frame   # the one way to change the pixel frame of pixel data
export rectify             # stereo rectification (rig types + fused warp)
export view_angle_limit  # the honest θ-domain of a lens
export image_circle      # the lit region of the sensor: from the lens, or a datasheet through the pitch
export imaged_region     # Rect(extent) ∩ image_circle: what a bounded camera images
export free_nodes, node_mask   # which of a model's parameters a camera can see (cameras/support.jl)
public SupportTrait, GlobalSupport, LocalSupport, support_trait, nodes, support, node_position, cubic_bspline_footprint
public radial_part      # a lens's radially symmetric part (shared with VisualGeometryToolkitEstimation's fit_lens)
export parameter_vector, parameter_names, parameter_bounds, with_parameters  # the lens parameter chart
public convert_eltype                     # a camera, lens or intrinsics rebuilt in another float type
public LensManifold                       # the lens family's Manifold (minimal_manifold(lens)); restrict it to a target's free coefficients
export CentralGenericModel, NoncentralGenericModel, Ray3  # generic grid models
export default_generic_grid  # the grid model's window interface (0.22.2)
public spline_support, spline_weights, window_controls, window_indices, direction_tangents
public GenericModelWindow, window, nominal_intrinsics, interpolated_direction, interpolated_point   # a view onto a block of a grid model's controls; its raw splines
public spline_window, spline_value, quasi_interpolant   # the uniform cubic lattice for any control type; the 4th-order quasi-interpolant
export with_model          # same camera/rig, new model (validated rebuild)
# `ids(rig)` (0.9): the rig's view-identity symbols, READ FROM the labeled
# extrinsics' frame parameters (single source of truth — no stored duplicate).
# `essential_form(rig)`/`EpipolarForm(rig)` mint the labeled epipolar forms.
export view_frames, image_frames, ids, essential_form
export validate_gt_inliers, validate_gt_metric
export p3p, sample_p3p, RotationVec
export PerspectiveMap # Perspective division P^n → R^(n-1)
# Re-export CoordinateTransformations callable-transform types (VGC extends
# their constructors with VGC-primitive overloads in transforms/coord_maps.jl
# and primitives/feature_frames.jl).
export AffineMap, Translation

# -----------------------------------------------------------------------------
# 4. COMPOSITION & INTERFACE - Abstract types for feature packages
# -----------------------------------------------------------------------------
# Core composition primitive (defined in base/types.jl)
export Attributed
export AbstractDetection           # = Attributed{<:Union{FeatureFrame, Uncertain{<:FeatureFrame}}}
export feature_frame               # Interface function stub: a detection's feature frame
public BoundedRange                # Validated range type
export Cspond, csponds, ScoredCspond
# Columnar correspondence collection + access protocol + lazy lifts (#21).
# NOTE: `Csponds` (collection of pair-correspondences) is distinct from
# VGF's `CspondSet` (ONE n-tuple of mutually corresponding elements).
export Csponds, sources, targets, hasattr, withattrs
public block_order, split_blocks   # grouped data: the sorted-key block order (key sets checked); a block-ordered vector back per group
export points
public planar_lift, MappedColumn
public lift   # not exported: Makie exports its own `lift`

# -----------------------------------------------------------------------------
# 5. UNITS AND UTILITIES
# -----------------------------------------------------------------------------
# Unit constants; px, pd, dpi and pt are registered into FlexUnits' registry at
# `__init__` (base/units.jl).
export px, dpi, pt, mm, cm, μm, m, inch, ft
public BaseQuantity   # FlexUnits' concrete stored form of a base quantity (base/units.jl)
export ustrip, uconvert, dimension, DimensionError, ConversionError
export @ud_str, uparse, qparse
export parse_unit, unit_string, strip_or_bare
public has_units, with_unit, strip_metres   # plain or unitful; with_unit is ustrip's inverse on the stack's carriers, strip_metres the boundary's strip
export Size2
export Gray, N0f8, N0f16  # Image types
public filter_kwargs, validate_dir
# FileCache moved to VisualGeometryToolkitFeatures

# Serialization traits (DimTrait, canon_value, etc.) moved to VisualGeometryToolkitFeatures

# -----------------------------------------------------------------------------
# 6. ADVANCED/INTERNAL - Matrix types, traits
# -----------------------------------------------------------------------------
# Matrix types (for low-level operations)
# AbstractHomConicMat, HomEllipseMat, HomCircleMat are in base/types.jl
export AbstractHomMat  # abstract parent types
public AbstractHomConicMat
export HomRotationMat, HomTranslationMat, HomUniformScalingMat, HomScalingMat
export HomEuclideanMat, HomSimilarityMat, HomAffineMat, HomographyMat
export HomEllipseMat, HomCircleMat, Conic
export AbstractFundMat, FundMat, EssentialMat, FixedModels

# Primitive traits (behavioral dispatch)
public HasBoundaryTrait, HasBoundary, NoBoundary, has_boundary
public OverlapMethod, AnalyticOverlap, BoundarySampled

# Conic traits
export ConicTrait,
    CircleTrait, EllipseTrait, ParabolaTrait, HyperbolaTrait, DegenerateTrait
public FrameScaleTrait, IsotropicScale, ShapeMatrixScale, frame_scale

# Solver backend trait (select implementation: NativeSolver, PoseLibSolver)
export AbstractSolverBackend, NativeSolver, PoseLibSolver
export classify_conic
export is_ellipse, gradient
export canonical_basis

# -----------------------------------------------------------------------------
# 7. ROBUST ESTIMATION - Loss functions and scale estimators
# -----------------------------------------------------------------------------
# Abstract types and interfaces
# Robust-estimation interfaces (robust/estimation.jl)
public scale_s2   # the squared scale ŝ² of a ConsensusSet, DeployedLoss or FixedS2
export AbstractScoring, AbstractLocalOptimization, RefitFamily, FixedScale,
    AbstractWeightPolicy, update_weights!, update_nuisance!
# The interface machinery behind them: public, reached qualified or imported by
# name. ConsensusSet is read through its accessors (inlier_count, scale_s2, consensus_score,
# is_fresh/check_fresh), never by field name.
public ConsensusSet, consensus_score, is_fresh, check_fresh, NativeRefit, ExternalRefit, optimization_path,
    path_suffix, path_marker, ScaleInterface, ConsensusAnchored, FollowsScorer, SelfScaling, NoScale,
    UnspecifiedScale, scale_interface, lo_label,
    lo_marker, loss_family, NormalizedWeightPolicy, inlier_weight, robust_weight, is_inlier,
    inlier_mask
# -----------------------------------------------------------------------------
# K+ CONTRACTS (robust/problem.jl, scale.jl, mixture.jl, posterior.jl,
# normaliser.jl, scoring_contract.jl, baseline_scorers.jl, lo_contract.jl):
# public, reached qualified or imported by name.
# -----------------------------------------------------------------------------
# The constraint-problem protocol.
public AbstractConstraintProblem, data_size, codimension, minimal_sample_size, residuals!,
    residual_context, whitened_residual, solver_covariance, provides_solver_covariance,
    refine!, refine_workspace
# The robust-statistics toolkit.
public IGPrior, JeffreysPrior, seed_posterior!, reset_posterior!, fit_scale, partition_student_t, partition_rss, weighted_rss,
    weighted_sq, square_residuals!, StudentTMixture, inlier_posterior, posterior_weight,
    posterior_weight_derivative, mixture_nll, next_nu, soft_posterior_weights!,
    PosteriorWeightState, sq_residuals, s_hat_sq, student_t_dof, inlier_prior,
    Estimated, FixedS2, FixedPrior, FixedNu, ConsensusDof, update_scale!, update_nu!,
    update_inlier_prior!, mixture_ecm!, median_seeded_posterior, posterior_inlier_probs!,
    posterior_weights!, GauInlier, decision_threshold, truncated_log_normaliser
# The scoring contract and the baseline scorers.
public scoring_workspace, score!, inlier_count, consensus_order, inlier_threshold,
    has_inlier_threshold, is_predictive, scored_model, scale_prior, outlier_halfwidth,
    score_bound, check_workspace, consensus_s2, consensus_set, inlier_probabilities!,
    score_model!, consensus_mask!, ScoreBuffers, residuals, scores, consensus_mask,
    with_width, RansacScoring, MsacScoring, ThresholdScoringWorkspace, residual_quality
# The local-optimization contract.
public lo_workspace, locally_optimize!, lo_estimate, LoInfo, lo_width, loss_at,
    deployed_scale_multiple, width_scale_interface, width_s2, check_composition, width_value,
    refine_external!, external_workspace, prepare_external!
# Selection over a benchmark pool record (robust/pool_selection.jl).
public best_pool_model, selection_staging, stage_selection, select_on_grid!
# The contract conformance helpers (ext/VisualGeometryToolkitCoreTestExt with Test).
public test_scoring_contract, test_local_optimization_contract, test_constraint_problem, test_manifold_contract
# The MAP inlier rule on a posterior membership weight (posterior > ½), the one
# definition every posterior decision in the stack calls.
export is_map_inlier

# Unified robust-loss family (robust/losses.jl, see LOSSES.md): shape tags,
# traits, and the verbs that read a loss's own width. `decision_width` is the
# one "where the decision flips" conversion (methods per owner).
export AbstractLossShape, GaussianShape, WelschShape, GemanMcclureShape,
    CauchyShape, SmoothTLSShape, TLSShape, HuberShape, liftable, bounded,
    lift_wmax, lift_rho, lift_conjugate, lift_weight, lift_zblock, lift_cost,
    decision_width
# A shape's loss family and display short-name (robust/losses.jl).
public loss_family, loss_label
# The width quantities as dispatch tags, their persisted names, and the scale
# multiple → native width map (robust/width_axes.jl); RANSACScoringDoneRight
# adds the OutlierHalfwidth method of `to_native`. Public, not exported.
public WidthAxis, ScaleMultiple, LossWidth, OutlierHalfwidth, width_axis, width_axis_name, to_native,
    native_map, outlier_halfwidth_to_loss_width, loss_width_to_outlier_halfwidth
# The chart-rows contract (robust/rows.jl): what a lifted local optimizer reads
# of a chart problem — its provider's whitened rows, Jacobian and per-point
# norms — and the one task-local reuse policy every per-call scratch workspace
# draws from. The lifts themselves are LocalOptimizationDoneRight's.
public RowsWorkspace, rows_workspace, task_buffers, point_norms!,
    rows_provider, lift_init, lift_refresh!, lift_jacobian!, lift_retract, lift_model,
    lift_valid, lift_rows, lift_residuals, lift_size
# M-estimator width interface: subtype AbstractMEstimator, define
# `efficiency_constant(::Type{MyShape})`, inherit the rest. `loss_width` and
# `sq_loss_width` stay UNEXPORTED (public, robust/losses.jl) — they are also the
# natural keyword and variable names at every call site; call them qualified.
export AbstractMEstimator, efficiency_constant
public codimension_factor
# A DEPLOYED loss carries the width it runs at as `scale_multiple` (the multiple
# of the noise scale) and `s2` (noise VARIANCE, always supplied by the caller);
# `AnyLoss` is either a deployed loss or an annealed one, and is what the
# width-reading verbs above dispatch on.
export deploy, DeployedLoss, loss_shape, rescale, retune
public AnyLoss, target_loss
# GNC graduation: the loss IS the homotopy's target, and a point along the
# path is `anneal(loss, μ)` — the target wearing its position, as an
# `Attributed`, because μ is not one quantity across the families.
export gnc_surrogate
public anneal, AnnealedLoss, GncSurrogate, MuScaled, WidthAnnealed, ConvexityAnnealed,
    BandAnnealed, NoSurrogate
# GNC surrogate facet, dispatched on `gnc_surrogate`: the Black–Rangarajan band
# weight (the one surrogate with no counterpart in the shape math — GM/Welsch
# are `lift_weight` at the annealed scale μ·c̄², and every target loss is the
# shape's own ρ) and the engine vocabulary (the family's μ-schedule is
# LocalOptimizationDoneRight's, with the GNC lift).
export gnc_tls_weight
export gnc_weight, gnc_target, gnc_mu0
public gnc_entry_mu0, gnc_mu_cap
# Barron α-family loss (facet B; the smooth-and-lift α-dial)
export BarronShape
# The model manifolds (the contract in base/manifold.jl, the instances in
# geometry/manifolds/): local parameterizations + the first-order covariance
# change between charts (`transform_covariance`; the ambient space is the
# chart `EuclideanManifold(ch)`). The `model` accessor stays unexported
# (qualified access) — the name is too generic for the export surface. The three
# covariance verbs: `propagate` (through a map), `transform_covariance` (between
# charts), `implicit_covariance` (of an estimate).
export Manifold, EssentialManifold5, FundMatManifold7, HomographyManifold8, RigidManifold6,
    transform_covariance
public tangent_size, EssentialManifold6, SubsetManifold, restrict, retract, recenter,
    recenter_pose, retract_jacobian, action_jacobian, pose_action_jacobian, retract_directions, s2_tangent_basis,
    SL3_BASIS
public anchor_svd   # FundMatManifold7's anchor (U, σ, V)
public tangent_part   # a manifold's coordinates in a stacked tangent vector (product parts; NLLSsolver variables, robust/lm_core.jl)
public LeftRigidManifold6   # the left-perturbation pose chart constraint_with_jacobians(pose, …) differentiates
public RigidExpManifold6    # the decoupled pose chart: exp on R from the left, t free (hand-eye)
public RigidChart, RotationParam, MRPParam, ExpParam, TranslationCoupling, FreeT, CarriedT   # the rigid chart family the three pose charts alias
public EssentialManifold, EssentialTranslation, S2TangentT, RenormalizedT   # the essential chart family EssentialManifold5/6 alias
public zero_tangent   # a chart's anchor coordinates, v = 0, in its float type
public ambient_coordinates   # a model's ambient coordinates, which a manifold's default retract_jacobian differentiates
public SphereManifold       # S^{N−1}: a model up to scale and sign (minimal_manifold(::RadialCamera))
public EuclideanManifold, ProductManifold   # a plain vector; a model of independent parts (minimal_manifold(::RadialHomography))
public Line2DManifold       # a 2D line: normal angle ⊕ offset (minimal_manifold(::Line2D))
# Implicit-function solver covariance in the manifold coordinates
# (geometry/manifolds/covariance.jl): ONE function, `implicit_covariance` — the
# square solve Σ̃_v = J·Σ̃_x·Jᵀ, J = −(∂_v g)⁻¹·∂_x g, and the whitened-constraint
# sandwich of an overdetermined one; from a chart or from blocks — plus the
# inverse retraction used by the MC calibration harness.
export implicit_covariance, minimal_manifold
public local_coordinates, constraint_dim
public manifold_constraints, constraint_jacobians   # the stacked constraints of a sample and their Jacobians, at any chart coordinates
# Image-domain confidence regions (geometry/confidence_regions.jl):
# confidence ellipses and Hartley-Zisserman epipolar-line bands (a transferred
# point's model covariance: `propagate(transform_deriv_params(H, u), Σθ)`).
export confidence_ellipse, LineConfidenceBand,
    epipolar_band, band_boundary, in_band
public band_conic
# Generic block-coordinate-descent loop (robust/block_coordinate_descent.jl,
# hoisted from VisualGeometryToolkitEstimation): the model/nuisance/weights loop + its
# extension verbs.
export AbstractBlockCoordinateMethod, block_coordinate_descent!
public init_state!, update_model!, update_residuals!, post_iteration!,
    stopping_criterion, block_coordinate_result
# The loop in Manopt.jl's names (public, never exported: a future `using Manopt`
# must not clash): the solver steps, the stopping criteria and the recorder.
public initialize_solver!, step_solver!, stop_solver!,
    StoppingCriterion, stop_reason, StopAfterIteration, StopWhenSolveFailed, StopNever, StopWhenAny,
    Recorded, current_cost, current_s2
# Best-tracking of an instance that claims descent (robust/block_coordinate_descent.jl).
public Incumbent, keep_best
# The chart LM (robust/lm_core.jl): NLLSsolver on the row protocol below;
# `lmsolve_chart!` stays internal (import-by-name), the workspace type is public.
public ManifoldLmWorkspace
# The chart-LM protocol a caller's linearization state implements (robust/lm_core.jl).
public residual_rows!, residual_jacobian_rows!, lm_retcode, is_stationary, nlls_retcode, loss_robustifier
# The one map of the stack's LM knobs onto NLLSsolver's options, and which of the
# fork's additions the loaded NLLSsolver has (robust/lm_core.jl).
public nlls_options, nlls_features
# The stack's naming checker (ext/VisualGeometryToolkitCoreTestExt; loads with Test): every stack package's
# quality test runs it on itself; reached qualified.
public check_naming, NamingViolation, naming_scope
# The stack's shared test helpers (ext/VisualGeometryToolkitCoreTestExt with Test;
# test_quality_gates in ext/VisualGeometryToolkitCoreQualityExt with Test, Aqua,
# ExplicitImports and JET): every stack package's tests call them qualified.
public rows_cost, check_rows_jacobian, test_naming, test_quality_gates
public sqrt_weights   # its √weight buffer

# -----------------------------------------------------------------------------
# 8. PROJECTIVE SOLVERS - Homography and fundamental matrix
# -----------------------------------------------------------------------------
# Solver layer (#21): the normalizations (strategies and CspondNormalization), the
# minimal-solver protocol, registry.
# `solve(solver, cs, idx)` (FixedModels) is THE minimal protocol; the one-step
# solvers are called by their algorithm names (`homography_dlt(cs)`).
public CspondNormalization, source_map, destination_map   # a correspondence set's normalization as a value (models move by ∘ with its sides)
export HartleyNormalization, DiagonalNormalization   # normalization strategies, applied as AffineMap(strategy, points)
export MinimalSolver, model_type
public sample_size
public max_solutions
export solve
export Homography4pt, Homography5ptRadial, Fundamental7pt, Essential5pt, P3P
# Public, not exported (2026-09-15 audit): the solver registry is consumed only by tests.
public minimal_solvers, minimal_solver, induced_models

# Homography solvers
export homography_dlt, homography_dlt!
public homography_4pt   # positional minimal kernel under Homography4pt (public, not exported)
export RadialHomography, AbstractRadialModel
public homography_5pt_radial   # positional minimal kernel under Homography5ptRadial
public homography, division_coeff   # RadialHomography accessors (plane part H, lens coefficient k)
# `center` and `direction` are the radial family's accessors (radial_models.jl,
# cameras/calibration.jl). PUBLIC, not exported: both names are far too generic
# for a foundational geometry package to inject into every `using` — the same
# collision class that made a downstream alias named `Detection` unloadable
# against VisualGeometryToolkitFeatures' blob `Detection`.
public center, direction
export radial_transfer
public radial_outline, radial_in_front, OUT_OF_FIELD_RESIDUAL   # the division model's image outline; its front-hemisphere extent test; the one out-of-field residual
# `project`/`backproject` gain RadialHomography methods (the division model
# as an uncalibrated projection/back-projection model) — exported with the
# camera verbs above.
public sign_normalize, unit_normalize
export transfer_ellipse
export constraint_with_jacobians
public test_model, test_cheirality, test_conditioning
public cheiral_pose_fast   # the 4-way cheirality vote over motion(E), on bearings
public dlt_conditioning, test_dlt_conditioning   # the DLT DESIGN matrix: its null-space gap, and the gate over it
public test_sample, test_consensus

# Fundamental matrix solvers
export fundamental_dlt, fundamental_dlt!
public fundamental_7pt   # positional minimal kernel under Fundamental7pt
export tls_line                   # the total-least-squares line with its (φ, d) covariance
export direct_ellipse             # the ellipse-specific algebraic fit (Halir–Flusser), one closed-form step
public ellipse_log_shape, ellipse_from_log_shape   # the ellipse log-shape chart (cx, cy, S₁₁, S₁₂, S₂₂)
public tls_line_statistics        # the same step's line, weighted residual sum and (φ, d) information
export sampson_distance, signed_sampson_distance
export triangulate   # ray-domain triangulation: the point AND both ray depths (cheirality is λ > 0, not z > 0)
public epipolar_rounding   # the relative degeneracy scales of the one Sampson rule
public constraint_rounding_scale   # a model's rounding scales at a correspondence: what `whiten` needs to apply that rule
public signed_sampson_jacobian, signed_sampson_row, sampson_rows!, sampson_jacobian_rows!   # LM rows: scalar + one KA kernel each
public signed_tangent_sampson_row   # the bearing per-row form (E + M₁/M₂); the rows! methods dispatch to it
public TransferForm, transfer_whitening!, transfer_rows!, transfer_jacobian_rows!   # the whitened homography transfer rows
public sampson_rss   # Σ sampson_distance² of a correspondence set (σ̂² = RSS/dof)
public sampson_residuals!   # the n × M Float32 matrix of sampson_distance, models by points (a pool's residuals)
public radial_transfer_rows!, radial_transfer_jacobian_rows!   # the division-model transfer LM rows
# Slot-labeled epipolar forms (0.8, geometry/forms.jl): the forms layer.
# `EpipolarForm` labels the two argument SLOTS of the bilinear form
# x₂ᵀMx₁ = 0 (FundMat/EssentialMat stay frame-free algebra); `epipolar_map` is
# the honest point → line correlation (NO inv — rank 2), whose frame change is
# `∘` with collineations (M ↦ H₂⁻ᵀMH₁⁻¹); `epipoles` exposes the kernels.
export EpipolarForm, EpipolarMap, epipolar_map, epipoles
public compose_jacobian   # ∂vec(outer ∘ m ∘ inner)/∂vec(m): pulls a composed constraint's G_θ back to the model
# `motion(::EssentialMat)` extracts (R, t) cheirality candidates via native
# SVD (in this package).  `motion(::HomographyMat)` extracts (R, t, n)
# Malis–Vargas candidates and is provided by the PoseLib.jl extension.
# Type alone disambiguates which algorithm runs.
export motion
# `essential_from_fundamental` is removed — use the typed
# `EssentialMat(F::FundMat, K1, K2)` constructor instead (defined in
# geometry/solvers/fundmat.jl).  The reverse direction is `FundMat(E, K1, K2)`.
export translation_angle, skew
export exp_so3, log_so3, left_jacobian, exp_se3, log_se3
public SO3        # group tag: the type-argument slot of left_jacobian(SO3, ω)
export kabsch, chordal_mean, mirror   # SO(3) projection is Rotations.nearest_rotation

# PoseLib-only solvers (no native VGC implementation)
export essential_dlt
public essential_5pt   # positional minimal kernel under Essential5pt

# NOTE: fit_line, LineFitResult moved to VisualGeometryToolkitEstimation

# SVD workspace for zero-alloc null-space computation (used by solvers)
public SvdWorkspace, svd_nullvec!

# Covariance propagation: propagate(J, Σ) = J * Σ * J'
export propagate, whiten
public whitening_factor   # L⁻¹ of the propagated constraint covariance, frozen once
# The scalar kernel gᵀΣg behind `whiten`: public, not exported (an underscore
# name has no place in `using`); RANSACScoringDoneRight imports it by name.
public _propagate_scalar

# -----------------------------------------------------------------------------
# 9. STACK QUALITY CONTRACTS - tag registries, package boundaries
# -----------------------------------------------------------------------------
# The helpers the stack's estimator and scorer packages build on (the chart LM's
# workspace and solves, the product chart's parts, the rotation algorithms'
# stationarity conditions, the radial-coefficient law, the PnP pixel whitening,
# the covariance block rows, the radial-H centred ray, the correspondence
# attribute columns, the weights checks and the whitened constraint residual):
# public, unexported — reached qualified or by explicit import.
public lm_workspace_type, lmsolve_chart!, fill_sqrt_weights!, manifold_parts,
    kabsch_stationarity, chordal_mean_stationarity, rescale_radial_coeff, pnp_pixel_whitening,
    block_rows, radial_ray, attrs_of, attribute_column, cd_pair, sigma_max, check_weights,
    weights_column, sq_norm, whitened_constraint_residual, whitened_distance, zero_sample!
# The one TOML tag → type map and its spelling rule (quality/tag_registry.jl),
# the dependency allow-list, the private-name scan and the extension points
# (quality/stack_contracts.jl). Reached qualified; the quality gates
# (ext/VisualGeometryToolkitCoreQualityExt) run them over every stack package.
public snake_case, TagRegistry, tag_of, expected_tag, tag_rule_violations, SCALE_INTERFACE_TAGS, TARGET_CALIBRATION_TAGS, calibration_tag, length_mask,
    tag_family, tag_paths, retired_tags, tag_registries,
    tag_table_violations, config_tag_violations
public stack_dependencies, stack_dependency_violations, private_name_violations,
    CONTRACT_GENERICS, contract_generic_violations
# What the agent instruction files may not carry (quality/agent_docs.jl).
public AGENT_DOC_FORBIDDEN, agent_doc_violations
# The one copy of the stack conventions (quality/conventions.jl).
public conventions_dir, convention_files

# =============================================================================
# INCLUDE ORDER - Dependency-aware loading
# =============================================================================
#
# The include order is carefully managed due to circular dependencies between
# transforms, primitives, and detection types. This documentation explains
# WHY the order matters.
#
# DEPENDENCY GRAPH (arrows show "depends on"):
#
#   base/units.jl          (no dependencies)
#        ↓
#   base/types.jl          (loads interface.jl, defines conic matrices)
#        ↓                  Provides: AbstractDetection, HomEllipseMat, HomCircleMat
#   base/utils.jl          (utilities, uses types)
#        ↓
#        ↓                  Pure math, no VGC deps
#   transforms/matrices.jl  (transform matrix hierarchy: composition promotion)
#        ↓
#   transforms/euclidean_map.jl  (EuclideanMap for camera poses)
#        ↓
#   primitives/primitives.jl  (submodule: Ellipse, Circle, Disc, etc.)
#        ↓                     Uses: HomEllipseMat, HomCircleMat from types.jl
#   transforms/conversions.jl  (coordinate conversions)
#   transforms/coord_maps.jl   (coordinate mappings)
#   transforms/region_maps.jl  (canonical and log-polar maps)
#   transforms/homography_map.jl   (projective transforms)
#        ↓                     All use: Ellipse, Circle from Primitives
#   geometry/solvers/solvers.jl        (P3P and geometric solvers)
#   geometry/solvers/homography_solvers.jl  (homography solvers, public API)
#   geometry/solvers/fundmat.jl             (F-matrix solvers, public API)
#        ↓
#   cameras/cameras.jl         (submodule: camera models, projection)
#        ↓                     Uses: Primitives

#                              Uses: Cameras, Primitives
#
# CRITICAL DEPENDENCIES:
# 1. types.jl MUST define HomEllipseMat/HomCircleMat BEFORE Primitives
#    (Primitives extends these with Ellipse constructors)
# 2. Primitives MUST be loaded BEFORE coord_maps.jl, region_maps.jl
#    (those files use Ellipse type from Primitives)
#
# =============================================================================

# Base types and utilities

# Shared machinery for the package family (unexported — import by name):
public present_kwargs, check_keys, toml_skip_fields, toml_key, toml_value,
    toml_field, toml_table, allowed_keys, read_table, convert_field, toml_default, write_toml,
    toml_field_name, parse_defaults
public lgamma_lanczos, digamma_stirling, trigamma_stirling, lu_solve!, tree_sum!,
    bilinear_sample_clamped
public draw_ellipse!, draw_projected_ellipse!, draw_polyline!
# Reached by qualified access or explicit import from the family's packages
# (VisualGeometryToolkitEstimation, VisualGeometryToolkitEval, RANSACScoringDoneRight, BlobBoards,
# VisualGeometryToolkitIO, PoseLib, VisualGeometryToolkitFeatures):
public Cameras, Primitives, Synthetic, Pools, SpatialQueries, reference_frame, is_valid_solution,
    poly!, thin_ratio, is_orientation_preserving,
    rect_corners, is_convex_quad   # the extent checks' corner order and winding test

include("base/units.jl")    # unit constants px, mm, dpi, …; the stack units registered at __init__ - no dependencies
include("base/size2.jl")    # Size2
include("base/feature_coords.jl")   # _strip_coords: the feature-coordinate compute boundary
include("base/frames.jl")   # Frame{N} tag hierarchy — BEFORE types.jl (CovMat/PrecMat frame params)
include("base/image_frames.jl")   # 2-D image frames (retinal/pixel/target-plane/distorted) — AFTER frames, BEFORE types.jl
include("base/types.jl")    # Loads interface.jl, defines AbstractDetection, conic matrices
include("base/cspond_set.jl") # Csponds columnar container + sources/targets protocol + lift
include("base/coordinates.jl")    # to_affine / to_euclidean — used by Primitives + transforms
include("base/fit_result.jl")     # FitResult{P,M} — the one iterative-fit return type
include("base/utils.jl")    # Geometry utilities - uses types
include("base/config_toml.jl")  # TOML config skeleton shared by every package with a configuration of record
include("quality/tag_registry.jl")  # TagRegistry: the one tag → type map (before the kinds that register tags: the target calibrations)
include("base/device.jl")   # default_device — the family's one KA backend resolver (CUDA ext registers)
include("base/svd_workspace.jl")  # SvdWorkspace + svd_nullvec! (used by solvers + RVG)
include("base/propagate.jl")     # propagate(J, Σ) = J * Σ * J' with specializations
include("base/measurement_noise.jl")  # exact, pair_covariance, noise_columns: the measurement-noise rule
include("base/manifold.jl")      # the Manifold CONTRACT (retraction + covariance transport); instances in geometry/manifolds/
# FileCache moved to VisualGeometryToolkitFeatures

# Robust estimation core: losses, the shared RANSAC/LO interfaces, the generic
# block-coordinate-descent loop, the chart-rows contract and the chart LM (NLLSsolver).
# NOTE: No dependencies on other VisualGeometryToolkitCore modules.
# The RANSAC engine and the model problems live in VisualGeometryToolkitEstimation.jl.
include("robust/losses.jl")       # unified robust-loss family + GR-lift facet (LOSSES.md)
include("robust/width_axes.jl")   # WidthAxis tags + to_native(::LossWidth) (uses TLSShape, decision_width)
include("robust/estimation.jl")   # shared RANSAC/LO interface hierarchy (types + contracts only)
include("robust/block_coordinate_descent.jl") # generic loop (block_coordinate_descent! + extension verbs; hoisted from RVG)
include("robust/rows.jl")         # the chart-rows contract + RowsWorkspace + task_buffers
include("robust/lm_core.jl")      # the chart LM: NLLSsolver on the row protocol (ManifoldLmWorkspace)
# The contracts the RANSAC framework composes through (K+): the constraint-problem
# protocol, the robust-statistics toolkit (scale, the Student-t + uniform mixture,
# the posterior-weight state), the scoring contract with the baseline scorers,
# and the local-optimization contract.
include("robust/problem.jl")          # AbstractConstraintProblem + data_size/codimension/residuals!/refine!
include("robust/scale.jl")            # IGPrior, fit_scale, partition_student_t, partition_rss, weighted_rss
include("robust/mixture.jl")          # StudentTMixture, the E-step, next_nu, GauInlier/decision_threshold
include("robust/posterior.jl")        # PosteriorWeightState, the nuisance policies, mixture_ecm!
include("robust/normaliser.jl")       # truncated_log_normaliser
include("robust/scoring_contract.jl") # score!/inlier_count/consensus_order, ScoreBuffers, score_model!
include("robust/baseline_scorers.jl") # RansacScoring, MsacScoring + their contract glue
include("robust/lo_contract.jl")      # lo_workspace/locally_optimize!/lo_estimate, loss_at

# Geometry Transforms (partial - matrices and rigid transforms only)
# NOTE: coord_maps, region_maps, homography_map come AFTER Primitives
include("geometry/transforms/matrices.jl")      # Transform matrix types + Holy Traits
include("geometry/transforms/euclidean_map.jl") # EuclideanMap (rigid transforms for camera poses)
include("geometry/transforms/lie.jl")           # SO(3)/SE(3) exp/log + left Jacobian (AD-safe exp side)
include("geometry/transforms/rotation_fits.jl") # Kabsch/Wahba, chordal mean, SO(3) projection

# Primitives submodule - geometric primitives (Ellipse, Circle, Disc, etc.)
# NOTE: Uses HomEllipseMat/HomCircleMat from base/types.jl
include("geometry/primitives/primitives.jl")
using .Primitives  # Import all exported symbols into parent namespace
using .Primitives: rect_corners, _rect_edge_samples, _snap_outward   # the one corner order and per-edge rect walk (primitives/rectangles.jl)

# Geometry Transforms (remaining - these use Ellipse from Primitives)
include("geometry/transforms/conversions.jl")      # Coordinate system conversions
include("geometry/transforms/ct_contravariant.jl") # CT types callable on Circle, Ellipse, Line2D
include("geometry/transforms/coord_maps.jl")       # Coordinate mappings (Circle→Ellipse)
include("geometry/transforms/region_maps.jl")      # Canonical and log-polar maps
include("geometry/transforms/homography_map.jl")  # HomographyMap (point/line/conic transforms)
include("geometry/transforms/propagate_framed.jl") # checked frame-labeled propagate over framed maps (0.9)
include("geometry/transforms/bounded_map.jl")     # BoundedMap (output extent decorator)

# PerspectiveMap (from CoordinateTransformations) is NOT invertible - dimension reduction

include("geometry/solvers/solvers.jl")                     # Forward declarations for PoseLib-only solvers
include("geometry/solvers/p3p_native.jl")                  # Native P3P (PoseLib p3p.cc port, NativeSolver backend)
include("geometry/solvers/essential_5pt_native.jl")        # Native 5-pt essential (PoseLib relpose_5pt.cc port)
include("geometry/solvers/homography_solvers.jl")         # Homography solvers (public API)
include("geometry/solvers/fundmat.jl")                     # F-matrix solvers (public API)
include("geometry/solvers/tls_line.jl")                    # tls_line: the total-least-squares line (Line2D, Uncertain)
include("geometry/solvers/direct_ellipse.jl")              # direct_ellipse: the ellipse-specific algebraic fit; the log-shape chart
include("geometry/radial_models.jl")                # AbstractRadialModel — the plane→pencil family (see the header for the layering)
include("geometry/solvers/homography_5pt_radial.jl")       # 5-pt one-sided radial homography (HomLib Wadenbäck 3DV'26 port)
include("geometry/forms.jl")                       # slot-labeled epipolar forms (EpipolarForm + actions)
# The concrete model manifolds, one family per file (the contract is base/manifold.jl).
include("geometry/manifolds/subset.jl")      # SubsetManifold — restrict(): free/fixed as a manifold operation (contract-generic; BEFORE the instances, which extend it)
include("geometry/manifolds/essential.jl")   # EssentialManifold5/6 — the production lift + eval charts
include("geometry/manifolds/fundmat.jl")     # FundMatManifold7 — Bartoli–Sturm orthonormal representation
include("geometry/manifolds/homography.jl")  # HomographyManifold8 + the sl(3) basis
include("geometry/manifolds/rigid.jl")       # RigidManifold6 — the rigid-pose retraction (needs EuclideanMap; extends SubsetManifold)
include("geometry/manifolds/sphere.jl")      # SphereManifold — S^{N−1}, the chart of a model known up to scale (RadialCamera's rows)
include("geometry/manifolds/product.jl")     # EuclideanManifold, ProductManifold — a model of independent parts (sl(3) ⊕ k)
include("geometry/manifolds/line.jl")        # Line2DManifold — a 2D line's normal angle and offset
include("geometry/manifolds/covariance.jl")  # implicit-function solver covariance in the manifold coordinates
include("geometry/confidence_regions.jl")          # image-domain confidence regions (confidence ellipses, epipolar bands)
include("geometry/model_validity.jl")              # test_sample/test_model/test_consensus per model class
include("geometry/solvers/solver_layer.jl")                # CspondNormalization + the MinimalSolver protocol + registry (#21)

# Core warping API - builders returning BoundedMap with image-side conventions baked in
include("geometry/transforms/warp.jl")
# Pointwise maps over arrays (one broadcast driver; the host threading policy).
include("geometry/transforms/pointwise.jl")
# Iterative Cox–de Boor B-spline kernels (0.14; hoisted from
# RayTracedCalibration). Uses the KernelAbstractions imports of warp.jl;
# rrules attach in the ChainRulesCore extension.
include("numerics/bspline.jl")
include("numerics/special.jl")    # device-safe lgamma/digamma/trigamma (adaptation kernels)
include("numerics/small_lu.jl")   # in-register partial-pivot LU for fixed-size systems
include("numerics/ka_reduce.jl")  # KernelAbstractions workgroup tree reduction
include("numerics/thin_plate_spline.jl")   # thin-plate spline: host-side solver, isbits allocation-free evaluator
include("geometry/primitives/surfaces.jl")   # Quad, HeightSurface: the surface a ray meets (needs the thin-plate evaluator)
include("base/raster.jl")         # array plotting without Makie (masks, debug frames)
include("geometry/primitives/bbox.jl")   # shape → SpatialIndexing.Rect (needs frames + dilate)
# Allocation-free R-tree queries (intersection, nearest neighbour) and the
# in-place bulk load, on SpatialIndexing's trees. Not `using`'d: reached as `VisualGeometryToolkitCore.SpatialQueries`.
include("geometry/spatial_queries.jl")

# Cameras submodule - camera models, sensors, projection
include("geometry/cameras/cameras.jl")
using .Cameras  # Import all exported symbols into parent namespace
# Internal camera helper (hoisted from VisualGeometryToolkitEstimation p3p.jl): the PnP
# lift's frozen pixel whitening. RVG imports it by name and adds the
# problem-level method.
import .Cameras: pnp_pixel_whitening

# Synthetic data generation — depends on fundmat.jl, homography_solvers.jl
include("synthetic/synthetic.jl")

# The benchmark pool record (PoolPair) and the raw-pair interface (StereoRawPair).
# NOTE: AFTER Cameras (FramedAffine, similarity_scale, StereoRig) and the solver
# layer (CspondNormalization, DiagonalNormalization). Not `using`'d here: its
# accessor names (num_points, gt_pose, …) stay out of every `using VisualGeometryToolkitCore`.
include("pools/pools.jl")
include("robust/pool_selection.jl")   # best_pool_model, select_on_grid! over a pool record (needs Pools)

# Visualization stubs for GLMakie extension (LSP indexing support)
include("visualization_stubs.jl")

# The stack's naming checker: its result type and entry points (the linter is
# the Test extension, ext/VisualGeometryToolkitCoreTestExt).
include("quality/naming.jl")
include("quality/test_helpers.jl")   # the rows checks, the quality gates (entry points)
include("quality/contract_helpers.jl")   # the contract conformance helpers (entry points)
# The stack's quality contracts: the tag registry's scans (the registry itself
# is included with the base layer, before the kinds that register tags); the
# package boundaries (dependency allow-list, private names, CONTRACT_GENERICS —
# last, as it lists functions from every section above).

"""
    SCALE_INTERFACE_TAGS

The scale-interface labels as TOML tags (a `TagRegistry` of
[`ScaleInterface`](@ref) values' types): `"consensus_anchored"`,
`"fixed_scale"`, `"follows_scorer"`, `"self_scaling"`, `"no_scale"`,
`"unspecified_scale"` — `snake_case` of the type's name. `"fixed"` and
`"partition_anchored"` are retired. A config that names an interface (an
ablation's delivered interface) reads it here.
"""
const SCALE_INTERFACE_TAGS = TagRegistry(:scale_interface, "",
    "consensus_anchored" => ConsensusAnchored, "fixed_scale" => FixedScale, "follows_scorer" => FollowsScorer,
    "self_scaling" => SelfScaling, "no_scale" => NoScale, "unspecified_scale" => UnspecifiedScale;
    retired = ("fixed" => "fixed_scale", "partition_anchored" => "consensus_anchored"),
    paths = [r"^(.*\.)?scale_interface$"])
include("quality/stack_contracts.jl")
include("quality/agent_docs.jl")
include("quality/conventions.jl")

# FlexUnits' registry is filled at load: a registration made while precompiling
# does not persist into the loaded image.
function __init__()
    _register_stack_units()
    return nothing
end

end # module VisualGeometryToolkitCore

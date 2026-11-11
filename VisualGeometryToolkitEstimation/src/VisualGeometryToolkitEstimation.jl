module VisualGeometryToolkitEstimation

import VisualGeometryToolkitCore as VgtkCore

# =============================================================================
# DEPENDENCIES
# =============================================================================

using LinearAlgebra: LinearAlgebra, I, dot, norm, normalize, cross, svd, opnorm,
    Diagonal, Symmetric,
    eigen                                                          # the ellipse refit rows' axes frame
using StaticArrays: StaticArrays, MVector, SA, SMatrix, SVector, StaticVector
using FixedSizeArrays: FixedSizeArray
using Random: Random
using Rotations: Rotations, RotMatrix

# VisualGeometryToolkitCore — explicit named imports only (no bare `using`).
using Statistics: Statistics, cov
using Distributions: Chisq, cdf   # SIMFIT++'s χ² noise model (estimators/ransac/simfitpp.jl)
using VisualGeometryToolkitCore: VisualGeometryToolkitCore,
    AbstractFundMat,
    AbstractSolverBackend,
    CameraModel, EssentialMat, EuclideanMap,
    FixedModels, FundMat, HomographyMap, HomographyMat,
    HyperRectangle, Line2D, NativeSolver, Point2, Point3,
    Extrinsics3, UnlabeledFrame, unlabel, relabel, Vec2, Vec3,
    Uncertain, CovMat,
    # Slot-labeled epipolar forms + per-view frame tags (VGC 0.9): the
    # measurement-frame bookkeeping of the 2D-2D RANSAC problems is
    # declared through these at the construction boundary (cspond.jl
    # `_measurement_map`/`_check_measurement_frames`/
    # `_measurement_epipolar`); the @inbounds Sampson kernels and scoring
    # conjugations keep the raw matrices (bit-pinned inline products).
    EpipolarForm, EpipolarMap, to_homogeneous, GnomonicFrame, ImageFrame,

    backproject, fundamental_dlt!,
    homography_dlt!,
    is_orientation_preserving, rect_corners, is_convex_quad,   # the extent gates (extent_traits.jl, radial_homography.jl)
    # Geometric degeneracy helpers relocated from RVG into VGC's model_validity.jl;
    # imported back for the P3P degeneracy check.
    thin_ratio, _are_points_distinct,
    normal, skew, whitening_factor,   # the frozen whitening of the reweighted fits (radial camera, division lift)
    weighted_whitened_residuals!, weighted_whitened_jacobian_rows!,   # the chart LM rows of every Sampson-constraint problem (one KA kernel each)
    kabsch, chordal_mean, log_so3,   # SO(3) log (hand-eye, robot-world)
    s2_tangent_basis,                        # the tangent chart of the P3P ray-coplanarity gate
    tls_line,                                # fit_line's unweighted seed direction
    tls_line_statistics,                     # the fixed-weight TLS step fit_line / the line LO reweight
    ellipse_log_shape   # the ellipse refit rows' chart (problems/ellipse.jl)

# Shared robust-estimation interfaces (VGC src/robust/estimation.jl): the
# LO-strategy root, the paper's two classification traits (optimization_path,
# scale_interface) with the score→LO width handoff (the width modes), and the
# weight-policy hierarchy/verbs. VGC owns and exports that vocabulary; RVG
# imports what it extends or calls (eval/ extends the traits as
# VisualGeometryToolkitCore.optimization_path etc.).
# THE estimator result type and the accessors RVG reads it through (VGC
# base/fit_result.jl). Every fit here returns a FitResult.
using VisualGeometryToolkitCore: FitResult, estimate, retcode, converged, iterations, weights
# Units at the boundary (src/units.jl; VGC conventions/units.md): a public
# estimator strips a physical input in metres at entry and reattaches metres at
# exit; the inner loop takes plain maps (`PlainEuclideanMap`) and plain points.
using FlexUnits: Quantity
using VisualGeometryToolkitCore: PlainEuclideanMap, UnitfulEuclideanMap, ustrip, rms, metadata
using VisualGeometryToolkitCore: CspondNormalization   # the correspondence normalization, chosen per model (VGC solver layer)
# The benchmark pool record and the raw-pair interface (VGC Pools): RVG writes
# the record (`PoolPair(rng, problem, raw)`, a constructor method on VGC's type
# whose problem argument is RVG's) and builds its pool-frame problems.
import VisualGeometryToolkitCore.Pools: PoolPair
using VisualGeometryToolkitCore.Pools: StereoRawPair, pair_csponds, pair_snn, pair_K1, pair_K2,
    pair_R_gt, pair_t_gt, pair_gt_inliers, pool_normalization, pool_frame_calibrations, pool_pixel_scale,
    pixel_residuals, best_model_index   # SIMFIT++'s pool adapter: a pair's pixel residuals, the one pool argmax
using VisualGeometryToolkitCore: sampson_distances!, sampson_distance, homography_dlt, test_dlt_conditioning
using VisualGeometryToolkitCore.Synthetic: optimistic_pose_error
# VGC state is read through its accessors, never by field name (a by-field read
# breaks at RUNTIME here when VGC renames the field; an accessor breaks at load).
using VisualGeometryToolkitCore: angle_between                 # the angle of two vectors (atan form)
using VisualGeometryToolkitCore: rotation, translation,        # EuclideanMap
    Translation,                                         # a frame's origin shift, composed onto a pose
    homography, division_coeff, center,                  # RadialHomography (center: RadialCamera too)
    radial_rows,                                         # RadialCamera
    scale_s2,                                            # ConsensusSet / DeployedLoss s2
    sqrt_weights,                                        # ManifoldLmWorkspace buffer
    lens, intrinsics,                                    # CameraModel
    geom,                                                # Uncertain's value
    propagate,                                           # J·Σ·Jᵀ: a chart covariance into vec(M) (covariance.jl)
    loss_shape                                           # DeployedLoss's shape
# The division-camera step, the radial-camera fit (moved from
# VisualGeometryToolkitCore, estimator taxonomy v2) and fit_camera reach VGC through these
# public names only: lens families and their manifold, the intrinsics, the
# frame accessors and the lens-profile queries.
using VisualGeometryToolkitCore: AbstractLensModel, AbstractRadialModel,
    Lens, LiftLens, DoubleSphere, Equidistant, Division, Perspective,
    Stereographic, Equisolid, Orthographic,
    LensManifold, ParameterManifold, Manifold, restrict, tangent_size, parameter_names,
    CameraCalibrationMat, LogicalIntrinsics, calibration_matrix, parameter_vector,
    RadialCamera, source_map, destination_map,
    projected_radius, view_angle, view_angle_limit, radial_part,
    similarity_scale   # a normalization's length ratio: pixel tolerances inside (radial homography)
import VisualGeometryToolkitCore: AbstractOptimizationStrategy,
    AbstractWeightPolicy, NormalizedWeightPolicy,
    update_weights!, update_nuisance!,
    inlier_weight, robust_weight, is_inlier,
    ConsensusSet, check_fresh
# The contracts the RANSAC loop and its local optimizers compose through
# (VisualGeometryToolkitCore src/robust/): the constraint-problem protocol (RVG's
# problems implement it — `import`), the scoring contract (the loop CALLS it —
# `using`; RVG adds no scorer), the local-optimization contract (RVG's hosted
# strategies implement `locally_optimize!`/`lo_workspace`/`lo_width`/
# `with_width`) and the robust-statistics toolkit.
import VisualGeometryToolkitCore: data_size, codimension, minimal_sample_size, mahalanobis_distances!,
    solver_covariance, provides_solver_covariance, refine!, refine_workspace,
    residual_context, mahalanobis_distance, whitened_residual,
    locally_optimize!, lo_workspace, lo_width
# The core optimization strategies and the strategy registry machinery are
# VisualGeometryToolkitCore's (robust/strategies.jl, A2); this package registers its
# own (TerminalLO) with the same macro and re-publishes the machinery for its
# downstream users.
using VisualGeometryToolkitCore: Irls, Triggs, LOSS_TAGS, STRATEGY_TAGS, lo_loss, fixed_width,
    lo_width_value, lo_admits_shape, lo_construct, lo_has_default, lo_takes_width, @lo_registry
# the registry hooks `@lo_registry` and this package's strategies add methods to: imported
import VisualGeometryToolkitCore: strategy_loss, lo_tag, lo_defaults, lo_parametric, lo_admits, lo_key, lo_field,
    lo_field_from_toml, lo_absent, lo_validate, lo_efficiency_constant_shape
using VisualGeometryToolkitCore: AbstractConstraintProblem, AbstractScoring, ScoreBuffers, score_model!,
    consensus_mask!, consensus_set, consensus_order, consensus_s2, inlier_count, inlier_probabilities!,
    scored_model, is_predictive, check_workspace, scoring_workspace, score!,
    scale_prior, outlier_halfwidth, lo_estimate, check_composition,
    EfficiencyConstant, LossWidth, TagRegistry, MsacScoring, RansacScoring, decision_boundary,
    StudentTUniformScoring, FixedHalfwidth, IGPrior, JeffreysPrior, BetaLabelPrior,
    LABEL_PRIOR_TAGS,   # the Student-t scorer's table (config_toml.jl scoring_from_toml)
    PosteriorWeightState, JeffreysPrior, seed_posterior!, partition_rss, weighted_rss,
    median_seeded_posterior, Estimated, FixedS2, FixedPrior, FixedNu, ConsensusDof, inlier_prior,
    square_distances!, sq_distances, s_hat_sq, student_t_dof,
    update_scale!, update_nu!, update_inlier_prior!,
    mixture_nll
# Generic block-coordinate-descent loop (hoisted to VGC
# robust/block_coordinate_descent.jl): the loop + its extension verbs are VGC generics now — `import`
# so the RVG problem files' method definitions extend the VGC functions.
import VisualGeometryToolkitCore: AbstractBlockCoordinateMethod, block_coordinate_descent!, update_model!,
    update_residuals!, post_iteration!, stopping_criterion, stop_reason,
    block_coordinate_result, current_cost, current_s2
using VisualGeometryToolkitCore: StoppingCriterion
# The chart-rows contract (VGC robust/rows.jl: the rows a lifted local
# optimizer reads, and the one task-local reuse policy) — RVG's chart problems
# provide it (`import`) — the chart-LM core and protocol (VGC
# robust/lm_core.jl) and the incumbent of a descent run, `Attributed(θ, (; cost))`
# (VGC robust/block_coordinate_descent.jl). Internal helpers imported by name; the
# chart-LM protocol is `import`ed so every problem's linearization state adds
# its methods to VGC's functions.
using VisualGeometryToolkitCore: RowsWorkspace, rows_workspace, task_buffers, point_norms!,
    ManifoldLmWorkspace, lm_workspace_type, lmsolve_chart!, fill_sqrt_weights!, keep_best, Attributed,
    lm_retcode, loss_robustifier
import VisualGeometryToolkitCore: residual_rows!, residual_jacobian_rows!, residual_cost!, residual_normal_equations!
import VisualGeometryToolkitCore: division_coeffs   # RadialHomography's coefficients; DivisionCameraState's method is ours
import VisualGeometryToolkitCore: rows_provider, lift_init, lift_refresh!, lift_jacobian!, lift_retract,
    lift_model, lift_valid, lift_rows, lift_residuals, lift_size
# The robot-world, division-camera and lens-regression fits (fitting/): the
# product chart and its parts, the rotation algorithms' stationarity
# conditions, grouped data's block order, the radial-coefficient law and the
# pixel intrinsics — each VGC's one spelling.
using VisualGeometryToolkitCore: ProductManifold, EuclideanManifold, model,
    kabsch_stationarity, chordal_mean_stationarity, block_order, split_blocks,
    rescale_radial_coeff
using VisualGeometryToolkitCore.Cameras: pixel_intrinsics
# Small hoisted helpers: PnP pixel whitening (problem-level method added in
# problems/p3p.jl — hence `import`), 3-D triangle thinness (model_validity.jl).
import VisualGeometryToolkitCore: pnp_pixel_whitening
# Ray-domain triangulation and cheirality (VGC B5): `triangulate` is exported
# there; the bundle triangulates in the ray domain (the pixel-domain DLT
# survives only as an independent oracle in test/relpose_bundle_tests.jl). The cheirality is the
# 4-way vote over motion(E) (`cheiral_pose_fast`, VGC geometry/solvers/fundmat.jl,
# which applies `test_cheirality` itself); RVG calls it and does not re-export it.
using VisualGeometryToolkitCore: triangulate, cheiral_pose_fast
# Unified robust-loss family (VGC's loss shapes) — the GR-lift facet
# verbs moved there verbatim. The shapes this package's defaults and
# dispatch name are imported by name (the strategy types and the loss table
# are VGC's, robust/strategies.jl).
using VisualGeometryToolkitCore: GaussianShape, TLSShape
# A width is COMPOSED onto a shape (VGC 0.29) rather than carried by it:
# `deploy(shape, s2; efficiency_constant)` pairs the tag with the squared scale it
# runs at, and the loss verbs then take no width argument. `bounded` is the redescending
# axis — whether a shape rejects at all — which decides whether a frozen width
# is a consensus threshold or just a scale (scoring_glue.jl `is_inlier`).
using VisualGeometryToolkitCore: deploy, DeployedLoss, sq_loss_width, bounded
using VisualGeometryToolkitCore: EssentialManifold5,
    HomographyManifold8, pose_action_jacobian,   # the relpose bundle's pose chart
    LeftRigidManifold6, recenter, retract, retract_jacobian, action_jacobian,
    minimal_manifold,   # the circle and radial-camera charts
    RigidExpManifold6, local_coordinates,   # the hand-eye and robot-world chart (the reported covariances'; the hand-eye constraint)
    Line2DManifold,   # the line's Hesse chart (LineFittingProblem)
    implicit_covariance, block_rows,   # the one model covariance and its block rows (estimators/ransac/covariance.jl)
    radial_transfer_rows!, radial_transfer_jacobian_rows!,          # the radial-H LM rows (likewise)
    radial_ray                                                     # the radial-H centred ray T₋c·H·s̃ (k seed, extent gate)
# RVG defines no methods on any of these — it only CALLS them — so `using`, not
# `import`. (`import` advertises an extension point that does not exist.)
using VisualGeometryToolkitCore: weight
import VisualGeometryToolkitCore:
    SvdWorkspace, # used by RANSAC + solvers
    test_model, # RANSAC problems extend this
    test_sample, # RANSAC problems extend; delegate to VGC model-class methods
    test_consensus, # RANSAC problems extend; delegate to VGC model-class methods
    constraint_with_jacobians, # RANSAC problems add (problem, model, i) methods
    constraint_rounding_scale, # ... and their rounding scales (problem, model, i)
    measurement_covariance,    # ... and their points' noise (problem, i)
    solve, model_type        # #21 solver-layer verbs — problems add `solve`/`model_type` methods
# #21 solver/correspondence layer: the consolidated MinimalSolveProblem is
# built on VGC's correspondence protocol (solver instances carry the
# NativeSolver/PoseLibSolver backend). The `CspondNormalization`/`MinimalSolver`
# machinery and the solver tags (Homography4pt/Fundamental7pt/Essential5pt/
# P3P) are reached through the solver objects themselves, not by name here.
using VisualGeometryToolkitCore: sources, targets, unit_normalize, Cspond, withattrs
# The perspective division's Jacobian — VGC's single definition, used for the
# homography dehomogenization (`_dpi`) and the relpose bundle's `_Dproj`.
using VisualGeometryToolkitCore: PerspectiveMap, transform_deriv

# PoseLib (optional, NOT a hard dep since 0.8.3): provides the PoseLibSolver
# methods for p3p / essential_5pt / essential_dlt / motion(::HomographyMat) /
# homography_4pt / fundamental_7pt on VGC's solver stubs. The default P3P is
# VGC's native port; `using PoseLib` in the consumer enables the C++ backends.
# Symbols are already exported via VGC. RVG does NOT load PoseLib itself — the
# consumer's own `using PoseLib` is what populates the backend method tables.
# RVG's suite does not load it either (it sat in the test target unloaded until
# 2026-09-21, where its VGC compat could make the test env unresolvable), so the
# PoseLibSolver paths are exercised by PoseLib's and VGC's suites, not this one.
using ForwardDiff: ForwardDiff
import NLLSsolver   # the block-structured LMs (the relpose bundle, refine_robot_world_hand_eye): qualified NLLSProblem/optimize! and its residual protocol
using VisualGeometryToolkitCore: nlls_retcode, nlls_rows_and_jacobian, nlls_stop_reasons   # NLLSsolver's termination bits + the stall postcondition, as a retcode
using VisualGeometryToolkitCore: nlls_options, nlls_features, finite_cost, Eliminable   # the stack's one LM knob map; the Schur system (upstream #60), detected
using SparseArrays: sparse   # the relpose bundle's stacked Jacobian (its stall postcondition); the rotation averaging's update
# fit_camera's spline families: the penalized normal equations (a sparse
# Cholesky), the direction chart, Core's lattice and its batched maps.
using SparseArrays: SparseMatrixCSC, nzrange, rowvals, nonzeros   # (gcv.jl: the Hutchinson path's pattern)
import SparseArrays
import Graphs   # the view and observation graphs' algorithms (components, articulation, bridges, spanning trees)
using LinearAlgebra: cholesky, cholesky!, lu, issuccess, diag
using VisualGeometryToolkitCore: ChartSpline, SplineLens, NoncentralSplineModel, Rect2, spline_window, pixel_centres,
    pointwise!, BoundedCamera, extent, spline_lattice, central_model, difference_rows, default_device,
    mean_positive_eigenvalue, penalty_scale, row_stencils   # the penalty weight's normalization and an operator's rows as blocks (Core)
using VisualGeometryToolkitCore: nearest_point, backproject_with_jacobian


# =============================================================================
# EXPORTS — the user API; the machinery is `public`
# =============================================================================
#
# Export what a user of the package calls: the pipelines, the RANSAC entry
# points, the problems a user builds, and the local-optimization strategies a
# user picks. The machinery — workspaces, samplers, tag tables, plumbing types,
# extension points, research arms — is `public`: named, documented and
# importable (`using VisualGeometryToolkitEstimation: DltWorkspace`), not exported.
#
# Names another package owns are exported ONLY by their owner: RVG re-exports
# nothing. Import VisualGeometryToolkitCore's result accessors, loss shapes, scale
# interfaces, solver verbs, scorings, priors and problem-interface verbs from
# it. (User decision, 2026-09-21: 79
# RVG exports → 29, 88 re-exports dropped.)

# -----------------------------------------------------------------------------
# 1. PIPELINES — robust estimation end to end
# -----------------------------------------------------------------------------
export find_homography, find_fundmat, find_essential, find_absolute_pose,
    find_hand_eye, find_radial_homography, find_radial_camera, find_rotation, find_circle, find_ellipse
# Each find_*'s local-optimization step alone, from a given model (no sampling).
export refine_homography, refine_fundmat, refine_essential, refine_absolute_pose,
    refine_hand_eye, refine_radial_homography, refine_radial_camera, refine_rotation, refine_circle, refine_ellipse
export fit_robot_world_hand_eye, refine_robot_world_hand_eye, fit_division_camera
export fit_residual_mixture   # the inlier/outlier classification of a held model's residuals
export refine_camera_and_poses   # a camera (intrinsics + lens) and one pose per view from 3D → 2D correspondences
export fit_rotation           # the robust rotation between two sets of directions
export optical_axis           # the radially symmetric camera closest to a camera: its axis and principal point (fit_camera)
export fit_linear             # the robust coefficients of a linear model (reweighted least squares)
export fit_rotation_averaging # robust rotation averaging on a view graph (spectral or spanning-tree seed, Lie-algebra IRLS)
export filtered_rotation_averaging   # the same behind the cycle and orientation filters (Theia's global pipeline)
export ViewGraph, ObservationGraph   # the frames and their verified pairs; the frames and what they observe (graphs/)
public views, edgelist, nedges, adjacent_views, components, label_components, require_connected, induced, keep_largest_component,
    articulation, bridges, spanning_tree, chain, triangles, filter_cycles, rotation_loop_angle, filter_by_orientation,
    observers, observed, observed_in, observers_of, covisibility   # their queries and algorithms, public, not exported
export fit_radial_camera, DivisionCameraError   # moved from VisualGeometryToolkitCore (estimator taxonomy v2)
export fit_camera             # one model-to-model regression: a camera model of type T onto a source over a region
export pspline_grid           # the spline targets' one P-spline solve on a given lattice (fit_camera's inner step)

# -----------------------------------------------------------------------------
# 2. RANSAC — the loop, its configuration and result, and seeded refinement
# -----------------------------------------------------------------------------
export ransac, locally_optimize, RansacConfig, RansacEstimate

# -----------------------------------------------------------------------------
# 3. PROBLEMS a user builds
# -----------------------------------------------------------------------------
export HomographyProblem, FundMatProblem, EssentialProblem, AbsolutePoseProblem, BearingPoseProblem, RotationProblem, CircleProblem, EllipseProblem,
    HandEyeProblem, LineFittingProblem, InhomLineFittingProblem,
    RadialHomographyProblem, RadialCameraProblem, RelposeBundleProblem

# -----------------------------------------------------------------------------
# 4. LOCAL-OPTIMIZATION strategies a user picks
# -----------------------------------------------------------------------------
export Irls, Triggs
public synthetic_scenes   # the shared synthetic test scenes (the Test extension): every stack package's tests draw from them
public CountedLO   # instrumentation: counts a strategy's locally_optimize! calls and iterations
# The terminal placement (one LO of a raw race's incumbent) and the placement trait.
public TerminalLO, lo_placement, InLoopPlacement, TerminalPlacement
# The zero-allocation RANSAC entry and its result buffers.
export ransac!
# SIMFIT++ (Edstedt, CVPRW 2025): the loss width from held-out residuals, and its pool adapter
# (estimators/ransac/simfitpp.jl; moved from RANSACScoringDoneRight)
export SimFitPP, simfitpp_loss_width, simfitpp_select
public unbiased_median_variance, with_residual_unit

# -----------------------------------------------------------------------------
# PUBLIC, not exported — the machinery
# -----------------------------------------------------------------------------
public gcv_select, gcv_curve, gcv_score, check_penalty_weight   # the one GCV of a penalty weight (RadiometricCamera's solves call it; its normalization is Core's penalty_scale, mean_positive_eigenvalue)
# Problem plumbing and RVG's own RANSAC-interface verbs. `Rigid3` is the
# unlabeled EuclideanMap alias find_hand_eye returns (generic enough to collide:
# PoseLib has its own), `refine!` the LO refit an extender implements.
public AbstractRansacProblem, AbstractCspondProblem, MinimalSolveProblem, PoseContext, BearingPoseContext,
    BoundedProblem, ExtentTrait, NoExtent, WithExtent, extent_trait, Rigid3,
    refine!, fit_hypothesis, draw_sample!, residual_dof, oracle_residual_dof
# Benchmark pools: the hypothesis generator, and how a pool record's σ̂² is
# measured (`PoolPair(rng, problem, raw; …)` is a method of VGC's record type).
public draw_hypotheses, Hypotheses, PoolVariance, InlierVariance, TrackVariance
# Samplers, and the protocol a new sampler implements (draw!, reset!).
public AbstractSampler, UniformSampler, ProsacSampler, GaussianSampler, ExhaustiveSampler, sampler,
    draw!, reset!, trial_budget, adapts
# Workspaces.
public DltWorkspace, RansacWorkspace, RansacResultBuffers, LoBuffers
# The result's metadata, the lean LO result, and posterior helpers. The last two
# have a documented downstream consumer (LocalOptimizationDoneRight).
public RansacMetadata, inlier_ratio, PosteriorLikelihood, LossPlainWeights,
    consensus_s2!, fixed_width
# The posterior-weight iteration a caller can drive outside RANSAC: its method,
# state and model updates, and the EM spec of `fit_residual_mixture` (its TOML
# loader and tag tables), with that fit's metadata.
public PosteriorIrlsMethod, PosteriorIrlsState, WeightedRefine, KeepModel,
    posterior_irls!, consensus_posterior, required_outlier_halfwidth, locally_optimize_rescored!,
    StopWhenNoImprovement, iterations_without_improvement,
    MixtureEmConfig, mixture_em_from_toml, MODEL_UPDATE_TAGS, NUISANCE_TAGS, ResidualMixtureMetadata,
    NoImprovementStop, ParameterChangeStop, StopWhenParameterChangeLess, change_parameters, STOP_TAGS
# The local-optimization extension point: a strategy adds a `locally_optimize!`
# method (downstream: LocalOptimizationDoneRight); a loss-parametric strategy
# names its loss (`strategy_loss`) and deploys it through `lo_loss`, its
# `efficiency_constant` the plain multiple whatever the shape; the
# chart problems' rows provider and the problem family a lift runs on.
public lo_loss, strategy_loss, lo_width_value,
    ManifoldLiftProvider, LiftableProblem
# Relative-pose bundle and robot-world calibration.
# The division-camera step, its state, the lift constraint and fit_camera's metadata.
public division_camera_step, DivisionCameraState, DivisionCameraProblem, board_sq_residuals,
    RadialLift, CameraFitMetadata
public bundle_refine_relpose, RobotWorldHandEyeCalibration, RobotWorldHandEyeMetadata
public RotationFitMetadata, LinearFitMetadata, RotationAveragingMetadata, CameraPosesRefineMetadata
# Configuration: configs/default.toml is the config of record; these select a
# strategy by tag and write a config value back as a table.
public default_config_path, optimization_strategy_from_toml, lo_from_toml, scoring_from_toml, config_dict, LO_TAGS, CONTRACT_GENERICS,
    LOSS_TAGS
# The strategy machinery every package registers its strategies with,
# re-published from VisualGeometryToolkitCore (whose CONTRACT_GENERICS hold the
# registry hooks): the registry macro, the keyword constructor, the shared shape
# admission and the hooks; this package's own are the generic writer and the
# TOML reader's hooks (CONTRACT_GENERICS below).
public @lo_registry, lo_construct, lo_config_dict, lo_admits_shape,
    lo_tag, lo_defaults, lo_parametric, lo_admits, lo_key, lo_field, lo_field_from_toml, lo_absent,
    lo_validate, lo_concrete, lo_efficiency_constant_shape, lo_resolve_efficiency_constant


# =============================================================================
# INCLUDE ORDER
# =============================================================================

include("interface.jl")
# configs/default.toml → `_dflt`: MUST precede every config-carrying struct.
include("config_defaults.jl")
include("units.jl")              # the estimators' one strip at entry and one reattachment at exit

# The generic loop (block_coordinate_descent!) is hoisted to VGC
# robust/block_coordinate_descent.jl (imported above). The scale, mixture and
# posterior-weighting toolkit is VisualGeometryToolkitCore's; the PosteriorLikelihood weight-policy adapter
# is in scoring_glue.jl below.

# RANSAC framework
include("estimators/ransac/types.jl")
include("estimators/ransac/samplers.jl")
include("estimators/ransac/interface.jl")
include("estimators/ransac/scoring_glue.jl")   # LO strategy types + width modes, PosteriorLikelihood, the strategy registry
include("estimators/ransac/loop.jl")
include("estimators/ransac/local_optimization.jl")   # the hosted strategies' locally_optimize! (posterior IRLS, BCD)
include("estimators/ransac/hypotheses.jl")   # draw_hypotheses: the RANSAC trial, repeated, no scoring
include("estimators/ransac/extent_traits.jl")

# RANSAC problem implementations
include("estimators/ransac/problems/cspond.jl")
include("estimators/ransac/problems/p3p.jl")
include("estimators/ransac/problems/bearing_pose.jl")   # the absolute pose on caller-supplied bearings (central or generalized): the tangent-plane rows
include("estimators/ransac/problems/homography.jl")
include("estimators/ransac/problems/radial_homography.jl")
include("estimators/ransac/problems/radial_camera.jl")    # the 1-D radial camera of a board: distortion-free verification
include("estimators/ransac/problems/fundmat.jl")
include("estimators/ransac/problems/essential.jl")
include("estimators/ransac/problems/circle.jl")     # the 2D circle through points: Circle3pt + the point-on-circle Sampson rows
include("estimators/ransac/problems/ellipse.jl")    # the 2D ellipse through points: Ellipse5pt + the point-on-ellipse Sampson rows
include("estimators/ransac/problems/rotation.jl")   # the pure rotation on bearings (the infinite homography): Rotation2pt + the bearing transfer rows
include("estimators/ransac/problems/sampson_lm.jl")   # the chart-LM rows of every Sampson-constraint problem (F, E, H, circle, radial camera, rotation)
include("estimators/ransac/problems/relpose_bundle.jl")
include("estimators/ransac/problems/line.jl")
include("estimators/ransac/problems/hand_eye.jl")
include("estimators/ransac/problems/pool.jl")   # the pool-frame problems of a pool record; PoolPair(rng, problem, raw)
include("estimators/ransac/simfitpp.jl")        # SIMFIT++ loss-width estimation (Edstedt 2025) + its pool adapter
include("estimators/ransac/covariance.jl")   # the one model covariance: the constraint sandwich (all providers)
# The chart-rows provider of every chart problem (VGC's rows contract; needs
# the per-problem chart/residual helpers above).
include("estimators/ransac/problems/manifold_providers.jl")

# Fitting pipelines — RANSAC-based
include("fitting/_pipeline.jl")
# Provides `fit_line` and the covariance-weighted line step `_weighted_line`,
# which problems/line.jl's LO-RANSAC `refine!` calls.
include("fitting/line.jl")
include("fitting/ellipse.jl")      # find_ellipse and refine_ellipse
include("fitting/homography.jl")
include("fitting/radial_homography.jl")
include("fitting/radial_camera.jl")
include("fitting/fit_radial_camera.jl")        # one board's radial camera, an iterative whitened fit (moved from VGC)
include("fitting/division_camera_step.jl")     # one iteration of the stacked linear focal solve + its state and failure (moved from VGC)
include("fitting/gcv.jl")                       # the stack's one GCV of a penalty weight (gcv_curve, gcv_select; the weight's normalization is Core's)
include("fitting/fit_camera.jl")                # one model-to-model regression: a lens family, or a SplineLens camera or non-central spline model, onto a source over a region
include("fitting/essential.jl")
include("fitting/rotation.jl")
include("fitting/circle.jl")
include("fitting/pose.jl")
include("fitting/fundmat.jl")
include("fitting/hand_eye.jl")
include("fitting/robot_world_hand_eye.jl")
include("fitting/refine_robot_world_hand_eye.jl")   # reprojection refinement of a RobotWorldHandEyeCalibration (static-N chart LM; uses robot_world_hand_eye.jl's type)
include("fitting/refine_camera_and_poses.jl")   # a camera and one pose per view from 3D → 2D correspondences (NLLSsolver on VGC's parameter and pose charts)
include("fitting/refine_spline_camera_and_poses.jl")   # its SplineLens route: 2 × 2 patches of sphere-charted controls, implicit-differentiation blocks, the Δ³ penalty, the rotation gauge
include("synthetic_scenes.jl")                    # synthetic_scenes: entry point (the scenes are the Test extension)
include("fitting/irls_weights.jl")           # the IRLS weight step of the three reweighted fits (σ̂ is Core's mad_scale)
include("fitting/fit_rotation.jl")           # the robust rotation between direction pairs (reweighted Kabsch)
include("fitting/fit_linear.jl")             # the robust coefficients of a linear model (reweighted least squares)
include("graphs/view_graph.jl")              # ViewGraph: frames and verified pairs (Graphs.jl underneath)
include("graphs/observation_graph.jl")       # ObservationGraph: frames and what they observe; covisibility → ViewGraph
include("fitting/fit_rotation_averaging.jl") # robust rotation averaging on a view graph (spectral or spanning-tree seed, Lie-algebra IRLS)
include("fitting/residual_mixture.jl")     # fit_residual_mixture: a held model's residuals → mixture EM → MAP labels
include("fitting/fit_division_camera.jl")    # the focal stack on the shared posterior-weight iteration (after residual_mixture.jl: its MixtureEmConfig)
# TOML loaders, the strategy-tag registry and the writer — dispatches on every
# config-carrying type, so it is included last.
include("config_toml.jl")

"""
    CONTRACT_GENERICS

The generic functions of VisualGeometryToolkitEstimation another package may add methods
to across a package boundary (VisualGeometryToolkitCore's quality gate,
`contract_generic_violations`): the TOML reader's hooks (`lo_concrete`,
`lo_resolve_efficiency_constant`), the writer (`config_dict`, for a strategy whose
table carries a key beyond the generic one), a wrapper's placement
(`lo_placement`), the posterior-weight stop rule's
counter (`iterations_without_improvement`), the parameter-change stop's measure of a
problem (`change_parameters`) and the sampler / running-best
protocol (`reset!`, `draw!`).
"""
const CONTRACT_GENERICS = (lo_concrete, lo_resolve_efficiency_constant,
                           config_dict, scoring_from_toml, lo_placement,
                           iterations_without_improvement, change_parameters, reset!, draw!)

end # module VisualGeometryToolkitEstimation

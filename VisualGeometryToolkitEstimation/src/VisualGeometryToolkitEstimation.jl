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
using PrecompileTools: @setup_workload, @compile_workload   # src/precompile.jl
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
    CameraExtrinsics, UnlabeledFrame, unlabel, relabel, Vec2, Vec3,
    Uncertain, CovMat,
    # Slot-labeled epipolar forms + per-view frame tags (VGC 0.9): the
    # observation-frame bookkeeping of the 2D-2D RANSAC problems is
    # declared through these at the construction boundary (correspondence.jl
    # `_observation_map`/`_check_observation_frames`/
    # `_observation_epipolar`); the @inbounds Sampson kernels and scoring
    # conjugations keep the raw matrices (bit-pinned inline products).
    EpipolarForm, EpipolarMap, to_homogeneous, GnomonicFrame, ImageFrame,

    backproject, fundamental_dlt!,
    homography_dlt!,
    is_orientation_preserving, rect_corners, is_convex_quad,   # the extent gates (extent_traits.jl, radial_homography.jl)
    # Geometric degeneracy helpers relocated from RVG into VGC's model_validity.jl;
    # imported back for the P3P degeneracy check.
    thin_ratio, are_points_distinct,
    normal, skew, whitening_factor,   # the frozen whitening of the reweighted fits (radial camera, division lift)
    weighted_whitened_residuals!, weighted_whitened_jacobian_rows!,   # the chart LM rows of every Sampson-constraint problem (one KA kernel each)
    log_so3,   # SO(3) log (hand-eye, robot-world)
    Kabsch, ChordalMean, TotalLeastSquares, NearestPoint,   # Core's closed forms, solved by their algorithms in the inner loops
    EstimationProblem, RefineProblem, MinimalProblem, NullParameters, ClassDefault, SolveStats,
    AbstractEstimationAlgorithm, AbstractEstimationProblem, FitKind, FindKind, RefineKind, ClosedFormKind, LmDamping, MarquardtDamping, LevenbergMarquardt,
    s2_tangent_basis,                        # the tangent chart of the P3P ray-coplanarity gate
    tls_line_statistics,                     # the fixed-weight TLS step of the covariance line fit / the line LO reweight
    ellipse_log_shape   # the ellipse refit rows' chart (problems/ellipse.jl)

# Shared robust-estimation interfaces (VGC src/robust/estimation.jl): the
# LO-strategy root, the paper's two classification traits (optimization_path,
# scale_interface) with the score→LO width handoff (the width modes), and the
# weight-policy hierarchy/verbs. VGC owns and exports that vocabulary; RVG
# imports what it extends or calls (eval/ extends the traits as
# VisualGeometryToolkitCore.optimization_path etc.).
# THE estimator result type and the accessors RVG reads it through (VGC
# base/fit_result.jl). Every fit here returns a FitResult.
using VisualGeometryToolkitCore: FitResult, weights, successful_retcode, noise_scale, scale_source, dof
# Its termination record (VGC base/return_code.jl): every result records its ReturnCode and StopRule
# through FitResult's constructor.
using VisualGeometryToolkitCore: ReturnCode, StopRule, found_model,
    worst_retcode, NoRule, ClosedForm, CostDecrease, IncrementSize, ParameterChange, NoImprovement,
    ToleranceMet, IterationCap, NonFinite, TrialBudget, AdaptiveCount, SeedKept
# The robust contract's scale record and weight kinds (VGC base/scale_source.jl,
# robust/weights.jl): RansacMetadata records the scorer's scale and its kind; the refine_* results
# (fitting/_refine.jl) the scale they were handed.
using VisualGeometryToolkitCore: ScaleSource, ScaleEstimated, ScaleNone, BinaryWeights,
    PosteriorWeights, unit_weights
# Units at the boundary (src/units.jl; VGC conventions/units.md): a public
# estimator strips a physical input in metres at entry and reattaches metres at
# exit; the inner loop takes plain maps (`PlainEuclideanMap`) and plain points.
using FlexUnits: Quantity
using VisualGeometryToolkitCore: PlainEuclideanMap, UnitfulEuclideanMap, ustrip, rms
using VisualGeometryToolkitCore: RetiredKeyword, refuse_retired_keyword   # the one refusal of a retired keyword (Core base/utils.jl)
using VisualGeometryToolkitCore: CorrespondenceNormalization   # the correspondence normalization, chosen per model (VGC solver layer)
using VisualGeometryToolkitCore: HartleyNormalization, AffineMap   # the 3-D strategy of a least squares over lengths (robot_world_hand_eye.jl)
# The benchmark pool record and the raw-pair interface (VGC Pools): RVG writes
# the record (`PoolPair(rng, problem, raw)`, a constructor method on VGC's type
# whose problem argument is RVG's) and builds its pool-frame problems.
import VisualGeometryToolkitCore.Pools: PoolPair
using VisualGeometryToolkitCore.Pools: StereoRawPair, pair_correspondences, pair_snn, pair_K1, pair_K2,
    pair_R_gt, pair_t_gt, pair_gt_inliers, pool_normalization, pool_frame_calibrations, pool_pixel_scale,
    pixel_residuals, best_model_index, check_residual_matrix   # SIMFIT++'s pool adapter: a pair's pixel residuals, the one pool argmax
using VisualGeometryToolkitCore: sampson_distances!, sampson_distance, test_dlt_conditioning
using VisualGeometryToolkitCore.Synthetic: optimistic_pose_error
# VGC state is read through its accessors, never by field name (a by-field read
# breaks at RUNTIME here when VGC renames the field; an accessor breaks at load).
using VisualGeometryToolkitCore: angle_between                 # the angle of two vectors (atan form)
using VisualGeometryToolkitCore: rotation, translation,        # EuclideanMap
    Translation,                                         # a frame's origin shift, composed onto a pose
    homography, division_coeff, center,                  # RadialHomography (center: RadialCamera too)
    radial_rows,                                         # RadialCamera
    scale_s2,                                            # ConsensusSet / DeployedLoss s2
    lens, intrinsics,                                    # CameraModel
    geom,                                                # Uncertain's value
    propagate,                                           # J·Σ·Jᵀ: a chart covariance into vec(M) (covariance.jl)
    loss_shape                                           # DeployedLoss's shape
# The division-camera step, the radial-camera fit (moved from
# VisualGeometryToolkitCore, estimator taxonomy v2) and CameraRegression reach VGC through these
# public names only: lens families and their manifold, the intrinsics, the
# frame accessors and the lens-profile queries.
using VisualGeometryToolkitCore: AbstractLensModel, AbstractRadialModel,
    Lens, LiftLens, DoubleSphere, Equidistant, Division, Perspective,
    Stereographic, Equisolid, Orthographic,
    LensManifold, ParameterManifold, Manifold, restrict, tangent_size, parameter_names, with_parameters,
    Radial, Even, Tangential, ThinPrism,   # CameraRegression's linear forms: the radial factor and the decentering deltas
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
    solver_covariance, provides_solver_covariance, refine!, refit_workspace,
    residual_context, mahalanobis_distance, whitened_residual,
    locally_optimize!, lo_workspace, lo_width
# The core optimization strategies and the strategy registry machinery are
# VisualGeometryToolkitCore's (robust/strategies.jl, A2); this package registers its
# own (TerminalLO) with the same macro and uses the rest internally (it
# re-publishes none of them: callers reach them through VisualGeometryToolkitCore).
using VisualGeometryToolkitCore: Irls, Triggs, LOSS_TAGS, STRATEGY_TAGS, lo_loss,
    lo_construct, lo_has_default, lo_takes_width, @lo_registry
# the registry hooks `@lo_registry` and this package's strategies add methods to: imported
import VisualGeometryToolkitCore: strategy_loss, lo_tag, lo_defaults, lo_parametric,
    lo_field_from_toml, lo_efficiency_constant_shape
using VisualGeometryToolkitCore: AbstractConstraintProblem, AbstractScoring, ScoreBuffers, score_model!,
    consensus_mask!, consensus_mask, consensus_set, consensus_order, consensus_s2, inlier_count, inlier_probabilities!,
    scored_model, is_predictive, check_workspace, scoring_workspace, score!,
    scale_prior, outlier_halfwidth, lo_estimate, check_composition,
    EfficiencyConstant, LossWidth, TagRegistry, MsacScoring, RansacScoring, AbstractThresholdScoring, decision_boundary,
    StudentTUniformScoring, FixedHalfwidth, IGPrior, JeffreysPrior, BetaLabelPrior,
    LABEL_PRIOR_TAGS,   # the Student-t scorer's table (config_toml.jl scoring_from_toml)
    PosteriorWeightState, JeffreysPrior, seed_posterior!, partition_rss, weighted_rss,
    MedianNormScale, Estimated, FixedS2, FixedPrior, FixedNu, ConsensusDof, inlier_prior,
    square_distances!, sq_distances, s_hat_sq, student_t_dof,
    update_scale!, update_nu!, update_inlier_prior!,
    mixture_nll
# Generic block-coordinate-descent loop (hoisted to VGC
# robust/block_coordinate_descent.jl): the loop + its extension verbs are VGC generics now — `import`
# so the RVG problem files' method definitions extend the VGC functions.
import VisualGeometryToolkitCore: AbstractBlockCoordinateMethod, update_model!,
    update_residuals!, post_iteration!, stopping_criterion, stop_reason,
    block_coordinate_result, current_cost, current_s2
using VisualGeometryToolkitCore: StoppingCriterion, CONTINUE, BlockCoordinateDescent, StopAfterIteration   # the loop's front end, its budget
# The EM fits' (DivisionCameraFit, ResidualMixture) scale source and weight kind: imported above.
# The chart-rows contract (VGC robust/rows.jl: the rows a lifted local
# optimizer reads, and the one task-local reuse policy) — RVG's chart problems
# provide it (`import`) — the chart-LM core and protocol (VGC
# robust/lm_core.jl) and the incumbent of a descent run, `Attributed(θ, (; cost))`
# (VGC robust/block_coordinate_descent.jl). Internal helpers imported by name; the
# chart-LM protocol is `import`ed so every problem's linearization state adds
# its methods to VGC's functions.
using VisualGeometryToolkitCore: RowsWorkspace, rows_workspace, task_buffers, point_norms!, keep_best, Attributed
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
# Ray-domain triangulation and cheirality (VGC B5): `Triangulation` is exported
# there; the bundle triangulates in the ray domain (the pixel-domain DLT
# survives only as an independent oracle in test/relpose_bundle_tests.jl). The cheirality is the
# 4-way vote over motion(E) (`cheiral_pose_fast`, VGC geometry/solvers/fundmat.jl,
# which applies `test_cheirality` itself); RVG calls it and does not re-export it.
using VisualGeometryToolkitCore: cheiral_pose_fast, triangulate_bearings
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
    observation_covariance,    # ... and their points' noise (problem, i)
    model_type               # the problems' model types
# The estimation interface (Core's solve_interface.jl): Estimation adds `__solve`/`__init` (and `solve!`/`reinit!` on
# its caches) for its OWN algorithm types, declares their `estimator_kind`/`supports_solve_rng`, and the defaults of
# its own problem types; the chart contract of the LevenbergMarquardt refine is implemented on its family problems.
import VisualGeometryToolkitCore: __solve, __init, estimator_kind, supports_solve_rng,
    model_chart, chart_rows, chart_model, chart_size, guess_model, solution_model, chart_result, default_optimization_strategy,
    block_structure, block_variables, residual_blocks, residual_summary, block_result, with_robustifier, default_stops,
    reinit!
import CommonSolve: solve!
using VisualGeometryToolkitCore: SquaredScale   # ConsensusScale's problem tag (the squared scale)
using VisualGeometryToolkitCore: solve, init, AbstractBlockProblem, AbstractFitMetadata,
    normalized_loss_weight, weighted_support_rms, NoiseScale, RowsWithJacobian
# #21 solver/correspondence layer: the consolidated MinimalSolveProblem is
# built on VGC's correspondence protocol (solver instances carry the
# NativeSolver/PoseLibSolver backend). The `CorrespondenceNormalization`/`MinimalSolver`
# machinery and the solver tags (Homography4pt/Fundamental7pt/Essential5pt/
# P3P) are reached through the solver objects themselves, not by name here.
using VisualGeometryToolkitCore: sources, targets, unit_normalize, Correspondence, withattrs
# The perspective division's Jacobian — VGC's single definition, used for the
# homography dehomogenization (`_dpi`) and the relpose bundle's `_Dproj`.
using VisualGeometryToolkitCore: PerspectiveMap, transform_deriv

# PoseLib (optional, NOT a hard dep since 0.8.3): provides the PoseLibSolver
# methods for p3p / essential_5pt / essential_8pt / motion(::HomographyMat) /
# homography_4pt / fundamental_7pt on VGC's solver stubs. The default P3P is
# VGC's native port; `using PoseLib` in the consumer enables the C++ backends.
# Symbols are already exported via VGC. RVG does NOT load PoseLib itself — the
# consumer's own `using PoseLib` is what populates the backend method tables.
# RVG's suite does not load it either (it sat in the test target unloaded until
# 2026-09-21, where its VGC compat could make the test env unresolvable), so the
# PoseLibSolver paths are exercised by PoseLib's and VGC's suites, not this one.
using ForwardDiff: ForwardDiff
import NLLSsolver   # the block-structured LMs' residual protocol (the relpose bundle, the camera-and-poses bundles)
using VisualGeometryToolkitCore: LmStops   # Core's one LM: its stop settings
using VisualGeometryToolkitCore: finite_cost, Eliminable   # a finite cost; a variable the Schur system eliminates
using SparseArrays: sparse   # the rotation averaging's update
# CameraRegression's spline families: the penalized normal equations (a sparse
# Cholesky), the direction chart, Core's lattice and its batched maps.
using SparseArrays: SparseMatrixCSC, nzrange, rowvals, nonzeros   # (gcv.jl: the Hutchinson path's pattern)
import SparseArrays
import Graphs   # the view and observation graphs' algorithms (components, articulation, bridges, spanning trees)
using LinearAlgebra: cholesky, cholesky!, lu, issuccess, diag
using VisualGeometryToolkitCore: ChartSpline, SplineLens, NoncentralSplineModel, Rect2, spline_window, pixel_centres,
    pointwise!, BoundedCamera, extent, spline_lattice, central_model, difference_rows, default_device,
    mean_positive_eigenvalue, penalty_scale   # the penalty weight's normalization (Core)


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
# 1. FINDS AND REFINES — `solve(EstimationProblem(f, p), Ransac(…); rng)`,
#    `solve(RefineProblem(f, θ₀), LevenbergMarquardt(…); scale | width)`, `f` a family problem below
# -----------------------------------------------------------------------------
export Ransac, LocalOptimization, RansacConfig, RansacEstimate

# -----------------------------------------------------------------------------
# 2. FITS — one algorithm type each, solved on an `EstimationProblem` (or a `RefineProblem` for the bundles' `f`s)
# -----------------------------------------------------------------------------
export ViewGraph, ObservationGraph   # the frames and their verified pairs; the frames and what they observe (graphs/)
public views, edgelist, nedges, adjacent_views, components, label_components, require_connected, induced, keep_largest_component,
    articulation, bridges, spanning_tree, chain, triangles, filter_cycles, rotation_loop_angle, filter_by_orientation,
    observers, observed, observed_in, observers_of, covisibility   # their queries and algorithms, public, not exported
export DivisionCameraError
public ViewAngleLimitError    # the camera regressions' refusal of a region past the target family's view-angle limit (its fields: the measured angle, the limit)
export PooledScale            # one noise scale for a set of results that share a noise model (EstimationProblem(NoiseScale, results))
# (the fits' algorithm types and their problem tags: added below by the fits' migration)
export ReweightedLeastSquares, LinearModel   # the fixed-scale reweighted fits (a linear model, a rotation, a rigid alignment, a robot-world calibration)
export RotationAveraging, FilteredRotationAveraging   # robust rotation averaging on a view graph, and behind the view-graph filters
export ResidualMixture        # the Student-t + uniform mixture of a held model's residuals (classification)
export RobotWorldHandEyeProblem, RelativePoseProblem   # the reprojection refine of a robot-world calibration; the two-view bundle of a relative pose (block problems)
export Hyperparameter, GeneralizedCrossValidation   # a hyperparameter's selection (its problem tag), and the penalty weight by generalized cross-validation
export ConsensusScale   # the consensus scale ŝ² of a model under a scorer (EstimationProblem(SquaredScale, problem, (; model)))
export LikelihoodRatioSelection   # model selection between refines: the most likely of several seeded local optimizations
export ScoreSelection   # the best-scoring stored model of a hypothesis pool (EstimationProblem(PoolSelection, pair => R))
export PoolSelection   # SimFitPP's tag of a pool's selected model (its loss-width problem's is Core's `LossWidth`)
export ImageMeasure, SolidAngleMeasure, UniformMeasure   # a camera regression's sample density (CameraRegressionProblem(…; weight))
public AbstractImageMeasure
export CameraRegressionProblem, CameraRegression, PSpline, ForwardSpline, OpticalAxis   # a camera model regressed onto a source over a region (its refine: LevenbergMarquardt on the problem)
export CameraPosesProblem     # a camera and one pose per view from 3D → 2D correspondences (refined by LevenbergMarquardt)
export RadialCameraFit, DivisionCameraFit   # one board's radial camera; the division camera from many boards' radial cameras

# -----------------------------------------------------------------------------
# 3. PROBLEMS a user builds — the `f` of every find and refine
# -----------------------------------------------------------------------------
export HomographyProblem, FundMatProblem, EssentialProblem, AbsolutePoseProblem, BearingPoseProblem, RotationProblem, CircleProblem, EllipseProblem,
    HandEyeProblem, LineFittingProblem, InhomLineFittingProblem,
    RadialHomographyProblem, RadialCameraProblem, RelposeBundleProblem

# -----------------------------------------------------------------------------
# 4. LOCAL-OPTIMIZATION: the core strategies Irls and Triggs are VisualGeometryToolkitCore's
# (exported there); this package publishes its own wrappers and instrumentation
# -----------------------------------------------------------------------------
public synthetic_scenes   # the shared synthetic test scenes (the Test extension): every stack package's tests draw from them
public CountedLO   # instrumentation: counts a strategy's locally_optimize! calls and iterations, and records each call's termination
# The terminal placement (one LO of a raw race's incumbent) and the placement trait.
public TerminalLO, lo_placement, InLoopPlacement, TerminalPlacement
# The RANSAC cache (`init(problem, Ransac(…); rng)` → `solve!`/`reinit!`) and its result buffers.
public RansacCache
# SIMFIT++ (Edstedt, CVPRW 2025): the loss width from held-out residuals (estimators/ransac/simfitpp.jl)
export SimFitPP
public with_residual_unit

# -----------------------------------------------------------------------------
# PUBLIC, not exported — the machinery
# -----------------------------------------------------------------------------
public gcv_curve, gcv_score, check_penalty_weight   # the GCV curve of a penalized least squares, its score, and a penalty weight's check (RadiometricCamera's solves call them; the weight's normalization is Core's penalty_scale, mean_positive_eigenvalue)
# Problem plumbing and RVG's own RANSAC-interface verbs. The LO refit an extender implements, `refine!`, is
# VisualGeometryToolkitCore's.
public AbstractRansacProblem, AbstractCorrespondenceProblem, MinimalSolveProblem, PoseContext, BearingPoseContext,
    BoundedProblem, with_extent, ExtentTrait, NoExtent, WithExtent, extent_trait,
    sample_models, sample_hypothesis, draw_sample!, residual_dof, oracle_residual_dof
# Benchmark pools: how a pool record's σ̂² is measured (`PoolPair(rng, problem, raw; …)`, a method of VGC's
# record type, draws its hypotheses with the RANSAC trial; the generator is internal).
public PoolVariance, InlierVariance, TrackVariance
# Samplers, and the protocol a new sampler implements (draw!, reset!).
public AbstractSampler, UniformSampler, ProsacSampler, GaussianSampler, ExhaustiveSampler, sampler,
    draw!, reset!, trial_budget, adapts
# Workspaces.
public DltWorkspace, RansacWorkspace, RansacResultBuffers, LoBuffers
# The result's metadata, the lean LO result, and posterior helpers. The last two
# have a documented downstream consumer (LocalOptimizationDoneRight).
public RansacMetadata, inlier_ratio, PooledScaleMetadata, PosteriorLikelihood, LossPlainWeights
# The posterior-weight local optimization, for strategy authors: its method, state and model updates and its LO
# contract step `posterior_irls!` (called only inside a `locally_optimize!` method: LocalOptimizationDoneRight's
# strategies); and the EM spec of `ResidualMixture` (its TOML loader and tag tables), with that fit's metadata.
# Outside RANSAC the posterior weighting is `solve(EstimationProblem(PosteriorWeightState, d, (; model_dof)),
# ResidualMixture())`, or `solve(RefineProblem(f, θ), LocalOptimization(strategy; scoring))`.
public PosteriorIrlsMethod, PosteriorIrlsState, WeightedRefine, KeepModel,
    posterior_irls!, consensus_posterior, required_outlier_halfwidth, rescore_and_optimize!,
    StopWhenNoImprovement, iterations_without_improvement,
    MixtureEmConfig, mixture_em_from_toml, MODEL_UPDATE_TAGS, NUISANCE_TAGS, ResidualMixtureMetadata,
    NoImprovementStop, ParameterChangeStop, StopWhenParameterChangeLess, change_parameters, STOP_TAGS
# The chart problems' rows provider (a lift runs on the `MinimalSolveProblem`
# family). (The local-optimization extension point — `locally_optimize!`, `strategy_loss`,
# `lo_loss` — is VisualGeometryToolkitCore's.)
public ManifoldLiftProvider
# Relative-pose bundle and robot-world calibration.
# The division-camera stack, its state, the lift constraint and CameraRegression's metadata.
public DivisionCameraState, DivisionCameraProblem, board_sq_residuals,
    RadialLift, CameraFitMetadata
public RobotWorldHandEyeCalibration, RobotWorldHandEyeMetadata
public FitMetadata, RotationAveragingMetadata, CameraPosesRefineMetadata, DivisionCameraMetadata
public RotationCenterRefineMetadata   # the rotation-centre refine's metadata (its dots without a bearing or a projection are its fields)
public irls_retcode, irls_stop_rule   # the reweighted fits' termination record, its two fields: Estimation's IRLS fits and downstream ones (BlobBoards' per-blob IRLS)
public uncertain_rotation   # a pairwise rotation fit as an averaging edge: Uncertain(R, noise_scale² × solver_covariance), rad²
public RotationCenterViews, rotation_center_pose, AbstractViewResidual, BearingResidual, ReprojectionResidual   # CameraPosesProblem's rotation-centre view parametrization (its views and their data)
public held_parameters   # the `fixed` that holds every parameter of a camera in a refine (the rotation-centre views require it)
public pixel_whitening_defect   # why a back-projection Jacobian cannot whiten a bearing residual into pixels (BearingPoseProblem's test of its M column)
# Configuration: configs/default.toml is the config of record; these select a
# strategy by tag and write a config value back as a table.
public ESTIMATION_CONFIG_VERSIONS, default_config_path, optimization_strategy_from_toml, lo_from_toml, scoring_from_toml, default_student_t_uniform_scoring, config_dict, OPTIMIZATION_STRATEGY_TAGS, CONTRACT_GENERICS
# The strategy machinery every package registers its strategies with — the
# registry macro `@lo_registry`, `lo_construct`, `LOSS_TAGS` and the registry
# hooks — is VisualGeometryToolkitCore's (its CONTRACT_GENERICS); a name is
# published only by the package that owns it. This package's own are the
# generic writer and the TOML reader's hooks (CONTRACT_GENERICS below).
public lo_config_dict, lo_concrete, lo_resolve_efficiency_constant


# =============================================================================
# INCLUDE ORDER
# =============================================================================

include("interface.jl")
# configs/default.toml → `_dflt`: MUST precede every config-carrying struct.
include("config_defaults.jl")
include("units.jl")              # the estimators' one strip at entry and one reattachment at exit

# The generic loop (BlockCoordinateDescent) is hoisted to VGC
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
include("estimators/ransac/hypotheses.jl")   # _draw_hypotheses: the RANSAC trial, repeated, no scoring
include("estimators/ransac/extent_traits.jl")

# RANSAC problem implementations
include("estimators/ransac/problems/correspondence.jl")
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
include("estimators/ransac/simfitpp.jl")        # SimFitPP: the SIMFIT++ loss width (Edstedt 2025) and its pool selection
include("estimators/ransac/score_selection.jl") # ScoreSelection: the best-scoring stored model of a pool (its streaming cache)
include("estimators/ransac/covariance.jl")   # the one model covariance: the constraint sandwich (all providers)
# The chart-rows provider of every chart problem (VGC's rows contract; needs
# the per-problem chart/residual helpers above).
include("estimators/ransac/problems/manifold_providers.jl")
include("estimators/ransac/solve.jl")   # the finds and the seeded local optimization as algorithms: Ransac, LocalOptimization, RansacCache, the class defaults (after the problems: their classes)
include("estimators/ransac/likelihood_ratio_selection.jl")   # LikelihoodRatioSelection: model selection between seeded local optimizations

# Fits
# Provides the covariance-weighted line fit and its step `_weighted_line`,
# which problems/line.jl's LO-RANSAC `refine!` calls.
include("fitting/line.jl")
include("fitting/radial_camera_fit.jl")        # RadialCameraFit: one board's radial camera, an iterative whitened fit
include("fitting/division_camera_stack.jl")    # the stacked linear focal solve (one iteration of DivisionCameraFit) + its state and failure
include("fitting/gcv.jl")                       # the stack's one GCV of a penalty weight (gcv_curve, GeneralizedCrossValidation on Hyperparameter; the weight's normalization is Core's)
include("fitting/camera_regression.jl")         # CameraRegressionProblem, CameraRegression, PSpline, ForwardSpline: a camera model regressed onto a source over a region
include("fitting/camera_regression_refine.jl")  # the regression's LevenbergMarquardt refine (the chart LM from a guess), OpticalAxis
include("fitting/pose.jl")
include("fitting/irls_weights.jl")           # ReweightedLeastSquares and the fixed-scale IRLS of the reweighted fits: scale input, loss width, weights (before the robot-world fit, one of its problems)
include("fitting/robot_world_hand_eye.jl")
include("fitting/robot_world_hand_eye_refine.jl")   # RobotWorldHandEyeProblem: the reprojection refine of a RobotWorldHandEyeCalibration (a block problem; uses robot_world_hand_eye.jl's type)
include("fitting/camera_poses.jl")              # CameraPosesProblem: a camera and one pose per view from 3D → 2D correspondences (Core's LevenbergMarquardt on its block contract, over VGC's parameter and pose charts)
include("fitting/rotation_center_views.jl")      # CameraPosesProblem's rotation-centre views: one rotation per view, a shared centre and offset
include("fitting/camera_poses_spline.jl")   # its SplineLens route: 2 × 2 patches of sphere-charted controls, implicit-differentiation blocks, the Δ³ penalty, the rotation gauge
include("synthetic_scenes.jl")                    # synthetic_scenes: entry point (the scenes are the Test extension)
include("fitting/reweighted_rotation.jl")           # ReweightedLeastSquares on a rotation between direction pairs (reweighted Kabsch)
include("fitting/reweighted_linear.jl")             # LinearModel; ReweightedLeastSquares on its coefficients
include("fitting/reweighted_rigid_alignment.jl")    # ReweightedLeastSquares on a rigid transform between point sets (reweighted Kabsch; ATE alignment)
include("graphs/view_graph.jl")              # ViewGraph: frames and verified pairs (Graphs.jl underneath)
include("graphs/observation_graph.jl")       # ObservationGraph: frames and what they observe; covisibility → ViewGraph
include("fitting/rotation_averaging.jl") # RotationAveraging, FilteredRotationAveraging: robust rotation averaging on a view graph (spectral or spanning-tree seed, Lie-algebra IRLS)
include("fitting/residual_mixture.jl")     # ResidualMixture: a held model's residuals → mixture EM → MAP labels
include("fitting/division_camera_fit.jl")    # DivisionCameraFit: the focal stack on the shared posterior-weight iteration (after residual_mixture.jl: its MixtureEmConfig)
# TOML loaders, the strategy-tag registry and the writer — dispatches on every
# config-carrying type, so it is included last.
include("config_toml.jl")
# The refine contract's strategies of record (configs/default.toml [refine.<class>] and the bundles' tables), read
# once the TOML reader exists: constants, so a refine's default strategy is inferred (estimators/ransac/solve.jl).
const _REFINE_STRATEGY_DEFAULTS = _read_refine_strategy_defaults()
include("precompile.jl")    # the PrecompileTools workload: the common find and refine (decision 13), after everything it calls

"""
    CONTRACT_GENERICS

The generic functions of VisualGeometryToolkitEstimation another package may add methods
to across a package boundary (VisualGeometryToolkitCore's quality gate,
`contract_generic_violations`): the TOML reader's hooks (`lo_concrete`,
`lo_resolve_efficiency_constant`), the writer (`config_dict`, for a strategy whose
table carries a key beyond the generic one), a wrapper's placement
(`lo_placement`), the posterior-weight stop rule's
counter (`iterations_without_improvement`), the parameter-change stop's measure of a
problem (`change_parameters`), the sampler / running-best
protocol (`reset!`, `draw!`).
"""
const CONTRACT_GENERICS = (lo_concrete, lo_resolve_efficiency_constant,
                           config_dict, scoring_from_toml, lo_placement,
                           iterations_without_improvement, change_parameters, reset!, draw!)

end # module VisualGeometryToolkitEstimation

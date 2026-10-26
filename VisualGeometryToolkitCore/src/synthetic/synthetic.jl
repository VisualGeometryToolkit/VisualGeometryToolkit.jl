# =============================================================================
# Synthetic Data Generation
# =============================================================================
#
# Ground-truth data generators for benchmarking geometric estimators.
# Produces point sets and evaluation metrics for:
#
#   - Conic/ellipse fitting (ellipse points + outliers)
#   - Evaluation metrics (relative error, inlier recall, breakdown curves)
#
# NOTE: Correspondence generators (stereo_correspondences, planar_correspondences,
# synthetic_stereo_rig) moved to VisualGeometryToolkitFeatures.
#
# DEPENDENCY GRAPH:
#
#   conics.jl            (ellipse/conic point generation)
#                         Independent of cameras
#   evaluation.jl        (relative error, inlier recall, breakdown curves)
#                         Independent of generation
#
# =============================================================================

"""
    Synthetic

Synthetic data generation for benchmarking geometric estimators.

Provides ground-truth scene generators with configurable noise and outlier
contamination for conic fitting, and evaluation metrics.

# Conic Points
```julia
pts, mask = ellipse_points(; n=100, outlier_frac=0.3, sigma=0.5, seed=42)
```

# Evaluation
```julia
err = imc_pose_error(F_est, K, R_gt, t_gt)
score = mean_average_accuracy(errs; T = 10)   # mAA@10°: the threshold is required
```

See also: `VisualGeometryToolkitFeatures` for correspondence generators
(`synthetic_stereo_rig`, `stereo_correspondences`, `planar_correspondences`).
"""
module Synthetic

using Random: Random, AbstractRNG, MersenneTwister, randperm
using StaticArrays: SMatrix, SVector, SOneTo, StaticMatrix, StaticVector
using LinearAlgebra: I, Symmetric, cholesky, dot, eigen, issuccess, lu, mul!, norm, normalize, tr, qr, ColumnNorm, pinv
using Statistics: cov, mean
using Distributions: Chisq, median, quantile
using ForwardDiff: ForwardDiff

# Import geometry functions from parent (for evaluation)
using ..VisualGeometryToolkitCore: EssentialMat, HomographyMat, motion,
    translation_angle,
    StereoRig, calibration_matrix, source, extrinsics,
    Point2, Vec3
# ... and the chart and covariance machinery the calibration harness drives
using ..VisualGeometryToolkitCore: FundMat, Rect2, Line2D, LineConfidenceBand, epipolar_band, PerspectiveMap,
    minimal_manifold, implicit_covariance, local_coordinates, retract, tangent_size,
    model, transform_covariance, EuclideanManifold, ambient_coordinates, propagate,
    transform_deriv_params, homography_4pt, to_euclidean, to_affine,
    clip_to_rect, in_band, unit_normalize, transform_deriv,
    manifold_constraints, constraint_jacobians,
    constraint_dim, whitening_factor, EuclideanMap, LeftRigidManifold6, action_jacobian,
    retract_jacobian, _measurement_dim
# ... and the bearing machinery the CAMERA-AWARE E trials need: the harness
# hands an E problem what an E problem takes (3-vector bearings plus their pixel
# Jacobians), with the noise model moving into the same frame.
using ..VisualGeometryToolkitCore: backproject_with_jacobian, project, Csponds, withattrs, sources, targets
using ..VisualGeometryToolkitCore: cd_pair, _chart_rows   # a correspondence's per-pair constraint factors (manifolds/covariance.jl)
using ..VisualGeometryToolkitCore.Cameras: _pixel_K   # the pixel K literal (cameras/calibration.jl)
using ..VisualGeometryToolkitCore: has_units, m   # a physical rig is read in metres (evaluation.jl)
using FlexUnits: ustrip
using Rotations: Rotations, RotMatrix, AngleAxis, rotation_angle
using CoordinateTransformations: AffineMap

# =============================================================================
# EXPORTS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. CONIC POINTS - Ellipse points with noise and outliers
# -----------------------------------------------------------------------------
export ellipse_points

# -----------------------------------------------------------------------------
# 2. EVALUATION - Metrics for comparing estimates to ground truth
# -----------------------------------------------------------------------------
export pose_error, imc_pose_error, optimistic_pose_error, projective_distance
export maa, mean_average_accuracy, maa_score
export normalize_map, perspective_augmentation
public perturbed_pose

# -----------------------------------------------------------------------------
# 3. MONTE-CARLO CALIBRATION of the uncertainty-propagation chain (calibration.jl):
#    public, not exported — reached as `Synthetic.run_calibration` or imported
# -----------------------------------------------------------------------------
public CalibrationResult, run_calibration, is_calibrated, noise_ladder,
    format_result, COVERAGE_LEVELS,
    make_point_covs, perturb_pairs, make_query_cov,
    sample_homography_scene, sample_twoview_scene, track_manifold_solution,
    homography_trial, fundmat_trial, essential_trial, F_NORM_SCALE, fundmat_norm_xform,
    ContainmentResult, run_containment, containment_nominal,
    h_transfer_trial, epipolar_band_trial, second_order_moments,
    sample_pose_scene, pnp_trial, p3p_trial, reprojection_trial
# =============================================================================
# INCLUDE ORDER
# =============================================================================

include("conics.jl")
include("evaluation.jl")
include("augmentation.jl")
include("calibration.jl")

end # module Synthetic

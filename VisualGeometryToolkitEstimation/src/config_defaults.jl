# =============================================================================
# Default configuration values — read from configs/default.toml
# =============================================================================
#
# Included BEFORE every config-carrying struct (RansacConfig, the local-
# optimization strategies, the tuning-constant registry, the fitting
# pipelines), because their keyword defaults read from here. That makes
# `configs/default.toml` the SINGLE source of truth for "the default": there is
# no second set of Julia literals to drift from it, and no sync test standing
# in for a design. (Adopted from BlobBoards' src/config_defaults.jl, 2026-09-03.)
#
# Consequences worth knowing:
#
#   * `RansacConfig()` IS the file, and so is every `Irls{K}()`'s `max_iter`,
#     every `find_*`'s MSAC `threshold`, and the LO strategy each pipeline
#     deploys. Partial keyword construction inherits file values for everything
#     it does not name (`RansacConfig(max_trials = 500)` means "the file,
#     with max_trials overridden").
#   * The loaders (`RansacConfig(path)`, `optimization_strategy_from_toml`)
#     override per key via `_table_kwargs`, so a run config that sets only
#     some keys inherits the rest from THIS file rather than from a parallel
#     set of literals.
#   * Values are baked at PRECOMPILE time, so editing the TOML needs a
#     recompile to take effect — a Revise session must be restarted.
#     That is the intended trade: these are code-level defaults, and per-run
#     variation goes through the loaders.
#
# `include_dependency` is load-bearing: Julia's staleness check tracks
# `include`d files, not arbitrary files read at top level. Without it, editing
# default.toml would leave stale values baked into the .ji and nothing would
# notice — the exact invisible-drift class this file exists to remove.
# =============================================================================

using VisualGeometryToolkitCore: toml_default, parse_defaults, ConfigVersions, ConfigMigration

"""
    default_config_path() -> String

Path to `configs/default.toml` — the configuration of record for every entry
point that is not handed an explicit config.
"""
default_config_path() = normpath(joinpath(@__DIR__, "..", "configs", "default.toml"))

"""
    ESTIMATION_CONFIG_VERSIONS

VisualGeometryToolkitEstimation's configuration converter (Core's
[`ConfigVersions`](@extref VisualGeometryToolkitCore.ConfigVersions)): `configs/default.toml` is version 3, each table
named after the algorithm or problem it configures (conventions/naming.md, decision 11). Read a stored file at the
current version with `VisualGeometryToolkitCore.migrate_config(TOML.parsefile(path), ESTIMATION_CONFIG_VERSIONS)`.

- Version 1 → 2 (0.13): version 1 (0.12 and earlier, no `config_version` key) named its tables after the retired
  estimator functions. The eight estimator tables move: the division-camera fit's to `[division_camera_fit]`, the
  bundles' to `[robot_world_hand_eye_problem]`, `[camera_poses_problem]` and `[camera_regression_problem]`, the three
  reweighted fits' to `[reweighted_least_squares.<rotation | linear | rigid_alignment>]` and the rotation averaging's to
  `[rotation_averaging]`.
- Version 2 → 3 (0.14): a family problem's `[pipelines.<class>]` (its `Ransac` defaults) is
  `[<problem>.ransac]` and its `[refine.<class>]` `[<problem>.levenberg_marquardt]` (`[homography_problem.ransac]`,
  `[fund_mat_problem.levenberg_marquardt]`, …); the hand-eye problem's `motion_pairs` and `[robot_world_hand_eye]`'s
  `min_rel_angle_deg` are `[hand_eye_problem]`'s, that table's `tol` the robot-world fit's
  `[reweighted_least_squares.robot_world_hand_eye_calibration]`; the radial homography's lift seed is
  `[radial_homography_problem.lift_seed]`, `[relpose_bundle]` `[relative_pose_problem]`, `[simfitpp]` `[sim_fit_pp]` and
  `[mixture_em]` `ResidualMixture`'s `[residual_mixture.config]`.
"""
const ESTIMATION_CONFIG_VERSIONS = ConfigVersions("VisualGeometryToolkitEstimation",
    # version 1 → 2 (0.13): the estimator tables named after the algorithm or problem, not the retired functions
    ConfigMigration(;
        tables = ["fit_division_camera", "fit_linear", "fit_rigid_alignment", "fit_rotation", "fit_rotation_averaging",
                  "mixture_em", "optical_axis", "optimization_strategy", "pipelines", "ransac", "refine", "refine_camera",
                  "refine_camera_and_poses", "refine_robot_world_hand_eye", "relpose_bundle", "robot_world_hand_eye",
                  "scoring", "simfitpp"],
        renames = ["fit_division_camera" => "division_camera_fit",
                   "refine_robot_world_hand_eye" => "robot_world_hand_eye_problem",
                   "refine_camera_and_poses" => "camera_poses_problem",
                   "refine_camera" => "camera_regression_problem",
                   "fit_rotation" => "reweighted_least_squares.rotation",
                   "fit_linear" => "reweighted_least_squares.linear",
                   "fit_rigid_alignment" => "reweighted_least_squares.rigid_alignment",
                   "fit_rotation_averaging" => "rotation_averaging"]),
    # version 2 → 3 (0.14): the family problems' tables, and every remaining table, named after the algorithm or problem
    # (a key moved out of a table moves before the table)
    ConfigMigration(;
        tables = ["camera_poses_problem", "camera_regression_problem", "division_camera_fit", "mixture_em", "optical_axis",
                  "optimization_strategy", "pipelines", "ransac", "refine", "relpose_bundle", "reweighted_least_squares",
                  "robot_world_hand_eye", "robot_world_hand_eye_problem", "rotation_averaging", "scoring", "simfitpp"],
        renames = ["pipelines.hand_eye.motion_pairs" => "hand_eye_problem.motion_pairs",
                   "robot_world_hand_eye.min_rel_angle_deg" => "hand_eye_problem.min_rel_angle_deg",
                   "robot_world_hand_eye.tol" => "reweighted_least_squares.robot_world_hand_eye_calibration.tol",
                   "pipelines.radial_homography.lift_seed" => "radial_homography_problem.lift_seed",
                   "pipelines.homography" => "homography_problem.ransac",
                   "pipelines.fundmat" => "fund_mat_problem.ransac",
                   "pipelines.essential" => "essential_problem.ransac",
                   "pipelines.rotation" => "rotation_problem.ransac",
                   "pipelines.circle" => "circle_problem.ransac",
                   "pipelines.ellipse" => "ellipse_problem.ransac",
                   "pipelines.absolute_pose" => "absolute_pose_problem.ransac",
                   "pipelines.bearing_pose" => "bearing_pose_problem.ransac",
                   "pipelines.radial_homography" => "radial_homography_problem.ransac",
                   "pipelines.radial_camera" => "radial_camera_problem.ransac",
                   "pipelines.hand_eye" => "hand_eye_problem.ransac",
                   "refine.homography" => "homography_problem.levenberg_marquardt",
                   "refine.fundmat" => "fund_mat_problem.levenberg_marquardt",
                   "refine.essential" => "essential_problem.levenberg_marquardt",
                   "refine.rotation" => "rotation_problem.levenberg_marquardt",
                   "refine.circle" => "circle_problem.levenberg_marquardt",
                   "refine.ellipse" => "ellipse_problem.levenberg_marquardt",
                   "refine.absolute_pose" => "absolute_pose_problem.levenberg_marquardt",
                   "refine.bearing_pose" => "bearing_pose_problem.levenberg_marquardt",
                   "refine.radial_homography" => "radial_homography_problem.levenberg_marquardt",
                   "refine.radial_camera" => "radial_camera_problem.levenberg_marquardt",
                   "refine.hand_eye" => "hand_eye_problem.levenberg_marquardt",
                   "relpose_bundle" => "relative_pose_problem",
                   "simfitpp" => "sim_fit_pp",
                   "mixture_em" => "residual_mixture.config"]);
    tables = ["absolute_pose_problem", "bearing_pose_problem", "camera_poses_problem", "camera_regression_problem",
              "circle_problem", "division_camera_fit", "ellipse_problem", "essential_problem", "fund_mat_problem",
              "hand_eye_problem", "homography_problem", "optical_axis", "optimization_strategy", "radial_camera_problem",
              "radial_homography_problem", "ransac", "relative_pose_problem", "residual_mixture", "reweighted_least_squares",
              "robot_world_hand_eye_problem", "rotation_averaging", "rotation_problem", "scoring", "sim_fit_pp"])

const _TOML_DEFAULTS = parse_defaults(default_config_path(); versions = ESTIMATION_CONFIG_VERSIONS)

"""
    _dflt(table, key)

Look a default up in `configs/default.toml`. `table` is a dotted path
(`"ransac"`, `"optimization_strategy.hq"`, `"homography_problem.ransac"`). Throws on a
missing table or key BY DESIGN: a keyword default that is not in the file
would be a silent second source of truth, which is what this mechanism exists
to prevent. A strategy's width is deliberately NOT a default: with none it
follows the scorer's decision point, and an explicit `efficiency_constant` or
`loss_width` is the caller's.
"""
_dflt(table::AbstractString, key::AbstractString) = toml_default(_TOML_DEFAULTS, table, key)

# The relative-pose bundle's iteration cap and cost tolerance (configs/default.toml [relative_pose_problem]; its stall
# tolerance is Core's one, `[levenberg_marquardt.stops] stationarity_tol`), read once: a lookup per call cost the
# bundle 1–2 allocations each (it runs per LO refit).
const _RELPOSE_BUNDLE_MAX_ITER = _dflt("relative_pose_problem", "max_iter")::Int
const _RELPOSE_BUNDLE_FTOL = _dflt("relative_pose_problem", "ftol")::Float64

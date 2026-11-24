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
[`ConfigVersions`](@extref VisualGeometryToolkitCore.ConfigVersions)): `configs/default.toml` is version 2. Version 1
(0.12 and earlier, no `config_version` key) named its tables after the retired estimator functions; version 2 names
each after the algorithm or problem that reads it. Read a stored file at the current version with
`VisualGeometryToolkitCore.migrate_config(TOML.parsefile(path), ESTIMATION_CONFIG_VERSIONS)`.

Its one migration (`only(ESTIMATION_CONFIG_VERSIONS.migrations).renames`) moves the eight estimator tables: the division-camera
fit's to `[division_camera_fit]`, the bundles' to `[robot_world_hand_eye_problem]`, `[camera_poses_problem]` and
`[camera_regression_problem]`, the three reweighted fits' to `[reweighted_least_squares.<rotation | linear |
rigid_alignment>]` and the rotation averaging's to `[rotation_averaging]`.
"""
const ESTIMATION_CONFIG_VERSIONS = ConfigVersions("VisualGeometryToolkitEstimation",
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
                   "fit_rotation_averaging" => "rotation_averaging"]);
    tables = ["camera_poses_problem", "camera_regression_problem", "division_camera_fit", "mixture_em", "optical_axis",
              "optimization_strategy", "pipelines", "ransac", "refine", "relpose_bundle", "reweighted_least_squares",
              "robot_world_hand_eye", "robot_world_hand_eye_problem", "rotation_averaging", "scoring", "simfitpp"])

const _TOML_DEFAULTS = parse_defaults(default_config_path(); versions = ESTIMATION_CONFIG_VERSIONS)

"""
    _dflt(table, key)

Look a default up in `configs/default.toml`. `table` is a dotted path
(`"ransac"`, `"optimization_strategy.hq"`, `"pipelines.homography"`). Throws on a
missing table or key BY DESIGN: a keyword default that is not in the file
would be a silent second source of truth, which is what this mechanism exists
to prevent. A strategy's width is deliberately NOT a default: with none it
follows the scorer's decision point, and an explicit `efficiency_constant` or
`loss_width` is the caller's.
"""
_dflt(table::AbstractString, key::AbstractString) = toml_default(_TOML_DEFAULTS, table, key)

# The relative-pose bundle's iteration cap and cost tolerance (configs/default.toml [relpose_bundle]; its stall
# tolerance is Core's one, `[lm] stationarity_tol`), read once: a lookup per call cost the
# bundle 1–2 allocations each (it runs per LO refit).
const _RELPOSE_BUNDLE_MAX_ITER = _dflt("relpose_bundle", "max_iter")::Int
const _RELPOSE_BUNDLE_FTOL = _dflt("relpose_bundle", "ftol")::Float64

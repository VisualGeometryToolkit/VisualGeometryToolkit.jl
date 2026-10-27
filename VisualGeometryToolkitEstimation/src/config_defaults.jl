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
#   * The loaders (`RansacConfig(path)`, `local_optimization_from_toml`)
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

using VisualGeometryToolkitCore: toml_default, parse_defaults

"""
    default_config_path() -> String

Path to `configs/default.toml` — the configuration of record for every entry
point that is not handed an explicit config.
"""
default_config_path() = normpath(joinpath(@__DIR__, "..", "configs", "default.toml"))

const _TOML_DEFAULTS = parse_defaults(default_config_path())

"""
    _dflt(table, key)

Look a default up in `configs/default.toml`. `table` is a dotted path
(`"ransac"`, `"local_optimization.hq"`, `"pipelines.homography"`). Throws on a
missing table or key BY DESIGN: a keyword default that is not in the file
would be a silent second source of truth, which is what this mechanism exists
to prevent. A strategy's width is deliberately NOT a default: with none it
follows the scorer's decision point, and an explicit `scale_multiple` or
`loss_width` is the caller's.
"""
_dflt(table::AbstractString, key::AbstractString) = toml_default(_TOML_DEFAULTS, table, key)

# The chart LM's iteration cap, stall tolerance and cost tolerance
# (configs/default.toml [chart_lm]), read once: the refit runs per LO
# iteration, so it does not look the file up per call. The stall tolerance is
# every chart LM's (the LO refit, `fit_lens`) and the stall postcondition of
# `refine_robot_world_hand_eye`'s NLLSsolver LM (`is_stationary`); the cost
# tolerance `ftol` is the LO refit's alone.
const _CHART_LM_MAX_ITER = _dflt("chart_lm", "max_iter")::Int
const _CHART_LM_STATIONARITY_TOL = _dflt("chart_lm", "stationarity_tol")::Float64
const _CHART_LM_FTOL = _dflt("chart_lm", "ftol")::Float64

# The relative-pose bundle's iteration cap, stall tolerance and cost tolerance
# (configs/default.toml [relpose_bundle]), read once: a lookup per call cost the
# bundle 1–2 allocations each (it runs per LO refit).
const _RELPOSE_BUNDLE_MAX_ITER = _dflt("relpose_bundle", "max_iter")::Int
const _RELPOSE_BUNDLE_STATIONARITY_TOL = _dflt("relpose_bundle", "stationarity_tol")::Float64
const _RELPOSE_BUNDLE_FTOL = _dflt("relpose_bundle", "ftol")::Float64

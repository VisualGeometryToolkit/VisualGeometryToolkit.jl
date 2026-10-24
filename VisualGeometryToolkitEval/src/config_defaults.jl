# =============================================================================
# config_defaults.jl — the default mechanism of VisualGeometryToolkitEval:
# configs/default.toml, read through `_config_default`.
#
# Every value of this package that changes a reported statistic, a rendered
# panel or a raw dataset read and is not a tuning parameter (configs/tuning.toml, Tuning's
# parser) or a scene split (configs/splits.toml) is declared once in
# configs/default.toml; every keyword default, struct default and dataset
# constant reads it here, so there is no second set of Julia literals to drift
# from it. The lookup is VisualGeometryToolkitCore's `toml_default`, which throws on a
# missing table or key by design.
#
# Included first (before the submodules), which reach it as `.._config_default`. The
# file is parsed at precompile; `include_dependency` makes an edit force a
# recompile (a Revise session needs a restart to see one).
# =============================================================================

using TOML: TOML
using VisualGeometryToolkitCore: toml_default

"The defaults of record (configs/default.toml)."
const DEFAULTS_PATH = normpath(joinpath(@__DIR__, "..", "configs", "default.toml"))
include_dependency(DEFAULTS_PATH)
const _TOML_DEFAULTS = TOML.parsefile(DEFAULTS_PATH)

"""
    _config_default(table, key)

The default `key` of `[table]` (a dotted path, `"pool.phototourism"`) in
configs/default.toml. Throws a `KeyError` naming a missing table or key.
"""
_config_default(table::AbstractString, key::AbstractString) = toml_default(_TOML_DEFAULTS, table, key)

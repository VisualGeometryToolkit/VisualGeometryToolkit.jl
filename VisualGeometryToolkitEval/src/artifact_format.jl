# =============================================================================
# artifact_format.jl — the format stamp of every persisted artifact kind.
#
# A persisted artifact (a scoring pool, a tuning sidecar or checkpoint) carries
# `<kind>_format`, the layout of its contents, written by the one writer
# (`_save_artifact`) and checked by the one reader check
# (`_check_artifact_format`) before anything else is read. A file without the
# key is format 1. A change to what an artifact stores bumps its version here;
# older files are then REFUSED with an `ArgumentError` naming the file and
# saying what changed and what to do — never migrated, never read through a
# shim.
#
# History, per kind (the entry's `change` is what the CURRENT version
# introduced, quoted in the refusal; `remedy` is what to do with an older file):
#   pool    1 — the maps stored under their names before 2026-10-01 (or not at all)
#           2 — `StereoScoringData.source_map` / its destination map (2026-10-01)
#           3 — Julia-version-stable streams: the pair subsample and the
#               minimal-sample draws seeded by `stable_seed` (SHA-256) on
#               StableRNGs' `StableRNG`, recorded as `pool_rng` with the
#               `julia_version` (2026-10-01)
#           4 — PLAIN data only (no stack type serialized): the light record,
#               its residual matrices split into a `:pool_residuals` file bound
#               to it by `pool_key`, the per-correspondence `snn` (2026-10-05;
#               the format spec is Pools `save_pool`)
#           5 — the destination map's pair key, `destination_map` (2026-10-26;
#               `Pools.migrate_pool` carries a format-4 file over)
#   pool_residuals
#           1 — the residual matrices of a v4 pool, with `pool_key` and
#               `residual_kind` (2026-10-05)
#   pool_samples
#           1 — the minimal-sample starts of a v4 pool's pairs, bound to it by
#               `pool_key` (2026-10-08)
#   pnp_pool
#           2 — the per-query pseudo-GT `gt_variance` (px²); a file without the
#               stamp is v1
#   tuning  1 — tuning sidecars/checkpoints/bundles before 2026-10-01: realized
#               grids only, no tuning spec
#           2 — the `[tuning]` spec, its hash, the grid SPECS (realized grids are
#               rebuilt from them) and the code identity (2026-10-01)
#           3 — the codified cross-validation: the spec's `cross_validation`
#               (registered scene lists, unit samplings, stable name-keyed
#               seeds) and `budget_sampling`, and the `sampling_unit`
#               (2026-10-01)
#           4 — Julia-version-stable draws: every unit sampling, budget draw
#               and hypothesis stream on StableRNGs' `StableRNG` (the only
#               `rng` a spec may name), and the Julia version + RNG names in
#               the code identity (2026-10-01)
#           5 — the EVALUATED REGION of a tune-phase checkpoint (`evaluated`:
#               label => the selection columns actually computed), so a resumed
#               run knows what is filled (2026-10-03)
#           6 — the tuned point as a NATIVE VALUE rather than a grid index
# =============================================================================

"""
    ARTIFACT_FORMATS

The format version this package writes and reads, per artifact kind
(`:pool`, `:pool_residuals`, `:pool_samples`, `:pnp_pool`, `:tuning`), with the change the
version introduced and the remedy for an older file. The file key is
`"<kind>_format"`; an absent key is version 1.
"""
const ARTIFACT_FORMATS = (
    pool = (version = 5,
            change = "the destination map's pair key, destination_map",
            remedy = "migrate a format-4 pool with Pools.migrate_pool; convert an older one to format 4 first with accv2026 scripts/tools/convert_pools_v4.jl"),
    pool_residuals = (version = 1,
            change = "the residual matrices of a v4 pool",
            remedy = "convert its pool with accv2026 scripts/tools/convert_pools_v4.jl"),
    pool_samples = (version = 1,
            change = "the minimal-sample starts of a v4 pool",
            remedy = "re-derive it with Pools.derive_sample_starts"),
    pnp_pool = (version = 2,
            change = "the per-query pseudo-GT gt_variance",
            remedy = "rebuild it with its accv2026 pool builder"),
    tuning = (version = 6,
            change = "the tuned point as a NATIVE VALUE (`tuned_selection`/`tuned_width`) rather than a grid index",
            remedy = "regenerate it by re-running its tuning"),
)

_format_key(kind::Symbol) = string(kind, "_format")

"The format version an open JLD2 file `f` declares for `kind` (absent key = 1)."
_artifact_format(f, kind::Symbol) = haskey(f, _format_key(kind)) ? Int(f[_format_key(kind)]) : 1

function _artifact_format_error(path, kind::Symbol, v::Integer)
    cur = ARTIFACT_FORMATS[kind]
    v < cur.version && throw(ArgumentError(
        "$path: $kind format v$v predates $(cur.change) (v$(cur.version)) — $(cur.remedy)"))
    throw(ArgumentError("$path: $kind format v$v is newer than this VisualGeometryToolkitEval " *
                        "reads (v$(cur.version))"))
end

"""
    _check_artifact_format(path, kind)

Refuse `path` unless it declares the current `kind` format
([`ARTIFACT_FORMATS`](@ref)): an `ArgumentError` naming the file. Every reader
of a `kind` artifact calls this before reading anything else.
"""
function _check_artifact_format(path::AbstractString, kind::Symbol)
    isfile(path) || throw(ArgumentError("$path: no such $kind artifact"))
    v = JLD2.jldopen(f -> _artifact_format(f, kind), path, "r")
    v == ARTIFACT_FORMATS[kind].version || _artifact_format_error(path, kind, v)
    return nothing
end

"""
    _save_artifact(path, kind; keys...) -> path

The writer of a `kind` artifact: `keys` to `path` (JLD2), stamped
`<kind>_format` at the current version. A key named like the stamp is refused.
"""
function _save_artifact(path::AbstractString, kind::Symbol; keys...)
    stamp = Symbol(_format_key(kind))
    haskey(keys, stamp) && throw(ArgumentError("_save_artifact: `$stamp` is the writer's own stamp"))
    JLD2.jldsave(path; keys..., stamp => ARTIFACT_FORMATS[kind].version)
    return path
end

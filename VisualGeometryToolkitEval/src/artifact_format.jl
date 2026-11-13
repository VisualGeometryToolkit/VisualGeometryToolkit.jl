# =============================================================================
# artifact_format.jl — the format stamp of every persisted artifact kind.
#
# A persisted artifact (a scoring pool, a tuning sidecar or checkpoint) carries
# `<kind>_format`, the layout of its contents, written by the one writer
# (`_save_artifact`) and checked by the one reader check
# (`_check_artifact_format`) before anything else is read. A file without the
# key is format 1. A change to what an artifact stores bumps its version here;
# older files are then REFUSED with an `ArgumentError` naming the file and
# saying what changed and what to do — never read through a shim. The one
# exception is a version a kind declares `read_from`: a file from that version
# on is read through its version steps (VisualGeometryToolkitIO `migrate_record`),
# in memory, and never rewritten (a tuning sidecar is immutable provenance).
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
#           6 — the correspondences' pair key, `correspondences` (the
#               terminology ruling, 2026-11-13; `Pools.migrate_pool` carries a
#               format-4 or -5 file over)
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
#           7 — the efficiency-constant names (2026-11-11, the robust-estimation
#               contract): the scale multiple's names → `efficiency_constant`
#               and `efficiency_constant_window` in the spec, the grid specs
#               and the axis-named data, and the efficiency
#               constant always the PLAIN multiple (`width = c·ŝ`). A v6 file is
#               read through its version step (`migrate_schema` of
#               "vgtk_eval/tuning/v6", persist.jl): the keys renamed, its
#               recorded hash checked against the stored table; its
#               efficiency constants are read through each arm's own
#               `format6_constant` hook (`tuned_width`, the identity unless
#               the roster declares one)
# =============================================================================

"""
    ARTIFACT_FORMATS

The format version this package writes and reads, per artifact kind
(`:pool`, `:pool_residuals`, `:pool_samples`, `:pnp_pool`, `:tuning`), with the change the
version introduced and the remedy for an older file. A kind that reads an
earlier version through its version steps names the oldest it reads,
`read_from`, and what that version introduced (`read_from_change`, quoted when
an even older file is refused). The file key is `"<kind>_format"`; an absent
key is version 1.
"""
const ARTIFACT_FORMATS = (
    pool = (version = 6,
            change = "the correspondences' pair key, correspondences",
            remedy = "migrate a format-4 or -5 pool with Pools.migrate_pool (a tree: scripts/migrate_pools.jl); convert an older one to format 4 first with accv2026 scripts/tools/convert_pools_v4.jl"),
    pool_residuals = (version = 1,
            change = "the residual matrices of a v4 pool",
            remedy = "convert its pool with accv2026 scripts/tools/convert_pools_v4.jl"),
    pool_samples = (version = 1,
            change = "the minimal-sample starts of a v4 pool",
            remedy = "re-derive it with Pools.derive_sample_starts"),
    pnp_pool = (version = 2,
            change = "the per-query pseudo-GT gt_variance",
            remedy = "rebuild it with its accv2026 pool builder"),
    tuning = (version = 7,
            change = "the efficiency-constant names (`efficiency_constant`, `efficiency_constant_window`)",
            remedy = "regenerate it by re-running its tuning",
            read_from = 6,
            read_from_change = "the tuned point as a NATIVE VALUE (`tuned_selection`/`tuned_width`) rather than a grid index"),
)

_format_key(kind::Symbol) = string(kind, "_format")

"The format version an open JLD2 file `f` declares for `kind` (absent key = 1)."
_artifact_format(f, kind::Symbol) = haskey(f, _format_key(kind)) ? Int(f[_format_key(kind)]) : 1

# The oldest version a kind reads (through its version steps), and what it introduced.
_oldest_read(kind::Symbol) = (k = ARTIFACT_FORMATS[kind];
    haskey(k, :read_from) ? (k.read_from, k.read_from_change) : (k.version, k.change))

function _artifact_format_error(path, kind::Symbol, v::Integer)
    cur = ARTIFACT_FORMATS[kind]
    oldest, change = _oldest_read(kind)
    v < oldest && throw(ArgumentError(
        "$path: $kind format v$v predates $change (v$oldest) — $(cur.remedy)"))
    throw(ArgumentError("$path: $kind format v$v is newer than this VisualGeometryToolkitEval " *
                        "reads (v$(cur.version))"))
end

"""
    _check_artifact_format(path, kind) -> Int

Refuse `path` unless it declares a `kind` format this package reads
([`ARTIFACT_FORMATS`](@ref): the current one, or one from the kind's
`read_from` on, which its reader carries over through its version steps): an
`ArgumentError` naming the file. Returns the format the file declares. Every
reader of a `kind` artifact calls this before reading anything else.
"""
function _check_artifact_format(path::AbstractString, kind::Symbol)
    isfile(path) || throw(ArgumentError("$path: no such $kind artifact"))
    v = JLD2.jldopen(f -> _artifact_format(f, kind), path, "r")
    first(_oldest_read(kind)) <= v <= ARTIFACT_FORMATS[kind].version || _artifact_format_error(path, kind, v)
    return v
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

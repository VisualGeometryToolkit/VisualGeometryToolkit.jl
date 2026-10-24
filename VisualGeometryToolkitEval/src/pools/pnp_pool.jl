# =============================================================================
# src/pools/pnp_pool.jl — the on-disk layout of a 2D-3D (PnP) pool file: one
# writer and one reader for every producer and consumer. PnP pools are in
# PIXELS (`ImageFrame`): queries carry their own camera.
# =============================================================================

"""
    save_pnp_pool(path, pools, misses; score) -> path

The writer of a PnP pool file (artifact kind `:pnp_pool`, format 2), JLD2,
plain data:

| key | meaning |
|---|---|
| `pnp_pool_format` | `2` |
| `frame` | `"ImageFrame"` — the 2-D observations are pixels |
| `pools` | per-query plain records: `name`, `camera_model`, `width`, `height`, `camera_params`, `u` (n×2 px), `X` (n×3), `snn`, `p3d`, `nobs` [+ `gt_R`, `gt_t`, `gt_variance` (px², NaN: untrusted)] |
| `misses` | the listed queries with no pool |
| `score` | what `snn` holds: `"snn"` (raw Lowe ratio), `"one_minus_lg"` or `"none"` |
"""
function save_pnp_pool(path::AbstractString, pools, misses; score::AbstractString)
    mkpath(dirname(abspath(path)))
    _save_artifact(path, :pnp_pool; frame = "ImageFrame", pools, misses, score = String(score))
    @info "pool file written" path pools = length(pools) misses = length(misses) score
    return path
end

"""
    load_pnp_pool(path) -> (; pools, misses, score)

The reader of a PnP pool file ([`save_pnp_pool`](@ref) is the format): refuses
another `pnp_pool_format` and a frame other than `ImageFrame`.
"""
function load_pnp_pool(path::AbstractString)
    _check_artifact_format(path, :pnp_pool)
    fr, pools, misses, score = JLD2.load(path, "frame", "pools", "misses", "score")
    fr == "ImageFrame" ||
        throw(ArgumentError("$path: a PnP pool is stored in ImageFrame (pixels), this one declares $(repr(fr))"))
    return (; pools, misses, score)
end

# =============================================================================
# StereoRawPair — the raw-pair interface every dataset reader satisfies
# =============================================================================
#
# A raw pair carries the matched correspondences, calibration, image sizes and
# GT of one image pair BEFORE a pool is drawn from it, all in PIXELS. The
# concrete readers (PhotoTourism, HEB, RoMa) live in the evaluation package;
# the pool-pair constructor touches a raw pair ONLY through these accessors.
# Raw pairs are never persisted.

"""
    StereoRawPair

Abstract supertype of a dataset reader's raw image pair: the input a pool pair
is drawn from, in PIXELS. Read through its accessors, whose defaults read the
field of the same name (a reader that stores a quantity differently overrides
the accessor):

| Accessor | Default field | Meaning |
|---|---|---|
| [`pair_csponds`](@ref) | `csponds` | pixel correspondences `x₁ => x₂` |
| [`pair_snn`](@ref) | `snn` | per-correspondence second-nearest-neighbour ratio (low = good) |
| [`pair_K1`](@ref), [`pair_K2`](@ref) | `K1`, `K2` | pixel calibration matrices |
| [`pair_size1`](@ref), [`pair_size2`](@ref) | `size1`, `size2` | image sizes (`Size2`) |
| [`pair_R_gt`](@ref), [`pair_t_gt`](@ref) | `R_gt`, `t_gt` | GT relative motion, source → target |
| [`pair_gt_inliers`](@ref) | `gt_inliers` | GT inlier mask |
| [`pair_scene`](@ref) | — | the pair's source-scene label, `nothing` unless overridden |

Its pool frame is [`pool_normalization`](@ref)`(raw)`, the normalization of
its two image sizes.
"""
abstract type StereoRawPair end

"`pair_csponds(raw::StereoRawPair)`: the pixel correspondences `x₁ => x₂` (see [`StereoRawPair`](@ref))."
pair_csponds(raw::StereoRawPair) = raw.csponds

"`pair_snn(raw::StereoRawPair)`: the per-correspondence second-nearest-neighbour ratio, low = good (see [`StereoRawPair`](@ref))."
pair_snn(raw::StereoRawPair) = raw.snn

"`pair_K1(raw::StereoRawPair)`: the source camera's pixel calibration matrix (see [`StereoRawPair`](@ref))."
pair_K1(raw::StereoRawPair) = raw.K1

"`pair_K2(raw::StereoRawPair)`: the target camera's pixel calibration matrix (see [`StereoRawPair`](@ref))."
pair_K2(raw::StereoRawPair) = raw.K2

"`pair_size1(raw::StereoRawPair)`: the source image's size, a `Size2` (see [`StereoRawPair`](@ref))."
pair_size1(raw::StereoRawPair) = raw.size1

"`pair_size2(raw::StereoRawPair)`: the target image's size, a `Size2` (see [`StereoRawPair`](@ref))."
pair_size2(raw::StereoRawPair) = raw.size2

"`pair_R_gt(raw::StereoRawPair)`: the GT relative rotation, source → target (see [`StereoRawPair`](@ref))."
pair_R_gt(raw::StereoRawPair) = raw.R_gt

"`pair_t_gt(raw::StereoRawPair)`: the GT relative translation, source → target (see [`StereoRawPair`](@ref))."
pair_t_gt(raw::StereoRawPair) = raw.t_gt

"`pair_gt_inliers(raw::StereoRawPair)`: the GT inlier mask of the correspondences (see [`StereoRawPair`](@ref))."
pair_gt_inliers(raw::StereoRawPair) = raw.gt_inliers

"`pair_scene(raw::StereoRawPair)`: the pair's source-scene label, `nothing` unless the reader overrides it (see [`StereoRawPair`](@ref))."
pair_scene(::StereoRawPair) = nothing

# A raw pair's pool frame: the construction from its two image sizes.
pool_normalization(raw::StereoRawPair) = pool_normalization(pair_size1(raw), pair_size2(raw))

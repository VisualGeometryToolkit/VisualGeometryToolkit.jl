# =============================================================================
# PoolPair — one image pair of a scoring pool (the light record)
#
# Stores correspondences, precomputed minimal-sample models and their pose
# errors in NormalizedImageFrame (the pool frame: image centre, half-diagonal
# unit, through the stored per-image maps), with the pixel calibration and GT
# pose. Used by the Shekhovtsov IJCV 2025 evaluation protocol (Sections 6.1 and
# 6.2). The per-model residual matrix is not stored here.
# =============================================================================

# The view ids of a pool pair: the rig's cameras and the framed maps' image
# frames are labeled with them (one source for both).
const _POOL_VIEW_IDS = (:cam1, :cam2)

# The stored correspondence type: `csponds(sources, targets)` of pool-frame 2-vectors.
const _PoolCsponds = typeof(csponds(SVector{2, Float64}[], SVector{2, Float64}[]))
_pool_csponds(cs::_PoolCsponds) = cs
_pool_csponds(cs) = csponds([SVector{2, Float64}(c.first) for c in cs], [SVector{2, Float64}(c.second) for c in cs])

"""
    PoolPair{M}

One image pair of a scoring pool: the light record — everything about the pair
except the per-model residual matrix, which is a separate file bound to the
record by a content key.

The pool is the one place in the stack that stores data normalized: every
point datum and model is in **[`NormalizedImageFrame`](@ref)** (image centre,
half-diagonal unit), through the per-image maps the record carries. See
"Coordinate frame" below.

# Type Parameters
- `M`: Model type, expressed in `NormalizedImageFrame` (`FundMat{Float64}` for
  the E and F pools, `HomographyMat{Float64}` for the H pools).

# Fields
- `csponds`: Correspondences in `NormalizedImageFrame` — NOT pixels, stored as
  the columnar `csponds(sources, targets)` of `SVector{2,Float64}` sides (any
  vector of 2-vector pairs is converted to it at construction). Each side is
  mapped pixel → normalized by its own map (`source_map`, `target_map`).
  [`pixel_csponds`](@ref) is the exact way back to pixels.
- `K1::SMatrix{3,3,Float64,9}`, `K2::SMatrix{3,3,Float64,9}`: the **pixel**
  calibration of the source and target cameras ([`calibration_matrices`](@ref)).
- `R_gt::SMatrix{3,3,Float64,9}`, `t_gt::SVector{3,Float64}`: the GT relative
  motion, source → target (`X₂ = R_gt·X₁ + t_gt`; [`gt_pose`](@ref)).
- `models::Vector{M}`: Precomputed minimal-sample models, expressed in
  `NormalizedImageFrame` (e.g. fundamentals conjugated by the per-image maps).
- `sample_indices::Matrix{Int}`: `(m, num_models)` — 1-based sample indices per model.
- `errors_r::Vector{Float64}`: `(num_models,)` — rotation error [deg] per model vs GT.
- `errors_t::Vector{Float64}`: `(num_models,)` — translation error [deg] per model vs GT.
- `gt_inliers::BitVector`: `(n,)` — GT inlier mask (from bundle adjustment), one
  entry per correspondence (checked at construction).
- `gt_variance::Float64`: σ̂²_p = RSS/ν of the residuals under the **oracle**
  pose (from the GT `R, t`) on the SfM-track keypoints — NOT a DLT refit.
  `ν = n·d_g` (oracle: no fitted DOF). NaN if too few tracks.
- `source_map::SMatrix{3,3,Float64,9}`, `target_map::SMatrix{3,3,Float64,9}`:
  the per-image maps `T` (pixel → `NormalizedImageFrame`) the data is stored in
  — the stored truth of the pool's frame, and the two sides of its
  [`pool_normalization`](@ref) (`source_map(N)`, `target_map(N)`); framed, they
  are [`pool_frame_maps`](@ref).
- `snn::Union{Nothing,Vector{Float32}}`: `(n,)` — the per-correspondence
  second-nearest-neighbour ratio the pool's hypotheses were sampled by, where
  the generator's is known; `nothing` otherwise.

# Coordinate frame

All point data (`csponds`, `models`, and the separate residuals) lives in
**`NormalizedImageFrame`**: each image is normalized by its own `T`
(`source_map`/`target_map`), **isotropically** — translate to the image centre
and scale BOTH axes by `1/s`, `s = ½√(W²+H²)` HALF the image diagonal:
`(ũ,ṽ) = (u−W/2, v−H/2)/s`, which puts the image corners at unit radius
([`DiagonalNormalization`](@ref)`(size)` per side:
[`pool_normalization`](@ref)`(size1, size2)`, what the pool generator writes).
Every width on pool data — the outlier half-width, the loss width, the IG
scale prior — is in this half-diagonal unit. The stored maps are the truth of
the frame: every conversion goes through them and none is reconstructed from
the image size.

Isotropy is the key property. Feature noise is isotropic in pixels
(`Σ_px = σ²·I`); a single scale maps it to `Σ_pool = (σ²/s²)·I`, still
isotropic, so the scalar `σ²`, the single global outlier half-width, the IG
scale prior, and the Sampson distance all stay correctly specified. (A per-axis
box normalization would make `Σ_pool` anisotropic and force a full 2×2 noise
covariance through the scoring.) Normalizing by image size — not focal — also
keeps the outlier half-width resolution/FOV-uniform: the `K⁻¹` calibrated
CHART — the z = 1 plane, VisualGeometryToolkitCore's `GnomonicFrame` — would not (extent
`≈ tan(fov/2)`, so a single outlier half-width drifts ~4× with focal across
PhotoTourism), which is why the pool frame is the scoring frame even for the
calibrated (essential) problem.

# Re-solving / local-optimization applications

Because the pool-frame maps `T` are stored, the record is self-contained for
re-solving and LO: the metric frame is recovered EXACTLY (no reconstruction
from image size or principal point). With `T` and the pixel `K`, the
**pool-frame calibration** is `Kc = T·K` ([`pool_frame_calibrations`](@ref)),
and:
- pool frame → the calibrated z = 1 CHART (`GnomonicFrame`, not a unit ray):
  `x_chart = inv(Kc) · x_pool`, a unit BEARING being `x_chart / ‖x_chart‖`;
- a pool-frame fundamental `F` → metric essential: `EssentialMat(F, Kc₁, Kc₂)`;
- scoring an essential `E` in the pool frame: `FundMat(E, Kc₁, Kc₂)` Sampson.
This is exact, so the global pool-frame outlier half-width needs **no**
per-pair conversion. A pixel tolerance enters the pool frame only through
[`pool_pixel_scale`](@ref).

# Construction

    PoolPair(; csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r,
             errors_t, gt_inliers, gt_variance, source_map, target_map, snn = nothing)
    PoolPair(; csponds, …, gt_variance, normalization::CspondNormalization, snn = nothing)
    PoolPair{M}(; …)

The keyword form is the one to build a record with (every field named; `M`
from `models` unless given): the maps come either as `source_map` and
`target_map` or as the `normalization` whose sides they are, never both. The
positional forms below take the fields in declaration order.

    PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices,
                errors_r, errors_t, gt_inliers, gt_variance,
                source_map, target_map, snn)
    PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices,
                errors_r, errors_t, gt_inliers, gt_variance,
                N::CspondNormalization, snn)

(`M` is inferred from `models` when omitted.) The maps, or the
[`pool_normalization`](@ref) whose sides they are, are REQUIRED: a record
without them is refused with an `ArgumentError`, never read as the identity —
an identity default once silently read a map-less record, a pool from a
generator that did not persist its maps, as if it were already in the pool
frame. Synthetic data built directly in the pool frame supplies the identity
EXPLICITLY. Each map must be affine (last row `[0 0 1]`, else `ArgumentError`),
and the columns must agree in length (`DimensionMismatch`): per point,
`gt_inliers` and `snn` (when present) have one entry per correspondence; per
model, `sample_indices` has one column and `errors_r`, `errors_t` one entry. Every
pool on disk satisfies this (all are format v4, loaded through these checks).
The record is built by the loader from plain data; it is never serialized as a
struct.
"""
struct PoolPair{M}
    csponds::_PoolCsponds
    K1::SMatrix{3, 3, Float64, 9}
    K2::SMatrix{3, 3, Float64, 9}
    R_gt::SMatrix{3, 3, Float64, 9}
    t_gt::SVector{3, Float64}
    models::Vector{M}
    sample_indices::Matrix{Int}
    errors_r::Vector{Float64}
    errors_t::Vector{Float64}
    gt_inliers::BitVector
    gt_variance::Float64
    # Per-image pool-frame maps `T` (pixel → NormalizedImageFrame) that the stored
    # `csponds`/`models` are expressed in. The pool-frame calibration is `Kc = T·K`.
    source_map::SMatrix{3, 3, Float64, 9}
    target_map::SMatrix{3, 3, Float64, 9}
    snn::Union{Nothing, Vector{Float32}}

    function PoolPair{M}(
            csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
            gt_inliers, gt_variance, source_map, target_map, snn
        ) where {M}
        _check_pool_map(:source_map, source_map)
        _check_pool_map(:target_map, target_map)
        n = length(csponds); m = length(models)
        size(sample_indices, 2) == m || throw(DimensionMismatch(
            "PoolPair: sample_indices has $(size(sample_indices, 2)) columns for $m models"))
        (length(errors_r) == m && length(errors_t) == m) || throw(DimensionMismatch(
            "PoolPair: errors_r/errors_t have $(length(errors_r))/$(length(errors_t)) entries for $m models"))
        length(gt_inliers) == n || throw(DimensionMismatch(
            "PoolPair: gt_inliers has $(length(gt_inliers)) entries for $n correspondences"))
        (snn === nothing || length(snn) == n) || throw(DimensionMismatch(
            "PoolPair: snn has $(length(snn)) entries for $n correspondences"))
        return new{M}(_pool_csponds(csponds), K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
            gt_inliers, gt_variance, source_map, target_map, snn)
    end
end

# A stored map is a 2-D affine map in homogeneous form: its last row is exactly
# [0 0 1] (what `to_homogeneous` of an AffineMap writes).
function _check_pool_map(name::Symbol, T::AbstractMatrix)
    size(T) == (3, 3) || throw(DimensionMismatch("PoolPair: $name must be 3×3, got $(size(T))"))
    (T[3, 1] == 0 && T[3, 2] == 0 && T[3, 3] == 1) || throw(ArgumentError(
        "PoolPair: $name must be an affine map pixel → NormalizedImageFrame (last row [0 0 1]), " *
            "got last row $(T[3, :])"))
    return nothing
end

PoolPair(csponds, K1, K2, R_gt, t_gt, models::AbstractVector{M}, sample_indices, errors_r, errors_t,
    gt_inliers, gt_variance, source_map, target_map, snn) where {M} =
    PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
        gt_inliers, gt_variance, source_map, target_map, snn)

# The record from its two-sided normalization: the sides are the stored maps.
PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
    gt_inliers, gt_variance, N::CspondNormalization, snn) where {M} =
    PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
        gt_inliers, gt_variance, to_homogeneous(source_map(N)), to_homogeneous(target_map(N)), snn)
PoolPair(csponds, K1, K2, R_gt, t_gt, models::AbstractVector{M}, sample_indices, errors_r, errors_t,
    gt_inliers, gt_variance, N::CspondNormalization, snn) where {M} =
    PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
        gt_inliers, gt_variance, N, snn)

# The keyword form: every field named; the maps as `source_map`/`target_map` or
# as the `normalization` whose sides they are.
function PoolPair{M}(; csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
        gt_inliers, gt_variance, source_map = nothing, target_map = nothing,
        normalization::Union{Nothing, CspondNormalization} = nothing, snn = nothing) where {M}
    if normalization !== nothing
        (source_map === nothing && target_map === nothing) || throw(ArgumentError(
            "PoolPair: give the maps as `source_map`/`target_map` or as `normalization`, not both"))
        return PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
            gt_inliers, gt_variance, normalization, snn)
    end
    (source_map === nothing || target_map === nothing) && _refuse_mapless()
    return PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
        gt_inliers, gt_variance, source_map, target_map, snn)
end
PoolPair(; models::AbstractVector, kwargs...) = PoolPair{eltype(models)}(; models, kwargs...)

# A pool record without its maps is refused, never read as the identity.
PoolPair{M}(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
    gt_inliers, gt_variance) where {M} = _refuse_mapless()
PoolPair(csponds, K1, K2, R_gt, t_gt, models, sample_indices, errors_r, errors_t,
    gt_inliers, gt_variance) = _refuse_mapless()
_refuse_mapless() = throw(ArgumentError(
    "a PoolPair pool record needs its maps `source_map`, `target_map` " *
        "(pixel → NormalizedImageFrame, the frame its data is stored in), or the " *
        "pool_normalization whose sides they are; a record without them is refused, " *
        "never read as the identity. Supply the maps the data was normalized with — " *
        "the identity explicitly for synthetic data built in the pool frame."))

# =============================================================================
# Accessors
# =============================================================================

"""
    num_points(pair::PoolPair) -> Int

The number of correspondences of the pair: the residual matrix's row count.
"""
num_points(pair::PoolPair) = length(pair.csponds)

"""
    num_models(pair::PoolPair) -> Int

The number of stored minimal-sample models: the residual matrix's column count.
"""
num_models(pair::PoolPair) = length(pair.models)

"""
    pool_imc(pair::PoolPair, j::Integer) -> Float64

The stored selection pose error [deg] of pool model `j` — the "imc" every pool
evaluation reports for the model it selects — dispatched on the pool's model
type:

- a homography pool (`HomographyMat`): the rotation error `errors_r[j]` alone —
  a homography's translation is known only up to the plane's scale, so its
  stored `errors_t` is not part of the selection error;
- every other pool (`FundMat`, `EssentialMat`): `max(errors_r[j], errors_t[j])`.

The one definition: the scorers' staged pairs bake it in, and so every staged
grid.
"""
pool_imc(pair::PoolPair, j::Integer) = max(pair.errors_r[j], pair.errors_t[j])
pool_imc(pair::PoolPair{<:HomographyMat}, j::Integer) = pair.errors_r[j]

"""
    calibration_matrices(pair::PoolPair) -> (K1, K2)

The PIXEL calibration matrices of the source and target cameras, as
`SMatrix{3,3,Float64,9}`. The pool-frame calibrations are
[`pool_frame_calibrations`](@ref).
"""
calibration_matrices(pair::PoolPair) = (pair.K1, pair.K2)

"""
    gt_pose(pair::PoolPair) -> (R_gt, t_gt)

The ground-truth relative pose (source → target, `X₂ = R_gt·X₁ + t_gt`) as
`(SMatrix{3,3,Float64,9}, SVector{3,Float64})`. `t_gt` is the stored
(unnormalized) translation; pose-error callers use it direction-only.
"""
gt_pose(pair::PoolPair) = (pair.R_gt, pair.t_gt)

"""
    rig(pair::PoolPair) -> StereoRig

The GT stereo rig of the pair, built on access from the stored pixel
calibrations and GT pose (camera 1 at the origin, camera 2 at the GT pose; view
ids `:cam1`, `:cam2`, the ids of [`pool_frame_maps`](@ref)). The record stores
plain matrices, not the rig: a change to the rig type never makes a stored pool
unreadable.
"""
rig(pair::PoolPair) = StereoRig(pair.K1, pair.K2, pair.R_gt, pair.t_gt; check = false, ids = _POOL_VIEW_IDS)

"""
    pool_normalization(pair::PoolPair) -> CspondNormalization
    pool_normalization(size1, size2) -> CspondNormalization
    pool_normalization(raw::StereoRawPair) -> CspondNormalization

The pool's two-sided normalization (pixel → `NormalizedImageFrame` per image),
with its inverses. Of a record: its stored maps, `pair.source_map` and
`pair.target_map`. Of two image sizes: the map a pool of such images is written
in, [`DiagonalNormalization`](@ref) per side (image centre, half diagonal,
corners at unit radius) — THE construction of the pool frame from an image
size; a raw pair's is that of its two image sizes; a record built from it
(`PoolPair{M}(…, N, snn)`) stores its sides. A
stored pool model `M` comes back to pixels by composition with the two sides,
`inv(D) ∘ M ∘ S` with `S, D = source_map(N), target_map(N)`, and goes the other
way by `D ∘ M ∘ inv(S)`. It returns a normalization, not a frame: `*Frame`
names frame tags only. The same maps, frame-typed, are
[`pool_frame_maps`](@ref).
"""
pool_normalization(pair::PoolPair) =
    CspondNormalization(AffineMap(pair.source_map), AffineMap(pair.target_map))   # its own analytic inverses
pool_normalization(size1, size2) =
    CspondNormalization(AffineMap(DiagonalNormalization(size1)), AffineMap(DiagonalNormalization(size2)))

"""
    pool_frame_maps(pair::PoolPair) -> (S, D)

The stored per-image maps as frame-typed affine maps,
`S::FramedAffine{ImageFrame{:cam1}, NormalizedImageFrame{:cam1}}` and
`D::FramedAffine{ImageFrame{:cam2}, NormalizedImageFrame{:cam2}}`: pixels of
each view in, the pool frame out; `inv` goes back. The frame lives on the maps
in memory only — the stored maps are plain matrices.
"""
function pool_frame_maps(pair::PoolPair)
    a, b = _POOL_VIEW_IDS
    return (
        _framed_affine(ImageFrame{a}, NormalizedImageFrame{a}, pair.source_map),
        _framed_affine(ImageFrame{b}, NormalizedImageFrame{b}, pair.target_map),
    )
end

function _framed_affine(::Type{Src}, ::Type{Dst}, T::SMatrix{3, 3, Float64, 9}) where {Src, Dst}
    a = AffineMap(T)   # the one homogeneous 3×3 → (linear, translation) split
    return FramedAffine{Src, Dst, Float64}(a.linear, a.translation)
end

"""
    pool_frame_calibrations(pair::PoolPair) -> (Kc1, Kc2)

The POOL-FRAME calibration matrices `Kc = T·K` of the source and target cameras:
the pixel calibrations of [`calibration_matrices`](@ref) composed with the
stored maps `source_map`, `target_map`, which map pool-frame coordinates ↔ the
calibrated z = 1 chart (`GnomonicFrame`) exactly. What a problem built
on the stored pool data takes as its `K`.
"""
function pool_frame_calibrations(pair::PoolPair)
    K1, K2 = calibration_matrices(pair)
    return (pair.source_map * K1, pair.target_map * K2)
end

"""
    pool_pixel_scale(pair::PoolPair) -> Float64

Pool units per PIXEL of the pair: the isotropic scale
([`similarity_scale`](@ref)) of its pixel → `NormalizedImageFrame` maps, the
larger of the two views (a gate takes one tolerance; the two half-diagonals
differ by a few percent, so the denser view's 0.1 px reads as ≈ 0.1 px in the
other). THE entry of a pixel tolerance into the pool frame:
`τ_pool = pool_pixel_scale(pair) · τ_px`.
"""
function pool_pixel_scale(pair::PoolPair)
    S, D = pool_frame_maps(pair)
    return max(similarity_scale(S), similarity_scale(D))
end

"""
    pixel_csponds(pair::PoolPair) -> StructArray

The pair's correspondences back in PIXELS: each side through the inverse of
its stored map ([`pool_frame_maps`](@ref)), so the result is in the pixel frame
the pool's maps were built in (the generator's input frame). The exact
inverse of the pool's normalization up to rounding; the way out of the pool
frame for a consumer that works in pixels.
"""
function pixel_csponds(pair::PoolPair)
    S, D = pool_frame_maps(pair)
    return csponds(map(inv(S), sources(pair.csponds)), map(inv(D), targets(pair.csponds)))
end

"""
    pixel_residuals(pair::PoolPair) -> Matrix{Float32}

The image-plane Sampson residuals of the pair's stored models in the PIXEL
frame, `num_points × num_models`. Pixels are the frame the SIMFIT++ reference works in, so
its pixel bounds apply to them without any scalar approximation (the pool frame
scales the two images independently, so [`pool_pixel_scale`](@ref) is a bound on
the scale, not a conversion). One
n×M scan of the light record: the correspondences come back to pixels through
the stored maps ([`pixel_csponds`](@ref)), each model by composition with the
pool normalization's two sides, `inv(D) ∘ M ∘ S` (a pool `F` → `T₂ᵀ F T₁`, a
pool `H` → `T₂⁻¹ H T₁`). A model type with neither law (a bare matrix) has no
`sampson_distance` and is a `MethodError`.

It never reads the pool's stored residual matrix, so it does not depend on that
matrix's residual kind. The stored residuals of an essential pool are TANGENT
Sampson distances (the pool generator's `residual_kind = :tangent_sampson`),
those of an F or H pool image-plane Sampson (`:sampson`); this exit is
image-plane Sampson for every pool, the quantity SIMFIT++'s pixel bounds are
stated in.
"""
function pixel_residuals(pair::PoolPair)
    n, M = num_points(pair), num_models(pair)
    (n == 0 || M == 0) && return Matrix{Float32}(undef, n, M)
    N = pool_normalization(pair)
    S, D = source_map(N), target_map(N)
    iD = inv(D)
    px = pixel_csponds(pair)
    u = [SVector{2,Float64}(x) for x in sources(px)]
    v = [SVector{2,Float64}(x) for x in targets(px)]
    out = Matrix{Float32}(undef, n, M)
    @inbounds for (j, m) in enumerate(pair.models)
        mp = iD ∘ m ∘ S
        for i in 1:n
            out[i, j] = Float32(sampson_distance(mp, u[i], v[i]))
        end
    end
    return out
end

# =============================================================================
# Pool pairs — the pool-frame problems of a pool record, and the record's
# construction from a raw pair
# =============================================================================
#
# A scoring pool (VisualGeometryCore.Pools) stores, per image pair, minimal-sample
# models with their pose errors and ground truth, all in NormalizedImageFrame
# (the pool frame), and a separate residual matrix. This file holds the two
# halves of the family-specific logic, in ONE place:
#
#   - the POOL-FRAME problem of a record, `EssentialProblem(pair)`,
#     `FundMatProblem(pair)`, `HomographyProblem(pair)`: the problem every pool
#     residual, σ̂² and pool-frame LO refit is computed on;
#   - the record's construction, `PoolPair(rng, problem, raw; …)`: hypotheses of
#     a PIXEL-frame problem (`draw_hypotheses`), carried into the pool frame once
#     by `pool_normalization(raw)`, with their pose errors, the GT mask, σ̂² and
#     the residual matrix of the record's own pool-frame problem.
#
# The two problems per pair have different roles, and both are deliberate: the
# hypotheses are SOLVED in pixels (the problem's sample gate reads its pixel
# tolerances there), while every stored distance and σ̂² is a POOL-FRAME
# quantity, the unit the pool's outlier half-widths and loss widths are stated
# in. Frame rule 5 (the benchmark pool): the record is the one declared
# exception to "frame in = frame out", and it names its frame on every exit.
# =============================================================================

# -----------------------------------------------------------------------------
# The pool-frame problems of a record
# -----------------------------------------------------------------------------

# The pool frame's coincidence tolerance: 0.1 px through the record's own
# pixel → pool scale, the one entry of a pixel tolerance into the pool frame.
# Always set; a caller cannot state it in other units.
function _pool_coincident_tol(pair::PoolPair, kw)
    haskey(kw, :coincident_tol) && throw(ArgumentError(
        "a pool-frame problem's coincident_tol is always COINCIDENT_PX · pool_pixel_scale(pair) " *
            "(0.1 px in pool units); it is not a keyword"))
    return VisualGeometryCore.COINCIDENT_PX * pool_pixel_scale(pair)
end

"""
    EssentialProblem(pair::PoolPair{<:FundMat}; kw...) -> EssentialProblem

The pool-frame essential problem of a pool record: its correspondences
(`NormalizedImageFrame`) back-projected to unit bearings by the pool-frame
cameras `CameraModel(Kcᵢ)`, `Kcᵢ = Tᵢ·Kᵢ` ([`pool_frame_calibrations`]).
Residuals are the tangent Sampson whitened through `Kc`, so they are pool-frame
quantities: the unit of the pool's stored distances, σ̂², outlier half-widths
and loss widths. The sample gate's coincidence tolerance is 0.1 px in pool
units (`COINCIDENT_PX · pool_pixel_scale(pair)`), always. `kw` go to
`EssentialProblem` (e.g. a `sampler`).

A pool's `Kc` is a matrix, so this problem is pinhole by construction.
"""
function EssentialProblem(pair::PoolPair{<:FundMat}; kw...)
    tol = _pool_coincident_tol(pair, kw)
    Kc1, Kc2 = pool_frame_calibrations(pair)
    return EssentialProblem(CameraModel(Kc1), CameraModel(Kc2), pair.csponds; coincident_tol = tol, kw...)
end

"""
    FundMatProblem(pair::PoolPair{<:FundMat}; kw...) -> FundMatProblem

The pool-frame fundamental problem of a pool record: its correspondences
(`NormalizedImageFrame`) verbatim. Its sample gate reads the scale-free
tolerance on its normalized view, so it takes no `coincident_tol`. `kw` go to
`FundMatProblem`.
"""
FundMatProblem(pair::PoolPair{<:FundMat}; kw...) = FundMatProblem(pair.csponds; kw...)

"""
    HomographyProblem(pair::PoolPair{<:HomographyMat}; kw...) -> HomographyProblem

The pool-frame homography problem of a pool record: its correspondences
(`NormalizedImageFrame`) verbatim, no measurement maps. Its sample gate reads
the scale-free tolerance on its normalized view, so it takes no
`coincident_tol`. `kw` go to `HomographyProblem`.
"""
HomographyProblem(pair::PoolPair{<:HomographyMat}; kw...) = HomographyProblem(pair.csponds; kw...)

# -----------------------------------------------------------------------------
# σ̂² of a pool record: how the GT noise variance is measured
# -----------------------------------------------------------------------------

"""
    PoolVariance

How a pool record's `gt_variance` (σ̂², pool units²) is measured — a dataset
fact, chosen by the pool's producer:

- [`InlierVariance`](@ref)`(; min_inliers)`: over the raw pair's GT-inlier
  correspondences;
- [`TrackVariance`](@ref)`(source, target)`: over separate pixel point pairs
  (SfM-track keypoints), under the oracle model.
"""
abstract type PoolVariance end

"""
    InlierVariance(; min_inliers)

σ̂² over the raw pair's GT-inlier correspondences, in the pool frame, with the
fitted-model residual DOF ([`residual_dof`](@ref)): for E and F the RSS of the
ORACLE model (the GT pose) in the family's own residual (E: tangent Sampson on
the record's [`EssentialProblem`](@ref); F: plane Sampson); for H, which has no
oracle model, the RSS of an unconstrained DLT refit to the GT inliers (NaN when
the masked design matrix fails VGC's `test_dlt_conditioning`). NaN when fewer
than `min_inliers` GT inliers.
"""
struct InlierVariance <: PoolVariance
    min_inliers::Int
end
InlierVariance(; min_inliers::Integer) = InlierVariance(Int(min_inliers))

"""
    TrackVariance(source, target)

σ̂² over separate pixel point pairs `source[i] => target[i]` (e.g. SfM-track
keypoints, independent of the matcher's correspondences), carried into the pool
frame by the raw pair's normalization: the RSS of the ORACLE model (the GT pose)
in the family's own residual, over the ORACLE DOF ([`oracle_residual_dof`](@ref),
`n·d_g`: nothing is fitted). E and F only. A minimum point count is the reader's
policy (it drops the pair); with no points σ̂² is NaN.
"""
struct TrackVariance <: PoolVariance
    source::Vector{SVector{2, Float64}}
    target::Vector{SVector{2, Float64}}
    function TrackVariance(source, target)
        length(source) == length(target) || throw(DimensionMismatch(
            "TrackVariance: $(length(source)) source points for $(length(target)) targets"))
        return new(SVector{2, Float64}.(source), SVector{2, Float64}.(target))
    end
end

# σ̂² = RSS/dof; NaN when there is no residual dof.
_rss_variance(rss::Float64, dof::Integer) = dof > 0 ? rss / dof : NaN

# The oracle F from the GT pose, in pixels and in the pool frame.
function _oracle_fundmat(raw::StereoRawPair, N::CspondNormalization)
    F_px = FundMat(EssentialMat(pair_R_gt(raw), pair_t_gt(raw)), pair_K1(raw), pair_K2(raw))
    return target_map(N) ∘ F_px ∘ inv(source_map(N))
end

# RSS of the oracle model over pool-frame points `src[i] => dst[i]` (optionally
# only where `mask`), in the SAME residual the family's stored distances are.
_oracle_rss(::FundMatProblem, raw::StereoRawPair, draft, src, dst, mask...) =
    sampson_rss(_oracle_fundmat(raw, pool_normalization(raw)), src, dst, mask...)
function _oracle_rss(::EssentialProblem, raw::StereoRawPair, draft, src, dst, mask...)
    Kc1, Kc2 = pool_frame_calibrations(draft)
    ep = EssentialProblem(CameraModel(Kc1), CameraModel(Kc2),
                          VisualGeometryCore.Csponds(Point2{Float64}.(src), Point2{Float64}.(dst)))
    r = residuals!(Vector{Float64}(undef, data_size(ep)), ep, EssentialMat(pair_R_gt(raw), pair_t_gt(raw)))
    return isempty(mask) ? sum(abs2, r) : sum(abs2, view(r, only(mask)))
end

function _pool_variance(v::InlierVariance, problem::Union{EssentialProblem, FundMatProblem}, raw::StereoRawPair, draft)
    mask = draft.gt_inliers
    n_in = count(mask)
    n_in < v.min_inliers && return NaN
    cs = draft.csponds
    return _rss_variance(_oracle_rss(problem, raw, draft, sources(cs), targets(cs), mask),
                         residual_dof(problem, n_in))
end

function _pool_variance(v::TrackVariance, problem::Union{EssentialProblem, FundMatProblem}, raw::StereoRawPair, draft)
    N = pool_normalization(raw)
    rss = _oracle_rss(problem, raw, draft, source_map(N).(v.source), target_map(N).(v.target))
    return _rss_variance(rss, oracle_residual_dof(problem, length(v.source)))
end

# H has no oracle model, so no track variance (a refusal, not a MethodError).
_pool_variance(::TrackVariance, ::HomographyProblem, ::StereoRawPair, draft) = throw(ArgumentError(
    "TrackVariance needs the oracle model of an E or F pool; a homography pool measures σ̂² " *
        "with InlierVariance"))

# H: an unconstrained 8-DOF DLT fit to the GT inliers in the pool frame, through
# VGC's one-step public DLT (it owns its conditioning: pool frame in, pool-frame
# H out). The refusal is this caller's policy: a degenerate GT-inlier set gives a
# design matrix whose null space is not isolated, and a σ̂² off that fit is
# meaningless, so NaN. The gate conditions the MASKED design matrix — the one
# solved — with the stack's one DLT threshold (`test_dlt_conditioning`'s
# default), read immediately after the call whose matrix the workspace holds.
function _pool_variance(v::InlierVariance, problem::HomographyProblem, raw::StereoRawPair, draft)
    mask = draft.gt_inliers
    n_in = count(mask)
    n_in < v.min_inliers && return NaN
    cs = draft.csponds
    ws = SvdWorkspace{Float64}(2 * length(cs), 9)
    H = homography_dlt(cs; mask, svd_ws = ws)
    (H === nothing || !test_dlt_conditioning(ws)) && return NaN
    rss = sampson_rss(H, sources(cs), targets(cs), mask)
    return _rss_variance(rss, residual_dof(problem, n_in))
end

# -----------------------------------------------------------------------------
# Hypotheses → the pool frame: pose errors and the frame change, per family
# -----------------------------------------------------------------------------
#
# A hypothesis is in the generation problem's MODEL frame: E calibrated (unit
# bearings), F pixels, H pixels or — for a mapped homography whose maps are
# declared into the calibrated chart (`GnomonicFrame`, the `K⁻¹` maps) —
# calibrated. Its pose error is read in that frame, and it moves into the pool
# frame by composition with the pool normalization's sides,
# `target_map(N) ∘ M ∘ inv(source_map(N))` of its pixel form.

_pool_model_class(::EssentialProblem) = FundMat{Float64}
_pool_model_class(::FundMatProblem) = FundMat{Float64}
_pool_model_class(::HomographyProblem) = HomographyMat{Float64}

_pose_errors(::EssentialProblem, E, raw) =
    optimistic_pose_error(SMatrix{3, 3, Float64, 9}(E), pair_R_gt(raw), pair_t_gt(raw))
_pose_errors(::FundMatProblem, F, raw) =
    optimistic_pose_error(EssentialMat(F, pair_K1(raw), pair_K2(raw)), pair_R_gt(raw), pair_t_gt(raw))
function _pose_errors(p::HomographyProblem, H, raw)
    Hd = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(H)))
    _homography_model_calibrated(p) && return optimistic_pose_error(Hd, pair_R_gt(raw), pair_t_gt(raw))
    return optimistic_pose_error(Hd,
                                 pair_K1(raw), pair_K2(raw), pair_R_gt(raw), pair_t_gt(raw))
end

_to_pool_frame(::EssentialProblem, E, raw, N::CspondNormalization) =
    target_map(N) ∘ FundMat(E, pair_K1(raw), pair_K2(raw)) ∘ inv(source_map(N))
_to_pool_frame(::FundMatProblem, F, raw, N::CspondNormalization) =
    target_map(N) ∘ FundMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(F))) ∘ inv(source_map(N))
function _to_pool_frame(p::HomographyProblem, H, raw, N::CspondNormalization)
    Hd = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(H)))
    _homography_model_calibrated(p) || return target_map(N) ∘ Hd ∘ inv(source_map(N))
    # the calibrated H back to pixels through the problem's own measurement maps
    return target_map(N) ∘ inv(p.dst_frame_map) ∘ Hd ∘ p.src_frame_map ∘ inv(source_map(N))
end

# The model frame of a homography problem: the data (pixel) frame when it
# carries no measurement maps; the calibrated chart when its maps are declared
# into `GnomonicFrame`. A mapped problem whose maps declare anything else names
# no frame a pose error can be read in, so it is refused.
_map_target(::HomographyMap{N, T, M, S, D}) where {N, T, M, S, D} = D
function _homography_model_calibrated(p::HomographyProblem)
    _has_measurement_maps(p) || return false
    (_map_target(p.src_frame_map) <: GnomonicFrame && _map_target(p.dst_frame_map) <: GnomonicFrame) &&
        return true
    throw(ArgumentError(
        "a pool homography problem is either unmapped (pixels) or mapped into the calibrated chart: " *
            "label its K⁻¹ maps into GnomonicFrame (got $(_map_target(p.src_frame_map)) => " *
            "$(_map_target(p.dst_frame_map)))"))
end

# -----------------------------------------------------------------------------
# The residual matrix of a record, on its own pool-frame problem
# -----------------------------------------------------------------------------
#
# E: the tangent Sampson of the metric `E = Kc₂ᵀ·F·Kc₁` on the record's
# `EssentialProblem(pair)` (`residuals!`), the residual a pool-frame LO scores
# refits with. F and H: the plane Sampson of the pool-frame model through VGC
# `sampson_distance` per point, NOT the problem's `residuals!`: it is the
# residual RSDR's `pixel_residuals` recomputes, and this route is kept for
# bitwise equality with pools already written. Stored as `Float32(|r|)`.

function _pool_residuals!(R::AbstractMatrix{Float32}, ::Union{FundMatProblem, HomographyProblem}, pair::PoolPair)
    cs = pair.csponds
    src = sources(cs); dst = targets(cs)
    @inbounds for j in eachindex(pair.models), i in eachindex(src)
        R[i, j] = Float32(sampson_distance(pair.models[j],
            SVector{2, Float64}(src[i]), SVector{2, Float64}(dst[i])))
    end
    return R
end
function _pool_residuals!(R::AbstractMatrix{Float32}, ::EssentialProblem, pair::PoolPair)
    ep = EssentialProblem(pair)
    Kc1, Kc2 = pool_frame_calibrations(pair)
    r = Vector{Float64}(undef, data_size(ep))
    for j in eachindex(pair.models)
        residuals!(r, ep, EssentialMat(pair.models[j], Kc1, Kc2))
        R[:, j] .= Float32.(abs.(r))
    end
    return R
end

# -----------------------------------------------------------------------------
# The pool-pair constructor
# -----------------------------------------------------------------------------

"""
    PoolPair(rng, problem, raw::StereoRawPair; target, max_trials, variance) -> (pair, residuals)
    PoolPair(hypotheses::Hypotheses, problem, raw::StereoRawPair; variance)   -> (pair, residuals)

THE pool pair of a raw image pair: the hypotheses of `problem`, a PIXEL-frame
`EssentialProblem`/`FundMatProblem`/`HomographyProblem` built on `raw`'s
correspondences with the pool's sampler, drawn by [`draw_hypotheses`](@ref)
(`target` accepted samples, at most `max_trials` draws), carried into the pool
frame once by `pool_normalization(raw)`. Returns the light record
`pair::PoolPair` (`NormalizedImageFrame`) and its residual matrix
`residuals::Matrix{Float32}` (`num_points × num_models`, pool units).

- **Models** are stored as pool-frame `FundMat{Float64}` (E and F; an E
  hypothesis is lifted through the pixel `K`s) or `HomographyMat{Float64}`.
  A mapped homography problem must declare its maps into the calibrated chart
  (`GnomonicFrame`); it is pulled back to pixels through them.
- **Pose errors** (`errors_r`, `errors_t`, degrees) are VGC's
  `optimistic_pose_error` of each hypothesis in its model frame against the GT
  pose of `raw`; a hypothesis with no decomposition (`errors_r ≥ 180`) is
  dropped with its sample.
- **The GT mask** is `pair_gt_inliers(raw)`.
- **σ̂²** (`gt_variance`) is measured as `variance::PoolVariance` states
  ([`InlierVariance`](@ref), [`TrackVariance`](@ref)) on the record's own
  pool-frame problem.
- **The residual matrix** is that problem's: the tangent Sampson of
  `EssentialProblem(pair)` for E, the plane Sampson of the pool-frame model for
  F and H, stored as `Float32(|r|)`.
- **`snn`** is `pair_snn(raw)`.

The second form assembles the record from hypotheses already drawn (in
`problem`'s model frame, `problem` built on `raw`).
"""
function PoolPair(rng::Random.AbstractRNG, problem::RelativeProblem, raw::StereoRawPair;
                  target::Integer, max_trials::Integer, variance::PoolVariance)
    hyps = draw_hypotheses(rng, problem; target, max_trials)
    return PoolPair(hyps, problem, raw; variance)
end

function PoolPair(hyps::Hypotheses, problem::RelativeProblem, raw::StereoRawPair; variance::PoolVariance)
    cs_px = pair_csponds(raw)
    data_size(problem) == length(cs_px) || throw(DimensionMismatch(
        "PoolPair: the problem holds $(data_size(problem)) correspondences, the raw pair $(length(cs_px))"))
    size(hyps.sample_indices, 2) == length(hyps.models) || throw(DimensionMismatch(
        "PoolPair: $(size(hyps.sample_indices, 2)) sample-index columns for $(length(hyps.models)) hypotheses"))
    N = pool_normalization(raw)
    cs_n = normalize(N, cs_px)
    cs_pool = VisualGeometryCore.csponds(sources(cs_n), targets(cs_n))

    errs = [_pose_errors(problem, m, raw) for m in hyps.models]
    keep = findall(e -> first(e) < 180.0, errs)       # drop pose-decomposition failures
    Mc = _pool_model_class(problem)
    models = Mc[_to_pool_frame(problem, hyps.models[j], raw, N) for j in keep]
    sidx = hyps.sample_indices[:, keep]
    errors_r = Float64[first(errs[j]) for j in keep]
    errors_t = Float64[last(errs[j]) for j in keep]

    K1 = SMatrix{3, 3, Float64, 9}(pair_K1(raw)); K2 = SMatrix{3, 3, Float64, 9}(pair_K2(raw))
    R_gt = SMatrix{3, 3, Float64, 9}(pair_R_gt(raw)); t_gt = SVector{3, Float64}(pair_t_gt(raw))
    gt_inliers = BitVector(pair_gt_inliers(raw))
    snn = Vector{Float32}(pair_snn(raw))
    # σ̂² is read off a DRAFT of the record (its own pool-frame csponds and Kc),
    # then the record is sealed with it.
    fields = (; csponds = cs_pool, K1, K2, R_gt, t_gt, models, sample_indices = sidx,
              errors_r, errors_t, gt_inliers, normalization = N, snn)
    draft = PoolPair{Mc}(; fields..., gt_variance = NaN)
    σ² = _pool_variance(variance, problem, raw, draft)
    pair = PoolPair{Mc}(; fields..., gt_variance = σ²)
    R = _pool_residuals!(Matrix{Float32}(undef, length(cs_pool), length(models)), problem, pair)
    return pair, R
end

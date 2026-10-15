# =============================================================================
# KA cross-warp NMS — packed imageAtomicMax splat + suppression.
# =============================================================================
#
# Backend-agnostic port of CrossWarpSplat.comp + CrossWarpNms.comp. Every
# feature packs its (quantized |intensity|, index) into a 64-bit value
#   packed = (clamp(|I|·65535/0.5, 0, 65535) << 32) | idx      (`_pool_pack`)
# and atomic-maxes it into a W×H splat image over its footprint
# (:mxm = the single centroid pixel; :ellipse = its Mahalanobis interior).
# A feature survives iff no pixel in its footprint holds a STRICTLY greater
# packed value. Ties break toward the higher index (larger low bits).
# =============================================================================

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
import Atomix

# Scale band of a feature with shape matrix S = [sxx sxy; sxy syy]: the
# geometric-mean scale √det S (σ√t for a tilted warp) binned in `w`-octave
# bands; `w == 0` puts everything in band 0, a degenerate shape gets the
# sentinel. See `_band_pool!` below for the why.
const _NO_BAND = typemin(Int32)
@inline function _scale_band(sxx::Float32, sxy::Float32, syy::Float32, w::Float32)
    det = sxx * syy - sxy * sxy
    det > 0.0f0 || return _NO_BAND
    return w > 0.0f0 ? Int32(floor(0.5f0 * log2(det) / w)) : Int32(0)
end

# The splat value: (quantized |intensity| << 32) | feature index. The C shader
# packs 32 bits ((q<<16)|idx16), capping a pool at 65535 features: at 24 MP the
# IMAS-25 pool is ~160k, and the cap silently dropped an arbitrary subset. For
# pools under that cap the winners are bit-identical (same quantizer, ties
# still break toward the higher index).
const _POOL_INT_SCALE = 65535.0f0 / 0.5f0
@inline function _pool_pack(I::Float32, idx::Integer)
    q = UInt64(clamp(round(Int, abs(I) * _POOL_INT_SCALE), 0, 65535))
    return (q << 32) | (UInt64(idx) & 0x00000000ffffffff)
end

# Argument checks shared by the NMS entry points (`asift_detect!`,
# `cross_warp_nms`). Each guards a
# kernel that would otherwise throw InexactError mid-launch: a tiny band width
# overflows `_scale_band`'s Int32, a non-finite cutoff hits `ceil(Int, …)`, and a
# negative window silently suppresses nothing.
function _check_nms_params(
        window_half::Integer, cutoff::Real, band_octaves::Real,
        band_reach::Integer
    )
    window_half >= 0 || throw(ArgumentError("window_half must be ≥ 0, got $window_half"))
    isfinite(cutoff) && cutoff > 0 ||
        throw(ArgumentError("cutoff must be finite and > 0, got $cutoff"))
    band_octaves == 0 || (isfinite(band_octaves) && band_octaves >= 1.0e-3) ||
        throw(ArgumentError("band_octaves must be 0 (off) or ≥ 1e-3, got $band_octaves"))
    band_reach >= 0 || throw(ArgumentError("band_reach must be ≥ 0, got $band_reach"))
    return nothing
end

# ----- the NMS kernels, over a (9,·) feature pool ---------------------------

# Cross-warp NMS over a feature pool, one SCALE BAND at a time. `band[i]` is
# the feature's band from `_band_pool!` (the sentinel `_NO_BAND` for invalid
# features, so no validity test is needed here); a splat sweep writes the
# features with `lo ≤ band ≤ hi`, a check sweep decides only the features with
# `band == b`. With every valid feature in band 0 (banding off) a single sweep
# with lo ≤ 0 ≤ hi is exactly the original whole-pool rule.
@kernel function _splat_mxm_pool!(
        splat, @Const(feat), @Const(band), n::Int32,
        lo::Int32, hi::Int32
    )
    i = @index(Global)
    @inbounds if i <= n && lo <= band[i] <= hi
        W = size(splat, 1); H = size(splat, 2)
        cx = clamp(round(Int, feat[1, i]) + 1, 1, W)
        cy = clamp(round(Int, feat[2, i]) + 1, 1, H)
        Atomix.@atomic max(splat[cx, cy], _pool_pack(feat[4, i], i))
    end
end

@kernel function _nms_mxm_pool!(
        keep, @Const(splat), @Const(feat), @Const(band),
        n::Int32, wh::Int32, b::Int32
    )
    i = @index(Global)
    @inbounds if i <= n && band[i] == b
        W = size(splat, 1); H = size(splat, 2)
        me = _pool_pack(feat[4, i], i)
        cx = clamp(round(Int, feat[1, i]) + 1, 1, W)
        cy = clamp(round(Int, feat[2, i]) + 1, 1, H)
        survive = true
        for dy in -wh:wh, dx in -wh:wh
            x = cx + dx; y = cy + dy
            (1 <= x <= W && 1 <= y <= H) || continue
            if splat[x, y] > me
                survive = false
            end
        end
        keep[i] = survive ? Int32(1) : Int32(0)
    end
end

@kernel function _splat_ell_pool!(
        splat, @Const(feat), @Const(band), n::Int32,
        K::Float32, lo::Int32, hi::Int32
    )
    i = @index(Global)
    @inbounds if i <= n && lo <= band[i] <= hi
        W = size(splat, 1); H = size(splat, 2)
        sxx = feat[5, i]; sxy = feat[6, i]; syy = feat[7, i]
        det = sxx * syy - sxy * sxy
        if det > 0.0f0
            si00 = syy / det; si01 = -sxy / det; si11 = sxx / det
            cx = round(Int, feat[1, i]) + 1; cy = round(Int, feat[2, i]) + 1
            rx = min(ceil(Int, K * sqrt(max(sxx, 1.0f-6))), 64)
            ry = min(ceil(Int, K * sqrt(max(syy, 1.0f-6))), 64)
            pk = _pool_pack(feat[4, i], i); K2 = K * K
            for dy in -ry:ry, dx in -rx:rx
                md2 = si00 * dx * dx + 2.0f0 * si01 * dx * dy + si11 * dy * dy
                md2 <= K2 || continue
                x = cx + dx; y = cy + dy
                (1 <= x <= W && 1 <= y <= H) || continue
                Atomix.@atomic max(splat[x, y], pk)
            end
        end
    end
end

@kernel function _nms_ell_pool!(
        keep, @Const(splat), @Const(feat), @Const(band),
        n::Int32, K::Float32, b::Int32
    )
    i = @index(Global)
    @inbounds if i <= n && band[i] == b
        keepv = Int32(0)
        W = size(splat, 1); H = size(splat, 2)
        sxx = feat[5, i]; sxy = feat[6, i]; syy = feat[7, i]
        det = sxx * syy - sxy * sxy
        if det > 0.0f0
            si00 = syy / det; si01 = -sxy / det; si11 = sxx / det
            cx = round(Int, feat[1, i]) + 1; cy = round(Int, feat[2, i]) + 1
            rx = min(ceil(Int, K * sqrt(max(sxx, 1.0f-6))), 64)
            ry = min(ceil(Int, K * sqrt(max(syy, 1.0f-6))), 64)
            me = _pool_pack(feat[4, i], i); K2 = K * K
            survive = true
            for dy in -ry:ry, dx in -rx:rx
                md2 = si00 * dx * dx + 2.0f0 * si01 * dx * dy + si11 * dy * dy
                md2 <= K2 || continue
                x = cx + dx; y = cy + dy
                (1 <= x <= W && 1 <= y <= H) || continue
                if splat[x, y] > me
                    survive = false
                end
            end
            keepv = survive ? Int32(1) : Int32(0)
        end
        keep[i] = keepv
    end
end

# Scale band of every pooled feature. The whole-pool rule lets a feature be
# suppressed by ANY stronger overlapping feature, so a board-sized extremum or
# a tilted warp's merged-blob streak (both legitimate DoG extrema, both far
# stronger) erased the 1–3 px blobs it was built from — on a panel of six
# 70×100 px boards the default kept 171 of ~500 true blobs. Banding by the
# geometric-mean scale √det S (σ√t for a tilted warp, so a tilted copy of a
# blob lands in the blob's own band and is still deduplicated) restricts
# suppression to comparable scales: band = ⌊log2(√det S) / w⌋ for a width of
# `w` octaves, and a feature is checked against bands b−reach … b+reach. With
# w = 1, reach = 1 the admitted scale ratio is < 4; measured 634 survivors on
# that panel, every board populated. `w == 0` puts every valid feature in
# band 0 — the original rule, bit for bit.
@kernel function _band_pool!(band, @Const(feat), n::Int32, w::Float32)
    i = @index(Global)
    @inbounds if i <= n
        band[i] = feat[9, i] == 1.0f0 ?
            _scale_band(feat[5, i], feat[6, i], feat[7, i], w) : _NO_BAND
    end
end


# The NMS kernels, instantiated for one backend. `AsiftPlan` holds them in its
# `kern` NamedTuple; `cross_warp_nms` builds its own.
_nms_kernels(dev) = (
    band = _band_pool!(dev, (256,)),
    smx = _splat_mxm_pool!(dev, (256,)), nmx = _nms_mxm_pool!(dev, (256,)),
    sel = _splat_ell_pool!(dev, (256,)), nel = _nms_ell_pool!(dev, (256,)),
)

# Cross-warp NMS over the first `n` features of a (9,·) pool (rows x, y, σ, I,
# s_xx, s_xy, s_yy, warp, valid): sets `keep[i] = 1` for the survivors, 0 for
# every other feature. Scale bands first, then one splat+check sweep per
# populated band, reusing the single splat image: sweep b splats bands b±reach
# and decides band b. A feature is decided exactly once (its own band); the keep
# flags of the other features are untouched, hence the fill. Banding off ⇒
# every valid feature is in band 0 ⇒ one sweep ⇒ the original rule. `bandh` is
# host scratch of length ≥ n. The one implementation behind `asift_detect!`
# (the plan's device pool) and `cross_warp_nms` (a pool handed in).
function _pool_nms!(
        dev, K, keep, splat, feat, band, bandh, n::Int, nms::Symbol,
        window_half, cutoff, band_octaves, band_reach
    )
    K.band(band, feat, Int32(n), Float32(band_octaves); ndrange = n)
    KA.synchronize(dev)
    copyto!(bandh, 1, band, 1, n)
    bands = sort!(unique(filter(!=(_NO_BAND), view(bandh, 1:n))))
    fill!(view(keep, 1:n), Int32(0))
    reach = Int32(band_reach)
    for b in bands
        lo = b - reach; hi = b + reach
        fill!(splat, UInt64(0))
        if nms === :mxm
            K.smx(splat, feat, band, Int32(n), lo, hi; ndrange = n)
            K.nmx(keep, splat, feat, band, Int32(n), Int32(window_half), b; ndrange = n)
        else  # :ellipse
            K.sel(splat, feat, band, Int32(n), Float32(cutoff), lo, hi; ndrange = n)
            K.nel(keep, splat, feat, band, Int32(n), Float32(cutoff), b; ndrange = n)
        end
    end
    KA.synchronize(dev)
    return nothing
end

# The features `i ≤ n` of a host (9,·) pool copy `fh` whose `flags[i] == on`.
function _pool_features(fh::AbstractMatrix{Float32}, n::Int, flags, on)
    out = ImasFeatureKA[]
    sizehint!(out, n)
    for i in 1:n
        flags[i] == on || continue
        push!(
            out, ImasFeatureKA(
                fh[1, i], fh[2, i], fh[3, i], fh[4, i],
                fh[5, i], fh[6, i], fh[7, i],
                Int32(fh[8, i])
            )
        )
    end
    return out
end

# The covariances (a host (6,·) copy `ch`) of the same features.
function _pool_covs(ch::AbstractMatrix{Float32}, n::Int, flags, on)
    out = _FrameCov[]
    sizehint!(out, n)
    for i in 1:n
        flags[i] == on || continue
        push!(out, _keypoint_cov(view(ch, :, i)))
    end
    return out
end

"""
    cross_warp_nms(feats, W, H; device, mode=:mxm, window_half=3, cutoff=3.0,
                   band_octaves=0.0, band_reach=1) -> Vector{ImasFeatureKA}

Cross-warp NMS over an aggregated feature pool (input-frame). `mode` is `:mxm`
(centroid window `window_half`) or `:ellipse` (Mahalanobis footprint,
cutoff `cutoff`). `band_octaves > 0` restricts suppression to features of
comparable scale — band `⌊log2(√det S) / band_octaves⌋ ± band_reach`. It runs
the kernels `asift_detect!` runs on its device pool, so on the pool
`asift_detect(img; nms = :none, raw = true)` returns it keeps the features
`asift_detect(img; nms = mode, raw = true)` keeps, given the same detection
keywords — up to exact ties in the quantized response, which break toward the
later pool index (and pool order varies between calls on a multithreaded or
GPU backend). Returns the survivors.
"""
function cross_warp_nms(
        feats::AbstractVector{ImasFeatureKA}, W::Integer, H::Integer;
        device::KA.Backend = default_device(), mode::Symbol = :mxm,
        window_half::Integer = 3, cutoff::Real = 3.0,
        band_octaves::Real = 0.0, band_reach::Integer = 1
    )
    mode in (:mxm, :ellipse) || throw(ArgumentError("mode must be :mxm or :ellipse"))
    _check_nms_params(window_half, cutoff, band_octaves, band_reach)
    n = length(feats)
    n == 0 && return ImasFeatureKA[]
    fh = Matrix{Float32}(undef, 9, n)   # the pool layout, every feature valid
    for (i, f) in enumerate(feats)
        fh[:, i] .= (f.x, f.y, f.σ, f.intensity, f.s_xx, f.s_xy, f.s_yy, Float32(f.warp_idx), 1.0f0)
    end
    feat = KA.allocate(device, Float32, 9, n)
    copyto!(feat, fh)
    keep = KA.allocate(device, Int32, n)
    _pool_nms!(
        device, _scheduled(_nms_kernels, device), keep, KA.allocate(device, UInt64, W, H), feat,
        KA.allocate(device, Int32, n), Vector{Int32}(undef, n), n, mode,
        window_half, cutoff, band_octaves, band_reach
    )
    return _pool_features(fh, n, Array(keep), Int32(1))
end

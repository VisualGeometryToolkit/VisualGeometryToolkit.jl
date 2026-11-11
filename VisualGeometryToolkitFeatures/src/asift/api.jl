# =============================================================================
# Public API for the KernelAbstractions ASIFT (IMAS-25) detector.
# =============================================================================
#
# `asift_detect` is the first-class entry point: image in, VGF
# `Detection{EllipticFrame}` out, running the fused batched pipeline in
# `driver.jl` on any KA device (`KA.CPU()` by default, `CUDABackend()`
# etc. for GPU). Plans are cached per (size, device, tilt-schedule) — the
# first call at a given size pays the buffer allocation, subsequent calls
# are allocation-free on device.
#
# The extremum test breaks exact DoG ties (strict against half the 26
# neighbours) rather than rejecting both pixels (2026-09-22): only features at
# exact ties moved.
# =============================================================================

# ----- input coercion: VGF images are H×W [row,col]; the pipeline runs on
# (W,H)[x,y] Float32 in [0,1] (x-fastest layout). 8-bit inputs
# (Gray{N0f8}, UInt8) are forwarded as raw H×W bytes — asift_detect! uploads
# them 4× smaller and transposes/scales on the device, which is ~free vs the
# ~18 ms host-side permutedims at 2 MP. Other eltypes convert on the host. --

_asift_input(img::AbstractMatrix{UInt8}) = gray_u8(img)             # H×W bytes
_asift_input(img::AbstractMatrix{Gray{N0f8}}) = gray_u8(img)        # its bytes, copied
# Float colorants keep their precision. Quantising them to `N0f8` here would
# discard bits the adaptation stage uses for sub-pixel fitting, and it happened
# silently: `Gray{Float32}` escaped only because `{<:Gray}` below is a more
# specific method, so `RGB{Float32}` lost 8 bits while `Gray{Float32}` did not.
_asift_input(img::AbstractMatrix{<:Colorant{<:AbstractFloat}}) =
    _asift_input(Gray{Float32}.(img))
# Fixed-point colorants (`RGB{N0f8}` from any JPEG loader) still take the
# UInt8 upload path via the `Gray{N0f8}` method above.
_asift_input(img::AbstractMatrix{<:Colorant}) = _asift_input(Gray{N0f8}.(img))
_asift_input(img::AbstractMatrix{Float32}) = permutedims(img, (2, 1))   # (W,H) xy
_asift_input(img::AbstractMatrix{<:Gray}) =
    permutedims(Float32.(channelview(img)), (2, 1))
_asift_input(img::AbstractMatrix{<:Real}) = permutedims(Float32.(img), (2, 1))

# ----- plan cache ------------------------------------------------------------

# Keyed by everything a plan is built from, compared by VALUE: the device
# instance (not its type — two backends of one type with different settings must
# not share a plan), the tilt schedule, upsampling and the seed blur.
const _PlanKey = Tuple{Int, Int, KA.Backend, Tuple, Bool, Float64}
const _ASIFT_PLAN_CACHE = Dict{_PlanKey, Any}()
const _ASIFT_PLAN_ORDER = _PlanKey[]      # least-recently-used first
# Serialises every use of the cache. A cached plan is ONE set of mutable device
# buffers, so two tasks detecting at the same resolution would write the same
# pyramid and pool, and eviction under a concurrent caller is a use-after-free.
# Reentrant, so the public entry points can nest.
const _ASIFT_LOCK = ReentrantLock()

# How many plans may stay resident. ONE by default: a plan owns the whole
# scale-space pyramid for its resolution — 11.09 GiB measured for a 1600x1200
# frame with the 25-tilt schedule and upsampling — so an unbounded cache turns
# "this corpus has N distinct resolutions" into "this run needs N x that".
# A 13-resolution training corpus reached a 173 GiB pool that way.
#
# Retaining one plan costs nothing when frames are processed GROUPED BY
# RESOLUTION (every frame in a group hits the same plan); raise it only if
# resolutions must interleave.
const _MAX_CACHED_PLANS = Ref(1)

"""
    set_max_cached_plans!(n) -> Int

Set how many [`AsiftPlan`](@ref)s stay resident (default 1). Evicted plans have
their device buffers released immediately via `KernelAbstractions.unsafe_free!`.
"""
function set_max_cached_plans!(n::Integer)
    n >= 1 || throw(ArgumentError("need at least one resident plan, got $n"))
    return lock(_ASIFT_LOCK) do
        _MAX_CACHED_PLANS[] = Int(n)
        _evict_plans!(_MAX_CACHED_PLANS[])
        _MAX_CACHED_PLANS[]
    end
end

# Drop least-recently-used plans until at most `keep` remain.
function _evict_plans!(keep::Int)
    while length(_ASIFT_PLAN_ORDER) > keep
        old = popfirst!(_ASIFT_PLAN_ORDER)
        plan = pop!(_ASIFT_PLAN_CACHE, old, nothing)
        plan === nothing || _free_plan!(plan)
    end
    return nothing
end

# Call with `_ASIFT_LOCK` held (the public entry points take it): the cache is a
# global `Dict`, and eviction under a concurrent caller frees buffers in use.
function _asift_plan(
        W::Int, H::Int, device::KA.Backend, tilts, upsample::Bool,
        seed_sigma::Real = 1.6
    )
    key = (W, H, device, Tuple(tilts), upsample, Float64(seed_sigma))
    cached = get(_ASIFT_PLAN_CACHE, key, nothing)
    if cached !== nothing
        i = findfirst(==(key), _ASIFT_PLAN_ORDER)   # touch: mark most-recent
        if i !== nothing
            deleteat!(_ASIFT_PLAN_ORDER, i)
            push!(_ASIFT_PLAN_ORDER, key)
        end
        return cached::AsiftPlan
    end
    _evict_plans!(max(_MAX_CACHED_PLANS[] - 1, 0))   # make room for the new one
    plan = AsiftPlan(W, H; device, tilts, upsample, seed_sigma)
    _ASIFT_PLAN_CACHE[key] = plan
    push!(_ASIFT_PLAN_ORDER, key)
    return plan
end

# Release a plan's device buffers EAGERLY via KernelAbstractions'
# `unsafe_free!` (device-agnostic: a no-op on CPU and on any backend that does
# not implement it).
#
# Waiting for the GC is not good enough here. A plan's `gauss`/`dogs` pyramids
# are multi-GiB on device but only a few hundred bytes of host wrapper, so they
# generate almost no GC pressure — the collector has no reason to run while the
# device fills. Measured: two plans for 1.9 MP and 3.1 MP frames (upsample, the
# 25-tilt schedule) hold ~17 GiB of a 24 GiB card between them.
_free_plan_buffers!(x::AbstractVector{<:AbstractArray}) =
    (foreach(_free_plan_buffers!, x); nothing)
_free_plan_buffers!(x::AbstractArray) = (KA.unsafe_free!(x); nothing)
_free_plan_buffers!(::Any) = nothing   # scalars, and `geo`'s NamedTuples

function _free_plan!(p::AsiftPlan)
    KA.synchronize(p.dev)   # a failed detect can leave kernels queued on these buffers
    for f in fieldnames(typeof(p))
        _free_plan_buffers!(getfield(p, f))
    end
    return nothing
end

"""
    clear_asift_plans!()

Drop all cached [`AsiftPlan`](@ref)s, releasing their device buffers
immediately rather than leaving them to a finalizer.

The cache is keyed by `(W, H, device, tilts, upsample, seed_sigma)` and holds at
most [`set_max_cached_plans!`](@ref) plans (default 1), evicting the least
recently used. Call this to release the resident plans outright — e.g. between
sequences in a mixed-resolution corpus, or before handing the device to another
workload. The freed memory is returned to the device (on CUDA, CUDA.jl's pool is
reclaimed), so an allocator outside the backend's pool, such as ONNX Runtime's,
can use it: a [`LoMaDescriptor`](@ref) or [`LoMaMatcher`](@ref) on ONNX
Runtime's CUDA provider calls this before it runs.
"""
function clear_asift_plans!()
    lock(_ASIFT_LOCK) do
        devices = unique(p.dev for p in values(_ASIFT_PLAN_CACHE))
        foreach(_free_plan!, values(_ASIFT_PLAN_CACHE))
        empty!(_ASIFT_PLAN_CACHE)
        empty!(_ASIFT_PLAN_ORDER)
        # `unsafe_free!` releases a buffer only when no array derived from it (a reshape made during a detection) is
        # still alive: those wrappers are garbage now, but may have been promoted past a young collection, so a full
        # one drops them (measured: a panorama frame's 10.6 GiB plan stayed in use after the frees until it ran)
        isempty(devices) || GC.gc()
        foreach(reclaim_device!, devices)
    end
    return nothing
end

# ----- Detection conversion --------------------------------------------------

"""
    asift_detections(feats::Vector{ImasFeatureKA}) -> Vector{Detection}
    asift_detections(feats, cov) -> Vector{Detection}

Convert raw ASIFT features (0-indexed opencv pixel coords, input frame) into
`Detection{EllipticFrame}`s in COLMAP convention (+0.5 on both axes, so the
top-left pixel's centre is (0.5, 0.5); see [`asift_detect`](@ref)). The
metadata is `(response, edge_score, laplacian_scale_score, warp_idx)`:
`laplacian_scale_score = sign(response)` (+1 dark, −1 light — the DoG sign
convention `polarity` gates on), which is what [`is_dark`](@ref) and
[`is_light`](@ref) read; `edge_score` is a placeholder (always 0). Given the
features' covariances (`asift_detect!(…; uncertainty = FeatureCovariance())`),
each frame is `Uncertain(frame, Σ̃)` (see [`FeatureCovariance`](@ref)); the
+0.5 shift leaves a covariance unchanged.
"""
asift_detections(feats::Vector{ImasFeatureKA}) = map(f -> Detection(_asift_frame(f), _asift_meta(f)), feats)

function asift_detections(feats::Vector{ImasFeatureKA}, covs::Vector{<:CovMat{3}})
    length(feats) == length(covs) ||
        throw(DimensionMismatch("$(length(feats)) features, $(length(covs)) covariances"))
    return map((f, c) -> Detection(Uncertain(_asift_frame(f), c), _asift_meta(f)), feats, covs)
end

function _asift_frame(f::ImasFeatureKA)
    center = Point2(Float64(f.x) + 0.5, Float64(f.y) + 0.5)
    S = SMatrix{2, 2, Float64}(
        Float64(f.s_xx), Float64(f.s_xy),
        Float64(f.s_xy), Float64(f.s_yy)
    )
    return EllipticFrame(center, S)
end

_asift_meta(f::ImasFeatureKA) = (
    response = Float64(f.intensity),
    edge_score = 0.0,
    # The polarity helpers read this field's sign. It was a constant 0.0,
    # so is_dark/is_light were false for every ASIFT detection.
    laplacian_scale_score = Float64(sign(f.intensity)),
    warp_idx = Int(f.warp_idx),
)

# ----- first-class entry -------------------------------------------------------

"""
    asift_detect(img; device=default_device(), nms=:mxm, window_half=3,
                 cutoff=3.0, intensity_thr=0.04, edge_thr=10.0,
                 polarity=:none, band_octaves=0.0, band_reach=1, boundary_K=3.0,
                 tilts=:imas25, mask=nothing, upsample=true, seed_sigma=1.6,
                 raw=false, uncertainty=nothing)

Run the IMAS-25 ASIFT detector on `img` (H×W; `Gray`, `UInt8`, or real-valued
in [0,1]) and return `Vector{Detection{EllipticFrame}}` in COLMAP coords.

**Coordinates.** A detection's `center` is `(x, y)`, with x along the image
columns and y along the rows, in COLMAP pixel-centre coordinates: the top-left
pixel's centre is (0.5, 0.5), so a blob centred on the pixel `img[r, c]` is
returned at `(c − 0.5, r − 0.5)`. Measured on isotropic Gaussian blobs on the
CPU (pinned by the test suite):
- a blob centred on a pixel centre comes back exact (within 0.005 px);
- between pixel centres, the quadratic sub-pixel refinement pulls the centre
  toward the nearest pixel centre, by at most 0.03 px at σ = 4 (−0.007, −0.015,
  −0.021, −0.027 px at fractions 0.1–0.4, mirrored above 0.5) and 0.06 px for
  σ from 2 to 8;
- an exact half-pixel centre, where the DoG ties between the two pixels, is
  refined like any other: the extremum test breaks the tie toward the first
  pixel, so it carries that side's bias (−0.031 px at σ = 4). A test strict
  against all 26 neighbours would reject both pixels;
- every detection is refined: extrema are searched on the interior scale slices
  only (each with a slice above and below), and an extremum whose refinement
  fails is dropped (`plan.refine_dropped` counts them), never returned on the
  octave's pixel grid.
With `raw = true` the features are in 0-indexed OpenCV coordinates (top-left
pixel centre (0, 0)), 0.5 less on both axes.

`nms` selects the cross-warp non-max suppression: `:mxm` (centroid window,
default), `:ellipse` (Mahalanobis
footprint), or `:none` (full per-warp feature pool). `polarity` gates by DoG
sign (`:dark` → response > 0, `:light` → response < 0, `:none` → both) BEFORE
the cross-warp NMS, so a blob of the other sign never suppresses a kept one.
`band_octaves > 0` makes the NMS scale-banded: a feature can only be suppressed
by features whose geometric-mean scale band `⌊log2(√det S) / band_octaves⌋` is
within `band_reach` of its own, so board-sized extrema and tilted-warp streaks
stop erasing the small blobs inside them (see `asift_detect!`).
`raw=true` returns the `ImasFeatureKA` vector instead of Detections.
`device`, `tilts`, `upsample` and `seed_sigma` pick the plan; the others but
`raw` and `mask` go to [`asift_detect!`](@ref), which refuses a keyword it does
not take.

**Tilts.** `tilts` is a [`tilt_schedule`](@ref) name or a vector of `(t, φ)`
pairs: `:imas25` (the default) the 25 tilts of IMAS-25, `:untilted` the
identity warp alone (plain DoG-SIFT), whose keypoints localize best (about
0.3 px rms on real panorama frames, against about 0.95 px for a tilted warp's):
the schedule for views that differ by a rotation, such as a panorama's. Under
`:untilted` every detection has `warp_idx == 1`, and they are exactly the
identity warp's detections under `:imas25`.

**Detection mask.** `mask`, an H×W `AbstractMatrix{Bool}` (`true` where
detection is not allowed: clipped or saturated pixels, a vignetted rim, an
overlay), drops every detection whose centre lies within three of its scales
(`characteristic_scale`, the DoG's support) of a masked pixel, so no keypoint
is read off masked pixels. It applies after the cross-warp NMS, to `raw`
features and covariances alike. The detector on a [`StabilizedImage`](@ref)
masks its saturated pixels by this rule.

**Determinism.** The result is a function of the image and the keywords: the
same keypoints, in the same order, run to run, on one thread or many, launched
statically or from inside a threaded region. The extrema pool fills in the order
the threads reach it, so before the cross-warp NMS (which breaks ties in its
quantized response by pool order) it is sorted into a canonical order: by
position, scale and response, then every other field (the covariance's too). The
detections come back in that order.

**Keypoint uncertainty.** `uncertainty = FeatureCovariance()` returns each
detection's frame as `Uncertain(frame, Σ̃)`, `Σ̃ = cov(d)` a 3×3 `CovMat` over
`(x, y, log σ)` for unit iid pixel noise in the detector's intensity units — see
[`FeatureCovariance`](@ref) for the model, the parameterization and the units.
With `raw = true` the result is then `(; features, cov)`. The default `nothing`
runs the detector without it, the kernels compiled exactly as before.

The plan for `(size, device, tilts, upsample, seed_sigma)` is cached (see
[`set_max_cached_plans!`](@ref)) and every call holds a process-wide lock, so
concurrent calls are safe but do not run in parallel.
On a `KA.CPU()` device every kernel launch is statically scheduled over the
threads, or dynamically when the call runs inside a threaded region (e.g. a
`Threads.@threads` loop over frames); the features are the same either way.
"""
function asift_detect(
        img::AbstractMatrix;
        device::KA.Backend = default_device(),
        nms::Symbol = :mxm, polarity::Symbol = :none,
        tilts = :imas25, raw::Bool = false,
        upsample::Bool = true, seed_sigma::Real = 1.6,
        uncertainty::Union{Nothing, FeatureCovariance} = nothing,
        mask::Union{Nothing, AbstractMatrix{Bool}} = nothing, kwargs...
    )
    _check_mask(mask, size(img))
    inp = _asift_input(img)
    W, H = inp isa Matrix{UInt8} ? (size(inp, 2), size(inp, 1)) : size(inp)
    ts = tilt_schedule(tilts)
    feats = lock(_ASIFT_LOCK) do
        plan = _asift_plan(W, H, device, ts, upsample, seed_sigma)
        asift_detect!(plan, inp; nms, polarity, uncertainty, kwargs...)
    end
    mask === nothing || (feats = _unmasked(feats, mask))
    return raw ? feats : _asift_detections(feats)
end

_asift_detections(feats::Vector{ImasFeatureKA}) = asift_detections(feats)
_asift_detections(r::NamedTuple{(:features, :cov)}) = asift_detections(r.features, r.cov)

# ----- the detection mask ------------------------------------------------------

# A detection whose centre lies within this many of its scales (√det of its shape: the DoG's support) of a masked
# pixel is dropped: its DoG read the masked pixels.
const _MASK_REACH = 3

_check_mask(::Nothing, sz) = nothing
_check_mask(mask::AbstractMatrix{Bool}, sz) = size(mask) == sz ? nothing : throw(DimensionMismatch(
    "asift_detect: the image is $(join(sz, '×')), the detection mask $(join(size(mask), '×'))"))

# The features (and their covariances) clear of `mask` (H×W, `true` where detection is not allowed): the distance
# from the pixel holding a feature's centre to the nearest masked pixel exceeds `_MASK_REACH` of its scales.
function _unmasked(feats::Vector{ImasFeatureKA}, mask::AbstractMatrix{Bool})
    return feats[_clear_of_mask(feats, mask)]
end
function _unmasked(r::NamedTuple{(:features, :cov)}, mask::AbstractMatrix{Bool})
    keep = _clear_of_mask(r.features, mask)
    return (features = r.features[keep], cov = r.cov[keep])
end

function _clear_of_mask(feats::Vector{ImasFeatureKA}, mask::AbstractMatrix{Bool})
    any(mask) || return trues(length(feats))
    H, W = size(mask)
    # the masked pixels' summed-area table: a disc's bounding box holding none is clear at once, in O(1); a box
    # holding some is read chord by chord (the discs near the mask only, not the frame's distance transform)
    S = zeros(Int32, H + 1, W + 1)
    @inbounds for j in 1:W, i in 1:H
        S[i + 1, j + 1] = S[i, j + 1] + S[i + 1, j] - S[i, j] + Int32(mask[i, j])
    end
    return map(feats) do f
        fr = _asift_frame(f)
        c = centroid(fr)
        i, j = clamp(floor(Int, c[2]) + 1, 1, H), clamp(floor(Int, c[1]) + 1, 1, W)
        _clear_at(mask, S, i, j, _MASK_REACH * characteristic_scale(fr))
    end
end

# Whether pixel (i, j)'s Euclidean distance √(Δi² + Δj²) to every masked pixel exceeds `reach`: the rule of the
# distance transform, read row by row over the disc (each row's chord counted in the table, O(1)).
function _clear_at(mask, S, i, j, reach)
    H, W = size(mask)
    within(di, dj) = !(sqrt(Float64(di^2 + dj^2)) > reach)
    r = isfinite(reach) ? floor(Int, max(reach, 0.0)) : max(H, W)
    i0, i1, j0, j1 = max(i - r, 1), min(i + r, H), max(j - r, 1), min(j + r, W)
    _masked_in(S, i0, i1, j0, j1) == 0 && return true
    for k in i0:i1
        di = k - i
        within(di, 0) || continue
        # the chord's half-width: the largest w with (di, w) within the disc, the float rule's own edge
        w = isfinite(reach) ? min(floor(Int, sqrt(max(Float64(reach)^2 - di^2, 0.0))), r) : r
        while w < r && within(di, w + 1); w += 1; end
        while w > 0 && !within(di, w); w -= 1; end
        _masked_in(S, k, k, max(j - w, 1), min(j + w, W)) == 0 || return false
    end
    return true
end

# The masked pixels in rows i0:i1, columns j0:j1, from the summed-area table `S` ((H + 1) × (W + 1)).
@inline _masked_in(S, i0, i1, j0, j1) = @inbounds S[i1 + 1, j1 + 1] - S[i0, j1 + 1] - S[i1 + 1, j0] + S[i0, j0]

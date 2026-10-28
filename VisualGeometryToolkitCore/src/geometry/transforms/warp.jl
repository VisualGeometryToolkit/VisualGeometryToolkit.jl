# =============================================================================
# Image warp — adapter, fusion, and the KernelAbstractions resampler
# =============================================================================
#
# `image_warp(math, output_rect)` wraps a math-space `Transformation` into a
# `BoundedMap` callable from `(row, col)` Julia array indices.  The wrapping
# composition is FIXED:
#
#   _RC_SWAP ∘ _OUT_BRIDGE ∘ math ∘ _IN_BRIDGE ∘ _RC_SWAP
#
# That is:
#   1.  `_RC_SWAP`: convert `(row, col)` ↔ `(x, y)`.
#   2.  `_IN_BRIDGE = EuclideanMap(PixelFrame{:julia}, PixelFrame{:colmap})`:
#       convert Julia array index → COLMAP math coord (pixel center at
#       half-integer) — the typed frame map, translation (-0.5, -0.5).
#   3.  `math`: the user's math-space transform — assumed to operate in
#       COLMAP convention (the package-wide invariant).  Camera intrinsics,
#       `canonical_map`, `logpolar_map`, K-derived homographies, etc. all
#       agree on COLMAP.
#   4.  `_OUT_BRIDGE = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})`:
#       convert COLMAP source-coord → Julia source-index for the sampler —
#       translation (+0.5, +0.5).
#   5.  `_RC_SWAP`: convert `(x, y)` ↔ `(row, col)`.
#
# The chain is then FUSED (`fuse`): every run of matrix-representable links
# (Translation / LinearMap / AffineMap / EuclideanMap / RCSwap /
# HomographyMap) collapses into a single `AffineMap` or `HomographyMap`, and
# an affine link left of a `PerspectiveMap` is pulled through the division
# into the homography.  The three canonical shapes after fusion:
#
#   affine math      →  one AffineMap                      (1 mat-vec per pixel)
#   projective math  →  PerspectiveMap ∘ one HomographyMap (1 mat-vec + division)
#   nonlinear math   →  AffineMap ∘ core ∘ AffineMap       (e.g. log-polar, lens)
#
# Perspective division is auto-injected for projective chains
# (`is_projective`) that don't already divide.
#
# There are no kwargs and no convention choices — everything in VGC speaks
# COLMAP, and the boundaries are Julia.  Users with non-COLMAP math compose
# the frame map at the boundary themselves, e.g.
# `math ∘ EuclideanMap(PixelFrame{:colmap}, PixelFrame{:opencv})`.
#
# Sampling is ONE KernelAbstractions bilinear kernel (`_warp_kernel!`, one warp
# or a batch of them) that runs on whatever backend owns the destination
# array — CPU and CUDA share the implementation.  The warp math executes
# per output pixel ON the device: the fused `BoundedMap` is isbits.
# =============================================================================

using CoordinateTransformations: Transformation, Translation, LinearMap, AffineMap,
    PerspectiveMap, ComposedTransformation
using ImageCore: ImageCore, Colorant, clamp01, floattype

# -----------------------------------------------------------------------------
# Row/col ↔ (x, y) permutation — singleton Transformation, zero arithmetic.
# -----------------------------------------------------------------------------

"""
    RCSwap <: Transformation

Singleton permutation `(p[1], p[2]) ↔ (p[2], p[1])`.  Self-inverse, zero fp
ops (no matrix multiply, just element reorder).  Used at the array/math
boundary of `image_warp` to bridge `(row, col)` and `(x, y)`.  `fuse`
canonicalizes it to a `LinearMap` permutation matrix so it participates in
affine-chain flattening.
"""
struct RCSwap <: Transformation end
@inline (::RCSwap)(p::StaticVector{2}) = SVector(p[2], p[1])
@inline (::RCSwap)(p::Tuple) = (p[2], p[1])
Base.inv(::RCSwap) = RCSwap()

const _RC_SWAP = RCSwap()
# The COLMAP↔Julia boundary bridges, expressed as typed frame maps (0.7):
# the "VGC math is COLMAP" invariant is now a labeled endpoint, not a bare
# ±0.5 constant. `fuse` canonicalizes them to AffineMaps like any
# EuclideanMap{2}, so the fused per-pixel math is unchanged.
const _IN_BRIDGE = EuclideanMap(PixelFrame{:julia}, PixelFrame{:colmap})   # Julia idx → COLMAP math
const _OUT_BRIDGE = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})  # COLMAP math → Julia idx

# =============================================================================
# is_projective — does the transform involve a projective (homogeneous) stage?
# =============================================================================

"""
    is_projective(t) -> Bool

`true` when `t` contains a projective stage (a `HomographyMap`) anywhere in
its composition tree — i.e. its output is only defined up to scale unless a
perspective division is applied somewhere downstream.  Recurses through BOTH
sides of `Base.ComposedFunction` and `CT.ComposedTransformation` (the
predecessor `needs_perspective_division` only checked the inner side, and
missed CT-typed compositions entirely).
"""
is_projective(::Any) = false
is_projective(::HomographyMap) = true
is_projective(f::ComposedFunction) = is_projective(f.outer) || is_projective(f.inner)
is_projective(f::ComposedTransformation) = is_projective(f.t1) || is_projective(f.t2)
is_projective(bm::BoundedMap) = is_projective(bm.map)

# Does the chain already contain a perspective division?  Guards against
# double-injecting `PerspectiveMap` when the caller composed one themselves —
# or composed a lens, which consumes a homogeneous ray (cameras extend this).
_has_division(::Any) = false
_has_division(::PerspectiveMap) = true
_has_division(f::ComposedFunction) = _has_division(f.outer) || _has_division(f.inner)
_has_division(f::ComposedTransformation) = _has_division(f.t1) || _has_division(f.t2)

# =============================================================================
# fuse — flatten matrix-representable runs in a composition chain
# =============================================================================

# Flatten a composition tree (both CT and Base spellings) into an
# outer-to-inner leaf tuple.
_leaves(t) = (t,)
_leaves(f::ComposedFunction) = (_leaves(f.outer)..., _leaves(f.inner)...)
_leaves(f::ComposedTransformation) = (_leaves(f.t1)..., _leaves(f.t2)...)

# Canonicalize leaves that block CT's affine algebra into forms it merges.
_canonical(t) = t
_canonical(::RCSwap) = LinearMap(SMatrix{2, 2, Float64}(0.0, 1.0, 1.0, 0.0))
_canonical(e::EuclideanMap{2}) = AffineMap(SMatrix{3, 3, Float64}(to_homogeneous(e)))

# Matrix-representable leaves: CT's ∘ (affine algebra) or VGC's HomographyMap
# bridges collapse any adjacent pair of these into ONE link.
const _MatrixLeaf = Union{
    Translation{<:StaticVector{2}},
    LinearMap,
    AffineMap,
    HomographyMap{3},
}

# Lift a matrix leaf to a HomographyMap (for pulling affines through a
# perspective division: A ∘ PerspectiveMap == PerspectiveMap ∘ lift(A)).
_as_homography(h::HomographyMap{3}) = h
_as_homography(a::Union{Translation{<:StaticVector{2}}, AffineMap}) =
    HomographyMap(to_homogeneous(a))
function _as_homography(a::LinearMap)
    M = SMatrix{2, 2}(a.linear)
    return HomographyMap(to_homogeneous(M, zero(SVector{2, eltype(M)})))
end

# Right-to-left greedy merge over the canonical leaf tuple.  Recursive on
# tuples so the common chains infer to concrete fused types.
_fuse_seq(xs::Tuple{T}) where {T} = xs
function _fuse_seq(xs::Tuple)
    rest = _fuse_seq(Base.tail(xs))
    a, b = first(xs), first(rest)
    if a isa _MatrixLeaf && b isa _MatrixLeaf
        return (a ∘ b, Base.tail(rest)...)
    elseif a isa _MatrixLeaf && b isa PerspectiveMap
        # Pull the affine through the division: division-then-affine equals
        # (homogeneous-lift ∘ homography)-then-division.
        return (PerspectiveMap(), _fuse_seq((_as_homography(a), Base.tail(rest)...))...)
    else
        return (a, rest...)
    end
end

_compose_all(xs::Tuple{T}) where {T} = first(xs)
_compose_all(xs::Tuple) = first(xs) ∘ _compose_all(Base.tail(xs))

"""
    fuse(t) -> Transformation

Flatten every run of matrix-representable links in a composition chain into
a single link.  `Translation`/`LinearMap`/`AffineMap`/`EuclideanMap`/`RCSwap`
runs merge into one `AffineMap`; runs containing a `HomographyMap` merge into
one `HomographyMap`; an affine link immediately left of a `PerspectiveMap` is
pulled through the division into the homography.  Nonlinear links (log-polar,
lens models, …) are preserved in place.  Semantically a no-op; structurally
the difference between five closure calls per pixel and one matrix multiply.

`fuse(bm::BoundedMap)` fuses the wrapped map and keeps the rect.
"""
fuse(t::Transformation) = _compose_all(_fuse_seq(map(_canonical, _leaves(t))))
fuse(bm::BoundedMap) = BoundedMap(fuse(bm.map), bm.rect)

# =============================================================================
# Precision conversion — Float32 warp math for GPU-resident pipelines
# =============================================================================
#
# `_precision(T, t)` rebuilds a (fused) chain with all stored floats in
# precision `T`.  Leaves that carry no floats are returned unchanged; a lens or
# grid model forwards to the public `convert_eltype` (the one bridge is in
# cameras/generic_models.jl), so a downstream lens's method reaches the kernels.

_precision(::Type{T}, x) where {T <: AbstractFloat} = x
_precision(::Type{T}, x::Translation) where {T <: AbstractFloat} =
    Translation(T.(x.translation))
_precision(::Type{T}, x::LinearMap) where {T <: AbstractFloat} =
    LinearMap(T.(x.linear))
_precision(::Type{T}, x::AffineMap) where {T <: AbstractFloat} =
    AffineMap(T.(x.linear), T.(x.translation))
_precision(::Type{T}, x::HomographyMap{3, S, M, Src, Dst}) where {T <: AbstractFloat, S, M, Src, Dst} =
    relabel(Src, Dst, HomographyMap(SMatrix{3, 3, T, 9}(to_homogeneous(x))))
_precision(::Type{T}, x::ComposedFunction) where {T <: AbstractFloat} =
    _precision(T, x.outer) ∘ _precision(T, x.inner)
_precision(::Type{T}, x::ComposedTransformation) where {T <: AbstractFloat} =
    _precision(T, x.t1) ∘ _precision(T, x.t2)
_precision(::Type{T}, bm::BoundedMap) where {T <: AbstractFloat} =
    BoundedMap(_precision(T, bm.map), bm.rect)

# =============================================================================
# image_warp — the single warp adapter
# =============================================================================

# `math` is typed to the documented contract (a `Transformation`, not `Any`).
# Untyped, `∘` on a plain function yields a `Base.ComposedFunction`, for which
# no `BoundedMap` constructor exists — so a non-Transformation argument failed
# downstream with an opaque MethodError on `BoundedMap` instead of here (JET
# flagged the union split, 2026-07-26). Nothing that works today is excluded:
# a ComposedFunction never reached `BoundedMap` successfully.
"""
    image_warp([T=Float64], math, output_rect::Rect{2}) -> BoundedMap

Wrap a math-space `Transformation` into a fused `BoundedMap` callable from
Julia array indices.  Composes (left-to-right when reading a call sequence):

    _RC_SWAP ∘ _OUT_BRIDGE ∘ math ∘ _IN_BRIDGE ∘ _RC_SWAP

then flattens the chain with [`fuse`](@ref).  Perspective division is
injected automatically when `math` is projective ([`is_projective`](@ref))
and doesn't already divide.

The output `BoundedMap` consumes `(row, col)` indices into the OUTPUT array
and returns `(sr, sc)` indices into the SOURCE array — both Julia 1-indexed.
The internal bridge pair — the typed frame maps
`EuclideanMap(PixelFrame{:julia}, PixelFrame{:colmap})` and its reverse —
converts to/from VGC's package-wide COLMAP math convention.

# Arguments
- `T` — optional leading element type of the fused warp math (the standard
  `rand(Float32, …)`-style convention).  `image_warp(Float32, …)` keeps the
  per-pixel math single-precision on GPU backends.
- `math` — any `CoordinateTransformations.Transformation`.  Assumed COLMAP-math.
- `output_rect::Rect{2}` — the output region in array coords (e.g.,
  `Rect(0, 0, width, height)`).

# Example
```julia
math = canonical_map(ellipse, 64; bound=1.0)
bm   = image_warp(math, 64)
patch = Matrix{Float32}(undef, 64, 64)
image_warp!(patch, bm, img_f32; fillvalue=0.5f0)
```
"""
function image_warp(
        ::Type{T}, math::Transformation, output_rect::Rect{2}
    ) where {T <: AbstractFloat}
    inner = (is_projective(math) && !_has_division(math)) ?
        PerspectiveMap() ∘ math : math
    fused = fuse(_RC_SWAP ∘ _OUT_BRIDGE ∘ inner ∘ _IN_BRIDGE ∘ _RC_SWAP)
    return BoundedMap(_precision(T, fused), output_rect)
end

image_warp(math::Transformation, output_rect::Rect{2}) =
    image_warp(Float64, math, output_rect)

"""
    image_warp([T=Float64], math, n::Integer) -> BoundedMap

Convenience overload for a square `n × n` output array.  Equivalent to
`image_warp(T, math, Rect(0, 0, n, n))`.  Typically paired with the
size-form math builders (`canonical_map(geom, n; …)`,
`logpolar_map(geom, n, r; …)`) for the common case where the patch domain
and output array share `n`.
"""
image_warp(::Type{T}, math, n::Integer) where {T <: AbstractFloat} =
    image_warp(T, math, Rect(0, 0, n, n))
image_warp(math, n::Integer) = image_warp(Float64, math, Rect(0, 0, n, n))

# =============================================================================
# The KernelAbstractions resampler — ONE bilinear kernel, every backend
# =============================================================================

# Whether the position (r, c) is within a bilinear tap's reach of the
# 1-based H × W grid. Decided in float, so a non-finite position (NaN fails
# every comparison) or a huge one (which unsafe_trunc would wrap to an
# arbitrary index) is out of reach before any index is formed.
@inline _within_reach(r, c, H, W) = 0 < r < H + 1 && 0 < c < W + 1

# Element-wise bilinear sample with fillvalue boundary.  The single sampling
# implementation shared by the plain and batched kernels.
@inline function _bilinear_sample(
        src::AbstractMatrix{S}, r::Real, c::Real,
        fillvalue::T, H::Integer, W::Integer
    ) where {S, T}
    # Outside the taps' reach every tap is fillvalue; a non-finite position
    # (the lens contract's "not imaged") is out of reach, so its NaN blend
    # weights never poison the sample.
    _within_reach(r, c, H, W) || return fillvalue
    r0 = unsafe_trunc(Int, floor(r)); r1 = r0 + 1
    c0 = unsafe_trunc(Int, floor(c)); c1 = c0 + 1
    dr = r - r0; dc = c - c0
    v00 = (1 <= r0 <= H && 1 <= c0 <= W) ? T(src[r0, c0]) : fillvalue
    v01 = (1 <= r0 <= H && 1 <= c1 <= W) ? T(src[r0, c1]) : fillvalue
    v10 = (1 <= r1 <= H && 1 <= c0 <= W) ? T(src[r1, c0]) : fillvalue
    v11 = (1 <= r1 <= H && 1 <= c1 <= W) ? T(src[r1, c1]) : fillvalue
    return T(
        (1 - dr) * (1 - dc) * v00 + (1 - dr) * dc * v01 +
            dr * (1 - dc) * v10 + dr * dc * v11
    )
end

# Accumulation type for a destination eltype: sample in float, store back.
_sample_type(::Type{T}) where {T <: AbstractFloat} = T
_sample_type(::Type{UInt8}) = Float32
_sample_type(::Type{C}) where {C <: Colorant} = floattype(C)

# Store a float-domain sample into the destination eltype.
@inline _store(::Type{T}, v) where {T <: AbstractFloat} = T(v)
@inline _store(::Type{UInt8}, v) = round(UInt8, clamp(v, zero(v), oftype(v, 255)))
@inline _store(::Type{C}, v) where {C <: Colorant} = convert(C, clamp01(v))

# ONE warp kernel: output (i, j) of warp k samples `src` at warps[k]((i, j)).
# `warps` is a device vector of isbits BoundedMaps (the batch, launched over the
# (P_r, P_c, N) stack) or a 1-tuple (a single warp, supplied by value — no device
# allocation — launched over the destination matrix, where k is 1).
@kernel function _warp_kernel!(out, @Const(src), @Const(warps), fv)
    I = @index(Global, Cartesian)
    H, W = size(src)
    @inbounds begin
        p = warps[_warp_index(I)](SVector(I[1], I[2]))   # Ints promote to the warp math's precision
        out[I] = _store(eltype(out), _bilinear_sample(src, p[1], p[2], fv, H, W))
    end
end
@inline _warp_index(::CartesianIndex{2}) = 1
@inline _warp_index(I::CartesianIndex{3}) = I[3]

"""
    image_warp!(dst, bm::BoundedMap, src; fillvalue=0) -> dst

Resample `src` into `dst` through the warp `bm` (built by [`image_warp`](@ref)):
for each `(row, col)` of `dst`, bilinearly sample `src` at `bm((row, col))`.
Runs the shared KernelAbstractions kernel on the backend that owns `dst` —
supply `CuArray`s for GPU execution (the warp math evaluates on-device; `bm`
is isbits).

`dst` and `src` may differ in eltype: samples accumulate in float and store
back through the destination eltype (`UInt8` rounds saturating, colorants
clamp to `[0, 1]`).  `fillvalue` is the out-of-bounds tap value, in the
float accumulation domain.
"""
function image_warp!(
        dst::AbstractMatrix, bm::BoundedMap, src::AbstractMatrix;
        fillvalue = zero(_sample_type(eltype(dst)))
    )
    backend = get_backend(dst)
    fv = _sample_type(eltype(dst))(fillvalue)
    _warp_kernel!(backend, (32, 32))(dst, src, (bm,), fv; ndrange = size(dst))
    KernelAbstractions.synchronize(backend)
    return dst
end

"""
    image_warp_batch_to!(out, warps, src; fillvalue=0) -> out

Sample `src` once per output pixel of every warp in `warps`, writing into
the pre-allocated 3D stack `out::AbstractArray{T, 3}` of shape
`(P_r, P_c, N)` where `N = length(warps)`.  Each `warps[k]` is a
`BoundedMap` produced by `image_warp`.  Same kernel and backend dispatch as
[`image_warp!`](@ref) — supply `CuArray`s (including the warp vector) for GPU
execution.

# Example (CPU)
```julia
warps = [image_warp(canonical_map(d, 64; bound=1.0), 64) for d in discs]
out   = Array{Float32}(undef, 64, 64, length(warps))
image_warp_batch_to!(out, warps, src; fillvalue=Float32(0.5))
```

# Example (GPU)
```julia
d_warps = CuArray(warps)
d_src   = CuArray(src_f32)
d_out   = CUDA.zeros(Float32, 64, 64, length(warps))
image_warp_batch_to!(d_out, d_warps, d_src; fillvalue=0.5f0)
```
"""
function image_warp_batch_to!(
        out::AbstractArray{T, 3}, warps::AbstractVector{<:BoundedMap},
        src::AbstractMatrix;
        fillvalue = zero(_sample_type(T))
    ) where {T}
    P_r, P_c, N = size(out)
    length(warps) == N || throw(
        DimensionMismatch(
            "size(out, 3) = $N must equal length(warps) = $(length(warps))"
        )
    )
    backend = get_backend(out)
    fv = _sample_type(T)(fillvalue)
    _warp_kernel!(backend, (16, 16, 1))(out, src, warps, fv; ndrange = size(out))
    KernelAbstractions.synchronize(backend)
    return out
end

# =============================================================================
# FramedImage — a canvas that remembers where it sits in the pixel frame
# =============================================================================

"""
    FramedImage(data, rect)

A resampled canvas TOGETHER with its placement: `rect` is the window of
the source pixel frame (COLMAP convention — origin at the window's top-left
corner, widths its footprint) that `data` covers; the sampling density is
`widths(rect) ./ size(data)` (reversed: array is row-major). Named `rect`
— not `extent` — deliberately: VGC's `extent(camera)` is a SIZE
(`Size2{Int}`), while this is a POSITIONED window. This is the
warp layer's currency at the return boundary — a bare matrix loses exactly
the information every consumer then has to carry by hand (the `ext` + `σ`
pair the rect form of [`undistort`](@ref) used to drop), and misplacing a
rectified image by its offset or scale is the half-pixel bug class the
frame conventions exist to kill.

`imshow(pos, ci::FramedImage)` (Makie extension) is self-placing: it draws
`data` over `rect`, so a rectified view lands in the same axes as the raw
frame — principal point, footprint and all — with nothing supplied alongside.

Produced by the rect form of [`undistort`](@ref); construct directly for
any canvas rendered over a known window of a pixel frame.
"""
struct FramedImage{M <: AbstractMatrix, R <: GeometryBasics.HyperRectangle{2}}
    data::M
    rect::R
end

Base.size(ci::FramedImage) = size(ci.data)
Base.size(ci::FramedImage, d::Integer) = size(ci.data, d)

"""
    bilinear_sample_clamped(::Type{Ta}, img, yf, xf, H::Int32, W::Int32) -> (in_bounds::Bool, value::Ta)

Bilinear sample of `img` at the 1-based array position `(yf, xf)` with the
OTHER boundary policy from `_bilinear_sample`'s fill: a position whose floor
cell lies outside `1:H × 1:W` is rejected (`(false, 0)`), and the `+1`
neighbour is clamped to the last row/column, so the border pixel is sampled
exactly rather than blended with a fill value. Exact (no interpolation
error) at integer positions, which is what keeps an integer-step scan
bit-identical to a pixel loop. Device-safe (`Int32` indices, no allocation);
hoisted verbatim from BlobBoards' adaptation kernels (2026-09).
"""
@inline function bilinear_sample_clamped(::Type{Ta}, img, yf, xf, H::Int32, W::Int32) where {Ta}
    # A non-finite or huge position is rejected before unsafe_trunc forms an index.
    _within_reach(yf, xf, H, W) || return (false, zero(Ta))
    ix0 = unsafe_trunc(Int32, floor(xf))
    iy0 = unsafe_trunc(Int32, floor(yf))
    if ix0 < Int32(1) || iy0 < Int32(1) || ix0 > W || iy0 > H
        return (false, zero(Ta))
    end
    tx = Ta(xf) - Ta(ix0)
    ty = Ta(yf) - Ta(iy0)
    ix1 = min(ix0 + Int32(1), W)
    iy1 = min(iy0 + Int32(1), H)
    v00 = Ta(img[iy0, ix0]); v01 = Ta(img[iy0, ix1])
    v10 = Ta(img[iy1, ix0]); v11 = Ta(img[iy1, ix1])
    a = v00 + (v01 - v00) * tx
    b = v10 + (v11 - v10) * tx
    return (true, a + (b - a) * ty)
end

# -----------------------------------------------------------------------------
# A texture lookup: bilinear in the half-texel chart, extended at the edges
# (hoisted from RadiometricCamera src/board/board.jl, `bilinear` and
# `_bilinear_setup`, its arithmetic unchanged)
# -----------------------------------------------------------------------------
#
# Not `bilinear_sample_clamped`'s policy: that one addresses 1-based array
# positions, rejects a position off the image and is exact at integers (a scan
# over pixels); this one addresses the unit square of the texture, never
# rejects (a position off it reads the edge), returns several channels at once
# and differentiates in the position. Two lookups with two contracts, not one
# with a flag.

# The value part of a (nested) Dual: a texel is chosen on it, the weights carry the partials.
@inline _value_part(x::ForwardDiff.Dual) = _value_part(ForwardDiff.value(x))
@inline _value_part(x) = x

# The documented minimum, checked once per lookup before any texel is read (a
# 1-wide or 1-high image has no second texel to blend with: the clamped index
# would read past it).
@noinline _too_small_texture(H, W) = throw(ArgumentError("bilinear_sample: the image needs at least 2 × 2 texels; got $H × $W"))
@inline function _bilinear_setup(H::Int, W::Int, uv::StaticVector{2})
    (H >= 2 && W >= 2) || _too_small_texture(H, W)
    h = one(eltype(uv)) / 2
    x = uv[1] * W - h; y = uv[2] * H - h
    xv = _value_part(x); yv = _value_part(y)
    Tv = typeof(xv)
    isfinite(xv) || (x = zero(x); xv = zero(Tv))        # a non-finite position reads the corner texel
    isfinite(yv) || (y = zero(y); yv = zero(Tv))
    xv = clamp(xv, zero(Tv), Tv(W - 1)); yv = clamp(yv, zero(Tv), Tv(H - 1))
    x0 = min(floor(Int, xv), W - 2); y0 = min(floor(Int, yv), H - 2)
    x0 = max(x0, 0); y0 = max(y0, 0)
    fx = clamp(x - x0, zero(x), one(x)); fy = clamp(y - y0, zero(y), one(y))
    return x0 + 1, y0 + 1, fx, fy          # 1-based indices of the lower-left texel
end

"""
    bilinear_sample(img::AbstractMatrix, uv::StaticVector{2}) -> value
    bilinear_sample(img::AbstractArray{T, 3}, Val(C), uv::StaticVector{2}) -> SVector{C}

Bilinear lookup of a texture at `uv` in its unit-square chart, the
half-texel convention: `img[i, j]` (row `i` of `H`, column `j` of `W`) is the
texel centred at `uv = ((j − ½)/W, (i − ½)/H)`, so `(0, 0)` is the outer
corner of `img[1, 1]` and `(1, 1)` that of `img[H, W]`. A position off the
centres' rectangle EXTENDS the edge (clamp to edge, never a rejection), and a
non-finite position reads the corner texel `img[1, 1]`. The image needs at
least 2 × 2 texels (an `ArgumentError` otherwise).

A matrix returns one value (any element type that scales and adds: numbers,
`SVector`s); a 3-D array `(H, W, C)` returns its first `C` channels as an
`SVector{C}` — RGB, or a signed distance and its orientation tensor.
Allocation-free and generic: under ForwardDiff the texel is chosen on the
position's value and the derivative flows through the weights, which are
continuous across texels (the value is C⁰).

The other bilinear lookup, [`bilinear_sample_clamped`](@ref), addresses
1-based array positions and rejects positions off the image.
"""
@inline function bilinear_sample(img::AbstractMatrix, uv::StaticVector{2})
    H, W = size(img)
    i, j, fx, fy = _bilinear_setup(H, W, uv)
    @inbounds return (1 - fx) * (1 - fy) * img[j, i] + fx * (1 - fy) * img[j, i + 1] +
        (1 - fx) * fy * img[j + 1, i] + fx * fy * img[j + 1, i + 1]
end
@inline function bilinear_sample(img::AbstractArray{T, 3}, ::Val{C}, uv::StaticVector{2}) where {T, C}
    H, W = size(img, 1), size(img, 2)
    i, j, fx, fy = _bilinear_setup(H, W, uv)
    w00 = (1 - fx) * (1 - fy); w10 = fx * (1 - fy); w01 = (1 - fx) * fy; w11 = fx * fy
    @inbounds return SVector{C}(
        ntuple(c -> w00 * img[j, i, c] + w10 * img[j, i + 1, c] + w01 * img[j + 1, i, c] + w11 * img[j + 1, i + 1, c], Val(C))
    )
end

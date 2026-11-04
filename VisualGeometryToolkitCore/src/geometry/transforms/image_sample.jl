# =============================================================================
# image_sample! — the bilinear sample of an image at scattered points
# =============================================================================
#
# `image_warp!` samples a source at the image of every pixel of a destination
# raster under one map; `image_sample!` samples it at points the caller already
# holds — the projections of a set of directions, the sensor points of
# features — one per destination element, in any array shape. Both run the
# same tap (`_bilinear_sample`, warp.jl) with the same fill rule in ONE
# KernelAbstractions kernel on the backend that owns the destination, so a
# value sampled at a point is the value a warp landing on that point gives.
# =============================================================================

@kernel function _sample_kernel!(out, @Const(src), @Const(points), fv)
    I = @index(Global, Linear)
    H, W = size(src)
    @inbounds begin
        p = points[I]
        # a COLMAP point (x, y) is the Julia array position (row y + ½, column x + ½)
        out[I] = _store(eltype(out), _bilinear_sample(src, p[2] + oftype(p[2], 0.5), p[1] + oftype(p[1], 0.5), fv, H, W))
    end
end

"""
    image_sample!(dst, src::AbstractMatrix, points; fillvalue = 0) -> dst

Bilinearly sample `src` at `points[i]` into `dst[i]`, for every index of
`points` (an array of `StaticVector{2}` of any shape; `dst` of the same size):
`points` are COLMAP coordinates in `src`'s pixel frame — `(x, y)`, the centre
of `src[i, j]` at `(j − ½, i − ½)` — the frame every Core map speaks, and the
one [`image_warp`](@ref)'s math lands in. The tap and its boundary are
[`image_warp!`](@ref)'s: a tap off the image reads `fillvalue`, so a point
within half a pixel outside the centres' rectangle blends toward it and a point
farther out, or a non-finite one (a map's "not imaged"), is `fillvalue`.

Runs one KernelAbstractions kernel on the backend that owns `dst`: supply
device arrays (`dst`, `src` and `points` on one device) for GPU execution.
`dst` and `src` may differ in eltype, as in `image_warp!`; `fillvalue` is in
the float accumulation domain.
"""
function image_sample!(
        dst::AbstractArray, src::AbstractMatrix, points::AbstractArray{<:StaticVector{2}};
        fillvalue = zero(_sample_type(eltype(dst)))
    )
    size(dst) == size(points) ||
        throw(DimensionMismatch("image_sample!: dst is $(size(dst)), points $(size(points))"))
    isempty(dst) && return dst
    backend = get_backend(dst)
    fv = _sample_type(eltype(dst))(fillvalue)
    _sample_kernel!(backend, 256)(dst, src, points, fv; ndrange = length(dst))
    KernelAbstractions.synchronize(backend)
    return dst
end

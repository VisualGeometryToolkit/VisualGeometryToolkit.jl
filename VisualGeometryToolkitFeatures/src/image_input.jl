# =============================================================================
# Image input ladders — one gray-float and one gray-byte coercion
# =============================================================================
#
# Every detector-side entry point in the family starts by coercing "whatever
# the loader gave us" (RGB{N0f8} from a JPEG, Gray{Float32} from a video
# decoder, a bare Float32 device array) into the form its kernels consume.
# Two forms recur — float gray (adaptation, canonicalisation, float ASIFT) and
# 8-bit gray bytes (the ASIFT byte upload, the undistort-then-detect
# path) — and until 2026-09 BlobBoards and this package each carried their
# own ladder. These are the two, exported; `_asift_input` builds on them.
# =============================================================================

"""
    gray_f32(img) -> AbstractMatrix

`img` as float gray: a float `Gray` matrix or a bare `Float32` matrix (a
device array already in the kernels' form) is returned UNTOUCHED — no
copy, no wrap, so a `CuArray{Float32}` stays on the device; anything else
(fixed-point gray, any colorant) converts to `Gray{Float32}`.
"""
gray_f32(img::AbstractMatrix{<:Gray{<:AbstractFloat}}) = img
gray_f32(img::AbstractMatrix{Float32}) = img
gray_f32(img::AbstractMatrix) = Gray{Float32}.(img)

"""
    gray_u8(img) -> Matrix{UInt8}

`img` as H×W gray bytes, in a `Matrix{UInt8}` the caller owns. A
`Matrix{UInt8}` is returned as is; a `Gray{N0f8}` matrix is copied byte for byte
(its pixels already are bytes); everything else is quantised through
`Gray{N0f8}` (float colorants LOSE precision here by design — use
[`gray_f32`](@ref) where sub-pixel photometry matters).
"""
gray_u8(img::Matrix{UInt8}) = img
gray_u8(img::AbstractMatrix{UInt8}) = collect(img)
# A COPY, never a wrap. This used to `unsafe_wrap` a `Gray{N0f8}` Array's memory
# as a `Matrix{UInt8}` (zero-copy), but nothing kept the image alive behind the
# wrapper: after the caller's last use of the image the GC could free the pixels
# while the detector still read them, and the `Gray{N0f8}.(img)` temporary every
# other colorant goes through was freed almost at once. Measured 2026-09-15: after
# one `GC.gc` and a few allocations, 15 782 of 4 000 000 bytes of a wrapped
# temporary were intact, and in a fresh worker process the same read crashed it.
# One memcpy per frame.
gray_u8(img::AbstractMatrix{Gray{N0f8}}) = collect(reinterpret(UInt8, img))::Matrix{UInt8}
gray_u8(img::AbstractMatrix) = gray_u8(Gray{N0f8}.(img))

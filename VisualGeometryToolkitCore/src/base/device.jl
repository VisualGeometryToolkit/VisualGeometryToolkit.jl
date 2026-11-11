# =============================================================================
# Default compute device — ONE resolver for the package family
# =============================================================================
#
# Every KernelAbstractions-generic stage in the family takes a
# `device::KA.Backend`. Defaulting that to `KA.CPU()` at each entry point
# makes the wrong thing both invisible and expensive: VisualGeometryToolkitFeatures'
# chunker sizes its resident pyramids at 80% of `Sys.free_memory()` on the host
# path, so a caller that forgot `device =` once measured ~20 GiB of anon RSS for
# a single 12.6 MP frame (48.5 GB at full resolution) and was OOM-killed rather
# than told. So "unspecified" resolves to an ACCELERATOR when one is
# functional, else `KA.CPU()` — exactly the old behaviour on a CPU-only host.
#
# Vendor packages are weakdeps — this file cannot name `CUDABackend` — so the
# resolution is a hook the accelerator extension REGISTERS into
# (`ext/VisualGeometryToolkitCoreCUDAExt.jl` assigns it in `__init__`). A method
# override instead would be method overwriting, which Julia refuses during
# extension precompilation.
#
# Hoisted from BlobBoards' src/device.jl on 2026-09-07, where the same
# resolver had existed since 2026-08 while this package's own consumers kept
# the hardcoded host default.
# =============================================================================

import KernelAbstractions as KA
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, get_backend

const _DEFAULT_DEVICE = Ref{Any}(nothing)

"""
    _DEVICE_RESOLVER

The thunk `default_device` calls once. Accelerator extensions assign it at
load time; the registered thunk is gated on the vendor package being
FUNCTIONAL (`CUDA.functional()`), not merely loaded, so a machine with the
package but no working driver still gets the host path.
"""
const _DEVICE_RESOLVER = Ref{Any}(() -> KA.CPU())

"""
    default_device() -> KA.Backend

The device every KernelAbstractions-generic entry point in the package family
uses when the caller supplies none: an accelerator when one is loaded AND
functional, else `KA.CPU()`.

Memoized: resolution asks the vendor package whether it is functional, which
is not free and cannot change within a session. `_DEVICE_RESOLVER[]` overrides
it (including back to `KA.CPU()`); [`reset_default_device!`](@ref) clears the
memo so the next call resolves again.

Supplying `device =` explicitly always wins — this only decides what
"unspecified" means, and it is deliberately NOT the cheapest device (see the
file header).
"""
function default_device()
    d = _DEFAULT_DEVICE[]
    d === nothing || return d
    d = _DEVICE_RESOLVER[]()
    _DEFAULT_DEVICE[] = d
    return d
end

"""
    reset_default_device!() -> nothing

Forget the memoized default so the next `default_device()` re-resolves —
for tests, and for a session that changed `_DEVICE_RESOLVER[]` after the
first resolution.
"""
reset_default_device!() = (_DEFAULT_DEVICE[] = nothing; nothing)

"""
    _DEVICE_MEMORY

The accelerator extensions' device-memory queries, `(; free, reclaim, names)` or
`nothing`: `free(dev)` the bytes free on a backend it defines (`nothing` for another),
`reclaim(dev)` its pool handed back to the device, `names()` its devices' names.
Assigned at load time, as `_DEVICE_RESOLVER` is (Core's CUDA extension).
"""
const _DEVICE_MEMORY = Ref{Any}(nothing)

"""
    device_free_bytes(dev) -> Int

Free memory on the KernelAbstractions backend `dev`, in bytes: the host's for
`KA.CPU()`, the accelerator extension's query for its backend (CUDA's
`free_memory`), else `typemax(Int)` with a one-time warning (a caller sizing
chunks from it then does not chunk).
"""
function device_free_bytes(dev)
    dev isa KA.CPU && return Int(Sys.free_memory())
    q = _DEVICE_MEMORY[]
    f = q === nothing ? nothing : q.free(dev)
    f === nothing || return Int(f)
    @warn "no free-memory query for $(typeof(dev)): tilt and score chunking are disabled" maxlog = 1
    return typemax(Int)
end

"""
    reclaim_device!(dev) -> nothing

Hand the memory `dev`'s backend pool holds back to the device (CUDA.jl's `reclaim`),
so an allocator outside that pool (ONNX Runtime's) can use it; nothing for the host
or a backend without a pool query.
"""
function reclaim_device!(dev)
    dev isa KA.CPU && return nothing
    q = _DEVICE_MEMORY[]
    q === nothing || q.reclaim(dev)
    return nothing
end

"""
    device_names() -> Vector{String}

The names of the accelerator devices this process sees (the extension's query: CUDA's
devices when CUDA is functional); none without an accelerator extension.
"""
device_names() = (q = _DEVICE_MEMORY[]; q === nothing ? String[] : q.names())

# =============================================================================
# Moving arrays to a device and back — ONE upload for the package family
# =============================================================================
#
# A user-facing call that takes `device` (undistort, rectify, a spline camera's
# batched project/backproject; Features' matchers and detectors) uploads its
# input once (`on_device(device, A)`), runs on `device`, and hands the result
# back in the CALLER's array kind (`on_device(get_backend(A), result)`): an
# `Array` in gives an `Array` out, a device array in stays on its device with no
# copy. `ka_upload` is hoisted from VisualGeometryToolkitFeatures'
# matching_ka.jl (2026-11-10); the hand-rolled `allocate` + `copyto!` uploads
# (eleven: the ASIFT driver's, its cross-warp NMS's, the matchers') call it, and
# "as it is when it is there in `T`, else a copy" (Features' score matrices,
# Core's 4-D B-spline weights) is `on_device(T, device, A)`.

"""
    ka_upload(T, device, A) -> array

A copy of `A` with element type `T` on `device`: one `Array{T}` on the host
(`Array{T}(A)`, which also downloads a device array), else a `KA.allocate`d
array filled by one `copyto!` — from `A` itself when it is a dense array of `T`,
else from `T.(A)` (a view, an adjoint, a reinterpret, another eltype). An
integer image (`UInt8`, `UInt16`, `N0f8`, `N0f16`: an integer or fixed-point
eltype narrower than a float `T`) crosses as it is and is converted ON the
device by one KernelAbstractions kernel, `T(x)` per element, on every backend
(the CPU's included): bit for bit the host's `T.(A)`, at `sizeof(eltype(A))`
bytes a pixel moved instead of `sizeof(T)`.
"""
ka_upload(::Type{T}, dev::KA.CPU, A::AbstractArray) where {T} = _narrow_upload(T, eltype(A)) ? _convert_on(T, dev, A) : Array{T}(A)
function ka_upload(::Type{T}, dev, A::AbstractArray) where {T}
    _narrow_upload(T, eltype(A)) && return _convert_on(T, dev, A)
    d = KA.allocate(dev, T, size(A)...)
    # A dense array of `T` is copied as it is; anything else — another eltype, a view, an
    # adjoint, a reinterpret — is materialized first (`T.(A)`: a host wrapper becomes an
    # `Array`, a device one stays on its device), since a device `copyto!` from the host
    # takes a dense array only.
    copyto!(d, A isa DenseArray && eltype(A) === T ? A : T.(A))
    return d
end

# An integer image (UInt8, UInt16, a FixedPointNumbers `Normed`, any integer or fixed-point eltype narrower than the
# target float) goes to the device as it is, `sizeof(S)` bytes a pixel instead of `sizeof(T)`, and becomes `T` there:
# ONE KernelAbstractions kernel, `T(x)` per element (the host's `T.(A)` element for element, so bit for bit), on every
# backend, the CPU's included (there the "upload" is the image itself, read in place).
_narrow_upload(::Type{T}, ::Type{S}) where {T, S} =
    T <: AbstractFloat && S !== T && (S <: Integer || S <: FixedPointNumbers.FixedPoint) && sizeof(S) < sizeof(T)
@kernel function _convert_elements!(out, @Const(src))
    i = @index(Global, Linear)
    @inbounds out[i] = eltype(out)(src[i])
end
function _convert_on(::Type{T}, dev, A::AbstractArray{S}) where {T, S}
    src = on_device(S, dev, A)                    # the narrow eltype crosses as it is (on the host: the image itself)
    out = KA.allocate(dev, T, size(A)...)
    _convert_elements!(dev)(out, src; ndrange = length(out))
    KA.synchronize(dev)
    return out
end

"""
    on_device([T = eltype(A)], device, A) -> array

`A` itself when it already lives on `device` with element type `T`, else one
[`ka_upload`](@ref)`(T, device, A)`. Also the way back: a result handed back in
the caller's array kind is `on_device(get_backend(input), result)`.
"""
on_device(::Type{T}, dev, A::AbstractArray) where {T} = get_backend(A) == dev && eltype(A) === T ? A : ka_upload(T, dev, A)
on_device(dev, A::AbstractArray) = on_device(eltype(A), dev, A)

"""
    device_zeros(T, device, dims...) -> array

A zero array of element type `T` and size `dims` on the KernelAbstractions
backend `device` (an `Array` on `KA.CPU()`), as [`ka_upload`](@ref) puts one
there: the stack's one spelling of it (KernelAbstractions' own `zeros` is not
public).
"""
device_zeros(::Type{T}, dev, dims::Integer...) where {T} = fill!(KA.allocate(dev, T, dims...), zero(T))
device_zeros(::Type{T}, dev, dims::Tuple{Vararg{Integer}}) where {T} = device_zeros(T, dev, dims...)


# =============================================================================
# One row kernel for every per-correspondence KernelAbstractions fill
# =============================================================================
#
# The LM row fills (Sampson, tangent Sampson, homography transfer, radial
# transfer) and the per-pair Sampson squares all have ONE shape: row `i` of
# `out` is a pure function of `i` and a few arrays. Each supplies that function
# in the uniform form `row!(out, i, args...)` and `_launch_rows!` runs it: one
# generic kernel, `_row_kernel!`, on the backend that owns `out`, the CPU's
# included (the owner's one-kernel policy, 2026-11-11: the plain CPU loop this
# replaced was 1.3× to 4.9× faster at 1000 to 100 rows; the slower CPU launch is
# accepted).
#
# Device safety: `row!` is a function SINGLETON and the state travels in `args`
# (arrays, static matrices, tuples of them) — never a closure over non-isbits
# state, which a device kernel cannot receive. Row functions are
# `Base.@propagate_inbounds`, so the kernel's `@inbounds` reaches their indexing.

@kernel function _row_kernel!(row!, out, args...)
    i = @index(Global)
    @inbounds row!(out, i, args...)
end

"""
    _launch_rows!(row!, out, n, args...) -> out

Run `row!(out, i, args...)` for `i in 1:n` as one `_row_kernel!` launch over
`ndrange = n` on the backend that owns `out` (every backend, the CPU's
included), followed by a synchronize. `n = 0` launches nothing.
"""
function _launch_rows!(row!::F, out, n::Integer, args::Vararg{Any, N}) where {F, N}
    n > 0 || return out
    backend = get_backend(out)
    _row_kernel!(backend)(row!, out, args...; ndrange = n)
    KA.synchronize(backend)
    return out
end

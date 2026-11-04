# =============================================================================
# Pointwise maps over arrays — one driver, every backend
# =============================================================================
#
# A batch of points goes through a per-point function by BROADCASTING it:
# `out .= f.(xs...)` is a SIMD loop on an `Array` and a compiled kernel on a
# GPU array (GPUArrays), so the per-point function — a lens's `project`, a
# camera, a framed stage — is the kernel body, with no per-family kernel code.
# `pointwise!` adds only the choice of how a HOST batch is split across
# threads: the same broadcast on contiguous chunks, one task each.
#
# Failure semantics are the per-point function's own: a point it does not map
# comes back NaN (`all(isfinite, out[i])` is its validity), never a throw.

"""
    ExecutionPolicy

How [`pointwise!`](@ref) runs a host batch: [`Auto`](@ref) (the default),
[`Serial`](@ref) or [`Threaded`](@ref). A device array (CUDA, Metal, …) is
one broadcast kernel under these; [`OnBackend`](@ref) launches one kernel on
a backend value the caller gives.
"""
abstract type ExecutionPolicy end

"One broadcast loop on the calling task — for callers that already parallelize an outer loop."
struct Serial <: ExecutionPolicy end

"The broadcast split into contiguous chunks, one task per available thread."
struct Threaded <: ExecutionPolicy end

"""
    Auto()

[`Serial`](@ref) below [`THREADED_MIN_LENGTH`](@ref) points or on one thread,
[`Threaded`](@ref) above it; a device array is always one broadcast kernel.
"""
struct Auto <: ExecutionPolicy end

"""
    OnBackend(backend; workgroupsize = nothing)

One kernel launched on `backend` itself: the backend value, so its compiler
options hold (`CUDABackend(; always_inline = true)`), which a broadcast cannot
give, since it launches on `get_backend(out)`, a backend with the defaults.
`workgroupsize` fixes the launch's workgroup (`nothing`: the backend's
choice). `out` and `xs` live on `backend` and share one shape (no
broadcasting). The launch is asynchronous, as a device broadcast is:
`KernelAbstractions.synchronize(backend)` before `out` is read on the host.
Each element is `f(xs[1][i], …)`, as in every policy. A functor whose
fields are each one of its type parameters or an isbits value reaches the
kernel as its fields, rebuilt on the device: one large argument costs local
memory there.
"""
struct OnBackend{B <: KernelAbstractions.Backend, W} <: ExecutionPolicy
    backend::B
    workgroupsize::W
end
OnBackend(backend::KernelAbstractions.Backend; workgroupsize = nothing) = OnBackend(backend, workgroupsize)

"""
    THREADED_MIN_LENGTH

The batch length from which [`Auto`](@ref) splits a host batch across threads:
below it a task's start-up costs more than the work it saves. Measured
(benchmark/lenses/new.toml, 32 threads): at 256 points Serial and Threaded tie
for the cheapest lens (one-coefficient division), at 1024 Threaded is 1.2×
faster for it and 3.4× for OpenCV's 5-coefficient undistortion.
"""
const THREADED_MIN_LENGTH = 1024

"""
    pointwise!(f, out, xs...; exec = Auto()) -> out

`out[i] = f(xs[1][i], xs[2][i], …)` for every index — `out .= f.(xs...)` —
on the backend that owns `out`: a SIMD loop on host arrays (split across
threads per `exec`), one broadcast kernel on a GPU array (one kernel on the
given backend with [`OnBackend`](@ref)). `f` is any
per-point function; for a GPU array it must be isbits and kernel-lawful (no
allocation, no throw, no dynamic dispatch), as every lens and camera map is.
Every policy gives bitwise identical results.
"""
pointwise!(f::F, out::AbstractArray, xs::AbstractArray...; exec::ExecutionPolicy = Auto()) where {F} =
    _pointwise!(_policy(exec, out), f, out, xs...)

_policy(p::ExecutionPolicy, out) = p
_policy(::Auto, out) =
    (KernelAbstractions.get_backend(out) isa KernelAbstractions.CPU && length(out) >= THREADED_MIN_LENGTH &&
        Threads.nthreads() > 1) ? Threaded() : Serial()

_pointwise!(::Serial, f::F, out, xs...) where {F} = (out .= f.(xs...); out)

# OnBackend's kernels. A per-point functor reaches the kernel as its FIELDS, separate kernel arguments, and is rebuilt on the device
# (`_device_functor`): one aggregate argument holding every field makes the device compiler copy it to local memory once any of its
# parts is used by reference — RadiometricCamera's per-pixel flux functor (2.3 KB of fields) needed 2800 bytes of local memory as
# one argument against 2456 as its fields, and ran 5 % slower. A functor whose type cannot be rebuilt from its fields' device types
# (`_splattable`), or with more fields than the kernels take, goes as one argument.
const _MAX_FIELDS = 16

# `F`'s fields are each one of its type parameters or an isbits value (which the device conversion keeps): then the device type is
# `F` with those parameters replaced by the fields' device types.
_splattable(::Type{F}) where {F} =
    isconcretetype(F) && !ismutabletype(F) && fieldcount(F) <= _MAX_FIELDS &&
    all(i -> (T = fieldtype(F, i); any(p -> p === T, F.parameters) || isbitstype(T)), 1:fieldcount(F))
# The functor of host type `F` from its fields' device values.
@generated function _device_functor(::Type{F}, fields...) where {F}
    params = collect(Any, F.parameters)
    for i in 1:fieldcount(F)
        j = findfirst(p -> p === fieldtype(F, i), params)
        j === nothing || (params[j] = fields[i])
    end
    T = isempty(params) ? F : F.name.wrapper{params...}
    return Expr(:new, T, (:(fields[$i]) for i in 1:fieldcount(F))...)
end
for n in 0:_MAX_FIELDS
    as = [Symbol(:a, i) for i in 1:n]
    @eval @kernel function $(Symbol(:_pointwise_fields_kernel_, n, :!))(out, xs, ::Type{F}, $(as...)) where {F}
        I = @index(Global, Cartesian)
        f = _device_functor(F, $(as...))
        @inbounds out[I] = f(map(x -> @inbounds(x[I]), xs)...)
    end
end
const _POINTWISE_FIELDS_KERNELS = Tuple(getfield(@__MODULE__, Symbol(:_pointwise_fields_kernel_, n, :!)) for n in 0:_MAX_FIELDS)
# The functor as one argument.
@kernel function _pointwise_kernel!(out, xs, f)
    I = @index(Global, Cartesian)
    @inbounds out[I] = f(map(x -> @inbounds(x[I]), xs)...)
end
_launch(p::OnBackend, k) = p.workgroupsize === nothing ? k(p.backend) : k(p.backend, p.workgroupsize)
function _pointwise!(p::OnBackend, f::F, out, xs...) where {F}
    all(x -> axes(x) == axes(out), xs) ||
        throw(DimensionMismatch("pointwise!: the inputs' axes $(map(axes, xs)) are not the output's $(axes(out))"))
    isempty(out) && return out
    if _splattable(F)
        _launch(p, _POINTWISE_FIELDS_KERNELS[fieldcount(F) + 1])(out, xs, F, ntuple(i -> getfield(f, i), Val(fieldcount(F)))...;
                                                                  ndrange = size(out))
    else
        _launch(p, _pointwise_kernel!)(out, xs, f; ndrange = size(out))
    end
    return out
end

function _pointwise!(::Threaded, f::F, out, xs...) where {F}
    KernelAbstractions.get_backend(out) isa KernelAbstractions.CPU || return _pointwise!(Serial(), f, out, xs...)
    n = length(out)
    nchunks = min(Threads.nthreads(), cld(n, 256))
    nchunks <= 1 && return _pointwise!(Serial(), f, out, xs...)
    ov, xv = vec(out), map(vec, xs)
    @sync for r in Iterators.partition(1:n, cld(n, nchunks))
        Threads.@spawn begin
            o = view(ov, r)
            o .= f.(map(x -> view(x, r), xv)...)
        end
    end
    return out
end

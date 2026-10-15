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
always one broadcast kernel, whatever the policy.
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
threads per `exec`), one broadcast kernel on a GPU array. `f` is any
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

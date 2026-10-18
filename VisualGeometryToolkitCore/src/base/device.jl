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

# =============================================================================
# One row driver for every per-correspondence KernelAbstractions fill
# =============================================================================
#
# The LM row fills (Sampson, tangent Sampson, homography transfer, radial
# transfer) and the per-pair Sampson squares all have ONE shape: row `i` of
# `out` is a pure function of `i` and a few arrays. Each supplies that function
# in the uniform form `row!(out, i, args...)` and this driver runs it — a plain
# loop on the CPU, one generic kernel on a device backend.
#
# The CPU fast path is not a nicety: KA's CPU launch costs 1.3× (1000 rows) to
# 4.9× (100 rows) over a plain loop, and these rows run every iteration of an LM
# refit. The loop and the kernel run the SAME per-row function; only the driver
# differs.
#
# Device safety: `row!` is a function SINGLETON and the state travels in `args`
# (arrays, static matrices, tuples of them) — never a closure over non-isbits
# state, which a device kernel cannot receive. Row functions are
# `Base.@propagate_inbounds`, so the driver's `@inbounds` reaches their indexing.

@kernel function _row_kernel!(row!, out, args...)
    i = @index(Global)
    @inbounds row!(out, i, args...)
end

"""
    _foreach_row!(row!, out, n, args...) -> out

Run `row!(out, i, args...)` for `i in 1:n` on the backend that owns `out`: a
plain `@inbounds` loop on `KA.CPU()`, else one `_row_kernel!` launch
over `ndrange = n` followed by a synchronize.
"""
function _foreach_row!(row!::F, out, n::Integer, args::Vararg{Any, N}) where {F, N}
    backend = get_backend(out)
    if backend isa KA.CPU
        @inbounds for i in 1:n
            row!(out, i, args...)
        end
        return out
    end
    _row_kernel!(backend)(row!, out, args...; ndrange = n)
    KA.synchronize(backend)
    return out
end

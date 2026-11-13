# The GPU becomes the family's default device once CUDA is loaded AND
# functional. `src/` cannot name `CUDABackend` (CUDA is a weakdep), so this
# registers the resolver at load time. It must be an ASSIGNMENT in `__init__`,
# not a method definition: overwriting `default_device` here would be method
# overwriting, which Julia refuses during extension precompilation.
module VisualGeometryToolkitCoreCUDAExt
    import VisualGeometryToolkitCore as VgtkCore

using CUDA: CUDA, CUDABackend
import KernelAbstractions as KA
import VisualGeometryToolkitCore

# CUDA's memory queries live in the package that defines the backend: `CUDA` (5.x) or `CUDACore` (6.x); the old
# `available_memory` (gone in 6) returned typemax through a bare `catch`, so chunking switched off silently.
_cuda_function(name::Symbol) = getfield(Base.moduleroot(parentmodule(CUDABackend)), name)

# Every CUDA launch compiles with `always_inline = true`: the stack's kernels are
# deep call trees of small `@inline`-less functions (a lens's `project`, a static
# LM, a spline's window sums) that GPUCompiler otherwise leaves as device calls,
# each with its stack frame. `OnBackend` launches on this value whichever
# `CUDABackend` the caller holds (`default_device()`, `get_backend(::CuArray)`,
# one built by hand), so one kernel on the same data compiles once, one way.
VgtkCore._launch_backend(b::CUDABackend) = CUDABackend(; prefer_blocks = b.prefer_blocks, always_inline = true)

function __init__()
    VgtkCore._DEVICE_RESOLVER[] =
        () -> CUDA.functional() ? VgtkCore._launch_backend(CUDABackend()) : KA.CPU()
    VgtkCore._DEVICE_MEMORY[] = (;
        free = dev -> dev isa CUDABackend ? Base.invokelatest(_cuda_function(:free_memory)) : nothing,
        reclaim = dev -> (dev isa CUDABackend && Base.invokelatest(_cuda_function(:reclaim)); nothing),
        names = () -> CUDA.functional() ? String[string(CUDA.name(d)) for d in CUDA.devices()] : String[],
    )
    return nothing
end

end # module

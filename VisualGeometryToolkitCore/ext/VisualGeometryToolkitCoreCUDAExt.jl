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

# The default GPU backend compiles every kernel with `always_inline = true`: the
# stack's kernels are deep call trees of small `@inline`-less functions (a lens's
# `project`, a static LM, a spline's window sums) that GPUCompiler otherwise
# leaves as device calls, each with its stack frame. A caller who names a device
# supplies their own backend value, which `OnBackend` keeps as it is.
function __init__()
    VgtkCore._DEVICE_RESOLVER[] =
        () -> CUDA.functional() ? CUDABackend(; always_inline = true) : KA.CPU()
    VgtkCore._DEVICE_MEMORY[] = (;
        free = dev -> dev isa CUDABackend ? Base.invokelatest(_cuda_function(:free_memory)) : nothing,
        reclaim = dev -> (dev isa CUDABackend && Base.invokelatest(_cuda_function(:reclaim)); nothing),
        names = () -> CUDA.functional() ? String[string(CUDA.name(d)) for d in CUDA.devices()] : String[],
    )
    return nothing
end

# `device_resident_threads` on the CUDA backend (Core's `_resident_threads`, one method per backend): the threads of a
# KernelAbstractions kernel resident on the device at once: the kernel compiled as the CUDA backend
# compiles it for this launch (CUDA's KernelAbstractions backend, `(::KA.Kernel{CUDABackend})(args...; ndrange)`: the
# same context, `always_inline` and static `maxthreads`), its block size the launch's (static, or the occupancy
# choice a dynamic workgroup takes, `prefer_blocks` included), then CUDA's occupancy of that function at that block
# size — the registers and shared memory it uses — times the block size times the multiprocessors.
function VgtkCore._resident_threads(backend::CUDABackend, obj::KA.Kernel, args, ndrange)
    nd, _, iterspace, _ = KA.launch_config(obj, ndrange, nothing)
    ctx = KA.mkcontext(obj, nd, iterspace)
    static = KA.workgroupsize(obj) <: KA.StaticSize
    maxthreads = static ? prod(KA.get(KA.workgroupsize(obj))) : nothing
    k = CUDA.@cuda launch = false always_inline = backend.always_inline maxthreads = maxthreads obj.f(ctx, args...)
    threads = if static
        maxthreads
    else
        config = CUDA.launch_configuration(k.fun; max_threads = prod(nd))
        backend.prefer_blocks ? cld(prod(nd), max(cld(prod(nd), min(prod(nd), config.threads)), config.blocks)) : config.threads
    end
    sms = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    return CUDA.active_blocks(k.fun, threads) * threads * sms
end

end # module

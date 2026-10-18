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

function __init__()
    VgtkCore._DEVICE_RESOLVER[] =
        () -> CUDA.functional() ? CUDABackend() : KA.CPU()
    return nothing
end

end # module

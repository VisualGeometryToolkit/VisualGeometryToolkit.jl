# =============================================================================
# ONNX Runtime sessions: the execution provider and the thread pool
# =============================================================================
#
# One session builder for the package's networks (LoMa's, loma.jl). It picks the
# execution provider (`:auto`, `:cpu` or `:cuda`) and sizes ONNX Runtime's thread
# pools from the Julia process's threads.
#
# CUDA. ONNXRunTime's GPU build (onnxruntime-linux-x64-gpu_cuda12) links CUDA's
# 12.x runtime (libcudart.so.12, libcublas(Lt).so.12, libcurand.so.10) and
# dlopens cuDNN 9 (libcudnn.so.9) by soname; it does not take CUDA.jl's runtime,
# which is 13.x on a CUDA 13 driver. So the CUDA path loads its own: the CUDA 12
# builds of CUDA_Runtime_jll's and CUDNN_jll's artifacts, chosen here by their
# `cuda` platform tag (not by the JLLs' host selection, which follows CUDA.jl),
# downloaded on first use (both JLLs' artifacts are lazy) and dlopened by path
# before the session is created, so the provider's sonames resolve to them. The
# two JLL *modules* are never loaded: loading CUDNN_jll would dlopen the build of
# the host's CUDA (13 on a CUDA 13 driver) under the same soname libcudnn.so.9,
# which is why this is no package extension. They are weak dependencies, pinned
# by [compat], located in the active environment by UUID: an environment that
# wants the GPU lists them (`CUDA_Runtime_jll`, `CUDNN_jll`); one that does not
# never resolves or downloads them. CUDA 12's libraries and CUDA.jl's 13 ones
# coexist in one process (different sonames); a cuDNN or CUDA 12 library already
# loaded from another file (cuDNN.jl, a system install) is refused by name.
#
# Threads. ONNX Runtime sizes its intra-op pool from the machine (all cores) and
# pins each thread to a core when it chooses the size itself; inside a Slurm
# cgroup that oversubscribes the allotment and the pinning fails
# (`pthread_setaffinity_np failed`). The size here is the caller's, by default
# `Threads.nthreads()`, the stack's measure of the process's allotment (Slurm
# jobs start Julia with `--threads=$SLURM_CPUS_PER_TASK`, VisualGeometryToolkitEval's
# launcher). An explicit size disables ONNX Runtime's pinning. The graph runs
# sequentially (ONNX Runtime's default execution mode), so the inter-op pool is
# set to 1: no thread beyond the intra-op pool's.
# =============================================================================

using Libdl: Libdl
using Artifacts: load_artifacts_toml, unpack_platform
using Base.BinaryPlatforms: HostPlatform, arch, os, libc, tags
using Pkg.Artifacts: ensure_artifact_installed

const _CUDA_RUNTIME_JLL = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
const _CUDNN_JLL = Base.PkgId(Base.UUID("62b44479-cb7b-5706-934f-f13b2eb2e645"), "CUDNN_jll")

# The CUDA 12 runtime libraries ONNX Runtime's CUDA provider links or opens, in load order (a library after those it needs).
const _CUDA_RUNTIME_LIBRARIES = ("libcudart.so.12", "libnvJitLink.so.12", "libnvrtc.so.12", "libcublasLt.so.12",
                                 "libcublas.so.12", "libcurand.so.10", "libcufft.so.11")
const _CUDNN_LIBRARIES = ("libcudnn_graph.so.9", "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
                          "libcudnn_engines_tensor_ir.so.9", "libcudnn_heuristic.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9",
                          "libcudnn_adv.so.9", "libcudnn.so.9")

"""
    OnnxCudaStatus

Whether ONNX Runtime's CUDA provider can run in this process, probed once:
`available`, and `reason` (empty when available) naming what is missing. With
the provider available, `driver` (the CUDA driver API version), `cuda_runtime`
and `cudnn` (the versions of the libraries loaded for it) are set.
"""
struct OnnxCudaStatus
    available::Bool
    reason::String
    driver::Union{Nothing, VersionNumber}
    cuda_runtime::Union{Nothing, VersionNumber}
    cudnn::Union{Nothing, VersionNumber}
end
_cuda_unavailable(reason, driver = nothing) = OnnxCudaStatus(false, reason, driver, nothing, nothing)

const _ONNX_CUDA_STATUS = Ref{Union{Nothing, OnnxCudaStatus}}(nothing)
const _ONNX_CUDA_LOCK = ReentrantLock()

"""
    onnx_cuda_status() -> OnnxCudaStatus

ONNX Runtime's CUDA provider in this process: probed (and its libraries loaded)
on the first call, the same answer after. `:auto` runs on CUDA exactly when
`onnx_cuda_status().available`.
"""
function onnx_cuda_status()
    @lock _ONNX_CUDA_LOCK begin
        s = _ONNX_CUDA_STATUS[]
        s === nothing ? (_ONNX_CUDA_STATUS[] = _probe_onnx_cuda()) : s
    end
end

function _probe_onnx_cuda()
    Sys.islinux() && arch(HostPlatform()) == "x86_64" ||
        return _cuda_unavailable("ONNX Runtime's CUDA build is set up here for x86-64 Linux only")
    driver = _cuda_driver()
    driver isa String && return _cuda_unavailable(driver)
    driver >= v"12" || return _cuda_unavailable("the CUDA driver supports CUDA $driver; CUDA 12 needs a driver of at least 525.60", driver)
    rt = _jll_cuda12_artifact(_CUDA_RUNTIME_JLL, "CUDA_Runtime", v -> v >= ORT.cuda_runtime_supported_version)
    rt isa String && return _cuda_unavailable(rt, driver)
    dnn = _jll_cuda12_artifact(_CUDNN_JLL, "CUDNN", _ -> true)
    dnn isa String && return _cuda_unavailable(dnn, driver)
    libs = vcat([joinpath(rt.dir, "lib", l) for l in _CUDA_RUNTIME_LIBRARIES],
                [joinpath(dnn.dir, "lib", l) for l in _CUDNN_LIBRARIES])
    for l in libs
        isfile(l) || return _cuda_unavailable("$(basename(l)) is not in $(dirname(l))", driver)
        h = Libdl.dlopen(basename(l), Libdl.RTLD_NOLOAD; throw_error = false)
        h === nothing && continue
        loaded = Libdl.dlpath(h)
        realpath(loaded) == realpath(l) || return _cuda_unavailable(
            "$(basename(l)) is already loaded from $loaded (cuDNN.jl, or a system CUDA 12), which ONNX Runtime's CUDA " *
            "provider would take instead of $(l); load neither in this process", driver)
    end
    # local: the provider's and cuDNN's lookups by soname find them loaded; their symbols stay out of the global scope
    for l in libs
        Libdl.dlopen(l, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL)
    end
    # ONNX Runtime's GPU build itself (an artifact of ONNXRunTime's, downloaded on first use), and one of its own kernels run
    # on the device (ONNXRunTime's one-op test graph, x + 1): a device its build has no code for fails here, not in a run
    try
        ORT.CAPI.libptr(:cuda)
        s = _onnx_session(ORT.testdatapath("increment2x3.onnx"), :cuda, 1)
        only(values(s(Dict("input" => zeros(Float32, 2, 3))))) == ones(Float32, 2, 3) || error("x + 1 on the device is not 1")
    catch err
        return _cuda_unavailable("ONNX Runtime's CUDA build does not run here: $(sprint(showerror, err))", driver)
    end
    return OnnxCudaStatus(true, "", driver, rt.version, dnn.version)
end

# The CUDA driver API version, or why there is none to use.
function _cuda_driver()
    h = Libdl.dlopen("libcuda.so.1"; throw_error = false)
    h === nothing && return "no NVIDIA driver (libcuda.so.1 is not on the library path)"
    rc = ccall(Libdl.dlsym(h, :cuInit), Cint, (Cuint,), 0)
    visible = get(ENV, "CUDA_VISIBLE_DEVICES", "unset")
    rc == 100 && return "no CUDA device is visible (CUDA_VISIBLE_DEVICES=$visible)"   # CUDA_ERROR_NO_DEVICE
    rc == 0 || return "the CUDA driver failed to initialize (cuInit error $rc; CUDA_VISIBLE_DEVICES=$visible)"
    n = Ref{Cint}(0)
    ccall(Libdl.dlsym(h, :cuDeviceGetCount), Cint, (Ref{Cint},), n)
    n[] > 0 || return "no CUDA device is visible (CUDA_VISIBLE_DEVICES=$visible)"
    v = Ref{Cint}(0)
    ccall(Libdl.dlsym(h, :cuDriverGetVersion), Cint, (Ref{Cint},), v)
    return VersionNumber(v[] ÷ 1000, (v[] % 1000) ÷ 10)
end

# The newest CUDA 12 build (`cuda` tag 12.x for which `accept(v)`) of `jll`'s artifact `name` for this host, installed:
# (; dir, version): the artifact's CUDA tag for the runtime, the library's own (libcudnn.so.9.x.y) for cuDNN; or why there
# is none.
function _jll_cuda12_artifact(jll::Base.PkgId, name, accept)
    entry = Base.locate_package(jll)
    entry === nothing && return "$(jll.name) is not in the active environment (add CUDA_Runtime_jll and CUDNN_jll to it " *
                                "for ONNX Runtime's CUDA provider)"
    root = dirname(dirname(entry))
    toml = joinpath(root, "Artifacts.toml")
    host = HostPlatform()
    best = nothing
    for meta in load_artifacts_toml(toml)[name]
        p = unpack_platform(meta, name, toml)
        p === nothing && continue
        (arch(p), os(p), libc(p)) == (arch(host), os(host), libc(host)) && !haskey(tags(p), "cuda_platform") || continue
        v = tryparse(VersionNumber, get(tags(p), "cuda", ""))
        (v !== nothing && v.major == 12 && accept(v)) || continue
        (best === nothing || v > best[2]) && (best = (meta, v))
    end
    best === nothing && return "$(jll.name) at $root has no CUDA 12 build of $name for this host"
    dir = try
        ensure_artifact_installed(name, best[1], toml)
    catch err
        return "$(jll.name)'s CUDA 12 build of $name did not install: $(sprint(showerror, err))"
    end
    name == "CUDA_Runtime" && return (; dir, version = best[2])
    for f in readdir(joinpath(dir, "lib"))
        m = Base.match(r"^libcudnn\.so\.(\d+\.\d+\.\d+)$", f)
        m === nothing || return (; dir, version = VersionNumber(something(m[1])))
    end
    return "$(jll.name)'s CUDA 12 build of $name at $dir holds no libcudnn.so.9.x.y"
end

"""
    resolve_execution_provider(execution_provider) -> Symbol

The ONNX Runtime provider `execution_provider` runs on here: `:auto` is `:cuda`
when [`onnx_cuda_status`](@ref) finds the CUDA provider available, and `:cpu`
with a warning naming why otherwise; `:cuda` is `:cuda`, or an error naming why
it is unavailable; `:cpu` is `:cpu`. Anything else is an `ArgumentError`. What
[`LoMaDescriptor`](@ref) and [`LoMaMatcher`](@ref) run on; a caller that keys a
cache on the provider resolves it first.
"""
function resolve_execution_provider(execution_provider::Symbol, status::Function = onnx_cuda_status)
    # `status` is a parameter so a test can hand in an unavailable one
    execution_provider === :cpu && return :cpu
    execution_provider in (:auto, :cuda) || throw(ArgumentError(
        "execution_provider = $(repr(execution_provider)) is not :auto, :cpu or :cuda"))
    s = status()
    s.available && return :cuda
    execution_provider === :cuda && error("execution_provider = :cuda: ONNX Runtime's CUDA provider is unavailable: $(s.reason)")
    @warn "ONNX Runtime's CUDA provider is unavailable, running on the CPU: $(s.reason)"
    return :cpu
end

# ONNXRunTime's `load_inference`, with the session options it does not take: the provider resolved, the intra-op pool
# `threads` (which also disables ONNX Runtime's core pinning), the inter-op pool 1.
function _onnx_session(path::AbstractString, execution_provider::Symbol, threads::Integer)
    threads >= 1 || throw(ArgumentError("threads must be at least 1, got $threads"))
    capi = ORT.CAPI
    api = capi.GetApi(; execution_provider)
    env = capi.CreateEnv(api; name = "VisualGeometryToolkitFeatures")
    options = capi.CreateSessionOptions(api)
    capi.check_and_release(api, @ccall $(api.SetIntraOpNumThreads)(options::Ptr{Cvoid}, threads::Cint)::Ptr{Cvoid})
    capi.check_and_release(api, @ccall $(api.SetInterOpNumThreads)(options::Ptr{Cvoid}, 1::Cint)::Ptr{Cvoid})
    execution_provider === :cuda &&
        capi.SessionOptionsAppendExecutionProvider_CUDA(api, options, capi.OrtCUDAProviderOptions())
    session = capi.CreateSession(api, env, path, options)
    meminfo = capi.CreateCpuMemoryInfo(api)
    allocator = capi.CreateAllocator(api, session, meminfo)
    return ORT.InferenceSession(api, execution_provider, session, meminfo, allocator,
                                ORT.input_names(api, session, allocator), ORT.output_names(api, session, allocator), true)
end

# What ran a session, as plain data for a result to record.
function _onnx_provenance(execution_provider::Symbol, threads::Integer)
    api_base = ORT.CAPI.OrtGetApiBase(; execution_provider)
    p = (; execution_provider, threads, onnxruntime = ORT.CAPI.GetVersionString(api_base))
    execution_provider === :cuda || return p
    s = onnx_cuda_status()
    return (; p..., cuda_driver = string(s.driver), cuda_runtime = string(s.cuda_runtime), cudnn = string(s.cudnn))
end

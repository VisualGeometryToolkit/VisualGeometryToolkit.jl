# =============================================================================
# ONNX Runtime sessions: the execution provider and the thread pool
# =============================================================================
#
# One session builder for the package's networks (LoMa's, loma.jl) and any caller's (`onnx_session`). It picks the
# execution provider (`:auto`, `:cpu` or `:cuda`) and sizes ONNX Runtime's thread
# pools from the Julia process's threads.
#
# CUDA. Microsoft's GPU builds of ONNX Runtime (onnxruntime-linux-x64-gpu_cuda12
# and _cuda13, 1.30 and 1.31; the CUDA plugin EP too) carry device code for
# sm_60/70/75/80/86/89/90 and sm_120 only, no sm_100 and no PTX (their CI sets
# CMAKE_CUDA_ARCHITECTURES "...;90-real;120-real"), so they do not run on a
# B200 (cudaErrorNoKernelImageForDevice). The CUDA path therefore runs conda-
# forge's build of the same release (onnxruntime-cpp, cuda129: "70-real;...;
# 90-real;100-real;120"), sm_89 and sm_100 alike, and through sm_100's code any
# 10.x device and through sm_120's any 12.x one. Pkg cannot unpack a .conda
# (a zip of zstd tarballs), so the package installs it itself: the four
# conda-forge files below (the build and the shared libraries it links:
# protobuf, abseil, re2), each pinned by sha256, unpacked with Julia's own 7z
# and merged into one artifact, whose tree hash Artifacts.toml pins
# (`onnxruntime_cuda`, lazy: installed by the first CUDA probe, never without a
# device). Their RPATH is $ORIGIN, so the merged lib/ resolves itself;
# libstdc++ is Julia's (they need GLIBCXX 3.4.31).
#
# The build links CUDA's 12.x runtime (libcudart.so.12, libcublas(Lt).so.12)
# and dlopens cuDNN 9 (libcudnn.so.9) and CUDA's other libraries by soname; it
# does not take CUDA.jl's runtime, which is 13.x on a CUDA 13 driver. So the
# CUDA path loads its own: the CUDA 12 builds of CUDA_Runtime_jll's and
# CUDNN_jll's artifacts, chosen here by their `cuda` platform tag (not by the
# JLLs' host selection, which follows CUDA.jl), downloaded on first use (both
# JLLs' artifacts are lazy) and dlopened by path before the session is created,
# so the provider's sonames resolve to them. The two JLL *modules* are never
# loaded: loading CUDNN_jll would dlopen the build of the host's CUDA (13 on a
# CUDA 13 driver) under the same soname libcudnn.so.9, which is why this is no
# package extension. They are weak dependencies, pinned by [compat], located in
# the active environment by UUID: an environment that wants the GPU lists them
# (`CUDA_Runtime_jll`, `CUDNN_jll`); one that does not never resolves or
# downloads them. CUDA 12's libraries and CUDA.jl's 13 ones coexist in one
# process (different sonames); a cuDNN or CUDA 12 library already loaded from
# another file (cuDNN.jl, a system install) is refused by name. The CPU path is
# ONNXRunTime's own (Microsoft's CPU build).
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

import ONNXRunTime as ORT   # the sessions (here and loma.jl)
using Libdl: Libdl
using Artifacts: artifact_hash, load_artifacts_toml, unpack_platform
using Base.BinaryPlatforms: HostPlatform, arch, os, libc, tags, triplet
using Pkg.Artifacts: artifact_exists, artifact_path, create_artifact, ensure_artifact_installed, remove_artifact
using Tar: Tar
using p7zip_jll: p7zip

const _CUDA_RUNTIME_JLL = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
const _CUDNN_JLL = Base.PkgId(Base.UUID("62b44479-cb7b-5706-934f-f13b2eb2e645"), "CUDNN_jll")

# The CUDA 12 runtime libraries ONNX Runtime's CUDA provider links or opens, in load order (a library after those it needs).
const _CUDA_RUNTIME_LIBRARIES = ("libcudart.so.12", "libnvJitLink.so.12", "libnvrtc.so.12", "libcublasLt.so.12",
                                 "libcublas.so.12", "libcurand.so.10", "libcufft.so.11", "libcusparse.so.12")
const _CUDNN_LIBRARIES = ("libcudnn_graph.so.9", "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
                          "libcudnn_engines_tensor_ir.so.9", "libcudnn_heuristic.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9",
                          "libcudnn_adv.so.9", "libcudnn.so.9")

# ONNX Runtime's CUDA build: conda-forge's files (name, sha256), merged into the artifact `onnxruntime_cuda` of
# Artifacts.toml. The first is the build (its CUDA dependencies, cuda-cudart >= 12.9.79, cudnn >= 9.10, are the JLLs'),
# the others the shared libraries it links, at the versions its own metadata requires (libprotobuf 7.35.1, libabseil
# 20260817, libre2-11).
const _ORT_CUDA_CHANNEL = "https://conda.anaconda.org/conda-forge/linux-64"
const _ORT_CUDA_FILES = (
    ("onnxruntime-cpp-1.31.1-h5ee9576_201_cuda129", "a867fd86014ee3dbdc0eafcdd1177d6c3d21f7b48c181b5e24e79e5e9f3d39f9"),
    ("libprotobuf-7.35.1-h8a00109_4", "132df0c122cc211ad51a4622036a2a31e4751fcc3702d3663d95ae27f98f9ea7"),
    ("libabseil-20260817.0-cxx17_hbb87c76_2", "478f0a05d1310dbf9e9a1887aeb482ebc0cb8fbe4ca9e206aec11479c0da89f9"),
    ("libre2-11-2025.11.05-h02ca46f_4", "0c49508a9525f89e9195155e106ae8c9fb57ddf46e5111f98d7f3227976c6a15"),
)
const _ARTIFACTS_TOML = joinpath(dirname(@__DIR__), "Artifacts.toml")

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

# The build, its CUDA 12 runtime and its cuDNN that the probe loads: (; build, ort, cuda_runtime = (; dir, version),
# cudnn = (; dir, version)), each `dir` a lib/ directory; or why there are none. `_package_cuda_source` is the package's own
# (the conda-forge artifact, the JLLs' CUDA 12 builds); `_conda_prefix_cuda_source(prefix)` an installed conda environment
# holding the same build and its CUDA libraries (a pixi environment's `.pixi/envs/default`).
function _package_cuda_source()
    rt = _jll_cuda12_artifact(_CUDA_RUNTIME_JLL, "CUDA_Runtime", v -> v >= v"12.9")   # the build's cuda-cudart >= 12.9.79
    rt isa String && return rt
    dnn = _jll_cuda12_artifact(_CUDNN_JLL, "CUDNN", _ -> true)
    dnn isa String && return dnn
    ort = _onnxruntime_cuda_artifact()
    ort isa String && return ort
    return (; build = "conda-forge $(first(first(_ORT_CUDA_FILES)))", ort = joinpath(ort.dir, "lib"),
            cuda_runtime = (; dir = joinpath(rt.dir, "lib"), rt.version), cudnn = (; dir = joinpath(dnn.dir, "lib"), dnn.version))
end

function _conda_prefix_cuda_source(prefix)
    lib = joinpath(prefix, "lib")
    meta = filter(f -> startswith(f, "onnxruntime-cpp-") && endswith(f, ".json"), readdir(joinpath(prefix, "conda-meta")))
    isempty(meta) && return "$prefix holds no onnxruntime-cpp (conda-meta)"
    libversion(re) = (f = filter(x -> occursin(re, x), readdir(lib)); isempty(f) ? v"0" : _library_version(only(f)))
    return (; build = "conda environment $prefix: $(chopsuffix(only(meta), ".json"))", ort = lib,
            cuda_runtime = (; dir = lib, version = libversion(r"^libcudart\.so\.12\.\d+\.\d+$")),
            cudnn = (; dir = lib, version = libversion(r"^libcudnn\.so\.9\.\d+\.\d+$")))
end

const _ONNX_CUDA_BUILD = Ref("")   # the build the probe loaded, for the provenance

function _probe_onnx_cuda(source::Function = _package_cuda_source)
    Sys.islinux() && arch(HostPlatform()) == "x86_64" ||
        return _cuda_unavailable("ONNX Runtime's CUDA build is set up here for x86-64 Linux only")
    driver = _cuda_driver()
    driver isa String && return _cuda_unavailable(driver)
    driver >= v"12" || return _cuda_unavailable("the CUDA driver supports CUDA $driver; CUDA 12 needs a driver of at least 525.60", driver)
    src = source()
    src isa String && return _cuda_unavailable(src, driver)
    libs = vcat([joinpath(src.cuda_runtime.dir, l) for l in _CUDA_RUNTIME_LIBRARIES],
                [joinpath(src.cudnn.dir, l) for l in _CUDNN_LIBRARIES])
    # A library of the same soname already loaded from another file is what the provider will bind to. A CUDA library's
    # soname carries its ABI major, so one at least as new as ours serves (CUDA.jl's CUDA 13 CURAND is libcurand.so.10 as
    # CUDA 12's is, self-contained); an older one may lack symbols, and a cuDNN's soname does not say which CUDA it was
    # built for (CUDNN_jll's CUDA 13 build is libcudnn.so.9 too): both are refused by name.
    for l in libs
        isfile(l) || return _cuda_unavailable("$(basename(l)) is not in $(dirname(l))", driver)
        h = Libdl.dlopen(basename(l), Libdl.RTLD_NOLOAD; throw_error = false)
        h === nothing && continue
        loaded = realpath(Libdl.dlpath(h))
        ours = realpath(l)
        loaded == ours && continue
        cudnn = startswith(basename(l), "libcudnn")
        !cudnn && _library_version(loaded) >= _library_version(ours) && continue
        why = cudnn ? "a cuDNN built for an unknown CUDA (cuDNN.jl's?)" : "older than $ours"
        return _cuda_unavailable("$(basename(l)) is already loaded from $loaded, $why, and ONNX Runtime's CUDA provider " *
                                 "would bind to it; do not load it in this process", driver)
    end
    # local: the provider's and cuDNN's lookups by soname find them loaded; their symbols stay out of the global scope
    for l in libs
        Libdl.dlopen(l, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL)
    end
    # ONNX Runtime's CUDA build itself (ONNXRunTime's :cuda library, set to the artifact's), and one of its own kernels run
    # on the device (ONNXRunTime's one-op test graph, x + 1): a device the build has no code for fails here, not in a run
    try
        ORT.CAPI.set_lib!(joinpath(src.ort, "libonnxruntime.so"), :cuda)
        # the probe's session reserves no arena room for the detector's plans and is released here, its arena with it
        s = _onnx_session(ORT.testdatapath("increment2x3.onnx"), :cuda, 1; reserve = false)
        try
            only(values(s(Dict("input" => zeros(Float32, 2, 3))))) == ones(Float32, 2, 3) || error("x + 1 on the device is not 1")
        finally
            ORT.release(s)
        end
    catch err
        return _cuda_unavailable("ONNX Runtime's CUDA build does not run here: $(sprint(showerror, err))", driver)
    end
    _ONNX_CUDA_BUILD[] = src.build
    return OnnxCudaStatus(true, "", driver, src.cuda_runtime.version, src.cudnn.version)
end

# The directory of ONNX Runtime's CUDA build (Artifacts.toml's `onnxruntime_cuda`), installed from conda-forge's files on
# first use, as (; dir); or why there is none.
function _onnxruntime_cuda_artifact()
    hash = artifact_hash("onnxruntime_cuda", _ARTIFACTS_TOML)
    hash === nothing && return "this package has no ONNX Runtime CUDA build for $(triplet(HostPlatform()))"
    artifact_exists(hash) && return (; dir = artifact_path(hash))
    @info "Installing ONNX Runtime's CUDA build: conda-forge's $(first(first(_ORT_CUDA_FILES))) and the libraries it links " *
          "($(length(_ORT_CUDA_FILES)) files, 0.34 GB; once per depot)"
    got = try
        create_artifact(_install_conda_files!)
    catch err
        return "ONNX Runtime's CUDA build did not install: $(sprint(showerror, err))"
    end
    got == hash && return (; dir = artifact_path(hash))
    remove_artifact(got)
    return "ONNX Runtime's CUDA build unpacked to the tree $got, not Artifacts.toml's $hash"
end

# conda-forge's files, each downloaded, checked against its sha256 and its package tarball (a .conda is a zip holding
# pkg-<name>.tar.zst and info-<name>.tar.zst) unpacked into `dir`, one tree. Julia's own 7z reads the zip and the zstd.
function _install_conda_files!(dir, files = _ORT_CUDA_FILES, channel = _ORT_CUDA_CHANNEL)
    z7 = p7zip()   # bound outside the commands: a name read only inside a backtick literal is invisible to static analysis
    mktempdir() do tmp
        for (name, digest) in files
            conda = _download_verified("$channel/$name.conda", digest, joinpath(tmp, "$name.conda"))
            run(pipeline(`$z7 x -y -o$tmp $conda pkg-$name.tar.zst`; stdout = devnull))
            pkg = joinpath(tmp, name)
            open(io -> Tar.extract(io, pkg), `$z7 x -so $(joinpath(tmp, "pkg-$name.tar.zst"))`)
            _merge_tree!(pkg, dir)
        end
    end
    return dir
end

# Move the tree `src` into `dst`, merging directories; two packages holding the same file is an error.
function _merge_tree!(src, dst)
    for name in readdir(src)
        s, d = joinpath(src, name), joinpath(dst, name)
        if isdir(s) && !islink(s) && isdir(d) && !islink(d)
            _merge_tree!(s, d)
        elseif ispath(d) || islink(d)
            error("two conda packages hold $d")
        else
            mv(s, d)
        end
    end
end

# A shared library's version from its file name (libcurand.so.10.3.10.19 → v"10.3.10"); v"0" without one.
function _library_version(path)
    m = Base.match(r"\.so\.(\d+(?:\.\d+){0,2})", basename(path))
    return m === nothing ? v"0" : VersionNumber(something(m[1]))
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
        occursin(r"^libcudnn\.so\.\d+\.\d+\.\d+$", f) && return (; dir, version = _library_version(f))
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

# ONNX Runtime's CUDA provider allocates outside the ASIFT detector's device pool. Each CUDA session's arena is capped
# (`gpu_mem_limit`, grown by what each run requests: ONNX Runtime's kSameAsRequested), and the detector's plans size their
# resident warps from the free device memory less the room the live CUDA sessions' arenas may still take
# (`_onnx_cuda_reserve`), so the two never compete for the device and no plan is released for a session's run. The default
# cap holds a LoMa session on a panorama frame's green cells; a run that needs more is ONNX Runtime's out-of-memory error,
# naming the limit.
#
# The room of one session is its cap less what its arena already holds: the device's free memory already excludes the held
# part, so reserving the whole cap would count it twice. What an arena holds is measured around each of the package's own
# runs and at its creation (`_onnx_run`: the drop in the device's free memory across the run or the creation, the arena
# only growing; it includes the CUDA context and library workspaces, which the budget's slack covers), so it assumes no other
# allocation on the device during a run (the detector and LoMa run in turn) and that the session runs on the device the
# default backend queries (device 0, ONNX Runtime's default); with no device query nothing is measured and the whole cap
# stays reserved. An arena is the session's handle (its `OrtSession`), held weakly: a session collected or released
# (`ONNXRunTime.release`) reserves nothing from then on.
const _ONNX_CUDA_ARENA_BYTES = 8 * 2^30

mutable struct _CudaArena
    const session::WeakRef   # the session's handle (an `OrtSession`)
    const cap::Int
    held::Int                # bytes the arena holds, measured; at most `cap`
end
const _ONNX_CUDA_ARENAS = _CudaArena[]
const _ONNX_ARENA_LOCK = ReentrantLock()

_session_alive(::Any) = true
_session_alive(o::ORT.CAPI.OrtSession) = ORT.CAPI.isalive(o)
_arena_alive(a::_CudaArena) = (o = a.session.value; o !== nothing && _session_alive(o))
# The arenas still in use, the others dropped. Called under the lock.
_live_arenas!() = filter!(_arena_alive, _ONNX_CUDA_ARENAS)

# A new arena of `cap` bytes for the session `handle`, nothing held yet.
function _reserve_arena!(handle, cap::Integer)
    a = _CudaArena(WeakRef(handle), Int(cap), 0)
    @lock _ONNX_ARENA_LOCK push!(_live_arenas!(), a)
    return a
end

# The arena of `handle` grew by `bytes` (a drop in free memory; a rise is not a shrink); nothing for an untracked handle.
function _arena_grew!(handle, bytes::Integer)
    @lock _ONNX_ARENA_LOCK for a in _live_arenas!()
        a.session.value === handle && (a.held = clamp(a.held + max(Int(bytes), 0), 0, a.cap))
    end
    return nothing
end

# The room the live CUDA sessions' arenas may still take: each cap less what that arena holds.
_onnx_cuda_reserve() = @lock _ONNX_ARENA_LOCK sum(a -> a.cap - a.held, _live_arenas!(); init = 0)

# The default backend's free device memory, or `nothing` on the host or without a query.
function _onnx_device_free()
    d = default_device()
    d isa KA.CPU && return nothing
    f = device_free_bytes(d)
    return f == typemax(Int) ? nothing : f
end

# `f()` with the device's free memory (`free`: bytes, or nothing without a query) read around it: its value and the
# drop in free memory, `nothing` without a query. The one measured arena growth (a run's, a session's creation).
function _measure_growth(f, free::Function)
    before = free()
    out = f()
    after = before === nothing ? nothing : free()
    return out, (after === nothing ? nothing : before - after)
end

# A run of the session `s` on `inputs`; on a tracked CUDA session the arena's growth is measured around it (`free`, the
# device's free memory or nothing).
function _onnx_run(s::ORT.InferenceSession, inputs, free::Function = _onnx_device_free)
    s.execution_provider === :cuda || return s(inputs)
    out, grown = _measure_growth(() -> s(inputs), free)
    grown === nothing || _arena_grew!(s.session, grown)
    return out
end

"""
    onnx_session(path; execution_provider = :auto, threads = Threads.nthreads(), gpu_mem_limit = 8 * 2^30)
        -> ONNXRunTime.InferenceSession

An ONNX Runtime session of the model file `path`, built as this package builds
its own networks' (LoMa's): ONNXRunTime's `load_inference` with the session
options it does not take. `execution_provider` is resolved by
[`resolve_execution_provider`](@ref) (`:auto` runs on CUDA when
[`onnx_cuda_status`](@ref) finds a device, on the CPU with a warning naming why
otherwise; `:cuda` is an error where CUDA is unavailable; `:cpu` is the CPU),
so a CUDA session runs ONNX Runtime's CUDA build this package installs, which
has device code for every device from sm_70 to sm_120, the B200's sm_100
among them (Microsoft's GPU builds lack sm_100). The session's
`execution_provider` field holds the resolved provider.

ONNX Runtime's intra-op pool has `threads` threads (its core pinning off), its
inter-op pool one. On CUDA, the session's device arena is capped at
`gpu_mem_limit` bytes and grows by what each run requests; a run that needs
more is ONNX Runtime's out-of-memory error. The ASIFT detector's device plans
leave free the room the process's live CUDA sessions may still take (each cap
less what its arena holds, measured around the package's own runs; a session
released or collected reserves nothing), so a session and the detector share
the device without either releasing the other's memory. The plans cached
before a CUDA session is created are released once, then.
On the CPU `gpu_mem_limit` is ignored.
"""
function onnx_session(path::AbstractString; execution_provider::Symbol = :auto, threads::Integer = Threads.nthreads(),
                      gpu_mem_limit::Integer = _ONNX_CUDA_ARENA_BYTES)
    gpu_mem_limit >= 1 || throw(ArgumentError("gpu_mem_limit must be at least 1 byte, got $gpu_mem_limit"))
    return _onnx_session(path, resolve_execution_provider(execution_provider), threads, gpu_mem_limit)
end

# The session on a resolved provider (`onnx_session`; the CUDA probe builds its test session here, before a provider can
# be resolved, with `reserve = false`): the intra-op pool `threads`, the inter-op pool 1; on CUDA the arena capped at
# `gpu_mem_limit`, the arena tracked in the plans' reserve, and the detector's plans cached before the session (sized
# without it) released once.
function _onnx_session(path::AbstractString, execution_provider::Symbol, threads::Integer,
                       gpu_mem_limit::Integer = _ONNX_CUDA_ARENA_BYTES; reserve::Bool = true)
    threads >= 1 || throw(ArgumentError("threads must be at least 1, got $threads"))
    capi = ORT.CAPI
    api = capi.GetApi(; execution_provider)
    env = capi.CreateEnv(api; name = "VisualGeometryToolkitFeatures")
    options = capi.CreateSessionOptions(api)
    capi.check_and_release(api, @ccall $(api.SetIntraOpNumThreads)(options::Ptr{Cvoid}, threads::Cint)::Ptr{Cvoid})
    capi.check_and_release(api, @ccall $(api.SetInterOpNumThreads)(options::Ptr{Cvoid}, 1::Cint)::Ptr{Cvoid})
    if execution_provider === :cuda
        cuda = capi.OrtCUDAProviderOptions(; gpu_mem_limit = Int(gpu_mem_limit), arena_extend_strategy = 1)
        capi.SessionOptionsAppendExecutionProvider_CUDA(api, options, cuda)
    end
    if execution_provider === :cuda && reserve
        session, grown = _measure_growth(() -> capi.CreateSession(api, env, path, options), _onnx_device_free)
        _reserve_arena!(session, gpu_mem_limit)
        grown === nothing || _arena_grew!(session, grown)   # the initializers the arena took
        clear_asift_plans!()
    else
        session = capi.CreateSession(api, env, path, options)
    end
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
    return (; p..., onnxruntime_build = _ONNX_CUDA_BUILD[], cuda_driver = string(s.driver),
            cuda_runtime = string(s.cuda_runtime), cudnn = string(s.cudnn))
end

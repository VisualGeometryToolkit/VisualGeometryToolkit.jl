# =============================================================================
# The GPU test tier — ONE decision for every stack package's suite
# =============================================================================
#
# GPU first: a package's kernels and device paths are its main paths, so its GPU
# items run whenever the machine can run them, and never quietly record zero
# tests when it cannot. Hoisted from RadiometricCamera's test/gpu_decision.jl
# (RC 0.8.0) so every package takes the decision the same way:
#
#   - the RUNNER (test/runtests.jl) decides once, `gpu_tier(prefix)`: a
#     functional accelerator is Core's `default_device()` (the vendor extension's
#     resolver, so this file names no vendor), `<PREFIX>_TEST_GPU=0` opts out,
#     and the device must have `min_free_bytes` free (GPU items allocate
#     per-view device caches; they failed "Out of GPU memory" next to running
#     experiments). A yes is handed to the GPU tier's workers as
#     `<PREFIX>_GPU_DECIDED=1`; a no says why (`@info`).
#   - each GPU item starts with `require_gpu(prefix)`: the worker re-checks
#     (`gpu_problem(prefix)`) and ERRORS with the reason when the runner decided
#     yes and the device is gone, so a GPU tier never passes on the CPU path.
# =============================================================================

_gpu_env(prefix::AbstractString, what::AbstractString) = string(prefix, "_", what)

"""
    gpu_tier(prefix::AbstractString; min_free_bytes::Integer = 20 * 2^30) -> Bool

The test runner's ONE decision whether to run a package's GPU tier: `true` when
an accelerator is functional ([`default_device`](@ref) is not `KA.CPU()`, so the
vendor package — `CUDA` — must be loaded), the environment does not opt out
(`<prefix>_TEST_GPU=0`) and the device has at least `min_free_bytes` free
([`device_free_bytes`](@ref)). A yes is handed to the GPU tier's worker
processes as `<prefix>_GPU_DECIDED=1` (set here, in `ENV`); a no logs why
(`@info`) and leaves it unset. Called once, in `test/runtests.jl`, before the
GPU tier; each GPU item then starts with [`require_gpu`](@ref)`(prefix)`.
"""
function gpu_tier(prefix::AbstractString; min_free_bytes::Integer = 20 * 2^30)
    dev = default_device()
    functional = !(dev isa KA.CPU)
    opted_out = get(ENV, _gpu_env(prefix, "TEST_GPU"), "1") == "0"
    free = functional ? device_free_bytes(dev) : 0
    ok = functional && !opted_out && free >= min_free_bytes
    if ok
        ENV[_gpu_env(prefix, "GPU_DECIDED")] = "1"
    else
        @info "GPU test tier skipped" functional opted_out free_GiB = round(free / 2^30; digits = 1) min_free_GiB = round(min_free_bytes / 2^30; digits = 1)
    end
    return ok
end

"""
    gpu_problem(prefix::AbstractString; min_free_bytes::Integer = 20 * 2^30) -> Union{Nothing, String}

On a GPU-tier worker: `nothing` when the runner decided to run the tier
(`<prefix>_GPU_DECIDED=1`, [`gpu_tier`](@ref)) and the accelerator is still
functional with `min_free_bytes` free; otherwise the reason the GPU items
cannot run, as a sentence.
"""
function gpu_problem(prefix::AbstractString; min_free_bytes::Integer = 20 * 2^30)
    decided = _gpu_env(prefix, "GPU_DECIDED")
    get(ENV, decided, "0") == "1" ||
        return "the GPU items run only in the GPU tier the test runner decided on (`gpu_tier`; $(decided)=1 is not set)"
    dev = default_device()
    dev isa KA.CPU && return "the runner decided to run the GPU tier, but no accelerator is functional on this worker (is CUDA loaded?)"
    free = device_free_bytes(dev)
    free >= min_free_bytes ||
        return "the runner decided to run the GPU tier, but this worker sees only $(round(free / 2^30; digits = 1)) GiB free (< $(round(min_free_bytes / 2^30; digits = 1)) GiB)"
    return nothing
end

"""
    require_gpu(prefix::AbstractString; min_free_bytes::Integer = 20 * 2^30) -> KA.Backend

The first line of every GPU test item: the accelerator's backend
([`default_device`](@ref)) when [`gpu_problem`](@ref)`(prefix)` finds none, and
an error with the reason otherwise — never a silent skip, so a GPU tier cannot
pass on the CPU path.
"""
function require_gpu(prefix::AbstractString; min_free_bytes::Integer = 20 * 2^30)
    p = gpu_problem(prefix; min_free_bytes)
    p === nothing || error(p)
    return default_device()
end

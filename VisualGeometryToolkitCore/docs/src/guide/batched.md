# Batches

```@meta
CurrentModule = VisualGeometryToolkitCore
```

A batch of points goes through the same per-point function a single point
does, by broadcasting it. [`pointwise!`](@ref)`(f, out, xs...)` is
`out .= f.(xs...)`: a SIMD loop on a host array and one compiled kernel on a
GPU array (a `CuArray`, an `MtlArray`), so a lens's `project`, a camera, or
a framed stage is the kernel body — there is no per-model kernel code, and a
batch can be any composition of per-point maps (`K ∘ lens ∘ pose`), fused
into one evaluation per point.

The lens surfaces are [`project!`](@ref)`(u, m, X)`, [`backproject!`](@ref),
[`distort_normalized!`](@ref) and [`undistort_normalized!`](@ref). A lens is
converted once to the output's number type, so a `Float32` batch computes in
`Float32` end to end (Metal has no `Float64`).

## One kernel

A batch runs as ONE KernelAbstractions kernel on the backend that owns the
output, the CPU's included: the same per-point function, bit for bit, on every
backend. The `exec` keyword, [`OnBackend`](@ref)`(backend; workgroupsize)`,
launches it on a backend value the caller gives, so its compiler options hold
(`CUDABackend(; always_inline = true)`), with a fixed workgroup size. On the CPU
the launch has finished on return; on a device it does not wait: synchronize
the backend before reading. `OnBackend` is the stack's one kernel-launch policy: every stack
kernel is instantiated through it, `OnBackend(backend; workgroupsize)(kernel)`,
the workgroup per backend (the backend's choice; one work item per workgroup on
the CPU and one warp on a device; or a size as given). (The host's broadcast policies, `Serial`,
`Threaded` and `Auto`, are retired: the owner's one-kernel policy, 2026-11-11.)

## Failure

A point the map does not image comes back NaN — the per-point function's own
answer, never a throw. Its validity is `all(isfinite, out[i])`.

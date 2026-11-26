# Devices & Kernels API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The one KernelAbstractions backend resolver, the accelerator's memory and devices, the one upload to a device, and
the per-backend choices of one kernel — its float type ([`kernel_float`](@ref)), its ForwardDiff chunk
([`dual_chunk`](@ref)) and its host resident threads ([`device_resident_threads`](@ref)) — each stated once, so a
downstream kernel's caller asks Core rather than branching on the backend itself. The host/GPU predicate is
KernelAbstractions' own `isgpu(backend)`. The kernel-launch policy, [`OnBackend`](@ref), is on the
[Batch API](batched.md) page.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["base/device.jl"]
```

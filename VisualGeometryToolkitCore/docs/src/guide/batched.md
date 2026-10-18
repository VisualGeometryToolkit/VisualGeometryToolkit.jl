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

## Execution policy

The `exec` keyword chooses how a **host** batch runs; a device array is always
one broadcast kernel.

- [`Serial`](@ref)`()` — one loop on the calling task. Use it when the caller
  already parallelizes an outer loop (RANSAC across image pairs), so threads
  are not oversubscribed.
- [`Threaded`](@ref)`()` — the same broadcast on contiguous chunks, one task
  per thread.
- [`Auto`](@ref)`()` (the default) — `Serial` below
  [`THREADED_MIN_LENGTH`](@ref) points, `Threaded` above it.

Every policy gives bitwise identical results.

## Failure

A point the map does not image comes back NaN — the per-point function's own
answer, never a throw. Its validity is `all(isfinite, out[i])`.

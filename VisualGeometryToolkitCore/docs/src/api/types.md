# Core Types API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The core composition and uncertainty types (`Attributed`, `Uncertain`,
`CovMat`/`PrecMat`), the conic and epipolar matrix types, interface stubs,
and traits. See the [Design: Attributed{V,A}](../guide/design.md) and
[Stereo Rigs and Uncertainty](../guide/stereo.md) guides.

## Core Types

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/types.jl", "base/interface.jl", "base/interface_functions.jl", "base/traits.jl"]
```

## Uncertainty Accessors

`prec` is documented above with the core types; `cov` extends
`Statistics.cov` (unexported — reachable as `Statistics.cov`):

```@docs
VisualGeometryToolkitCore.cov
```

## Fit Results

Every estimator returns a [`FitResult`](@ref): a fixed core — the estimate,
why it stopped, how many iterations, the residual RMS over `n` — plus
solver-specific `metadata` selected by type parameter. Read it through the
accessors below (`estimate(r)`, `retcode(r)`, …, and the metadata's own such as
`weights(r)`), never by field name. A property spelling still forwards to the
metadata for older call sites; a name on neither is an error, never `nothing`.

```@docs
FitResult
BundleMetadata
estimate
retcode
converged
iterations
rms
metadata
weights
dof
nested_metadata
```

A robust result records ONE scale — the one it ran with, or the one it
estimated — and where it came from: its metadata's `scale` and
`scale_source`, read by [`noise_scale`](@ref) and [`scale_source`](@ref). A
downstream stage takes the hand-off as `scale = result`; [`scale_input`](@ref)
parses a stage's `scale`/`width` keywords, exactly one of which it requires.

```@docs
ScaleSource
ScaleGiven
ScaleUpstream
ScaleEstimated
ScaleNone
noise_scale
scale_source
ScaleInput
scale_input
```

Why a fit stopped is its termination record: the verdict, a
[`ReturnCode`](@ref) (`ReturnCode.T`, SciMLBase's names and success rule),
read by [`retcode`](@ref), and the rule that fired, a [`StopRule`](@ref), read
by [`stop_rule`](@ref). [`converged`](@ref) is strict,
[`successful_retcode`](@ref) of the code: a cap or a stall is not converged.
Whether a RANSAC found a model is [`found_model`](@ref).

```@docs
ReturnCode
successful_retcode
StopRule
return_code
stop_rule
found_model
retcode_symbol
retcode_from_symbol
worst_retcode
```

Its per-datum weights in [0, 1] are [`weights`](@ref)`(result)`, a plain vector;
what they mean — and so where [`inlier_mask`](@ref)`(result)` cuts them — is
the producer type's [`WeightKind`](@ref), read by [`weight_kind`](@ref)
(Estimation Interfaces).

## Index

```@index
Pages = ["types.md"]
```

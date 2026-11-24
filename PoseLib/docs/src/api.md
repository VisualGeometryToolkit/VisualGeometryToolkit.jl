# API reference

```@meta
CurrentModule = PoseLib
```

Every docstring of the package: the module, the exported [`LeZach`](@ref)
strategy, the `PoseLibRefineMetadata` result metadata,
[`is_available`](@ref PoseLib.is_available), and the `PoseLibSolver` methods
this package adds to VisualGeometryToolkitCore's backend contract generics: the
minimal-solver kernels, the eight-point kernel `essential_8pt` and the
Levenberg–Marquardt refine `lm_backend_refine`.

```@autodocs
Modules = [PoseLib]
Order = [:module, :type, :constant, :function]
```

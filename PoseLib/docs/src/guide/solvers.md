# Minimal solvers

A minimal solver estimates a model from the smallest sample of
correspondences that determines it, and returns every solution. The random
sample consensus (RANSAC) loop calls one per hypothesis. VisualGeometryToolkitCore
defines the solvers and their interface, `solve(solver, cs, idx)`; each solver
takes a backend that chooses the implementation. Loading PoseLib.jl adds the
backend `PoseLibSolver()`.

| Solver | Model | Sample | Correspondences `cs` |
|---|---|---|---|
| `Homography4pt(PoseLibSolver())` | `HomographyMat` | 4 | 2D points `x1 => x2` |
| `Fundamental7pt(PoseLibSolver())` | `FundMat`, up to 3 | 7 | 2D image points `x1 => x2` |
| `P3P(PoseLibSolver())` | `Extrinsics3`, up to 4 | 3 | 3D world points `X => f`, `f` a unit bearing |
| `Essential5pt(PoseLibSolver())` | `EssentialMat`, up to 10 | 5 | unit bearings `f1 => f2` |

`idx` names the correspondences of the sample. The solver reads only those, so
a RANSAC loop hands over the whole set with a new `idx` each time:

```julia
models = solve(Essential5pt(PoseLibSolver()), cs, idx)
```

VisualGeometryToolkitCore's native P3P and five-point solvers are ports of
PoseLib's. Having both backends lets either check the other, and lets a
benchmark compare against PoseLib itself.

## Calibrated solvers take bearings

P3P and the five-point solver are calibrated: their image-side data are unit
bearings, 3-vectors of unit length, as everywhere in the stack (see the
[bearing convention](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/dev/guide/conventions/)).
A 2-vector is a `MethodError`, never silently lifted to `(x, y, 1)`. The
solvers do not check the length of each bearing: the camera's back-projection,
which builds the bearings, guarantees it.

## The eight-point essential matrix

`essential_dlt(cs)` estimates an essential matrix from eight or more bearing
correspondences by the direct linear transform (DLT), taking every
correspondence as an inlier. VisualGeometryToolkitCore has no native version,
so without a backend argument it calls PoseLib's.
`essential_dlt(PoseLibSolver(), cs)` names the backend explicitly.

## The positional kernels

Underneath `solve`, each solver calls a positional kernel:
`VisualGeometryToolkitCore.homography_4pt`, `fundamental_7pt`, `p3p` and
`essential_5pt`. PoseLib.jl adds a `PoseLibSolver` method to each. Of these
only `p3p` is exported; the others are public but not exported. Most code has
no reason to call them directly.

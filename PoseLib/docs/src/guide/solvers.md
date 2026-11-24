# Minimal solvers

A minimal solver estimates a model from the smallest sample of
correspondences that determines it, and returns every solution. The random
sample consensus (RANSAC) loop calls one per hypothesis. VisualGeometryToolkitCore
defines the solvers and their problem, `solve(MinimalProblem(M, cs, idx),
solver)`; each solver takes a backend that chooses the implementation. Loading
PoseLib.jl adds the backend `PoseLibSolver()`.

| Solver | Model `M` | Sample | Correspondences `cs` |
|---|---|---|---|
| `Homography4pt(PoseLibSolver())` | `HomographyMat` | 4 | 2D points `x1 => x2` |
| `Fundamental7pt(PoseLibSolver())` | `FundMat`, up to 3 | 7 | 2D image points `x1 => x2` |
| `P3P(PoseLibSolver())` | `CameraExtrinsics`, up to 4 | 3 | 3D world points `X => f`, `f` a unit bearing |
| `Essential5pt(PoseLibSolver())` | `EssentialMat`, up to 10 | 5 | unit bearings `f1 => f2` |

`idx` names the correspondences of the sample. The solver reads only those, so
a RANSAC loop hands over the whole set with a new `idx` each time:

```julia
models = solve(MinimalProblem(EssentialMat, cs, idx), Essential5pt(PoseLibSolver()))
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

`solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))` estimates
an essential matrix from eight or more bearing correspondences by the direct
linear transform (DLT), taking every correspondence as an inlier: a closed
form, whose `FitResult` holds the matrix as `u` (`nothing` when PoseLib's
solution is not finite). VisualGeometryToolkitCore has no native version, so
the backend is required: `DLT()` on an essential matrix is refused. Its kernel
is VisualGeometryToolkitCore's `essential_8pt`, to which PoseLib.jl adds the
`PoseLibSolver` method.

## The positional kernels

Underneath `solve`, each solver calls a positional kernel:
`VisualGeometryToolkitCore.homography_4pt`, `fundamental_7pt`, `p3p`,
`essential_5pt` and `essential_8pt`. PoseLib.jl adds a `PoseLibSolver` method
to each. Of these only `p3p` is exported; the others are public but not
exported. Most code has no reason to call them directly.

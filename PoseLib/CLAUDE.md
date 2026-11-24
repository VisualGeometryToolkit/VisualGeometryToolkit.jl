# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PoseLib.jl: Julia bindings to the PoseLib C++ library, as the `PoseLibSolver`
backend of VisualGeometryToolkitCore's `solve(problem, algorithm)` interface:
the minimal solvers' kernels (homography 4-pt, fundamental 7-pt, P3P, essential
5-pt), the eight-point essential matrix (`DLT(PoseLibSolver())`, the kernel
`essential_8pt`), and PoseLib's robust refines,
`solve(RefineProblem(cs, guess[, p]), LevenbergMarquardt(PoseLibSolver(); optimization_strategy); stops = LmStops(; max_iter), scale | width)`
with the guess a `HomographyMat`, a `FundMat` or an `EuclideanMap{3}` and the context `p` the
intrinsics where PoseLib needs them (`(; K1, K2)`, `(; K)`). Core's `solve` forwards it to the backend
contract generic `lm_backend_refine`, which PoseLib implements for `PoseLibSolver`: PoseLib adds no `__solve`.

PoseLib adds methods to Core's functions only where Core lists them in
`CONTRACT_GENERICS` (the backend kernels, `lm_backend_refine`, `default_damping`), always on
`PoseLibSolver`; `check_algorithm_traits` (R16) holds in the quality test.
Rebuilding the C wrapper (`wrapper/`) is in `docs/src/developer.md`.

## Testing

Plain `@testset` (`test/runtests.jl`): `Pkg.test()` runs the whole suite; there
is no per-item filter. `test/readme_tests.jl` runs the README's one julia
block in a fresh environment built from its `pkg> add` install line, and checks
that the landing page (`docs/src/index.md`) shows the same line and the same
block. `test/docs_tests.jl` checks that every public name has an attached
docstring whose signature lines match the methods.

## Documentation

The README is for users: purpose, installation, a quick start, links. Reference
and maintainer material lives in the Documenter pages under `docs/src/`
(`guide/` for users, `developer.md` for maintainers). `docs/make.jl` runs with
`checkdocs = :public`, doctests and linkcheck, and no `warnonly`: any warning
fails the build.

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "🤖 Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself

## Stack Conventions

The conventions every stack package follows — naming and the estimator verbs,
signatures, data arguments, coordinate frames and normalization ("frame in =
frame out"), pixel origins, bearings, Jacobians, units and the stack
contracts — live ONCE, in VisualGeometryToolkitCore's `conventions/` folder
(`conventions/README.md` indexes it). Read them before changing code here.
This file carries no copy of them, only how they land in this package. Where
the folder is:

- in a checkout of the stack's monorepo: `../VisualGeometryToolkitCore/conventions/`
  (the root CLAUDE.md imports it);
- in VisualGeometryToolkitCore's own repository: `conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`, i.e.
  `joinpath(pkgdir(VisualGeometryToolkitCore), "conventions")`.

The agent-docs test checks that this file points there and holds no copy.

## Coordinate Frames and Normalization

The rule is the stack's (`conventions/frames.md`); how it lands here:

### Frames here

The bindings take caller-frame data and return results in that frame; a
`width` is in the unit of the data handed over. Any conditioning inside the
C++ is private to it. A binding that needs a camera takes it (not a pre-divided
loss width), and says which camera model it hands to C++. The loss is a
VisualGeometryToolkitCore loss shape (or `LeZach()`), mapped to PoseLib's
integer `LossType` only at the ccall.

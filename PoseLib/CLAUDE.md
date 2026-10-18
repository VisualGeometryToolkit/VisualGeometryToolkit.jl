# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PoseLib.jl: Julia bindings to the PoseLib C++ library: the `PoseLibSolver`
backends of VisualGeometryToolkitCore's minimal solvers (homography 4-pt, fundamental
7-pt, P3P, essential 5-pt / 8-pt) and PoseLib's truncated-loss refiners
(`refine_relpose`, `refine_fundamental`, `refine_homography`,
`refine_absolute_pose`).

## Testing

Plain `@testset` (`test/runtests.jl`): `Pkg.test()` runs the whole suite; there
is no per-item filter.

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
`loss_width` is in the unit of the data handed over. Any conditioning inside the
C++ is private to it. A binding that needs a camera takes it (not a pre-divided
loss width), and says which camera model it hands to C++. The loss is a
VisualGeometryToolkitCore loss shape (or `LeZachTruncation()`), mapped to PoseLib's
integer `LossType` only at the ccall.

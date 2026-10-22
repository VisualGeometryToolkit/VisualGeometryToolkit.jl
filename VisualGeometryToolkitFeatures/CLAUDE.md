# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

VisualGeometryToolkitFeatures: feature detection, description and matching for the
VisualGeometry stack (ASIFT detection, descriptor matching: `asift_detect`,
`nn_match`, `match`), returning VisualGeometryToolkitCore types.

## Testing

ReTestItems: `Pkg.test()` runs every item; `Pkg.test(; test_args = ["<regex>"])`
filters items by name. Check the reported total.

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

Detectors return COLMAP pixel centres (top-left pixel centre at (0.5, 0.5)),
pinned by a test. `raw = true` output is OpenCV-convention (0.5 less) and is
documented as such. The NN matchers work in descriptor space and never touch
coordinates. LoMa (`src/loma.jl`) takes keypoints too: it maps COLMAP pixels
into the network's [-1, 1] chart privately, and returns index matches
(`Vector{NNMatch}`), never coordinates.

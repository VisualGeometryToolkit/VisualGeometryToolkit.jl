# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

VisualGeometryFeatures: feature detection, description and matching for the
VisualGeometry stack (ASIFT detection, descriptor matching: `asift_detect`,
`nn_match`, `match`), returning VisualGeometryCore types.

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
contracts — live ONCE, in VisualGeometryCore's `conventions/` folder
(`conventions/README.md` indexes it). Read them before changing code here.
This file carries no copy of them, only how they land in this package. Where
the folder is:

- in a checkout of the stack's monorepo: `../VisualGeometryCore/conventions/`
  (the root CLAUDE.md imports it);
- in VisualGeometryCore's own repository: `conventions/`;
- from Julia, the installed copy: `VisualGeometryCore.conventions_dir()`, i.e.
  `joinpath(pkgdir(VisualGeometryCore), "conventions")`.

The agent-docs test checks that this file points there and holds no copy.

## Coordinate Frames and Normalization

The rule is the stack's (`conventions/frames.md`); how it lands here:

### Frames here

Detectors return COLMAP pixel centres (top-left pixel centre at (0.5, 0.5)),
pinned by a test. `raw = true` output is OpenCV-convention (0.5 less) and is
documented as such. Matchers work in descriptor space and never touch
coordinates.

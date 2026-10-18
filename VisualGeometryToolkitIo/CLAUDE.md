# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

VisualGeometryToolkitIo: serialization for transforms and cameras across the
VisualGeometry stack: a COLMAP model reader/writer and the self-describing
`vgio/<kind>/v2` JSON schema, with a unit-contract boundary and quaternion
rotations.

## Testing

ReTestItems: `Pkg.test()` runs every item; `Pkg.test(; test_args =
["migrate|v1"])` filters items by name; `RETESTITEMS_NWORKERS=0` runs serially
for debugging. Check the reported total.

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

Persisted labels (pixel frame, length unit, pose from/to) are DERIVED FROM
THE DATA'S TYPES: the writer writes the type's frame and never converts. Readers
return typed frames (VGC's `TargetPoints2D{F}`, labelled
`EuclideanMap`s); moving the numbers is `convert_frame`, explicitly. Third-party
readers state the file's convention (`file_frame`, `file_unit`). Nothing here
normalizes.

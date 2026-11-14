# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

VisualGeometryToolkitIO: serialization for transforms and cameras across the
VisualGeometry stack: a COLMAP model reader/writer and the self-describing
`vgio/<kind>/v<major>.<minor>` JSON kinds (one envelope: `generator`, one
`units` object, one `conventions` object), with a unit-contract boundary and
quaternion rotations.

## Testing

ReTestItems: `Pkg.test()` runs every item; `Pkg.test(; test_args =
["migrate|v1"])` filters items by name; `RETESTITEMS_NWORKERS=0` runs serially
for debugging. Check the reported total.

## Schema versions

Every earlier schema version a published file holds is READ; only the
current version is written. Reason: a schema change must never force the
published data (Zenodo, Hugging Face; multi-GB archives at ~100 KB/s) to be
uploaded again, so a published file keeps loading forever.

- Each old version has ONE step, `migrate_schema(::Val{schema}, tree)` (a
  parsed record → the next version's record), in `src/migrate.jl` (the
  IO 0.10 steps in `src/migrate_v010.jl`). The versions read are derived
  from the steps (`schema_versions()`); no list is kept by hand. The
  readers run the steps on load (`_read_store` → `migrate_record`), silently
  (an `@debug` names the version); `vgio_migrate` rewrites a file with the
  same steps (the file form `migrate_schema(::Val{schema}, src, dst)`).
- A minor version adds optional keys only (a reader reads any minor of its
  major); a renamed, removed or re-meant key is a major version.
- A schema change adds a step for the version it replaces (and its file
  form, and a fixture under `test/fixtures/old/` or built in
  `schema_versions_tests.jl`). Never delete a step whose version a published
  file holds, and never make a writer write an old version — with ONE rule
  for a kind whose new major version only ADDS a form (owner ruling (a),
  IO 0.10.2): its writer may write the earlier version for a record that
  version can hold, the earlier version's very bytes, so a reader of that
  version keeps reading every such file. The choice dispatches on what the
  record holds (`sensor_characterisation`: v2.0 iff an entry's black is a
  `Darks`, else v1.0; `_written_version`), and the versions a writer can emit
  are ONE table, `_written_versions(kind)` (validate.jl), which the writer's
  `_write_kind(...; version)` checks and the JSON Schema generator publishes
  (one schema per written version, none other: json_schema_tests.jl). Such
  an earlier version keeps its read step but needs no file form: it is
  still written, so no file of it needs rewriting.
- A version no published file holds, and one that lacks what the current
  record needs (transforms v1's free-form frame names), keeps a refusal by
  name. Transforms and trajectory v2 have a version step for the one layout a
  published file holds (the 2026-09 hand-eye record, CaliBlobs' 2026-09 track
  segment); any other v2 layout is refused in that step.
- A schema owner (BlobBoards' boards) adds its own steps and calls
  `migrate_record` in its reader.

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
return typed frames (VGC's `TargetObservations{F}`, labelled
`EuclideanMap`s); moving the numbers is `convert_frame`, explicitly. Third-party
readers state the file's convention (`file_frame`, `file_unit`). Nothing here
normalizes.

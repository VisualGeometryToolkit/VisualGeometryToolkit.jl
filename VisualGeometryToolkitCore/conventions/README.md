# The stack conventions

The one canonical copy of the conventions every stack package follows. It ships
with VisualGeometryToolkitCore: in an installed copy the folder is
`VisualGeometryToolkitCore.conventions_dir()`, and
`VisualGeometryToolkitCore.convention_files()` lists the files in reading order:

1. `overview.md` — what these files are
2. `naming.md` — the verb states the kind; the iterative-estimator and width vocabularies
3. `signatures.md` — the model leads, then its data, then parameters; mutation order
4. `data-arguments.md` — correspondences in, dicts for groups
5. `frames.md` — frame in = frame out; normalization; the benchmark-pool exception
6. `pixel-origins.md` — the three pixel-origin conventions; COLMAP by default
7. `bearings.md` — calibrated geometry is on unit bearings
8. `measurement-noise.md` — plain = unit isotropic, `Uncertain` = its covariance, `exact` = none
9. `persisted-geometry.md` — on disk, `covariance` present iff `Uncertain`; the pose chart; the JSON + payload storage rule
10. `jacobians.md` — derivative and Jacobian names
11. `units.md` — FlexUnits with a pixel dimension; values in base units, units named at `ustrip`
12. `stack-contracts.md` — tags, dependencies, package boundaries, agent docs

**Who reads them.** VisualGeometryToolkitCore's `CLAUDE.md` imports every file. Each
other package's `CLAUDE.md` carries a "Stack conventions" pointer to this folder
and no copy of the text; its test suite checks that. The docs' Conventions page
is generated from the same files by `docs/make.jl`, in the order above.

**Editing.** Change the text here and nowhere else. The files are
Documenter-flavoured Markdown: the `[`name`](@ref)` links are deliberate
protection, not decoration. `makedocs` sets no `warnonly`, so an unresolvable
`@ref` is a hard error and a rename that does not update these files breaks the
docs build rather than rotting quietly (`PixelChart` once survived here as a
plain code span after the name became `PixelFrame`). Link only names that are
VisualGeometryToolkitCore's AND documented; leave as code spans other packages' API,
verb families quoted as words, math placeholders, Ceres and GTSAM names, and
above all the RETIRED names in `frames.md`, which must not resolve. A link to a
guide page is written relative to this folder
(`../docs/src/guide/frames.md`); `docs/make.jl` rewrites it for the generated
page.

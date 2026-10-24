# Releasing

Every package is registered from a subdirectory of this repository in
[CauRegistry](https://github.com/prittjam/CauRegistry.jl) (public), and
submitted to General. Releases never wait for General: a Manifest records UUID,
version and tree hash, never the registry, so an environment resolves the same
from either. Users add CauRegistry until a package is in General (README,
"Installation").

## Registration

1. All checks green on `main` (CI, Nightly, Docs).
2. Local pre-release checks on the release machine (no hosted runner can do them):
   - GPU items: `CUDA_VISIBLE_DEVICES=0 julia --project=VisualGeometryToolkitCore -e 'using Pkg; Pkg.test()'`
     (and BlobBoards' GPU suites downstream). Never device 1.
   - LO timing gate: `LO_TIMING_GATE=1` in VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
     and LocalOptimizationDoneRight, against the pinned baselines, on a quiet machine.
   - Strict run: `Pkg.test(; julia_args = ["--depwarn=error"])` for each package.
3. Bump `version` in `<Package>/Project.toml`; widen dependents' compat in the
   same commit if the bump is breaking.
4. CauRegistry: register the bumped package from a clean checkout of that commit
   (`LocalRegistry.register("<Package>"; registry = "CauRegistry", repo =
   "https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git")`;
   the subdirectory is detected), then push the registry.
5. General: comment on the same commit `@JuliaRegistrator register subdir=<Package>`
   (one comment per package, dependency order — VisualGeometryToolkitCore first). Nothing
   waits for the merge.
6. TagBot tags `<Package>-v<version>` and creates the GitHub release; the tag
   push builds that package's docs into `gh-pages/<Package>/v<version>`.
7. Zenodo archives the whole repository at that release as a new version of the
   toolkit's one record (metadata from `CITATION.cff`). Every package's release
   is a version of the same record; the concept DOI in `CITATION.cff` and the
   README always resolves to the latest, and is the one to cite.

## The Zenodo DOI

The toolkit has ONE concept DOI, not one per package. Zenodo's GitHub
integration is switched on for this repository: sign in to Zenodo with GitHub,
have an organization owner approve the Zenodo OAuth app for
`VisualGeometryToolkit` (until then the repository is not listed, and Sync does
not help), then toggle `VisualGeometryToolkit.jl` on Zenodo's GitHub page. It
archives every published GitHub release, so a release that is not a package's
— the PoseLib binaries — becomes a toolkit version too unless the toggle is off
while it is published. If a package release has no Zenodo version, check that
the toggle is still on.

Only releases published after the toggle are archived. The first one mints the
concept DOI. Zenodo reads `CITATION.cff` from the tagged commit, and
VisualGeometryToolkitCore 0.1.0's commit predates the file, so that first
record's title and authors are set by hand on Zenodo. Then record the concept
DOI as `doi:` in `CITATION.cff`, as a DOI badge in the README, and as
`doi = {…}` in the one BibTeX entry repeated in every "Citing" section: the
root README, each package's README and the docs index of
VisualGeometryToolkitCore, VisualGeometryToolkitEstimation and
VisualGeometryToolkitEval.

## The first release (0.1.0)

CauRegistry goes public with the release, in one commit
(`registry/cauregistry_golive.jl` of the assembly tooling): for the nine
released packages (the six here, RANSACScoringDoneRight,
LocalOptimizationDoneRight, BlobBoards) every earlier, private version is
removed, and 0.1.0 is registered from the public repositories. The registry's
other entries are unchanged.

General registrations follow in waves; each wave starts after the previous one
merges:

| wave | packages | needs |
|---|---|---|
| 1 | VisualGeometryToolkitCore | — |
| 2 | VisualGeometryToolkitIO, VisualGeometryToolkitFeatures, VisualGeometryToolkitEstimation (+ RANSACScoringDoneRight) | VisualGeometryToolkitCore |
| 3 | VisualGeometryToolkitEval | VisualGeometryToolkitIO, VisualGeometryToolkitEstimation |
| 4 | PoseLib, the last of the six | VisualGeometryToolkitCore; its binaries uploaded as release assets first |
| 5 | (LocalOptimizationDoneRight) | VisualGeometryToolkitEstimation; PoseLib (a weak dependency, which General also requires registered) |
| 6 | (BlobBoards) | the above |

New packages wait three days in General; `PoseLib` needs a manual merge
(name distance 2 from `PDELib`).

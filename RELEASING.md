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
   - GPU items: `CUDA_VISIBLE_DEVICES=0 julia --project=VisualGeometryCore -e 'using Pkg; Pkg.test()'`
     (and BlobBoards' GPU suites downstream). Never device 1.
   - LO timing gate: `LO_TIMING_GATE=1` in VisualGeometryCore, RobustVisualGeometry
     and LocalOptimizationDoneRight, against the pinned baselines, on a quiet machine.
   - Strict run: `Pkg.test(; julia_args = ["--depwarn=error"])` for each package.
3. Bump `version` in `<Package>/Project.toml`; widen dependents' compat in the
   same commit if the bump is breaking.
4. CauRegistry: register the bumped package from a clean checkout of that commit
   (`LocalRegistry.register("<Package>"; registry = "CauRegistry", repo =
   "https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git")`;
   the subdirectory is detected), then push the registry.
5. General: comment on the same commit `@JuliaRegistrator register subdir=<Package>`
   (one comment per package, dependency order — VisualGeometryCore first). Nothing
   waits for the merge.
6. TagBot tags `<Package>-v<version>` and creates the GitHub release; the tag
   push builds that package's docs into `gh-pages/<Package>/v<version>`.

## The first release (0.1.0)

CauRegistry goes public with the release, in one commit
(`registry/cauregistry_golive.jl` of the assembly tooling): for the nine
released packages (the six here, RansacScoringDoneRight,
LocalOptimizationDoneRight, BlobBoards) every earlier, private version is
removed, and 0.1.0 is registered from the public repositories. The registry's
other entries are unchanged.

General registrations follow in waves; each wave starts after the previous one
merges:

| wave | packages | needs |
|---|---|---|
| 1 | VisualGeometryCore | — |
| 2 | VisualGeometryIo, VisualGeometryFeatures, RobustVisualGeometry (+ RansacScoringDoneRight) | VisualGeometryCore |
| 3 | VisualGeometryEval | VisualGeometryIo, RobustVisualGeometry |
| 4 | PoseLib, the last of the six | VisualGeometryCore; its binaries uploaded as release assets first |
| 5 | (LocalOptimizationDoneRight) | RobustVisualGeometry; PoseLib (a weak dependency, which General also requires registered) |
| 6 | (BlobBoards) | the above |

New packages wait three days in General; `PoseLib` needs a manual merge
(name distance 2 from `PDELib`).

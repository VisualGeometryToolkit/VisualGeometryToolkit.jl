# VisualGeometryFeatures.jl

Feature detection, description and matching for the VisualGeometry stack.

- **Detection** — ASIFT (IMAS-25) on KernelAbstractions. `asift_detect` runs on
  the CPU or any KernelAbstractions GPU backend and returns `Detection`s (a
  feature frame plus detector metadata), which round-trip through JSON.
- **Matching** — nearest-neighbour descriptor matching, one KernelAbstractions
  implementation for the CPU and every GPU backend (`nn_match`, `match`), and a
  stacked matcher over many galleries (`nn_match_stacked`).
- **Extensions** — JLD2 backs `FileCache`; Makie draws detections and
  correspondences.

The geometric types are VisualGeometryCore's (VGC); this package adds the
detector, the descriptors and the matchers over them.

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryFeatures")
```

Requires Julia 1.12. Part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack.

## Conventions

The stack's conventions are VGC's, and VGC's
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryCore.jl/dev/guide/conventions/)
is the canonical text — function verbs, argument order, data arguments, the
frame tags and the pixel conventions. Read it there; this section records only
what a VGF reader needs on top of it.

### Pixel origin

Every VGF detector returns **COLMAP** pixel centres: the top-left pixel's centre
is `(0.5, 0.5)`, so the pixel `img[r, c]` has its centre at `(c − 0.5, r − 0.5)`.
A `center` is `(x, y)` — x along the image columns, y along the rows.

A detection's frame convention is declared in two places, both pinned by the
test suite: the `Detection` docstring (`src/detection_types.jl`) for the type,
and the `asift_detect` docstring (`src/asift/api.jl`) for the detector, which
also documents its measured sub-pixel accuracy. `asift_detect(…; raw = true)`
returns 0-indexed OpenCV coordinates instead — 0.5 less on both axes — and says
so at the call site.

**Frame in = frame out** (VGC's rule, repeated in `CLAUDE.md`): pixels in,
pixels out. The matchers work in descriptor space and never touch coordinates,
so nothing here normalizes.

### Feature frames are shapes, not coordinate frames

`FeatureFrame` is "frame" in the VLFeat/Kornia sense: the SHAPE of a detected
feature, not a coordinate system. It is a VGC type, with four concrete forms
ordered by degrees of freedom:

| Frame | DOF | Shape |
|---|---|---|
| `Disc` | 3 | centre + isotropic scale |
| `OrientedDisc` | 4 | centre + scale + orientation |
| `EllipticFrame` | 5 | centre + moment matrix (anisotropic) |
| `AffineFrame` | 6 | centre + full affine shape |

A `Detection{F, M}` composes one of these with detector metadata. The accessor
is **`feature_frame(d)`** — VGC's name, which VGF's methods follow since
`083ee3b`; the bare `frame(d)` is retired. `asift_detect` emits
`Detection{EllipticFrame}`.

Coordinate-frame TAGS are a separate vocabulary and live in VGC — `ImageFrame`,
`NormalizedImageFrame`, `GnomonicFrame` (the z = 1 chart). The two vocabularies share the word "frame" and nothing else.

### The refinement bias

`asift_detect`'s quadratic sub-pixel refinement pulls a centre toward the
nearest pixel centre. On noise-free isotropic Gaussian blobs (CPU), a blob on a
pixel centre comes back exact to within 0.005 px, and between centres the pull
is about 0.03 px at σ = 4. It is not bounded by 0.06 px across scales: the Monte
Carlo study under `test/mc` (`benchmark/mc/results/zeisl/findings.md`) measured a
centre bias of 0.1–0.2 px on clean blobs with σ ≥ 6.5, and up to ≈ 1 px in
clutter.

**Other repos depend on that number, so it is not to be changed casually.** It
is what the stack's coincidence tolerances are anchored on: two detections of
the SAME feature differ by at most 0.06 px, so VGC sets its coincidence gate at
0.1 px to clear it with margin (`tol_px`, and the reasoning in VGC's
`src/geometry/model_validity.jl`). Loosening VGF's bound is a stack-wide change,
not a VGF one — raise it with the consumers rather than re-tuning here.

### Keypoint uncertainty

`asift_detect(img; uncertainty = FeatureCovariance())` returns each detection
with its frame as VGC's `Uncertain(frame, Σ̃)`: `cov(d)` is a 3×3 `CovMat` over
`(x, y, log σ)` in the image frame, for unit iid pixel noise in the detector's
intensity units (the `[0, 1]` scale for 8-bit input), so the physical covariance
is `σ_n²·Σ̃`. Position-only consumers take its top-left 2×2 block. It is the
implicit-function-theorem covariance of the refined DoG extremum, its noise
term tabulated exactly for the discrete pyramid once per plan; the
`FeatureCovariance` docstring states the model, and benchmark/mc holds the Monte
Carlo study that calibrates it. The default `uncertainty = nothing` compiles the
detector exactly as without the option.

## Testing

ReTestItems; test items live in `test/*_tests.jl`.

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable: after a pull, checkout or rebase that changes
`Project.toml`, they re-resolve the environment (resolve only; precompilation
waits for the next `using`). They are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## License

MIT

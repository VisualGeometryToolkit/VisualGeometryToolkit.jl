# VisualGeometryToolkitFeatures.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitFeatures)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitFeatures)

VisualGeometryToolkitFeatures detects, describes and matches image features:
the ASIFT detector on the CPU or a GPU, nearest-neighbour descriptor matching,
and the learned LoMa descriptors and matchers. It returns the geometric types
of [VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/).

## Installation

```julia-repl
pkg> add VisualGeometryToolkitFeatures
```

Julia 1.12 or later. Every dependency is in the General registry.

## Example

The example detects the blobs of a synthetic image and matches two sets of
descriptors. It needs no files and no network.

```julia
using VisualGeometryToolkitFeatures
using Random   # a standard library, shipped with Julia

# A 120×160 gray image with three Gaussian blobs, given as
# (row, column, σ, amplitude): two light blobs and one dark one.
blobs = [(40, 50, 4.0, 0.35), (70, 100, 3.0, -0.3), (30, 130, 6.0, 0.3)]
img = [0.5 + sum(a * exp(-((r - y)^2 + (c - x)^2) / (2s^2)) for (y, x, s, a) in blobs)
       for r in 1:120, c in 1:160]

# Detect with the default IMAS-25 tilt schedule.
dets = asift_detect(img)
length(dets) == 3                      # true
count(is_dark, dets) == 1              # true

# A centre is (x, y) in COLMAP pixel coordinates, where the top-left pixel's
# centre is (0.5, 0.5): the blob at row 70, column 100 is found at (99.5, 69.5).
dark = only(filter(is_dark, dets))
round.(dark.center; digits = 1) == [99.5, 69.5]   # true

# Match descriptors, one per row. Here they are random unit vectors, and the
# second set is the first shuffled and perturbed.
rng = Xoshiro(1)
unit_rows(D) = D ./ sqrt.(sum(abs2, D; dims = 2))
D1 = unit_rows(randn(rng, Float32, 100, 32))
perm = randperm(rng, 100)
D2 = unit_rows(D1[perm, :] .+ 0.05f0 .* randn(rng, Float32, 100, 32))
matches = nn_match(D1, D2)             # mutual nearest neighbours, cosine similarity
all(perm[m.second] == m.first for m in matches)   # true
```

The [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/stable/guide/getting-started/)
guide walks through it step by step.

## Documentation

- [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/stable/guide/getting-started/):
  the example, step by step.
- [Detection](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/stable/guide/detection/):
  the ASIFT detector, its coordinates, accuracy and keypoint covariance.
- [Matching](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/stable/guide/matching/):
  strategies, metrics, stacked galleries and LoMa.
- [API Reference](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/stable/api/detection/).

VisualGeometryToolkitIO reads and writes detections as JSON, and
VisualGeometryToolkitEstimation fits models to the matches. The
[portal](https://visualgeometrytoolkit.github.io) links every package of the
toolkit.

## Citing

VisualGeometryToolkitFeatures is part of VisualGeometryToolkit. Cite the toolkit. Its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry, and GitHub's "Cite this repository" button exports it:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  doi    = {10.5281/zenodo.23548372},
  year   = {2026}
}
```

Cite also the papers behind the methods you use, listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).

`LoMaDescriptor` and `LoMaMatcher` run the networks of
[LoMa](https://github.com/davnords/LoMa) by David Nordström, Johan Edstedt,
Georg Bökman, Jonathan Astermark, Anders Heyden, Viktor Larsson, Mårten
Wadenbäck, Michael Felsberg and Fredrik Kahl. The architectures, the trained
weights and the ONNX exports are theirs, downloaded from the authors' release.
The descriptors are [DeDoDe](https://github.com/Parskatt/DeDoDe)'s (Edstedt
*et al.*). If you use them in published work, cite both:

```bibtex
@inproceedings{nordstrom2026loma,
  title     = {{LoMa: Local Feature Matching Revisited}},
  author    = {David Nordström and Johan Edstedt and Georg Bökman and Jonathan Astermark and Anders Heyden and Viktor Larsson and Mårten Wadenbäck and Michael Felsberg and Fredrik Kahl},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}

@inproceedings{edstedt2024dedode,
  title     = {{DeDoDe: Detect, Don't Describe --- Describe, Don't Detect for Local Feature Matching}},
  author    = {Johan Edstedt and Georg Bökman and Mårten Wadenbäck and Michael Felsberg},
  booktitle = {2024 International Conference on 3D Vision (3DV)},
  year      = {2024},
  organization = {IEEE}
}
```

LoMa's code is MIT-licensed except its matcher, which inherits LightGlue's
Apache-2.0; DeDoDe is MIT except the DINOv2 backbone of DeDoDe-G (Apache-2.0).

## License

MIT

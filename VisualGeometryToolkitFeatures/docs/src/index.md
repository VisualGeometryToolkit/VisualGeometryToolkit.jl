# VisualGeometryToolkitFeatures.jl

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

VisualGeometryToolkitFeatures detects, describes and matches image features:
the ASIFT detector on the CPU or a GPU, nearest-neighbour descriptor matching,
and the learned LoMa descriptors and matchers. The geometric types it returns
(feature frames, correspondences, covariances) are
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/)'s;
this package adds the detectors, the descriptors and the matchers that work on
them.

## What it provides

- **Detection.** ASIFT (IMAS-25) on KernelAbstractions: [`asift_detect`](@ref)
  runs on the CPU or any KernelAbstractions GPU backend and returns
  [`Detection`](@ref)s (a feature frame plus the detector's metadata),
  optionally with each keypoint's covariance ([`FeatureCovariance`](@ref)).
  Beside it: MSER ([`detect_mser`](@ref)) and the Förstner operator
  ([`detect_foerstner`](@ref)). See [Detection](guide/detection.md) and
  [The Förstner operator](guide/foerstner.md).
- **Image input.** The grayscale ladders ([`gray_f32`](@ref),
  [`gray_u8`](@ref)), a PNG as linear radiance ([`load_linear_image`](@ref))
  and a camera RAW file as exact linear values ([`load_raw`](@ref)).
- **Matching.** Nearest-neighbour descriptor matching with one
  KernelAbstractions implementation for the CPU and every GPU backend
  ([`nn_match`](@ref), [`match`](@ref)), and a stacked matcher over many
  galleries ([`nn_match_stacked`](@ref)). See [Matching](guide/matching.md).
- **LoMa.** The learned DeDoDe descriptors and LoMa matchers, run from the
  authors' ONNX exports through ONNX Runtime on the CPU or CUDA
  ([`LoMaDescriptor`](@ref), [`LoMaMatcher`](@ref)).
- **Extensions.** JLD2 backs [`FileCache`](@ref), and Makie draws detections
  and correspondences ([`detections!`](@ref)).

Detectors return pixel coordinates in the COLMAP convention (the top-left
pixel's centre is (0.5, 0.5)); the matchers work in descriptor space and
return index matches, never coordinates.

## Installation

```julia-repl
pkg> add VisualGeometryToolkitFeatures
```

Julia 1.12 or later; every dependency is in the General registry. Add
VisualGeometryToolkitCore as well when you construct frames or points
yourself.

## A first example

The example detects the blobs of a synthetic image and matches two sets of
descriptors (the README's example; the test suite runs both).

```julia
using VisualGeometryToolkitFeatures
# Random is a standard library, shipped with Julia.
using Random

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

## Documentation

**Start here.** [Getting Started](guide/getting-started.md) walks through the
example one step at a time.

**How to.** Detect features and read their coordinates, accuracy and
covariance ([Detection](guide/detection.md)); find corners and circular points
([The Förstner operator](guide/foerstner.md)); match descriptors on the CPU, a
GPU, over stacked galleries or with LoMa ([Matching](guide/matching.md)).

**Explanation.** The stack's shared conventions (function names, argument
order, coordinate frames and pixel origins) are VisualGeometryToolkitCore's
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/);
[Developer Notes](developer.md) records the accuracy bounds other packages rely
on.

**Reference.** The API Reference pages ([Detection](api/detection.md),
[Image Input](api/image.md), [Matching](api/matching.md), [Cache and Visualization](api/cache.md)) document
every public name; [References](references.md) lists the literature.

## Role in the VisualGeometryToolkit stack

VisualGeometryToolkitFeatures builds on VisualGeometryToolkitCore.
[VisualGeometryToolkitIO](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/serialization/)
writes and reads detections as JSON (`json_value`, `from_json`), and
[VisualGeometryToolkitEstimation](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/)
fits models to the matches. All toolkit documentation is aggregated at the
portal: <https://visualgeometrytoolkit.github.io>.

## Citing

VisualGeometryToolkitFeatures is part of VisualGeometryToolkit. Cite the
toolkit; its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry (GitHub's "Cite this repository" button exports
it):

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

### LoMa

[`LoMaDescriptor`](@ref) and [`LoMaMatcher`](@ref) run the networks of
[LoMa](https://github.com/davnords/LoMa) by David Nordström, Johan Edstedt,
Georg Bökman, Jonathan Astermark, Anders Heyden, Viktor Larsson, Mårten
Wadenbäck, Michael Felsberg and Fredrik Kahl. The architectures, the trained
weights and the ONNX exports are theirs, downloaded from the authors' release;
this package only feeds them keypoints and reads their matches back. The
descriptors are [DeDoDe](https://github.com/Parskatt/DeDoDe)'s (Edstedt
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

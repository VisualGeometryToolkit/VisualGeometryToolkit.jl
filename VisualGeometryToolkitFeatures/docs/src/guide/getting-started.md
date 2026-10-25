# Getting Started

This page installs the package and walks through the README's quick start one
step at a time. Every block on this page is a doctest, run by the docs build.

## Installation

Install VisualGeometryToolkitFeatures from the General registry (Julia 1.12 or
later):

```julia
using Pkg
Pkg.add("VisualGeometryToolkitFeatures")
```

The geometric types come from VisualGeometryToolkitCore, which is installed as
a dependency. Add it to your environment as well (`Pkg.add("VisualGeometryToolkitCore")`)
when you construct frames or points yourself.

## A synthetic image

The detector takes an `H×W` matrix: `Gray` pixels, `UInt8`, or real values in
[0, 1]. The image below is a gray field with three Gaussian blobs, each given
as `(row, column, σ, amplitude)`. A positive amplitude makes a light blob and a
negative one a dark blob.

```jldoctest quickstart
julia> using VisualGeometryToolkitFeatures

julia> blobs = [(40, 50, 4.0, 0.35), (70, 100, 3.0, -0.3), (30, 130, 6.0, 0.3)];

julia> img = [0.5 + sum(a * exp(-((r - y)^2 + (c - x)^2) / (2s^2)) for (y, x, s, a) in blobs)
              for r in 1:120, c in 1:160];

julia> size(img)
(120, 160)
```

## Detecting features

[`asift_detect`](@ref) runs the IMAS-25 ASIFT detector: the image is warped by
25 simulated camera tilts, DoG extrema are found in each warp, and the
detections are mapped back to the input image and merged by a cross-warp
non-maximum suppression. Each result is a [`Detection`](@ref): an elliptic
feature frame with the detector's metadata.

```jldoctest quickstart
julia> dets = asift_detect(img);

julia> length(dets)
3

julia> dark = only(filter(is_dark, dets));

julia> round.(dark.center; digits = 1)    # (x, y): column, then row
2-element GeometryBasics.Point{2, Float64} with indices SOneTo(2):
 99.5
 69.5

julia> round(dark.σ; digits = 1)
3.0

julia> count(is_light, dets)
2
```

The centre is in COLMAP pixel coordinates, where the top-left pixel's centre is
(0.5, 0.5). The blob placed at row 70 and column 100 is therefore found at
`(99.5, 69.5)`. [Detection](detection.md) explains the coordinates, the
accuracy of the sub-pixel refinement and the other options of the detector.

The metadata reads as properties of the detection:

```jldoctest quickstart
julia> propertynames(dark.metadata)
(:response, :edge_score, :laplacian_scale_score, :warp_idx)

julia> dark.response > 0    # a dark blob has a positive DoG response
true
```

## Matching descriptors

[`nn_match`](@ref) matches two descriptor matrices, one descriptor per row. By
default it keeps mutual nearest neighbours under the cosine similarity, so the
descriptors should have unit length. Here the second set is the first,
shuffled and perturbed:

```jldoctest quickstart
julia> using Random

julia> rng = Xoshiro(1);

julia> unit_rows(D) = D ./ sqrt.(sum(abs2, D; dims = 2));

julia> D1 = unit_rows(randn(rng, Float32, 100, 32));

julia> perm = randperm(rng, 100);

julia> D2 = unit_rows(D1[perm, :] .+ 0.05f0 .* randn(rng, Float32, 100, 32));

julia> matches = nn_match(D1, D2);

julia> length(matches)
100

julia> all(perm[m.second] == m.first for m in matches)
true
```

Each match is an [`NNMatch`](@ref): `m.first` indexes the rows of `D1`,
`m.second` those of `D2`, and `m.score` and `m.ratio` are the similarity and
Lowe's ratio. [Matching](matching.md) covers the other strategies and metrics,
the GPU and the learned LoMa matchers.

## Saving detections

Detections are written to and read from JSON by VisualGeometryToolkitIO
(`json_value(d, contract)` and `from_json(Detection, obj, contract)`), which
owns the stack's quantity JSON format; its
[Serialization](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/serialization/)
guide describes the format.

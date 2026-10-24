# Matching

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

## Nearest-neighbour matching

[`nn_match`](@ref) and [`match`](@ref) compare two descriptor matrices, one
descriptor per row, and return a `Vector{NNMatch}` sorted best first. Both run
the same KernelAbstractions kernels: `match` on the CPU, `nn_match` on any
device (`device = CUDABackend()`, for instance), and a GPU returns what the
CPU returns. `match` is public but not exported, because it would shadow
`Base.match`; call it as `VisualGeometryToolkitFeatures.match`.

The strategy is a positional argument:

- [`MutualNN`](@ref) (the default) keeps pairs that are each other's nearest
  neighbour;
- [`NN`](@ref) keeps every row's nearest neighbour;
- [`KnnMatch`](@ref)`(k)` keeps every row's `k` nearest neighbours, for a
  downstream geometric check to disambiguate.

The metric is a keyword: [`CosineSimilarity`](@ref) (the default, for unit
descriptors) or [`L2`](@ref). `ratio_threshold` applies Lowe's ratio test,
and `min_similarity` or `max_distance` bound the score.

```jldoctest matching
julia> using VisualGeometryToolkitFeatures

julia> D1 = Float32[1 0; 0 1; 0.6 0.8];

julia> D2 = Float32[0 1; 1 0];

julia> [(m.first, m.second) for m in nn_match(D1, D2)]
2-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (2, 1)

julia> length(nn_match(D1, D2, NN()))    # row 3 matches too, one way
3
```

## Matches as correspondences

An [`NNMatch`](@ref) holds indices. `csponds(features1, features2, matches)`
turns them into VisualGeometryToolkitCore correspondences between the matched
features, with each match's score and ratio as attribute columns, ready for a
robust estimator.

[`DescribedFeature`](@ref) pairs a detection with its descriptor.
[`described_features`](@ref) builds a `StructArray` of them from detections and
a descriptor matrix, and [`descriptor_matrix`](@ref) returns the descriptors as
a matrix without copying; `match` and `csponds` accept the `StructArray`
directly.

## Many galleries at once

[`nn_match_stacked`](@ref) matches one set of features against many galleries
(a dictionary of descriptor matrices) in one matrix product, with the same
result per gallery as matching that gallery alone. A
[`StackedDescriptorIndex`](@ref) keeps the stacked galleries on the device
between calls.

## LoMa

[`LoMaDescriptor`](@ref) and [`LoMaMatcher`](@ref) run the learned DeDoDe
descriptors and LoMa matchers of Nordström et al. (ECCV 2026) from the authors'
ONNX exports, through ONNX Runtime on the CPU or CUDA. The models are
downloaded on first use and cached in the package's scratch space. A
descriptor describes the strongest detections of an image; a matcher matches
two described sets and returns `Vector{NNMatch}`, like `match`:

```julia
describe = LoMaDescriptor(:b)
matcher = LoMaMatcher(:b128)
fa = describe(img_a, asift_detect(img_a))
fb = describe(img_b, asift_detect(img_b))
matches = matcher(fa, fb, size(img_a), size(img_b))
```

The package's `examples/loma.jl` runs this on a pair of downloaded images and
draws the matches. Cite the LoMa and DeDoDe papers when you use these networks
(see the README).

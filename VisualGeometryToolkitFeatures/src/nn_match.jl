# =============================================================================
# Nearest-Neighbor Descriptor Matching
# =============================================================================
#
# Modular NN matching with orthogonal strategy, metric, and filters:
#   - Strategy: NN (one-way), MutualNN (intersection of both directions), KnnMatch(k)
#   - Metric: CosineSimilarity or L2 (determines pairwise matrix + NN direction)
#   - Filters: ratio_threshold (Lowe's SNN), min_similarity / max_distance
#   - dm: optional precomputed pairwise matrix (Kornia-style)
#
# This file holds the vocabulary and the `match` entry point. The matching
# itself is the KernelAbstractions matcher in matching_ka.jl, which runs on the
# CPU for `match` and on any backend for `nn_match`.
#
# =============================================================================

# =============================================================================
# Strategy Types
# =============================================================================

"""
    NNStrategy

Abstract type for nearest-neighbor matching strategies.
"""
abstract type NNStrategy end

"""
    NN <: NNStrategy

One-way nearest-neighbor matching: for each descriptor in `desc1`,
find the nearest neighbor in `desc2`.
"""
struct NN <: NNStrategy end

"""
    MutualNN <: NNStrategy

Mutual nearest-neighbor matching: find NN in both directions,
keep only pairs where both directions agree.
"""
struct MutualNN <: NNStrategy end

"""
    KnnMatch(k) <: NNStrategy

k-nearest-neighbor matching: for each `desc1` row, keep its `k` nearest `desc2`
columns instead of only the single best. Permissive by design — surfaces true
matches that sit at rank 2..k or just under a strict cutoff (common with
repetitive patterns, where a confuser can out-rank the true neighbor).
Downstream geometry (e.g. RANSAC) disambiguates the extra candidates. Pair with
`min_similarity` to bound the kept scores.
"""
struct KnnMatch <: NNStrategy
    k::Int
    function KnnMatch(k::Int)
        k ≥ 1 || throw(ArgumentError("KnnMatch k must be ≥ 1, got $k"))
        return new(k)
    end
end

# =============================================================================
# Metric Types
# =============================================================================

"""
    DescriptorMetric

Abstract type for descriptor comparison metrics.
"""
abstract type DescriptorMetric end

"""
    DescriptorSimilarity <: DescriptorMetric

Similarity metrics: higher values = more similar. NN finds maxima.
"""
abstract type DescriptorSimilarity <: DescriptorMetric end

"""
    DescriptorDistance <: DescriptorMetric

Distance metrics: lower values = more similar. NN finds minima.
"""
abstract type DescriptorDistance <: DescriptorMetric end

"""
    CosineSimilarity <: DescriptorSimilarity

Cosine similarity for L2-NORMALIZED descriptors, computed as the dot product
`a·b` — no normalization is applied, so un-normalized input gives raw dot
products. Range [-1, 1] for unit vectors, 1 = identical.
"""
struct CosineSimilarity <: DescriptorSimilarity end

"""
    L2 <: DescriptorDistance

Euclidean distance: `‖a - b‖`. Range [0, 2] for unit vectors, 0 = identical.
"""
struct L2 <: DescriptorDistance end

# =============================================================================
# Match Types
# =============================================================================

"""
    NNMatchMeta

Metadata for a nearest-neighbor match.

# Fields
- `score::Float32`: Match quality in the metric's native units
  (similarity for CosineSimilarity, distance for L2)
- `ratio::Float32`: Lowe's ratio — `d_best / d_2nd` for a distance metric,
  `s_2nd / s_best` for a similarity, so that smaller = more distinctive whatever
  the metric. `Inf32` when there is no second neighbour, and when a similarity's
  best score is non-positive: the ratio is undefined there, and no ratio test
  admits it.
"""
struct NNMatchMeta
    score::Float32
    ratio::Float32
end

"""
    NNMatch

A nearest-neighbor match: `Attributed{Cspond{Int,Int}, NNMatchMeta}`.

Property forwarding gives direct access:
- `m.first` — index into desc1
- `m.second` — index into desc2
- `m.score` — match quality (similarity or distance, depending on metric)
- `m.ratio` — Lowe's SNN ratio
"""
const NNMatch = Attributed{Cspond{Int, Int}, NNMatchMeta}

NNMatch(i::Int, j::Int, score::Float32, ratio::Float32) =
    Attributed(Cspond{Int, Int}(i, j), NNMatchMeta(score, ratio))

# =============================================================================
# Internal Helpers
# =============================================================================

# Ratio: always best/second_best for distances, second_best/best for similarities
# Convention: smaller ratio = more distinctive match
_ratio(v1, v2, ::DescriptorDistance) = v1 / v2    # d_best / d_2nd

# A Lowe ratio measures distinctiveness as a ratio of MAGNITUDES, so it is
# defined only for a POSITIVE best similarity. At s_best ≤ 0 the quotient stops
# measuring that (measured, 2026-09-22): it is ≥ 1 whenever both scores are
# negative — rejected, but for the wrong reason, since a MORE negative pair
# scores closer to 1 — and at s_best = 0 it is −Inf, which satisfies any ratio test
# trivially and keeps a match no better than orthogonal. An undefined ratio is
# therefore reported as Inf, which no ratio test admits. A non-positive best
# score survives to here only when `min_similarity ≤ 0`; the test item
# "ratios and the ratio test when min_similarity ≤ 0" pins both branches.
function _ratio(v1, v2, ::DescriptorSimilarity)   # s_2nd / s_best
    r = v2 / v1
    return v1 > 0 ? r : oftype(r, Inf)
end

# Sort order for final output: distances ascending, similarities descending
_sort_rev(::DescriptorDistance) = false
_sort_rev(::DescriptorSimilarity) = true

# =============================================================================
# Main API
# =============================================================================

"""
    match(desc1, desc2, [strategy]; metric, dm, ratio_threshold,
          min_similarity, max_distance) -> Vector{NNMatch}

Match descriptors using nearest-neighbor search, on the CPU. It runs the same
KernelAbstractions matcher as [`nn_match`](@ref) on a GPU, so a device returns
what this returns. Descriptors are compared in their own float type.

Public but not exported, because it would shadow `Base.match`: import it
(`using VisualGeometryToolkitFeatures: match`) or call `VisualGeometryToolkitFeatures.match`.

# Arguments
- `desc1::AbstractMatrix{<:Real}`: N1×D descriptor matrix (rows = descriptors)
- `desc2::AbstractMatrix{<:Real}`: N2×D descriptor matrix
- `strategy::NNStrategy`: `MutualNN()` (default), `NN()` or `KnnMatch(k)`

# Keyword Arguments
- `metric::DescriptorMetric`: `CosineSimilarity()` (default) or `L2()`
- `dm`: Precomputed N1×N2 pairwise matrix (skip `pairwise_matrix` computation)
- `ratio_threshold`: Lowe's SNN ratio threshold; `nothing` = disabled
- `min_similarity`: Minimum score for `DescriptorSimilarity` metrics
- `max_distance`: Maximum score for `DescriptorDistance` metrics

# Returns
`Vector{NNMatch}` sorted by score (best first).

# Usage
```julia
using VisualGeometryToolkitFeatures: match, pairwise_matrix, MutualNN, NN, CosineSimilarity, L2

# Cosine similarity (default) — good for L2-normalized descriptors
match(d1, d2, MutualNN())
match(d1, d2, MutualNN(); min_similarity=0.7)

# L2 distance
match(d1, d2, MutualNN(); metric=L2(), max_distance=0.8)

# Precomputed distance matrix
dm = pairwise_matrix(CosineSimilarity(), d1, d2)
match(d1, d2, MutualNN(); dm)

# With Lowe's ratio test
match(d1, d2, NN(); ratio_threshold=0.8)
```
"""
function match(
        desc1::AbstractMatrix{<:Real}, desc2::AbstractMatrix{<:Real},
        strategy::NNStrategy = MutualNN();
        metric::DescriptorMetric = CosineSimilarity(),
        dm = nothing,
        ratio_threshold = nothing,
        min_similarity = nothing,
        max_distance = nothing
    )
    return _match(
        desc1, desc2, strategy, KA.CPU();
        metric, dm, ratio_threshold, min_similarity, max_distance
    )
end

# =============================================================================
# Convenience: NNMatch → geometric correspondences
# =============================================================================

"""
    csponds(features1, features2, matches::AbstractVector{NNMatch})

Convert index-based NN matches to geometric correspondences.

Returns a columnar `Csponds` whose endpoints are the matched features and
whose attribute columns carry BOTH match statistics:

- `score::Vector{Float64}` — higher-is-better sampling prior
  (similarity metrics are used as is; distance metrics are inverted
  `1/(d + 1e-6)`)
- `ratio::Vector{Float32}` — the Lowe's-ratio value of each match
  (previously computed and then dropped; it now survives to downstream
  consumers)

Access endpoints via `sources`/`targets`, attributes via `cs.score` /
`cs.ratio`; subsetting keeps all columns aligned.
"""
function csponds(
        features1::AbstractVector, features2::AbstractVector,
        matches::AbstractVector{NNMatch};
        metric::DescriptorMetric = CosineSimilarity()
    )
    n = length(matches)
    S, T = eltype(features1), eltype(features2)
    firsts = Vector{S}(undef, n)
    seconds = Vector{T}(undef, n)
    scores = Vector{Float64}(undef, n)
    ratios = Vector{Float32}(undef, n)
    _score = metric isa DescriptorSimilarity ? m -> Float64(m.score) : m -> Float64(1 / (m.score + 1.0f-6))
    @inbounds for (i, m) in enumerate(matches)
        firsts[i] = features1[m.first]
        seconds[i] = features2[m.second]
        scores[i] = _score(m)
        ratios[i] = m.ratio
    end
    return Csponds(firsts, seconds; score = scores, ratio = ratios)
end

# =============================================================================
# Greedy one-to-one resolution
# =============================================================================

"""
    resolve_one_to_one!(used_a::AbstractVector{Bool}, used_b::AbstractVector{Bool},
                        candidates; a = c -> c.first, b = c -> c.second) -> Vector{Int}

Greedy one-to-one reduction over an ALREADY-ORDERED candidate list: visit
`candidates` in the given order, keep a candidate when neither of its
endpoints is used yet, and mark both endpoints used. Returns the kept
indices into `candidates` (in visit order); `used_a`/`used_b` are mutated
in place, so pre-seeded entries (e.g. matches certified by an earlier
stage) are respected and callers can chain several candidate lists through
one pair of masks.

`a`/`b` extract the two endpoint indices from a candidate — the defaults
serve `NNMatch` (`m.first`/`m.second`); supply accessors for tuple lists.

The caller owns the ORDER: sort best-first (descending similarity, ascending
distance) for best-first assignment (the reduction `MutualNN` sets already satisfy, and the one
`KnnMatch(k)` sets need), by overlap for geometric association, etc.
"""
function resolve_one_to_one!(
        used_a::AbstractVector{Bool}, used_b::AbstractVector{Bool},
        candidates; a = c -> c.first, b = c -> c.second
    )
    kept = Int[]
    for (k, c) in enumerate(candidates)
        ia = a(c)
        ib = b(c)
        (used_a[ia] || used_b[ib]) && continue
        used_a[ia] = true
        used_b[ib] = true
        push!(kept, k)
    end
    return kept
end

"""
    resolve_one_to_one(matches::AbstractVector{NNMatch}, n_a, n_b;
                       metric = CosineSimilarity()) -> Vector{NNMatch}

Best-first one-to-one reduction of an `NNMatch` list: visit the best match
first and keep each `first`/`second` index at most once. `metric` must be the
metric the matches were scored with, because it decides what "best" means: the
highest score for a `DescriptorSimilarity` (the default, `CosineSimilarity()`),
the lowest for a `DescriptorDistance` such as `L2()`. `n_a`/`n_b` bound the two
index spaces. `MutualNN` output is returned unchanged (already one-to-one);
`KnnMatch(k)` output collapses to its best consistent assignment.
"""
function resolve_one_to_one(
        matches::AbstractVector{NNMatch}, n_a::Int, n_b::Int;
        metric::DescriptorMetric = CosineSimilarity()
    )
    # The key is metric-dependent BECAUSE it once was not: sorting by descending
    # score whatever the metric made an L2 match list keep each point's worst
    # candidate first.
    best_first = metric isa DescriptorSimilarity ? (m -> -Float64(m.score)) : (m -> Float64(m.score))
    perm = sortperm(matches; by = best_first)
    kept = resolve_one_to_one!(falses(n_a), falses(n_b), view(matches, perm))
    return matches[perm[kept]]
end

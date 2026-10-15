# Shared fixtures for the test items in test/*_tests.jl. ReTestItems evaluates
# this module once in each worker that runs an item listing it in `setup`.
@testsetup module Fixtures

using Colors: Gray, RGB
using FixedPointNumbers: N0f8
import KernelAbstractions as KA

export synth_blobs, BLOBS, IMG, FakeGPU, KA, Gray, RGB, N0f8

# =============================================================================
# KA ASIFT (IMAS-25) detector — CPU KernelAbstractions backend (GPU-free CI).
#
# These tests cover the public API,
# the CPU KA execution path, and detection semantics on synthetic blobs
# with known positions, scales, and polarities.
# =============================================================================

"Synthetic gray field with Gaussian blobs at known (row, col, σ, amplitude)."
function synth_blobs(H, W, blobs)
    img = fill(0.5f0, H, W)
    for (cy, cx, s, a) in blobs, j in 1:W, i in 1:H
        img[i, j] += a * exp(-((i - cy)^2 + (j - cx)^2) / (2s^2))
    end
    return Gray{N0f8}.(clamp.(img, 0, 1))
end

const BLOBS = [
    (40, 50, 4.0, 0.35),    # intensity peak → LIGHT blob (response < 0)
    (70, 96, 3.0, -0.3),   # intensity valley → DARK blob (response > 0)
    (30, 130, 6.0, 0.3),
]   # light
const IMG = synth_blobs(96, 160, BLOBS)

# A GPU backend no package defines a free-memory query for (see device_free_bytes).
struct FakeGPU <: KA.GPU end

# =============================================================================
# The matching rules, written out directly — full stable sorts, no kernels — as
# the reference the one KernelAbstractions matcher is tested against. `dm` has
# the gallery on its rows; `similarity = false` means a distance (smaller is
# better). Of equal scores the lower index ranks first, the kernels' rule.
# Returns sorted (i, j, score, ratio) tuples, as `match_key` makes of a match list.
# =============================================================================
using VisualGeometryFeatures: MutualNN, KnnMatch
export ref_match, match_key

function ref_match(
        dm::AbstractMatrix, strategy; similarity::Bool = true, ratio_threshold = nothing,
        min_similarity = nothing, max_distance = nothing
    )
    M, N = size(dm)
    best_first(v) = sortperm(v; rev = similarity)
    k = strategy isa KnnMatch ? min(strategy.k, N) : 1
    has_second = strategy isa KnnMatch ? k >= 2 : N >= 2
    out = Tuple{Int, Int, Float32, Float32}[]
    for i in 1:M
        p = best_first(dm[i, :])
        s1 = Float32(dm[i, p[1]])
        # No second neighbour, or an undefined similarity ratio (best ≤ 0), is
        # Inf32 — the matcher's rule, written out independently here.
        ratio = if !has_second || (similarity && s1 <= 0)
            Inf32
        else
            s2 = Float32(dm[i, p[2]])
            similarity ? s2 / s1 : s1 / s2
        end
        (ratio_threshold === nothing || ratio < ratio_threshold) || continue
        for r in 1:k
            j = p[r]
            strategy isa MutualNN && best_first(dm[:, j])[1] != i && continue
            s = Float32(dm[i, j])
            kept = similarity ? (min_similarity === nothing || s >= min_similarity) :
                (max_distance === nothing || s <= max_distance)
            kept && push!(out, (i, j, s, ratio))
        end
    end
    return sort(out)
end

match_key(ms) = sort([(m.first, m.second, m.score, m.ratio) for m in ms])
end # module Fixtures

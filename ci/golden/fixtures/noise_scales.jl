# Fixtures of the noise-scale estimators: residual vectors for Core's median-of-norms scale (noise_scale_tests'
# shapes: d = 1, 2, 3, 6, contaminated, biased, Float32), and RANSAC results / recorded numbers for Estimation's
# degrees-of-freedom-weighted pool (ransac_weights_tests' homography scenes).
using Test, StableRNGs, StaticArrays, StructArrays, Random
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation

const _NS_HS = VgtkEstimation.synthetic_scenes().HomographyScenes

function ns_residuals(seed)
    rng = StableRNG(seed)
    out = Pair{String, Any}[]
    for d in (1, 2, 3, 6), n in (5, 200)
        push!(out, "gauss_d$(d)_n$(n)_s0.37" => [SVector{d}(0.37 .* randn(rng, d)) for _ in 1:n])
    end
    clean = [SVector{2}(0.5 .* randn(rng, 2)) for _ in 1:150]
    push!(out, "contaminated_d2" => vcat(clean, [SVector{2}(40.0 .* randn(rng, 2)) for _ in 1:50]))
    push!(out, "biased_d1" => [SVector{1}(0.9 + 0.37 * randn(rng)) for _ in 1:100])
    push!(out, "float32_d2" => [SVector{2, Float32}(randn(rng, Float32, 2)) for _ in 1:64])
    return out
end
ns_exact_residuals() = [
    "rayleigh_513" => [SVector(3.0, 4.0), SVector(1.0, 0.0), SVector(6.0, 8.0)],
    "d1_signed" => SVector{1}.([-2.0, 1.0, 3.0]),
    "float32_three" => [SVector(1.0f0, 0.0f0), SVector(0.0f0, 2.0f0), SVector(0.0f0, 3.0f0)],
    "one" => [SVector(2.0, 0.0)],
]

# The homography scenes the pool test draws its RANSAC results on (one per "view").
ns_views(seed) = [let (s, d) = _NS_HS.homography_pairs(StableRNG(100 * seed + k), _NS_HS.H_TRUE;
        n_in = 60 + 20k, n_out = 20, σ = 0.3 + 0.2k); StructArray(s .=> d) end for k in 1:4]
ns_too_few() = let (s, d) = _NS_HS.homography_pairs(StableRNG(42), _NS_HS.H_TRUE; n_in = 3, n_out = 0, σ = 0.5); StructArray(s .=> d) end

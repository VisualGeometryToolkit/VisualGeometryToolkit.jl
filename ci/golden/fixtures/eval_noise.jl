# Fixtures of the noise-prior family: VisualGeometryToolkitEval's prior_specs_tests.jl σ̂² samples (the six values with
# one heavy-tailed, the same with its largest value moved to 1e3, a single value, none, and the inverse-gamma draw
# IG(3, 2e-3) of 200 000 values), with the IG draw repeated at each noise seed.
using StableRNGs, Random
using Random: MersenneTwister
using Distributions: InverseGamma as IGDist

function eval_noise_samples()
    σ2s = [1.0e-4, 4.0e-4, 2.5e-5, 9.0e-4, 1.6e-4, 3.0e-2]
    return [
        "six_heavy_tail" => σ2s,
        "six_tail_moved" => vcat(σ2s[1:end-1], 1.0e3),
        "single" => [1.0e-4],
        "empty" => Float64[],
        "ig3_2e-3_200k" => rand(MersenneTwister(4), IGDist(3.0, 2.0e-3), 200_000),
    ]
end

# The IG(3, 2e-3) draw at a noise seed (200 000 values, as the suite's).
eval_noise_ig_draw(seed) = rand(StableRNG(seed), IGDist(3.0, 2.0e-3), 200_000)

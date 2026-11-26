# =============================================================================
# Synthetic board surfaces — the stack's test warpings of a planar target
# =============================================================================

"""
    saddle_height(𝐛) -> Float64

The stack's synthetic saddle warping of a board: the height, in metres, at the chart position
`𝐛 ∈ [0, 1]²` (the board's page fractions), with `q = 𝐛 .- 0.5`,

    0.08 q₁² − 0.04 q₁ q₂ + 0.06 q₂²

— a quadratic about the board's centre (zero there, with no affine part: a pure bending), 0.025 m
and 0.045 m at the corners (the test scenes' name, "saddle", is historical). A `HeightField`'s nodal
heights are `[saddle_height(𝐛) for 𝐛 in nodes]`. The warping of Core's height-surface fixture
(`test/fixtures/height_surface/generate.jl`) and of RadiometricCamera's test scenes.
"""
saddle_height(𝐛) = (q = 𝐛 .- 0.5; 0.08 * q[1]^2 - 0.04 * q[1] * q[2] + 0.06 * q[2]^2)

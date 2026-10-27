# =============================================================================
# The contract conformance helpers — entry points (implemented in
# VisualGeometryToolkitCoreTestExt, loaded with `Test`). A scorer's package runs the
# first on every scorer, a local optimizer's the second on every strategy, a
# problem's the third on every problem.
# =============================================================================

"""
    test_scoring_contract(scoring, residuals; n = length(residuals), d_g = 1, sample = nothing)

The scoring contract on `scoring` over one residual vector, as tests:
[`score!`](@ref) type-stable and allocation-free after warm-up and
deterministic; [`inlier_count`](@ref) in `0:n`; [`consensus_order`](@ref) a
permutation of `1:n`; the derived [`consensus_set`](@ref) and
[`consensus_s2`](@ref) inferred; [`check_workspace`](@ref) accepting the
scorer's own workspace and refusing a foreign one; [`inlier_threshold`](@ref) a `Float64` where
[`has_inlier_threshold`](@ref); and, for a bounded-sum scorer
([`score_bound`](@ref)), the early exit tie-safe: a `bound` the score beats or
ties never changes it, and an exit returns a value at or below its bound.
Implemented by VisualGeometryToolkitCoreTestExt.
"""
function test_scoring_contract end

"""
    test_local_optimization_contract(lo, problem, θ, scoring; T = Float64, allocations = true)

The local-optimization contract on `lo` for `problem` from the hypothesis `θ`
under `scoring`, as tests: [`lo_workspace`](@ref) built once,
[`locally_optimize!`](@ref) type-stable, returning a [`lo_estimate`](@ref)
result of the problem's model type, honest about `residuals_current` (the
buffers then hold the refined model's residuals), allocation-free after
warm-up (`allocations = true`). Implemented by VisualGeometryToolkitCoreTestExt.
"""
function test_local_optimization_contract end

"""
    test_manifold_contract(m::Manifold; atol = 1e-12, allocations = true)

The manifold contract on `m`, as tests — what an optimizer that holds `m` as
its variable relies on: the tangent size a constant of the type
([`zero_tangent`](@ref) inferred, static); `recenter` inferred to
`typeof(m)` for a step of `m`'s own float type in any static container
(`SVector`, `MVector`; a plain `Vector` returns that type too) and allocation-free
(`allocations = false` records a known allocation as `@test_broken`); [`retract_jacobian`](@ref) at the
anchor and at a step equal to ForwardDiff through `retract` within
`atol`; and the chart's own invariants (the essential chart's translation on
the unit sphere after a step). Run it on every manifold family a package
defines. Implemented by VisualGeometryToolkitCoreTestExt.
"""
function test_manifold_contract end

"""
    test_constraint_problem(problem, θ)

The constraint-problem protocol on `problem` at the model `θ`, as tests:
`data_size`/`codimension`/`minimal_sample_size` integers, [`residuals!`](@ref)
type-stable and allocation-free after warm-up and equal to
[`whitened_residual`](@ref) at every point. Implemented by
VisualGeometryToolkitCoreTestExt.
"""
function test_constraint_problem end

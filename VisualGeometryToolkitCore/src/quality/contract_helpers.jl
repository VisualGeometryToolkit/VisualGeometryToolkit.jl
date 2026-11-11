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
scorer's own workspace and refusing a foreign one; [`decision_boundary`](@ref) a `Float64` where
[`has_decision_boundary`](@ref); and, for a bounded-sum scorer
([`score_bound`](@ref)), the early exit tie-safe: a `bound` the score beats or
ties never changes it, and an exit returns a value at or below its bound.
Implemented by VisualGeometryToolkitCoreTestExt.
"""
function test_scoring_contract end

"""
    test_local_optimization_contract([T = Float64,] lo, problem, θ, scoring; allocations = true)

The local-optimization contract on `lo` for `problem` from the hypothesis `θ`
under `scoring`, in the element type `T` (the workspace's and score buffers'),
as tests: [`lo_workspace`](@ref) built once,
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
`data_size`/`codimension`/`minimal_sample_size` integers,
[`mahalanobis_distances!`](@ref) type-stable, allocation-free after warm-up,
`≥ 0`, and equal to [`mahalanobis_distance`](@ref) at every point to rounding:
`rtol = 1e-11` of the point's distance or `1e-14` of the largest finite
distance (a non-finite one identical); and [`whitened_residual`](@ref) in the
constraint's shape at every point — a number at `d_g = 1`, `d_g` components
otherwise, so a scalar override at `d_g ≥ 2` (a distance under the residual's
name, the pre-0.7 hook) is refused — and with its norm equal to the point's
distance to the same rounding (a non-finite distance, a gate's, matched by a
non-finite residual), so the two hooks describe one point. The rounding is of the measurements, not of the residual: a
reprojection residual is a difference of ~1000 px coordinates, so two paths
differ by ~1000·eps px whatever the residual (PnP measured 4.8e-14 px on a
1.08 px residual, E 5.7e-14 px; 2026-11). Not bit for bit: the batch loop and a single call may contract
StaticArrays' `muladd`s into FMAs differently once the loop vectorizes, so a
bitwise rule held for a problem only by forbidding inlining (PnP's
`@noinline`, 15–22% per point) — result accuracy, not bit accuracy.
Implemented by VisualGeometryToolkitCoreTestExt.
"""
function test_constraint_problem end

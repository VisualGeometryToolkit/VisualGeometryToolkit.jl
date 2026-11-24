# =============================================================================
# FitResult — the solution of every estimation problem (Core 0.14)
# =============================================================================
#
# SciML's solution interface first (`u`, `resid`, `retcode`, `stats`, `prob`,
# `alg`, `original`; SciMLBase's NonlinearSolution), extended by the stack's
# `stop_rule`, `rms`, `n` and `metadata` (the design record,
# docs/design/sciml-mapping.md §4, and the owner's rulings of 2026-11-23).
#
# TYPE STABILITY IS A HARD REQUIREMENT (2026-09-02 design round): no
# Union-typed fields — the metadata VARIETY is carried by the type parameter
# `M`, one CONCRETE type per solver family, selected at COMPILE TIME (supply
# the type, never a runtime flag that switches the return type). Derived
# quantities (covariance) are functions computed on demand, never
# maybe-slots.
#
# One exception: a RANSAC's `u` may be absent, its `U` the union of the model
# type and `Nothing` (VisualGeometryToolkitEstimation's `RansacEstimate`) — a
# search that finds no model has no estimate, and an absent model is neither
# runtime metadata variety nor an optional derived quantity. The union is
# stored inline with a one-byte tag and split, not boxed, and a result never
# goes to the device (measured 2026-11-03: a `Union{Ellipse{Float64}, Nothing}`
# field takes 48 bytes against 40). Every other result keeps a concrete `U`.
#
# ONE WAY PER QUANTITY (owner, 2026-11-23): a result's own data is read by
# field (`r.u`, `r.rms`, `r.metadata.quality`), a cross-type contract quantity
# by its function only (`weights(r)`, `noise_scale(r)`), so no property is
# forwarded into the metadata (Core 0.13's forwarding, a second spelling of
# every metadata field, is gone).

"""
    SolveStats(; nsteps = 0, nf = 0, njacs = 0, nfactors = 0, nsolve = 0)

The work counters of a solve, a [`FitResult`](@ref)'s `stats`, under SciMLBase's
`NLStats` field names: `nsteps` the iterations (an LM's accepted steps, an IRLS
fit's cycles; a RANSAC's trials; `0` for a closed form), `nf` the residual
evaluations, `njacs` the Jacobians, `nfactors` the factorizations, `nsolve` the
linear solves. A counter an algorithm does not keep is `0` (SciML's
`OptimizationStats` convention); each algorithm documents the ones it keeps.
Immutable and isbits, unlike SciML's mutable `NLStats`, so a device solve
returns it (design record, deviation D11).
"""
struct SolveStats
    nsteps::Int
    nf::Int
    njacs::Int
    nfactors::Int
    nsolve::Int
end
SolveStats(; nsteps::Integer = 0, nf::Integer = 0, njacs::Integer = 0, nfactors::Integer = 0, nsolve::Integer = 0) =
    SolveStats(Int(nsteps), Int(nf), Int(njacs), Int(nfactors), Int(nsolve))

"""
    FitResult{U, M, R, Prob, Alg, Orig}

The solution of an estimation problem: SciML's solution fields first, then the
stack's extensions. Read every field AS A FIELD — `r.u`, `r.retcode`,
`r.stats.nsteps`, `r.stop_rule`, `r.rms`, `r.n`, `r.metadata.<field>` — and the
success test is [`successful_retcode`](@ref)`(r)`. The cross-type contract
quantities (implemented by other types too) are read by their functions only:
[`noise_scale`](@ref), [`scale_source`](@ref), [`weights`](@ref),
[`inlier_mask`](@ref), [`dof`](@ref), `s2`, [`found_model`](@ref) and
[`retcode_symbol`](@ref) — one way per quantity (conventions/data-arguments.md).

SciML's fields (`SciMLBase.NonlinearSolution`):
- `u::U` — the estimate: the model, camera, calibration record… A find that
  found no model, and a degenerate closed form, hold `nothing` (design record D10).
- `resid::R` — the per-datum whitened residual distances at `u`, computed once
  at the solution (`nothing` for a solve that reports none; D12).
- `retcode::ReturnCode.T` — the verdict ([`ReturnCode`](@ref)).
- `stats::SolveStats` — the work counters ([`SolveStats`](@ref)).
- `prob::Prob`, `alg::Alg` — the problem solved and the algorithm that solved
  it (`nothing` for a result built inside a solver, before the front end
  attaches them).
- `original::Orig` — a wrapped backend's own return (PoseLib's); `nothing` for
  the stack's own solvers.

The stack's extensions:
- `stop_rule::StopRule` — which rule ended the solve ([`StopRule`](@ref)).
- `rms::Float64` over `n::Int` residuals (a robust fit's support).
- `metadata::M` — the solver family's diagnostics, ONE concrete type per family
  (`nothing` for a lean result): never a Union field.

The type parameters lead with `U` and `M`, so `FitResult{U, M}` names the
results of a model type and a metadata type whatever the rest.

Like SciML's solutions, a result's TYPE carries its problem's and its
algorithm's types (`Prob`, `Alg`), and the result keeps a reference to its data
through `prob`. To collect the results of different problems, hold `r.u` (or a
concretely typed container of results), not a `Vector{FitResult}`, and drop
the results whose data should be freed.

    FitResult(u, retcode, stop_rule, stats, rms, n, metadata; resid = nothing, prob = nothing, alg = nothing, original = nothing)
"""
struct FitResult{U, M, R, Prob, Alg, Orig}
    u::U
    resid::R
    retcode::ReturnCode.T
    stats::SolveStats
    prob::Prob
    alg::Alg
    original::Orig
    stop_rule::StopRule
    rms::Float64
    n::Int
    metadata::M
end
# THE constructor: `FitResult{U}(…)` fixes the result's model type (a `Union{M, Nothing}` for a solve that can return
# none), every other parameter inferred; `FitResult(u, …)` takes `U` from `u`. Called without keywords (a device solve's
# form) it is an ordinary positional call.
@inline function FitResult{U}(u, code::ReturnCode.T, rule::StopRule, stats::SolveStats, rms::Real, n::Integer, metadata::M;
        resid::R = nothing, prob::Prob = nothing, alg::Alg = nothing, original::Orig = nothing) where {U, M, R, Prob, Alg, Orig}
    return FitResult{U, M, R, Prob, Alg, Orig}(u, resid, code, stats, prob, alg, original, rule, Float64(rms), Int(n), metadata)
end
@inline FitResult(u::U, code::ReturnCode.T, rule::StopRule, stats::SolveStats, rms::Real, n::Integer, metadata; kwargs...) where {U} =
    FitResult{U}(u, code, rule, stats, rms, n, metadata; kwargs...)

# THE rebuild: `r` with the estimate `u` (of the result model type `U`), the problem and the algorithm given, every
# other field as it was.
@inline _rebuild(r::FitResult, ::Type{U}, u, prob, alg) where {U} =
    FitResult{U}(u, r.retcode, r.stop_rule, r.stats, r.rms, r.n, r.metadata; resid = r.resid, prob, alg, original = r.original)

"""
    with_problem(r::FitResult, prob, alg) -> FitResult

`r` with the problem and algorithm that produced it recorded (`prob`, `alg`),
every other field as it was: what the generic `solve` does to the result of a
`__solve` method, the one place they are set.
"""
@inline with_problem(r::FitResult{U}, prob, alg) where {U} = _rebuild(r, U, r.u, prob, alg)
with_problem(x, prob, alg) = x   # a minimal solve's FixedModels carries no problem

# `BundleMetadata`, the bundle refinements' metadata, is in base/bundle_metadata.jl (it records a ScaleSource).

"""
    rms(residuals::AbstractVector, n::Integer = length(residuals)) -> Float64

The root-mean-square residual `√(Σ rᵢ² / n)` a fit reports (its `rms`): over `n`
points, so a point's stacked residual rows (two per pixel point) enter as its
squared residual norm — `rms(rows, n_points)` — and scalar errors as themselves,
`rms(errors)`. The one place it is computed.
"""
rms(residuals::AbstractVector, n::Integer = length(residuals)) = sqrt(sum(abs2, residuals) / n)

"""
    map_estimate(f, r::FitResult) -> FitResult

`r` with `f` applied to its estimate `r.u`, every other field as it was: the
one way a stage hands on a result whose model it re-expresses (a unit
attached, frame labels restored). A RANSAC result's `u` is `Union{M, Nothing}`:
`f` maps a present model, an absent one stays `nothing`, and the result keeps
the union over `f`'s image of `M` (`f` needs no method for `nothing`). Any
other result's `u` is concrete and simply mapped.
"""
function map_estimate(f, r::FitResult{P}) where {P}
    E = r.u
    Nothing <: P || (u = f(E); return _rebuild(r, typeof(u), u, r.prob, r.alg))
    N = Base.promote_op(f, Base.nonnothingtype(P))   # f's image of the present model (f needs no `nothing` method)
    return _rebuild(r, Union{N, Nothing}, E === nothing ? nothing : f(E), r.prob, r.alg)
end

# The cross-type contract quantities a result answers through its metadata: read by these functions ONLY
# (a FitResult forwards no property into its metadata, so `weights(r)` has no second spelling).
"""
    weights(r::FitResult) -> AbstractVector

Per-datum weights in [0, 1], recorded in the metadata: a plain
vector in the data's order (a lazy all-ones `unit_weights` for an L2 result).
What they mean is the producer's `weight_kind`, and `inlier_mask(r)` cuts them
by it. Errors on a metadata type that carries none — including the lean
`nothing`. A contract generic: a metadata that stores its weights otherwise
adds a method.
"""
weights(r::FitResult) = r.metadata.weights

"""
    dof(r::FitResult) -> Int

Residual degrees of freedom, recorded in the metadata. Errors on
a metadata type that carries none.
"""
dof(r::FitResult) = r.metadata.dof

"""
    s2(r::FitResult) -> Float64

The residual variance `ŝ² = RSS/ν` a fit recorded in its metadata (a bundle's, a robot-world calibration's): read by
this function only (spec §2). Errors on a metadata type that carries none. A contract generic, as `dof`.
"""
s2(r::FitResult) = r.metadata.s2

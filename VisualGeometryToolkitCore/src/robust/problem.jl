# =============================================================================
# The constraint-problem protocol — the residual-level contract scoring and
# local optimization read (moved from RANSACScoringDoneRight's interface.jl).
# =============================================================================
#
# A scorer scores models from per-point residuals. It does NOT know about
# minimal samples, solvers, or RANSAC workspaces — only this thin contract:
#
#   data_size(p)                           — number of data points
#   codimension(p)                         — constraint codimension d_g
#   minimal_sample_size(p)                 — model DOF m (for ν = (k−m)·d_g)
#   constraint_with_jacobians(p, model, i) — (g, ∂ₓg, ∂_θg) at point i
#   measurement_covariance(p, i)           — Σ̃_x of point i's measurements (default I)
#   constraint_rounding_scale(p, model, i) — the rounding scales at point i (default
#                                            `nothing`: the exact-zero degeneracy rule)
#   solver_covariance(p, model, idx)       — Σ̃_θ of the model estimated from the points
#                                            idx; REQUIRED for predictive scoring (the
#                                            fallback throws), `nothing` = degenerate sample
#   provides_solver_covariance(P)          — trait: whether P defines solver_covariance
#
# `residuals!` is derived from `constraint_with_jacobians` (Mahalanobis distance);
# a problem supplies a faster path through `residual_context` (per model, once)
# and `whitened_residual` (per point), never by rewriting the loop.
#
# A local optimization reads one more verb of a problem: the weighted refit
# `refine!(R, problem, θ, w)` on the problem's own workspace
# `refine_workspace(P, n)` (VisualGeometryToolkitEstimation implements both per problem).
# =============================================================================

"""
    AbstractConstraintProblem

Supertype of a problem that scoring and local optimization can run on.
Implementors provide `data_size`, `codimension`, `minimal_sample_size`, and
`constraint_with_jacobians`; optionally `measurement_covariance` and
`constraint_rounding_scale` (the point's noise and its degeneracy scales),
`solver_covariance` with
`provides_solver_covariance` (predictive scoring), `residual_context` with
`whitened_residual` (a Jacobian-free residual path), and `refine!` with
`refine_workspace` (the weighted refit local optimizations call).
"""
abstract type AbstractConstraintProblem end

"""
    data_size(problem) -> Int

Total number of data points the problem scores over.
"""
function data_size end

"""
    codimension(problem) -> Int

Constraint codimension `d_g`: the number of independent constraint equations a
true datum satisfies (1 for F/E/line scalar residual, 2 for homography / PnP).
A scoring reports the codimension it was built for through the same verb.
"""
function codimension end

"""
    minimal_sample_size(problem) -> Int

Model degrees of freedom `m` used in the residual DOF `ν = (k − m)·d_g`.
"""
function minimal_sample_size end

"""
    solver_covariance(problem, model, sample_indices) -> SMatrix or nothing

Model parameter covariance `Σ̃_θ` propagated through the minimal solver, for
predictive scoring. A provider returns `nothing` for a DEGENERATE minimal
sample, one whose solver Jacobian does not exist, and predictive scoring then
SKIPS that hypothesis like a failed solve.

A provider declares itself with [`provides_solver_covariance`](@ref). A problem
type that defines no provider cannot be scored predictively: the fallback method
throws an `ArgumentError` naming the problem type.
"""
solver_covariance(p::AbstractConstraintProblem, _model, ::AbstractVector{Int}) =
    throw(_no_solver_covariance(typeof(p)))

"""
    provides_solver_covariance(::Type{<:AbstractConstraintProblem}) -> Bool

Whether problem type `P` defines [`solver_covariance`](@ref), so that it can be
scored predictively. Default `false`; a provider declares
`provides_solver_covariance(::Type{<:P}) = true` beside its method.
"""
provides_solver_covariance(::Type{<:AbstractConstraintProblem}) = false

function _no_solver_covariance(::Type{P}) where {P}
    name = nameof(P)
    return ArgumentError(
        "predictive scoring needs a solver covariance, and $name provides none: define " *
            "`VisualGeometryToolkitCore.solver_covariance(::$name, model, sample_indices)` and " *
            "declare `VisualGeometryToolkitCore.provides_solver_covariance(::Type{<:$name}) = true`, " *
            "or score $name model-certain"
    )
end

"""
    residual_context(problem, model) -> ctx

What [`residuals!`](@ref) computes ONCE per model before its per-point loop, and
hands to every [`whitened_residual`](@ref)`(problem, ctx, i)`. The default is the
model itself, so a problem that defines neither hook scores exactly as before.

A problem overrides it when part of the residual depends on the model alone and
is costly: a relative-pose bundle's cheiral pose decomposition and inverse
intrinsics, an essential matrix widened to `Float64` once rather than per point.
It then defines `whitened_residual(problem, ctx::C, i)` for its context type `C`.
"""
residual_context(::AbstractConstraintProblem, model) = model

"""
    whitened_residual(problem, ctx, i) -> Real

The whitened residual of data point `i` under `model`: the norm of `L⁻¹gᵢ`,
where `LLᵀ = Σ̃_{gᵢ} = ∂ₓgᵢ·Σ̃_x·∂ₓgᵢᵀ`, so that its square is the Mahalanobis
distance `gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ` (`~ χ²(d_g)` under H₀). SIGNED for a scalar
constraint (`d_g = 1`: `gᵢ/√Σ̃_{gᵢ}`), the non-negative norm for a vector one.

The default is [`whitened_constraint_residual`](@ref)`(problem, model, i)`: the
Sampson residual of `constraint_with_jacobians(problem, model, i)` (the geometry
layer of [`sampson_residual_with_normal`](@ref), at the problem's own currency
of `G_x` and its `constraint_rounding_scale`), whitened by the point's
`measurement_covariance` — the stack's one Sampson path and its one
degeneracy rule (`±Inf`
for an unsatisfiable degenerate point, `0` for one that vanishes exactly). A
problem overrides it when a cheaper Jacobian-free route to the SAME quantity
exists; the override must keep the sign contract. [`residuals!`](@ref) is this
per point, at `ctx = `[`residual_context`](@ref)`(problem, model)`.
"""
@inline whitened_residual(p::AbstractConstraintProblem, model, i::Int) =
    whitened_distance(whitened_constraint_residual(p, model, i))

"""
    residuals!(r, problem, model)

Fill `r[1:data_size(problem)]` with whitened residuals such that `r[i]^2` is the
Mahalanobis distance `gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ` (so `r[i]^2 ~ χ²(d_g)` under H₀).
It is [`whitened_residual`](@ref) at every point, SIGNED for a scalar constraint.
The per-model work is [`residual_context`](@ref), computed once; a problem
overrides that hook and `whitened_residual`, not this loop.
"""
function residuals!(r::AbstractVector, p::AbstractConstraintProblem, model)
    ctx = residual_context(p, model)
    @inbounds for i in eachindex(r)
        r[i] = whitened_residual(p, ctx, i)
    end
    return r
end

"""
    measurement_covariance(problem, i) -> Σ̃_x

The covariance of data point `i`'s measurements, over the coordinates its
constraint's `G_x` differentiates (a correspondence's pair block): what
[`whitened_constraint_residual`](@ref) whitens by. A problem's measurements
carry their noise (conventions/measurement-noise.md) — `I` for plain points,
the default — and a problem that stores covariances (a correspondence set's
`Σ₁`/`Σ₂` columns, a line fit's per-point shapes) returns point `i`'s.
"""
measurement_covariance(::AbstractConstraintProblem, ::Int) = I

"""
    whitened_constraint_residual(problem, model, i) -> T | SVector{dg}

Data point `i`'s constraint (`constraint_with_jacobians(problem, model, i)`) as
a whitened residual, in the constraint's shape (a number for a scalar
constraint): its Sampson residual at the problem's
[`constraint_rounding_scale`](@ref)`(problem, model, i)` (base/sampson.jl's
geometry layer and its one degeneracy rule), whitened by the point's
[`measurement_covariance`](@ref) (one Cholesky of `G_x·Σ̃_x·G_xᵀ`).
"""
@inline function whitened_constraint_residual(p::AbstractConstraintProblem, model, i::Int)
    g, G_x, G_θ = constraint_with_jacobians(p, model, i)
    s, _ = _whitened_constraint(g, G_x, G_θ, constraint_rounding_scale(p, model, i), measurement_covariance(p, i), nothing)
    return _constraint_shape(g, s)
end

"""
    whitened_constraint_rows(problem, model, i) -> (r, J_w)

Data point `i`'s constraint as a least-squares row: its
[`whitened_constraint_residual`](@ref) and its model rows `L_Σ⁻¹·G_θ` under the
SAME factor, held fixed (the Gauss–Newton row of a frozen whitening;
`L_Σ L_Σᵀ = G_x·Σ̃_x·G_xᵀ`). In the constraint's shape: a number and an
`SVector` row for a scalar constraint, an `SVector` and a matrix for a vector
one. Degenerate (the gradient vanishes at the rounding scale, or `Σ̃_x` is
singular along the normal): the rule's residual (`0` when the constraint
vanishes, else `±Inf`) and ZERO rows. What a predictive scorer reads per point:
the residual and the rows the model covariance is propagated through.
"""
@inline function whitened_constraint_rows(p::AbstractConstraintProblem, model, i::Int)
    g, G_x, G_θ = constraint_with_jacobians(p, model, i)
    s, J = _whitened_constraint(g, G_x, G_θ, constraint_rounding_scale(p, model, i), measurement_covariance(p, i), _as_row(G_θ))
    return _constraint_shape(g, s), _rows_shape(G_θ, J)
end
# Rows back in `G_θ`'s shape.
@inline _rows_shape(::StaticVector{n}, J::StaticMatrix{1, n, T}) where {n, T} = SVector{n, T}(Tuple(J))
@inline _rows_shape(::StaticMatrix, J::StaticMatrix) = J

@inline whitened_distance(r::Real) = r
@inline whitened_distance(r::StaticVector) = norm(r)

# Squared norm: `rᵢ²` for scalar, `rᵢᵀrᵢ` for vector — ‖rᵢ‖² from the
# shape-whitened residual.
@inline sq_norm(r::Real) = r * r
@inline sq_norm(r::StaticVector) = dot(r, r)

"""
    refine!(R, problem, θ, w) -> θ′ | nothing

The weighted refit of `problem` from `θ` under per-point weights `w`
(`w[1:data_size(problem)]`), on the problem's refit workspace `R`
([`refine_workspace`](@ref)) — the model block of every IRLS-family local
optimization. `nothing` means the solve failed (an underdetermined or singular
system); the caller keeps its incumbent.
"""
function refine! end

"""
    refine_workspace(::Type{P}, n::Int) -> R

The scratch of [`refine!`](@ref) for a problem of type `P` with up to `n` data
points, built once per composed workspace and reused across calls. `nothing`
for a problem whose refit needs none.
"""
function refine_workspace end

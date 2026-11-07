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
# `mahalanobis_distances!` is derived from `constraint_with_jacobians` (the
# norm of each point's `whitened_residual`); a problem supplies a faster path
# through `residual_context` (per model, once) and `mahalanobis_distance` (per
# point), never by rewriting the loop. `whitened_residuals!` stacks the signed
# `whitened_residual` rows.
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
`mahalanobis_distance` (a Jacobian-free distance path), and `refine!` with
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

What [`mahalanobis_distances!`](@ref) and [`whitened_residuals!`](@ref)
compute ONCE per model before their per-point loops, and hand to every
[`mahalanobis_distance`](@ref)`(problem, ctx, i)` and
[`whitened_residual`](@ref)`(problem, ctx, i)`. The default is the model
itself, so a problem that defines neither hook scores exactly as before.

A problem overrides it when part of the residual depends on the model alone and
is costly: a relative-pose bundle's cheiral pose decomposition and inverse
intrinsics, an essential matrix widened to `Float64` once rather than per point.
It then defines the per-point hooks it answers for its context type `C`
(`constraint_with_jacobians(problem, ctx::C, i)`, which the defaults read, or
`mahalanobis_distance(problem, ctx::C, i)` directly).
"""
residual_context(::AbstractConstraintProblem, model) = model

"""
    whitened_residual(problem, ctx, i) -> T | SVector{d_g}

Data point `i`'s constraint as a whitened residual, SIGNED and in the
constraint's shape: a number for a scalar constraint (`d_g = 1`, the signed
distance `gᵢ/√Σ̃_{gᵢ}`), an `SVector` of its `d_g` components for a vector one
(`L⁻¹gᵢ`, `LLᵀ = Σ̃_{gᵢ} = ∂ₓgᵢ·Σ̃_x·∂ₓgᵢᵀ`), so that `‖rᵢ‖²` is the squared
Mahalanobis distance `gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ` (`~ χ²(d_g)` under H₀). `ctx` is
[`residual_context`](@ref)`(problem, model)`, the model by default. GTSAM's
`NoiseModelFactor::whitenedError` for this point.

The default is the Sampson residual of `constraint_with_jacobians(problem, ctx,
i)` at the problem's [`constraint_rounding_scale`](@ref)`(problem, ctx, i)`
(base/sampson.jl's geometry layer, the stack's one Sampson path and its one
degeneracy rule: `±Inf` for an unsatisfiable degenerate point, `0` for one that
vanishes exactly), whitened by the point's [`measurement_covariance`](@ref) (one
Cholesky of `G_x·Σ̃_x·G_xᵀ`). An override returns the same shape: a scalar for a
vector constraint is a distance, not a residual, and
[`test_constraint_problem`](@ref) refuses it. The distance is
[`mahalanobis_distance`](@ref); the rows over every point
[`whitened_residuals!`](@ref).
"""
@inline function whitened_residual(p::AbstractConstraintProblem, ctx, i::Int)
    g, G_x, G_θ = _three(constraint_with_jacobians(p, ctx, i))
    s, _ = _whitened_constraint(g, G_x, G_θ, constraint_rounding_scale(p, ctx, i), measurement_covariance(p, i), nothing)
    return _constraint_shape(g, s)
end

"""
    mahalanobis_distance(problem, ctx, i) -> Real

The Mahalanobis distance of data point `i`, `≥ 0`: the norm of its
[`whitened_residual`](@ref), so that its square is the squared Mahalanobis
distance `gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ` (`~ χ²(d_g)` under H₀). `ctx` is
[`residual_context`](@ref)`(problem, model)`, the model by default. What every
scorer reads per point: scores take its square, which the residual's sign does
not move.

A problem overrides it when a cheaper Jacobian-free route to the SAME distance
exists (a reprojection error's norm), or to put a gate before it (a point the
lens cannot place: `OUT_OF_FIELD_RESIDUAL`); an override returns a value
`≥ 0`. [`mahalanobis_distances!`](@ref) is this at every point.
"""
@inline mahalanobis_distance(p::AbstractConstraintProblem, ctx, i::Int) =
    _residual_norm(_as_vector(whitened_residual(p, ctx, i)))   # `|r|` for a scalar constraint (base/sampson.jl)

"""
    mahalanobis_distances!(d, problem, model) -> d

Fill `d[i]` with [`mahalanobis_distance`](@ref)`(problem, ctx, i)` for every
index of `d` (`1:data_size(problem)` in the RANSAC loop), `≥ 0`, so that
`d[i]^2 ~ χ²(d_g)` under H₀. The per-model work is
[`residual_context`](@ref), computed once; a problem overrides that hook and
`mahalanobis_distance`, not this loop. The signed rows are
[`whitened_residuals!`](@ref).
"""
function mahalanobis_distances!(d::AbstractVector, p::AbstractConstraintProblem, model)
    ctx = residual_context(p, model)
    @inbounds for i in eachindex(d)
        d[i] = mahalanobis_distance(p, ctx, i)
    end
    return d
end

"""
    whitened_residuals!(r, problem, model) -> r

The [`whitened_residual`](@ref) of every data point, stacked: point `i`'s `d_g`
signed components in rows `d_g(i−1)+1 … d_g·i` of `r`, `d_g =
`[`codimension`](@ref)`(problem)`, unweighted, at `ctx = `[`residual_context`](@ref)`(problem,
model)`, computed once. `r` holds every point, `d_g·data_size(problem)` rows;
another length is a `DimensionMismatch`, and so is a point whose residual is
not `d_g` components (a scalar override of a vector constraint): it fails
loudly rather than reading past the data. The problem-level form of the
model-level `whitened_residuals!(r, M, xs)` (unweighted too); its row norms
are [`mahalanobis_distances!`](@ref)'s.
"""
whitened_residuals!(r::AbstractVector, p::AbstractConstraintProblem, model) = _stack_whitened_residuals!(r, p, model)
# A model that is itself a vector would also match the model-level
# `whitened_residuals!(r, M, xs::AbstractVector)`: the problem decides.
whitened_residuals!(r::AbstractVector, p::AbstractConstraintProblem, model::AbstractVector) = _stack_whitened_residuals!(r, p, model)
function _stack_whitened_residuals!(r::AbstractVector, p::AbstractConstraintProblem, model)
    dg, n = codimension(p), data_size(p)
    length(r) == dg * n || throw(DimensionMismatch(
        "whitened_residuals!: r has $(length(r)) rows; $n points of $dg components need $(dg * n)"))
    n == 0 && return r
    ctx = residual_context(p, model)
    for i in 1:n
        _store_rows!(r, whitened_residual(p, ctx, i), i, dg)
    end
    return r
end
# Point `i`'s components in its rows; a number is its one component.
@inline function _store_rows!(r, w, i::Int, dg::Int)
    length(w) == dg || throw(DimensionMismatch(
        "whitened_residuals!: point $i's whitened_residual has $(length(w)) components, the constraint $dg " *
            "(a scalar for a vector constraint is a distance: override `mahalanobis_distance`)"))
    @inbounds for k in 1:dg
        r[dg * (i - 1) + k] = w[k]
    end
    return nothing
end

"""
    measurement_covariance(problem, i) -> Σ̃_x

The covariance of data point `i`'s measurements, over the coordinates its
constraint's `G_x` differentiates (a correspondence's pair block): what
[`whitened_residual`](@ref) whitens by. A problem's measurements
carry their noise (conventions/measurement-noise.md) — `I` for plain points,
the default — and a problem that stores covariances (a correspondence set's
`Σ₁`/`Σ₂` columns, a line fit's per-point shapes) returns point `i`'s.
"""
measurement_covariance(::AbstractConstraintProblem, ::Int) = I

"""
    whitened_constraint_rows(problem, model, i) -> (r, J_w)

Data point `i`'s constraint as a least-squares row: its
[`whitened_residual`](@ref) and its model rows `L_Σ⁻¹·G_θ` under the
SAME factor, held fixed (the Gauss–Newton row of a frozen whitening;
`L_Σ L_Σᵀ = G_x·Σ̃_x·G_xᵀ`). In the constraint's shape: a number and an
`SVector` row for a scalar constraint, an `SVector` and a matrix for a vector
one. Degenerate (the gradient vanishes at the rounding scale, or `Σ̃_x` is
singular along the normal): the rule's residual (`0` when the constraint
vanishes, else `±Inf`) and ZERO rows. What a predictive scorer reads per point:
the residual and the rows the model covariance is propagated through.

The residual is GTSAM's `NoiseModelFactor::whitenedError` for this point (its
`unwhitenedError` is the constraint `g`), with the Sampson residual's noise
along the constraint's normal; the rows are the frozen-whitening Gauss–Newton
rows (the LM's exact rows are [`whitened_jacobian_rows!`](@ref)'s).
"""
@inline function whitened_constraint_rows(p::AbstractConstraintProblem, model, i::Int)
    g, G_x, G_θ = _three(constraint_with_jacobians(p, model, i))
    s, J = _whitened_constraint(g, G_x, G_θ, constraint_rounding_scale(p, model, i), measurement_covariance(p, i), _as_row(G_θ))
    return _constraint_shape(g, s), _rows_shape(G_θ, J)
end
# A problem's constraint is exactly `(g, G_x, G_θ)`: a longer tuple (the retired
# noise and rounding-scale slots) is refused rather than silently truncated.
@inline _three(cd::NTuple{3, Any}) = cd
@noinline _three(cd::Tuple) = throw(ArgumentError(
    "constraint_with_jacobians(problem, model, i) returns (g, G_x, G_θ), got $(length(cd)) values: " *
        "a point's noise is `measurement_covariance(problem, i)` and its rounding scale " *
        "`constraint_rounding_scale(problem, model, i)`"))
# Rows back in `G_θ`'s shape.
@inline _rows_shape(::StaticVector{n}, J::StaticMatrix{1, n, T}) where {n, T} = SVector{n, T}(Tuple(J))
@inline _rows_shape(::StaticMatrix, J::StaticMatrix) = J

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

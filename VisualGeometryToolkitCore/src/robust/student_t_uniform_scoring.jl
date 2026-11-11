# =============================================================================
# StudentTUniformScoring — the Student-t + uniform marginal score on the
# scoring contract (moved from RANSACScoringDoneRight's scoring.jl and
# extreme_value_sweep.jl, 2026-11-11, not rewritten)
#
#   1. StudentTUniformScoring{P, DG, L, O} (P = Nothing | Predictive)
#   2. its workspace and the buffer-based score_residuals!/score_sq_residuals!
#   3. the sweeps it dispatches to: the supplied half-width, the predictive
#      penalties, and the estimated-support sweep (an outlier model's hooks)
#   4. its scoring contract (scoring_workspace, score!, inlier_count, …)
#
# How the score eliminates its nuisances (the paper's table):
#
#                 scale supplied            scale marginalised
#   profile z     RANSAC, MSAC              StudentTUniformScoring  <-- here
#   marginal z    MLESAC, MAGSAC++          intractable (2^N partitions)
#
# The scale σ² is integrated out against the scorer's scale prior (Jeffreys or
# the inverse-gamma `IGPrior`), the labels are profiled over the sorted
# partitions, and the label prior adds a term per inlier count. Its priors are
# REQUIRED: the stack's default is its configuration's (VisualGeometryToolkitEstimation's
# configs/default.toml `[scoring.student_t_uniform]`, read by `scoring_from_toml`).
# =============================================================================

# -----------------------------------------------------------------------------
# Uncertain{M} dispatch
# -----------------------------------------------------------------------------
#
#   model::M             → model-certain (Σ_θ = 0, treat θ as exact)
#   model::Uncertain{M}  → model-uncertain (Σ_θ = cov(model)), the predictive score
#
# Every problem's `mahalanobis_distances!` returns the norms of its whitened
# residuals, ‖rᵢ‖² the squared Mahalanobis distance:
#
#   model::M (certain)     → scores[i] = rᵢ²,         penalties[i] = 0
#   model::Uncertain{M}   → scores[i] = rᵢᵀ Ṽᵢ⁻¹ rᵢ, penalties[i] = log|Ṽᵢ|
#
# Measurement-space ℓᵢ = 0: the outlier density in whitened constraint space
# rᵢ = Lᵢ⁻¹gᵢ is (2·outlier_halfwidth)^{-dg}, constant and θ-independent. The inlier
# Gaussian normalization |Σ̃_{gᵢ}|^{-1/2} cancels with the whitening Jacobian.
# Model-uncertainty ℓᵢ = log|Ṽᵢ|: the predictive inflation Ṽᵢ = I + G Σ̃_θ Gᵀ ≥ I
# deflates qᵢ ≤ ‖rᵢ‖², so without the penalty an uncertain model would always
# score at least as well as a certain one; the sweep accumulates −½ Σᵢ∈I ℓᵢ.
# -----------------------------------------------------------------------------

"""
    Predictive

Type tag for the predictive variant of [`StudentTUniformScoring`](@ref):
`StudentTUniformScoring{Predictive}` incorporates the model's estimation
uncertainty Σ̃_θ (`solver_covariance`, of a minimal-sample model or of a refit
on its consensus set) into the per-point scores; `StudentTUniformScoring{Nothing}`
treats the model as exact (Σ_θ = 0). An option, not a default: measured
against model-certain scoring it ties or loses on pooled ΔAUC@10° and costs
13–565× per hypothesis (VisualGeometryToolkitEstimation's docs/src/guide/scoring.md).
"""
struct Predictive end

"""
    StudentTUniformScoring{P, DG, L, O} <: AbstractScoring

The scale-free marginal score of the Student-t inlier + uniform outlier model
(Section 3, Eq. 12 of "RANSAC Scoring Done Right") — **spec only**: the
immutable configuration of the score. Its scratch lives on its
[`ScoringWorkspace`](@ref), the scoring contract's `scoring_workspace`.

- `P`: `Nothing` (model-certain) or [`Predictive`](@ref) (model-uncertain);
- `DG::Int`: the codimension `d_g` of the model's constraint;
- `L`, `O`: the label prior and the outlier model (type parameters, so the
  sweep dispatches on them statically).

The score marginalizes σ² under the scale prior:

    S(θ, I) = log Γ(ν/2 + α₀) − (ν/2)·log 2π − (ν/2 + α₀)·log(RSS_I/2 + β₀) − n_o·d_g·log(2h) [+ label prior]

with `ν = d_g·(n_I − m)`, `RSS_I = Σᵢ∈I rᵢ²`, `n_o = N − n_I` and `h` the
outlier half-width.

Build it with

    StudentTUniformScoring(problem | model_dof, outlier; scale_prior, label_prior[, codimension])
    StudentTUniformScoring{Predictive}(problem | model_dof, outlier; scale_prior, label_prior[, codimension])

`scale_prior` (an `IGPrior`: `JeffreysPrior()` or `IGPrior(α₀, β₀)`) and
`label_prior` ([`FlatLabelPrior`](@ref), [`ProfiledLabelPrior`](@ref),
[`BetaLabelPrior`](@ref)) are REQUIRED: the stack's default is its
configuration's, a scorer table read by VisualGeometryToolkitEstimation's
`scoring_from_toml` (its defaults in Estimation's configs/default.toml). The
`model_dof` form takes `codimension` (1 when absent: a scalar constraint).

The outlier component `outlier` is a supplied half-width (a number, or
[`FixedHalfwidth`](@ref): the profile-marginal score) or an outlier model that
estimates the support (an [`AbstractOutlierModel`](@ref) of another package:
RANSACScoringDoneRight's extreme-value and generalized Pareto models). Those
have no length parameter and take model-certain scalar residuals (`d_g = 1`),
since the scorer receives one distance per point.

Accessors: [`minimal_sample_size`](@ref) (`m`), [`codimension`](@ref),
[`scale_prior`](@ref), [`label_prior`](@ref), [`outlier_model`](@ref),
[`outlier_halfwidth`](@ref) (a supplied width only).
"""
struct StudentTUniformScoring{P, DG, L<:AbstractLabelPrior, O<:AbstractOutlierModel} <: AbstractScoring
    model_dof::Int
    label_prior::L
    scale_prior::IGPrior
    outlier_model::O
    # TYPED field constructor, replacing the default one that `convert`s every
    # argument: a stale positional call would otherwise convert silently.
    StudentTUniformScoring{P, DG}(model_dof::Int, label_prior::L, scale_prior::IGPrior,
                                  outlier_model::O) where {P, DG, L<:AbstractLabelPrior, O<:AbstractOutlierModel} =
        new{P, DG, L, O}(model_dof, label_prior, scale_prior, outlier_model)
end

codimension(::StudentTUniformScoring{P, DG}) where {P, DG} = DG

"""
    outlier_halfwidth(scoring::StudentTUniformScoring{P, DG, L, FixedHalfwidth}) -> Float64

The supplied half-width of the uniform outlier support per axis. The sweep
charges `log(2·outlier_halfwidth)` per point and axis. A scorer whose outlier
model estimates the support has none: the scoring contract's `nothing`.
"""
outlier_halfwidth(s::StudentTUniformScoring{P, DG, L, FixedHalfwidth}) where {P, DG, L} =
    s.outlier_model.outlier_halfwidth

"""
    outlier_model(scoring::StudentTUniformScoring) -> AbstractOutlierModel

The scorer's outlier component.
"""
outlier_model(s::StudentTUniformScoring) = s.outlier_model

"""
    label_prior(scoring::StudentTUniformScoring) -> AbstractLabelPrior

The scorer's prior of the labels.
"""
label_prior(s::StudentTUniformScoring) = s.label_prior

# The model DOF `m` of `ν = (k − m)·d_g` — the problem's minimal sample size,
# which is what every problem-aware constructor fills the field from.
minimal_sample_size(s::StudentTUniformScoring) = s.model_dof

scale_prior(s::StudentTUniformScoring) = s.scale_prior

# -----------------------------------------------------------------------------
# Constructors: the priors are required keywords
# -----------------------------------------------------------------------------

# A number is the supplied half-width; an outlier model that estimates the
# support needs model-certain scalar residuals (the sweep sees one distance per
# point, so the coordinates the tail is fitted to exist only at d_g = 1, and the
# predictive penalties' parametric sweep has no tail fit).
_outlier_model(a::Real) = FixedHalfwidth(a)
_outlier_model(m::AbstractOutlierModel) = m
_check_outlier_model(::Type, ::Val, ::FixedHalfwidth) = nothing
function _check_outlier_model(P::Type, ::Val{DG}, m::AbstractOutlierModel) where {DG}
    DG == 1 || throw(ArgumentError(
        "$(nameof(typeof(m))) fits the residual coordinates, and a scorer receives one " *
        "distance per point: it supports codimension 1, got $DG"))
    P === Nothing || throw(ArgumentError(
        "$(nameof(typeof(m))) is model-certain only (the predictive parametric sweep has no tail fit)"))
    return nothing
end

function StudentTUniformScoring{P, DG}(model_dof::Int, outlier::Union{Real, AbstractOutlierModel};
                                       scale_prior::IGPrior, label_prior::AbstractLabelPrior) where {P, DG}
    m = _outlier_model(outlier)
    _check_outlier_model(P, Val(DG), m)
    return StudentTUniformScoring{P, DG}(model_dof, label_prior, scale_prior, m)
end

# Bare model DOF: the codimension is a keyword (1 when absent: a scalar constraint).
StudentTUniformScoring{P}(model_dof::Int, outlier; codimension::Int = 1, kw...) where {P} =
    StudentTUniformScoring{P, codimension}(model_dof, outlier; kw...)
StudentTUniformScoring(model_dof::Int, outlier; codimension::Int = 1, kw...) =
    StudentTUniformScoring{Nothing, codimension}(model_dof, outlier; kw...)

# Problem-aware: m and d_g from the problem (an instance or its type).
StudentTUniformScoring(problem::Union{AbstractConstraintProblem, Type{<:AbstractConstraintProblem}},
                       outlier; kw...) =
    StudentTUniformScoring{Nothing, codimension(problem)}(minimal_sample_size(problem), outlier; kw...)
StudentTUniformScoring{P}(problem::Union{AbstractConstraintProblem, Type{<:AbstractConstraintProblem}},
                          outlier; kw...) where {P} =
    (_check_provider(P, problem isa Type ? problem : typeof(problem));
     StudentTUniformScoring{P, codimension(problem)}(minimal_sample_size(problem), outlier; kw...))

_check_provider(::Type, ::Type) = nothing
_check_provider(::Type{Predictive}, ::Type{P}) where {P} =
    provides_solver_covariance(P) ? nothing : throw(ArgumentError(
    "predictive scoring needs a solver covariance, and $(nameof(P)) provides none: define " *
    "`VisualGeometryToolkitCore.solver_covariance(::$(nameof(P)), model, sample_indices)` and declare " *
    "`VisualGeometryToolkitCore.provides_solver_covariance(::Type{<:$(nameof(P))}) = true`, or score $(nameof(P)) model-certain"))

# =============================================================================
# alloc_scoring_workspace — build ScoringWorkspace from problem + scoring
# =============================================================================

"""
    alloc_scoring_workspace(problem, scoring::StudentTUniformScoring) -> ScoringWorkspace
    alloc_scoring_workspace(n::Int, dg::Int, scoring::StudentTUniformScoring) -> ScoringWorkspace

The scratch of the Algorithm 1 sweep for `n` points (`data_size(problem)`) of
codimension `dg` (`codimension(problem)`): the log Γ table at the scorer's scale
prior, the label prior's table, the estimated-support buffers when the outlier
model estimates the support. The scoring contract's
`scoring_workspace(scoring, n, dg)` is the second form.
"""
alloc_scoring_workspace(problem::AbstractConstraintProblem, scoring::StudentTUniformScoring) =
    alloc_scoring_workspace(data_size(problem), codimension(problem), scoring)

function alloc_scoring_workspace(n::Int, dg::Int, scoring::StudentTUniformScoring)
    lg = log_gamma_table(n, dg; alpha0 = scoring.scale_prior.alpha0)
    pt = label_prior_table(n, scoring.label_prior)
    tail = estimates_outlier_support(scoring.outlier_model) ? Vector{Float64}(undef, 3n + 2) : Float64[]
    ScoringWorkspace(
        Vector{Int}(undef, n),
        Vector{Int}(undef, n),
        lg,
        zeros(Float64, n),
        Vector{Float64}(undef, n),
        pt,
        scoring.scale_prior;
        label_prior = scoring.label_prior,
        tail,
    )
end

"""
    score_sq_residuals!(sws, q, scoring, problem, model, n) -> (S, k)

Score a model by computing per-point squared whitened residuals and sweeping
for the optimal partition (Algorithm 1, Eq. 12). Fills `q[1:n]` with squared
whitened residuals, `q[n+1:end]` with `typemax`, and overwrites the `sws` sweep
buffers. Returns `(S, k)`: sweep score and optimal inlier count.
"""
function score_sq_residuals!(sws::ScoringWorkspace, q::AbstractVector{T}, s::StudentTUniformScoring,
                             problem::AbstractConstraintProblem, model,
                             n::Int) where T
    @inbounds for i in 1:n
        # squared norm of the whitened residual itself, not of its norm — no
        # sqrt round trip for a vector constraint
        q[i] = sq_norm(whitened_residual(problem, model, i))
    end
    @inbounds for i in n+1:length(q)
        q[i] = typemax(T)
    end
    _zero_penalties!(sws, s)
    return sweep!(sws, s, q, n)
end

"""
    score_residuals!(sws, q, scoring, residuals, n; sample_idx = nothing) -> (S, k)

Score from precomputed residuals (model-certain, isotropic noise): squares
`residuals[1:n]` into `q`, pads `q[n+1:end]` with `typemax`, zeros the `sws`
penalties, and sweeps for the optimal partition (overwriting the `sws` sweep
buffers). The `sample_idx` points get `q = 0`. Returns `(S, k)`: sweep score
and optimal inlier count. The residuals are in the caller's unit, the one
`scoring`'s outlier half-width is in.

```jldoctest
julia> using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, FlatLabelPrior,
           alloc_scoring_workspace, score_residuals!

julia> r = [0.010, -0.020, 0.015, 0.005, -0.010, 0.012, -0.008, 0.40, -0.45, 0.30];

julia> scoring = StudentTUniformScoring(2, 0.5; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior());

julia> sws = alloc_scoring_workspace(length(r), 1, scoring);

julia> S, k = score_residuals!(sws, zeros(length(r)), scoring, r, length(r));

julia> k   # the seven small residuals are the inliers
7
```
"""
function score_residuals!(sws::ScoringWorkspace, q::AbstractVector{T}, s::StudentTUniformScoring,
                          residuals::AbstractVector{<:Real},
                          n::Int;
                          sample_idx::Union{Nothing,AbstractVector{Int}}=nothing) where T
    square_distances!(q, residuals, n)
    @inbounds for i in n+1:length(q)
        q[i] = typemax(T)
    end
    zero_sample!(q, sample_idx)     # bounds-checked: an invalid index throws
    _zero_penalties!(sws, s)
    return sweep!(sws, s, q, n)
end

"""
    score_sq_residuals!(sws, q, scoring::StudentTUniformScoring{Predictive}, problem, model, sample_idx, n) -> (S, k)

Predictive variant: the model-uncertain scores `qᵢ` from the solver
covariance `solver_covariance(problem, model, sample_idx)`, then the
sort-and-sweep (Algorithm 1, Eq. 20). A DEGENERATE sample — the provider
returns `nothing` — is no hypothesis: `(-Inf, 0)`, with `q` and `sws`
untouched; it is never scored model-certain.
"""
function score_sq_residuals!(sws::ScoringWorkspace, q::AbstractVector,
                             s::StudentTUniformScoring{Predictive},
                             problem::AbstractConstraintProblem, model,
                             sample_idx::AbstractVector{Int}, n::Int)
    Σ̃_θ = solver_covariance(problem, model, sample_idx)
    isnothing(Σ̃_θ) && return (-Inf, 0)
    return score_sq_residuals!(sws, q, s, problem, model, Σ̃_θ, sample_idx, n)
end

"""
    score_sq_residuals!(sws, q, scoring::StudentTUniformScoring{Predictive}, problem, model, Σ̃_θ,
                        sample_idx, n; residuals = nothing) -> (S, k)

Predictive scoring at a GIVEN unit-scale model covariance `Σ̃_θ` (multiplicative
convention, `Σ_θ = s²·Σ̃_θ`): fills `q[i]` with the predictive squared residual
`rᵢᵀ Ṽᵢ⁻¹ rᵢ` and the `sws` penalties with `log|Ṽᵢ|`, then sort-and-sweeps
(Algorithm 1, Eq. 20). The minimal sample's points (`sample_idx`, or
`nothing`) get zero `q` and penalty. When `residuals` is supplied it receives
each point's Mahalanobis distance, the norm of its `whitened_residual` from
the same constraint evaluation (`≥ 0`).
"""
function score_sq_residuals!(sws::ScoringWorkspace, q::AbstractVector{T},
                             s::StudentTUniformScoring{Predictive},
                             problem::AbstractConstraintProblem, model, Σ̃_θ,
                             sample_idx::Union{Nothing,AbstractVector{Int}}, n::Int;
                             residuals::Union{Nothing,AbstractVector{<:Real}} = nothing) where T
    @inbounds for i in 1:n
        rᵢ, ∂θgᵢ_w = whitened_constraint_rows(problem, model, i)
        residuals === nothing || (residuals[i] = norm(rᵢ))   # the distance, never the signed residual
        q[i], sws.penalties[i] = _predictive_score_penalty(rᵢ, ∂θgᵢ_w, Σ̃_θ)
        # sort_key = q (unit-scale Mahalanobis): with heterogeneous penalties
        # sweep! takes the exact parametric path, whose key t_i(μ) = g_i − μq_i
        # carries the covariance penalty.
        sws.sort_key[i] = q[i]
    end
    # Sample points satisfy constraints exactly — their prediction variance
    # is zero (model is determined by them), so their penalty is spurious.
    zero_sample!(q, sample_idx)
    zero_sample!(sws.penalties, sample_idx)
    @inbounds for i in n+1:length(q)
        q[i] = typemax(T)
        sws.sort_key[i] = typemax(T)
    end
    return sweep!(sws, s, q, n)
end

_maybe_prior(sws::ScoringWorkspace) = isempty(sws.prior_table) ? nothing : sws.prior_table

"""
    sweep!(sws::ScoringWorkspace, scoring::StudentTUniformScoring, scores, n) -> (S*, k*)

The scorer's sort-and-sweep of the per-point scores `scores[1:n]` on its
workspace: the supplied half-width's prefix sweep, the predictive score's
(with the workspace's `penalties`), or the estimated-support sweep of an
outlier model that estimates the support. Records the half-width the selected
partition was scored with on the workspace.
"""
sweep!(sws::ScoringWorkspace, s::StudentTUniformScoring{Nothing, DG},
       scores::Vector{T}, n::Int) where {DG, T} =
    _outlier_sweep!(s.outlier_model, sws, s, scores, n)
# A supplied width: the profile-marginal sweep.
function _outlier_sweep!(m::FixedHalfwidth, sws::ScoringWorkspace, s::StudentTUniformScoring{Nothing, DG},
                         scores::Vector{T}, n::Int) where {DG, T}
    sws.outlier_halfwidth = m.outlier_halfwidth
    return sweep!(sws.perm, sws.lg_table, s.model_dof, log(2 * m.outlier_halfwidth), DG, scores, n;
                  scratch=sws.scratch, prior_table=_maybe_prior(sws), scale_prior=s.scale_prior,
                  radix=sws.radix, launch=sws.launch)
end
sweep!(sws::ScoringWorkspace, s::StudentTUniformScoring{Predictive, DG},
       scores::Vector{T}, n::Int) where {DG, T} =
    (sws.outlier_halfwidth = outlier_halfwidth(s);
     sweep!(sws.perm, sws.lg_table, s.model_dof, log(2 * outlier_halfwidth(s)), DG, scores, n;
           scratch=sws.scratch, penalties=sws.penalties,
           prior_table=_maybe_prior(sws), scale_prior=s.scale_prior,
           radix=sws.radix, launch=sws.launch))

# =============================================================================
# The estimated-support sweep: an outlier model that estimates the support
# (another package's: `outlier_cost`, `scored_outlier_halfwidth`, optionally
# `outlier_context!`). The sort is the sweep's own (`_sort_perm!`: the Θ(n)
# radix on the exact-size path), and the per-partition score is
# `_table_partition_score_cost`, the one formula every sweep shares; the label
# prior is the workspace's table. Only the outlier term differs per model.
# Allocation-free: every buffer is the workspace's.
# =============================================================================

# The sorted distances and their suffix sums in the workspace's tail buffers
# (z[1:n], T1[1:n+1], T2[1:n+1]), and the sorted squared distances in its
# sort_key buffer, gathered together after the sort. A non-finite distance
# (the sort puts it last) is clamped to the largest finite one for the tail
# fit; its squared distance stays non-finite, so no finite-RSS partition holds
# it — it is an outlier, as in the supplied-width sweep.
function _tail_buffers!(sws::ScoringWorkspace, scores, n::Int)
    t = sws.tail
    length(t) >= 3n + 2 || throw(ArgumentError(
        "the workspace's tail buffers cover $(div(length(t) - 2, 3)) points, not $n; build it from this scoring"))
    z = view(t, 1:n)
    S1 = view(t, n+1:2n+1)
    S2 = view(t, 2n+2:3n+2)
    q = view(sws.sort_key, 1:n)
    zfinite = 0.0
    @inbounds for i in 1:n
        qi = Float64(scores[sws.perm[i]])
        q[i] = qi
        zi = sqrt(qi)                      # the distance (d_g = 1: the absolute residual)
        isfinite(zi) ? (zfinite = zi) : (zi = zfinite)
        z[i] = zi
    end
    suffix_sums!(S1, S2, z)
    return z, S1, S2, q
end

# The sweep over the prefixes of the sorted squared distances `q`, given the
# complement's cost of each (an empty complement costs nothing).
@inline function _tail_sweep!(cost, sws::ScoringWorkspace, s::StudentTUniformScoring, q, n::Int)
    m = s.model_dof
    prior_table = _maybe_prior(sws)
    RSS = 0.0
    best_S, best_k = -Inf, 0
    @inbounds for k in 1:n
        RSS += q[k]
        k <= m && continue
        S = _table_partition_score_cost(s.scale_prior, sws.lg_table, RSS, k, m, 1.0, k == n ? 0.0 : cost(k))
        prior_table === nothing || (S += prior_table[k])
        if S > best_S
            best_S, best_k = S, k
        end
    end
    return best_S, best_k
end

# (Any codimension in the signature, so the supplied half-width's method is the more specific one; the constructor
# admits an estimating model at d_g = 1 only.)
function _outlier_sweep!(model::AbstractOutlierModel, sws::ScoringWorkspace,
                         s::StudentTUniformScoring{Nothing, DG}, scores::Vector{T}, n::Int) where {DG, T}
    n == 0 && return (-Inf, 0)
    _sort_perm!(sws.perm, scores, n; scratch = sws.scratch, radix = sws.radix)
    z, S1, S2, q = _tail_buffers!(sws, scores, n)
    ctx = outlier_context!(model, z, S1, S2, n)
    S, k = _tail_sweep!(k -> outlier_cost(model, ctx, z, S1, S2, k, n), sws, s, q, n)
    sws.outlier_halfwidth = k == n ? z[n] : scored_outlier_halfwidth(model, ctx, z, S1, S2, k, n)
    return S, k
end

# =============================================================================
# The predictive per-point score and penalty
# =============================================================================

# Model-uncertain score and penalty for a scalar constraint (dg = 1), in the
# MULTIPLICATIVE predictive-covariance convention of the generalized sweep
# lemma: Ṽᵢ = s²·Vᵢ with the unit-scale shape Vᵢ = 1 + (∂θgᵢ_w)ᵀ Σ̃_unit (∂θgᵢ_w),
# Σ̃_unit the propagation of UNIT measurement noise through the fit. The returned
# qᵢ = rᵢ²/Vᵢ and ℓᵢ = log Vᵢ are s-free; the s-dependence factors into the
# common scale marginal exactly as the lemma's separation requires.
@inline function _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T},
                                            Σ̃_unit) where {n_θ,T}
    Vᵢ = one(T) + dot(∂θgᵢ_w, Σ̃_unit * ∂θgᵢ_w)
    Vᵢ > eps(T) || return (typemax(T), zero(T))
    qᵢ = rᵢ^2 / Vᵢ
    return (qᵢ, log(Vᵢ))
end

# The vector-constraint (dg ≥ 2) form: Vᵢ = I + (∂θgᵢ_w) Σ̃_unit (∂θgᵢ_w)ᵀ,
# qᵢ = rᵢᵀ Vᵢ⁻¹ rᵢ and ℓᵢ = log|Vᵢ|. `StaticVector`/`StaticMatrix`, not narrowly
# `SVector`: the whitened homography residual arrives as a `GeometryBasics.Vec{2}`.
@inline function _predictive_score_penalty(rᵢ::StaticVector{dg,T}, ∂θgᵢ_w::StaticMatrix{dg,n_θ,T},
                                            Σ̃_unit) where {dg,n_θ,T}
    Vᵢ = SMatrix{dg,dg,T}(I) + ∂θgᵢ_w * Σ̃_unit * ∂θgᵢ_w'
    det_Vᵢ = det(Vᵢ)
    det_Vᵢ > eps(T) || return (typemax(T), zero(T))
    qᵢ = dot(rᵢ, Vᵢ \ SVector{dg,T}(rᵢ))
    return (qᵢ, log(det_Vᵢ))
end

_zero_penalties!(::ScoringWorkspace, ::StudentTUniformScoring{Nothing}) = nothing
_zero_penalties!(sws::ScoringWorkspace, ::StudentTUniformScoring{Predictive}) = fill!(sws.penalties, 0.0)

# =============================================================================
# The scoring contract (robust/scoring_contract.jl)
# =============================================================================
#
# The sort-and-sweep IS the contract's fused path: one `score!` produces the
# score, k* (cached on the workspace for `inlier_count`) and the sorted order
# (`consensus_order`), whose first k* entries are the consensus set.

scoring_workspace(s::StudentTUniformScoring, n::Int, d_g::Int) = alloc_scoring_workspace(n, d_g, s)

# Model-certain: the residuals are given; `q` receives rᵢ² (the sample's points
# zeroed). Under predictive scoring a PLAIN model is scored this way too (a
# seed scored model-certain to find its consensus set): the penalties are zeroed.
function score!(sws::ScoringWorkspace, s::StudentTUniformScoring, q::AbstractVector,
                residuals::AbstractVector, n::Int; sample = nothing, bound::Float64 = -Inf)
    S, k = score_residuals!(sws, q, s, residuals, n; sample_idx = sample)
    sws.k = k
    return S
end

# Predictive: the model carries its unit-scale covariance (`Uncertain`); the
# residuals receive each point's whitened residual from the same constraint
# evaluation.
function score!(sws::ScoringWorkspace, s::StudentTUniformScoring{Predictive}, q::AbstractVector,
                residuals::AbstractVector, problem, θ::Uncertain, n::Int;
                sample = nothing, bound::Float64 = -Inf)
    S, k = score_sq_residuals!(sws, q, s, problem, geom(θ), Statistics.cov(θ),
                               sample, n; residuals)
    sws.k = k
    return S
end

inlier_count(::StudentTUniformScoring, sws::ScoringWorkspace) = sws.k
consensus_order(::StudentTUniformScoring, sws::ScoringWorkspace) = sws.perm
is_predictive(::StudentTUniformScoring{Predictive}) = true

"""
    scored_model(scoring::StudentTUniformScoring{Predictive}, problem, θ, sample) -> Uncertain | nothing

The model with the unit-scale covariance of its fit on the points `sample`
(`solver_covariance`); `nothing` for a degenerate sample — no hypothesis,
skipped like a failed solve, never scored model-certain.
"""
function scored_model(::StudentTUniformScoring{Predictive}, problem, θ, sample)
    Σ̃_θ = solver_covariance(problem, θ, sample)
    isnothing(Σ̃_θ) && return nothing
    return Uncertain(θ, CovMat(Σ̃_θ))
end

# The decision point of the last score: the Gaussian-uniform MAP crossover of the
# consensus set's Student-t scale (at the local optimizer's dof, m + 1) against
# the uniform outlier box.
function inlier_threshold(s::StudentTUniformScoring{P, DG}, buf::ScoreBuffers) where {P, DG}
    sws = buf.scoring_ws
    k = sws.k
    s2 = consensus_s2(s, residuals(buf), sws.perm, k; model_dof = s.model_dof + 1, d_g = DG)
    return decision_threshold(GauInlier(s2; d_g = DG), sws.outlier_halfwidth)
end
has_inlier_threshold(::Type{<:StudentTUniformScoring}) = true

# A workspace's tables are built for one scale prior, one label prior and one
# kind of outlier model (the tail buffers or none).
function check_workspace(s::StudentTUniformScoring, sws::ScoringWorkspace)
    sws.scale_prior == s.scale_prior && sws.label_prior == s.label_prior &&
        isempty(sws.tail) == !estimates_outlier_support(s.outlier_model) && return nothing
    throw(ArgumentError(
        "the workspace's scoring tables were built for scale prior $(sws.scale_prior), " *
        "label prior $(sws.label_prior) and $(isempty(sws.tail) ? "a supplied" : "an estimated") " *
        "outlier half-width, but the scoring has $(s.scale_prior), $(s.label_prior) and " *
        "$(nameof(typeof(s.outlier_model))); build the workspace from this scoring"))
end

"""
    with_width(scoring::StudentTUniformScoring, width::OutlierHalfwidth) -> StudentTUniformScoring

The same scorer at the supplied outlier half-width `width` (every other field
kept); a scorer whose outlier model estimates the support takes none (the
contract's `ArgumentError`).
"""
with_width(s::StudentTUniformScoring{P, DG, L, FixedHalfwidth}, w::OutlierHalfwidth{Float64}) where {P, DG, L} =
    StudentTUniformScoring{P, DG}(s.model_dof, s.label_prior, s.scale_prior, FixedHalfwidth(w.value))
# Only a supplied width has an axis; a scorer whose outlier model estimates the
# support takes no width (no `width_axis`, so a width grid refuses it up front).
width_axis(::StudentTUniformScoring{P, DG, L, FixedHalfwidth}) where {P, DG, L} = OutlierHalfwidth()

"""
    SCORING_TAGS

The [`TagRegistry`](@ref) of the scoring families: `"ransac"`
([`RansacScoring`](@ref)), `"msac"` ([`MsacScoring`](@ref)) and
`"student_t_uniform"` ([`StudentTUniformScoring`](@ref)), `snake_case` of the
type's name with the family's suffix `Scoring` stripped. A package that adds
families (RANSACScoringDoneRight's MLESAC, MAGSAC++, …) declares its own
registry of the same family and suffix, `TagRegistry(:scoring, "Scoring", …;
retired)`, and its loaders read `merge(VgtkCore.SCORING_TAGS, its_tags)`.
VisualGeometryToolkitEstimation's `scoring_from_toml` reads a scorer table through it.
"""
const SCORING_TAGS = TagRegistry(:scoring, "Scoring", "ransac" => RansacScoring, "msac" => MsacScoring,
                                 "student_t_uniform" => StudentTUniformScoring)

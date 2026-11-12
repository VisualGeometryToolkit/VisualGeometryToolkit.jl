# =============================================================================
# ManifoldLiftProvider — the chart-rows provider of every chart problem
# =============================================================================
#
# VisualGeometryToolkitCore's chart-rows contract (robust/rows.jl) for the chart
# problems whose codimension the lifts support (H, PnP, F, E, the radial
# homography and the radial camera — the `MinimalSolveProblem` family): ONE
# provider type. Its state IS the problem's chart (`_chart`, the refit's own
# seeding), its rows are the refit's rows (`ChartLm`, interface.jl) evaluated
# at the chart's origin with unit weights, and a step re-anchors by the chart's
# own `recenter`. So a lift (LocalOptimizationDoneRight's HQ and GNC drivers)
# and the chart-LM refit cannot drift apart — they share the chart, the rows
# and the model read-out (`_model_at`).
#
# Provider protocol (generic; one line per exception below):
#   rows_provider(problem)      -> the provider (rows from the one reuse policy)
#   lift_init(prov, model)      -> the chart at the model          (`_chart`)
#   lift_refresh!(prov, ch)     rows (and per-point norms for codim 2) at ch
#   lift_jacobian!(prov, ch)    the rows' Jacobian at ch
#   lift_retract(prov, ch, Δθ)  -> the chart re-anchored at the step (`recenter`)
#   lift_model(prov, ch)        -> the RANSAC model                 (`_model_at`)
#   lift_valid(prov)            participation mask (default `nothing`)
#   lift_rows / lift_residuals / lift_size
# The exceptions: PnP carries its pixel block and participation mask (`_lift_aux`,
# `_lift_state`, `lift_valid`), F re-anchors by the SVD (`lift_retract`), and a
# plain-homography hypothesis seeds the radial homography's k (`lift_init`).
# =============================================================================

"""
    ManifoldLiftProvider{P, D, C, X}

The chart-rows provider (VisualGeometryToolkitCore's `rows_provider`) of a problem
`P` on a `D`-dof chart with codimension-`C` residuals: the problem, its rows
(VGC `RowsWorkspace{D, C}`, one per (task, size)), the point count, and the
problem's frozen per-run data `aux` (`nothing` unless the problem has some —
PnP's pixel block).
"""
struct ManifoldLiftProvider{P, D, C, X}
    problem::P
    lws::RowsWorkspace{D, C}
    n::Int
    aux::X
end

_lift_aux(::Any) = nothing

# The provider's (D, C): the chart's dof and the codimension — the problem's two
# facts, not restated for the lift.
_lift_dims(problem) = (Val(chart_dof(problem)), Val(codimension(problem)))

# The problems with a lift — the chart problems whose codimension the lift core
# supports (1 and 2): exactly the `MinimalSolveProblem` family, whose every
# instantiation carries a minimal chart of its model and a codimension-1 or -2
# constraint. The lifts' `locally_optimize!` methods (LocalOptimizationDoneRight)
# are typed on it, so the contract's refusal keeps guarding every other problem.
"""
    LiftableProblem

The problems that provide VisualGeometryToolkitCore's chart-rows contract to a
lifted local optimizer: the [`MinimalSolveProblem`](@ref) family, whose every
instance has a minimal chart of its model and a codimension-1 or -2
constraint. A lifted strategy's `locally_optimize!` methods dispatch on it,
so the contract still refuses every other problem.
"""
const LiftableProblem = MinimalSolveProblem

function rows_provider(problem::LiftableProblem)
    n = data_size(problem)
    D, C = _lift_dims(problem)
    return ManifoldLiftProvider(problem, rows_workspace(D, C, n), n, _lift_aux(problem))
end

lift_rows(p::ManifoldLiftProvider) = p.lws
lift_size(p::ManifoldLiftProvider) = p.n
# The residual currency: per-point norms for codim 2, signed scalars for codim 1.
lift_residuals(p::ManifoldLiftProvider{<:Any, <:Any, 1}) = p.lws.r
lift_residuals(p::ManifoldLiftProvider{<:Any, <:Any, 2}) = p.lws.rn
_lift_norms!(::ManifoldLiftProvider{<:Any, <:Any, 1}) = nothing
_lift_norms!(p::ManifoldLiftProvider{<:Any, <:Any, C}) where {C} = (point_norms!(p.lws.rn, p.lws.r, p.n, C); nothing)
lift_valid(::ManifoldLiftProvider) = nothing

# The generic provider: the chart problem's own chart, rows and read-out.
lift_init(p::ManifoldLiftProvider, model) = _chart(p.problem, _as_model(p.problem, model))
_lift_state(p::ManifoldLiftProvider, ch) = _lm_state(p.problem, ch, p.lws.onev)
function _refresh_rows!(p::ManifoldLiftProvider, ch)
    residual_rows!(p.lws.r, _lift_state(p, ch), VgtkCore.zero_tangent(ch))
    return _lift_norms!(p)
end
lift_refresh!(p::ManifoldLiftProvider, ch) = _refresh_rows!(p, ch)
function lift_jacobian!(p::ManifoldLiftProvider, ch)
    residual_jacobian_rows!(p.lws.J, _lift_state(p, ch), VgtkCore.zero_tangent(ch))
    return nothing
end
@inline lift_retract(::ManifoldLiftProvider, ch, Δθ) = recenter(ch, Δθ)
lift_model(p::ManifoldLiftProvider, ch) = _model_at(p.problem, ch)

# PnP — the frozen pixel block (aux) and the per-iteration participation mask,
# written by the rows (a point the lens cannot place: out-of-field rows, masked
# out of the θ/z systems — the convention in problems/p3p.jl).
const _PnPProvider = ManifoldLiftProvider{<:AbsolutePoseProblem}
_lift_aux(problem::AbsolutePoseProblem) = _lift_pixel_block(lens(problem.context.model), problem)  # frozen for the run
_lift_state(p::_PnPProvider, ch) = ChartLm(ch, p.problem, p.lws.onev, _PnPAux(p.aux, p.lws.valid))
lift_valid(p::_PnPProvider) = p.lws.valid

# F — a step re-anchors by the chart's SVD constructor at the stepped F
# (`minimal_manifold`): ONE SVD per step. The SVD re-anchor, not `recenter`,
# keeps σ canonical (s₂/s₁ ∈ [0, 1]): `recenter` measured a different
# trajectory (up to 1.3 in normalized F on narrow-FOV scenes) with no gain in
# inlier RMS.
@inline lift_retract(::ManifoldLiftProvider{<:FundMatProblem}, ch, Δθ) = minimal_manifold(retract(ch, Δθ))

# RadialHomography — the lift has no LO weights, so a plain-homography
# hypothesis (no distortion term) seeds k on the correspondences its H already
# explains: the TLS weights of the plain-H reprojection norms at
# `efficiency_constant · σ̂`, σ̂ their noise scale (Core's stack-internal `_mad_scale_norms`, codimension
# 2) and the multiple configs/default.toml's [pipelines.radial_homography.lift_seed]
# (3·√quantile(Chisq(2), 0.5): three medians of the norms); the refit's own
# `_radial_seed_chart` otherwise. The scale is floored at floatmin, so on an
# exact plain H the seed is the exactly explained correspondences.
const _LIFT_SEED_EFFICIENCY_CONSTANT = _dflt("pipelines.radial_homography.lift_seed", "efficiency_constant")::Float64
function lift_init(p::ManifoldLiftProvider{<:RadialHomographyProblem}, m::VgtkCore.RadialHomography)
    iszero(division_coeff(m)) || return _radial_seed_chart(p.problem, m, nothing)
    H = unit_normalize(SMatrix{3, 3, Float64, 9}(homography(m)))
    lift_refresh!(p, _radial_chart(H, 0.0, center(m)))
    rn = view(p.lws.rn, 1:p.n)
    σ = VgtkCore._mad_scale_norms(; norms = rn, codimension = codimension(p.problem))   # the lift workspace holds only the norms
    loss = deploy(TLSShape(), max(σ^2, floatmin(Float64)); efficiency_constant = _LIFT_SEED_EFFICIENCY_CONSTANT)
    c² = sq_loss_width(loss)
    return _radial_seed_chart(p.problem, m, [weight(TLSShape(), x^2, c²) for x in rn])
end

# =============================================================================
# Triggs — the robustified chart LM of every chart problem (scoring_glue.jl)
# =============================================================================

"""
    locally_optimize!(L, lo::Triggs, problem::LiftableProblem, model, cs, buf, scoring) -> FitResult

The robustified chart LM: the loss the width mode deploys (`lo_loss`:
`efficiency_constant·ŝ` at θ's Student-t scale), frozen, as VisualGeometryToolkitCore's
`loss_robustifier`, and ONE `lmsolve_chart!` from `model` over every datum at
unit weight, on the chart-LM refit's own chart and state (`_chart`,
`_refit_state`: the rows `refine!` descends) in its workspace (`L.refine`, the
one reuse policy) at its LM settings (`[chart_lm]`). NLLSsolver's Triggs
correction carries the loss; nothing is reweighted here and no inlier set is
formed. The model comes back through the chart's `recenter` and `_model_at`,
as from `refine!`.
"""
locally_optimize!(L::LoBuffers, lo::Triggs, problem::LiftableProblem, model, cs::ConsensusSet, buf::ScoreBuffers, scoring) =
    _triggs_lm(L.refine, strategy_loss(lo)(), problem, model, lo_loss(lo, cs, scoring, buf))

# The rows the robustified LM robustifies, and a refine's plain least squares
# minimises (`_refine_run(::Nothing, …)`, fitting/_refine.jl): the refit's
# (`_refit_state`), whose per-datum norm must be the residual's own unit — the
# unit the loss width is in (the problem's `mahalanobis_distance`). A problem
# whose refit rows are in another unit (PnP's pinhole rows, calibrated) states
# its pixel rows here.
_triggs_state(problem, ch, sqrtw) = _refit_state(problem, ch, sqrtw)

# THE robustified chart LM — the RANSAC local optimization's (above) and every
# refine_*'s Triggs (fitting/_refine.jl): `shape` at the deployed `loss`, one
# `lmsolve_chart!` from `model` over every datum at unit weight, on the refit's
# chart and state, in the chart-LM workspace `refine_ws` (the one reuse policy).
function _triggs_lm(refine_ws, shape::VgtkCore.AbstractLossShape, problem, model, loss::DeployedLoss)
    ch = _chart(problem, _as_model(problem, model))
    rows, dof = _lm_size(problem)
    ews = _lm_workspace(refine_ws, rows, dof)
    fill!(sqrt_weights(ews), 1.0)
    lm = _triggs_state(problem, ch, sqrt_weights(ews))
    robustifier = loss_robustifier(shape, sqrt(sq_loss_width(loss)))
    x, _, _, n_iter, (code, rule) = lmsolve_chart!(ews, lm, robustifier, Val(codimension(problem));
        maxit = _CHART_LM_MAX_ITER, stationarity_tol = _CHART_LM_STATIONARITY_TOL, ftol = _CHART_LM_FTOL)
    any(isnan, x) && return lo_estimate(model, ReturnCode.Failure, NonFinite)
    return lo_estimate(_model_at(problem, recenter(ch, x)), code, rule; iterations = n_iter)
end

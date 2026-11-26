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
# and the model read-out (`chart_model`).
#
# Provider protocol (generic; one line per exception below):
#   rows_provider(problem)      -> the provider (rows from the one reuse policy)
#   lift_init(prov, model)      -> the chart at the model          (`_chart`)
#   lift_refresh!(prov, ch)     rows (and per-point norms for codim 2) at ch
#   lift_jacobian!(prov, ch)    the rows' Jacobian at ch
#   lift_retract(prov, ch, Δθ)  -> the chart re-anchored at the step (`recenter`)
#   lift_model(prov, ch)        -> the RANSAC model                 (`chart_model`)
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

# The problems with a lift: the chart problems whose codimension the lift core
# supports (1 and 2), exactly the `MinimalSolveProblem` family, whose every
# instantiation carries a minimal chart of its model and a codimension-1 or -2
# constraint. The lifts' `locally_optimize!` methods (LocalOptimizationDoneRight)
# are typed on `MinimalSolveProblem`, so the contract's refusal keeps guarding
# every other problem.

function rows_provider(problem::MinimalSolveProblem)
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
lift_model(p::ManifoldLiftProvider, ch) = chart_model(p.problem, ch)

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
    σ = solve(EstimationProblem(NoiseScale, rn, (; codimension = codimension(p.problem))), MedianNormScale()).u   # the norms' robust scale
    loss = deploy(TLSShape(), max(σ^2, floatmin(Float64)); efficiency_constant = _LIFT_SEED_EFFICIENCY_CONSTANT)
    c² = sq_loss_width(loss)
    return _radial_seed_chart(p.problem, m, [weight(TLSShape(), x^2, c²) for x in rn])
end

# =============================================================================
# Triggs — the robustified chart LM of every chart problem (scoring_glue.jl)
# =============================================================================

"""
    locally_optimize!(L, lo::Triggs, problem::MinimalSolveProblem, model, cs, buf, scoring) -> FitResult

The robustified chart LM: the loss the width mode deploys (`lo_loss`:
`efficiency_constant·ŝ` at θ's Student-t scale), frozen, as VisualGeometryToolkitCore's
`loss_robustifier`, and ONE chart LM (the slot's Triggs refine cache) from `model` over every datum at
unit weight, on the chart-LM refit's own chart and state (`_chart`,
`chart_rows`: the rows `refine!` descends) in its workspace (`L.refine`, the
one reuse policy) at Core's LM stop settings (`LmStops`, `[lm]`). NLLSsolver's Triggs
correction carries the loss; nothing is reweighted during the LM and no inlier set is
formed (the loop's re-score decides the inliers). The model comes back through the
chart's `recenter` and `chart_model`, as from `refine!`.

The two weight rules of `Irls{K}`'s `locally_optimize!` fix the one LM's
per-datum weights, 0 or 1 (unit weight on every row when neither excludes a
datum): a datum whose residual at `model` (`buf.residuals` on entry) is not
finite weighs 0 under every shape — its row would make the LM's cost infinite
at its start and hand back the seed — and `Triggs{GaussianShape}` weighs only
the scorer's consensus set `cs`. The weights are fixed at the seed for the whole
LM (a datum an LM increment puts out of reach keeps its weight; the increment's cost is
non-finite and it is rejected). A zero weight gives an exactly zero row
whatever the problem's row is (Core's refine's zero-weight row rule).
"""
function locally_optimize!(L::LoBuffers, lo::Triggs, problem::MinimalSolveProblem, model, cs::ConsensusSet, buf::ScoreBuffers, scoring)
    check_fresh(cs, buf)
    loss = lo_loss(lo, cs, scoring, buf)
    weights = _triggs_weights!(L.posterior_w, _lo_consensus(lo, cs, buf, scoring), buf.residuals, data_size(problem))
    # unit weight (no datum excluded) as ones on the same buffer: the cache holds one weights type (√1 = 1, bit for bit)
    weights === nothing && (weights = fill!(view(L.posterior_w, 1:data_size(problem)), 1.0))
    r = _triggs_solve!(L.refine, lo, problem, model, loss, weights)
    r.u === nothing && return lo_estimate(model, ReturnCode.Failure, NonFinite)   # a non-finite solve: the seed back
    return lo_estimate(r.u, r.retcode, r.stop_rule; iterations = r.stats.nsteps)
end

# The robustified refit: the slot's Triggs cache re-pointed at the problem, the seed, the frozen loss and the weights;
# a slot built for no Triggs strategy (a wrapper the workspace did not see through) refits one-shot.
function _triggs_solve!(slot::RefitSlot{<:Any, R}, lo::Triggs{K}, problem, model, loss, weights) where {R, K}
    (R === Nothing || !_slot_holds(slot, problem)) && return _triggs_one_shot(lo, problem, model, loss, weights)
    c = _slot_robust!(slot, problem, lo)
    return solve!(reinit!(c, model; f = problem, width = loss, weights))
end
_triggs_solve!(::Nothing, lo, problem, model, loss, weights) = _triggs_one_shot(lo, problem, model, loss, weights)
_triggs_one_shot(::Triggs{K}, problem, model, loss, weights) where {K} =
    solve(RefineProblem(problem, model), LevenbergMarquardt(; optimization_strategy = Triggs{K}()); width = loss, weights,
          summary = nothing)

# The Triggs LO's per-datum weights, 0 or 1 (`_weighs`: a finite seed residual,
# inside the consensus set when one is given), in the buffer `w`; `nothing` — unit
# weight on every row, the LM as before — when no datum is excluded. `w` is the
# LO workspace's `posterior_w` (the posterior iteration's weight buffer, which
# every posterior seeding rewrites): it holds these weights after a Triggs LO.
function _triggs_weights!(w::AbstractVector{Float64}, consensus, r::AbstractVector, n::Int)
    consensus === nothing && all(i -> @inbounds(isfinite(r[i])), 1:n) && return nothing
    @inbounds for i in 1:n
        w[i] = _weighs(consensus, r[i], i) ? 1.0 : 0.0
    end
    return view(w, 1:n)
end

# The chart contract of Core's LevenbergMarquardt refine (robust/refine.jl) on every chart problem: the chart at a model
# (`_chart`, the refit's own seeding), the model a chart stands for (`chart_model`, defined per problem), the LM's size, and the rows the
# robustified LM robustifies and a refine's least squares minimises — the refit's (`_refit_state`), whose per-datum
# norm must be the residual's own unit, the unit the loss width is in (the problem's `mahalanobis_distance`). A problem
# whose refit rows are in another unit (PnP's pinhole rows, calibrated) states its pixel rows (problems/p3p.jl).
model_chart(p::AbstractRansacProblem, θ) = _chart(p, _as_model(p, θ))
chart_size(p::AbstractRansacProblem) = _lm_size(p)
chart_rows(problem::AbstractRansacProblem, ch, sqrtw) = _refit_state(problem, ch, sqrtw)

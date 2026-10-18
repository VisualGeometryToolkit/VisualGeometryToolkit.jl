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
# The exceptions: H freezes its transfer whitening at each anchor (refresh),
# PnP carries its pixel block and participation mask (`_lift_aux`,
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
lift_init(p::ManifoldLiftProvider, model) = _chart(p.problem, model)
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

# H — the transfer whitening is frozen at each anchor: the refresh computes it
# into the refit's buffer (the one reuse policy, `_transfer_whitening_buffer`),
# the Jacobian at the same anchor reuses it.
const _HProvider = ManifoldLiftProvider{<:HomographyProblem}
_lift_state(p::_HProvider, ch) = ChartLm(ch, p.problem, p.lws.onev, _transfer_whitening_buffer(p.n))
lift_refresh!(p::_HProvider, ch) = (_freeze_whitening!(_transfer_whitening_buffer(p.n), ch, p.problem); _refresh_rows!(p, ch))

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
# explains: the plain-H reprojection within three medians (the refit's own
# `_radial_seed_chart` otherwise).
function lift_init(p::ManifoldLiftProvider{<:RadialHomographyProblem}, m::VgtkCore.RadialHomography)
    iszero(division_coeff(m)) || return _radial_seed_chart(p.problem, m, nothing)
    H = unit_normalize(SMatrix{3, 3, Float64, 9}(homography(m)))
    lift_refresh!(p, _radial_chart(H, 0.0, center(m)))
    rn = p.lws.rn
    gate = 3 * Statistics.median(view(rn, 1:p.n))
    return _radial_seed_chart(p.problem, m, [rn[i] <= gate ? 1.0 : 0.0 for i in 1:p.n])
end

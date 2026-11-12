# =============================================================================
# The refine contract — the shared body of every RANSAC-class `refine_*`
# =============================================================================
#
# A refine starts from the caller's model and NEVER estimates a scale: it runs
# no scorer, forms no consensus set and has no thin-partition guard. Its robust
# loss runs at a width the caller states — `scale = σ` (or an upstream result,
# its `noise_scale`) times the strategy's efficiency constant, or `width = w` —
# and the result records the scale it ran with (`ScaleGiven`, `ScaleUpstream`,
# or `ScaleNone` for a width alone and for plain least squares). The strategies:
#
#   Triggs{K}  ONE robustified chart LM (Core's `loss_robustifier` handed to
#              NLLSsolver) — the default, `[refine.<class>]`
#   Irls{K}    reweighted chart LMs at the loss weights ρ′, until the robust
#              cost stops decreasing or `max_iter` cycles
#   nothing    plain least squares over every datum (one chart LM), no scale
#
# The result is `FitResult{M, RefineMetadata}`: the refined model, the
# per-datum weights and residuals at it, and the scale record. No inlier mask:
# Irls/Triggs weights are loss weights (`IrlsWeights`, `inlier_mask(r, cut)`),
# L2 weights are all ones (`BinaryWeights`).
#
# The RANSAC-family seeded refine — a scorer, its consensus set, the strategy
# anchored on it — is `locally_optimize`, not a refine_*.
# =============================================================================

"""
    RefineMetadata{L, T, W}

The metadata of every RANSAC-class `refine_*` (`FitResult{M, RefineMetadata}`),
read through the accessors — [`weights`](@extref VisualGeometryToolkitCore.weights), [`weight_kind`](@extref VisualGeometryToolkitCore.weight_kind),
[`noise_scale`](@extref VisualGeometryToolkitCore.noise_scale),
[`scale_source`](@extref VisualGeometryToolkitCore.scale_source) — or as
properties of the result:

- `weights::W`: one per datum at the estimate. For a robust refine (`L` an
  `Irls` or `Triggs` strategy) the loss weight `ρ′(r²)`, maximum 1, exactly 0
  below `eps` ([`IrlsWeights`](@extref VisualGeometryToolkitCore.IrlsWeights):
  no cut of its own, `inlier_mask(result, cut)`); for plain least squares
  (`L === Nothing`) a lazy all-ones vector
  ([`BinaryWeights`](@extref VisualGeometryToolkitCore.BinaryWeights)).
- `residuals::Vector{T}`: each datum's residual norm at the estimate, in the
  problem's residual unit.
- `scale::T`, `scale_source::ScaleSource`: the scale the refine ran with, as it
  was given (`ScaleGiven`, `ScaleUpstream`); `NaN` and `ScaleNone` for a width
  given alone and for plain least squares. A refine never estimates one.
- `loss_width::T`: the width the loss ran at — `width`, or the efficiency
  constant times `scale`; `NaN` for plain least squares.
- `residual_scale::T`: a DIAGNOSTIC, the MAD scale of the residual norms at the
  problem's codimension (`median(‖r‖)/√χ²_d(½)`). It is never handed on: the
  hand-off is `noise_scale`, the scale the refine was given.

`L` is the strategy type (the producer, read by `weight_kind`), `T` the float
type, `W` the weights' vector type.
"""
struct RefineMetadata{L, T <: AbstractFloat, W <: AbstractVector{T}}
    weights::W
    residuals::Vector{T}
    scale::T
    scale_source::ScaleSource
    loss_width::T
    residual_scale::T
end
RefineMetadata{L}(weights::W, residuals::Vector{T}, scale::Real, source::ScaleSource, loss_width::Real,
                  residual_scale::Real) where {L, T, W <: AbstractVector{T}} =
    RefineMetadata{L, T, W}(weights, residuals, T(scale), source, T(loss_width), T(residual_scale))

# The weight kind is the producer's: a robust strategy's loss weights, or (L2,
# `Nothing`, Core's method) all ones.
VgtkCore.weight_kind(::Type{<:RefineMetadata{L}}) where {L} = VgtkCore.weight_kind(L)   # Core's for Irls, Triggs, nothing

# -----------------------------------------------------------------------------
# The keywords: the strategy, the scale input, the workspace — and the refusals
# -----------------------------------------------------------------------------

# A keyword a refine does not take, refused with the way to say it. (The retired width multiple's name is spelled in
# two parts: the stack's naming check refuses it anywhere else.)
const _RETIRED_MULTIPLE = Symbol("scale" * "_multiple")
const _REFINE_KEYWORDS = "optimization_strategy, scale, width, workspace"
function _refine_refusal(key::Symbol)
    seeded = "the RANSAC local-optimization step from a seed, with a scorer and its consensus set, is " *
        "`locally_optimize(problem, θ₀; scoring, local_optimization_strategy)`"
    msg = if key in (:scoring, :threshold, :outlier_halfwidth)
        "`$key` is no keyword of a refine_*: a refine runs no scorer — it estimates no scale and forms no " *
            "consensus set. Give the scale it runs at: `scale = σ` (or an upstream result) or `width = w`; $seeded"
    elseif key === _RETIRED_MULTIPLE
        "the keyword `$key` is retired: give `efficiency_constant` — a field of the strategy " *
            "(`optimization_strategy = Triggs{Shape}(efficiency_constant = c)`), always the plain multiple of `scale`"
    elseif key in (:loss, :loss_width, :efficiency_constant)
        "`$key` is no keyword of a refine_*: the loss is the strategy, `optimization_strategy = Triggs{Shape}(…)` " *
            "or `Irls{Shape}(…)` (its efficiency constant a field of the strategy), and its width `width = w` or " *
            "the efficiency constant times `scale = σ`"
    elseif key in (:robust, :local_optimization, :local_optimization_strategy)
        "the keyword `$key` is retired: give `optimization_strategy` (a refine's strategy: Triggs{Shape}, Irls{Shape} " *
            "or nothing)"
    elseif key === :extent
        "`extent` is no keyword of a refine_*: the extent check validates a consensus set, and a refine forms " *
            "none; $seeded, on `BoundedProblem(problem, extent)`"
    elseif key in (:max_trials, :min_trials, :config, :rng)
        "`$key` is no keyword of a refine_*: a refine draws no samples"
    else
        "`$key` is no keyword of a refine_* (its keywords: $_REFINE_KEYWORDS, and its find_*'s model keywords)"
    end
    return ArgumentError(msg)
end
_refuse_refine_keywords(rest) = (isempty(rest) || throw(_refine_refusal(first(keys(rest)))); nothing)

# A refine runs `Triggs{K}`, `Irls{K}` or `nothing` — nothing else hosts a
# refine without a consensus set.
_refine_strategy(lo::Nothing) = lo
function _refine_strategy(lo::Union{Irls, Triggs})
    strategy_loss(lo) === GaussianShape && throw(ArgumentError(
        "$(VgtkCore.lo_label(lo)) is plain least squares: a refine spells it `optimization_strategy = nothing`"))
    lo_width(lo) isa LossWidth && throw(ArgumentError(
        "a refine's loss width is its keyword `width = w`, not the strategy's `loss_width`"))
    return lo
end
_refine_strategy(lo) = throw(ArgumentError(
    "a refine_* takes `optimization_strategy = Triggs{Shape}(…)`, `Irls{Shape}(…)` or `nothing` (plain least " *
        "squares), got $(nameof(typeof(lo))): it needs a scorer's consensus set, which a refine has none — " *
        "run it from the seed with `locally_optimize(problem, θ₀; scoring, local_optimization_strategy)`"))

# The efficiency constant a refine's strategy `lo` multiplies the scale by, at
# the problem's codimension `d`: the strategy's own (stated on it, ALWAYS the
# plain multiple, `loss_width = c × scale`), or Core's `efficiency_constant(K, d)`
# for a strategy that states none. A shape with no constant (TLS: a truncation)
# needs one stated, or `width`.
_refine_constant(lo, d::Integer) = _refine_constant(lo_width(lo), strategy_loss(lo), d)
_refine_constant(w::EfficiencyConstant{Float64}, _K, _d) = w.value
function _refine_constant(::Nothing, K, d)
    hasmethod(VgtkCore.efficiency_constant, Tuple{Type{K}}) || throw(ArgumentError(
        "$(VgtkCore.loss_label(K)) has no efficiency constant (it is a truncation): state one on the strategy, or give `width = w`"))
    return VgtkCore.efficiency_constant(K, d)
end

# The parsed keywords: the strategy, its scale input and the loss it deploys
# (`nothing` both for L2), the workspace.
struct _RefineSpec{L, S, D, W}
    lo::L
    input::S
    loss::D
    workspace::W
end
_loss_width(spec::_RefineSpec) = sqrt(_sq_loss_width(spec.loss))
# A deployed loss's squared width; NaN for least squares (no loss).
_sq_loss_width(::Nothing) = NaN
_sq_loss_width(loss) = sq_loss_width(loss)

# The loss a robust refine (or bundle) deploys from its scale input at
# codimension `d`, frozen: the shape at `c × scale` (`c` the strategy's
# efficiency constant), or at the caller's `width` — factored as the RANSAC
# local optimization deploys a multiple of its scale and an absolute width
# (`deploy(shape, s2; efficiency_constant)`), so the same width is the same loss.
function _refine_loss(lo, input::ScaleInput, d::Integer)
    shape = strategy_loss(lo)()
    isnan(input.width) &&
        return deploy(shape, max(Float64(input.scale)^2, eps()); efficiency_constant = _refine_constant(lo, d))
    lo_width(lo) === nothing || throw(ArgumentError(
        "`width = $(input.width)` is the loss width itself: the strategy's efficiency constant would be ignored — " *
            "give the strategy none, or give `scale`"))
    return deploy(shape, max(Float64(input.width)^2, eps()); efficiency_constant = 1.0)
end

function _refine_spec(::Type{P}, class::Val;
                      optimization_strategy = _refine_default(class), scale = nothing, width = nothing,
                      workspace = nothing, rest...) where {P}
    _refuse_refine_keywords(rest)
    lo = _refine_strategy(optimization_strategy)
    if lo === nothing
        (scale === nothing && width === nothing) || throw(ArgumentError(
            "plain least squares (optimization_strategy = nothing) reads no scale; got " *
                (scale === nothing ? "width = $width" : "scale = $(repr(scale))")))
        return _RefineSpec(lo, nothing, nothing, workspace)
    end
    input = scale_input(scale, width)
    return _RefineSpec(lo, input, _refine_loss(lo, input, codimension(P)), workspace)
end

# The strategies a bundle refine runs: `Triggs{K}` (the default, its TOML table's `optimization_strategy`), `Irls{K}`
# or `nothing` (least squares), refused otherwise as on every refine (`_refine_strategy`), and the loss a robust one
# deploys at the reprojection residual's codimension 2 from the caller's `scale` or `width` (`_refine_loss`). A
# refine estimates no scale; least squares reads none.
function _bundle_loss(fname, lo, scale, width)
    lo = _refine_strategy(lo)
    if lo === nothing
        (scale === nothing && width === nothing) || throw(ArgumentError(
            "$fname: plain least squares (optimization_strategy = nothing) reads no scale; got " *
                (scale === nothing ? "width = $width" : "scale = $(repr(scale))")))
        return lo, nothing, nothing
    end
    input = scale_input(scale, width)
    return lo, input, _refine_loss(lo, input, 2)
end

# The strategy of record of a refine class (or bundle): configs/default.toml
# [refine.<class>] `optimization_strategy`, or the bundle's own table. Read once,
# into the constant `_REFINE_STRATEGY_DEFAULTS` (after config_toml.jl), and
# looked up by a class known at compile time (`Val`), so the default is inferred.
const _REFINE_CLASSES = (:homography, :fundmat, :essential, :rotation, :circle, :ellipse, :absolute_pose,
                         :bearing_pose, :radial_homography, :radial_camera, :hand_eye)
const _BUNDLE_SECTIONS = (:refine_camera_and_poses, :refine_robot_world_hand_eye)
_dflt_refine_strategy(class::Symbol) =
    _dflt_strategy(class in _BUNDLE_SECTIONS ? String(class) : "refine.$class", class)
_dflt_strategy(section::AbstractString, class::Symbol) =
    optimization_strategy_from_toml(_dflt(section, "optimization_strategy"); class, section = "$section.optimization_strategy")
_read_refine_strategy_defaults() =
    NamedTuple{(_REFINE_CLASSES..., _BUNDLE_SECTIONS...)}(map(_dflt_refine_strategy, (_REFINE_CLASSES..., _BUNDLE_SECTIONS...)))
_refine_default(::Val{C}) where {C} = getfield(_REFINE_STRATEGY_DEFAULTS, C)

# -----------------------------------------------------------------------------
# The pipeline
# -----------------------------------------------------------------------------

"""
    _guarded_refine(problem, P, n, class, θ₀; optimization_strategy, scale, width, workspace)
        -> FitResult{M, RefineMetadata}

The shared body of every RANSAC-class `refine_*` on `n` data of the problem type
`P`: the keywords parsed (and the ones a refine does not take refused), then —
when the data hold a minimal sample — the strategy run once from `θ₀` on
`problem()` (built only then) and the result assembled; otherwise the guess
comes back unrefined, `retcode` `:degenerate`, `iterations` 0, `n` 0 and NaN
residuals.
"""
function _guarded_refine(problem, ::Type{P}, n::Integer, class::Val, θ₀; kwargs...) where {P}
    spec = _refine_spec(P, class; kwargs...)
    _holds_minimal_sample(P, n) || return _refine_too_few(P, spec, θ₀, Int(n))
    return _refine_pipeline(problem(), spec, θ₀)
end

function _refine_pipeline(problem, spec::_RefineSpec{L}, θ₀) where {L}
    _prepare!(problem)
    θ = _seed_convert(model_type(problem), θ₀)
    M = model_type(typeof(problem))
    θ̂, code, iters = _refine_run(spec.lo, problem, θ, spec)
    θ̂ = convert(M, θ̂)::M   # the problem's model type on every path (a failed LM returns its seed), so the result is inferred
    n = data_size(problem)
    r = Vector{Float64}(undef, n)
    mahalanobis_distances!(r, problem, θ̂)
    sq = [abs2(x) for x in r]
    w = _refine_weights(spec, sq)
    support, rms = _weighted_support_rms(w, sq)   # the stack's one rule of a weighted result (loop.jl): n the support, rms weighted
    rs = VgtkCore._mad_scale_norms(; norms = r, codimension = codimension(problem))   # the residual norms (mahalanobis_distances!)
    md = RefineMetadata{L}(w, r, _recorded_scale(spec), _recorded_source(spec), _loss_width(spec), rs)
    return FitResult(θ̂, code, iters, rms, support, md)
end

_recorded_scale(spec::_RefineSpec) = spec.input === nothing ? NaN : Float64(spec.input.scale)
_recorded_source(spec::_RefineSpec) = spec.input === nothing ? ScaleNone : spec.input.source

# The weights at the estimate from the squared residuals `sq`: the loss's (`_loss_weights`), or L2's lazy all ones.
_refine_weights(::_RefineSpec{Nothing}, sq) = unit_weights(Float64, length(sq))
_refine_weights(spec::_RefineSpec, sq) = _loss_weights(strategy_loss(spec.lo)(), sq_loss_width(spec.loss), sq)

# THE loss weights of every refine (the RANSAC classes and both bundles) from the squared residuals `sq`: ρ′ at the
# loss's squared width `c²`, max 1, and a weight below eps an exact 0 (out of the support, §12).
_loss_weights(shape, c², sq) = [(x = Float64(weight(shape, s, c²)); x < eps(Float64) ? 0.0 : x) for s in sq]

# Too few data for a minimal sample: the guess, unrefined, with no residual.
function _refine_too_few(::Type{P}, spec::_RefineSpec{L}, θ₀, n::Int) where {P, L}
    w = L === Nothing ? unit_weights(Float64, n) : zeros(n)   # L2's weights are its unit weights; nothing was fitted (n = 0)
    md = RefineMetadata{L}(w, fill(NaN, n), _recorded_scale(spec), _recorded_source(spec), _loss_width(spec), NaN)
    return FitResult(_seed_convert(model_type(P), θ₀), :degenerate, 0, NaN, 0, md)
end

# The chart LM's workspace from the caller's `workspace` (a RansacWorkspace, its
# local optimizer's buffers, or a chart-LM workspace); the one reuse policy
# (`_lm_workspace`) takes over when it does not fit.
_refine_lm_ws(::Nothing) = nothing
_refine_lm_ws(ws::RansacWorkspace) = _refine_lm_ws(ws.lo_ws)
_refine_lm_ws(L::LoBuffers) = L.refine
_refine_lm_ws(ws) = ws

# -----------------------------------------------------------------------------
# The three strategies — (model, retcode, iterations)
# -----------------------------------------------------------------------------

# Plain least squares: ONE chart LM from θ over every datum at unit weight.
function _refine_run(::Nothing, problem, θ, spec::_RefineSpec)
    ch = _chart(problem, _as_model(problem, θ))
    rows, dof = _lm_size(problem)
    ews = _lm_workspace(_refine_lm_ws(spec.workspace), rows, dof)
    fill!(sqrt_weights(ews), 1.0)
    lm = _refit_state(problem, ch, sqrt_weights(ews))
    x, _, code, n_iter = lmsolve_chart!(ews, lm; maxit = _CHART_LM_MAX_ITER, stationarity_tol = _CHART_LM_STATIONARITY_TOL,
                                        ftol = _CHART_LM_FTOL)
    any(isnan, x) && return θ, :solve_failed, n_iter
    return _model_at(problem, recenter(ch, x)), lm_retcode(code), n_iter
end

# Triggs: ONE robustified chart LM at the frozen width (`_triggs_lm`, the RANSAC
# local optimization's own).
function _refine_run(lo::Triggs, problem, θ, spec::_RefineSpec)
    r = _triggs_lm(_refine_lm_ws(spec.workspace), strategy_loss(lo)(), problem, θ, spec.loss)
    return estimate(r), retcode(r), iterations(r)
end

# Irls: the weights ρ′ at the current model, one weighted chart LM (`refine!`),
# repeated while the robust cost Σρ(r²) decreases, at most `max_iter` cycles.
# `iterations` counts the cycles.
function _refine_run(lo::Irls, problem, θ, spec::_RefineSpec)
    shape = strategy_loss(lo)()
    c² = sq_loss_width(spec.loss)
    n = data_size(problem)
    ws = _refine_lm_ws(spec.workspace)
    r = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    cost_at!(θ) = (mahalanobis_distances!(r, problem, θ); sum(x -> ρ(shape, x^2, c²), r))
    cost = cost_at!(θ)
    for k in 1:lo.max_iter
        w .= weight.(Ref(shape), r .^ 2, c²)
        θ′ = refine!(ws, problem, θ, w)
        θ′ === nothing && return θ, :solve_failed, k
        cost′ = cost_at!(θ′)
        cost′ < cost || return θ, :converged, k
        θ, cost = θ′, cost′
    end
    return θ, :max_iterations, lo.max_iter
end

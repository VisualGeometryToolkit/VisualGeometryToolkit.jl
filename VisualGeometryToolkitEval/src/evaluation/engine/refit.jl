# =============================================================================
# evaluation/engine/refit.jl — one refit layer for every local optimizer.
#
# `RefitFamily(lo)` (VisualGeometryToolkitCore's Holy trait) routes a strategy to its
# execution path; `guarded_refit` is the single seed-fallback + non-finite-
# fallback + timing bracket; `refit_imc!` is the one call every consumer
# (budget selectors, LO matrices, the basin ladder) makes. Written against the
# contracts only:
#
#   NativeRefit   — the default: the strategy runs in VisualGeometryToolkitEstimation's
#                   production `locally_optimize` (VisualGeometryToolkitCore's
#                   `locally_optimize!`), seeded by the refit scoring at the
#                   refit's own outlier half-width — the caller's scoring
#                   TEMPLATE rebuilt `with_width(template, OutlierHalfwidth(nat′))`.
#                   Its loss width comes from its width mode (the consensus
#                   scale), or from an absolute scale `s2` the caller supplies
#                   (`with_width(lo, fixed_width(lo, s2))`).
#   ExternalRefit — a third-party solver's model → model refine at an absolute
#                   loss width (VisualGeometryToolkitCore's `refine_external!`, on its
#                   per-pair `external_workspace`): PoseLib's truncated refiners
#                   (LocalOptimizationDoneRight's PoseLib extension), RansacLib's
#                   classical LO (a paper project's).
# =============================================================================

"""
    ScoreGuarded(inner::AbstractLocalOptimization)

LO-RANSAC best-so-far monotonicity wrapper for IRLS-family arms.
The inner LO runs unchanged, but the refined model is kept only if its SELECTION score (the
same marginal that seeded it) does not drop below the seed's
(VisualGeometryToolkitEstimation's `score_guard`). It preserves score monotonicity while
the optimizer descends a different objective.

Its `RefitFamily` and the external-refit bracket (`external_workspace`,
`prepare_external!`, `refine_external!`) are `inner`'s, as on
VisualGeometryToolkitEstimation's `CountedLO` and `TerminalLO`: a guarded external
arm is routed, prepared and refined as the bare arm. The guard is
`score_guard` on the native path, so an external refit is forwarded unguarded.
"""
struct ScoreGuarded{L<:AbstractLocalOptimization} <: AbstractLocalOptimization
    inner::L
end
VgtkCore.RefitFamily(g::ScoreGuarded) = RefitFamily(g.inner)
VgtkCore.external_workspace(g::ScoreGuarded, problem, frame, scoring) =
    external_workspace(g.inner, problem, frame, scoring)
VgtkCore.prepare_external!(ws, g::ScoreGuarded, problem, θ, width, scoring) =
    prepare_external!(ws, g.inner, problem, θ, width, scoring)
# (A fixed noise scale reaches the `LossWidth` method through VGC's `FixedS2` default.)
VgtkCore.refine_external!(ws, g::ScoreGuarded, problem, θ, width::LossWidth, scoring) =
    refine_external!(ws, g.inner, problem, θ, width, scoring)
# Identity + interface traits are forwarded to the guarded strategy.
VgtkCore.optimization_path(g::ScoreGuarded) = optimization_path(g.inner)
VgtkCore.scale_interface(g::ScoreGuarded) = scale_interface(g.inner)
VgtkCore.loss_family(g::ScoreGuarded) = loss_family(g.inner)
VgtkCore.lo_label(g::ScoreGuarded) = lo_label(g.inner)
VgtkCore.lo_marker(g::ScoreGuarded) = lo_marker(g.inner)
_unwrap_lo(lo::AbstractLocalOptimization) = lo
_unwrap_lo(lo::ScoreGuarded) = lo.inner
_is_guarded(::AbstractLocalOptimization) = false
_is_guarded(::ScoreGuarded) = true

"""
    NativeAxis

The native width axes, `Union{OutlierHalfwidth, LossWidth}` (VisualGeometryToolkitCore
`WidthAxis`): what a selector grid, a selection sweep or a refiner's own native
lives on. `ScaleMultiple()` is a width parameterization, never a native.
"""
const NativeAxis = Union{OutlierHalfwidth, LossWidth}

"""
    consumed_axis(lo) -> NativeAxis

The native axis refiner `lo` consumes, by its refit family: a native refiner
its scoring outlier half-width (`OutlierHalfwidth()`), an external one its
truncated loss's width (`LossWidth()`).
"""
consumed_axis(lo::AbstractLocalOptimization) = consumed_axis(RefitFamily(lo), lo)
consumed_axis(::NativeRefit, lo) = OutlierHalfwidth()
consumed_axis(::ExternalRefit, lo) = LossWidth()
# VisualGeometryToolkitEstimation's wrappers and ScoreGuarded consume what their
# inner strategy does (a strategy may state its own axis, not only its family's)
consumed_axis(lo::Union{CountedLO, TerminalLO, ScoreGuarded}) = consumed_axis(lo.inner)

"""
    consumed_native(lo, axis::NativeAxis, native::Real;
                    operating_scale_multiple = OPERATING_SCALE_MULTIPLE, d_g) -> Float64

The native width THIS refiner consumes ([`consumed_axis`](@ref)), given a
selector grid value `native` on native axis `axis`. Same-axis is identity;
cross-axis converts at the `operating_scale_multiple` crossover
(VisualGeometryToolkitCore `outlier_halfwidth_to_loss_width` /
`loss_width_to_outlier_halfwidth`) for residual codimension `d_g`
(`codimension(problem_type)`: 1 for E/F, 2 for H).
"""
consumed_native(lo, axis::NativeAxis, native::Real;
                operating_scale_multiple::Real = OPERATING_SCALE_MULTIPLE, d_g::Integer) =
    _convert_native(axis, consumed_axis(lo), Float64(native), Float64(operating_scale_multiple), Int(d_g))
_convert_native(::A, ::A, x, operating_scale_multiple, d_g) where {A<:NativeAxis} = x
_convert_native(::OutlierHalfwidth, ::LossWidth, x, operating_scale_multiple, d_g) =
    outlier_halfwidth_to_loss_width(x; operating_scale_multiple, d_g)
_convert_native(::LossWidth, ::OutlierHalfwidth, x, operating_scale_multiple, d_g) =
    loss_width_to_outlier_halfwidth(x; operating_scale_multiple, d_g)

"""
    guarded_refit(opt, score, on_fail; warn_label = nothing)
        -> (err::Float64, lo_ns::Int64)

The seed-fallback + timing bracket.

- `opt() -> Union{Nothing, M}` — the bracketed optimizer call (data-in →
  result-out). Throws are caught; `nothing` signals bail / no consensus.
- `score(m::M) -> Float64` — GT scoring of the refined model, NOT timed.
  A non-finite score falls back to `on_fail` (a GT-free decision: the refit
  produced an unusable model).
- `on_fail` policy is the caller's (`refit_imc!`'s fallback): budget/producer supply the seed's error
  (LO-RANSAC invariant — refinement never worsens the running best); the
  basin ladder supplies `NaN` (a failed refit did not travel).
- `lo_ns` spans exactly `opt()`.
"""
function guarded_refit(opt::F, score::G, on_fail::Float64;
                       warn_label::Union{Nothing,String} = nothing) where {F,G}
    t0 = time_ns()
    m = try
        opt()
    catch e
        warn_label === nothing ||
            @warn "$warn_label failed (fallback applied)" exception = e
        nothing
    end
    lo_ns = Int64(time_ns() - t0)
    m === nothing && return (on_fail, lo_ns)
    e = Float64(score(m))
    return (isfinite(e) ? e : on_fail, lo_ns)
end

"""
    refit_imc!(ctx::PairCtx, lo, seed, nat′::Float64;
               scoring = ctx.scoring, on_fail = nothing, bundle::Bool = false,
               s2 = nothing) -> (imc::Float64, lo_ns::Int64)

Refine the seed model `seed` with strategy `lo` at ITS native parameter `nat′`
(see [`consumed_native`](@ref)) and return the refined model's imc error [deg]
plus the bracketed optimizer wall-clock.

- `scoring` (NativeRefit): the scoring TEMPLATE the refit is seeded under,
  rebuilt at `nat′` (`with_width(scoring, OutlierHalfwidth(nat′))`) — the
  context's refit template by default; `ctx.solo_scoring` for an
  absolute-width (solo) refit.
- `on_fail`: value on optimizer bail / non-finite refit. Default = the seed's
  own imc (never-worsen); the ladder supplies `NaN`.
- `bundle`: route the native refit through `ctx.bp` (Schur reprojection bundle)
  instead of the pose-only `ctx.ep`.
- `s2`: an externally supplied squared scale — the one conduit for a
  non-consensus width:
  * NativeRefit: the strategy is deployed at the absolute width
    `with_width(lo, fixed_width(lo, s2))` (its scale multiple at the scale
    `√s2`) — the role-isolated controls (solo CV, oracle-scale controls, the
    anchoring ablation). The seed sweep still runs; only the LO's width moves.
  * ExternalRefit: the fixed scale `FixedS2(s2)` replaces `nat′` (by default
    the loss width `√s2`).
"""
function refit_imc!(ctx::PairCtx, lo::AbstractLocalOptimization, seed, nat′::Float64;
                    scoring = ctx.scoring, on_fail = nothing, bundle::Bool = false,
                    s2 = nothing)
    fallback = on_fail === nothing ? model_imc(seed, ctx) : Float64(on_fail)
    refit_imc!(ctx, RefitFamily(lo), lo, seed, nat′, scoring, fallback, bundle, s2)
end

"""
    refit_imc!(ctx, family::RefitFamily, lo, seed, nat′, scoring, fallback, bundle, s2)
        -> (imc::Float64, lo_ns::Int64)

The execution path of one `RefitFamily` (VisualGeometryToolkitCore's Holy trait:
`NativeRefit`, `ExternalRefit`). [`refit_imc!`](@ref)`(ctx, lo, seed, nat′; …)`
resolves the keywords and calls this with `RefitFamily(lo)`. A new strategy
plugs in through the contracts (`locally_optimize!`, or `refine_external!`
with its `external_workspace`), never by a method here.
"""
function refit_imc!(ctx, ::NativeRefit, lo, seed, nat′, scoring, fallback, bundle, s2)
    scoring === nothing && throw(ArgumentError(
        "refit_imc!: a native refit needs a scoring template (build_pair_ctx(…; scoring))"))
    prob = bundle ? (ctx.bp === nothing ?
                     error("refit_imc!: bundle=true needs build_pair_ctx(...; bundle=true)") :
                     ctx.bp) : ctx.ep
    sc = with_width(scoring, OutlierHalfwidth(nat′))
    ws = _ctx_workspace!(ctx, _ws_key(Symbol(bundle ? :bp_ : :ep_, _template_tag(ctx, scoring)), lo), prob, sc)
    # Function barrier: `sc` and `ws` arrive abstractly typed; inside
    # `_native_refit_imc` they are concrete, so `locally_optimize` specializes
    # statically per (problem, scoring, LO) triple.
    _native_refit_imc(prob, seed, lo, sc, m -> model_imc(m, ctx), fallback, ws,
                      s2 === nothing ? nothing : Float64(s2))
end

# The workspace-key tag of a refit's scoring template: the context's refit
# template, its solo template, or another (keyed by its value at unit width).
# A scorer's workspace may depend on its configuration (the marginal's log Γ
# table on its prior), so templates never share one.
_template_tag(ctx, scoring) =
    scoring === ctx.scoring ? :refit :
    scoring === ctx.solo_scoring ? :solo :
    Symbol(:sc, hash(with_width(scoring, OutlierHalfwidth(1.0))))

# Reusable per-pair workspaces (see PairCtx.lo_ws). Keyed by problem variant ×
# scoring template × LO TYPE — buffers are size/type-only, so any refit of the
# same (problem, scoring template, strategy) reuses them; every call re-scores
# its seed.
_ws_key(base::Symbol, lo) = Symbol(base, :_, nameof(typeof(_unwrap_lo(lo))))
_ctx_workspace!(ctx, key::Symbol, prob, sc) =
    get!(() -> RansacWorkspace(prob, sc), ctx.lo_ws, key)

"""
    refit_model(problem, seed, lo, scoring; workspace = nothing, s2 = nothing)
        -> Union{Nothing, model}

The native (`locally_optimize`) refit of the refit layer, returning the refined
MODEL — `nothing` when the strategy produced none. `lo` may be `ScoreGuarded`
(the guard becomes `score_guard`). `s2` deploys the same strategy at an
absolute squared scale (`with_width(lo, fixed_width(lo, s2))`; the scoring
sweep is untouched, only the LO's width moves); `nothing` keeps the strategy's
own width mode. The refit layer's imc rows wrap exactly this in
`guarded_refit`; a caller that needs the refined model itself (a polish warm
start) calls it directly.
"""
function refit_model(prob, seed, lo::AbstractLocalOptimization, sc;
                     workspace = nothing, s2::Union{Nothing,Real} = nothing)
    lo_run = _unwrap_lo(lo)
    lo_run = s2 === nothing ? lo_run : with_width(lo_run, fixed_width(lo_run, Float64(s2)))
    r = locally_optimize(prob, seed; local_optimization = lo_run, scoring = sc,
                         workspace, score_guard = _is_guarded(lo))
    return converged(r) ? estimate(r) : nothing
end

function _native_refit_imc(prob, seed, lo, sc, imcf::F, fallback::Float64, ws,
                           s2_fixed::Union{Nothing,Float64}) where {F}
    opt = () -> refit_model(prob, seed, lo, sc; workspace = ws, s2 = s2_fixed)
    guarded_refit(opt, imcf, fallback; warn_label = "locally_optimize ($(nameof(typeof(lo))))")
end

# --- ExternalRefit: the solver's own model → model refine at an absolute loss
# width — the refiner's own native `nat′`, or a fixed scale `FixedS2(s2)` when
# the caller supplies one (by default the loss width `√s2`; an adaptive arm
# reads the scale itself) — on its per-pair workspace. `prepare_external!` runs outside
# the bracket (a consensus-scale bootstrap); `false` is no refit, no call.
function refit_imc!(ctx, ::ExternalRefit, lo, seed, nat′, scoring, fallback, bundle, s2)
    prob = ctx.ep
    width = s2 === nothing ? LossWidth(Float64(nat′)) : FixedS2(Float64(s2))
    ws = get!(() -> external_workspace(lo, prob, ctx.geom, scoring), ctx.lo_ws,
              Symbol(:ext_, _template_tag(ctx, scoring), :_, nameof(typeof(_unwrap_lo(lo)))))
    prepare_external!(ws, lo, prob, seed, width, scoring) || return (fallback, Int64(0))
    guarded_refit(() -> refine_external!(ws, lo, prob, seed, width, scoring),
                  m -> model_imc(m, ctx), fallback; warn_label = "external refine ($(nameof(typeof(lo))))")
end

# --- An ABSOLUTE loss width `w` (the solo validations, the loss-width tuning
# columns): a native refiner sweeps its seed at `nat′` under the context's solo
# template and runs at `with_width(lo, fixed_width(lo, w²))` (its scale
# multiple at the scale `w` — the unit-multiple strategy runs at exactly `w`);
# an external refiner refines at `w` itself.
_width_refit_imc!(ctx, ::NativeRefit, lo, seed, w::Float64, nat′::Float64; kw...) =
    refit_imc!(ctx, lo, seed, nat′; scoring = ctx.solo_scoring, s2 = w^2, kw...)
_width_refit_imc!(ctx, ::ExternalRefit, lo, seed, w::Float64, nat′::Float64; kw...) =
    refit_imc!(ctx, lo, seed, w; kw...)

# -----------------------------------------------------------------------------
# The refit half-width INSIDE a RANSAC loop (live end-to-end runs; any loop that
# hands its LO a scoring, e.g. an absolute-pose race)
# -----------------------------------------------------------------------------
"""
    RefitHalfwidth(lo, outlier_halfwidth, scoring_of)
    RefitHalfwidth(lo, outlier_halfwidth, template::AbstractScoring)

`lo` refitting under the refit scoring at ITS OWN outlier half-width instead of
the selection's: the in-loop form of the refit layer's native port
(`refit_imc!`'s `nat′`, `refit_model`). The self-scaled arms (Stu-tU IRLS, BCD)
read their half-width from the scoring the loop hands them, so inside a RANSAC
loop they would otherwise refit at the SELECTION half-width — not the refit
half-width their sequential CV tuned at the fixed a*. `scoring_of(problem)`
builds the refit scoring; the template form is `with_width(template,
OutlierHalfwidth(outlier_halfwidth))`.

In the loop, each call re-scores the model under the refit scoring in the
loop's own buffers and runs the loop's own Phase-2 step on that consensus set
(VisualGeometryToolkitEstimation's `locally_optimize_rescored!`): at `outlier_halfwidth` =
the selection's it is the score the loop just ran, so the wrapped arm
reproduces the bare arm exactly. The refined model goes back to the loop, which
re-scores it under the SELECTION scoring and keeps it only if it beats the
incumbent.
"""
struct RefitHalfwidth{L <: AbstractLocalOptimization, F} <: AbstractLocalOptimization
    lo::L
    outlier_halfwidth::Float64
    scoring_of::F
    function RefitHalfwidth(lo::L, outlier_halfwidth::Real, scoring_of::F) where {L <: AbstractLocalOptimization, F <: Function}
        isfinite(outlier_halfwidth) && outlier_halfwidth > 0 ||
            throw(ArgumentError("RefitHalfwidth: the refit outlier half-width must be finite and positive, got $outlier_halfwidth"))
        return new{L,F}(lo, Float64(outlier_halfwidth), scoring_of)
    end
end
# `template` is any scoring with an outlier half-width (`with_width(template,
# OutlierHalfwidth(a))`); the function form above is more specific.
function RefitHalfwidth(lo::AbstractLocalOptimization, outlier_halfwidth::Real, template)
    a = OutlierHalfwidth(Float64(outlier_halfwidth))::OutlierHalfwidth{Float64}
    sc = with_width(template, a)
    return RefitHalfwidth(lo, outlier_halfwidth, _ -> sc)
end
VgtkCore.lo_workspace(::Type{P}, ::Type{T}, r::RefitHalfwidth, n::Int) where {P <: AbstractRansacProblem, T} =
    lo_workspace(P, T, r.lo, n)
function VgtkCore.locally_optimize!(L, r::RefitHalfwidth, problem, model,
                                              ::ConsensusSet, buf::ScoreBuffers, _scoring)
    # Model-certain scoring only: the re-score scores the PLAIN model; a
    # predictive scoring would need its covariance-wrapped form.
    return locally_optimize_rescored!(L, r.lo, problem, model, buf, r.scoring_of(problem))
end

# =============================================================================
# fit_rotation — a robust rotation between two sets of directions
# =============================================================================
#
# The rotation R with bᵢ ≈ R·aᵢ from direction pairs `a => b` (star directions
# in a camera and on the sky, a gravity vector in two frames), with some of
# the pairs wrong. An iteratively reweighted Kabsch/Wahba:
#
#   seed      VisualGeometryToolkitCore's `kabsch` of the pairs (prior weights, if any);
#   repeat    the residuals rᵢ = ∠(R·aᵢ, bᵢ) (radians: the norm of a 2-D
#             tangent residual, so codimension 2); the IRLS weights of the
#             `Irls{K}` strategy's loss at ONE loss width, set once from the
#             caller's scale input (`scale = σ` or an upstream result: the
#             efficiency constant × σ, Huber's 1.5011 at codimension 2; or
#             `width = w`) and never re-estimated (irls_weights.jl); the
#             weighted Kabsch.
#   stop      when an iteration moves R by less than `tol` (radians), or at
#             the strategy's `max_iter`.
#
# Hoisted from RadiometricCamera's experiments/lib/stars.jl
# (`fit_rotation_irls`, `wahba`), 2026-10-27, with two corrections: the scale
# of a 2-D residual's norm is its median over 1.1774 (Rayleigh), not 1.4826 ×
# median (a 1-D absolute deviation's, which reads 2-D norms 1.745× too wide);
# and the loss and its width are the stack's (an `Irls{K}` strategy) instead
# of a hard-wired Huber at 3 × that scale. Since the robust-estimation
# contract (0.9) the scale is the caller's and fixed (RC's width: Huber at
# `width = 3 · 1.4826 · median(angles)` of its own scale estimate).
# =============================================================================

"""
    RotationFitMetadata{L, W}

`FitResult` metadata of [`fit_rotation`](@ref), all at the returned rotation
(`L` the `optimization_strategy`'s type, `Nothing` for least squares): `weights::W`,
the loss weight `ρ′(r²)` of every pair normalized to a maximum of 1
(`IrlsWeights`; a lazy all-ones vector, `BinaryWeights`, for least squares —
the solve's weight is the pair's prior times this); `residuals`, the angle
`∠(R·aᵢ, bᵢ)` of every pair in radians; `scale`, `scale_source` (radians; the
scale record, `noise_scale(result)`); `loss_width` (radians, `NaN` for least
squares).
"""
struct RotationFitMetadata{L, W <: AbstractVector{Float64}}
    weights::W
    residuals::Vector{Float64}
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    loss_width::Float64
end
RotationFitMetadata{L}(w::W, r, s, src, lw) where {L, W} = RotationFitMetadata{L, W}(w, r, s, src, lw)
VgtkCore.weight_kind(::Type{<:RotationFitMetadata{L}}) where {L} = VgtkCore.weight_kind(L)   # Core's trait of the strategy (Irls: IrlsWeights; nothing: binary)

"""
    fit_rotation(cs; optimization_strategy, scale | width, tol)
        -> FitResult{RotMatrix{3, Float64}, RotationFitMetadata}

The robust rotation `R` with `bᵢ ≈ R·aᵢ` from direction pairs `cs` (`aᵢ => bᵢ`,
3-vectors; normalized at entry, so any nonzero length is a direction), with a
fraction of the pairs gross outliers. A fit — it seeds itself with
VisualGeometryToolkitCore's [`kabsch`](@extref VisualGeometryToolkitCore.kabsch) — iterating a reweighted Kabsch
at ONE loss width: residual angles `rᵢ = ∠(R·aᵢ, bᵢ)`, the weights of the
strategy's loss at that width, and the weighted Kabsch. The width is fixed by
the caller's scale input and never re-estimated.

- `optimization_strategy` — the robust weighting, an [`Irls`](@extref VisualGeometryToolkitCore.Irls)`{K}` strategy (its loss
  `K`, its efficiency constant and its `max_iter`; default
  configs/default.toml `[fit_rotation] optimization_strategy`, Huber), or `nothing` for the
  plain (prior-weighted) Kabsch.
- `scale` | `width` — with an `Irls` strategy, EXACTLY ONE of: `scale = σ`
  (the residual angles' noise scale per tangent component, radians),
  `scale = result` (an upstream robust result's `noise_scale`), or
  `width = w` (the loss width, radians). The width is `w`, or `c · σ` with `c`
  the strategy's efficiency constant, by default the shape's at codimension 2
  (`efficiency_constant(K, 2)`: Huber 1.5011); a shape with none
  (`Irls{TLSShape}()`) needs the strategy's or a `width`. Neither, both, or
  one given with `optimization_strategy = nothing` is an `ArgumentError`.
- `tol` — converged when an iteration moves `R` by less than `tol` radians
  (default configs/default.toml [fit_rotation]).
- A `weights` column on `cs` (`withattrs(cs; weights)`) multiplies every pair's
  loss weight (fixed prior weights).

`retcode_symbol(result)` is `:converged`, `:max_iterations`, `:solved` (least squares), or
`:degenerate` when fewer than two pairs keep a positive weight. `n` is the
support (the pairs of positive solve weight), `rms` the weight-weighted root
mean square residual angle. The metadata ([`RotationFitMetadata`](@ref))
carries the weights, the residual angles, the scale record and the loss
width.

Accuracy: with a minority of gross outliers and the inliers' scale given, the
estimate reaches the noise floor of the inliers' own least-squares rotation
(≈ σ/√m for m inliers at direction noise σ); the suite pins 20% random
outliers among 200 pairs at σ = 1e-3 rad to within 5e-4 rad, where the plain
Kabsch is off by over 1e-2.
"""
function fit_rotation(
        cs::AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{3, <:Real}}};
        optimization_strategy::Union{Nothing, AbstractOptimizationStrategy} = _fit_strategy_default("fit_rotation"),
        local_optimization = _RetiredKeyword(),   # refused, naming its successor
        scale = nothing, width::Union{Nothing, Real} = nothing,
        tol::Real = _dflt("fit_rotation", "tol"),
    )
    n = length(cs)
    n >= 2 || throw(ArgumentError("fit_rotation needs at least 2 direction pairs, got $n"))
    _refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    f = _fit_irls("fit_rotation", optimization_strategy, scale, width, 2)
    as = SVector{3, Float64}[_unit_direction(a, i) for (i, a) in enumerate(sources(cs))]
    bs = SVector{3, Float64}[_unit_direction(b, i) for (i, b) in enumerate(targets(cs))]
    w_col = VgtkCore.weights_column(cs)
    prior = w_col === nothing ? ones(n) : Float64.(VgtkCore.check_weights(w_col))
    w = _fit_weights_storage(f, n)
    R = _weighted_kabsch(as, bs, prior)
    R === nothing && return _rotation_degenerate(optimization_strategy, f, n, w)
    r = _rotation_residuals(R, as, bs)
    tol_met = false
    iter = 0
    while iter < _fit_max_iter(optimization_strategy)
        iter += 1
        _irls_weights!(w, f, r)
        R_new = _weighted_kabsch(as, bs, _solve_weights(prior, w))
        R_new === nothing && return _rotation_degenerate(optimization_strategy, f, n, w, _irls_record(false, false; solve_failed = true))
        Δ = Rotations.rotation_angle(R_new * R')
        R = R_new
        r = _rotation_residuals(R, as, bs)
        if Δ < tol
            tol_met = true
            break
        end
    end
    _irls_weights!(w, f, r)                    # the metadata at the returned rotation
    sw = _solve_weights(prior, w)
    used, rms = _weighted_support_rms(sw, (abs2(x) for x in r))
    return FitResult(R, _irls_record(tol_met, f === nothing)..., iter, rms, used, RotationFitMetadata{typeof(optimization_strategy)}(w, r, _fit_scale_record(f)...))
end

function _unit_direction(v, i)
    x = SVector{3, Float64}(v)
    nx = norm(x)
    (isfinite(nx) && nx > 0) || throw(ArgumentError("fit_rotation: direction $i is zero or not finite"))
    return x / nx
end

# Σ wᵢ bᵢaᵢᵀ through VGC's `kabsch` (its pairs scaled by the weights); `nothing`
# when fewer than two pairs carry weight (a one-parameter family of rotations).
function _weighted_kabsch(as, bs, w)
    count(>(0), w) >= 2 || return nothing
    return RotMatrix{3, Float64}(kabsch([w[i] * as[i] => bs[i] for i in eachindex(as) if w[i] > 0]))
end

_rotation_residuals(R, as, bs) = [angle_between(R * a, b) for (a, b) in zip(as, bs)]   # VGC's atan(‖Ra × b‖, Ra · b)

# A solve that failed: before the loop, a degenerate start `(InitialFailure, NoRule)`; inside it, a weighted
# solve that could not be solved `(InternalLinearSolveFailed, NoRule)`.
_rotation_degenerate(strategy, f, n, w, record::Tuple{ReturnCode.T, StopRule} = (ReturnCode.InitialFailure, NoRule)) =
    FitResult(one(RotMatrix{3, Float64}), record..., 0, NaN, 0, RotationFitMetadata{typeof(strategy)}(w, fill(NaN, n), _fit_scale_record(f)...))

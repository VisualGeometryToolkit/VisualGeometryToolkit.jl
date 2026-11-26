# =============================================================================
# ReweightedLeastSquares on a rotation — a robust rotation between two sets of directions
# =============================================================================
#
# The rotation R with bᵢ ≈ R·aᵢ from direction pairs `a => b` (star directions
# in a camera and on the sky, a gravity vector in two frames), with some of
# the pairs wrong. An iteratively reweighted Kabsch/Wahba:
#
#   seed      VisualGeometryToolkitCore's `Kabsch` of the pairs (prior weights, if any);
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
    solve(EstimationProblem(RotMatrix{3}, cs), ReweightedLeastSquares(; optimization_strategy, tol); scale | width)
        -> FitResult{RotMatrix{3, Float64}, FitMetadata}

The robust rotation `R` with `bᵢ ≈ R·aᵢ` from direction pairs `cs` (`aᵢ => bᵢ`,
3-vectors; normalized at entry, so any nonzero length is a direction), with a
fraction of the pairs gross outliers. A fit — it seeds itself with
VisualGeometryToolkitCore's [`Kabsch`](@extref VisualGeometryToolkitCore.Kabsch) — iterating a reweighted Kabsch
at ONE loss width: residual angles `rᵢ = ∠(R·aᵢ, bᵢ)`, the weights of the
strategy's loss at that width, and the weighted Kabsch. The width is fixed by
the caller's scale input and never re-estimated.

- `optimization_strategy` — the robust weighting, an [`Irls`](@extref VisualGeometryToolkitCore.Irls)`{K}` strategy (its loss
  `K`, its efficiency constant and its `max_iter`; default
  configs/default.toml `[reweighted_least_squares.rotation] optimization_strategy`, Huber), or `nothing` for the
  plain (prior-weighted) Kabsch.
- `scale` | `width` (solve keywords) — with an `Irls` strategy, EXACTLY ONE of: `scale = σ`
  (the residual angles' noise scale per tangent component, radians),
  `scale = result` (an upstream robust result's `noise_scale`), or
  `width = w` (the loss width, radians). The width is `w`, or `c · σ` with `c`
  the strategy's efficiency constant, by default the shape's at codimension 2
  (`efficiency_constant(K, 2)`: Huber 1.5011); a shape with none
  (`Irls{TLSShape}()`) needs the strategy's or a `width`. Neither, both, or
  one given with `optimization_strategy = nothing` is an `ArgumentError`.
- `tol` — converged when an iteration moves `R` by less than `tol` radians
  (default configs/default.toml [reweighted_least_squares.rotation]).
- A `weights` column on `cs` (`withattrs(cs; weights)`) multiplies every pair's
  loss weight (fixed prior weights).

`retcode_symbol(result)` is `:converged`, `:max_iterations`, `:solved` (least squares), or
`:degenerate` when fewer than two pairs keep a positive weight. `n` is the
support (the pairs of positive solve weight), `rms` the weight-weighted root
mean square residual angle. The metadata ([`FitMetadata`](@ref))
carries the weights, the residual angles, the scale record and the loss
width.

Accuracy: with a minority of gross outliers and the inliers' scale given, the
estimate reaches the noise floor of the inliers' own least-squares rotation
(≈ σ/√m for m inliers at direction noise σ); the suite pins 20% random
outliers among 200 pairs at σ = 1e-3 rad to within 5e-4 rad, where the plain
Kabsch is off by over 1e-2.
"""
function __solve(
        prob::EstimationProblem{<:RotMatrix{3}, <:AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{3, <:Real}}}, NullParameters},
        alg::ReweightedLeastSquares; scale = nothing, width::Union{Nothing, Real} = nothing,
    )
    cs = prob.f
    who = "ReweightedLeastSquares (rotation)"
    n = length(cs)
    n >= 2 || throw(ArgumentError("$who needs at least 2 direction pairs, got $n"))
    optimization_strategy, tol = _reweighted_settings(alg, "reweighted_least_squares.rotation", who)
    f = _fit_irls(who, optimization_strategy, scale, width, 2)
    as = SVector{3, Float64}[_unit_direction(a, i) for (i, a) in enumerate(sources(cs))]
    bs = SVector{3, Float64}[_unit_direction(b, i) for (i, b) in enumerate(targets(cs))]
    w_col = VgtkCore.weights_column(cs)
    prior = w_col === nothing ? ones(n) : Float64.(VgtkCore.check_weights(w_col))
    w = _fit_weights_storage(f, n)
    # VGC's weighted Kabsch (Σ wᵢbᵢaᵢᵀ); fewer than two pairs of positive weight leave a one-parameter family
    count(>(0), prior) >= 2 || return _reweighted_failure(one(RotMatrix{3, Float64}), optimization_strategy, f, n, w)
    R = _weighted_kabsch(as, bs, prior)
    r = _rotation_residuals(R, as, bs)
    tol_met = false
    iter = 0
    while iter < _fit_max_iter(optimization_strategy)
        iter += 1
        _irls_weights!(w, f, r)
        sw = _solve_weights(prior, w)
        count(>(0), sw) >= 2 || return _reweighted_failure(one(RotMatrix{3, Float64}), optimization_strategy, f, n, w, ReturnCode.InternalLinearSolveFailed, NoRule)
        R_new = _weighted_kabsch(as, bs, sw)
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
    used, rms = weighted_support_rms(sw, (abs2(x) for x in r))
    return FitResult(R, irls_retcode(tol_met, f === nothing), irls_stop_rule(tol_met, f === nothing), SolveStats(; nsteps = iter), rms, used,
                     FitMetadata{typeof(optimization_strategy)}(w, r, _fit_scale_record(f)...))
end

# Core's Kabsch of the weighted pairs (its `weights` column: Σ wᵢbᵢaᵢᵀ), as a RotMatrix — the one weighted Kabsch of
# the reweighted fits (the rotation's and the rigid alignment's).
_weighted_kabsch(as, bs, w) =
    RotMatrix{3, Float64}(solve(EstimationProblem(RotMatrix{3}, VgtkCore.Correspondences(as, bs; weights = w)), Kabsch()).u)

function _unit_direction(v, i)
    x = SVector{3, Float64}(v)
    nx = norm(x)
    (isfinite(nx) && nx > 0) || throw(ArgumentError("ReweightedLeastSquares (rotation): direction $i is zero or not finite"))
    return x / nx
end

_rotation_residuals(R, as, bs) = [angle_between(R * a, b) for (a, b) in zip(as, bs)]   # VGC's atan(‖Ra × b‖, Ra · b)


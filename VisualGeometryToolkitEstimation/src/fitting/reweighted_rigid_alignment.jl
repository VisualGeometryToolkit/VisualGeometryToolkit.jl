# =============================================================================
# ReweightedLeastSquares on an EuclideanMap{3} — the robust rigid transform between two point sets
# =============================================================================
#
# The rigid T = (R, t) with bᵢ ≈ R·aᵢ + t from point pairs `a => b` (a
# trajectory's camera centres against the ground truth's, for an absolute
# trajectory error), with some of the pairs gross outliers. The weighted
# Kabsch on weighted centroids (Horn 1987; Umeyama 1991 without the scale):
#
#   seed      the prior-weighted Kabsch of the centred points, t from the
#             weighted centroids — the classical ATE alignment;
#   repeat    the residuals rᵢ = ‖bᵢ − (R·aᵢ + t)‖ (codimension 3); the IRLS
#             weights of the `Irls{K}` strategy's loss at ONE loss width, set
#             once from the caller's scale input (irls_weights.jl); the
#             weighted Kabsch;
#   stop      when an iteration moves R by less than `tol` radians and t by
#             less than `tol · loss_width`, or at the strategy's `max_iter`.
#   optimization_strategy = nothing: the seed, one solve.
#
# Motivation (icra2026 "failure B"): five sparse, badly registered frames
# placed 40–176° and 0.9–2.4 m wrong tilted an all-centre Kabsch ATE alignment
# by 8–11°, which read as a 374 mm mount-phase ATE where the honest one was
# 30 mm. With a Huber or Welsch weighting at the walk's scale those centres
# get weight ≈ 0 and the alignment is the clean centres'.
# =============================================================================

"""
    solve(EstimationProblem(EuclideanMap{3}, cs), ReweightedLeastSquares(; optimization_strategy, tol); scale | width)
        -> FitResult{EuclideanMap{3}, FitMetadata}

The rigid transform `T` (`T(a) = R·a + t`) with `bᵢ ≈ T(aᵢ)` from point pairs
`cs` (`aᵢ => bᵢ`, 3-vectors): the alignment of an estimated trajectory's
positions onto the ground truth's for an absolute trajectory error (ATE), or
of any two point clouds in correspondence. A fit — it seeds itself with the
weighted Kabsch of the centred points (VisualGeometryToolkitCore's
[`Kabsch`](@extref VisualGeometryToolkitCore.Kabsch), `t` from the weighted
centroids) — iterating, when robust, a reweighted Kabsch at ONE loss width.

- `optimization_strategy` — `nothing` (the class default: the classical
  least-squares alignment, one solve), or an [`Irls`](@extref VisualGeometryToolkitCore.Irls)`{K}` strategy
  (its loss, its efficiency constant and its `max_iter`). A Triggs strategy is
  refused.
- `scale` | `width` (solve keywords) — with an `Irls` strategy, EXACTLY ONE of: `scale = σ`
  (the points' noise scale per coordinate, in their unit), `scale = result`
  (an upstream robust result's `noise_scale`), or `width = w` (the loss width
  on the residual distance). The width is `w`, or `c · σ` with `c` the
  strategy's efficiency constant, by default the shape's at codimension 3
  (`efficiency_constant(K, 3)`: Huber 1.628, Welsch 3.4691). Neither, both, or
  one given to the least-squares fit is an `ArgumentError`.
- `tol` — converged when an iteration moves `R` by less than `tol` radians and
  `t` by less than `tol` times the loss width (default configs/default.toml
  `[reweighted_least_squares.rigid_alignment] tol`).
- A `weights` column on `cs` (`withattrs(cs; weights)`) multiplies every pair's
  loss weight (fixed prior weights).

At least three pairs of positive weight are needed, not all on one line
(`ArgumentError`; a robust iteration whose weights leave fewer returns
`:degenerate`). `retcode_symbol(result)` is `:converged`, `:max_iterations`, `:solved` (least
squares) or `:degenerate`; `n` is the support (pairs of positive solve
weight), `rms` the weight-weighted root mean square distance — the ATE of the
pairs the weighting keeps. Physical points (lengths) are read in metres and
the transform comes back physical; the metadata are then in metres.
"""
function __solve(
        prob::EstimationProblem{<:EuclideanMap{3}, <:AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{3, <:Real}}}, NullParameters},
        alg::ReweightedLeastSquares; scale = nothing, width::Union{Nothing, Real} = nothing,
    )
    cs = prob.f
    who = "ReweightedLeastSquares (rigid alignment)"
    n = length(cs)
    optimization_strategy, tol = _reweighted_settings(alg, "reweighted_least_squares.rigid_alignment", who; strategy = false)
    f = _fit_irls(who, optimization_strategy, scale, width, 3)
    as = SVector{3, Float64}[SVector{3, Float64}(a) for a in sources(cs)]
    bs = SVector{3, Float64}[SVector{3, Float64}(b) for b in targets(cs)]
    (all(p -> all(isfinite, p), as) && all(p -> all(isfinite, p), bs)) ||
        throw(ArgumentError("$who: a point is not finite"))
    w_col = VgtkCore.weights_column(cs)
    prior = w_col === nothing ? ones(n) : Float64.(VgtkCore.check_weights(w_col))
    count(>(0), prior) >= 3 || throw(ArgumentError(
        "$who needs at least 3 pairs of positive weight, got $(count(>(0), prior))"))
    w = _fit_weights_storage(f, n)
    T = _weighted_rigid(as, bs, prior)
    T === nothing && throw(ArgumentError("$who: the points of positive weight are collinear"))
    r = _rigid_residuals(T, as, bs)
    tol_met = false
    iter = 0
    while iter < _fit_max_iter(optimization_strategy)
        iter += 1
        _irls_weights!(w, f, r)
        T_new = _weighted_rigid(as, bs, _solve_weights(prior, w))
        T_new === nothing && return _reweighted_failure(EuclideanMap(one(RotMatrix{3, Float64}), zero(SVector{3, Float64}); check = false),
                                                    optimization_strategy, f, n, w, irls_termination(false, false; linear_solve_failed = true))
        ΔR = Rotations.rotation_angle(RotMatrix{3, Float64}(rotation(T_new) * rotation(T)'))
        Δt = norm(translation(T_new) - translation(T))
        T = T_new
        r = _rigid_residuals(T, as, bs)
        if ΔR < tol && Δt < tol * f.loss_width
            tol_met = true
            break
        end
    end
    _irls_weights!(w, f, r)                           # the metadata at the returned transform
    sw = _solve_weights(prior, w)
    used, rms = weighted_support_rms(sw, (abs2(x) for x in r))
    return FitResult(T, irls_termination(tol_met, f === nothing)..., SolveStats(; nsteps = iter), rms, used,
                     FitMetadata{typeof(optimization_strategy)}(w, r, _fit_scale_record(f)...))
end

# Physical points: stripped in metres once (a `scale`/`width` given as a length too), the plain fit, the transform
# read back in metres once.
function __solve(
        prob::EstimationProblem{<:EuclideanMap{3}, <:AbstractVector{<:Correspondence{<:StaticVector{3, <:Quantity}, <:StaticVector{3, <:Quantity}}}, NullParameters},
        alg::ReweightedLeastSquares; scale = nothing, width = nothing,
    )
    cs = prob.f
    plain = [VgtkCore.strip_metres(a) => VgtkCore.strip_metres(b) for (a, b) in zip(sources(cs), targets(cs))]
    w_col = VgtkCore.weights_column(cs)
    plain = w_col === nothing ? plain : withattrs(VgtkCore.Correspondences(sources(plain), targets(plain)); weights = w_col)
    s = scale isa Union{Nothing, FitResult} ? scale : _metres(scale)
    return _attach_metres(__solve(EstimationProblem(EuclideanMap{3}, plain), alg; scale = s, width = _metres(width)))
end

function _weighted_rigid(as, bs, w)
    keep = findall(>(0), w)
    length(keep) >= 3 || return nothing
    sw = sum(w[keep])
    ca = sum(w[i] * as[i] for i in keep) / sw
    cb = sum(w[i] * bs[i] for i in keep) / sw
    # collinear (or coincident) centred sources leave a rotation about their line free
    S = sum(w[i] * (as[i] - ca) * (as[i] - ca)' for i in keep)
    λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(SMatrix{3, 3, Float64, 9}(S)))
    λ[2] > 1.0e-12 * max(λ[3], eps()) || return nothing
    R = _weighted_kabsch([as[i] - ca for i in keep], [bs[i] - cb for i in keep], w[keep])
    return EuclideanMap(R, cb - R * ca; check = false)
end

_rigid_residuals(T, as, bs) = [norm(b - (rotation(T) * a + translation(T))) for (a, b) in zip(as, bs)]


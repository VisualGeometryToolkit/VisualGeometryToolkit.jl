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
#             tangent residual, so codimension 2); the noise scale σ̂ from
#             their median (`median(r) / √chisq_median(2)`, the Rayleigh
#             median — the scale of a 2-D residual's norm, VGC's
#             `chisq_median`); the loss at width `scale_multiple · σ̂`
#             (`deploy`, default multiple the shape's efficiency constant at
#             codimension 2); the loss's IRLS weights; the weighted Kabsch.
#   stop      when an iteration moves R by less than `tol` (radians), or at
#             `max_iter`.
#
# Hoisted from RadiometricCamera's experiments/lib/stars.jl
# (`fit_rotation_irls`, `wahba`), 2026-10-27, with two corrections: the scale
# of a 2-D residual's norm is its median over 1.1774 (Rayleigh), not 1.4826 ×
# median (a 1-D absolute deviation's, which reads 2-D norms 1.745× too wide);
# and the loss and its width multiple are the stack's (`AbstractLossShape`,
# `scale_multiple`) instead of a hard-wired Huber at 3 × that scale. RC's
# exact width is `scale_multiple = 3 · 1.4826 · 1.1774 ≈ 5.237`.
# =============================================================================

"""
    RotationFitMetadata

`FitResult` metadata of [`fit_rotation`](@ref), all at the returned rotation:
`weights`, the loss's IRLS weight of every pair (times its prior weight);
`residuals`, the angle `∠(R·aᵢ, bᵢ)` of every pair in radians; `sigma`, the
robust noise scale σ̂ (radians) the final weights were computed at. They
forward as properties: `result.weights`, `result.residuals`, `result.sigma`.
"""
struct RotationFitMetadata
    weights::Vector{Float64}
    residuals::Vector{Float64}
    sigma::Float64
end

"""
    fit_rotation(cs; loss, scale_multiple = nothing, max_iter, tol)
        -> FitResult{RotMatrix{3, Float64}, RotationFitMetadata}

The robust rotation `R` with `bᵢ ≈ R·aᵢ` from direction pairs `cs` (`aᵢ => bᵢ`,
3-vectors; normalized at entry, so any nonzero length is a direction), with a
fraction of the pairs gross outliers. A fit — it seeds itself with
VisualGeometryToolkitCore's [`kabsch`](@ref) — iterating a reweighted Kabsch:
residual angles `rᵢ = ∠(R·aᵢ, bᵢ)`, the robust scale
`σ̂ = median(r) / √chisq_median(2)` (the median of a 2-D residual norm), the
weights of `loss` (an `AbstractLossShape`, e.g. `HuberShape()`) at width
`scale_multiple · σ̂`, and the weighted Kabsch.

- `loss` — the robust loss shape (default the TOML's, Huber).
- `scale_multiple` — the width as a multiple of σ̂; `nothing` takes the shape's
  efficiency constant at codimension 2 (`deploy`'s default).
- `max_iter`, `tol` — at most `max_iter` iterations; converged when one moves
  `R` by less than `tol` radians (defaults: configs/default.toml [fit_rotation]).
- A `weights` column on `cs` (`withattrs(cs; weights)`) multiplies every pair's
  loss weight (fixed prior weights).

`retcode` is `:converged`, `:max_iterations`, or `:degenerate` when fewer than
two pairs keep a positive weight; on exact data (every residual zero) the seed
is returned, `:converged` after one iteration. `rms` is the root mean square
residual angle over the `n` pairs (outliers included); the metadata carries
the weights, the residual angles and σ̂ at the returned rotation.

Accuracy: with a minority of gross outliers the estimate reaches the noise
floor of the inliers' own least-squares rotation (≈ σ/√m for m inliers at
direction noise σ); the suite pins 20% random outliers among 200 pairs at
σ = 1e-3 rad to within 5e-4 rad, where the plain Kabsch is off by over 1e-2.
"""
function fit_rotation(
        cs::AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{3, <:Real}}};
        loss::VgtkCore.AbstractLossShape = LOSS_TAGS[_dflt("fit_rotation", "loss")](),
        scale_multiple::Union{Nothing, Real} = nothing,
        max_iter::Integer = _dflt("fit_rotation", "max_iter"),
        tol::Real = _dflt("fit_rotation", "tol"),
    )
    n = length(cs)
    n >= 2 || throw(ArgumentError("fit_rotation needs at least 2 direction pairs, got $n"))
    as = SVector{3, Float64}[_unit_direction(a, i) for (i, a) in enumerate(sources(cs))]
    bs = SVector{3, Float64}[_unit_direction(b, i) for (i, b) in enumerate(targets(cs))]
    w_col = VgtkCore.weights_column(cs)
    prior = w_col === nothing ? ones(n) : Float64.(VgtkCore.check_weights(w_col))
    R = _weighted_kabsch(as, bs, prior)
    R === nothing && return _rotation_degenerate(n, prior)
    r = _rotation_residuals(R, as, bs)
    w = copy(prior)
    σ = 0.0
    retcode = :max_iterations
    iter = 0
    while iter < max_iter
        iter += 1
        σ = Statistics.median(r) / sqrt(VgtkCore.chisq_median(2))
        if iszero(σ)                       # an exact fit: every weight is the prior's, R stands
            w .= prior
            retcode = :converged
            break
        end
        L = isnothing(scale_multiple) ? deploy(loss, σ^2; codim = 2) : deploy(loss, σ^2; codim = 2, scale_multiple)
        c² = VgtkCore.sq_loss_width(L)
        @inbounds for i in 1:n
            w[i] = prior[i] * weight(loss, r[i]^2, c²)
        end
        R_new = _weighted_kabsch(as, bs, w)
        R_new === nothing && return _rotation_degenerate(n, w)
        Δ = Rotations.rotation_angle(R_new * R')
        R = R_new
        r = _rotation_residuals(R, as, bs)
        if Δ < tol
            retcode = :converged
            break
        end
    end
    return FitResult(R, retcode, iter, sqrt(sum(abs2, r) / n), n, RotationFitMetadata(w, r, σ))
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

_rotation_residuals(R, as, bs) = [atan(norm(cross(R * a, b)), dot(R * a, b)) for (a, b) in zip(as, bs)]

_rotation_degenerate(n, w) =
    FitResult(one(RotMatrix{3, Float64}), :degenerate, 0, NaN, n, RotationFitMetadata(collect(Float64, w), fill(NaN, n), NaN))

# =============================================================================
# fit_rotation_averaging — robust rotation averaging on a view graph
# =============================================================================
#
# The absolute rotations of a set of frames from their pairwise relative
# rotations (the edges of a view graph, some of them wrong), after Chatterjee &
# Govindu (ICCV 2013, TPAMI 2018): IRLS in the Lie algebra.
#
#   seed      VisualGeometryToolkitCore's `spectral_rotation_averaging` (every edge).
#   repeat    per edge (i, j) the residual rotation E_ij = R_j·R_ij·R_iᵀ (the
#             identity when the edge agrees) and its log r_ij (radians, a
#             3-vector, codimension 3); the robust scale σ̂ = median‖r‖ /
#             √chisq_median(3); the loss's IRLS weights at width
#             `scale_multiple · σ̂`; the weighted sparse linear least squares
#             of the first-order update ω_j − ω_i = −r_ij over every edge,
#             the gauge frame's ω = 0; R_k ← exp(ω_k)·R_k.
#   stop      when an update moves every frame by less than `tol` radians, or at
#             `max_iter`.
# =============================================================================

"""
    RotationAveragingMetadata{K}

`FitResult` metadata of [`fit_rotation_averaging`](@ref) at the returned
rotations: per edge `(i, j)`, `residuals` the angle (radians) of the residual
rotation `R_j·R_ij·R_iᵀ` and `weights` the loss's IRLS weight times the edge's
prior weight; `sigma` the robust scale σ̂ (radians). They forward as properties.
"""
struct RotationAveragingMetadata{K}
    residuals::Dict{Tuple{K, K}, Float64}
    weights::Dict{Tuple{K, K}, Float64}
    sigma::Float64
end

"""
    fit_rotation_averaging(relative[, prior]; loss, scale_multiple = nothing, max_iter, tol)
        -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationAveragingMetadata{K}}

Robust rotation averaging on a view graph: the absolute rotations `R_k` of the
frames (`d = R_k·f_k`, every frame's directions into the common frame, the
first of the sorted frames the identity — the gauge) from the relative
rotations `relative`, an `AbstractDict` keyed by frame pairs `(i, j)` whose value
`R_ij` takes frame `i`'s directions into frame `j`'s (`f_j = R_ij·f_i`, so
`R_i = R_j·R_ij`), some of them wrong. A fit: it seeds itself with
VisualGeometryToolkitCore's `spectral_rotation_averaging` and iterates a reweighted
first-order Lie-algebra least squares over every edge (Chatterjee & Govindu
2013), each a sparse linear solve.

- `prior` — fixed non-negative edge weights (an `AbstractDict` with the same
  keys: inlier counts); all ones by default.
- `loss`, `scale_multiple` — the robust loss shape (default the TOML's) and its
  width as a multiple of σ̂ (the median of the residual angles over
  `√chisq_median(3)`); `nothing` takes the shape's efficiency constant at
  codimension 3, which only an M-estimator has.
- `max_iter`, `tol` — at most `max_iter` iterations; converged when one moves
  every frame by less than `tol` radians (configs/default.toml
  [fit_rotation_averaging]).

`retcode` is `:converged` or `:max_iterations`; `rms` the root mean square
residual angle over the edges. A disconnected graph is an `ArgumentError`
(VisualGeometryToolkitCore's seed refuses it).
"""
function fit_rotation_averaging(
        relative::AbstractDict{<:Tuple{K, K}}, prior::Union{Nothing, AbstractDict} = nothing;
        loss::VgtkCore.AbstractLossShape = LOSS_TAGS[_dflt("fit_rotation_averaging", "loss")](),
        scale_multiple::Union{Nothing, Real} = nothing,
        max_iter::Integer = _dflt("fit_rotation_averaging", "max_iter"),
        tol::Real = _dflt("fit_rotation_averaging", "tol"),
    ) where {K}
    loss isa VgtkCore.AbstractMEstimator || scale_multiple !== nothing || throw(ArgumentError(
        "fit_rotation_averaging: the loss $(nameof(typeof(loss))) has no efficiency default; give its width as `scale_multiple`"))
    edges = sort!(collect(keys(relative)))
    p = [prior === nothing ? 1.0 : Float64(prior[e]) for e in edges]
    all(>=(0), p) || throw(ArgumentError("fit_rotation_averaging: a prior edge weight is negative"))
    Rrel = [SMatrix{3, 3, Float64, 9}(relative[e]) for e in edges]
    seed = VgtkCore.spectral_rotation_averaging(relative, prior)
    frames = sort!(collect(keys(seed)))
    col = Dict(k => 3 * (a - 2) for (a, k) in enumerate(frames))       # -3: the gauge frame
    R = Dict(k => SMatrix{3, 3, Float64, 9}(seed[k]) for k in frames)
    m = length(edges); r = Vector{SVector{3, Float64}}(undef, m); w = copy(p)
    residuals!() = for (q, (i, j)) in enumerate(edges)
        r[q] = log_so3(R[j] * Rrel[q] * R[i]')
    end
    residuals!()
    retcode = :max_iterations; iter = 0; σ = NaN
    while iter < max_iter
        iter += 1
        σ = _averaging_weights!(w, r, p, loss, scale_multiple)
        iszero(σ) && (retcode = :converged; break)                     # every edge agrees
        ω = _averaging_update(edges, col, r, w, 3 * (length(frames) - 1))
        Δ = 0.0
        for k in frames
            c = col[k]; c < 0 && continue
            δ = SVector(ω[c + 1], ω[c + 2], ω[c + 3])
            R[k] = SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(δ)) * R[k]
            Δ = max(Δ, norm(δ))
        end
        residuals!()
        Δ < tol && (retcode = :converged; break)
    end
    σ = _averaging_weights!(w, r, p, loss, scale_multiple)
    out = Dict(k => RotMatrix{3, Float64, 9}(Rotations.nearest_rotation(R[k])) for k in frames)
    meta = RotationAveragingMetadata{K}(Dict(e => norm(r[q]) for (q, e) in enumerate(edges)), Dict(e => w[q] for (q, e) in enumerate(edges)), σ)
    return FitResult(out, retcode, iter, sqrt(sum(norm(x)^2 for x in r) / m), m, meta)
end

# σ̂ from the residual angles of the edges of positive prior weight (the χ²₃
# median), and the loss's IRLS weights times the prior into `w`; σ̂ = 0 leaves
# the prior. Returns σ̂.
function _averaging_weights!(w, r, prior, loss, scale_multiple)
    σ = Statistics.median(norm(r[q]) for q in eachindex(r) if prior[q] > 0) / sqrt(VgtkCore.chisq_median(3))
    if iszero(σ)
        w .= prior
        return σ
    end
    c² = VgtkCore.sq_loss_width(_averaging_loss(loss, σ^2, scale_multiple))
    for q in eachindex(w)
        w[q] = prior[q] * weight(loss, norm(r[q])^2, c²)
    end
    return σ
end

# The loss at the noise scale s2 (codimension 3), as `fit_rotation`'s at 2.
_averaging_loss(loss::VgtkCore.AbstractMEstimator, s2, ::Nothing) = deploy(loss, s2; codim = 3)
_averaging_loss(loss::VgtkCore.AbstractMEstimator, s2, scale_multiple::Real) = deploy(loss, s2; codim = 3, scale_multiple)
_averaging_loss(loss::VgtkCore.AbstractLossShape, s2, scale_multiple::Real) = deploy(loss, s2; scale_multiple)

# The weighted sparse least squares of ω_j − ω_i = −r_ij over the edges (3 rows
# each, the gauge frame's columns absent): its normal equations, sparse.
function _averaging_update(edges, col, r, w, nunk)
    Is = Int[]; Js = Int[]; Vs = Float64[]; b = zeros(3 * length(edges))
    for (q, (i, j)) in enumerate(edges)
        s = sqrt(w[q])
        for a in 1:3
            row = 3 * (q - 1) + a
            col[j] >= 0 && (push!(Is, row); push!(Js, col[j] + a); push!(Vs, s))
            col[i] >= 0 && (push!(Is, row); push!(Js, col[i] + a); push!(Vs, -s))
            b[row] = -s * r[q][a]
        end
    end
    A = sparse(Is, Js, Vs, 3 * length(edges), nunk)
    return Vector((A' * A) \ (A' * b))
end

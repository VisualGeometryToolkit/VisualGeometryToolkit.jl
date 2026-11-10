# =============================================================================
# Generalized cross-validation of a penalty weight, and the weight's normalization
# =============================================================================
#
# The stack's one GCV (Golub, Heath & Wahba 1979) and one penalty-weight convention.
# A penalized least squares `min_y ‖J y − r‖² + λ ‖D y − c‖²` (n data rows) has the
# influence matrix `H_λ = J (JᵀJ + λ DᵀD)⁻¹ Jᵀ` of the data on the fit and
# `GCV(λ) = n ‖(I − H_λ) r‖² / (n − tr H_λ)²`.
#
# THE WEIGHT. `λ = β · penalty_scale(energy, N, m̄)`, `penalty_scale = energy / N / m̄`:
# β relative to the data's energy per parameter (`energy / N`) and to the penalty
# operator's mean positive eigenvalue `m̄` of `DᵀD` (`mean_positive_eigenvalue(D)`), so β
# means the same for every operator, lattice and region; β = 1 weighs the penalty like
# the data. The energy is the caller's: a linear solve's data energy (RadiometricCamera's
# Σϕ²), or a nonlinear refine's Gauss–Newton energy `tr JᵀJ` over its parameters (the
# residuals of exact data vanish, so they cannot set a scale).
#
# THE SEARCH. `gcv_select(score, scale)`: `score(λ)` returns `(; gcv, rss, dof)` — a
# closed-form curve of one Gauss–Newton system (`gcv_curve`), or a caller's own
# evaluation (a nonlinear refine solved at each λ) — minimized over log₁₀ β by Core's
# grid-and-golden minimizer, the scan evaluated from the largest β down.
# =============================================================================

"""
    mean_positive_eigenvalue(D) -> Float64
    mean_positive_eigenvalue(Symmetric(DᵀD)) -> Float64

The mean of the positive eigenvalues of a penalty operator's normal matrix `DᵀD`: the
trace over the rank, from the operator `D` itself (`‖D‖²_F / rank D`; the rank by a
sparse QR for a sparse `D`, so a lattice of 10⁴ nodes needs no dense eigendecomposition),
or from the normal matrix given as `Symmetric` (`tr G / rank G`). The two forms agree; a
plain matrix is read as the operator, so the normal matrix must come wrapped. The `m̄` of
the penalty weight's normalization ([`penalty_scale`](@ref)).
"""
mean_positive_eigenvalue(D::AbstractMatrix) = sum(abs2, D) / _rank(D)
mean_positive_eigenvalue(G::Symmetric) = LinearAlgebra.tr(G) / LinearAlgebra.rank(Matrix{Float64}(G))
_rank(D::SparseMatrixCSC) = LinearAlgebra.rank(LinearAlgebra.qr(SparseMatrixCSC{Float64, Int}(D)))
_rank(D::AbstractMatrix) = LinearAlgebra.rank(Matrix{Float64}(D))

"""
    penalty_scale(energy, N, m̄) -> Float64

The unit of a penalty weight: `λ = β · energy / N / m̄`, β relative to the data's
`energy` per parameter (`N` parameters) and to the penalty operator's mean positive
eigenvalue `m̄` ([`mean_positive_eigenvalue`](@ref)). β = 1 weighs the penalty like the data.
"""
penalty_scale(energy::Real, N::Integer, m̄::Real) = Float64(energy) / N / Float64(m̄)

"""
    check_penalty_weight(w, what; allow_nothing = false) -> w

A penalty weight as an option states it: a finite β ≥ 0, or `:gcv` (chosen by
[`gcv_select`](@ref)); with `allow_nothing`, also `nothing` (no penalty). An
`ArgumentError` naming `what` otherwise.
"""
function check_penalty_weight(w, what::AbstractString; allow_nothing::Bool = false)
    (w === :gcv || (allow_nothing && w === nothing)) && return w
    w isa Real && isfinite(w) && w >= 0 && return w
    throw(ArgumentError("$what = $(repr(w)) is not a finite weight β ≥ 0$(allow_nothing ? ", nothing," : "") or :gcv"))
end

"""
    gcv_score(n, rss, dof) -> Float64

The generalized cross-validation score `n·rss / (n − dof)²` of a fit with `n` data rows,
residual sum of squares `rss` and `dof = tr H` effective parameters.
"""
gcv_score(n::Integer, rss::Real, dof::Real) = n * Float64(rss) / (n - Float64(dof))^2

"""
    gcv_curve(G, h, rr, Q, q, n) -> (λ -> (; gcv, rss, dof))

GCV of the penalized least squares with the normal-equation pieces `G = JᵀJ`, `h = Jᵀr`,
`rr = ‖r‖²`, `Q = DᵀD`, `q = Dᵀc` and `n` data rows, as a function of λ: one Cholesky
`B = G + Q = L Lᵀ` and one symmetric eigendecomposition `L⁻¹ Q L⁻ᵀ = U diag(μ) Uᵀ`
(`V = L⁻ᵀ U`: `VᵀBV = I`, `VᵀQV = diag μ`, `VᵀGV = I − diag μ`) give, with
`sᵢ = 1 − μᵢ + λ μᵢ` and `w = (Vᵀh + λ Vᵀq) ./ s` (the solution `y = V w`),
`rss = rr − 2 w⋅Vᵀh + Σ wᵢ² (1 − μᵢ)` and `dof = tr H_λ = Σ (1 − μᵢ)/sᵢ`. Parameters that
neither the data nor the penalty reads (zero diagonal in both) are dropped; a ridge of
1e-12 × the mean diagonal of `B` keeps the Cholesky defined. Dense: `O(p³)` in the
parameters. (Moved from RadiometricCamera's linear solves.)
"""
function gcv_curve(G::AbstractMatrix, h::AbstractVector, rr::Real, Q::AbstractMatrix, q::AbstractVector, n::Integer)
    keep = [j for j in axes(G, 1) if G[j, j] > 0 || Q[j, j] > 0]
    Gk = convert(Matrix{Float64}, G[keep, keep]); Qk = convert(Matrix{Float64}, Q[keep, keep])
    B = Gk .+= Qk
    ε = 1.0e-12 * LinearAlgebra.tr(B) / max(size(B, 1), 1)
    for j in axes(B, 1)
        B[j, j] += ε
    end
    F = LinearAlgebra.cholesky!(Symmetric(B))
    μ, U = LinearAlgebra.eigen!(Symmetric(LinearAlgebra.rdiv!(LinearAlgebra.ldiv!(F.L, Qk), F.U)))
    μ = clamp.(μ, 0.0, 1.0)
    V = F.U \ U
    zh = V' * Vector{Float64}(h[keep]); zq = V' * Vector{Float64}(q[keep])
    return function (λ::Real)
        s = 1 .- μ .+ λ .* μ
        w = (zh .+ λ .* zq) ./ s
        rss = max(rr - 2 * dot(w, zh) + sum(w .^ 2 .* (1 .- μ)), 0.0)
        dof = sum((1 .- μ) ./ s)
        return (; gcv = gcv_score(n, rss, dof), rss, dof)
    end
end

"""
    gcv_select(score, scale; log10_bracket = (-8.0, 4.0), points = 25, xtol = 1e-3, refine_ends = true)
        -> (; β, λ, gcv, rss, dof, log10_β_bracket, at_bracket_end)

The penalty weight β minimizing GCV, `score(λ)` returning `(; gcv, rss, dof)` at
`λ = β · scale` ([`penalty_scale`](@ref)) — a [`gcv_curve`](@ref), or a caller's own
evaluation: a scan of `points` values of log₁₀ β, evenly spaced over `log10_bracket` and
evaluated from the largest down (a caller's evaluation may warm-start each from the
last), then golden-section steps between the best scan point's neighbours until the
bracket is `2·xtol` wide (VisualGeometryToolkitCore's `grid_golden_minimum`). GCV need not
be unimodal over many decades: the scan picks the basin. A least value at a scan end is
refined on its one neighbouring interval when `refine_ends` (else returned as it is, a
caller whose every evaluation is costly may skip it); `at_bracket_end` says the result
lies at that end (the bracket did not contain the minimum; never silently taken as
interior). `gcv` is the least value evaluated, `dof` the effective number of parameters
`tr H_λ` there. A score of `Inf` (a weight the caller could not evaluate) is never the
minimum.
"""
function gcv_select(score, scale::Real; log10_bracket::Tuple{Real, Real} = (-8.0, 4.0), points::Integer = 25, xtol::Real = 1.0e-3,
        refine_ends::Bool = true)
    lo, hi = Float64.(log10_bracket)
    ts = range(hi, lo; length = points)
    seen = Dict{Float64, NamedTuple{(:gcv, :rss, :dof), NTuple{3, Float64}}}()
    g(t) = (c = get!(() -> _gcv_record(score(exp10(t) * scale)), seen, t); c.gcv)
    iters = max(0, ceil(Int, log(xtol / abs(step(ts))) / log((sqrt(5) - 1) / 2)))
    found = VgtkCore.grid_golden_minimum(g, ts; iters, refine_ends)
    c = seen[found.x]
    return (; β = exp10(found.x), λ = exp10(found.x) * scale, c.gcv, c.rss, c.dof, log10_β_bracket = [lo, hi],
        at_bracket_end = found.at_end)
end
_gcv_record(c) = (; gcv = Float64(c.gcv), rss = Float64(c.rss), dof = Float64(c.dof))

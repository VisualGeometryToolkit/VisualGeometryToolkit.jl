# =============================================================================
# Generalized cross-validation of a penalty weight
# =============================================================================
#
# The stack's one GCV (Golub, Heath & Wahba 1979) and one penalty-weight convention.
# A penalized least squares `min_y ‖J y − r‖² + λ ‖D y − c‖²` (n data rows) has the
# influence matrix `H_λ = J (JᵀJ + λ DᵀD)⁻¹ Jᵀ` of the data on the fit and
# `GCV(λ) = n ‖(I − H_λ) r‖² / (n − tr H_λ)²`.
#
# THE WEIGHT (Core's `penalty_scale` and `mean_positive_eigenvalue`, numerics/penalized_least_squares.jl).
# `λ = β · penalty_scale(energy, N, m̄)`, `penalty_scale = energy / N / m̄`:
# β relative to the data's energy per parameter (`energy / N`) and to the penalty
# operator's mean positive eigenvalue `m̄` of `DᵀD` (`mean_positive_eigenvalue(D)`), so β
# means the same for every operator, lattice and region; β = 1 weighs the penalty like
# the data. The energy is the caller's: a linear solve's data energy (RadiometricCamera's
# Σϕ²), or a nonlinear refine's Gauss–Newton energy `tr JᵀJ` over its parameters (the
# residuals of exact data vanish, so they cannot set a scale).
#
# THE SEARCH. `gcv_select(score, scale)`: `score(λ)` returns `(; gcv, rss, dof)` — a
# closed-form curve of one Gauss–Newton system (`gcv_curve`), or a caller's own
# evaluation — minimized over log₁₀ β by a coarse scan and Brent's method (Core's
# `grid_brent_minimum`), the scan evaluated from the largest β down.
# =============================================================================

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
    gcv_curve(G, h, rr, Q, q, n) -> (λ -> (; gcv, rss, dof, y))

GCV of the penalized least squares `min_y ‖J y − r‖² + λ‖D y − c‖²` with the
normal-equation pieces `G = JᵀJ`, `h = Jᵀr`, `rr = ‖r‖²`, `Q = DᵀD`, `q = Dᵀc` and `n` data
rows, as a function of λ. One Cholesky `B = G + Q = L Lᵀ` and one symmetric
eigendecomposition `L⁻¹ Q L⁻ᵀ = U diag(μ) Uᵀ` — the generalized eigenproblem of the pencil
`(Q, B)`, `V = L⁻ᵀ U`: `VᵀBV = I`, `VᵀQV = diag μ` — diagonalize every `M_λ = G + λQ` at
once, `VᵀM_λV = diag s`, `sᵢ = 1 − μᵢ + λμᵢ`. With `zh = Vᵀh`, `zq = Vᵀq` and
`w = (zh + λ zq) ./ s` (the solution `y = V w`, returned: a nonlinear fit's Gauss–Newton
step at the chosen λ): `rss = rr − 2 w⋅zh + Σ (1 − μᵢ) wᵢ²`, `dof = tr H_λ = Σ (1 − μᵢ)/sᵢ`.
Parameters nothing reads (zero diagonal in `B`) are dropped (0 in `y`). `B` must then
be positive definite: a direction no row holds is a gauge, which its caller holds
exactly (Core's rule: no penalty or ridge fixes it); an `ArgumentError` otherwise.
Dense, `O(p³)` once in the parameters, then `O(p²)` per λ. (Moved from
RadiometricCamera's linear solves.)

    gcv_curve(J, r, D, c; exact_max = 3000, probes = 64, seed) -> (λ -> (; gcv, rss, dof, y))

The same from the rows themselves (`J`, `D` dense or sparse): the pencil above for at
most `exact_max` parameters, the data residual `r − J y` read from the rows (exact where
it is small: the normal-equation form's `rr − 2w⋅zh + …` cancels); past it, per λ one
sparse Cholesky of `M_λ` (CHOLMOD, its symbolic analysis done once: the pattern does not
depend on λ) for the solution and a Hutchinson estimate of `tr H_λ = tr(M_λ⁻¹JᵀJ)` from
`probes` Rademacher vectors drawn from `seed` (deterministic; the probes read the
symmetric `zᵀJM_λ⁻¹Jᵀz`, `H`'s eigenvalues in [0, 1], so its standard deviation is
`≤ √(2 tr H / probes)`). Either way the parameters nothing reads are dropped (0 in `y`).
"""
function gcv_curve(G::AbstractMatrix, h::AbstractVector, rr::Real, Q::AbstractMatrix, q::AbstractVector, n::Integer)
    P = _gcv_pencil(G, Q)
    return _pencil_curve(P, h, q, n, size(G, 1), (w, zh, yk) -> max(rr - 2 * dot(w, zh) + dot(w, P.k .* w), 0.0))
end
function gcv_curve(J::AbstractMatrix, r::AbstractVector, D::AbstractMatrix, c::AbstractVector;
        exact_max::Integer = 3000, probes::Integer = 64, seed::Integer = 20261110)
    n, p = size(J)
    Js = convert(SparseMatrixCSC{Float64, Int}, J); Ds = convert(SparseMatrixCSC{Float64, Int}, D)
    G = Js' * Js; Q = Ds' * Ds
    h = Js' * r; q = Ds' * c
    if p <= exact_max
        # the pencil, and the data residual read from the rows (`r − J y`, no cancellation where it is small)
        P = _gcv_pencil(G, Q)
        Jk = Js[:, P.keep]
        return _pencil_curve(P, h, q, n, p, (w, zh, yk) -> sum(abs2, r - Jk * yk))
    end
    keep = _read_parameters(G, Q)
    Gk = G[keep, keep]; Qk = Q[keep, keep]; hk = h[keep]; qk = q[keep]; Jk = Js[:, keep]
    M = Gk + Qk                                          # the pattern every M_λ shares
    Gp = _on_pattern(Gk, M); Qp = _on_pattern(Qk, M)
    F = cholesky(Symmetric(M); check = false)            # its symbolic analysis, reused at every λ
    W = Jk' * rand(Random.Xoshiro(seed), (-1.0, 1.0), n, probes)
    return function (λ::Real)
        M.nzval .= Gp.nzval .+ λ .* Qp.nzval
        cholesky!(F, Symmetric(M); check = false)
        issuccess(F) || throw(ArgumentError("gcv_curve: J'J + λD'D is not positive definite at λ = $λ: a gauge is held exactly by the caller, never by a ridge"))
        yk = F \ (hk + λ * qk)
        rss = sum(abs2, r - Jk * yk)
        dof = sum(W .* (F \ W)) / probes
        return (; gcv = gcv_score(n, rss, dof), rss, dof, y = _scatter(p, keep, yk))
    end
end
_scatter(p::Integer, keep::AbstractVector{Int}, yk::AbstractVector) = (y = zeros(p); y[keep] = yk; y)
# The parameters something reads: a positive diagonal of `G + Q`.
_read_parameters(G::AbstractMatrix, Q::AbstractMatrix) = [j for j in axes(G, 1) if G[j, j] + Q[j, j] > 0]
# The sparse matrix `A` stored on the pattern of `M` (which holds `A`'s).
function _on_pattern(A::SparseMatrixCSC, M::SparseMatrixCSC)
    out = copy(M); fill!(out.nzval, 0.0)
    for j in axes(A, 2), k in nzrange(A, j)
        r = rowvals(M); rg = nzrange(M, j)
        out.nzval[rg[searchsortedfirst(view(r, rg), rowvals(A)[k])]] = nonzeros(A)[k]
    end
    return out
end
# The pencil's curve over λ: `w = (zh + λzq)./s` in its coordinates, `y = V w` scattered
# over the parameters, `rss_of(w, zh, yk)` the data residual's sum of squares.
function _pencil_curve(P, h::AbstractVector, q::AbstractVector, n::Integer, p::Integer, rss_of)
    zh = P.V' * Vector{Float64}(h[P.keep]); zq = P.V' * Vector{Float64}(q[P.keep])
    return function (λ::Real)
        s = 1 .- P.μ .+ λ .* P.μ
        w = (zh .+ λ .* zq) ./ s
        yk = P.V * w
        rss = rss_of(w, zh, yk)
        dof = sum(P.k ./ s)
        return (; gcv = gcv_score(n, rss, dof), rss, dof, y = _scatter(p, P.keep, yk))
    end
end

# The generalized eigendecomposition of the pencil `(Q, B)`, `B = G + Q` over the
# parameters something reads (`keep`, a positive diagonal of `B`): one Cholesky `B = L Lᵀ`
# and one symmetric eigendecomposition `L⁻¹ Q L⁻ᵀ = U diag(μ) Uᵀ`, `V = L⁻ᵀ U` (`VᵀBV = I`,
# `VᵀQV = diag μ`); `k = 1 − μ`, the diagonal of `VᵀGV`. A `B` that is not positive
# definite is a gauge: an `ArgumentError`, never a ridge.
function _gcv_pencil(G::AbstractMatrix, Q::AbstractMatrix)
    keep = _read_parameters(G, Q)
    Qk = Matrix{Float64}(Q[keep, keep])
    F = LinearAlgebra.cholesky!(Symmetric(Matrix{Float64}(G[keep, keep]) + Qk); check = false)
    issuccess(F) || throw(ArgumentError("gcv_curve: J'J + D'D is not positive definite: a direction neither the data nor the penalty reads (a gauge) is held exactly by the caller, never by a ridge"))
    μ, U = LinearAlgebra.eigen!(Symmetric(LinearAlgebra.rdiv!(LinearAlgebra.ldiv!(F.L, Qk), F.U)))
    μ = clamp.(μ, 0.0, 1.0)
    V = F.U \ U
    return (; keep, μ, V, k = 1 .- μ)
end

"""
    gcv_select(score, scale; log10_bracket = (-8.0, 4.0), points = 25)
        -> (; β, λ, gcv, rss, dof, log10_β_bracket, at_bracket_end)

The penalty weight β minimizing GCV, `score(λ)` returning `(; gcv, rss, dof)` at
`λ = β · scale` (Core's `penalty_scale`) — a [`gcv_curve`](@ref), or a caller's own
evaluation: a coarse scan of `points` values of log₁₀ β, evenly spaced over
`log10_bracket` and evaluated from the largest down, then Brent's method on the scan's
interval around its best point, to Brent's own floor (Core's `grid_brent_minimum`; no
tolerance to choose: each evaluation of a curve is closed form). GCV need not be
unimodal over many decades: the scan picks the basin. A least value at a scan end is
refined on its one neighbouring interval, and `at_bracket_end` says the result lies at
that end (the bracket did not contain the minimum; never silently taken as interior).
`gcv` is the least value evaluated, `dof` the effective number of parameters `tr H_λ`
there. A score of `Inf` (a weight the caller could not evaluate) is never the minimum.
(RadiometricCamera's scan-and-Brent search, adopted.)
"""
function gcv_select(score, scale::Real; log10_bracket::Tuple{Real, Real} = (-8.0, 4.0), points::Integer = 25)
    lo, hi = Float64.(log10_bracket)
    seen = Dict{Float64, NamedTuple{(:gcv, :rss, :dof), NTuple{3, Float64}}}()
    g(t) = (c = get!(() -> _gcv_record(score(exp10(t) * scale)), seen, t); c.gcv)
    found = VgtkCore.grid_brent_minimum(g, range(hi, lo; length = points))
    c = seen[found.x]
    return (; β = exp10(found.x), λ = exp10(found.x) * scale, c.gcv, c.rss, c.dof, log10_β_bracket = [lo, hi],
        at_bracket_end = found.at_end)
end
_gcv_record(c) = (; gcv = Float64(c.gcv), rss = Float64(c.rss), dof = Float64(c.dof))

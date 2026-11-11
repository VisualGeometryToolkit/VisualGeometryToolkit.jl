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
    gcv_curve(G, h, rr, Q, q, n; hold = nothing) -> (λ -> (; gcv, rss, dof, y))

GCV of the penalized least squares `min_y ‖J y − r‖² + λ‖D y − c‖² + ‖R y‖²` with the
normal-equation pieces `G = JᵀJ`, `h = Jᵀr`, `rr = ‖r‖²`, `Q = DᵀD`, `q = Dᵀc` and `n` data
rows, as a function of λ. `hold = RᵀR` is a λ-independent term that is not data (a
nonlinear solve's rows holding coordinates no other row reads, zero at the linearization
point): it enters the system, not `tr H`. One Cholesky `B = G + RᵀR + Q = L Lᵀ` and one
symmetric eigendecomposition `L⁻¹ Q L⁻ᵀ = U diag(μ) Uᵀ` — the generalized eigenproblem of
the pencil `(Q, B)`, `V = L⁻ᵀ U`: `VᵀBV = I`, `VᵀQV = diag μ` — diagonalize every
`M_λ = G + RᵀR + λQ` at once, `VᵀM_λV = diag s`, `sᵢ = 1 − μᵢ + λμᵢ`. With `K = VᵀGV`
(`I − diag μ` without a hold), `zh = Vᵀh`, `zq = Vᵀq` and `w = (zh + λ zq) ./ s` (the
solution `y = V w`, returned: a nonlinear fit's Gauss–Newton step at the chosen λ): `rss =
rr − 2 w⋅zh + wᵀKw`, `dof = tr H_λ = Σ Kᵢᵢ/sᵢ`. Parameters nothing reads (zero diagonal in
`B`) are dropped (0 in `y`). `B` must then be positive definite: a
direction no row holds is a gauge, which its caller holds exactly (Core's rule: no penalty
or ridge fixes it); an `ArgumentError` otherwise. Dense, `O(p³)` once in the parameters,
then `O(p²)` per λ. (Moved from RadiometricCamera's linear solves.)

    gcv_curve(J, r, D, c; hold = nothing, exact_max = 3000, probes = 64, seed) -> (λ -> (; gcv, rss, dof, y))

The same from the rows themselves (`J`, `D` and the hold's rows `R` dense or sparse): the
pencil above for at most `exact_max` parameters, the data residual `r − J y` read from the
rows (exact where it is small: the normal-equation form's `rr − 2w⋅zh + wᵀKw` cancels);
past it, per λ one sparse Cholesky of
`M_λ` (CHOLMOD) for the solution and a Hutchinson estimate of `tr H_λ = tr(M_λ⁻¹JᵀJ)` from
`probes` Rademacher vectors drawn from `seed` (deterministic; the probes read the
symmetric `zᵀJM_λ⁻¹Jᵀz`, `H`'s eigenvalues in [0, 1], so its standard deviation is
`≤ √(2 tr H / probes)`).
"""
function gcv_curve(G::AbstractMatrix, h::AbstractVector, rr::Real, Q::AbstractMatrix, q::AbstractVector, n::Integer;
        hold::Union{Nothing, AbstractMatrix} = nothing)
    P = _gcv_pencil(G, Q, hold)
    zh = P.V' * Vector{Float64}(h[P.keep]); zq = P.V' * Vector{Float64}(q[P.keep])
    return function (λ::Real)
        s = 1 .- P.μ .+ λ .* P.μ
        w = (zh .+ λ .* zq) ./ s
        rss = max(rr - 2 * dot(w, zh) + dot(w, P.K * w), 0.0)
        dof = sum(P.k ./ s)
        return (; gcv = gcv_score(n, rss, dof), rss, dof, y = _scatter(size(G, 1), P.keep, P.V * w))
    end
end
function gcv_curve(J::AbstractMatrix, r::AbstractVector, D::AbstractMatrix, c::AbstractVector;
        hold::Union{Nothing, AbstractMatrix} = nothing, exact_max::Integer = 3000, probes::Integer = 64, seed::Integer = 20261110)
    n, p = size(J)
    Js = SparseMatrixCSC{Float64, Int}(J); Ds = SparseMatrixCSC{Float64, Int}(D)
    G = Js' * Js; Q = Ds' * Ds
    h = Js' * r; q = Ds' * c
    RR = hold === nothing ? nothing : (R = SparseMatrixCSC{Float64, Int}(hold); R' * R)
    if p <= exact_max
        # the pencil, and the data residual read from the rows (`r − J y`, no cancellation where it is small)
        P = _gcv_pencil(G, Q, RR)
        zh = P.V' * Vector{Float64}(h[P.keep]); zq = P.V' * Vector{Float64}(q[P.keep])
        Jk = Js[:, P.keep]
        return function (λ::Real)
            s = 1 .- P.μ .+ λ .* P.μ
            yk = P.V * ((zh .+ λ .* zq) ./ s)
            rss = sum(abs2, r - Jk * yk)
            dof = sum(P.k ./ s)
            return (; gcv = gcv_score(n, rss, dof), rss, dof, y = _scatter(p, P.keep, yk))
        end
    end
    Gh = RR === nothing ? G : G + RR
    W = Js' * rand(Random.Xoshiro(seed), (-1.0, 1.0), n, probes)
    return function (λ::Real)
        F = cholesky(Symmetric(Gh + λ * Q); check = false)
        issuccess(F) || throw(ArgumentError("gcv_curve: J'J + R'R + λD'D is not positive definite at λ = $λ: a gauge is held exactly by the caller, never by a ridge"))
        y = F \ (h + λ * q)
        rss = sum(abs2, r - Js * y)
        dof = sum(W .* (F \ W)) / probes
        return (; gcv = gcv_score(n, rss, dof), rss, dof, y)
    end
end
_scatter(p::Integer, keep::AbstractVector{Int}, yk::AbstractVector) = (y = zeros(p); y[keep] = yk; y)

# The generalized eigendecomposition of the pencil `(Q, B)`, `B = G + RᵀR + Q` over the
# parameters something reads (`keep`, a positive diagonal of `B`): one Cholesky `B = L Lᵀ`
# and one symmetric eigendecomposition `L⁻¹ Q L⁻ᵀ = U diag(μ) Uᵀ`, `V = L⁻ᵀ U` (`VᵀBV = I`,
# `VᵀQV = diag μ`); `K = VᵀGV` (`I − diag μ` without a hold) and `k` its diagonal. A `B`
# that is not positive definite is a gauge: an `ArgumentError`, never a ridge.
function _gcv_pencil(G::AbstractMatrix, Q::AbstractMatrix, hold::Union{Nothing, AbstractMatrix})
    Bf = hold === nothing ? G + Q : G + hold + Q
    keep = [j for j in axes(Bf, 1) if Bf[j, j] > 0]
    Gk = Matrix{Float64}(G[keep, keep]); Qk = Matrix{Float64}(Q[keep, keep])
    F = LinearAlgebra.cholesky!(Symmetric(Matrix{Float64}(Bf[keep, keep])); check = false)
    issuccess(F) || throw(ArgumentError("gcv_curve: J'J + R'R + D'D is not positive definite: a direction neither the data, the hold nor the penalty reads (a gauge) is held exactly by the caller, never by a ridge"))
    μ, U = LinearAlgebra.eigen!(Symmetric(LinearAlgebra.rdiv!(LinearAlgebra.ldiv!(F.L, Qk), F.U)))
    μ = clamp.(μ, 0.0, 1.0)
    V = F.U \ U
    K = hold === nothing ? LinearAlgebra.Diagonal(1 .- μ) : Symmetric(V' * (Gk * V))
    return (; keep, μ, V, K, k = LinearAlgebra.diag(K))
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

# =============================================================================
# Quadrature rules and low-discrepancy points (hoisted from RadiometricCamera
# src/camera/quadrature.jl, `gauss_hermite_prob` and `hammersley`, their
# arithmetic unchanged)
# =============================================================================

using LinearAlgebra: SymTridiagonal, eigen

"""
    gauss_hermite_prob(n) -> (nodes::Vector{Float64}, weights::Vector{Float64})

The `n`-point Gauss–Hermite rule for the standard normal weight
(probabilists' form): `Σ wᵢ f(xᵢ) ≈ E[f(X)]`, `X ~ N(0, 1)`, the weights
summing to one, exact for polynomials of degree `2n − 1`. Closed forms for
`n ≤ 3`; beyond, Golub–Welsch — the eigen-decomposition of the Jacobi matrix
of the monic Hermite polynomials `He_k` (off-diagonal `√k`), the weights the
squared first components of its eigenvectors. Ascending nodes. An
`ArgumentError` for `n < 1`.

For `N(μ, σ²)`, use the nodes `μ + σ·xᵢ` with the same weights.
"""
function gauss_hermite_prob(n::Integer)
    n == 1 && return [0.0], [1.0]
    n == 2 && return [-1.0, 1.0], [0.5, 0.5]
    n == 3 && return [-sqrt(3.0), 0.0, sqrt(3.0)], [1 / 6, 2 / 3, 1 / 6]
    n >= 4 || throw(ArgumentError("gauss_hermite_prob: n ≥ 1 nodes, got $n"))
    x, V = eigen(SymTridiagonal(zeros(n), sqrt.(Float64.(1:(n - 1)))))
    w = V[1, :] .^ 2
    return x, w ./ sum(w)
end

"""
    hammersley([T = Float64,] i, N) -> (u₁, u₂)

The `i`-th point (`1 ≤ i ≤ N`) of the `N`-point 2-D Hammersley set in
`[0, 1)²`: `u₁ = (i − ½)/N`, `u₂` the base-2 van der Corput radical inverse of
`i − 1`. Scalar and isbits-only (a kernel computes its own point from its
index); the type argument leads, as in `rand(T, …)`.
"""
@inline hammersley(i::Integer, N::Integer) = hammersley(Float64, i, N)
@inline function hammersley(::Type{T}, i::Integer, N::Integer) where {T}
    u1 = (T(i) - T(1) / 2) / T(N)
    b = UInt32(i - 1); r = zero(T); f = T(1) / 2
    while b != 0
        (b & 1) == 1 && (r += f)
        f /= 2; b >>= 1
    end
    return u1, r
end

"""
    square_rule(x, w) -> (nodes::Vector{SVector{2, Float64}}, weights::Vector{Float64})

The tensor-product rule over a square of a one-dimensional rule (nodes `x`,
weights `w`, e.g. QuadGK's `gauss(n)` or [`gauss_hermite_prob`](@ref)): node
`(x[i], x[j])` with weight `w[i]·w[j]`, the first coordinate fastest. The
weights sum to the square of the 1-D rule's sum; it integrates exactly the
products of the polynomials the 1-D rule integrates exactly.
"""
function square_rule(x::AbstractVector, w::AbstractVector)
    length(x) == length(w) || throw(DimensionMismatch("square_rule: $(length(x)) nodes, $(length(w)) weights"))
    n = length(x)
    nodes = SVector{2, Float64}[]; weights = Float64[]
    for j in 1:n, i in 1:n
        push!(nodes, SVector{2, Float64}(x[i], x[j]))
        push!(weights, w[i] * w[j])
    end
    return nodes, weights
end

"""
    unit_disc_rule(n_ρ, n_φ) -> (nodes::Vector{SVector{2, Float64}}, weights::Vector{Float64})

The polar product rule over the unit disc `‖x‖ ≤ 1`: Gauss–Legendre in the
radius `ρ ∈ [0, 1]` (`n_ρ` nodes, QuadGK's `gauss` mapped from [−1, 1]) times
`n_φ` equally spaced angles `φ_j = 2π(j − ½)/n_φ`; node `(ρ cos φ, ρ sin φ)`
with weight `ρ·w_ρ·2π/n_φ` (the polar area element), radius-major (node
`(k − 1)·n_φ + j`). The weights sum to π, the disc's area: `Σ wᵢ f(xᵢ)`
approximates `∫ f` over the disc, exactly for `ρ^a cos(mφ), ρ^a sin(mφ)` with
`a + 1 ≤ 2n_ρ − 1` and `|m| < n_φ`.
"""
function unit_disc_rule(n_ρ::Integer, n_φ::Integer)
    (n_ρ >= 1 && n_φ >= 1) || throw(ArgumentError("unit_disc_rule: n_ρ, n_φ ≥ 1, got $(n_ρ), $(n_φ)"))
    x, w = gauss(n_ρ)
    ρ = (x .+ 1) ./ 2; wρ = w ./ 2                    # [−1, 1] → [0, 1]
    φ = [2π * (j - 0.5) / n_φ for j in 1:n_φ]
    nodes = SVector{2, Float64}[]; weights = Float64[]
    for k in 1:n_ρ, j in 1:n_φ
        push!(nodes, SVector{2, Float64}(ρ[k] * cos(φ[j]), ρ[k] * sin(φ[j])))
        push!(weights, ρ[k] * wρ[k] * 2π / n_φ)
    end
    return nodes, weights
end

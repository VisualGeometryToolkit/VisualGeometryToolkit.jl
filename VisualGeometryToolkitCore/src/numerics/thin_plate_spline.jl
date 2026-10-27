# =============================================================================
# Thin-plate spline on scattered nodes (hoisted from RadiometricCamera
# src/camera/tps.jl and the gradient of src/board/warping.jl, 2026-10)
#
#   f(𝐩) = a₁ + a_x 𝐩_x + a_y 𝐩_y + Σᵢ wᵢ U(𝐩, 𝐩ᵢ),   U(𝐩, 𝐪) = ½ r² ln r²,  r = ‖𝐩 − 𝐪‖
#
# with (w; a) the solution of [K P; Pᵀ 0] (w; a) = (values; 0), K = [U(𝐩ᵢ, 𝐩ⱼ)]
# (zero diagonal), P = [1 𝐩_x 𝐩_y]. The system matrix depends only on the
# nodes, so its inverse is formed once and (w, a) is a LINEAR map of the nodal
# values.
#
# The arithmetic is RadiometricCamera's, operation for operation: its golden
# tests pin it, and test/thin_plate_spline_tests.jl checks every value here
# against numbers that RadiometricCamera's own functions produced
# (test/fixtures/thin_plate_spline): `==` for everything after (w, a), within
# a measured bound for what goes through `inv` and BLAS (the BLAS build owns
# those last bits). Do not "improve" it — in particular the solver keeps `inv`
# rather than a factorization (see `ThinPlateSplineSystem`).
#
# Two halves, as in RadiometricCamera: the solver object `ThinPlateSplineSystem`
# (RadiometricCamera's `ThinPlateSpline`, renamed here because that name is the
# target calibration kind's, geometry/cameras/targets.jl) holds heap matrices and stays host-side; the evaluator takes isbits inputs
# (an `SVector` of nodes, `w`, `a`), allocates nothing and runs in a
# KernelAbstractions kernel on the CPU backend (tested; no device backend is),
# in any float type or `ForwardDiff.Dual`.
#
# Element types: `tps_evaluate` PROMOTES the nodes with (w, 𝐩), while
# `tps_gradient` CONVERTS the nodes to the promoted type of (w, 𝐩) — the rules
# of RadiometricCamera's `tps_evaluate` and `board_height_grad`, kept as they
# are and deliberately not unified (either change would move bits). Supplying the
# nodes in the evaluation's type (Float32 nodes for a Float32 evaluation) makes
# the two agree.
# =============================================================================

"""
    tps_radial_basis(𝐩, 𝐪) -> scalar

The thin-plate spline's radial basis function `U(𝐩, 𝐪) = ½ r² ln r²` with
`r = ‖𝐩 − 𝐪‖` (the same as `r² ln r`), and `U = 0` at `r = 0`. The zero branch
is taken on `r² > 0`, not by evaluating `0 · log 0`, so a `ForwardDiff.Dual`
evaluated on a node gets a zero value with zero partials (the true derivative
there) instead of `NaN`.
"""
@inline function tps_radial_basis(𝐩::StaticVector{2}, 𝐪::StaticVector{2})
    r2 = (𝐩[1] - 𝐪[1])^2 + (𝐩[2] - 𝐪[2])^2
    return r2 > 0 ? r2 * log(r2) / 2 : zero(r2)
end

"""
    ThinPlateSplineSystem(nodes) -> ThinPlateSplineSystem{N}

The solver object of a thin-plate spline through `N` nodes in the plane: the
node positions, the `(N + 3) × N` matrix `coeff` that maps nodal values to the
spline's weights and affine part, `(w; a) = coeff · values`, and the `N × N`
bending-energy form `Q` (`valuesᵀ Q values = wᵀ K w`, `K = [U(𝐩ᵢ, 𝐩ⱼ)]`, see
[`bending_energy`](@ref)).

`coeff` is the first `N` columns of the INVERTED system matrix `[K P; Pᵀ 0]`,
formed with `inv` as RadiometricCamera forms it: its golden tests pin these
numbers, and a factorization would move them in the last bits. The matrices
live on the host; [`tps_coefficients`](@ref) turns nodal values into the
isbits `(w, a)` that [`tps_evaluate`](@ref) and [`tps_gradient`](@ref) take.
"""
struct ThinPlateSplineSystem{N}
    nodes::SVector{N, SVector{2, Float64}}
    coeff::Matrix{Float64}      # (N+3) × N
    Q::Matrix{Float64}          # N × N
end

function ThinPlateSplineSystem(nodes::AbstractVector{<:StaticVector{2}})
    N = length(nodes)
    Umat = [tps_radial_basis(SVector{2, Float64}(nodes[i]), SVector{2, Float64}(nodes[j])) for i in 1:N, j in 1:N]
    P = [ones(N) getindex.(nodes, 1) getindex.(nodes, 2)]
    K = [Umat P; P' zeros(3, 3)]
    Kinv = inv(K)   # `inv`, not `\`: RadiometricCamera's goldens depend on these exact bits
    coeff = Kinv[:, 1:N]
    Cw = coeff[1:N, :]
    Q = Matrix(Symmetric(Cw' * Umat * Cw))
    return ThinPlateSplineSystem{N}(SVector{N}(SVector{2, Float64}.(nodes)), coeff, Q)
end

"""
    grid_nodes(n; lo = -0.5, hi = 0.5) -> Vector{SVector{2, Float64}}

The `n × n` equally spaced nodes over the square `[lo, hi]²`, column-major
(`x` fastest). The default square `[-½, ½]²` is RadiometricCamera's SENSOR
chart (its focus surface); a spline over another domain — a target's chart,
say — supplies its own `lo` and `hi`.
"""
function grid_nodes(n::Integer; lo::Real = -0.5, hi::Real = 0.5)
    g = range(lo, hi; length = n)
    return [SVector(float(g[i]), float(g[j])) for j in 1:n for i in 1:n]
end

"""
    tps_coefficients(tps::ThinPlateSplineSystem{N}, values) -> (w::SVector{N}, a::SVector{3})

The spline's weights `w` and affine part `a = (a₁, a_x, a_y)` through the
nodal values `values` (length `N`). Linear in `values`, and generic in their
element type: a `ForwardDiff.Dual` or a `Float32` is kept (the product
with the `Float64` matrix is formed first, then converted to the values'
element type).
"""
function tps_coefficients(tps::ThinPlateSplineSystem{N}, t::AbstractVector{T}) where {N, T}
    length(t) == N || throw(DimensionMismatch("expected $N nodal values, got $(length(t))"))
    wa = tps.coeff * t
    return SVector{N, T}(ntuple(i -> wa[i], Val(N))), SVector{3, T}(wa[N + 1], wa[N + 2], wa[N + 3])
end

"""
    tps_evaluate(nodes, w, a, 𝐩) -> scalar
    tps_evaluate(tps::ThinPlateSplineSystem, w, a, 𝐩) -> scalar

The spline at `𝐩`: `a₁ + a_x 𝐩_x + a_y 𝐩_y + Σᵢ wᵢ U(𝐩, 𝐩ᵢ)` with `(w, a)` from
[`tps_coefficients`](@ref) and `U` the [`tps_radial_basis`](@ref). The first form
takes isbits inputs only (`nodes::SVector{N, <:StaticVector{2}}`, `w`, `a`,
`𝐩`): it allocates nothing and runs in a KernelAbstractions kernel on the CPU
backend. It is generic in the element type and PROMOTES the nodes with `w` and
`𝐩`: a `Float32` evaluation with `Float64` nodes returns `Float64`. This
differs from [`tps_gradient`](@ref), which converts the nodes, as
RadiometricCamera's two functions do; supply the nodes in the evaluation's type
(`Float32` nodes for `Float32`) and both work in that type.
"""
@inline function tps_evaluate(nodes::SVector{N, <:StaticVector{2}}, w::StaticVector{N}, a::StaticVector{3}, 𝐩::StaticVector{2}) where {N}
    a_1, a_x, a_y = a[1], a[2], a[3]
    s = a_1 + a_x * 𝐩[1] + a_y * 𝐩[2]
    for i in 1:N
        s += w[i] * tps_radial_basis(𝐩, nodes[i])
    end
    return s
end
tps_evaluate(tps::ThinPlateSplineSystem, w, a, 𝐩) = tps_evaluate(tps.nodes, w, a, 𝐩)

"""
    tps_gradient(nodes, w, a, 𝐩) -> SVector{2}
    tps_gradient(tps::ThinPlateSplineSystem, w, a, 𝐩) -> SVector{2}

The analytic gradient `∇f(𝐩)` of [`tps_evaluate`](@ref): `(a_x, a_y) + Σᵢ wᵢ
(𝐩 − 𝐩ᵢ)(ln r² + 1)`, since `∇_𝐩 ½ r² ln r² = (𝐩 − 𝐩ᵢ)(ln r² + 1)`; a node at
`r = 0` contributes zero. The sum is RadiometricCamera's `board_height_grad`
term for term (its height field is this spline without the affine part, which
is added last). Isbits inputs and allocation-free, and it runs in a
KernelAbstractions kernel on the CPU backend, as `tps_evaluate`. Unlike
[`tps_evaluate`](@ref), which promotes, it CONVERTS the node coordinates to the
promoted element type of `w` and `𝐩` (RadiometricCamera's `board_height_grad`
rule): `Float32` `w` and `𝐩` with `Float64` nodes give a `Float32` gradient.
Supply the nodes in the evaluation's type and the two agree.
"""
@inline function tps_gradient(nodes::SVector{N, <:StaticVector{2}}, w::StaticVector{N}, a::StaticVector{3}, 𝐩::StaticVector{2}) where {N}
    S = promote_type(eltype(w), eltype(𝐩))
    g1 = zero(S); g2 = zero(S)
    @inbounds for i in 1:N
        n = nodes[i]
        d1 = 𝐩[1] - S(n[1]); d2 = 𝐩[2] - S(n[2])
        r2 = d1 * d1 + d2 * d2
        f = r2 > 0 ? w[i] * (log(r2) + 1) : zero(S)
        g1 += f * d1; g2 += f * d2
    end
    return SVector(a[2] + g1, a[3] + g2)
end
tps_gradient(tps::ThinPlateSplineSystem, w, a, 𝐩) = tps_gradient(tps.nodes, w, a, 𝐩)

"""
    nodal_weights(tps::ThinPlateSplineSystem{N}, 𝐩) -> SVector{N, Float64}

The sensitivity of the spline at `𝐩` to each nodal value, `∂f(𝐩)/∂valuesᵢ`.
With `f(𝐩) = φ(𝐩)ᵀ (coeff · values)` and `φ = (U(𝐩, 𝐩₁) … U(𝐩, 𝐩_N), 1, 𝐩_x,
𝐩_y)`, node `i`'s weight is `Σ_r coeff[r, i] φ_r(𝐩)` — exact, since the spline
is linear in its values. Host-side (it reads `coeff`), in `Float64`.
"""
function nodal_weights(tps::ThinPlateSplineSystem{N}, 𝐩::StaticVector{2}) where {N}
    N <= _NODAL_STATIC_MAX || return _nodal_weights_loop(tps, 𝐩)
    φ = ntuple(r -> r <= N ? tps_radial_basis(𝐩, tps.nodes[r]) : (r == N + 1 ? 1.0 : (r == N + 2 ? Float64(𝐩[1]) : Float64(𝐩[2]))), Val(N + 3))
    c = tps.coeff
    return SVector{N}(ntuple(i -> sum(ntuple(r -> c[r, i] * φ[r], Val(N + 3))), Val(N)))
end
# The unrolled form compiles N(N + 3) terms: past this many nodes (a 3 × 3 grid
# and a little more) the same sums run as loops, left to right as `sum` folds a
# tuple, so both forms give the same bits.
const _NODAL_STATIC_MAX = 16
function _nodal_weights_loop(tps::ThinPlateSplineSystem{N}, 𝐩::StaticVector{2}) where {N}
    c = tps.coeff
    φ = Vector{Float64}(undef, N + 3)
    for r in 1:N
        φ[r] = tps_radial_basis(𝐩, tps.nodes[r])
    end
    φ[N + 1] = 1.0; φ[N + 2] = Float64(𝐩[1]); φ[N + 3] = Float64(𝐩[2])
    out = MVector{N, Float64}(undef)
    @inbounds for i in 1:N
        s = c[1, i] * φ[1]
        for r in 2:(N + 3)
            s += c[r, i] * φ[r]
        end
        out[i] = s
    end
    return SVector(out)
end

"""
    bending_energy(tps::ThinPlateSplineSystem, values) -> scalar

The bending energy `valuesᵀ Q values` of the spline through the nodal values
`values` (`= wᵀ K w` with `K = [U(𝐩ᵢ, 𝐩ⱼ)]` the radial basis between the
nodes; zero for affine values up to rounding); a quadratic regularizer on the
values.
"""
bending_energy(tps::ThinPlateSplineSystem, t::AbstractVector) = dot(t, tps.Q * t)

"""
    bending_factor(tps::ThinPlateSplineSystem{N}) -> Matrix{Float64}

A factor `L` (`N × N`) of the bending-energy form: `‖L · values‖² =`
[`bending_energy`](@ref)`(tps, values)` to rounding, so `L · values` are a
regularizer's residual rows. `Q` is positive semidefinite with the affine
values in its null space, so it has no Cholesky factor: `L = Λ₊^½ Vᵀ` from the
eigendecomposition `Q = V Λ Vᵀ`, `Λ₊` the eigenvalues clamped at zero (the null
space's rounding residue may come out negative). An eigenbasis is not unique,
so only `LᵀL` is defined — never compare the rows one by one.
"""
function bending_factor(tps::ThinPlateSplineSystem)
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(tps.Q))
    return LinearAlgebra.Diagonal(sqrt.(max.(F.values, 0.0))) * F.vectors'
end

# =============================================================================
# The bending energy and the natural boundary condition of a B-spline — read
# from BSplineKit (the CPU-side B-spline algebra; Core keeps only the device
# kernels, numerics/bspline.jl)
#
#   E(f) = ∫∫ f_xx² + 2 f_xy² + f_yy² dx dy  over the spline's domain:
#   for f = Σ c_ij N_i(x) M_j(y) the quadratic form
#   vec(c)ᵀ (G₂ˣ ⊗ G₀ʸ + 2 G₁ˣ ⊗ G₁ʸ + G₀ˣ ⊗ G₂ʸ) vec(c),
#   Gᵣ = galerkin_matrix(B, (Derivative(r), Derivative(r))) per axis
#   (Hermitian, banded; seven terms a row for a cubic).
#
# The natural boundary condition f'' = 0 at a border knot is BSplineKit's
# `RecombinedBSplineBasis(B, Natural())`: the recombined basis keeps the
# interior parameters and makes the outermost coefficient follow them. On a
# uniform cubic lattice its recombination is the linear extrapolation
# c₀ = 2c₁ − c₂ (`natural_extrapolation`), the rule `BoundaryRule` applies
# outward from an imaged region of any shape.
# =============================================================================

using BSplineKit: BSplineKit, BSplineBasis, BSplineOrder, Derivative, Natural, RecombinedBSplineBasis,
                  galerkin_matrix, recombination_matrix

"""
    cubic_lattice_basis(n; spacing = 1) -> BSplineKit.BSplineBasis

The uniform cubic B-spline lattice of `n` nodes (Core's generic-model
lattice) as a BSplineKit basis: knots `(−3:n) · spacing`, so basis `j`
(node `j`) is centred at `(j − 2) · spacing` and the spline's domain is the
`n − 3` cells `[0, (n − 3) · spacing]` where the basis sums to one.
"""
function cubic_lattice_basis(n::Integer; spacing::Real = 1.0)
    n >= 4 || throw(ArgumentError("cubic_lattice_basis: a uniform cubic lattice needs ≥ 4 nodes, got $n"))
    return BSplineBasis(BSplineOrder(4), collect((-3:n) .* Float64(spacing)); augment = Val(false))
end

_gram(B, r) = Matrix{Float64}(galerkin_matrix(B, (Derivative(r), Derivative(r))))

"""
    bending_gram(B) -> Matrix{Float64}
    bending_gram(Bx, By) -> Matrix{Float64}

The bending energy of a B-spline as a quadratic form in its coefficients:
`∫ f''²` of a curve on the BSplineKit basis `B`; `∫∫ f_xx² + 2 f_xy² + f_yy²`
of the tensor-product surface on `Bx ⊗ By`, `vec(c)ᵀ G vec(c)` with `c` laid
`[i, j]` (the first index fastest). BSplineKit's Galerkin matrices of the
basis's derivatives, over the bases' domains. Positive semidefinite, the
affine functions its null space. A regularizer reads it as rows
([`bending_rows`](@ref)) or, on a large uniform lattice, cell by cell
([`bending_stencils`](@ref)); a lattice of vector-valued controls applies it
per component.
"""
bending_gram(B) = _gram(B, 2)
bending_gram(Bx, By) =
    LinearAlgebra.kron(_gram(By, 0), _gram(Bx, 2)) .+ 2 .* LinearAlgebra.kron(_gram(By, 1), _gram(Bx, 1)) .+
    LinearAlgebra.kron(_gram(By, 2), _gram(Bx, 0))

"""
    bending_rows(G) -> Matrix{Float64}

A factor `L` of a bending Gram matrix ([`bending_gram`](@ref)): `‖L c‖² =
cᵀ G c` to rounding, so `√λ · L (c − c₀)` are a regularizer's residual rows
(the chart LM's row protocol). `L = Λ₊^½ Vᵀ` from `G = V Λ Vᵀ`, one row per
positive eigenvalue (the null space's rounding residue dropped); only `LᵀL`
is defined.
"""
function bending_rows(G::AbstractMatrix)
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Matrix{Float64}(G)))
    keep = F.values .> 1.0e-12 * max(maximum(F.values), eps())
    return LinearAlgebra.Diagonal(sqrt.(F.values[keep])) * F.vectors[:, keep]'
end

"""
    bending_stencils(nx, ny; spacing = (1, 1)) -> (; windows, L)

The bending energy of an `nx × ny` uniform cubic lattice
([`cubic_lattice_basis`](@ref) per axis, cell size `spacing`) cell by cell,
for a lattice too large for a dense form: `E = Σₖ ‖L c[windows[k]]‖²` over the
`(nx − 3)(ny − 3)` cells, `windows[k]` the 16 linear indices of cell `k`'s
4 × 4 window (the first index fastest) and `L` the factor of one cell's
16 × 16 form — [`bending_gram`](@ref) of a one-cell lattice, shared by every
cell of a uniform lattice. A row involves one window only, so a solve that
blocks the lattice keeps its rows local.
"""
function bending_stencils(nx::Integer, ny::Integer; spacing = (1.0, 1.0))
    (nx >= 4 && ny >= 4) || throw(ArgumentError("bending_stencils: a uniform cubic lattice needs ≥ 4 × 4 nodes, got $nx × $ny"))
    L = bending_rows(bending_gram(cubic_lattice_basis(4; spacing = spacing[1]), cubic_lattice_basis(4; spacing = spacing[2])))
    lin = LinearIndices((nx, ny))
    windows = [ntuple(k -> lin[cx + (k - 1) % 4, cy + (k - 1) ÷ 4], Val(16)) for cy in 1:(ny - 3) for cx in 1:(nx - 3)]
    return (; windows, L)
end

"""
    natural_extrapolation() -> (w₁, w₂)

The natural boundary condition `f'' = 0` at a border knot of a uniform cubic
B-spline as the outermost coefficient's extrapolation from the two inward,
`c₀ = w₁ c₁ + w₂ c₂`: the relation BSplineKit's
`RecombinedBSplineBasis(B, Natural())` imposes (its recombination matrix's
span), `(2, −1)` — the rule [`BoundaryRule`](@ref) applies outward from an
imaged region.
"""
function natural_extrapolation()
    M = Matrix{Float64}(recombination_matrix(RecombinedBSplineBasis(cubic_lattice_basis(6), Natural())))
    v = LinearAlgebra.nullspace(M[1:3, 1:2]')[:, 1]           # (c₀, c₁, c₂) ⟂ the recombined span: v ⋅ c = 0
    return (-v[2] / v[1], -v[3] / v[1])
end

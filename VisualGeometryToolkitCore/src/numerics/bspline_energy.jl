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
([`bending_stencils`](@ref), and over a mask of held nodes `bending_rows(mask)`); a lattice of vector-valued controls applies it
per component.
"""
bending_gram(B) = _gram(B, 2)
bending_gram(Bx, By) =
    LinearAlgebra.kron(_gram(By, 0), _gram(Bx, 2)) .+ 2 .* LinearAlgebra.kron(_gram(By, 1), _gram(Bx, 1)) .+
    LinearAlgebra.kron(_gram(By, 2), _gram(Bx, 0))

"""
    bending_rows(G) -> Matrix{Float64}
    bending_rows(tps::ThinPlateSplineSystem) -> Matrix{Float64}
    bending_rows(sys::SphericalSplineSystem) -> Matrix{Float64}

The bending energy's rows, the stack's second penalty operator kind beside
[`difference_rows`](@ref): a factor `L` of a positive semidefinite bending form —
a B-spline lattice's Gram matrix ([`bending_gram`](@ref)), or a thin-plate or
spherical spline system's `Q` ([`bending_energy`](@ref)) — with `‖L c‖² = cᵀ G c`
to rounding, so `√λ · L (c − c₀)` are a regularizer's residual rows (the chart LM's
row protocol) and `L` a penalty operator for [`mean_positive_eigenvalue`](@ref) and
[`cgls`](@ref). `L = Λ₊^½ Vᵀ` from `G = V Λ Vᵀ`, one row per positive eigenvalue (the
null space — a system's affine or constant values — and its rounding residue have
no row; [`bending_factor`](@ref) is the same factor with a row for every node, the
fixed shape a target calibration kind stores); only `LᵀL` is defined.
"""
function bending_rows(G::AbstractMatrix)
    L, λ = _eigen_factor(G)
    return L[λ .> 1.0e-12 * max(maximum(λ), eps()), :]
end

# THE factor of a positive semidefinite form `G = V Λ Vᵀ`: `L = Λ₊^½ Vᵀ` (the eigenvalues
# clamped at zero: the null space's rounding residue may come out negative), one row per
# eigenvalue `λ` (ascending), `‖L c‖² = cᵀGc` to rounding — `bending_rows` and the
# thin-plate kinds' `bending_factor` (numerics/thin_plate_spline.jl) read it.
function _eigen_factor(G::AbstractMatrix)
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(float(Matrix(G))))
    return LinearAlgebra.Diagonal(sqrt.(max.(F.values, zero(eltype(F.values))))) * F.vectors', F.values
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
    bending_rows(mask::AbstractMatrix{Bool}; spacing = (1, 1)) -> SparseMatrixCSC{Float64, Int}

The bending energy's rows over the nodes of a uniform cubic lattice that `mask`
holds (`true`: a node the rows may read), as a sparse operator on the lattice's
column-major vector. These are the [`bending_stencils`](@ref) of the
`size(mask)` lattice (cell size `spacing`) over the cells whose 16 nodes are all
in `mask`: each cell's rows `L` in cell order, the first index fastest. It is
the masked bending operator beside the masked differences of
[`difference_rows`](@ref)`(mask, order)`. A regularizer of a lattice that holds
only the nodes an imaged region reaches reads it as a penalty operator
([`mean_positive_eigenvalue`](@ref), [`row_stencils`](@ref)). A lattice of
vector-valued controls applies it per component. A mask that holds no complete
cell gives an operator with no rows.
"""
function bending_rows(mask::AbstractMatrix{Bool}; spacing = (1.0, 1.0))
    nx, ny = size(mask)
    min(nx, ny) < 4 && return sparse(Int[], Int[], Float64[], 0, nx * ny)   # a lattice below 4 × 4 has no cell: no rows
    st = bending_stencils(nx, ny; spacing)
    I, J, V = Int[], Int[], Float64[]
    row = 0
    for w in st.windows
        all(n -> mask[n], w) || continue
        for r in axes(st.L, 1)
            row += 1
            for (m, n) in enumerate(w)
                push!(I, row); push!(J, n); push!(V, st.L[r, m])
            end
        end
    end
    return sparse(I, J, V, row, nx * ny)
end

"""
    difference_stencil(order) -> NTuple{order + 1, Float64}

The coefficients of the forward difference `Δᵒʳᵈᵉʳ` of a lattice's
coefficients, `Δᵒc₀ = Σₖ sₖ cₖ₋₁` over `order + 1` consecutive nodes: the
binomial coefficients with alternating sign, `(1, −3, 3, −1)` for `order = 3`.
The one source of a P-spline's difference penalty
(VisualGeometryToolkitEstimation's `fit_camera`) and of [`BoundaryRule`](@ref)'s
curvature-preserving extrapolation (`Δ³c = 0` solved for the outermost
coefficient).
"""
difference_stencil(order::Integer) =
    ntuple(k -> Float64((-1)^(k - 1) * binomial(Int(order), k - 1)), Int(order) + 1)

"""
    difference_rows(mask::AbstractMatrix{Bool}, order; axes = (1, 2)) -> SparseMatrixCSC{Float64, Int}
    difference_rows(dims::Dims{2}, order; axes = (1, 2))

The forward differences `Δᵒʳᵈᵉʳ` ([`difference_stencil`](@ref), any order ≥ 1: 1 a
gradient, 2 the natural rule's, 3 the P-spline's) of a 2-D lattice of coefficients along
the lattice `axes` (a subset of `(1, 2)`, in that order), as a sparse operator on the
lattice's column-major vector: one row per window of `order + 1` consecutive nodes that
lies wholly in `mask` (`true`: a node the rows may read), first the windows along the
first axis (`(i, j) … (i + order, j)`, `j` outer), then those along the second. `dims` is
the unmasked lattice (`trues(dims)`): a full texel page `(TW, TH)` at `order = 1` is its
gradient, the horizontal differences first. A lattice of vector-valued controls applies
it per component.

`DᵀD` is the P-spline difference penalty (Eilers & Marx 1996): at `order = 3` it
annihilates a quadratic along each axis, so only the variation beyond it is penalized.
An `N × 1` mask is a profile (rows along its one axis only). The isotropic pairing of
the order-1 rows is [`gradient_pairs`](@ref); a device solve takes the operator as a
[`CSRMatrix`](@ref).
"""
function difference_rows(mask::AbstractMatrix{Bool}, order::Integer; axes::Tuple{Vararg{Int}} = (1, 2))
    order >= 1 || throw(ArgumentError("difference_rows: order must be ≥ 1, got $order"))
    (!isempty(axes) && issubset(axes, (1, 2)) && issorted(axes; lt = <=)) ||
        throw(ArgumentError("difference_rows: axes is (1,), (2,) or (1, 2), got $axes"))
    s = difference_stencil(order)
    L = LinearIndices(mask)
    n1, n2 = size(mask)
    I, J, V = Int[], Int[], Float64[]
    row = 0
    for (axis, ni, nj) in ((1, n1 - order, n2), (2, n1, n2 - order)), j in 1:nj, i in 1:ni
        axis in axes || continue
        δ = axis == 1 ? CartesianIndex(1, 0) : CartesianIndex(0, 1)
        all(k -> mask[CartesianIndex(i, j) + k * δ], 0:order) || continue
        row += 1
        for k in 0:order
            push!(I, row); push!(J, L[CartesianIndex(i, j) + k * δ]); push!(V, s[k + 1])
        end
    end
    return sparse(I, J, V, row, length(mask))
end
difference_rows(dims::Dims{2}, order::Integer; axes::Tuple{Vararg{Int}} = (1, 2)) = difference_rows(trues(dims), order; axes)

"""
    gradient_pairs(mask::AbstractMatrix{Bool}) -> Matrix{NTuple{2, Int}}
    gradient_pairs(dims::Dims{2})

Each node's isotropic gradient pair among the rows of `difference_rows(mask, 1)`: at
node `(i, j)`, `(r₁, r₂)` the rows of its forward differences along the first axis
(`(i, j) → (i + 1, j)`) and along the second (`(i, j) → (i, j + 1)`), `0` where that
difference is not a row (the lattice's last node on the axis, or a neighbour outside
`mask`). A penalty on the gradient's magnitude (`(D x)[r₁]² + (D x)[r₂]²`, an
L0 or total-variation step) reads both entries of a pair together.
"""
function gradient_pairs(mask::AbstractMatrix{Bool})
    n1, n2 = size(mask)
    P = fill((0, 0), n1, n2)
    row = 0
    for j in 1:n2, i in 1:(n1 - 1)
        (mask[i, j] && mask[i + 1, j]) || continue
        row += 1; P[i, j] = (row, 0)
    end
    for j in 1:(n2 - 1), i in 1:n1
        (mask[i, j] && mask[i, j + 1]) || continue
        row += 1; P[i, j] = (P[i, j][1], row)
    end
    return P
end
gradient_pairs(dims::Dims{2}) = gradient_pairs(trues(dims))

"""
    clamped_knots(n, p, lo, hi) -> SVector{n + p + 1, Float64}

The clamped (open) uniform knot vector of `n` basis functions of degree `p`
on `[lo, hi]`: `p + 1` knots at each end and `n − p − 1` interior knots
spaced evenly, BSplineKit's `knots(BSplineBasis(BSplineOrder(p + 1),
range(lo, hi; length = n − p + 1)))` as the static vector
[`bspline_basis`](@ref) takes (a kernel's knots). The spline interpolates its
first and last coefficient at `lo` and `hi`. Needs `n > p`.
"""
clamped_knots(n::Integer, p::Integer, lo::Real, hi::Real) =
    (n > p || throw(ArgumentError("clamped_knots: need more than p = $p control points, got $n"));
     SVector{n + p + 1}(Float64.(BSplineKit.knots(BSplineBasis(BSplineOrder(p + 1), range(Float64(lo), Float64(hi); length = n - p + 1))))))

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

# =============================================================================
# Penalized least squares — the stack's one owner of each piece
# =============================================================================
#
# Every penalized solve of the stack is
#
#   min_x ‖A x − b‖² + λ·‖D (x − c)‖²,   λ = β · penalty_scale(energy, N, m̄)
#
# differing only in the operator D, the target c and the solve:
#
# THE OPERATOR. `difference_rows` (numerics/bspline_energy.jl): forward differences
#   of orders 1–3 on a masked lattice, a lattice of controls or a full texel page,
#   along chosen axes; `gradient_pairs` groups its order-1 rows into each node's
#   isotropic (∂₁, ∂₂) pair. `bending_rows`: the thin-plate bending energy's rows,
#   the second operator kind. `row_stencils`: an operator's rows as the nodes and
#   coefficients each reads, the form a nonlinear solve's residual blocks take (an
#   NLLSsolver block per row on the variables holding its nodes).
# THE NORMALIZATION. `mean_positive_eigenvalue` (m̄ of DᵀD) and `penalty_scale`
#   (β relative to the data's energy per parameter and to m̄).
# THE LINEAR SOLVE. `cgls` (min ‖Ax − b‖² + ‖√w∘(Dx − h)‖², warm start, target,
#   KernelAbstractions on any backend) on sparse operators, `CSRMatrix` their device
#   form; `normal_cg` the same iteration on a normal operator AᵀA + DᵀWD that is
#   only applied (rows never stored); either Jacobi-preconditioned. A dense or
#   CHOLMOD-sized problem solves its normal equations directly.
# THE WEIGHT'S CHOICE. VisualGeometryToolkitEstimation's `gcv_curve`/`gcv_select`
#   (fitting/gcv.jl) — GCV is a fit's, not a numerical primitive.
#
# NO PENALTY FIXES A GAUGE. A direction neither the data nor the penalty holds is
# held exactly by its caller (a parametrization, a held coordinate, a post-solve
# transform), never by a ridge or a pull toward the start.
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
mean_positive_eigenvalue(D::AbstractMatrix) = sum(abs2, D) / _operator_rank(D)
mean_positive_eigenvalue(G::LinearAlgebra.Symmetric) = LinearAlgebra.tr(G) / LinearAlgebra.rank(Matrix{Float64}(G))
_operator_rank(D::SparseMatrixCSC) = LinearAlgebra.rank(LinearAlgebra.qr(SparseMatrixCSC{Float64, Int}(D)))
_operator_rank(D::AbstractMatrix) = LinearAlgebra.rank(Matrix{Float64}(D))

"""
    penalty_scale(energy, N, m̄) -> Float64

The unit of a penalty weight: `λ = β · energy / N / m̄`, β relative to the data's
`energy` per parameter (`N` parameters) and to the penalty operator's mean positive
eigenvalue `m̄` ([`mean_positive_eigenvalue`](@ref)), so β means the same for every
operator, lattice and region; β = 1 weighs the penalty like the data. The energy is the
caller's: a linear solve's data energy (`tr AᵀA`, or a radiometric Σϕ²), or a nonlinear
refine's Gauss–Newton energy `tr JᵀJ` over its parameters (the residuals of exact data
vanish, so they cannot set a scale).
"""
penalty_scale(energy::Real, N::Integer, m̄::Real) = Float64(energy) / N / Float64(m̄)

"""
    row_stencils(D::SparseMatrixCSC) -> Vector{Tuple{Vector{Int}, Vector{Float64}}}

The rows of a sparse operator as the columns (nodes) each reads and its coefficients
there, in row order, the nodes ascending: the form of a penalty as a nonlinear solve's
residual blocks (one block per row on the variables holding its nodes, its residual
`√λ·Σₘ sₘ xₙₘ`, a caller's chart and row weighting applied to it).
"""
function row_stencils(D::SparseMatrixCSC)
    T = SparseMatrixCSC(transpose(D))                     # column m: row m of D
    return [(T.rowval[T.colptr[m]:(T.colptr[m + 1] - 1)], T.nzval[T.colptr[m]:(T.colptr[m + 1] - 1)]) for m in 1:size(T, 2)]
end

# =============================================================================
# The device form of a sparse operator, and the one CGLS
# =============================================================================

"""
    CSRMatrix(A::SparseMatrixCSC; device = default_device()) -> CSRMatrix

A sparse operator in compressed sparse rows on a KernelAbstractions `device`
([`default_device`](@ref)`()` when none is supplied: an accelerator when one is functional,
else `KA.CPU()`): `mul!(y, C,
x)` is one work item per row (each row's sum in its stored order: deterministic on every
backend, no atomics). `Adapt.adapt(backend, C)` moves it. The operator a device solve
([`cgls`](@ref)) applies; its transpose is a second `CSRMatrix` of `transpose(A)`
([`PenalizedOperator`](@ref) holds both).
"""
struct CSRMatrix{I <: AbstractVector{<:Integer}, V <: AbstractVector{Float64}}
    m::Int
    n::Int
    rowptr::I
    colval::I
    nzval::V
end
CSRMatrix(A::SparseMatrixCSC; device = default_device()) = _transposed_csr(SparseMatrixCSC{Float64, Int}(transpose(A)); device)   # its columns are A's rows
# The CSR form of `transpose(A)`: A's own compressed columns, copied to `device` (no transpose).
_transposed_csr(A::SparseMatrixCSC; device) =
    CSRMatrix(size(A, 2), size(A, 1), ka_upload(Int, device, A.colptr), ka_upload(Int, device, A.rowval), ka_upload(Float64, device, A.nzval))
Base.size(C::CSRMatrix) = (C.m, C.n)
Base.size(C::CSRMatrix, d::Integer) = d == 1 ? C.m : d == 2 ? C.n : 1
Adapt.adapt_structure(to, C::CSRMatrix) = CSRMatrix(C.m, C.n, Adapt.adapt(to, C.rowptr), Adapt.adapt(to, C.colval), Adapt.adapt(to, C.nzval))
KA.get_backend(C::CSRMatrix) = get_backend(C.nzval)

@kernel function _csr_mul_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x))
    i = @index(Global)
    s = zero(eltype(y))
    @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
        s += nzval[k] * x[colval[k]]
    end
    @inbounds y[i] = s
end
function LinearAlgebra.mul!(y::AbstractVector, C::CSRMatrix, x::AbstractVector)
    C.m == 0 && return y
    _csr_mul_kernel!(get_backend(y))(y, C.rowptr, C.colval, C.nzval, x; ndrange = C.m)
    return y
end

"""
    PenalizedOperator(A::SparseMatrixCSC, D::SparseMatrixCSC = spzeros(0, size(A, 2)); device = default_device())

The operators of a penalized least squares `min ‖A x − b‖² + ‖√w ∘ (D x − h)‖²` on a
KernelAbstractions `device`, built once for every solve on them ([`cgls`](@ref)): `A`,
`D` and their transposes as [`CSRMatrix`](@ref)es. `D` is any of the stack's penalty
operators ([`difference_rows`](@ref), [`bending_rows`](@ref) as a sparse matrix), its rows'
weights `w` the solve's.
"""
struct PenalizedOperator{C <: CSRMatrix, B}
    A::C
    At::C
    D::C
    Dt::C
    device::B
end
function PenalizedOperator(A::SparseMatrixCSC, D::SparseMatrixCSC = SparseMatrixCSC{Float64, Int}(undef, 0, size(A, 2)); device = default_device())
    size(D, 2) == size(A, 2) || throw(DimensionMismatch("PenalizedOperator: A has $(size(A, 2)) columns, D $(size(D, 2))"))
    # a transpose's CSR form is the operator's own compressed columns
    return PenalizedOperator(CSRMatrix(A; device), _transposed_csr(A; device), CSRMatrix(D; device), _transposed_csr(D; device), device)
end

"""
    cgls(op::PenalizedOperator, b; w = 1, h = nothing, diagonal = nothing, x0 = nothing, rtol = 1e-6, max_iter = 500)
        -> (; x, iterations, converged, relres)

The least-squares solution of `min_x ‖A x − b‖² + ‖√w ∘ (D x − h)‖²` (`op`'s `A` and `D`;
`w` a weight per penalty row, or one for all; `h` the penalty's target, `nothing` for 0)
by CGLS — conjugate gradients on the normal equations `(AᵀA + DᵀWD) x = Aᵀb + DᵀWh`
applied as the stacked operator `[A; √W D]`, never formed — from `x0` (a warm start;
`nothing`: 0), on `op`'s device (the vectors moved there; `x` returned on it).
`diagonal`, the normal matrix's diagonal `diag(AᵀA + DᵀWD)` (positive), preconditions the
iteration (Jacobi: the unknowns' columns scaled to unit norm), `nothing` none.

The stopping rule is relative to the normal equations' right-hand side, `‖sₖ‖ ≤ rtol ·
‖Aᵀb + DᵀWh‖`, `sₖ = Aᵀ(b − A xₖ) + DᵀW(h − D xₖ)` the normal-equation residual: a
reference fixed by the problem, not by the start, so a warm start that is already close
needs few or no iterations (a rule relative to `‖s₀‖` asks a warm start for `rtol` more
digits of improvement than a cold one). `converged` when the rule is met within
`max_iter` iterations; `relres` is `‖sₖ‖ / ‖Aᵀb + DᵀWh‖`.

The rows of `A` stored; a problem whose normal operator is only applied (its rows never
formed) solves with [`normal_cg`](@ref), the same iteration.
"""
function cgls(op::PenalizedOperator, b::AbstractVector; w = 1.0, h::Union{Nothing, AbstractVector} = nothing,
        diagonal::Union{Nothing, AbstractVector} = nothing, x0::Union{Nothing, AbstractVector} = nothing,
        rtol::Real = 1.0e-6, max_iter::Integer = 500)
    dev = op.device
    n = op.A.n; mD = op.D.m
    length(b) == op.A.m || throw(DimensionMismatch("cgls: b has $(length(b)) entries, A $(op.A.m) rows"))
    vec_on(v) = on_device(Float64, dev, v)
    bd = vec_on(b)
    sw = w isa Real ? fill!(KA.allocate(dev, Float64, mD), sqrt(Float64(w))) : sqrt.(vec_on(w))
    hd = h === nothing ? device_zeros(Float64, dev, mD) : vec_on(h)
    x = x0 === nothing ? device_zeros(Float64, dev, n) : copy(vec_on(x0))
    sys = _StackedNormal(op, sw, similar(bd), similar(hd), similar(bd), similar(hd), similar(x))
    # the reference: Aᵀb + DᵀWh
    s = similar(x)
    LinearAlgebra.mul!(s, op.At, bd)
    sys.r2 .= sw .* sw .* hd
    LinearAlgebra.mul!(sys.t, op.Dt, sys.r2); s .+= sys.t
    sref = norm(s)
    # the residuals at x and the normal-equation residual
    LinearAlgebra.mul!(sys.r1, op.A, x); sys.r1 .= bd .- sys.r1
    LinearAlgebra.mul!(sys.r2, op.D, x); sys.r2 .= sw .* (hd .- sys.r2)
    _normal_residual!(s, sys)
    return _normal_cg!(sys, x, s, _jacobi_inverse(diagonal, dev, n), sref, rtol, max_iter)
end

"""
    normal_cg(N, g; diagonal = nothing, x0 = nothing, rtol = 1e-6, max_iter = 500)
        -> (; x, iterations, converged, relres)

The solution of the normal equations `N x = g` of a penalized least squares, `N = AᵀA +
DᵀWD` and `g = Aᵀb + DᵀWh` ([`cgls`](@ref)'s problem), by conjugate gradients on `N`
APPLIED: `N` a function `N(y, x)` writing `y = N x` into `y`, or an operator with `mul!(y,
N, x)` (a matrix, a [`CSRMatrix`](@ref)). The problem whose rows are never stored — a
panorama's pixels accumulated into a lattice stencil and a Schur block — solves here; one
whose rows are stored solves with `cgls`, which applies the rows and so loses no digits to
forming `N`. Both are one iteration with one stopping rule.

`diagonal`, `N`'s diagonal (positive), preconditions the iteration (Jacobi), `nothing`
none; `x0` is a warm start (`nothing`: 0). The vectors live on `g`'s KernelAbstractions
backend (`diagonal` and `x0` moved there; `x` returned on it), and `N` applies there.
The stopping rule is cgls's, relative to the right-hand side: `‖g − N xₖ‖ ≤ rtol · ‖g‖`,
so a warm start at the solution takes 0 iterations. `converged` when it is met within
`max_iter` iterations; `relres` is `‖g − N xₖ‖ / ‖g‖`.
"""
function normal_cg(N, g::AbstractVector; diagonal::Union{Nothing, AbstractVector} = nothing,
        x0::Union{Nothing, AbstractVector} = nothing, rtol::Real = 1.0e-6, max_iter::Integer = 500)
    dev = get_backend(g)
    n = length(g)
    gd = on_device(Float64, dev, g)
    x = x0 === nothing ? device_zeros(Float64, dev, n) : copy(on_device(Float64, dev, x0))
    sys = _AppliedNormal(N, similar(gd))
    s = copy(gd)
    x0 === nothing || (_apply_normal!(sys.Np, N, x); s .-= sys.Np)
    return _normal_cg!(sys, x, s, _jacobi_inverse(diagonal, dev, n), norm(gd), rtol, max_iter)
end

# -----------------------------------------------------------------------------
# The one (preconditioned) conjugate-gradient iteration on the normal equations
# N x = g, N = AᵀA + DᵀWD, shared by cgls and normal_cg. A system supplies the
# curvature pᵀN p of a search direction (keeping its image) and the normal-equation
# residual after a move of α along it: cgls from its stacked residuals [b − Ax;
# √W(h − Dx)], normal_cg from N p. The stopping rule ‖sₖ‖ ≤ rtol·‖g‖ and the
# iteration count are this loop's alone.
# -----------------------------------------------------------------------------

struct _StackedNormal{O <: PenalizedOperator, V <: AbstractVector{Float64}, U <: AbstractVector{Float64}}
    op::O
    sw::U      # √w per penalty row
    r1::V      # b − A x
    r2::U      # √W (h − D x)
    q1::V      # A p
    q2::U      # √W D p
    t::V       # an n-vector of scratch (Dᵀ q2)
end
function _normal_residual!(s, sys::_StackedNormal)
    LinearAlgebra.mul!(s, sys.op.At, sys.r1)
    sys.q2 .= sys.sw .* sys.r2
    LinearAlgebra.mul!(sys.t, sys.op.Dt, sys.q2)
    s .+= sys.t
    return s
end
function _curvature!(sys::_StackedNormal, p)
    LinearAlgebra.mul!(sys.q1, sys.op.A, p)
    LinearAlgebra.mul!(sys.q2, sys.op.D, p); sys.q2 .= sys.sw .* sys.q2
    return dot(sys.q1, sys.q1) + dot(sys.q2, sys.q2)
end
function _descend!(s, sys::_StackedNormal, α)
    sys.r1 .-= α .* sys.q1; sys.r2 .-= α .* sys.q2
    return _normal_residual!(s, sys)
end

struct _AppliedNormal{F, V <: AbstractVector{Float64}}
    N::F
    Np::V      # N p
end
_apply_normal!(y, N::Function, x) = (N(y, x); y)
_apply_normal!(y, N, x) = LinearAlgebra.mul!(y, N, x)
_curvature!(sys::_AppliedNormal, p) = dot(p, _apply_normal!(sys.Np, sys.N, p))
_descend!(s, sys::_AppliedNormal, α) = (s .-= α .* sys.Np; s)

function _jacobi_inverse(diagonal, dev, n)
    diagonal === nothing && return nothing
    length(diagonal) == n || throw(DimensionMismatch("the Jacobi diagonal has $(length(diagonal)) entries, the unknowns $n"))
    d = on_device(Float64, dev, diagonal)
    all(>(0), d) || throw(ArgumentError("the Jacobi diagonal must be positive (the normal matrix's diagonal)"))
    return 1 ./ d
end

function _normal_cg!(sys, x, s, dinv, sref, rtol, max_iter)
    z = dinv === nothing ? s : dinv .* s       # the preconditioned residual (s itself without one)
    p = copy(z)
    γ = dot(s, z); σ = dot(s, s)
    goal = (rtol * sref)^2
    it = 0
    while it < max_iter && σ > goal
        qq = _curvature!(sys, p)
        qq > 0 || break
        α = γ / qq
        x .+= α .* p
        _descend!(s, sys, α)
        dinv === nothing || (z .= dinv .* s)
        γn = dot(s, z); σ = dot(s, s)
        p .= z .+ (γn / γ) .* p
        γ = γn
        it += 1
    end
    return (; x, iterations = it, converged = σ <= goal, relres = sref > 0 ? sqrt(σ) / sref : 0.0)
end

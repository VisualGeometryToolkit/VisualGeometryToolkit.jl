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
# THE LINEAR SOLVE. `_cgls` (min ‖Ax − b‖² + ‖√w∘(Dx − h)‖², warm start, target,
#   any KernelAbstractions backend): Krylov.jl's CGLS on the stacked operator
#   [A; √W D], its products the operators' own — `CSRMatrix`, the device form of a
#   sparse operator, or on CUDA a cuSPARSE CSR matrix (the CUDA extension);
#   `_normal_cg`, Krylov.jl's CG on a normal operator AᵀA + DᵀWD that is only
#   applied (rows never stored); either Jacobi-preconditioned. A dense or
#   CHOLMOD-sized problem solves its normal equations directly.
# THE WEIGHT'S CHOICE. VisualGeometryToolkitEstimation's `GeneralizedCrossValidation`
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
else `KA.CPU()`). `mul!(y, C, x)` sums each row in its stored order, no atomics, the same
bits on every run: on CUDA the product is cuSPARSE's CSR product of these arrays (its
reproducible algorithm; the CUDA extension), elsewhere one work item per row.
`Adapt.adapt(backend, C)` moves it. The operator a device solve (`_cgls`,
`_normal_cg`) applies; its transpose is a second `CSRMatrix` of `transpose(A)`
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
_transposed_csr(A::SparseMatrixCSC; device) = (I = _csr_index_type(device);
    CSRMatrix(size(A, 2), size(A, 1), ka_upload(I, device, A.colptr), ka_upload(I, device, A.rowval), ka_upload(Float64, device, A.nzval)))
# The integers an uploaded operator's indices are stored in on `device`: an accelerator extension states its sparse library's
# (CUDA: cuSPARSE's Int32, so the operator's arrays ARE a cuSPARSE matrix's and its product is the library's).
_csr_index_type(device) = Int
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
    OnBackend(get_backend(y))(_csr_mul_kernel!)(y, C.rowptr, C.colval, C.nzval, x; ndrange = C.m)
    return y
end

"""
    PenalizedOperator(A::SparseMatrixCSC, D::SparseMatrixCSC = spzeros(0, size(A, 2)); device = default_device())

The operators of a penalized least squares `min ‖A x − b‖² + ‖√w ∘ (D x − h)‖²` on a
KernelAbstractions `device`, built once for every solve on them (`_cgls`): `A`,
`D` and their transposes as [`CSRMatrix`](@ref)es. `D` is any of the stack's penalty
operators ([`difference_rows`](@ref), [`bending_rows`](@ref) as a sparse matrix), its rows'
weights `w` the solve's.
"""
struct PenalizedOperator{CA <: CSRMatrix, CD <: CSRMatrix, B}
    A::CA
    At::CA
    D::CD
    Dt::CD
    device::B
end
function PenalizedOperator(A::SparseMatrixCSC, D::SparseMatrixCSC = SparseMatrixCSC{Float64, Int}(undef, 0, size(A, 2)); device = default_device())
    size(D, 2) == size(A, 2) || throw(DimensionMismatch("PenalizedOperator: A has $(size(A, 2)) columns, D $(size(D, 2))"))
    # a transpose's CSR form is the operator's own compressed columns
    return PenalizedOperator(CSRMatrix(A; device), _transposed_csr(A; device), CSRMatrix(D; device), _transposed_csr(D; device), device)
end

"""
    _cgls(op::PenalizedOperator, b; w = 1, h = nothing, diagonal = nothing, x0 = nothing, rtol = 1e-6, max_iter = 500)
        -> (; x, iterations, converged, relres)

The least-squares solution of `min_x ‖A x − b‖² + ‖√w ∘ (D x − h)‖²` (`op`'s `A` and `D`;
`w` a weight per penalty row, or one for all; `h` the penalty's target, `nothing` for 0)
by CGLS (Krylov.jl's `cgls`) — conjugate gradients on the normal equations `(AᵀA + DᵀWD)
x = Aᵀb + DᵀWh` applied as the stacked operator `S = [A; √W D]`, never formed — from `x0`
(a start: the iteration solves for the correction to it, so a direction neither `A` nor
`D` holds keeps `x0`'s value; `nothing`: 0), on `op`'s device (the vectors moved there;
`x` returned on it). Every product is a row product of a stored operator (`A`, `D` and
their transposes as rows of their own: no transpose product, so the result is the same on
every run).

`diagonal`, the normal matrix's diagonal `diag(AᵀA + DᵀWD)` (positive), preconditions the
iteration (Jacobi: the unknowns' columns scaled to unit norm, `S C` with `C =
diagonal^-½`), `nothing` none.

The stopping rule is relative to the normal equations' right-hand side, `‖C sₖ‖ ≤ rtol ·
‖C (Aᵀb + DᵀWh)‖`, `sₖ = Aᵀ(b − A xₖ) + DᵀW(h − D xₖ)` the normal-equation residual (`C`
the Jacobi scaling, 1 without one): a reference fixed by the problem, not by the start,
so a start that is already close needs few or no iterations. `converged` when the rule is
met within `max_iter` iterations; `relres` is the left side over the right side's norm.

The rows of `A` stored; a problem whose normal operator is only applied (its rows never
formed) solves with `_normal_cg`.
"""
function _cgls(op::PenalizedOperator, b::AbstractVector; w = 1.0, h::Union{Nothing, AbstractVector} = nothing,
        diagonal::Union{Nothing, AbstractVector} = nothing, x0::Union{Nothing, AbstractVector} = nothing,
        rtol::Real = 1.0e-6, max_iter::Integer = 500)
    dev = op.device
    n = op.A.n; m = op.A.m; mD = op.D.m
    length(b) == m || throw(DimensionMismatch("_cgls: b has $(length(b)) entries, A $m rows"))
    vec_on(v) = on_device(Float64, dev, v)
    sw = w isa Real ? fill!(KA.allocate(dev, Float64, mD), sqrt(Float64(w))) : sqrt.(vec_on(w))
    dinv = _jacobi_inverse(diagonal, dev, n)
    S = _StackedOperator(map(_solve_operator, (op.A, op.At, op.D, op.Dt))..., sw, dinv === nothing ? nothing : sqrt.(dinv),
                         device_zeros(Float64, dev, n), similar(sw))
    rhs = device_zeros(Float64, dev, m + mD)                       # [b; √W h]
    copyto!(view(rhs, 1:m), vec_on(b))
    h === nothing || (view(rhs, (m + 1):(m + mD)) .= sw .* vec_on(h))
    g = device_zeros(Float64, dev, n)
    sref = norm(LinearAlgebra.mul!(g, S', rhs))                    # the reference: ‖C (Aᵀb + DᵀWh)‖
    x = x0 === nothing ? device_zeros(Float64, dev, n) : copy(vec_on(x0))
    if x0 !== nothing                                              # the correction's right-hand side: [b; √W h] − S x0
        r = similar(rhs)
        LinearAlgebra.mul!(r, S, S.c === nothing ? x : x ./ S.c)
        rhs .-= r
    end
    goal = Float64(rtol) * sref
    max_iter >= 1 || (s = norm(LinearAlgebra.mul!(g, S', rhs)); return (; x, iterations = 0, converged = s <= goal, relres = sref > 0 ? s / sref : 0.0))
    z, stats = Krylov.cgls(S, rhs; atol = goal, rtol = 0.0, itmax = Int(max_iter), history = true)
    x .+= S.c === nothing ? z : S.c .* z
    s = Float64(last(stats.Aresiduals))                            # ‖C sₖ‖ of the last iterate
    return (; x, iterations = Int(stats.niter), converged = s <= goal, relres = sref > 0 ? s / sref : 0.0)
end

# The operator a solve applies for a stored sparse operator: the `CSRMatrix` itself (its row kernel) — an accelerator
# extension returns its library's matrix of the same arrays (CUDA: a cuSPARSE CSR matrix under the reproducible product).
_solve_operator(C::CSRMatrix) = C

# S = [A; √W D] C of a penalized least squares and its adjoint, each a row product of a stored operator (`A`, `D`, and
# their transposes `At`, `Dt` as operators of their own); `C` the Jacobi column scaling (`nothing`: none). The operator
# Krylov.jl's solvers take: `size`, `eltype`, `adjoint` and `mul!`.
struct _StackedOperator{OA, OD, U <: AbstractVector{Float64}, C <: Union{Nothing, AbstractVector{Float64}}, V <: AbstractVector{Float64}}
    A::OA
    At::OA
    D::OD
    Dt::OD
    sw::U      # √w per penalty row
    c::C       # the unknowns' scaling
    t::V       # an n-vector of scratch
    q::U       # a penalty-row vector of scratch
end
struct _StackedAdjoint{S <: _StackedOperator}
    S::S
end
Base.size(S::_StackedOperator) = (size(S.A, 1) + length(S.sw), size(S.A, 2))
Base.size(T::_StackedAdjoint) = reverse(size(T.S))
Base.size(S::Union{_StackedOperator, _StackedAdjoint}, d::Integer) = d <= 2 ? size(S)[d] : 1
Base.eltype(::Union{_StackedOperator, _StackedAdjoint}) = Float64
Base.adjoint(S::_StackedOperator) = _StackedAdjoint(S)
Base.adjoint(T::_StackedAdjoint) = T.S
function LinearAlgebra.mul!(y::AbstractVector, S::_StackedOperator, x::AbstractVector)
    m = size(S.A, 1); mD = length(S.sw)
    xs = S.c === nothing ? x : (S.t .= S.c .* x; S.t)
    m > 0 && LinearAlgebra.mul!(view(y, 1:m), S.A, xs)
    if mD > 0
        LinearAlgebra.mul!(S.q, S.D, xs)
        view(y, (m + 1):(m + mD)) .= S.sw .* S.q
    end
    return y
end
function LinearAlgebra.mul!(x::AbstractVector, T::_StackedAdjoint, y::AbstractVector)
    S = T.S; m = size(S.A, 1); mD = length(S.sw)
    m > 0 ? LinearAlgebra.mul!(x, S.At, view(y, 1:m)) : fill!(x, 0)
    if mD > 0
        S.q .= S.sw .* view(y, (m + 1):(m + mD))
        LinearAlgebra.mul!(S.t, S.Dt, S.q)
        x .+= S.t
    end
    S.c === nothing || (x .*= S.c)
    return x
end

"""
    _normal_cg(N, g; diagonal = nothing, x0 = nothing, rtol = 1e-6, max_iter = 500)
        -> (; x, iterations, converged, relres)

The solution of the normal equations `N x = g` of a penalized least squares, `N = AᵀA +
DᵀWD` and `g = Aᵀb + DᵀWh` (`_cgls`'s problem), by conjugate gradients (Krylov.jl's
`cg`) on `N` APPLIED: `N` a function `N(y, x)` writing `y = N x` into `y`, or an operator
with `mul!(y, N, x)` (a matrix, a [`CSRMatrix`](@ref)). The problem whose rows are never
stored — a panorama's pixels accumulated into a lattice stencil and a Schur block — solves
here; one whose rows are stored solves with `_cgls`, which applies the rows and so loses no
digits to forming `N`.

`diagonal`, `N`'s diagonal (positive), preconditions the iteration (Jacobi: the symmetric
scaling `C N C`, `C = diagonal^-½`), `nothing` none; `x0` is a start (`nothing`: 0; the
iteration solves for the correction to it). The vectors live on `g`'s KernelAbstractions
backend (`diagonal` and `x0` moved there; `x` returned on it), and `N` applies there. The
stopping rule is relative to the right-hand side, `‖C (g − N xₖ)‖ ≤ rtol · ‖C g‖` (`C` the
Jacobi scaling, 1 without one), so a start at the solution takes 0 iterations. `converged`
when it is met within `max_iter` iterations; `relres` is the left side over the right
side's norm.
"""
function _normal_cg(N, g::AbstractVector; diagonal::Union{Nothing, AbstractVector} = nothing,
        x0::Union{Nothing, AbstractVector} = nothing, rtol::Real = 1.0e-6, max_iter::Integer = 500)
    dev = get_backend(g)
    n = length(g)
    gd = on_device(Float64, dev, g)
    dinv = _jacobi_inverse(diagonal, dev, n)
    S = _ScaledNormal(N isa CSRMatrix ? _solve_operator(N) : N, dinv === nothing ? nothing : sqrt.(dinv), similar(gd), n)
    scaled(v) = S.c === nothing ? v : S.c .* v
    sref = norm(scaled(gd))                                        # the reference: ‖C g‖
    goal = Float64(rtol) * sref
    x = x0 === nothing ? device_zeros(Float64, dev, n) : copy(on_device(Float64, dev, x0))
    rhs = copy(gd)                                                 # the correction's right-hand side: C (g − N x0)
    x0 === nothing || (_apply_normal!(S.t, S.N, x); rhs .-= S.t)
    S.c === nothing || (rhs .*= S.c)
    result(s, iterations) = (; x, iterations, converged = s <= goal, relres = sref > 0 ? s / sref : 0.0)
    (max_iter >= 1 && norm(rhs) > goal) || return result(norm(rhs), 0)
    z, stats = Krylov.cg(S, rhs; atol = goal, rtol = 0.0, itmax = Int(max_iter), history = true)
    x .+= scaled(z)
    return result(Float64(last(stats.residuals)), Int(stats.niter))
end

# C N C of an applied normal operator `N` (a function `N(y, x)`, or an operator with `mul!`) under the Jacobi scaling `C`
# (`nothing`: none): the symmetric operator Krylov.jl's `cg` takes (`size`, `eltype`, `mul!`).
struct _ScaledNormal{F, C <: Union{Nothing, AbstractVector{Float64}}, V <: AbstractVector{Float64}}
    N::F
    c::C
    t::V       # an n-vector of scratch
    n::Int
end
Base.size(S::_ScaledNormal) = (S.n, S.n)
Base.size(S::_ScaledNormal, d::Integer) = d <= 2 ? S.n : 1
Base.eltype(::_ScaledNormal) = Float64
_apply_normal!(y, N::Function, x) = (N(y, x); y)
_apply_normal!(y, N, x) = LinearAlgebra.mul!(y, N, x)
function LinearAlgebra.mul!(y::AbstractVector, S::_ScaledNormal, z::AbstractVector)
    S.c === nothing && return _apply_normal!(y, S.N, z)
    S.t .= S.c .* z
    _apply_normal!(y, S.N, S.t)
    y .*= S.c
    return y
end

function _jacobi_inverse(diagonal, dev, n)
    diagonal === nothing && return nothing
    length(diagonal) == n || throw(DimensionMismatch("the Jacobi diagonal has $(length(diagonal)) entries, the unknowns $n"))
    d = on_device(Float64, dev, diagonal)
    all(>(0), d) || throw(ArgumentError("the Jacobi diagonal must be positive (the normal matrix's diagonal)"))
    return 1 ./ d
end

# =============================================================================
# The front end — the penalized least squares as problems solved through `solve`
# =============================================================================

"""
    ToleranceStops(; max_iter = 500, rtol = 1e-6)

The stop settings of an iteration with an iteration cap and a relative tolerance — the stack's Krylov least squares
([`CGLS`](@ref), [`NormalCG`](@ref)) and a bisection: the solve keyword `stops`. `max_iter` is the iteration cap
(`0`: no iteration, the start judged as it is); `rtol` the relative tolerance the solver states (CGLS and CG: the
normal-equation residual relative to the right-hand side's). Isbits.
"""
struct ToleranceStops
    max_iter::Int
    rtol::Float64
end
function ToleranceStops(; max_iter::Integer = 500, rtol::Real = 1.0e-6)
    max_iter >= 0 || throw(ArgumentError("ToleranceStops: max_iter is a count of iterations, ≥ 0; got $max_iter"))
    rtol >= 0 || throw(ArgumentError("ToleranceStops: rtol is a tolerance, ≥ 0; got $rtol"))
    return ToleranceStops(Int(max_iter), Float64(rtol))
end

# The parent of the numerics' problems: solved by a fit-kind algorithm (iterative, from its own start).
abstract type _NumericsProblem <: AbstractEstimationProblem end
@inline _check_kind(::_NumericsProblem, ::FitKind, alg) = nothing
@inline _check_kind(::_NumericsProblem, ::EstimatorKind, alg) = throw(ArgumentError(
    "$(nameof(typeof(alg))) is no solver of a numerics problem: CGLS() solves a PenalizedLeastSquaresProblem, " *
        "NormalCG() a NormalEquationsProblem, GridBrent() a BracketedMinimizationProblem"))

"""
    PenalizedLeastSquaresProblem(op::PenalizedOperator, b; w = 1, h = nothing, diagonal = nothing, u0 = nothing)

The penalized least squares `min_x ‖A x − b‖² + ‖√w ∘ (D x − h)‖²` of `op`'s `A` and `D` (`w` a weight per penalty row,
or one for all; `h` the penalty's target, `nothing` for 0; `diagonal` the normal matrix's diagonal `diag(AᵀA + DᵀWD)`,
the Jacobi preconditioner, `nothing` none; `u0` a start, `nothing` for 0). Solved by [`CGLS`](@ref)`()`:
`solve(PenalizedLeastSquaresProblem(op, b; w), CGLS(); stops = ToleranceStops(; rtol, max_iter))`.
"""
struct PenalizedLeastSquaresProblem{O, B, W, H, D, U} <: _NumericsProblem
    op::O
    b::B
    w::W
    h::H
    diagonal::D
    u0::U
end
PenalizedLeastSquaresProblem(op::PenalizedOperator, b::AbstractVector; w = 1.0, h = nothing, diagonal = nothing, u0 = nothing) =
    PenalizedLeastSquaresProblem(op, b, w, h, diagonal, u0)

"""
    NormalEquationsProblem(N, g; diagonal = nothing, u0 = nothing)

The normal equations `N x = g` of a penalized least squares whose rows are never stored, `N` APPLIED (a function
`N(y, x)` writing `y = N x`, or an operator with `mul!`), `diagonal` its diagonal (the Jacobi preconditioner, `nothing`
none), `u0` a start (`nothing`: 0). Solved by [`NormalCG`](@ref)`()`.
"""
struct NormalEquationsProblem{N, G, D, U} <: _NumericsProblem
    N::N
    g::G
    diagonal::D
    u0::U
end
NormalEquationsProblem(N, g::AbstractVector; diagonal = nothing, u0 = nothing) = NormalEquationsProblem(N, g, diagonal, u0)

"""
    CGLS()
    NormalCG()

The stack's Krylov solvers of a penalized least squares, fit-kind algorithms: `CGLS()` (Krylov.jl's `cgls` on the
stacked operator `[A; √W D]`, its rows applied, never the normal matrix formed) solves a
[`PenalizedLeastSquaresProblem`](@ref); `NormalCG()` (Krylov.jl's `cg` on the applied normal operator) a
[`NormalEquationsProblem`](@ref). Both stop at `stops = `[`ToleranceStops`](@ref)`(; rtol, max_iter)` — the
normal-equation residual relative to the right-hand side's, `‖C sₖ‖ ≤ rtol · ‖C g‖` (`C` the Jacobi scaling) — and
return a `FitResult`: `u` the solution (on the operator's device), `(Success, ToleranceMet)` when the rule is met
within the cap, else `(MaxIters, IterationCap)`, `stats.nsteps` the iterations, `metadata = (; relres)` the final
relative residual.
"""
struct CGLS <: AbstractEstimationAlgorithm end
@doc (@doc CGLS) struct NormalCG <: AbstractEstimationAlgorithm end
estimator_kind(::CGLS) = FitKind()
estimator_kind(::NormalCG) = FitKind()

_krylov_result(s) = FitResult(s.x, s.converged ? ReturnCode.Success : ReturnCode.MaxIters, s.converged ? ToleranceMet : IterationCap,
    SolveStats(; nsteps = s.iterations), NaN, 0, (; relres = s.relres))
__solve(prob::PenalizedLeastSquaresProblem, ::CGLS; stops::ToleranceStops = ToleranceStops()) =
    _krylov_result(_cgls(prob.op, prob.b; prob.w, prob.h, prob.diagonal, x0 = prob.u0, rtol = stops.rtol, max_iter = stops.max_iter))
__solve(prob::NormalEquationsProblem, ::NormalCG; stops::ToleranceStops = ToleranceStops()) =
    _krylov_result(_normal_cg(prob.N, prob.g; prob.diagonal, x0 = prob.u0, rtol = stops.rtol, max_iter = stops.max_iter))

public ToleranceStops, PenalizedLeastSquaresProblem, NormalEquationsProblem, CGLS, NormalCG

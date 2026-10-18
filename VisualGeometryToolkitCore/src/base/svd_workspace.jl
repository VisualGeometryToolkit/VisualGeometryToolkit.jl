# =============================================================================
# SVD Workspace — Pre-allocated buffers for zero-alloc LAPACK gesdd!
# =============================================================================
#
# Used by homography_solvers.jl and fundmat.jl for zero-allocation
# SVD null-space computation. Also used by VisualGeometryToolkitEstimation for RANSAC.
#
# =============================================================================

"""
    SvdWorkspace{T}(max_rows, ncols)

Pre-allocated workspace for in-place SVD via LAPACK `gesdd!` (divide-and-conquer).
Eliminates all heap allocations from repeated SVD calls on matrices up to
`max_rows × ncols`.

`Float64` only: the workspace calls the double-precision `dgesdd` directly, so
`SvdWorkspace{Float64}` is the one constructible instance (the type parameter
stays so that a caller's `SvdWorkspace{T}` field type still reads).
"""
struct SvdWorkspace{T <: AbstractFloat}
    U::Matrix{T}
    S::Vector{T}
    Vt::Matrix{T}
    work::Vector{T}
    iwork::Vector{LinearAlgebra.BlasInt}
end

function _gesdd_lwork(::Type{T}, jobz::Char, m::Int, n::Int) where {T <: Float64}
    m >= 1 || return 1
    work_query = Vector{T}(undef, 1)
    U = Matrix{T}(undef, 1, 1); Vt = Matrix{T}(undef, 1, 1); A = Matrix{T}(undef, 1, 1)
    S = Vector{T}(undef, 1); iwork = Vector{LinearAlgebra.BlasInt}(undef, 1)
    info = Ref{LinearAlgebra.BlasInt}(0)
    ccall(
        (LinearAlgebra.BLAS.@blasfunc(dgesdd_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
        (
            Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{T},
            Ref{LinearAlgebra.BlasInt}, Ptr{T}, Ptr{T}, Ref{LinearAlgebra.BlasInt},
            Ptr{T}, Ref{LinearAlgebra.BlasInt}, Ptr{T}, Ref{LinearAlgebra.BlasInt},
            Ptr{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Clong,
        ),
        jobz, m, n, A, max(1, m), S, U, max(1, m),
        Vt, n, work_query, LinearAlgebra.BlasInt(-1), iwork, info, 1
    )
    LinearAlgebra.LAPACK.chklapackerror(info[])
    return round(LinearAlgebra.BlasInt, nextfloat(real(work_query[1])))
end

function SvdWorkspace{T}(max_rows::Int, ncols::Int) where {T <: Float64}
    U = Matrix{T}(undef, max_rows, ncols)
    S = Vector{T}(undef, ncols)
    Vt = Matrix{T}(undef, ncols, ncols)
    iwork = Vector{LinearAlgebra.BlasInt}(undef, 8 * ncols)
    # Query the optimal workspace size for both job kinds svd_nullvec! uses:
    # 'S' for a tall-enough system, 'A' for a wide one (fewer rows than columns,
    # whose null vector the thin factorization does not compute). The wide
    # case's requirement grows with the row count, so its largest instance
    # (ncols - 1 rows, or max_rows if fewer) bounds every smaller one.
    lwork = max(
        _gesdd_lwork(T, 'S', max_rows, ncols),
        _gesdd_lwork(T, 'A', min(max_rows, ncols - 1), ncols),
    )
    work = Vector{T}(undef, lwork)
    return SvdWorkspace{T}(U, S, Vt, work, iwork)
end

"""
    svd_nullvec!(ws::SvdWorkspace, A, m, Val(N)) -> SVector{N}

Compute null space vector (last row of Vt) of the first `m` rows of `A`
(with `N` columns) using pre-allocated workspace. Zero allocations.
`A` is overwritten by LAPACK. `N` must be a compile-time constant via `Val`.

A wide system (`m < N`, e.g. the 8 × 9 design matrix of a 4-point homography
or an 8-point fundamental DLT) is factored in full (`JOBZ = 'A'`): its thin
factorization has only `m` right singular vectors, none of them the null
vector. The singular values it does not have are zeroed, so
[`dlt_conditioning`](@ref) reads such a system as rank-deficient only when it
is. A LAPACK failure throws `LAPACKException`.

# Example
```julia
ws = SvdWorkspace{Float64}(100, 9)
v = svd_nullvec!(ws, A, size(A,1), Val(9))  # -> SVector{9,Float64}
```
"""
function svd_nullvec!(ws::SvdWorkspace{T}, A::AbstractMatrix{T}, m::Int, ::Val{N}) where {T <: Float64, N}
    info = Ref{LinearAlgebra.BlasInt}(0)
    lda = stride(A, 2)   # stride, not size — handles SubArray views correctly
    ldu = size(ws.U, 1)
    # thin ('S') keeps only min(m, N) right singular vectors: a wide system's
    # null vector needs the full factorization
    jobz = m < N ? 'A' : 'S'
    m < N && fill!(view(ws.S, (m + 1):N), zero(T))
    ccall(
        (LinearAlgebra.BLAS.@blasfunc(dgesdd_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
        (
            Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{T},
            Ref{LinearAlgebra.BlasInt}, Ptr{T}, Ptr{T}, Ref{LinearAlgebra.BlasInt},
            Ptr{T}, Ref{LinearAlgebra.BlasInt}, Ptr{T}, Ref{LinearAlgebra.BlasInt},
            Ptr{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Clong,
        ),
        jobz, m, N, A, max(1, lda), ws.S, ws.U, max(1, ldu),
        ws.Vt, N, ws.work, length(ws.work), ws.iwork, info, 1
    )
    LinearAlgebra.LAPACK.chklapackerror(info[])
    @inbounds return SVector{N, T}(ntuple(i -> ws.Vt[N, i], Val(N)))
end

"""
    _dlt_nullvec(A, m, svd_ws) -> AbstractVector (9 entries)

The null vector of the first `m` rows of a 9-column DLT design matrix `A`: the
ONE null-vector path of `_dlt!`. With a `SvdWorkspace` and a `Float64` `A` it is
[`svd_nullvec!`](@ref) (zero allocations, `A` overwritten, and the singular
values left in the workspace for [`dlt_conditioning`](@ref)). Without a
workspace, or for another eltype (the workspace calls `dgesdd`, which is
`Float64` only), it is an allocating SVD of the compacted rows, `A` untouched;
that SVD is full only when the system is wide (the minimum's 8 × 9, whose thin
`Vt` has no null row), since a tall system's full `U` is m².
"""
_dlt_nullvec(A::AbstractMatrix{Float64}, m::Int, svd_ws::SvdWorkspace{Float64}) =
    svd_nullvec!(svd_ws, A, m, Val(9))
_dlt_nullvec(A::AbstractMatrix, m::Int, ::Union{Nothing, SvdWorkspace}) =
    svd(view(A, 1:m, :); full = m < 9).Vt[end, :]

"""
    dlt_conditioning(ws::SvdWorkspace; null_dim::Int = 1) -> Float64

The DLT design matrix's null-space gap from the solve that filled `ws`:
`S[1] / max(S[n - null_dim], eps())`, the largest singular value over the
smallest NON-null one (`n = length(ws.S)`; for the 9-column homography and
fundamental systems, `S[1]/S[8]`). The `eps()` floor is load-bearing: an
exactly rank-deficient design matrix returns an enormous ratio rather than
dividing by zero, which is how a collinear correspondence set announces
itself.

A MEASUREMENT, not a policy: it neither rejects nor alters the solve. Pair it
with [`test_dlt_conditioning`](@ref), whose default `100.0` is the one
threshold: VisualGeometryToolkitEstimation's `refine!(::DltWorkspace, …)` refuses a
relative-family hypothesis by calling it at that default.

ONLY MEANINGFUL IN THE COORDINATES THE SOLVE USED. The ratio is a property of
the design matrix as built, so it depends on the normalization AND on the
weights column that went into it. VGC's public DLT forms build a
`HartleyNormalization`, which is what makes a threshold of `100.0` mean the
same thing here as in VisualGeometryToolkitEstimation — a caller that solves in another
frame must recalibrate its threshold rather than carry this one across.

REFLECTS THE LAST SOLVE that filled this workspace, and nothing checks that
for you: read it immediately after the `homography_dlt!`/`fundamental_dlt!`
call whose matrix you mean. There is deliberately no staleness flag — adding
one means a field on `SvdWorkspace`, which VisualGeometryToolkitEstimation constructs
too, so it would turn a local guard into a cross-repo API change and a
lockstep window, to protect against a mistake this sentence prevents.
"""
@inline function dlt_conditioning(ws::SvdWorkspace{T}; null_dim::Int = 1) where {T}
    n = length(ws.S)
    @boundscheck (1 <= n - null_dim <= n) || throw(ArgumentError("null_dim = $null_dim out of range for $n singular values"))
    @inbounds return Float64(ws.S[1] / max(ws.S[n - null_dim], eps(T)))
end

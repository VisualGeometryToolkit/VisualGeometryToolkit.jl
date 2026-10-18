# =============================================================================
# In-register small LU — one partial-pivot solve for every fixed-size system
# =============================================================================

"""
    lu_solve!(A::MMatrix{N, N}, b::MVector{N}) -> Int32

Solve `A x = b` in place by LU with partial pivoting — `A` is overwritten with
its factors and the solution replaces `b`. Zero-allocation on static
mutable arrays (StaticArrays' generic `lu` falls back to a heap path at these
sizes), and pure register arithmetic, so it runs inside a KernelAbstractions
kernel on any backend.

Returns `0`, or `k` when the `k`-th pivot is EXACTLY zero — mirroring LAPACK
getrf's `info`, which a batched driver uses to invalidate degenerate systems
rather than propagate `Inf`s. Hoisted verbatim from BlobBoards' 6×6 adaptation
solve (2026-09); its op sequence is the one the CPU adaptation path is pinned
to.

(`_e5_solve10!` in the native 5-point essential solver is a multi-right-hand-side
elimination that multiplies by the reciprocal pivot on back-substitution, ported
from Eigen for fidelity to PoseLib; it stays separate because unifying the two
would change one of them at the ulp level.)
"""
@inline function lu_solve!(A::MMatrix{N, N}, b::MVector{N}) where {N}
    T = eltype(A)
    @inbounds for k in 1:N
        p = k
        amax = abs(A[k, k])
        for i in (k + 1):N
            a = abs(A[i, k])
            if a > amax
                amax = a
                p = i
            end
        end
        A[p, k] == zero(T) && return Int32(k)
        if p != k
            for j in 1:N
                tmp = A[k, j]
                A[k, j] = A[p, j]
                A[p, j] = tmp
            end
            tmp = b[k]
            b[k] = b[p]
            b[p] = tmp
        end
        inv_kk = one(T) / A[k, k]
        for i in (k + 1):N
            l = A[i, k] * inv_kk
            A[i, k] = l
            for j in (k + 1):N
                A[i, j] -= l * A[k, j]
            end
            b[i] -= l * b[k]
        end
    end
    @inbounds for k in N:-1:1
        for j in (k + 1):N
            b[k] -= A[k, j] * b[j]
        end
        b[k] /= A[k, k]
    end
    return Int32(0)
end

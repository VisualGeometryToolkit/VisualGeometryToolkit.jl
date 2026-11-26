# fused_products.jl — explicit fused products: one rounding order in every compiled copy
#
# `muladd`, StaticArrays' small products (built on `muladd`) and its `dot`
# (`@simd`) let LLVM fuse a multiply and an add, or not, as it sees fit — and it
# decides per compiled copy. A root inlined into two callers (two hits of one
# surface in one function) then converged 2–3 ulp apart on one ray
# (MappedSurface, Julia 1.12 and 1.13; test/mapped_surface_determinism_tests.jl),
# and the epipolar constraint's products rounded differently in the Sampson
# residual a scorer reads and in the Jacobian row an LM reads (measured on an
# essential matrix's bearings: 243 of 300 residuals apart in their last bits,
# Core 0.14). These forms leave LLVM no choice: every product-sum is an explicit
# `fma` chain in a fixed order — entry (i, j) of A·B is
# `fma(A[i, K], B[K, j], … fma(A[i, 2], B[2, j], A[i, 1] * B[1, j]))`, the terms in
# index order, the first a plain product (StaticArrays' order) — an `fma` is one
# rounding on every target, the device included, and everything else is plain
# arithmetic, which Julia never contracts on the CPU.
#
# Every body is INLINED (`Expr(:meta, :inline)`): `Base.fma(::Float64, …)` keeps
# its `have_fma` branch and the `fma_emulated` call in its typed body, so the
# inliner prices every fma as a call and would leave even a four-term product a
# call. A call per product kept the Sampson Jacobian rows' loop from vectorizing
# (an essential matrix's LO 2× slower, Core 0.14 before this). Inlined, the
# product is the same chain of the same fmas, so its bits do not depend on where
# it is compiled.
#
# Who computes with them: the Sampson layer (base/sampson.jl) and every
# constraint it reads (`constraint_with_jacobians`, the observation-gradient
# contractions, the whitening), the spline lenses' 2-D LM (lm_core.jl), the
# MappedSurface root (mapped_surface.jl, target_surfaces.jl) and a Quad's plane
# coordinates (surfaces.jl).
@generated function _fma_dot(a::StaticVector{N}, b::StaticVector{N}) where {N}
    ex = :(a[1] * b[1])
    for k in 2:N
        ex = :(fma(a[$k], b[$k], $ex))
    end
    return Expr(:block, Expr(:meta, :inline), :(@inbounds $ex))
end
# A B, as one fused chain per entry (column-major)
@generated function _fma_mul(A::StaticMatrix{M, K}, B::StaticMatrix{K, N}) where {M, K, N}
    entry(i, j) = foldl((ex, k) -> :(fma(A[$i, $k], B[$k, $j], $ex)), 2:K; init = :(A[$i, 1] * B[1, $j]))
    return Expr(:block, Expr(:meta, :inline), :(@inbounds SMatrix{$M, $N}($([entry(i, j) for j in 1:N for i in 1:M]...))))
end
# A x
@generated function _fma_mul(A::StaticMatrix{M, K}, x::StaticVector{K}) where {M, K}
    entry(i) = foldl((ex, k) -> :(fma(A[$i, $k], x[$k], $ex)), 2:K; init = :(A[$i, 1] * x[1]))
    return Expr(:block, Expr(:meta, :inline), :(@inbounds SVector{$M}($([entry(i) for i in 1:M]...))))
end
# Aᵀ B and Aᵀ x
@generated function _fma_tmul(A::StaticMatrix{K, M}, B::StaticMatrix{K, N}) where {M, K, N}
    entry(i, j) = foldl((ex, k) -> :(fma(A[$k, $i], B[$k, $j], $ex)), 2:K; init = :(A[1, $i] * B[1, $j]))
    return Expr(:block, Expr(:meta, :inline), :(@inbounds SMatrix{$M, $N}($([entry(i, j) for j in 1:N for i in 1:M]...))))
end
@generated function _fma_tmul(A::StaticMatrix{K, M}, x::StaticVector{K}) where {M, K}
    entry(i) = foldl((ex, k) -> :(fma(A[$k, $i], x[$k], $ex)), 2:K; init = :(A[1, $i] * x[1]))
    return Expr(:block, Expr(:meta, :inline), :(@inbounds SVector{$M}($([entry(i) for i in 1:M]...))))
end

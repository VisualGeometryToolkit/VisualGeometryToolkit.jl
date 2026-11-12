# fused_products.jl — explicit fused products: one rounding order in every compiled copy
#
# `muladd`, StaticArrays' small products (built on `muladd`) and its `dot`
# (`@simd`) let LLVM fuse a multiply and an add, or not, as it sees fit — and it
# decides per compiled copy. A root inlined into two callers (two hits of one
# surface in one function) then converged 2–3 ulp apart on one ray
# (MappedSurface, Julia 1.12 and 1.13; test/mapped_surface_determinism_tests.jl).
# These forms leave LLVM no choice: every product-sum is an explicit `fma`
# chain in a fixed order (an `fma` is one rounding on every target, the device
# included), and everything else plain arithmetic, which Julia never contracts.
# The spline lenses' 2-D LM (spline_lens.jl), the MappedSurface root
# (mapped_surface.jl, target_surfaces.jl) and a Quad's plane coordinates
# (surfaces.jl) compute with them, so they stay inlined (on the device too) and give
# one answer.
@generated function _fma_dot(a::StaticVector{N}, b::StaticVector{N}) where {N}
    ex = :(a[1] * b[1])
    for k in 2:N
        ex = :(fma(a[$k], b[$k], $ex))
    end
    return :(@inbounds $ex)
end
# A B, as one fused chain per entry (column-major)
@generated function _fma_mul(A::StaticMatrix{M, K}, B::StaticMatrix{K, N}) where {M, K, N}
    entry(i, j) = foldl((ex, k) -> :(fma(A[$i, $k], B[$k, $j], $ex)), 2:K; init = :(A[$i, 1] * B[1, $j]))
    return :(@inbounds SMatrix{$M, $N}($([entry(i, j) for j in 1:N for i in 1:M]...)))
end
# A x
@generated function _fma_mul(A::StaticMatrix{M, K}, x::StaticVector{K}) where {M, K}
    entry(i) = foldl((ex, k) -> :(fma(A[$i, $k], x[$k], $ex)), 2:K; init = :(A[$i, 1] * x[1]))
    return :(@inbounds SVector{$M}($([entry(i) for i in 1:M]...)))
end
# Aᵀ B and Aᵀ x
@generated function _fma_tmul(A::StaticMatrix{K, M}, B::StaticMatrix{K, N}) where {M, K, N}
    entry(i, j) = foldl((ex, k) -> :(fma(A[$k, $i], B[$k, $j], $ex)), 2:K; init = :(A[1, $i] * B[1, $j]))
    return :(@inbounds SMatrix{$M, $N}($([entry(i, j) for j in 1:N for i in 1:M]...)))
end
@generated function _fma_tmul(A::StaticMatrix{K, M}, x::StaticVector{K}) where {M, K}
    entry(i) = foldl((ex, k) -> :(fma(A[$k, $i], x[$k], $ex)), 2:K; init = :(A[1, $i] * x[1]))
    return :(@inbounds SVector{$M}($([entry(i) for i in 1:M]...)))
end

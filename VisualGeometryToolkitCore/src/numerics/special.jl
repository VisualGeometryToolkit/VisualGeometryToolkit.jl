# =============================================================================
# Device-safe special functions — branch-free, +*/log only
# =============================================================================
#
# The Student-t adaptation kernels need lgamma, digamma and trigamma INSIDE a
# KernelAbstractions kernel, on any backend and at any float width. SpecialFunctions
# is host-only, and CUDA's libdevice `__nv_lgamma` exists only there (the CUDA
# extension of the consumer substitutes it for bit-exactness with legacy
# kernels). These are the portable defaults: pure arithmetic, identical op
# sequence everywhere. Hoisted from BlobBoards' adapt kernels (2026-09).
# =============================================================================

"""
    lgamma_lanczos(x::AbstractFloat)

log Γ(x) by the Lanczos approximation (g = 7, n = 9) — branch-free, only
`+ * / log`, so it compiles on every KernelAbstractions backend and every
float width. ~1 ulp in Float64 for x ≥ 1; no reflection, so callers keep
their argument ≥ 1 (every adaptation call site does).
"""
@inline function lgamma_lanczos(x::T) where {T <: AbstractFloat}
    z = x - one(T)
    base = z + T(7.5)
    s = T(0.99999999999980993)
    s += T(676.5203681218851) / (z + T(1.0))
    s += T(-1259.1392167224028) / (z + T(2.0))
    s += T(771.32342877765313) / (z + T(3.0))
    s += T(-176.61502916214059) / (z + T(4.0))
    s += T(12.507343278686905) / (z + T(5.0))
    s += T(-0.13857109526572012) / (z + T(6.0))
    s += T(9.9843695780195716e-6) / (z + T(7.0))
    s += T(1.5056327351493116e-7) / (z + T(8.0))
    # 0.9189385332046727 = log(2π)/2
    return T(0.9189385332046727) + (z + T(0.5)) * log(base) - base + log(s)
end

"""
    digamma_stirling(x)

ψ(x) by the Stirling series through the x⁻⁴ term — pure arithmetic, for the
Newton step on a Student-t's degrees of freedom inside a device kernel.
"""
@inline digamma_stirling(x::T) where {T} =
    log(x) - T(0.5) / x - T(1.0) / (T(12.0) * x * x) + T(1.0) / (T(120.0) * x * x * x * x)

"""
    trigamma_stirling(x)

ψ′(x) by the Stirling series through the x⁻³ term; companion of
[`digamma_stirling`](@ref).
"""
@inline function trigamma_stirling(x::T) where {T}
    ix = T(1.0) / x
    return ix + T(0.5) * ix * ix + ix * ix * ix / T(6.0)
end

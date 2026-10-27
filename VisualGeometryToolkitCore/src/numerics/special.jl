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

# =============================================================================
# The normal CDF and the dilogarithm (hoisted from RadiometricCamera:
# src/board/printed_edge.jl `_ncdf`, src/camera/focus_charts.jl `li2`; their
# arithmetic unchanged) — generic over `Real`, so ForwardDiff `Dual`s flow
# through, and every constant converted to the argument's value type, so a
# Float32 kernel never promotes.
# =============================================================================

"""
    normcdf_as(x::Real)

The standard normal CDF `Φ(x)` by Abramowitz & Stegun 7.1.26 on `erf`
(|error| < 1.5e-7): branch-free but for the sign, only `+ * / exp`, so it
compiles in any KernelAbstractions kernel at any float width. A `Dual`
flows through the polynomial, so its derivative is that of the
approximation (`φ(x)` to ≈ 1e-6) — the exact derivative of the evaluated
objective.
"""
@inline function normcdf_as(x::Real)
    T = typeof(x)
    z = abs(x) * T(0.7071067811865476)
    t = one(T) / (one(T) + T(0.3275911) * z)
    p = ((((T(1.061405429) * t - T(1.453152027)) * t + T(1.421413741)) * t - T(0.284496736)) * t + T(0.254829592)) * t
    y = one(T) - p * exp(-z * z)                                       # erf(z)
    return x >= zero(T) ? (one(T) + y) / 2 : (one(T) - y) / 2
end

# Bernoulli-series coefficients B_n/(n+1)! of Li₂(y) = Σ c_k u^k, u = −ln(1 − y) (k = n + 1 = 1 … 17)
const _LI2_C = (
    1.0, -0.25, 1 / 36, 0.0, -1 / 3600, 0.0, 1 / 211680, 0.0, -1 / 10886400, 0.0, 1 / 526901760, 0.0,
    -691 / (2730 * 6227020800.0), 0.0, 7 / (6 * 1307674368000.0), 0.0, -3617 / (510 * 355687428096000.0),
)
# Li₂(y) for 0 ≤ y ≤ ½: the Bernoulli series in u = −ln(1 − y), by Horner, the
# coefficients in the value type of `y`.
@inline function _dilog_series(y)
    u = -log1p(-y)
    c(k) = oftype(_value_part(u), _LI2_C[k])
    acc = zero(u) + c(17)
    for k in 16:-1:1
        acc = acc * u + c(k)
    end
    return acc * u
end

"""
    dilog(x::Real)

The dilogarithm `Li₂(x) = −∫₀ˣ ln(1 − t)/t dt` on `[0, 1]`: the Bernoulli
series in `u = −ln(1 − x)` (17 terms, by Horner) for `x ≤ ½`, the reflection
`Li₂(x) = π²/6 − ln x·ln(1 − x) − Li₂(1 − x)` above; `x ≥ 1` reads
`Li₂(1) = π²/6`. Accurate to a few eps in Float64; generic over `Real`
(Float32 kernels, ForwardDiff `Dual`s), the constants in the argument's value
type.
"""
@inline dilog(x::Real) =
    x <= 0.5 ? _dilog_series(x) : x >= 1 ? oftype(x, π^2 / 6) : (oftype(x, π^2 / 6) - log(x) * log1p(-x) - _dilog_series(1 - x))

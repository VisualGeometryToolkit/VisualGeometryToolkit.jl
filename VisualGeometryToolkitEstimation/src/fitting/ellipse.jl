# =============================================================================
# Ellipse refinement — orthogonal distance, chart LM on the log-shape chart
# =============================================================================
#
# The maximum-likelihood ellipse under isotropic Gaussian noise on the points:
# minimize Σᵢ dᵢ², dᵢ the signed orthogonal distance of point i to the ellipse,
# its distance to the foot point (the boundary point closest to it). The start
# value is VGC's `direct_ellipse` (one closed-form algebraic step); the chart is
# VGC's log-shape chart (`ellipse_log_shape`), boundary x(u) = c + exp(S)·u.
#
# The foot point is a 1-D root solve (Eberly's robust bisection), which AD must
# not see. At the foot point the residual's derivative is −nᵀ ∂x/∂θ with the
# boundary parameter u held fixed — the u-derivative of x is tangent, so normal
# to n — and that closed form is the Jacobian.
#
# =============================================================================
# Dependencies: StaticArrays, ForwardDiff (∂exp(S)/∂S), the chart LM
#               VGC: Ellipse, direct_ellipse, ellipse_log_shape, ellipse_from_log_shape
# =============================================================================

"""
    EllipseFitMetadata{T}

`FitResult` metadata of [`refine_ellipse`](@ref). The core carries the fitted
ellipse — `estimate` is an `Uncertain{Ellipse{T}, <:CovMat{5,T}}` whose
covariance is that of the log-shape chart `(cx, cy, S₁₁, S₁₂, S₂₂)`
(`ellipse_log_shape`) — the LM's `retcode` and `iterations`, and `rms`, the
root-mean-square orthogonal distance over the `n` points. Here:
- `s²::T`: estimated residual variance, `S/(n-5)`
- `ν::Int`: degrees of freedom, `n - 5`

Both forward as properties: `result.s²`, `result.ν`.
"""
struct EllipseFitMetadata{T}
    s²::T
    ν::Int
end

# exp(S) of the symmetric 2×2 S = [s₁₁ s₁₂; s₁₂ s₂₂], Dual-friendly and smooth at
# a circle: exp(S) = eᵐ·(cosh δ·I + (sinh δ/δ)·(S − m·I)), m the mean of the
# diagonal, δ² = h² + s₁₂², h half the diagonal difference. cosh δ and sinh δ/δ
# are series in δ² near 0, so no √ is differentiated at δ = 0.
function _expm_sym2(s11, s12, s22)
    m = (s11 + s22) / 2
    h = (s11 - s22) / 2
    δ² = h^2 + s12^2
    if δ² < 1.0e-6
        ch = 1 + δ² / 2 + δ²^2 / 24 + δ²^3 / 720
        shδ = 1 + δ² / 6 + δ²^2 / 120 + δ²^3 / 5040
    else
        δ = sqrt(δ²)
        ch = cosh(δ)
        shδ = sinh(δ) / δ
    end
    em = exp(m)
    return SMatrix{2, 2}(em * (ch + shδ * h), em * shδ * s12, em * shδ * s12, em * (ch - shδ * h))
end

# The foot point of (y₀, y₁) ≥ 0 on the axis-aligned ellipse with semi-axes
# e₀ ≥ e₁ > 0, in the first quadrant (Eberly, "Distance from a Point to an
# Ellipse, an Ellipsoid, or a Hyperellipsoid", robust bisection on the root of
# F(s) = (r₀z₀/(s + r₀))² + (z₁/(s + 1))² − 1).
function _ellipse_foot_quadrant(y0::T, y1::T, e0::T, e1::T) where {T}
    if y1 > 0
        y0 > 0 || return (zero(T), e1)
        z0, z1 = y0 / e0, y1 / e1
        g = z0^2 + z1^2 - 1
        iszero(g) && return (y0, y1)
        r0 = (e0 / e1)^2
        n0 = r0 * z0
        s0, s1 = z1 - 1, g < 0 ? zero(T) : hypot(n0, z1) - 1
        s = s0
        for _ in 1:200
            s = (s0 + s1) / 2
            (s == s0 || s == s1) && break
            f = (n0 / (s + r0))^2 + (z1 / (s + 1))^2 - 1
            f > 0 ? (s0 = s) : f < 0 ? (s1 = s) : break
        end
        return (r0 * y0 / (s + r0), y1 / (s + 1))
    end
    numer0, denom0 = e0 * y0, e0^2 - e1^2
    if numer0 < denom0
        xde0 = numer0 / denom0
        return (e0 * xde0, e1 * sqrt(1 - xde0^2))
    end
    return (e0, zero(T))
end

# The chart-LM linearization state: the points and the anchor's log-shape chart
# vector m₀; the chart vector θ is the update, m = m₀ + θ.
struct _EllipseLm{P}
    points::P
    m0::SVector{5, Float64}
end

# The boundary geometry at the chart point m: centre, the axes' frame V (major
# axis first) and semi-axes, from the eigen-decomposition S = V·diag(l)·Vᵀ.
function _ellipse_frame(m)
    F = eigen(Symmetric(SMatrix{2, 2, Float64}(m[3], m[4], m[4], m[5])))
    V = hcat(F.vectors[:, 2], F.vectors[:, 1])
    return SVector{2, Float64}(m[1], m[2]), SMatrix{2, 2, Float64}(V), exp(F.values[2]), exp(F.values[1])
end

# Point p's signed orthogonal distance (> 0 outside), and at its foot point the
# outward unit normal n and the boundary parameter u (x = c + exp(S)·u), all in
# the input frame.
function _ellipse_foot(p, c, V, e0, e1)
    q = V' * (p - c)
    x0, x1 = _ellipse_foot_quadrant(abs(q[1]), abs(q[2]), e0, e1)
    x = SVector(copysign(x0, q[1]), copysign(x1, q[2]))
    nl = SVector(x[1] / e0^2, x[2] / e1^2)
    nn = norm(nl)
    n = iszero(nn) ? SVector(one(nn), zero(nn)) : V * (nl / nn)
    u = V * SVector(x[1] / e0, x[2] / e1)
    d = dot(n, p - (c + V * x))
    return d, n, u
end

function residual_rows!(r, lm::_EllipseLm, θ)
    c, V, e0, e1 = _ellipse_frame(lm.m0 + θ)
    @inbounds for (i, p) in enumerate(lm.points)
        r[i] = first(_ellipse_foot(p, c, V, e0, e1))
    end
    return r
end

# −nᵀ ∂x/∂θ at the foot point, u fixed: −n for the centre, −nᵀ(∂exp(S)/∂Sₖ)·u for
# the shape (∂exp(S)/∂S once per call, by ForwardDiff through `_expm_sym2`).
function residual_jacobian_rows!(J, lm::_EllipseLm, θ)
    m = lm.m0 + θ
    c, V, e0, e1 = _ellipse_frame(m)
    dA = ForwardDiff.jacobian(s -> vec(_expm_sym2(s[1], s[2], s[3])), SVector(m[3], m[4], m[5]))   # 4×3
    @inbounds for (i, p) in enumerate(lm.points)
        _, n, u = _ellipse_foot(p, c, V, e0, e1)
        J[i, 1] = -n[1]
        J[i, 2] = -n[2]
        for k in 1:3
            dAk = SMatrix{2, 2, Float64}(dA[1, k], dA[2, k], dA[3, k], dA[4, k])
            J[i, 2 + k] = -dot(n, dAk * u)
        end
    end
    return J
end

"""
    refine_ellipse(points, init = direct_ellipse(points); max_iter = 100)
        -> FitResult{Uncertain{Ellipse}, EllipseFitMetadata}

The ellipse minimizing the summed squared orthogonal distance `Σᵢ dᵢ²` of the
2-D `points` to it, `dᵢ` the distance of point `i` to its foot point: the
maximum-likelihood ellipse under isotropic Gaussian noise on the points. A
Levenberg–Marquardt refinement from `init` (by default VisualGeometryToolkitCore's
algebraic [`direct_ellipse`](@ref)) on the log-shape chart
(`ellipse_log_shape`), which cannot leave the ellipses and is smooth at a
circle.

The estimate carries the covariance of the log-shape chart vector, `s²·(JᵀJ)⁻¹`
with `s² = Σdᵢ²/(n−5)`, `J` the Jacobian of the distances at the solution.
`retcode` is the LM's (`:converged`, `:max_iterations` or `:stalled`). Throws
`ArgumentError` for fewer than 6 points, or when the default start does not
exist (`direct_ellipse` returns `nothing`).
"""
function refine_ellipse(points::AbstractVector{<:StaticVector{2}}, init::Ellipse = _direct_start(points); max_iter::Integer = 100)
    n = length(points)
    n >= 6 || throw(ArgumentError("refine_ellipse requires at least 6 points, got $n"))
    pts = [SVector{2, Float64}(p[1], p[2]) for p in points]
    lm = _EllipseLm(pts, SVector{5, Float64}(ellipse_log_shape(init)))
    ews = ManifoldLmWorkspace(n, 5)
    x, _, conv, iterations = lmsolve_chart!(ews, lm; maxit = max_iter, stationarity_tol = _CHART_LM_STATIONARITY_TOL)
    r = residual_rows!(zeros(n), lm, x)
    J = residual_jacobian_rows!(zeros(n, 5), lm, x)
    S = sum(abs2, r)
    ν = n - 5
    s² = S / ν
    Σ = CovMat(SMatrix{5, 5, Float64}(s² * inv(Symmetric(J' * J))))
    e = ellipse_from_log_shape(lm.m0 + x)
    return FitResult(Uncertain(e, Σ), VgtkCore.lm_retcode(conv), iterations, sqrt(S / n), n, EllipseFitMetadata(s², ν))
end

function _direct_start(points)
    e = direct_ellipse(points)
    e === nothing && throw(ArgumentError("refine_ellipse: the points admit no direct_ellipse start value; supply `init`"))
    return e
end

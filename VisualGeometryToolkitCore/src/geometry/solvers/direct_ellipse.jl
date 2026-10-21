# =============================================================================
# direct_ellipse — the ellipse-specific algebraic fit (one closed-form step)
# =============================================================================
#
# Fitzgibbon, Pilu & Fisher's direct least-squares ellipse fit in the
# numerically stable form of Halir & Flusser: minimize Σ (xᵢᵀ C xᵢ)² subject to
# 4AC − B² = 1, a 3×3 eigenproblem with exactly one elliptic solution, in
# Hartley-normalized coordinates. One deterministic step, so an algorithm name
# (taxonomy v2); the orthogonal-distance refinement from it is
# VisualGeometryToolkitEstimation's `refine_ellipse`.
#
# The log-shape chart below is the ellipse's minimal parameterization that the
# refinement and its covariance live in.

"""
    direct_ellipse(points) -> Ellipse{T} | nothing

The ellipse through the 2-D `points` minimizing the summed squared algebraic
residual `Σᵢ (xᵢᵀ C xᵢ)²` under the ellipse constraint `4AC − B² = 1`
(Fitzgibbon, Pilu & Fisher 1999, in the numerically stable form of Halir &
Flusser 1998), computed in Hartley-normalized coordinates: one closed-form step
whose solution is always an ellipse. `nothing` when the points admit none (too
few in general position, e.g. collinear). Throws `ArgumentError` for fewer than
5 points.

A start value: the algebraic residual weighs points by where they sit on the
curve. VisualGeometryToolkitEstimation's `refine_ellipse` minimizes the
orthogonal distance from it.
"""
function direct_ellipse(points::AbstractVector{<:StaticVector{2}})
    n = length(points)
    n >= 5 || throw(ArgumentError("direct_ellipse requires at least 5 points, got $n"))
    T = float(eltype(eltype(points)))
    pts = [SVector{2, T}(p[1], p[2]) for p in points]
    N = AffineMap(HartleyNormalization(), pts)

    # The scatter blocks of the quadratic (x², xy, y²) and linear (x, y, 1) parts.
    S1 = zero(SMatrix{3, 3, T})
    S2 = zero(SMatrix{3, 3, T})
    S3 = zero(SMatrix{3, 3, T})
    for p in pts
        q = N(p)
        d1 = SVector{3, T}(q[1]^2, q[1] * q[2], q[2]^2)
        d2 = SVector{3, T}(q[1], q[2], one(T))
        S1 += d1 * d1'
        S2 += d1 * d2'
        S3 += d2 * d2'
    end
    F3 = cholesky(Symmetric(S3); check = false)
    issuccess(F3) || return nothing                      # the points span no plane of (x, y, 1)
    Tl = -(F3 \ S2')                                     # linear part in terms of the quadratic one
    M = S1 + S2 * Tl
    # C₁⁻¹·M with C₁ = [0 0 2; 0 −1 0; 2 0 0], the constraint 4AC − B² = aᵀC₁a
    Mc = vcat(M[3, :]' / 2, -M[2, :]', M[1, :]' / 2)
    E = eigen(Matrix(Mc))
    a1 = nothing
    for k in 1:3
        v = E.vectors[:, k]
        maximum(abs ∘ imag, v) <= sqrt(eps(T)) * maximum(abs, v) || continue
        w = SVector{3, T}(real.(v))
        4 * w[1] * w[3] - w[2]^2 > 0 && (a1 = w; break)
    end
    a1 === nothing && return nothing
    a2 = Tl * a1
    A, B, C = a1
    D, Ec, F = a2
    Cn = SMatrix{3, 3, T}(A, B / 2, D / 2, B / 2, C, Ec / 2, D / 2, Ec / 2, F)
    H = SMatrix{3, 3, T}(to_homogeneous(N))
    Cp = H' * Cn * H                                     # back to the input coordinates
    Cp = (Cp + Cp') / 2
    all(isfinite, Cp) || return nothing
    return Ellipse(HomEllipseMat{T}(Tuple(Cp)))
end

# =============================================================================
# The log-shape chart of an ellipse
# =============================================================================

"""
    ellipse_log_shape(e::Ellipse) -> SVector{5}

The log-shape chart of `e`: `(cx, cy, S₁₁, S₁₂, S₂₂)`, the centre and the
symmetric `S = R·diag(log a, log b)·Rᵀ` (`R` the rotation by `e.θ`), so that the
boundary is `{c + exp(S)·u : ‖u‖ = 1}`. Minimal, unable to leave the ellipse
cone (the exponential of every symmetric `S` is positive definite) and smooth
at a circle, where `(a, b, θ)` swaps its axes. Inverse of
[`ellipse_from_log_shape`](@ref).
"""
function ellipse_log_shape(e::Ellipse{T}) where {T}
    s, c = sincos(e.θ)
    la, lb = log(e.a), log(e.b)
    S11 = c^2 * la + s^2 * lb
    S12 = s * c * (la - lb)
    S22 = s^2 * la + c^2 * lb
    return SVector{5, T}(e.center[1], e.center[2], S11, S12, S22)
end

"""
    ellipse_from_log_shape(m) -> Ellipse

The ellipse of the log-shape chart vector `m = (cx, cy, S₁₁, S₁₂, S₂₂)`. Inverse
of [`ellipse_log_shape`](@ref).
"""
function ellipse_from_log_shape(m::AbstractVector)
    T = float(eltype(m))
    F = eigen(Symmetric(SMatrix{2, 2, T}(m[3], m[4], m[4], m[5])))
    v = F.vectors[:, 2]                                  # eigenvalues ascend: the major axis is the last
    return Ellipse{T}(Point2{T}(m[1], m[2]), exp(F.values[2]), exp(F.values[1]), mod(atan(v[2], v[1]), T(π)))
end

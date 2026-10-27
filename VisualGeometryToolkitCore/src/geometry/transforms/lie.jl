# =============================================================================
# SO(3) / SE(3) Lie-group helpers
# =============================================================================
#
# Exponential-map side (exp_so3, left_jacobian, exp_se3) is written as
# explicit Rodrigues/Taylor SMatrix arithmetic rather than delegating to
# Rotations.jl: these run inside reverse-mode AD (Zygote) in downstream
# packages, and Zygote cannot differentiate through Rotations' RotationVec →
# RotMatrix conversion (ChainRules Tangent mismatch, verified 2026-07).
# The logarithm (log_so3, log_se3) is the same kind of arithmetic — a Shepperd
# quaternion and one atan — generic over `Real`, so ForwardDiff `Dual`s flow
# through it, INCLUDING at the identity, where Rotations' RotationVec path
# differentiates to NaN (0/0) and where a hand-eye or robot-world residual sits
# at convergence.
#
# Hoisted from RayTracedCalibration's se3.jl (masters-thesis), where the
# Float32 and Float64 variants existed as separate copies; here they are one
# generic implementation. The Rodrigues coefficients are functions of θ² —
# `_sinc` = sin θ/θ, `_cosc` = (1 − cos θ)/θ², `_sinc3` = (θ − sin θ)/θ³ —
# each with its own series switch (below): `exp_so3` reads `_sinc` (its second
# coefficient keeps its own closed form, see there), `left_jacobian` `_cosc` and
# `_sinc3`.
# =============================================================================

# Taylor-vs-trig switch on θ². Must be at least eps(T): below θ² ≈ eps(T),
# cos(θ) rounds to one(T) and the trig branch's (1−cosθ)/θ² collapses to 0
# instead of ½ — the hoisted originals used 1f-8/1e-16 and silently produced
# a zeroed linear coefficient for Float32 θ ∈ (1e-4, ~3.5e-4). eps(T) keeps
# every previously-correct input on its old branch (bit-identical) and moves
# only the broken window onto the Taylor series.
_small_angle_sq(::Type{T}) where {T <: Real} = eps(T)   # a ForwardDiff Dual answers its value type's eps

# sin θ/θ of θ²: no cancellation, so the closed form is accurate down to the
# switch, and the two-term series below it (θ² < eps(T)) is exact to rounding.
@inline function _sinc(θ²::T) where {T <: Real}
    θ² < _small_angle_sq(T) && return one(T) - θ² / T(6)
    θ = sqrt(θ²)
    return sin(θ) / θ
end

# Series-vs-closed-form switch of `_cosc` and `_sinc3` on θ² (θ = 1/4): below
# it the closed forms lose digits to cancellation (thousands of eps at θ =
# 1e-5); the six-term series is within 4 eps there (lie_tests.jl, against a
# 300-bit reference), and above it the closed forms are exact to rounding.
const _left_jacobian_series_sq = 1 // 16

# (1 − cos θ)/θ² of θ²: Σₖ (−θ²)ᵏ/(2k+2)!, k = 0:5, by Horner, below the switch.
@inline function _cosc(θ²::T) where {T <: Real}
    θ² < _left_jacobian_series_sq && return T(1 // 2) + θ² * (-T(1 // 24) + θ² * (T(1 // 720) +
        θ² * (-T(1 // 40320) + θ² * (T(1 // 3628800) - θ² * T(1 // 479001600)))))
    return (one(T) - cos(sqrt(θ²))) / θ²
end

# (θ − sin θ)/θ³ of θ²: Σₖ (−θ²)ᵏ/(2k+3)!, k = 0:5, by Horner, below the switch.
@inline function _sinc3(θ²::T) where {T <: Real}
    θ² < _left_jacobian_series_sq && return T(1 // 6) + θ² * (-T(1 // 120) + θ² * (T(1 // 5040) +
        θ² * (-T(1 // 362880) + θ² * (T(1 // 39916800) - θ² * T(1 // 6227020800)))))
    θ = sqrt(θ²)
    return (θ - sin(θ)) / (θ² * θ)
end

"""
    exp_so3(ω::SVector{3,T}) -> SMatrix{3,3,T,9}

Rodrigues exponential map so(3) → SO(3): `R = I + a·[ω]× + b·[ω]×²` with the
second-order Taylor branch below the per-type small-angle threshold.

Plain SMatrix arithmetic — safe inside Zygote-tracked code (unlike
`RotMatrix(RotationVec(ω...))`, which reverse-mode AD cannot penetrate) and
generic over `Real`, so ForwardDiff `Dual`s flow through it too.
See also [`skew`](@ref), [`left_jacobian`](@ref), [`exp_se3`](@ref).

GTSAM spells this `Rot3::Expmap` (and `log_so3` is `Rot3::Logmap`), Sophus
`SO3::exp` / `SO3::log`. Manifolds.jl's name is `exp_lie(G, X)` / `log_lie`:
its `exp(M, p, X)` is the exponential at a BASE POINT, while this is the
Lie-group exponential at the identity — which is why the group is not a
first argument here.
"""
function exp_so3(ω::SVector{3, T}) where {T <: Real}
    θ² = ω[1] * ω[1] + ω[2] * ω[2] + ω[3] * ω[3]
    K = skew(ω)
    a = _sinc(θ²)
    # b is (1 − cos θ)/θ² by its closed form down to θ² = eps(T), NOT `_cosc`:
    # RadiometricCamera's goldens pin exp_so3's bits. The closed form's relative
    # error near the switch (~eps/θ²) is invisible in R, where b multiplies
    # [ω]×², of size θ².
    b = θ² < _small_angle_sq(T) ? T(0.5) - θ² / T(24) : (one(T) - cos(sqrt(θ²))) / θ²
    return SMatrix{3, 3, T, 9}(I) + a * K + b * (K * K)
end

"""
    SO3

The rotation group, as a type tag — the group a Lie-group helper is being
asked about, in the slot a type argument takes (`left_jacobian(SO3, ω)`).
Carries no data. Public, not exported.
"""
struct SO3 end

"""
    left_jacobian(SO3, ω::SVector{3,T}) -> SMatrix{3,3,T,9}

Left Jacobian of SO(3): `J = I + a·[ω]× + b·[ω]×²` with
`a = (1−cosθ)/θ²`, `b = (θ−sinθ)/θ³`. Maps a
rotation-vector perturbation to the translation part of the SE(3)
exponential: `t = J(ω)·ρ`. Not provided by Rotations.jl. Below `θ = 1/4`, where
the closed forms cancel, `a` and `b` are their six-term Taylor series (within
4 eps of the exact value); above it the closed forms.

This is Sophus' `SO3::leftJacobian`. **GTSAM's `Rot3::ExpmapDerivative` is
its TRANSPOSE**, the RIGHT Jacobian `Jr(ω) = Jl(ω)ᵀ = Jl(−ω)`, because GTSAM
retracts by right multiplication — do not equate the two. Measured here
(2026-09-22): the left perturbation identity
`exp(ω+δ) = exp(Jl(ω)·δ)·exp(ω)` holds to 5.8e-13 while the right-multiplied
one is off by 1.8e-6, and `Jl(ω) = Jl(−ω)ᵀ` to 0.0.
"""
function left_jacobian(::Type{SO3}, ω::SVector{3, T}) where {T <: Real}
    θ² = ω[1] * ω[1] + ω[2] * ω[2] + ω[3] * ω[3]
    K = skew(ω)
    return SMatrix{3, 3, T, 9}(I) + _cosc(θ²) * K + _sinc3(θ²) * (K * K)
end

"""
    exp_se3(ω::SVector{3}, ρ::SVector{3}) -> EuclideanMap

SE(3) exponential of the twist `(ω, ρ)`:
`(exp_so3(ω), left_jacobian(SO3, ω)·ρ)` as an [`EuclideanMap`](@ref)
(constructed with `check = false`; the Rodrigues result is orthonormal to
floating-point precision by construction).

GTSAM's `Pose3::Expmap` (`log_se3` is `Pose3::Logmap`), Sophus' `SE3::exp`,
Manifolds.jl's `exp_lie`. Note the twist ORDER differs between libraries —
ours is `(ω, ρ)`, rotation first.

Generic over `Real`: the two parts are promoted to one float type, so
ForwardDiff `Dual`s flow through it (a pose retraction `exp_se3(ω, ρ) ∘ T`
differentiated by an NLLS solver), as through `exp_so3` and `left_jacobian`.
"""
function exp_se3(ω::SVector{3, <:Real}, ρ::SVector{3, <:Real})
    T = float(promote_type(eltype(ω), eltype(ρ)))
    ωT, ρT = SVector{3, T}(ω), SVector{3, T}(ρ)
    R = exp_so3(ωT)
    t = left_jacobian(SO3, ωT) * ρT
    return EuclideanMap(RotMatrix{3, T}(R), t; check = false)
end

"""
    log_so3(R::StaticMatrix{3,3,T}) -> SVector{3,T}

Rotation vector (axis · angle, angle ∈ [0, π]) of `R` — the SO(3) logarithm:
the unit quaternion `(w, v)` of `R` by Shepperd's method (the largest of
`w, x, y, z` computed from the diagonal, so every branch, the near-π one
included, divides by a number ≥ ½), sign fixed to `w ≥ 0`, then
`2·atan(‖v‖, w)·v/‖v‖` — with its Taylor series `(2/w)(1 − ‖v‖²/(3w²))·v` below
the per-type small-angle threshold, so the derivative at the identity is exact.
Generic over `Real`: ForwardDiff `Dual`s flow through (within 2.5 ulp of
Rotations' `RotationVec`, measured 2026-09-22).
"""
function log_so3(R::SMatrix{3, 3, T, 9}) where {T <: Real}
    r11, r21, r31, r12, r22, r32, r13, r23, r33 = Tuple(R)
    tr = r11 + r22 + r33
    if tr >= r11 && tr >= r22 && tr >= r33
        w = sqrt(1 + tr) / 2; d = 4w
        x = (r32 - r23) / d; y = (r13 - r31) / d; z = (r21 - r12) / d
    elseif r11 >= r22 && r11 >= r33
        x = sqrt(1 + r11 - r22 - r33) / 2; d = 4x
        w = (r32 - r23) / d; y = (r12 + r21) / d; z = (r13 + r31) / d
    elseif r22 >= r33
        y = sqrt(1 - r11 + r22 - r33) / 2; d = 4y
        w = (r13 - r31) / d; x = (r12 + r21) / d; z = (r23 + r32) / d
    else
        z = sqrt(1 - r11 - r22 + r33) / 2; d = 4z
        w = (r21 - r12) / d; x = (r13 + r31) / d; y = (r23 + r32) / d
    end
    if w < 0
        w, x, y, z = -w, -x, -y, -z
    end
    s² = x * x + y * y + z * z
    f = if s² < _small_angle_sq(T)
        2 / w * (1 - s² / (3 * w * w))
    else
        s = sqrt(s²)
        2 * atan(s, w) / s
    end
    return SVector{3, T}(f * x, f * y, f * z)
end
# Any static 3×3 (a `RotMatrix`, a `Rotation`, a `MMatrix`): the same logarithm on
# its `SMatrix`, the element type kept (Duals included).
log_so3(R::StaticMatrix{3, 3, T}) where {T <: Real} = log_so3(SMatrix{3, 3, T, 9}(R))

# The rotation a drifted rotation matrix stands for, by its unit quaternion — the
# ONE re-anchoring policy of every chart with an SO(3) part (`recenter` of the rigid
# and essential charts): a long run of multiplied rotation updates otherwise
# accumulates rounding until an `EuclideanMap` rebuild's rotation check refuses the
# iterate. Retractions do not re-project; only re-anchoring does.
_reproject_so3(R::AbstractMatrix{T}) where {T} =
    SMatrix{3, 3, T, 9}(RotMatrix{3, T}(Rotations.QuatRotation(RotMatrix{3, T}(SMatrix{3, 3, T, 9}(R)))))

"""
    log_se3(E::EuclideanMap) -> (ω, ρ)

SE(3) logarithm of a rigid transform: `ω = log_so3(R)`,
`ρ = left_jacobian(SO3, ω) \\ t`. Inverse of [`exp_se3`](@ref). Element type
follows the map's rotation eltype. NOT AD-safe under Zygote (see
[`log_so3`](@ref)).
"""
function log_se3(E::PlainEuclideanMap{3})
    T = eltype(E.R)
    R = SMatrix{3, 3, T, 9}(E.R)
    t = SVector{3, T}(E.t)
    ω = log_so3(R)
    ρ = left_jacobian(SO3, ω) \ t
    return ω, ρ
end

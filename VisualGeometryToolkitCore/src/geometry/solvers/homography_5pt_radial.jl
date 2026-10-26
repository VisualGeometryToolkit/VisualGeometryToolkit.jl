# =============================================================================
# 5-point radial homography, one-sided — port of HomLib's
# get_wadenback_3dv_2026_one_sided.cpp
# =============================================================================
#
# Source of truth: HomLib (Valtonen Örnhag, MIT), solvers/wadenback_3dv_2026/
# {get_wadenback_3dv_2026_one_sided.cpp, wadenback_common.cpp}, implementing
# Wadenbäck, Valtonen Örnhag, Edstedt, "Radially Distorted Homographies,
# Revisited" (3DV 2026, arXiv:2508.21190). HomLib's citation entry:
#
#   @InProceedings{wadenback-etal-2026-3dv,
#       author    = {Wadenb{\"a}ck, M{\aa}rten and Valtonen~{\"O}rnhag, Marcus
#                    and Edstedt, Johan},
#       title     = {Radially Distorted Homographies, Revisited},
#       booktitle = {Proceedings of the International Conference on
#                    3D Vision (3DV)},
#       year      = {2026},
#   }
#
# Model — one-parameter DIVISION distortion on the TARGET side only:
#
#     ûᵢ = dᵢ / (1 + k·‖dᵢ‖²)        (undistort the observed targets)
#     ûᵢ ≃ π(H s̃ᵢ)                   (plain homography on the sources)
#
# Five correspondences make (H, k) exactly determined: k satisfies a cubic
# whose coefficients come from the adjugate construction over the targets
# (`_radial1_coeffs`) crossed with a source-side triple-product vector
# (`_radial1_rhs`). Each real root yields one candidate: the targets are
# undistorted with k and H is the unique interpolating homography of the
# FIRST FOUR correspondences (the fifth is consumed by the distortion
# constraint) — matching the C++ reference.
#
# The problem is "4.5-point" OVERDETERMINED: (H, k) has 9 DOF against the
# 10 constraints of five correspondences. The cubic uses one of the fifth
# point's two equations, so every candidate interpolates the first four
# correspondences plus that scalar equation; only the TRUE root also
# satisfies the redundant second equation (`_radial1_unused_residual`,
# which `extra_check` minimizes) and interpolates all five points on
# noise-free data. (Numerically a spurious root can undistort the target
# quad into near-degeneracy, collapsing even its 4-point interpolation —
# the C++ reference emits the identical candidate; robust scoring or
# `extra_check` discards these downstream.)
#
# Deliberate substitutions (validated against the C++ golden test):
#   - PoseLib's solve_cubic_real → `_real_cubic_roots` (fundmat.jl). Both
#     enumerate Vieta's trigonometric roots at θ, θ−2π/3, θ−4π/3 from the
#     scale-invariant depressed form, so root ORDER matches; the same single
#     Newton polish PoseLib applies is kept here. `_real_cubic_roots`
#     additionally falls back to the quadratic when the leading coefficient
#     vanishes (the C++ divides by it unguarded).
#   - homography_4pt_guo → `homography_4pt`. The interpolating homography of
#     4 points is unique, so the kernels agree up to scale; ours returns the
#     sign+Frobenius-normalized representative (H[3,3] ≥ 0).
#
# Coordinates are used RAW (no Hartley normalization), like the reference:
# k is division-model curvature in squared target-frame units, so callers
# keep sources/targets in comparable, roughly unit-scale frames (k transforms
# as k/s² under uniform scaling of the targets).

"""
    RadialHomography{T}

Radially distorted homography model: a [`HomographyMat`](@ref) `H` paired
with a one-parameter division-model coefficient `k` acting on the TARGET
side, about the distortion `center` `c`. A correspondence `(s, d)` is
explained when the division-model RAY LIFT of the observation is
proportional to the recentered `H` ray:

    (d − c ; 1 + k‖d − c‖²) ≃ λ · T₋c · H s̃

The division model is an uncalibrated projection/back-projection model
([`project`](@ref)/[`backproject`](@ref)) — the lift's signed third
component represents rays beyond 90° (omnidirectional lenses), which no
image-space "undistortion" can. The solver frame is
origin-centered (`c = 0` — [`homography_5pt_radial`](@ref) constructs it
that way); composing back through a fixed-center normalization produces the
pixel-frame model with the center carried along, so consumers
([`radial_transfer`](@ref), quad projection) need no side-channel frame
data.

Produced by [`homography_5pt_radial`](@ref) / [`Homography5ptRadial`](@ref).
"""
struct RadialHomography{T} <: AbstractRadialModel
    H::HomographyMat{T}
    k::T
    center::Point2{T}
end

"The plane part `H` of a [`RadialHomography`](@ref): plane point → undistorted pixel."
homography(m::RadialHomography) = m.H

"The division coefficient `k` of a [`RadialHomography`](@ref)'s image-side lens, about [`center`](@ref)."
division_coeff(m::RadialHomography) = m.k

# Solver-frame convenience: distortion about the origin.
RadialHomography{T}(H::HomographyMat, k) where {T} =
    RadialHomography{T}(HomographyMat{T}(Tuple(H)), T(k), zero(Point2{T}))
RadialHomography(H::HomographyMat{T}, k::Real) where {T} =
    RadialHomography{T}(H, k)

Base.zero(::Type{RadialHomography{T}}) where {T} =
    RadialHomography{T}(
    HomographyMat{T}(ntuple(_ -> zero(T), Val(9))), zero(T),
    zero(Point2{T})
)
Base.zero(m::RadialHomography) = zero(typeof(m))

@inline is_valid_solution(m::RadialHomography) =
    all(isfinite, m.H) && isfinite(m.k) && all(isfinite, m.center)

# -----------------------------------------------------------------------------
# Coefficient construction (wadenback_common.cpp: compute_coeffs)
# -----------------------------------------------------------------------------

# The 12-vector of cubic-in-k monomial coefficients built from the five
# DISTORTED target points. Layout matches the C++ row-major order:
# (n1_3, n2_3, n3_3, n1_2, n2_2, n3_2, n1_1, n2_1, n3_1, n1_0, n2_0, n3_0).
@inline function _radial1_coeffs(x1, x2, x3, x4, x5)
    # M = adj([x̃1 x̃2 x̃3]) with the division-model lift folded in:
    # m_ij = cij_1*k + cij_0 (third column is k-free, hence cij3_1 omitted)
    c11_1 = -x2[1]^2 * x3[2] - x2[2]^2 * x3[2] + x2[2] * x3[1]^2 + x2[2] * x3[2]^2
    c11_0 = x2[2] - x3[2]
    c12_1 = x2[1]^2 * x3[1] - x2[1] * x3[1]^2 - x2[1] * x3[2]^2 + x2[2]^2 * x3[1]
    c12_0 = -x2[1] + x3[1]
    c21_1 = x1[1]^2 * x3[2] + x1[2]^2 * x3[2] - x1[2] * x3[1]^2 - x1[2] * x3[2]^2
    c21_0 = -x1[2] + x3[2]
    c22_1 = -x1[1]^2 * x3[1] + x1[1] * x3[1]^2 + x1[1] * x3[2]^2 - x1[2]^2 * x3[1]
    c22_0 = x1[1] - x3[1]
    c31_1 = -x1[1]^2 * x2[2] - x1[2]^2 * x2[2] + x1[2] * x2[1]^2 + x1[2] * x2[2]^2
    c31_0 = x1[2] - x2[2]
    c32_1 = x1[1]^2 * x2[1] - x1[1] * x2[1]^2 - x1[1] * x2[2]^2 + x1[2]^2 * x2[1]
    c32_0 = -x1[1] + x2[1]
    c13_0 = x2[1] * x3[2] - x2[2] * x3[1]
    c23_0 = -x1[1] * x3[2] + x1[2] * x3[1]
    c33_0 = x1[1] * x2[2] - x1[2] * x2[1]

    # g = M * x̃4 (each entry affine in k; the lift contributes r² to the
    # k-coefficient through the third column)
    r42 = x4[1] * x4[1] + x4[2] * x4[2]
    g1_1 = c11_1 * x4[1] + c12_1 * x4[2] + c13_0 * r42
    g1_0 = c13_0 + c11_0 * x4[1] + c12_0 * x4[2]
    g2_1 = c21_1 * x4[1] + c22_1 * x4[2] + c23_0 * r42
    g2_0 = c23_0 + c21_0 * x4[1] + c22_0 * x4[2]
    g3_1 = c31_1 * x4[1] + c32_1 * x4[2] + c33_0 * r42
    g3_0 = c33_0 + c31_0 * x4[1] + c32_0 * x4[2]

    # h = M * x̃5
    r52 = x5[1] * x5[1] + x5[2] * x5[2]
    h1_1 = c11_1 * x5[1] + c12_1 * x5[2] + c13_0 * r52
    h1_0 = c13_0 + c11_0 * x5[1] + c12_0 * x5[2]
    h2_1 = c21_1 * x5[1] + c22_1 * x5[2] + c23_0 * r52
    h2_0 = c23_0 + c21_0 * x5[1] + c22_0 * x5[2]
    h3_1 = c31_1 * x5[1] + c32_1 * x5[2] + c33_0 * r52
    h3_0 = c33_0 + c31_0 * x5[1] + c32_0 * x5[2]

    # Diagonal of T = adj(diag(g)) — quadratic in k
    t1_2 = g2_1 * g3_1
    t1_1 = g2_0 * g3_1 + g2_1 * g3_0
    t1_0 = g2_0 * g3_0
    t2_2 = g1_1 * g3_1
    t2_1 = g1_0 * g3_1 + g1_1 * g3_0
    t2_0 = g1_0 * g3_0
    t3_2 = g1_1 * g2_1
    t3_1 = g1_0 * g2_1 + g1_1 * g2_0
    t3_0 = g1_0 * g2_0

    # n = T * h — cubic in k
    n1_3 = h1_1 * t1_2
    n1_2 = h1_0 * t1_2 + h1_1 * t1_1
    n1_1 = h1_0 * t1_1 + h1_1 * t1_0
    n1_0 = h1_0 * t1_0
    n2_3 = h2_1 * t2_2
    n2_2 = h2_0 * t2_2 + h2_1 * t2_1
    n2_1 = h2_0 * t2_1 + h2_1 * t2_0
    n2_0 = h2_0 * t2_0
    n3_3 = h3_1 * t3_2
    n3_2 = h3_0 * t3_2 + h3_1 * t3_1
    n3_1 = h3_0 * t3_1 + h3_1 * t3_0
    n3_0 = h3_0 * t3_0

    return SVector(n1_3, n2_3, n3_3, n1_2, n2_2, n3_2, n1_1, n2_1, n3_1, n1_0, n2_0, n3_0)
end

# Source-side triple-product vector (get_rhs): built from the five
# UNDISTORTED source points, k-free.
@inline function _radial1_rhs(x1, x2, x3, x4, x5)
    m11 = x2[2] - x3[2]
    m12 = -x2[1] + x3[1]
    m13 = x2[1] * x3[2] - x2[2] * x3[1]
    m21 = -x1[2] + x3[2]
    m22 = x1[1] - x3[1]
    m23 = -x1[1] * x3[2] + x1[2] * x3[1]
    m31 = x1[2] - x2[2]
    m32 = -x1[1] + x2[1]
    m33 = x1[1] * x2[2] - x1[2] * x2[1]
    g1 = m13 + m11 * x4[1] + m12 * x4[2]
    g2 = m23 + m21 * x4[1] + m22 * x4[2]
    g3 = m33 + m31 * x4[1] + m32 * x4[2]
    h1 = m13 + m11 * x5[1] + m12 * x5[2]
    h2 = m23 + m21 * x5[1] + m22 * x5[2]
    h3 = m33 + m31 * x5[1] + m32 * x5[2]
    return SVector(g2 * g3 * h1, g1 * g3 * h2, g1 * g2 * h3)
end

# The residual of the (redundant) first equation of the 3-equation system —
# used by `extra_check` to pick the most consistent root.
@inline function _radial1_unused_residual(d::SVector{12, T}, rhs, k) where {T}
    k2 = k * k
    k3 = k2 * k
    return abs(
        (rhs[1] * d[3] - rhs[3] * d[1]) * k3 +
            (rhs[1] * d[6] - rhs[3] * d[4]) * k2 +
            (rhs[1] * d[9] - rhs[3] * d[7]) * k +
            rhs[1] * d[12] - rhs[3] * d[10]
    )
end

# One candidate: undistort the first four targets with k, interpolate H.
@inline function _radial1_candidate(s₁, s₂, s₃, s₄, d₁, d₂, d₃, d₄, k::T) where {T}
    u₁ = d₁ / (1 + k * (d₁[1]^2 + d₁[2]^2))
    u₂ = d₂ / (1 + k * (d₂[1]^2 + d₂[2]^2))
    u₃ = d₃ / (1 + k * (d₃[1]^2 + d₃[2]^2))
    u₄ = d₄ / (1 + k * (d₄[1]^2 + d₄[2]^2))
    H = homography_4pt(s₁, s₂, s₃, s₄, u₁, u₂, u₃, u₄)
    H === nothing && return nothing
    return RadialHomography{T}(H, k)
end

# -----------------------------------------------------------------------------
# Public kernel
# -----------------------------------------------------------------------------

"""
    homography_5pt_radial(s₁, …, s₅, d₁, …, d₅; extra_check=false)
        -> FixedModels{3, RadialHomography}

Minimal 5-point solver for a homography with one-sided one-parameter
division-model radial distortion on the TARGET side ([Wadenback2026](@citet) —
port of HomLib's `get_one_sided`). Sources `sᵢ` are distortion-free; targets `dᵢ` are
observed (distorted). Each candidate `(H, k)` satisfies

    dᵢ / (1 + k‖dᵢ‖²) ≃ π(H s̃ᵢ)

exactly for `i = 1…4`, plus ONE scalar equation of the fifth
correspondence: the problem is "4.5-point" overdetermined (9 DOF vs 10
constraints), so only the true root also satisfies the redundant remaining
equation and interpolates all five points. The distortion coefficient
solves a cubic, so up to 3 candidates are returned (`count == 0` on
degenerate samples). With `extra_check = true` only the root minimizing
the redundant-equation residual is kept (at most 1 candidate) — the
reference enables this in its RANSAC wrapper.

Coordinates are used raw: supply roughly unit-scale, centered coordinates
(`k` lives in squared target units and rescales as `k/s²`).
"""
function homography_5pt_radial(
        s₁::StaticVector{2, T}, s₂::StaticVector{2, T}, s₃::StaticVector{2, T},
        s₄::StaticVector{2, T}, s₅::StaticVector{2, T},
        d₁::StaticVector{2, T}, d₂::StaticVector{2, T}, d₃::StaticVector{2, T},
        d₄::StaticVector{2, T}, d₅::StaticVector{2, T};
        extra_check::Bool = false
    ) where {T}
    M = RadialHomography{T}

    dd = _radial1_coeffs(d₁, d₂, d₃, d₄, d₅)
    rhs = _radial1_rhs(s₁, s₂, s₃, s₄, s₅)

    # Cubic in k from the second and third equations (C++ 0-based d[]
    # indices shifted by one)
    c₃ = -rhs[2] * dd[3] + rhs[3] * dd[2]
    c₂ = -rhs[2] * dd[6] + rhs[3] * dd[5]
    c₁ = -rhs[2] * dd[9] + rhs[3] * dd[8]
    c₀ = -rhs[2] * dd[12] + rhs[3] * dd[11]

    raw = _real_cubic_roots(c₃, c₂, c₁, c₀)

    # Compact the real roots and apply PoseLib's single Newton polish.
    ks = MVector{3, T}(undef)
    nroots = 0
    @inbounds for i in 1:3
        k = raw[i]
        isnan(k) && continue
        f = ((c₃ * k + c₂) * k + c₁) * k + c₀
        df = (3 * c₃ * k + 2 * c₂) * k + c₁
        if abs(df) > zero(T)
            k -= f / df
        end
        nroots += 1
        ks[nroots] = k
    end

    cands = MVector{3, M}(undef)
    n_out = 0

    if extra_check && nroots > 0
        # Keep only the root most consistent with the unused equation.
        best = 1
        if nroots > 1
            best_res = T(Inf)
            @inbounds for i in 1:nroots
                res = _radial1_unused_residual(dd, rhs, ks[i])
                if res < best_res
                    best_res = res
                    best = i
                end
            end
        end
        m = _radial1_candidate(s₁, s₂, s₃, s₄, d₁, d₂, d₃, d₄, @inbounds ks[best])
        if m !== nothing && is_valid_solution(m)
            n_out = 1
            @inbounds cands[1] = m
        end
    else
        @inbounds for i in 1:nroots
            m = _radial1_candidate(s₁, s₂, s₃, s₄, d₁, d₂, d₃, d₄, ks[i])
            (m === nothing || !is_valid_solution(m)) && continue
            n_out += 1
            cands[n_out] = m
        end
    end

    z = zero(M)
    @inbounds for i in (n_out + 1):3
        cands[i] = z
    end
    return FixedModels{3, M}(n_out, Tuple(cands))
end

# =============================================================================
# Model ops — division-model point maps, transfer, coordinate normalization
# =============================================================================

# The division model is an UNCALIBRATED PROJECTION/BACK-PROJECTION model,
# not image-space warping: back-projection lifts an observed point to a RAY
# whose third component `1 + k‖d − c‖²` may be zero (a 90° ray) or negative
# (beyond the hemisphere), which is how one coefficient represents
# omnidirectional lenses with FOV > 180° (Fitzgibbon's lifted
# representation). The "undistorted image point" is merely the
# dehomogenization of that ray and exists only on the front hemisphere —
# so the ops below extend VGC's camera verbs `project`/`backproject`
# instead of pretending to be image warps.

"""
    backproject(m::RadialHomography, d) -> SVector{3}

Back-project an observed target point to its (unnormalized) ray in the
model's CENTERED frame: `(d − c ; 1 + k‖d − c‖²)` — the division-model
lift. Total; the third component's sign carries front/back hemisphere
(zero at exactly 90°, negative beyond — valid for omnidirectional lenses).
The `H` part of the model relates source points to these rays:
`backproject(m, d) ≃ λ · T₋c · H s̃` (with `T₋c` the recentering
translation).
"""
@inline function backproject(m::RadialHomography{T}, d::StaticVector{2}) where {T}
    dx = T(d[1]) - m.center[1]
    dy = T(d[2]) - m.center[2]
    return Vec3{T}(dx, dy, 1 + m.k * (dx * dx + dy * dy))
end

"""
    project(m::RadialHomography, ray::StaticVector{3}) -> SVector{2} or nothing

Project a ray (in the model's CENTERED frame, as produced by the `H` part)
to the observed image point whose [`backproject`](@ref) lift is
proportional to it: `c + μ·(x, y)` with `μ = 2 / (z + √(z² − 4k ρ²))`,
`ρ² = x² + y²` (the branch continuous with `k → 0`, where it reduces to
perspective division `μ = 1/z`). For `k < 0` (fisheye/omni) this is total
over front AND back rays; for `k > 0` it returns `nothing` when
`z² − 4kρ² < 0` (the ray lies outside the model's representable field).
"""
@inline function project(m::RadialHomography, ray::StaticVector{3})
    p = _radial_projection(m, ray)
    p.ok || return nothing
    d = _radial_point(m, p)
    return Point2{eltype(d)}(d[1], d[2])
end

# The ONE edge rule of the division-model projection, shared by `project`, its
# derivatives and the LM rows (and so by every scorer that calls
# `radial_transfer`): a ray is in the model's field iff it is off-axis with
# `z² − 4kρ² > 0` and `z + √(z² − 4kρ²)` not lost to rounding RELATIVE to `z`, or
# on-axis (ρ² below rounding relative to z²) and forward. Relative thresholds
# only, so a rescaled ray or frame gets the same verdict. Generic in the element
# type (Duals included) and device-safe.
@inline function _radial_projection(m::RadialHomography, ray::StaticVector{3})
    T = float(promote_type(typeof(m.k), eltype(ray)))
    x = T(ray[1]); y = T(ray[2]); z = T(ray[3]); k = T(m.k)
    ρ² = x * x + y * y
    disc = z * z - 4 * k * ρ²
    S = sqrt(max(disc, zero(T)))
    den = z + S
    axial = ρ² < eps(T) * max(z * z, one(T))
    ok = axial ? z > zero(T) : disc > zero(T) && abs(den) >= eps(T) * max(abs(z), one(T))
    return (; x, y, z, k, ρ², S, μ = 2 / den, axial, ok)
end

"""
    radial_in_front(m::RadialHomography, s) -> Bool

Whether source point `s` lands in `m`'s field ([`project`](@ref)'s one edge rule)
AND in its FRONT hemisphere: the observed point's division lift `μ·(x, y, z)` has
a positive third component (`μ·z > 0`, the ray within 90° of the axis). For
`k ≥ 0` every in-field ray is in front (the lift's third component `1 + kρ²`
is positive), so this adds `z > 0` for `k < 0` only — where `project` is total
over back rays too (omnidirectional lenses). The extent gate of a planar target:
a board corner at or past 90° means the hypothesis spans ≥ 180° across a plane,
which no camera images.
"""
@inline function radial_in_front(m::RadialHomography, s)
    p = _radial_projection(m, radial_ray(m, s))
    return p.ok && p.μ * p.z > zero(p.μ)
end

# The predicted observation of an in-field ray: `c + μ·(x, y)`, the centre exactly
# on the axis.
@inline function _radial_point(m::RadialHomography, p)
    T = typeof(p.μ)
    c1 = T(m.center[1]); c2 = T(m.center[2])
    return p.axial ? SVector{2, T}(c1, c2) : SVector{2, T}(c1 + p.μ * p.x, c2 + p.μ * p.y)
end

# ∂(c + μ·(x, y))/∂(x, y, z) and ∂/∂k of an in-field ray, closed form: with
# S = √(z² − 4kρ²) and μ = 2/(z + S),
#     ∂μ/∂ρ² = μ²k/S,   ∂μ/∂z = −μ²(1 + z/S)/2,   ∂μ/∂k = μ²ρ²/S.
@inline function _radial_derivs(p)
    μ² = p.μ * p.μ
    A = μ² * p.k / p.S
    B = -μ² * (1 + p.z / p.S) / 2
    xy = 2 * p.x * p.y * A
    D = SMatrix{2, 3}(p.μ + 2 * p.x * p.x * A, xy, xy, p.μ + 2 * p.y * p.y * A, p.x * B, p.y * B)
    gk = μ² * p.ρ² / p.S
    return D, SVector(p.x * gk, p.y * gk)
end

"""
    transform_deriv(m::RadialHomography, ray::StaticVector{3}) -> SMatrix{2,3}
    transform_deriv_params(m::RadialHomography, ray::StaticVector{3}) -> SMatrix{2,1}

The derivatives of [`project`](@ref)`(m, ray)`: along the (centred) ray, and along
the division coefficient `k` (the model's one lens parameter; `H` and the centre
act before the ray). Closed form; NaN where `project` returns `nothing` — the
same edge rule, as for the lens models beyond their field.
"""
@inline function CoordinateTransformations.transform_deriv(m::RadialHomography, ray::StaticVector{3})
    p = _radial_projection(m, ray)
    T = typeof(p.μ)
    p.ok || return fill(T(NaN), SMatrix{2, 3, T, 6})
    return _radial_derivs(p)[1]
end
@inline function CoordinateTransformations.transform_deriv_params(m::RadialHomography, ray::StaticVector{3})
    p = _radial_projection(m, ray)
    T = typeof(p.μ)
    p.ok || return fill(T(NaN), SMatrix{2, 1, T, 2})
    return SMatrix{2, 1, T, 2}(_radial_derivs(p)[2])
end

"""
    radial_transfer(m::RadialHomography, s) -> SVector{2} or nothing

Predicted OBSERVED target of source `s` through `m`: recenter the ray
`H s̃` and [`project`](@ref) it. Ray-based — no intermediate perspective
point, so it stays defined through and beyond 90° for `k < 0`.
"""
@inline radial_transfer(m::RadialHomography, s) = project(m, radial_ray(m, s))

# The centred ray `T₋c·H·s̃` of a source point, and of a direction `dH` along H.
@inline radial_ray(m::RadialHomography{T}, s) where {T} = radial_ray(SMatrix{3, 3, T, 9}(m.H), m.center, s)
@inline function radial_ray(H::StaticMatrix{3, 3}, c, s)
    T = eltype(H)
    h = H * SVector{3, T}(T(s[1]), T(s[2]), one(T))
    return Vec3{T}(h[1] - c[1] * h[3], h[2] - c[2] * h[3], h[3])
end

# =============================================================================
# LM rows: the √w-weighted transfer residuals and their chart Jacobian
# =============================================================================

"""
    OUT_OF_FIELD_RESIDUAL = 1.0e6

The residual of an observation a model cannot place — a ray outside a lens's or
division model's field, a point behind the camera — in the caller's units: the
ONE constant for such rows. In an LM row it is a CONSTANT (with a zero Jacobian
row), so a step that pushes observations out is charged for it and rejected —
never a zero residual, which would reward exactly that step; in a scoring
residual it is finite (not `Inf`/`NaN`), so sorting and squaring stay defined
while the point lands among the outliers. `1e6` rather than larger: its square
(`1e12`) leaves Float64 cost sums a resolution of ~`1e-4`, so the in-field rows
stay visible to LM next to a constant out-of-field one.
"""
const OUT_OF_FIELD_RESIDUAL = 1.0e6

# The two rows of one correspondence, in the uniform `(out, i, args...)` form
# `_foreach_row!` runs (a plain loop on the CPU, one kernel on a device).
Base.@propagate_inbounds function _radial_rows!(r, i, m, u₁, u₂, sqrtw)
    s = u₁[i]; d = u₂[i]
    p = _radial_projection(m, radial_ray(m, s))
    T = eltype(r)
    if p.ok
        d̂ = _radial_point(m, p)
        r[2i - 1] = sqrtw[2i - 1] * (d̂[1] - d[1])
        r[2i] = sqrtw[2i] * (d̂[2] - d[2])
    else
        r[2i - 1] = sqrtw[2i - 1] * T(OUT_OF_FIELD_RESIDUAL)
        r[2i] = sqrtw[2i] * T(OUT_OF_FIELD_RESIDUAL)
    end
    return nothing
end

Base.@propagate_inbounds function _radial_jacobian_rows!(J, i, m, dH, u₁, sqrtw)
    s = u₁[i]
    p = _radial_projection(m, radial_ray(m, s))
    N = length(dH)
    if !p.ok
        for q in 1:(N + 1)
            J[2i - 1, q] = zero(eltype(J))
            J[2i, q] = zero(eltype(J))
        end
        return nothing
    end
    D, ∂k = _radial_derivs(p)
    for q in 1:N
        g = D * radial_ray(dH[q], m.center, s)    # the centred ray is linear in H
        J[2i - 1, q] = sqrtw[2i - 1] * g[1]
        J[2i, q] = sqrtw[2i] * g[2]
    end
    J[2i - 1, N + 1] = sqrtw[2i - 1] * ∂k[1]
    J[2i, N + 1] = sqrtw[2i] * ∂k[2]
    return nothing
end

"""
    radial_transfer_rows!(r, m::RadialHomography, u₁, u₂, sqrtw) -> r
    radial_transfer_jacobian_rows!(J, m::RadialHomography, dH, u₁, sqrtw) -> J

The √w-weighted measurement-space transfer rows of `m` over `u₁[i] => u₂[i]`, two
per correspondence (`r[2i-1:2i] = sqrtw[2i-1:2i] .* (radial_transfer(m, u₁[i]) − u₂[i])`),
and their Jacobian: columns `1:N` along the directions `dH = (∂H/∂θ₁, …, ∂H/∂θ_N)`
of `H` (e.g. the reshaped columns of `retract_jacobian(HomographyManifold8(H₀), v)`),
column `N + 1` along `k` (`transform_deriv_params`). A ray outside the
model's field ([`project`](@ref)'s one edge rule) is a constant residual with a
ZERO Jacobian row, so LM rejects a step that pushes rays out rather than being
rewarded for it. One KernelAbstractions kernel each on device backends, a plain
loop over the same row functions on the CPU.
"""
function radial_transfer_rows!(r::AbstractVector, m::RadialHomography, u₁::AbstractVector, u₂::AbstractVector, sqrtw::AbstractVector)
    return _foreach_row!(_radial_rows!, r, length(u₁), m, u₁, u₂, sqrtw)
end

function radial_transfer_jacobian_rows!(J::AbstractMatrix, m::RadialHomography, dH::Tuple, u₁::AbstractVector, sqrtw::AbstractVector)
    return _foreach_row!(_radial_jacobian_rows!, J, length(u₁), m, dH, u₁, sqrtw)
end

"""
    CspondNormalization(::Type{<:AbstractRadialModel}, cs; center, mask = nothing)

The radial family's normalization of `plane_point => pixel` correspondences
(a [`RadialHomography`](@ref), a `RadialCamera`; the H/F one lives in
homography_solvers.jl): Hartley at the centroid on the plane side, Hartley
about the REQUIRED `center` (the distortion centre, the pencil's centre) on the
image side — the origin the minimal kernels are written about, with an
isotropic scale that keeps a division model in its class (`k′ = k/s²`) — over
`mask`'s correspondences and weighted by the `weights` column when there is
one. A centroid image-side frame would move the centre off the origin
(measured: 1.4 px misfit against 4e-13 px).
"""
CspondNormalization(::Type{<:AbstractRadialModel}, cs; center, mask = nothing) =
    CspondNormalization(HartleyNormalization(), HartleyNormalization(; center), cs, _bitmask(mask, length(cs)), weights_column(cs))

# -----------------------------------------------------------------------------
# Frame change — composition with the plane-side and image-side maps.
#
# A radial homography is a plane → pixel map, so a frame change is `∘`, as for
# every map: `D ∘ m ∘ B` for an image-side `D` and a plane-side `B` (a
# normalization moves it by `destination_map(N) ∘ m ∘ inv(source_map(N))`, and back
# by `inv(destination_map(N)) ∘ m ∘ source_map(N)`).
#
#   plane side (any collineation B):  H′ = H·B, k and c unchanged
#   image side (an ISOTROPIC similarity D = s·x + b — every normalization
#   strategy produces these; the division model commutes only with
#   similarities about its centre, and the centre moves with them):
#       H′ = D·H,   k′ = k / s²,   c′ = D(c)
#
# A non-isotropic image-side map would break the radial symmetry, and a
# projective one would not keep a division model: both are an `ArgumentError`.
# -----------------------------------------------------------------------------
@inline function Base.:∘(a::Union{HomographyMap{3}, _CTLeafTransformation}, m::RadialHomography{T}) where {T}
    D = SMatrix{3, 3}(to_homogeneous(a))
    s = D[1, 1]
    (iszero(D[3, 1]) && iszero(D[3, 2]) && isone(D[3, 3]) && iszero(D[1, 2]) && iszero(D[2, 1]) && D[2, 2] == s) ||
        throw(ArgumentError("∘ on a RadialHomography: the image-side map must be an isotropic similarity (s·x + b); got $D"))
    return RadialHomography{T}(
        HomographyMat{T}(Tuple(D * SMatrix{3, 3}(m.H))), T(rescale_radial_coeff(m.k, inv(s), 1)),
        Point2{T}(AffineMap(D)(m.center))
    )
end
@inline Base.:∘(m::RadialHomography{T}, b::Union{HomographyMap{3}, _CTLeafTransformation}) where {T} =
    RadialHomography{T}(HomographyMat{T}(Tuple(SMatrix{3, 3}(m.H) * SMatrix{3, 3}(to_homogeneous(b)))), m.k, m.center)

# -----------------------------------------------------------------------------
# Model-validity predicates (model_validity.jl vocabulary)
# -----------------------------------------------------------------------------

"""
    test_sample(::Type{<:RadialHomography}, cs, idx; target_tol = 1.0e-3) -> Bool

Degeneracy gate for the 5-point radial sample: the homography checks of the
four fully-used correspondences (orientation, coincidence, collinearity —
run on the distorted targets, an adequate proxy since the division model is
locally orientation-preserving) plus pairwise distinctness of all five.
Orientation and collinearity are scale-free; the SOURCE distinctness is a
fixed separation `1e-3` in `cs`'s units, and the TARGET distinctness is an
absolute separation `target_tol` in `cs`'s units — a caller holding the
normalization converts a pixel separation through its scale
(`target_tol = τ_px * similarity_scale(destination_map(normalization))`), so the
gate does not move with the frame.
"""
function test_sample(
        ::Type{<:RadialHomography}, cs, idx::AbstractVector{<:Integer};
        target_tol::Real = 1.0e-3
    )
    u = sources(cs); v = targets(cs)
    @inbounds src = (u[idx[1]], u[idx[2]], u[idx[3]], u[idx[4]])
    @inbounds dst = (v[idx[1]], v[idx[2]], v[idx[3]], v[idx[4]])
    # Both separations are in `cs`'s own units, never the F/E/H pixel default.
    source_tol = 1.0e-3
    return _orientation_preserving(src, dst) &&
        _not_collinear(src) && _not_collinear(dst) &&
        _points_distinct_idx(u, idx; tol_px = source_tol) &&
        _points_distinct_idx(v, idx; tol_px = target_tol)
end

"""
    test_model(m::RadialHomography, cs, idx) -> Bool

Feasibility of a radial-homography candidate: SVD conditioning of the `H`
part plus ray-space cheirality — the recentered `H s̃ᵢ` must be
proportional to `backproject(m, dᵢ)` with a UNIFORM sign across the
sample (`m` and `cs` in the same frame). Ray-based, so front/back
hemisphere consistency is checked without any perspective division.

The conditioning bound (`test_conditioning`, κ ≤ 100) is a property of the
frame `m` is written in, and it is calibrated for Hartley frames — the H/F
convention. `m` and `cs` are therefore meant to be in the normalization of
`CspondNormalization(RadialHomography, cs; center)`: Hartley on both sides,
the target about the distortion centre. (In a smaller-scale frame, e.g. the
earlier full-diagonal target frame, κ rejected true-model candidates:
measured 1 of 124.)
"""
function test_model(m::RadialHomography, cs, idx::AbstractVector{<:Integer})
    test_conditioning(m.H) || return false
    return _radial_cheirality(m, cs, idx)
end
test_model(m::RadialHomography, cs, mask::AbstractVector{Bool}) =
    test_conditioning(m.H) && _radial_cheirality(m, cs, _mask_indices(mask))

test_consensus(m::RadialHomography, cs, mask::AbstractVector{Bool}) =
    _radial_cheirality(m, cs, _mask_indices(mask))

# Uniform-sign cheirality of the H part against the division-model ray lifts:
# the homography's rule (`_test_uniform_cheirality`) with the recentred H ray
# `radial_ray` predicted and the division lift observed — the lift's signed
# third component makes front/back rays genuinely distinguishable (no
# perspective division).
function _radial_cheirality(m::RadialHomography, cs, indices)
    u = sources(cs); v = targets(cs)
    return _test_uniform_cheirality(i -> (radial_ray(m, u[i]), backproject(m, v[i])), indices)
end

"""
    radial_outline(m::RadialHomography, rect::HyperRectangle{2}; samples = 20) -> Vector{Point2{Float64}}

Boundary of a world-plane `rect` transferred through the radial model: each edge
sampled `samples` times (`_rect_edge_samples`) and pushed through
[`radial_transfer`](@ref); samples whose ray leaves the representable field are
DROPPED — the polygon shortens, it never carries a `nothing`. PER EDGE, not at
constant arc length (`boundary_point(::Rect, θ)`): every corner is sampled
exactly, because this outline is a board's image-plane QUAD and a quad whose
corners are missed is a different polygon. The one sampler behind a board's
image-plane quad and its Makie overlay (hoisted from BlobBoards, 2026-09).
"""
radial_outline(m::RadialHomography, r::HyperRectangle{2}; samples::Int = 20) =
    _rect_edge_samples(s -> radial_transfer(m, s), r, samples)

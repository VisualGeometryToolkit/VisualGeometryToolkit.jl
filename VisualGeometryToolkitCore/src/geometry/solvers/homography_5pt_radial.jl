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
    RadialHomography{T, N}

Radially distorted homography model: a [`HomographyMat`](@ref) `H` (plane point →
undistorted pixel) and a PROJECTIVE division camera on the TARGET side — the
`N` division coefficients `λ = (λ₁, …, λ_N)` (in squared pixel units, `λₚ`
multiplying `ρ²ᵖ`) about the distortion centre `c` (the camera's principal
point, COLMAP pixels). A correspondence `(s, d)` is explained when the
division-model RAY LIFT of the observation is proportional to the recentered
`H` ray:

    (d − c ; 1 + Σₚ λₚ‖d − c‖²ᵖ) ≃ λ · T₋c · H s̃

The camera is [`CameraModel`](@ref)`(m)` =
`CameraModel(ProjectiveIntrinsics(c), Division(λ...))`, built on demand: the
struct stores its primitives (`H`, `λ`, `center`), so it is isbits and runs on a
device. `RadialHomography{T}` (any `N`) stays a valid partial type; `N = 1` is
the one-coefficient model of [`homography_5pt_radial`](@ref), for which
[`division_coeff`](@ref) and the property `m.k` read `λ₁` (both throw an
`ArgumentError` for `N > 1` — never a silent `λ₁`).

The division model is an uncalibrated projection/back-projection model
([`project`](@ref)/[`backproject`](@ref)) — the lift's signed third
component represents rays beyond 90° (omnidirectional lenses), which no
image-space "undistortion" can. The solver frame is
origin-centered (`c = 0` — [`homography_5pt_radial`](@ref) constructs it
that way); composing back through a fixed-center normalization produces the
pixel-frame model with the center carried along, so consumers
([`radial_transfer`](@ref), quad projection) need no side-channel frame
data.

Constructors: `RadialHomography(H, camera)` (a `CameraModel{<:ProjectiveIntrinsics,
<:Division}`), and for `N = 1` `RadialHomography(H, k, c)`,
`RadialHomography(H, k)` and `RadialHomography{T}(H, k[, c])` (the centre
defaults to the origin). Produced by [`homography_5pt_radial`](@ref) /
[`Homography5ptRadial`](@ref).
"""
struct RadialHomography{T, N} <: AbstractRadialModel
    H::HomographyMat{T}
    λ::NTuple{N, T}
    center::Point2{T}
end

"The plane part `H` of a [`RadialHomography`](@ref): plane point → undistorted pixel."
homography(m::RadialHomography) = getfield(m, :H)

"""
    division_coeffs(m::RadialHomography{T, N}) -> NTuple{N, T}

The `N` division coefficients `(λ₁, …, λ_N)` of `m`'s image-side lens, about
[`center`](@ref), in squared centred-pixel units (`λₚ` multiplies `ρ²ᵖ`).
"""
division_coeffs(m::RadialHomography) = getfield(m, :λ)

"""
    division_coeff(m::RadialHomography{T, 1}) -> T

The division coefficient `k = λ₁` of a ONE-coefficient [`RadialHomography`](@ref)'s
image-side lens, about [`center`](@ref) (the property `m.k` is this). Throws an
`ArgumentError` for `N > 1`: read [`division_coeffs`](@ref) there.
"""
division_coeff(m::RadialHomography{T, 1}) where {T} = getfield(m, :λ)[1]
division_coeff(::RadialHomography{T, N}) where {T, N} = throw(
    ArgumentError("division_coeff: a RadialHomography with $N division coefficients has no single k; use division_coeffs")
)

# `m.k` is the one-coefficient model's λ₁ (the pre-N field); every other name is a field.
@inline Base.getproperty(m::RadialHomography, s::Symbol) = s === :k ? division_coeff(m) : getfield(m, s)
Base.propertynames(::RadialHomography{T, 1}, private::Bool = false) where {T} = (:H, :λ, :center, :k)
Base.propertynames(::RadialHomography, private::Bool = false) = (:H, :λ, :center)

# N = 1 constructors (the pre-N signatures), and coefficients as a tuple. The
# centre defaults to the origin — the solver frame.
RadialHomography{T}(H::AbstractMatrix, λ::Tuple, c::AbstractVector = zero(Point2{T})) where {T} =
    RadialHomography{T, length(λ)}(HomographyMat{T}(Tuple(H)), map(T, λ), Point2{T}(c[1], c[2]))
RadialHomography{T}(H::AbstractMatrix, k::Real, c::AbstractVector = zero(Point2{T})) where {T} =
    RadialHomography{T}(H, (k,), c)
RadialHomography(H::HomographyMat{T}, k::Real) where {T} = RadialHomography{T}(H, k)
RadialHomography(H::AbstractMatrix, k::Real, c::AbstractVector) =
    RadialHomography{float(promote_type(eltype(H), typeof(k), eltype(c)))}(H, k, c)

Base.zero(::Type{RadialHomography{T, N}}) where {T, N} =
    RadialHomography{T, N}(HomographyMat{T}(ntuple(_ -> zero(T), Val(9))), ntuple(_ -> zero(T), Val(N)), zero(Point2{T}))
Base.zero(::Type{RadialHomography{T}}) where {T} = zero(RadialHomography{T, 1})
Base.zero(m::RadialHomography) = zero(typeof(m))

@inline is_valid_solution(m::RadialHomography) =
    all(isfinite, m.H) && all(isfinite, m.λ) && all(isfinite, m.center)

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
    M = RadialHomography{T, 1}

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

# The lift's third component `1 + Σₚ λₚ ρ²ᵖ` at ρ² (one coefficient: `1 + k ρ²`).
@inline _lift_den(λ::NTuple{1}, ρ²) = 1 + λ[1] * ρ²
@inline _lift_den(λ::Tuple, ρ²) = 1 + ρ² * evalpoly(ρ², λ)

"""
    backproject(m::RadialHomography, d) -> SVector{3}

Back-project an observed target point to its (unnormalized) ray in the
model's CENTERED frame: `(d − c ; 1 + Σₚ λₚ‖d − c‖²ᵖ)` — the division-model
lift (`1 + k‖d − c‖²` for one coefficient). Total; the third component's sign
carries front/back hemisphere (zero at exactly 90°, negative beyond — valid
for omnidirectional lenses). The `H` part of the model relates source points
to these rays: `backproject(m, d) ≃ λ · T₋c · H s̃` (with `T₋c` the
recentering translation).
"""
@inline function backproject(m::RadialHomography{T}, d::StaticVector{2}) where {T}
    dx = T(d[1]) - m.center[1]
    dy = T(d[2]) - m.center[2]
    return Vec3{T}(dx, dy, _lift_den(m.λ, dx * dx + dy * dy))
end

"""
    project(m::RadialHomography, ray::StaticVector{3}) -> SVector{2} or nothing

Project a ray (in the model's CENTERED frame, as produced by the `H` part)
to the observed image point whose [`backproject`](@ref) lift is
proportional to it. One coefficient, in closed form: `c + μ·(x, y)` with
`μ = 2 / (z + √(z² − 4k ρ²))`, `ρ² = x² + y²` (the branch continuous with
`k → 0`, where it reduces to perspective division `μ = 1/z`). For `k < 0`
(fisheye/omni) this is total over front AND back rays; for `k > 0` it returns
`nothing` when `z² − 4kρ² < 0` (the ray lies outside the model's representable
field). `N > 1` coefficients: `c + project(Division(λ...), ray)`, the lens's own
solve on its monotone branch, `nothing` where the lens images nothing (past
its fold).
"""
@inline project(m::RadialHomography, ray::StaticVector{3}) = _radial_project(m, ray, _projection_lens(m))

@inline function _radial_project(m::RadialHomography, ray, lens)
    p = _radial_projection(m, ray, lens)
    p.ok || return nothing
    d = _radial_point(m, p)
    return Point2{eltype(d)}(d[1], d[2])
end

# The lens a model projects through: none for one coefficient (the closed form
# below), the `Division{N}` lens otherwise — built once per call site (building it
# solves for its branch ends), so the rows are handed it once per launch.
@inline _projection_lens(::RadialHomography{T, 1}) where {T} = nothing
@inline _projection_lens(m::RadialHomography) = Division(m.λ...)

# The ONE edge rule of the one-coefficient division-model projection, shared by
# `project`, its derivatives and the LM rows (and so by every scorer that calls
# `radial_transfer`): a ray is in the model's field iff it is off-axis with
# `z² − 4kρ² > 0` and `z + √(z² − 4kρ²)` not lost to rounding RELATIVE to `z`, or
# on-axis (ρ² below rounding relative to z²) and forward. Relative thresholds
# only, so a rescaled ray or frame gets the same verdict. Generic in the element
# type (Duals included) and device-safe. `N > 1`: the lens's rule (its image
# point finite).
@inline function _radial_projection(m::RadialHomography{<:Any, 1}, ray::StaticVector{3}, ::Nothing)
    T = float(promote_type(eltype(m.λ), eltype(ray)))
    x = T(ray[1]); y = T(ray[2]); z = T(ray[3]); k = T(m.λ[1])
    ρ² = x * x + y * y
    disc = z * z - 4 * k * ρ²
    S = sqrt(max(disc, zero(T)))
    den = z + S
    axial = ρ² < eps(T) * max(z * z, one(T))
    ok = axial ? z > zero(T) : disc > zero(T) && abs(den) >= eps(T) * max(abs(z), one(T))
    return (; x, y, z, k, ρ², S, μ = 2 / den, axial, ok)
end
@inline function _radial_projection(::RadialHomography, ray::StaticVector{3}, lens)
    u = project(lens, ray)
    return (; u, z = ray[3], ok = all(isfinite, u))
end

"""
    radial_in_front(m::RadialHomography, s) -> Bool

Whether source point `s` lands in `m`'s field ([`project`](@ref)'s edge rule)
AND in its FRONT hemisphere: the observed point's division lift has a positive
third component (one coefficient: `μ·z > 0`, the ray within 90° of the axis;
`N > 1`: the ray's `z > 0`, the lift on the lens's branch pointing the ray's
way). For `k ≥ 0` every in-field ray is in front (the lift's third component
`1 + kρ²` is positive), so this adds `z > 0` for `k < 0` only — where `project`
is total over back rays too (omnidirectional lenses). The extent gate of a
planar target: a board corner at or past 90° means the hypothesis spans ≥ 180°
across a plane, which no camera images.
"""
@inline function radial_in_front(m::RadialHomography, s)
    p = _radial_projection(m, radial_ray(m, s), _projection_lens(m))
    return p.ok && _in_front(m, p)
end
@inline _in_front(::RadialHomography{<:Any, 1}, p) = p.μ * p.z > zero(p.μ)
@inline _in_front(::RadialHomography, p) = p.z > zero(p.z)

# The predicted observation of an in-field ray: `c + μ·(x, y)`, the centre exactly
# on the axis (`N > 1`: `c + u`, the lens's image point).
@inline function _radial_point(m::RadialHomography{<:Any, 1}, p)
    T = typeof(p.μ)
    c1 = T(m.center[1]); c2 = T(m.center[2])
    return p.axial ? SVector{2, T}(c1, c2) : SVector{2, T}(c1 + p.μ * p.x, c2 + p.μ * p.y)
end
@inline function _radial_point(m::RadialHomography, p)
    T = eltype(p.u)
    return SVector{2, T}(T(m.center[1]) + p.u[1], T(m.center[2]) + p.u[2])
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

# Both derivatives of an in-field ray: along the ray (2 × 3) and along the
# coefficients (2 × N). One coefficient: the closed form above; `N > 1`: the
# lens's (ForwardDiff through its `project`).
@inline function _radial_jacobians(::RadialHomography{<:Any, 1}, p, ray, ::Nothing)
    D, gk = _radial_derivs(p)
    return D, SMatrix{2, 1}(gk[1], gk[2])
end
@inline _radial_jacobians(::RadialHomography, p, ray, lens) =
    (transform_deriv(lens, ray), transform_deriv_params(lens, ray))

"""
    transform_deriv(m::RadialHomography, ray::StaticVector{3}) -> SMatrix{2,3}
    transform_deriv_params(m::RadialHomography{T, N}, ray::StaticVector{3}) -> SMatrix{2,N}

The derivatives of [`project`](@ref)`(m, ray)`: along the (centred) ray, and along
the division coefficients `λ₁ … λ_N` (the model's lens parameters; `H` and the
centre act before the ray). Closed form for one coefficient; for `N > 1` the
`Division{N}` lens's own (ForwardDiff through its `project`). NaN where
`project` returns `nothing` — the same edge rule, as for the lens models beyond
their field.
"""
@inline function CoordinateTransformations.transform_deriv(m::RadialHomography, ray::StaticVector{3})
    lens = _projection_lens(m)
    p = _radial_projection(m, ray, lens)
    T = _projection_eltype(p)
    p.ok || return fill(T(NaN), SMatrix{2, 3, T, 6})
    return SMatrix{2, 3, T, 6}(_radial_jacobians(m, p, ray, lens)[1])
end
@inline function CoordinateTransformations.transform_deriv_params(m::RadialHomography{<:Any, N}, ray::StaticVector{3}) where {N}
    lens = _projection_lens(m)
    p = _radial_projection(m, ray, lens)
    T = _projection_eltype(p)
    p.ok || return fill(T(NaN), SMatrix{2, N, T})
    return SMatrix{2, N, T}(_radial_jacobians(m, p, ray, lens)[2])
end
@inline _projection_eltype(p) = haskey(p, :μ) ? typeof(p.μ) : eltype(p.u)

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
# `_foreach_row!` runs (a plain loop on the CPU, one kernel on a device). `lens`
# is `_projection_lens(m)`, built once per launch.
Base.@propagate_inbounds function _radial_rows!(r, i, m, lens, u₁, u₂, sqrtw)
    s = u₁[i]; d = u₂[i]
    p = _radial_projection(m, radial_ray(m, s), lens)
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

# The row functions without a lens argument (one row at a time, the lens built per
# row: free for one coefficient, whose lens is `nothing`).
Base.@propagate_inbounds _radial_rows!(r, i, m, u₁, u₂, sqrtw) = _radial_rows!(r, i, m, _projection_lens(m), u₁, u₂, sqrtw)
Base.@propagate_inbounds _radial_jacobian_rows!(J, i, m, dH, u₁, sqrtw) = _radial_jacobian_rows!(J, i, m, _projection_lens(m), dH, u₁, sqrtw)

Base.@propagate_inbounds function _radial_jacobian_rows!(J, i, m::RadialHomography{<:Any, N}, lens, dH, u₁, sqrtw) where {N}
    s = u₁[i]
    ray = radial_ray(m, s)
    p = _radial_projection(m, ray, lens)
    nH = length(dH)
    if !p.ok
        for q in 1:(nH + N)
            J[2i - 1, q] = zero(eltype(J))
            J[2i, q] = zero(eltype(J))
        end
        return nothing
    end
    D, ∂λ = _radial_jacobians(m, p, ray, lens)
    for q in 1:nH
        g = D * radial_ray(dH[q], m.center, s)    # the centred ray is linear in H
        J[2i - 1, q] = sqrtw[2i - 1] * g[1]
        J[2i, q] = sqrtw[2i] * g[2]
    end
    for j in 1:N
        J[2i - 1, nH + j] = sqrtw[2i - 1] * ∂λ[1, j]
        J[2i, nH + j] = sqrtw[2i] * ∂λ[2, j]
    end
    return nothing
end

"""
    radial_transfer_rows!(r, m::RadialHomography, u₁, u₂, sqrtw) -> r
    radial_transfer_jacobian_rows!(J, m::RadialHomography, dH, u₁, sqrtw) -> J

The √w-weighted measurement-space transfer rows of `m` over `u₁[i] => u₂[i]`, two
per correspondence (`r[2i-1:2i] = sqrtw[2i-1:2i] .* (radial_transfer(m, u₁[i]) − u₂[i])`),
and their Jacobian: columns `1:M` along the directions `dH = (∂H/∂θ₁, …, ∂H/∂θ_M)`
of `H` (e.g. the reshaped columns of `retract_jacobian(HomographyManifold8(H₀), v)`),
columns `M + 1 … M + N` along the division coefficients `λ₁ … λ_N`
(`transform_deriv_params`). A ray outside the
model's field ([`project`](@ref)'s edge rule) is a constant residual with a
ZERO Jacobian row, so LM rejects a step that pushes rays out rather than being
rewarded for it. One KernelAbstractions kernel each on device backends, a plain
loop over the same row functions on the CPU.
"""
function radial_transfer_rows!(r::AbstractVector, m::RadialHomography, u₁::AbstractVector, u₂::AbstractVector, sqrtw::AbstractVector)
    return _foreach_row!(_radial_rows!, r, length(u₁), m, _projection_lens(m), u₁, u₂, sqrtw)
end

"""
    radial_transfer_jacobian_rows!(J, m::RadialHomography, dH, u₁, sqrtw) -> J

The Jacobian of [`radial_transfer_rows!`](@ref): columns `1:M` along the
`M` directions `dH` of `H`, columns `M + 1 … M + N` along the `N` division
coefficients, a zero row for a ray outside the model's field.
"""
function radial_transfer_jacobian_rows!(J::AbstractMatrix, m::RadialHomography, dH::Tuple, u₁::AbstractVector, sqrtw::AbstractVector)
    return _foreach_row!(_radial_jacobian_rows!, J, length(u₁), m, _projection_lens(m), dH, u₁, sqrtw)
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
#   plane side (any collineation B):  H′ = H·B, λ and c unchanged
#   image side (an ISOTROPIC similarity D = s·x + b — every normalization
#   strategy produces these; the division model commutes only with
#   similarities about its centre, and the centre moves with them):
#       H′ = D·H,   λₚ′ = λₚ / s²ᵖ,   c′ = D(c)
#
# A non-isotropic image-side map would break the radial symmetry, and a
# projective one would not keep a division model: both are an `ArgumentError`.
# -----------------------------------------------------------------------------
@inline function Base.:∘(a::Union{HomographyMap{3}, _CTLeafTransformation}, m::RadialHomography{T, N}) where {T, N}
    D = SMatrix{3, 3}(to_homogeneous(a))
    s = D[1, 1]
    (iszero(D[3, 1]) && iszero(D[3, 2]) && isone(D[3, 3]) && iszero(D[1, 2]) && iszero(D[2, 1]) && D[2, 2] == s) ||
        throw(ArgumentError("∘ on a RadialHomography: the image-side map must be an isotropic similarity (s·x + b); got $D"))
    return RadialHomography{T}(
        HomographyMat{T}(Tuple(D * SMatrix{3, 3}(m.H))), ntuple(p -> T(rescale_radial_coeff(m.λ[p], inv(s), p)), Val(N)),
        Point2{T}(affine_part(D)(m.center))
    )
end
@inline Base.:∘(m::RadialHomography{T}, b::Union{HomographyMap{3}, _CTLeafTransformation}) where {T} =
    RadialHomography{T}(HomographyMat{T}(Tuple(SMatrix{3, 3}(m.H) * SMatrix{3, 3}(to_homogeneous(b)))), m.λ, m.center)

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

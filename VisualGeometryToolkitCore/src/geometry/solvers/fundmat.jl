# =============================================================================
# Fundamental Matrix Solvers — Public API for epipolar geometry estimation
# =============================================================================
#
# Pure geometry solvers for estimating fundamental matrices from point
# correspondences. These functions are independent of RANSAC.
#
# DEPENDENCY: Uses StaticArrays, LinearAlgebra.
#             Uses HartleyNormalization, sign_normalize from homography_solvers.jl
#             Uses Line2D from .Primitives (for typed epipolar-line callable).
#
# =============================================================================

using .Primitives: Line2D

# =============================================================================
# Public: 7-Point Fundamental Matrix Solver
# =============================================================================

"""
    fundamental_7pt(s₁..s₇, d₁..d₇) -> FixedModels{3,FundMat{T}} | nothing

Compute fundamental matrices from exactly 7 point correspondences; `T` is the
points' element type.

Returns `FixedModels{3,FundMat{T}}` with 1–3 Frobenius-normalized solutions
(F[3,3] >= 0), or `nothing` if the system is degenerate. Zero heap allocations.

# Algorithm
1. Build 7×9 DLT matrix from Kronecker products
2. Partitioned A\\B solve for 2D null space (zero-allocation)
3. Cubic determinant constraint: det(λ f₁ + f₂) = 0
4. 1 or 3 real roots → 1 or 3 F matrices

# Example
```julia
result = fundamental_7pt(s₁, s₂, s₃, s₄, s₅, s₆, s₇,
                         d₁, d₂, d₃, d₄, d₅, d₆, d₇)
if result !== nothing
    for F in result  # iterates 1–3 FundMat solutions
        # use F...
    end
end
```
"""
function fundamental_7pt(
        s₁::StaticVector{2, T}, s₂::StaticVector{2, T}, s₃::StaticVector{2, T},
        s₄::StaticVector{2, T}, s₅::StaticVector{2, T}, s₆::StaticVector{2, T},
        s₇::StaticVector{2, T},
        d₁::StaticVector{2, T}, d₂::StaticVector{2, T}, d₃::StaticVector{2, T},
        d₄::StaticVector{2, T}, d₅::StaticVector{2, T}, d₆::StaticVector{2, T},
        d₇::StaticVector{2, T}
    ) where {T}
    o = one(T)

    # Build 7×9 DLT matrix: each row = kron(u₂, u₁)ᵀ
    M = @SMatrix [
        d₁[1] * s₁[1]  d₁[1] * s₁[2]  d₁[1]  d₁[2] * s₁[1]  d₁[2] * s₁[2]  d₁[2]  s₁[1]  s₁[2]  o;
        d₂[1] * s₂[1]  d₂[1] * s₂[2]  d₂[1]  d₂[2] * s₂[1]  d₂[2] * s₂[2]  d₂[2]  s₂[1]  s₂[2]  o;
        d₃[1] * s₃[1]  d₃[1] * s₃[2]  d₃[1]  d₃[2] * s₃[1]  d₃[2] * s₃[2]  d₃[2]  s₃[1]  s₃[2]  o;
        d₄[1] * s₄[1]  d₄[1] * s₄[2]  d₄[1]  d₄[2] * s₄[1]  d₄[2] * s₄[2]  d₄[2]  s₄[1]  s₄[2]  o;
        d₅[1] * s₅[1]  d₅[1] * s₅[2]  d₅[1]  d₅[2] * s₅[1]  d₅[2] * s₅[2]  d₅[2]  s₅[1]  s₅[2]  o;
        d₆[1] * s₆[1]  d₆[1] * s₆[2]  d₆[1]  d₆[2] * s₆[1]  d₆[2] * s₆[2]  d₆[2]  s₆[1]  s₆[2]  o;
        d₇[1] * s₇[1]  d₇[1] * s₇[2]  d₇[1]  d₇[2] * s₇[1]  d₇[2] * s₇[2]  d₇[2]  s₇[1]  s₇[2]  o
    ]

    # 2D null space via partitioned solve (zero-allocation).
    # Partition M = [A | B] where A is 7×7 (cols 1-7), B is 7×2 (cols 8-9).
    # If A is nonsingular, null(M) is spanned by [-A⁻¹B; I₂].
    # The 7×7 LU solve is stack-allocated via StaticArrays.
    A = M[:, SOneTo(7)]
    B = M[:, SA[8, 9]]
    F = lu(A; check = false)         # `check=false` → no SingularException
    issuccess(F) || return nothing # zero pivot → degenerate sample
    X = F \ (-B)                   # 7×2 SMatrix, zero-alloc
    any(isnan, X) && return nothing
    f₁ = SVector{9, T}(X[1, 1], X[2, 1], X[3, 1], X[4, 1], X[5, 1], X[6, 1], X[7, 1], one(T), zero(T))
    f₂ = SVector{9, T}(X[1, 2], X[2, 2], X[3, 2], X[4, 2], X[5, 2], X[6, 2], X[7, 2], zero(T), one(T))

    return _rank_two_constraint(f₁, f₂, T)
end

# =============================================================================
# Public: N-Point DLT with Hartley Normalization + Rank-2 Enforcement
# =============================================================================

"""
    fundamental_dlt(cs; mask = nothing, svd_ws = nothing) -> FundMat or nothing

Compute a fundamental matrix from N >= 8 correspondences `cs` (`u₁ => u₂`
pairs; any [`sources`](@ref)/[`targets`](@ref) container, a
[`Csponds`](@ref) preferred) by the Direct Linear Transform with Hartley
normalization and rank-2 enforcement.

Per-correspondence weights are the container's `weights` column (set with
[`withattrs`](@ref)), never a keyword. When `mask` (a `BitVector` or an
index list) is given, only those correspondences are used.

# Example
```julia
cs = Csponds(pts1, pts2)
F = fundamental_dlt(cs)
F = fundamental_dlt(cs; mask = inlier_mask)
```
"""
function fundamental_dlt(
        cs::AbstractVector{<:Pair{<:StaticVector{2, T}, <:StaticVector{2, T}}};
        mask = nothing, svd_ws = nothing
    ) where {T}
    n = length(cs)
    A = Matrix{T}(undef, n, 9)
    # the one normalization of this estimator: built here, once, from the
    # data (and its `weights` column); the `!` form only uses it
    return fundamental_dlt!(A, CspondNormalization(FundMat, cs; mask), cs; mask, svd_ws)
end

# One correspondence's DLT row in normalized homogeneous points `s`, `d`: the
# Kronecker product `d ⊗ s` (row-major vec(F)), so `A·vec(F) = dᵀ F s`. The
# per-point filler of the shared `_dlt!` pipeline (homography_solvers.jl).
Base.@propagate_inbounds function _fundmat_dlt_row!(A, r, s, d)
    A[r, 1] = d[1] * s[1]; A[r, 2] = d[1] * s[2]; A[r, 3] = d[1] * s[3]
    A[r, 4] = d[2] * s[1]; A[r, 5] = d[2] * s[2]; A[r, 6] = d[2] * s[3]
    A[r, 7] = d[3] * s[1]; A[r, 8] = d[3] * s[2]; A[r, 9] = d[3] * s[3]
    return nothing
end

"""
    fundamental_dlt!(A, N::CspondNormalization, cs; mask = nothing, svd_ws = nothing) -> FundMat or nothing

In-place N-point DLT for the fundamental matrix of the correspondences `cs`
using the pre-allocated buffer `A` (N × 9). Weights are the container's
`weights` column, as for [`fundamental_dlt`](@ref).

When `mask::BitVector` is provided, only masked points are used and rows are
compacted into the top of `A`. The inner-loop form: `N` is the enclosing
estimator's [`CspondNormalization`](@ref), built once at its entry (the public
form builds it from the data); this form applies it row by row and composes the
model back into the data frame (`inv(D) ∘ F ∘ S`), and never
normalizes on its own.

No covariance is returned — as for [`homography_dlt!`](@ref): the SVD's resolvent
covariance is not this estimator's under pixel noise (algebraic rows are
heteroscedastic); its first-order covariance, if ever needed, is
[`implicit_covariance`](@ref)'s block form with the DLT's identity weighting.
"""
function fundamental_dlt!(
        A::AbstractMatrix{T},
        N::CspondNormalization{T},
        cs::AbstractVector{<:Pair{<:StaticVector{2, T}, <:StaticVector{2, T}}};
        mask = nothing,
        svd_ws = nothing
    ) where {T}
    # N is the enclosing estimator's normalization: used here, never built here
    F_norm = _dlt!(_fundmat_dlt_row!, A, N, cs, 1, mask, svd_ws)
    F_norm === nothing && return nothing
    return sign_normalize(inv(destination_map(N)) ∘ enforce_rank_two(FundMat{T}(Tuple(F_norm))) ∘ source_map(N))   # composed back to the data frame, then the gauge
end

"""
    triangulate(pose::EuclideanMap{3}, f₁, f₂) -> (X, λ₁, λ₂)

Triangulate a correspondence of UNIT BEARINGS: the midpoint of the mutual
perpendicular between the two rays, in VIEW 1's frame, together with both ray
depths. `pose` maps view 1 to view 2 (`X₂ = R·X₁ + t`).

The depths are the point of returning them. Cheirality on rays is `λ₁ > 0 && λ₂
> 0` — the point is in front of both cameras along their own rays — which is
the plane-free replacement for testing `X[3] > 0`. A z-depth test silently
assumes a forward hemisphere the ray domain does not have, so it is wrong for
any camera that sees past 90°.

Returns `λ₁ = λ₂ = NaN` for parallel rays (`|f₁·R'f₂| → 1`), which is
degenerate rather than exceptional: it is what zero parallax looks like, and
the caller decides. Never throws.

# Why this form, and what it costs

It is the only form that GENERALISES. The pixel-domain DLT (`uᵢ × Pᵢ X̃ = 0`)
builds `P = K[I|0]` — pinhole by construction — and calibrated geometry in this
stack must work for fisheye and omnidirectional cameras, where that matrix does
not exist. That is the deciding argument, and it is not a number.

MEASURED against that DLT rather than assumed better, paired (identical
scenes, points and noise draws), binned by parallax, on relative 3-D error.
Reprojection error cannot discriminate the two — it sits at the noise level
even where the 3-D error reaches 363 — so it is reported only as a sanity
check. The trade is real and mixed:

| apex angle | med DLT → ray | q99 DLT → ray | ray wins (paired) |
|---|---|---|---|
| [0, 0.1°)  | 0.488 → 0.567 | **21.9 → 1.73** | 43.3% |
| [0.1, 0.3°)| 0.170 → 0.174 | **3.69 → 1.38** | 45.4% |
| [0.3, 1°)  | 0.0592 → 0.0585 | 0.354 → 0.336 | 48.0% |
| [1, 3°)    | 0.0202 → 0.0201 | 0.092 → 0.0925 | 47.8% |
| [3, 90°)   | 0.00597 → 0.00595 | 0.0279 → 0.0279 | 47.6% |

So: it WINS ONLY IN THE TAIL, and decisively there — a q99 of 21.9 is a point
placed at 22× its true distance, and those are what corrupt a bundle or a
cheirality vote downstream. It is slightly WORSE on the median below 0.1°
(43.3% of paired points, i.e. it loses more often than it wins), in a regime
where both answers are already about half the true distance. Above 0.3° the
two are indistinguishable to three significant figures. Noise-free both are
exact to 1e-13–1e-16, so the entire difference is noise propagation near
degeneracy, not conditioning.

A NEGATIVE RESULT worth keeping: ray-depth cheirality does NOT reduce the
behind-camera rate. Below 0.1° parallax it is 20.1% for the DLT's z-test and
**20.4%** here; in [0.1, 0.3°) both are 0.23%. That was predicted to improve
and measured not to — at that parallax the noisy rays genuinely cross behind a
camera, so it is the geometry, not the test. Ray depth is still the right
statement (a z-test assumes a forward hemisphere the ray domain lacks), but it
buys correctness of MEANING, not accuracy.
"""
@inline function triangulate(pose::PlainEuclideanMap{3}, f₁::StaticVector{3}, f₂::StaticVector{3})
    T = float(promote_type(eltype(f₁), eltype(f₂)))
    R = SMatrix{3, 3, T, 9}(pose.R)
    t = SVector{3, T}(pose.t)
    a = SVector{3, T}(f₁[1], f₁[2], f₁[3])
    d = R' * SVector{3, T}(f₂[1], f₂[2], f₂[3])   # ray 2's direction in view 1
    w = -(R' * t)                                  # camera 2's centre in view 1
    # CLAMP before squaring: for exactly parallel rays `dot(a, d)` rounds to
    # either side of ±1, and an unclamped `1 - b²` then lands at ±1e-16 — so the
    # degeneracy guard would fire or not depending on the rounding direction.
    # Clamped, parallel rays give exactly zero and the guard is deterministic.
    b = clamp(dot(a, d), -one(T), one(T))
    den = one(T) - b * b                           # sin² of the parallax angle; zero iff parallel
    abs(den) > eps(T) || return (Point3{T}(T(NaN), T(NaN), T(NaN)), T(NaN), T(NaN))
    aw = dot(a, w)
    dw = dot(d, w)
    λ₁ = (aw - b * dw) / den
    λ₂ = (b * aw - dw) / den
    p₁ = λ₁ * a
    p₂ = w + λ₂ * d
    X = (p₁ + p₂) / 2
    return (Point3{T}(X[1], X[2], X[3]), λ₁, λ₂)
end

"""
    test_cheirality(pose::EuclideanMap{3}, f₁, f₂) -> Bool

Ray-domain cheirality: triangulate the bearing pair and require BOTH depths
positive. The plane-free replacement for `X[3] > 0`, which assumes a forward
hemisphere that a fisheye or omnidirectional camera does not have.

Parallel rays (no parallax) are rejected — `NaN > 0` is false — which is the
honest answer: nothing was triangulated.
"""
@inline function test_cheirality(pose::PlainEuclideanMap{3}, f₁::StaticVector{3}, f₂::StaticVector{3})
    _, λ₁, λ₂ = triangulate(pose, f₁, f₂)
    return λ₁ > 0 && λ₂ > 0
end

"""
    cheiral_pose_fast(E, f₁, f₂) -> (R::SMatrix{3,3}, t::SVector{3})

The cheirality vote: of [`motion`](@ref)`(E)`'s four `(R, t)` candidates, the one
with the most bearing pairs in front of BOTH cameras by the ray-domain
[`test_cheirality`](@ref) (ties go to the earlier candidate). `f₁`, `f₂` are
equal-length vectors of unit bearings — the caller back-projects with its own
camera; the z = 1 chart this used to build through K⁻¹ does not exist for a ray
at 90°. `t` is unit (the essential's translation direction).

Ray depth is right in meaning and camera-general, not a rate fix: see
[`triangulate`](@ref)'s recorded negative result before re-attempting it as one.
Moved here from VisualGeometryToolkitEstimation's relative-pose bundle, which
the evaluation protocol's cheirality also called.
"""
function cheiral_pose_fast(E, f₁::AbstractVector, f₂::AbstractVector)
    length(f₁) == length(f₂) || throw(DimensionMismatch(
        "cheiral_pose_fast: $(length(f₁)) source bearings but $(length(f₂)) target bearings"
    ))
    bestR = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    bestt = SVector{3, Float64}(0.0, 0.0, 1.0); best_cnt = -1
    for (R, t) in motion(SMatrix{3, 3, Float64, 9}(E))
        Rs = SMatrix{3, 3, Float64, 9}(R); ts = SVector{3, Float64}(t)
        pose = EuclideanMap(Rs, ts)
        cnt = 0
        @inbounds for (a, b) in zip(f₁, f₂)
            test_cheirality(pose, a, b) && (cnt += 1)
        end
        cnt > best_cnt && (best_cnt = cnt; bestR = Rs; bestt = ts)
    end
    return bestR, bestt
end

# =============================================================================
# Public: Sampson Distance
# =============================================================================

# THE Sampson distance, of every correspondence model: the whitened residual
# of `constraint_with_jacobians` (`whiten`) at the model's rounding scales
# (`constraint_rounding_scale`), so the distance, the scorers that whiten the
# same constraint, and the one degeneracy rule cannot disagree. Plain points
# are unit isotropic noise (`Σ_x = I`); an `Uncertain` point carries its own
# covariance (the Mahalanobis Sampson distance), an `exact` one none
# (conventions/measurement-noise.md). A model's per-type facts
# are its `constraint_with_jacobians` and its rounding scales — nothing here.
"""
    sampson_distance(F::FundMat, u₁, u₂)
    sampson_distance(E::EssentialMat, x̂₁, x̂₂)
    sampson_distance(H::HomographyMat, u₁, u₂)
    sampson_distance(M, c::Pair)

First-order approximation to the geometric error of the correspondence
`u₁ => u₂` under the model, model first: the norm of the whitened constraint,

    d = ‖L⁻¹g‖,   LLᵀ = G_x·Σ_x·G_xᵀ,

`g` and `G_x = ∂g/∂(u₁, u₂)` from [`constraint_with_jacobians`](@ref) — for the
epipolar constraint `u₂ᵀ F u₁ = 0` the familiar
`|u₂ᵀ F u₁| / sqrt((Fu₁)₁² + (Fu₁)₂² + (Fᵀu₂)₁² + (Fᵀu₂)₂²)`, for a homography
`√(gᵀC⁻¹g)` over its two transfer rows. Pixel-space inputs for `FundMat` and
`HomographyMat`, gnomonic ([`GnomonicFrame`](@ref)-coordinate) inputs for
`EssentialMat`. Accepts plain `Point2`/`SVector{2}` (unit isotropic, `Σ_x = I`),
[`Uncertain`](@ref) points (Mahalanobis, the per-point covariance or precision)
and [`exact`](@ref) points (zero covariance; not both endpoints) — the
measurement-noise rule of [`pair_covariance`](@ref); a 3-vector
bearing is a `MethodError`, never read as its first two components. The
frame-checked face is `sampson_distance(::EpipolarForm, x₁, x₂)`.

DEGENERATE (the measurement gradient vanishes to rounding — both points at the
epipoles, an epipolar line at infinity, a point a homography maps to the line
at infinity): `0` when the constraint vanishes too (the pair satisfies it
exactly), `Inf` when it does not (no nearby point can) — the one rule of
[`whiten`](@ref), at the model's [`constraint_rounding_scale`](@ref), so the
answer does not depend on the coordinates' scale. NaN propagates.
"""
@inline sampson_distance(M::Union{AbstractFundMat, HomographyMat}, u₁, u₂) =
    _whitened_norm(_sampson_residual(M, u₁, u₂))
@inline sampson_distance(M::Union{AbstractFundMat, HomographyMat}, c::Pair) = sampson_distance(M, c.first, c.second)

"""
    signed_sampson_distance(M, u, v) -> T

Signed Sampson distance of a codimension-1 model `M` (fundamental / essential)
at the correspondence `u => v`: `g / ‖∂g‖` — [`whiten`](@ref)'s residual of
[`constraint_with_jacobians`](@ref), whose magnitude is [`sampson_distance`](@ref).
The **sign** encodes which side of the epipolar line the point falls on.
Keeping it signed is what makes an LM residual smooth and its Jacobian
sign-consistent.

DEGENERATE (the point gradient vanishes to rounding): the unsigned distance's
rule, signed — `0` when the constraint vanishes with it, `±Inf` (the sign of
`g`) when it does not. Both tests are relative ([`epipolar_rounding`](@ref)),
so a model's homogeneous scale never moves the result.
"""
@inline signed_sampson_distance(M::AbstractFundMat, u, v) = _sampson_residual(M, u, v)::Real

# The whitened constraint residual of one correspondence, in the promoted
# element type of the model and the points.
@inline function _sampson_residual(M, u₁, u₂)
    a = geom(u₁); b = geom(u₂)
    T = float(promote_type(eltype(M), eltype(a), eltype(b)))
    Mt = _with_eltype(M, T); x₁ = _plane_point(T, a); x₂ = _plane_point(T, b)
    g, G_x, _ = constraint_with_jacobians(Mt, x₁, x₂)
    return _whiten_residual(g, G_x, _pair_covariance(u₁, u₂), constraint_rounding_scale(Mt, x₁, x₂))
end
# A plane measurement as the model's 2-vector: typed, so a 3-vector is refused.
@inline _plane_point(::Type{T}, u::StaticVector{2}) where {T} = SVector{2, T}(u[1], u[2])
# A model matrix in the element type T (the same wrapper).
@inline _with_eltype(M::StaticMatrix, ::Type{T}) where {T} =
    eltype(M) === T ? M : Base.typename(typeof(M)).wrapper{T}(Tuple(M))
# The 4×4 measurement covariance of (u₁, u₂) is `_pair_covariance`
# (base/measurement_noise.jl): `I` for plain points (the isotropic Sampson
# distance), the block diagonal of the points' own otherwise, `exact` a zero block.
# The distance of a whitened residual: its magnitude for a scalar, its norm for a vector.
@inline _whitened_norm(r::Real) = abs(r)
@inline _whitened_norm(r::StaticVector) = norm(r)

"""
    epipolar_rounding(F, s, d) -> (tol_J, tol_g)

The degeneracy scales of the epipolar constraint `dᵀ F s` at the homogeneous
points `s = (x₁, y₁, 1)`, `d = (x₂, y₂, 1)`: the rounding bound `tol_J` of
every point-gradient component (|F|·|s|, |F|ᵀ·|d|) and `tol_g` of the
constraint value (|d|ᵀ·|F|·|s|), loose by a constant. A correspondence whose
squared point gradient is `≤ tol_J²` is DEGENERATE, and the one Sampson
degeneracy rule answers it with `0` when `|g| ≤ tol_g` and `Inf` otherwise
(see [`sampson_distance`](@ref), [`signed_sampson_distance`](@ref)).

Relative to the model's own entries, so its homogeneous scale never decides.
"""
@inline function epipolar_rounding(F, s, d)
    nF = sum(abs, F); ns = sum(abs, s); nd = sum(abs, d)
    tol = 8 * eps(typeof(nF))
    return tol * nF * max(ns, nd), tol * nF * ns * nd
end

# The epipolar models' rounding scales (`constraint_rounding_scale`), at the
# homogeneous points — the verdicts `whiten` and the LM rows both read.
@inline function constraint_rounding_scale(M::AbstractFundMat, u::StaticVector{2}, v::StaticVector{2})
    T = float(promote_type(eltype(M), eltype(u), eltype(v)))
    gradient, value = epipolar_rounding(M, _lift_image_point(u, T), _lift_image_point(v, T))
    return (; gradient, value)
end

# =============================================================================
# Typed callable: point → epipolar Line2D
# =============================================================================
#
# Mirrors the `(::HomographyMap)(::Line2D)` typed-callable pattern from
# `transforms/ct_contravariant.jl`, but with one-way semantics: the action
# is a rank-2 CORRELATION — point to hyperplane — not a collineation (point
# to point), so F (and E) are NOT invertible transformations:
#
#   `F(u₁) = ℓ₂` produces the epipolar line in image 2 of point u₁ in image 1.
#   `F'(u₂) = ℓ₁` produces the epipolar line in image 1 of point u₂ in image 2.
#
# The transpose direction is reached via `Adjoint{<:Any,<:AbstractFundMat}`
# which Julia constructs from `F'` / `transpose(F)` automatically.
#
# Calling convention is identical to homography's contravariant line API
# in surface form (typed return, single positional arg) — but the math is
# a bilinear-form contraction, not a contravariant transformation. The
# frame-labeled face of the same action is `epipolar_map(::EpipolarForm)`
# (geometry/forms.jl), which also documents WHY there is no inverse.

"""
    (F::FundMat)(u::SVector{2})    -> Line2D
    (E::EssentialMat)(x̂::SVector{2}) -> Line2D

The epipolar CORRELATION, applied: the epipolar line in image 2 of the
input point in image 1 (`ℓ₂ = F ũ₁` — a point-to-hyperplane action, one-way
by rank 2; contrast the collineation action of `HomographyMap`). Pixel
input for `FundMat`, gnomonic ([`GnomonicFrame`](@ref)-coordinate) input for
`EssentialMat`. Use `F'(u)` / `E'(x̂)` for the reverse direction.

The Mat callable is the frame-free surface; the slot-labeled equivalent is
[`epipolar_map`](@ref) on an [`EpipolarForm`](@ref), whose frame tags make
the direction checkable. The degenerate directions are [`epipoles`](@ref).
"""
@inline function (F::AbstractFundMat{T})(u::StaticVector{2, T}) where {T}
    return Line2D{T}(F * to_affine(u))
end

@inline function (Fᵀ::Adjoint{<:Any, <:AbstractFundMat{T}})(u::StaticVector{2, T}) where {T}
    return Line2D{T}(Fᵀ * to_affine(u))
end

# =============================================================================
# Public: Sampson Jacobians — Constraint + Jacobians for AbstractFundMat
# =============================================================================

"""
    constraint_with_jacobians(F::FundMat, u₁::SVector{2}, u₂::SVector{2})
    constraint_with_jacobians(E::EssentialMat, x̂₁::SVector{2}, x̂₂::SVector{2})
        -> (g::T, G_x::SVector{4,T}, G_θ::SVector{9,T})

Compute the epipolar constraint `g` and its derivatives.

The scalar constraint `g = ũ₂ᵀ F ũ₁` (one equation, d_g = 1).  Identical
algebra for F and E — the wrapper type signals input coordinate space
(pixels for `FundMat`, gnomonic points for `EssentialMat`).

Returns:
- `g::T`              — epipolar constraint value
- `G_x::SVector{4,T}` — ∂g/∂(x₁,y₁,x₂,y₂) (measurement Jacobian)
- `G_θ::SVector{9,T}` — ∂g/∂vec(F) (model Jacobian, column-major)
"""
@inline function constraint_with_jacobians(
        F::AbstractFundMat{T},
        u₁::StaticVector{2, T},
        u₂::StaticVector{2, T}
    ) where {T}
    s = _lift_image_point(u₁, T)
    d = _lift_image_point(u₂, T)

    g = dot(d, F * s)

    Fs = F * s     # epipolar line in image 2
    Ftd = F' * d   # epipolar line in image 1

    G_x = SVector{4, T}(Ftd[1], Ftd[2], Fs[1], Fs[2])
    G_θ = SVector{9, T}(
        d[1] * s[1], d[2] * s[1], d[3] * s[1],
        d[1] * s[2], d[2] * s[2], d[3] * s[2],
        d[1] * s[3], d[2] * s[3], d[3] * s[3]
    )

    return (g, G_x, G_θ)
end

@inline constraint_with_jacobians(F::AbstractFundMat, c::Pair{<:StaticVector{2}, <:StaticVector{2}}) =
    constraint_with_jacobians(F, c.first, c.second)

# =============================================================================
# The TANGENT form: E on unit bearings, with backprojection Jacobians
# =============================================================================
#
# The plane methods above lift a 2-vector to (x, y, 1) and differentiate in the
# z = 1 chart. That is correct for `FundMat`, which is a pixel-plane model by
# definition, and wrong for `EssentialMat` on any camera whose rays leave the
# forward hemisphere — the chart is degenerate at 90° and cannot represent them
# at all. The bearing form below has no such limit.

"""
    constraint_with_jacobians(E::EssentialMat, f₁, f₂; M₁, M₂) -> (g, G_x, G_θ)

The epipolar constraint on UNIT BEARINGS and its Jacobians:
`g = f₂ᵀ E f₁`, with

- `G_x = (f₂ᵀE·M₁, f₁ᵀEᵀ·M₂)` — 1×4, in PIXEL coordinates, because `M` is the
  backprojection Jacobian `∂f/∂u` from [`backproject_with_jacobian`](@ref);
- `G_θ = vec(f₂ f₁ᵀ)` — ∂g/∂vec(E), column-major, unchanged from the plane
  form and feeding the same covariance machinery.

`M` is TANGENT (`fᵀM = 0`), so no explicit projection onto the tangent plane is
needed here or downstream — that is the property `backproject_with_jacobian`
is built to guarantee.

Why the Jacobians are arguments rather than recomputed: they are per-point
data, so they ride as attribute COLUMNS on the correspondence set
(`withattrs(cs; M₁, M₂)`), per the correspondence standard. Recomputing them
per residual would repeat a camera backprojection in the inner loop.

The plane sibling `constraint_with_jacobians(F, u₁, u₂)` is the same question
in the z = 1 chart. Use that one for `FundMat` and this one for
`EssentialMat`: F is a pixel-plane model, E is not.
"""
@inline function constraint_with_jacobians(
        E::EssentialMat{T}, f₁::StaticVector{3}, f₂::StaticVector{3};
        M₁::StaticMatrix{3, 2}, M₂::StaticMatrix{3, 2}
    ) where {T}
    Em = SMatrix{3, 3, T, 9}(E)
    s = SVector{3, T}(f₁[1], f₁[2], f₁[3])
    d = SVector{3, T}(f₂[1], f₂[2], f₂[3])
    Es = Em * s        # the epipolar plane normal seen in view 2
    Etd = Em' * d      # and in view 1
    g = dot(d, Es)
    r1 = Etd' * M₁     # 1×2, pixels in view 1
    r2 = Es' * M₂      # 1×2, pixels in view 2
    G_x = SVector{4, T}(r1[1], r1[2], r2[1], r2[2])
    G_θ = SVector{9, T}(
        d[1] * s[1], d[2] * s[1], d[3] * s[1],
        d[1] * s[2], d[2] * s[2], d[3] * s[2],
        d[1] * s[3], d[2] * s[3], d[3] * s[3]
    )
    return (g, G_x, G_θ)
end

"""
    sampson_distance(E::EssentialMat, f₁, f₂; M₁, M₂, Σ = nothing) -> T

The WHITENED TANGENT Sampson distance of a bearing correspondence:
`g / √(G_x Σ G_xᵀ)` with `g`, `G_x` from
[`constraint_with_jacobians`](@ref). Because `G_x` is expressed in pixels, the
result is in PIXELS and the threshold stays a pixel σ — the unit the detector
actually guarantees — for any lens, including one whose rays the z = 1 chart
cannot represent.

`Σ` is the 4×4 measurement covariance of `(u₁, u₂)` in pixels, SUPPLIED BY THE
CALLER and defaulting to the identity (unit isotropic pixel noise). It is not
baked in on purpose: whitening by raw pixels and whitening through a pool's
canonical camera differ at FIRST order on any pair whose two images differ in
size, which is common, so the choice belongs to whoever knows what frame the
threshold is quoted in.

Degenerate (the gradient vanishes to rounding) follows the plane form's rule:
`0` when the constraint vanishes with it, `Inf` otherwise.

This is PoseLib's `FixCameraRelativePoseRefiner` residual with `Σ = I`, and to
first order it agrees with the plane Sampson of the same correspondence on a
pinhole; they differ at second order, where only this one is defined past 90°.
"""
@inline sampson_distance(
    E::EssentialMat, f₁::StaticVector{3}, f₂::StaticVector{3};
    M₁::StaticMatrix{3, 2}, M₂::StaticMatrix{3, 2}, Σ = nothing
) = abs(signed_sampson_distance(E, f₁, f₂; M₁, M₂, Σ))

"""
    signed_sampson_distance(E::EssentialMat, f₁, f₂; M₁, M₂, Σ = nothing) -> T

The bearing (TANGENT) Sampson distance of [`sampson_distance`](@ref)`(E, f₁, f₂;
M₁, M₂, Σ)`, signed: `g / √(G_x Σ G_xᵀ)`. Degenerate is the unsigned rule with
`g`'s sign (`0`, or `±Inf`). The one body: the unsigned form is its `abs`.
"""
@inline function signed_sampson_distance(
        E::EssentialMat, f₁::StaticVector{3}, f₂::StaticVector{3};
        M₁::StaticMatrix{3, 2}, M₂::StaticMatrix{3, 2}, Σ = nothing
    )
    # `whiten`'s residual at the rows' rounding scales (`_tangent_rounding`), so
    # the scorer and the LO agree on which pairs are degenerate whatever E's
    # scale or eltype; a singular Σ along G_x is degenerate too
    g, G_x, _ = constraint_with_jacobians(E, f₁, f₂; M₁, M₂)
    ρ = constraint_rounding_scale(E, f₁, f₂; M₁, M₂)
    return _whiten_residual(g, G_x, Σ === nothing ? I : Σ, ρ)
end

# The bearing form's rounding scales (`constraint_rounding_scale`): the
# `_tangent_rounding` the residual and Jacobian rows read.
@inline function constraint_rounding_scale(
        E::EssentialMat, f₁::StaticVector{3}, f₂::StaticVector{3};
        M₁::StaticMatrix{3, 2}, M₂::StaticMatrix{3, 2}
    )
    gradient, value = _tangent_rounding(E, f₁, f₂, M₁, M₂)
    return (; gradient, value)
end

# The per-correspondence parts every signed-Sampson form shares, in ONE
# arithmetic order: the homogeneous points, the two epipolar lines, the
# constraint value g = x₂ᵀ·M·x₁, the squared point gradient
# ‖∂ₓg‖² = (M·x₁)₁² + (M·x₁)₂² + (Mᵀ·x₂)₁² + (Mᵀ·x₂)₂², and the verdicts of the
# one degeneracy rule. Eltype-generic (Float32, ForwardDiff Duals) and
# device-safe: the distance, the LM rows and their kernels all read these.
# The points are 2-vectors in the model's measurement frame: typed, so a 3-vector
# bearing is refused (MethodError) rather than read as its first two components.
@inline function _epipolar_parts(M, u::StaticVector{2}, v::StaticVector{2})
    T = float(promote_type(eltype(M), eltype(u), eltype(v)))
    x1 = _lift_image_point(u, T)
    x2 = _lift_image_point(v, T)
    Fx1 = M * x1
    Fᵀx2 = M' * x2
    g = dot(x2, Fx1)
    J² = Fx1[1]^2 + Fx1[2]^2 + Fᵀx2[1]^2 + Fᵀx2[2]^2
    tol_J, tol_g = epipolar_rounding(M, x1, x2)
    return (; x1, x2, Fx1, Fᵀx2, g, J², degenerate = J² <= tol_J^2, vanishes = abs(g) <= tol_g)
end

"""
    signed_sampson_jacobian(g, G_x, G_θ, ∂G_x, degenerate = iszero(G_x)) -> SVector

The derivative along model coordinates `θ` of the signed Sampson distance
`g/‖G_x‖` of a codimension-1 constraint: `g` its value, `G_x` its measurement
gradient, `G_θ = ∂g/∂θ` and `∂G_x = ∂G_x/∂θ` (`length(G_x) × length(G_θ)`),

    ∂(g/‖G_x‖)/∂θ_q = (G_θ[q] − (g/‖G_x‖)·(G_xᵀ·∂G_x[:, q])/‖G_x‖) / ‖G_x‖.

`degenerate` (the one Sampson degeneracy rule, e.g. [`epipolar_rounding`](@ref)'s
verdict) gives a ZERO row: a least-squares row cannot carry the rule's `±Inf`,
and a zero row is what the reweighting that follows gives it. Generic in the
element type and device-safe; [`sampson_jacobian_rows!`](@ref) runs it per
correspondence for the epipolar models.
"""
@inline function signed_sampson_jacobian(
        g, G_x::StaticVector, G_θ::StaticVector{N}, ∂G_x::StaticMatrix,
        degenerate::Bool = iszero(sum(abs2, G_x))
    ) where {N}
    J² = sum(abs2, G_x)
    denom = sqrt(J²)
    samp = g / denom
    row = SVector(ntuple(q -> (G_θ[q] - samp * (dot(G_x, ∂G_x[:, q]) / denom)) / denom, Val(N)))
    return degenerate ? zero(row) : row
end

"""
    signed_sampson_row(M, dM, u, v) -> (d, ∂d::SVector)

One correspondence's signed Sampson distance under the epipolar model `M`
(fundamental or essential, in the measurement frame of `u`, `v`) and its
derivative along the chart directions `dM = (∂M/∂θ₁, …)` — e.g. the columns of
`retract_jacobian(ch, v)` reshaped to 3×3. A degenerate pair (the one
rule) is `(0, 0)`: LM-safe. Device-safe; the per-row function of
[`sampson_rows!`](@ref) and [`sampson_jacobian_rows!`](@ref).
"""
@inline signed_sampson_row(M, dM::Tuple, u, v) = _sampson_row(M, _direction_matrix(dM), u, v)

# The chart directions `dM = (∂M/∂θ₁, …, ∂M/∂θ_N)` as the ROWS of an N×9 matrix
# (row q = vec(∂M/∂θ_q), column-major): every direction's bilinear form
# `xᵀ·(∂M/∂θ_q)·y` is then row q of `B·vec(x yᵀ)`, so one N×9 product replaces
# N separate 3×3 contractions per correspondence. Loop-invariant over a sweep.
@inline _direction_matrix(dM::NTuple{N, Any}) where {N} =
    SMatrix{N, 9}(ntuple(k -> dM[(k - 1) % N + 1][(k - 1) ÷ N + 1], Val(9 * N)))

# The signed Sampson row of `signed_sampson_jacobian`, CONTRACTED. Every entry of
# that formula is a bilinear form in the direction ∂M/∂θ_q:
#   G_θ[q]              = x₂ᵀ·D_q·x₁
#   G_xᵀ·∂G_x[:, q]      = aᵀ·D_q·x₁ + x₂ᵀ·D_q·b,   a = (Mx₁)₁,₂ and b = (Mᵀx₂)₁,₂ with a 0 third entry,
# so the whole row is `B·vec(W) / ‖G_x‖` with ONE 3×3 matrix
#   W = x₂x₁ᵀ − (g/‖G_x‖²)·(a x₁ᵀ + x₂ bᵀ)
# — the same algebra as `signed_sampson_jacobian(g, G_x, G_θ, ∂G_x)` (equal to
# rounding; pinned in test/sampson_rows_tests.jl), at a third of the
# arithmetic. Degenerate (the one rule) is `(0, 0)`. `ifelse`, not a branch:
# a branch in the row stops the CPU loop from vectorizing well (measured 2× on
# the residual row, 2026-10-07).
@inline _sampson_row(M, B::SMatrix{N, 9}, u, v) where {N} = _scaled_sampson_row(_sampson_terms(M, B, u, v)...)
@inline function _sampson_terms(M, B::SMatrix{N, 9}, u, v) where {N}
    p = _epipolar_parts(M, u, v)
    z = zero(p.g)
    a = SVector(p.Fx1[1], p.Fx1[2], z)
    b = SVector(p.Fᵀx2[1], p.Fᵀx2[2], z)
    return _contracted_sampson_terms(B, p.g, p.J², p.degenerate, p.x1, p.x2, a, b)
end

# The shared tail of the plane and tangent rows, as (g, B·vec(W), 1/‖G_x‖) with
# W = x₂x₁ᵀ − (g/J²)(a x₁ᵀ + x₂ bᵀ): the distance is g/‖G_x‖ and the chart row
# B·vec(W)/‖G_x‖, the scale 0 when degenerate. The scale is returned apart so
# a Jacobian row applies it with its √w in one scalar product.
@inline function _contracted_sampson_terms(B::SMatrix{N, 9}, g, J², degenerate::Bool, x₁, x₂, a, b) where {N}
    c = ifelse(degenerate, zero(g), g / J²)       # finite when degenerate (J² may be 0): the 0 scale then zeroes the row
    W = x₂ * (x₁ - c * b)' - (c * a) * x₁'      # x₂x₁ᵀ − c(a x₁ᵀ + x₂ bᵀ), two outer products
    return g, B * vec(W), ifelse(degenerate, zero(J²), inv(sqrt(J²)))
end
@inline function _scaled_sampson_row(g, row, scale)
    return ifelse(iszero(scale), zero(g), g * scale), scale * row
end

# One row each, in the uniform `(out, i, args...)` form `_foreach_row!` runs
# (a plain loop on the CPU, one kernel on a device). The Jacobian rows take the
# packed direction matrix `B` (`_direction_matrix`, formed once per sweep).
Base.@propagate_inbounds function _sampson_residual_row!(r, i, M, u₁, u₂, sqrtw)
    p = _epipolar_parts(M, u₁[i], u₂[i])
    r[i] = ifelse(p.degenerate, zero(eltype(r)), eltype(r)(sqrtw[i] * (p.g / sqrt(p.J²))))
    return nothing
end
Base.@propagate_inbounds function _sampson_jacobian_row!(J, i, M, B, u₁, u₂, sqrtw)
    _, row, scale = _sampson_terms(M, B, u₁[i], u₂[i])
    w = sqrtw[i] * scale
    for q in 1:size(B, 1)
        J[i, q] = w * row[q]
    end
    return nothing
end

"""
    sampson_rows!(r, M, u₁, u₂, sqrtw) -> r

The √w-weighted signed Sampson residual rows of the epipolar model `M` over the
correspondences `u₁[i] => u₂[i]`: `r[i] = sqrtw[i]·g/‖∂ₓg‖`, a degenerate pair a
zero row. ONE KernelAbstractions kernel on the backend that owns `r` (supply device
arrays for the GPU), a plain loop over the same row function on the CPU; `M` in
the measurement frame of the points (for framed data `C₂ᵀ·M·C₁`, formed once
per model).
"""
function sampson_rows!(
        r::AbstractVector, M,
        u₁::AbstractVector{<:StaticVector{2}}, u₂::AbstractVector{<:StaticVector{2}},
        sqrtw::AbstractVector
    )
    return _foreach_row!(_sampson_residual_row!, r, length(u₁), M, u₁, u₂, sqrtw)
end

"""
    sampson_jacobian_rows!(J, M, dM, u₁, u₂, sqrtw) -> J

The √w-weighted Jacobian rows of [`sampson_rows!`](@ref) along the chart
directions `dM = (∂M/∂θ₁, …, ∂M/∂θ_N)`: `J[i, q] = sqrtw[i]·∂dᵢ/∂θ_q`
([`signed_sampson_jacobian`](@ref)), a degenerate pair a zero row. One kernel on
the backend that owns `J`.
"""
function sampson_jacobian_rows!(
        J::AbstractMatrix, M, dM::Tuple,
        u₁::AbstractVector{<:StaticVector{2}}, u₂::AbstractVector{<:StaticVector{2}},
        sqrtw::AbstractVector
    )
    return _foreach_row!(_sampson_jacobian_row!, J, length(u₁), M, _direction_matrix(dM), u₁, u₂, sqrtw)
end

# =============================================================================
# The TANGENT LM rows: E on unit bearings
# =============================================================================
#
# The rows above are the IMAGE-PLANE form: `_epipolar_parts` reads `u[1], u[2]`
# and lifts to (x, y, 1), so a 3-vector bearing handed to them has its third
# component DISCARDED and replaced by 1 — a different function, not a scaled
# one. VisualGeometryToolkitEstimation measured the two disagreeing by up to 446× per
# point after its scoring path moved to bearings while `_EssentialLm` did not:
# the optimizer descended something unrelated to what was scored, failed to
# reduce its own cost (0.006378 → 0.006416), and reported `:converged` while a
# pose went from a 3.4° seed to 26°. It would have shipped as "the LO
# converged".
#
# Hence BOTH halves of the dispatch are typed: the image-plane rows take
# `StaticVector{2}` only, these take `StaticVector{3}` plus the per-point
# Jacobians, and neither can be reached by the other's data.
#
# ONE IMPLEMENTATION, which is the whole point. The constraint and its
# gradients come from `constraint_with_jacobians(E, f₁, f₂; M₁, M₂)` — the same
# scalar the scoring path uses — and the WHITENING DERIVATIVE comes from
# `_contracted_sampson_terms`, the same function the image-plane rows use (the
# contracted form of `signed_sampson_jacobian`). Nothing about `g/‖∇g‖` is
# re-derived here; only the vectors G_x pairs with, which differ because G_x
# carries the backprojection Jacobians.

# The degeneracy rule, in the family's own arithmetic. `epipolar_rounding` sizes
# the bearing-space gradient; G_x is that gradient carried into PIXELS by M, so
# its rounding scale carries the same factor — measured in the same 1-norm style
# the rule itself uses, rather than with a new constant.
@inline function _tangent_rounding(E, f₁, f₂, M₁, M₂)
    tol_J, tol_g = epipolar_rounding(E, f₁, f₂)
    s = max(sum(abs, M₁), sum(abs, M₂))
    return tol_J * s, tol_g
end

# The per-correspondence parts every tangent-Sampson form shares — the bearing
# counterpart of `_epipolar_parts`: the constraint `g = f₂ᵀ E f₁`, its pixel
# gradient `G_x` and model gradient `G_θ` (`constraint_with_jacobians`), the
# squared gradient `J² = ‖G_x‖²`, and the verdicts of the one degeneracy rule at
# the `_tangent_rounding` scales. The scalar distance, the residual rows and the
# Jacobian rows all read these.
@inline function _tangent_parts(E, f₁, f₂, M₁, M₂)
    g, G_x, G_θ = constraint_with_jacobians(E, f₁, f₂; M₁, M₂)
    J² = sum(abs2, G_x)
    tol_J, tol_g = _tangent_rounding(E, f₁, f₂, M₁, M₂)
    return (; g, G_x, G_θ, J², degenerate = J² <= tol_J^2, vanishes = abs(g) <= tol_g)
end

"""
    signed_tangent_sampson_row(E, dE, f₁, f₂, M₁, M₂) -> (d, ∂d::SVector)

One bearing correspondence's signed TANGENT Sampson distance under `E` and its
derivative along the chart directions `dE = (∂E/∂θ₁, …)`. The bearing
counterpart of [`signed_sampson_row`](@ref); `M₁`/`M₂` are the 3×2
backprojection Jacobians from [`backproject_with_jacobian`](@ref), which is
what puts `d` and the rows in PIXELS.

Degenerate (the gradient vanishes to rounding) is `(0, 0)`, LM-safe, the same
rule the image-plane row uses. Device-safe; the per-row function of the
bearing methods of [`sampson_rows!`](@ref) and
[`sampson_jacobian_rows!`](@ref).
"""
@inline signed_tangent_sampson_row(E::EssentialMat, dE::Tuple, f₁, f₂, M₁, M₂) =
    _tangent_sampson_row(E, _direction_matrix(dE), f₁, f₂, M₁, M₂)

# The bearing row through the SAME contraction as the plane row
# (`_contracted_sampson_terms`): G_x carries the backprojection Jacobians, so
# G_xᵀ·∂G_x[:, q] = f₂ᵀ·D_q·(M₁ G_x₁,₂) + (M₂ G_x₃,₄)ᵀ·D_q·f₁ — `a` and `b` are the
# two pixel gradients carried back to bearings. M is per-point DATA and does not
# depend on θ, so it goes through the derivative unchanged.
@inline _tangent_sampson_row(E::EssentialMat, B::SMatrix{N, 9}, f₁, f₂, M₁, M₂) where {N} =
    _scaled_sampson_row(_tangent_sampson_terms(E, B, f₁, f₂, M₁, M₂)...)
@inline function _tangent_sampson_terms(E::EssentialMat, B::SMatrix{N, 9}, f₁, f₂, M₁, M₂) where {N}
    p = _tangent_parts(E, f₁, f₂, M₁, M₂)
    s = SVector{3}(f₁[1], f₁[2], f₁[3])
    d = SVector{3}(f₂[1], f₂[2], f₂[3])
    a = M₁ * SVector(p.G_x[1], p.G_x[2])
    b = M₂ * SVector(p.G_x[3], p.G_x[4])
    # x₁ = f₁, x₂ = f₂; the view-1 gradient pairs with f₂ on the left, so it is the
    # `b`-slot of the plane form and the view-2 one its `a`-slot.
    return _contracted_sampson_terms(B, p.g, p.J², p.degenerate, s, d, b, a)
end

# One row each, in the uniform `(out, i, args...)` form `_foreach_row!` runs
# (a plain loop on the CPU, one kernel on a device).
Base.@propagate_inbounds function _tangent_residual_row!(r, i, E, f₁, f₂, M₁, M₂, sqrtw)
    p = _tangent_parts(E, f₁[i], f₂[i], M₁[i], M₂[i])
    r[i] = ifelse(p.degenerate, zero(eltype(r)), eltype(r)(sqrtw[i] * (p.g / sqrt(p.J²))))
    return nothing
end

Base.@propagate_inbounds function _tangent_jacobian_row!(J, i, E, B, f₁, f₂, M₁, M₂, sqrtw)
    _, row, scale = _tangent_sampson_terms(E, B, f₁[i], f₂[i], M₁[i], M₂[i])
    w = sqrtw[i] * scale
    for q in 1:size(B, 1)
        J[i, q] = w * row[q]
    end
    return nothing
end

"""
    sampson_rows!(r, E::EssentialMat, f₁, f₂, sqrtw; M₁, M₂) -> r

The √w-weighted signed TANGENT Sampson residual rows of `E` over the BEARING
correspondences `f₁[i] => f₂[i]`, with the per-point backprojection Jacobians
`M₁[i]`, `M₂[i]` (3×2, from [`backproject_with_jacobian`](@ref)) carrying the
residual into PIXELS — so an LM threshold stays a pixel σ for any lens,
including one whose rays the z = 1 chart cannot represent.

Positionally identical to the image-plane method so an LM protocol stays
uniform; the Jacobians ride as keywords exactly as they do on the scalar
[`sampson_distance`](@ref)`(E, f₁, f₂; M₁, M₂)`, with which these rows agree
(pinned in `test/tangent_sampson_rows_tests.jl`).

A 2-vector endpoint cannot reach this method and a bearing cannot reach the
image-plane one — see the section note above for the 446× disagreement that
made the separation a type rather than a convention.
"""
function sampson_rows!(
        r::AbstractVector, E::EssentialMat,
        f₁::AbstractVector{<:StaticVector{3}}, f₂::AbstractVector{<:StaticVector{3}},
        sqrtw::AbstractVector; M₁::AbstractVector, M₂::AbstractVector
    )
    return _foreach_row!(_tangent_residual_row!, r, length(f₁), E, f₁, f₂, M₁, M₂, sqrtw)
end

"""
    sampson_jacobian_rows!(J, E::EssentialMat, dE, f₁, f₂, sqrtw; M₁, M₂) -> J

The √w-weighted Jacobian rows of the bearing [`sampson_rows!`](@ref) along the
chart directions `dE = (∂E/∂θ₁, …, ∂E/∂θ_N)`: `J[i, q] = sqrtw[i]·∂dᵢ/∂θ_q`, a
degenerate pair a zero row.

The whitening derivative is the contracted [`signed_sampson_jacobian`](@ref)
(`_contracted_sampson_terms`) — the SAME function the image-plane rows use — so
`g/‖∇g‖` has one implementation here and there. What this adds is the pair of
vectors `G_x` contracts with, which differ only because `G_x` carries the
backprojection Jacobians; `M` is per-point data and does not depend on `θ`.
Finite-differenced against the residual rows in the tests, with two different
focal lengths, rather than derived and asserted against itself.
"""
function sampson_jacobian_rows!(
        J::AbstractMatrix, E::EssentialMat, dE::Tuple,
        f₁::AbstractVector{<:StaticVector{3}}, f₂::AbstractVector{<:StaticVector{3}},
        sqrtw::AbstractVector; M₁::AbstractVector, M₂::AbstractVector
    )
    return _foreach_row!(_tangent_jacobian_row!, J, length(f₁), E, _direction_matrix(dE), f₁, f₂, M₁, M₂, sqrtw)
end

# =============================================================================
# Public: Rank-2 Enforcement
# =============================================================================

"""
    enforce_rank_two(F::StaticMatrix{3,3}) -> typeof(F)

Enforce the rank-2 constraint on a fundamental matrix by zeroing the
smallest singular value via SVD.

A valid fundamental matrix must have rank 2 (det(F) = 0). This function
projects an arbitrary 3×3 matrix onto the closest rank-2 matrix in
Frobenius norm. Preserves the input type (FundMat, SMatrix, etc.).
"""
function enforce_rank_two(F::StaticMatrix{3, 3, T}) where {T}
    S = svd(F)   # works directly on any StaticMatrix{3,3} (FundMat, SMatrix, etc.)
    Σ = SMatrix{3, 3, T, 9}(
        S.S[1], zero(T), zero(T),
        zero(T), S.S[2], zero(T),
        zero(T), zero(T), zero(T)
    )
    result = SMatrix{3, 3, T, 9}(S.U * Σ * S.Vt)
    return similar_type(typeof(F), T)(Tuple(result))
end

# =============================================================================
# Public: Typed forward / reverse constructors — EssentialMat ↔ FundMat
# =============================================================================
#
# These canonical typed constructors replace the hand-rolled
# `skew(t) * R` / `inv(K2)' * E * inv(K1)` / `K₂ᵀ F K₁` snippets that
# historically proliferated across bench code, and replace the older
# `essential_from_fundamental` helper.  Centralizing them here eliminates a
# class of silent algebraic-typo bugs and gives downstream callers a
# strongly-typed API: `EssentialMat`/`FundMat` *types* signal the
# calibrated/uncalibrated distinction at every dispatch site, both as
# inputs and outputs.

"""
    EssentialMat(R::AbstractMatrix, t::AbstractVector) -> EssentialMat
    EssentialMat(X::EuclideanMap{3})                   -> EssentialMat

Construct the calibrated epipolar matrix `E = [t]× R` from the relative
rigid transform `(R, t)` in the EXTRINSIC direction, `x₂ = R·x₁ + t`
(monogram `X_{V₂V₁}` — view 1 coordinates in, view 2 coordinates out).
Returns an `EssentialMat{T}` (typed wrapper) so that downstream dispatch
can distinguish E from F by type, not by comment.

The `EuclideanMap` form reads `.R`/`.t` directly — labeled maps are
welcome (e.g. a `CameraExtrinsics` or a rig's
`ViewFrame{id₁} → ViewFrame{id₂}` transform); the frame tags are irrelevant
to the algebra here and are simply dropped. This is the constructor behind
[`essential_form`](@ref)`(rig)`, which additionally labels the resulting
form's slots with the rig's [`GnomonicFrame`](@ref) tags.

`t` is treated as a direction (no normalization here — caller can scale
for sign or rescale to unit norm if desired).
"""
function EssentialMat(R::AbstractMatrix, t::AbstractVector)
    @assert size(R) == (3, 3) "EssentialMat: R must be 3×3"
    @assert length(t) == 3    "EssentialMat: t must have length 3"
    T = promote_type(eltype(R), eltype(t))
    R_s = SMatrix{3, 3, T, 9}(R)
    t_s = SVector{3, T}(t)
    return EssentialMat{T}(Tuple(skew(t_s) * R_s))
end

# Convenience: pull R and t out of an EuclideanMap (or duck-typed pose)
# and forward.
# Pose-like: an EuclideanMap, an (R, t) tuple, or a NamedTuple with R and t.
# (Typed, not `Any`: an untyped method is ambiguous with StaticArrays'
# generic constructors for the wrapper type.)
EssentialMat(pose::Union{EuclideanMap, Tuple{AbstractMatrix, AbstractVector}, NamedTuple}) =
    EssentialMat(_em_R(pose), _em_t(pose))
# A physical pose strips in metres at entry: E = [t]×R is a plain matrix (its
# scale is a gauge), equal to the plain pose's in metres.
EssentialMat(pose::UnitfulEuclideanMap{3}) = EssentialMat(ustrip(m, pose))

# Internal pose-decomposition shims — kept loose-typed so they accept
# both `EuclideanMap` and any duck-typed `(R, t)` tuple.
# NOTE (0.10 fix): these read `.R`/`.t` — EuclideanMap's actual fields.
# The 0.5-era originals read `.linear`/`.translation` (CT AffineMap's field
# names), which EuclideanMap never had, so the EuclideanMap path always
# threw; no caller had exercised it (regression-pinned in
# epipolar_forms_tests.jl).
@inline _em_R(p::Tuple) = p[1]
@inline _em_t(p::Tuple) = p[2]
@inline _em_R(p) = p.R   # EuclideanMap's rotation field
@inline _em_t(p) = p.t

"""
    EssentialMat(F::FundMat, K1::AbstractMatrix, K2::AbstractMatrix) -> EssentialMat
    EssentialMat(F::FundMat, K::AbstractMatrix)                      -> EssentialMat

Construct the calibrated epipolar matrix `E = K₂ᵀ F K₁` from a
fundamental matrix and per-view intrinsics.  Inverse of
`FundMat(E, K1, K2)`.  Replaces the older `essential_from_fundamental`
helper which returned a raw `SMatrix` and hence lost the F→E type
distinction at the call site.

This is the frame-free algebra; the checked, frame-labeled counterpart is
composition on an [`EpipolarMap`](@ref) — the pixel → gnomonic frame
change per argument slot, with the slot labels verifying that each `K` met
the view it was meant for (see [`GnomonicFrame`](@ref)).

`K1`, `K2` may be supplied as raw `SMatrix{3,3}`, `CameraCalibrationMat`
(any `T<:Real`), or any `AbstractMatrix` of size 3×3.
"""
function EssentialMat(F::FundMat, K1::AbstractMatrix, K2::AbstractMatrix)
    @assert size(K1) == (3, 3) "EssentialMat: K1 must be 3×3"
    @assert size(K2) == (3, 3) "EssentialMat: K2 must be 3×3"
    T = promote_type(eltype(F), eltype(K1), eltype(K2))
    F_s = SMatrix{3, 3, T, 9}(F)
    K1_s = SMatrix{3, 3, T, 9}(K1)
    K2_s = SMatrix{3, 3, T, 9}(K2)
    return EssentialMat{T}(Tuple(transpose(K2_s) * F_s * K1_s))
end

EssentialMat(F::FundMat, K::AbstractMatrix) = EssentialMat(F, K, K)

# Loose-typed F entry points — accept any 3×3 matrix (e.g. raw `SMatrix`
# from a solver that returns un-wrapped F).  Wraps into `FundMat`
# internally so the formula lives in one place.
EssentialMat(F::AbstractMatrix, K1::AbstractMatrix, K2::AbstractMatrix) =
    EssentialMat(
    FundMat{eltype(F)}(Tuple(SMatrix{3, 3, eltype(F), 9}(F))),
    K1, K2
)
EssentialMat(F::AbstractMatrix, K::AbstractMatrix) = EssentialMat(F, K, K)

"""
    FundMat(E::EssentialMat, K1::AbstractMatrix, K2::AbstractMatrix) -> FundMat
    FundMat(E::EssentialMat, K::AbstractMatrix)                      -> FundMat

Construct the (uncalibrated) epipolar matrix `F = K₂⁻ᵀ E K₁⁻¹` from
an essential matrix and per-view intrinsic matrices.  The shared-K
overload assumes `K₁ = K₂ = K` (e.g. PhotoTourism scenes captured by
one camera).

Inverse of `EssentialMat(F, K1, K2)`.  Strongly-typed in both input
(`EssentialMat`) and output (`FundMat`) so the calibrated /
uncalibrated distinction is enforced at compile time.

This is the frame-free algebra of the gnomonic → pixel frame change; the
checked, slot-labeled form of the same law is composition (`∘`) of an
[`EpipolarMap`](@ref) with the `HomographyMap{GnomonicFrame{id}, ImageFrame{id}}(K)` lifts, as minted by
`EpipolarForm(rig)` on a `StereoRig` (see [`essential_form`](@ref)).

`K1`, `K2` may be supplied as raw `SMatrix{3,3}`, `CameraCalibrationMat`
(any `T<:Real`), or any `AbstractMatrix` of size 3×3.
"""
function FundMat(E::EssentialMat, K1::AbstractMatrix, K2::AbstractMatrix)
    @assert size(K1) == (3, 3) "FundMat: K1 must be 3×3"
    @assert size(K2) == (3, 3) "FundMat: K2 must be 3×3"
    T = promote_type(eltype(E), eltype(K1), eltype(K2))
    E_s = SMatrix{3, 3, T, 9}(E)
    K1_s = SMatrix{3, 3, T, 9}(K1)
    K2_s = SMatrix{3, 3, T, 9}(K2)
    return FundMat{T}(Tuple(transpose(inv(K2_s)) * E_s * inv(K1_s)))
end

FundMat(E::EssentialMat, K::AbstractMatrix) = FundMat(E, K, K)

# =============================================================================
# Public: Essential Matrix Decomposition
# =============================================================================

# -----------------------------------------------------------------------------
# Task-local zero-alloc 3×3 SVD for motion(E)
#
# `svd(::SMatrix{3,3,Float64})` does NOT lower to the generic
# `svd(Matrix(A))`/gesdd path: StaticArrays (≥ v1.7, src/blas.jl) special-cases
# Float64/Float32 SMatrix and calls **dgesvd_** (QR iteration) directly with
# jobu = jobvt = 'S', lda = ldu = M, ldvt = K, and a FIXED
# `lwork = max(3K + max(M,N), 5K)` — no workspace query. For 3×3 that is
# lwork = 15. Its per-call MMatrix/MVector temporaries and the `Ref` escape
# into the ccall and are the allocation cost we remove here.
#
# `_svd33!` issues the IDENTICAL dgesvd call (same jobu/jobvt, dims, leading
# dimensions, and the same fixed lwork = 15) into cached buffers, so outputs
# are bit-identical to `svd(::SMatrix{3,3,Float64})` — same LAPACK routine,
# same arguments; buffer reuse cannot change the computed bits because dgesvd
# fully overwrites U/S/Vt and `work` is pure scratch (verified empirically:
# poisoned buffers give identical bits). Do NOT change the routine (e.g. to
# gesdd) or lwork: gesdd differs from gesvd by ~1 ulp in U/Vt.
#
# The workspace is cached in `task_local_storage` — one per task, never shared,
# safe under task migration across threads (unlike `threadid()` indexing).
# -----------------------------------------------------------------------------
struct MotionSvdWorkspace
    A::Matrix{Float64}     # dgesvd destroys its input; scratch copy of E
    U::Matrix{Float64}     # 3×3 left singular vectors
    S::Vector{Float64}     # 3 singular values (descending)
    Vt::Matrix{Float64}    # 3×3 right singular vectors (transposed)
    work::Vector{Float64}  # lwork = max(3K + max(M,N), 5K) = 15 for 3×3
end
MotionSvdWorkspace() = MotionSvdWorkspace(
    Matrix{Float64}(undef, 3, 3), Matrix{Float64}(undef, 3, 3),
    Vector{Float64}(undef, 3), Matrix{Float64}(undef, 3, 3),
    Vector{Float64}(undef, 15)
)

@inline _motion_svd_workspace() =
    task_buffers(MotionSvdWorkspace, :vgc_motion_svd_workspace)::MotionSvdWorkspace

# dgesvd_('S', 'S', 3, 3, …, lwork=15) into cached buffers; returns (U, Vt)
# as SMatrices, bit-identical to `(F.U, F.Vt)` of `svd(Es)`.
function _svd33!(ws::MotionSvdWorkspace, Es::SMatrix{3, 3, Float64, 9})
    A = ws.A
    @inbounds for i in 1:9
        A[i] = Es[i]
    end
    info = Ref{LinearAlgebra.BlasInt}(1)
    ccall(
        (LinearAlgebra.BLAS.@blasfunc(dgesvd_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
        (
            Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt},
            Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ptr{Float64},
            Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BlasInt},
            Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Clong, Clong,
        ),
        'S', 'S', 3, 3, A, 3, ws.S, ws.U, 3,
        ws.Vt, 3, ws.work, 15, info, 1, 1
    )
    LinearAlgebra.LAPACK.chklapackerror(info[])  # same error path as StaticArrays' _svd
    U = SMatrix{3, 3, Float64, 9}(ntuple(i -> @inbounds(ws.U[i]), Val(9)))
    Vt = SMatrix{3, 3, Float64, 9}(ntuple(i -> @inbounds(ws.Vt[i]), Val(9)))
    return U, Vt
end

"""
    motion(E::AbstractMatrix) -> NTuple{4, Tuple{SMatrix{3,3}, Vec3{Float64}}}

Extract candidate `(R, t)` motion tuples from an essential matrix `E`.

Returns all 4 combinations of the two possible rotations and two possible
translation directions. Only one is physically valid (positive depth for
both cameras), but for pose error evaluation we take the minimum over all 4.

Parallels `motion(::HomographyMat)` (Malis-Vargas decomposition for
homographies — same shape, multi-candidate).
"""
function motion(E::AbstractMatrix)
    FU, FVt = _svd33!(_motion_svd_workspace(), SMatrix{3, 3, Float64}(E))
    W = SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1)

    # Ensure proper rotation (det = +1)
    U = FU * sign(det(FU))
    Vt = FVt * sign(det(FVt))

    R1 = U * W * Vt
    R2 = U * W' * Vt
    t = Vec3{Float64}(U[:, 3])  # essential-matrix translation = free vector

    return ((R1, t), (R1, -t), (R2, t), (R2, -t))
end

"""
    motion(H::HomographyMat) -> Vector{Tuple{RotMatrix{3,Float64}, Vec3{Float64}, Vec3{Float64}}}

Malis–Vargas decomposition [MalisVargas2007](@citep) of a **calibrated** homography into up to 4 candidate
`(R, t, n)` motions: rotation `R`, translation `t = t_phys / d_phys`, and unit
plane normal `n` (sign fixed so `n₃ ≥ 0`).  Native port of PoseLib's
`motion_from_homography` (single-SVD variant); the candidate order matches
PoseLib exactly: `(R₁,t₁,n₁), (R₁,−t₁,−n₁), (R₂,t₂,n₂), (R₂,−t₂,−n₂)`.

For the degenerate near-equal-outer-singular-value case (planar / pure rotation)
it returns the 1-element list `[(H̄, 0, 0)]` (normalized homography, zero t/n),
as PoseLib does.  Parallels [`motion`](@ref)`(E)` for essential matrices.
"""
motion(H::HomographyMat) = _decompose_homography(SMatrix{3, 3, Float64, 9}(H))

function _decompose_homography(HH::SMatrix{3, 3, Float64, 9})
    RT = Tuple{RotMatrix{3, Float64}, Vec3{Float64}, Vec3{Float64}}
    det(HH) < 0 && (HH = -HH)                 # fix overall sign
    F = svd(HH)
    S = F.S                                  # descending singular values
    H2 = HH / S[2]                            # normalize by the middle σ
    V = SMatrix{3, 3, Float64, 9}(F.V)
    # Degenerate: σ₁ ≈ σ₃ ⇒ normal unconstrained (planar / pure rotation).
    if abs(S[1] - S[3]) < 1.0e-6 * S[1]
        z = Vec3{Float64}(0.0, 0.0, 0.0)
        return RT[(RotMatrix{3, Float64}(H2), z, z)]
    end
    det(V) < 0 && (V = -V)
    s1 = (S[1] * S[1]) / (S[2] * S[2])
    s3 = (S[3] * S[3]) / (S[2] * S[2])
    v1 = V[:, 1]; v2 = V[:, 2]; v3 = V[:, 3]
    a = sqrt(1.0 - s3); b = sqrt(s1 - 1.0); c = sqrt(s1 - s3)
    u1 = (a * v1 + b * v3) / c
    u2 = (a * v1 - b * v3) / c
    R1 = _onb3(H2 * v2, H2 * u1) * _onb3(v2, u1)'
    R2 = _onb3(H2 * v2, H2 * u2) * _onb3(v2, u2)'
    n1 = cross(v2, u1); n1 = n1[3] < 0 ? -n1 : n1; t1 = (H2 - R1) * n1
    n2 = cross(v2, u2); n2 = n2[3] < 0 ? -n2 : n2; t2 = (H2 - R2) * n2
    return RT[
        (RotMatrix{3, Float64}(R1), Vec3{Float64}(t1), Vec3{Float64}(n1)),
        (RotMatrix{3, Float64}(R1), Vec3{Float64}(-t1), Vec3{Float64}(-n1)),
        (RotMatrix{3, Float64}(R2), Vec3{Float64}(t2), Vec3{Float64}(n2)),
        (RotMatrix{3, Float64}(R2), Vec3{Float64}(-t2), Vec3{Float64}(-n2)),
    ]
end

# 3×3 frame [a, b, a×b] from two 3-vectors (columns) — Malis–Vargas ONB blocks.
@inline _onb3(a::AbstractVector, b::AbstractVector) =
    hcat(SVector{3, Float64}(a), SVector{3, Float64}(b), cross(a, b))

# =============================================================================
# Internal: Rank-2 Constraint via Cubic (for 7-point solver)
# =============================================================================

function _rank_two_constraint(f₁::SVector{9, T}, f₂::SVector{9, T}, ::Type{T}) where {T}
    # Compute cubic coefficients for det(λ f₁ + f₂) = 0
    # F is stored column-major: indices [1,2,3] = col1, [4,5,6] = col2, [7,8,9] = col3

    # 2×2 minors of f₁
    t3 = f₁[5] * f₁[9] - f₁[6] * f₁[8]
    t4 = f₁[4] * f₁[9] - f₁[6] * f₁[7]
    t5 = f₁[4] * f₁[8] - f₁[5] * f₁[7]

    # 2×2 minors of f₂
    t0 = f₂[5] * f₂[9] - f₂[6] * f₂[8]
    t1 = f₂[4] * f₂[9] - f₂[6] * f₂[7]
    t2 = f₂[4] * f₂[8] - f₂[5] * f₂[7]

    # Cubic coefficients: det(λ f₁ + f₂) = a λ³ + b λ² + c λ + d
    a = f₁[1] * t3 - f₁[2] * t4 + f₁[3] * t5

    b = (
        f₂[1] * t3 - f₂[2] * t4 + f₂[3] * t5 -
            f₂[4] * (f₁[2] * f₁[9] - f₁[3] * f₁[8]) +
            f₂[5] * (f₁[1] * f₁[9] - f₁[3] * f₁[7]) -
            f₂[6] * (f₁[1] * f₁[8] - f₁[2] * f₁[7]) +
            f₂[7] * (f₁[2] * f₁[6] - f₁[3] * f₁[5]) -
            f₂[8] * (f₁[1] * f₁[6] - f₁[3] * f₁[4]) +
            f₂[9] * (f₁[1] * f₁[5] - f₁[2] * f₁[4])
    )

    c = (
        f₁[1] * t0 - f₁[2] * t1 + f₁[3] * t2 -
            f₁[4] * (f₂[2] * f₂[9] - f₂[3] * f₂[8]) +
            f₁[5] * (f₂[1] * f₂[9] - f₂[3] * f₂[7]) -
            f₁[6] * (f₂[1] * f₂[8] - f₂[2] * f₂[7]) +
            f₁[7] * (f₂[2] * f₂[6] - f₂[3] * f₂[5]) -
            f₁[8] * (f₂[1] * f₂[6] - f₂[3] * f₂[4]) +
            f₁[9] * (f₂[1] * f₂[5] - f₂[2] * f₂[4])
    )

    d = f₂[1] * t0 - f₂[2] * t1 + f₂[3] * t2

    roots = _real_cubic_roots(a, b, c, d)

    # Build F matrices from real roots (0 allocs).
    # Cubic always yields 1 or 3 real roots.
    # Return: FixedModels{3,FundMat{T}} or nothing
    n = 0
    F₁ = F₂ = F₃ = FundMat{T}(Tuple(zero(SMatrix{3, 3, T, 9})))
    for k in 1:3
        isnan(roots[k]) && continue
        λ = roots[k]
        fv = λ * f₁ + f₂
        # fv is row-major: [F₁₁,F₁₂,F₁₃, F₂₁,F₂₂,F₂₃, F₃₁,F₃₂,F₃₃]
        # SMatrix constructor is column-major: transpose to get correct layout
        F_mat = FundMat{T}(
            fv[1], fv[4], fv[7],
            fv[2], fv[5], fv[8],
            fv[3], fv[6], fv[9]
        )
        F_norm = sign_normalize(F_mat)::FundMat{T}
        n += 1
        if n == 1
            F₁ = F_norm
        elseif n == 2
            F₂ = F_norm
        else
            F₃ = F_norm
        end
    end

    n == 0 && return nothing
    return FixedModels{3, FundMat{T}}(n, (F₁, F₂, F₃))
end

# =============================================================================
# Internal: Cubic Solver
# =============================================================================

function _real_cubic_roots(a::T, b::T, c::T, d::T) where {T}
    if abs(a) < eps(T) * (abs(b) + abs(c) + abs(d))
        if abs(b) < eps(T) * (abs(c) + abs(d))
            return SVector{3, T}(T(NaN), T(NaN), T(NaN))
        end
        disc = c * c - 4 * b * d
        if disc < zero(T)
            return SVector{3, T}(T(NaN), T(NaN), T(NaN))
        end
        sq = sqrt(disc)
        m = c < zero(T) ? -c + sq : -c - sq
        x₁ = m / (2 * b)
        x₂ = (2 * d) / m
        return SVector{3, T}(x₁, x₂, T(NaN))
    end

    p = (3 * a * c - b * b) / (3 * a * a)
    q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a)
    discriminant = q * q / 4 + p * p * p / 27
    shift = -b / (3 * a)

    if abs(p) < eps(T) && abs(q) < eps(T)
        return SVector{3, T}(shift, shift, shift)
    elseif discriminant <= zero(T)
        # Three real roots (Vieta's trigonometric method)
        r = 2 * sqrt(-p / 3)
        θ₃ = acos(clamp(3 * q / (p * r), -one(T), one(T))) / 3

        x₁ = r * cos(θ₃) + shift
        x₂ = r * cos(θ₃ - 2 * T(π) / 3) + shift
        x₃ = r * cos(θ₃ + 2 * T(π) / 3) + shift
        return SVector{3, T}(x₁, x₂, x₃)
    else
        # One real root (Cardano)
        sq = sqrt(discriminant)
        S = cbrt(-q / 2 + sq)
        U = cbrt(-q / 2 - sq)
        x₁ = S + U + shift
        return SVector{3, T}(x₁, T(NaN), T(NaN))
    end
end

# =============================================================================
# Internal: Oriented Epipolar Constraint
# =============================================================================
#
# For F satisfying u₂ᵀ F u₁ = 0, the oriented constraint checks that all
# sample correspondences have consistent geometric orientation.
#
# Formula (Chum et al. ICPR 2004):
#   sᵢ = (Fᵀ ũ₂ⁱ) · (e₁ × ũ₁ⁱ) = μᵢ ‖e₁ × ũ₁ⁱ‖²
# where e₁ = epipole in image 1 (right null vector of F, from row cross products).
# All sᵢ must have the same sign (pairwise chirality consistency).

"""
    test_model(F::FundMat, u₁_sample, u₂_sample) -> Bool

Oriented epipolar constraint [Chum2004](@citep). For an exact correspondence,
the epipolar line `l₁ = Fᵀũ₂` and the line `e₁ × ũ₁` are the same line in
image 1 (both go through `e₁` and `ũ₁`), so `Fᵀũ₂ = μᵢ(e₁ × ũ₁)` for
some scalar `μᵢ`. The signum uses the dot product of all three components:

    sᵢ = (Fᵀũ₂ⁱ) · (e₁ × ũ₁ⁱ) = μᵢ · ‖e₁ × ũ₁ⁱ‖²

Since `‖e₁ × ũ₁ⁱ‖² ≥ 0`, `sign(sᵢ) = sign(μᵢ)`. The code checks
`sᵢ · sⱼ > 0` for all pairs, i.e., that all `μᵢ` share the same sign.
Using the full dot product (rather than a single component) avoids
numerical instability when one component happens to be near zero.

This is a chirality parity check. The absolute sign of `μᵢ` is meaningless
(F and e₁ are projective, defined up to scale), but if two correspondences
give different signs, they cannot both be in front of both cameras. The test
rejects F matrices that are algebraically valid (`u₂ᵀFu₁ = 0`) but
geometrically inconsistent. An inner-loop gate: `F` and the points in the same
frame, the estimator's normalization (sign-based, so any positive-scale
normalization gives the same answer).
"""
test_model(F::FundMat, u₁_sample, u₂_sample) =
    _oriented_epipolar_check(u₁_sample, u₂_sample, SMatrix{3, 3}(F))

function test_model(F::FundMat, cs, idx::AbstractVector{<:Integer})
    return _oriented_epipolar_check(view(sources(cs), idx), view(targets(cs), idx), SMatrix{3, 3}(F))
end

# (Fᵀ ũ₂) · (e₁ × ũ₁), where ũ = [x, y, 1] (homogeneous).
@inline function _oriented_signum(F::StaticMatrix{3, 3, T}, ec, u₁, u₂) where {T}
    # l = Fᵀ ũ₂  (epipolar line in image 1)
    l₁ = F[1, 1] * u₂[1] + F[2, 1] * u₂[2] + F[3, 1]
    l₂ = F[1, 2] * u₂[1] + F[2, 2] * u₂[2] + F[3, 2]
    l₃ = F[1, 3] * u₂[1] + F[2, 3] * u₂[2] + F[3, 3]
    c = cross(SVector(ec[1], ec[2], ec[3]), SVector(u₁[1], u₁[2], one(u₁[1])))   # e₁ × ũ₁
    return l₁ * c[1] + l₂ * c[2] + l₃ * c[3]
end

function _oriented_epipolar_check(u₁, u₂, F::StaticMatrix{3, 3, T}) where {T}
    n = length(u₁)
    n < 2 && return true

    # Epipole e₁: right null vector of F (Fe₁=0), computed as row1 × row3
    ec = SA[
        F[1, 2] * F[3, 3] - F[1, 3] * F[3, 2],
        F[1, 3] * F[3, 1] - F[1, 1] * F[3, 3],
        F[1, 1] * F[3, 2] - F[1, 2] * F[3, 1],
    ]

    if ec[1]^2 + ec[2]^2 + ec[3]^2 < eps(T)
        ec = SA[
            F[2, 2] * F[3, 3] - F[2, 3] * F[3, 2],
            F[2, 3] * F[3, 1] - F[2, 1] * F[3, 3],
            F[2, 1] * F[3, 2] - F[2, 2] * F[3, 1],
        ]
    end

    @inbounds begin
        # sᵢ = (Fᵀ ũ₂) · (e₁ × ũ₁) = μᵢ ‖e₁ × ũ₁‖²
        # Full dot product over all 3 components for numerical robustness.
        s_ref = _oriented_signum(F, ec, u₁[1], u₂[1])

        check_n = min(n, 20)
        for i in 2:check_n
            s_i = _oriented_signum(F, ec, u₁[i], u₂[i])
            if s_ref * s_i < zero(T)
                return false
            end
        end
    end
    return true
end

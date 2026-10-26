using LinearAlgebra: norm, svd, dot   # IAC stack + radial camera

# ==============================================================================
# Zhang-style intrinsics regression from plane-to-image homographies
# ==============================================================================
#
# Overdetermined linear solve for the calibration from a set of plane→image
# homographies via the image of the absolute conic (IAC), following
# Zhang, "A flexible new technique for camera calibration" (TPAMI 2000),
# restricted to zero skew and a FIXED principal point. When the division-model
# distortion has been peeled off by the radial-homography arm first
# (the BabelCalib back-end split — Lochman et al., ICCV 2021), each board's H
# is a genuine pinhole homography and these constraints hold at any FOV.
#
# FRAME CONTRACT: every H maps plane coordinates (any planar similarity frame;
# per-board frames may differ) to image coordinates whose ORIGIN IS THE
# PRINCIPAL POINT (centered pixels — the frame the radial models' lift already
# uses). Then ω = K⁻ᵀK⁻¹ = diag(1/fx², 1/fy², 1) and each H = [h₁ h₂ h₃]
# contributes Zhang's two linear constraints on ω:
#
#     h₁ᵀ ω h₂ = 0                    (r₁ ⟂ r₂)
#     h₁ᵀ ω h₁ − h₂ᵀ ω h₂ = 0        (‖r₁‖ = ‖r₂‖)

# One homography's two constraint rows on (a, b) = (1/fx², 1/fy²), with the
# constant (ω₃₃ = 1) contribution moved to the right-hand side. The whole H
# is rescaled by ‖h₁‖ first so every board enters the stack at unit scale.
function _iac_rows(H::HomographyMat)
    s = 1 / norm(SVector(H[1, 1], H[2, 1], H[3, 1]))
    h1 = SVector(H[1, 1], H[2, 1], H[3, 1]) * s
    h2 = SVector(H[1, 2], H[2, 2], H[3, 2]) * s
    r1 = SVector(h1[1] * h2[1], h1[2] * h2[2], h1[3] * h2[3])
    r2 = SVector(h1[1]^2 - h2[1]^2, h1[2]^2 - h2[2]^2, h1[3]^2 - h2[3]^2)
    return r1, r2
end

function _iac_system(Hs, weights)
    n = length(Hs)
    n ≥ 1 || throw(ArgumentError("zhang_intrinsics needs at least one homography"))
    check_weights(weights, n)
    A = Matrix{Float64}(undef, 2n, 2)
    c = Vector{Float64}(undef, 2n)
    for (i, H) in enumerate(Hs)
        w = weights === nothing ? 1.0 : sqrt(Float64(weights[i]))
        r1, r2 = _iac_rows(H)
        A[2i - 1, 1] = w * r1[1]; A[2i - 1, 2] = w * r1[2]; c[2i - 1] = -w * r1[3]
        A[2i, 1] = w * r2[1]; A[2i, 2] = w * r2[2]; c[2i] = -w * r2[3]
    end
    return A, c
end

function _iac_focal(x)
    all(>(0), x) || throw(
        ArgumentError(
            "IAC solve produced a non-positive 1/f² — the homography set is " *
                "degenerate (near fronto-parallel views) or violates the " *
                "centered-frame contract (principal point must be the origin)"
        )
    )
    return 1 ./ sqrt.(x)
end

"""
    zhang_intrinsics(Hs; center, weights = nothing) -> LogicalIntrinsics{Float64}

The calibration from plane→image homographies by Zhang's IAC constraints —
one linear least-squares step, assuming zero skew and the principal point at
the known `center` (pixels, required). Each `H` maps its plane to the
caller's pixels; they are read about `center` internally, and the result is
in the caller's pixels: `K = [fx 0 cx; 0 fy cy; 0 0 1]` with `(cx, cy) =
center`. Each homography contributes two linear equations on
`(1/fx², 1/fy²)`; the overdetermined system is solved by (optionally
weighted) least squares. `weights` holds one non-negative vote strength per
homography (e.g. inlier counts). Throws `ArgumentError` when the set is
degenerate — all views near fronto-parallel leave the IAC unobservable.
"""
function zhang_intrinsics(Hs::AbstractVector{<:HomographyMat}; center, weights = nothing)
    A, c = _iac_system(_about_center(Hs, center), weights)
    F = svd(A)
    F.S[2] > 1.0e-10 * F.S[1] || throw(
        ArgumentError(
            "IAC system is rank-deficient — homography set cannot separate " *
                "fx from fy (add tilted views, or supply the sensor pitch to " *
                "the single-f method)"
        )
    )
    x = A \ c
    fx, fy = _iac_focal(x)
    return pixel_intrinsics(fx, fy, center)
end

# The homographies read about the principal point (their targets translated
# by −center: the IAC system is written with the principal point at the
# origin), and the K back in the caller's pixels.
_about_center(Hs, center) =
    [HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, -center[1], -center[2], 1) * SMatrix{3, 3, Float64}(H))) for H in Hs]
# The pixel K literal (focal lengths `fx`, `fy`, principal point `center`, no
# skew) — the one spelling — and the intrinsics wrapping it.
_pixel_K(fx, fy, center) = SMatrix{3, 3, Float64, 9}(fx, 0, 0, 0, fy, 0, center[1], center[2], 1)
pixel_intrinsics(fx, fy, center) = LogicalIntrinsics(CameraCalibrationMat(_pixel_K(fx, fy, center)))

"""
    zhang_intrinsics(Hs, pitch::Size2; center, weights = nothing) -> LogicalIntrinsics{Float64}

Known-sensor variant: with the pixel pitch given, the aspect ratio is fixed
(`fy/fx = pitch.width / pitch.height`) and only a single focal length remains.
Each homography's two constraints collapse onto one unknown `1/fx²`, solved by
weighted least squares — this stays well-posed even for view sets that cannot
separate `fx` from `fy`. The metric focal length is `fx * pitch.width`.
"""
function zhang_intrinsics(
        Hs::AbstractVector{<:HomographyMat}, pitch::Size2;
        center, weights = nothing
    )
    # px is dimensionless, so a length per pixel is a length (3.9μm == 3.9μm/px)
    all(c -> dimension(c) == dimension(mm / px), pitch) ||
        throw(ArgumentError("pitch must be a length per pixel, got $pitch"))
    τ = ustrip(pitch.width / pitch.height)   # fy/fx, dimensionless: fx = f/pitch_x, fy = f/pitch_y
    A, c = _iac_system(_about_center(Hs, center), weights)
    u = A[:, 1] .+ A[:, 2] ./ τ^2            # b = a/τ²
    den = dot(u, u)
    den > 0 || throw(ArgumentError("IAC system is degenerate — no constraint on f"))
    a = dot(u, c) / den
    fx = only(_iac_focal(SVector(a)))
    return pixel_intrinsics(fx, τ * fx, center)
end

# ==============================================================================
# The 1-D radial camera (BabelCalib back-end, Lochman, Liepieshov, Chen,
# Perdoch, Zach, Pritts — ICCV 2021)
# ==============================================================================
#
# Rows 1–2 of a plane→image homography are the 1-D RADIAL CAMERA: with square
# pixels, K = diag(f, f, 1) and the principal point at the origin,
#     H₁:₂ = λf · [r₁₁ r₁₂ t₁; r₂₁ r₂₂ t₂]
# — invariant to f and to ANY radial distortion about the origin. Its
# constraint is the point–radial-line incidence, exposed through
# `constraint_with_jacobians(P, X, x) → (g, G_x, G_θ)` like every model here; its
# minimal solver is `radial_camera_5pt` (`RadialCamera5pt` behind `solve`).
# The focal and division-model stack built on it — the radial camera's fit,
# the division lift, the whitened linear focal solve and its robust driver —
# lives in VisualGeometryToolkitEstimation (`fit_radial_camera`, `division_camera_step`,
# `fit_division_camera`).

# ------------------------------------------------------------------------------
# RadialCamera — the 1-D radial camera as a value, with its constraint
# ------------------------------------------------------------------------------

"""
    RadialCamera{T}

The 1-D radial camera of one planar board: the 2×3 block `P` such that the
image point of a plane point X̃ lies on the radial line through `center` in
direction `P·X̃` (rows 1–2 of the plane homography taken about that centre, up
to scale — invariant to f and to any radially symmetric distortion about it).

The centre is part of the camera, as it is for [`RadialHomography`](@ref).
A radial camera without one is only half a model: its `P` predicts a
direction, and a direction is not a locus in the image until you know where
the pencil is centred. Carrying it here is what lets [`project`](@ref) return
the `Line2D` the pixel must lie on, and lets `constraint_with_jacobians` take the
RAW pixel — so no caller has to pre-centre its correspondences and keep that
convention in its head.

`RadialCamera(H)` reads it off a `HomographyMat` (about the origin unless a
centre is given); `HomographyMat(P)` returns the homography with the
placeholder third row `(0, 0, 1)` that the focal solve accepts (it never reads
row 3), carrying the centre translation.
"""
struct RadialCamera{T <: Real} <: AbstractRadialModel
    P::SMatrix{2, 3, T, 6}
    center::Point2{T}
end
RadialCamera{T}(P::SMatrix{2, 3, T, 6}) where {T} = RadialCamera{T}(P, Point2{T}(zero(T), zero(T)))
# The padding element of a `FixedModels` of radial cameras (zero rows, centre at
# the origin): the unused slots, never read as a solution.
Base.zero(::Type{RadialCamera{T}}) where {T} = RadialCamera{T}(zero(SMatrix{2, 3, T, 6}), zero(Point2{T}))
Base.zero(m::RadialCamera) = zero(typeof(m))

"The 2×3 block `P` of a [`RadialCamera`](@ref): rows 1–2 of the plane homography about its [`center`](@ref), up to scale."
radial_rows(m::RadialCamera) = m.P

# The radial camera's minimal manifold: its rows up to scale and sign are a point
# of S⁵ (vec(P), column-major — the vectorization `constraint_with_jacobians`' G_θ
# differentiates), the centre riding along. One constraint per plane ↔ pixel
# correspondence, the pixel its measurement: the covariance machinery
# (`implicit_covariance`) takes it as is.
struct _RadialCameraRows{T}
    center::Point2{T}
end
(e::_RadialCameraRows)(p::StaticVector{6}) = RadialCamera(SMatrix{2, 3}(Tuple(p)), e.center)
minimal_manifold(m::RadialCamera) = SphereManifold(SVector{6}(Tuple(m.P)), _RadialCameraRows(m.center))
ambient_coordinates(m::RadialCamera) = SVector{6}(Tuple(m.P))
local_coordinates(ch::SphereManifold{6}, m::RadialCamera) = local_coordinates(ch, SVector{6}(Tuple(m.P)))
constraint_dim(::Type{<:RadialCamera}) = 1
_measurement_dim(::Type{<:RadialCamera}) = 2
RadialCamera(P::SMatrix{2, 3, T, 6}) where {T} = RadialCamera{T}(P)
RadialCamera(P::SMatrix{2, 3, T, 6}, center) where {T} =
    RadialCamera{T}(P, Point2{T}(center[1], center[2]))

# `RadialCamera(H, c)` is the exact inverse of `HomographyMat(::RadialCamera)`:
# `H` is a plane→PIXEL homography in the SAME (raw) image coordinates the centre
# is given in, so the image origin is moved to `c` first — rows 1–2 of T(−c)·H —
# and only then read off. With the default origin centre T(0) = I, so the
# one-argument form is rows 1–2 of `H`, unchanged.
function RadialCamera(H::HomographyMat{T}, center = Point2{T}(zero(T), zero(T))) where {T}
    c = Point2{T}(center[1], center[2])
    M = SMatrix{3, 3, T, 9}(H)
    return RadialCamera{T}(
        SMatrix{2, 3, T, 6}(
            M[1, 1] - c[1] * M[3, 1], M[2, 1] - c[2] * M[3, 1],
            M[1, 2] - c[1] * M[3, 2], M[2, 2] - c[2] * M[3, 2],
            M[1, 3] - c[1] * M[3, 3], M[2, 3] - c[2] * M[3, 3]
        ),
        c
    )
end

# The radial camera of a radial homography — the SAME law as the raw-pixel
# constructor above, because `m.H` is a raw-pixel homography too. The model's
# contract is `(d − c ; 1 + k‖d − c‖²) ≃ λ · T₋c · H s̃` (homography_5pt_radial.jl):
# the recentring is applied TO `H s̃`, so `H` maps the plane into RAW pixels and
# the radial camera about `m.center` is rows 1–2 of `T(−m.center)·m.H`.
#
# An earlier version of this took the rows unchanged, on the false claim that
# `m.H` was already centred. Measured against the model's own `radial_transfer`,
# that put the predicted ray ~62° off the true one and 121 px from the observed
# pixel whenever the centre was not the origin.
RadialCamera(m::RadialHomography) = RadialCamera(m.H, m.center)

"""
    RadialCamera(cam::PosedCamera) -> RadialCamera

The 1-D radial camera of a POSED camera viewing the plane `z = 0` of its
extrinsics' source frame: rows 1–2 of the induced homography `H = K·[r₁ r₂ t]`,
read about `K`'s principal point.

An IDENTITY, not a fit — no correspondences, no least squares, the pose taken
exactly as it stands. That is what makes this the safe way to hand a calibrated
arm's `Extrinsics3` to VisualGeometryToolkitEstimation's `division_camera_step`: re-estimating a radial camera
from the same board's correspondences re-runs the metric upgrade's sign decision
on a slightly different matrix, and can flip the board's pose.

**The lens is ignored by construction, not by omission.** A radially symmetric
distortion about that same principal point moves an image point along its own
ray from the centre, so it cannot change the ray — and the ray is all this
model claims (see [`RadialCamera`](@ref)). That invariance is why a camera with
ANY radially symmetric lens is admissible here, while
`HomographyMat(::PosedPinholeCamera)` stays Pinhole-only: the plane map of a
distorted camera is not a homography, but its radial camera is exact.
"""
function RadialCamera(cam::PosedCamera{<:CameraModel})
    K = calibration_matrix(cam)
    return RadialCamera(_induced_plane_homography(cam), Point2{Float64}(K[1, 3], K[2, 3]))
end

HomographyMat(P::RadialCamera{T}) where {T} = HomographyMat{T}(
    Tuple(
        SMatrix{3, 3, T, 9}(1, 0, 0, 0, 1, 0, P.center[1], P.center[2], 1) *
            SMatrix{3, 3, T, 9}(P.P[1, 1], P.P[2, 1], 0, P.P[1, 2], P.P[2, 2], 0, P.P[1, 3], P.P[2, 3], 1)
    )
)
# A radial camera is a plane point → image LINE map, so a frame change is `∘`,
# as for every map (a normalization moves it by
# `target_map(N) ∘ P ∘ inv(source_map(N))`, back by
# `inv(target_map(N)) ∘ P ∘ source_map(N)`; an image-side `FramedAffine` by
# `HomographyMap(m) ∘ P`):
#
#   plane side (any collineation B):  rows P·B, centre unchanged
#   image side (any AFFINE D = A·x + b):  the line through c in direction P·X̃
#       goes to the line through D(c) in direction A·P·X̃ — rows A·P, centre D(c)
#
# A projective image-side map does not keep the pencil's centre a point of the
# model: an `ArgumentError`. To re-attach a caller's DATUM centre exactly after
# moving back (the inverse map carrying the normalized origin back rounds it):
# `RadialCamera(radial_rows(inv(target_map(N)) ∘ P ∘ source_map(N)), c)`.
function Base.:∘(a::Union{HomographyMap{3}, _CTLeafTransformation}, P::RadialCamera)
    D = SMatrix{3, 3}(to_homogeneous(a))
    (iszero(D[3, 1]) && iszero(D[3, 2]) && isone(D[3, 3])) ||
        throw(ArgumentError("∘ on a RadialCamera: the image-side map must be affine; got $D"))
    A = SMatrix{2, 2}(D[1, 1], D[2, 1], D[1, 2], D[2, 2])
    return RadialCamera(A * P.P, AffineMap(D)(P.center))
end
Base.:∘(P::RadialCamera, b::Union{HomographyMap{3}, _CTLeafTransformation}) =
    RadialCamera(P.P * SMatrix{3, 3}(to_homogeneous(b)), P.center)

"""
    project(P::RadialCamera, X) -> Line2D

The 1-D radial camera's image of the plane point `X`: the radial LINE through
the camera's centre in direction `P·X̃`. The camera knows where a point's ray
points, not how far along it the point lands, so a point of the plane images
to a line — and the RAW pixel of `X` lies on it under any radially symmetric
distortion about that centre. [`constraint_with_jacobians`](@ref) is that
statement as a residual.

[`direction`](@ref) gives the bare `P·X̃` for callers working up to scale.
"""
function project(P::RadialCamera, X::StaticVector{2})
    d = direction(P, X)
    c = P.center
    return Line2D(Point2(c[1], c[2]), Point2(c[1] + d[1], c[2] + d[2]))
end

"""
    direction(P::RadialCamera, X) -> Vec2

The direction `P·X̃` of the radial line [`project`](@ref) returns — the raw
2-vector, for callers that work up to scale and do not need the locus.
"""
direction(P::RadialCamera, X::StaticVector{2}) = Vec2(P.P * SVector(X[1], X[2], one(eltype(X))))

"""
    constraint_with_jacobians(P::RadialCamera, X::SVector{2}, x::SVector{2})
        -> (g::T, G_x::SVector{2,T}, G_θ::SVector{6,T})

Radial-camera constraint of one correspondence (plane point `X`, RAW pixel
`x`): `g = x₁·(P₂·X̃) − x₂·(P₁·X̃)` with `x` taken relative to the camera's
`center` — the pixel lies on the radial line [`project`](@ref) returns.
`G_x = ∂g/∂x = (P₂·X̃, −P₁·X̃)`, so `g/‖G_x‖` is exactly the perpendicular
distance of the pixel from that line; `G_θ = ∂g/∂vec(P)` (column-major:
P₁₁, P₂₁, P₁₂, P₂₂, P₁₃, P₂₃).

The centre lives on the camera, so correspondences stay in the frame they
were measured in and no caller pre-centres them.
"""
@inline function constraint_with_jacobians(P::RadialCamera, X::StaticVector{2}, x::StaticVector{2})
    # Element types promote (Dual-safe: ForwardDiff differentiates these bodies).
    X̃ = SVector(X[1], X[2], one(eltype(X)))
    p1 = dot(SVector(P.P[1, 1], P.P[1, 2], P.P[1, 3]), X̃)
    p2 = dot(SVector(P.P[2, 1], P.P[2, 2], P.P[2, 3]), X̃)
    x1 = x[1] - P.center[1]
    x2 = x[2] - P.center[2]
    g = x1 * p2 - x2 * p1
    G_x = SVector(p2, -p1)
    G_θ = SVector(-x2 * X̃[1], x1 * X̃[1], -x2 * X̃[2], x1 * X̃[2], -x2 * X̃[3], x1 * X̃[3])
    return g, G_x, G_θ
end

"""
    radial_camera_5pt(pts; center = Point2(0, 0)) -> RadialCamera{Float64} | nothing

The exactly determined radial camera from five `plane_point => raw_pixel`
pairs about `center` — the minimal solver a RANSAC over board correspondences samples.
Closed form and homogeneous, the way [`homography_4pt`](@ref VisualGeometryToolkitCore.homography_4pt) is: the first
three plane points form a projective basis `B`, so `P·B = [λ₁y₁ λ₂y₂ λ₃y₃]`
with `yⱼ` the pixels' ray directions and three unknown scales; the fourth and
fifth points, written in that basis (`c = B⁻¹X̃`), each give one linear
equation in `λ` (their pixel must be parallel to `Σⱼ cⱼλⱼyⱼ`), and `λ` is the
cross product of the two equation rows. Then `P = [λⱼyⱼ]·B⁻¹`, unit Frobenius
norm. Two 3×3 static solves and a cross product — no SVD, no scale fixed by
hand, no root to pick. `nothing` when the basis points are collinear or the
two equations are dependent (the configuration does not determine `P`).
"""
function radial_camera_5pt(pts::AbstractVector; center = Point2(0.0, 0.0))
    length(pts) == 5 || throw(ArgumentError("radial_camera_5pt takes exactly 5 correspondences, got $(length(pts))"))
    c = Point2{Float64}(center[1], center[2])
    X = ntuple(i -> (p = pts[i][1]; SVector{3, Float64}(p[1], p[2], 1.0)), Val(5))
    y = ntuple(i -> (q = pts[i][2]; SVector{2, Float64}(q[1] - c[1], q[2] - c[2])), Val(5))
    B = hcat(X[1], X[2], X[3])
    abs(LinearAlgebra.det(B)) > 1.0e-12 * maximum(abs, B)^3 || return nothing        # collinear basis
    Binv = inv(B)
    c4 = Binv * X[4]; c5 = Binv * X[5]
    a = SVector(c4[1] * _cross2d(y[4], y[1]), c4[2] * _cross2d(y[4], y[2]), c4[3] * _cross2d(y[4], y[3]))
    b = SVector(c5[1] * _cross2d(y[5], y[1]), c5[2] * _cross2d(y[5], y[2]), c5[3] * _cross2d(y[5], y[3]))
    λ = cross(a, b)
    nλ = norm(λ)
    (isfinite(nλ) && nλ > 1.0e-12 * max(norm(a) * norm(b), 1.0e-300)) || return nothing   # dependent equations
    Pm = hcat(λ[1] * y[1], λ[2] * y[2], λ[3] * y[3]) * Binv
    nP = norm(Pm)
    (isfinite(nP) && nP > 0) || return nothing
    return RadialCamera{Float64}(SMatrix{2, 3, Float64, 6}(Pm / nP), c)
end

"""
    RadialCamera5pt(; center = Point2(0, 0))

Minimal 5-point radial-camera solver ([`solve`](@ref) verb over the
[`radial_camera_5pt`](@ref) kernel) on `plane_point => raw_pixel`
correspondences. The pencil's `center` belongs to the solver, not the data:
it is the problem's geometry, carried onto every returned camera.
`sample_size` 5, `max_solutions` 1; a degenerate sample (collinear basis,
dependent equations) gives `count == 0`.

Deliberately NOT entered in the `minimal_solvers` registry: generic
`Cspond{Point2, Point2}` data carries no pencil centre (the same reason
`Homography5ptRadial` stays out). Construct it where the centre is known.
"""
struct RadialCamera5pt <: MinimalSolver{RadialCamera{Float64}}
    center::Point2{Float64}
end
RadialCamera5pt(; center = Point2(0.0, 0.0)) = RadialCamera5pt(Point2{Float64}(center[1], center[2]))
sample_size(::RadialCamera5pt) = 5
max_solutions(::RadialCamera5pt) = 1

function solve(s::RadialCamera5pt, cs, idx::AbstractVector{<:Integer})
    u = sources(cs); v = targets(cs)
    pts = SVector{5}(ntuple(k -> (@inbounds SVector{2, Float64}(u[idx[k]]) => SVector{2, Float64}(v[idx[k]])), Val(5)))
    P = radial_camera_5pt(pts; center = s.center)
    M = RadialCamera{Float64}
    return P === nothing ?
        FixedModels{1, M}(0, (zero(M),)) :
        FixedModels{1, M}(1, (P,))
end

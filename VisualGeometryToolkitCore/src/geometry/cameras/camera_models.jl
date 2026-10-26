# ==============================================================================
# CameraModel - Composable camera model combining intrinsics and lens
# ==============================================================================

"""
    CameraModel{I, L} <: AbstractCamera

Camera model combining intrinsics and lens model: an unposed camera
([`has_extrinsics`](@ref) is `false`), the parametric intrinsics × lens
family. The generic grid models ([`CentralGenericModel`](@ref),
[`NoncentralGenericModel`](@ref)) are the other unposed cameras; they have no
K/lens split.

Use `project(model, X)` to map a 3D camera-space point to 2D pixel
coordinates, and `backproject(model, u)` / `unproject(model, u, depth)`
for the inverse directions.

# Fields
- `intrinsics::I`: Camera intrinsics (LogicalIntrinsics)
- `lens::L`: Lens model (`Pinhole`, a `Lens`, a `LiftLens`, …)

# Examples
```julia
# Logical camera with pinhole lens
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
logical = LogicalIntrinsics(K)
model = CameraModel(logical, Pinhole())

# Physical camera with pinhole lens
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
physical = LogicalIntrinsics(f, pitch, pp)
model = CameraModel(physical, Pinhole())

# With Brown-Conrady distortion
model = CameraModel(logical, Lens(Perspective(), Radial((-0.1, 0.01))))

# Project a point
X = SVector(10.0mm, 5.0mm, 100.0mm)
u = project(model, X)
```
"""
struct CameraModel{I <: AbstractIntrinsics, L <: AbstractLensModel} <: AbstractCamera
    intrinsics::I
    lens::L
end


# ============================================================================
# CameraModel Constructors
# ============================================================================

# Convenience constructor with default pinhole lens
CameraModel(intrinsics::AbstractIntrinsics; lens = Pinhole()) =
    CameraModel(intrinsics, lens)

"""
    CameraModel(f::Quantity, pitch::Size2{<:Quantity}, pp::AbstractVector{<:Quantity}; lens=Pinhole()) -> CameraModel

Convenience constructor to build a complete physical camera model from parameters.

# Example
```julia
camera = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
```
"""
CameraModel(
    f::Quantity, pitch::Size2{<:Quantity}, pp::AbstractVector{<:Quantity};
    lens = Pinhole()
) =
    CameraModel(LogicalIntrinsics(f, pitch, pp), lens)

"""
    CameraModel(f::Quantity, pp::AbstractVector{<:Quantity}; lens=Pinhole()) -> CameraModel

Convenience constructor to build a logical camera model from focal length and principal point (both in pixels).

# Example
```julia
camera = CameraModel(800.0px, [320.0px, 240.0px])
```
"""
CameraModel(
    f::Quantity, pp::AbstractVector{<:Quantity};
    lens = Pinhole()
) =
    CameraModel(LogicalIntrinsics(f, pp), lens)

"""
    CameraModel(f::Tuple{T,T}, pp::AbstractVector{<:Quantity}; lens=Pinhole()) where T<:Quantity -> CameraModel

Convenience constructor to build a logical camera model with anisotropic focal lengths (fx ≠ fy).

# Example
```julia
camera = CameraModel((800.0px, 805.0px), [320.0px, 240.0px])
```
"""
CameraModel(
    f::Tuple{T, T}, pp::AbstractVector{<:Quantity};
    lens = Pinhole()
) where {T <: Quantity} =
    CameraModel(LogicalIntrinsics(f, pp), lens)

"""
    CameraModel(map::AffineMap; lens=Pinhole()) -> CameraModel

Construct a logical camera model from a `CoordinateTransformations.AffineMap`.

The 2×2 linear part encodes focal lengths (and optional skew), the 2D translation
encodes the principal point. This constructor enables idiomatic composition:

```julia
K = CameraCalibrationMat(M)
model = CameraModel(EuclideanMap(PixelFrame{:opencv}, PixelFrame{:colmap}) ∘ AffineMap(K); lens=lens)
```
"""
function CameraModel(map::AffineMap; lens::AbstractLensModel = Pinhole())
    K = CameraCalibrationMat{Float64}(Tuple(to_homogeneous(map)))
    return CameraModel(LogicalIntrinsics(K), lens)
end

"""
    CameraModel(K::CameraCalibrationMat; lens=Pinhole(), image_origin=:colmap) -> CameraModel

Convenience constructor from a calibration matrix K directly.

Wraps K in `LogicalIntrinsics` internally. The `image_origin` keyword specifies
which pixel coordinate convention K is expressed in (`:opencv`, `:colmap`, or
`:julia`). The principal point is shifted to COLMAP convention (pixel centers at
0.5) before storage.

# Example
```julia
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
model = CameraModel(K; lens=Lens(Perspective(), Radial((-0.1,))))

# From OpenCV calibration (pixel centers at 0,0)
K_cv = CameraCalibrationMat(M)
model = CameraModel(K_cv; lens=lens, image_origin=:opencv)
```
"""
function CameraModel(
        K::CameraCalibrationMat; lens::AbstractLensModel = Pinhole(),
        image_origin::Symbol = :colmap
    )
    return if image_origin != :colmap
        # Origin shift via the 0.7 frame-map factory. The removed 0.6 surface
        # `Translation(::Symbol, ::Symbol)` must NOT be called here: without
        # the VGC method, that expression silently matches CT's generic
        # `Translation(x, y)` and builds a `Translation{SVector{2,Symbol}}`
        # (the 0.7.0 leftover bug). Wrapped back into a CT `Translation` (via
        # `.t`, unitless `Vec2{Float64}`) so `∘ AffineMap(K)` fuses eagerly
        # into a plain `AffineMap`, as before.
        origin_shift = Translation(
            EuclideanMap(PixelFrame{image_origin}, PixelFrame{:colmap}).t
        )
        CameraModel(origin_shift ∘ AffineMap(K); lens)
    else
        CameraModel(LogicalIntrinsics(K), lens)
    end
end

"""
    CameraModel(K::SMatrix{3,3,<:Real}; lens=Pinhole(), image_origin=:colmap) -> CameraModel

A camera model from a plain static 3×3 calibration matrix, as dataset readers
and pool records store it: `K` is validated and wrapped as a
[`CameraCalibrationMat`](@ref), then built by the method above. With the
default `Pinhole()` lens this is the pinhole camera of `K`. (A plain `Matrix`
goes through `CameraCalibrationMat(K)` first.)
"""
CameraModel(K::SMatrix{3, 3, <:Real}; lens::AbstractLensModel = Pinhole(), image_origin::Symbol = :colmap) =
    CameraModel(CameraCalibrationMat(K); lens, image_origin)

"""
    CameraModel(; focal, pp, lens=Pinhole()) -> CameraModel

Unitless convenience constructor for a logical camera model.

Accepts plain `Real` values (no `px` units required). Useful for synthetic data
generation and quick prototyping.

# Arguments
- `focal`: Focal length in pixels — scalar `Real` for isotropic, or `Tuple{Real,Real}` for anisotropic
- `pp::Tuple{Real,Real}`: Principal point (cx, cy) in pixels
- `lens::AbstractLensModel=Pinhole()`: Lens model

# Examples
```julia
model = CameraModel(; focal=800.0, pp=(320.0, 240.0))
model = CameraModel(; focal=(800.0, 805.0), pp=(320.0, 240.0))  # anisotropic
```
"""
function CameraModel(;
        focal, pp::Tuple{Real, Real},
        lens::AbstractLensModel = Pinhole()
    )
    # Build K directly — the former path manufactured px quantities only to
    # strip them one call later (audit #15 round-trip kill).
    fx, fy = focal isa Tuple ? (Float64(focal[1]), Float64(focal[2])) :
        (Float64(focal), Float64(focal))
    K = CameraCalibrationMat{Float64}(
        Tuple(
            SMatrix{3, 3, Float64}(
                fx, 0.0, 0.0,
                0.0, fy, 0.0,
                Float64(pp[1]), Float64(pp[2]), 1.0
            )
        )
    )
    return CameraModel(LogicalIntrinsics(K), lens)
end

# ============================================================================
# CameraModel Accessors
# ============================================================================

"""
    calibration_matrix(model::CameraModel) -> SMatrix{3,3,T,9}

Return the 3×3 calibration matrix K as a plain `SMatrix`, preserving the
intrinsics' stored element type (`Float64` for the default constructors,
`Float32` for e.g. `LogicalIntrinsics(CameraCalibrationMat{Float32}(…))`).

# Example
```julia
model = CameraModel(; focal=800.0, pp=(320.0, 240.0))
K = calibration_matrix(model)  # [800 0 320; 0 800 240; 0 0 1]
```
"""
function calibration_matrix(model::CameraModel)
    K = model.intrinsics.K
    T = eltype(K)
    return SMatrix{3, 3, T, 9}(K)
end

# ============================================================================
# Projection and Backprojection Methods
# ============================================================================

# Pixel coordinates are *locations* in the image plane, so `project` returns
# a `Point2`.  CoordinateTransformations' `AffineMap` doesn't propagate the
# Point2/Vec2 affine-type distinction (it walks generic-array arithmetic,
# returning `Vec2` or `SVector{2}` depending on the lens function's return),
# so we coerce here at the boundary.  Lens distortion math internally still
# uses `Vec2` (math vectors) for its intermediate normalized coords.
"""
    project(model::CameraModel, X::AbstractVector) -> Point2

Project 3D point X in camera coordinates to 2D pixel coordinates.

# Arguments
- `model::CameraModel`: Camera model
- `X::AbstractVector`: 3D point in camera coordinates [X, Y, Z] (with units)

# Returns
- Point2 with pixel coordinates (including px units)

# Example
```julia
model = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
X = SVector(10.0mm, 5.0mm, 100.0mm)
u = project(model, X)  # Returns Point2 in pixels
```
"""
@inline function project(model::CameraModel, X::AbstractVector)
    raw = _pixel_affine(model.intrinsics)(
        project(model.lens, SVector{3}(X[1], X[2], X[3]))
    )
    # Natural eltype: Float64 pipelines are unchanged; Float32 intrinsics +
    # Float32 points stay Float32 instead of being silently widened.
    return Point2(raw[1], raw[2])
end

"""
    reprojection_error(camera, pose, X, u) -> SVector{2}

The reprojection residual of a scene point, Jacobian-free:
`project(camera, pose(X)) − u`, with `pose` any map into the camera frame (a
world → camera `EuclideanMap`, `X_c = R·X + t`; a camera → world pose enters as
`inv(pose)`, which is `x ↦ Rᵀ(x − t)`). NaN where the lens cannot image the
point (`project` answers on the ray domain). Element type follows the inputs,
so ForwardDiff duals flow through the pose and the lens.
"""
@inline function reprojection_error(camera, pose, X, u)
    û = project(camera, pose(X))
    return SVector(û[1] - u[1], û[2] - u[2])
end

"""
    backproject(model::CameraModel, u::StaticVector{2,T<:Real}) -> Vec3

Backproject 2D pixel coordinates to 3D ray direction.

Inputs and outputs are unitless (assumed units: px for input, unitless
direction for output). The output eltype follows the usual promotion of the
input and the intrinsics: Float64 pipelines are unchanged; a Float32 camera
(`LogicalIntrinsics{Float32}`) with Float32 pixels stays Float32; ForwardDiff
`Dual` pixels give a `Dual` direction, so the bearing can be differentiated
with respect to the pixel; integer pixels give the bearing of their
floating-point value, as the lens-level `backproject(lens, xd)` and the grid
models' `backproject` do.

The return type and semantics depend on the intrinsics type:
- LogicalIntrinsics: Returns normalized unitless direction


# Arguments
- `model::CameraModel`: Camera model with intrinsics and lens
- `u::StaticVector{2,T}`: 2D pixel coordinates [u, v] (unitless, in px)

# Returns
- `Vec3`: Normalized ray direction (unitless)

# Example
```julia
camera = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
u = SVector(400.0, 300.0)  # px (unitless)
ray = backproject(camera, u)  # Returns normalized Vec3 direction
```
"""
backproject(model::CameraModel, u::StaticVector{2, <:Real}) =
    Vec3(backproject(model.intrinsics, model.lens, u))

# Batched version for multiple points. (It once handed the whole vector to the
# scalar-only 3-arg method — a MethodError on every call, found by JET.)
backproject(model::CameraModel, points::AbstractVector{<:StaticVector{2, <:Real}}) =
    map(u -> backproject(model, u), points)

"""
    backproject_with_jacobian(model::CameraModel, u::StaticVector{2}) -> (f, M)

The unit bearing of pixel `u` and its 3×2 Jacobian `M = ∂f/∂u`, in ONE evaluation.
`f` is what [`backproject`](@ref) returns; `M` is what propagates a PIXEL
covariance onto the sphere, so the pair is the entry point for calibrated
two-view geometry on bearings (the tangent Sampson residual and its rows).

`M` is TANGENT by construction: `fᵀM = 0` to rounding, because it is built in
a tangent basis rather than by differentiating a normalization. Consumers
therefore need no explicit projection onto the tangent plane — which is what
keeps an inner loop free of the allocation such a projection would cost.

# The tangency assertion is LOAD-BEARING — do not simplify it away

`fᵀM = 0` is not decorative, and it is not merely an internal consistency
check. It is the ONLY test in the suite that catches a correspondence whose
two views' Jacobians have been SWAPPED — `M₂` handed where `M₁` belongs, the
natural mistake when porting a residual onto bearings.

Nothing else sees it, and that is measured rather than assumed
(`test/essential_tangent_mc_tests.jl`). The swap is very nearly
norm-preserving: the two halves of the tangent Sampson's `G_x` are scaled by
`|Eᵀf₂|` and `|Ef₁|`, whose ratio has median 1.018 and 5–95% range
[0.972, 1.066], so `‖G_x‖` barely moves and the residual stays plausible. Every
Monte Carlo calibration cell keeps reporting NEES ≈ 1 — the best of them gets
to 0.970, inside its own 4 SE band. Tangency, by contrast, fails loudly:
`fᵀM` measures 4.8e-19 with the right Jacobian against 1.0e-3 with the swapped
one, a factor of 2e15.

So a future cleanup that reads the tangency assertion as redundant with the
statistical gates would delete the one check that catches a view swap, and the
statistical gates would go on succeeding while it did.

Construction, given that a camera's forward map is the differentiable one:
with `B` an orthonormal basis of `T_f S²` ([`s2_tangent_basis`](@ref), shared
with `EssentialManifold5` rather than re-derived) and
`A = ∂project(model, ·)/∂f · B` the 2×2 restriction of the projection
differential to that plane, `M = B·A⁻¹`. The projection differential is taken
by automatic differentiation through the camera's OWN `project`, so every
lens — division, Kannala-Brandt, double sphere, a generic grid — is covered
without a per-model derivation, the same derivation-risk-free stance
`retract_jacobian` takes.

This is PoseLib's `unproject_with_jac`; GTSAM's nearest relative is
`PinholeBase::backprojectPointAtInfinity(p, Dresult_dpose, Dresult_dpoint)`,
which returns a `Unit3`. GTSAM spells the pixel → calibrated step `calibrate`,
which collides with THIS stack's task verb `calibrate` (estimating a camera),
so the name here stays `backproject`.

Returns `(NaN bearing, NaN Jacobian)` wherever `backproject` does — outside
the lens's domain, past a fold, or where the tangent restriction `A` is
singular (a ray the camera cannot resolve in two directions). Never throws.
"""
@inline function backproject_with_jacobian(model::CameraModel, u::StaticVector{2})
    f = backproject(model, u)
    T = eltype(f)
    nanM = SMatrix{3, 2, T, 6}(ntuple(_ -> T(NaN), Val(6)))
    all(isfinite, f) || return f, nanM
    b1, b2 = s2_tangent_basis(SVector{3, T}(f))
    B = SMatrix{3, 2, T, 6}(b1[1], b1[2], b1[3], b2[1], b2[2], b2[3])
    # The closure must stay ELEMENT-TYPE GENERIC: `_value_and_jacobian` feeds it
    # ForwardDiff `Dual`s, so forcing the camera's own `T` here would convert
    # them back to Float64 and lose the derivative (a MethodError, in fact).
    _, Jp = _value_and_jacobian(X -> SVector{2}(Tuple(project(model, X))), SVector{3, T}(f))
    A = Jp * B                                  # 2×2: the projection differential on T_f S²
    d = det(A)
    (isfinite(d) && !iszero(d)) || return f, nanM
    return f, B * inv(A)
end

# =============================================================================
# Trait-dispatched backprojection (generic entry point)
# =============================================================================

"""
    pinhole(model::CameraModel) -> CameraModel{I, Pinhole}
    imaging_transform(model::CameraModel)

The lens-stripped pinhole twin (downstream packages hand-rolled
`CameraModel(x.intrinsics, Pinhole())` in three places) and the composed
camera-space → pixel map as a value. `intrinsics`/`lens`
accessors live in cameras.jl.
"""
pinhole(model::CameraModel) = CameraModel(model.intrinsics, Pinhole())
imaging_transform(model::CameraModel) =
    imaging_transform(model.intrinsics, model.lens)

"""
    constraint_with_jacobians(model::CameraModel, pose::EuclideanMap{3}, X, u)
        -> (e::SVector{2}, G_x::SMatrix{2,2}, G_θ::SMatrix{2,6})

The reprojection constraint of a FULL camera in RAW pixels: the residual
`e = project(model, R·X + t) − u` between the observed pixel `u` and the
world point `X` imaged through the intrinsics AND the lens, its measurement
Jacobian `G_x = ∂e/∂u = −I₂` (the observed pixel is the measurement, `X` is
exact — the PnP model; the triple every `constraint_with_jacobians` returns, so a
predictive whitening reads it like any other), and its Jacobian in the
left-perturbation pose chart `δ = (δω, δt)` of the calibrated
`constraint_with_jacobians(pose, X, ray)`:

    ∂e/∂δ = ∂project/∂X_cam · [−[X_cam]×  I₃]

where the first factor is `A · transform_deriv(lens, X_cam)`, the chain rule
through the lens's ray-domain hook `project(lens, X)` — the same map the
residual goes through. One residual currency — the pixels
the detector measured — for every lens family, so a Student-t scale, a
Welsch cutoff or an inlier gate mean the same thing at the rim of a
fisheye as at its centre. For a `Pinhole` this is exactly `A · g` of the
calibrated form. Directions the lens does not image (for a gnomonic-frame
lens, every `X_cam[3] ≤ 0`) project to non-finite pixels; callers mask on
`isfinite`, as with `project`.
"""
@inline function constraint_with_jacobians(
        model::CameraModel, pose::EuclideanMap{3, Float64, <:Rotations.Rotation{3, Float64}, <:StaticVector{3, Float64}},
        X::StaticVector{3, Float64}, u::StaticVector{2, Float64}
    )
    X_cam = pose.R * X + pose.t
    p = project(model, X_cam)
    e = SVector{2, Float64}(p[1] - u[1], p[2] - u[2])
    # The chain rule through `A ∘ project(lens, ·)`, with the SAME hook the
    # residual above used. It was spelled out here as `A ∘ lens ∘ π` — which is
    # the perspective frame, so the gradient disagreed with the residual for
    # any lens that images past the horizon.
    A = SMatrix{2, 2, Float64, 4}(_pixel_affine(model.intrinsics).linear)
    Jp = A * transform_deriv(model.lens, X_cam)
    G_θ = Jp * hcat(skew(-X_cam), one(SMatrix{3, 3, Float64}))
    return (e, -one(SMatrix{2, 2, Float64}), G_θ)
end

# The lens-level maps `project(lens, X)` / `backproject(lens, xd)` and their
# Jacobians are the lens contract: geometry/cameras/lens_rays.jl.

"""
    backproject(intrinsics::AbstractIntrinsics, lens::AbstractLensModel, u::StaticVector{2,T<:Real}) -> Vec3

The pixel-frame half of back-projection: `K⁻¹` takes the pixel `u` to the
normalized distorted point, and the bearing is the lens's own
[`backproject`](@ref)`(lens, xd)` — which, not this method, is what a lens
family extends.
"""
@inline function backproject(
        intrinsics::AbstractIntrinsics, lens::AbstractLensModel,
        u::StaticVector{2, <:Real}
    )
    h = to_affine(u)
    xd = intrinsics.K_inv * h
    # The ONLY thing this layer owns is the pixel frame. The bearing itself is
    # the lens's, and it is already a unit `Vec` — a DIRECTION, not the position
    # at unit depth, which is `unproject`'s answer rather than this one.
    return backproject(lens, SVector(xd[1], xd[2]))
end

# The IsPhysical branches that used to double this dispatch table scaled the
# ray by f and then normalized the scale away — direction-identical to the
# logical path. They are gone; physical intrinsics only matter for METRIC
# unproject and the focal accessors.

# =============================================================================
# Unproject (2D + depth → 3D)
# =============================================================================

"""
    unproject(model::CameraModel, u::AbstractVector, depth::Real) -> Point3
    unproject(intrinsics, lens, u, depth) -> Point3

The camera-frame point on the ray behind pixel `u` whose Z-DEPTH is `depth` —
the convention of depth images: distance along the optical axis. It is
`b·(depth / b₃)` with `b = backproject(intrinsics, lens, u)`, for every lens,
and NaN where the ray has `b₃ ≤ 0`: a Z-depth names no point on such a ray. A
point at distance `d` ALONG any ray, past 90° included, is
`d · backproject(model, u)`.
"""
unproject(model::CameraModel, u::AbstractVector, depth::Real) =
    Point3(unproject(model.intrinsics, model.lens, u, depth))

function unproject(
        intrinsics::AbstractIntrinsics, lens::AbstractLensModel,
        u::AbstractVector, depth::Real
    )
    T = eltype(intrinsics.K)
    b = backproject(intrinsics, lens, SVector{2, float(promote_type(eltype(u), T))}(u[1], u[2]))
    b[3] > 0 || return Point3{T}(T(NaN), T(NaN), T(NaN))
    Z = T(depth)
    s = Z / T(b[3])
    # `Point`, not `Vec`: this one HAS a distance in it.
    return Point3{T}(T(b[1]) * s, T(b[2]) * s, Z)
end

# =============================================================================
# Undistortion trait — CameraModel convenience method
# =============================================================================

has_distortion(model::CameraModel) = has_distortion(model.lens)

# =============================================================================
# Image undistortion
# =============================================================================

"""
    CameraModel(m::RadialHomography) -> CameraModel{LogicalIntrinsics, Division}

Bridge a fitted uncalibrated radial model to a camera: unit-focal
intrinsics with the principal point at `m.center` and a [`Division`](@ref)
lens with `m.k` — normalized coordinates coincide with centered pixels, so
the lens coefficient IS the model's pixel-frame `k`. The resulting camera
drives the fused KA [`undistort!`](@ref) warp directly:
`undistort(CameraModel(m), img)`.
"""
CameraModel(m::RadialHomography{T}) where {T} = CameraModel(
    LogicalIntrinsics(
        CameraCalibrationMat(
            T[1 0 m.center[1]; 0 1 m.center[2]; 0 0 1]
        )
    ),
    Division(m.k)
)

# =============================================================================
# Frame of record — the intrinsics K IS the normalization frame of the lens
# =============================================================================
#
# A lens coefficient is a number about a frame: the same physical lens has
# k = −5e-7 in centered pixels and −0.6 in unit-focal normalized coordinates.
# Every estimation frame (centered pixels, a Hartley or diagonal normalization)
# is an affine map of the image plane — and so is K:
# `normalized = K⁻¹·pixel`. Hence no bare coefficients cross an API boundary:
# they travel as `CameraModel(intrinsics, lens)`, whose K is the frame of
# record — physical, or a placeholder (unit focal at the center; the image
# diagonal) for a partial calibration. The same camera under another K is a
# constructor (`CameraModel(model, K′)`) with a per-family law; its EXISTENCE
# records which families tolerate a placeholder K (scale-covariant radial
# polynomials do; angle-based models — Kannala–Brandt, DoubleSphere — need a
# metric K). Not a frame change of a map (that is `∘`): the camera's pixel map
# is unchanged, only its split into K and lens moves.

"""
    CameraModel(model::CameraModel, K′::CameraCalibrationMat) -> CameraModel

The same camera with intrinsics `K′`: the lens re-expressed under the new
intrinsics matrix sharing the principal point, so the pixel map is unchanged: `normalized′ = normalized / s` with `s = f′/f`, so a scale-covariant
radial lens transforms its coefficients by the law aᵢ → aᵢ·s^(2i),
bⱼ → bⱼ·s^(2j) (the generalized `k′ = k·s²`). Throws for a moved principal
point (the lens center is data, not a frame choice), for anisotropic
rescaling of a radial lens (radial symmetry would be broken), and for
families whose coefficients are only meaningful under a metric K.
"""
function CameraModel(model::CameraModel{<:LogicalIntrinsics}, K′::CameraCalibrationMat)
    K = model.intrinsics.K
    (
        isapprox(K[1, 3], K′[1, 3]; atol = 1.0e-9 * max(1, abs(K[1, 3]))) &&
            isapprox(K[2, 3], K′[2, 3]; atol = 1.0e-9 * max(1, abs(K[2, 3])))
    ) ||
        throw(ArgumentError("CameraModel(model, K′): the principal point must be shared — the lens center is data, not a frame choice"))
    (iszero(K[1, 2]) && iszero(K′[1, 2])) || throw(ArgumentError("CameraModel(model, K′): skewed intrinsics are not supported"))
    sx, sy = K′[1, 1] / K[1, 1], K′[2, 2] / K[2, 2]
    return CameraModel(LogicalIntrinsics(K′), _rescale_lens(model.lens, sx, sy))
end

# The scale law, leaf by leaf (radius r ↦ r/s): a radial coefficient of r^(2i)
# becomes c·s^(2i) (`rescale_radial_coeff`), the decentering deltas — degree 2
# and 2i + 1 in the point — p·s and sxᵢ·s^(2i−1). Only the perspective frame
# and the Even lift rescale with the image: an angular frame's radius is an
# angle, fixed by a metric K.
_rescale_lens(l::Lens{Perspective, Nothing}, sx, sy) = l
_rescale_lens(l::Lens{Perspective}, sx, sy) = Lens(l.projection, _rescale_node(l.distortion, _isotropic_scale(l, sx, sy)))
_rescale_lens(l::LiftLens{<:Radial{Even}}, sx, sy) = LiftLens(_rescale_node(l.radial, _isotropic_scale(l, sx, sy)))
_rescale_lens(lens::AbstractLensModel, sx, sy) = _not_scale_covariant(lens)
function _isotropic_scale(l, sx, sy)
    isapprox(sx, sy; rtol = 1.0e-12) ||
        throw(ArgumentError("CameraModel(model, K′): anisotropic rescaling ($sx, $sy) breaks the radial symmetry of $(typeof(l))"))
    T = _coeff_type(l)
    return (T === Union{} ? Float64 : T)(sx)
end
_not_scale_covariant(x) = throw(
    ArgumentError(
        "CameraModel(model, K′): $(nameof(typeof(x))) is not scale-covariant — its coefficients are " *
            "meaningful only under a metric K, so no placeholder-frame re-expression exists"
    )
)
_rescale_node(r::Radial{Even, P, Q, T}, s) where {P, Q, T} = Radial{Even, P, Q, T}(
    ntuple(i -> rescale_radial_coeff(r.k[i], s, i), Val(P)), ntuple(j -> rescale_radial_coeff(r.d[j], s, j), Val(Q))
)
_rescale_node(t::Tangential, s) = Tangential(t.p1 * s, t.p2 * s)
_rescale_node(t::ThinPrism{N, T}, s) where {N, T} =
    ThinPrism{N, T}(ntuple(i -> t.sx[i] * s^(2i - 1), Val(N)), ntuple(i -> t.sy[i] * s^(2i - 1), Val(N)))
_rescale_node(a::Additive, s) = Additive(map(t -> _rescale_node(t, s), a.terms))
_rescale_node(c::Chain, s) = Chain(map(t -> _rescale_node(t, s), c.stages))
_rescale_node(t, s) = _not_scale_covariant(t)

# One authority for "how close to the division model's horizon r_∞ may
# content sit": division_bounded (the predicate) and undistorted_rect /
# valid_rect (the clip) must agree, or the predicate can answer "bounded"
# for a frame the clip then trims. Was 0.95 vs 0.92 across six sites.
const HORIZON_MARGIN = 0.92
"""
    DEFAULT_MAX_FOV

Recommended half-angle cap for flattening a view onto a perspective plane, as
the `max_fov` of [`undistorted_rect`](@ref). 70° gives tan θ ≈ 2.75, so the
rectified plane stays a small multiple of the frame; past ~80° (tan 5.7) it
grows faster than the detail it reveals, and the last degree before 90° is
unbounded.

A RECOMMENDATION, not the default: `undistorted_rect` defaults `max_fov` to
`Inf` so that it and [`valid_rect`](@ref) stay bounded by the same rule. The
callers that want a viewing cap — `plot_rectification` — supply this.
"""
const DEFAULT_MAX_FOV = deg2rad(70.0)

"""
    division_bounded(k, center, sz; margin = HORIZON_MARGIN) -> Bool

Whether the perspective (undistorted) image of a frame of size `sz = (width, height)` is
BOUNDED under a division model `k` about `center` (all in the same pixel
units — e.g. a `RadialHomography`'s `k`/`center`). For `k < 0` the
undistorted radius `r_u = r_d/(1 + k r_d²)` diverges at the 90° circle
`r_∞ = 1/√(−k)`: content at or beyond it undistorts to infinity, and no
finite canvas holds it. True iff the farthest image corner's distorted
radius stays inside `margin·r_∞` (`k ≥ 0` is always bounded).

When true, the corner stretch factor is `1/(1 + k·r_max²)` — the canvas
scale an undistorted image needs to keep every input pixel in bounds.
"""
function division_bounded(k::Real, center, sz; margin::Real = HORIZON_MARGIN)
    width, height = sz[1], sz[2]
    k >= 0 && return true
    r_max = sqrt(
        max(
            (0 - center[1])^2, (width - center[1])^2
        ) + max(
            (0 - center[2])^2, (height - center[2])^2
        )
    )
    return r_max < margin / sqrt(-k)
end
division_bounded(m::RadialHomography, sz; kwargs...) =
    division_bounded(m.k, m.center, sz; kwargs...)

"""
    division_bounded(model::CameraModel{<:LogicalIntrinsics, <:LiftLens}, sz; margin = HORIZON_MARGIN) -> Bool

Does the whole image lie inside `margin · monotone_radius(lens)` (in the
lens's normalized distorted radius)? The general-family form of the
coefficient-level predicate above.
"""
function division_bounded(
        model::CameraModel{<:LogicalIntrinsics, <:LiftLens}, sz; margin::Real = HORIZON_MARGIN
    )
    width, height = sz[1], sz[2]
    K = model.intrinsics.K
    cx, cy = K[1, 3], K[2, 3]
    rmax = sqrt(max((0 - cx)^2, (width - cx)^2) / K[1, 1]^2 + max((0 - cy)^2, (height - cy)^2) / K[2, 2]^2)
    return rmax < margin * monotone_radius(model.lens)
end

"""
    distortion_map(model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel}) -> Transformation

Math-space transform mapping undistorted COLMAP pixel coordinates to
distorted ones: `pixel → K⁻¹ → lens → K → pixel`.  Spelled as a plain
composition (`AffineMap(K) ∘ lens ∘ AffineMap(K⁻¹)`) so `image_warp`'s
`fuse` merges the calibration affines into the COLMAP/Julia index bridges —
the fused per-pixel path is one affine, the lens polynomial, one affine.
"""
function distortion_map(model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel})
    T = eltype(model.intrinsics.K)
    return AffineMap(SMatrix{3, 3, T}(model.intrinsics.K)) ∘ model.lens ∘
        AffineMap(SMatrix{3, 3, T}(model.intrinsics.K_inv))
end

"""
    undistort(model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel}, img)

Remove lens distortion from an image.

For each output (undistorted) pixel, computes the corresponding source
(distorted) pixel through `model.lens` and resamples. Uses VGC's
`image_warp` builder to bake pixel coordinate conventions into a
`BoundedMap` (K is assumed to be in COLMAP convention: image corner at
origin, pixel centers at 0.5).

Each output pixel is sampled through [`project`](@ref), which NaNs a ray past
a radial factor's fold (the pixel takes the fill value) but does not test the
other folds: past a fold of an [`Additive`](@ref)'s deltas or a
[`Chain`](@ref)'s [`Tilt`](@ref) the pixel samples a finite source pixel whose
[`backproject`](@ref) is a different ray, so the output repeats content of
the imaged branch there. Where the lens overlaps itself over the sensor, no
undistortion is one-to-one, and a map built from `backproject` may jump
between preimages (see [`backproject`](@ref)): injectivity is checked on the
lens over the sensor rectangle, not per pixel.

# Arguments
- `model::CameraModel`: Camera model (any lens family)
- `img`: Input distorted image (any AbstractMatrix — Gray, RGB, etc.)

# Returns
Undistorted image of the same size and element type.

# Example
```julia
model = CameraModel(; focal=2458.0, pp=(2099.0, 1494.0),
                      lens=Lens(Perspective(), Additive(Radial((-0.159, 0.048)), Tangential(0.006, -0.001))))
img_clean = undistort(model, img)
```
"""
undistort(model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel}, img::AbstractMatrix) =
    undistort!(similar(img), model, img)
# "No camera" is the identity: BlobBoards' pipeline (_undistort_adapt) supplies
# `nothing` when a stage has no calibration. Consumed downstream; keep.
undistort(::Nothing, img::AbstractMatrix) = img

# No-op for distortion-free lenses and missing camera
undistort(::CameraModel{<:AbstractIntrinsics, Pinhole}, img::AbstractMatrix) = img
"""
    undistort!(dst, [T = Float64], model, src; fillvalue = zero(eltype(dst))) -> dst

In-place [`undistort`](@ref): resample `src` into the pre-allocated `dst`
through the fused lens warp, on the backend that owns `dst` (any
KernelAbstractions backend — supply `CuArray`s for GPU execution).  `dst` and
`src` may differ in eltype (see [`image_warp!`](@ref)).  `T` sets the warp
math's float type: `undistort!(dst, Float32, model, src)` keeps the per-pixel
math single-precision for GPU throughput. The order is Julia Base's: the
mutated `dst`, then the type, then the inputs.
"""
undistort!(
    dst::AbstractMatrix, model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel},
    src::AbstractMatrix; kwargs...
) = undistort!(dst, Float64, model, src; kwargs...)

function undistort!(
        dst::AbstractMatrix, ::Type{T},
        model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel}, src::AbstractMatrix;
        kwargs...
    ) where {T <: AbstractFloat}
    # Same-size form = the rect form at unit scale over the frame (the
    # identity placement fuses exactly, so this is the historical warp).
    return undistort!(dst, T, model, src, Rect(0, 0, size(dst, 2), size(dst, 1)); kwargs...)
end

"""
    undistort!(dst, [T = Float64], model, src, rect; fillvalue = zero(eltype(dst))) -> dst

GENERAL form: resample the undistorted-plane window `rect` (COLMAP pixel
coordinates) into `dst`. `dst`'s size sets the sampling density — supply a
smaller `dst` to downscale; the scale and offset fold into the fused warp
affines. The rect-free methods above render `Rect(0, 0, w, h)` at unit
scale (the classic same-size undistortion, which CROPS any content the
undistortion pushes outside the frame); pair this form with
[`undistorted_rect`](@ref) to render the full extent instead.
"""
undistort!(
    dst::AbstractMatrix, model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel},
    src::AbstractMatrix, rect::GeometryBasics.HyperRectangle{2}; kwargs...
) = undistort!(dst, Float64, model, src, rect; kwargs...)

function undistort!(
        dst::AbstractMatrix, ::Type{T},
        model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel},
        src::AbstractMatrix, rect::GeometryBasics.HyperRectangle{2};
        fillvalue = zero(eltype(dst))
    ) where {T <: AbstractFloat}
    Hout, Wout = size(dst)
    o = GeometryBasics.origin(rect); wd = GeometryBasics.widths(rect)
    place = AffineMap(
        SMatrix{2, 2, Float64}(wd[1] / Wout, 0, 0, wd[2] / Hout),
        SVector(Float64(o[1]), Float64(o[2]))
    )
    bm = image_warp(T, distortion_map(model) ∘ place, Rect(0, 0, Wout, Hout))
    return image_warp!(dst, bm, src; fillvalue)
end

"""
    undistort(model, img, rect; scale = 1, fillvalue = zero(eltype(img))) -> FramedImage

Allocating rect form of [`undistort!`](@ref): the canvas is sized
`ceil.(widths(rect) ./ scale)`, `rect` is resampled into it, and the result
is returned WITH its placement as a [`FramedImage`](@ref) — `rect` is the
window of the camera's pixel frame the canvas covers, so
`imshow(pos, undistort(model, img, ext; scale))` is self-placing and the
`ext`/`scale` pair never travels beside the pixels by hand. The in-place
[`undistort!`](@ref) forms are unchanged (the caller owns `dst` and its
placement); the rect-free same-size forms return bare matrices (their
placement is trivially the frame).
"""
function undistort(
        model::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel},
        img::AbstractMatrix,
        rect::GeometryBasics.HyperRectangle{2}; scale::Real = 1,
        fillvalue = zero(eltype(img))
    )
    wd = GeometryBasics.widths(rect)
    # The destination FOLLOWS the source's backend (similar, not zeros):
    # CuArrays in, CuArray FramedImage out — the fused KA warp runs on the
    # device (rectify's convention, applied here too).
    #
    # NOT pre-filled: the warp writes EVERY destination pixel, so a canvas
    # painted here is discarded. Out-of-source taps come from the kernel's
    # `fillvalue`, which is why that had to be a keyword rather than a `fill!`.
    dst = similar(img, ceil(Int, wd[2] / scale), ceil(Int, wd[1] / scale))
    return FramedImage(undistort!(dst, model, img, rect; fillvalue), rect)
end

"""
    undistorted_rect(m::RadialHomography, (width, height); margin = HORIZON_MARGIN, max_fov = Inf) -> Rect

The undistorted-plane extent of a `width`×`height` frame under `m`'s
division model: the frame boundary undistorted in closed form
(`u = c + (d − c)/(1 + k‖d − c‖²)`), with boundary radii at or beyond
`margin·r_∞` (the 90° divergence circle, `r_∞ = 1/√(−k)`) clipped to it —
the honest cap when [`division_bounded`](@ref) is false and the true
extent is infinite. Feed the rect to the rect form of
[`undistort`](@ref)/[`undistort!`](@ref):

    ext = undistorted_rect(m, (w, h))
    canvas = undistort(CameraModel(m), img, ext;
                       scale = max(1, GeometryBasics.widths(ext)[1] / 5000))

`max_fov` (radians) additionally bounds the extent by FIELD OF VIEW, capping it
at the bounding square of that cone about the principal point. `Inf`, the
default, leaves the frame's own field alone; a finite value is what makes a
fisheye — whose horizon can sit inside the frame — rectify to a usable plane at
all, since `tan θ` is unbounded in the last degree before the pole.
[`DEFAULT_MAX_FOV`](@ref) is the recommended cap.
"""
undistorted_rect(m::RadialHomography{Float64}, sz; margin::Real = HORIZON_MARGIN, max_fov::Real = Inf) =
    undistorted_rect(CameraModel(m), sz; margin, max_fov)

# ONE authority for the frame-boundary walk both rect builders share: the
# closed-form undistortion of a boundary pixel, with radii at or beyond the
# horizon clip landing AT the clip (the honest cap when the true landing is
# at infinity). Returns the undistorted PIXEL for a distorted boundary pixel.
# FAMILY-GENERIC, through the lens contract. This used to call
# `radial_ratio`, which only the rational lift answered, so the two rect
# functions below — and with them `plot_rectification`, the panel documented
# as the way to judge one fit against another — worked for the division family
# alone and raised a MethodError on Brown–Conrady, Kannala–Brandt, double
# sphere, Scaramuzza and the perspective fisheyes. `undistort_normalized` is
# the same quantity every lens already provides (`x_u` for `x_d`); on the
# division family the two agree to 4 ulps (measured 2026-09-16 over three
# lenses and 2000 radii: 8.2e-16 relative), which is rounding, not a change of
# answer. NaN where the lens does not image the sample — see the callers.
@inline function _boundary_undistort(lens, fx, fy, cx, cy, rclip, d::SVector{2, Float64})
    n = SVector((d[1] - cx) / fx, (d[2] - cy) / fy)
    ρ = hypot(n[1], n[2])
    # Past the clip, the sample is taken ON the clip circle in the same
    # direction — the honest cap when the true landing is at infinity.
    nc = ρ >= rclip ? n * (rclip / max(ρ, eps())) : n
    nu = undistort_normalized(lens, nc)
    return SVector(cx + fx * nu[1], cy + fy * nu[2])
end

# THE CLIP RADIUS of a rect walk: the normalized distorted radius past which a
# boundary sample is taken on the clip circle instead of where it really lands.
#
# `monotone_radius` is it whenever the lens folds — and for a division lens with
# a pole the two coincide exactly, because that pole IS the 90° circle (measured:
# 1.8257418583502216 vs 1.8257418583505538 for `Division(-0.3)`). A lens that
# never folds has no such radius, and `monotone_radius` says `Inf`; for those the
# horizon is the bound, since the perspective view diverges as the ray approaches
# 90°. Without it a fisheye's rect was decided by whichever boundary sample
# happened to land nearest the horizon — an equidistant fisheye at f = 400 over
# 1280×960 returned a 112575×134216 px rect, a number with no meaning.
@inline function _rect_clip_radius(lens, margin::Real)
    r = Float64(monotone_radius(lens))
    isfinite(r) && return margin * r
    h = project(lens, SVector(1.0, 0.0, 0.0))        # a ray at exactly 90°
    r90 = hypot(Float64(h[1]), Float64(h[2]))
    return isfinite(r90) ? margin * r90 : Inf        # neither: nothing to clip at
end

# THE frame-boundary walk both rect builders share: the frame's edges on a
# closed 400-sample grid (`rect_edges`, corners included — `valid_rect` takes
# per-edge extrema over them) and the setup: the frame size, the principal point
# and the undistortion of a boundary pixel (`_boundary_undistort` at this
# model's clip radius).
_frame_edges(W::Float64, H::Float64) = rect_edges(Rect(0.0, 0.0, W, H), EdgeGrid{Closed}(400))
function _frame_walk(model::CameraModel, sz, margin::Real)
    K = model.intrinsics.K
    fx, fy, cx, cy = Float64(K[1, 1]), Float64(K[2, 2]), Float64(K[1, 3]), Float64(K[2, 3])
    lens = _precision(Float64, model.lens)
    rclip = _rect_clip_radius(lens, margin)     # normalized distorted radius
    return Float64(sz[1]), Float64(sz[2]), cx, cy, d -> _boundary_undistort(lens, fx, fy, cx, cy, rclip, d)
end

function undistorted_rect(
        model::CameraModel{<:LogicalIntrinsics, <:AbstractLensModel}, sz;
        margin::Real = HORIZON_MARGIN, max_fov::Real = Inf
    )
    Wimg, Himg, cx, cy, undistort_boundary = _frame_walk(model, sz, margin)
    xmin = ymin = Inf; xmax = ymax = -Inf
    n_imaged = 0
    for edge in _frame_edges(Wimg, Himg), d in edge
        u = undistort_boundary(d)
        # A SAMPLE THE LENS DOES NOT IMAGE contributes nothing. The clip keeps
        # a family with a fold inside its monotone branch, so this only fires
        # for a lens whose horizon lies within the frame and whose
        # `monotone_radius` is `Inf` — a view with no finite perspective rect
        # at all, which the check below reports rather than returning NaN.
        all(isfinite, u) || continue
        n_imaged += 1
        xmin = min(xmin, u[1]); xmax = max(xmax, u[1])
        ymin = min(ymin, u[2]); ymax = max(ymax, u[2])
    end
    # Fewer than two imaged samples means the walk bounds NOTHING — the lens's
    # horizon lies inside the frame. The accumulators are the empty interval
    # `[Inf, -Inf]` there, so widen them to the whole plane and let the cap, if
    # there is one, be the entire answer; with no cap the emptiness check below
    # reports it rather than returning a NaN rect.
    if n_imaged < 2
        xmin, ymin, xmax, ymax = -Inf, -Inf, Inf, Inf
    end
    # FIELD-OF-VIEW CAP. `margin` clips a fraction of the DISTORTED radius,
    # which is nearly linear in the view angle — so on a fisheye even 0.92·r∞
    # sits ~0.3° from the pole and `tan θ` sends the flattened plane to
    # hundreds of thousands of pixels (measured: 369 441 px for a 4256 px
    # frame, an 88x downscale that leaves a 48 px thumbnail). The angle is
    # what has to be bounded, so the extent is intersected with the bounding
    # square of the `max_fov` cone about the principal point. OPT-IN — `Inf`,
    # the default, leaves the frame's own field alone, so that `valid_rect`
    # (which has no cap) stays inscribed in this rect as the contract tests
    # assert. The viewing policy lives with the caller that wants it.
    if isfinite(max_fov)
        rcap = fov_radius(model, max_fov)
        xmin = max(xmin, cx - rcap); xmax = min(xmax, cx + rcap)
        ymin = max(ymin, cy - rcap); ymax = min(ymax, cy + rcap)
    end
    (isfinite(xmax - xmin) && isfinite(ymax - ymin) && xmax > xmin && ymax > ymin) || throw(
        ArgumentError(
            "undistorted_rect: no finite extent — the lens images $(n_imaged) of the frame's " *
                "boundary samples and max_fov = $(rad2deg(max_fov))° bounds nothing that is " *
                "left (lens $(nameof(typeof(model.lens))), margin = $margin). Supply a finite " *
                "`max_fov` to bound the extent by field of view instead of by the frame."
        )
    )
    return Rect(xmin, ymin, xmax - xmin, ymax - ymin)
end

"""
    horizon_radius(model) -> Float64

The PIXEL radius of the camera's horizon: the circle at which rays reach the
lens's monotone limit, beyond which nothing can be placed on a perspective
plane at all. `Inf` for a lens with no such limit (a pinhole, a mild
polynomial). This is the circle `plot_rectification` draws on the raw panel,
and the guide for how much of a frame it is meaningful to rectify.

It is NOT a safe rectification bound by itself. The distorted radius is very
nearly linear in the view angle, while the perspective magnification is
`tan θ`, so a generous FRACTION of this radius is still a fraction of a degree
from the pole: for a 927 px fisheye with a 1419 px horizon, 92% of it is
θ ≈ 89.7°, `tan θ ≈ 199`, and the rectified plane comes out 369 000 px wide.
Bound the ANGLE instead — see `undistorted_rect`'s `max_fov`.
"""
horizon_radius(model::CameraModel) =
    Float64(calibration_matrix(model)[1, 1]) * Float64(monotone_radius(lens(model)))

"""
    fov_radius(model, θ) -> Float64

The UNDISTORTED (perspective-plane) pixel radius of the cone of half-angle
`θ` radians: `f·tan(θ)`. The companion of [`horizon_radius`](@ref) — one says
where the lens stops imaging, this says how big a chosen field of view is once
flattened.
"""
fov_radius(model::CameraModel, θ::Real) =
    Float64(calibration_matrix(model)[1, 1]) * tan(Float64(θ))

"""
    valid_rect(m::RadialHomography, (width, height); margin = HORIZON_MARGIN) -> Rect
    valid_rect(model::CameraModel{<:LogicalIntrinsics, <:AbstractLensModel}, sz;
               margin = HORIZON_MARGIN) -> Rect

The INSCRIBED counterpart of [`undistorted_rect`](@ref) — OpenCV's
`alpha = 0` endpoint: the largest axis-aligned window of the undistorted
plane containing NO invalid pixels, so
`undistort(model, img, valid_rect(model, size))` renders with no fill —
the no-black-borders mode. Same frame-boundary walk as
[`undistorted_rect`](@ref) (one authority, `_boundary_undistort`), but
taking per-edge extrema INWARD — the left constraint is the maximum x over
the undistorted left edge, and so on — instead of the global bounding box
(sampled, like the outer rect; corners are the edge-sample endpoints, so
both bow directions are covered).

Boundary radii at or beyond `margin·r_∞` clip to the horizon exactly as in
`undistorted_rect`: when the whole horizon circle lies inside the frame the
valid region is unbounded and the clip is the honest cap. Throws when the
inscribed window is empty (boundary folded past the principal point).
"""
function valid_rect(
        model::CameraModel{<:LogicalIntrinsics, <:AbstractLensModel}, sz; margin::Real = HORIZON_MARGIN
    )
    Wimg, Himg, _, _, undistort_boundary = _frame_walk(model, sz, margin)
    xlo = ylo = -Inf; xhi = yhi = Inf
    # An unimaged sample constrains nothing (see `undistorted_rect`); the
    # emptiness check below is what reports a boundary that never resolved.
    edge(d, k, f, acc) = (u = undistort_boundary(d); isfinite(u[k]) ? f(acc, u[k]) : acc)
    bottom, right, top, left = _frame_edges(Wimg, Himg)   # y = 0, x = W, y = H, x = 0
    for d in left; xlo = edge(d, 1, max, xlo); end
    for d in right; xhi = edge(d, 1, min, xhi); end
    for d in bottom; ylo = edge(d, 2, max, ylo); end
    for d in top; yhi = edge(d, 2, min, yhi); end
    all(isfinite, (xlo, xhi, ylo, yhi)) || throw(
        ArgumentError(
            "valid_rect: the lens images none of one frame edge — its horizon lies inside the " *
                "frame (lens $(nameof(typeof(model.lens))), margin = $margin)"
        )
    )
    (xhi > xlo && yhi > ylo) || throw(
        ArgumentError(
            "valid_rect: empty inscribed window — the undistorted frame boundary " *
                "folds past the principal point (margin = $margin)"
        )
    )
    return Rect(xlo, ylo, xhi - xlo, yhi - ylo)
end
valid_rect(m::RadialHomography{Float64}, sz; margin::Real = HORIZON_MARGIN) =
    valid_rect(CameraModel(m), sz; margin)

"""
    pnp_pixel_whitening(proj) -> SMatrix{2,2,Float64,4}

The frozen pixel whitening of the PnP lift: the constant 2×2 calibration
block `A = K[1:2,1:2]` of a Pinhole imaging transform (`AffineMap(K) ∘
Pinhole()`), so that `A·g` (with `g` the calibrated constraint of
`constraint_with_jacobians`) is exactly the pixel reprojection residual the
scoring pipeline measures. Errors for non-affine (distortion-lens) imaging
transforms — the lift's one-currency contract requires an affine pixel map.
Downstream (VisualGeometryToolkitEstimation) adds the problem-level accessor method.
"""
pnp_pixel_whitening(proj::CoordinateTransformations.ComposedTransformation{<:AffineMap, Pinhole}) =
    SMatrix{2, 2, Float64, 4}(proj.t1.linear)
pnp_pixel_whitening(proj) = error(
    "The PnP loss lift requires a Pinhole camera (affine pixel map " *
        "AffineMap(K) ∘ Pinhole()); got $(typeof(proj)). Undistort the " *
        "measurements to a pinhole model, or use an IRLS local optimization."
)

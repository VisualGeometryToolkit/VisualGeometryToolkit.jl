# ==============================================================================
# Homography - 2D projective transformations for planar scenes
# ==============================================================================

# Note: HomographyMat is defined in geometry/transforms/matrices.jl as part of
# the transform hierarchy.  `Base.inv(::AbstractHomMat)` (in matrices.jl)
# already returns the same leaf type via `similar_type`, so no explicit
# `inv(::HomographyMat)` override is needed.

"""
    HomographyMat(camera::PinholeCamera) -> HomographyMat{Float64}

Compute the homography H that maps from a planar board (z=0 plane) to the image.

Only valid for pinhole cameras. For points on the z=0 plane, the homography is:

    H = K [r1 r2 t]

where K is intrinsics, r1 and r2 are the first two columns of the rotation matrix R,
and t is the translation from camera extrinsics.

# Arguments
- `camera::PinholeCamera`: Camera with pinhole projection

# Returns
- `HomographyMat{Float64}`: Homography matrix (board coords → image coords)

# Example
```julia
camera = Camera(model, extrinsics)
H = HomographyMat(camera)
H_inv = inv(H)  # For warping: image → board
```
"""
HomographyMat(camera::PinholeCamera) = _induced_plane_homography(camera)
# a physical pinhole: the plane coordinates have no unit — refused, naming the fix
const _UnitfulPinholeCamera = Camera{<:CameraModel{<:AbstractIntrinsics, Pinhole}, <:UnitfulEuclideanMap{3}}
HomographyMat(::_UnitfulPinholeCamera) = _refuse_physical_camera("HomographyMat(camera)")
HomographyMat(::_UnitfulPinholeCamera, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
    _refuse_physical_camera("HomographyMat(camera, origin, u, v)")

"""
    Extrinsics3(H::HomographyMat, intrinsics::AbstractIntrinsics) -> Extrinsics3

The pose of a PLANAR target from its plane → image homography — the inverse of
[`HomographyMat`](@ref)`(camera)`: `H ∝ K·[r₁ r₂ t]` maps the target's `z = 0`
plane (its own units) to the pixels of `intrinsics` (a pinhole calibration in
the same pixels: frame in = frame out, nothing normalized). Returns the
world(target) → camera map `x_cam = R·X + t`, `t` in the target's units.

`B = K⁻¹H` is scaled by the mean norm of its first two columns and signed so
the target lies in front of the camera (`t₃ > 0`); `R` is the proper rotation
nearest (Frobenius) to `[r₁ r₂ r₁×r₂]`. An IDENTITY for an exact `H` — no
correspondences, no refinement; for an estimated `H` it is the closed-form
pose its planar model implies ([Zhang2000](@citet), §3.1).
"""
function Extrinsics3(H::HomographyMat, intrinsics::AbstractIntrinsics)
    K = SMatrix{3, 3, Float64, 9}(intrinsics.K)
    B = K \ SMatrix{3, 3, Float64, 9}(H)
    λ = 2 / (norm(B[:, 1]) + norm(B[:, 2]))
    B[3, 3] < 0 && (λ = -λ)
    r1, r2, t = λ * B[:, 1], λ * B[:, 2], λ * B[:, 3]
    F = svd(hcat(r1, r2, cross(r1, r2)))
    R = F.U * F.Vt
    det(R) < 0 && (R = F.U * SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, 1, 0, 0, 0, -1) * F.Vt)
    return Extrinsics3(RotMatrix{3, Float64}(SMatrix{3, 3, Float64, 9}(R)), Vec3{Float64}(t))
end

# H = K·[r₁ r₂ t] for ANY parametric camera, lens ignored.
#
# For a PINHOLE camera this IS the plane → image map, which is why the public
# constructor above is Pinhole-only: the plane map of a distorted camera is not
# a homography, and offering one would be a lie about the geometry. The other
# caller is `RadialCamera(::Camera)` (geometry/cameras/calibration.jl), where
# the lens genuinely drops out — rows 1–2 of this matrix about the principal
# point are the board's radial camera whatever the lens, because a radially
# symmetric distortion about that point cannot move an image point off its own
# ray from it.
#
# The plane `(origin, u, v)` in world coordinates (default: the z = 0 board
# plane, `origin = 0`, `u = e₁`, `v = e₂`): H = K·[R·u  R·v  R·origin + t].
function _induced_plane_homography(
        camera::Camera{<:CameraModel}, origin::AbstractVector = Point3{Float64}(0, 0, 0),
        u::AbstractVector = Vec3{Float64}(1, 0, 0), v::AbstractVector = Vec3{Float64}(0, 1, 0)
    )
    K = camera.model.intrinsics.K
    R = camera.extrinsics.R
    t = camera.extrinsics.t
    uu = Vec3{Float64}(u...)        # plane basis vectors (free vectors)
    vv = Vec3{Float64}(v...)
    oo = Point3{Float64}(origin...)  # plane origin (position)
    return HomographyMat{Float64}(K * hcat(R * uu, R * vv, R * oo + t))
end

"""
    HomographyMat(camera::PinholeCamera,
                  origin::AbstractVector,
                  u::AbstractVector,
                  v::AbstractVector) -> HomographyMat{Float64}

Compute the homography H that maps from an arbitrary world plane to the
image, where the plane is parameterized by a frame `(u, v)` centered at
`origin` (all in world coordinates).

    H = K · [R·u  R·v  R·origin + t]

A point (s, t, 1) in the plane-local frame corresponds to world point
`origin + s·u + t·v`, and its image is `H · (s, t, 1)` in homogeneous
coordinates. Only valid for pinhole cameras.
"""
function HomographyMat(
        camera::PinholeCamera,
        origin::AbstractVector,
        u::AbstractVector,
        v::AbstractVector
    )
    return _induced_plane_homography(camera, origin, u, v)
end

# =============================================================================
# Labeled Mat → Map lifts — the induced homographies know their frames
# =============================================================================

"""
    HomographyMap(camera::PinholeCamera) -> HomographyMap

The plane-induced homography of `camera` as a LABELED map:
`Src = TargetPlaneFrame, Dst = PixelFrame{:colmap}` (monogram `H_PB` — consumes
target-plane-frame coordinates, produces COLMAP pixel coordinates). The matrix is
`HomographyMat(camera)` (`H = K [r₁ r₂ t]`, the `z = 0` board plane); the
frames are knowable here because the constructor fixes both endpoints: the
source IS the target-plane frame of the `z = 0` world plane, and VGC intrinsics
produce COLMAP pixels (the package-wide invariant).

```julia
H = HomographyMap(camera)          # TargetPlaneFrame → PixelFrame{:colmap}
u = H(Point2(s, t))                # target-plane frame → homogeneous pixel
H_warp = inv(H)                    # PixelFrame{:colmap} → TargetPlaneFrame
```
"""
HomographyMap(camera::PinholeCamera) =
    HomographyMap{TargetPlaneFrame, PixelFrame{:colmap}}(HomographyMat(camera))

"""
    HomographyMap(camera::PinholeCamera, origin, u, v) -> HomographyMap

The homography induced by the arbitrary world plane `(origin, u, v)`, as a
partially labeled map: `Dst = PixelFrame{:colmap}` (VGC intrinsics produce
COLMAP pixels), but `Src = UnlabeledFrame{2}` — the plane-local frame is
parameterized by the CALLER's choice of `origin`/`u`/`v`, so no VGC frame
tag can name it. Callers that maintain their own frame tag for the plane
should [`relabel`](@ref) the source:

```julia
struct FacadeFrame <: Frame{2} end
H = relabel(FacadeFrame, PixelFrame{:colmap}, HomographyMap(camera, o, u, v))
```
"""
HomographyMap(
    camera::PinholeCamera,
    origin::AbstractVector, u::AbstractVector, v::AbstractVector
) = relabel(
    UnlabeledFrame{2}, PixelFrame{:colmap},
    HomographyMap(HomographyMat(camera, origin, u, v))
)

# =============================================================================
# project — Camera projection of geometric primitives on the z=0 plane
# =============================================================================
#
# Dispatch on lens model:
#   Pinhole          → exact via induced homography H = K[r₁ r₂ t]
#   AbstractLensModel → approximate via point sampling through full model
#
# Pinhole preserves conics exactly (projective geometry). Distorting lenses
# (distorted, fisheye) have no closed-form conic image, so we sample
# boundary points and project each through the full camera model.
# =============================================================================

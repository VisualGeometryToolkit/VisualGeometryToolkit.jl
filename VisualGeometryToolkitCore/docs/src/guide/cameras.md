# Camera Models

```@meta
CurrentModule = VisualGeometryToolkitCore
```

VisualGeometryToolkitCore provides a composable camera system with physical units
throughout.

## Camera Model Components

A camera is assembled from three components:

1. **Intrinsics** — the calibration matrix `K`
   ([`LogicalIntrinsics`](@ref); pixel units)
2. **Lens model** — a map between rays and normalized image points
   ([`Pinhole`](@ref), any [`Lens`](@ref) or [`LiftLens`](@ref); see
   [Lens Families](lenses.md))
3. **Extrinsics** — the camera pose in world coordinates, as a
   frame-labeled [`EuclideanMap`](@ref) (see
   [Coordinate Frames](frames.md))

[`CameraModel`](@ref) pairs intrinsics with a lens;
[`PosedCamera`](@ref VisualGeometryToolkitCore.Cameras.PosedCamera) pairs a model with a
pose.

Every camera — a model, a posed camera, a [`BoundedCamera`](@ref) of either —
is an [`AbstractCamera`](@ref). Whether it carries a pose is the trait
[`is_posed`](@ref VisualGeometryToolkitCore.is_posed): `false` for a
model, `true` for a `PosedCamera`, the wrapped camera's for a `BoundedCamera`. The
posed-only verbs (`pose`, `extrinsics`, rig membership) refuse an unposed camera
with an `ArgumentError`, and `PosedCamera(model, X)` refuses a model that is already
posed or bounded — the bound goes outside the pose,
`BoundedCamera(PosedCamera(model, X), extent)`. The trait is shared with the
calibration targets: a [`PosedTarget`](@ref) is posed, a `PlanarTarget` or
`PointTarget` is not, and `model`, `pose` and `extrinsics` answer on a posed
member of either kind.

## Quick Example

```julia
using VisualGeometryToolkitCore

# Select a sensor from the built-in database
s = sensor("Sony", "IMX174")

# Focal length from field of view
f = focal_length_for_fov(s, deg2rad(40.0); dimension=:horizontal)

# Principal point at the image center (Rect(s) is COLMAP convention)
pp = centroid(Rect(s)) .* px

# Camera model (intrinsics + lens)
model = CameraModel(f, s.pitch, pp)

# Posed camera (extrinsics)
camera = PosedCamera(model, lookat(SVector(0.0, 0.0, 500.0),
                              SVector(0.0, 0.0, 0.0),
                              SVector(0.0, -1.0, 0.0)))
```

## Intrinsics

[`LogicalIntrinsics`](@ref) stores the calibration matrix `K` (and its
inverse) in pixel units:

```julia
intrinsics = LogicalIntrinsics(1000.0px, Point2(960.0px, 540.0px))
```

Physical quantities enter through the constructors — a focal length in
`mm` plus a pixel pitch produce the equivalent logical intrinsics
(`CameraModel(f, pitch, pp)`); physical scale lives with the [`Sensor`](@ref),
not redundantly in the intrinsics.

### Built-in Sensor Database

```julia
s = sensor("Sony", "IMX174")      # look up by manufacturer and model
s.resolution                      # Size2 in px
s.pitch                           # Size2 in μm/px
```

See [`CMOS_SENSORS`](@ref) for the registry.

## Lens Models

A lens model is a map between rays and normalized image points:
[`project`](@ref)`(lens, X)` and [`backproject`](@ref)`(lens, x_d)`, valid
past 90° wherever the lens images. [`has_distortion`](@ref) classifies the
models; [`distortion_frame`](@ref) names each model's defining ray frame
(see [The Framed Imaging Chain](imaging-chain.md)).

There are two kinds of lens, built from a handful of parts — see
[Lens Families](lenses.md):

- [`Lens`](@ref)`(projection, distortion)` — a projection onto a frame
  ([`Perspective`](@ref), [`Equidistant`](@ref), [`Stereographic`](@ref),
  [`Equisolid`](@ref), [`Orthographic`](@ref), or the parametric
  [`DoubleSphere`](@ref), [`Unified`](@ref), [`ExtendedUnified`](@ref))
  followed by a distortion in that frame ([`Radial`](@ref),
  [`Tangential`](@ref), [`ThinPrism`](@ref), [`Tilt`](@ref),
  [`FieldOfView`](@ref), combined by [`Additive`](@ref) and [`Chain`](@ref)).
  [`Pinhole`](@ref) and [`KannalaBrandt`](@ref) are named instances.
- [`LiftLens`](@ref)`(radial)` — the image point lifts to its ray in closed
  form: [`Division`](@ref), BabelCalib's rational model,
  [`Scaramuzza`](@ref).

Every other lens map — the perspective views, Jacobians, the radial
profile, undistortion — is derived from that pair; see
[Back-projection and Undistortion](undistortion.md).

## Projection and Backprojection

```julia
u = project(camera, X_world)          # 3D world point → pixel
ray = backproject(model, u)           # pixel → unit ray (camera frame)
X = unproject(model, u, depth)        # pixel + depth → 3D point
```

[`project`](@ref) composes extrinsics with the model's imaging map;
[`backproject`](@ref) and [`unproject`](@ref) are the sections of the
projection, on the ray domain of the lens (no `tan θ` blow-up at the
equator; NaN where the lens does not image, or where a Z-depth names no point). [`transfer`](@ref) moves
depth-attributed points between views (see
[Stereo Rigs and Uncertainty](stereo.md)).

The camera's ray ↦ pixel map is available in two forms: the frame-free
[`imaging_transform`](@ref), `AffineMap(K) ∘ lens`
with the lens applied to the ray (so it is `project(model, X)` for every ray),
and the fully-typed [`imaging_chain`](@ref) with every junction frame-checked —
see [The Framed Imaging Chain](imaging-chain.md).

## Camera Pose

Camera poses are frame-labeled rigid maps: [`CameraPose`](@ref) (`X_WC`,
camera-to-world) and [`CameraExtrinsics`](@ref) (`X_CW`, world-to-camera)
are frame-pair aliases of `EuclideanMap`, and `inv` converts between them
by swapping the labels. `PosedCamera(model, pose)` and
`PosedCamera(model, extrinsics)` both work — the constructor normalizes to the
extrinsics slot, so handing a pose where extrinsics are expected is
unrepresentable.

```julia
extrinsics = lookat(eye, target, up)      # world → camera EuclideanMap
p = lookat_pose(eye, target, up)          # the inverse convention
camera = PosedCamera(model, extrinsics)
pose(camera)                              # CameraPose (X_WC)
```

## P3P Pose Estimation

```julia
# 3D world points and 2D image points
csp = csponds(X, u)               # StructArray of Pt3-to-Pt2 correspondences

result = p3p(model, csp)          # FixedModels{4, Extrinsics3}, empty when no solution
for ext in result                 # 0–4 Extrinsics3 solutions
    camera_i = PosedCamera(model, ext)
    # score by reprojection error ...
end
```

The solver currency is the labeled [`Extrinsics3`](@ref) alias
(world-to-camera, a dispatch fact). [`sample_p3p`](@ref) draws minimal
samples; the `P3P` solver type plugs into the
[solver layer](../api/matching.md).

## Homographies

For a camera viewing the plane `z = 0`, the plane-induced homography is
minted with frame labels (`TargetPlaneFrame → PixelFrame{:colmap}`):

```julia
H = HomographyMap(HomographyMat(camera))
u = H(Point2(100.0, 50.0))        # board point → pixel
Q = H(circle_world)               # a Conic: conics transform contravariantly
```

## See Also

- [Cameras API](../api/cameras.md)
- [The Framed Imaging Chain](imaging-chain.md)
- [Back-projection and Undistortion](undistortion.md)
- [Stereo Rigs and Uncertainty](stereo.md)

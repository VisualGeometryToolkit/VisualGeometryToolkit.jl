# Refiners

The four refiners run PoseLib's robust Levenberg–Marquardt (LM) optimization
from a seed model. In the stack's naming, a `refine_*` function needs a
starting guess and returns a model of the same kind. It does not sample
hypotheses or decide inliers on its own; that is the job of a random sample
consensus (RANSAC) estimator, which may call a refiner as its local
optimization.

| Refiner | Seed | Correspondences | Residual |
|---|---|---|---|
| [`refine_homography`](@ref)`(H, cs)` | `HomographyMat` | 2D `x1 => x2` | symmetric transfer error |
| [`refine_fundamental`](@ref)`(F, cs)` | `FundMat` | 2D `x1 => x2` | Sampson error |
| [`refine_relpose`](@ref)`(pose, K1, K2, cs)` | [`Rigid3`](@ref PoseLib.Rigid3) | pixel points `x1 => x2` | tangent Sampson error |
| [`refine_relpose`](@ref)`(pose, cs)` | `Rigid3` | unit bearings `f1 => f2` with Jacobians | tangent Sampson error |
| [`refine_absolute_pose`](@ref)`(pose, K, cs)` | `Rigid3` | 3D to pixel `X => u` | reprojection error |

Every refiner takes three keywords:

- `max_iter`, the iteration budget. It has no default; the caller chooses it.
- `loss`, the robust loss. The default is `TLSShape()`, a truncated quadratic.
- `loss_width`, the width of the loss, in the unit of the residual.

## Losses

A loss is one of VisualGeometryToolkitCore's loss shapes. Each is parametrized
by one width `w`, and PoseLib's own `loss_scale` parameter equals that width
for every loss bound here. The width reaches PoseLib unchanged.

| `loss` | `ρ(r)` | |
|---|---|---|
| `TLSShape()` (default) | `min(r², w²)` | inliers have `r < w`; outliers get zero weight |
| `HuberShape()` | `r²` up to `w`, linear beyond | |
| `CauchyShape()` | `w² log(1 + r²/w²)` | |
| `GaussianShape()` | `r²` | plain least squares; takes no `loss_width` |
| [`LeZachTruncation`](@ref)`()` | `min(r², w²)` | the truncated loss under Le and Zach's reweighting scheme |

The truncated loss with a fixed width is the cost that MSAC (M-estimator
sample consensus) scores hypotheses with, so the default refiner optimizes
what an MSAC-scored RANSAC loop measures.

Under a truncated loss, a correspondence with a residual above `w` contributes
nothing. A seed far from the answer, with most residuals above the width,
therefore comes back almost unchanged.

## Frames and units

The refiners follow the stack's rule that results come back in the frame the
inputs came in. They hand the data to PoseLib as given and normalize nothing.
`loss_width` is in the unit of the residual, which is the unit of the data:

- A homography or fundamental matrix in pixels takes pixel points and a pixel
  width. A model in another frame takes points and width in that frame.
- The camera-aware `refine_relpose(pose, K1, K2, cs)` and
  `refine_absolute_pose(pose, K, cs)` take the intrinsic matrices of the frame
  the points are in, and a width in that frame's unit. PoseLib back-projects
  through each camera, so one width serves two images even when their focal
  lengths differ.

Each intrinsic matrix reaches PoseLib as its `SIMPLE_PINHOLE` camera model:
one focal length, `(K₁₁ + K₂₂)/2`, and the principal point `(K₁₃, K₂₃)`. Skew
and any difference between `K₁₁` and `K₂₂` are dropped. The model is exact
only for square pixels and zero skew.

## Relative pose on bearings

Calibrated geometry in this stack works on unit bearings, because the z = 1
image plane has no point for a ray at or beyond 90° from the optical axis.
`refine_relpose(pose, cs)` takes bearings `f1 => f2` together with the 3×2
Jacobians of the back-projection, as the attribute columns `M₁` and `M₂`:

```julia
using VisualGeometryToolkitCore: backproject_with_jacobian

b1 = backproject_with_jacobian.(Ref(camera1), u1)   # (bearing, Jacobian) per point
b2 = backproject_with_jacobian.(Ref(camera2), u2)
cs = withattrs(Csponds(first.(b1), first.(b2)); M₁ = last.(b1), M₂ = last.(b2))
refine_relpose(pose, cs; max_iter = 25, loss_width = 2.0)
```

The residual is the tangent Sampson error

    r = (f₂ᵀ E f₁) / √(‖M₂ᵀ E f₁‖² + ‖M₁ᵀ Eᵀ f₂‖²)

Its denominator is the gradient of the epipolar constraint with respect to the
coordinates the Jacobians differentiate against. So the Jacobians set the unit
of `r`, and of `loss_width`:

| `M` is | `r` and `loss_width` are in |
|---|---|
| `∂f/∂u` from `backproject_with_jacobian` | pixels of `u` |
| an orthonormal basis of the tangent plane of the sphere at `f` | radians |

A width in the wrong unit is off by a factor of the focal length, and nothing
reports it: a truncated loss simply keeps the wrong inliers.

With pixel points and pinhole cameras, the camera-aware form reduces to this
one. PoseLib back-projects each point and runs the same refiner, and the two
forms agree to rounding error.

### What is refused

Both calibrated refiners refuse calibrated 2-vectors (points on the z = 1
plane) with an `ArgumentError` that names the supported form.
`refine_relpose(pose, cs)` also refuses 3-vectors that are not of unit length.
A point `(x, y, 1)` has the right type, and for a narrow field of view it even
refines to a plausible pose, but it is not a bearing.

## Absolute pose

PoseLib has no absolute-pose refiner on bearings: its residual is a
reprojection error in the image. `refine_absolute_pose(pose, K, cs)` is
therefore the only form, with pixel points and a pixel width. Only the six
degrees of freedom of the pose are refined; `K` stays fixed. Because the
camera is a pinhole, this package offers no correct absolute-pose refinement
for a fisheye camera.

`PoseLib.bundle_adjust`, PoseLib's own name for this refinement, is an
unexported alias of `refine_absolute_pose` with the same arguments.

## Poses

The pose refiners take and return a [`Rigid3`](@ref PoseLib.Rigid3), an
unlabeled rigid motion `x ↦ R·x + t`. `EuclideanMap(R, t)` builds one from a
rotation and a translation. A labeled pose, such as the `Extrinsics3` that P3P
returns, becomes one through `unlabel`.

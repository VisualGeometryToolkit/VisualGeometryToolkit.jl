# Refiners

PoseLib's robust Levenberg–Marquardt (LM) optimization polishes a guess. In the
stack's naming it is a refine: an algorithm that needs a starting guess and
returns a model of the same kind. It does not sample hypotheses or decide
inliers on its own; that is the job of a random sample consensus (RANSAC)
estimator, which may call a refine as its local optimization.

Every refine is one call of the stack's estimation interface:

```julia
solve(RefineProblem(cs, guess[, p]),
      LevenbergMarquardt(PoseLibSolver(); optimization_strategy);
      stops = LmStops(; max_iter), scale | width)
```

- `RefineProblem(cs, guess, p)` holds the correspondences, the guess and the
  context: the guess's type and the data's shape select PoseLib's residual, and
  `p` holds the intrinsic matrices where PoseLib needs them (a refine that reads
  none takes none: a stray context has no method). VisualGeometryToolkitCore's
  `solve` forwards it to the backend contract generic `lm_backend_refine`, whose
  `PoseLibSolver` method this package is.
- `LevenbergMarquardt(PoseLibSolver())` is VisualGeometryToolkitCore's one
  host LM with PoseLib's backend. Its fields are its configuration: the
  strategy, `optimization_strategy`, and the damping, PoseLib's λ·I
  (`LevenbergDamping()`, the backend's default; another is refused).
- The solve keywords are the run settings: `stops`, VisualGeometryToolkitCore's
  `LmStops`, of which PoseLib applies `max_iter`, its iteration budget, and the
  scale input, `scale` or `width`.

| guess | `cs` | context `p` | Residual |
|---|---|---|---|
| `HomographyMat` | 2D `x1 => x2` | none | symmetric transfer error |
| `FundMat` | 2D `x1 => x2` | none | Sampson error |
| `EuclideanMap{3}`, a relative pose | pixel points `x1 => x2` | `(; K1, K2)` | Sampson error through the cameras |
| `EuclideanMap{3}`, a relative pose | unit bearings `f1 => f2` with Jacobians | none | tangent Sampson error |
| `EuclideanMap{3}`, an absolute pose | 3D to pixel `X => u` | `(; K)` | reprojection error |

The settings:

- `max_iter` (of `stops = LmStops(; max_iter)`), PoseLib's Levenberg–Marquardt
  iteration budget, 1000 by default. PoseLib's tolerances are its own: any
  other `LmStops` field that is not `===` to `LmStops()`'s is refused with an
  `ArgumentError` naming it, never ignored.
- `optimization_strategy`, the strategy: VisualGeometryToolkitCore's `Irls{K}`
  with one of the shapes below, PoseLib's [`LeZach`](@ref)`()`, or `nothing`
  for plain least squares. It has no default: the strategy of a robust
  estimate is the caller's configuration, and PoseLib's refines have no class
  default. (An `Irls` strategy's own `max_iter`, its reweighting count, is not
  read: PoseLib's budget is the stops'.)
- exactly one of `width = w`, the width of the loss in the unit of the
  residual, and `scale = σ`, the noise scale (or an earlier robust result,
  whose `noise_scale` is handed on). Least squares takes neither.

A keyword the solve does not take is a `MethodError`, as for every algorithm.

## Strategies

PoseLib's Levenberg–Marquardt weighs each residual by its loss weight `ρ′`
alone, without the `ρ″` term, so its strategies are `Irls{K}`; a `Triggs{K}`
(the `ρ″`-corrected LM of the stack's native refines) is refused. On this
backend a strategy SELECTS PoseLib's own robust loss (its C loss enum): an
`Irls{K}` here is PoseLib's robustified Levenberg–Marquardt at that loss, not
VisualGeometryToolkitCore's reweighting loop, and `LeZach()` exists for this
backend only (every other backend refuses it). Each shape
is parametrized by one width `w`, and PoseLib's own `loss_scale` parameter
equals that width for every loss bound here. The width reaches PoseLib
unchanged.

| `optimization_strategy` | `ρ(r)` | |
|---|---|---|
| `Irls{TLSShape}()` | `min(r², w²)` | inliers have `r < w`; outliers get zero weight |
| `Irls{HuberShape}()` | `r²` up to `w`, linear beyond | |
| `Irls{CauchyShape}()` | `w² log(1 + r²/w²)` | |
| [`LeZach`](@ref)`()` | `min(r², w²)` | the truncated loss under Le and Zach's reweighting scheme; TOML `type = "le_zach"` ([`POSELIB_STRATEGY_TAGS`](@ref PoseLib.POSELIB_STRATEGY_TAGS)) |
| `nothing` | `r²` | plain least squares; takes no `scale` or `width` |

Given `scale = σ`, the loss runs at `w = c·σ`: `c` is the strategy's own
efficiency constant, else VisualGeometryToolkitCore's
`efficiency_constant(shape, d)` at the codimension `d` of the refine's
residual (1 for the Sampson and tangent-Sampson errors, 2 for the transfer and
reprojection errors). A truncation (TLS, Le–Zach) has no efficiency constant,
so it takes `width`.

The truncated loss with a fixed width is the cost that MSAC (M-estimator
sample consensus) scores hypotheses with, so `Irls{TLSShape}()` optimizes
what an MSAC-scored RANSAC loop measures.

Under a truncated loss, a correspondence with a residual above `w` contributes
nothing. A guess far from the answer, with most residuals above the width,
therefore comes back almost unchanged.

## The result

Every refine returns a VisualGeometryToolkitCore `FitResult`, read by field:

- `result.u`, the refined model;
- `result.retcode`, its termination record's `ReturnCode`: `ReturnCode.Success`
  (with `result.stop_rule == NoRule`: PoseLib states no rule) when PoseLib
  stopped on its own tolerances, `ReturnCode.MaxIters` when it used the whole
  budget; `result.stats.nsteps`, the iterations PoseLib took. Fewer
  correspondences than the model's minimal sample (4 for a homography, 7 for a
  fundamental matrix, 5 for a relative pose, 3 for an absolute pose) is
  `ReturnCode.InitialFailure`: the guess comes back unrefined after 0
  iterations, and PoseLib is not called;
- `result.n`, the number of correspondences refined over (all of them: PoseLib
  reports no weights, so this is not a weighted support), 0 below the minimal
  sample; `result.rms` is `NaN` and `result.resid` `nothing` (PoseLib reports
  no residuals);
- `result.original`, the C wrapper's raw output: its return value, PoseLib's
  iteration count (`BundleStats.iterations`), and the output buffers as PoseLib
  wrote them, column-major (`H` or `F`; a pose's `R` and `t`). When PoseLib
  was not called it holds 0 iterations and the guess's buffers, so a refine's
  result has one type whatever the data;
- `result.prob` and `result.alg`, the problem and the algorithm solved;
- the scale record in its metadata ([`PoseLibRefineMetadata`](@ref PoseLib.PoseLibRefineMetadata)):
  the scale it was given and its source, read by `noise_scale(result)` and
  `scale_source(result)`, and `result.metadata.width`, the width PoseLib ran
  at. `noise_scale(result)` hands a given scale to the next stage; a result run
  at a width or by least squares records none (`ScaleNone`) and refuses it.

## Frames and units

The refines follow the stack's rule that results come back in the frame the
inputs came in. They hand the data to PoseLib as given and normalize nothing.
The width is in the unit of the residual, which is the unit of the data:

- A homography or fundamental matrix in pixels takes pixel points and a pixel
  width. A model in another frame takes points and width in that frame.
- The camera-aware relative and absolute poses take the intrinsic matrices of
  the frame the points are in, as their context, and a width in that frame's
  unit. PoseLib back-projects through each camera, so one width serves two
  images even when their focal lengths differ.

Each intrinsic matrix reaches PoseLib as its `SIMPLE_PINHOLE` camera model:
one focal length, `(K₁₁ + K₂₂)/2`, and the principal point `(K₁₃, K₂₃)`. Skew
and any difference between `K₁₁` and `K₂₂` are dropped. The model is exact
only for square pixels and zero skew. [`calibrated_loss_width`](@ref PoseLib.calibrated_loss_width)
converts a pixel width to the bearing unit PoseLib's relative pose works at.

## Relative pose on bearings

Calibrated geometry in this stack works on unit bearings, because the z = 1
image plane has no point for a ray at or beyond 90° from the optical axis.
A relative pose on bearings `f1 => f2` takes the 3×2 Jacobians of
the back-projection, as the attribute columns `M₁` and `M₂`, and no context:

```julia
using VisualGeometryToolkitCore: backproject_with_jacobian

b1 = backproject_with_jacobian.(Ref(camera1), u1)   # (bearing, Jacobian) per point
b2 = backproject_with_jacobian.(Ref(camera2), u2)
cs = withattrs(Correspondences(first.(b1), first.(b2)); M₁ = last.(b1), M₂ = last.(b2))
solve(RefineProblem(cs, pose),
      LevenbergMarquardt(PoseLibSolver(); optimization_strategy = Irls{TLSShape}());
      stops = LmStops(; max_iter = 25), width = 2.0)
```

The residual is the tangent Sampson error

    r = (f₂ᵀ E f₁) / √(‖M₂ᵀ E f₁‖² + ‖M₁ᵀ Eᵀ f₂‖²)

Its denominator is the gradient of the epipolar constraint with respect to the
coordinates the Jacobians differentiate against. So the Jacobians set the unit
of `r`, and of the width:

| `M` is | `r` and the width are in |
|---|---|
| `∂f/∂u` from `backproject_with_jacobian` | pixels of `u` |
| an orthonormal basis of the tangent plane of the sphere at `f` | radians |

A width in the wrong unit is off by a factor of the focal length, and nothing
reports it: a truncated loss simply keeps the wrong inliers.

With pixel points and pinhole cameras, the camera-aware form reduces to this
one. PoseLib back-projects each point and runs the same refinement, and the two
forms agree to rounding error.

### What is refused

Both calibrated poses refuse calibrated 2-vectors (points on the z = 1 plane:
2D points without the intrinsics) with an `ArgumentError` that names the
supported form. The bearing form also refuses 3-vectors that are not finite or
not of unit length. A point `(x, y, 1)` has the right type, and for a narrow
field of view it even refines to a plausible pose, but it is not a bearing. A
physical relative pose is refused too: its translation is a direction, up to
scale, with no unit to strip.

## Absolute pose

PoseLib has no absolute-pose refine on bearings: its residual is a
reprojection error in the image. A refine of `X => u` with the
context `(; K)` is therefore the only form, with pixel points and a pixel
width. Only the six degrees of freedom of the pose are refined; `K` stays
fixed. Because the camera is a pinhole, this package offers no correct
absolute-pose refinement for a fisheye camera.

A physical guess (a translation in FlexUnits lengths) with physical world
points is stripped in metres once, refined, and comes back in metres, with the
plain refinement's record; a physical guess with plain points, or the reverse,
is refused.

## Poses

The pose refines take, and return as their `u`, an unlabeled `EuclideanMap{3}`
(a `RotMatrix{3, Float64}` rotation and a `Vec3{Float64}` translation), the rigid motion `x ↦ R·x + t`. `EuclideanMap(R, t)` builds one from a
rotation and a translation. A labeled pose, such as the `CameraExtrinsics` that P3P
returns, becomes one through `unlabel`.

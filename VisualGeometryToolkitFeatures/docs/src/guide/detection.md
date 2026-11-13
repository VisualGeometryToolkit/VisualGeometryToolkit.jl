# Detection

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

## The ASIFT detector

[`asift_detect`](@ref) is the package's detector. It implements IMAS-25, the
affine-invariant SIFT of the IMAS family. The image is resampled under 25
simulated camera tilts (the identity, 8 tilts of 2.62 and 16 tilts of 5.18 at
evenly spaced longitudes). DoG extrema are found in each warp, refined to
sub-pixel accuracy, mapped back to the input image as elliptic frames, and
merged across warps by a non-maximum suppression. The whole pipeline runs as
KernelAbstractions kernels, so the same code runs on the CPU (the default) and
on any GPU backend (`device = CUDABackend()`, for instance).

The tilt schedule is a keyword, a [`tilt_schedule`](@ref) name or a vector of
`(t, φ)` pairs. `tilts = :untilted`, the identity alone, is plain DoG-SIFT. Its
detections localize best (about 0.3 px rms on real panorama frames, against
about 0.95 px for a tilted warp's), so it is the schedule for views that differ
by a rotation, such as a panorama's:

```julia
dets = asift_detect(img)                       # IMAS-25 (tilts = :imas25)
sift = asift_detect(img; tilts = :untilted)    # DoG-SIFT: the identity warp alone
```

`exclusion_mask`, an H×W `Bool` matrix, marks the pixels where detection is not allowed
(clipped highlights, a vignetted rim): a detection whose centre lies within
three of its scales of a masked pixel is dropped. The detector on a
variance-stabilized RAW ([`variance_stabilize`](@ref)) masks the sensor's
saturated pixels by the same rule.

```julia
clipped = Float32.(Gray.(img)) .>= 0.99f0
dets = asift_detect(img; exclusion_mask = clipped)
```

The other keywords choose the non-maximum suppression (`nms = :mxm`,
`:ellipse` or `:none`), gate by polarity (`polarity = :dark` or `:light`),
restrict suppression to features of similar scale (`band_octaves`), and set
the detection thresholds. The [`asift_detect`](@ref) docstring lists them.

The result does not depend on the threads or the device's scheduling: the same
image gives the same detections, in the same order, run to run.

### Plans and their cache

The detector's buffers for one image size, device and tilt schedule are an
[`AsiftPlan`](@ref). `asift_detect` builds plans on demand and keeps the most
recent one: a plan for a large image under the full schedule holds several GiB
of device memory. Process frames grouped by resolution to reuse it.
[`set_max_cached_plans!`](@ref) keeps more plans, and
[`clear_asift_plans!`](@ref) releases them. To manage a plan yourself, build it
and call [`asift_detect!`](@ref), which returns the raw
[`ImasFeatureKA`](@ref) records; [`asift_detections`](@ref) converts them.

## Pixel origin

Every detector of the package returns COLMAP pixel centres. The top-left
pixel's centre is `(0.5, 0.5)`, so the pixel `img[r, c]` has its centre at
`(c − 0.5, r − 0.5)`. A `center` is `(x, y)`: x along the image columns and y
along the rows.

`asift_detect(img; raw = true)` returns the raw features instead, in 0-indexed
OpenCV coordinates (top-left pixel centre at `(0, 0)`), which are 0.5 less on
both axes.

The package follows the stack's rule that a function returns results in the
frame its inputs came in: pixels in, pixels out. The matchers work in
descriptor space and never touch coordinates, so nothing here normalizes.

## Feature frames are shapes

A [`Detection`](@ref) composes a feature frame with the detector's metadata.
"Frame" is meant in the VLFeat and Kornia sense: the shape of a detected
feature, not a coordinate system. The frames are VisualGeometryToolkitCore
types, ordered by degrees of freedom:

| Frame | DOF | Shape |
|---|---|---|
| `Disc` | 3 | centre and isotropic scale |
| `OrientedDisc` | 4 | centre, scale and orientation |
| `EllipticFrame` | 5 | centre and shape matrix (anisotropic) |
| `AffineFrame` | 6 | centre and full affine shape |

`asift_detect` returns `Detection{EllipticFrame}`. The accessor is
VisualGeometryToolkitCore's `feature_frame(d)`. Properties of the frame and of the metadata read directly:
`d.center`, `d.σ` (the characteristic scale), `d.Σ` (the shape matrix),
`d.response`.

Coordinate-frame tags, such as `ImageFrame` and `NormalizedImageFrame`, are a
separate vocabulary that lives in VisualGeometryToolkitCore. The two share the
word "frame" and nothing else.

## Polarity

ASIFT records the sign of the DoG response in the metadata field
`laplacian_scale_score`: positive for a dark blob on a lighter background and
negative for a light one. [`is_dark`](@ref) and [`is_light`](@ref) read it.
Detections whose metadata has no such field are neither; the trait
[`polarity_trait`](@ref) tells the two kinds of metadata apart.

## Accuracy of the sub-pixel refinement

The quadratic sub-pixel refinement pulls a centre toward the nearest pixel
centre. On noise-free isotropic Gaussian blobs on the CPU, measured by the test
suite:

- a blob centred on a pixel centre comes back exact, within 0.005 px;
- between pixel centres the pull is at most 0.03 px at σ = 4, and at most
  0.06 px for σ from 2 to 8.

Larger blobs and clutter do worse. The Monte Carlo study in the package's
`benchmark/mc` directory measured a centre bias of 0.1 to 0.2 px on clean
blobs with σ ≥ 6.5, and up to about 1 px in clutter. Treat 0.06 px as the bound
for small, isolated blobs only.

## Detection uncertainty

`asift_detect(img; uncertainty = FeatureCovariance())` returns each detection
with its frame as VisualGeometryToolkitCore's `Uncertain(frame, Σ̃)`. `cov(d)`
is a 3×3 covariance over `(x, y, log σ)` in the image frame, for unit iid pixel
noise in the detector's intensity units (the [0, 1] scale for 8-bit input). The
physical covariance is `σ_n²·Σ̃` for a noise standard deviation `σ_n`. Consumers
that need only the position take the top-left 2×2 block.

The covariance follows from the implicit function theorem at the refined DoG
extremum; its noise term is tabulated exactly for the discrete pyramid once per
plan. The [`FeatureCovariance`](@ref) docstring states the model. The Monte
Carlo study that calibrates it is `benchmark/mc/detection_uncertainty_mc.jl`,
and `benchmark/mc/zeisl_comparison.jl` compares it with the covariance of
Zeisl et al. (BMVC 2009). The default, `uncertainty = nothing`, compiles the
detector exactly as without the option.

## Continuous refinement (opt-in)

`asift_detect(img; refinement = ContinuousRefinement())` refines each detection again after the
detector's one-step quadratic refinement. The detection candidates are the same. The second
refinement is an iterative solve, Core's device Levenberg–Marquardt `static_lm`, run inside a
kernel on the CPU and on a GPU, and it works on a continuous model of the DoG response in
`(x, y, log σ)`. With `uncertainty = FeatureCovariance()`, each refined detection's covariance is
recomputed at the refined point from that model's own derivatives. A detection whose continuous
refinement fails keeps the one-step position. Without the keyword, the detector is unchanged.

`ContinuousRefinement()` is one recipe. The spatial gradient comes from Farid and Simoncelli's
matched prefilter and first-derivative filters (IEEE TIP 2004), interpolated. The scale row comes
from the cardinal-spline interpolant of the unfiltered DoG, so the prefilter's blur does not bias
the scale. Its keywords are the bounds and tolerances of the refinement, and their defaults come
from `configs/continuous_refinement.toml`.

On noise-free synthetic blobs (`benchmark/continuous_refine`), the largest centre bias over a
full sub-pixel period drops from 0.02–0.14 level pixels (one-step) to 0.0001–0.0005. The
half-sample jump of the one-step refinement disappears. The scale error matches the spline
model's, at most 0.0041 in `log σ` at octaves 1–3. The covariance is calibrated, with NEES₂
between 1.94 and 2.09. On a GPU the refinement costs about twice the detector's own time per
detection; on the CPU it adds about 16 %.

## The image circle of a lens

[`detect_image_circle`](@ref) measures where light falls on the sensor, the lit
region a lens's field stop leaves on a dark surround, from photographs of one
camera, lens and focus. No lens model predicts it. Each frame lights a different
part of the rim, and their per-pixel maximum is the union of their light. Its
dark maximally stable regions ([`detect_mser`](@ref), every region up to the
whole image, duplicates kept) carry the rim as the boundary of the surround.
The convex hull of every region's boundary, without the frame's border, small
holes and vertices within `border_margin` of the border, is the rim's contour.
The model, an `Ellipse` (the default) or a `Circle`, is a RANSAC fit through
that contour, scored by the Student-t + uniform marginal at the caller's
`outlier_halfwidth`.

The detection is this package's. The fit is VisualGeometryToolkitEstimation's
`find_ellipse` / `find_circle`, reached through a package extension, so load
that package first. The optional L0 smoothing (`smoothing_weight > 0`) is
ImageSmooth's, also an extension:

```julia
using VisualGeometryToolkitFeatures, VisualGeometryToolkitEstimation
using Random: Xoshiro

res = detect_image_circle(Xoshiro(1), frames; outlier_halfwidth = 1000, min_hole_area = 256)
res === nothing || (res.circle, res.rim_gap < π)   # the circle; whether the rim was seen all round
```

A RAW frame (a `LinearRaw` or a `RawMosaic`) is measured on its green cells,
the geometry image, with the keywords in green-cell pixels. Only the result is
mapped: the circle and the contour come back in the raw's pixels through the
cells' `raw_frame_map`. The circle is about 1 px outside the true rim, because the hull follows
the dark side of the rim's partial-coverage band. Treat it as a mask that errs
toward including the rim's last pixels.

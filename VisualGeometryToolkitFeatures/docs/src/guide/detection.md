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

The tilt schedule is a keyword. `tilts = [(1.0, 0.0)]`, the identity alone, is
plain DoG-SIFT:

```julia
dets = asift_detect(img)                          # IMAS-25
sift = asift_detect(img; tilts = [(1.0, 0.0)])    # DoG-SIFT
```

The other keywords choose the non-maximum suppression (`nms = :mxm`,
`:ellipse` or `:none`), gate by polarity (`polarity = :dark` or `:light`),
restrict suppression to features of similar scale (`band_octaves`), and set
the detection thresholds. The [`asift_detect`](@ref) docstring lists them.

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
| `EllipticFrame` | 5 | centre and moment matrix (anisotropic) |
| `AffineFrame` | 6 | centre and full affine shape |

`asift_detect` returns `Detection{EllipticFrame}`. The accessor is
`feature_frame(d)`. Properties of the frame and of the metadata read directly:
`d.center`, `d.σ` (the characteristic scale), `d.Σ` (the moment matrix),
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

## Keypoint uncertainty

`asift_detect(img; uncertainty = FeatureCovariance())` returns each detection
with its frame as VisualGeometryToolkitCore's `Uncertain(frame, Σ̃)`. `cov(d)`
is a 3×3 covariance over `(x, y, log σ)` in the image frame, for unit iid pixel
noise in the detector's intensity units (the [0, 1] scale for 8-bit input). The
physical covariance is `σ_n²·Σ̃` for a noise standard deviation `σ_n`. Consumers
that need only the position take the top-left 2×2 block.

The covariance follows from the implicit function theorem at the refined DoG
extremum; its noise term is tabulated exactly for the discrete pyramid once per
plan. The [`FeatureCovariance`](@ref) docstring states the model. The Monte
Carlo study that calibrates it is `benchmark/mc/keypoint_uncertainty_mc.jl`,
and `benchmark/mc/zeisl_comparison.jl` compares it with the covariance of
Zeisl et al. (BMVC 2009). The default, `uncertainty = nothing`, compiles the
detector exactly as without the option.

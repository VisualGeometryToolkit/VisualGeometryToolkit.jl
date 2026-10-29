# The Förstner operator

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

[`detect_foerstner`](@ref) is the interest operator of Förstner and Gülch
[Foerstner1987](@cite): it finds windows of a grey image whose gradients
locate a point well, and estimates in each a junction point and a circular
point by least squares, keeping the one a significance test prefers. It
follows the MATLAB reference implementation `ip_fop.m` [Luxen2005](@cite)
step by step, including the two extensions it takes from Köthe
[Koethe2003](@cite): the image is processed at doubled resolution, and the
`:koethe` window criterion ranks windows by the smaller eigenvalue of the
structure tensor.

```julia
using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore

points  = detect_foerstner(img)                          # Float32, default_device()
corners = filter(d -> d.label === :corner, points)
Ellipse(first(corners); sigma = 1)                       # the 1σ error ellipse (ip_errell.m)
```

## What it returns

One row per window, a `StructArray` of
`Detection(Uncertain(PointFrame(x), Σ), (; label, window, F))`. The frame is
VisualGeometryToolkitCore's `PointFrame`, a position without a scale: neither
point model estimates a size.

| `label` | the row's point and covariance |
|---|---|
| `:corner` | the junction point (the reference's `corner`) |
| `:circular` | the reference's circular point (`dot_point`): the centre of a rotationally symmetric pattern, a dot or a ring. Förstner & Gülch's term; it carries no size estimate and is not a scale-space blob |
| `:unknown` | the window centre and a NaN covariance: the significance test labelled neither (the reference's `noclass`, which has no estimate) |

`window` is the window centre (the reference's `win`), and `F` the statistic
`Ω_corner / Ω_circular` the label was decided on.

## Coordinates

Positions are COLMAP pixel centres of the input image, `(x, y) =
(column, row)`, the top-left pixel's centre at `(0.5, 0.5)`, as every detector
of the package returns them. MATLAB indexes the pixel `(r, c)` at its centre,
so a position `(r, c)` of the reference is `(c − ½, r − ½)` here.

The operator works on the doubled image. The package's one 2× upsampling (the
ASIFT seed's, VLFeat's grid) makes the doubled pixel `(r′, c′)` (1-based) the
bilinear sample at input position `(r′/2, c′/2)`, 1-based, so a doubled-grid
position is at COLMAP `((c′ − 1)/2, (r′ − 1)/2)` in the input, and a
covariance shrinks by 4. The conversion is made once, at the output.

## Devices and precision

The image-wide steps — the doubling, the gradient and its integration, the
window measures, the regional maxima and the non-maximum suppression, and the
two least-squares estimates per window — are KernelAbstractions kernels, the
same code on `KA.CPU()` and on a GPU (`device = CUDABackend()`). The F-test
and the result are assembled on the host. `T` sets the precision: `Float32` by
default, `Float64` as the reference. On the reference's example image
`Quadrate.tif` (500 × 500, 100 windows) a CUDA device and the CPU give the same
labels and positions within 3.1 × 10⁻⁵ px in `Float32` and 3.2 × 10⁻¹⁴ px in
`Float64`; the test suite checks it whenever a CUDA device is present.

## Accuracy

On area-sampled synthetic dots (radius 2.5 px, sub-pixel centres spread over
the whole frame up to 8 px from every border) every circular point comes back
within 0.0054 px of the truth. A junction is less exact: at an X-junction the
window measure is flat around the vertex, so the window can land a few doubled
pixels off it, and the point estimate in an off-centre window is biased by up
to ≈ 0.1 px (within its reported covariance); this is the operator's, not a
difference from the reference.

On `Quadrate.tif` with the defaults, both window criteria find 100 windows,
all labelled `:corner`; with the reference example's parameters
(`noise_sigma = 1`, `integration = :gaussian`, `derivative_sigma = 0.7`,
`integration_sigma = 2`) also 100 corners, median position standard deviation
0.057 px.

## Differences from `ip_fop.m`

The implementation follows the reference line for line except in three
places, each recorded here with its evidence.

### 1. Undoing the doubling: fixed

The reference doubles the image with `imresize(g, 2, 'bilinear')` (line 172)
and maps positions back with `transf_H`, `transf_v` (lines 340–350, applied at
436, 446 and 457): `x = (1 − n)/(1 − 2n)·x′ + (1 − 2)n/(1 − 2n)` per axis, `n`
the image size. That map is the inverse of an align-corners grid, on which the
doubled pixels `1` and `2n` sit on the input pixels `1` and `n`. `imresize`'s
grid is not that: its doubled pixel `j` samples the input at `j/2 + 1/4`
(1-based), so the exact inverse is `x = x′/2 + 1/4`, and the reference's points
are off by `((n + 1)/2 − x)/(2n − 1)`: zero at the image centre, up to 0.25 px
toward the centre at the borders.

What this rests on, since MATLAB cannot be run here: MATLAB's `imresize`
documentation states that antialiasing applies only when shrinking, so a
2× bilinear enlargement is plain bilinear interpolation; and `imresize.m`
maps output to input coordinates by `u = x/scale + 0.5·(1 − 1/scale)`, i.e.
`u = x/2 + 1/4` for `scale = 2`, with the out-of-range neighbour of the first
sample taken from the edge pixel (index mirroring, which at this offset
replicates the border). The tests emulate that grid to measure the reference's
map.

Measured on 30–42 area-sampled dots spread over the frame (sub-pixel centres,
up to 8 px from every border), sampled on `imresize`'s grid:

| back-map | largest error |
|---|---|
| the reference's `transf_H`, `transf_v` | 0.236 px (borders; ≈ 0 at the centre), matching the predicted `((n + 1)/2 − x)/(2n − 1)` to 0.002 px at every dot |
| `imresize`'s exact inverse | 0.0054 px |
| the package's upsampling grid, its exact inverse (what `detect_foerstner` does) | 0.0054 px |

The package's grid (input `j/2`) is a quarter input pixel from `imresize`'s
(input `j/2 + 1/4`); the doubled samples differ accordingly, and the measured
effect on the positions is nil at this precision (0.0054 px either way).

### 2. The circular point's residual sum: kept as the reference has it

Line 423 computes the circular point's weighted residual sum as
`Σ γ_cc·Δr_dot² − 2γ_rc·Δr_corner·Δc_corner + γ_rr·Δc_corner²`: two of its
three terms use the junction model's residuals. Förstner and Gülch's circular
model has, for each pixel `i`, the residual `∇gᵢ⊥ᵀ(x̂ − xᵢ)` (the distance of
the point from the line through the pixel along its gradient, weighted by the
gradient), so its residual sum is `Σ (xᵢ − x̂)ᵀ Γᵢ⊥ (xᵢ − x̂)` with all three
terms from the circular estimate. That sum enters the covariance
`Ω/redundancy · N⁻¹` (line 449) and the test statistic (line 427).

A Monte Carlo test of the reported covariance does not support the derived
form. On noisy synthetic dots (radius 2.5 px, random sub-pixel centres,
400 trials each; NEES `eᵀΣ⁻¹e`, nominal mean 2):

| noise (of the [0, 1] range) | | mean NEES | 95 % coverage | 99.9 % coverage |
|---|---|---|---|---|
| 0.01 | line 423 (kept) | 3.0 | 0.85 | 0.965 |
| 0.01 | derived residual sum | 13.7 | 0.375 | 0.65 |
| 0.03 | line 423 (kept) | 8.7 | 0.578 | 0.805 |
| 0.03 | derived residual sum | 20.4 | 0.315 | 0.52 |

Neither form is calibrated for circular points (both under-cover: the variance
factor treats the gradients of a window as independent, which the smoothing
makes them not), and the reference's form is the closer one. The labels were
the same under both in all 1 600 trials (dots and X-junctions at both noise
levels), the junction covariance does not involve it (NEES 0.91 and 1.72,
95 % coverage 0.998 and 0.96), and the reference's sum was never ≤ 0. So line
423 is kept as written; a test item reruns a 100-trial version of the
comparison.

### 3. Defaults: the code's

The reference's header documents `SIGMA_N = 2.5` grey values and
`SIGMA_DIFF = 0.7` px (lines 37 and 50); its code runs with `2.0` and `1.0`
(lines 521 and 524). The defaults here are the code's — what `ip_fop.m`
actually ran. This is a conflict in the reference's documentation, not a
correction.

### Also as the reference

`noise_sigma` is in grey levels of an 8-bit image, as the reference's for a
`uint8` image: the operator reads the image in `[0, 1]` and uses
`noise_sigma / 255`. The reference stretches a floating-point image to its own
range; this does not. With a Gaussian integration the estimation window still
weighs its pixels equally, which the reference's header flags as an
inconsistency.

The full docstring: [`detect_foerstner`](@ref).

## Measurement noise lives on the measurements

A correspondence residual is a constraint `g(θ; x₁, x₂) = 0` whitened by the
noise of its measurements. That noise is a property of each MEASUREMENT, stated
where the measurement is built, never a property of a problem type:

| Endpoint | Covariance | Spelled |
|---|---|---|
| plain point | `Σ = I`, unit isotropic noise **in its own frame** | `p` |
| uncertain point | its own `Σ` (or precision) | [`Uncertain`](@ref)`(p, `[`CovMat`](@ref)`(Σ))`, `Uncertain(p, `[`PrecMat`](@ref)`(Λ))` |
| exact point | `Σ = 0` | [`exact`](@ref)`(p)` |

With `G_x = ∂g/∂(x₁, x₂) = [G₁ G₂]`, the constraint covariance is the first-order
propagation of both endpoints and the residual is the constraint whitened by it:

    Σ̃ = G₁·Σ₁·G₁ᵀ + G₂·Σ₂·G₂ᵀ,    r = L⁻¹·g,    L·Lᵀ = Σ̃.

[`pair_covariance`](@ref) is the 4×4 block diagonal over `(x₁, y₁, x₂, y₂)` that
[`whiten`](@ref) and [`whitening_factor`](@ref) take as `Σ_x`. **A correspondence
whose two endpoints are both exact is an error** (`ArgumentError`): `Σ̃ = 0`
leaves nothing to whiten, and no residual unit exists.

**What a threshold means.** `r` is in units of the noise. For plain image points
that is the pixel: a 3 px threshold means 3 px, exactly as before this rule was
written down. With real covariances `r` is in SIGMAS — `r² ~ χ²(d_g)` — and a
threshold is a number of standard deviations. An exact source with a plain
target is the plain pixel case on one side: the residual is the transfer error
in the target image.

**Plain is bit-for-bit what it always was.** A plain set carries no covariance
columns, every reader takes its `Σ = I` path unchanged, and the stack's
equivalence goldens are identical across the change. Image↔image matches, pool
pairs and F/E/H estimation on them do not move.

### Why the old formulation was wrong — a worked example

Until 2026-10 every correspondence problem hard-coded `Σ₁ = Σ₂ = I` over
`(x₁, y₁, x₂, y₂)`, i.e. assumed both endpoints were noisy points measured in
the SAME unit. A calibration board breaks that silently.

Take a 5 cm board in METRES seen at 0.3 m by an f = 800 px camera. The board →
image Jacobian is about `f/Z ≈ 2670` px/m (measured: singular values 2500–2600 at
the board centre). For the homography's transfer form `G_x = [A −I]`,
`A = ∂u₂/∂x₁`, so

    Σ̃ = A·Aᵀ + I ≈ 2600²·I,    r ≈ e / 2600,

with `e` the transfer error in pixels: the residual comes out in BOARD METRES.
A 0.2 px inlier scores ≈ 7e-5 and a 40 px mismatch ≈ 0.015. Both are far below a
3 px threshold, MSAC counts everything as an inlier, and the fit lands tens of
pixels off (25–40 px across draws of the regression test,
VisualGeometryToolkitEstimation's `metre-unit board`).

With the board `exact`, `Σ₁ = 0`: `Σ̃ = I`, `r = e` in pixels, and the 40 px
mismatch is rejected at 3 px. The fit is then the classical reprojection
estimator for a known target.

**Why it went unnoticed.** Before the K+ refactor the default scorer was a
scale-estimating marginal (Student-t + uniform): it estimated the residual
scale from the data, so a residual in metres simply produced a scale in metres
and the classification came out right. MSAC's FIXED pixel threshold, made the
`find_*` default in K+, had no scale to absorb the unit and exposed it. A
scale-estimating scorer still hides a unit error; it does not make one correct
— its outlier half-width is still stated in the residual's unit.

### When to use which

- **Image ↔ image matches** (detections in two images, pool pairs): plain.
- **Board, model, world or map points** — anything known by construction rather
  than measured: `exact`. A board → image fit is `exact.(board) .=> pixels`.
- **Detections with a covariance** (a detector's per-point uncertainty, a
  propagated estimate): `Uncertain`, in the caller's frame.

A problem class whose measurement model already treats one side as exact is
consistent with the rule as it stands: absolute pose takes plain 3D points and
whitens only the image point; the radial homography and radial camera score the
reprojection distance in the target image, accept their board side `exact` (or
plain, its long-standing meaning there) and refuse any other covariance. A problem class whose every residual route (scoring, refit
rows, lift rows) does not yet read covariances refuses `Uncertain`/`exact`
endpoints with an `ArgumentError` rather than scoring one function and
descending another (VisualGeometryToolkitEstimation: F and E today).

### Normalization

Covariances are in the CALLER's frame, like the points (frame in = frame out).
An estimator's private normalization moves them with the points: an affine side
map `x ↦ A·x + b` takes `Σ` to `A·Σ·Aᵀ` (`s²·Σ` for a similarity of scale `s`);
`normalize(N, cs)` of a [`CspondNormalization`](@ref) does it for the
correspondence set's covariance columns, and a projective measurement map uses
its Jacobian at each point. The Mahalanobis residual is invariant to the move,
so thresholds stay in the caller's units — pixels for plain pixel data, sigmas
for covariances — and nothing downstream converts them.

### Storage

[`noise_columns`](@ref) turns `Uncertain` endpoints into two attribute columns
of the correspondence set, `Σ₁` (sources) and `Σ₂` (targets), static 2×2
matrices in the caller's frame; a side whose endpoints are all plain has NO
column (unit isotropic). An estimator's constructor calls it once; the inner
loop reads the columns ([`pair_covariance`](@ref)`(cs, i)`,
[`transfer_whitening!`](@ref)) and never an `Uncertain` value. `exact` is an
ordinary zero `CovMat`: its four numbers per point cost nothing past the
constructor.

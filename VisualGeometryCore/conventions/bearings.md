## Calibrated geometry is on bearings

Calibrated two-view and pose geometry — the essential matrix, relative pose,
P3P/PnP, their RANSAC problems, refiners, scoring and benchmark pools — works on
**unit bearings** (unit rays on S², typed `StaticVector{3}`), never on the affine z = 1
plane ([`GnomonicFrame`](@ref), the chart).

The reason is not tidiness: a ray at or beyond 90° has no coordinate on that
plane, so any `K⁻¹u` dehomogenized to a 2-vector, any silent `[x, y, 1]` lift and
any scoring through `F = K⁻ᵀEK⁻¹` excludes the fisheye and omnidirectional
cameras this stack calibrates. Calibrated solvers and problems are typed
`Cspond{<:StaticVector{3},<:StaticVector{3}}` so a 2-vector is a `MethodError`
rather than a silent narrowing.

**The type is a necessary condition, not a guarantee.** A three-vector still on
the z = 1 plane, `(x, y, 1)`, satisfies the signature, and because the epipolar
algebra is scale-invariant per endpoint it even returns the right answer for
every ray the chart can represent. What it cannot represent is a ray at or past
90°, so the fisheye exclusion survives the type and merely stops being visible
in it. The guarantee comes from where bearings are **built** — the camera's
back-projection, defined for every ray the camera can see — and the essential
problem checks unit norm once, at construction. A unit-norm check inside the
solver was considered and deliberately not taken: it would be a per-sample
tolerance in a hot loop, checking data the caller built. At the boundary it is a
structural discriminator rather than a tuned constant, since a z = 1 point has
norm ≥ 1 (≈ 1.41 at 45° off axis) and coincides with a unit bearing only on the
optical axis, where the two are the same vector.

Consequences worth stating:

- Bearings come from the camera's own back-projection,
  `backproject_with_jacobian(camera, u) -> (f, M)`, whose Jacobian is tangent to
  the sphere (`fᵀM = 0`).
- The residual is a tangent-plane (spherical) Sampson distance whitened by the
  pixel covariance propagated through that Jacobian, which keeps thresholds in
  pixel σ where the caller set them.
- Calibrated data takes **no normalization**: only rotations preserve the
  essential structure, and unit bearings need no conditioning.
- **The tangency assertion `fᵀM = 0` is load-bearing, not decorative.** Swapping
  the two views' Jacobians moves the residual by under 2% (0.19% at equal focal
  lengths) but its Jacobian by 341%: a swap destroys the descent *direction*, not
  the cost, and is invisible to Monte Carlo calibration averages. Only tangency,
  checked per view, catches it — and a test of per-view wiring needs two
  different focal lengths to be able to fail at all.

**Three declared pinhole islands**, one fact with three instances. A matrix
cannot express a general camera's nonlinear back-projection, so (1) the
mapped-homography path, where `K⁻¹` declares the calibration as measurement
maps, and (2) RobustVisualGeometry's relative-pose bundle, which holds K
matrices rather than camera models, are pinhole by construction. (3)
LocalOptimizationDoneRight's vendored RansacLib baseline is third-party and exempt.
Making (1) or (2) camera-general means carrying camera models instead of
matrices — a separate decision, not a rider on this rule. Separately, the Monte
Carlo harness's epipolar **band** trial keeps its query in the z = 1 chart (an
epipolar line is a calibrated-plane object), so it cannot validate an essential
matrix's covariance at a fisheye periphery: a green band trial is not fisheye
coverage.

When checking whether something "runs on bearings", trace the data PREPARATION
path, not just the kernel. The five-point kernel had taken bearings all along,
while the affine-plane assumption was found in five other places by five
different routes: a pool generator dehomogenizing `K⁻¹u`, a 2-vector lift the
solver registry documented as "the caller's frame convention", a cheirality test
building `(x/z, y/z, 1)`, a helper named for bearings that returned plane points,
and the Monte Carlo harness's own interface — the last of which no test could
catch, because the harness is what the tests are written in.

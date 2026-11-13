# VisualGeometryToolkitEstimation changelog

## 0.11.0 (unreleased)

The breaking wave 10, on VisualGeometryToolkitCore 0.10.

### Breaking

- **`optical_axis` fits 4 radial terms (θ³…θ⁹), not Fisheye624's 6** (the owner's ruling of 2026-11-13, from the order
  study `~/logs/dgx-agent/wave10/optical-axis-order.md`): the smallest order whose principal point lies within ½σ of
  6 terms' on every converging RC camera, its cond(J) 47–160× lower. The order is configurable,
  `configs/default.toml [optical_axis] radial_terms` (read once). A camera fitted by `optical_axis`, and so the gauge
  `fit_camera(SplineLens, …)` takes for a SplineLens source, changes: its radial coefficients (four), and its focal by
  up to ~1e-5 relative (GRID's whole-image fit: 299.99977 → 300.00290 px); its principal point and axis by ≤ 1e-6 px
  and ≤ 1e-8° on the symmetric test lenses (the decentred lens: 0.007 px).

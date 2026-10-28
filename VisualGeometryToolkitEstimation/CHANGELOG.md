# VisualGeometryToolkitEstimation changelog

## 0.2.0 (unreleased)

### Added

- **`fit_linear(A, y[, prior])`**, the robust coefficients of a linear model:
  a reweighted least squares under a loss shape (default Huber at its
  efficiency constant 1.345) at the residuals' MAD scale, with fixed prior row
  weights; `configs/default.toml` `[fit_linear]`. Hoisted from
  RadiometricCamera's rooftop vignetting fit.

### Known limits

- **`refine_absolute_pose` whitens a generalized camera's residuals only
  approximately.** This applies to the generalized form (an `origins`
  column) with pixel whitening (an `M` column). The whitening maps the error
  in each ray's direction to pixels and holds the ray's origin fixed. It
  ignores `∂o/∂u`, how the origin moves with the pixel. For a non-central
  camera the residual is therefore a pixel residual to first order in the
  direction only. A column for each origin's Jacobian is a later extension.

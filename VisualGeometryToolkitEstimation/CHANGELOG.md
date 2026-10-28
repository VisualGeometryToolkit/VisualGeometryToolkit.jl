# VisualGeometryToolkitEstimation changelog

## 0.2.0 (unreleased)

### Known limits

- **`refine_absolute_pose` whitens a generalized camera's residuals only
  approximately.** This applies to the generalized form (an `origins`
  column) with pixel whitening (an `M` column). The whitening maps the error
  in each ray's direction to pixels and holds the ray's origin fixed. It
  ignores `∂o/∂u`, how the origin moves with the pixel. For a non-central
  camera the residual is therefore a pixel residual to first order in the
  direction only. A column for each origin's Jacobian is a later extension.

# VisualGeometryToolkitEstimation changelog

## 0.11.0 (unreleased)

The breaking wave 10, on VisualGeometryToolkitCore 0.10.

### Breaking

- **A camera fit over a region beyond its source's calibrated support is refused** (the owner's ruling of 2026-11-13):
  `optical_axis`, `approximate_camera`, `refine_camera`, `fit_camera` (every form) and `pspline_grid` call Core's
  `check_region_support` once at entry and throw its typed `RegionBeyondSupport`, never a warning. The support is the
  new keyword `support`, the source's `valid_domain` by default, or the calibration a re-gridded camera came from
  (RadiometricCamera's siemens start camera: refused over its whole sensor, extrapolated to 179.8°, fitted over its
  Schöps support). `optical_axis(camera, region)` takes a region (`region ∩ imaged_region(camera)`), and `fit_camera`
  / `refine_camera` of a bounded source fit over `region ∩ imaged_region(source)`, as `approximate_camera` does (the
  0.10.2 patch line, ported).

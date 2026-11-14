# Wave 11 removals

The deprecated names and keywords to be removed in wave 11 (the next breaking wave). Each one still works, with a
deprecation warning, until then. Each package's CHANGELOG points here; add a row when you deprecate something.

| package | name | deprecated in | replacement / reason | removal |
|---|---|---|---|---|
| VisualGeometryToolkitFeatures | `detect_image_circle(…; min_hole_area)` | 0.9.2 | none: the hull fit it tuned is gone. The coverage is measured on the rim's log-polar profiles. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitFeatures | `detect_image_circle(…; smoothing_weight)` | 0.9.2 | none: the L0 smoothing fed the MSER path, which is gone. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitFeatures | `detect_image_circle(…; border_margin)` | 0.9.2 | none: the hull's border vertices are gone. Off-frame bins are skipped instead. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitCore | single-frame `photon_transfer(mosaic::RawMosaic; …)` | 0.10.1 | the multi-frame `photon_transfer(groups, darks)` (owner ruling (B)) | wave 11 |
| VisualGeometryToolkitCore | the LM's tolerance keywords: `LmStops(; residual_resolution, ftol, xtol, stationarity_tol)`, `StaticLmStops`'s, and `is_stationary`'s absolute `ftol` | — (renamed in wave 11, owner 2026-11-14; no deprecation, no alias) | rename per ONE survey of SciML NonlinearSolve and MINPACK: the standard names adopted together (the survey's candidates include `abstol` for the residual-norm floor and `reltol` for the relative cost change); `is_stationary`'s absolute floor takes the survey's term too, never a third vocabulary (`cost_floor` only if the survey has no standard term). The old keywords refused by name with the migration text | wave 11 |
| VisualGeometryToolkitCore | `photon_transfer(…; normalize::Bool)` | — (owner 2026-11-14) | a strategy type in place of the Bool, named per the stack's conventions | wave 11 |
| VisualGeometryToolkitCore | `photon_transfer(groups, darks::AbstractDict)` | 0.10.4 (superseded) | `photon_transfer(groups, darks::Darks)`: the darks matched by band, the black per group | wave 11 |

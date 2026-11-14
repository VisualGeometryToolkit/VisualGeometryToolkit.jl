# Wave 11 removals

The deprecated names and keywords to be removed in wave 11 (the next breaking wave). Each one still works, with a
deprecation warning, until then. Each package's CHANGELOG points here; add a row when you deprecate something.

| package | name | deprecated in | replacement / reason | removal |
|---|---|---|---|---|
| VisualGeometryToolkitFeatures | `detect_image_circle(…; min_hole_area)` | 0.9.2 | none: the hull fit it tuned is gone. The coverage is measured on the rim's log-polar profiles. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitFeatures | `detect_image_circle(…; smoothing_weight)` | 0.9.2 | none: the L0 smoothing fed the MSER path, which is gone. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitFeatures | `detect_image_circle(…; border_margin)` | 0.9.2 | none: the hull's border vertices are gone. Off-frame bins are skipped instead. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitCore | single-frame `photon_transfer(mosaic::RawMosaic; …)` | 0.10.1 | the multi-frame `photon_transfer(groups, darks)` (owner ruling (B)) | wave 11 |
| VisualGeometryToolkitCore | `LmStops(; residual_resolution)`, `StaticLmStops(…; residual_resolution)` | — (renamed in wave 11, owner 2026-11-14; no deprecation, no alias) | `abstol`: the absolute tolerance on the residual norm (SciML NonlinearSolve's meaning). The old keyword is refused by name with the migration text. | wave 11 |
| VisualGeometryToolkitCore | `LmStops(; ftol)` (and `StaticLmStops`'s counterpart, if any) | — (renamed in wave 11, owner 2026-11-14; no deprecation, no alias) | `reltol`: the relative cost-change tolerance. The old keyword is refused by name with the migration text. | wave 11 |
| VisualGeometryToolkitCore | `LmStops(; xtol, stationarity_tol)`, `StaticLmStops`'s | — (surveyed in the same change, owner 2026-11-14) | the standard names, from SciML NonlinearSolve and MINPACK, adopted in the same change; no aliases; the old keywords refused by name with the migration text | wave 11 |

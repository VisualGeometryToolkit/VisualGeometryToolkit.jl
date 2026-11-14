# Wave 11 removals

The deprecated names and keywords to be removed in wave 11 (the next breaking wave). Each one still works, with a
deprecation warning, until then. Each package's CHANGELOG points here; add a row when you deprecate something.

| package | name | deprecated in | replacement / reason | removal |
|---|---|---|---|---|
| VisualGeometryToolkitFeatures | `detect_image_circle(…; min_hole_area)` | 0.9.2 | none: the hull fit it tuned is gone. The coverage is measured on the rim's log-polar profiles. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitFeatures | `detect_image_circle(…; smoothing_weight)` | 0.9.2 | none: the L0 smoothing fed the MSER path, which is gone. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitFeatures | `detect_image_circle(…; border_margin)` | 0.9.2 | none: the hull's border vertices are gone. Off-frame bins are skipped instead. Ignored with a depwarn. | wave 11 (0.10.0) |
| VisualGeometryToolkitCore | single-frame `photon_transfer(mosaic::RawMosaic; …)` | 0.10.1 | the multi-frame `photon_transfer(groups, darks)` (owner ruling (B)) | wave 11 |

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
| VisualGeometryToolkitFeatures | `gaussian_gradient(…; derivative::Symbol)` (`:gaussian2d`, `:gaussian1d`) | — (scheduled in 0.9.4; public since 0.9.1, so not changed in a patch) | a type by dispatch (the Gaussian across the derivative or none), checked by the method table rather than a Symbol test; `detect_foerstner`'s `derivative` keyword follows it | wave 11 (0.10.0) |

**Image-circle revision candidates (owner review, not scheduled):** `detect_image_circle`'s behaviours documented as
they are in VisualGeometryToolkitFeatures 0.9.4, unchanged there (its outputs are 0.9.3's bit for bit), for a future
owner-reviewed revision: (a) a bin more than half saturated is flagged (`:saturated`) but keeps its median in the profile,
where the 0.9.2 and 0.9.3 docs said it was skipped; (b) a column's `pixels` (a bin's pixel count per row, the arc) has no floor of
one pixel: inside r_max a bin counts fewer pixels than its rows, and its noise is larger by that; (c) the `:auto` ladder, when no binning has a
fit with 20 wedges with an inlier, returns the finest binning tried; (d) a noise of zero is used with no floor: every
positive drop is significant and the edges' uncertainty is 0. The hull keywords above stay on the wave-11 schedule.

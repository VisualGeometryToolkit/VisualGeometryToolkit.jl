"""
    synthetic_scenes() -> Module

The stack's shared synthetic test scenes: a module whose submodules are the
scene families — `EpipolarScenes`, `HomographyScenes`, `PoseScenes`,
`DivisionBoards`, `P3PScenes`, `Cameras` and `Contamination` — each
exporting its builders (`calibrated_pairs`, `stereo_pixel_pairs`, `fundmat_scene` (THE two-camera
fundamental-matrix scene),
`homography_pairs`, `frustum_point`, `noisy_board_views`, `p3p_scene`,
`contaminated_pairs`, …). Each draws from the caller's RNG in a fixed order
(pinned fixtures depend on it). Every stack package's tests draw from these
instead of copying them:

```julia
using Test, VisualGeometryToolkitEstimation
const EpipolarScenes = VisualGeometryToolkitEstimation.synthetic_scenes().EpipolarScenes
u₁, u₂ = EpipolarScenes.calibrated_pairs(rng, R, t; n = 50, σ = 1.0e-3)
```

Implemented by VisualGeometryToolkitEstimationTestExt: `using Test` loads it.
"""
function synthetic_scenes end

# Detection

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

```@docs
VisualGeometryToolkitFeatures
```

The detection type and its polarity helpers, the ASIFT detector and its plans,
the detection covariance, the Gaussian scale space and gradient, the MSER,
Förstner and SFOP detectors, the spatially
spread selection of detections (SSC), and the image circle of a lens measured
from photographs. The image-input
helpers and the PNG and RAW readers are on [Image Input](image.md).

```@autodocs
Modules = [VisualGeometryToolkitFeatures]
Pages = ["detection_types.jl", "mser.jl", "foerstner.jl", "batching.jl", "asift/api.jl", "asift/driver.jl",
         "asift/backproject.jl", "asift/uncertainty.jl", "asift/pyramid.jl", "asift/warp.jl",
         "asift/crosswarp.jl", "sfop.jl", "spread.jl", "image_circle.jl", "image_circle_rim.jl"]
```

The opt-in continuous refinement (`asift_detect(img; refinement = ContinuousRefinement())`):

```@docs
ContinuousRefinement
```

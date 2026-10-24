# Ellipse Fitting

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

`refine_ellipse` fits an ellipse to points by minimizing the orthogonal
distance. It starts from VisualGeometryToolkitCore's algebraic `direct_ellipse`
unless a start value is supplied as `init`.

```@docs
refine_ellipse
EllipseFitMetadata
```

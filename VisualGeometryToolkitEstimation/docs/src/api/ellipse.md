# Ellipse Fitting

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

`refine_ellipse` fits an ellipse to points by minimizing the orthogonal
distance. It starts from VisualGeometryToolkitCore's algebraic `direct_ellipse`
unless a start value is supplied as `init`.

From contaminated points, the ellipse is one more model of the RANSAC family,
as the circle ([Circle Fitting](circle.md)): `EllipseProblem` is the
`MinimalSolveProblem` of an `Ellipse{Float64}` through one image's points, with
VisualGeometryToolkitCore's minimal solver `Ellipse5pt` (the conic through five
points, kept when it is a real ellipse), the point-on-ellipse constraint
`constraint_with_jacobians(e, x)` with `g = (x − c)ᵀ Q (x − c) − 1`, and the
chart `EllipseManifold` (the centre and the log shape). The residual is the
family's Sampson distance `g/‖∂g/∂x‖`, the geometric distance to first order, in
pixels. `find_ellipse` is its RANSAC, with the robustified LM
([`Triggs`](@ref)) as its local optimization.

```@docs
refine_ellipse
EllipseFitMetadata
EllipseProblem
find_ellipse
```

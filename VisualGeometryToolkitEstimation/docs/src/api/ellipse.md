# Ellipse Fitting

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

The ellipse is one more model of the RANSAC family,
as the circle ([Circle Fitting](circle.md)): `EllipseProblem` is the
`MinimalSolveProblem` of an `Ellipse{Float64}` through one image's points, with
VisualGeometryToolkitCore's minimal solver `Ellipse5pt` (the conic through five
points, kept when it is a real ellipse), the point-on-ellipse constraint
`constraint_with_jacobians(e, x)` with `g = (x − c)ᵀ Q (x − c) − 1`, and the
chart `EllipseManifold` (the centre and the log shape). The residual is the
family's Sampson distance `g/‖∂g/∂x‖`, the geometric distance to first order, in
pixels, and it scores; the refit rows every local optimization descends are the
points' ORTHOGONAL distances to the ellipse (the distance to the foot point),
the maximum-likelihood objective under isotropic Gaussian noise on the points,
which the Sampson distance approximates only to first order (on a
high-curvature ellipse the Sampson least squares is measurably the worse
estimator). `find_ellipse` is its RANSAC, with the robustified LM
([`Triggs`](@ref)) as its local optimization; `refine_ellipse` is that local
optimization alone from a given ellipse — with `Irls{GaussianShape}()` at a
threshold above every residual, the orthogonal-distance least squares, its
covariance `s²·solver_covariance(problem, e, inliers)`.

```@docs
EllipseProblem
find_ellipse
refine_ellipse
```

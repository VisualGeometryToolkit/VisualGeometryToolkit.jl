# Circle Fitting

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

The 2D circle is one more model of the RANSAC family: `CircleProblem` is the
`MinimalSolveProblem` of a `Circle{Float64}` through one image's points. The
circle supplies only what a model supplies, all of it VisualGeometryToolkitCore's —
the minimal solver `Circle3pt` (the circumcircle of three points), the
point-on-circle constraint `constraint_with_jacobians(c, x)` with
`g = ‖x − c‖² − r²`, and the chart `CircleManifold` (the centre and log r).
The residual is the family's Sampson distance `g/‖∂g/∂x‖ = (d² − r²)/(2d)`, the
geometric distance `d − r` to first order, in the points' units (pixels).
`solve(EstimationProblem(CircleProblem(points)), Ransac(); rng)` is its find, with
the robustified LM ([`Triggs`](@extref VisualGeometryToolkitCore.Triggs)) as its
class's local optimization, and `solve(RefineProblem(CircleProblem(points), c₀),
LevenbergMarquardt(); scale | width)` its refine.

```@docs
CircleProblem
```

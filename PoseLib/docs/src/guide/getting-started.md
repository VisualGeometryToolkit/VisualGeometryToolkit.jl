# Getting started

The examples on this page are doctests: the documentation build runs them and
checks their output. They share one scene. A camera at the origin looks down
the z axis at a grid of points on a gently curved surface, about six units
away. Its intrinsic matrix `K` has a focal length of 800 pixels.

```jldoctest gs
julia> using PoseLib, VisualGeometryToolkitCore

julia> K = [800.0 0.0 320.0; 0.0 800.0 240.0; 0.0 0.0 1.0];

julia> X = [Point3(x, y, 6.0 + 0.5x * y) for x in -1.0:0.5:1.0 for y in -1.0:0.5:1.0];

julia> pixel(K, x) = (y = K * x; Point2(y[1] / y[3], y[2] / y[3]));

julia> u = [pixel(K, x) for x in X];
```

## A minimal solver

The P3P solver estimates a camera pose from three world points and their
bearings, the unit rays from the camera centre towards them. Selecting the
backend `PoseLibSolver()` runs PoseLib's solver in place of
VisualGeometryToolkitCore's native one. P3P has up to four real solutions;
one of them is the true pose, the identity.

```jldoctest gs
julia> f = [Vec3(x / sqrt(sum(abs2, x))) for x in X];

julia> poses = solve(P3P(PoseLibSolver()), Csponds(X, f), [1, 13, 25]);

julia> 1 <= length(poses) <= 4
true

julia> any(p -> maximum(abs, p.t) < 1e-8, poses)
true
```

## Refining an absolute pose

A refiner starts from a seed and polishes it. The pose refiners take and
return a [`Rigid3`](@ref PoseLib.Rigid3), which `EuclideanMap(R, t)` builds
from a rotation and a translation. Here the seed is a few pixels off.

`loss_width` is in the unit of the image points, pixels here. Under the
default truncated loss, a point whose residual exceeds it counts as an outlier
and gets zero weight. A seed must therefore be close enough that the inliers
fall within the width.

```jldoctest gs
julia> seed = EuclideanMap(RotationVec(0.002, -0.001, 0.0), Vec3(0.005, 0.0, 0.01));

julia> pose = refine_absolute_pose(seed, K, Csponds(X, u); max_iter = 50, loss_width = 2.0);

julia> maximum(abs, pose.t) < 1e-8
true
```

A P3P solution is a labeled `Extrinsics3`, a map from the world frame to the
camera frame. `unlabel` turns it into a `Rigid3` seed.

```jldoctest gs
julia> best = poses[argmin([maximum(abs, p.t) for p in poses])];

julia> refine_absolute_pose(unlabel(best), K, Csponds(X, u); max_iter = 50, loss_width = 2.0) isa PoseLib.Rigid3
true
```

## Refining a relative pose

A second camera, rotated and moved sideways by one unit, sees the same points.
The relative pose maps the first camera's frame to the second's. Its
translation is known only up to scale, and the refiner returns it at unit
length.

```jldoctest gs
julia> truth = EuclideanMap(RotationVec(0.0, 0.1, 0.0), Vec3(1.0, 0.0, 0.0));

julia> x2 = [pixel(K, truth(x)) for x in X];

julia> seed = EuclideanMap(RotationVec(0.0, 0.101, 0.001), Vec3(0.99995, 0.01, 0.0));

julia> rel = refine_relpose(seed, K, K, Csponds(u, x2); max_iter = 50, loss_width = 2.0);

julia> maximum(abs, rel.R - truth.R) < 1e-6
true
```

The same refinement runs on unit bearings, the form that also serves fisheye
cameras. Each bearing carries the Jacobian of the back-projection, which sets
the unit of `loss_width`: pixels for these Jacobians. See
[Refiners](refiners.md) for why.

```jldoctest gs
julia> using VisualGeometryToolkitCore: backproject_with_jacobian

julia> camera = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K)), Pinhole());

julia> b1 = backproject_with_jacobian.(Ref(camera), u);

julia> b2 = backproject_with_jacobian.(Ref(camera), x2);

julia> cs = withattrs(Csponds(first.(b1), first.(b2)); M₁ = last.(b1), M₂ = last.(b2));

julia> maximum(abs, refine_relpose(seed, cs; max_iter = 50, loss_width = 2.0).R - rel.R) < 1e-12
true
```

## Refining a homography

The homography refiner takes the seed matrix and the point correspondences it
should explain. The points here are exact, so the refined matrix matches the
true one up to scale.

```jldoctest gs
julia> H = HomographyMat([1.1 0.02 5.0; -0.03 0.95 -3.0; 1e-4 2e-4 1.0]);

julia> x1 = [Point2(a, b) for a in 0.0:100.0:400.0 for b in 0.0:100.0:300.0];

julia> y = [(h = H * [x; 1.0]; Point2(h[1] / h[3], h[2] / h[3])) for x in x1];

julia> H0 = HomographyMat([1.1 0.02 6.0; -0.03 0.96 -3.0; 1e-4 2e-4 1.0]);

julia> H1 = refine_homography(H0, Csponds(x1, y); max_iter = 25, loss_width = 2.0);

julia> H1 / H1[3, 3] ≈ H
true
```

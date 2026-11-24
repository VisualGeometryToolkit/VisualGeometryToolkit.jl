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
bearings, the unit rays from the camera centre towards them. Every estimator
of the stack is `solve(problem, algorithm)`; a minimal sample is a
`MinimalProblem` of the model, the correspondences and the indices of the
sample. Selecting the backend `PoseLibSolver()` runs PoseLib's solver in place
of VisualGeometryToolkitCore's native one. P3P has up to four real solutions;
one of them is the true pose, the identity.

```jldoctest gs
julia> f = [Vec3(x / sqrt(sum(abs2, x))) for x in X];

julia> poses = solve(MinimalProblem(CameraExtrinsics, Correspondences(X, f), [1, 13, 25]), P3P(PoseLibSolver()));

julia> 1 <= length(poses) <= 4
true

julia> any(p -> maximum(abs, p.t) < 1e-8, poses)
true
```

## Refining an absolute pose

A refine starts from a guess and polishes it. Its problem is a
`RefineProblem(cs, guess, p)`: the correspondences, the guess, and `p` the
context PoseLib needs, here the camera's `K`; the guess's type and the data
select PoseLib's residual. The algorithm is the stack's Levenberg–Marquardt
with PoseLib's backend, `LevenbergMarquardt(PoseLibSolver(); optimization_strategy)`,
and its iteration budget is the solve keyword `stops = LmStops(; max_iter)`
(VisualGeometryToolkitCore's; PoseLib applies `max_iter` alone). The solve
returns a `FitResult`, whose `u` is the refined model; a pose refine takes and
returns an unlabeled `EuclideanMap{3}`, which `EuclideanMap(R, t)` builds from
a rotation and a translation. Here the guess is a few pixels off.

The strategy is `Irls{TLSShape}()`, the truncated loss, and its `width` is in
the unit of the image points, pixels here. A point whose residual exceeds it
counts as an outlier and gets zero weight. A guess must therefore be close
enough that the inliers fall within the width.

```jldoctest gs
julia> using VisualGeometryToolkitCore: LmStops

julia> guess = EuclideanMap(RotationVec(0.002, -0.001, 0.0), Vec3(0.005, 0.0, 0.01));

julia> tls = LevenbergMarquardt(PoseLibSolver(); optimization_strategy = Irls{TLSShape}());

julia> problem = RefineProblem(Correspondences(X, u), guess, (; K));

julia> r = solve(problem, tls; stops = LmStops(; max_iter = 50), width = 2.0);

julia> maximum(abs, r.u.t) < 1e-8
true

julia> successful_retcode(r)
true
```

A P3P solution is a labeled `CameraExtrinsics`, a map from the world frame to the
camera frame. `unlabel` turns it into an unlabeled guess.

```jldoctest gs
julia> best = poses[argmin([maximum(abs, p.t) for p in poses])];

julia> problem = RefineProblem(Correspondences(X, u), unlabel(best), (; K));

julia> solve(problem, tls; stops = LmStops(; max_iter = 50), width = 2.0).u isa EuclideanMap{3}
true
```

## Refining a relative pose

A second camera, rotated and moved sideways by one unit, sees the same points.
The relative pose maps the first camera's frame to the second's. Its
translation is known only up to scale, and the refine returns it at unit
length. On pixel points its context is the two cameras' intrinsics,
`(; K1, K2)`.

```jldoctest gs
julia> truth = EuclideanMap(RotationVec(0.0, 0.1, 0.0), Vec3(1.0, 0.0, 0.0));

julia> x2 = [pixel(K, truth(x)) for x in X];

julia> guess = EuclideanMap(RotationVec(0.0, 0.101, 0.001), Vec3(0.99995, 0.01, 0.0));

julia> problem = RefineProblem(Correspondences(u, x2), guess, (; K1 = K, K2 = K));

julia> rel = solve(problem, tls; stops = LmStops(; max_iter = 50), width = 2.0).u;

julia> maximum(abs, rel.R - truth.R) < 1e-6
true
```

The same refine runs on unit bearings, the form that also serves fisheye
cameras; it needs no context. Each bearing carries the Jacobian of the
back-projection, which sets the unit of the width: pixels for these Jacobians.
See [Refiners](refiners.md) for why.

```jldoctest gs
julia> using VisualGeometryToolkitCore: backproject_with_jacobian

julia> camera = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K)), Lens(Perspective()));

julia> b1 = backproject_with_jacobian.(Ref(camera), u);

julia> b2 = backproject_with_jacobian.(Ref(camera), x2);

julia> cs = withattrs(Correspondences(first.(b1), first.(b2)); M₁ = last.(b1), M₂ = last.(b2));

julia> bearings = solve(RefineProblem(cs, guess), tls; stops = LmStops(; max_iter = 50), width = 2.0);

julia> maximum(abs, bearings.u.R - rel.R) < 1e-12
true
```

## Refining a homography

The homography's problem holds the point correspondences the guess, a
`HomographyMat`, should explain. The points here are exact, so the refined matrix matches the true one
up to scale.

```jldoctest gs
julia> H = HomographyMat([1.1 0.02 5.0; -0.03 0.95 -3.0; 1e-4 2e-4 1.0]);

julia> x1 = [Point2(a, b) for a in 0.0:100.0:400.0 for b in 0.0:100.0:300.0];

julia> y = [(h = H * [x; 1.0]; Point2(h[1] / h[3], h[2] / h[3])) for x in x1];

julia> H0 = HomographyMat([1.1 0.02 6.0; -0.03 0.96 -3.0; 1e-4 2e-4 1.0]);

julia> H1 = solve(RefineProblem(Correspondences(x1, y), H0), tls; stops = LmStops(; max_iter = 25), width = 2.0).u;

julia> H1 / H1[3, 3] ≈ H
true
```

## The eight-point essential matrix

From eight or more bearing correspondences, all inliers, the direct linear
transform (DLT) estimates the essential matrix; VisualGeometryToolkitCore has
no native version, so its backend is PoseLib's.

```jldoctest gs
julia> f2 = [Vec3(y / sqrt(sum(abs2, y))) for y in (truth(x) for x in X)];

julia> E = solve(EstimationProblem(EssentialMat, Correspondences(f, f2)), DLT(PoseLibSolver())).u;

julia> E isa EssentialMat{Float64}
true
```

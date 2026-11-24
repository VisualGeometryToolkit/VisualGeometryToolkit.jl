# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The geometry of PoseLib's refiner tests (test_refine.jl, test_contract.jl) and of the golden records of
# ci/golden/fixtures/poselib.jl: two pinhole cameras, a relative pose, 200 points (MersenneTwister(7)), a seed pose (the guess; `seed` is the golden records' field name);
# and test_contract.jl's scene of every refiner. Data constructors only — no refiner is called; include-once safe
# (functions only). Its includer first includes VisualGeometryToolkitCore's test/fixtures/pinhole_points.jl
# (`pinhole_point`, `pinhole_pixel`).
using Random: MersenneTwister
using LinearAlgebra: normalize
using Rotations: AngleAxis, RotMatrix
using StaticArrays: SVector, SMatrix
import PoseLib
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: Correspondences, withattrs, backproject_with_jacobian, CameraModel, LogicalIntrinsics,
    CameraCalibrationMat, Lens, Perspective, Vec3

# The shared geometry of test_refine.jl and test_contract.jl (rng MersenneTwister(7)).
function refine_geometry()
    rng = MersenneTwister(7)
    n = 200; nout = n ÷ 5
    K1 = SMatrix{3,3}(800.0, 0, 0, 0, 800.0, 0, 320.0, 240.0, 1.0)
    K2 = SMatrix{3,3}(760.0, 0, 0, 0, 760.0, 0, 330.0, 250.0, 1.0)
    R = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.2, normalize([0.3, 1.0, 0.1])...)))
    t = normalize(SVector(1.0, 0.1, 0.05))
    X = [SVector(4 * randn(rng), 3 * randn(rng), 8.0 + 4 * rand(rng)) for _ in 1:n]
    Rs = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.01, 0.0, 1.0, 0.0))) * R
    seed = VgtkCore.EuclideanMap(RotMatrix{3}(Rs), Vec3{Float64}(normalize(t + SVector(0.0, 0.02, 0.0))))
    tx = SMatrix{3,3}(0.0, t[3], -t[2], -t[3], 0.0, t[1], t[2], -t[1], 0.0)
    return (; n, nout, K1, K2, R, t, X, seed, tx)
end

# test_contract.jl's scene, one draw per refiner, its outlier rng `r2` a parameter (the suite's MersenneTwister(99); the
# golden records' StableRNG(seed) noise draws).
function contract_scene(r2)
    (; n, nout, K1, K2, R, t, X, seed, tx) = refine_geometry()
    out2() = SVector(640 * rand(r2), 480 * rand(r2))
    x1 = [pinhole_pixel(K1, Y) for Y in X]
    x2 = [pinhole_pixel(K2, R * Y + t) for Y in X]
    x2[1:nout] .= [out2() for _ in 1:nout]
    rel = Correspondences(x1, x2)
    F = inv(K2)' * tx * R * inv(K1)
    F0 = F .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(r2, 9)))
    H = K2 * (R + t * SVector(0.0, 0.0, 0.1)') * inv(K1)
    h1 = [out2() for _ in 1:n]
    h2 = [pinhole_pixel(H, SVector(u[1], u[2], 1.0)) for u in h1]
    h2[1:nout] .= [out2() for _ in 1:nout]
    H0 = H .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(r2, 9)))
    cam1 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K1)), Lens(Perspective()))
    cam2 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K2)), Lens(Perspective()))
    bm1 = backproject_with_jacobian.(Ref(cam1), x1)
    bm2 = backproject_with_jacobian.(Ref(cam2), x2)
    bear = withattrs(Correspondences(first.(bm1), first.(bm2)); M₁ = last.(bm1), M₂ = last.(bm2))
    f = 800.0
    xa = [pinhole_point(R * Y + t) for Y in X]
    xa[1:nout] .= [SVector(randn(r2), randn(r2)) for _ in 1:nout]
    ua = [SVector(f * c[1] + K1[1, 3], f * c[2] + K1[2, 3]) for c in xa]
    abs_cs = Correspondences(X, ua)
    # the same 2D-3D data as unit bearings (the chart points lifted and normalized): P3P's minimal samples
    abs_bear = Correspondences(X, [normalize(SVector(c[1], c[2], 1.0)) for c in xa])
    # the same absolute-pose data in metres (world points and seed), as the suite's physical refinement builds it
    abs_cs_m = Correspondences([Y .* VgtkCore.m for Y in X], ua)
    seedm = VgtkCore.EuclideanMap(seed.R, seed.t .* VgtkCore.m)
    return (; seed, K1, K2, rel, F0, H0, hom = Correspondences(h1, h2), bear, abs_cs, abs_bear, abs_cs_m, seedm)
end

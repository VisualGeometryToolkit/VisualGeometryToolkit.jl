# Fixtures of the PoseLib family: the scenes of PoseLib's suite (test_contract.jl's `_baseline_data`, test_refine.jl's
# two-view / 2D-3D scene and its per-testset draws, the typed-loss homography, runtests.jl's smoke data, test_p3p.jl's
# lookat cameras and test_p3p_poselib.jl's reference rays), built from data constructors only.
using Test, StableRNGs, StaticArrays, StructArrays, Random
using LinearAlgebra: normalize, norm
using Random: MersenneTwister, Xoshiro
using Rotations: AngleAxis, RotMatrix
using VisualGeometryToolkitCore
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: Correspondences, withattrs, backproject_with_jacobian, s2_tangent_basis, CameraModel,
    LogicalIntrinsics, CameraCalibrationMat, Pinhole, Vec3, Point3, PosedCamera
using PoseLib

_pl_nrm(Y) = SVector(Y[1] / Y[3], Y[2] / Y[3])
_pl_proj(K, Y) = _pl_nrm(K * Y)

# The shared geometry of test_refine.jl and test_contract.jl (rng MersenneTwister(7)).
function _pl_geometry()
    rng = MersenneTwister(7)
    n = 200; nout = n ÷ 5
    K1 = SMatrix{3,3}(800.0, 0, 0, 0, 800.0, 0, 320.0, 240.0, 1.0)
    K2 = SMatrix{3,3}(760.0, 0, 0, 0, 760.0, 0, 330.0, 250.0, 1.0)
    R = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.2, normalize([0.3, 1.0, 0.1])...)))
    t = normalize(SVector(1.0, 0.1, 0.05))
    X = [SVector(4 * randn(rng), 3 * randn(rng), 8.0 + 4 * rand(rng)) for _ in 1:n]
    Rs = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.01, 0.0, 1.0, 0.0))) * R
    seed = PoseLib.Rigid3(RotMatrix{3}(Rs), Vec3{Float64}(normalize(t + SVector(0.0, 0.02, 0.0))))
    tx = SMatrix{3,3}(0.0, t[3], -t[2], -t[3], 0.0, t[1], t[2], -t[1], 0.0)
    return (; n, nout, K1, K2, R, t, X, seed, tx)
end

# test_contract.jl's `_baseline_data()`, verbatim, its outlier rng `r2` a parameter: the suite's MersenneTwister(99)
# ("test"), or a StableRNG(seed) noise draw.
function poselib_contract_scene(r2)
    (; n, nout, K1, K2, R, t, X, seed, tx) = _pl_geometry()
    out2() = SVector(640 * rand(r2), 480 * rand(r2))
    x1 = [_pl_proj(K1, Y) for Y in X]
    x2 = [_pl_proj(K2, R * Y + t) for Y in X]
    x2[1:nout] .= [out2() for _ in 1:nout]
    rel = Correspondences(x1, x2)
    F = inv(K2)' * tx * R * inv(K1)
    F0 = F .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(r2, 9)))
    H = K2 * (R + t * SVector(0.0, 0.0, 0.1)') * inv(K1)
    h1 = [out2() for _ in 1:n]
    h2 = [_pl_proj(H, SVector(u[1], u[2], 1.0)) for u in h1]
    h2[1:nout] .= [out2() for _ in 1:nout]
    H0 = H .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(r2, 9)))
    cam1 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K1)), Pinhole())
    cam2 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K2)), Pinhole())
    bm1 = backproject_with_jacobian.(Ref(cam1), x1)
    bm2 = backproject_with_jacobian.(Ref(cam2), x2)
    bear = withattrs(Correspondences(first.(bm1), first.(bm2)); M₁ = last.(bm1), M₂ = last.(bm2))
    f = 800.0
    xa = [_pl_nrm(R * Y + t) for Y in X]
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

poselib_contract_scenes() = [
    "test" => poselib_contract_scene(MersenneTwister(99)),
    ("seed$s" => poselib_contract_scene(StableRNG(s)) for s in SEEDS)...,
]

# test_refine.jl's scene: every testset's own draw, in the suite's order of rngs.
function poselib_refine_scene()
    (; n, nout, K1, K2, R, t, X, seed, tx) = _pl_geometry()
    outlier2(r) = SVector(640 * rand(r), 480 * rand(r))
    I3 = SMatrix{3,3}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)
    # z = 1 chart (refused)
    chart = (x1 = _pl_nrm.(X), x2 = [_pl_nrm(R * Y + t) for Y in X])
    # camera-aware relative pose, pixels (rng 101)
    rngc = MersenneTwister(101)
    x1 = [_pl_proj(K1, Y) for Y in X]
    x2 = [_pl_proj(K2, R * Y + t) for Y in X]
    x2[1:nout] .= [outlier2(rngc) for _ in 1:nout]
    n1 = [_pl_nrm(inv(K1) * SVector(u[1], u[2], 1.0)) for u in x1]
    n2 = [_pl_nrm(inv(K1) * SVector(u[1], u[2], 1.0)) for u in [_pl_proj(K1, R * Y + t) for Y in X]]
    p2 = [_pl_proj(K1, R * Y + t) for Y in X]
    n2[1:nout] .= [_pl_nrm(inv(K1) * SVector(u[1], u[2], 1.0)) for u in x2[1:nout]]
    p2[1:nout] .= x2[1:nout]
    relcam = (; cs = Correspondences(x1, x2), cs_K1 = Correspondences(x1, p2), cs_cal = Correspondences(n1, n2), I3)
    # fundamental (rng 102)
    rngf = MersenneTwister(102)
    F = inv(K2)' * tx * R * inv(K1)
    fx1 = [_pl_proj(K1, Y) for Y in X]
    fx2 = [_pl_proj(K2, R * Y + t) for Y in X]
    fx2[1:nout] .= [outlier2(rngf) for _ in 1:nout]
    F0 = F .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(rngf, 9)))
    fund = (; F0, cs = Correspondences(fx1, fx2), F0_2x3 = Matrix(F0)[1:2, :])
    # homography (rng 103)
    rngh = MersenneTwister(103)
    H = K2 * (R + t * SVector(0.0, 0.0, 0.1)') * inv(K1)
    hx1 = [outlier2(rngh) for _ in 1:n]
    hx2 = [_pl_proj(H, SVector(u[1], u[2], 1.0)) for u in hx1]
    hx2[1:nout] .= [outlier2(rngh) for _ in 1:nout]
    H0 = H .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(rngh, 9)))
    hom = (; H0, cs = Correspondences(hx1, hx2), H0_2x2 = Matrix(H0)[1:2, 1:2])
    # bearings with tangent Jacobians (rng 31)
    rngb = MersenneTwister(31)
    Kb2 = SMatrix{3,3}(1100.0, 0, 0, 0, 1100.0, 0, 330.0, 250.0, 1.0)
    u1 = [_pl_proj(K1, Y) for Y in X]
    u2 = [_pl_proj(Kb2, R * Y + t) for Y in X]
    u2[1:nout] .= [SVector(660 * rand(rngb), 500 * rand(rngb)) for _ in 1:nout]
    cam1 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K1)), Pinhole())
    cam2 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(Kb2)), Pinhole())
    bm1 = backproject_with_jacobian.(Ref(cam1), u1)
    bm2 = backproject_with_jacobian.(Ref(cam2), u2)
    f1 = first.(bm1); M₁ = last.(bm1)
    f2 = first.(bm2); M₂ = last.(bm2)
    chart1 = [SVector(f[1] / f[3], f[2] / f[3], 1.0) for f in f1]
    chart2 = [SVector(f[1] / f[3], f[2] / f[3], 1.0) for f in f2]
    nan1 = copy(f1); nan1[3] = SVector(NaN, NaN, NaN)
    bearings = (; Kb2, pix = Correspondences(u1, u2), cs = withattrs(Correspondences(f1, f2); M₁, M₂),
                no_jacobians = Correspondences(f1, f2), chart_sources = withattrs(Correspondences(chart1, f2); M₁, M₂),
                chart_targets = withattrs(Correspondences(f1, chart2); M₁, M₂), nan_source = withattrs(Correspondences(nan1, f2); M₁, M₂))
    # bearings past 90° (rngs 11, 13)
    rngf2 = MersenneTwister(11); rngn = MersenneTwister(13)
    m = 150
    Xs = [(v = normalize(SVector(randn(rngf2), randn(rngf2), randn(rngf2))); (3.0 + 4rand(rngf2)) * v) for _ in 1:m]
    Xs[1] = SVector(5.0, 0.0, 0.0)
    Rg = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.15, normalize([0.2, 1.0, 0.4])...)))
    tg = normalize(SVector(0.6, -0.2, 0.3))
    tb(f) = (b = s2_tangent_basis(SVector{3,Float64}(f)); SMatrix{3,2,Float64,6}(b[1][1], b[1][2], b[1][3], b[2][1], b[2][2], b[2][3]))
    jitter(f, σ) = (b = s2_tangent_basis(SVector{3,Float64}(f)); normalize(SVector{3,Float64}(f) + σ * (randn(rngn) * b[1] + randn(rngn) * b[2])))
    g1 = [normalize(Y) for Y in Xs]
    g2 = [jitter(normalize(Rg * Y + tg), deg2rad(0.02)) for Y in Xs]
    Rsg = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.02, 0.0, 1.0, 0.0))) * Rg
    seedf = PoseLib.Rigid3(RotMatrix{3}(Rsg), Vec3{Float64}(normalize(tg + SVector(0.0, 0.03, 0.0))))
    past90 = (; seed = seedf, cs = withattrs(Correspondences(g1, g2); M₁ = tb.(g1), M₂ = tb.(g2)),
              chart = Correspondences([SVector(f[1] / f[3], f[2] / f[3]) for f in g1], [SVector(f[1] / f[3], f[2] / f[3]) for f in g2]))
    # camera-aware absolute pose, pixels (rng 104)
    rnga = MersenneTwister(104)
    fa = (K1[1,1] + K1[2,2]) / 2
    xcal = [_pl_nrm(R * Y + t) for Y in X]
    xcal[1:nout] .= [SVector(randn(rnga), randn(rnga)) for _ in 1:nout]
    upix = [SVector(fa * c[1] + K1[1,3], fa * c[2] + K1[2,3]) for c in xcal]
    Xm = [Y .* VgtkCore.m for Y in X]
    seedm = VgtkCore.EuclideanMap(seed.R, seed.t .* VgtkCore.m)
    absolute = (; f = fa, cs = Correspondences(X, upix), cs_m = Correspondences(Xm, upix), seedm,
                labeled = VgtkCore.CameraExtrinsics(seedm), K_3x4 = hcat(Matrix(K1), zeros(3)),
                Kbad = SMatrix{3,3}(fa, 0, 0, 0, fa, 0, 0.0, 0.0, 1.0), chart = Correspondences(X, [_pl_nrm(R * Y + t) for Y in X]))
    # unit bearings of the scene (essential_dlt, the 5-point kernel) and their chart (refused by type)
    b1 = [normalize(Y) for Y in X]
    b2 = [normalize(R * Y + t) for Y in X]
    c1 = [_pl_nrm(Y) for Y in X]
    c2 = [_pl_nrm(R * Y + t) for Y in X]
    return (; seed, K1, K2, chart, relcam, fund, hom, bearings, past90, absolute, b1, b2, c1, c2, X)
end

# test_refine.jl's typed-loss homography (rng 211): a Euclidean H, 3 px outliers, and the 1.2–1.5 px noise.
function poselib_typed_loss_scene()
    rngw = MersenneTwister(211)
    θ = 0.1
    He = SMatrix{3,3}(cos(θ), sin(θ), 0.0, -sin(θ), cos(θ), 0.0, 12.0, -7.0, 1.0)
    m = 120; mout = 20
    y1 = [SVector(640 * rand(rngw), 480 * rand(rngw)) for _ in 1:m]
    exact = [_pl_proj(He, SVector(u[1], u[2], 1.0)) for u in y1]
    H0 = He + SMatrix{3,3}(0.0, 0, 0, 0, 0, 0, 0.05, -0.05, 0)
    offset = SVector(3.0, 0.0)
    with_outliers = copy(exact); with_outliers[1:mout] .+= Ref(offset)
    noisy = [x + (1.2 + 0.3 * rand(rngw)) * (a = 2π * rand(rngw); SVector(cos(a), sin(a))) for x in exact]
    noisy_out = copy(noisy); noisy_out[1:mout] .+= Ref(offset)
    return (; H0, outliers = Correspondences(y1, with_outliers), noisy = Correspondences(y1, noisy),
            noisy_out = Correspondences(y1, noisy_out))
end

# runtests.jl's smoke data: three points seen by the identity pose; a near-square homography sample; five random
# bearings per view (unseeded in the suite: drawn here from StableRNG(seed)).
function poselib_smoke_scene()
    X = [Point3(1.0, 0.5, 5.0), Point3(-0.5, 1.0, 4.0), Point3(0.0, -0.5, 6.0)]
    rays = [Vec3(x ./ norm(x)) for x in X]
    s = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)]
    d = [SVector(0.1, 0.1), SVector(1.1, 0.0), SVector(1.0, 1.1), SVector(0.0, 1.0)]
    five(rng) = ntuple(_ -> SVector{3,Float64}(normalize([randn(rng), randn(rng), 1.0])), Val(5))
    bearings5 = [(rng = StableRNG(seed); (five(rng), five(rng))) for seed in SEEDS]
    return (; X, rays, s, d, bearings5)
end

# test_p3p.jl's lookat scenes (each testset's `Random.seed!(k)` is the task RNG at Xoshiro(k); the suite draws
# nothing after seeding but the random trials) and test_p3p_poselib.jl's reference rays.
function poselib_p3p_scenes()
    s = sensor("Sony", "IMX174")
    f = focal_length(s, deg2rad(40.0); dimension = :horizontal)
    pp = s.resolution ./ 2
    model = CameraModel(f, s.pitch, pp)
    seen(X, eye, target) = (cam = PosedCamera(model, lookat(eye, target, Vec3(0.0, -1.0, 0.0)));
                            (backproject.(Ref(model), project.(Ref(cam), X)), X))
    out = Pair{String, Any}[]
    rng = Xoshiro(42)
    for trial in 1:10
        X = [Point3(rand(rng, 100.0:700.0), rand(rng, 100.0:500.0), 0.0) for _ in 1:3]
        eye = Point3(rand(rng, 200.0:600.0), rand(rng, 200.0:400.0), rand(rng, 1500.0:2500.0))
        push!(out, "random_trial$trial" => seen(X, eye, Point3(400.0, 300.0, 0.0)))
    end
    eye = Point3(400.0, 300.0, 2000.0)
    push!(out, "collinear" => seen([Point3(100.0, 100.0, 0.0), Point3(200.0, 200.0, 0.0), Point3(300.0, 300.0, 0.0)], eye, Point3(200.0, 200.0, 0.0)))
    cam = PosedCamera(model, lookat(eye, Point3(200.0, 200.0, 0.0), Vec3(0.0, -1.0, 0.0)))
    Xsmall = [Point3(400.0, 300.0, 0.0), Point3(400.1, 300.0, 0.0), Point3(400.0, 300.1, 0.0)]
    push!(out, "small_triangle" => (backproject.(Ref(model), project.(Ref(cam), Xsmall)), Xsmall))
    push!(out, "multiple" => seen([Point3(0.0, 0.0, 0.0), Point3(100.0, 0.0, 0.0), Point3(50.0, 100.0, 0.0)],
        Point3(50.0, 50.0, 500.0), Point3(50.0, 50.0, 0.0)))
    Xmm = [Point3(100.0, 100.0, 0.0), Point3(500.0, 100.0, 0.0), Point3(300.0, 400.0, 0.0)]
    push!(out, "scale_mm" => seen(Xmm, Point3(300.0, 250.0, 2000.0), Point3(300.0, 250.0, 0.0)))
    push!(out, "scale_x10" => seen([Point3(p[1] * 10.0, p[2] * 10.0, p[3] * 10.0) for p in Xmm],
        Point3(3000.0, 2500.0, 20000.0), Point3(3000.0, 2500.0, 0.0)))
    push!(out, "poselib_reference" => (
        [SVector(-0.623795, 0.113042, 0.773370), SVector(-0.429263, -0.138400, 0.892513), SVector(-0.691148, -0.095083, 0.716431)],
        [SVector(406.59, 153.83, 0.00), SVector(80.96, 455.07, 0.00), SVector(564.71, 432.72, 0.00)]))
    return out
end

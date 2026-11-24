@testitem "units_boundary" setup = [PoseScenes] begin
    using VisualGeometryToolkitCore: weights, noise_scale, scale_source
    using VisualGeometryToolkitCore: solve, EstimationProblem
    using VisualGeometryToolkitEstimation: Ransac, HomographyProblem
    using VisualGeometryToolkitCore: retcode_symbol
    using VisualGeometryToolkitCore: inlier_mask
# The units layering at the public estimators (VGC conventions/units.md): a
# physical input (lengths) is stripped in metres once at entry, the inner loop
# runs plain, the model is reattached in metres once at exit. A physical input
# must give, after `ustrip(m, ·)`, BIT FOR BIT the plain result on the inputs'
# `ustrip(m, ·)` with the same seed; one built in millimetres gives the plain
# result on its stored metres. The problems and the inner loop refuse a
# physical value (`MethodError`), and the plane-model fits refuse physical
# plane points (their matrix cannot record the unit).
#
# Inputs "from millimetres" are built by multiplication (`x .* mm`):
# VisualGeometryToolkitIO's `read_length` is not a dependency this package may
# have (the stack allow-list), and an estimator sees only the stored metres
# either way — which is what these items pin.

using Test
using StableRNGs
using LinearAlgebra: I, normalize
using StaticArrays: SVector, SMatrix
using Rotations: AngleAxis, RotMatrix, RotXYZ
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
using VisualGeometryToolkitCore: EuclideanMap, CameraModel, Correspondences, HyperRectangle, HomographyMat, Vec2, Vec3, rms, rotation, translation, project, m, mm, ustrip, DimensionError
using VisualGeometryToolkitEstimation: ReweightedLeastSquares, RobotWorldHandEyeProblem, RelativePoseProblem, RobotWorldHandEyeCalibration, AbsolutePoseProblem, HandEyeProblem, RadialCameraFit, DivisionCameraFit
using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

# A plain pose with its translation read in `u`: Core's attachment.
phys(E, u = m) = VgtkCore.with_unit(E, u)
phys(E::AbstractVector, u = m) = phys.(E, Ref(u))
same_pose(a, b) = rotation(a) == rotation(b) && translation(a) == translation(b)
# The parts of two RANSAC results that must agree bit for bit (the model apart).
same_diagnostics(a, b) = a.retcode === b.retcode && a.stats.nsteps == b.stats.nsteps && a.rms === b.rms &&
    inlier_mask(a) == inlier_mask(b) && a.resid == b.resid && scale_source(a) === scale_source(b) && (scale_source(a) === VgtkCore.ScaleNone || noise_scale(a) === noise_scale(b)) && weights(a) == weights(b)

@testset "Ransac on a AbsolutePoseProblem" begin
    rng = StableRNG(9)
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    R, t = RotMatrix(RotXYZ(0.15, -0.25, 0.1)), Vec3{Float64}(0.3, -0.2, 0.6)
    obs = [Pair(frustum_point(rng, cam, R, t)...) for _ in 1:80]
    obs = [(k % 5 == 0 ? first(p) => last(p) + SVector(40.0, -25.0) : p) for (k, p) in enumerate(obs)]   # 20 % outliers
    X, u = first.(obs), last.(obs)

    plain = solve(EstimationProblem(AbsolutePoseProblem(cam, X .=> u)), Ransac(); rng = StableRNG(1))
    phy = solve(EstimationProblem(AbsolutePoseProblem(cam, [x .* m for x in X] .=> u)), Ransac(); rng = StableRNG(1))
    @test VgtkCore.has_units(phy.u) && !VgtkCore.has_units(plain.u)
    @test same_pose(ustrip(m, phy.u), plain.u)
    @test same_diagnostics(phy, plain)
    # the labels of the model type survive the round trip (world → camera)
    frames(E) = typeof(E).parameters[5:6]
    @test frames(phy.u) == frames(plain.u)

    # built in millimetres: the plain search on the stored metres
    Xmm = [(1000 .* x) .* mm for x in X]
    from_mm = solve(EstimationProblem(AbsolutePoseProblem(cam, Xmm .=> u)), Ransac(); rng = StableRNG(1))
    stored = solve(EstimationProblem(AbsolutePoseProblem(cam, [map(q -> ustrip(m, q), x) for x in Xmm] .=> u)), Ransac(); rng = StableRNG(1))
    @test same_pose(ustrip(m, from_mm.u), stored.u)
    @test isapprox(translation(stored.u), translation(plain.u); rtol = 1.0e-9)
    # plain millimetres and physical millimetres: the P3P coincidence gate is
    # 1e-3 in the problem's world unit (1 mm for physical input, stripped in
    # metres; 1 μm for plain millimetres), which no sample of a fixture spread
    # over metres meets — the same samples, the same answer in either form
    plain_mm = solve(EstimationProblem(AbsolutePoseProblem(cam, [1000 .* x for x in X] .=> u)), Ransac(); rng = StableRNG(1))
    @test inlier_mask(plain_mm) == inlier_mask(from_mm) && plain_mm.stats.nsteps == from_mm.stats.nsteps
    @test isapprox(rotation(plain_mm.u), rotation(from_mm.u); atol = 1.0e-12)
    @test isapprox(translation(plain_mm.u), 1000 .* translation(ustrip(m, from_mm.u)); rtol = 1.0e-9)

    # a Correspondences keeps its columns: the PROSAC score orders the same samples
    s = collect(range(1.0, 0.1; length = length(X)))
    @test same_pose(
        ustrip(m, solve(EstimationProblem(AbsolutePoseProblem(cam, Correspondences([x .* m for x in X], u; score = s))), Ransac(); rng = StableRNG(2)).u),
        solve(EstimationProblem(AbsolutePoseProblem(cam, Correspondences(X, u; score = s))), Ransac(); rng = StableRNG(2)).u
    )
    # a physical extent goes with physical points
    ext = HyperRectangle(Vec2(-2.0, -2.0), Vec2(4.0, 4.0))
    ext_m = HyperRectangle(Vec2(-2.0, -2.0) .* m, Vec2(4.0, 4.0) .* m)
    @test same_pose(
        ustrip(m, solve(EstimationProblem(AbsolutePoseProblem(cam, [x .* m for x in X] .=> u), (; extent = ext_m)), Ransac(); rng = StableRNG(3)).u),
        solve(EstimationProblem(AbsolutePoseProblem(cam, X .=> u), (; extent = ext)), Ransac(); rng = StableRNG(3)).u
    )
    # an extent of the other form than the points is no extent of the problem (no method)
    @test_throws MethodError solve(EstimationProblem(AbsolutePoseProblem(cam, [x .* m for x in X] .=> u), (; extent = ext)), Ransac())
    @test_throws MethodError solve(EstimationProblem(AbsolutePoseProblem(cam, X .=> u), (; extent = ext_m)), Ransac())
    # fewer than a minimal sample: nothing, either form
    @test let r = solve(EstimationProblem(AbsolutePoseProblem(cam, [x .* m for x in X[1:2]] .=> u[1:2])), Ransac())
        retcode_symbol(r) === :no_model && r.stats.nsteps == 0 && r.u === nothing
    end
    # the problem strips physical points in metres once, at construction: the same problem as the plain metres'
    @test AbsolutePoseProblem(cam, [x .* m for x in X] .=> u).cs == AbsolutePoseProblem(cam, X .=> u).cs
end

# AX = XB: rig poses A_i and pattern poses B_i = Y⁻¹ A_i X with noise.
function hand_eye_data(rng; N = 40)
    sc = robot_world_hand_eye_scene(rng; N, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1), A = (ang_max = π / 3, t_scale = 0.5))
    A = [perturb_pose(rng, a; σ_R = 1.0e-3, σ_t = 1.0e-3) for a in sc.A]
    B = [perturb_pose(rng, marker_pose(sc.Y, a, only(sc.X)); σ_R = 1.0e-3, σ_t = 1.0e-3) for a in sc.A]
    return A, B
end

@testset "Ransac on a HandEyeProblem" begin
    A, B = hand_eye_data(StableRNG(11))
    plain = solve(EstimationProblem(HandEyeProblem(A .=> B)), Ransac(); rng = StableRNG(5))
    phy = solve(EstimationProblem(HandEyeProblem(phys(A) .=> phys(B))), Ransac(); rng = StableRNG(5))
    @test VgtkCore.has_units(phy.u)
    @test same_pose(ustrip(m, phy.u), plain.u)
    @test same_diagnostics(phy, plain)
    # lengths beside physical poses are lengths, and a scorer's width is in metres: the same search as the plain
    # numbers in metres
    sc2mm = VgtkCore.MsacScoring(HandEyeProblem; threshold = ustrip(m, 2.0mm))
    @test same_pose(
        ustrip(m, solve(EstimationProblem(HandEyeProblem(phys(A) .=> phys(B); char_length = 0.4m)), Ransac(; scoring = sc2mm); rng = StableRNG(6)).u),
        solve(EstimationProblem(HandEyeProblem(A .=> B; char_length = 0.4)), Ransac(; scoring = sc2mm); rng = StableRNG(6)).u
    )
    @test_throws ArgumentError solve(EstimationProblem(HandEyeProblem(phys(A) .=> phys(B))), Ransac(; threshold = 2.0mm))   # a width is its scorer's
    @test_throws DimensionError solve(EstimationProblem(HandEyeProblem(phys(A) .=> phys(B); char_length = 0.4)), Ransac())
    # built in millimetres: the plain search on the stored metres
    Amm = phys([EuclideanMap(rotation(a), 1000 .* translation(a)) for a in A], mm)
    Bmm = phys([EuclideanMap(rotation(b), 1000 .* translation(b)) for b in B], mm)
    @test same_pose(
        ustrip(m, solve(EstimationProblem(HandEyeProblem(Amm .=> Bmm)), Ransac(); rng = StableRNG(5)).u),
        solve(EstimationProblem(HandEyeProblem([ustrip(m, a) for a in Amm] .=> [ustrip(m, b) for b in Bmm])), Ransac(); rng = StableRNG(5)).u
    )
    # a mix of forms is a MethodError; physical poses are stripped in metres by the problem, once
    @test_throws MethodError HandEyeProblem(A .=> phys(B))
    @test HandEyeProblem(phys(A) .=> phys(B)).motions == HandEyeProblem(A .=> B).motions
end

@testset "robot-world fit" begin
    rng = StableRNG(12)
    (; Y, X, A) = robot_world_hand_eye_scene(rng; N = 20, M = 3, Y = (t_scale = 0.5,), X = (ang_max = 0.5, t_scale = 0.3), A = (t_scale = 1.0,))
    markers = Dict(k => [a => perturb_pose(rng, marker_pose(Y, a, X[k]); σ_R = 1.0e-3, σ_t = 1.0e-3) for a in A] for k in 1:3)
    markers_m = Dict(k => [phys(first(c)) => phys(last(c)) for c in cs] for (k, cs) in markers)
    plain = rwhe_fit(markers)
    phy = rwhe_fit(markers_m)
    cal, calu = plain.u, phy.u
    @test calu isa RobotWorldHandEyeCalibration{Int} && VgtkCore.has_units(calu.Y) && !VgtkCore.has_units(cal.Y)
    @test calu.ids == cal.ids
    @test same_pose(ustrip(m, calu.Y), cal.Y) && all(k -> same_pose(ustrip(m, calu.X[k]), cal.X[k]), cal.ids)
    @test phy.rms === plain.rms && phy.metadata.cov == plain.metadata.cov && VgtkCore.s2(phy) === VgtkCore.s2(plain)
    # σ_trans is a length beside physical poses
    @test rwhe_fit(markers_m; σ_trans = 1.0mm).metadata.cov ==
        rwhe_fit(markers; σ_trans = ustrip(m, 1.0mm)).metadata.cov
    @test_throws DimensionError rwhe_fit(markers_m; σ_trans = 1.0e-3)
    # the calibration's constructor: one form, a physical one stored in metres
    c = RobotWorldHandEyeCalibration(phys(Y), Dict(k => phys(X[k]) for k in 1:3))
    @test VgtkCore.has_units(c.Y) && same_pose(ustrip(m, c.Y), RobotWorldHandEyeCalibration(Y, Dict(k => X[k] for k in 1:3)).Y)
    @test_throws MethodError RobotWorldHandEyeCalibration(phys(Y), Dict(k => X[k] for k in 1:3))
end

@testset "robot-world reprojection refine" begin
    rng = StableRNG(13)
    camera = CameraModel(; focal = 600.0, pp = (400.0, 300.0))
    Y = rigid(AngleAxis(2.0, normalize(SVector(0.3, -1.0, 0.4))...), SVector(0.2, -0.1, 0.3))
    X = [rigid(AngleAxis(0.2, normalize(randn(rng, 3))...), SVector(0.08 * (k - 1.5), 0.0, 0.0)) for k in 1:2]
    grid = [SVector(0.03 * (i - 3), 0.03 * (j - 3), 0.0) for i in 1:5 for j in 1:5]
    A = [Y ∘ rigid(AngleAxis(0.35 * rand(rng), normalize(randn(rng, 3))...), SVector(0.1 * randn(rng), 0.1 * randn(rng), 1.2)) for _ in 1:8]
    markers = Dict(
        k => [
                a => [P => SVector{2, Float64}(project(camera, marker_pose(Y, a, X[k])(P))...) + 0.3 * randn(rng, SVector{2, Float64}) for P in grid]
                for a in A
            ] for k in 1:2
    )
    guess = RobotWorldHandEyeCalibration(perturb_pose(rng, Y; σ_R = 0.01, σ_t = 0.01), Dict(k => perturb_pose(rng, X[k]; σ_R = 0.01, σ_t = 0.01) for k in 1:2))
    markers_m = Dict(k => [phys(first(e)) => [(P .* m) => q for (P, q) in last(e)] for e in els] for (k, els) in markers)
    guess_m = RobotWorldHandEyeCalibration(phys(guess.Y), Dict(k => phys(guess.X[k]) for k in guess.ids))
    plain = rwhe_refine(guess, camera, markers; optimization_strategy = nothing)
    phy = rwhe_refine(guess_m, camera, markers_m; optimization_strategy = nothing)
    @test VgtkCore.has_units(phy.u.Y)
    @test same_pose(ustrip(m, phy.u.Y), plain.u.Y)
    @test all(k -> same_pose(ustrip(m, phy.u.X[k]), plain.u.X[k]), plain.u.ids)
    @test phy.retcode === plain.retcode && phy.stats.nsteps == plain.stats.nsteps && phy.rms === plain.rms
    @test phy.metadata.cov == plain.metadata.cov
    # one form: a physical guess with plain observations is a MethodError
    @test_throws MethodError rwhe_refine(guess_m, camera, markers; optimization_strategy = nothing)
    @test_throws MethodError rwhe_refine(guess, camera, markers_m; optimization_strategy = nothing)
end

@testset "the plane-model fits refuse physical plane points" begin
    src = [SVector(0.1 * i, 0.07 * j) for i in 1:4 for j in 1:4]
    dst = [SVector(100.0 + 900 * p[1], 50.0 + 880 * p[2]) for p in src]
    phy = [p .* m for p in src] .=> dst
    @test_throws ArgumentError solve(EstimationProblem(HomographyProblem(phy)), Ransac())
    # the fix it names strips in metres, the unit of the stack's other boundaries
    msg = try solve(EstimationProblem(HomographyProblem(phy)), Ransac()); "" catch e; sprint(showerror, e) end
    @test occursin("ustrip.(m, X)", msg) && !occursin("ustrip.(mm", msg)
    @test_throws ArgumentError solve(EstimationProblem(RadialCamera, phy, (; center = SVector(400.0, 300.0))), RadialCameraFit())
    @test_throws ArgumentError solve(EstimationProblem(CameraModel, (Dict(1 => HomographyMat(SMatrix{3, 3, Float64}(I))), Dict(1 => phy)), (; focal_seed = 500.0)), DivisionCameraFit())
    # plain plane points are unchanged
    @test solve(EstimationProblem(HomographyProblem(src .=> dst)), Ransac(); rng = StableRNG(1)) !== nothing
end

@testset "a relative-pose seed is plain (its translation is a direction)" begin
    seed = rigid(RotMatrix(RotXYZ(0.01, 0.02, 0.0)), SVector(1.0, 0.0, 0.0))
    K = SMatrix{3, 3, Float64}(500, 0, 0, 0, 500, 0, 320, 240, 1)
    cs = [SVector(100.0 + i, 100.0 + 2i) => SVector(110.0 + i, 100.0 + 2i) for i in 1:10]
    @test_throws MethodError solve(VgtkCore.RefineProblem(RelativePoseProblem(K, K, cs), phys(seed)), VgtkCore.LevenbergMarquardt())
end
end

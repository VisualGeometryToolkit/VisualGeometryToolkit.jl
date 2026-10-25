@testitem "units_boundary" setup = [PoseScenes] begin
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
using VisualGeometryToolkitCore: EuclideanMap, CameraModel, Csponds, HyperRectangle, HomographyMat, Vec2, Vec3,
    estimate, retcode, iterations, rms, metadata, rotation, translation, project, m, mm, ustrip, DimensionError
using VisualGeometryToolkitEstimation: find_absolute_pose, find_hand_eye, fit_robot_world_hand_eye,
    refine_robot_world_hand_eye, RobotWorldHandEyeCalibration, AbsolutePoseProblem, HandEyeProblem,
    find_homography, fit_radial_camera, fit_division_camera

# A plain pose with its translation read in `u` (construction: the attachment).
phys(E, u = m) = EuclideanMap(rotation(E), translation(E) .* u)
phys(E::AbstractVector, u = m) = phys.(E, Ref(u))
same_pose(a, b) = rotation(a) == rotation(b) && translation(a) == translation(b)
# The parts of two RANSAC results that must agree bit for bit (the model apart).
same_diagnostics(a, b) = retcode(a) === retcode(b) && iterations(a) == iterations(b) && rms(a) === rms(b) &&
    a.inlier_mask == b.inlier_mask && a.residuals == b.residuals && a.sigma === b.sigma

@testset "find_absolute_pose" begin
    rng = StableRNG(9)
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    R, t = RotMatrix(RotXYZ(0.15, -0.25, 0.1)), Vec3{Float64}(0.3, -0.2, 0.6)
    obs = [Pair(frustum_point(rng, cam, R, t)...) for _ in 1:80]
    obs = [(k % 5 == 0 ? first(p) => last(p) + SVector(40.0, -25.0) : p) for (k, p) in enumerate(obs)]   # 20 % outliers
    X, u = first.(obs), last.(obs)

    plain = find_absolute_pose(cam, X .=> u; rng = StableRNG(1))
    phy = find_absolute_pose(cam, [x .* m for x in X] .=> u; rng = StableRNG(1))
    @test VgtkCore.has_units(estimate(phy)) && !VgtkCore.has_units(estimate(plain))
    @test same_pose(ustrip(m, estimate(phy)), estimate(plain))
    @test same_diagnostics(phy, plain)
    # the labels of the model type survive the round trip (world → camera)
    frames(E) = typeof(E).parameters[5:6]
    @test frames(estimate(phy)) == frames(estimate(plain))

    # built in millimetres: the plain search on the stored metres
    Xmm = [(1000 .* x) .* mm for x in X]
    from_mm = find_absolute_pose(cam, Xmm .=> u; rng = StableRNG(1))
    stored = find_absolute_pose(cam, [map(q -> ustrip(m, q), x) for x in Xmm] .=> u; rng = StableRNG(1))
    @test same_pose(ustrip(m, estimate(from_mm)), estimate(stored))
    @test isapprox(translation(estimate(stored)), translation(estimate(plain)); rtol = 1.0e-9)

    # a Csponds keeps its columns: the PROSAC score orders the same samples
    s = collect(range(1.0, 0.1; length = length(X)))
    @test same_pose(
        ustrip(m, estimate(find_absolute_pose(cam, Csponds([x .* m for x in X], u; score = s); rng = StableRNG(2)))),
        estimate(find_absolute_pose(cam, Csponds(X, u; score = s); rng = StableRNG(2)))
    )
    # a physical extent goes with physical points
    ext = HyperRectangle(Vec2(-2.0, -2.0), Vec2(4.0, 4.0))
    ext_m = HyperRectangle(Vec2(-2.0, -2.0) .* m, Vec2(4.0, 4.0) .* m)
    @test same_pose(
        ustrip(m, estimate(find_absolute_pose(cam, [x .* m for x in X] .=> u; extent = ext_m, rng = StableRNG(3)))),
        estimate(find_absolute_pose(cam, X .=> u; extent = ext, rng = StableRNG(3)))
    )
    @test_throws TypeError find_absolute_pose(cam, [x .* m for x in X] .=> u; extent = ext)
    @test_throws TypeError find_absolute_pose(cam, X .=> u; extent = ext_m)
    # fewer than a minimal sample: nothing, either form
    @test find_absolute_pose(cam, [x .* m for x in X[1:2]] .=> u[1:2]) === nothing
    # the problem is the inner loop: physical points are a MethodError
    @test_throws MethodError AbsolutePoseProblem(cam, [x .* m for x in X] .=> u)
end

# AX = XB: rig poses A_i and pattern poses B_i = Y⁻¹ A_i X with noise.
function hand_eye_data(rng; N = 40)
    sc = robot_world_hand_eye_scene(rng; N, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1), A = (ang_max = π / 3, t_scale = 0.5))
    A = [perturb_pose(rng, a; σ_R = 1.0e-3, σ_t = 1.0e-3) for a in sc.A]
    B = [perturb_pose(rng, marker_pose(sc.Y, a, only(sc.X)); σ_R = 1.0e-3, σ_t = 1.0e-3) for a in sc.A]
    return A, B
end

@testset "find_hand_eye" begin
    A, B = hand_eye_data(StableRNG(11))
    plain = find_hand_eye(A .=> B; rng = StableRNG(5))
    phy = find_hand_eye(phys(A) .=> phys(B); rng = StableRNG(5))
    @test VgtkCore.has_units(estimate(phy))
    @test same_pose(ustrip(m, estimate(phy)), estimate(plain))
    @test same_diagnostics(phy, plain)
    # lengths beside physical poses are lengths: the same search as the plain numbers in metres
    @test same_pose(
        ustrip(m, estimate(find_hand_eye(phys(A) .=> phys(B); threshold = 2.0mm, char_length = 0.4m, rng = StableRNG(6)))),
        estimate(find_hand_eye(A .=> B; threshold = ustrip(m, 2.0mm), char_length = 0.4, rng = StableRNG(6)))
    )
    @test_throws DimensionError find_hand_eye(phys(A) .=> phys(B); threshold = 0.002)
    @test_throws DimensionError find_hand_eye(phys(A) .=> phys(B); char_length = 0.4)
    # built in millimetres: the plain search on the stored metres
    Amm = phys([EuclideanMap(rotation(a), 1000 .* translation(a)) for a in A], mm)
    Bmm = phys([EuclideanMap(rotation(b), 1000 .* translation(b)) for b in B], mm)
    @test same_pose(
        ustrip(m, estimate(find_hand_eye(Amm .=> Bmm; rng = StableRNG(5)))),
        estimate(find_hand_eye([ustrip(m, a) for a in Amm] .=> [ustrip(m, b) for b in Bmm]; rng = StableRNG(5)))
    )
    # a mix of forms, and a physical pose reaching the problem, are MethodErrors
    @test_throws MethodError find_hand_eye(A .=> phys(B))
    @test_throws MethodError HandEyeProblem(phys(A) .=> phys(B))
end

@testset "fit_robot_world_hand_eye" begin
    rng = StableRNG(12)
    (; Y, X, A) = robot_world_hand_eye_scene(rng; N = 20, M = 3, Y = (t_scale = 0.5,), X = (ang_max = 0.5, t_scale = 0.3), A = (t_scale = 1.0,))
    markers = Dict(k => [a => perturb_pose(rng, marker_pose(Y, a, X[k]); σ_R = 1.0e-3, σ_t = 1.0e-3) for a in A] for k in 1:3)
    markers_m = Dict(k => [phys(first(c)) => phys(last(c)) for c in cs] for (k, cs) in markers)
    plain = fit_robot_world_hand_eye(markers)
    phy = fit_robot_world_hand_eye(markers_m)
    cal, calu = estimate(plain), estimate(phy)
    @test calu isa RobotWorldHandEyeCalibration{Int} && VgtkCore.has_units(calu.Y) && !VgtkCore.has_units(cal.Y)
    @test calu.ids == cal.ids
    @test same_pose(ustrip(m, calu.Y), cal.Y) && all(k -> same_pose(ustrip(m, calu.X[k]), cal.X[k]), cal.ids)
    @test rms(phy) === rms(plain) && metadata(phy).cov == metadata(plain).cov && metadata(phy).s2 === metadata(plain).s2
    # σ_trans is a length beside physical poses
    @test metadata(fit_robot_world_hand_eye(markers_m; σ_trans = 1.0mm)).cov ==
        metadata(fit_robot_world_hand_eye(markers; σ_trans = ustrip(m, 1.0mm))).cov
    @test_throws DimensionError fit_robot_world_hand_eye(markers_m; σ_trans = 1.0e-3)
    # the calibration's constructor: one form, a physical one stored in metres
    c = RobotWorldHandEyeCalibration(phys(Y), Dict(k => phys(X[k]) for k in 1:3))
    @test VgtkCore.has_units(c.Y) && same_pose(ustrip(m, c.Y), RobotWorldHandEyeCalibration(Y, Dict(k => X[k] for k in 1:3)).Y)
    @test_throws MethodError RobotWorldHandEyeCalibration(phys(Y), Dict(k => X[k] for k in 1:3))
end

@testset "refine_robot_world_hand_eye" begin
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
    plain = refine_robot_world_hand_eye(guess, camera, markers)
    phy = refine_robot_world_hand_eye(guess_m, camera, markers_m)
    @test VgtkCore.has_units(estimate(phy).Y)
    @test same_pose(ustrip(m, estimate(phy).Y), estimate(plain).Y)
    @test all(k -> same_pose(ustrip(m, estimate(phy).X[k]), estimate(plain).X[k]), estimate(plain).ids)
    @test retcode(phy) === retcode(plain) && iterations(phy) == iterations(plain) && rms(phy) === rms(plain)
    @test metadata(phy).cov == metadata(plain).cov
    # one form: a physical guess with plain observations is a MethodError
    @test_throws MethodError refine_robot_world_hand_eye(guess_m, camera, markers)
    @test_throws MethodError refine_robot_world_hand_eye(guess, camera, markers_m)
end

@testset "the plane-model fits refuse physical plane points" begin
    src = [SVector(0.1 * i, 0.07 * j) for i in 1:4 for j in 1:4]
    dst = [SVector(100.0 + 900 * p[1], 50.0 + 880 * p[2]) for p in src]
    phy = [p .* m for p in src] .=> dst
    @test_throws ArgumentError find_homography(phy)
    # the fix it names strips in metres, the unit of the stack's other boundaries
    msg = try find_homography(phy); "" catch e; sprint(showerror, e) end
    @test occursin("ustrip.(m, X)", msg) && !occursin("ustrip.(mm", msg)
    @test_throws ArgumentError fit_radial_camera(phy; center = SVector(400.0, 300.0))
    @test_throws ArgumentError fit_division_camera(
        Dict(1 => HomographyMat(SMatrix{3, 3, Float64}(I))), Dict(1 => phy); focal_seed = 500.0
    )
    # plain plane points are unchanged
    @test find_homography(src .=> dst; rng = StableRNG(1)) !== nothing
end

@testset "a relative-pose seed is plain (its translation is a direction)" begin
    seed = rigid(RotMatrix(RotXYZ(0.01, 0.02, 0.0)), SVector(1.0, 0.0, 0.0))
    K = SMatrix{3, 3, Float64}(500, 0, 0, 0, 500, 0, 320, 240, 1)
    cs = [SVector(100.0 + i, 100.0 + 2i) => SVector(110.0 + i, 100.0 + 2i) for i in 1:10]
    @test_throws MethodError VgtkEstimation.bundle_refine_relpose(phys(seed), K, K, cs)
end
end

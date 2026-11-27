# The optical axis of a SplineLens camera: the fit-then-refine recipe of a radially symmetric family onto a gauge-K source frees the
# principal point and the axis; the camera re-gauged so ẑ is that axis (Core's axis_gauged).

@testsetup module OffAxisSpline
using StaticArrays: SA, SVector
using Rotations: RotMatrix, AngleAxis
using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, KannalaBrandt, Ellipse, Rect, px
export FISHEYE, ON_AXIS, Q, OFF_AXIS, OFF_BOUNDED, REGION, DISC

# a Kannala–Brandt fisheye (f = 300 px, its axis at the pixel (500, 500)), its 40 × 40 spline; then the same spline in
# a camera frame turned 5° about a tilted axis, gauged as a never-fitted spline is (its K's principal point the pixel
# imaging the turned frame's ẑ, ~26 px from the axis's pixel)
const FISHEYE = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
const ON_AXIS = CameraModel(SplineLens, FISHEYE, Size2(1000, 1000); grid_size = (40, 40))
const Q = RotMatrix(AngleAxis(deg2rad(5.0), 0.6, 0.8, 0.0))
const OFF_AXIS = CameraModel(SplineLens, map(c -> Q * c, ON_AXIS.lens.backward.controls), Size2(1000, 1000))
const OFF_BOUNDED = BoundedCamera(OFF_AXIS, Size2(1000, 1000))
const DISC = Ellipse(SVector(500.0, 500.0), 480.0, 480.0, 0.0)   # the lit disc: the bound a lens coverage gives
const REGION = Rect(0, 0, 1000, 1000) ∩ DISC
end

@testitem "OpticalAxis: a spline turned 5° off its axis — the axis and the principal point recovered, every image kept after the re-gauge" setup = [CameraRecipes, OffAxisSpline] begin
    import VisualGeometryToolkitCore as VgtkCore
    
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, dot, normalize
    using Rotations: RotMatrix, AngleAxis
    using VisualGeometryToolkitCore: calibration_matrix, rotation, map_frames, ViewFrame, project, backproject, axis_gauged, EuclideanMap, relabel, WorldFrame, CameraFrame, model, with_lens_coverage, imaged_region
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

    K0 = calibration_matrix(OFF_AXIS)
    @test norm(SVector(K0[1, 3], K0[2, 3]) - SVector(500.0, 500.0)) > 20          # the never-fitted gauge: off the axis's pixel
    r = solve(EstimationProblem(CameraRegressionProblem(with_lens_coverage(OFF_BOUNDED, DISC))), OpticalAxis())
    @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize)
    R = r.metadata.rotation
    @test map_frames(R) == (ViewFrame{:source}, ViewFrame{:fitted})
    axis = rotation(R)' * SVector(0.0, 0.0, 1.0)                                   # the axis in the source frame
    @test acosd(min(1.0, dot(axis, Q * SVector(0.0, 0.0, 1.0)))) < 1.0e-3          # the turned frame's true axis, Q·ẑ
    K = calibration_matrix(r.u)
    pp = SVector(K[1, 3], K[2, 3])
    @test norm(pp - SVector(500.0, 500.0)) < 0.02                                  # the axis's pixel
    @test r.metadata.symmetry_residual < 1.0e-3 && r.metadata.max_px < 0.05      # a symmetric lens: nothing azimuthal left
    # the fit-then-refine recipe's fit of the family it names, its decentering groups held, from the whole-order start: the same
    # axis and principal point (OpticalAxis reaches the order by continuation, 1 → 4 terms)
    r2 = solve(EstimationProblem(CameraRegressionProblem(typeof(r.u.lens), OFF_BOUNDED, imaged_region(with_lens_coverage(OFF_BOUNDED, DISC)); fixed = (:p, :sx, :sy))), CameraRegression(; refine = LevenbergMarquardt()))
    @test acosd(min(1.0, dot(rotation(r2.metadata.rotation)' * SVector(0.0, 0.0, 1.0), axis))) < 1.0e-4
    @test norm(SVector(calibration_matrix(r2.u)[1, 3], calibration_matrix(r2.u)[2, 3]) - pp) < 1.0e-3
    # the re-gauge: ẑ is the axis, the principal point its pixel; every image kept with the poses turned
    g = axis_gauged(OFF_AXIS, rotation(R), K)
    @test norm(project(g, SVector(0.0, 0.0, 1.0)) - pp) < 1.0e-6
    @test norm(normalize(backproject(g, pp)) - SVector(0.0, 0.0, 1.0)) < 1.0e-4
    pose = relabel(WorldFrame, CameraFrame, EuclideanMap(RotMatrix(AngleAxis(0.3, 0.2, -1.0, 0.4)), (0.1, -0.2, 2.0)))
    turned = axis_gauged(pose, rotation(R))
    X = [SVector(x, y, 0.0) for x in -0.8:0.2:0.8 for y in -0.6:0.2:0.6]
    @test all(x -> norm(project(g, turned(x)) - project(OFF_AXIS, pose(x))) < 1.0e-6, X)
end

@testitem "OpticalAxis: a decentred lens reports its symmetry residual; a physical K is its own axis" setup = [CameraRecipes, OffAxisSpline] begin
    using Test
    using StaticArrays: SA, SVector
    using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, Lens, Equidistant, Radial, Additive, Tangential, rotation, calibration_matrix, px, KannalaBrandt, NoncentralSplineModel, ChartSpline, with_lens_coverage
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

    # a fisheye with a tangential (decentring) term: no radially symmetric camera about any axis images it
    dec = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = Lens(Equidistant(), Additive(Radial((-0.02,)), Tangential(2.0e-3, -1.5e-3))))
    spl = BoundedCamera(CameraModel(SplineLens, dec, Size2(1000, 1000); grid_size = (40, 40)), Size2(1000, 1000))
    rd = solve(EstimationProblem(CameraRegressionProblem(with_lens_coverage(spl, DISC))), OpticalAxis())
    rs = solve(EstimationProblem(CameraRegressionProblem(with_lens_coverage(OFF_BOUNDED, DISC))), OpticalAxis())
    @test rd.metadata.symmetry_residual > 0.1                                     # px: reported, not absorbed
    @test rd.metadata.symmetry_residual > 100 * rs.metadata.symmetry_residual
    # a physical K: its principal point and its ẑ are the source's own — the identity, not fitted
    rp = solve(EstimationProblem(CameraRegressionProblem(BoundedCamera(FISHEYE, Size2(1000, 1000); lens_coverage = DISC))), OpticalAxis())
    @test rotation(rp.metadata.rotation) == one(rotation(rp.metadata.rotation))
    @test calibration_matrix(rp.u)[1:2, 3] == calibration_matrix(FISHEYE)[1:2, 3]
    @test isnan(rp.metadata.symmetry_residual)                                     # a radial profile has no azimuth to measure
    @test typeof(rp.metadata.rotation) === typeof(rs.metadata.rotation)          # one rotation type for every source
    # a non-central source: not supported yet
    G = ON_AXIS
    nc = NoncentralSplineModel(G.intrinsics, G.lens.backward, ChartSpline(G.lens.backward.area, fill(SVector(0.0, 0.0, 0.0), size(G.lens.backward.controls))))
    @test_throws r"not supported yet" solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, nc, REGION)), CameraRegression())
    @test_throws r"not supported yet" solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, BoundedCamera(nc, Size2(1000, 1000)), REGION)), CameraRegression())
    @test_throws r"not supported yet" solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, nc, REGION)), CameraRegression(; refine = LevenbergMarquardt()))
    @test_throws r"not supported yet" solve(RefineProblem(CameraRegressionProblem(nc, REGION), rp.u), LevenbergMarquardt())
end

@testitem "CameraRegression(SplineLens): a gauge-K source is gauged on its optical axis first; a physical source is unchanged" setup = [OffAxisSpline] begin
    import VisualGeometryToolkitCore as VgtkCore
    
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: rotation, calibration_matrix, backproject, project, lens, SplineLens, BoundedCamera, Size2
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, OFF_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-3))
    cam = r.u
    R = rotation(r.metadata.rotation)
    oa = solve(EstimationProblem(CameraRegressionProblem(OFF_BOUNDED)), OpticalAxis())                    # the gauge reads the source's bound, not the fit's region
    @test R == rotation(oa.metadata.rotation) && r.metadata.symmetry_residual == oa.metadata.symmetry_residual
    @test calibration_matrix(cam)[1:2, 3] == calibration_matrix(oa.u)[1:2, 3]
    # the backward is the source's turned by R, exactly as axis_gauged turns it; the forward fitted on it
    @test lens(cam).backward.controls == map(c -> R * c, lens(OFF_AXIS).backward.controls)
    pix = [SVector(500.0 + ρ * cos(a), 500.0 + ρ * sin(a)) for ρ in (0.0, 150.0, 400.0) for a in range(0, 2π; length = 9)]
    @test all(u -> norm(backproject(cam, u) - R * backproject(OFF_AXIS, u)) < 1.0e-12, pix)
    @test all(u -> norm(project(cam, backproject(cam, u)) - u) < 1.0e-3, pix)
    @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.ToleranceMet) && r.metadata.family.forward.max_px <= 1.0e-3
    # a parametric source: its ẑ is the axis already — the identity, the source's K
    rp = solve(EstimationProblem(CameraRegressionProblem(SplineLens, BoundedCamera(FISHEYE, Size2(1000, 1000)), REGION)), CameraRegression(; tolerance = 1.0e-2, forward = false))
    @test rotation(rp.metadata.rotation) == one(R) && isnan(rp.metadata.symmetry_residual)
    @test calibration_matrix(rp.u) == calibration_matrix(FISHEYE)
end

@testitem "OpticalAxis: a bounded camera only — its region is the camera's own bound or a given region within it; a circular fisheye's axis from its image circle" setup = [OffAxisSpline] begin
    import VisualGeometryToolkitCore as VgtkCore
    
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, dot, normalize, I
    using Rotations: RotMatrix, AngleAxis
    using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, Ellipse, Rect, KannalaBrandt, backproject, px, rotation, calibration_matrix, imaged_region
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

    # a bare model has no bound: refused, with the route (the region is the camera's bound, never a default)
    @test_throws r"BoundedCamera" solve(EstimationProblem(CameraRegressionProblem(OFF_AXIS)), OpticalAxis())
    # a circular fisheye (an equidistant f = 250 px lens imaging to 100°, its circle r ≈ 436 px on a 1400 × 1000 sensor;
    # its frame turned 3° off its axis): its spline images nothing past the circle, so its corners are the lattice's
    # extrapolation (as the RC golden siemens grid's). Over its image-circle bound the axis is found; over the whole
    # sensor the corners' made-up rays break the fit
    fish = CameraModel(250.0px, SA[700.0px, 500.0px]; lens = KannalaBrandt(0.0))
    Q3 = RotMatrix(AngleAxis(deg2rad(3.0), 1.0, -0.4, 0.0))
    θmax = deg2rad(100.0)
    f(u) = (d = backproject(fish, u); acos(clamp(d[3], -1, 1)) <= θmax ? Q3 * d : nothing)
    spl = CameraModel(SplineLens, f, Size2(1400, 1000); grid_size = (30, 22), init = :quasi_interpolate)
    circle = Ellipse(SVector(700.0, 500.0), 0.95 * 250 * θmax, 0.95 * 250 * θmax, 0.0)
    r = solve(EstimationProblem(CameraRegressionProblem(BoundedCamera(spl, Size2(1400, 1000); lens_coverage = circle))), OpticalAxis())
    @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    axis = rotation(r.metadata.rotation)' * SVector(0.0, 0.0, 1.0)
    @test acosd(min(1.0, dot(axis, Q3 * SVector(0.0, 0.0, 1.0)))) < 0.01
    K = calibration_matrix(r.u)
    @test norm(SVector(K[1, 3], K[2, 3]) - SVector(700.0, 500.0)) < 0.1
    @test r.metadata.symmetry_residual < 0.05
    whole = solve(EstimationProblem(CameraRegressionProblem(BoundedCamera(spl, Size2(1400, 1000)))), OpticalAxis())
    @test whole.metadata.symmetry_residual > 10 * r.metadata.symmetry_residual
    # an explicit region (0.10.2): the fit over `region ∩ imaged_region(camera)`. The whole-sensor camera over the image
    # circle finds the axis its coverage-bound twin finds (RadiometricCamera's siemens start camera stalled over its
    # whole image and converges over its board's rectangle: the same case on a real lattice)
    rc = solve(EstimationProblem(CameraRegressionProblem(BoundedCamera(spl, Size2(1400, 1000)), circle)), OpticalAxis())
    @test (rc.retcode, rc.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    @test acosd(min(1.0, dot(rotation(rc.metadata.rotation)' * SVector(0.0, 0.0, 1.0), Q3 * SVector(0.0, 0.0, 1.0)))) < 0.01
    @test rc.metadata.symmetry_residual < 0.05
    # a region past the camera's image is clipped to it: the sensor rectangle on the coverage-bound camera is its bound
    bounded = BoundedCamera(spl, Size2(1400, 1000); lens_coverage = circle)
    rr = solve(EstimationProblem(CameraRegressionProblem(bounded, Rect(0, 0, 1400, 1000))), OpticalAxis())
    @test (rr.retcode, rr.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease) && length(rr.metadata.residuals) == length(r.metadata.residuals)
    @test calibration_matrix(rr.u) == K       # the same pixels, in the same order: the same problem, the same bits
    # no region: the camera's own imaged region, the one path, the same bits
    @test solve(EstimationProblem(CameraRegressionProblem(bounded, imaged_region(bounded))), OpticalAxis()).metadata.residuals == r.metadata.residuals
end

# OpticalAxis's LM takes every chart LM's stop settings (Core's [lm] ftol = 1e-10, Estimation 0.10.2; until 0.10.1 it ran at
# the chart LM's eps): on the whole sensor of the 5°-off-axis spline the default family's cost settles before its
# steps reach the increment floor, so it stops on the relative cost decrease (at eps: IncrementSize, one iteration
# later; the principal point moves by < 1e-6 px; wave10/optical-axis-order.md)
@testitem "OpticalAxis: the chart LM's cost tolerance ends a settled regression (CostDecrease)" setup = [OffAxisSpline] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore: CostDecrease
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    r = solve(EstimationProblem(CameraRegressionProblem(OFF_BOUNDED)), OpticalAxis())
    @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    @test r.stop_rule === CostDecrease
end

@testitem "the camera-regression refine's regression on the device: the CPU and CUDA backends of the one kernel agree to rounding" setup = [CameraRecipes, OffAxisSpline] begin
    import VisualGeometryToolkitCore as VgtkCore
    
    using Test
    using CUDA: CUDA
    using StaticArrays: SMatrix
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VgtkCore, rotation, calibration_matrix, parameter_vector, lens, imaged_region, with_lens_coverage, KannalaBrandt, model, free_parameter_names, tangent_size, view_angle_limit
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VgtkEstimation, CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # the reduction kernel (each sample's 2-D residual and Jacobian rows, Σ‖r‖², Jᵀr, JᵀJ) is the same KernelAbstractions
    # kernel on every backend: on the SAME samples the CPU's and the GPU's regressions agree to the reduction's rounding
    # (the work items' sums in another order), with the same steps. Since 0.11.0 `device` also draws a gauge-K source's
    # samples (its rays back-projected on that device, E6), so the public calls no longer share them; the kernel alone
    # is compared on the host's samples, and the public call on a physical K, whose samples are its host-side profile.
    cpu = VgtkCore.KA.CPU()
    # a physical K: the samples are the lens's radial profile, on the host whatever the device
    rc = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, FISHEYE, REGION; device = cpu)), CameraRegression(; refine = LevenbergMarquardt()))
    @test (rc.retcode, rc.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize)
    # a gauge K (the off-axis spline over its disc): the regression state built as the camera-regression refine builds it, its samples
    # the host's, then the one kernel on each backend
    b = with_lens_coverage(OFF_BOUNDED, DISC)
    src = model(b); region = imaged_region(b); L = typeof(VgtkEstimation._AXIS_LENS); fixed = VgtkEstimation._AXIS_FIXED
    role = VgtkCore.Cameras.IntrinsicsRole(lens(src)); K0 = SMatrix{3, 3, Float64}(calibration_matrix(src)); M = VgtkEstimation._offset_matrix(K0)
    guess = VgtkEstimation._approximation_start(VgtkEstimation._linear_form(L), L, src, region, VgtkEstimation.ImageMeasure(), fixed, cpu)
    gl = lens(guess); moving = free_parameter_names(guess, fixed)
    start = VgtkEstimation._refine_start(role, SMatrix{3, 3, Float64}(calibration_matrix(guess)), K0, M)
    pr = VgtkEstimation._profile_samples(role, L, src, region, K0, VgtkEstimation.ImageMeasure(), cpu)
    P = VgtkEstimation._regression_manifold(role, VgtkEstimation._lens_family_manifold(role, gl, moving), start, moving)
    N = tangent_size(P)
    θcap = (t = Float64(view_angle_limit(gl)) - 1.0e-9; t < pr.θ_max - 1.0e-6 ? t : Inf)
    function regress(dev)
        lm = VgtkEstimation._LensRegressionLm(dev, P, false, θcap, M, VgtkEstimation._regression_device_data(dev, pr, tangent_size(P)))
        x, record, its = VgtkEstimation._regress_lens(lm, Val(N), VgtkCore.LmStops(; max_iter = VgtkEstimation._dflt("camera_regression_problem", "max_iter")))
        e = K0[1, 1] .* VgtkEstimation._sample_errors(lm, x, θcap)                                    # px
        return (; x, record, its, rms = sqrt(sum(i -> pr.w[i] * sum(abs2, e[i]), eachindex(e)) / sum(pr.w)), max_px = maximum(norm, e))
    end
    c = regress(cpu)
    @test c.record[1] == VgtkCore.ReturnCode.Success
    if CUDA.functional()
        gpu = VgtkCore.default_device()
        g = regress(gpu)
        @test g.record == c.record && g.its == c.its
        @test maximum(abs, g.x - c.x) < 1.0e-9                                                  # chart coordinates
        @test abs(g.rms - c.rms) < 1.0e-12 && abs(g.max_px - c.max_px) < 1.0e-12                 # px
        # the physical K's public call: the same samples on both backends, so the same comparison end to end
        rg = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, FISHEYE, REGION; device = gpu)), CameraRegression(; refine = LevenbergMarquardt()))
        @test (rg.retcode, rg.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize) && rg.stats.nsteps == rc.stats.nsteps
        @test maximum(abs, calibration_matrix(rg.u) - calibration_matrix(rc.u)) < 1.0e-9
        @test maximum(abs, collect(parameter_vector(lens(rg.u))) - collect(parameter_vector(lens(rc.u)))) < 1.0e-9
        @test abs(rg.rms - rc.rms) < 1.0e-12 && abs(rg.metadata.max_px - rc.metadata.max_px) < 1.0e-12   # px
    end
end

# The axis chart's rotation is Rotations' `rotation_between(a, ẑ)`, bit for bit, on both of its branches (the regular
# one and the axis opposite ẑ, about a perpendicular vector), and under duals.
@testitem "the axis chart's rotation: Rotations' rotation_between(a, ẑ) bit for bit" begin
    using Test
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize
    import Rotations, ForwardDiff
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VgtkEstimation
    z = SVector(0.0, 0.0, 1.0)
    ref(a) = SMatrix{3, 3}(Rotations.rotation_between(a, z))
    for a in (z, -z, SVector(1.0, 0.0, 0.0), SVector(0.0, -1.0, 0.0), normalize(SVector(0.3, -0.2, 0.9)),
              normalize(SVector(-0.7, 0.1, -0.4)), normalize(SVector(1.0e-9, 2.0e-9, -1.0)), normalize(SVector(0.2, 0.2, 0.1)))
        @test VgtkEstimation._axis_rotation(a) == ref(a)
        @test VgtkEstimation._axis_rotation(a) * a ≈ z atol = 1.0e-8    # (the half turn of an axis within rounding of −ẑ is exact to the axis's own offset)
    end
    for u in (SVector(0.0, 0.0, -1.0), SVector(0.5, -0.5, 0.5), SVector(-2.0, 1.0, 1.0), SVector(0.1, 0.3, -0.2), SVector(1.0, 1.0, 1.0))
        @test VgtkEstimation._perpendicular(u) == Rotations.perpendicular_vector(u)
    end
    a0 = normalize(SVector(0.3, -0.2, 0.9))
    J = ForwardDiff.jacobian(a -> vec(VgtkEstimation._axis_rotation(a)), a0)
    @test J == ForwardDiff.jacobian(a -> vec(ref(a)), a0)
end

# The optical axis of a gauge-K source and the exact resample that turns onto it, on a CUDA device: the reduction
# kernel of the axis fit (each sample's residual and its Jacobian rows by duals of the chart vector) compiles for the
# device and gives the host's result. (Until 0.12.1 the public calls were tested on the CPU backend only, and the
# kernel's normal-equations row did not compile for CUDA with the axis fit's chart.)
@testitem "OpticalAxis and CameraRegression(SplineLens) of a gauge-K source on a CUDA device: the host's result" setup = [OffAxisSpline] begin
    using Test
    using CUDA: CUDA
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VgtkCore, imaged_region, SplineLens, rotation, calibration_matrix
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    if CUDA.functional()
        cpu = VgtkCore.KA.CPU(); gpu = VgtkCore.default_device()
        region = imaged_region(OFF_BOUNDED)
        rc = solve(EstimationProblem(CameraRegressionProblem(OFF_BOUNDED; device = cpu)), OpticalAxis())
        rg = solve(EstimationProblem(CameraRegressionProblem(OFF_BOUNDED; device = gpu)), OpticalAxis())
        @test VgtkCore.successful_retcode(rc) && VgtkCore.successful_retcode(rg)
        # the two devices draw the gauge-K source's samples themselves (its rays back-projected there), so the fits
        # agree to the samples' rounding, not bit for bit
        @test maximum(abs, calibration_matrix(rg.u) - calibration_matrix(rc.u)) < 1.0e-6
        fc = solve(EstimationProblem(CameraRegressionProblem(SplineLens, OFF_BOUNDED, region; device = cpu)), CameraRegression(; tolerance = 1.0e-2))
        fg = solve(EstimationProblem(CameraRegressionProblem(SplineLens, OFF_BOUNDED, region; device = gpu)), CameraRegression(; tolerance = 1.0e-2))   # its optical axis first, on the device
        @test VgtkCore.successful_retcode(fc) && VgtkCore.successful_retcode(fg)
        @test maximum(abs, calibration_matrix(fg.u) - calibration_matrix(fc.u)) < 1.0e-6
    else
        @test_skip "no functional CUDA device: the device path of OpticalAxis is not exercised"
    end
end

# The caller's `device` reaches a SplineLens source's ray sampling — the expensive step: with the default device's
# resolution made to throw, every camera verb given `device` still runs (until 0.11.0 `OpticalAxis` took none, and
# the samples of a gauge-K source were drawn on the default device whatever the caller gave).
@testitem "device: a SplineLens source's rays are sampled on the caller's device — the default is never used" setup = [CameraRecipes, OffAxisSpline] begin
    using Test
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VgtkCore, imaged_region, KannalaBrandt, SplineLens
    struct _NoDefaultBackend <: VgtkCore.KA.Backend end
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    cpu = VgtkCore.KA.CPU()
    region = imaged_region(OFF_BOUNDED)
    # the default device, for the block, a backend nothing can run on (Core's one scope of the default): a call the
    # caller's `device` does not reach fails on it (no method), so every call below runs on `cpu`
    VgtkCore.with_default_device(_NoDefaultBackend()) do
        r = solve(EstimationProblem(CameraRegressionProblem(OFF_BOUNDED; device = cpu)), OpticalAxis())
        @test VgtkCore.successful_retcode(r)
        @test VgtkCore.successful_retcode(solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, OFF_BOUNDED, region; device = cpu)), CameraRegression()))
        @test VgtkCore.successful_retcode(solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, OFF_BOUNDED, region; device = cpu)), CameraRegression(; refine = LevenbergMarquardt())))
        @test solve(RefineProblem(CameraRegressionProblem(OFF_AXIS, region; device = cpu), r.u), LevenbergMarquardt()).u isa VgtkCore.CameraModel   # from its own optimum: runs on `cpu`
        @test VgtkCore.successful_retcode(solve(EstimationProblem(CameraRegressionProblem(SplineLens, OFF_BOUNDED, region; device = cpu)), CameraRegression(; tolerance = 1.0e-2)))   # its optical axis first
    end
end

# The axis family's radial order is a configurable literal (configs/default.toml [OpticalAxis] radial_terms, read once):
# 4 terms, θ³…θ⁹, the owner's ruling of 2026-11-13 from the wave-10 order study (
# the smallest order whose principal point is within ½σ of 6 terms' on every converging RC camera, cond(J) 47–160×
# lower). Until 0.11.0 it was Fisheye624's 6 (θ³…θ¹³).
@testitem "OpticalAxis: the axis family's radial order is the config's (4 terms)" setup = [OffAxisSpline] begin
    using Test
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VgtkCore, Lens, Equidistant, Additive, Tangential, ThinPrism, Radial, lens, parameter_names, with_lens_coverage
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VgtkEstimation, CameraRegressionProblem, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    n = VgtkEstimation._dflt("optical_axis", "radial_terms")
    @test n == 4
    # the order continuation's first order: the 1-term linear fit and refine, then 2, 3 and 4 terms, each from the last
    @test VgtkEstimation._dflt("optical_axis", "start_radial_terms") == 1
    @test typeof(VgtkEstimation._AXIS_LENS) === typeof(Lens(Equidistant(), Additive(Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0))) ∘ Radial(ntuple(_ -> 0.0, n))))
    names = parameter_names(lens(solve(EstimationProblem(CameraRegressionProblem(with_lens_coverage(OFF_BOUNDED, DISC))), OpticalAxis()).u))
    @test :k4 in names && !(:k5 in names)
end

# The region-support guard (Core's `check_region_support`, the owner's ruling of 2026-11-13): a fit over a region that
# reaches beyond the source camera's calibrated support is REFUSED (`RegionBeyondSupport`), never fitted over the model's
# extrapolation. RadiometricCamera's siemens start camera is a re-gridded copy of a Schöps calibration whose support is
# Rect(1616, 593, 2159, 2321): over its whole sensor (corners extrapolated to 179.8°) the axis fit stalled at every
# radial order (wave10/optical-axis-order.md); over the support it converges.
@testitem "region support: a fit beyond the source's calibrated support is refused; inside it, the axis converges" setup = [CameraRecipes] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using StaticArrays: SVector, SMatrix
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VgtkCore, CameraModel, SplineLens, ChartSpline, BoundedCamera, Size2, Rect, LogicalIntrinsics, CameraCalibrationMat, RegionBeyondSupport, imaged_region, model, KannalaBrandt
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    c = include(joinpath(@__DIR__, "fixtures", "rc_siemens_start_camera.jl"))
    C = [SVector{3, Float64}(c.controls[(i - 1) * c.gw + j]) for i in 1:c.gh, j in 1:c.gw]
    K = SMatrix{3, 3, Float64}(c.fx, 0.0, 0.0, c.skew, c.fy, 0.0, c.cx, c.cy, 1.0)
    cam = BoundedCamera(CameraModel(LogicalIntrinsics(CameraCalibrationMat(K; check = false)), SplineLens(ChartSpline(Rect(c.area...), C))),
                        Size2(c.width, c.height))
    support = Rect(1616.0, 593.0, 2159.0, 2321.0)   # the Schöps source's valid_domain
    # the whole sensor: refused by each fit at its entry, naming the fit
    for (who, f) in (("OpticalAxis", () -> solve(EstimationProblem(CameraRegressionProblem(cam; support)), OpticalAxis())),
                     ("CameraRegression", () -> solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, cam, imaged_region(cam); support)), CameraRegression(; refine = LevenbergMarquardt()))),
                     ("CameraRegression", () -> solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, cam, imaged_region(cam); support)), CameraRegression())))
        e = try f(); nothing catch err; err end
        @test e isa RegionBeyondSupport && e.who == who && e.support == support
    end
    # the support itself: the axis converges
    r = solve(EstimationProblem(CameraRegressionProblem(cam, support; support)), OpticalAxis())
    @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    # the camera-regression refine refuses the whole sensor too, and fits inside
    @test_throws RegionBeyondSupport solve(RefineProblem(CameraRegressionProblem(cam, imaged_region(cam); support), r.u), LevenbergMarquardt())
    @test (let rs = solve(RefineProblem(CameraRegressionProblem(cam, support; support), r.u), LevenbergMarquardt()); (rs.retcode, rs.stop_rule) end) === (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    # without `support`, a camera's own valid_domain bounds it: a region past a lens's injectivity disc is refused
    kb = CameraModel(SMatrix{3, 3, Float64}(300, 0, 0, 0, 300, 0, 320, 240, 1); lens = KannalaBrandt(-0.135))
    @test_throws RegionBeyondSupport solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, kb, Rect(-1.0e5, -1.0e5, 2.0e5, 2.0e5))), CameraRegression())
end

# The axis fit's order continuation (0.11.8). `OpticalAxis` fits the axis family at 1 radial term (its linear fit, then the
# refine), then at 2, 3 and 4, each order's refine from the last's camera (the higher coefficients held at 0: `fixed`);
# the result is the highest order whose refine converged without its lens folding inside the region (`_folds_inside`,
# `family.fitted_terms`). The whole-order linear start and its refits inside the fold (`_axis_fold_refits`) are the
# fallback when no order holds. A low order does not follow a source's profile where it bends at the
# region's edge, so its start does not fold there; a direct lift of the 1-term fit to 4 terms can end in a worse local
# minimum (the slope-collapse fisheye below), the stepwise continuation does not.
@testsetup module FoldingSources
using StaticArrays: SMatrix, SVector
using LinearAlgebra: norm
using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, Ellipse, CameraCalibrationMat, calibration_matrix, KannalaBrandt, parameter_vector, lens
using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
export AXIS_PIXEL, folding_source, theta_fold, theta_collapse, axis_pixel, order_continuation_kb
import VisualGeometryToolkitEstimation as VgtkEstimation
include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "folding_sources.jl"))   # `AXIS_PIXEL`, `folding_source`: shared with the golden records
axis_pixel(r) = (K = calibration_matrix(r.u); SVector(K[1, 3], K[2, 3]))
# the stepwise continuation by the public verbs on KannalaBrandt{1…4}: the reference the axis is compared with
function order_continuation_kb(src, region)
    g = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{1}, src, region)), CameraRegression()).u; r = nothing
    for N in 1:4
        k = Tuple(Float64.(parameter_vector(lens(g))))
        guess = CameraModel(SMatrix{3, 3, Float64}(calibration_matrix(g)); lens = KannalaBrandt(ntuple(i -> i <= length(k) ? k[i] : 0.0, N)...))
        r = solve(RefineProblem(CameraRegressionProblem(src, region), guess), LevenbergMarquardt()); g = r.u
    end
    return r
end
end

# 0.11.6: a linear start that folds inside the region. The source images a pinhole (f = 600 px, its axis at (512, 494))
# out to 350 px, and past it a ray field that turns away fast (25° more by 480 px): the axis family's 4-term linear fit
# turns over (folds) at 64° while the region reaches 67°, and the refine from it sits with the axis unmoved — on the
# re-gauged K (principal point (500, 500)) a silent 0° turn, the Sigma 14 mm Schöps source's failure (2026-11-15). Since
# 0.11.8 the continuation fits it over the whole region: its 1- and 2-term fits fold inside the region with the axis
# unmoved, its 3-term fit finds the axis (its 4-term one folds 0.001° inside the region: 3 terms kept); the
# fallback's refit inside the fold is kept and tested.
@testitem "OpticalAxis: a folding linear start is refitted inside the fold, never a silent rotation 0" setup = [FoldingSources] begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    import VisualGeometryToolkitEstimation as VgtkEstimation
    f = 600.0; r0 = 350.0
    θof(r) = theta_fold(r; turn = deg2rad(25.0), f, r0)
    src = folding_source(θof)
    part = VgtkCore.imaged_region(src)
    # the continuation: the whole region, the 1-term start folding inside it (recorded), the axis from a lower order
    r = solve(EstimationProblem(CameraRegressionProblem(src)), OpticalAxis())
    md = r.metadata
    @test successful_retcode(r)
    @test md.region == part
    @test md.family.start_fold isa Float64 && md.family.start_fold < md.family.θ_max
    @test md.family.fitted_terms > 1                                   # the 1-term fit does not hold
    @test norm(axis_pixel(r) - AXIS_PIXEL) < 1.0                      # the axis pixel: (512, 494), not (500, 500)
    # the fallback: the whole-order start folds inside the region (its widest angle θ(478 px) ≈ 67°), recorded, and the
    # axis is refitted inside the fold
    rf, fold, ok = VgtkEstimation._axis_fold_refits(VgtkCore.model(src), part, VgtkCore.default_device())
    @test ok && successful_retcode(rf)
    @test fold isa Float64 && fold < θof(478.0) && rf.metadata.family.θ_max < fold
    @test rf.metadata.region != part
    @test norm(axis_pixel(rf) - AXIS_PIXEL) < 1.0
    # a start that does not fold records none, and every order converges
    plain = solve(EstimationProblem(CameraRegressionProblem(BoundedCamera(CameraModel(SplineLens, CameraModel(; pp = (512.0, 494.0), focal = 600.0), Size2(1000, 1000); grid_size = (30, 30)),
                                       Size2(1000, 1000)))), OpticalAxis())
    @test plain.metadata.family.start_fold === nothing && successful_retcode(plain) && plain.metadata.family.fitted_terms == 4
end

# A linear start that folds just PAST the region's widest angle. The source is the folding test's pinhole and rim turn
# (19.5° by 480 px) with an azimuthal misfit (θ scaled by 1 + 0.02·Δx/480): the 4-term start folds at 62.67° while the
# region reaches 62.53° — RadiometricCamera's ag10fe start (an 8 mm fisheye's Schöps source over the sensor ∩ its image
# circle ∩ his rectangle, its start folding 0.03° past the region's 96.36°). The refine from it stalls ~12 px off the
# axis; under a termination that reads a tolerance stop converged (Core 0.11) it ends there converged, its lens folding
# outside the region, and a whole-order fit would take it silently. The continuation lands on the stepwise
# KannalaBrandt path's axis.
@testitem "OpticalAxis: a start folding just past the region's widest angle — the continuation's axis, never the start's silent one" setup = [FoldingSources] begin
    using Test
    using LinearAlgebra: norm
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: view_angle_limit, lens
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    import VisualGeometryToolkitEstimation as VgtkEstimation
    f = 600.0; r0 = 350.0
    θof(r) = theta_fold(r; turn = deg2rad(19.5), f, r0)
    src = folding_source(θof; ε = 0.02)
    part = VgtkCore.imaged_region(src)
    # the premise: the 4-term start folds past the region's widest angle, within the band a refit leaves out, and its
    # refine does not find the axis: it stalls (Core ≤ 0.10.11), or (a tolerance stop converged, Core 0.11) ends ~12 px off
    # with its lens folding outside the region, where no fold gate sees it
    start = VgtkEstimation._approximation_start(VgtkEstimation._linear_form(typeof(VgtkEstimation._AXIS_LENS)), typeof(VgtkEstimation._AXIS_LENS), VgtkCore.model(src), part, VgtkEstimation.ImageMeasure(), VgtkEstimation._AXIS_FIXED,
                                   VgtkCore.default_device())
    plain = VgtkEstimation._refine_camera(start, VgtkCore.model(src), part; fixed = VgtkEstimation._AXIS_FIXED)
    θfold, θmax = Float64(view_angle_limit(lens(start))), plain.metadata.family.θ_max
    @test θmax < θfold < θmax / VgtkEstimation._INSIDE_FOLD
    @test !VgtkCore.successful_retcode(plain) || (!VgtkEstimation._folds_inside(plain) && norm(axis_pixel(plain) - AXIS_PIXEL) > 5.0)
    # Estimation's own validity gate refuses that fit whatever the LM's verdict (on the axis: its regression stationary;
    # no fold inside the region), and the whole-order fallback never reports an off-axis fit as holding
    @test !VgtkEstimation._axis_fit_valid(plain)
    rf, _, ok = VgtkEstimation._axis_fold_refits(VgtkCore.model(src), part, VgtkCore.default_device())
    @test !ok || norm(axis_pixel(rf) - AXIS_PIXEL) < 5.0
    # the axis: the continuation over the whole region, on the stepwise path's axis, ~2 px from (512, 494) (the misfit's bias)
    r = solve(EstimationProblem(CameraRegressionProblem(src)), OpticalAxis())
    md = r.metadata
    @test successful_retcode(r)
    @test md.region == part
    @test md.family.fitted_terms == 4
    @test norm(axis_pixel(r) - axis_pixel(order_continuation_kb(src, part))) < 0.3
    @test norm(axis_pixel(r) - AXIS_PIXEL) < 2.5
end

# A fisheye whose profile's slope collapses at the region's edge, as the ag10fe Schöps source's does (rc-convergence
# optaxis/order, 2026-11-15): equidistant (f = 300 px) to 420 px, then θ grows by 0.2·((r − 420)/60)³ rad, so dr/dθ falls
# to f/4 at the edge (103.5°). Lifting the 1-term fit straight to 4 terms converges ~1.8 px off the axis at a higher cost;
# the stepwise continuation finds the axis.
@testitem "OpticalAxis: a fisheye's slope collapsing at the edge — the stepwise continuation finds the axis a direct lift misses" setup = [FoldingSources] begin
    using Test
    using LinearAlgebra: norm
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    import VisualGeometryToolkitEstimation as VgtkEstimation
    f = 300.0; r0 = 420.0
    θof(r) = theta_collapse(r; f, r0)
    src = folding_source(θof)
    part = VgtkCore.imaged_region(src); m = VgtkCore.model(src)
    # the direct lift: the 1-term linear start, then the 4-term refine at once
    s1 = VgtkEstimation._approximation_start(VgtkEstimation._linear_form(typeof(VgtkEstimation._AXIS_LENS)), typeof(VgtkEstimation._AXIS_LENS), m, part, VgtkEstimation.ImageMeasure(), (VgtkEstimation._AXIS_FIXED..., VgtkEstimation._axis_held(1)...),
                                VgtkCore.default_device())
    direct = VgtkEstimation._refine_camera(s1, m, part; fixed = VgtkEstimation._AXIS_FIXED)
    @test norm(axis_pixel(direct) - AXIS_PIXEL) > 1.0
    r = solve(EstimationProblem(CameraRegressionProblem(src)), OpticalAxis())
    md = r.metadata
    @test successful_retcode(r)
    @test md.region == part && md.family.fitted_terms >= 2
    @test norm(axis_pixel(r) - AXIS_PIXEL) < 0.2
    @test r.rms < direct.rms
end

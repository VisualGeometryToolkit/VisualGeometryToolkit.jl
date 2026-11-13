# The optical axis of a SplineLens camera: approximate_camera of a radially symmetric family onto a gauge-K source frees the
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

@testitem "optical_axis: a spline turned 5° off its axis — the axis and the principal point recovered, every image kept after the re-gauge" setup = [OffAxisSpline] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, dot, normalize
    using Rotations: RotMatrix, AngleAxis
    using VisualGeometryToolkitCore: estimate, metadata, retcode, calibration_matrix, rotation, map_frames, ViewFrame, project,
        backproject, axis_gauged, EuclideanMap, relabel, WorldFrame, CameraFrame, model, with_lens_coverage, imaged_region
    using VisualGeometryToolkitEstimation: optical_axis, approximate_camera

    K0 = calibration_matrix(OFF_AXIS)
    @test norm(SVector(K0[1, 3], K0[2, 3]) - SVector(500.0, 500.0)) > 20          # the never-fitted gauge: off the axis's pixel
    r = optical_axis(with_lens_coverage(OFF_BOUNDED, DISC))
    @test retcode_symbol(r) === :converged
    R = metadata(r).rotation
    @test map_frames(R) == (ViewFrame{:source}, ViewFrame{:fitted})
    axis = rotation(R)' * SVector(0.0, 0.0, 1.0)                                   # the axis in the source frame
    @test acosd(min(1.0, dot(axis, Q * SVector(0.0, 0.0, 1.0)))) < 1.0e-3          # the turned frame's true axis, Q·ẑ
    K = calibration_matrix(estimate(r))
    pp = SVector(K[1, 3], K[2, 3])
    @test norm(pp - SVector(500.0, 500.0)) < 0.02                                  # the axis's pixel
    @test metadata(r).symmetry_residual < 1.0e-3 && metadata(r).max_px < 0.05      # a symmetric lens: nothing azimuthal left
    # the same call as approximate_camera's: optical_axis is that one call, on the radially symmetric family it names,
    # its decentering groups held
    r2 = approximate_camera(typeof(estimate(r).lens), OFF_BOUNDED, imaged_region(with_lens_coverage(OFF_BOUNDED, DISC)); fixed = (:p, :sx, :sy))
    @test rotation(metadata(r2).rotation) == rotation(R) && calibration_matrix(estimate(r2)) == K
    # the re-gauge: ẑ is the axis, the principal point its pixel; every image kept with the poses turned
    g = axis_gauged(OFF_AXIS, rotation(R), K)
    @test norm(project(g, SVector(0.0, 0.0, 1.0)) - pp) < 1.0e-6
    @test norm(normalize(backproject(g, pp)) - SVector(0.0, 0.0, 1.0)) < 1.0e-4
    pose = relabel(WorldFrame, CameraFrame, EuclideanMap(RotMatrix(AngleAxis(0.3, 0.2, -1.0, 0.4)), (0.1, -0.2, 2.0)))
    turned = axis_gauged(pose, rotation(R))
    X = [SVector(x, y, 0.0) for x in -0.8:0.2:0.8 for y in -0.6:0.2:0.6]
    @test all(x -> norm(project(g, turned(x)) - project(OFF_AXIS, pose(x))) < 1.0e-6, X)
end

@testitem "optical_axis: a decentred lens reports its symmetry residual; a physical K is its own axis" setup = [OffAxisSpline] begin
    using Test
    using StaticArrays: SA, SVector
    using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, Lens, Equidistant, Radial, Additive, Tangential,
        estimate, metadata, rotation, calibration_matrix, px, KannalaBrandt, NoncentralSplineModel, ChartSpline, with_lens_coverage
    using VisualGeometryToolkitEstimation: optical_axis, fit_camera, approximate_camera, refine_camera

    # a fisheye with a tangential (decentring) term: no radially symmetric camera about any axis images it
    dec = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = Lens(Equidistant(), Additive(Radial((-0.02,)), Tangential(2.0e-3, -1.5e-3))))
    spl = BoundedCamera(CameraModel(SplineLens, dec, Size2(1000, 1000); grid_size = (40, 40)), Size2(1000, 1000))
    rd = optical_axis(with_lens_coverage(spl, DISC))
    rs = optical_axis(with_lens_coverage(OFF_BOUNDED, DISC))
    @test metadata(rd).symmetry_residual > 0.1                                     # px: reported, not absorbed
    @test metadata(rd).symmetry_residual > 100 * metadata(rs).symmetry_residual
    # a physical K: its principal point and its ẑ are the source's own — the identity, not fitted
    rp = optical_axis(BoundedCamera(FISHEYE, Size2(1000, 1000); lens_coverage = DISC))
    @test rotation(metadata(rp).rotation) == one(rotation(metadata(rp).rotation))
    @test calibration_matrix(estimate(rp))[1:2, 3] == calibration_matrix(FISHEYE)[1:2, 3]
    @test isnan(metadata(rp).symmetry_residual)                                     # a radial profile has no azimuth to measure
    @test typeof(metadata(rp).rotation) === typeof(metadata(rs).rotation)          # one rotation type for every source
    # a non-central source: not supported yet
    G = ON_AXIS
    nc = NoncentralSplineModel(G.intrinsics, G.lens.backward, ChartSpline(G.lens.backward.area, fill(SVector(0.0, 0.0, 0.0), size(G.lens.backward.controls))))
    @test_throws r"not supported yet" fit_camera(KannalaBrandt{4}, nc, REGION)
    @test_throws r"not supported yet" fit_camera(KannalaBrandt{4}, BoundedCamera(nc, Size2(1000, 1000)), REGION)
    @test_throws r"not supported yet" approximate_camera(KannalaBrandt{4}, nc, REGION)
    @test_throws r"not supported yet" refine_camera(estimate(rp), nc, REGION)
end

@testitem "fit_camera(SplineLens): a gauge-K source is gauged on its optical axis first; a physical source is unchanged" setup = [OffAxisSpline] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: estimate, metadata, retcode, rotation, calibration_matrix, backproject, project, lens,
        SplineLens, BoundedCamera, Size2
    using VisualGeometryToolkitEstimation: fit_camera, optical_axis

    r = fit_camera(SplineLens, OFF_BOUNDED, REGION; tolerance = 1.0e-3)
    cam = estimate(r)
    R = rotation(metadata(r).rotation)
    oa = optical_axis(OFF_BOUNDED)                    # the gauge reads the source's bound, not the fit's region
    @test R == rotation(metadata(oa).rotation) && metadata(r).symmetry_residual == metadata(oa).symmetry_residual
    @test calibration_matrix(cam)[1:2, 3] == calibration_matrix(estimate(oa))[1:2, 3]
    # the backward is the source's turned by R, exactly as axis_gauged turns it; the forward fitted on it
    @test lens(cam).backward.controls == map(c -> R * c, lens(OFF_AXIS).backward.controls)
    pix = [SVector(500.0 + ρ * cos(a), 500.0 + ρ * sin(a)) for ρ in (0.0, 150.0, 400.0) for a in range(0, 2π; length = 9)]
    @test all(u -> norm(backproject(cam, u) - R * backproject(OFF_AXIS, u)) < 1.0e-12, pix)
    @test all(u -> norm(project(cam, backproject(cam, u)) - u) < 1.0e-3, pix)
    @test retcode_symbol(r) === :converged && metadata(r).family.forward.max_px <= 1.0e-3
    # a parametric source: its ẑ is the axis already — the identity, the source's K
    rp = fit_camera(SplineLens, BoundedCamera(FISHEYE, Size2(1000, 1000)), REGION; tolerance = 1.0e-2, forward = false)
    @test rotation(metadata(rp).rotation) == one(R) && isnan(metadata(rp).symmetry_residual)
    @test calibration_matrix(estimate(rp)) == calibration_matrix(FISHEYE)
end

@testitem "optical_axis: a bounded camera only — its region is the camera's own bound or a given region within it; a circular fisheye's axis from its image circle" setup = [OffAxisSpline] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, dot, normalize, I
    using Rotations: RotMatrix, AngleAxis
    using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, Ellipse, Rect, KannalaBrandt, backproject, px,
        estimate, metadata, retcode, rotation, calibration_matrix, imaged_region
    using VisualGeometryToolkitEstimation: optical_axis

    # a bare model has no bound: refused, with the route (the region is the camera's bound, never a default)
    @test_throws r"BoundedCamera" optical_axis(OFF_AXIS)
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
    r = optical_axis(BoundedCamera(spl, Size2(1400, 1000); lens_coverage = circle))
    @test retcode_symbol(r) === :converged
    axis = rotation(metadata(r).rotation)' * SVector(0.0, 0.0, 1.0)
    @test acosd(min(1.0, dot(axis, Q3 * SVector(0.0, 0.0, 1.0)))) < 0.01
    K = calibration_matrix(estimate(r))
    @test norm(SVector(K[1, 3], K[2, 3]) - SVector(700.0, 500.0)) < 0.1
    @test metadata(r).symmetry_residual < 0.05
    whole = optical_axis(BoundedCamera(spl, Size2(1400, 1000)))
    @test metadata(whole).symmetry_residual > 10 * metadata(r).symmetry_residual
    # an explicit region (0.10.2): the fit over `region ∩ imaged_region(camera)`. The whole-sensor camera over the image
    # circle finds the axis its coverage-bound twin finds (RadiometricCamera's siemens start camera stalled over its
    # whole image and converges over its board's rectangle: the same case on a real lattice)
    rc = optical_axis(BoundedCamera(spl, Size2(1400, 1000)), circle)
    @test retcode_symbol(rc) === :converged
    @test acosd(min(1.0, dot(rotation(metadata(rc).rotation)' * SVector(0.0, 0.0, 1.0), Q3 * SVector(0.0, 0.0, 1.0)))) < 0.01
    @test metadata(rc).symmetry_residual < 0.05
    # a region past the camera's image is clipped to it: the sensor rectangle on the coverage-bound camera is its bound
    bounded = BoundedCamera(spl, Size2(1400, 1000); lens_coverage = circle)
    rr = optical_axis(bounded, Rect(0, 0, 1400, 1000))
    @test retcode_symbol(rr) === :converged && length(metadata(rr).residuals) == length(metadata(r).residuals)
    @test calibration_matrix(estimate(rr)) == K       # the same pixels, in the same order: the same problem, the same bits
    # no region: the camera's own imaged region, the one path, the same bits
    @test metadata(optical_axis(bounded, imaged_region(bounded))).residuals == metadata(r).residuals
end

# optical_axis's LM takes every chart LM's stop settings ([chart_lm] ftol = 1e-10, Estimation 0.10.2; until 0.10.1 it ran at
# lmsolve_chart!'s eps): on the whole sensor of the 5°-off-axis spline the default family's cost settles before its
# steps reach the increment floor, so it stops on the relative cost decrease (at eps: IncrementSize, one iteration
# later; the principal point moves by < 1e-6 px; wave10/optical-axis-order.md)
@testitem "optical_axis: the chart LM's cost tolerance ends a settled regression (CostDecrease)" setup = [OffAxisSpline] begin
    using Test
    using VisualGeometryToolkitCore: retcode_symbol, stop_rule, iterations, CostDecrease
    using VisualGeometryToolkitEstimation: optical_axis
    r = optical_axis(OFF_BOUNDED)
    @test retcode_symbol(r) === :converged
    @test stop_rule(r) === CostDecrease
end

@testitem "refine_camera's regression on the device: the CPU and CUDA backends of the one kernel agree to rounding" setup = [OffAxisSpline] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using CUDA: CUDA
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, estimate, metadata, retcode, iterations, rotation,
        calibration_matrix, parameter_vector, lens, imaged_region, with_lens_coverage, KannalaBrandt, rms
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, approximate_camera, optical_axis
    # the reduction kernel (each sample's 2-D residual and Jacobian rows, Σ‖r‖², Jᵀr, JᵀJ) is the same KernelAbstractions
    # kernel on every backend: the CPU's and the GPU's fits agree to the reduction's rounding (the work items' sums in
    # another order), with the same steps
    cpu = VC.KA.CPU()
    b = with_lens_coverage(OFF_BOUNDED, DISC)
    fits = [(VE._AXIS_FAMILY, b, imaged_region(b), VE._AXIS_FIXED), (KannalaBrandt{4}, FISHEYE, REGION, ())]   # a gauge K, a physical K
    for (L, src, region, fixed) in fits
        rc = approximate_camera(L, src, region; fixed, device = cpu)
        @test retcode_symbol(rc) === :converged
        CUDA.functional() || continue
        rg = approximate_camera(L, src, region; fixed, device = VC.default_device())
        @test retcode_symbol(rg) === :converged && iterations(rg) == iterations(rc)
        @test maximum(abs, calibration_matrix(estimate(rg)) - calibration_matrix(estimate(rc))) < 1.0e-9
        @test maximum(abs, collect(parameter_vector(lens(estimate(rg)))) - collect(parameter_vector(lens(estimate(rc))))) < 1.0e-9
        @test maximum(abs, rotation(metadata(rg).rotation) - rotation(metadata(rc).rotation)) < 1.0e-12
        @test abs(rms(rg) - rms(rc)) < 1.0e-12 && abs(metadata(rg).max_px - metadata(rc).max_px) < 1.0e-12   # px
    end
end

# The caller's `device` reaches a SplineLens source's ray sampling — the expensive step: with the default device's
# resolution made to throw, every camera verb given `device` still runs (until 0.11.0 `optical_axis` took none, and
# the samples of a gauge-K source were drawn on the default device whatever the caller gave).
@testitem "device: a SplineLens source's rays are sampled on the caller's device — the default is never resolved" setup = [OffAxisSpline] begin
    using Test
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, estimate, imaged_region, KannalaBrandt, SplineLens
    using VisualGeometryToolkitEstimation: optical_axis, refine_camera, fit_camera, approximate_camera
    cpu = VC.KA.CPU()
    region = imaged_region(OFF_BOUNDED)
    resolver = VC._DEVICE_RESOLVER[]
    VC._DEVICE_RESOLVER[] = () -> error("default_device() resolved: the caller's `device` did not reach this call")
    VC.reset_default_device!()
    try
        r = optical_axis(OFF_BOUNDED; device = cpu)
        @test VC.converged(r)
        @test VC.converged(fit_camera(KannalaBrandt{4}, OFF_BOUNDED, region; device = cpu))
        @test VC.converged(approximate_camera(KannalaBrandt{4}, OFF_BOUNDED, region; device = cpu))
        @test estimate(refine_camera(estimate(r), OFF_AXIS, region; device = cpu)) isa VC.CameraModel   # from its own optimum: runs on `cpu`
        @test VC.converged(fit_camera(SplineLens, OFF_BOUNDED, region; tolerance = 1.0e-2, device = cpu))   # its optical axis first
    finally
        VC._DEVICE_RESOLVER[] = resolver
        VC.reset_default_device!()
    end
end

# The axis family's radial order is a configurable literal (configs/default.toml [optical_axis] radial_terms, read once):
# 4 terms, θ³…θ⁹, the owner's ruling of 2026-11-13 from the order study (~/logs/dgx-agent/wave10/optical-axis-order.md:
# the smallest order whose principal point is within ½σ of 6 terms' on every converging RC camera, cond(J) 47–160×
# lower). Until 0.11.0 it was Fisheye624's 6 (θ³…θ¹³).
@testitem "optical_axis: the axis family's radial order is the config's (4 terms)" setup = [OffAxisSpline] begin
    using Test
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, Lens, Equidistant, Additive, Tangential, ThinPrism, Radial, estimate,
        lens, parameter_names, with_lens_coverage
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, optical_axis
    n = VE._dflt("optical_axis", "radial_terms")
    @test n == 4
    @test VE._AXIS_FAMILY === typeof(Lens(Equidistant(), Additive(Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0))) ∘ Radial(ntuple(_ -> 0.0, n))))
    names = parameter_names(lens(estimate(optical_axis(with_lens_coverage(OFF_BOUNDED, DISC)))))
    @test :k4 in names && !(:k5 in names)
end

# The region-support guard (Core's `check_region_support`, the owner's ruling of 2026-11-13): a fit over a region that
# reaches beyond the source camera's calibrated support is REFUSED (`RegionBeyondSupport`), never fitted over the model's
# extrapolation. RadiometricCamera's siemens start camera is a re-gridded copy of a Schöps calibration whose support is
# Rect(1616, 593, 2159, 2321): over its whole sensor (corners extrapolated to 179.8°) the axis fit stalled at every
# radial order (wave10/optical-axis-order.md); over the support it converges.
@testitem "region support: a fit beyond the source's calibrated support is refused; inside it, the axis converges" begin
    using Test
    using StaticArrays: SVector, SMatrix
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, ChartSpline, BoundedCamera, Size2, Rect,
        LogicalIntrinsics, CameraCalibrationMat, RegionBeyondSupport, retcode_symbol, imaged_region, estimate, model, KannalaBrandt
    using VisualGeometryToolkitEstimation: optical_axis, approximate_camera, refine_camera, fit_camera
    c = include(joinpath(@__DIR__, "fixtures", "rc_siemens_start_camera.jl"))
    C = [SVector{3, Float64}(c.controls[(i - 1) * c.gw + j]) for i in 1:c.gh, j in 1:c.gw]
    K = SMatrix{3, 3, Float64}(c.fx, 0.0, 0.0, c.skew, c.fy, 0.0, c.cx, c.cy, 1.0)
    cam = BoundedCamera(CameraModel(LogicalIntrinsics(CameraCalibrationMat(K; check = false)), SplineLens(ChartSpline(Rect(c.area...), C))),
                        Size2(c.width, c.height))
    support = Rect(1616.0, 593.0, 2159.0, 2321.0)   # the Schöps source's valid_domain
    # the whole sensor: refused by each fit at its entry, naming the fit
    for (who, f) in (("optical_axis", () -> optical_axis(cam; support)),
                     ("approximate_camera", () -> approximate_camera(KannalaBrandt{4}, cam, imaged_region(cam); support)),
                     ("fit_camera", () -> fit_camera(KannalaBrandt{4}, cam, imaged_region(cam); support)))
        e = try f(); nothing catch err; err end
        @test e isa RegionBeyondSupport && e.who == who && e.support == support
    end
    # the support itself: the axis converges
    r = optical_axis(cam, support; support)
    @test retcode_symbol(r) === :converged
    # refine_camera refuses the whole sensor too, and fits inside
    @test_throws RegionBeyondSupport refine_camera(estimate(r), cam, imaged_region(cam); support)
    @test retcode_symbol(refine_camera(estimate(r), cam, support; support)) in (:converged, :solved)
    # without `support`, a camera's own valid_domain bounds it: a region past a lens's injectivity disc is refused
    kb = CameraModel(SMatrix{3, 3, Float64}(300, 0, 0, 0, 300, 0, 320, 240, 1); lens = KannalaBrandt(-0.135))
    @test_throws RegionBeyondSupport fit_camera(KannalaBrandt{4}, kb, Rect(-1.0e5, -1.0e5, 2.0e5, 2.0e5))
end

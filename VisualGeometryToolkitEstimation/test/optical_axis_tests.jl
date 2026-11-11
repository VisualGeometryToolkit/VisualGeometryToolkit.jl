# The optical axis of a SplineLens camera: fit_camera of a radially symmetric family onto a gauge-K source frees the
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
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, dot, normalize
    using Rotations: RotMatrix, AngleAxis
    using VisualGeometryToolkitCore: estimate, metadata, retcode, calibration_matrix, rotation, map_frames, ViewFrame, project,
        backproject, axis_gauged, EuclideanMap, relabel, WorldFrame, CameraFrame, model, with_lens_coverage, imaged_region
    using VisualGeometryToolkitEstimation: optical_axis, fit_camera

    K0 = calibration_matrix(OFF_AXIS)
    @test norm(SVector(K0[1, 3], K0[2, 3]) - SVector(500.0, 500.0)) > 20          # the never-fitted gauge: off the axis's pixel
    r = optical_axis(with_lens_coverage(OFF_BOUNDED, DISC))
    @test retcode(r) === :converged
    R = metadata(r).rotation
    @test map_frames(R) == (ViewFrame{:source}, ViewFrame{:fitted})
    axis = rotation(R)' * SVector(0.0, 0.0, 1.0)                                   # the axis in the source frame
    @test acosd(min(1.0, dot(axis, Q * SVector(0.0, 0.0, 1.0)))) < 1.0e-3          # the turned frame's true axis, Q·ẑ
    K = calibration_matrix(estimate(r))
    pp = SVector(K[1, 3], K[2, 3])
    @test norm(pp - SVector(500.0, 500.0)) < 0.02                                  # the axis's pixel
    @test metadata(r).symmetry_residual < 1.0e-3 && metadata(r).max_px < 0.05      # a symmetric lens: nothing azimuthal left
    # the same call as fit_camera's: optical_axis is that one call, on the radially symmetric family it names
    r2 = fit_camera(typeof(estimate(r).lens), OFF_BOUNDED, imaged_region(with_lens_coverage(OFF_BOUNDED, DISC)))
    @test rotation(metadata(r2).rotation) == rotation(R) && calibration_matrix(estimate(r2)) == K
    # the re-gauge: ẑ is the axis, the principal point its pixel; every image kept with the poses turned
    g = axis_gauged(OFF_AXIS, rotation(R), pp)
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
    using VisualGeometryToolkitEstimation: optical_axis, fit_camera

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
end

@testitem "fit_camera(SplineLens): a gauge-K source is gauged on its optical axis first; a physical source is unchanged" setup = [OffAxisSpline] begin
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
    @test retcode(r) === :converged && metadata(r).family.forward.max_px <= 1.0e-3
    # a parametric source: its ẑ is the axis already — the identity, the source's K
    rp = fit_camera(SplineLens, BoundedCamera(FISHEYE, Size2(1000, 1000)), REGION; tolerance = 1.0e-2, forward = false)
    @test rotation(metadata(rp).rotation) == one(R) && isnan(metadata(rp).symmetry_residual)
    @test calibration_matrix(estimate(rp)) == calibration_matrix(FISHEYE)
end

@testitem "optical_axis: a bounded camera only — its region is the camera's own bound; a circular fisheye's axis from its image circle" setup = [OffAxisSpline] begin
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, dot, normalize, I
    using Rotations: RotMatrix, AngleAxis
    using VisualGeometryToolkitCore: CameraModel, SplineLens, BoundedCamera, Size2, Ellipse, Rect, KannalaBrandt, backproject, px,
        estimate, metadata, retcode, rotation, calibration_matrix, EuclideanMap
    using VisualGeometryToolkitEstimation: optical_axis, refine_camera_and_poses

    # a bare model has no bound: refused, with the route (the region is the camera's bound, never a default)
    @test_throws r"BoundedCamera" optical_axis(OFF_AXIS)
    @test_throws r"BoundedCamera" refine_camera_and_poses(OFF_AXIS, [EuclideanMap(RotMatrix{3}(1.0I), SVector(0.0, 0.0, 0.0))], [[SVector(0.0, 0.0, 2.0) => SVector(500.0, 500.0)]])
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
    @test retcode(r) === :converged
    axis = rotation(metadata(r).rotation)' * SVector(0.0, 0.0, 1.0)
    @test acosd(min(1.0, dot(axis, Q3 * SVector(0.0, 0.0, 1.0)))) < 0.01
    K = calibration_matrix(estimate(r))
    @test norm(SVector(K[1, 3], K[2, 3]) - SVector(700.0, 500.0)) < 0.1
    @test metadata(r).symmetry_residual < 0.05
    whole = optical_axis(BoundedCamera(spl, Size2(1400, 1000)))
    @test metadata(whole).symmetry_residual > 10 * metadata(r).symmetry_residual
end

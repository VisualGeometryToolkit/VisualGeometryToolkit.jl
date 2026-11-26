# The sources the spline fits take: a bounded camera's image is Core's `extent` (Estimation's alias `image_size`
# is deleted, 0.11.0); any other source is refused by name. `ForwardSpline` refuses any camera that is not a
# bounded SplineLens camera, naming what it got.

@testitem "spline fits: a bounded camera's image is Core's extent; any other source has none" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, BoundedCamera, Size2, Rect, KannalaBrandt, SplineLens, NoncentralSplineModel,
        valid_domain, RegionBeyondSupport
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, PSpline
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # no alias of `extent`: the name is gone, and with it the declared extension point
    @test !isdefined(VgtkEstimation, :image_size)
    @test all(f -> nameof(f) !== :image_size, VgtkEstimation.CONTRACT_GENERICS)
    m = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    region = Rect(0, 0, 1280, 960)
    # a bare model has no image for the lattice: bound it first
    @test_throws r"has no image" solve(EstimationProblem(CameraRegressionProblem(SplineLens, m, region)), CameraRegression(; tolerance = 1.0e-2))
    @test_throws r"has no image" solve(EstimationProblem(CameraRegressionProblem(SplineLens, m, region)), PSpline(; grid_size = (8, 8)))
    @test_throws r"has no image" solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, m, region)), CameraRegression(; tolerance = 1.0e-2))
    # a bounded one: its extent is the lattice's image, the fit over the source's valid domain (KB's injectivity disc,
    # r ≈ 262 px: the whole image reaches past it, refused — the owner's ruling of 2026-11-13, refuse, never clip)
    bc = BoundedCamera(m, Size2(1280, 960))
    @test_throws RegionBeyondSupport solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, region)), PSpline(; grid_size = (8, 8)))
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, valid_domain(m))), PSpline(; grid_size = (8, 8)))
    @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.ClosedForm)
end

@testitem "ForwardSpline: a bounded non-central model is refused by name, not a MethodError" setup = [FitCameraNoncentral] begin
    using Test
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, ForwardSpline
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    @test_throws r"got a BoundedCamera of a NoncentralSplineModel" solve(EstimationProblem(CameraRegressionProblem(SOURCE)), ForwardSpline(; tolerance = 1.0e-3))
    @test_throws r"got a NoncentralSplineModel" solve(EstimationProblem(CameraRegressionProblem(MODEL)), ForwardSpline(; tolerance = 1.0e-3))
end

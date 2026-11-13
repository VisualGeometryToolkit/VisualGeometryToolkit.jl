# The sources the spline fits take: a bounded camera's image is Core's `extent` (Estimation's alias `image_size`
# is deleted, 0.11.0); any other source is refused by name. `fit_forward_spline` refuses any camera that is not a
# bounded SplineLens camera, naming what it got.

@testitem "spline fits: a bounded camera's image is Core's extent; any other source has none" begin
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, BoundedCamera, Size2, Rect, KannalaBrandt, SplineLens, NoncentralSplineModel, retcode_symbol
    using VisualGeometryToolkitEstimation: fit_camera, pspline_grid
    # no alias of `extent`: the name is gone, and with it the declared extension point
    @test !isdefined(VgtkEstimation, :image_size)
    @test all(f -> nameof(f) !== :image_size, VgtkEstimation.CONTRACT_GENERICS)
    m = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    region = Rect(0, 0, 1280, 960)
    # a bare model has no image for the lattice: bound it first
    @test_throws r"has no image" fit_camera(SplineLens, m, region; tolerance = 1.0e-2)
    @test_throws r"has no image" pspline_grid(SplineLens, m, region, (8, 8))
    @test_throws r"has no image" fit_camera(NoncentralSplineModel, m, region; tolerance = 1.0e-2)
    # a bounded one: its extent is the lattice's image
    r = pspline_grid(SplineLens, BoundedCamera(m, Size2(1280, 960)), region, (8, 8))
    @test retcode_symbol(r) === :converged
end

@testitem "fit_forward_spline: a bounded non-central model is refused by name, not a MethodError" setup = [FitCameraNoncentral] begin
    using Test
    using VisualGeometryToolkitEstimation: fit_forward_spline
    @test_throws r"got a BoundedCamera of a NoncentralSplineModel" fit_forward_spline(SOURCE; tolerance = 1.0e-3)
    @test_throws r"got a NoncentralSplineModel" fit_forward_spline(MODEL; tolerance = 1.0e-3)
end

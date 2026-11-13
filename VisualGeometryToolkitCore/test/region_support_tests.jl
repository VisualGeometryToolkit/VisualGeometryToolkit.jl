@testitem "region support: a fit's region beyond the source's calibrated support is refused, typed" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: check_region_support, RegionBeyondSupport, WholePlane

    # RC's whole-sensor siemens start camera (rc080-bisect/start_camera.json, 5208 × 3476): its source's calibrated support
    # (the Schöps calibration's valid_domain) is the box below, while the re-gridded camera's own region is the whole sensor,
    # where the model is extrapolated (to a field angle of 179.8° at the corners)
    support = Rect(1616.0, 593.0, 2159.0, 2321.0)
    sensor = Rect(0.0, 0.0, 5208.0, 3476.0)
    e = try check_region_support(support, sensor; who = "optical_axis"); nothing catch err; err end
    @test e isa RegionBeyondSupport && e.who == "optical_axis" && e.region == sensor && e.support == support
    @test occursin("beyond the source camera's calibrated support", sprint(showerror, e))
    @test check_region_support(support, Rect(1700.0, 700.0, 1000.0, 1000.0); who = "fit") == Rect(1700.0, 700.0, 1000.0, 1000.0)
    # the regions a fit takes: an ellipse, an intersection, the whole plane
    disc = Ellipse(Point2(2600.0, 1700.0), 300.0, 300.0, 0.0)
    @test check_region_support(support, disc; who = "fit") === disc
    @test check_region_support(support, sensor ∩ disc; who = "fit") isa VisualGeometryToolkitCore.RegionIntersection
    @test_throws RegionBeyondSupport check_region_support(support, WholePlane(); who = "fit")
    @test check_region_support(WholePlane(), sensor; who = "fit") === sensor
    # a camera's own form reads its valid_domain: a lens bounded by its injectivity disc refuses a region past it
    cam = CameraModel(SMatrix{3, 3, Float64}(300, 0, 0, 0, 300, 0, 320, 240, 1); lens = KannalaBrandt(-0.135))
    v = valid_domain(cam)
    @test v isa Ellipse
    @test_throws RegionBeyondSupport check_region_support(cam, Rect(-1.0e5, -1.0e5, 2.0e5, 2.0e5); who = "approximate_camera")
    @test check_region_support(cam, Rect(300.0, 220.0, 40.0, 40.0); who = "approximate_camera") == Rect(300.0, 220.0, 40.0, 40.0)
end

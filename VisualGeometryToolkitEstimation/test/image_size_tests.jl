@testitem "image_size: a bounded camera's image in whole pixels; any other source has none" begin
    # RadiometricCamera reached it by its former private name and extends it for its bounded lens type.
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, BoundedCamera, Size2, extent, KannalaBrandt
    @test Base.ispublic(VgtkEstimation, :image_size) && VgtkEstimation.image_size in VgtkEstimation.CONTRACT_GENERICS
    m = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    bc = BoundedCamera(m, Size2(1280, 960))
    @test VgtkEstimation.image_size(bc) === Size2(1280, 960)
    @test VgtkEstimation.image_size(bc) == Size2(Int(extent(bc).width), Int(extent(bc).height))
    @test VgtkEstimation.image_size(BoundedCamera(m, Size2(641, 3))) === Size2(641, 3)
    @test_throws r"has no image" VgtkEstimation.image_size(m)               # a bare model: bound it first
end

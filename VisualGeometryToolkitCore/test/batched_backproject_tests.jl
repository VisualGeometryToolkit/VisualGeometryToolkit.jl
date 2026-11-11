@testitem "backproject(camera, pixels; device): one batched form for every camera with a device form" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, SplineLens, NoncentralSplineModel, ChartSpline, px, backproject,
        backproject_with_jacobian, default_device
    import KernelAbstractions as KA

    # The batched backprojection (and its Jacobian) of a parametric camera runs on the device as one
    # `pointwise!`, each ray an SVector{3} — the scalar map, bit for bit on the CPU.
    cam = CameraModel(400.0px, SA[321.5px, 238.25px]; lens = KannalaBrandt(-0.02, 0.003, 0.0, 0.0))
    pixels = [SVector(x, y) for y in 0.5:37.0:479.5 for x in 0.5:41.0:639.5]
    for dev in (KA.CPU(), default_device())
        r = backproject(cam, pixels; device = dev)
        @test r isa Vector{SVector{3, Float64}}
        dev isa KA.CPU && @test r == [SVector{3}(backproject(cam, u)) for u in pixels]
        @test maximum(i -> maximum(abs, r[i] - backproject(cam, pixels[i])), eachindex(pixels)) <= 1.0e-12
        j = backproject_with_jacobian(cam, pixels; device = dev)
        @test eltype(j) <: Tuple{SVector{3, Float64}, SMatrix{3, 2, Float64}}
        dev isa KA.CPU && @test j == [(SVector{3}(first(backproject_with_jacobian(cam, u))), last(backproject_with_jacobian(cam, u))) for u in pixels]
        @test maximum(i -> maximum(abs, last(j[i]) - last(backproject_with_jacobian(cam, pixels[i]))), eachindex(pixels)) <= 1.0e-9
    end
    # a SplineLens camera and the non-central model: the same forms
    g = CameraModel(SplineLens, cam, Size2(640, 480); grid_size = (14, 12))
    @test backproject(g, pixels; device = KA.CPU()) == [SVector{3}(backproject(g, u)) for u in pixels]
    m = NoncentralSplineModel(cam, Size2(640, 480); grid_size = (14, 12))
    @test map(last, backproject_with_jacobian(m, pixels; device = KA.CPU())) == [last(backproject_with_jacobian(m, u)) for u in pixels]
    @test isempty(backproject(cam, SVector{2, Float64}[]; device = KA.CPU()))
end

@testitem "CameraCalibrationMat from a focal length and a principal point, image rect or size" begin
    using Test, StaticArrays, ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: px, mm
    using GeometryBasics: Rect2
    using FlexUnits: DimensionError

    sz = Size2(320, 240)
    K9 = CameraCalibrationMat(412.0, 0.0, 0.0, 0.0, 412.0, 0.0, 160.0, 120.0, 1.0)

    @testset "the image centre in Core's COLMAP convention" begin
        @test centroid(Rect(sz)) == SVector(160.0, 120.0)                 # the image [0, W] × [0, H]: (W/2, H/2)
        @test centroid(Rect2(0.5, 0.5, 4.0, 2.0)) == SVector(2.5, 1.5)   # origin + widths / 2
    end

    @testset "the same matrix by every form" begin
        K = CameraCalibrationMat(412.0, sz)
        @test K isa CameraCalibrationMat{Float64} && K == K9
        @test CameraCalibrationMat(412.0, Rect(sz)) == K
        @test CameraCalibrationMat(412.0, centroid(Rect(sz))) == K
        @test CameraCalibrationMat(412.0, Point2(160.0, 120.0)) == K
        @test CameraCalibrationMat(412.0, [160.0, 120.0]) == K
        @test CameraCalibrationMat((412.0, 412.0), sz) == K
        @test CameraCalibrationMat(412.0px, sz) == K                       # a pixel is dimensionless
        @test CameraCalibrationMat((412.0px, 412.0px), Rect(sz)) == K
        @test CameraCalibrationMat(412, sz) == K                           # integers give Float64
    end

    @testset "anisotropic focals, an offset rect" begin
        K = CameraCalibrationMat((412.0, 410.0), Rect2(10.0, 20.0, 300.0, 200.0))
        @test (K[1, 1], K[2, 2], K[1, 2], K[1, 3], K[2, 3]) == (412.0, 410.0, 0.0, 160.0, 120.0)
        @test CameraCalibrationMat((412.0px, 410.0px), Point2(161.5, 119.0)) ==
            CameraCalibrationMat(412.0, 0.0, 0.0, 0.0, 410.0, 0.0, 161.5, 119.0, 1.0)
    end

    @testset "element type: the floating-point promotion of the inputs" begin
        @test CameraCalibrationMat(412.0f0, Point2f(160, 120)) isa CameraCalibrationMat{Float32}
        @test CameraCalibrationMat(412.0f0, Size2(320.0f0, 240.0f0)) isa CameraCalibrationMat{Float32}
        g = ForwardDiff.derivative(f -> CameraCalibrationMat(f, sz)[1, 1], 412.0)
        @test g == 1.0
    end

    @testset "checked like the 9-entry constructor" begin
        @test_throws ArgumentError CameraCalibrationMat(-1.0, sz)
        @test_throws ArgumentError CameraCalibrationMat((412.0, 0.0), sz)
        @test CameraCalibrationMat(-1.0, sz; check = false)[1, 1] == -1.0
        @test_throws DimensionMismatch CameraCalibrationMat(412.0, [1.0, 2.0, 3.0])
        @test_throws DimensionError CameraCalibrationMat(16.0mm, sz)       # a length, not pixels
    end

    @testset "the camera of the README" begin
        K = CameraCalibrationMat(412.0, sz)
        cam = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), sz)
        @test extent(cam) == sz && CameraCalibrationMat(412.0, extent(cam)) == K
    end

    @testset "the existing forms unchanged" begin
        @test CameraCalibrationMat(800.0px, [320.0px, 240.0px]) == CameraCalibrationMat(800.0, [320.0, 240.0])
        @test CameraCalibrationMat((800.0px, 805.0px), [320.0px, 240.0px]) isa CameraCalibrationMat{Float64}
        @test CameraCalibrationMat(-800.0px, [320.0px, 240.0px])[1, 1] == -800.0   # unchecked, as before
        @test CameraModel(; focal = 800.0, pp = (320.0, 240.0)) == CameraModel(CameraCalibrationMat(800.0, Point2(320.0, 240.0)))
    end

    @testset "no ambiguities" begin
        @test isempty(Test.detect_ambiguities(VisualGeometryToolkitCore; recursive = true))
    end
end

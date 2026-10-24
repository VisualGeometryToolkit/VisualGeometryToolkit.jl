# =============================================================================
# Pixel-sized parameters on FlexUnits' quantities.
#
# FlexUnits has the seven SI dimensions only, so `px` is dimensionless: a pixel
# cannot be told from a radian or a ratio (the accepted loss). What survives is
# the length check: every public function that takes a pixel-sized quantity
# (a sensor's resolution, a pixel focal length or principal point, an image
# size) accepts a plain number or a dimensionless quantity, gives today's
# numbers, and refuses a LENGTH with FlexUnits' `DimensionError`.
# =============================================================================

@testitem "units: pixel-sized parameters" begin
    using Test
    using FlexUnits: Dimensions, FixRat32, Quantity
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: px, mm, μm, ustrip, strip_or_bare, Size2, Disc, Point2,
        DimensionError, DiagonalNormalization, CameraCalibrationMat, aspect_ratio, Rect
    using VisualGeometryToolkitCore.Cameras: Sensor, sensor, focal_length_for_fov, LogicalIntrinsics

    Q = Quantity{Float64, Dimensions{FixRat32}}

    @testset "Sensor: resolution in pixels, pitch a length" begin
        a = Sensor(Size2(width = 1920, height = 1200), 5.86μm)    # integers are pixels
        b = Sensor(Size2(width = 1920px, height = 1200px), 5.86μm)
        for s in (a, b)
            @test s isa Sensor{Q, Q}
            @test ustrip.(px, s.resolution) == Size2(1920.0, 1200.0)
            @test ustrip.(μm, s.pitch) ≈ Size2(5.86, 5.86)
            @test aspect_ratio(s) == 1.6
            @test Rect(s) == Rect(0, 0, 1920, 1200)
        end
        @test ustrip.(px, sensor("Sony", "IMX174").resolution) == Size2(1920.0, 1200.0)
        # a length where a pixel count is meant
        @test_throws DimensionError Sensor(Size2(width = 1920mm, height = 1200mm), 5.86μm)
        @test_throws DimensionError Sensor(Size2(width = 1920mm, height = 1200mm), Size2(width = 5.86μm, height = 5.86μm))
    end

    @testset "focal_length_for_fov: pixels from a resolution, a length from a sensor" begin
        θ = deg2rad(40.0)
        f_plain = focal_length_for_fov(Size2(width = 1280, height = 960), θ)
        @test f_plain == 640 / tan(θ / 2)
        f_px = focal_length_for_fov(Size2(width = 1280px, height = 960px), θ)
        @test ustrip(px, f_px) == f_plain
        @test ustrip(px, focal_length_for_fov(Size2(width = 1280px, height = 960px), θ; dimension = :vertical)) == 480 / tan(θ / 2)
        @test_throws DimensionError focal_length_for_fov(Size2(width = 1280mm, height = 960mm), θ)
        f_mm = focal_length_for_fov(sensor("Sony", "IMX174"), θ)
        @test ustrip(mm, f_mm) ≈ 960 / tan(θ / 2) * 5.86e-3
    end

    @testset "pixel intrinsics: a focal length and principal point in pixels" begin
        K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
        @test Matrix(K) == [800.0 0 320; 0 800 240; 0 0 1]
        K2 = CameraCalibrationMat((800.0px, 805.0px), [320.0px, 240.0px])
        @test Matrix(K2) == [800.0 0 320; 0 805 240; 0 0 1]
        @test LogicalIntrinsics(800.0px, [320.0px, 240.0px]).K == K
        # a metric focal length through the pitch gives the pixel one
        Km = CameraCalibrationMat(16.0mm, Size2(width = 5.86μm / px, height = 5.86μm / px), [320.0px, 240.0px])
        @test Km[1, 1] ≈ 16.0 / 5.86e-3 && Km[2, 2] ≈ 16.0 / 5.86e-3 && Km[1, 3] == 320.0
        # a length where a pixel focal length or principal point is meant
        @test_throws DimensionError CameraCalibrationMat(16.0mm, [320.0px, 240.0px])
        @test_throws DimensionError CameraCalibrationMat(800.0px, [3.2mm, 2.4mm])
        @test_throws DimensionError CameraCalibrationMat((16.0mm, 16.0mm), [320.0px, 240.0px])
        @test_throws DimensionError LogicalIntrinsics(16.0mm, [320.0px, 240.0px])
    end

    @testset "image sizes: Size2 stripping and the diagonal normalization" begin
        @test ustrip.(px, Size2(width = 800px, height = 600px)) === Size2(800.0, 600.0)
        @test strip_or_bare(px, 3.0px) === 3.0 && strip_or_bare(px, 3.0) === 3.0
        for size in (Size2(width = 640, height = 480), Size2(width = 640px, height = 480px), (640, 480))
            d = DiagonalNormalization(size)
            @test (d.width, d.height) === (640.0, 480.0) && d.center == [320.0, 240.0]
        end
        @test_throws DimensionError DiagonalNormalization(Size2(width = 640mm, height = 480mm))
    end

    @testset "feature frames: pixels or a length, never mixed" begin
        @test ustrip(px, Disc(Point2(100.0px, 200.0px), 5.0px)) == Disc(Point2(100.0, 200.0), 5.0)
        @test ustrip(mm, Disc(Point2(1.0mm, 2.0mm), 0.5mm)) == Disc(Point2(1.0, 2.0), 0.5)
        @test VgtkCore.bbox(Disc(Point2(100.0px, 200.0px), 5.0px); scale = 3.0) ==
            VgtkCore.bbox(Disc(Point2(100.0, 200.0), 5.0); scale = 3.0)
    end
end

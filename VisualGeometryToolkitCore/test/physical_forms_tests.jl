# The physical forms: a lens on a sensor, its metric focal length, a pure translation, a target at the world's origin and a
# bare board posed in the world.

@testitem "physical forms: a lens on a sensor and its metric focal length" begin
    using Test
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore

    sensor = Sensor(Size2(320, 240), 20μm)
    camera = BoundedCamera(8mm, sensor; lens = Lens(DoubleSphere(0.25, 0.4)))
    K = calibration_matrix(VgtkCore.model(camera))
    @test (K[1, 1], K[2, 2], K[1, 3], K[2, 3]) == (400.0, 400.0, 160.0, 120.0)   # 8 mm / 20 μm, the sensor's centre
    @test extent(camera) == Size2(320, 240) && VgtkCore.sensor(camera) !== nothing
    # the general form it composes, stated once
    @test camera == BoundedCamera(CameraModel(8mm, sensor.pitch, centroid(Rect(sensor)) .* px; lens = Lens(DoubleSphere(0.25, 0.4))), sensor)
    @test ustrip(mm, metric_focal_length(camera)) ≈ 8.0
    @test lens_coverage(BoundedCamera(8mm, sensor; lens_coverage = Circle(Point2(160.0, 120.0), 150.0))) isa Circle
    @test_throws ArgumentError metric_focal_length(BoundedCamera(VgtkCore.model(camera), Size2(320, 240)))   # no pitch
    @test_throws Exception BoundedCamera(8.0, sensor)                                                            # a plain f is no length
end

@testitem "physical forms: a pure translation, a target at the origin, a bare board" begin
    using Test
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using LinearAlgebra: I
    using StaticArrays: SVector

    E = EuclideanMap(I, SVector(0.0, 0.0, 5.0))
    @test E(SVector(1.0, 2.0, 3.0)) == SVector(1.0, 2.0, 8.0) && VgtkCore.has_units(E) == false
    @test VgtkCore.has_units(EuclideanMap(I, [0m, 0m, 1m])) && EuclideanMap(I, zeros(2))(SVector(1.0, 2.0)) == SVector(1.0, 2.0)

    physical = PosedTarget(Board(Size2(width = 2m, height = 2m), [-1m, -1m]))   # the board's frame is the world's
    @test VgtkCore.has_units(physical) && VgtkCore.board(VgtkCore.model(physical)).size == Size2(width = 2m, height = 2m)
    @test pose(physical) == EuclideanMap(I, [0m, 0m, 0m])
    plain = PosedTarget(Board(Size2(width = 2.0, height = 2.0)))
    @test !VgtkCore.has_units(plain) && pose(plain) == EuclideanMap(I, zeros(3))
    lifted = PosedTarget(Board(Size2(width = 2.0, height = 2.0)), EuclideanMap(I, SVector(0.0, 0.0, 1.0)))
    @test pose(lifted)(SVector(0.0, 0.0, 0.0)) == SVector(0.0, 0.0, 1.0)
    t = PlanarTarget([SVector(0.0, 0.0), SVector(0.1, 0.0), SVector(0.0, 0.1)])
    @test pose(PosedTarget(t)) == EuclideanMap(I, zeros(3))
end

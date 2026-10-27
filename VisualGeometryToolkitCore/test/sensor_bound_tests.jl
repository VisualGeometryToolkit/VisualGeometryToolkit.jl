# A camera bounded by its physical sensor (cameras.jl): `BoundedCamera(camera,
# sensor)` keeps the sensor beside the pixel extent its resolution states, so
# `extent` reads the same whichever bound was supplied and `sensor(bc)` carries
# the pitch a datasheet's millimetres need.
@testitem "sensor bound" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: model, with_unit, has_units
    using StaticArrays
    using Rotations: RotMatrix, RotXYZ

    K = @SMatrix [1200.0 0.0 1224.0; 0.0 1200.0 1024.0; 0.0 0.0 1.0]
    cm = CameraModel(CameraCalibrationMat(K))
    s = CMOS_SENSORS["Sony"]["IMX250"]                 # 2448 × 2048 @ 3.45 μm
    bs = BoundedCamera(cm, s)
    be = BoundedCamera(cm, Size2(2448, 2048))

    @testset "the extent is the sensor's resolution" begin
        @test extent(bs) == extent(be) == Size2(2448, 2048)
        @test bs.extent == be.extent
        @test extent(bs) isa Size2{Int}
        @test (@inferred extent(bs)) == Size2(2448, 2048)
        @test Rect(s) == Rect(0, 0, extent(bs)...)
        @test model(bs) === cm
    end

    @testset "the sensor accessor" begin
        @test sensor(bs) == s
        @test sensor(bs).pitch.width ≈ 3.45μm / px
        @test sensor(be) === nothing
        @test bs isa BoundedCamera{<:CameraModel}
    end

    @testset "rebuilds keep the sensor" begin
        E = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.3, -0.1, 2.0))
        bp = BoundedCamera(PosedCamera(cm, E), s)
        @test sensor(with_model(bp, cm)) == s
        bu = with_unit(bp, mm)
        @test has_units(bu) && sensor(bu) == s
        @test sensor(ustrip(mm, bu)) == s
        @test extent(ustrip(mm, bu)) == Size2(2448, 2048)
    end
end

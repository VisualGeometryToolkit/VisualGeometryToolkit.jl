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

# The bound's kind is a type parameter: a bounded camera of an isbits camera model is itself isbits — a device kernel takes it as it
# is — and `convert_eltype` converts the wrapped camera, keeping the bound.
@testitem "sensor bound: isbits, convert_eltype" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: model, convert_eltype
    using StaticArrays
    K = @SMatrix [930.0 0.0 1224.0; 0.0 930.0 1024.0; 0.0 0.0 1.0]
    cm = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K)), KannalaBrandt(-0.04, 0.0004, 0.0002, -0.00008))
    s = CMOS_SENSORS["Sony"]["IMX250"]
    for bc in (BoundedCamera(cm, s), BoundedCamera(cm, Size2(2448, 2048)))
        @test isbits(bc)
        b32 = convert_eltype(Float32, bc)
        @test isbits(b32) && eltype(calibration_matrix(b32)) === Float32 && extent(b32) == extent(bc) && sensor(b32) == sensor(bc)
        @test backproject(b32, SVector(1500.0f0, 900.0f0)) ≈ backproject(bc, SVector(1500.0, 900.0)) rtol = 1.0e-5
    end
end

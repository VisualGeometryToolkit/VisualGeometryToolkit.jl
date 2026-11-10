@testitem "rotation validity tolerance follows the element type" begin
    using Test, LinearAlgebra, StaticArrays, Random
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore: RotationMap, _assert_rotation

    # Float32 roundings of exact rotations have a residual of a few Float32 ulps,
    # above the Float64 tolerance 10·√eps(Float64) ≈ 1.5e-7: they are rotations.
    rng = MersenneTwister(20261110)
    for _ in 1:200
        R = SMatrix{3, 3, Float32}(Matrix(RotXYZ(2π * rand(rng), 2π * rand(rng), 2π * rand(rng))))
        @test _assert_rotation(R) === nothing
        @test RotationMap(RotMatrix{3, Float32}(R)) isa RotationMap
    end

    # A Float32 matrix whose Float32 residual exceeds the old Float64 tolerance
    # (the residual is computed in Float32 arithmetic by Rotations.isrotation).
    R = SMatrix{3, 3, Float32}(Matrix(RotXYZ(0.3, 0.2, 0.1)))
    R = R + SMatrix{3, 3, Float32}(1.0f-7 * [0 1 0; -1 0 1; 0 -1 0])
    @test norm(SMatrix{3, 3, Float64}(R) * SMatrix{3, 3, Float64}(R)' - I) > 5.0e-8
    @test _assert_rotation(R) === nothing

    # A genuine non-rotation is still refused, in both precisions, and the
    # message reports the quantity that decided.
    for T in (Float32, Float64)
        S = SMatrix{3, 3, T}(1.01, 0, 0, 0, 1, 0, 0, 0, 1)
        @test_throws ArgumentError _assert_rotation(S)
        F = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, -1)
        @test_throws ArgumentError _assert_rotation(F)
    end
    @test_throws ArgumentError _assert_rotation(SMatrix{3, 3, Float32}(1.001, 0, 0, 0, 1, 0, 0, 0, 1) * 1.1f0)

    # Float64 is unchanged: the 1.5e-7 tolerance still refuses 1e-6 errors.
    R64 = SMatrix{3, 3, Float64}(Matrix(RotXYZ(0.3, 0.2, 0.1)))
    @test _assert_rotation(R64) === nothing
    @test_throws ArgumentError _assert_rotation(R64 + 1.0e-6 * SMatrix{3, 3}(0, 1, 0, -1, 0, 1, 0, -1, 0))
    @test _assert_rotation(R64; atol = 1.0e-3) === nothing
end

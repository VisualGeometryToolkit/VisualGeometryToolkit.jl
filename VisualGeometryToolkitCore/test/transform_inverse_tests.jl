# The inverse of a structured transform is of its own class EXACTLY: each map inverts analytically (an EuclideanMap's
# rotation by its transpose, a SimilarityMap by (1/s, Rᵀ, −Rᵀt/s), a CT affine by its own inverse), and lifting one to a
# HomographyMap stores that analytic inverse, so the stored inverse's last row is exactly 0 0 1 (a 3×3 inversion leaves
# 1 ± ulp there). A calibration matrix's inverse is analytic with exact structural zeros; a homography keeps the 3×3 inverse.
@testitem "transform matrix inverse" begin
    using Test
    using StaticArrays: SMatrix, SVector, @SMatrix
    using LinearAlgebra: I, Diagonal
    using Rotations: RotMatrix
    using CoordinateTransformations: Translation, LinearMap, AffineMap
    using VisualGeometryToolkitCore: HomographyMap, HomographyMat, CameraCalibrationMat, EuclideanMap, SimilarityMap,
        to_homogeneous

    c, s = cos(0.7), sin(0.7)
    R = RotMatrix{2}(0.7)
    maps = (
        LinearMap(R),
        Translation(SVector(3.7, -1.3)),
        EuclideanMap(R, SVector(3.7, -1.3)),
        LinearMap(2.5I),
        SimilarityMap(2.0, R, SVector(3.7, -1.3)),
        LinearMap(Diagonal(SVector(1.5, 0.25))),
        AffineMap(@SMatrix([1.3 -0.4; 0.2 0.9]), SVector(3.7, -1.3)),
    )
    for a in maps
        h = HomographyMap(a)
        Hi = inv(h).forward
        @test Hi isa SMatrix{3, 3, Float64, 9}
        @test (Hi[3, 1], Hi[3, 2], Hi[3, 3]) === (0.0, 0.0, 1.0)              # exactly affine
        @test Hi ≈ to_homogeneous(inv(a)) rtol = 4 * eps()                   # the map's analytic inverse (to fusion)
        @test to_homogeneous(a) * Hi ≈ I(3) atol = 1.0e-12
    end
    E = maps[3]
    @test SMatrix{2, 2}(to_homogeneous(inv(E))[1:2, 1:2]) == SMatrix{2, 2}(R)'   # rigid: the transpose, exactly
    @test inv(maps[2]) == Translation(SVector(-3.7, 1.3))

    K = CameraCalibrationMat{Float64}(1234.5, 0.0, 0.0, 0.3, 1230.1, 0.0, 2016.7, 1511.2, 1.0)
    Ki = inv(K)
    @test Ki isa SMatrix{3, 3, Float64, 9}                                     # K⁻¹ is not a calibration
    @test (Ki[2, 1], Ki[3, 1], Ki[3, 2], Ki[3, 3]) === (0.0, 0.0, 0.0, 1.0)      # exact structural entries
    @test SMatrix(K) * Ki ≈ I(3) atol = 1.0e-12
    @test inv(HomographyMap(K)).forward == Ki                                  # the lift stores the analytic inverse

    H = HomographyMat{Float64}(1.0, 0.1, 1.0e-4, 0.2, 0.9, -2.0e-4, 5.0, -3.0, 1.0)
    @test inv(H) isa HomographyMat
    @test SMatrix(H) * SMatrix(inv(H)) ≈ I(3) atol = 1.0e-12
end

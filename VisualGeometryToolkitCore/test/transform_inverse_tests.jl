# The inverse of a homogeneous transform matrix is of its own class EXACTLY (transforms/matrices.jl): an affine-family
# matrix's inverse has its last row exactly 0 0 1 (the 3×3 inverse left 1 ± ulp there), a rigid one's linear block is the
# transpose; the type is kept; a homography keeps the 3×3 inverse.
@testitem "transform matrix inverse" begin
    using Test
    using StaticArrays: SMatrix, SVector, @SMatrix
    using LinearAlgebra: I
    using VisualGeometryToolkitCore: HomRotationMat, HomTranslationMat, HomEuclideanMat, HomUniformScalingMat, HomSimilarityMat,
        HomScalingMat, HomAffineMat, HomographyMat, CameraCalibrationMat

    c, s = cos(0.7), sin(0.7)
    cases = (
        HomRotationMat{Float64}(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0),
        HomTranslationMat(SVector(3.7, -1.3)),
        HomEuclideanMat{Float64}(c, s, 0.0, -s, c, 0.0, 3.7, -1.3, 1.0),
        HomUniformScalingMat{2}(2.5),
        HomSimilarityMat{Float64}(2.0c, 2.0s, 0.0, -2.0s, 2.0c, 0.0, 3.7, -1.3, 1.0),
        HomScalingMat{Float64}(1.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0),
        HomAffineMat{Float64}(1.3, 0.2, 0.0, -0.4, 0.9, 0.0, 3.7, -1.3, 1.0),
        CameraCalibrationMat{Float64}(1234.5, 0.0, 0.0, 0.3, 1230.1, 0.0, 2016.7, 1511.2, 1.0),
    )
    for M in cases
        Mi = inv(M)
        @test typeof(Mi) == typeof(M)
        @test (Mi[3, 1], Mi[3, 2], Mi[3, 3]) === (0.0, 0.0, 1.0)              # exactly affine
        @test SMatrix(M) * SMatrix(Mi) ≈ I(3) atol = 1.0e-12
    end
    R, E = cases[1], cases[3]
    @test inv(R) == R' && SMatrix{2, 2}(inv(E)[1:2, 1:2]) == SMatrix{2, 2}(E[1:2, 1:2])'   # rigid: the transpose, exactly
    @test inv(cases[2]) == HomTranslationMat(SVector(-3.7, 1.3))
    @test inv(cases[8])[2, 1] == 0                                              # calibration: upper triangular again
    H = HomographyMat{Float64}(1.0, 0.1, 1.0e-4, 0.2, 0.9, -2.0e-4, 5.0, -3.0, 1.0)
    @test inv(H) isa HomographyMat
    @test SMatrix(H) * SMatrix(inv(H)) ≈ I(3) atol = 1.0e-12
end

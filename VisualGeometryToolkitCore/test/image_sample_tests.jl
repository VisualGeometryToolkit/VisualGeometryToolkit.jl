# =============================================================================
# image_sample! (geometry/transforms/image_sample.jl): the bilinear tap at
# scattered points — its value, its boundary, and its agreement with
# image_warp! at the points a warp lands on.
# =============================================================================

@testitem "image_sample!: scattered points, the warp's tap and boundary" begin
    using Test, StaticArrays, Random
    using CoordinateTransformations: AffineMap
    using GeometryBasics: Rect
    using VisualGeometryToolkitCore: image_sample!, image_warp, image_warp!

    rng = Xoshiro(11)
    src = rand(rng, Float32, 30, 40)
    @testset "a pixel centre reads its pixel; between centres the bilinear blend" begin
        out = zeros(Float32, 2, 2)
        pts = reshape([SVector(4.5f0, 2.5f0), SVector(0.5f0, 0.5f0), SVector(9.75f0, 6.5f0), SVector(39.5f0, 29.5f0)], 2, 2)
        image_sample!(out, src, pts)
        @test out[1, 1] == src[3, 5]                                       # (x, y) = (j − ½, i − ½)
        @test out[1, 2] ≈ 0.75f0 * src[7, 10] + 0.25f0 * src[7, 11]
        @test out[2, 1] == src[1, 1] && out[2, 2] == src[30, 40]
    end
    @testset "off the image: fillvalue, blended within half a pixel; non-finite is fillvalue" begin
        out = zeros(Float32, 4)
        image_sample!(out, src, [SVector(-5.0f0, 3.0f0), SVector(0.0f0, 0.5f0), SVector(NaN32, 3.0f0), SVector(3.0f0, 1.0f6)]; fillvalue = 2.0f0)
        @test out[1] == 2.0f0 && out[3] == 2.0f0 && out[4] == 2.0f0
        @test out[2] ≈ 0.5f0 * src[1, 1] + 0.5f0 * 2.0f0
    end
    @testset "the warp's value at the point it lands on" begin
        A = AffineMap(SMatrix{2, 2}(0.37, 0.05, -0.04, 0.41), SVector(3.3, 2.1))   # output COLMAP px → source COLMAP px
        dst = zeros(Float32, 25, 35)
        image_warp!(dst, image_warp(A, Rect(0, 0, 35, 25)), src; fillvalue = 0.25f0)
        pts = [SVector{2, Float64}(A(SVector(j - 0.5, i - 0.5))) for i in 1:25, j in 1:35]
        out = similar(dst)
        image_sample!(out, src, pts; fillvalue = 0.25f0)
        @test maximum(abs.(out .- dst)) < 1.0e-6
    end
    @testset "sizes must agree" begin
        @test_throws DimensionMismatch image_sample!(zeros(Float32, 3), src, [SVector(1.0, 1.0)])
        @test image_sample!(Float32[], src, SVector{2, Float64}[]) == Float32[]
    end
end

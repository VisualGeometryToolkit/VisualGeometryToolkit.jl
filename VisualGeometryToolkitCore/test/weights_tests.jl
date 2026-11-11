# =============================================================================
# Per-datum weights and their kind: weight_kind, inlier_mask(result)
# (robust/weights.jl; the robust-estimation contract, Q3)
# =============================================================================

@testitem "weights" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: FitResult, weights, weight_kind, inlier_mask,
        WeightKind, BinaryWeights, PosteriorWeights, IrlsWeights, RansacScoring, MsacScoring, StudentTUniformScoring

    # A metadata whose kind is its producer's type parameter, as RansacMetadata{T, S}.
    struct _Meta{P, T}
        weights::Vector{T}
    end
    VgtkCore.weight_kind(::Type{<:_Meta{P}}) where {P} = weight_kind(P)
    struct _Irls end
    VgtkCore.weight_kind(::Type{_Irls}) = IrlsWeights()
    result(P, w) = FitResult(0.0, :converged, 1, 0.0, length(w), _Meta{P, eltype(w)}(w))

    @testset "the kind is the producer type's trait" begin
        @test weight_kind(RansacScoring{1, Float64}) === BinaryWeights()
        @test weight_kind(MsacScoring{2, Float32}) === BinaryWeights()
        @test weight_kind(StudentTUniformScoring) === PosteriorWeights()
        r = result(MsacScoring{1, Float64}, [1.0, 0.0, 1.0])
        @test @inferred(weight_kind(r)) === BinaryWeights()
        @test weights(r) == [1.0, 0.0, 1.0]
        @test_throws ArgumentError weight_kind(FitResult(0.0, :converged, 1, 0.0, 1, nothing))
    end

    @testset "inlier_mask dispatches on the kind" begin
        w = [1.0, 0.0, 0.6, 0.4, 0.5]
        @test inlier_mask(result(MsacScoring{1, Float64}, w)) == [true, false, true, true, true]          # w > 0
        @test inlier_mask(result(StudentTUniformScoring, w)) == [true, false, true, false, false]        # MAP: > ½
        @test inlier_mask(result(StudentTUniformScoring, Float32.(w))) == [true, false, true, false, false]
        @test_throws ArgumentError inlier_mask(result(_Irls, w))                                         # no cut of its own
        @test inlier_mask(result(_Irls, w), 0.45) == [true, false, true, false, true]                    # an explicit cut
        @test inlier_mask(result(StudentTUniformScoring, w), 0.0) == [true, false, true, true, true]
        @test @inferred(inlier_mask(result(MsacScoring{1, Float64}, w))) isa BitVector
    end
end

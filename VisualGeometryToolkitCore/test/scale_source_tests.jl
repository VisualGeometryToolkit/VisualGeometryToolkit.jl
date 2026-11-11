# =============================================================================
# The robust stages' scale hand-off: ScaleSource, noise_scale, scale_source,
# scale_input (base/scale_source.jl; the robust-estimation contract, Q1/Q2)
# =============================================================================

@testitem "scale_source" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: FitResult, ScaleSource, ScaleGiven, ScaleUpstream,
        ScaleEstimated, ScaleNone, noise_scale, scale_source, ScaleInput, scale_input

    # A robust metadata as every verb's will carry it: one scale, one source.
    struct _RobustMeta{T}
        scale::T
        scale_source::ScaleSource
        weights::Vector{T}
    end
    result(s, src; T = Float64) = FitResult(1.0, :converged, 3, 0.1, 10, _RobustMeta{T}(T(s), src, T[1, 1]))

    @testset "the enum: isbits, four values, prefixed" begin
        @test isbitstype(ScaleSource)
        @test sizeof(ScaleSource) == 1
        @test Set(instances(ScaleSource)) == Set((ScaleGiven, ScaleUpstream, ScaleEstimated, ScaleNone))
        # the prefix keeps clear of Core's public update-policy type `Estimated`
        @test VgtkCore.Estimated !== ScaleEstimated
        @test isconcretetype(fieldtype(_RobustMeta{Float64}, :scale_source))
    end

    @testset "noise_scale / scale_source read the record, type-stably" begin
        for src in (ScaleGiven, ScaleUpstream, ScaleEstimated)
            r = result(0.7, src)
            @test @inferred(noise_scale(r)) === 0.7
            @test @inferred(scale_source(r)) === src
        end
        r32 = result(0.5, ScaleEstimated; T = Float32)
        @test @inferred(noise_scale(r32)) === 0.5f0
        # ScaleNone: the NaN is never handed on
        none = result(NaN, ScaleNone)
        @test @inferred(scale_source(none)) === ScaleNone
        @test_throws ArgumentError noise_scale(none)
        @test_throws ArgumentError scale_input(none, nothing)
        # a lean result (no metadata) records no scale either
        lean = FitResult(1.0, :converged, 1, 0.0, 3, nothing)
        @test scale_source(lean) === ScaleNone
        @test_throws ArgumentError noise_scale(lean)
    end

    @testset "scale_input: exactly one of scale, an upstream result, width" begin
        s = @inferred scale_input(2.0, nothing)
        @test s isa ScaleInput{Float64}
        @test s.scale === 2.0 && isnan(s.width) && s.source === ScaleGiven
        w = @inferred scale_input(nothing, 3)
        @test w.width === 3.0 && isnan(w.scale) && w.source === ScaleGiven
        u = @inferred scale_input(result(0.4, ScaleEstimated), nothing)
        @test u.scale === 0.4 && isnan(u.width) && u.source === ScaleUpstream
        f = @inferred scale_input(0.25f0, nothing)
        @test f isa ScaleInput{Float32} && f.scale === 0.25f0
        # neither, both, and non-positive or non-finite values are refused
        @test_throws ArgumentError scale_input(nothing, nothing)
        @test_throws ArgumentError scale_input(1.0, 2.0)
        @test_throws ArgumentError scale_input(result(0.4, ScaleGiven), 2.0)
        for bad in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError scale_input(bad, nothing)
            @test_throws ArgumentError scale_input(nothing, bad)
        end
    end
end

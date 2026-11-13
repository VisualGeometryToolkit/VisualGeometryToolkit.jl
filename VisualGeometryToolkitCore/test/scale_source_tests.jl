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
    result(s, src; T = Float64) = FitResult(1.0, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 3, 0.1, 10, _RobustMeta{T}(T(s), src, T[1, 1]))

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
        lean = FitResult(1.0, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 1, 0.0, 3, nothing)
        @test scale_source(lean) === ScaleNone
        @test_throws ArgumentError noise_scale(lean)
    end

    @testset "scale_input: exactly one of scale, an upstream result, width" begin
        s = @inferred scale_input(2.0, nothing)
        @test s isa ScaleInput{Float64}
        @test s.scale === 2.0 && isnan(s.width) && s.source === ScaleGiven
        w = @inferred scale_input(nothing, 3)
        # D-w: a width-only stage records no scale (never w / c) and the width itself
        @test w.width === 3.0 && isnan(w.scale) && w.source === ScaleNone
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

    @testset "a scale input answers the accessor pair, as a result does" begin
        # a stage records what it was given through noise_scale / scale_source, never the fields
        for (inp, src) in ((scale_input(2.0, nothing), ScaleGiven), (scale_input(result(0.4, ScaleEstimated), nothing), ScaleUpstream))
            @test @inferred(scale_source(inp)) === src
            @test @inferred(noise_scale(inp)) === inp.scale
        end
        w = scale_input(nothing, 3.0)
        @test @inferred(scale_source(w)) === ScaleNone
        @test_throws ArgumentError noise_scale(w)            # a width hands on no scale
        @test scale_source(nothing) === ScaleNone            # no input (least squares)
    end
end

# The refine contract's loss width (hoisted from Estimation's `_refine_loss` and PoseLib's `_refine_width`, which had
# drifted): least squares takes neither keyword; `scale` multiplies the strategy's constant (its own, else the table's);
# `width` is the width itself and is refused beside a strategy's own width mode, named for what that mode is.
@testitem "refine_width: scale xor width, the strategy's constant, least squares" begin
    using Test
    using VisualGeometryToolkitCore: refine_width, Irls, WelschShape, TLSShape, EfficiencyConstant, LossWidth,
        efficiency_constant, ScaleGiven, ScaleNone, FitResult
    lsq = refine_width(nothing, nothing, nothing, 2)
    @test isnan(lsq.width) && isnan(lsq.efficiency_constant) && lsq.input === nothing
    @test_throws "has no loss width" refine_width(nothing, 0.5, nothing, 2)
    @test_throws "has no loss width" refine_width(nothing, nothing, 2.0, 2)
    w = Irls{WelschShape}(25, nothing)
    r = refine_width(w, 0.5, nothing, 2)
    @test r.efficiency_constant == efficiency_constant(WelschShape, 2) && r.width == r.efficiency_constant * 0.5 &&
          r.input.source === ScaleGiven
    r = refine_width(Irls{WelschShape}(25, EfficiencyConstant(3.0)), 0.5, nothing, 1)
    @test r.width == 1.5 && r.efficiency_constant == 3.0
    r = refine_width(w, nothing, 2.0, 2)
    @test r.width == 2.0 && r.efficiency_constant == 1.0 && r.input.source === ScaleNone && isnan(r.input.scale)
    @test_throws "give `scale` or `width`, not both" refine_width(w, 0.5, 2.0, 2)
    @test_throws ArgumentError refine_width(w, nothing, nothing, 2)
    # a stated constant would be overridden by `width`; a stated absolute width contradicted (not "a constant")
    @test_throws "states its efficiency constant 3.0" refine_width(Irls{WelschShape}(25, EfficiencyConstant(3.0)), nothing, 2.0, 2)
    @test_throws "states its own width" refine_width(Irls{TLSShape}(25, LossWidth(2.0)), nothing, 1.0, 2)
    # a truncation with no constant takes `width`, and refuses `scale` (no constant to multiply)
    @test refine_width(Irls{TLSShape}(25, nothing), nothing, 1.5, 2).width == 1.5
    @test_throws "no efficiency constant" refine_width(Irls{TLSShape}(25, nothing), 0.5, nothing, 2)
end

# The unit-bearing gate (hoisted from Estimation's `_assert_unit_bearings` and PoseLib's copy): finite first, then unit.
@testitem "assert_unit_bearings: finite, then unit, naming the caller and the remedy" begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: assert_unit_bearings
    b = [normalize(SVector(0.1i, -0.05i, 1.0)) for i in 1:10]
    @test assert_unit_bearings(b, "source", "MyProblem"; remedy = "Back-project.") === nothing
    @test assert_unit_bearings(map(x -> SVector{3, Float32}(x), b), "source", "MyProblem"; remedy = "") === nothing
    chart = [SVector(0.1i, -0.05i, 1.0) for i in 1:10]
    @test_throws "MyProblem needs UNIT bearings; the target at index 10" assert_unit_bearings(chart, "target", "MyProblem"; remedy = "x")
    @test_throws "Use the camera." assert_unit_bearings(chart, "target", "MyProblem"; remedy = "Use the camera.")
    withnan = copy(b); withnan[4] = SVector(NaN, 0.0, 1.0); withnan[7] = 2 .* withnan[7]
    @test_throws "MyProblem needs finite bearings; the source at index 4" assert_unit_bearings(withnan, "source", "MyProblem"; remedy = "")
    @test_throws UndefKeywordError assert_unit_bearings(b, "source", "MyProblem")
end

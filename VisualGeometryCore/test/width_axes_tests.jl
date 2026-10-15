# WidthAxis tags, their persisted names, and the scale multiple → loss width
# map (robust/width_axes.jl).

@testitem "width_axes" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: WidthAxis, ScaleMultiple, LossWidth, OutlierHalfwidth, width_axis, width_axis_name, to_native
    struct _UnnamedAxis <: WidthAxis end   # a WidthAxis outside the persisted table

    @testset "the persisted names" begin
        @test width_axis("scale_multiple") === ScaleMultiple()
        @test width_axis("loss_width") === LossWidth()
        @test width_axis("outlier_halfwidth") === OutlierHalfwidth()
        @test width_axis(SubString("xloss_width", 2)) === LossWidth()
        @test all(a -> a isa WidthAxis, (ScaleMultiple(), LossWidth(), OutlierHalfwidth()))
        @test_throws ArgumentError width_axis("k")
        @test_throws ArgumentError width_axis("LossWidth")
        @test width_axis(:loss_width) === LossWidth()
        @test_throws ArgumentError width_axis(:k)
    end

    @testset "width_axis_name is the inverse" begin
        for a in (ScaleMultiple(), LossWidth(), OutlierHalfwidth())
            @test width_axis(width_axis_name(a)) === a
            @test width_axis(String(width_axis_name(a))) === a
        end
        @test width_axis_name(OutlierHalfwidth()) === :outlier_halfwidth
        @test @inferred(width_axis_name(LossWidth())) === :loss_width
        # an axis outside the persisted table has no name: no method, not a fallback
        @test_throws MethodError width_axis_name(_UnnamedAxis())
    end

    @testset "to_native on the loss-width axis: the truncation flips at its width" begin
        @test to_native(LossWidth(), 2.5, 0.4; d_g = 1) == 2.5 * 0.4
        @test to_native(LossWidth(), 2.5, 0.4; d_g = 2) == to_native(LossWidth(), 2.5, 0.4; d_g = 1)
        @test to_native(LossWidth(), 3, 2; d_g = 1) === 6.0
        @test to_native(LossWidth(), 2.5, 0.4; d_g = 1) == decision_width(TLSShape(), 2.5 * 0.4)
        @test_throws UndefKeywordError to_native(LossWidth(), 2.5, 0.4)   # the codimension is always stated
        @test_throws MethodError to_native(ScaleMultiple(), 2.5, 0.4; d_g = 1)
    end
end

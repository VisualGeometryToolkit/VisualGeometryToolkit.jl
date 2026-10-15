# The scale multiple → native maps on the outlier-half-width axis, the curried
# per-axis map, the native-axis conversions at an operating scale multiple, and
# the absolute loss width a `LossWidth` value carries (robust/width_axes.jl;
# moved from RansacScoringDoneRight, K+ 2026-10-09).

@testitem "native_width_maps" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: ScaleMultiple, LossWidth, OutlierHalfwidth, to_native, native_map,
        outlier_halfwidth_to_loss_width, loss_width_to_outlier_halfwidth, loss_width,
        GauInlier, decision_width, decision_threshold, with_width, MsacScoring, RansacScoring

    @testset "to_native(::OutlierHalfwidth): the Gaussian-uniform crossover" begin
        for d_g in (1, 2), scale in (0.01, 1.0, 3.0), k in (0.8, 2.0, 6.0)
            h = to_native(OutlierHalfwidth(), k, scale; d_g)
            @test h == decision_width(GauInlier(scale^2; d_g), k * scale)
            # the MAP rule of that mixture flips at the residual k·scale
            @test decision_threshold(GauInlier(scale^2; d_g), h) ≈ k * scale rtol = 1e-10
        end
        # d_g is the outlier density's exponent: it moves the native width by decades
        @test to_native(OutlierHalfwidth(), 6.0, 1.0; d_g = 1) ≈ 8.2293e7 rtol = 1e-4
        @test to_native(OutlierHalfwidth(), 6.0, 1.0; d_g = 2) ≈ 1.0156e4 rtol = 1e-4
        @test_throws UndefKeywordError to_native(OutlierHalfwidth(), 2.0, 1.0)
    end

    @testset "native_map: the curried to_native" begin
        for axis in (OutlierHalfwidth(), LossWidth()), d_g in (1, 2)
            f = native_map(axis; d_g)
            @test f(3.0, 0.2) == to_native(axis, 3.0, 0.2; d_g)
        end
        @test_throws MethodError native_map(ScaleMultiple(); d_g = 1)
        @test_throws UndefKeywordError native_map(LossWidth())
    end

    @testset "outlier half-width <-> loss width at an operating scale multiple" begin
        # round trip
        for operating_scale_multiple in (1.0, 2.5, 4.0), outlier_halfwidth in (0.7, 5.0, 120.0), d_g in (1, 2)
            w = outlier_halfwidth_to_loss_width(outlier_halfwidth; operating_scale_multiple, d_g)
            @test loss_width_to_outlier_halfwidth(w; operating_scale_multiple, d_g) ≈
                  outlier_halfwidth rtol = 1.0e-12
        end
        # d_g = 1: the closed form around the unit-scale crossover, bitwise
        uh1(m) = decision_width(GauInlier(1.0; d_g = 1), Float64(m))
        for m in (1.0, 2.5, 4.0), x in (0.004, 0.05, 0.7, 120.0)
            @test outlier_halfwidth_to_loss_width(x; operating_scale_multiple = m, d_g = 1) === Float64(m) * x / uh1(m)
            @test loss_width_to_outlier_halfwidth(x; operating_scale_multiple = m, d_g = 1) === uh1(m) * x / Float64(m)
        end
        # d_g = 2: the codim-2 boundary itself — the half-width at scale ŝ maps
        # to the loss width m·ŝ
        for m in (1.0, 2.5, 4.0), ŝ in (0.01, 0.3, 2.0)
            h = decision_width(GauInlier(ŝ^2; d_g = 2), m * ŝ)
            @test outlier_halfwidth_to_loss_width(h; operating_scale_multiple = m, d_g = 2) ≈ m * ŝ rtol = 1.0e-12
            # ≡ to_native on the two axes at the same (m, ŝ)
            @test outlier_halfwidth_to_loss_width(to_native(OutlierHalfwidth(), m, ŝ; d_g = 2);
                                                  operating_scale_multiple = m, d_g = 2) ≈
                  to_native(LossWidth(), m, ŝ; d_g = 2) rtol = 1.0e-12
        end
        @test_throws UndefKeywordError outlier_halfwidth_to_loss_width(1.0; operating_scale_multiple = 2.5)
        @test_throws UndefKeywordError loss_width_to_outlier_halfwidth(1.0; operating_scale_multiple = 2.5)
    end

    @testset "loss_width(::LossWidth): the absolute width of a factored value" begin
        for w in (0.003, 0.5, 2.0, 17.0)
            @test loss_width(LossWidth(w)) === w || loss_width(LossWidth(w)) == 1.0 * sqrt(Float64(w)^2)
        end
        @test loss_width(LossWidth(2.5, 0.16)) == 2.5 * sqrt(0.16)
        @test_throws MethodError loss_width(LossWidth())        # the bare axis carries no width
    end

    @testset "with_width on the threshold scorers: the threshold is the loss width" begin
        for w in (0.01, 0.4, 3.0), dg in (1, 2)
            @test with_width(MsacScoring(dg), LossWidth(w)) == MsacScoring(dg; threshold = w)
            @test with_width(RansacScoring(dg), LossWidth(w)) == RansacScoring(dg; threshold = w)
            @test with_width(MsacScoring(dg; threshold = 9.0), LossWidth(w)) == MsacScoring(dg; threshold = w)
        end
        @test VisualGeometryCore.width_axis(MsacScoring(1; threshold = 0.5)) === LossWidth()
        @test VisualGeometryCore.width_axis(RansacScoring()) === LossWidth()
    end
end

@testitem "SplineLens: backward lens equals the old central grid" begin
    using Test
    using StaticArrays
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, CentralGenericModel, KannalaBrandt, px, backproject,
        calibration_matrix, with_present, valid_domain, Rect2, CameraCalibrationMat, LogicalIntrinsics
    using VisualGeometryToolkitCore.Cameras: _calibrated_rect
    using GeometryBasics: origin, widths

    # =============================================================================
    # The central grid as a CameraModel lens: the backward map against the golden
    # outputs of the old CentralGenericModel (test/fixtures/spline_lens_golden.jl,
    # recorded before the fold-in), the gauge, the re-gauge and the valid domain.
    # =============================================================================

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "spline_lens_golden.jl"))
    G = GOLDEN
    src = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
    image = (G.image[1], G.image[2])
    grid = (G.grid_size[1], G.grid_size[2])
    cam = CameraModel(SplineLens, src, image; grid_size = grid)
    masked = with_present(cam, G.masked.present)
    pixels = [SVector(G.pixels[1, i], G.pixels[2, i]) for i in axes(G.pixels, 2)]

    # The largest deviation from a golden 3 × N backprojection where it is finite, and
    # whether the camera is NaN exactly where the golden value is.
    function deviation(c, ref)
        worst, same_nan = 0.0, true
        for (i, u) in enumerate(pixels)
            b, r = backproject(c, u), SVector(ref[1, i], ref[2, i], ref[3, i])
            if any(isnan, r)
                same_nan &= all(isnan, b)
            else
                same_nan &= all(isfinite, b)
                worst = max(worst, norm(b - r))
            end
        end
        return worst, same_nan
    end

    @testset "backproject equals the old grid's, NaN where it is" begin
        worst, same_nan = deviation(cam, G.central.backproject)
        @test same_nan
        @test worst <= 1.0e-12
        worst_m, same_nan_m = deviation(masked, G.masked.backproject)
        @test same_nan_m
        @test worst_m <= 1.0e-12
        @test count(i -> !isnan(G.masked.backproject[1, i]), axes(G.pixels, 2)) >= 100   # the masked comparison bites
    end

    @testset "a camera source keeps its K; skew is an ArgumentError" begin
        @test calibration_matrix(cam) == calibration_matrix(src)
        skewed = CameraModel(
            LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(300.0, 0.0, 0.0, 1.0, 300.0, 0.0, 500.0, 500.0, 1.0))),
            KannalaBrandt(G.kb...)
        )
        @test_throws ArgumentError CameraModel(SplineLens, skewed, image)
    end

    @testset "the presence mask: all-true is nothing, all-false and a wrong size throw" begin
        b = cam.lens.backward
        @test SplineLens(b; present = trues(size(b.controls))).present === nothing
        @test masked.lens.present isa Matrix{Bool}
        @test_throws ArgumentError SplineLens(b; present = falses(size(b.controls)))
        @test_throws DimensionMismatch SplineLens(b; present = trues(3, 3))
        @test all(isnan, masked.lens.backward.controls[1, 1])                    # an absent slot holds NaN
    end

    @testset "the gauge of a function source (4256 × 2848 Kannala–Brandt)" begin
        f, pp = 1754.7, SVector(2133.6, 1422.5)
        kb = CameraModel(f * px, SA[pp[1] * px, pp[2] * px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))
        ray = u -> backproject(kb, u)
        for (gs, tol) in (((24, 16), 1.0e-3), ((70, 48), 1.0e-5))           # 70 × 48 is the default grid
            c = CameraModel(SplineLens, ray, (4256, 2848); grid_size = gs, init = :quasi_interpolate)
            K = calibration_matrix(c)
            @test norm(SVector(K[1, 3], K[2, 3]) - pp) <= tol
            @test abs(K[1, 1] / f - 1) <= tol
            @test abs(K[2, 2] / f - 1) <= tol
            @test K[1, 2] == 0
        end
        @test CameraModel(SplineLens, ray, (4256, 2848); init = :quasi_interpolate).lens.backward.controls |> size == (48, 70)
    end

    @testset "ẑ not imaged in the calibrated area: the gauge is the area's centre" begin
        ray = u -> backproject(src, u)
        c = CameraModel(SplineLens, ray, image; calib_area = (600, 600, 999, 999), init = :quasi_interpolate)
        K = calibration_matrix(c)
        @test (K[1, 3], K[2, 3]) == (800.0, 800.0)
        @test K[1, 1] > 0 && K[2, 2] > 0
        @test all(isfinite, backproject(c, SVector(800.0, 800.0)))
    end

    @testset "a re-gauge CameraModel(cam, K′) keeps every pixel's ray" begin
        K = calibration_matrix(cam)
        K′ = CameraCalibrationMat(SMatrix{3, 3, Float64}(0.5 * K[1, 1], 0.0, 0.0, 0.0, 0.5 * K[2, 2], 0.0, K[1, 3], K[2, 3], 1.0))
        cam′ = CameraModel(cam, K′)
        @test calibration_matrix(cam′) == K′
        worst = 0.0
        for u in pixels
            b, b′ = backproject(cam, u), backproject(cam′, u)
            @test isnan(b[1]) == isnan(b′[1])
            isnan(b[1]) || (worst = max(worst, norm(b′ - b)))
        end
        @test worst <= 1.0e-12
    end

    @testset "inference, with and without a forward and a mask" begin
        forward = ChartSpline(Rect2(-1.0, -1.0, 2.0, 2.0), fill(SVector(0.0, 0.0), 4, 4))
        with_forward = CameraModel(cam.intrinsics, SplineLens(cam.lens.backward; forward))
        for c in (cam, masked, with_forward)
            @test @inferred(backproject(c, SVector(500.0, 400.0))) isa StaticVector{3, Float64}
        end
        @test @inferred(backproject(cam, SVector(500.0f0, 400.0f0))) isa StaticVector{3, Float64}
    end

    @testset "valid_domain: the calibrated area in pixels; under a mask, the fully present windows" begin
        old = CentralGenericModel(src, image; grid_size = grid)
        r = @inferred valid_domain(cam)
        @test r isa Rect2{Float64}
        @test r == _calibrated_rect(old)
        # the golden mask, independently: window (ix, iy) reads controls [iy-2:iy+1, ix-2:ix+1] and covers the
        # cell from node ix − 1 to node ix (node j at pixel (j − 2)·h, h = 1000 / (n − 3))
        P = G.masked.present
        gh, gw = size(P)
        full = [(ix, iy) for iy in 3:(gh - 1), ix in 3:(gw - 1) if all(P[(iy - 2):(iy + 1), (ix - 2):(ix + 1)])]
        hx, hy = 1000.0 / (gw - 3), 1000.0 / (gh - 3)
        x0, x1 = 0.0 + (minimum(first, full) - 3) * hx, 0.0 + (maximum(first, full) - 2) * hx
        y0, y1 = 0.0 + (minimum(last, full) - 3) * hy, 0.0 + (maximum(last, full) - 2) * hy
        rm = @inferred valid_domain(masked)
        @test rm isa Rect2{Float64}
        @test rm == Rect2{Float64}(SVector(x0, y0), SVector(x1 - x0, y1 - y0))
        @test all(widths(rm) .< widths(r))                                       # the 3-ring mask shrinks it
        # every finite golden backprojection of the masked camera lies in it
        @test all(i -> isnan(G.masked.backproject[1, i]) || (origin(rm)[1] <= pixels[i][1] <= origin(rm)[1] + widths(rm)[1] &&
            origin(rm)[2] <= pixels[i][2] <= origin(rm)[2] + widths(rm)[2]), axes(G.pixels, 2))
    end
end

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

@testitem "SplineLens: project" begin
    using Test
    using StaticArrays
    using LinearAlgebra: norm, normalize
    using Adapt: adapt
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, px, project, backproject,
        calibration_matrix, with_present, Rect2, convert_eltype, equidistant_chart, LogicalIntrinsics, CameraCalibrationMat
    using VisualGeometryToolkitCore.Cameras: _lens_axis, _lattice_spacing, _spline_camera

    # =============================================================================
    # project through a SplineLens against the golden outputs of the old grid
    # models (test/fixtures/spline_lens_golden.jl): the LM without a forward spline
    # (the old CentralGenericModel's), the forward spline (the old SplineProjection's),
    # the batched and the Float32 device forms.
    # =============================================================================

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "spline_lens_golden.jl"))
    G = GOLDEN
    src = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
    image = (G.image[1], G.image[2])
    cam = CameraModel(SplineLens, src, image; grid_size = (G.grid_size[1], G.grid_size[2]))
    masked = with_present(cam, G.masked.present)
    K = calibration_matrix(cam)
    n = G.n_lattice
    lattice = [SVector(G.pixels[1, i], G.pixels[2, i]) for i in 1:n]
    rays_in = [SVector(G.central.backproject[1, i], G.central.backproject[2, i], G.central.backproject[3, i]) for i in 1:n]
    rays_out = [SVector(G.rays_out[1, i], G.rays_out[2, i], G.rays_out[3, i]) for i in axes(G.rays_out, 2)]

    # The largest deviation from a golden 2 × N projection where it is finite, and
    # whether the camera is NaN exactly where the golden value is.
    function deviation(c, rays, ref)
        worst, same_nan = 0.0, true
        for (i, X) in enumerate(rays)
            u, r = project(c, X), SVector(ref[1, i], ref[2, i])
            if any(isnan, r)
                same_nan &= all(isnan, u)
            else
                same_nan &= all(isfinite, u)
                worst = max(worst, norm(u - r))
            end
        end
        return worst, same_nan
    end

    # The old SplineProjection's chart is about the ray of the fit's quadrature centroid,
    # the pixel (499.5, 499.5), not of the principal point (500, 500): 2.4e-3 rad apart.
    # The lens's chart is about the ray of its principal point, so the forward spline is
    # read on the same pixel → ray map in the gauge whose principal point is that pixel
    # (K′, pp = (499.5, 499.5), the same controls over the same pixel rectangle), its
    # pixel output through K′⁻¹.
    S = G.spline
    K′ = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0.0, 0.0, 0.0, K[2, 2], 0.0, 499.5, 499.5, 1.0)))
    cam′ = _spline_camera(K′, Rect2(0.0, 0.0, 1000.0, 1000.0), cam.lens.backward.controls)
    C = [SVector((S.controls[1, i, j] - 499.5) / K[1, 1], (S.controls[2, i, j] - 499.5) / K[2, 2]) for i in axes(S.controls, 2), j in axes(S.controls, 3)]
    forward = ChartSpline(Rect2(S.area[1], S.area[2], S.area[3], S.area[4]), C)
    cam_fwd = CameraModel(K′, SplineLens(cam′.lens.backward; forward))
    masked_fwd = with_present(cam_fwd, G.masked.present)
    rays64 = [SVector(S.rays[1, i], S.rays[2, i], S.rays[3, i]) for i in axes(S.rays, 2)]

    @testset "without a forward: the old central grid's projection" begin
        worst, same_nan = deviation(cam, rays_in, G.central.project_in)
        @test same_nan
        @test worst <= 1.0e-9
        @test all(X -> all(isnan, project(cam, X)), rays_out)
        worst_m, same_nan_m = deviation(masked, rays_in, G.masked.project_in)
        @test same_nan_m
        @test worst_m <= 1.0e-9
        @test count(i -> !isnan(G.masked.project_in[1, i]), 1:n) >= 100        # the masked comparison bites
        @test all(X -> all(isnan, project(masked, X)), rays_out)
    end

    @testset "with a forward: the old SplineProjection on the lens axis" begin
        @test norm(_lens_axis(cam.lens) - S.chart_rotation[3, :]) > 1.0e-3            # the principal point's ray is not the golden chart's axis
        @test norm(_lens_axis(cam_fwd.lens) - S.chart_rotation[3, :]) <= 1.0e-12
        @test masked_fwd.lens.forward === forward
        worst, same_nan = deviation(cam_fwd, rays64, S.project)
        @test same_nan
        @test worst <= 1.0e-9
        # the forward agrees with the LM to 0.01 px on the 64 golden rays
        @test maximum(X -> norm(project(cam_fwd, X) - project(cam′, X)), rays64) <= 0.01
        # a ray two knot cells beyond the forward's chart is not imaged
        o, w = S.area[1:2], S.area[3:4]
        hx, _ = _lattice_spacing(forward.area, size(forward.controls))
        c = SVector(o[1] + w[1] + 2hx, o[2] + w[2] / 2)
        θ = norm(c)
        R = SMatrix{3, 3}(S.chart_rotation)
        X_out = R' * SVector(sin(θ) * c[1] / θ, sin(θ) * c[2] / θ, cos(θ))
        @test norm(equidistant_chart(_lens_axis(cam_fwd.lens))(X_out) - c) <= 1.0e-9
        @test all(isnan, project(cam_fwd, X_out))
    end

    @testset "batched: the golden batched outputs; the scalar solve exactly" begin
        for (c, ref) in ((cam, G.central), (masked, G.masked))
            b = project(c, rays_in, lattice)
            @test b.ok == ref.batch_project_ok
            @test any(b.ok)
            @test maximum(i -> norm(b.px[i] - SVector(ref.batch_project_px[1, i], ref.batch_project_px[2, i])), findall(b.ok)) <= 1.0e-9
            @test all(i -> all(isnan, b.px[i]), findall(!, b.ok))
        end
        # the same seed and budget as the scalar form: the same solve, bit for bit
        centre = fill(SVector(500.0, 500.0), length(rays64))
        for c in (cam, masked, cam_fwd, masked_fwd)
            b = project(c, rays64, centre; iters = 100)
            @test all(i -> isequal(b.px[i], SVector{2, Float64}(project(c, rays64[i]))), eachindex(rays64))
            @test b.ok == [all(isfinite, project(c, X)) for X in rays64]
        end
        @test project(cam, SVector{3, Float64}[], SVector{2, Float64}[]).px == SVector{2, Float64}[]
        @test_throws ArgumentError project(cam, rays_in, lattice[1:3])
    end

    @testset "batched backproject: ChartSpline's kernel after K⁻¹; the Float32 device form" begin
        pixels = [SVector(G.pixels[1, i], G.pixels[2, i]) for i in axes(G.pixels, 2)]
        for c in (cam, masked)
            r = backproject(c, pixels)
            @test all(i -> isequal(r[i], SVector{3, Float64}(backproject(c, pixels[i]))), eachindex(pixels))
        end
        c32 = convert_eltype(Float32, cam)
        @test eltype(c32.lens.backward.controls) == SVector{3, Float32}
        d = adapt(Array, c32)
        pixels32 = [SVector{2, Float32}(u) for u in pixels]
        r32 = backproject(d, pixels32)
        @test eltype(r32) == SVector{3, Float32}
        # The Float32 camera images every pixel the Float64 one does, the area's own
        # edge (u = 0 or v = 0) included: batched and scalar, Float32 and Float64 pixels.
        for (name, rays) in (
                ("batched, Float32 pixels", r32),
                ("batched, Float64 pixels", backproject(d, pixels)),
                ("scalar, Float32 pixels", [backproject(c32, u) for u in pixels32]),
                ("scalar, Float64 pixels", [backproject(c32, u) for u in pixels]),
            )
            worst, nan_mismatch, edge_nan = 0.0, 0, 0
            for (i, u) in enumerate(pixels)
                a, b = backproject(cam, u), SVector{3, Float64}(rays[i])
                if isnan(a[1]) != isnan(b[1])
                    nan_mismatch += 1
                    (u[1] == 0 || u[2] == 0) && (edge_nan += 1)
                end
                isnan(a[1]) || isnan(b[1]) || (worst = max(worst, norm(b - a)))
            end
            @test (name, edge_nan) == (name, 0)
            @test (name, nan_mismatch) == (name, 0)
            @test worst <= 1.0e-5
        end
        m32 = convert_eltype(Float32, masked_fwd)
        @test m32.lens.present == masked.lens.present
        @test eltype(m32.lens.forward.controls) == SVector{2, Float32}
    end

    @testset "inference, with and without a forward and a mask" begin
        X = SVector(0.1, -0.2, 1.0)
        for c in (cam, masked, cam_fwd, masked_fwd)
            @test @inferred(project(c, X)) isa StaticVector{2, Float64}
            @test @inferred(project(c.lens, X)) isa SVector{2, Float64}
        end
    end
end

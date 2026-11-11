@testitem "spline camera" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: SplineLens, ChartSpline, Division, is_posed, cubic_bspline_weights, spline_window,
        spline_lattice, calibration_matrix, Rect2, spline_weights, directions, grid_size
    using VisualGeometryToolkitCore.Cameras: _cubic_bspline_weights_deriv, _continued_window, _eval_window, _normalized_point
    using StaticArrays
    using FiniteDiff: FiniteDiff
    using GeometryBasics: Vec3
    using LinearAlgebra: norm, normalize, dot
    using Random: MersenneTwister
    using BSplineKit: BSplineBasis, BSplineOrder, Spline
    using ForwardDiff: ForwardDiff

    # =============================================================================
    # The central grid camera (Schöps et al. CVPR 2020) as a SplineLens camera
    # =============================================================================
    #
    # Covers: the ported uniform-cubic window weights against BSplineKit (the
    # oracle pattern of bspline_tests.jl — the standard package validates the
    # scalar port), project∘backproject round trips and parity with the
    # parametric source it was initialized from, the calibrated-area contract
    # (NaN outside), the z-depth unproject, the undistortion, the precision
    # bridge, the batched projection and the window a point reads. The
    # non-central model's cases are in noncentral_spline_model_tests.jl.

    @testset "the central grid camera (Schöps et al.)" begin
        @testset "spline window weights — BSplineKit oracle" begin
            rng = MersenneTwister(7)
            n = 11
            c = randn(rng, n)
            # Uniform cubic B-spline with integer knots: basis j (order 4) on
            # knots -3:n covers exactly n control points.
            basis = BSplineBasis(BSplineOrder(4), Float64.(collect(-3:n)); augment = Val(false))
            S = Spline(basis, c)
            # Build a 1-D-in-x test through the 2-D window: constant in y. The
            # ported window eval at grid coordinate g equals the BSplineKit
            # uniform cubic spline at parameter g − 1 (the reference's "+2 so the
            # SURROUNDING control points are used" convention vs BSplineKit's
            # basis-index origin on these knots) — verified to 1e-14.
            cxy = [Vec3{Float64}(c[gx], 0, 0) for _ in 1:4, gx in 1:n]  # 4 rows (y), n cols (x)
            for g in range(1.5, n - 2.5; length = 40)
                ix, iy, wx, wy = _continued_window(g, 1.0, n, 4)
                ours = _eval_window(cxy, ix, iy, wx, wy)[1]
                @test isapprox(ours, S(g - 1); atol = 1.0e-12)
            end
            # partition of unity + endpoint continuity of the ported weights
            for u in (0.0, 0.25, 0.7, 0.999999)
                @test isapprox(sum(cubic_bspline_weights(u)), 1.0; atol = 1.0e-12)
            end
        end

        @testset "spline window weights — Float32 against BigFloat" begin
            # The weights are cubics in u = x − 3; a Float32 evaluation stays
            # within 1e-7 of the exact weights at the same x (the form in x
            # itself lost 1.2e-5 to cancellation).
            # Their derivatives likewise (the x form lost 3.6e-6).
            exact(u) = ((1 - u)^3 / 6, (3u^3 - 6u^2 + 4) / 6, (-3u^3 + 3u^2 + 3u + 1) / 6, u^3 / 6)
            exact_deriv(u) = (-(1 - u)^2 / 2, (3u^2 - 4u) / 2, (-3u^2 + 2u + 1) / 2, u^2 / 2)
            worst = worst_deriv = 0.0
            worst_sum = worst_deriv_sum = 0.0
            for u in range(0.0f0, prevfloat(1.0f0); length = 20_001)
                x = 3.0f0 + u
                w = cubic_bspline_weights(x - 3)          # the window coordinate's local coordinate, as the windows form it
                dw = _cubic_bspline_weights_deriv(x - 3)
                worst = max(worst, maximum(Float64.(abs.(big.(w) .- exact(big(x) - 3)))))
                worst_deriv = max(worst_deriv, maximum(Float64.(abs.(big.(dw) .- exact_deriv(big(x) - 3)))))
                worst_sum = max(worst_sum, abs(sum(Float64.(w)) - 1))
                worst_deriv_sum = max(worst_deriv_sum, abs(sum(Float64.(dw))))
            end
            @test cubic_bspline_weights(0.5f0) isa NTuple{4, Float32}
            @test _cubic_bspline_weights_deriv(0.5f0) isa NTuple{4, Float32}
            @test cubic_bspline_weights(0.375) == (0.625^3 / 6, (3 * 0.375^3 - 6 * 0.375^2 + 4) / 6,
                (-3 * 0.375^3 + 3 * 0.375^2 + 3 * 0.375 + 1) / 6, 0.375^3 / 6)      # dyadic: every form is exact
            @test worst < 1.0e-7
            @test worst_deriv < 1.0e-7
            @test worst_sum < 4 * eps(Float32)
            @test worst_deriv_sum < 4 * eps(Float32)
        end

        # A smooth parametric source: division lens on a small frame.
        W, H = 640, 480
        K = @SMatrix [300.0 0.0 320.0; 0.0 300.0 240.0; 0.0 0.0 1.0]
        src = CameraModel(CameraCalibrationMat(K); lens = Division(-0.1))

        @testset "round trip + parity with the source" begin
            g = CameraModel(SplineLens, src, (W, H); grid_size = (20, 16))
            @test g isa AbstractCamera && !is_posed(g)
            rng = MersenneTwister(3)
            worst_rt = 0.0
            worst_angle = 0.0
            for _ in 1:25
                px = SVector(20 + (W - 40) * rand(rng), 20 + (H - 40) * rand(rng))
                d = backproject(g, px)
                @test all(isfinite, d) && isapprox(norm(d), 1; atol = 1.0e-12)
                # parity with the parametric source it was initialized from
                ds = normalize(SVector{3, Float64}(backproject(src, px)))
                worst_angle = max(worst_angle, translation_angle(d, ds))
                # project the direction back (any positive depth)
                px2 = project(g, 3.7 * d)
                @test all(isfinite, px2)
                worst_rt = max(worst_rt, norm(px2 - px))
            end
            @test worst_rt < 1.0e-3            # LM converges to the same pixel
            @test worst_angle < 2.0e-4         # prefilter-init floor (measured 8.9e-5)
            # the node samples taken as controls (Schoenberg's approximation) is no init: the
            # prefilter interpolates them, quasi-interpolation is the local alternative
            @test_throws "init must be :interpolate" CameraModel(SplineLens, src, (W, H); grid_size = (20, 16), init = :sample)
            # calibrated-area contract
            @test all(isnan, backproject(g, SVector(-1.0, 10.0)))
            @test all(isfinite, backproject(g, SVector(Float64(W), 10.0)))   # the far edge: the area is closed
            @test all(isnan, backproject(g, SVector(W + 1.0, 10.0)))
            @test backproject(g, SVector(-1.0, 10.0)) isa StaticVector{3, Float64}   # one return type: kernel-lawful
            @inferred backproject(g, SVector(W + 1.0, 10.0))
            # z-depth unproject (VGC convention)
            px = SVector(200.0, 150.0)
            X = unproject(g, px, 5.0)
            @test all(isfinite, X) && isapprox(X[3], 5.0; atol = 1.0e-12)
            @test all(isfinite, project(g, X))
            # a seed is honored: the batched form's per-ray warm start
            @test norm(project(g, [3.7 * backproject(g, px)], [px]; iters = 100).px[1] - px) < 1.0e-3
        end

        @testset "undistortion through the spline camera" begin
            # smooth synthetic image; compare undistort through the SOURCE
            # parametric camera vs through the spline camera fitted to it, in
            # the same target frame (the source's K, which the camera keeps)
            Ws, Hs = 160, 120
            Ks = @SMatrix [80.0 0.0 80.0; 0.0 80.0 60.0; 0.0 0.0 1.0]
            msrc = CameraModel(CameraCalibrationMat(Ks); lens = Division(-0.12))
            gsrc = CameraModel(SplineLens, msrc, (Ws, Hs); grid_size = (10, 8))
            @test calibration_matrix(gsrc) == Ks
            imgt = [
                exp(-((x - 70)^2 + (y - 55)^2) / 400) + 0.2 * sin(x / 9) * cos(y / 7)
                    for y in 1:Hs, x in 1:Ws
            ]
            rect = VgtkCore.Rect(30.0, 20.0, 100.0, 80.0)
            up = undistort(msrc, imgt, rect)                 # parametric (fused warp)
            ug = undistort(gsrc, imgt, rect)                 # the spline lens (the warp sampled on a coarse grid)
            @test ug isa FramedImage && ug.rect === rect
            @test size(ug.data) == size(up.data)
            # interiors agree to the fit's approximation floor (borders differ by
            # fill handling; compare an interior crop)
            di = abs.(ug.data[10:70, 10:90] .- up.data[10:70, 10:90])
            @test maximum(di) < 0.02
        end

        @testset "house-type acceptance (Size2 / Rect)" begin
            gt = CameraModel(SplineLens, src, (W, H); grid_size = (8, 6))
            gs = CameraModel(SplineLens, src, VgtkCore.Size2(W, H); grid_size = (8, 6))
            @test gs.lens.backward.controls == gt.lens.backward.controls && valid_domain(gs) == valid_domain(gt)
            gr = CameraModel(
                SplineLens, src, VgtkCore.Size2(W, H); grid_size = (8, 6),
                calib_area = VgtkCore.Rect(0, 0, W, H)
            )
            @test valid_domain(gr) == Rect2{Float64}(0.0, 0.0, Float64(W), Float64(H))
            @test gr.lens.backward.controls == gt.lens.backward.controls
        end

        @testset "_precision" begin
            g = CameraModel(SplineLens, src, (W, H); grid_size = (8, 6))
            l32 = VgtkCore.Cameras._precision(Float32, g.lens)
            @test eltype(l32.backward.controls) == SVector{3, Float32}
            @test all(isfinite, backproject(CameraModel(g.intrinsics, l32), SVector(100.0, 100.0)))
        end
    end

    @testset "batched device projection (KA kernel, CPU backend)" begin
        rng = MersenneTwister(41)   # seeded: an unseeded borderline draw tripped the parity bound once
        W, H = 160, 120
        Ks = @SMatrix [80.0 0.0 80.0; 0.0 80.0 60.0; 0.0 0.0 1.0]
        msrc = CameraModel(CameraCalibrationMat(Ks); lens = Division(-0.12))
        g = CameraModel(SplineLens, msrc, (W, H); grid_size = (12, 9))
        pts = [SVector(10.0 + 140.0 * rand(rng), 8.0 + 104.0 * rand(rng)) for _ in 1:200]
        rays = [backproject(g, p) for p in pts]
        keep = findall(r -> all(isfinite, r), rays)
        pts = pts[keep]; rays = SVector{3, Float64}.(rays[keep])
        seeds = [p + SVector(2.0 * randn(rng), 2.0 * randn(rng)) for p in pts]   # warm, ±2 px
        out = project(g, rays, seeds)
        @test all(out.ok)
        # parity vs the host per-ray LM (cold, from the area's centre)
        hostpx = [project(g, rays[i]) for i in eachindex(rays)]
        dmax = maximum(norm(out.px[i] - hostpx[i]) for i in eachindex(rays))
        @test dmax < 5.0e-4
        # …and both land on the true pixel
        @test maximum(norm(out.px[i] - pts[i]) for i in eachindex(pts)) < 1.0e-3
    end

    # =============================================================================
    # The window a point reads: its controls and weights, the tangent basis
    # =============================================================================
    @testset "the window a point reads" begin
        using VisualGeometryToolkitCore: direction_tangents
        W, H = 1200, 800
        src = CameraModel(CameraCalibrationMat(500.0px, [600.0px, 400.0px]); lens = Division(-0.25, 0.02))
        g = CameraModel(SplineLens, src, (W, H); grid_size = (20, 16))
        b = g.lens.backward
        for px in (SVector(613.7, 388.2), SVector(101.3, 702.9), SVector(1150.0, 40.0))
            x = _normalized_point(g.intrinsics, px)
            ix, iy, wx, wy = spline_window(b.area, size(b.controls), x)
            w = SMatrix{4, 4}([wy[r] * wx[c] for r in 1:4, c in 1:4])
            C = SMatrix{4, 4}([b.controls[iy - 3 + r, ix - 3 + c] for r in 1:4, c in 1:4])
            @test sum(w) ≈ 1 atol = 1.0e-12                                   # partition of unity
            d = sum(w .* C)
            @test normalize(SVector{3}(d)) ≈ backproject(g, px) atol = 1.0e-12   # the same direction backproject reads
            # spline_weights: the same window and weights, and their partials in x
            jx, jy, w2, (w1x, w1y) = spline_weights(b, x)
            @test (jx, jy) == (ix, iy) && w2 == w
            @test sum(w2 .* C) == d
            Jw = FiniteDiff.finite_difference_jacobian(p -> vec(spline_weights(b, SVector{2}(p))[3]), Vector(x), Val(:central))
            @test maximum(abs.(vec(w1x) .- Jw[:, 1])) < 1.0e-7 * maximum(abs, Jw)
            @test maximum(abs.(vec(w1y) .- Jw[:, 2])) < 1.0e-7 * maximum(abs, Jw)
            @test abs(sum(w1x)) < 1.0e-9 && abs(sum(w1y)) < 1.0e-9
        end
        # the control grid and its size, in the grid_size keyword's order
        @test directions(g) === b.controls && grid_size(g) == (20, 16)
        @test all(isnan, backproject(g, SVector(-5.0, 10.0)))
        t1, t2 = direction_tangents(SVector(0.0, 0.0, 1.0))
        @test norm(t1) ≈ 1 && norm(t2) ≈ 1 && abs(dot(t1, t2)) < 1.0e-12 && abs(dot(t1, SVector(0.0, 0.0, 1.0))) < 1.0e-12
        # the basis keeps the direction's element type: Float32 stays Float32 (a device kernel is not promoted), and a dual
        # direction gives dual tangents whose derivative is the derivative of the basis (central differences, off the branch switch)
        d32 = normalize(SVector(0.3f0, -0.2f0, 0.9f0))
        f1, f2 = direction_tangents(d32)
        @test eltype(f1) === Float32 && eltype(f2) === Float32
        @test abs(dot(f1, f2)) < 1.0f-6 && abs(dot(f1, d32)) < 1.0f-6 && norm(f1) ≈ 1 && norm(f2) ≈ 1
        for i in 1:3                                         # each branch at Float32, the same basis as at Float64
            d = normalize(SVector(i == 1 ? 0.95 : 0.2, 0.1, 0.3))
            g1, g2 = direction_tangents(Float32.(d))
            h1, h2 = direction_tangents(d)
            @test maximum(abs, g1 - h1) < 1.0f-6 && maximum(abs, g2 - h2) < 1.0f-6
        end
        dir(s) = normalize(SVector(0.3 + s, -0.2 + 2s, 0.9 - s))
        J = ForwardDiff.derivative(s -> vcat(direction_tangents(dir(s))...), 0.0)
        Jc = (vcat(direction_tangents(dir(1.0e-6))...) - vcat(direction_tangents(dir(-1.0e-6))...)) / 2.0e-6
        @test eltype(direction_tangents(dir(ForwardDiff.Dual(0.0, 1.0)))[1]) <: ForwardDiff.Dual
        @test maximum(abs, J - Jc) < 1.0e-8
    end

    @testset "projection through a degenerate grid fails, never throws" begin
        W, H = 640, 480
        src = CameraModel(CameraCalibrationMat(300.0px, [320.0px, 240.0px]); lens = Division(-0.2))
        g = CameraModel(SplineLens, src, (W, H); grid_size = (12, 10))
        bad = copy(g.lens.backward.controls)
        bad[5, 6] = -bad[5, 6]                       # an antipodal control: blends near it cross zero
        gb = CameraModel(g.intrinsics, SplineLens(ChartSpline(g.lens.backward.area, bad)))
        X = SVector(0.05, -0.02, 1.0)
        p = project(gb, [X], [SVector(330.0, 235.0)]).px[1]
        @test all(isnan, p) || all(isfinite, p)
        p = project(gb, X)
        @test all(isnan, p) || all(isfinite, p)
        b = project(gb, [X], [SVector(NaN, 1.0)])
        @test !b.ok[1] && all(isnan, b.px[1])
    end

    @testset "default_spline_grid" begin
        using VisualGeometryToolkitCore: default_spline_grid
        @test default_spline_grid(4240, 2832) == (70, 48)
        @test default_spline_grid(2570, 2569) == (44, 44)
        @test default_spline_grid(10, 10) == (4, 4)
    end

    @testset "spline_lattice: the constructor's layout, unsampled" begin
        src = CameraModel(CameraCalibrationMat(300.0px, [320.0px, 240.0px]); lens = Division(-0.2))
        built = CameraModel(SplineLens, src, (640, 480); grid_size = (12, 10), calib_area = (0, 0, 599, 399))
        lat = spline_lattice(SplineLens, src, (640, 480); grid_size = (12, 10), calib_area = (0, 0, 599, 399))
        @test lat isa CameraModel{<:Any, <:SplineLens}
        @test calibration_matrix(lat) == calibration_matrix(built) && lat.lens.backward.area == built.lens.backward.area
        @test size(lat.lens.backward.controls) == size(built.lens.backward.controls) == (10, 12)
        @test valid_domain(lat) == valid_domain(built)
        a, sz = built.lens.backward.area, size(built.lens.backward.controls)
        @test all(p -> spline_window(lat.lens.backward.area, sz, p) == spline_window(a, sz, p), (SVector(-0.9, -0.7), SVector(0.9, 0.5), SVector(2.0, -3.0)))
        @test all(==(SVector(0.0, 0.0, 1.0)), spline_lattice(SplineLens, src, (640, 480)).lens.backward.controls)
    end
end

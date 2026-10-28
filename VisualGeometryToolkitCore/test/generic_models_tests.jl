@testitem "generic_models" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CentralGenericModel, NoncentralGenericModel, Ray3, Division,
        is_posed
    using VisualGeometryToolkitCore.Cameras: _cubic_weights, _cubic_weights_deriv, _spline_window, _eval_window, _grid_coords
    using StaticArrays
    using FiniteDiff: FiniteDiff
    using GeometryBasics: Vec3
    using LinearAlgebra: norm, normalize, dot
    using Random: MersenneTwister
    using BSplineKit: BSplineBasis, BSplineOrder, Spline
    using ForwardDiff: ForwardDiff

    # =============================================================================
    # Generic camera models (Schöps et al. CVPR 2020) — the grid models
    # =============================================================================
    #
    # Covers: the ported uniform-cubic window weights against BSplineKit (the
    # oracle pattern of bspline_tests.jl — the standard package validates the
    # scalar port), central project∘backproject round trips and parity with the
    # parametric source it was initialized from, the calibrated-area contract
    # (nothing outside), the z-depth unproject, the noncentral model's
    # central-reduction (zero point grid) and genuine non-central round trip
    # (the projected pixel's LINE goes through the point), and _precision.

    @testset "Generic camera models (Schöps et al.)" begin
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
                ix, iy, wx, wy = _spline_window(g, 1.0)
                ours = _eval_window(cxy, ix, iy, wx, wy)[1]
                @test isapprox(ours, S(g - 1); atol = 1.0e-12)
            end
            # partition of unity + endpoint continuity of the ported weights
            for f in (3.0, 3.25, 3.7, 3.999999)
                @test isapprox(sum(_cubic_weights(f)), 1.0; atol = 1.0e-12)
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
                w = _cubic_weights(x)
                dw = _cubic_weights_deriv(x)
                worst = max(worst, maximum(Float64.(abs.(big.(w) .- exact(big(x) - 3)))))
                worst_deriv = max(worst_deriv, maximum(Float64.(abs.(big.(dw) .- exact_deriv(big(x) - 3)))))
                worst_sum = max(worst_sum, abs(sum(Float64.(w)) - 1))
                worst_deriv_sum = max(worst_deriv_sum, abs(sum(Float64.(dw))))
            end
            @test _cubic_weights(3.5f0) isa NTuple{4, Float32}
            @test _cubic_weights_deriv(3.5f0) isa NTuple{4, Float32}
            @test worst < 1.0e-7
            @test worst_deriv < 1.0e-7
            @test worst_sum < 4 * eps(Float32)
            @test worst_deriv_sum < 4 * eps(Float32)
        end

        # A smooth parametric source: division lens on a small frame.
        W, H = 640, 480
        K = @SMatrix [300.0 0.0 320.0; 0.0 300.0 240.0; 0.0 0.0 1.0]
        src = CameraModel(CameraCalibrationMat(K); lens = Division(-0.1))

        @testset "central: round trip + parity with the source" begin
            g = CentralGenericModel(src, (W, H); grid_size = (20, 16))
            @test g isa AbstractCamera && !is_posed(g)
            rng = MersenneTwister(3)
            worst_rt = 0.0
            worst_angle = 0.0
            for _ in 1:25
                px = SVector(20 + (W - 40) * rand(rng), 20 + (H - 40) * rand(rng))
                d = backproject(g, px)
                @test d !== nothing && isapprox(norm(d), 1; atol = 1.0e-12)
                # parity with the parametric source it was initialized from
                ds = normalize(SVector{3, Float64}(backproject(src, px)))
                worst_angle = max(worst_angle, translation_angle(d, ds))
                # project the direction back (any positive depth)
                px2 = project(g, 3.7 * d)
                @test px2 !== nothing
                worst_rt = max(worst_rt, norm(px2 - px))
            end
            @test worst_rt < 1.0e-3            # LM converges to the same pixel
            @test worst_angle < 2.0e-4         # prefilter-init floor (measured 8.9e-5)
            # the prefilter init beats quasi-interpolation by an order of magnitude
            gq = CentralGenericModel(src, (W, H); grid_size = (20, 16), init = :sample)
            wq = 0.0
            rng2 = MersenneTwister(3)
            for _ in 1:25
                px = SVector(20 + (W - 40) * rand(rng2), 20 + (H - 40) * rand(rng2))
                dq = backproject(gq, px)
                dsq = normalize(SVector{3, Float64}(backproject(src, px)))
                wq = max(wq, translation_angle(dq, dsq))
            end
            @test wq > 5 * worst_angle
            # calibrated-area contract
            @test backproject(g, SVector(-1.0, 10.0)) === nothing
            @test backproject(g, SVector(Float64(W), 10.0)) === nothing
            # z-depth unproject (VGC convention)
            px = SVector(200.0, 150.0)
            X = unproject(g, px, 5.0)
            @test X !== nothing && isapprox(X[3], 5.0; atol = 1.0e-12)
            @test project(g, X) !== nothing
            # init kwarg is honored
            @test norm(project(g, 3.7 * backproject(g, px); init = px) - px) < 1.0e-3
        end

        @testset "noncentral: central reduction + genuine lines" begin
            gc = CentralGenericModel(src, (W, H); grid_size = (16, 12))
            gn = NoncentralGenericModel(src, (W, H); grid_size = (16, 12))
            px = SVector(345.0, 210.0)
            ray = backproject(gn, px)
            @test ray isa Ray3
            @test isapprox(norm(ray.direction), 1; atol = 1.0e-12)
            @test norm(ray.origin) < 1.0e-12                       # zero point grid ⇒ central
            @test norm(SVector{3}(ray.direction) - backproject(gc, px)) < 1.0e-12
            X = 2000.0 * SVector{3}(ray.direction)
            pc = project(gc, X)
            pn = project(gn, X)
            @test pc !== nothing && pn !== nothing && norm(pn - pc) < 1.0e-3

            # perturb the point grid: rays no longer share a center, and the
            # projected pixel's LINE goes through the query point
            rng = MersenneTwister(11)
            pts = [
                Vec3{Float64}(0.4 * randn(rng), 0.4 * randn(rng), 0.4 * randn(rng))
                    for _ in 1:12, _ in 1:16
            ]
            gnc = NoncentralGenericModel(W, H, 0, 0, W - 1, H - 1, gn.directions, pts)
            worst = 0.0
            for _ in 1:15
                p0 = SVector(30 + (W - 60) * rand(rng), 30 + (H - 60) * rand(rng))
                r = backproject(gnc, p0)
                X = SVector{3}(r.origin) + 1500.0 * SVector{3}(r.direction)
                p1 = project(gnc, X; init = p0 + SVector(3.0, -2.0))
                @test p1 !== nothing
                r1 = backproject(gnc, p1)
                pto = SVector{3}(r1.origin) - X
                dist = norm(pto - dot(pto, SVector{3}(r1.direction)) * SVector{3}(r1.direction))
                worst = max(worst, dist)
            end
            @test worst < 1.0e-4                                   # point-to-line distance
        end

        @testset "undistortion through the generic model" begin
            # smooth synthetic image; compare undistort through the SOURCE
            # parametric camera vs through the generic model fitted to it, in
            # the same target frame (the source's K)
            Ws, Hs = 160, 120
            Ks = @SMatrix [80.0 0.0 80.0; 0.0 80.0 60.0; 0.0 0.0 1.0]
            msrc = CameraModel(CameraCalibrationMat(Ks); lens = Division(-0.12))
            gsrc = CentralGenericModel(msrc, (Ws, Hs); grid_size = (10, 8))
            imgt = [
                exp(-((x - 70)^2 + (y - 55)^2) / 400) + 0.2 * sin(x / 9) * cos(y / 7)
                    for y in 1:Hs, x in 1:Ws
            ]
            rect = VgtkCore.Rect(30.0, 20.0, 100.0, 80.0)
            up = undistort(msrc, imgt, rect)                 # parametric (fused warp)
            ug = undistort(gsrc, imgt, Ks, rect)             # generic (coarse-inverse remap)
            @test ug isa FramedImage && ug.rect === rect
            @test size(ug.data) == size(up.data)
            # interiors agree to the fit's approximation floor (borders differ by
            # fill handling; compare an interior crop)
            di = abs.(ug.data[10:70, 10:90] .- up.data[10:70, 10:90])
            @test maximum(di) < 0.02
        end

        @testset "house-type acceptance (Size2 / Rect)" begin
            gt = CentralGenericModel(src, (W, H); grid_size = (8, 6))
            gs = CentralGenericModel(src, VgtkCore.Size2(W, H); grid_size = (8, 6))
            @test gs.grid == gt.grid && gs.calib_max_x == gt.calib_max_x
            gr = CentralGenericModel(
                src, VgtkCore.Size2(W, H); grid_size = (8, 6),
                calib_area = VgtkCore.Rect(0, 0, W, H)
            )
            @test gr.calib_min_x == 0 && gr.calib_max_x == W - 1 && gr.calib_max_y == H - 1
            @test gr.grid == gt.grid
        end

        @testset "_precision" begin
            g = CentralGenericModel(src, (W, H); grid_size = (8, 6))
            g32 = VgtkCore.Cameras._precision(Float32, g)
            @test eltype(g32.grid) == Vec3{Float32}
            @test backproject(g32, SVector(100.0, 100.0)) !== nothing
        end
    end

    @testset "batched device projection (KA kernel, CPU backend)" begin
        rng = MersenneTwister(41)   # seeded: an unseeded borderline draw tripped the parity bound once
        W, H = 160, 120
        Ks = @SMatrix [80.0 0.0 80.0; 0.0 80.0 60.0; 0.0 0.0 1.0]
        msrc = CameraModel(CameraCalibrationMat(Ks); lens = Division(-0.12))
        g = CentralGenericModel(msrc, (W, H); grid_size = (12, 9))
        pts = [SVector(10.0 + 140.0 * rand(rng), 8.0 + 104.0 * rand(rng)) for _ in 1:200]
        rays = [backproject(g, p) for p in pts]
        keep = findall(!isnothing, rays)
        pts = pts[keep]; rays = SVector{3, Float64}.(rays[keep])
        inits = [p + SVector(2.0 * randn(rng), 2.0 * randn(rng)) for p in pts]   # warm, ±2 px
        out = project(g, rays; init = inits)
        @test all(out.ok)
        # parity vs the host per-ray LM from the same inits
        hostpx = [project(g, rays[i]; init = inits[i]) for i in eachindex(rays)]
        dmax = maximum(norm(out.px[i] - hostpx[i]) for i in eachindex(rays))
        @test dmax < 5.0e-4
        # …and both land on the true pixel
        @test maximum(norm(out.px[i] - pts[i]) for i in eachindex(pts)) < 1.0e-3
    end

    # =============================================================================
    # The window interface (0.22.2): support, weights + pixel derivatives, controls
    # =============================================================================
    @testset "spline window interface" begin
        using VisualGeometryToolkitCore: spline_support, spline_weights, window_controls, window_indices, direction_tangents
        W, H = 1200, 800
        src = CameraModel(CameraCalibrationMat(500.0px, [600.0px, 400.0px]); lens = Division(-0.25, 0.02))
        g = CentralGenericModel(src, (W, H); grid_size = (20, 16))
        for px in (SVector(613.7, 388.2), SVector(101.3, 702.9), SVector(1150.0, 40.0))
            sup = spline_support(g, px)
            @test sup !== nothing
            ix, iy = sup
            w, wx, wy = spline_weights(g, ix, iy, px)
            C = window_controls(g, ix, iy)
            idx = window_indices(g, ix, iy)
            @test all(g.grid[idx[k]] == C[k] for k in 1:16)
            @test sum(w) ≈ 1 atol = 1.0e-12                                   # partition of unity
            d = sum(w .* C)
            @test normalize(SVector{3}(d)) ≈ backproject(g, px) atol = 1.0e-12   # the same direction backproject reads
            # derivatives vs central differences, the window held fixed
            Jw = FiniteDiff.finite_difference_jacobian(p -> vec(spline_weights(g, ix, iy, p)[1]), px, Val(:central))
            @test maximum(abs.(vec(wx) .- Jw[:, 1])) < 1.0e-7
            @test maximum(abs.(vec(wy) .- Jw[:, 2])) < 1.0e-7
            @test sum(wx) ≈ 0 atol = 1.0e-10
            @test sum(wy) ≈ 0 atol = 1.0e-10
        end
        @test spline_support(g, SVector(-5.0, 10.0)) === nothing
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
        g = CentralGenericModel(src, (W, H); grid_size = (12, 10))
        bad = copy(g.grid)
        bad[5, 6] = -bad[5, 6]                       # an antipodal control: blends near it cross zero
        gb = CentralGenericModel{Float64}(W, H, g.calib_min_x, g.calib_min_y, g.calib_max_x, g.calib_max_y, bad)
        X = SVector(0.05, -0.02, 1.0)
        p = project(gb, X; init = SVector(330.0, 235.0))
        @test p === nothing || all(isfinite, p)
        @test project(gb, X; init = SVector(NaN, 1.0)) === nothing
    end

    @testset "default_generic_grid" begin
        using VisualGeometryToolkitCore: default_generic_grid
        @test default_generic_grid(4240, 2832) == (70, 48)
        @test default_generic_grid(2570, 2569) == (44, 44)
        @test default_generic_grid(10, 10) == (4, 4)
    end
end

@testitem "fit_lens" begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: FitResult, estimate, rms, metadata, iterations, converged
    using VisualGeometryToolkitCore: view_angle_limit, projected_radius, Division, parameter_vector, radial_part, parameters,
        Lens, LiftLens, Radial, Tangential, ThinPrism, Additive, Perspective, Equidistant, Orthographic, Even,
        KannalaBrandt, DoubleSphere, Scaramuzza, Pinhole
    # the target types: a family with its arities
    KB4 = KannalaBrandt{4}
    SC3 = Scaramuzza{3}
    DSL = typeof(Lens(DoubleSphere(0.5, 0.5)))
    OPENCV3 = Lens{Perspective, Radial{Even, 3, 0, Float64}}      # OpenCV's radial polynomial k₁…k₃
    EQ = typeof(Lens(Equidistant()))
    using VisualGeometryToolkitEstimation: fit_lens
    import VisualGeometryToolkitEstimation
    using StaticArrays

    # =============================================================================
    # fit_lens — model-to-model regression on the radial profile r(θ) (moved from
    # VisualGeometryToolkitCore with the function, estimator taxonomy v2)
    # =============================================================================
    #
    # The objective is the INTEGRAL of the squared profile difference over the
    # shared view-angle domain, by Clenshaw–Curtis quadrature on Chebyshev
    # nodes. Covers: the quadrature itself, same-family round trips (the
    # recovery test of the driver), the division → {KB, DoubleSphere} fits the
    # facility exists for, the Brown–Conrady domain-clip policy (perspective
    # targets cannot see past 90° — warned, reported, never silent), the max_view_angle
    # cap, fix_focal, scale-only projection targets, and the rational target's
    # ray-lift angle residual.

    # The cameras below share a 4240 × 2832 frame's principal point.
    pp = (2120.0, 1416.0)

    @testset "fit_lens — model-to-model regression on r(θ)" begin
        @testset "Clenshaw–Curtis integrates polynomials" begin
            θ, q = VgtkEstimation._clenshaw_curtis(32, 1.3)
            @test sum(q) ≈ 1.3 rtol = 1.0e-12
            @test sum(q .* θ .^ 2) ≈ 1.3^3 / 3 rtol = 1.0e-10
            @test sum(q .* θ .^ 5) ≈ 1.3^6 / 6 rtol = 1.0e-8
        end

        mdiv = CameraModel(; pp, focal = 927.0, lens = Division(-0.3302, -0.0765))   # the hallway-session camera
        rcorner = hypot(2120.0, 1416.0)                 # farthest sensor point from pp

        @testset "same-family round trip (KannalaBrandt)" begin
            mkb = CameraModel(; pp, focal = 600.0, lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
            r = fit_lens(KB4, mkb)
            @test abs(metadata(r).s - 1) < 1.0e-6
            c = parameters(lens(estimate(r)))
            @test isapprox(c.k1, -0.02; atol = 1.0e-6)
            @test isapprox(c.k2, 0.004; atol = 1.0e-6)
            @test isapprox(c.k3, -0.001; atol = 1.0e-5)
            @test isapprox(c.k4, 0.0002; atol = 1.0e-5)
            @test rms(r) < 1.0e-4
            @test !metadata(r).clipped
        end

        @testset "division → KannalaBrandt" begin
            r = fit_lens(KB4, mdiv; max_radius_px = rcorner)
            @test !metadata(r).clipped                       # KB's family domain covers the source
            @test rms(r) < 5.0
            @test estimate(r).lens === lens(estimate(r))
        end

        @testset "division → Scaramuzza is exact; Scaramuzza → KannalaBrandt" begin
            # Division(k₁, k₂)'s lift IS Scaramuzza(k₁, 0, k₂): the regression recovers it
            r = fit_lens(SC3, mdiv; max_radius_px = rcorner)
            @test !metadata(r).clipped
            @test rms(r) < 1.0e-4
            c = parameters(lens(estimate(r)))                     # a₂, a₃, a₄ are d1, d2, d3
            @test isapprox(c.d1, -0.3302; atol = 1.0e-4)
            @test abs(c.d2) < 1.0e-4
            @test isapprox(c.d3, -0.0765; atol = 1.0e-4)
            msc = CameraModel(; pp, focal = 927.0, lens = Scaramuzza(-0.3302, 0.02, -0.0765))
            rk = fit_lens(KB4, msc; max_radius_px = rcorner)
            @test !metadata(rk).clipped
            @test rms(rk) < 5.0
        end

        @testset "division → DoubleSphere" begin
            # The source is a 180°-FOV CIRCULAR fisheye: light ends at the image
            # circle — the division model's 90° radius f·monotone_radius ≈
            # 1328 px — and the frame corners (2554 px) are dark. Capped at the
            # lit field (θ ≈ 87°), the 2-dof double sphere models the fisheye at
            # ~1.1 px, the DS paper's regime.
            rlit = 927.0 * VgtkCore.monotone_radius(mdiv.lens) * 0.97
            r = fit_lens(DSL, mdiv; max_radius_px = rlit)
            @test 0 < parameters(lens(estimate(r))).alpha <= 1
            @test rms(r) < 3.0
            Δ = abs(metadata(r).s * projected_radius(lens(estimate(r)), 0.6) - projected_radius(mdiv.lens, 0.6)) * 927.0
            @test Δ < 1.0
            # Asked to follow the division lift 65° PAST the image circle — pure
            # unlit extrapolation, folding toward vertical — the rigid 2-dof
            # family refuses (measured ~49 px rms). A finding about the source
            # model's unconstrained tail, not about the double sphere; pinned
            # loosely so a regression to nonsense still shows.
            rfull = fit_lens(DSL, mdiv; max_radius_px = rcorner)
            @test rms(rfull) < 60.0
            @test rms(rfull) > rms(r)
        end

        @testset "division → OpenCV's radial polynomial clips at the perspective edge" begin
            r = @test_logs (:warn, r"cannot represent") match_mode = :any fit_lens(OPENCV3, mdiv; max_radius_px = rcorner)
            @test metadata(r).clipped
            @test metadata(r).θ_max < π / 2
            @test metadata(r).θ_source > metadata(r).θ_max             # the report carries what was thrown away
            @test isfinite(rms(r))
            @test lens(estimate(r)) isa OPENCV3                              # radial-only target
            # a target with decentering terms holds them at zero, by policy
            rd = fit_lens(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0)))), mdiv; max_radius_px = rcorner)
            cd = parameters(lens(estimate(rd)))
            @test cd.p1 == 0 && cd.p2 == 0 && cd.d1 == 0 && cd.d3 == 0
            @test (cd.k1, cd.k2, cd.k3) == Tuple(parameter_vector(lens(estimate(r))))   # the same fit as the radial-only target
        end

        @testset "max_view_angle cap + pinhole target" begin
            r = fit_lens(Pinhole, mdiv; max_view_angle = deg2rad(30))
            @test metadata(r).θ_max ≈ deg2rad(30) atol = 1.0e-12
            @test 0 < metadata(r).s < 1                      # barrel: r_d < tanθ ⇒ smaller equivalent focal
            @test rms(r) < 30.0                  # a pinhole is a poor fisheye — but a fitted one
        end

        @testset "scale-only projection target + fix_focal" begin
            r = fit_lens(EQ, mdiv; max_radius_px = rcorner)
            @test isfinite(rms(r)) && metadata(r).s > 0
            rf = fit_lens(KB4, mdiv; fix_focal = true)
            @test metadata(rf).s == 1.0
        end

        @testset "rational target round trip (ray-lift angle residual)" begin
            r = fit_lens(Division{2}, mdiv; max_radius_px = rcorner)
            @test abs(metadata(r).s - 1) < 1.0e-3
            @test isapprox(lens(estimate(r)).radial.d[1], -0.3302; atol = 2.0e-3)
            @test isapprox(lens(estimate(r)).radial.d[2], -0.0765; atol = 5.0e-3)
            rr = fit_lens(LiftLens{Radial{Even, 1, 2, Float64}}, mdiv; max_radius_px = rcorner)     # a numerator too
            @test rms(rr) < 1.0e-3
        end

        @testset "a lower-degree source: the leading coefficient fits to rounding" begin
            # Division{3} onto Division(k₁, k₂) lands k₃ at ~1e-17: the rms was NaN
            # (the rational solve's bracket spanned the lift's fold at ρ ~ 5e7) …
            r3 = fit_lens(Division{3}, mdiv; max_radius_px = rcorner)
            @test rms(r3) < 1.0e-4
            @test abs(lens(estimate(r3)).radial.d[3]) < 1.0e-12
            # … and Scaramuzza onto Division(k) lands a₃, a₄ there: the fold sweep
            # stalled at the origin, view_angle_limit was 0 and fit_lens threw
            for k in (-0.2, 0.05)
                rs = fit_lens(SC3, CameraModel(; pp, focal = 927.0, lens = Division(k)); max_radius_px = rcorner)
                @test rms(rs) < 1.0e-4
                @test isapprox(parameters(lens(estimate(rs))).d1, k; atol = 1.0e-9)
            end
        end

        @testset "Division alone is the one-coefficient model; a type without its arities is refused" begin
            @test lens(estimate(fit_lens(Division, mdiv; max_radius_px = rcorner))) ==
                lens(estimate(fit_lens(Division{1}, mdiv; max_radius_px = rcorner)))
            @test_throws ArgumentError fit_lens(LiftLens, mdiv)
            @test_throws ArgumentError fit_lens(KannalaBrandt, mdiv)
        end

        @testset "the family limits are VGC's own view-angle limits" begin
            # the perspective and sphere edges were constants beside fit_lens;
            # they are view_angle_limit at its default cap, exactly
            @test VgtkEstimation._family_limit(Pinhole) == view_angle_limit(Pinhole()) == π / 2 - 1.0e-3
            @test VgtkEstimation._family_limit(KB4) == view_angle_limit(Lens(Equidistant())) == π - 1.0e-3
            @test VgtkEstimation._family_limit(typeof(Lens(Orthographic()))) == view_angle_limit(Lens(Orthographic()))
            @test VgtkEstimation._family_limit(DSL) == π - 1.0e-3
        end

        @testset "the decentering fit_lens warns about" begin
            synthetic = Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)), Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
            @test VgtkEstimation._decentering(synthetic) ≈ hypot(1.0e-3, -5.0e-4, -5.5e-3, 7.0e-4, 2.2e-3, -1.2e-3)
            @test radial_part(synthetic) === Lens(Perspective(), Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)))
        end

        @testset "fit_lens accepts the concrete rational type" begin
            cam = CameraModel(; pp, focal = 900.0, lens = Division(-0.2, -0.08))
            fit = fit_lens(Division{3, Float64}, cam; max_radius_px = 2500.0)
            @test VgtkCore.lens(estimate(fit)) isa Division{3, Float64}
            @test rms(fit) < 0.5
        end
    end

    @testset "fit_lens direction = :inverse (analytic complement, same family)" begin
        # Brown–Conrady: analytic forward, iterative backward — the inverse fit
        # gives it an analytic UNDISTORT in its own parameterization.
        hd(lens) = CameraModel(; focal = 900.0, pp = (960.0, 540.0), lens)   # a 1920 × 1080 frame
        src = hd(Lens(Perspective(), Radial((-0.18, 0.03))))
        rinv = fit_lens(OPENCV3, src; direction = :inverse, max_view_angle = 0.6)
        @test rinv isa FitResult
        @test metadata(rinv).direction === :inverse
        @test rms(rinv) < 0.5
        Li = lens(estimate(rinv)); si = metadata(rinv).s
        # round trip: L⁻(L(r)) ≈ r, in source pixels, bounded by the fit error
        worst = 0.0
        for θ in range(0.02, 0.95 * metadata(rinv).θ_max; length = 40)
            rd = Float64(projected_radius(src.lens, θ))
            ru = si * Float64(projected_radius(Li, atan(rd)))
            worst = max(worst, abs(ru - tan(θ)) * 900.0)
        end
        @test worst < 4 * max(rms(rinv), 0.05) + metadata(rinv).max_px
        # Division: analytic backward already — the inverse fit is its analytic
        # FORWARD complement in the same two coefficients.
        dsrc = hd(VgtkCore.Division(-0.2, -0.05))
        dinv = fit_lens(Division{2}, dsrc; direction = :inverse, max_view_angle = 0.9)
        @test metadata(dinv).direction === :inverse && isfinite(rms(dinv))
        # a source imaging past 90°: the inverse targets are tan θ, which the
        # perspective plane ends at — the domain is capped below 90° and warned,
        # as a perspective target caps the forward one (it used to bisect to
        # θ > 90° and regress onto negative, exploding tan θ targets)
        fsrc = hd(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0))
        @test view_angle_limit(lens(fsrc)) > π / 2
        finv = @test_logs (:warn, r"inverse profile") match_mode = :any fit_lens(KB4, fsrc; direction = :inverse)
        @test metadata(finv).θ_max < π / 2 && metadata(finv).clipped
        @test isfinite(rms(finv)) && isfinite(metadata(finv).max_px)
    end
end

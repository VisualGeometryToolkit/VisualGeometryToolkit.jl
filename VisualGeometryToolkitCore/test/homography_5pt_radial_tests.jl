# 5-pt one-sided radial homography (port of HomLib's
# get_wadenback_3dv_2026_one_sided.cpp — Wadenbäck, Valtonen Örnhag, Edstedt,
# "Radially Distorted Homographies, Revisited", 3DV 2026).
#
# Golden values from HomLib tests/test_wadenback_3dv_2026.cpp ("one-sided"
# case, extra_check=false), verified against the compiled C++ suite
# (12/12 green vs Eigen 3.4.0, 2026-08-30).

@testitem "homography_5pt_radial" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore: max_solutions
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sample_size, sign_normalize   # public, not exported
    using CoordinateTransformations: AffineMap
    using VisualGeometryToolkitCore: homography_5pt_radial, RadialHomography, Homography5ptRadial
    using StaticArrays, LinearAlgebra, Random
    using FiniteDiff: FiniteDiff

    # Golden matrices are printed row-major; SMatrix fills column-major.
    _rowmat(a, b, c, d, e, f, g, h, i) = SMatrix{3, 3, Float64}(a, d, g, b, e, h, c, f, i)

    # Predicted (undistorted) target of a source through H.
    _transfer(H, s) = (PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(H)))(s)

    # Division-model forward distortion: the d with d/(1 + k‖d‖²) = u.
    function _distort_div(u, k)
        r = norm(u)
        (abs(k) < 1.0e-14 || r < 1.0e-14) && return SVector{2, Float64}(u[1], u[2])
        disc = 1 - 4k * r^2
        disc <= 0 && return nothing
        t = (1 - sqrt(disc)) / (2k * r)
        return SVector{2, Float64}(u[1] * t / r, u[2] * t / r)
    end

    # Max interpolation residual of a candidate over correspondences 1:n.
    # The problem is "4.5-pt" overdetermined (9 DOF vs 10 constraints): every
    # cubic root interpolates the four fully-used correspondences (n = 4); the
    # fifth is enforced through ONE scalar equation, so full 5-point
    # interpolation (n = 5) holds only for the true root.
    function _max_residual(m::RadialHomography, s, d, n = 4)
        worst = 0.0
        for i in 1:n
            û = d[i] / (1 + m.k * (d[i][1]^2 + d[i][2]^2))
            q = SMatrix{3, 3, Float64}(m.H) * SVector(s[i][1], s[i][2], 1.0)
            v = SVector(û[1], û[2], 1.0)
            # Projective (gauge-free) residual: sine of the angle between H s̃
            # and the lifted undistorted target — robust when a spurious root
            # sends a point toward the plane at infinity (q₃ → 0), where the
            # affine transfer residual overflows.
            worst = max(worst, norm(cross(q, v)) / (norm(q) * norm(v)))
        end
        return worst
    end

    @testset "homography_5pt_radial (one-sided, HomLib port)" begin
        # HomLib golden sample (sources p1, distorted targets p2)
        s = (
            SVector(0.242104535827652, -0.296297354299645),
            SVector(0.319776775526576, -0.114580331777873),
            SVector(-0.122936754370192, -0.532655969828257),
            SVector(0.303141152380151, -0.295669384794041),
            SVector(0.446905125282507, -0.528127859697035),
        )
        d = (
            SVector(-0.562787932824692, -0.0769625591786919),
            SVector(-0.612813481767545, -0.0311462459576149),
            SVector(-0.57724705213935, -0.252034297773712),
            SVector(-0.544031097770073, -0.0440402280844913),
            SVector(-0.424602042497788, 0.0211895908990797),
        )

        @testset "C++ golden values" begin
            sols = collect(homography_5pt_radial(s..., d...))
            @test length(sols) == 3

            # Distortion roots, in the C++ output order
            @test sols[1].k ≈ 5.11091 atol = 1.0e-5
            @test sols[2].k ≈ 2.73767 atol = 1.0e-5
            @test sols[3].k ≈ -0.3 atol = 1.0e-6

            Hg = (
                _rowmat(
                    3.24632054513494e-5, -4.95598175719004e-5, -1.87243913120627e-5,
                    -8.7180348266203e-7, -3.24711594819769e-6, -2.28713196044006e-7,
                    -0.000146151989535363, 0.000231387475606354, 8.596489883095e-5
                ),
                _rowmat(
                    7.88700481939429e-5, -0.000115003527994859, -5.31700374887011e-5,
                    6.35935433111556e-6, -6.84986213748709e-6, -3.56922455715305e-6,
                    -0.000263190991107698, 0.000381884387848442, 0.000176871066503982
                ),
                _rowmat(
                    0.000227753625098076, -7.00643123791218e-5, -0.0005259319838488,
                    0.000429490139445578, 3.06112333501461e-5, -0.00015645439655273,
                    0.000133561563704977, -0.00048633354193638, 0.000545809522795618
                ),
            )
            for i in 1:3
                # Candidates are sign+Frobenius-normalized; compare up to that gauge.
                @test SMatrix{3, 3, Float64}(sols[i].H) ≈ sign_normalize(Hg[i]) rtol = 1.0e-6
            end
            # Only the generating root (k = -0.3) interpolates all five
            # correspondences. Spurious root k₁ still interpolates the four
            # fully-used points but misses the fifth; spurious root k₂
            # undistorts the target quad into near-degeneracy, so even its
            # 4-point interpolation collapses numerically — the C++ reference
            # emits the identical candidate (robust scoring / extra_check
            # discard these downstream).
            @test _max_residual(sols[3], s, d, 5) < 1.0e-7
            @test _max_residual(sols[1], s, d) < 1.0e-7
            @test _max_residual(sols[1], s, d, 5) > 1.0e-3
            @test _max_residual(sols[2], s, d) > 1.0e-3
        end

        @testset "extra_check keeps the consistent root" begin
            sols = collect(homography_5pt_radial(s..., d...; extra_check = true))
            @test length(sols) == 1
            # The golden sample was generated with k = -0.3; the redundant
            # equation singles it out among the three cubic roots.
            @test sols[1].k ≈ -0.3 atol = 1.0e-6
        end

        @testset "synthetic ground-truth recovery" begin
            rng = MersenneTwister(3126)
            n_ok = 0
            n_run = 0
            for _ in 1:500
                # Well-conditioned homography: similarity + mild projective row
                θ = 2π * rand(rng)
                sc = 0.7 + 0.6 * rand(rng)
                v1, v2 = 0.2 * randn(rng), 0.2 * randn(rng)
                t1, t2 = 0.3 * randn(rng), 0.3 * randn(rng)
                H0 = SMatrix{3, 3, Float64}(
                    sc * cos(θ), sc * sin(θ), v1,
                    -sc * sin(θ), sc * cos(θ), v2,
                    t1, t2, 1.0
                )
                k_gt = -0.4 + 0.55 * rand(rng)

                src = ntuple(_ -> SVector(rand(rng) - 0.5, rand(rng) - 0.5), Val(5))
                dst_u = map(p -> _transfer(H0, p), src)
                any(p -> norm(p) > 1.2, dst_u) && continue
                dst_any = map(p -> _distort_div(p, k_gt), dst_u)
                any(isnothing, dst_any) && continue
                dst = map(p -> p::SVector{2, Float64}, dst_any)

                sols = homography_5pt_radial(src..., dst...)
                n_run += 1
                H0n = sign_normalize(H0)

                found = false
                for m in sols
                    # Only the ground-truth root carries an interpolation
                    # guarantee (spurious roots can degenerate — see the golden
                    # testset); it must reproduce (H, k) and interpolate all 5.
                    if abs(m.k - k_gt) < 1.0e-6 &&
                            isapprox(SMatrix{3, 3, Float64}(m.H), H0n; atol = 1.0e-6)
                        @test _max_residual(m, src, dst, 5) < 1.0e-6
                        found = true
                    end
                end
                found && (n_ok += 1)
            end
            @test n_run > 300
            @test n_ok / n_run > 0.99
        end

        @testset "degenerate samples do not throw" begin
            # Collinear sources: candidates with no interpolating homography are
            # filtered; whatever remains must be finite.
            s_bad = (
                SVector(0.0, 0.0), SVector(0.1, 0.1), SVector(0.2, 0.2),
                SVector(0.3, 0.3), SVector(0.4, -0.1),
            )
            sols = homography_5pt_radial(s_bad..., d...)
            # BOTH SIDES. The loop below asserted nothing whenever the filter
            # rejected every candidate, which is what it does here — so the
            # count is pinned too, and a solver that started returning three
            # garbage solutions could no longer succeed by returning none.
            @test length(sols) <= 3
            for m in sols
                @test all(isfinite, m.H) && isfinite(m.k)
            end
        end

        @testset "solver-layer wrapper" begin
            solver = Homography5ptRadial()
            @test sample_size(solver) == 5
            @test max_solutions(solver) == 3
            @test model_type(solver) == RadialHomography

            cs = Csponds(collect(s), collect(d))
            sols = solve(solver, cs, 1:5)
            @test length(sols) == 3
            @test collect(sols)[3].k ≈ -0.3 atol = 1.0e-6

            sols_ec = solve(Homography5ptRadial(extra_check = true), cs, 1:5)
            @test length(sols_ec) == 1
            @test collect(sols_ec)[1].k ≈ -0.3 atol = 1.0e-6
        end

        @testset "projection model ops (uncalibrated back-projection)" begin
            m = RadialHomography{Float64}(
                HomographyMat{Float64}(ntuple(i -> Float64(i in (1, 5, 9)), 9)),
                -0.4, SVector(0.2, -0.1)
            )
            # project ∘ backproject = identity on observations
            for d0 in (SVector(0.7, 0.3), SVector(-1.2, 0.8), SVector(0.21, -0.11))
                @test project(m, backproject(m, d0)) ≈ d0 atol = 1.0e-12
            end
            # omni: a ray BEYOND 90° (z < 0) still projects for k < 0, and its
            # lift points backward — image-space "undistortion" cannot express this
            ray = SVector(1.0, 0.4, -0.3)
            d = project(m, ray)
            @test d !== nothing
            ℓ = backproject(m, d)
            @test ℓ[3] < 0
            @test norm(cross(SVector{3, Float64}(ℓ), ray)) / (norm(ℓ) * norm(ray)) < 1.0e-10
            # k > 0: rays outside the representable field have no image point
            mp = RadialHomography{Float64}(m.H, 0.5, m.center)
            @test project(mp, SVector(5.0, 0.0, 0.1)) === nothing

            # composition with a fixed-center isotropic frame: k/s² law, the
            # model's center lands on the origin, exact round trip
            Hpx = HomographyMat{Float64}(
                (4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0)
            )
            mpx = RadialHomography{Float64}(Hpx, -1.1e-7, SVector(600.0, 450.0))
            f = VgtkCore.CspondNormalization(
                AffineMap(HartleyNormalization(), [SVector(0.1, 0.2), SVector(1.0, -0.5), SVector(-0.3, 0.4)]),
                AffineMap(DiagonalNormalization((1200.0, 900.0)))          # centred at (600, 450), the model's centre
            )
            s = 2 / hypot(1200.0, 900.0)
            Sf = HomographyMap(VgtkCore.source_map(f)); Df = HomographyMap(VgtkCore.target_map(f))
            mn = Df ∘ mpx ∘ inv(Sf)
            @test mn.k ≈ mpx.k / s^2 rtol = 1.0e-12
            @test norm(mn.center) < 1.0e-9
            back = inv(Df) ∘ mn ∘ Sf
            @test back.k ≈ mpx.k rtol = 1.0e-10
            @test back.center ≈ mpx.center atol = 1.0e-6
            @test SMatrix{3, 3, Float64}(back.H) ≈ SMatrix{3, 3, Float64}(mpx.H) rtol = 1.0e-9

            # transfer commutes with the frame
            s0 = SVector(30.0, 40.0)
            d0 = radial_transfer(mpx, s0)
            @test d0 !== nothing
            s0n = (f.src * SVector(s0[1], s0[2], 1.0))[SVector(1, 2)]
            d0n_expected = (f.dst * SVector(d0[1], d0[2], 1.0))[SVector(1, 2)]
            d0n = radial_transfer(mn, s0n)
            @test d0n ≈ d0n_expected atol = 1.0e-9
        end

        @testset "Division lens — closed-form camera + KA image undistortion" begin
            lens = Division(-0.4)
            # forward ∘ inverse = identity on the front hemisphere, both k signs
            for k in (-0.4, 0.15),
                    p in (SVector(0.3, -0.2), SVector(0.8, 0.6), SVector(0.0, 0.0))

                l = Division(k)
                d = l(p)
                @test VgtkCore.undistort_normalized(l, SVector(d[1], d[2])) ≈ p atol = 1.0e-12
            end
            # beyond the 90° circle (r_∞ = 1/√0.4 ≈ 1.581): a ray past 90°, no perspective point
            @test backproject(lens, SVector(1.7, 0.0))[3] < 0
            @test all(isnan, VgtkCore.undistort_normalized(lens, SVector(1.7, 0.0)))
            # forward Jacobian vs central differences
            x0 = SVector(0.31, -0.14)
            J = VgtkCore.transform_deriv(lens, x0)
            fd = FiniteDiff.finite_difference_jacobian(y -> SVector(Tuple(lens(y))), x0, Val(:central))
            @test maximum(abs, SMatrix{2, 2}(J) - fd) < 1.0e-6
            # bounded-undistortion predicate: r_∞ = 1/√(−k) vs farthest corner
            @test division_bounded(0.0, (600, 450), (1200, 900))
            @test division_bounded(-1.0 / 2000.0^2, (600, 450), (1200, 900))
            @test !division_bounded(-1.0 / 700.0^2, (600, 450), (1200, 900))  # r_max ≈ 750

            # CameraModel bridge: unit-focal K ⇒ lens-normalized = centered pixels,
            # so the camera's undistortion agrees with the model's back-projection
            m = RadialHomography{Float64}(
                HomographyMat{Float64}(ntuple(i -> Float64(i in (1, 5, 9)), 9)),
                -1.0e-7, SVector(600.0, 450.0)
            )
            cam = VgtkCore.CameraModel(m)
            @test cam.lens isa Division
            d_px = SVector(900.0, 650.0)
            ℓ = backproject(m, d_px)
            u_cam = VgtkCore.undistort_normalized(
                cam.lens, SVector(d_px[1] - 600.0, d_px[2] - 450.0)
            )
            @test PerspectiveMap()(ℓ) ≈ u_cam atol = 1.0e-12

            # full-image fused KA undistort!: a bright dot at distorted pixel D
            # lands at its back-projected position (convention-proof argmax check)
            img = zeros(Float64, 900, 1200)
            D = (700, 980)                                    # (row, col)
            img[D[1], D[2]] = 1.0
            out = VgtkCore.undistort(cam, img)
            p_col = SVector(D[2] - 0.5, D[1] - 0.5)           # COLMAP pixel center
            ℓd = backproject(m, p_col)
            exp_px = SVector(600.0, 450.0) + PerspectiveMap()(ℓd)
            ci = argmax(out)
            @test norm(SVector(ci[2] - 0.5, ci[1] - 0.5) - exp_px) < 1.5

            # whole-extent undistortion via the general rect API: extent covers
            # the expanded frame; the dot's position honors (origin, scale)
            m2 = RadialHomography{Float64}(m.H, -2.0e-7, m.center)
            ext = VgtkCore.undistorted_rect(m2, (1200, 900))
            wd = VgtkCore.GeometryBasics.widths(ext)
            og = VgtkCore.GeometryBasics.origin(ext)
            @test wd[1] > 1200 && wd[2] > 900              # extent ⊃ the source frame
            σ2 = max(1.0, wd[1] / 2000)
            canvas = VgtkCore.undistort(
                VgtkCore.CameraModel(m2), img, ext; scale = σ2
            )
            # the rect form returns the warp layer's currency: canvas WITH its window
            @test canvas isa FramedImage
            @test canvas.rect === ext
            @test size(canvas, 2) <= 2001
            cf = argmax(canvas.data)
            found_full = SVector(og[1], og[2]) +
                σ2 * SVector(cf[2] - 0.5, cf[1] - 0.5)
            ℓ2 = backproject(m2, p_col)
            exp2 = SVector(600.0, 450.0) + PerspectiveMap()(ℓ2)
            @test norm(found_full - exp2) < 2σ2

        end
    end

    @testset "the normalization: Hartley about the known distortion centre" begin
        using VisualGeometryToolkitCore: radial_transfer, test_sample, similarity_scale, target_map, CspondNormalization
        using Random: Xoshiro, randperm
        f0 = 1400.0; c = SVector(1010.0, 520.0)                     # not the image centre, not the centroid
        R = SMatrix{3, 3}(1.0, 0.0, 0.0, 0.0, cos(0.3), sin(0.3), 0.0, -sin(0.3), cos(0.3))
        t = SVector(40.0, -30.0, 900.0)
        Hm = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(f0, 0, 0, 0, f0, 0, c[1], c[2], 1) * hcat(R[:, 1], R[:, 2], t)))
        m = RadialHomography{Float64}(Hm, -3.0e-7, Point2{Float64}(c[1], c[2]))
        Xs = [SVector(x, y) for x in -150.0:50.0:150.0 for y in -150.0:50.0:150.0]
        cs = Csponds(Xs, [SVector{2, Float64}(radial_transfer(m, X)) for X in Xs])
        fr = CspondNormalization(RadialHomography, cs; center = c)
        @test target_map(fr)(c) == SVector(0.0, 0.0)                # the kernel's origin
        @test_throws UndefKeywordError CspondNormalization(RadialHomography, cs)
        # the 5-pt kernel in this frame recovers the model exactly; a centroid
        # target frame puts the distortion centre on the centroid and misses
        csn = LinearAlgebra.normalize(fr, cs)
        cent = CspondNormalization(HartleyNormalization(), HartleyNormalization(), cs)
        csc = LinearAlgebra.normalize(cent, cs)
        _xf(mb, X) = (d = radial_transfer(mb, X); d === nothing ? Inf : norm(d - radial_transfer(m, X)))
        _pulled(fr, m) = inv(HomographyMap(target_map(fr))) ∘ m ∘ VgtkCore.source_map(fr)
        err(fr, sols) = minimum((maximum(_xf(_pulled(fr, sols[i]), X) for X in Xs) for i in 1:length(sols)); init = Inf)
        rng = Xoshiro(3); best = Inf; bestc = Inf
        for _ in 1:100
            idx = randperm(rng, length(Xs))[1:5]
            best = min(best, err(fr, solve(Homography5ptRadial(), csn, idx)))
            bestc = min(bestc, err(cent, solve(Homography5ptRadial(), csc, idx)))
        end
        @test best < 1.0e-9
        @test bestc > 0.1
        # the target distinctness gate reads a PIXEL separation through the
        # scale: an accepted sample fails exactly past its closest target pair
        σ = similarity_scale(target_map(fr))
        idx = first(filter(i -> test_sample(RadialHomography, csn, i), [randperm(rng, length(Xs))[1:5] for _ in 1:50]))
        dpx = minimum(norm(targets(cs)[a] - targets(cs)[b]) for a in idx for b in idx if a < b)
        @test test_sample(RadialHomography, csn, idx; target_tol = 0.99 * dpx * σ)
        @test !test_sample(RadialHomography, csn, idx; target_tol = 1.01 * dpx * σ)
    end
end

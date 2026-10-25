# Lens contract: the perspective views, the rational lift family, the frame of
# record (CameraModel(model, K′) / K ↔ frame), and the contract on rays past 90°.
@testitem "lens_contract" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra: LinearAlgebra, norm, dot, cholesky, Symmetric
    using StaticArrays: SVector, SMatrix, @SMatrix
    using Rotations: RotXYZ, RotMatrix
    using FiniteDiff: FiniteDiff
    import ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: NoWhitening, SampsonWhitening, monotone_radius, source_map, target_map, radial_ratio, whitened_rows, calibration_stage   # public, not exported
    using VisualGeometryToolkitCore: project_with_jacobians, radial_camera_5pt   # kernel: public, not exported

    @testset "monotone_radius agrees with the fold view_angle_limit reports" begin
        # A family that folds must report the fold in BOTH currencies: the
        # incidence angle (`view_angle_limit`) and the distorted radius
        # (`monotone_radius`). Kannala–Brandt reported the angle and fell
        # through to `Inf` for the radius, so `_default_sampling_clamp` clamped
        # nothing and a `CentralGenericModel` built from such a camera filled
        # its control grid with NaN. The two must be the same fold.
        for l in (
                KannalaBrandt(-0.3, 0.0, 0.0, 0.0),
                KannalaBrandt(-0.2, 0.01),
                Lens(Perspective(), Radial((-0.5,))),
            )
            θ = view_angle_limit(l)
            isfinite(θ) && θ < π || continue
            R = monotone_radius(l)
            @test isfinite(R)
            # The profile's value at the limiting angle IS the limiting radius.
            # `view_angle_limit` locates the fold by a numerical scan and
            # `monotone_radius` by a root of the derivative, so they agree to a
            # few parts per million, not to machine precision; the tolerance is
            # wide enough for that and far too tight for a different fold.
            @test R ≈ norm(project(l, SVector(sin(θ), 0.0, cos(θ)))) rtol = 1.0e-4
        end
        # A family with no fold still says so.
        @test monotone_radius(Pinhole()) == Inf
        @test monotone_radius(Lens(Equidistant())) == Inf
    end

    @testset "lens contract: perspective views" begin
        @testset "the functor is sugar for distort_normalized" begin
            x = SVector(0.3, -0.2)
            for lens in (
                    Pinhole(), Lens(Equidistant()), Lens(Stereographic()), Division(-0.4),
                    Lens(DoubleSphere(0.3, 0.6)), KannalaBrandt(0.05),
                )
                @test lens(x) == distort_normalized(lens, x) == project(lens, SVector(x[1], x[2], 1.0))
            end
        end

        @testset "round trips, both directions" begin
            x = SVector(0.35, -0.21)
            for lens in (
                    Lens(Equidistant()), Division(-0.4), Division(0.15),
                    Lens(DoubleSphere(0.3, 0.6)), KannalaBrandt(0.05, -0.01),
                )
                xd = distort_normalized(lens, x)
                @test undistort_normalized(lens, SVector(xd[1], xd[2])) ≈ x atol = 1.0e-9
                xu = undistort_normalized(lens, x)
                @test distort_normalized(lens, SVector(xu[1], xu[2])) ≈ x atol = 1.0e-9
            end
        end

        @testset "two-coefficient division: closed-form backproject, root-on-the-ray project" begin
            lens = Division(-0.4, 0.05)
            for x in (SVector(0.3, -0.2), SVector(0.7, 0.5), SVector(0.0, 0.0), SVector(1.0, 0.2))
                xd = distort_normalized(lens, x)
                @test undistort_normalized(lens, SVector(xd[1], xd[2])) ≈ x atol = 1.0e-9
            end
            # Two kinds of monotone radius. A POLE lens (Q has a positive root: the
            # 90° circle) represents arbitrarily large undistorted radii, so the
            # projection must converge far past r_∞ in the undistorted plane —
            # the rectified image's outer region — instead of collapsing onto the rim.
            pole = Division(-0.2172, -0.1012)
            Rp = monotone_radius(pole)
            @test abs(radial_ratio(pole.radial, Rp^2)) > 1.0e6 || !isfinite(radial_ratio(pole.radial, Rp^2))   # it IS a pole (Q(Rp²) ≈ ±0)
            for ρu in (1.5, 3.0, 6.0, 30.0)
                xd = distort_normalized(pole, SVector(ρu, 0.0))
                @test xd[1] < Rp
                @test undistort_normalized(pole, SVector(xd[1], xd[2]))[1] ≈ ρu rtol = 1.0e-9
            end
            # A FOLD lens (k₂ > 0 here: Q never vanishes) has a finite maximum
            # undistorted radius g_max at the fold — a maximum field of view below
            # 180°. Targets under it project; rays beyond it are not imaged and come
            # back NaN, not as a point on the fold.
            Rf = monotone_radius(lens)
            gmax = Rf * radial_ratio(lens.radial, Rf^2)
            for ρu in (0.5gmax, 0.95gmax)
                xd = distort_normalized(lens, SVector(ρu, 0.0))
                @test xd[1] < Rf
                @test undistort_normalized(lens, SVector(xd[1], xd[2]))[1] ≈ ρu rtol = 1.0e-8
            end
            @test all(isnan, distort_normalized(lens, SVector(3gmax, 0.0)))
            g(ρ) = ρ * radial_ratio(lens.radial, ρ^2)                  # the back-projection profile
            R = monotone_radius(lens)
            @test g(0.999R) > g(0.99R)                           # increasing inside …
            @test !(g(1.05R) > g(R)) || !isfinite(g(1.05R))     # … not past the monotone radius
            @test monotone_radius(Division(-0.4)) ≈ 1 / sqrt(0.4)
            @test monotone_radius(Division(0.25)) ≈ 2.0
            @test monotone_radius(Pinhole()) == Inf
            @test monotone_radius(Division(0.0)) == Inf
            @test lens isa Division{2, Float64}
            @test Division(-0.4) === LiftLens(Radial((), (-0.4,)))
        end

        @testset "frame of record: CameraModel(model, K′) law, K ↔ frame" begin
            K1 = CameraCalibrationMat(@SMatrix [1.0 0.0 100.0; 0.0 1.0 80.0; 0.0 0.0 1.0])      # unit focal (centered pixels)
            K2 = CameraCalibrationMat(@SMatrix [500.0 0.0 100.0; 0.0 500.0 80.0; 0.0 0.0 1.0])
            m1 = CameraModel(LogicalIntrinsics(K1), Division(-4.0e-7, 2.0e-13))
            m2 = CameraModel(m1, K2)
            @test lens(m2).radial.d[1] ≈ -4.0e-7 * 500^2 rtol = 1.0e-12
            @test lens(m2).radial.d[2] ≈ 2.0e-13 * 500^4 rtol = 1.0e-12
            @test lens(CameraModel(m2, K1)).radial.d[1] ≈ -4.0e-7 rtol = 1.0e-12
            # the same physical map: pixel → undistorted pixel agrees under either K
            px = SVector(700.0, 300.0)
            topx(K, n) = AffineMap(K)(n)
            ninv(K) = inv(calibration_stage(LogicalIntrinsics(K)))          # pixel → K⁻¹·pixel
            u1 = topx(K1, undistort_normalized(lens(m1), ninv(K1)(px)))
            u2 = topx(K2, undistort_normalized(lens(m2), ninv(K2)(px)))
            @test u1 ≈ u2 rtol = 1.0e-10
            @test_throws ArgumentError CameraModel(m1, CameraCalibrationMat(@SMatrix [500.0 0.0 120.0; 0.0 500.0 80.0; 0.0 0.0 1.0]))  # moved principal point
            @test_throws ArgumentError CameraModel(m1, CameraCalibrationMat(@SMatrix [500.0 0.0 100.0; 0.0 520.0 80.0; 0.0 0.0 1.0]))  # anisotropic
            @test_throws ArgumentError CameraModel(CameraModel(LogicalIntrinsics(K1), KannalaBrandt(0.1)), K2)              # not scale-covariant
        end

        @testset "extent and boundedness generalize through the CameraModel" begin
            m = RadialHomography{Float64}(HomographyMat{Float64}(ntuple(i -> Float64(i in (1, 5, 9)), 9)), -6.0e-7, SVector(2120.0, 1416.0))
            e1 = undistorted_rect(m, (4240, 2832))
            e2 = undistorted_rect(CameraModel(m), (4240, 2832))
            @test e1 == e2
            @test division_bounded(CameraModel(m), (4240, 2832)) == division_bounded(m, (4240, 2832))
            two = CameraModel(CameraCalibrationMat(@SMatrix [1.0 0.0 2120.0; 0.0 1.0 1416.0; 0.0 0.0 1.0]); lens = Division(-2.5e-7, -1.5e-13))
            e3 = undistorted_rect(two, (4240, 2832))
            @test all(isfinite, (e3.origin..., e3.widths...)) && all(>(0), e3.widths)
        end

        @testset "the rect walk serves every lens family, not just the division one" begin
            # `undistorted_rect`/`valid_rect` were bound to
            # `CameraModel{<:LogicalIntrinsics, <:` the rational lift `}`, because the
            # boundary walk called `radial_ratio` — a method only that family
            # answers. Everything else raised a MethodError, and with it
            # `plot_rectification`, whose panel is the documented way to judge
            # one fit against another. The walk now goes through
            # `undistort_normalized`, which is the lens contract itself.
            cam_of(lens) = CameraModel(; focal = 400.0, pp = (640.0, 480.0), lens)
            sz = (1280, 960)
            for lens in (
                    Division(-0.3), Division(0.15), Lens(Perspective(), Radial((-0.3,))),
                    Lens(Perspective(), Additive(Radial((-0.159, 0.048)), Tangential(0.006, -0.001))),
                    KannalaBrandt(-0.3, 0.0, 0.0, 0.0), Scaramuzza(-0.3), Pinhole(),
                )
                cam = cam_of(lens)
                outer = undistorted_rect(cam, sz)
                inner = valid_rect(cam, sz)
                @test all(isfinite, outer.origin) && all(isfinite, outer.widths)
                @test all(>(0), outer.widths)
                # The inscribed window lies inside the bounding one, always.
                @test all(inner.origin .>= outer.origin .- 1.0e-6)
                @test all(inner.origin .+ inner.widths .<= outer.origin .+ outer.widths .+ 1.0e-6)
            end
            # A distortion-free lens rectifies to itself.
            pin = undistorted_rect(cam_of(Pinhole()), sz)
            @test pin.origin ≈ [0.0, 0.0] atol = 1.0e-9
            @test pin.widths ≈ [1280.0, 960.0] atol = 1.0e-9
        end

        @testset "a lens that never folds is clipped at its horizon" begin
            # `monotone_radius` is `Inf` for a perspective fisheye, so nothing
            # bounded the walk and the rect was decided by whichever boundary
            # sample landed nearest 90°: an equidistant fisheye at f = 400 over
            # 1280x960 returned 112575x134216 px. The clip is the horizon.
            cam = CameraModel(; focal = 400.0, pp = (640.0, 480.0), lens = Lens(Equidistant()))
            r = undistorted_rect(cam, (1280, 960))
            @test all(isfinite, r.widths)
            @test maximum(r.widths) < 20_000
            # Bounded, but genuinely larger than the frame — the view IS wide.
            @test maximum(r.widths) > 2_000
        end

        @testset "valid_rect — inscribed no-fill window" begin
            W, H = 4240, 2832
            m = RadialHomography{Float64}(HomographyMat{Float64}(ntuple(i -> Float64(i in (1, 5, 9)), 9)), -6.0e-7, SVector(2120.0, 1416.0))
            outer = undistorted_rect(m, (W, H))
            inner = valid_rect(m, (W, H))
            # inner ⊂ outer
            @test all(inner.origin .>= outer.origin)
            @test all(inner.origin .+ inner.widths .<= outer.origin .+ outer.widths .+ 1.0e-9)
            # dual forms agree
            @test valid_rect(CameraModel(m), (W, H)) == inner
            # NO-FILL CONTRACT: every pixel of the inner boundary forward-distorts
            # to INSIDE the source frame (this is the definition, tested directly)
            cm = CameraModel(m)
            K = cm.intrinsics.K
            fx, fy, cx, cy = K[1, 1], K[2, 2], K[1, 3], K[2, 3]
            o, wd = inner.origin, inner.widths
            for t in range(0, 1; length = 200), u in (
                        SVector(o[1] + t * wd[1], o[2]), SVector(o[1] + t * wd[1], o[2] + wd[2]),
                        SVector(o[1], o[2] + t * wd[2]), SVector(o[1] + wd[1], o[2] + t * wd[2]),
                    )
                n = SVector((u[1] - cx) / fx, (u[2] - cy) / fy)
                d = cm.lens(n)
                pix = SVector(cx + fx * d[1], cy + fy * d[2])
                @test -1.0e-6 <= pix[1] <= W + 1.0e-6 && -1.0e-6 <= pix[2] <= H + 1.0e-6
            end
            # near-zero distortion: the inscribed window IS the frame (within a pixel)
            m0 = RadialHomography{Float64}(HomographyMat{Float64}(ntuple(i -> Float64(i in (1, 5, 9)), 9)), -1.0e-13, SVector(2120.0, 1416.0))
            r0 = valid_rect(m0, (W, H))
            @test isapprox(r0.origin[1], 0; atol = 1) && isapprox(r0.origin[2], 0; atol = 1)
            @test isapprox(r0.widths[1], W; atol = 2) && isapprox(r0.widths[2], H; atol = 2)
        end

        @testset "projection function r(θ) and view angle" begin
            # pure fisheye family: the textbook projection functions
            for θ in (0.1, 0.6, 1.2)
                @test projected_radius(Lens(Equidistant()), θ) ≈ θ atol = 1.0e-12
                @test projected_radius(Lens(Stereographic()), θ) ≈ 2tan(θ / 2) atol = 1.0e-12
                @test projected_radius(Lens(Equisolid()), θ) ≈ 2sin(θ / 2) atol = 1.0e-12
                @test projected_radius(Lens(Orthographic()), θ) ≈ sin(θ) atol = 1.0e-12
                @test view_angle(Lens(Equidistant()), θ) ≈ θ atol = 1.0e-12
            end
            kb = KannalaBrandt(0.05, -0.01)
            θ = 0.7
            @test projected_radius(kb, θ) ≈ θ * (1 + 0.05θ^2 - 0.01θ^4) atol = 1.0e-12
            @test view_angle(kb, projected_radius(kb, θ)) ≈ θ atol = 1.0e-9
            # division family: 90° at the monotone radius, round trips both sides of it
            for lens in (Division(-0.4), Division(-0.2172, -0.1012))
                R = monotone_radius(lens)
                @test view_angle(lens, R) ≈ π / 2 atol = 1.0e-6
                for ρ in (0.2 * R, 0.6 * R, 0.95 * R)
                    @test projected_radius(lens, view_angle(lens, ρ)) ≈ ρ rtol = 1.0e-8
                end
                ρ100 = projected_radius(lens, deg2rad(100))          # past 90°: the omni half of the lift
                @test ρ100 > R
                @test view_angle(lens, ρ100) ≈ deg2rad(100) atol = 1.0e-9
            end
            @test_throws ArgumentError projected_radius(Division(-0.4), -0.1)
        end

        @testset "the radial camera and its 5-point solver" begin
            f0, k0 = 1226.3, -6.4e-7
            K = @SMatrix [f0 0.0 0.0; 0.0 f0 0.0; 0.0 0.0 1.0]
            R = RotXYZ(0.4, -0.3, 0.2); t = SVector(30.0, -20.0, 400.0)
            H = SMatrix{3, 3, Float64}(K * hcat(R[:, 1], R[:, 2], t))
            pts = Pair{SVector{2, Float64}, SVector{2, Float64}}[]
            for X in -80.0:40.0:80.0, Y in -80.0:40.0:80.0
                P = R * SVector(X, Y, 0.0) + t
                u = f0 * PerspectiveMap()(P)
                ρ = hypot(u[1], u[2]); r = ρ < 1.0e-12 ? 0.0 : (1 - sqrt(1 - 4k0 * ρ^2)) / (2k0 * ρ)
                push!(pts, SVector(X, Y) => (ρ < 1.0e-12 ? u : u * (r / ρ)))
            end
            # the exact radial camera: rows 1–2 of the true homography (its fit lives in
            # VisualGeometryToolkitEstimation, `fit_radial_camera`)
            Pc = RadialCamera(HomographyMat{Float64}(Tuple(H)))
            @test Pc isa RadialCamera
            a = vec(Pc.P)
            @test SMatrix{3, 3, Float64}(HomographyMat(Pc))[3, :] == SVector(0.0, 0.0, 1.0)
            # the exactly determined case: five pairs spread over the board give the same camera (up to scale)
            P5 = radial_camera_5pt(pts[[1, 5, 8, 21, 25]])                   # four corners of the 5×5 grid and an interior point
            @test P5 !== nothing
            @test abs(abs(dot(vec(P5.P), vec(Pc.P))) / (norm(vec(P5.P)) * norm(vec(Pc.P))) - 1) < 1.0e-6
            @test abs(norm(vec(P5.P)) - 1) < 1.0e-12                           # unit Frobenius norm, the scale never fixed by an entry
            @test_throws ArgumentError radial_camera_5pt(pts[1:6])
            @test radial_camera_5pt(pts[1:5]) === nothing                       # the first grid column: a collinear basis
            # project: the pixel of every pair lies ON the projected radial line, and
            # `direction` is the bare P·X̃ the line is spanned by
            for (X, x) in pts
                @test abs(VgtkCore.distance(VgtkCore.project(Pc, X), x)) < 1.0e-6
                d = VgtkCore.direction(Pc, X)
                @test abs(d[1] * x[2] - d[2] * x[1]) / (norm(d) * norm(x)) < 1.0e-6
            end
            # the centre rides on the camera: the same rows about c explain the RAW
            # pixels, every line goes through c, and the homography carries the translation
            c = SVector(931.0, -472.0)
            raw = [X => (x + c) for (X, x) in pts]
            Pr = RadialCamera(Pc.P, c)
            @test Tuple(Pr.center) == (c[1], c[2])
            for (X, x) in raw
                line = VgtkCore.project(Pr, X)
                @test abs(VgtkCore.distance(line, x)) < 1.0e-6
                @test abs(VgtkCore.distance(line, c)) < 1.0e-9
            end
            @test SMatrix{3, 3, Float64}(HomographyMat(Pr))[3, :] == SVector(0.0, 0.0, 1.0)
            @test SMatrix{3, 3, Float64}(HomographyMat(Pr))[1:2, 3] ≈ Pc.P[:, 3] + c
            # RadialHomography → RadialCamera obeys the SAME T(−c) law: `m.H` is a
            # raw-pixel homography (the model's contract recenters `H s̃`), so its
            # rows must be read about `m.center`. Checked against the model's own
            # `radial_transfer`, an independent path.
            let kd = -1.0e-8,
                    Hm = VgtkCore.HomographyMat{Float64}(
                    Tuple(SMatrix{3, 3, Float64}(f0, 0, 0, 0, f0, 0, c[1], c[2], 1) * hcat(R[:, 1], R[:, 2], t))
                ),
                    m = VgtkCore.RadialHomography{Float64}(Hm, kd, Point2{Float64}(c[1], c[2])),
                    Pm = RadialCamera(m)
                @test Tuple(Pm.center) == (c[1], c[2])
                for X in (SVector(37.0, -21.0), SVector(-64.0, 55.0), SVector(0.0, 0.0))
                    d = VgtkCore.radial_transfer(m, X)
                    d === nothing && continue
                    g, Gx, _ = VgtkCore.constraint_with_jacobians(Pm, X, SVector{2, Float64}(d))
                    @test abs(g) / hypot(Gx[1], Gx[2]) < 1.0e-8       # the observed pixel lies ON the ray
                end
            end
            P5r = radial_camera_5pt(raw[[1, 5, 8, 21, 25]]; center = c)
            # the radial camera's normalization: the pencil centre on the image
            # origin; a 5-point solve in that view, pulled back, is the camera
            Nr = VgtkCore.CspondNormalization(RadialCamera, Csponds(first.(raw), last.(raw)); center = c)
            @test VgtkCore.target_map(Nr)(c) == SVector(0.0, 0.0)
            nraw = LinearAlgebra.normalize(Nr, Csponds(first.(raw), last.(raw)))
            P5n = inv(HomographyMap(target_map(Nr))) ∘ radial_camera_5pt(nraw[[1, 5, 8, 21, 25]]; center = SVector(0.0, 0.0)) ∘ source_map(Nr)
            @test abs(abs(dot(vec(P5n.P), vec(P5r.P))) / (norm(P5n.P) * norm(P5r.P)) - 1) < 1.0e-12
            @test SVector(P5n.center...) ≈ c rtol = 1.0e-14
            @test_throws UndefKeywordError VgtkCore.CspondNormalization(RadialCamera, Csponds(first.(raw), last.(raw)))
            @test Tuple(P5r.center) == (c[1], c[2])
            @test abs(abs(dot(vec(P5r.P), vec(P5.P))) - 1) < 1.0e-9
            # degenerate: collinear plane points leave more than one null direction
            col = [SVector(t, 2t) => pts[1][2] .* (1 + 0.1t) for t in 1.0:5.0]
            @test radial_camera_5pt(col) === nothing
            # the solver over the same kernel: `solve(RadialCamera5pt(; center), cs, idx)`,
            # the centre a property of the solver, a degenerate sample `count == 0`
            cs5 = Csponds(first.(raw), last.(raw))
            s5 = RadialCamera5pt(; center = c)
            @test VgtkCore.sample_size(s5) == 5 && VgtkCore.max_solutions(s5) == 1
            @test VgtkCore.model_type(s5) === RadialCamera{Float64}
            sol = solve(s5, cs5, [1, 5, 8, 21, 25])
            @test sol isa VgtkCore.FixedModels{1, RadialCamera{Float64}} && length(sol) == 1
            @test sol[1].P == P5r.P && sol[1].center == P5r.center
            @test length(solve(RadialCamera5pt(), Csponds(first.(col), last.(col)), 1:5)) == 0
            _solve5(s, cs, idx) = solve(s, cs, idx)
            _solve5(s5, cs5, 1:5)
            @test (@allocated _solve5(s5, cs5, 1:5)) == 0
        end

        @testset "the radial camera of a POSED camera — an identity, and lens-blind" begin
            R = RotXYZ(0.12, -0.25, 0.4)
            ext = VgtkCore.EuclideanMap(RotMatrix{3, Float64}(R), SVector(0.03, -0.02, 1.4))
            model = CameraModel(; focal = 900.0, pp = (640.0, 480.0))
            cam = PosedCamera(model, ext)
            P = RadialCamera(cam)
            # For a PINHOLE camera this must be exactly rows 1–2 of the induced
            # homography about the principal point — the constructor is an identity.
            Pref = RadialCamera(HomographyMat(cam), Point2(640.0, 480.0))
            @test P.P ≈ Pref.P
            @test P.center == Point2(640.0, 480.0)
            # Swapping in a radially symmetric lens (same K, same pose) must leave the
            # radial camera BIT-identical: a distortion about that centre moves a
            # point along its own ray, and the ray is all this model claims.
            cam_div = PosedCamera(CameraModel(model.intrinsics, Division(-0.18, 0.03)), ext)
            Pd = RadialCamera(cam_div)
            @test Pd.P == P.P
            @test Pd.center == P.center
            # ... and the claim is geometric, not merely algebraic: the DISTORTED
            # pixel of a plane point lies on the line the radial camera predicts.
            Xs = (SVector(0.05, -0.07), SVector(-0.11, 0.09), SVector(0.2, 0.15), SVector(-0.25, -0.2))
            for X in Xs
                u = VgtkCore.project(cam_div, SVector(X[1], X[2], 0.0))
                line = VgtkCore.project(Pd, X)
                @test abs(VgtkCore.distance(line, Point2(u[1], u[2]))) < 1.0e-9
            end
            # composition with a normalization's sides — the family's frame law, both directions.
            cs = [X => SVector{2, Float64}(VgtkCore.project(cam_div, SVector(X[1], X[2], 0.0))) for X in Xs]
            fr = VgtkCore.CspondNormalization(HomographyMat, cs)
            Sf, Df = HomographyMap(source_map(fr)), HomographyMap(target_map(fr))
            Pback = inv(Df) ∘ (Df ∘ Pd ∘ inv(Sf)) ∘ Sf
            @test Pback.P ./ Pback.P[1, 1] ≈ Pd.P ./ Pd.P[1, 1]
            @test Pback.center ≈ Pd.center
        end

        @testset "constraint derivatives (radial camera) vs ForwardDiff" begin
            Pm = RadialCamera{Float64}(SMatrix{2, 3, Float64}(0.9, 0.1, -0.2, 1.1, 3.0, -2.0))
            X = SVector(12.0, -7.0); x = SVector(300.0, -150.0)
            g, G_x, G_θ = constraint_with_jacobians(Pm, X, x)
            gx(v) = constraint_with_jacobians(Pm, X, v)[1]
            gθ(v) = constraint_with_jacobians(RadialCamera{eltype(v)}(SMatrix{2, 3}(v[1], v[2], v[3], v[4], v[5], v[6])), X, x)[1]
            @test ForwardDiff.gradient(gx, x) ≈ G_x atol = 1.0e-9
            @test ForwardDiff.gradient(gθ, SVector(vec(Pm.P))) ≈ G_θ atol = 1.0e-9
            # policies: NoWhitening returns the row, Sampson whitens by G_x G_xᵀ (the
            # division lift's two-row form moved with it to VisualGeometryToolkitEstimation)
            r0, J0 = whitened_rows(NoWhitening(), g, G_x, G_θ)
            @test r0 === g && J0 === G_θ
            rs, Js = whitened_rows(SampsonWhitening(), g, G_x, G_θ)
            @test rs ≈ g / norm(G_x) atol = 1.0e-12
            @test Js ≈ G_θ / norm(G_x) atol = 1.0e-12
            # A two-row constraint (the division lift's form, VisualGeometryToolkitEstimation's
            # RadialLift): whitened by a factor W of Σ_g⁻¹ = (G_x G_xᵀ)⁻¹ — the
            # rows' Gram matrices are the Mahalanobis forms, whatever W is.
            # (Moved from VisualGeometryToolkitEstimation's fit_radial_camera_tests.jl.)
            g2 = SVector(0.3, -1.2); Gx2 = SMatrix{2, 3}(1.0, 0.2, -0.4, 0.9, 0.3, 0.5)
            Gθ2 = SMatrix{2, 4}(0.7, -0.1, 0.2, 1.3, -0.6, 0.4, 0.05, 0.8)
            Σg = Gx2 * Gx2'
            r2, J2 = whitened_rows(SampsonWhitening(), g2, Gx2, Gθ2)
            @test dot(r2, r2) ≈ dot(g2, Σg \ g2) rtol = 1.0e-12
            @test J2' * J2 ≈ Gθ2' * (Σg \ Gθ2) rtol = 1.0e-12
            @test J2' * r2 ≈ Gθ2' * (Σg \ g2) rtol = 1.0e-12
            @test whitened_rows(NoWhitening(), g2, Gx2, Gθ2) === (g2, Gθ2)
        end
    end

    # =============================================================================
    # The lens contract on rays — every family, the whole field
    # =============================================================================
    #
    # `project` / `backproject` ARE the lens contract (geometry/cameras/lens_rays.jl).
    # Every older route held a `Z > 0` gate or a `tan θ` frame at some point, each
    # independently "correct" for a rectilinear lens; a lens that images past 90° is
    # the only thing that tells them apart, and nothing in a pinhole suite can. So
    # the oracles here are the textbook projection formulas, not the package's
    # helpers, and the angles run from the optical axis through 90° to 175°.
    @testset "lens contract on rays" begin
        C = VgtkCore.Cameras
        ray(θ; ψ = atan(0.6, 0.8)) = SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))   # unit: θ IS the view angle
        viewangle(X) = atan(hypot(X[1], X[2]), X[3])
        profile(ρ) = X -> (r = hypot(X[1], X[2]); iszero(r) ? SVector(0.0, 0.0) : ρ(viewangle(X)) / r * SVector(X[1], X[2]))

        rneg = LiftLens(Radial((), (-0.31259823703309836, -0.08123929939414641)))    # k₂ < 0: never folds, the field runs to 180°
        kb = KannalaBrandt(-0.01, 0.001, -1.0e-4, 1.0e-5)
        tpf = Lens(Equidistant(), Additive(Radial((-0.02, 0.004, -0.0009, 0.0001)), Tangential(0.0006, -0.0004), ThinPrism((0.0002,), (-0.0001,))))
        ds = Lens(DoubleSphere(0.3, 0.6))                                            # field of view to ≈ 142°
        bc = Lens(Perspective(), Additive(Radial((-0.12, 0.03)), Tangential(1.0e-3, -5.0e-4)))
        sct = Scaramuzza(-0.1664, -0.0063, 0.035)              # MATLAB's tamron fit, normalized: folds at ≈ 67°
        scf = Scaramuzza(-0.4865, 0.239, -0.175)                # the fisheye camera's: never folds, images past 90°
        bct = Lens(Perspective(), Additive(Radial((-0.12, 0.03)), Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
        thinprism(X) = begin                                                        # OpenCV's formula, written out
            x, y = PerspectiveMap()(X)
            r2 = x * x + y * y
            c = parameters(bct)
            rad = 1 + c.k1 * r2 + c.k2 * r2^2
            SVector(
                rad * x + 2 * c.p1 * x * y + c.p2 * (r2 + 2 * x * x) + c.sx1 * r2 + c.sx2 * r2^2,
                rad * y + 2 * c.p2 * x * y + c.p1 * (r2 + 2 * y * y) + c.sy1 * r2 + c.sy2 * r2^2,
            )
        end

        # (lens, textbook image of a unit ray — `nothing` where there is no closed form, widest angle tested)
        families = (
            (Lens(Equidistant()), profile(θ -> θ), 175.0),
            (Lens(Stereographic()), profile(θ -> 2tan(θ / 2)), 175.0),
            (Lens(Equisolid()), profile(θ -> 2sin(θ / 2)), 175.0),
            (Lens(Orthographic()), profile(sin), 89.0),
            (kb, profile(θ -> (c = parameters(kb); θ * (1 + c.k1 * θ^2 + c.k2 * θ^4 + c.k3 * θ^6 + c.k4 * θ^8))), 175.0),
            (tpf, X -> (e = profile(θ -> θ)(X); C._distort(tpf.distortion, e)), 175.0),
            (ds, X -> C.FramedLensProjection(ds)(X), 140.0),
            (rneg, nothing, 175.0),
            (bc, X -> bc(PerspectiveMap()(X)), 45.0),
            (bct, thinprism, 45.0),
            (sct, nothing, 60.0),
            (scf, nothing, 175.0),
            (Pinhole(), PerspectiveMap(), 60.0),
        )
        angles = (0.0, 5.0, 30.0, 60.0, 89.999, 90.0, 95.0, 110.0, 140.0, 175.0)

        # FILTERED, not guarded. An `if θd <= θmax` inside the body left the
        # skipped combinations as testsets that ran no tests — 38 of them, each
        # a warning in a green suite, which is exactly how a family whose limit
        # is wrong comes to test nothing and say nothing.
        cases = [(l, image, θd) for (l, image, θmax) in families for θd in angles if θd <= θmax]
        @test !isempty(cases)

        @testset "$(nameof(typeof(l))) at $(θd)°" for (l, image, θd) in cases
            begin
                X = ray(deg2rad(θd))
                u = project(l, 2.5 * X)                                              # any positive scale
                image === nothing || @test u ≈ image(X) atol = 1.0e-10
                b = backproject(l, u)
                @test b ≈ X atol = 1.0e-9
                @test norm(b) ≈ 1 atol = 1.0e-12
                # the Jacobians of the maps themselves: finite everywhere, the optical axis included
                JX = transform_deriv(l, X)
                @test all(isfinite, JX)
                @test JX ≈ FiniteDiff.finite_difference_jacobian(v -> project(l, v), X, Val(:central)) rtol = 1.0e-5 atol = 1.0e-7
                p = parameter_vector(l)
                if !isempty(p)
                    Jp = transform_deriv_params(l, X)
                    @test Jp ≈ FiniteDiff.finite_difference_jacobian(q -> project(with_parameters(l, q), X), p, Val(:central)) rtol = 1.0e-5 atol = 1.0e-7
                    u2, JX2, Jp2 = project_with_jacobians(l, X)
                    @test u2 ≈ u && JX2 ≈ JX && Jp2 ≈ Jp
                end
            end
        end

        # A 190° field of view — 95° at the image-circle rim — pixel by pixel: every
        # pixel inside the circle back-projects to a ray that projects to the same
        # pixel, and the rim really is past the horizon.
        wide = [l for (l, _, θmax) in families if θmax >= 95]
        @test !isempty(wide)
        @testset "190° field of view: $(nameof(typeof(l)))" for l in wide
            begin
                R = 1000.0
                model = CameraModel(; focal = R / norm(project(l, ray(deg2rad(95.0)))), pp = (1100.0, 1100.0), lens = l)
                worst, widest = 0.0, 0.0
                for i in 0:40, j in 0:40
                    px = SVector(100.0 + 50i, 100.0 + 50j)
                    if hypot(px[1] - 1100, px[2] - 1100) <= R
                        b = backproject(model, px)
                        widest = max(widest, rad2deg(viewangle(b)))
                        worst = max(worst, norm(SVector{2}(Tuple(project(model, b))) - px))
                    end
                end
                @test worst < 1.0e-6
                @test widest > 94.99
            end
        end

        @testset "Scaramuzza: the division lift in the radius" begin
            # a₂ alone is Division(a₂)'s lift 1 + a₂ρ², and (a₂, a₄) is Division(a₂, a₄)'s —
            # the same rays, the same folds, the same NaNs, through a different solver
            for (sc, dv) in (
                    (Scaramuzza(-0.4), Division(-0.4)), (Scaramuzza(0.2), Division(0.2)),
                    (Scaramuzza(-0.3302, 0.0, -0.0765), Division(-0.3302, -0.0765)),
                )
                for θd in (0.0, 5.0, 30.0, 47.0, 60.0, 89.999, 90.0, 95.0, 140.0, 175.0)
                    X = ray(deg2rad(θd))
                    u, v = project(sc, X), project(dv, X)
                    @test (all(isnan, u) && all(isnan, v)) || isapprox(u, v; atol = 1.0e-10)
                end
                for xd in (SVector(0.0, 0.0), SVector(0.3, -0.2), SVector(0.9, 0.4), SVector(1.5, 0.0), SVector(2.5, 0.0))
                    b, c = backproject(sc, xd), backproject(dv, xd)
                    @test (all(isnan, b) && all(isnan, c)) || isapprox(b, c; atol = 1.0e-12)
                end
                @test monotone_radius(sc) ≈ monotone_radius(dv)
                @test C._fold_radius(sc) ≈ C._fold_radius(dv)
            end
            # the odd term is what the division family cannot carry
            @test C._fold_radius(Scaramuzza(0.0, 0.05)) ≈ cbrt(10.0) atol = 1.0e-9   # 1 − 2a₃ρ³ = 0
            @test !isfinite(C._fold_radius(Scaramuzza(0.0, -0.05)))
            @test view_angle(Scaramuzza(0.0, -0.05), 0.5) > atan(0.5)               # g < 1: wider than perspective
            @test all(isfinite, project(Scaramuzza(0.0, -0.05), ray(deg2rad(80.0))))
            @test all(isnan, project(Scaramuzza(0.0, 0.05), ray(deg2rad(80.0))))    # past the fold: θ_fold ≈ 55°
            # the fisheye camera's lift reaches 90° where g vanishes, and beyond
            ρ90 = monotone_radius(scf)
            a = parameters(scf)
            @test isfinite(ρ90) && abs(1 + a.d1 * ρ90^2 + a.d2 * ρ90^3 + a.d3 * ρ90^4) < 1.0e-9
            @test view_angle(scf, ρ90) ≈ π / 2 atol = 1.0e-9
            @test view_angle(scf, 2ρ90) > π / 2
            @test C.view_angle_limit(scf) > deg2rad(170)
            @test C.view_angle_limit(sct) ≈ atan(C._fold_radius(sct), C._den(sct.radial, C._fold_radius(sct))) atol = 1.0e-6
            # the axis: the lift is perspective to first order, so the Jacobians are the pinhole's
            @test all(iszero, transform_deriv_params(sct, SVector(0.0, 0.0, 1.0)))
            @test transform_deriv(sct, SVector(0.0, 0.0, 2.0)) ≈ SMatrix{2, 3, Float64}(0.5, 0.0, 0.0, 0.5, 0.0, 0.0)
            @test undistort_normalized(sct, SVector(0.0, 0.0)) == SVector(0.0, 0.0)
            @test transform_deriv(inv(sct), SVector(0.0, 0.0)) ≈ SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 1.0)
            # Float32 stays Float32
            @test project(Scaramuzza(Float32(-0.17f0), Float32(-0.006f0), Float32(0.035f0)), SVector{3, Float32}(0.3, 0.2, 1.0)) isa Vec2{Float32}
        end

        # cv2.projectPoints (OpenCV 4.12.0) with K = I on the tamron thin-prism
        # calibration (k1–k6, p1, p2, s1–s4 all non-zero) and on an exaggerated
        # synthetic set: our forward map is OpenCV's term for term, and our inverse
        # returns to the point (cv2.undistortPoints itself stops after five trips
        # and is off by ~1e-6 at ρ ≈ 0.9; ours runs to convergence).
        @testset "OpenCV thin prism against cv2.projectPoints" begin
            tamron = Lens(Perspective(), Additive(Radial((-2.7067990842834013, 4.003544352914542, 1.5827483616019924), (-2.5409744152370775, 3.5017575444899127, 2.4059525010963547)), Tangential(0.0028263106588956688, 0.0017230351557208537), ThinPrism((-3.137923848171712e-5, 7.363153842759605e-5), (-0.00322446080396022, 0.00027475714123090727))))
            synthetic = Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)), Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
            @test (parameters(tamron).sx1, parameters(tamron).sx2, parameters(tamron).sy1, parameters(tamron).sy2) ==
                (-3.137923848171712e-5, 7.363153842759605e-5, -0.00322446080396022, 0.00027475714123090727)
            xs = (SVector(0.3, -0.2), SVector(-0.05, 0.4), SVector(0.7, 0.55), SVector(-0.6, -0.45), SVector(0.0, 0.85))
            oracle = (
                (
                    tamron, (
                        SVector(0.2941092714796456, -0.195972524014775), SVector(-0.04857894075451882, 0.3907887892982418),
                        SVector(0.6388668681104548, 0.5007341608447827), SVector(-0.5523403240742476, -0.4151234080848689),
                        SVector(0.001260657523461596, 0.778607952399596),
                    ),
                ),
                (
                    synthetic, (
                        SVector(0.2937350983111105, -0.1959397922074071), SVector(-0.04990676153989657, 0.3920902798191727),
                        SVector(0.634167492819129, 0.5034474316346728), SVector(-0.5624490262301972, -0.4180513204538979),
                        SVector(-0.003969595624999999, 0.7830777145826354),
                    ),
                ),
            )
            for (lens, expected) in oracle, (x, xd) in zip(xs, expected)
                @test distort_normalized(lens, x) ≈ xd atol = 1.0e-13
                @test undistort_normalized(lens, xd) ≈ x atol = 1.0e-9
                @test project(lens, SVector(x[1], x[2], 1.0)) ≈ xd atol = 1.0e-13
            end
            @test distort_normalized(tamron, SVector(0.0, 0.0)) == SVector(0.0, 0.0)
            # a thin prism is not radial: `radial_part` keeps the radial factor only
            @test VgtkCore.radial_part(synthetic) === Lens(Perspective(), Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)))
            @test radial(synthetic) === Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003))
            @test VgtkCore.radial_part(Division(-0.2)) == Division(-0.2)          # a lift: as is
            @test Scaramuzza(0.0, 0.0, 0.0) isa LiftLens && Division(-0.2) isa LiftLens   # the division model is a rational lift
            @test !(KannalaBrandt(0.0, 0.0, 0.0, 0.0) isa LiftLens)
        end

        @testset "outside the lens's domain: NaN, never a throw" begin
            @test all(isnan, project(Lens(Stereographic()), SVector(0.0, 0.0, -1.0)))    # the backward axis images a circle
            @test all(isnan, project(Pinhole(), SVector(0.1, 0.0, -1.0)))                 # a perspective lens ends at 90°
            @test all(isnan, project(ds, ray(deg2rad(150.0))))                            # past the double-sphere field of view
            @test all(isnan, transform_deriv(ds, ray(deg2rad(150.0))))
            @test all(isnan, transform_deriv_params(ds, ray(deg2rad(150.0))))
            @test all(isnan, backproject(Lens(Equisolid()), SVector(1.6, 1.3)))          # past the equisolid radius 2
            @test all(isnan, backproject(KannalaBrandt(-0.3), SVector(0.9, 0.0)))   # θ_d(θ) peaks at 0.703: no θ images 0.9
            fold = Division(0.2)                                                          # the lift folds at 48.19°
            @test all(isfinite, project(fold, ray(deg2rad(47.0))))
            @test all(isnan, project(fold, ray(deg2rad(60.0))))
            # and past the fold radius no ray is behind the image point: no folded ray
            Rfold = C._fold_radius(fold)
            @test project(fold, backproject(fold, SVector(0.9Rfold, 0.0))) ≈ SVector(0.9Rfold, 0.0) atol = 1.0e-10
            @test all(isnan, backproject(fold, SVector(1.5Rfold, 0.0)))
            @test all(isnan, undistort_normalized(fold, SVector(1.5Rfold, 0.0)))
            # the fold is found exactly, not from the local sign: a lift whose view angle
            # turns back UP past a second fold is still off the branch there
            twofold = Division(11 / 6, -1 / 3, 1 / 30)                                   # lift numerator (1 − s)(1 − s/2)(1 − s/3)
            Qt(s) = 1 + (11 / 6) * s - (1 / 3) * s^2 + (1 / 30) * s^3
            dQt(s) = 11 / 6 - (2 / 3) * s + (1 / 10) * s^2
            numer(s) = Qt(s) - 2s * dQt(s)                                                # PQ + 2s(P′Q − PQ′) with P = 1
            @test C._fold_coeffs(twofold.radial) ≈ SVector(1.0, -11 / 6, 1.0, -1 / 6)
            @test all(s -> numer(s) ≈ evalpoly(s, Tuple(C._fold_coeffs(twofold.radial))), (0.3, 1.7, 2.5, 4.0))
            @test numer(2.5) > 0                                                          # positive again between the 2nd and 3rd roots …
            @test all(isnan, backproject(twofold, SVector(sqrt(2.5), 0.0)))              # … yet past the first fold
            @test C._fold_radius(twofold) ≈ 1.0 atol = 1.0e-9
            bt = backproject(twofold, SVector(0.9, 0.0))
            @test all(isfinite, bt)
            @test project(twofold, bt) ≈ SVector(0.9, 0.0) atol = 1.0e-9
            θtop = C.view_angle(twofold, 1.0 - 1.0e-9)
            @test all(isfinite, project(twofold, ray(θtop - 0.05)))
            @test all(isnan, project(twofold, ray(θtop + 0.05)))
            @test monotone_radius(twofold) ≈ 1.0 atol = 1.0e-9
            # Kannala–Brandt folds where 1 + 3k₁θ² + … first vanishes: past it no ray is imaged
            kfold = KannalaBrandt(-0.3)                                           # fold at θ = 1/√0.9
            @test all(isfinite, project(kfold, ray(1.0)))
            @test all(isnan, project(kfold, ray(1.1)))
            # where the ratio Newton from g = 1 misses a root that exists on the branch, the
            # safeguarded solve on [0, θ_fold) recovers it (the Newton-only inverse was NaN here)
            steep = KannalaBrandt(-0.2, 0.02, 0.0, 0.002)
            for θ in (2.4, 2.9)
                X = ray(θ)
                @test backproject(steep, project(steep, X)) ≈ X / norm(X) atol = 1.0e-8
            end
            # a lift that folds before 90°: finite inside its field, NaN past the fold
            tam = LiftLens(Radial((), (-0.17114550341263696, 0.029918349030169777)))
            θfold = C.view_angle(tam, C._fold_radius(tam))
            @test θfold < π / 2
            @test isfinite(C.projected_radius(tam, θfold * 0.9))
            @test isnan(C.projected_radius(tam, θfold * 1.05))
            @test isnan(C.projected_radius(tam, π / 2))
            @test !isfinite(C.projected_radius(rneg, Float64(π)))                         # 180° is a limit, not a value
        end

        @testset "the number type follows the inputs" begin
            kb32 = KannalaBrandt(Float32(-0.01f0), Float32(0.001f0))
            u32 = project(kb32, SVector{3, Float32}(ray(deg2rad(100.0))))
            @test eltype(u32) === Float32
            @test eltype(backproject(kb32, u32)) === Float32
        end

        @testset "allocation-free" begin
            allocs(f, a, b) = (f(a, b); @allocated f(a, b))
            X = ray(deg2rad(100.0))
            for l in (kb, tpf, ds, rneg, Lens(Equisolid()), bc)
                Y = l === bc ? ray(deg2rad(30.0)) : X
                u = project(l, Y)
                @test allocs(project, l, Y) == 0
                @test allocs(backproject, l, u) == 0
                @test allocs(transform_deriv, l, Y) == 0
                @test allocs(transform_deriv_params, l, Y) == 0
            end
        end

        # A perspective radial factor images nothing past its fold, like Kannala–Brandt and
        # the rational family.
        @testset "a perspective radial factor past its fold" begin
            bcf = Lens(Perspective(), Radial((-0.3,)))                  # fold at s = 1/0.9: θ ≈ 46.5°
            ray0(θ) = SVector(sin(θ), 0.0, cos(θ))
            @test all(isnan, project(bcf, ray0(deg2rad(60.0))))
            u = project(bcf, ray0(deg2rad(30.0)))
            @test all(isfinite, u)
            b = backproject(bcf, u)
            @test atan(hypot(b[1], b[2]), b[3]) ≈ deg2rad(30.0) atol = 1.0e-9
            ρ = sqrt(1 / 0.9)
            @test monotone_radius(bcf) ≈ ρ * (1 - 0.3 * ρ^2) rtol = 1.0e-9
            @test monotone_radius(Lens(Perspective(), Radial((0.1,)))) == Inf              # never folds
            @test monotone_radius(Lens(Perspective(), Radial((-0.12, 0.03)))) == Inf  # 1 − 0.36s + 0.15s² has no real root
        end

        @testset "a rounding-artefact leading coefficient" begin
            # A fit onto a lower-degree source lands its leading coefficient at
            # ~1e-17 — the normal outcome, not an exotic input — and Cauchy's root
            # bound and the fold then sit at powers of its inverse (~1e16, ~1e8).
            # The fold sweep stalled at the origin (a fold at 0: no field at all)
            # and the bracketed solves spanned the far fold on a fixed budget (NaN).
            sc_ro = Scaramuzza(-0.3302, 0.0, 1.0e-17)
            @test C._fold_radius(sc_ro) ≈ sqrt((0.3302 + sqrt(0.3302^2 + 12.0e-17)) / 6.0e-17) rtol = 1.0e-9   # 1 + 0.3302ρ² − 3a₄ρ⁴ = 0
            @test monotone_radius(sc_ro) ≈ monotone_radius(Scaramuzza(-0.3302)) rtol = 1.0e-9     # g's zero, as without a₄
            @test C.view_angle_limit(sc_ro) > deg2rad(170)
            # Division{3} onto Division(k₁, k₂): the lift numerator's fold at ρ ~ 5e7
            dv_ro = Division(-0.3302, -0.0765, 2.3e-17)
            for θ in (2.7, 2.9, 3.1)
                @test projected_radius(dv_ro, θ) ≈ projected_radius(Division(-0.3302, -0.0765), θ) rtol = 1.0e-9
            end
            # the Scaramuzza bracket at a₄ = 1e-14: the fold at ρ ~ 4e6 alone left the
            # solve unconverged; the doubling bound localizes it
            sc4 = Scaramuzza(-0.4, 0.0, 1.0e-14)
            r, z = sin(0.27), cos(0.27)
            ρ, ok = C._radius_bracketed(sc4.radial, r, z, NaN)
            @test ok && abs(C._h(sc4.radial, ρ, 1.0, r, z)) <= 1.0e-12
            # the bracket is the tighter of the fold and the first doubling at which h ≥ 0
            h(ρ) = ρ - 3.0
            @test C._lift_bracket(h, 1.0, Inf) == 4.0     # 1 → 2 → 4
            @test C._lift_bracket(h, 1.0, 2.5) == 2.5     # the fold is tighter
            @test C._lift_bracket(h, NaN, NaN) == 4.0     # no usable seed starts at 1; a NaN fold bounds nothing
        end
    end
end

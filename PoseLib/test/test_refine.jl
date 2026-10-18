# Robust-loss refinement wrappers and essential_dlt: correspondences in as
# one Cspond vector (model first). Synthetic two-view / 2D-3D data with 20 %
# gross outliers; every wrapper must pull a perturbed seed back to the truth,
# and every Cspond container (Csponds, Vector{Pair}) must give the same model.

import VisualGeometryToolkitCore as VgtkCore
using Random: MersenneTwister
using LinearAlgebra: normalize, norm, dot
using Rotations: AngleAxis
using VisualGeometryToolkitCore: Csponds, withattrs, backproject_with_jacobian, s2_tangent_basis,
                          CameraModel, LogicalIntrinsics, CameraCalibrationMat, Pinhole,
                          TLSShape, HuberShape, CauchyShape, GaussianShape

_rot_err_deg(Ra, Rb) = rad2deg(acos(clamp((sum(SMatrix{3,3}(Ra) .* SMatrix{3,3}(Rb)) - 1) / 2, -1, 1)))
_nrm(Y) = SVector(Y[1] / Y[3], Y[2] / Y[3])
_proj(K, Y) = _nrm(K * Y)
# Projective 3×3 models agree up to scale and sign.
_proj_dist(A, B) = (a = vec(Matrix(A)) / norm(Matrix(A)); b = vec(Matrix(B)) / norm(Matrix(B));
                    min(norm(a - b), norm(a + b)))

@testset "refine_* and essential_dlt take csponds" begin
    rng = MersenneTwister(7)
    n = 200; nout = n ÷ 5
    # SIMPLE_PINHOLE intrinsics (fx = fy): the camera-aware wrapper hands
    # PoseLib one focal per view, (fx + fy) / 2.
    K1 = SMatrix{3,3}(800.0, 0, 0, 0, 800.0, 0, 320.0, 240.0, 1.0)
    K2 = SMatrix{3,3}(760.0, 0, 0, 0, 760.0, 0, 330.0, 250.0, 1.0)
    R = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.2, normalize([0.3, 1.0, 0.1])...)))
    t = normalize(SVector(1.0, 0.1, 0.05))
    X = [SVector(4 * randn(rng), 3 * randn(rng), 8.0 + 4 * rand(rng)) for _ in 1:n]
    Rs = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.01, 0.0, 1.0, 0.0))) * R
    seed = PoseLib.Rigid3(RotMatrix{3}(Rs), Vec3{Float64}(normalize(t + SVector(0.0, 0.02, 0.0))))
    # Each testset draws from its OWN rng: with a shared one, adding or
    # removing a draw in any testset silently retunes every testset after it.
    outlier2(r) = SVector(640 * rand(r), 480 * rand(r))

    # The z = 1 chart is REFUSED for calibrated relative pose, in both Cspond
    # containers. The point of a throw rather than a deleted method is the
    # message: a bare MethodError would tell a caller that some argument is
    # wrong, not which of the two live forms to move to.
    @testset "refine_relpose refuses the z = 1 chart" begin
        x1 = _nrm.(X)
        x2 = [_nrm(R * Y + t) for Y in X]
        @test_throws ArgumentError refine_relpose(seed, Csponds(x1, x2); max_iter = 25, loss_width = 3e-3)
        @test_throws ArgumentError refine_relpose(seed, x1 .=> x2; max_iter = 25, loss_width = 3e-3)
        msg = try
            refine_relpose(seed, Csponds(x1, x2); max_iter = 25, loss_width = 3e-3)
        catch e
            sprint(showerror, e)
        end
        # Names both replacements and the producer, not just the complaint.
        @test occursin("M₁", msg) && occursin("backproject_with_jacobian", msg)
        @test occursin("refine_relpose(pose, K1, K2, cs; max_iter, loss_width)", msg)
    end

    @testset "refine_relpose (camera-aware, pixels)" begin
        rngc = MersenneTwister(101)
        x1 = [_proj(K1, Y) for Y in X]
        x2 = [_proj(K2, R * Y + t) for Y in X]
        x2[1:nout] .= [outlier2(rngc) for _ in 1:nout]
        ref = refine_relpose(seed, K1, K2, Csponds(x1, x2); max_iter = 25, loss_width = 2.0)
        @test _rot_err_deg(ref.R, R) < 1e-3
        @test dot(normalize(ref.t), t) > 1 - 1e-8
    end

    tx = SMatrix{3,3}(0.0, t[3], -t[2], -t[3], 0.0, t[1], t[2], -t[1], 0.0)
    @testset "refine_fundamental" begin
        rngf = MersenneTwister(102)
        F = inv(K2)' * tx * R * inv(K1)
        x1 = [_proj(K1, Y) for Y in X]
        x2 = [_proj(K2, R * Y + t) for Y in X]
        x2[1:nout] .= [outlier2(rngf) for _ in 1:nout]
        F0 = F .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(rngf, 9)))
        Fr = refine_fundamental(F0, Csponds(x1, x2); max_iter = 25, loss_width = 2.0)
        @test Fr isa FundMat{Float64}
        @test _proj_dist(Fr, F) < 1e-6 < _proj_dist(F0, F)
    end

    @testset "refine_homography" begin
        rngh = MersenneTwister(103)
        H = K2 * (R + t * SVector(0.0, 0.0, 0.1)') * inv(K1)
        x1 = [outlier2(rngh) for _ in 1:n]
        x2 = [_proj(H, SVector(u[1], u[2], 1.0)) for u in x1]
        x2[1:nout] .= [outlier2(rngh) for _ in 1:nout]
        H0 = H .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(rngh, 9)))
        Hr = refine_homography(H0, Csponds(x1, x2); max_iter = 25, loss_width = 2.0)
        @test Hr isa HomographyMat{Float64}
        @test _proj_dist(Hr, H) < 1e-6 < _proj_dist(H0, H)
    end

    @testset "refine_absolute_pose refuses the z = 1 chart" begin
        x = [_nrm(R * Y + t) for Y in X]
        @test_throws ArgumentError refine_absolute_pose(seed, Csponds(X, x); max_iter = 25, loss_width = 0.05)
        msg = try
            refine_absolute_pose(seed, Csponds(X, x); max_iter = 25, loss_width = 0.05)
        catch e
            sprint(showerror, e)
        end
        @test occursin("refine_absolute_pose(pose, K, cs; max_iter, loss_width)", msg)
        # The alias is the FUNCTION, so it survives a method withdrawal.
        @test PoseLib.bundle_adjust === refine_absolute_pose
    end

    # -------------------------------------------------------------------------
    # The BEARING refiner (poselib::refine_relpose(d1, d2, M1, M2, …),
    # bundle.cc:228). The camera-aware pixel wrapper above REDUCES to it: with
    # the intrinsics fixed, PoseLib calls camera.unproject_with_jac() and hands
    # (d, M) to this same FixCameraRelativePoseRefiner. So feeding it the same
    # (d, M) from VGC's backproject_with_jacobian must reproduce that wrapper's
    # answer — and does, to ~1e-16.
    #
    # WHAT THE 1e-12 BOUND CATCHES (each measured against this data, as
    # max|ΔR| from the camera-aware answer, which agrees to 5.6e-17):
    #   the loss width read in bearing units (width/f, the "normalized units" mistake)
    #       → the truncated loss zero-weights everything and the refiner
    #         returns the SEED: rot error stays 0.573°, ΔR 9.7e-3.
    #   M₁ and M₂ swapped between the views → ΔR 1.4e-5.
    #   M packed row-major instead of column-major → ΔR 1.9e-4.
    # The tightest of those is 7 orders the far side of the bound, and the
    # bound is 4 orders above the agreement it asserts. Two focals on purpose:
    # with f₁ = 800 and f₂ = 1100 the two M's differ in scale, which is what
    # makes the swap visible at all. Local rng so the outlier draw does not
    # depend on where this testset sits in the file.
    @testset "refine_relpose (bearings, tangent Sampson)" begin
        rngb = MersenneTwister(31)
        Kb2 = SMatrix{3,3}(1100.0, 0, 0, 0, 1100.0, 0, 330.0, 250.0, 1.0)
        u1 = [_proj(K1, Y) for Y in X]
        u2 = [_proj(Kb2, R * Y + t) for Y in X]
        u2[1:nout] .= [SVector(660 * rand(rngb), 500 * rand(rngb)) for _ in 1:nout]
        cam1 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K1)), Pinhole())
        cam2 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(Kb2)), Pinhole())
        bm1 = backproject_with_jacobian.(Ref(cam1), u1)
        bm2 = backproject_with_jacobian.(Ref(cam2), u2)
        f1 = first.(bm1); M₁ = last.(bm1)
        f2 = first.(bm2); M₂ = last.(bm2)
        cs = withattrs(Csponds(f1, f2); M₁, M₂)

        loss_width = 2.0                         # PIXELS: M = ∂f/∂u carries the unit
        pix = refine_relpose(seed, K1, Kb2, Csponds(u1, u2); max_iter = 25, loss_width)
        bear = refine_relpose(seed, cs; max_iter = 25, loss_width)
        @test _rot_err_deg(bear.R, R) < 0.05 < _rot_err_deg(seed.R, R)
        @test dot(normalize(bear.t), t) > 1 - 1e-5
        @test maximum(abs.(SMatrix{3,3}(bear.R) .- SMatrix{3,3}(pix.R))) < 1e-12
        @test norm(bear.t - pix.t) < 1e-12

        # The Jacobians are not optional: they, and nothing else, set the loss width's unit.
        @test_throws ArgumentError refine_relpose(seed, Csponds(f1, f2); max_iter = 25, loss_width)

        # THE CHART IN 3-VECTOR CLOTHING — the case the StaticVector{3}
        # signature cannot see. Lift each bearing to z = 1 and it stays a
        # 3-vector and keeps the same DIRECTION, so the refiner still answers
        # plausibly: measured by calling the C shim with the guard bypassed, the
        # lifted set returns a finite pose 0.0086° from truth against the honest
        # 0.0067°, from a 0.573° seed — it refines, it looks right, and it is
        # 2.1e-4 from the correct answer. Nothing downstream would flag it, and
        # the exclusion of rays past 90° has quietly survived. Only the norm
        # betrays it (1.41 at 45° off axis here), which is why the guard is a
        # norm check and why it lives at this boundary rather than in a loop.
        chart1 = [SVector(f[1] / f[3], f[2] / f[3], 1.0) for f in f1]
        @test maximum(norm, chart1) > 1.05          # a real chart, not the axis
        @test_throws ArgumentError refine_relpose(
            seed, withattrs(Csponds(chart1, f2); M₁, M₂); max_iter = 25, loss_width)
        gmsg = try
            refine_relpose(seed, withattrs(Csponds(chart1, f2); M₁, M₂); max_iter = 25, loss_width)
        catch e
            sprint(showerror, e)
        end
        @test occursin("source", gmsg) && occursin("backproject_with_jacobian", gmsg)
        # Targets are checked too, not just sources.
        chart2 = [SVector(f[1] / f[3], f[2] / f[3], 1.0) for f in f2]
        @test_throws ArgumentError refine_relpose(
            seed, withattrs(Csponds(f1, chart2); M₁, M₂); max_iter = 25, loss_width)
        # Genuine bearings succeed: the tolerance does not reject its own producer.
        @test refine_relpose(seed, cs; max_iter = 25, loss_width).R == bear.R
    end

    # The reason this entry point exists. The z = 1 chart cannot NAME a ray at
    # 90°: its coordinate there is (Inf, NaN). Point 1 below is exactly such a
    # ray, and 77 of the 150 bearings lie behind the camera (up to 173.6° off
    # axis). The bearing refiner pulls the seed from 1.146° to <0.05° through
    # it; handed the same geometry in the chart, the 2-vector sibling returns
    # the SEED UNTOUCHED — the NaN poisons the cost and LM never steps. That is
    # the failure an implementation which routed bearings through the chart
    # would have, and it is silent: the pose it returns is finite.
    @testset "refine_relpose (bearings) past 90°" begin
        rngf = MersenneTwister(11); rngn = MersenneTwister(13)
        m = 150
        Xs = [(v = normalize(SVector(randn(rngf), randn(rngf), randn(rngf)));
               (3.0 + 4rand(rngf)) * v) for _ in 1:m]
        Xs[1] = SVector(5.0, 0.0, 0.0)                     # bearing (1, 0, 0): z == 0
        Rg = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.15, normalize([0.2, 1.0, 0.4])...)))
        tg = normalize(SVector(0.6, -0.2, 0.3))
        # Orthonormal tangent basis ⇒ the local coordinate is arc length, so
        # here the loss width is in RADIANS. Same call, different unit — set by M, as the
        # docstring says.
        tb(f) = (b = s2_tangent_basis(SVector{3,Float64}(f));
                 SMatrix{3,2,Float64,6}(b[1][1], b[1][2], b[1][3], b[2][1], b[2][2], b[2][3]))
        jitter(f, σ) = (b = s2_tangent_basis(SVector{3,Float64}(f));
                        normalize(SVector{3,Float64}(f) + σ * (randn(rngn) * b[1] + randn(rngn) * b[2])))
        g1 = [normalize(Y) for Y in Xs]
        g2 = [jitter(normalize(Rg * Y + tg), deg2rad(0.02)) for Y in Xs]
        cs = withattrs(Csponds(g1, g2); M₁ = tb.(g1), M₂ = tb.(g2))

        @test g1[1][3] == 0.0                              # the chart has no coordinate here
        @test !all(isfinite, (g1[1][1] / g1[1][3], g1[1][2] / g1[1][3]))
        @test count(f -> f[3] <= 0, g1) > m ÷ 4            # the data really leaves the hemisphere

        Rs = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.02, 0.0, 1.0, 0.0))) * Rg
        seedf = PoseLib.Rigid3(RotMatrix{3}(Rs), Vec3{Float64}(normalize(tg + SVector(0.0, 0.03, 0.0))))
        ref = refine_relpose(seedf, cs; max_iter = 25, loss_width = deg2rad(0.5))
        @test _rot_err_deg(ref.R, Rg) < 0.05 < _rot_err_deg(seedf.R, Rg)
        @test dot(normalize(ref.t), tg) > 1 - 1e-6

        # The same geometry in the chart is now REFUSED rather than silently
        # mis-answered. Measured before the refusal landed, on exactly this
        # data: it returned a FINITE pose that was the seed to <1e-12 — the NaN
        # poisoned the cost and LM never took a step, so a caller saw a
        # plausible pose and no error at all. That is what the throw replaces.
        c1 = [SVector(f[1] / f[3], f[2] / f[3]) for f in g1]
        c2 = [SVector(f[1] / f[3], f[2] / f[3]) for f in g2]
        @test_throws ArgumentError refine_relpose(seedf, Csponds(c1, c2); max_iter = 25, loss_width = deg2rad(0.5))
    end

    # Camera-aware absolute pose: PIXEL points + K, loss width in PIXELS. The
    # residual is `project(camera, RX+t) − u`, exactly `focal ×` what the
    # withdrawn calibrated form measured — so the pixel width that truncates the
    # same points is `f ×` the calibrated one, and that factor is the whole
    # content of the unit claim.
    #
    # It is pinned by the two controls rather than by comparing against the
    # withdrawn method: a width handed over in CALIBRATED units under-truncates by
    # 900×, zero-weighting everything, and a K with its principal point dropped
    # mis-projects every point. Both leave the refiner standing on the seed, so
    # `< 1e-6` against the truth separates the right call from either mistake by
    # 4 orders. Without them, `loss_width` could be in any unit at all and the
    # accuracy assertion alone would still succeed.
    @testset "refine_absolute_pose (camera-aware, pixels)" begin
        rnga = MersenneTwister(104)
        f = (K1[1,1] + K1[2,2]) / 2
        xcal = [_nrm(R * Y + t) for Y in X]
        xcal[1:nout] .= [SVector(randn(rnga), randn(rnga)) for _ in 1:nout]
        upix = [SVector(f * c[1] + K1[1,3], f * c[2] + K1[2,3]) for c in xcal]
        calibrated_loss_width = 0.05

        pix = refine_absolute_pose(seed, K1, Csponds(X, upix); max_iter = 25, loss_width = f * calibrated_loss_width)
        @test _rot_err_deg(pix.R, R) < 1e-3
        @test norm(pix.t - t) < 1e-6

        wrong_unit = refine_absolute_pose(seed, K1, Csponds(X, upix); max_iter = 25, loss_width = calibrated_loss_width)
        @test norm(wrong_unit.t - t) > 1e-3          # stalls: the width is 900× too small
        Kbad = SMatrix{3,3}(f, 0, 0, 0, f, 0, 0.0, 0.0, 1.0)
        wrong_pp = refine_absolute_pose(seed, Kbad, Csponds(X, upix); max_iter = 25, loss_width = f * calibrated_loss_width)
        @test norm(wrong_pp.t - t) > 1e-3            # principal point is load-bearing
    end

    # Unit bearings, not (x/z, y/z, 1): the chart lifted to a 3-vector satisfies
    # these signatures, so the tests must not model the habit they exist to end.
    b1 = [normalize(Y) for Y in X]
    b2 = [normalize(R * Y + t) for Y in X]

    @testset "essential_dlt(cs)" begin
        E = VgtkCore.essential_dlt(Csponds(b1[1:20], b2[1:20]))
        @test E isa EssentialMat{Float64}
        @test _proj_dist(E, tx * R) < 1e-8
        @test VgtkCore.essential_dlt(PoseLibSolver(), b1[1:20] .=> b2[1:20]) == E
    end

    # The PoseLib backend of VGC's `Essential5pt`. VGC's `solve` hands the
    # kernel two `SVector{5}` of bearings; until this binding accepted that,
    # `solve(Essential5pt(PoseLibSolver()), cs, idx)` was a MethodError, and
    # `essential_5pt(PoseLibSolver(), cs, idx)` read a `.first` column a plain
    # `Vector{Pair}` does not have. Every route must give the same models, and
    # the true E must be among them.
    @testset "essential_5pt / Essential5pt (PoseLib backend)" begin
        idx = [3, 17, 42, 88, 150]                    # all inliers (> nout)
        csv = b1 .=> b2
        ref = VgtkCore.essential_5pt(PoseLibSolver(),
                  ntuple(k -> b1[idx[k]], Val(5)), ntuple(k -> b2[idx[k]], Val(5)))
        @test length(ref) >= 1
        @test minimum(E -> _proj_dist(E, tx * R), ref) < 1e-8
        for got in (VgtkCore.solve(Essential5pt(PoseLibSolver()), csv, idx),
                    VgtkCore.solve(Essential5pt(PoseLibSolver()), Csponds(b1, b2), idx),
                    VgtkCore.essential_5pt(PoseLibSolver(), csv, idx),
                    VgtkCore.essential_5pt(PoseLibSolver(), Csponds(b1, b2), idx))
            @test length(got) == length(ref) && all(collect(got) .== collect(ref))
        end
    end

    # Calibrated minimal kernels and the one-step E: a 2-vector endpoint is a
    # MethodError, as in VGC's own kernels — never lifted to z = 1. (No norm
    # check here: these are inner-loop pieces, and the epipolar and P3P algebra
    # is scale-invariant per ray; the guarantee comes from the producer.)
    @testset "calibrated kernels refuse 2-vectors" begin
        c1 = [_nrm(Y) for Y in X]
        c2 = [_nrm(R * Y + t) for Y in X]
        @test_throws MethodError VgtkCore.essential_5pt(PoseLibSolver(), c1 .=> c2, 1:5)
        @test_throws MethodError VgtkCore.essential_5pt(PoseLibSolver(), Csponds(c1, c2), 1:5)
        @test_throws MethodError VgtkCore.solve(Essential5pt(PoseLibSolver()), c1 .=> c2, 1:5)
        @test_throws MethodError VgtkCore.essential_dlt(PoseLibSolver(), Csponds(c1, c2))
        @test_throws MethodError VgtkCore.p3p(PoseLibSolver(), c1[1:3], X[1:3])
    end
    # -------------------------------------------------------------------------
    # The typed loss and its width. PoseLib builds every loss from one scalar
    # `loss_scale` (robust/robust_loss.h); the binding hands `loss_width` over
    # through UNCHANGED because, read off the C++, that scalar is exactly the
    # width w of VisualGeometryToolkitCore's closed forms: TruncatedLoss
    # ρ = min(r², w²), Huber's elbow at r = w, Cauchy w²·log1p(r²/w²). These
    # pin it behaviourally on a Euclidean homography (forward and backward
    # transfer residuals have equal length), so a factor, a square or a ½ at
    # the boundary fails:
    #   truncation at w, not w² nor w/2: outliers at 3 px are rejected at
    #     w = 2.5 (w² = 6.25 would keep them) and kept at w = 3.5 (w/2 = 1.75
    #     would reject them);
    #   Huber's elbow at w: residuals in [1.2, 1.5] are inside the quadratic
    #     core at w = 2 (an elbow at w/2 = 1 would leave it), and outliers at
    #     3 px are past it (an elbow at w² = 4 would not be);
    #   Cauchy's width is a SCALE: w → ∞ is least squares.
    # -------------------------------------------------------------------------
    @testset "typed loss: the width crosses unchanged" begin
        rngw = MersenneTwister(211)
        θ = 0.1
        He = SMatrix{3,3}(cos(θ), sin(θ), 0.0, -sin(θ), cos(θ), 0.0, 12.0, -7.0, 1.0)
        m = 120; mout = 20
        y1 = [SVector(640 * rand(rngw), 480 * rand(rngw)) for _ in 1:m]
        exact = [_proj(He, SVector(u[1], u[2], 1.0)) for u in y1]
        H0 = He + SMatrix{3,3}(0.0, 0, 0, 0, 0, 0, 0.05, -0.05, 0)
        offset = SVector(3.0, 0.0)
        with_outliers = copy(exact); with_outliers[1:mout] .+= Ref(offset)
        cs_out = Csponds(y1, with_outliers)
        fit(cs, loss, w) = refine_homography(H0, cs; max_iter = 100, loss, loss_width = w)

        # TLSShape: a truncation at w.
        @test _proj_dist(fit(cs_out, TLSShape(), 2.5), He) < 1e-8
        @test _proj_dist(fit(cs_out, TLSShape(), 3.5), He) > 1e-5
        # Le–Zach optimizes the same truncation: the same width decides. (At
        # w = 2.5 its TR-IRLS returns the seed on this data, a property of the
        # scheme rather than of the width, so it is checked at w = 1, which
        # also rejects the 3 px outliers, and at w = 3.5, which keeps them
        # where a width of w/2 would not.)
        @test _proj_dist(fit(cs_out, LeZachTruncation(), 1.0), He) < 1e-6
        @test _proj_dist(fit(cs_out, LeZachTruncation(), 3.5), He) > 1e-2

        # HuberShape: the elbow at w.
        noisy = [x + (1.2 + 0.3 * rand(rngw)) * (a = 2π * rand(rngw); SVector(cos(a), sin(a))) for x in exact]
        l2 = refine_homography(H0, Csponds(y1, noisy); max_iter = 100, loss = GaussianShape())
        @test _proj_dist(fit(Csponds(y1, noisy), HuberShape(), 2.0), l2) < 1e-12
        noisy_out = copy(noisy); noisy_out[1:mout] .+= Ref(offset)
        l2_out = refine_homography(H0, Csponds(y1, noisy_out); max_iter = 100, loss = GaussianShape())
        @test _proj_dist(fit(Csponds(y1, noisy_out), HuberShape(), 2.0), l2_out) > 1e-6
        @test _proj_dist(fit(Csponds(y1, noisy_out), HuberShape(), 6.0), l2_out) < 1e-12

        # CauchyShape: the width is a scale — wide is least squares, narrow is not.
        @test _proj_dist(fit(Csponds(y1, noisy_out), CauchyShape(), 1.0e6), l2_out) < 1e-9
        @test _proj_dist(fit(Csponds(y1, noisy_out), CauchyShape(), 2.0), l2_out) > 1e-6

        # The Gaussian loss has no width; every other loss needs one; the LM
        # budget is the caller's configuration, never a literal default.
        @test_throws ArgumentError refine_homography(H0, cs_out; max_iter = 10, loss = GaussianShape(), loss_width = 2.0)
        @test_throws ArgumentError refine_homography(H0, cs_out; max_iter = 10)
        @test_throws UndefKeywordError refine_homography(H0, cs_out; loss_width = 2.0)
        # The integer loss constants are retired, not aliased.
        @test !isdefined(PoseLib, Symbol("LOSS" * "_TRUNCATED"))
        @test_throws MethodError refine_homography(H0, cs_out; max_iter = 10, loss = 1, loss_width = 2.0)
    end
end

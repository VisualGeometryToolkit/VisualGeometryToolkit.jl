# The coincidence gate, pinned at the CHECK rather than at the gate's verdict.
#
# Why that distinction is the whole point: before the unit fix, homography's
# full gate read 98.6% duplicate capture at 200 px spread while its coincidence
# check was at 6.2% — `_not_collinear` was carrying it incidentally, because two
# near-identical points make every triangle through them thin. Asserting the
# VERDICT would therefore have succeeded for the wrong reason and pinned nothing.
# So: fat triangles, so collinearity cannot rescue the sample, and the
# coincidence predicate asserted directly.
#
# The old gate reduces EXACTLY to a pixel threshold, which is why the historical
# figures quoted here are computed rather than sampled: Hartley's translation is
# distance-invariant and its scale is uniform, so a normalized distance is
# `similarity_scale` times the pixel distance, and the old `1e-3` was therefore
# `t_old = 1e-3 / similarity_scale` pixels — verified to 5.2e-18 by RVG.
#
# Duplicate offset is 0.06 px throughout: VisualGeometryFeatures' pinned
# worst-case sub-pixel refinement error over σ ∈ [2,8]. A duplicate of the SAME
# feature cannot differ by more than that. The flattering 0.03 px was used
# earlier and hid a third of the defect.
@testitem "coincidence_gate" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryCore
    using VisualGeometryCore: _are_points_distinct, _points_distinct_idx,
        _not_collinear, COINCIDENT_PX

    DUP = 0.06          # VGF's pinned worst-case refinement error, px
    FAR = 1.0           # where genuinely distinct features live, px

    @testset "every gate FORWARDS tol_px — a dropped keyword is invisible" begin
        # This pins a bug that shipped in adca0b9: E's no-attribute fallback
        # called the helpers without forwarding `tol_px`, so all four of these
        # returned the same answer across four orders of magnitude of requested
        # tolerance. A dropped keyword produces no error and no warning — the
        # gate simply ignores you — so it needs a test that varies the value and
        # demands the verdict move with it.
        using VisualGeometryCore: HomographyMat, FundMat, EssentialMat, test_sample, Csponds
        pts = [SVector(100.0, 100.0), SVector(100.03, 100.0),      # 0.03 px apart
               SVector(300.0, 140.0), SVector(180.0, 320.0),
               SVector(240.0, 260.0), SVector(150.0, 210.0), SVector(275.0, 305.0)]
        cs = Csponds(pts, pts)
        # F and E through the full gate: coincidence IS their whole gate, so the
        # verdict is the check.
        for (M, k) in ((FundMat, 7), (EssentialMat, 5))
            idx = collect(1:k)
            @test !test_sample(M, cs, idx; tol_px = 0.1)      # 0.03 px → coincident
            @test test_sample(M, cs, idx; tol_px = 1.0e-3)    # → distinct
            @test !test_sample(M, cs, idx; tol_px = 10.0)     # verdict tracks the value
        end
        # Homography through the HELPER, not the gate. Its `test_sample` composes
        # coincidence with `_not_collinear`, and a 0.03 px pair makes every
        # triangle through it thin at this spread — so the gate rejects whatever
        # `tol_px` says, and asserting the verdict here would pin collinearity
        # while claiming to pin forwarding. (An earlier version of this test did
        # exactly that and failed for the right reason.)
        h = (pts[1], pts[2], pts[3], pts[4])
        @test !_are_points_distinct(h; tol_px = 0.1)
        @test _are_points_distinct(h; tol_px = 1.0e-3)
        @test !_are_points_distinct(h; tol_px = 10.0)
        # …and that H's gate does forward, shown where collinearity admits: a
        # clustered sample whose triangles stay fat enough to succeed.
        c4 = (SVector(320.0, 320.0), SVector(320.03, 320.0),
              SVector(328.0, 322.0), SVector(323.0, 330.0))
        if _not_collinear(c4)
            @test !test_sample(HomographyMat, Csponds(collect(c4), collect(c4)), 1:4; tol_px = 0.1)
            @test test_sample(HomographyMat, Csponds(collect(c4), collect(c4)), 1:4; tol_px = 1.0e-3)
        end
    end

    @testset "the tolerance is in pixels and is the same for both arities" begin
        @test COINCIDENT_PX == 0.1
        # The two helpers are the same check for different sample sizes. Calibrating
        # one without the other is the bug this pair used to have, so pin that they
        # agree on identical data.
        a = SVector(100.0, 100.0)
        for d in (0.0, DUP, 0.099, 0.101, FAR)
            b = a + SVector(d, 0.0)
            tup = (a, b, SVector(300.0, 140.0), SVector(180.0, 320.0))
            vec = collect(tup)
            @test _are_points_distinct(tup) == _points_distinct_idx(vec, 1:4)
        end
    end

    @testset "coincidence is SPREAD-INVARIANT; collinearity is not" begin
        # This is the defect in one assertion. `thin_ratio` is a RATIO, so a
        # fixed 0.06 px duplicate looks relatively larger as the data clusters
        # and eventually stops looking collinear — the old gate's protection
        # decayed exactly where it was needed. The pixel coincidence check has
        # no such dependence.
        #
        # MEASURED FROM TWO CONSTRUCTIONS THAT BRACKET THE CURVE. Collinearity's
        # incidental duplicate-catching, as a fraction ADMITTED:
        #
        #   spread   random samples   equilateral (this one)
        #   640 px        0%                   0%
        #   200 px        0.8%                 0%
        #    60 px       41.8%                 0%
        #    20 px       82%                  58%
        #
        # Same shape, identical at wide spread, differing when clustered purely
        # by triangle size — which is the signature of a real effect rather than
        # an artifact of either setup. Do not assert that collinearity "cannot"
        # admit: at wide spread it cannot, at clustered spread it routinely
        # does, and an earlier version of this comment claimed the former
        # universally on the strength of a count misread from truncated test
        # output.
        rng = Xoshiro(11)
        coll_by_spread = Int[]
        for spread in (640.0, 200.0, 60.0, 20.0)
            caught = 0
            coll = 0
            n = 400
            for _ in 1:n
                base = [SVector(320.0, 320.0) + spread * SVector(cosd(θ), sind(θ)) / 2
                        for θ in (0.0, 120.0, 240.0)]
                φ = 2π * rand(rng)
                dup = base[1] + DUP * SVector(cos(φ), sin(φ))
                pts = (base[1], dup, base[2], base[3])
                _not_collinear(pts) || (coll += 1)          # collinearity rejected it
                _are_points_distinct(pts) || (caught += 1)  # coincidence rejected it
            end
            @test caught == n                # INVARIANT: every duplicate, every spread
            push!(coll_by_spread, coll)
        end
        # Collinearity's capture is NOT invariant: total at the widest spread,
        # and it falls away as the sample clusters. That decay is the defect the
        # unit fix removes the dependence on.
        @test coll_by_spread[1] == 400                       # 640 px: catches all, incidentally
        @test coll_by_spread[end] < coll_by_spread[1]        # 20 px: strictly fewer
    end

    @testset "genuinely distinct features are admitted at every spread" begin
        rng = Xoshiro(12)
        for spread in (640.0, 200.0, 60.0, 20.0)
            admitted = 0
            n = 400
            for _ in 1:n
                base = [SVector(320.0, 320.0) + spread * SVector(cosd(θ), sind(θ)) / 2
                        for θ in (0.0, 120.0, 240.0)]
                φ = 2π * rand(rng)
                near = base[1] + FAR * SVector(cos(φ), sin(φ))     # 1 px away: distinct
                pts = (base[1], near, base[2], base[3])
                _are_points_distinct(pts) && (admitted += 1)
            end
            @test admitted == n          # exact: 1 px > 0.1 px, never rejected
        end
    end

    @testset "the boundary is the tolerance itself, and it does not move with spread" begin
        # The defect was that the effective threshold tracked the sample's own
        # spread. Pin that it no longer does: the same pair decides the same way
        # in a 640 px sample and a 20 px one.
        for spread in (640.0, 20.0)
            far = SVector(320.0, 320.0) + SVector(spread, 0.0) / 2
            a = SVector(320.0, 320.0)
            just_in = (a, a + SVector(0.0999, 0.0), far, far + SVector(0.0, spread / 3))
            just_out = (a, a + SVector(0.1001, 0.0), far, far + SVector(0.0, spread / 3))
            @test !_are_points_distinct(just_in)     # inside 0.1 px → coincident
            @test _are_points_distinct(just_out)     # outside → distinct
        end
    end

    @testset "the OLD constant is wrong in the pixel domain too" begin
        # The third independent angle, and the one that closes the argument.
        # 1e-3 was miscalibrated in the normalized domain (0% capture from 60 px
        # spread down). It is not merely miscalibrated in PIXELS — it is inert:
        # a 0.06 px duplicate is 60× the old tolerance, so nothing is ever
        # coincident. Together with the inverse failure (the new 0.1 px
        # tolerance meeting normalized coordinates gives a 17 px effective
        # threshold), the constant has no frame in which it is defensible.
        rng = Xoshiro(14)
        for spread in (640.0, 200.0, 60.0, 20.0)
            caught_old = 0
            n = 200
            for _ in 1:n
                base = [SVector(320.0, 320.0) + spread * SVector(cosd(θ), sind(θ)) / 2
                        for θ in (0.0, 120.0, 240.0)]
                φ = 2π * rand(rng)
                pts = (base[1], base[1] + DUP * SVector(cos(φ), sin(φ)), base[2], base[3])
                _are_points_distinct(pts; tol_px = 1.0e-3) || (caught_old += 1)
            end
            @test caught_old == 0          # the old constant catches NOTHING in pixels
        end
    end

    @testset "E on bearings: a pixel tolerance through the Jacobian" begin
        # The E gate never sees pixels — it sees directions. The claim under test
        # is that a duplicate DETECTION (0.06 px apart in the image) is still
        # rejected after that round trip, and that a genuinely distinct pair
        # (1 px) still is not.
        using VisualGeometryCore: CameraModel, LogicalIntrinsics, CameraCalibrationMat,
            Pinhole, KannalaBrandt, backproject_with_jacobian, Csponds, withattrs,
            EssentialMat, test_sample
        f = 800.0; c = 320.0
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(f, 0, 0, 0, f, 0, c, c, 1))
        rng = Xoshiro(21)
        for cam in (CameraModel(LogicalIntrinsics(K), Pinhole()),
                CameraModel(LogicalIntrinsics(K), KannalaBrandt(-0.02)))
            for (offset, expect_admit) in ((DUP, false), (FAR, true))
                admits = 0
                n = 60
                for _ in 1:n
                    us = [SVector(120 + 380rand(rng), 120 + 380rand(rng)) for _ in 1:5]
                    φ = 2π * rand(rng)
                    us[3] = us[2] + offset * SVector(cos(φ), sin(φ))
                    bm = [backproject_with_jacobian(cam, u) for u in us]
                    fs = [b[1] for b in bm]; Ms = [b[2] for b in bm]
                    cs = withattrs(Csponds(fs, fs); M₁ = Ms, M₂ = Ms)
                    test_sample(EssentialMat, cs, 1:5) && (admits += 1)
                end
                # exact: the conversion is deterministic given the camera
                @test admits == (expect_admit ? n : 0)
            end
        end
    end

    @testset "F and E carry the whole gate: no collinearity check exists for them" begin
        # Unlike homography, the epipolar families have nothing else, so the
        # coincidence check is the entire pre-solve gate — which is why its
        # being inoperative admitted duplicates straight into the solver.
        rng = Xoshiro(13)
        for (K, spread) in ((7, 60.0), (5, 60.0), (7, 20.0), (5, 20.0))
            caught = 0
            n = 300
            for _ in 1:n
                pts = [SVector(320 + spread * (rand(rng) - 0.5), 320 + spread * (rand(rng) - 0.5))
                       for _ in 1:K]
                φ = 2π * rand(rng)
                pts[3] = pts[2] + DUP * SVector(cos(φ), sin(φ))
                _points_distinct_idx(pts, 1:K) || (caught += 1)
            end
            @test caught == n
        end
    end

    @testset "E without M columns: the pixel gate, on any container" begin
        using VisualGeometryCore: Csponds, EssentialMat, FundMat, test_sample
        us = [SVector(100.0, 100.0), SVector(200.0, 150.0), SVector(300.0, 260.0),
            SVector(140.0, 330.0), SVector(260.0, 90.0)]
        vs = us .+ Ref(SVector(3.0, -2.0))
        dup = copy(us); dup[3] = dup[2] + SVector(DUP, 0.0)
        for (a, expect) in ((us, true), (dup, false))
            pairs = [x => y for (x, y) in zip(a, vs)]
            # a plain Pair vector carries no columns: the same answer as Csponds
            @test test_sample(EssentialMat, pairs, 1:5) == expect
            @test test_sample(EssentialMat, Csponds(a, vs), 1:5) == expect
            @test test_sample(FundMat, pairs, 1:5) == expect
        end
    end
end

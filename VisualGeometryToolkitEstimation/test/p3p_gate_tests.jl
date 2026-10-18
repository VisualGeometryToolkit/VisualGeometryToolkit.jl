@testitem "p3p_gate" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitEstimation

    const VGC = VisualGeometryToolkitCore
    const RVG = VisualGeometryToolkitEstimation

    # =========================================================================
    # Drift pin for P3P's coincidence tolerances
    # =========================================================================
    #
    # `test_sample(::AbsolutePoseProblem, idx)` calls VGC's shared
    # `_are_points_distinct` in TWO domains that are not pixels:
    #
    #   X — world control points, in the CALLER's units (m, mm, …)
    #   u — image points, in pixels (measured: `targets(p.cs)` are the pixel
    #       values supplied; the unit bearings live in `p.cs_solve`)
    #
    # B7 changed that helper's default from 1e-3 to `COINCIDENT_PX = 0.1`, a
    # pixel deduplication threshold. P3P had been riding the default, so it
    # silently acquired a tolerance calibrated for another domain: 0.1 read as
    # metres is a 10 cm coincidence radius, a hundredfold loosening. Both
    # tolerances are now supplied BY NAME at their pre-B7 values.
    #
    # This pins that. A dropped or un-forwarded keyword produces no error — it
    # just answers differently — so only a test that demands the verdict track
    # the constant will catch it. That is exactly how the B7 regression got in:
    # the E gate accepted `tol_px` and did not forward it in one branch, and
    # 1e-3, the default and 10.0 all decided alike.

    # A near-duplicate pair makes every triangle through it thin, so the
    # collinearity checks SHADOW the coincidence checks — the same structure
    # found in the homography gate, where `_not_collinear` had been catching
    # duplicates incidentally while the coincidence check sat inoperative.
    # Measured here with the third point at 0.3 m: at a world separation of
    # 2e-3 the coincidence check admits while `thin_ratio` rejects, so the
    # coincidence verdict is never the deciding one.
    #
    # Bringing the third point to 0.04 m keeps `thin_ratio` at 0.0125-0.0499,
    # clear of its 1e-2 bound, while the coincidence check still flips across
    # 1e-3. That isolates the path under test, exactly as the epipolar
    # regression cell keeps its triangles fat.
    const THIRD = 0.04

    function p3p_sample(; world_sep, img_sep)
        cam = VGC.CameraModel(800VGC.px, SA[320.0VGC.px, 240.0VGC.px])
        X1 = SA[0.0, 0.0, 2.0]
        X2 = X1 + SA[world_sep, 0.0, 0.0]
        X3 = SA[0.0, THIRD, 2.0]
        u1 = VGC.project(cam, X1)
        u3 = VGC.project(cam, X3)
        u2 = u1 + VGC.Point2(img_sep, 0.0)
        pairs = [
            VGC.Point3(X1) => u1,
            VGC.Point3(X2) => u2,
            VGC.Point3(X3) => u3,
        ]
        return RVG.AbsolutePoseProblem(cam, pairs)
    end

    @testset "world tolerance is honoured, and is the deciding check" begin
        # Below the world tolerance: rejected. The other three checks succeed at
        # this geometry, so the rejection can only come from world coincidence.
        p_near = p3p_sample(world_sep = 5.0e-4, img_sep = 0.2)
        @test !RVG.test_sample(p_near, [1, 2, 3])

        # Above it: admitted. THIS is the assertion that catches a reverted or
        # un-forwarded tolerance — if P3P falls back to COINCIDENT_PX = 0.1,
        # a 2e-3 m separation is a hundredfold inside it and this flips to
        # rejected. It is the whole point of the pin.
        p_far = p3p_sample(world_sep = 2.0e-3, img_sep = 0.8)
        @test RVG.test_sample(p_far, [1, 2, 3])
    end

    @testset "the world constant is the one the gate actually uses" begin
        # Straddle the CONSTANT rather than a literal, so the pin follows the
        # value if it is ever deliberately changed, and fails if the gate stops
        # reading it.
        coincident = RVG._P3P_COINCIDENT_WORLD
        @test !RVG.test_sample(p3p_sample(world_sep = 0.5coincident, img_sep = 0.2), [1, 2, 3])
        @test RVG.test_sample(p3p_sample(world_sep = 4.0coincident, img_sep = 0.8), [1, 2, 3])
    end

    @testset "a distorting lens: coplanar RAYS, not collinear pixels, are the degeneracy" begin
        K = VGC.CameraCalibrationMat(950.0VGC.px, [2120.0VGC.px, 1416.0VGC.px])
        cam = VGC.CameraModel(VGC.LogicalIntrinsics(K), VGC.Division(-0.3, -0.08))
        # three points on the plane y = z/2 through the centre: their rays are
        # coplanar, a P3P degeneracy, but the lens bends their line into a curve
        Xs = (SA[-1.0, 0.5, 1.0], SA[0.0, 1.0, 2.0], SA[1.5, 0.5, 1.0])
        u = map(X -> VGC.Point2{Float64}(VGC.project(cam, X)...), Xs)
        @test VGC.thin_ratio(u...) > RVG._P3P_DEGENERACY_TOL           # the pixels alone look healthy
        coplanar = RVG.AbsolutePoseProblem(cam, [VGC.Point3(Xs[i]) => u[i] for i in 1:3])
        @test !RVG.test_sample(coplanar, [1, 2, 3])                       # the rays decide
        # lift the middle point off the plane: a genuine triangle of rays is admitted
        X2 = SA[0.0, 1.6, 2.0]
        u2 = VGC.Point2{Float64}(VGC.project(cam, X2)...)
        spread = RVG.AbsolutePoseProblem(cam, [VGC.Point3(Xs[1]) => u[1], VGC.Point3(X2) => u2, VGC.Point3(Xs[3]) => u[3]])
        @test RVG.test_sample(spread, [1, 2, 3])
        # a Pinhole keeps the pixel test, bit for bit (collinear pixels ⟺ coplanar rays)
        @test RVG.test_sample(p3p_sample(world_sep = 2.0e-3, img_sep = 0.8), [1, 2, 3])
    end

    # NO PIN FOR THE IMAGE TOLERANCE, deliberately — it is unreachable.
    #
    # `_P3P_COINCIDENT_IMAGE` is 1e-3 px, about sixty times below VGF's pinned
    # 0.06 px worst-case sub-pixel refinement bound, so it cannot separate a
    # duplicate from a re-detection of the same feature. Worse, it is fully
    # shadowed: measured, at any image separation small enough for the
    # coincidence check to fire (<1e-3 px) the image `thin_ratio` is already
    # 3.1e-5, far under its 1e-2 bound, so collinearity has rejected the sample
    # first. At 0.2 px, where thin_ratio is healthy at 0.0125, coincidence
    # admits. There is no separation at which the image coincidence verdict
    # decides anything.
    #
    # So there is nothing to pin: any test would be asserting the collinearity
    # check's behaviour while appearing to assert coincidence — the same trap
    # that made the homography attribution wrong by a factor of twenty. The
    # value needs calibrating before it can be pinned, and that is its own
    # decision: P3P's image check guards a different degeneracy from the
    # epipolar backstop (two DISTINCT world points projecting together, forcing
    # a grazing pose), so 0.1 px is not automatically the right number either.
end

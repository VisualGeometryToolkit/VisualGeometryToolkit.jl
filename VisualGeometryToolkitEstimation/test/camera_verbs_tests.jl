# =============================================================================
# The camera-from-camera verbs (0.9): CameraRegression is linear (no guess),
# the camera-regression refine polishes a guess by the chart LM, the fit-then-refine recipe is the two
# =============================================================================

@testitem "camera_verbs: the rational linear fit of icra2026's fisheye lands at OpenCV's calibration (failure A)" setup = [CameraRecipes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Lens, Perspective, Additive, Radial, Tangential, ThinPrism, Rect, Ellipse, calibration_matrix, parameters, rms, lens, view_angle
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # The fisheye of icra2026's mocap captures (a 180° lens on a 4256 × 2848 sensor): its Kannala–Brandt calibration
    # from the blob capture (run `ncoef=2_subset=blob_mount_selfcal_train_cd9c4edf`, camera_mocap.json), and OpenCV's
    # rational calibration of the same lens from the AprilGrid capture (`model=rational_origin=opencv_subset=april…`):
    # f = 921.9 px, pp (2140.1, 1441.4), radial (0.346, 0.0101, 1.1e-5 | 0.696, 0.0614, 4.1e-4).
    kb = CameraModel(; pp = (2138.8425928470033, 1436.604235274649), focal = 927.4663595958699,
        lens = KannalaBrandt(-0.03371588391401308, 0.0013137235024557887, -0.0019529571546981683, 0.0001827493332399132))
    RAT = typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0), ThinPrism((0.0, 0.0), (0.0, 0.0)))))
    region = Rect(0, 0, 4256, 2848) ∩ Ellipse(SVector(2138.8425928470033, 1436.604235274649), 1250.0, 1250.0, 0.0)   # to 84° off axis
    # 0.8's chart LM from the zero-coefficient start stalled at the pinhole here (f 927.4 px, |k| < 1e-3, 6008 px at
    # the edge; the bundle then collapsed the focal to 25–184 px). The linear fit leaves it in one solve:
    r = solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region)), CameraRegression())
    @test retcode_symbol(r) === :solved
    K = calibration_matrix(r.u)
    @test abs(K[1, 1] - 921.9) < 10                                                    # px: OpenCV's focal (925.3)
    @test hypot(K[1, 3] - 2140.0957668437163, K[2, 3] - 1441.4063814989743) < 6        # px: the KB's principal point, kept
    c = parameters(lens(r.u))
    @test 0.2 < c.k1 < 0.6 && 0.4 < c.d1 < 1.0 && c.d2 > 0.01                           # far from zero, OpenCV's signs and sizes
    @test c.p1 == 0 && c.sx1 == 0                                                       # decentering: zero, not seen by the radial relation
    @test r.metadata.max_px < 1 && r.rms < 0.25                                       # px
    # a field past 90° off axis (1400 px; the lens images 90° at ≈ 1315 px): the rational target applies its distortion
    # on the perspective plane, which ends at 90°, so it is refused, naming the plane and the region's extent — never
    # fitted over a clipped field; an angle-parametrised family over the same region is fitted
    wide = Rect(0, 0, 4256, 2848) ∩ Ellipse(SVector(2138.8425928470033, 1436.604235274649), 1400.0, 1400.0, 0.0)
    @test_throws VgtkEstimation.ViewAngleLimitError solve(EstimationProblem(CameraRegressionProblem(RAT, kb, wide)), CameraRegression())
    @test_throws r"perspective plane" solve(EstimationProblem(CameraRegressionProblem(RAT, kb, wide)), CameraRegression())
    @test_throws r"Use a lift \(division\) model" solve(EstimationProblem(CameraRegressionProblem(RAT, kb, wide)), CameraRegression(; refine = LevenbergMarquardt()))
    # the fields: the region's angle about the source's axis (the KB profile's at 1400 px, radians) and the limit
    e = try solve(EstimationProblem(CameraRegressionProblem(RAT, kb, wide)), CameraRegression(; refine = LevenbergMarquardt())); nothing catch err; err end
    @test e isa VgtkEstimation.ViewAngleLimitError && e.family === RAT && e.limit == π / 2 - 1.0e-3
    @test e.field_angle ≈ Float64(view_angle(lens(kb), 1400.0 / 927.4663595958699)) rtol = 1.0e-6
    @test rad2deg(e.field_angle) > 90
    rk = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, kb, wide)), CameraRegression())
    @test retcode_symbol(rk) === :solved && rk.metadata.family.θ_source > π / 2 && rk.metadata.max_px < 1.0e-6
    # and the refine from it converges near it
    ra = solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region)), CameraRegression(; refine = LevenbergMarquardt()))
    @test retcode_symbol(ra) === :converged && abs(calibration_matrix(ra.u)[1, 1] - 921.9) < 10 && ra.rms <= r.rms
end

@testitem "camera_verbs: CameraRegression's linear residual is the returned camera's, and records no scale" setup = [CameraRecipes, FitCameraLensCases] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: rms, lens, calibration_matrix, projected_radius, project, noise_scale, scale_source, ScaleNone, Division, weights, weight_kind, BinaryWeights, inlier_mask
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, CameraFitMetadata
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    for L in (KB4, SC3, Division{2})
        region = disc(mdiv, 2000.0)
        r = solve(EstimationProblem(CameraRegressionProblem(L, mdiv, region)), CameraRegression())
        @test retcode_symbol(r) === :solved && r.stats.nsteps == 0
        m = r.metadata
        @test length(m.residuals) == 129 && maximum(m.residuals) == m.max_px
        # recomputed from the returned camera: the source's ray at each profile node (Clenshaw–Curtis on [0, θ_max]),
        # projected, against the source's pixel of it
        cam = r.u
        K0 = calibration_matrix(mdiv)
        θs = first(VgtkEstimation._clenshaw_curtis(VgtkEstimation._PROFILE_NODES, m.family.θ_max))
        e = map(θs) do θ
            u_src = SVector(K0[1, 3], K0[2, 3]) + K0[1, 1] * SVector(projected_radius(lens(mdiv), θ), 0.0)
            norm(project(cam, SVector(sin(θ), 0.0, cos(θ))) - u_src)
        end
        @test maximum(abs.(e .- m.residuals)) <= 1.0e-6
        @test scale_source(r) === ScaleNone && isnan(m.scale)
        @test_throws ArgumentError noise_scale(r)
        @test weights(r) == ones(length(m.residuals)) && weight_kind(r) === BinaryWeights()   # L2: unit weights
        @test all(inlier_mask(r))
        rr = solve(RefineProblem(CameraRegressionProblem(mdiv, region), r.u), LevenbergMarquardt())
        @test scale_source(rr) === ScaleNone
        @test_throws ArgumentError noise_scale(rr)
    end
    # KB from a physical source: the polynomial's linear fit IS the integral's least squares, the refine's minimum
    r = solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, disc(mdiv, rcorner))), CameraRegression())
    ra = solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, disc(mdiv, rcorner))), CameraRegression(; refine = LevenbergMarquardt()))
    @test abs(r.rms - ra.rms) <= 1.0e-6 * ra.rms
end

@testitem "camera_verbs: a family with no linear form is refused, naming the camera-regression refine; the fit-then-refine recipe starts it from its anchor" setup = [CameraRecipes, FitCameraLensCases] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using VisualGeometryToolkitCore: Lens, Perspective, Equidistant, Tilt, FieldOfView, Radial, Unified, ExtendedUnified, DoubleSphere
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    for L in (typeof(Lens(DoubleSphere(0.5, 0.5))), typeof(Lens(Unified(0.0))), typeof(Lens(ExtendedUnified(0.0, 0.0))), typeof(Lens(Perspective(), FieldOfView(0.0))),
              typeof(Lens(Perspective(), Tilt(0.0, 0.0) ∘ Radial((0.0, 0.0)))))
        @test_throws r"RefineProblem\(CameraRegressionProblem" solve(EstimationProblem(CameraRegressionProblem(L, mdiv, disc(mdiv, 1500.0))), CameraRegression())
        @test_throws r"the family's anchor" solve(EstimationProblem(CameraRegressionProblem(L, mdiv, disc(mdiv, 1500.0))), CameraRegression())
    end
    r = solve(EstimationProblem(CameraRegressionProblem(typeof(Lens(DoubleSphere(0.5, 0.5))), mdiv, disc(mdiv, 1500.0))), CameraRegression(; refine = LevenbergMarquardt()))
    @test retcode_symbol(r) === :converged
    # the refine's keywords: L2 only, no scale input
    g = solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, disc(mdiv, 1500.0))), CameraRegression()).u
    @test_throws ArgumentError solve(RefineProblem(CameraRegressionProblem(mdiv, disc(mdiv, 1500.0)), g), LevenbergMarquardt(); scale = 1.0)
    @test_throws ArgumentError solve(RefineProblem(CameraRegressionProblem(mdiv, disc(mdiv, 1500.0)), g), LevenbergMarquardt(); width = 1.0)
    @test_throws ArgumentError solve(RefineProblem(CameraRegressionProblem(mdiv, disc(mdiv, 1500.0)), g), LevenbergMarquardt(; optimization_strategy = :huber))
    @test_throws MethodError solve(RefineProblem(CameraRegressionProblem(mdiv, disc(mdiv, 1500.0)), g), LevenbergMarquardt(; local_optimization = nothing))   # the retired keyword
end

@testitem "camera_verbs: the camera-regression refine from a perturbed guess reaches the fit-then-refine recipe's minimum" setup = [CameraRecipes, FitCameraLensCases] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SMatrix
    using VisualGeometryToolkitCore: CameraModel, LogicalIntrinsics, CameraCalibrationMat, lens, calibration_matrix, parameter_vector, with_parameters, rms, Division
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    for (L, R) in ((KB4, rcorner), (SC3, rcorner), (Division{2}, 2500.0))
        region = disc(mdiv, R)
        a = solve(EstimationProblem(CameraRegressionProblem(L, mdiv, region)), CameraRegression(; refine = LevenbergMarquardt()))
        best = a.u
        p = collect(parameter_vector(lens(best)))
        K = SMatrix{3, 3, Float64}(calibration_matrix(best))
        guess = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1] * 1.05, 0, 0, 0, K[2, 2] * 1.05, 0, K[1, 3], K[2, 3], 1); check = false)),
            with_parameters(lens(best), p .* 0.8))
        r = solve(RefineProblem(CameraRegressionProblem(mdiv, region), guess), LevenbergMarquardt())
        @test retcode_symbol(r) === :converged
        @test maximum(abs.(collect(parameter_vector(lens(r.u))) .- p)) < 1.0e-5
        @test abs(calibration_matrix(r.u)[1, 1] - K[1, 1]) < 1.0e-5 * K[1, 1]
        @test abs(r.rms - a.rms) <= 1.0e-6 * max(a.rms, 1.0e-9) + 1.0e-12
    end
    # a physical source's principal point is data: a guess that moves it is refused
    g = solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, disc(mdiv, 1500.0))), CameraRegression(; refine = LevenbergMarquardt())).u
    K = SMatrix{3, 3, Float64}(calibration_matrix(g))
    moved = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0, 0, 0, K[2, 2], 0, K[1, 3] + 3, K[2, 3], 1); check = false)), lens(g))
    @test_throws r"principal point" solve(RefineProblem(CameraRegressionProblem(mdiv, disc(mdiv, 1500.0)), moved), LevenbergMarquardt())
end

@testitem "camera_verbs: the camera-regression refine's `fixed` holds parameters by name or group; the default frees the rational denominator" setup = [CameraRecipes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Lens, Perspective, Radial, Even, Ellipse, Rect, calibration_matrix, parameters, parameter_names, lens, rms
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    kb = CameraModel(; pp = (2138.8425928470033, 1436.604235274649), focal = 927.4663595958699,
        lens = KannalaBrandt(-0.03371588391401308, 0.0013137235024557887, -0.0019529571546981683, 0.0001827493332399132))
    RAT = Lens{Perspective, Radial{Even, 3, 3, Float64}}
    region = Rect(0, 0, 4256, 2848) ∩ Ellipse(SVector(2138.8425928470033, 1436.604235274649), 1250.0, 1250.0, 0.0)
    g = solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region)), CameraRegression()).u
    cg = parameters(lens(g))
    # the denominator group held: it keeps the guess's values exactly, the numerator and the focal move — the
    # denominator-held refine (0.8's free set, here from the linear start); the group is its names
    rh = solve(RefineProblem(CameraRegressionProblem(kb, region; fixed = (:d,)), g), LevenbergMarquardt())
    ch = parameters(lens(rh.u))
    @test retcode_symbol(rh) === :converged
    @test (ch.d1, ch.d2, ch.d3) === (cg.d1, cg.d2, cg.d3)
    @test ch.k1 != cg.k1 && calibration_matrix(rh.u)[1, 1] != calibration_matrix(g)[1, 1]
    @test lens(solve(RefineProblem(CameraRegressionProblem(kb, region; fixed = (:d1, :d2, :d3)), g), LevenbergMarquardt()).u) == lens(rh.u)
    # the default (`fixed = ()`) frees it, to a lower minimum; the fit-then-refine recipe forwards `fixed`
    rd = solve(RefineProblem(CameraRegressionProblem(kb, region), g), LevenbergMarquardt())
    cd = parameters(lens(rd.u))
    @test cd.d1 != cg.d1 && rd.rms < rh.rms
    @test solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region)), CameraRegression(; refine = LevenbergMarquardt())).rms == rd.rms
    # the fit-then-refine recipe forwards `fixed` to both stages: the linear start holds the denominator too
    @test lens(solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region; fixed = (:d,))), CameraRegression(; refine = LevenbergMarquardt())).u) ==
        lens(solve(RefineProblem(CameraRegressionProblem(kb, region; fixed = (:d,)), solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region; fixed = (:d,))), CameraRegression()).u), LevenbergMarquardt()).u)
    # everything held: the guess, unmoved
    r0 = solve(RefineProblem(CameraRegressionProblem(kb, region; fixed = (:f, :k, :d)), g), LevenbergMarquardt())
    @test lens(r0.u) == lens(g) && calibration_matrix(r0.u) == calibration_matrix(g)
    # a name or group the guess lacks, and the keywords that are not the stack's, are refused naming `fixed`
    @test_throws r"no parameter or group" solve(RefineProblem(CameraRegressionProblem(kb, region; fixed = (:xi,)), g), LevenbergMarquardt())
    for kw in (:free, :fix, :fixed_params, :hold)
        @test_throws r"give `fixed`" solve(RefineProblem(CameraRegressionProblem(kb, region; (kw => (:d,),)...), g), LevenbergMarquardt())
    end
end

@testitem "camera_verbs: CameraRegression's `fixed` drops the held columns from the one linear solve, at the target's values" setup = [CameraRecipes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Lens, LiftLens, Perspective, Radial, Even, Additive, Tangential, Division, Ellipse, Rect, calibration_matrix, parameters, parameter_names, parameter_vector, lens, rms
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    kb = CameraModel(; pp = (2138.8425928470033, 1436.604235274649), focal = 927.4663595958699,
        lens = KannalaBrandt(-0.03371588391401308, 0.0013137235024557887, -0.0019529571546981683, 0.0001827493332399132))
    region = Rect(0, 0, 4256, 2848) ∩ Ellipse(SVector(2138.8425928470033, 1436.604235274649), 1250.0, 1250.0, 0.0)
    RAT = Lens{Perspective, Radial{Even, 3, 3, Float64}}
    POLY = Lens{Perspective, Radial{Even, 3, 0, Float64}}
    # the denominator held at the anchor's zeros IS the free subproblem: the polynomial family's fit
    rh = solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region; fixed = (:d,))), CameraRegression())
    rp = solve(EstimationProblem(CameraRegressionProblem(POLY, kb, region)), CameraRegression())
    ch, cp = parameters(lens(rh.u)), parameters(lens(rp.u))
    @test (ch.d1, ch.d2, ch.d3) === (0.0, 0.0, 0.0)
    @test maximum(abs, (ch.k1 - cp.k1, ch.k2 - cp.k2, ch.k3 - cp.k3)) <= 1.0e-10
    @test abs(calibration_matrix(rh.u)[1, 1] - calibration_matrix(rp.u)[1, 1]) <= 1.0e-8
    @test abs(rh.rms - rp.rms) <= 1.0e-9
    # CaliBlobs' pattern: an OpenCV target freeing k, holding the tangential terms and the denominator at the TARGET
    # LENS's values; the source in the family's numerator: its k recovered, the held values exactly the target's
    src = CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = Lens(Perspective(), Radial((-0.12, 0.03, 0.004))))
    target = Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(1.0e-4, -2.0e-4)))
    disc = Ellipse(SVector(960.0, 540.0), 700.0, 700.0, 0.0)
    r = solve(EstimationProblem(CameraRegressionProblem(target, src, disc; fixed = (:p, :d))), CameraRegression())
    c = parameters(lens(r.u))
    @test retcode_symbol(r) === :solved
    @test (c.p1, c.p2, c.d1, c.d2, c.d3) === (1.0e-4, -2.0e-4, 0.0, 0.0, 0.0)
    @test maximum(abs, (c.k1 + 0.12, c.k2 - 0.03, c.k3 - 0.004)) < 1.0e-8
    @test abs(calibration_matrix(r.u)[1, 1] - 900.0) < 1.0e-6
    # the residual is the held tangential terms' alone (the target's p, not the source's: ~p·r² at the samples' azimuth
    # 0), so the median is sane, not CaliBlobs' 19.7 px; with them at zero the fit is exact
    @test sort(r.metadata.residuals)[end ÷ 2] < 1.0 && r.metadata.max_px < 2.0                # px
    r00 = solve(EstimationProblem(CameraRegressionProblem(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0))), src, disc; fixed = (:p, :d))), CameraRegression())
    @test r00.metadata.max_px < 1.0e-6
    # everything held: the target, unchanged, at the source's focal
    r0 = solve(EstimationProblem(CameraRegressionProblem(target, src, disc; fixed = parameter_names(CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = target)))), CameraRegression())
    @test lens(r0.u) == target && calibration_matrix(r0.u) == calibration_matrix(src)
    # a lift holds a coefficient only with the focal: d2 held at the target's value recovers d1 exactly
    mdiv = CameraModel(; pp = (2120.0, 1416.0), focal = 927.0, lens = Division(-0.3302, -0.0765))
    ddisc = Ellipse(SVector(2120.0, 1416.0), 2000.0, 2000.0, 0.0)
    @test_throws r"hold the focal too" solve(EstimationProblem(CameraRegressionProblem(Division(0.0, -0.0765), mdiv, ddisc; fixed = (:d2,))), CameraRegression())
    rl = solve(EstimationProblem(CameraRegressionProblem(Division(0.0, -0.0765), mdiv, ddisc; fixed = (:f, :d2))), CameraRegression())
    cl = parameters(lens(rl.u))
    @test cl.d2 === -0.0765 && abs(cl.d1 + 0.3302) < 1.0e-9 && rl.metadata.max_px < 1.0e-6
    # the held spellings that are not the stack's are refused; the fit-then-refine recipe forwards `fixed` to both stages
    for kw in (:free, :fix, :fixed_params, :hold)
        @test_throws r"give `fixed`" solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region; (kw => (:d,),)...)), CameraRegression())
    end
    ra = solve(EstimationProblem(CameraRegressionProblem(RAT, kb, region; fixed = (:d,))), CameraRegression(; refine = LevenbergMarquardt()))
    ca = parameters(lens(ra.u))
    @test (ca.d1, ca.d2, ca.d3) === (0.0, 0.0, 0.0) && ra.rms <= rh.rms * (1 + 1.0e-9)
end

# the camera-regression refine's regression state reduces its work items' partial sums through Core's two reductions (0.10.2; until
# 0.10.1 it unpacked them by hand): the same bits as that hand unpack, at the start and at a moved chart point.
@testitem "camera_verbs: the camera-regression refine's regression reduces its partial sums through Core's reduce_normal_equations and reduce_cost" setup = [FitCameraLensCases] begin
    using Test
    using StaticArrays: SVector, SMatrix
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: calibration_matrix, lens, tangent_size, residual_cost!, residual_normal_equations!
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    region = disc(mdiv, rcorner)
    guess = solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, region)), CameraRegression()).u
    # the camera-regression refine's own construction of the state (src/fitting/camera_regression_refine.jl)
    role = VgtkCore.Cameras.IntrinsicsRole(lens(mdiv))
    moving = VgtkCore.free_parameter_names(guess, ())
    K0 = SMatrix{3, 3, Float64}(calibration_matrix(mdiv)); M = VgtkEstimation._offset_matrix(K0)
    start = VgtkEstimation._refine_start(role, SMatrix{3, 3, Float64}(calibration_matrix(guess)), K0, M)
    pr = VgtkEstimation._profile_samples(role, KB4, mdiv, region, K0, VgtkEstimation.ImageMeasure(), VgtkCore.default_device())
    P = VgtkEstimation._regression_manifold(role, VgtkEstimation._lens_family_manifold(role, lens(guess), moving), start, moving)
    lm = VgtkEstimation._LensRegressionLm(VgtkCore.default_device(), P, false, Inf, M, pr.rays, pr.angles, pr.points, sqrt.(pr.w))
    N = tangent_size(P)
    @test N > 0 && size(lm.partials, 2) > 1
    for x in (zero(SVector{N, Float64}), SVector{N, Float64}(ntuple(a -> 1.0e-6 * (-1)^a / a, N)))
        c, g, H = residual_normal_equations!(nothing, nothing, lm, x)
        @test isfinite(c) && all(isfinite, g) && all(isfinite, H)
        t = Array(sum(lm.partials; dims = 2))          # 0.10.1's hand unpack of the same partial sums
        @test (c, g, H) === (0.5 * t[1], SVector{N, Float64}(ntuple(a -> t[1 + a], N)), SMatrix{N, N, Float64}(ntuple(k -> t[1 + N + k], N * N)))
        cost = residual_cost!(nothing, lm, x)
        @test cost === 0.5 * sum(view(lm.partials, 1, :))
        @test cost ≈ c
        # 0.10.2's two calls, the launch and then the reduction, against Core's launch-and-reduce (0.10.3): the same bits
        apart = similar(lm.partials)
        args = (lm.P, lm.angle_residual, lm.θcap, lm.M, lm.rays, lm.angles, lm.points, lm.sw, x)
        VgtkCore._launch_rows!(VgtkEstimation._normal_row!, apart, size(apart, 2), args...)
        @test residual_normal_equations!(nothing, nothing, lm, x) === VgtkCore.reduce_normal_equations(apart, Val(N))
        VgtkCore._launch_rows!(VgtkEstimation._cost_row!, apart, size(apart, 2), args...)
        @test residual_cost!(nothing, lm, x) === VgtkCore.reduce_cost(apart)
    end
end

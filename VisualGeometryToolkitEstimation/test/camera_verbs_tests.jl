# =============================================================================
# The camera-from-camera verbs (0.9): fit_camera is linear (no guess),
# refine_camera polishes a guess by the chart LM, approximate_camera is the two
# =============================================================================

@testitem "camera_verbs: the rational linear fit of icra2026's fisheye lands at OpenCV's calibration (failure A)" begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Lens, Perspective, Additive, Radial, Tangential, ThinPrism,
        Rect, Ellipse, estimate, metadata, retcode, calibration_matrix, parameters, rms, lens
    using VisualGeometryToolkitEstimation: fit_camera, approximate_camera
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
    r = fit_camera(RAT, kb, region)
    @test retcode(r) === :solved
    K = calibration_matrix(estimate(r))
    @test abs(K[1, 1] - 921.9) < 10                                                    # px: OpenCV's focal (925.3)
    @test hypot(K[1, 3] - 2140.0957668437163, K[2, 3] - 1441.4063814989743) < 6        # px: the KB's principal point, kept
    c = parameters(lens(estimate(r)))
    @test 0.2 < c.k1 < 0.6 && 0.4 < c.d1 < 1.0 && c.d2 > 0.01                           # far from zero, OpenCV's signs and sizes
    @test c.p1 == 0 && c.sx1 == 0                                                       # decentering: zero, not seen by the radial relation
    @test metadata(r).max_px < 1 && rms(r) < 0.25                                       # px
    # and the refine from it converges near it
    ra = approximate_camera(RAT, kb, region)
    @test retcode(ra) === :converged && abs(calibration_matrix(estimate(ra))[1, 1] - 921.9) < 10 && rms(ra) <= rms(r)
end

@testitem "camera_verbs: fit_camera's linear residual is the returned camera's, and records no scale" setup = [FitCameraLensCases] begin
    using Test
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: estimate, metadata, rms, lens, calibration_matrix, projected_radius, project, retcode,
        iterations, noise_scale, scale_source, ScaleNone, Division, weights, weight_kind, BinaryWeights, inlier_mask
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera, refine_camera, approximate_camera, CameraFitMetadata
    for L in (KB4, SC3, Division{2})
        region = disc(mdiv, 2000.0)
        r = fit_camera(L, mdiv, region)
        @test retcode(r) === :solved && iterations(r) == 0
        m = metadata(r)
        @test length(m.residuals) == 129 && maximum(m.residuals) == m.max_px
        # recomputed from the returned camera: the source's ray at each profile node (Clenshaw–Curtis on [0, θ_max]),
        # projected, against the source's pixel of it
        cam = estimate(r)
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
        rr = refine_camera(estimate(r), mdiv, region)
        @test scale_source(rr) === ScaleNone
        @test_throws ArgumentError noise_scale(rr)
    end
    # KB from a physical source: the polynomial's linear fit IS the integral's least squares, the refine's minimum
    r = fit_camera(KB4, mdiv, disc(mdiv, rcorner))
    ra = approximate_camera(KB4, mdiv, disc(mdiv, rcorner))
    @test abs(rms(r) - rms(ra)) <= 1.0e-6 * rms(ra)
end

@testitem "camera_verbs: a family with no linear form is refused, naming refine_camera; approximate_camera starts it from its anchor" setup = [FitCameraLensCases] begin
    using Test
    using VisualGeometryToolkitCore: Lens, Perspective, Equidistant, Tilt, FieldOfView, Radial, Unified, ExtendedUnified, DoubleSphere,
        estimate, metadata, retcode
    using VisualGeometryToolkitEstimation: fit_camera, approximate_camera, refine_camera
    for L in (DSL, typeof(Lens(Unified(0.0))), typeof(Lens(ExtendedUnified(0.0, 0.0))), typeof(Lens(Perspective(), FieldOfView(0.0))),
              typeof(Lens(Perspective(), Tilt(0.0, 0.0) ∘ Radial((0.0, 0.0)))))
        @test_throws r"refine_camera" fit_camera(L, mdiv, disc(mdiv, 1500.0))
        @test_throws r"approximate_camera" fit_camera(L, mdiv, disc(mdiv, 1500.0))
    end
    r = approximate_camera(DSL, mdiv, disc(mdiv, 1500.0))
    @test retcode(r) === :converged
    # the refine's keywords: L2 only, no scale input
    g = estimate(fit_camera(KB4, mdiv, disc(mdiv, 1500.0)))
    @test_throws ArgumentError refine_camera(g, mdiv, disc(mdiv, 1500.0); scale = 1.0)
    @test_throws ArgumentError refine_camera(g, mdiv, disc(mdiv, 1500.0); width = 1.0)
    @test_throws ArgumentError refine_camera(g, mdiv, disc(mdiv, 1500.0); optimization_strategy = :huber)
    @test_throws MethodError refine_camera(g, mdiv, disc(mdiv, 1500.0); local_optimization = nothing)   # the retired keyword
end

@testitem "camera_verbs: refine_camera from a perturbed guess reaches approximate_camera's minimum" setup = [FitCameraLensCases] begin
    using Test
    using StaticArrays: SMatrix
    using VisualGeometryToolkitCore: CameraModel, LogicalIntrinsics, CameraCalibrationMat, estimate, metadata, lens, retcode,
        calibration_matrix, parameter_vector, with_parameters, rms, Division
    using VisualGeometryToolkitEstimation: approximate_camera, refine_camera
    for (L, R) in ((KB4, rcorner), (SC3, rcorner), (Division{2}, 2500.0))
        region = disc(mdiv, R)
        a = approximate_camera(L, mdiv, region)
        best = estimate(a)
        p = collect(parameter_vector(lens(best)))
        K = SMatrix{3, 3, Float64}(calibration_matrix(best))
        guess = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1] * 1.05, 0, 0, 0, K[2, 2] * 1.05, 0, K[1, 3], K[2, 3], 1); check = false)),
            with_parameters(lens(best), p .* 0.8))
        r = refine_camera(guess, mdiv, region)
        @test retcode(r) === :converged
        @test maximum(abs.(collect(parameter_vector(lens(estimate(r)))) .- p)) < 1.0e-5
        @test abs(calibration_matrix(estimate(r))[1, 1] - K[1, 1]) < 1.0e-5 * K[1, 1]
        @test abs(rms(r) - rms(a)) <= 1.0e-6 * max(rms(a), 1.0e-9) + 1.0e-12
    end
    # a physical source's principal point is data: a guess that moves it is refused
    g = estimate(approximate_camera(KB4, mdiv, disc(mdiv, 1500.0)))
    K = SMatrix{3, 3, Float64}(calibration_matrix(g))
    moved = CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0, 0, 0, K[2, 2], 0, K[1, 3] + 3, K[2, 3], 1); check = false)), lens(g))
    @test_throws r"principal point" refine_camera(moved, mdiv, disc(mdiv, 1500.0))
end

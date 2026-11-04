# =============================================================================
# fit_camera — one model-to-model regression over a region of the sensor
# =============================================================================
#
# A lens family reproduces the profile regression it replaced (recorded), with
# the region's exact 1-D reduction; a SplineProjection of the synthetic fisheye
# grid meets its tolerance on held-out points, edge band included, and its size
# bisection rests on a monotone error; a grid fitted from a parametric model
# beats node interpolation and stays smooth outside the region.

@testitem "fit_camera: a lens family reproduces the recorded profile regressions" setup = [FitCameraLensCases] begin
    using Test
    using VisualGeometryToolkitCore: estimate, metadata, rms, lens, parameter_vector
    using VisualGeometryToolkitEstimation: fit_camera, CameraFitMetadata
    worst = 0.0
    for (name, (L, m, _, region)) in LENS_CASES
        r = name in ("OPENCV3", "OPENCV decentered") ?
            (@test_logs (:warn, r"cannot represent") match_mode = :any fit_camera(L, m, region)) : fit_camera(L, m, region)
        @test metadata(r) isa CameraFitMetadata
        p, s, max_px = RECORDED[name]
        q = parameter_vector(lens(estimate(r)))
        @test agrees(q, p)
        @test abs(metadata(r).family.focal_scale - s) <= 1.0e-6
        @test agrees_px(metadata(r).max_px, max_px)
        @test rms(r) == metadata(r).rms_px
        isempty(p) || (worst = max(worst, maximum(abs.(collect(q) .- collect(p)))))
    end
    @test worst < 1.0e-8                          # measured 2026-11-04: 1.4e-9 (the double sphere past its image circle)
end

@testitem "fit_camera: parametric equivalence with fit_lens (before its removal)" setup = [FitCameraLensCases] begin
    using Test
    using VisualGeometryToolkitCore: estimate, metadata, lens, parameter_vector
    using VisualGeometryToolkitEstimation: fit_camera, fit_lens
    for (name, (L, m, kw, region)) in LENS_CASES
        a = fit_lens(L, m; kw...)
        b = fit_camera(L, m, region)
        pa, _, max_px = RECORDED[name]
        @test agrees(parameter_vector(lens(estimate(a))), pa) && agrees_px(metadata(a).max_px, max_px)   # the record is fit_lens's
        @test agrees(parameter_vector(lens(estimate(b))), parameter_vector(lens(estimate(a))))
        @test agrees_px(metadata(b).max_px, metadata(a).max_px)
        # a view-angle cap clipped the predecessor's domain; the region is the domain here
        haskey(kw, :max_view_angle) || @test metadata(b).family.clipped == metadata(a).clipped
    end
end

@testitem "fit_camera: the region's share of each radius" setup = [FitCameraLensCases] begin
    using Test
    using StaticArrays: SMatrix
    using VisualGeometryToolkitCore: estimate, metadata, lens, parameter_vector, calibration_matrix, Rect, WholePlane
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    K = SMatrix{3, 3, Float64}(calibration_matrix(mdiv))
    # a disc about the principal point: every azimuth leaves it at its radius
    e = VgtkEstimation._exit_radii(disc(mdiv, 1000.0), K)
    @test maximum(abs.(e .- 1000.0 / 927.0)) < 1.0e-12
    # the frame: the exit radius along an azimuth is the distance to the frame's edge
    frame = Rect(0, 0, 4240, 2832)
    e = VgtkEstimation._exit_radii(frame, K)
    ψ = [2π * (k - 0.5) / length(e) for k in eachindex(e)]
    edge = [min(2120.0 / abs(cos(a)), 1416.0 / abs(sin(a))) / 927.0 for a in ψ]
    @test maximum(abs.(e .- edge)) < 1.0e-9
    @test VgtkEstimation._coverage(e, 1000.0 / 927.0) == 1.0            # inside the frame on every azimuth
    @test 0 < VgtkEstimation._coverage(e, 2000.0 / 927.0) < 1           # past the frame's half height, not its half width
    @test VgtkEstimation._coverage(e, 1.01 * rcorner / 927.0) == 0.0
    # the frame against the disc through its corners: the same domain (to the azimuth
    # sampling, which reaches the corners within half an azimuth step), less weight
    # past the edges
    rf = fit_camera(KB4, mdiv, frame)
    rd = fit_camera(KB4, mdiv, disc(mdiv, rcorner))
    @test metadata(rf).family.θ_max ≈ metadata(rd).family.θ_max rtol = 1.0e-3
    @test parameter_vector(lens(estimate(rf))) != parameter_vector(lens(estimate(rd)))
    @test metadata(rf).max_px_edge <= metadata(rf).max_px
    # a region that misses the principal point, the whole plane: refused
    @test_throws ArgumentError fit_camera(KB4, mdiv, Rect(0, 0, 100, 100))
    @test_throws ArgumentError fit_camera(KB4, mdiv, WholePlane())
end

@testitem "fit_camera: a SplineProjection of the synthetic fisheye grid" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, rms, ChartSpline, SplineProjection,
        project, backproject, transform_deriv
    using VisualGeometryToolkitEstimation: fit_camera, CameraFitMetadata
    using FiniteDiff: FiniteDiff

    r = fit_camera(SplineProjection, GRID, REGION; tolerance = 1.0e-3)
    sp = estimate(r)
    @test sp isa SplineProjection && sp.spline isa ChartSpline{2}
    @test metadata(r) isa CameraFitMetadata
    @test retcode(r) === :converged
    # the held-out maximum, the edge band's included (measured 2026-11-04: 9.2e-4 px,
    # in the band; rms 1.0e-4 px; 28 × 28 controls)
    @test metadata(r).max_px <= 1.0e-3
    @test metadata(r).max_px_edge <= 1.0e-3
    @test rms(r) < 2.0e-4
    @test metadata(r).family.counts[1] == metadata(r).family.counts[2]           # a full disc: isotropic counts
    @test metadata(r).family.tolerance == 1.0e-3 && iterations(r) > 1
    # the round trip at fresh points of the region, through the public verbs
    pts = [SVector(500.0 + ρ * cos(a), 500.0 + ρ * sin(a)) for ρ in (0.0, 120.0, 300.0, 470.0) for a in range(0, 2π; length = 13)]
    @test maximum(p -> norm(project(sp, backproject(GRID, p)) - p), pts) < 1.0e-3
    # a ray outside the region: its pixel lies outside the disc
    out = backproject(FISHEYE, SVector(500.0 + 495.0, 500.0))
    @test project(sp, out) === nothing
    @test sp(out) isa SVector{2}                                             # the map itself continues
    # the Jacobian against central differences
    for p in pts[1:6:end]
        X = backproject(GRID, p)
        J = transform_deriv(sp, X)
        Jfd = FiniteDiff.finite_difference_jacobian(x -> Vector(sp(SVector{3}(x))), Vector(X))
        @test J ≈ Jfd rtol = 1.0e-5 atol = 1.0e-4
    end
end

@testitem "fit_camera: the spline's error is non-increasing in its size" setup = [FitCameraFisheye] begin
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # the property the bisection relies on, at three spacings (measured 2026-11-04:
    # 5.1e-3, 2.7e-3, 5.4e-4 px at 8, 16, 32 cells)
    q = VgtkEstimation._chart_quadrature(GRID, REGION)
    errs = [VgtkEstimation._spline_projection_fit(q, c).max_px for c in (8, 16, 32)]
    @test issorted(errs; rev = true)
    @test errs[3] < 1.0e-3 < errs[1]
end

@testitem "fit_camera: region shapes" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SA
    using VisualGeometryToolkitCore: estimate, metadata, retcode, SplineProjection, CameraModel, KannalaBrandt, Rect, px
    using VisualGeometryToolkitEstimation: fit_camera
    # the disc cut by the frame: a non-square region, anisotropic counts
    cut = Rect(0, 150, 1000, 700) ∩ DISC
    r = fit_camera(SplineProjection, GRID, cut; tolerance = 1.0e-3)
    @test retcode(r) === :converged && metadata(r).max_px <= 1.0e-3
    n = metadata(r).family.counts
    @test n[1] > n[2]                                                           # wider than tall (measured 28 × 22)
    # the whole sensor of a parametric camera (corners at 81°), a CameraModel source
    wide = CameraModel(500.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    r = fit_camera(SplineProjection, wide, Rect(0, 0, 1000, 1000); tolerance = 1.0e-3)
    @test retcode(r) === :converged && metadata(r).max_px <= 1.0e-3
    # the grid's whole sensor (its corners near 150°, far past the lit disc) does not
    # reach 1e-3 px within the bracket: the fit says so and reports what it reached
    # (measured 1.4e-3 px at 169 × 169 controls)
    r = fit_camera(SplineProjection, GRID, Rect(0, 0, 1000, 1000); tolerance = 1.0e-3)
    @test retcode(r) === :max_iterations
    @test 1.0e-3 < metadata(r).max_px < 1.0e-2
    @test_throws ArgumentError fit_camera(SplineProjection, GRID, REGION; tolerance = 1.0e-3, weight = :solid_angle)
end

@testitem "fit_camera: a grid from a parametric model" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: estimate, metadata, retcode, BoundedCamera, Size2, CentralGenericModel, angle_between
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = fit_camera(CentralGenericModel, bc, REGION; grid_size = (40, 40), tolerance = 0.01)
    g = estimate(r)
    @test retcode(r) === :solved
    # Core's full-image lattice, the region as the domain
    @test (g.width, g.height, g.calib_min_x, g.calib_min_y, g.calib_max_x, g.calib_max_y) == (1000, 1000, 0, 0, 999, 999)
    @test size(g.grid) == (40, 40) && g.domain === REGION
    # the round trip over the region: no worse than node interpolation (measured
    # 2026-11-04: 5.4e-5 px against 6.3e-3 px)
    interpolated = CentralGenericModel(FISHEYE, (1000, 1000); grid_size = (40, 40))
    q = VgtkEstimation._image_quadrature(FISHEYE, REGION)
    e_fit = VgtkEstimation._grid_round_trip(g, q.held_rays, q.held_px)
    e_int = VgtkEstimation._grid_round_trip(interpolated, q.held_rays, q.held_px)
    @test maximum(e_fit) <= maximum(e_int)
    @test maximum(e_fit) ≈ metadata(r).max_px
    @test metadata(r).max_px < 1.0e-3
    # the nodes outside the region: finite, and smooth — the unit controls' third
    # differences along both axes (measured 0.0025, node interpolation's 0.11, ringing
    # in the corners); the nodes whose support meets the region within its bounding
    # cone ± 30° (measured 109° against 97°). The full-image lattice reaches the image
    # corners, ~150° off axis for this lens, so the cone bound is for those nodes only.
    @test all(c -> all(isfinite, c), g.grid)
    unit(G, k) = [normalize(SVector{3}(G[i, j]))[k] for i in axes(G, 1), j in axes(G, 2)]
    d3(A) = max(maximum(abs, diff(diff(diff(A; dims = 1); dims = 1); dims = 1)), maximum(abs, diff(diff(diff(A; dims = 2); dims = 2); dims = 2)))
    @test maximum(k -> d3(unit(g.grid, k)), 1:3) < 0.01
    @test maximum(k -> d3(unit(g.grid, k)), 1:3) < maximum(k -> d3(unit(interpolated.grid, k)), 1:3)
    axis = SVector(0.0, 0.0, 1.0)
    cone = maximum(f -> angle_between(f, axis), q.fit_rays)
    h = 1000 / 37                                                                # the lattice's cell
    near = [norm(SVector((gx - 2) * h, (gy - 2) * h) - SVector(500.0, 500.0)) <= 480 + 2h for gy in 1:40, gx in 1:40]
    @test maximum(angle_between(SVector{3}(g.grid[i]), axis) for i in findall(near)) <= cone + deg2rad(30)
    # a bare camera model has no image for the full-image lattice
    @test_throws ArgumentError fit_camera(CentralGenericModel, FISHEYE, REGION; grid_size = (40, 40), tolerance = 0.01)
end

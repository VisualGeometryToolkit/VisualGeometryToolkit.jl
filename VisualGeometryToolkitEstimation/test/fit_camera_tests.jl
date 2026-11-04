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
        @test 0 <= rms(r) <= metadata(r).max_px
        isempty(p) || (worst = max(worst, maximum(abs.(collect(q) .- collect(p)))))
    end
    @test worst < 1.0e-8                          # measured 2026-11-04: 1.4e-9 (the double sphere past its image circle)
end

@testitem "fit_camera: the profile regression's pieces" setup = [FitCameraLensCases] begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: estimate, metadata, rms, lens, parameters, Lens, LiftLens, KannalaBrandt, Division,
        Orthographic, Equidistant, Perspective, Radial, Additive, Tangential, ThinPrism, Pinhole, radial_part,
        view_angle_limit, CameraModel
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    @testset "Clenshaw–Curtis integrates polynomials" begin
        θ, q = VgtkEstimation._clenshaw_curtis(32, 1.3)
        @test sum(q) ≈ 1.3 rtol = 1.0e-12
        @test sum(q .* θ .^ 2) ≈ 1.3^3 / 3 rtol = 1.0e-10
        @test sum(q .* θ .^ 5) ≈ 1.3^6 / 6 rtol = 1.0e-8
    end
    @testset "the family limits are VGC's own view-angle limits" begin
        @test VgtkEstimation._family_limit(Pinhole) == view_angle_limit(Pinhole()) == π / 2 - 1.0e-3
        @test VgtkEstimation._family_limit(KB4) == view_angle_limit(Lens(Equidistant())) == π - 1.0e-3
        @test VgtkEstimation._family_limit(typeof(Lens(Orthographic()))) == view_angle_limit(Lens(Orthographic()))
        @test VgtkEstimation._family_limit(DSL) == π - 1.0e-3
    end
    @testset "a clip by the target family is reported and warned" begin
        r = @test_logs (:warn, r"cannot represent") match_mode = :any fit_camera(OPENCV3, mdiv, disc(mdiv, rcorner))
        @test metadata(r).family.clipped
        @test metadata(r).family.θ_max < π / 2 < metadata(r).family.θ_source
        @test lens(estimate(r)) isa OPENCV3
        @test !metadata(fit_camera(KB4, mdiv, disc(mdiv, rcorner))).family.clipped
    end
    @testset "decentering: a target holds it at zero, a source's is stripped and warned" begin
        rd = fit_camera(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0)))), mdiv, disc(mdiv, rcorner))
        cd = parameters(lens(estimate(rd)))
        @test cd.p1 == 0 && cd.p2 == 0 && cd.d1 == 0 && cd.d3 == 0
        synthetic = Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)), Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
        @test VgtkEstimation._decentering(synthetic) ≈ hypot(1.0e-3, -5.0e-4, -5.5e-3, 7.0e-4, 2.2e-3, -1.2e-3)
        @test radial_part(synthetic) === Lens(Perspective(), Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)))
        src = CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = synthetic)
        @test_logs (:warn, r"non-radial terms") match_mode = :any fit_camera(OPENCV3, src, disc(src, 500.0))
    end
    @testset "Division alone is the one-coefficient model; a type without its arities is refused" begin
        @test lens(estimate(fit_camera(Division, mdiv, disc(mdiv, rcorner)))) == lens(estimate(fit_camera(Division{1}, mdiv, disc(mdiv, rcorner))))
        @test_throws ArgumentError fit_camera(LiftLens, mdiv, disc(mdiv, rcorner))
        @test_throws ArgumentError fit_camera(KannalaBrandt, mdiv, disc(mdiv, rcorner))
        @test_throws ArgumentError fit_camera(KB4, mdiv, disc(mdiv, rcorner); weight = :pixels)
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

@testitem "fit_camera: a region past the source's fold" begin
    using Test
    using StaticArrays: SA
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Rect, metadata, retcode, px
    using VisualGeometryToolkitEstimation: fit_camera
    # KB(−0.02) at f = 200 px images radii to r(π) ≈ 2.52 (504 px); the frame's
    # corners lie at 707 px, where the source has no view angle (once a NaN
    # domain and an ArgumentError): the fit takes the source's own domain, clipped
    src = CameraModel(200.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    r = @test_logs (:warn, r"past the radii the source") match_mode = :any fit_camera(KannalaBrandt{4}, src, Rect(0, 0, 1000, 1000))
    fam = metadata(r).family
    @test fam.clipped && isfinite(fam.θ_max) && fam.θ_max == fam.θ_source > 3.1
    @test retcode(r) === :converged && metadata(r).max_px < 1.0e-9                  # KB4 holds KB1 exactly
end

@testitem "fit_camera: a SplineProjection of the synthetic fisheye grid" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, rms, ChartSpline, SplineProjection,
        project, backproject, transform_deriv, WholePlane
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
    @test metadata(r).family.tolerance == 1.0e-3 && iterations(r) == 0 && metadata(r).family.outer_iterations > 1
    # the round trip at fresh points of the region, through the public verbs
    pts = [SVector(500.0 + ρ * cos(a), 500.0 + ρ * sin(a)) for ρ in (0.0, 120.0, 300.0, 470.0) for a in range(0, 2π; length = 13)]
    @test maximum(p -> norm(project(sp, backproject(GRID, p)) - p), pts) < 1.0e-3
    # a ray outside the region: its pixel lies outside the disc
    out = backproject(FISHEYE, SVector(500.0 + 495.0, 500.0))
    @test project(sp, out) === nothing
    @test sp(out) isa SVector{2}                                             # the map itself continues
    # a ray far outside the fitted cone (~97°) is not imaged, wherever the continuation lands
    @test project(sp, SVector(sind(170.0), 0.0, cosd(170.0))) === nothing
    @test project(SplineProjection(sp.chart, sp.spline, WholePlane()), SVector(sind(170.0), 0.0, cosd(170.0))) === nothing
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
    errs = [VgtkEstimation._spline_projection_fit(q, c).errors.max for c in (8, 16, 32)]
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
    # the quadrature is on the image: a spline target takes no weight
    @test_throws MethodError fit_camera(SplineProjection, GRID, REGION; tolerance = 1.0e-3, weight = :image)
end

@testitem "fit_camera: a grid from a parametric model" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, BoundedCamera, Size2, CentralGenericModel,
        CameraModel, KannalaBrandt, Rect, Ellipse, angle_between, px
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = fit_camera(CentralGenericModel, bc, REGION; tolerance = 1.0e-4)
    g = estimate(r)
    md = metadata(r)
    # the tolerance chooses the lattice: the largest spacing that meets it (measured
    # 2026-11-04: 35 × 35 nodes, 9.6e-5 px; 40 × 40 reached 5.4e-5 px)
    @test retcode(r) === :converged && iterations(r) == 0 && md.family.outer_iterations > 1
    @test md.max_px <= 1.0e-4 && md.family.tolerance == 1.0e-4
    gh, gw = md.family.grid_size
    @test gh == gw && 20 <= gh <= 40 && VgtkEstimation._lattice_grid_size(1000, 1000, gw - 3) == (gw, gh)
    # Core's full-image lattice, the region as the domain
    @test (g.width, g.height, g.calib_min_x, g.calib_min_y, g.calib_max_x, g.calib_max_y) == (1000, 1000, 0, 0, 999, 999)
    @test size(g.grid) == (gh, gw) && g.domain === REGION
    # the round trip over the region: no worse than node interpolation on the same lattice
    interpolated = CentralGenericModel(FISHEYE, (1000, 1000); grid_size = (gw, gh))
    q = VgtkEstimation._image_quadrature(FISHEYE, REGION)
    e_fit = VgtkEstimation._grid_round_trip(g, q.held_rays, q.held_px)
    e_int = VgtkEstimation._grid_round_trip(interpolated, q.held_rays, q.held_px)
    @test maximum(e_fit) <= maximum(e_int)
    @test maximum(e_fit) ≈ md.max_px
    # the nodes outside the region: finite, and smooth — the unit controls' third
    # differences along both axes, below node interpolation's (ringing in the
    # corners); the nodes whose support meets the region within its bounding cone
    # ± 30° (measured 109° against 97° at 40 × 40). The full-image lattice reaches
    # the image corners, ~150° off axis for this lens, so the cone bound is for
    # those nodes only.
    @test all(c -> all(isfinite, c), g.grid)
    unit(G, k) = [normalize(SVector{3}(G[i, j]))[k] for i in axes(G, 1), j in axes(G, 2)]
    d3(A) = max(maximum(abs, diff(diff(diff(A; dims = 1); dims = 1); dims = 1)), maximum(abs, diff(diff(diff(A; dims = 2); dims = 2); dims = 2)))
    @test maximum(k -> d3(unit(g.grid, k)), 1:3) < 0.01
    @test maximum(k -> d3(unit(g.grid, k)), 1:3) < maximum(k -> d3(unit(interpolated.grid, k)), 1:3)
    axis = SVector(0.0, 0.0, 1.0)
    cone = maximum(f -> angle_between(f, axis), q.fit_rays)
    h = 1000 / (gw - 3)                                                          # the lattice's cell
    near = [norm(SVector((gx - 2) * h, (gy - 2) * h) - SVector(500.0, 500.0)) <= 480 + 2h for gy in 1:gh, gx in 1:gw]
    @test maximum(angle_between(SVector{3}(g.grid[i]), axis) for i in findall(near)) <= cone + deg2rad(30)
    # a tolerance out of reach at the finest lattice (a 60 px image: 13 × 13
    # nodes at most) says so and returns what it reached, never warns only
    small = BoundedCamera(CameraModel(18.0px, SA[30.0px, 30.0px]; lens = KannalaBrandt(-0.02)), Size2(60, 60))
    ru = fit_camera(CentralGenericModel, small, Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0); tolerance = 1.0e-6)
    @test retcode(ru) === :max_iterations && metadata(ru).max_px > 1.0e-6 && metadata(ru).family.grid_size == (13, 13)
    # a bare camera model has no image for the full-image lattice; grid_size is not an input
    @test_throws ArgumentError fit_camera(CentralGenericModel, FISHEYE, REGION; tolerance = 0.01)
    @test_throws MethodError fit_camera(CentralGenericModel, bc, REGION; grid_size = (40, 40), tolerance = 0.01)
    @test_throws ArgumentError fit_camera(CentralGenericModel, bc, REGION; tolerance = 0.0)
    # fewer than 4 nodes on an axis: a named ArgumentError
    @test_throws r"at least 4 × 4 nodes" VgtkEstimation._penalized_solve(Tuple{SVector{16, Int}, SVector{16, Float64}}[], zeros(0, 2), 3, 9, Val(2))
end

@testitem "fit_camera: a non-central grid from a non-central source" setup = [FitCameraNoncentral] begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, normalize, dot
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, rms, NoncentralGenericModel, Vec3, backproject,
        interpolated_direction, interpolated_point, nearest_point
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    r = fit_camera(NoncentralGenericModel, SOURCE, REGION; tolerance = 1.0e-3)
    g = estimate(r)
    md = metadata(r)
    # the tolerance chooses the lattice (measured 2026-11-04: 26 × 26 nodes)
    @test retcode(r) === :converged && iterations(r) == 0 && md.family.outer_iterations > 1
    gh, gw = md.family.grid_size
    # Core's full-image lattice, the region as the domain
    @test (g.width, g.height, g.calib_min_x, g.calib_min_y, g.calib_max_x, g.calib_max_y) == (1000, 1000, 0, 0, 999, 999)
    @test size(g.directions) == size(g.points) == (gh, gw) && g.domain === REGION
    # the anchor: Core's nearest point of every quadrature line of the source
    q = VgtkEstimation._image_quadrature(VgtkEstimation._source_lines, SOURCE, REGION)
    @test md.family.center ≈ nearest_point([o => d for (o, d) in q.fit_rays])
    # held out, interior and edge band (measured 2026-11-04 at 26 × 26: direction
    # 9.2e-4 px max, 1.3e-4 px rms, the max at the edge; line offset 5.1e-8 m max
    # against a 3 mm field; at 40 × 40 7.1e-4 px and 7.9e-9 m) — and far better
    # than sampling the source's lines at the nodes
    @test md.max_px <= 1.0e-3 && rms(r) < 5.0e-4 && md.max_px_edge <= md.max_px
    @test md.family.max_offset < 1.0e-7 && md.family.rms_offset < 5.0e-8
    h = 1000 / (gw - 3)
    lines = [backproject(SOURCE, SVector(clamp((j - 2) * h, 0.0, 999.0), clamp((i - 2) * h, 0.0, 999.0))) for i in 1:gh, j in 1:gw]
    sampled = NoncentralGenericModel{Float64}(
        1000, 1000, 0, 0, 999, 999, [Vec3{Float64}(l.direction) for l in lines], [Vec3{Float64}(l.origin) for l in lines]
    )
    e_s, o_s = VgtkEstimation._line_errors(sampled, SOURCE, q)
    @test md.max_px < e_s.max / 100 && md.family.max_offset < o_s.max / 100
    # the rays at and past 90° (to ~97°): the fitted lines stay on the source's
    # (measured 2.0e-6 rad, 7.2e-9 m)
    θ = [acos(clamp(d[3], -1, 1)) for (_, d) in q.held_rays]
    grazing = findall(>=(π / 2), θ)
    @test length(grazing) > 1000 && rad2deg(maximum(θ)) > 95
    angle(i) = acos(clamp(dot(normalize(SVector{3}(interpolated_direction(g, q.held_px[i]))), q.held_rays[i][2]), -1, 1))
    function offset(i)
        o, d = q.held_rays[i]
        Δ = SVector{3}(interpolated_point(g, q.held_px[i])) - o
        return norm(Δ - dot(Δ, d) * d)
    end
    @test maximum(angle, grazing) < 1.0e-4
    @test maximum(offset, grazing) < 1.0e-7
    # frame in = frame out: translating every source line by t translates the
    # fitted points and the anchor by t (absolute points, not relative to c*)
    t = SVector(0.5, -0.25, 2.0)
    rt = fit_camera(NoncentralGenericModel, noncentral_source(t), REGION; tolerance = 1.0e-3)
    @test metadata(rt).family.grid_size == md.family.grid_size
    @test maximum(norm.(estimate(rt).points .- g.points .- Ref(t))) < 1.0e-9
    @test norm(metadata(rt).family.center - md.family.center - t) < 1.0e-12
    @test estimate(rt).directions ≈ g.directions
    # a non-central grid source has its own image: no bound needed; a tolerance out
    # of reach at the finest lattice (169 × 169 for 1000 px) returns what it reached
    # (measured 4.3e-5 px), never throws
    ru = fit_camera(NoncentralGenericModel, SOURCE, REGION; tolerance = 1.0e-9)
    @test retcode(ru) === :max_iterations && 1.0e-9 < metadata(ru).max_px < 1.0e-3
    @test metadata(ru).family.grid_size == (169, 169)
end

@testitem "fit_camera: a non-central grid from a central source" setup = [FitCameraNoncentral] begin
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: estimate, metadata, retcode, NoncentralGenericModel, CentralGenericModel, BoundedCamera, Size2,
        backproject, CameraModel, KannalaBrandt, Rect, Ellipse, px
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = fit_camera(NoncentralGenericModel, bc, REGION; tolerance = 1.0e-3)
    g = estimate(r)
    @test retcode(r) === :converged
    # every line through the projection centre: c* is the centre and every
    # point is c* (offsets ≡ 0)
    @test metadata(r).family.center == SVector(0.0, 0.0, 0.0)
    @test all(iszero, g.points)
    @test metadata(r).family.max_offset == 0
    # the directions as the central grid target fits them: the same lattice
    # (measured 21 × 21, 8.5e-4 px)
    @test metadata(r).max_px <= 1.0e-3
    @test metadata(r).family.grid_size == metadata(fit_camera(CentralGenericModel, bc, REGION; tolerance = 1.0e-3)).family.grid_size
    # out of reach at the finest lattice of a 60 px image: said, not warned
    small = BoundedCamera(CameraModel(18.0px, SA[30.0px, 30.0px]; lens = KannalaBrandt(-0.02)), Size2(60, 60))
    ru = fit_camera(NoncentralGenericModel, small, Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0); tolerance = 1.0e-6)
    @test retcode(ru) === :max_iterations && metadata(ru).max_px > 1.0e-6
    # the batched central source lines are the scalar back-projection, normalized
    pix = [SVector(500.5, 400.5), SVector(20.5, 500.5), SVector(950.5, 980.5)]
    @test VgtkEstimation._source_lines(FISHEYE, pix) ==
        [(SVector(0.0, 0.0, 0.0), normalize(SVector{3, Float64}(backproject(FISHEYE, p)))) for p in pix]
    @test_throws ArgumentError fit_camera(NoncentralGenericModel, FISHEYE, REGION; tolerance = 0.01)
end

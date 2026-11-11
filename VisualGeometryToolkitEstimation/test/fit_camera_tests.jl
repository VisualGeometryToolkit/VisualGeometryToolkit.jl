# =============================================================================
# fit_camera — one model-to-model regression over a region of the sensor
# =============================================================================
#
# A lens family reproduces the profile regression it replaced (recorded), with
# the region's exact 1-D reduction; the forward spline of the synthetic fisheye's
# SplineLens camera meets its tolerance on held-out points, edge band included,
# its backward kept exactly, and its size bisection rests on a monotone error; a
# SplineLens backward fitted from a parametric model beats node interpolation and
# stays smooth outside the region.

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

@testitem "fit_camera: the :image density's dr/dθ is exact — the central difference it replaced to its truncation error" setup = [FitCameraLensCases] begin
    using Test
    using VisualGeometryToolkitCore: radial_part, lens, projected_radius, view_angle_limit
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # the replaced rule: h = 1e-5·b central differences (b the domain's edge); its
    # error O(h²·r‴) + O(eps·r/h) is below 1e-8 here. The fits it weights are the
    # recorded regressions' above, unchanged to their 1e-6.
    for (name, (_, m, _, _)) in LENS_CASES
        srcr = radial_part(lens(m))
        b = min(Float64(view_angle_limit(srcr)), 1.4) * 0.95
        h = 1.0e-5 * b
        for θ in range(0.05b, 0.95b; length = 7)
            cd = (Float64(projected_radius(srcr, θ + h)) - Float64(projected_radius(srcr, θ - h))) / 2h
            r, dr = VgtkEstimation._radius_with_jacobian(srcr, θ)     # one Dual evaluation: the value and dr/dθ
            @test abs(dr - cd) <= 1.0e-8 * max(1.0, abs(cd))
            @test abs(r - Float64(projected_radius(srcr, θ))) <= 4 * eps(max(1.0, abs(r)))   # the plain radius, to its rounding
        end
    end
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
        @test VgtkEstimation._decentering(VgtkEstimation.VgtkCore.Cameras.RadialSymmetry(synthetic), synthetic) ≈ hypot(1.0e-3, -5.0e-4, -5.5e-3, 7.0e-4, 2.2e-3, -1.2e-3)
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


@testitem "fit_camera(SplineLens): a SplineLens source keeps its backward, on its optical axis, and gains a forward spline" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, rms, ChartSpline, SplineLens, CameraModel,
        BoundedCamera, Size2, project, backproject, calibration_matrix, lens, rotation
    using VisualGeometryToolkitEstimation: fit_camera, CameraFitMetadata

    r = fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3)
    cam = estimate(r)
    @test cam isa CameraModel && lens(cam) isa SplineLens && lens(cam).forward isa ChartSpline{2}
    @test metadata(r) isa CameraFitMetadata
    @test retcode(r) === :converged
    # the backward is the source's on its optical axis (2026-11-11, changed on purpose: a gauge-K source is gauged
    # first, `optical_axis` then Core's `axis_gauged`): its controls turned by the fit's rotation, exactly, its K's
    # principal point moved to the axis's pixel. GRID is the fisheye's own spline, so the turn is its approximation
    # error alone (measured 2026-11-11: 6e-7 px of principal point)
    R = rotation(metadata(r).rotation)
    @test lens(cam).backward.controls == map(c -> R * c, lens(GRID).backward.controls)
    @test maximum(abs, R - one(R)) < 1.0e-8
    K, KG = calibration_matrix(cam), calibration_matrix(GRID)
    @test K[1, 1] == KG[1, 1] && K[2, 2] == KG[2, 2] && maximum(abs, K[1:2, 3] - KG[1:2, 3]) < 1.0e-5
    fam = metadata(r).family
    @test fam.λ === nothing && fam.grid_size == (40, 40) && fam.outer_iterations == 0
    @test metadata(r).max_px < 1.0e-9                                           # the kept backward against itself
    # the forward's held-out maximum, the edge band's included (the old forward map
    # it replaced measured 2026-11-04: 9.2e-4 px, in the band; rms 1.0e-4 px; 28 × 28 controls)
    @test fam.forward.max_px <= 1.0e-3
    @test fam.forward.max_px_edge <= 1.0e-3
    @test fam.forward.rms < 2.0e-4
    @test fam.forward.counts[1] == fam.forward.counts[2]                                        # a full disc: isotropic counts
    @test fam.tolerance == 1.0e-3 && iterations(r) == 0 && fam.forward.outer_iterations > 1
    # the round trip at fresh points of the region, through the public verbs: the
    # camera projects through its forward spline
    pts = [SVector(500.0 + ρ * cos(a), 500.0 + ρ * sin(a)) for ρ in (0.0, 120.0, 300.0, 470.0) for a in range(0, 2π; length = 13)]
    @test maximum(p -> norm(project(cam, backproject(GRID, p)) - p), pts) < 1.0e-3
    # a ray far outside the fitted cone (~97°) is not imaged, wherever the continuation lands
    @test all(isnan, project(cam, SVector(sind(170.0), 0.0, cosd(170.0))))
    # forward = false: the backward alone, the same controls; no forward record
    rb = fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3, forward = false)
    @test lens(estimate(rb)).forward === nothing && lens(estimate(rb)).backward.controls == lens(cam).backward.controls
    @test metadata(rb).family.forward === nothing
    # refitting the fitted camera, which holds a forward: the forward is dropped and
    # refitted, the backward gauged again — already on its axis, so the second turn is the axis fit's own tolerance
    # (changed on purpose, 2026-11-11: no longer `==` the first fit's, which the second gauge turns by ~1e-10 rad)
    r2 = fit_camera(SplineLens, BoundedCamera(cam, Size2(1000, 1000)), REGION; tolerance = 1.0e-3)
    R2 = rotation(metadata(r2).rotation)
    @test lens(estimate(r2)).backward.controls == map(c -> R2 * c, lens(cam).backward.controls)
    @test maximum(abs, R2 - one(R2)) < 1.0e-8
    @test maximum(norm.(lens(estimate(r2)).forward.controls .- lens(cam).forward.controls)) < 1.0e-6
    @test retcode(r2) === :converged && metadata(r2).family.λ === nothing && metadata(r2).family.forward.max_px <= 1.0e-3
    # a kept backward has no lattice to choose
    @test_throws r"kept as it is" fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3, grid_size = (40, 40))
end

@testitem "fit_camera: the forward spline's error is non-increasing in its size" setup = [FitCameraFisheye] begin
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # the property the bisection relies on, at three spacings (the forward map it
    # replaced measured 2026-11-04: 5.1e-3, 2.7e-3, 5.4e-4 px at 8, 16, 32 cells)
    q = VgtkEstimation._chart_quadrature(GRID, REGION, VgtkEstimation.default_device())
    errs = [VgtkEstimation._forward_fit(q, c).errors.max for c in (8, 16, 32)]
    @test issorted(errs; rev = true)
    @test errs[3] < 1.0e-3 < errs[1]
end

@testitem "fit_camera(SplineLens): region shapes" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SA
    using VisualGeometryToolkitCore: estimate, metadata, retcode, SplineLens, CameraModel, BoundedCamera, Size2, KannalaBrandt, Rect, px
    using VisualGeometryToolkitEstimation: fit_camera
    # the disc cut by the frame: a non-square region, anisotropic counts
    cut = Rect(0, 150, 1000, 700) ∩ DISC
    r = fit_camera(SplineLens, GRID_BOUNDED, cut; tolerance = 1.0e-3)
    @test retcode(r) === :converged && metadata(r).family.forward.max_px <= 1.0e-3
    n = metadata(r).family.forward.counts
    @test n[1] > n[2]                                                           # wider than tall (measured 28 × 22)
    # the whole sensor of a parametric camera (corners at 81°): its backward lattice
    # and its forward, both to the tolerance
    wide = BoundedCamera(CameraModel(500.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), Size2(1000, 1000))
    r = fit_camera(SplineLens, wide, Rect(0, 0, 1000, 1000); tolerance = 1.0e-3)
    @test retcode(r) === :converged && metadata(r).max_px <= 1.0e-3 && metadata(r).family.forward.max_px <= 1.0e-3
    # the grid's whole sensor (its corners near 150°, far past the lit disc): the
    # forward reaches 1e-3 px within the bracket, where the old forward map did
    # not. The cause, measured 2026-11-10 against the old code (004a004c): the
    # chart's axis. The source is the same — the 40 × 40 controls bit-identical to
    # the old grid's, its rays equal, the same 250 000 quadrature points (none
    # dropped) and bracket (166 cells) — and fitting x instead of pixels changes
    # nothing (1.41027e-3 px either way on the old axis). At 169 × 169 the old
    # chart's axis, the ray of the quadrature's centroid (499.5, 499.5) — 1.7 mrad
    # off ẑ — reaches 1.41e-3 px; the lens axis (ẑ, the ray of the principal
    # point) reaches 5.69e-4 px.
    r = fit_camera(SplineLens, GRID_BOUNDED, Rect(0, 0, 1000, 1000); tolerance = 1.0e-3)
    @test retcode(r) === :converged && metadata(r).family.forward.max_px <= 1.0e-3
    # a forward tolerance out of reach within the bracket: the fit says so and
    # reports what it reached, the kept backward notwithstanding
    r = fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-7)
    @test retcode(r) === :max_iterations && 1.0e-7 < metadata(r).family.forward.max_px < 1.0e-3 && metadata(r).max_px < 1.0e-9
    # the quadrature is on the image: a spline target takes no weight
    @test_throws MethodError fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3, weight = :image)
end

@testitem "fit_camera(SplineLens): a backward lattice from a parametric model" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, BoundedCamera, Size2, SplineLens,
        CameraModel, KannalaBrandt, Rect, Ellipse, angle_between, calibration_matrix, lens, grid_size, px
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-4, forward = false)
    g = estimate(r)
    G = lens(g).backward.controls
    md = metadata(r)
    # the tolerance chooses the lattice: the largest spacing that meets it (measured
    # 2026-11-04: 35 × 35 nodes, 9.6e-5 px; 40 × 40 reached 5.4e-5 px)
    @test retcode(r) === :converged && iterations(r) == 0 && md.family.outer_iterations > 1
    @test md.max_px <= 1.0e-4 && md.family.tolerance == 1.0e-4
    gw, gh = md.family.grid_size
    @test gh == gw && 20 <= gh <= 40 && VgtkEstimation._lattice_grid_size(1000, 1000, gw - 3) == (gw, gh)
    # Core's full-image lattice, with the source's K
    @test size(G) == (gh, gw) && calibration_matrix(g) == calibration_matrix(FISHEYE)
    @test lens(g).backward.area == lens(CameraModel(SplineLens, FISHEYE, Size2(1000, 1000); grid_size = (gw, gh))).backward.area
    # the held-out direction error over the region: no worse than node interpolation on the same lattice
    interpolated = CameraModel(SplineLens, FISHEYE, Size2(1000, 1000); grid_size = (gw, gh))
    q = VgtkEstimation._image_quadrature(FISHEYE, REGION)
    held = VgtkEstimation._held_out(SplineLens, FISHEYE, q, g.intrinsics, VgtkCore.default_device())
    e_fit = VgtkEstimation._direction_errors(g, held)
    e_int = VgtkEstimation._direction_errors(interpolated, held)
    @test held.n_dropped == 0 && length(e_fit) == length(q.held_px) && md.family.n_held_out_dropped == 0
    @test maximum(e_fit) <= maximum(e_int)
    @test maximum(e_fit) ≈ md.max_px
    # the held-out error has no solver floor: a tolerance far below what the finest
    # lattice attains is out of reach, never a false convergence (measured
    # 2026-11-04: 169 × 169 reaches 1.9e-6 px)
    rf = fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-12, forward = false)
    @test retcode(rf) === :max_iterations && metadata(rf).max_px > 1.0e-9 && metadata(rf).family.grid_size == (169, 169)
    # the nodes outside the region: finite, and smooth — the unit controls' third
    # differences along both axes, below node interpolation's (ringing in the
    # corners); the nodes whose support meets the region within its bounding cone
    # ± 30° (measured 109° against 97° at 40 × 40). The full-image lattice reaches
    # the image corners, ~150° off axis for this lens, so the cone bound is for
    # those nodes only.
    @test all(c -> all(isfinite, c), G)
    unit(G, k) = [normalize(SVector{3}(G[i, j]))[k] for i in axes(G, 1), j in axes(G, 2)]
    d3(A) = max(maximum(abs, diff(diff(diff(A; dims = 1); dims = 1); dims = 1)), maximum(abs, diff(diff(diff(A; dims = 2); dims = 2); dims = 2)))
    @test maximum(k -> d3(unit(G, k)), 1:3) < 0.01
    @test maximum(k -> d3(unit(G, k)), 1:3) < maximum(k -> d3(unit(lens(interpolated).backward.controls, k)), 1:3)
    axis = SVector(0.0, 0.0, 1.0)
    cone = maximum(f -> angle_between(f, axis), q.fit_rays)
    h = 1000 / (gw - 3)                                                          # the lattice's cell
    near = [norm(SVector((gx - 2) * h, (gy - 2) * h) - SVector(500.0, 500.0)) <= 480 + 2h for gy in 1:gh, gx in 1:gw]
    @test maximum(angle_between(SVector{3}(G[i]), axis) for i in findall(near)) <= cone + deg2rad(30)
    # a tolerance out of reach at the finest lattice (a 60 px image: 13 × 13
    # nodes at most) says so and returns what it reached, never warns only
    small = BoundedCamera(CameraModel(18.0px, SA[30.0px, 30.0px]; lens = KannalaBrandt(-0.02)), Size2(60, 60))
    ru = fit_camera(SplineLens, small, Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0); tolerance = 1.0e-6, forward = false)
    @test retcode(ru) === :max_iterations && metadata(ru).max_px > 1.0e-6 && metadata(ru).family.grid_size == (13, 13)
    # a given lattice: no search, the tolerance read off it
    rg = fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-4, grid_size = (40, 40), forward = false)
    @test metadata(rg).family.grid_size == (40, 40) && metadata(rg).family.outer_iterations == 0 && retcode(rg) === :converged
    # a non-square one: given as (columns, rows) and reported in that order, Core's grid_size's
    rn = fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-4, grid_size = (40, 30), forward = false)
    @test metadata(rn).family.grid_size == grid_size(estimate(rn)) == (40, 30) && size(lens(estimate(rn)).backward.controls) == (30, 40)
    # a bare camera model has no image for the full-image lattice
    @test_throws r"no image" fit_camera(SplineLens, FISHEYE, REGION; tolerance = 0.01)
    @test_throws ArgumentError fit_camera(SplineLens, bc, REGION; tolerance = 0.0)
    # fewer than 4 nodes on an axis: a named ArgumentError
    @test_throws r"at least 4 × 4 nodes" VgtkEstimation._penalized_solve(Tuple{SVector{16, Int}, SVector{16, Float64}}[], zeros(0, 2), 3, 9, Val(2))
end

@testitem "fit_camera: held-out pixels without a finite source Jacobian are dropped and counted" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SMatrix, SVector
    using LinearAlgebra: normalize
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: estimate, BoundedCamera, Size2, SplineLens, backproject_with_jacobian
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    # a source that images every ray but cannot differentiate at a few pixels
    struct NaNJacobianSource
        model::Any
        bad::Set{Any}
    end
    VgtkCore.backproject_with_jacobian(s::NaNJacobianSource, u) =
        u in s.bad ? (first(backproject_with_jacobian(s.model, u)), fill(NaN, SMatrix{3, 2, Float64})) : backproject_with_jacobian(s.model, u)
    VgtkCore.backproject_with_jacobian(s::NaNJacobianSource, us::AbstractVector{<:SVector{2}}; device) = map(u -> backproject_with_jacobian(s, u), us)
    g = estimate(fit_camera(SplineLens, BoundedCamera(FISHEYE, Size2(1000, 1000)), REGION; tolerance = 1.0e-3, forward = false))
    q = VgtkEstimation._image_quadrature(FISHEYE, REGION)
    bad = [3, 10, 50]
    src = NaNJacobianSource(FISHEYE, Set{Any}(q.held_px[bad]))
    held_all = VgtkEstimation._held_out(SplineLens, FISHEYE, q, g.intrinsics, VgtkCore.default_device())
    held = VgtkEstimation._held_out(SplineLens, src, q, g.intrinsics, VgtkCore.default_device())
    kept = setdiff(eachindex(q.held_px), bad)
    @test held_all.n_dropped == 0 && held.n_dropped == 3 && held.px == q.held_px[kept] && held.edge == q.edge[kept]
    e_all = VgtkEstimation._direction_errors(g, held_all)
    e = VgtkEstimation._direction_errors(g, held)
    @test all(isfinite, e)
    @test e ≈ e_all[kept] rtol = 1.0e-12     # (the wrapper's call compiles apart: the last bits may differ)
    errors, _ = VgtkEstimation._grid_errors(g, held, nothing)
    @test isfinite(errors.max) && errors.max == maximum(e)
    @test_throws r"not finite at any held-out pixel" VgtkEstimation._held_out(SplineLens, NaNJacobianSource(FISHEYE, Set{Any}(q.held_px)), q, g.intrinsics, VgtkCore.default_device())
end

@testitem "fit_camera: a non-central model's line offsets drop the held-out pixels its direction errors drop" setup = [FitCameraNoncentral] begin
    using Test
    using StaticArrays: SMatrix, SVector
    using LinearAlgebra: norm, dot
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: estimate, NoncentralSplineModel, backproject_with_jacobian
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    struct NaNJacobianLines
        model::Any
        bad::Set{Any}
    end
    VgtkCore.backproject_with_jacobian(s::NaNJacobianLines, u) =
        u in s.bad ? (first(backproject_with_jacobian(s.model, u)), fill(NaN, SMatrix{3, 2, Float64})) : backproject_with_jacobian(s.model, u)
    VgtkCore.backproject_with_jacobian(s::NaNJacobianLines, us::AbstractVector{<:SVector{2}}; device) = map(u -> backproject_with_jacobian(s, u), us)
    fitted = estimate(fit_camera(NoncentralSplineModel, SOURCE, REGION; tolerance = 1.0e-3))
    q = VgtkEstimation._image_quadrature(VgtkEstimation._lines_on(VgtkEstimation.default_device()), MODEL, REGION)
    xs = VgtkEstimation._normalized_points(fitted.intrinsics, q.held_px)
    offset(i) = VgtkEstimation._line_offset(fitted, xs[i], first(q.held_rays[i]), last(q.held_rays[i]))   # the batch's own scalar offset
    worst = partialsortperm(map(offset, eachindex(q.held_px)), 1:3; rev = true)   # the three largest offsets
    held = VgtkEstimation._held_out(NoncentralSplineModel, NaNJacobianLines(MODEL, Set{Any}(q.held_px[worst])), q, fitted.intrinsics, VgtkCore.default_device())
    direction, offsets = VgtkEstimation._line_errors(fitted, held)
    kept = setdiff(eachindex(q.held_px), worst)
    @test held.n_dropped == 3 && offsets.max < maximum(offset, worst)
    @test offsets.max == maximum(offset, kept)   # bit for bit: the offset's dot product is written out (`_line_offset`)
    # (measured 2026-11-04: 5.04e-8 m over the kept pixels; 5.10e-8 m when the dropped ones were still counted)
end

@testitem "pspline_grid: the SplineLens backward's one solve on a given lattice — fit_camera's at the lattice it chose" setup = [FitCameraFisheye] begin
    using Test
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, rms, BoundedCamera, Size2, SplineLens, lens
    using VisualGeometryToolkitEstimation: fit_camera, pspline_grid, CameraFitMetadata
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-3)
    rb = fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-3, forward = false)
    @test lens(estimate(r)).backward.controls == lens(estimate(rb)).backward.controls   # the forward does not move the backward
    gw, gh = metadata(r).family.grid_size
    p = pspline_grid(SplineLens, bc, REGION, (gw, gh))       # (columns, rows), Core's lattice convention
    @test retcode(p) === :solved && iterations(p) == 0 && metadata(p) isa CameraFitMetadata && lens(estimate(p)).forward === nothing
    @test lens(estimate(p)).backward.controls == lens(estimate(r)).backward.controls && rms(p) == rms(r) && metadata(p).max_px == metadata(r).max_px
    @test metadata(p).family == (; λ = metadata(r).family.λ, grid_size = (gw, gh), stride = metadata(r).family.stride, n_held_out_dropped = 0)
    for (cols, rows) in ((35, 25), (105, 75))
        g = pspline_grid(SplineLens, bc, REGION, (cols, rows))
        G = lens(estimate(g)).backward.controls
        @test size(G) == (rows, cols) && metadata(g).family.grid_size == (cols, rows) && retcode(g) === :solved
        @test isfinite(metadata(g).max_px) && all(c -> all(isfinite, c), G)
    end
    @test metadata(pspline_grid(SplineLens, bc, REGION, (105, 75))).max_px < metadata(pspline_grid(SplineLens, bc, REGION, (35, 25))).max_px
    @test_throws ArgumentError pspline_grid(SplineLens, FISHEYE, REGION, (35, 25))      # a bare model has no image
    @test_throws ArgumentError pspline_grid(SplineLens, bc, REGION, (3, 9))             # fewer than 4 nodes on an axis
end

@testitem "pspline_grid: the non-central model on a given lattice — fit_camera's at the lattice it chose" setup = [FitCameraNoncentral] begin
    using Test
    using VisualGeometryToolkitCore: estimate, metadata, retcode, rms, NoncentralSplineModel
    using VisualGeometryToolkitEstimation: fit_camera, pspline_grid
    r = fit_camera(NoncentralSplineModel, SOURCE, REGION; tolerance = 1.0e-3)
    gw, gh = metadata(r).family.grid_size
    p = pspline_grid(NoncentralSplineModel, SOURCE, REGION, (gw, gh))
    @test retcode(p) === :solved && estimate(p).directions.controls == estimate(r).directions.controls &&
        estimate(p).points.controls == estimate(r).points.controls
    @test metadata(p).max_px == metadata(r).max_px && metadata(p).family.max_offset == metadata(r).family.max_offset &&
        metadata(p).family.center == metadata(r).family.center && rms(p) == rms(r)
    for (cols, rows) in ((35, 25), (105, 75))
        g = pspline_grid(NoncentralSplineModel, SOURCE, REGION, (cols, rows))
        @test size(estimate(g).directions.controls) == (rows, cols) && isfinite(metadata(g).max_px) && isfinite(metadata(g).family.max_offset)
    end
end

@testitem "fit_camera(SplineLens): a function source goes through Core's source constructor, which gauges it" setup = [FitCameraFisheye] begin
    using Test
    using VisualGeometryToolkitCore: metadata, retcode, estimate, backproject, CameraModel, SplineLens, BoundedCamera, Size2, lens, rotation
    using VisualGeometryToolkitEstimation: fit_camera
    # a function of the pixel has neither an image nor a K: the fit refuses it, and
    # the route is the SplineLens camera of the function (its K the lattice's gauge),
    # whose backward the fit keeps and whose forward it fits
    f = u -> backproject(FISHEYE, u)
    @test_throws r"no image" fit_camera(SplineLens, f, REGION; tolerance = 1.0e-2)
    @test_throws r"CameraModel\(SplineLens, f, size" fit_camera(SplineLens, f, REGION; tolerance = 1.0e-2)   # the refusal names the route
    cam = CameraModel(SplineLens, f, Size2(1000, 1000); grid_size = (40, 40), init = :quasi_interpolate)
    r = fit_camera(SplineLens, BoundedCamera(cam, Size2(1000, 1000)), REGION; tolerance = 1.0e-2)
    @test retcode(r) === :converged && metadata(r).family.forward.max_px <= 1.0e-2
    # the backward is the camera's on its optical axis (changed on purpose, 2026-11-11: a gauge-K source is gauged
    # first), its controls turned by the fit's rotation exactly
    R = rotation(metadata(r).rotation)
    @test lens(estimate(r)).backward.controls == map(c -> R * c, lens(cam).backward.controls)
end

@testitem "fit_camera: a lens refitted onto its own family has exactly zero residuals" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore: CameraModel, Lens, Perspective, Equidistant, Stereographic, Division, Scaramuzza, Ellipse,
        metadata, rms
    using VisualGeometryToolkitEstimation: fit_camera
    # the regression's targets are the source's plain radii, so a source at its
    # family's anchor is reproduced exactly (a Dual radius differs in the last bits)
    region = Ellipse(SVector(320.0, 240.0), 200.0, 200.0, 0.0)
    for l in (Lens(Perspective()), Lens(Equidistant()), Lens(Stereographic()), zero(Division{2}), zero(Scaramuzza{3}))
        r = fit_camera(typeof(l), CameraModel(; pp = (320.0, 240.0), focal = 300.0, lens = l), region)
        @test metadata(r).max_px == 0 && rms(r) == 0
    end
end

@testitem "fit_camera: a non-central model from a non-central source" setup = [FitCameraNoncentral] begin
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, normalize, dot
    using VisualGeometryToolkitCore: estimate, metadata, retcode, iterations, rms, NoncentralSplineModel, ChartSpline, backproject,
        nearest_point, calibration_matrix
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    r = fit_camera(NoncentralSplineModel, SOURCE, REGION; tolerance = 1.0e-3)
    g = estimate(r)
    md = metadata(r)
    # the tolerance chooses the lattice (measured 2026-11-04: 26 × 26 nodes)
    @test retcode(r) === :converged && iterations(r) == 0 && md.family.outer_iterations > 1
    gw, gh = md.family.grid_size
    # Core's full-image lattice with the source's K
    @test g isa NoncentralSplineModel && g.intrinsics.K == MODEL.intrinsics.K && g.directions.area == MODEL.directions.area
    @test size(g.directions.controls) == size(g.points.controls) == (gh, gw)
    # the anchor: Core's nearest point of every quadrature line of the source
    q = VgtkEstimation._image_quadrature(VgtkEstimation._lines_on(VgtkEstimation.default_device()), MODEL, REGION)
    @test md.family.center ≈ nearest_point([o => d for (o, d) in q.fit_rays])
    # held out, interior and edge band (measured 2026-11-04 at 26 × 26: direction
    # 9.2e-4 px max, 1.3e-4 px rms, the max at the edge; line offset 5.1e-8 m max
    # against a 3 mm field; at 40 × 40 7.1e-4 px and 7.9e-9 m) — and far better
    # than sampling the source's lines at the nodes
    @test md.max_px <= 1.0e-3 && rms(r) < 5.0e-4 && md.max_px_edge <= md.max_px
    @test md.family.max_offset < 1.0e-7 && md.family.rms_offset < 5.0e-8
    h = 1000 / (gw - 3)
    lines = [backproject(MODEL, SVector(clamp((j - 2) * h, 0.0, 999.0), clamp((i - 2) * h, 0.0, 999.0))) for i in 1:gh, j in 1:gw]
    sampled = NoncentralSplineModel(
        g.intrinsics, ChartSpline(g.directions.area, [SVector{3, Float64}(l.direction) for l in lines]),
        ChartSpline(g.directions.area, [SVector{3, Float64}(l.origin) for l in lines])
    )
    e_s, o_s = VgtkEstimation._line_errors(sampled, VgtkEstimation._held_out(NoncentralSplineModel, MODEL, q, g.intrinsics, VgtkCore.default_device()))
    @test md.max_px < e_s.max / 100 && md.family.max_offset < o_s.max / 100
    # the rays at and past 90° (to ~97°): the fitted lines stay on the source's
    # (measured 2.0e-6 rad, 7.2e-9 m)
    θ = [acos(clamp(d[3], -1, 1)) for (_, d) in q.held_rays]
    grazing = findall(>=(π / 2), θ)
    @test length(grazing) > 1000 && rad2deg(maximum(θ)) > 95
    xs = VgtkEstimation._normalized_points(g.intrinsics, q.held_px)
    angle(i) = acos(clamp(dot(normalize(SVector{3}(g.directions(xs[i]))), q.held_rays[i][2]), -1, 1))
    function offset(i)
        o, d = q.held_rays[i]
        Δ = SVector{3}(g.points(xs[i])) - o
        return norm(Δ - dot(Δ, d) * d)
    end
    @test maximum(angle, grazing) < 1.0e-4
    @test maximum(offset, grazing) < 1.0e-7
    # frame in = frame out: translating every source line by t translates the
    # fitted points and the anchor by t (absolute points, not relative to c*)
    t = SVector(0.5, -0.25, 2.0)
    rt = fit_camera(NoncentralSplineModel, noncentral_source(t), REGION; tolerance = 1.0e-3)
    @test metadata(rt).family.grid_size == md.family.grid_size
    @test maximum(norm.(estimate(rt).points.controls .- g.points.controls .- Ref(t))) < 1.0e-9
    @test norm(metadata(rt).family.center - md.family.center - t) < 1.0e-12
    @test estimate(rt).directions.controls ≈ g.directions.controls
    # a tolerance out of reach at the finest lattice (169 × 169 for 1000 px)
    # returns what it reached (measured 4.3e-5 px), never throws
    ru = fit_camera(NoncentralSplineModel, SOURCE, REGION; tolerance = 1.0e-9)
    @test retcode(ru) === :max_iterations && 1.0e-9 < metadata(ru).max_px < 1.0e-3
    @test metadata(ru).family.grid_size == (169, 169)
    # the bare model has no image: bound it
    @test_throws r"no image" fit_camera(NoncentralSplineModel, MODEL, REGION; tolerance = 1.0e-3)
end

@testitem "fit_camera: a non-central model from a central source" setup = [FitCameraNoncentral] begin
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: estimate, metadata, retcode, NoncentralSplineModel, SplineLens, BoundedCamera, Size2,
        backproject, CameraModel, KannalaBrandt, Rect, Ellipse, px
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = fit_camera(NoncentralSplineModel, bc, REGION; tolerance = 1.0e-3)
    g = estimate(r)
    @test retcode(r) === :converged
    # every line through the projection centre: c* is the centre and every
    # point is c* (offsets ≡ 0)
    @test metadata(r).family.center == SVector(0.0, 0.0, 0.0)
    @test all(iszero, g.points.controls)
    @test metadata(r).family.max_offset == 0
    # the directions as the SplineLens target fits its backward: the same lattice
    # (measured 21 × 21, 8.5e-4 px)
    @test metadata(r).max_px <= 1.0e-3
    @test metadata(r).family.grid_size == metadata(fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-3, forward = false)).family.grid_size
    # out of reach at the finest lattice of a 60 px image: said, not warned
    small = BoundedCamera(CameraModel(18.0px, SA[30.0px, 30.0px]; lens = KannalaBrandt(-0.02)), Size2(60, 60))
    ru = fit_camera(NoncentralSplineModel, small, Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0); tolerance = 1.0e-6)
    @test retcode(ru) === :max_iterations && metadata(ru).max_px > 1.0e-6
    # the batched central source lines are the scalar back-projection, normalized
    pix = [SVector(500.5, 400.5), SVector(20.5, 500.5), SVector(950.5, 980.5)]
    @test VgtkEstimation._source_lines(FISHEYE, pix, VgtkEstimation.default_device()) ==
        [(SVector(0.0, 0.0, 0.0), normalize(SVector{3, Float64}(backproject(FISHEYE, p)))) for p in pix]
    @test_throws ArgumentError fit_camera(NoncentralSplineModel, FISHEYE, REGION; tolerance = 0.01)
end

@testitem "fit_camera: a parametric lens from a SplineLens camera, through the camera's rays (the exporters' route)" setup = [FitCameraFisheye] begin
    using Test
    using VisualGeometryToolkitCore: estimate, metadata, lens, rms, parameter_vector, calibration_matrix, KannalaBrandt, rotation
    using LinearAlgebra: I
    using VisualGeometryToolkitEstimation: fit_camera
    # A SplineLens camera has no OpenCV/COLMAP form: the exporters' refusal (VisualGeometryToolkitIO) says to fit a
    # parametric lens to it first, `fit_camera(KannalaBrandt{4}, camera, region)` — the bounded camera or its model.
    # Its K is a gauge, so its lens has no radial profile of its own: the fit samples the camera's rays. GRID is the
    # 40 × 40 SplineLens camera of a Kannala–Brandt fisheye (k₁ = −0.02, f = 300 px, its K kept).
    for src in (GRID_BOUNDED, GRID)
        r = fit_camera(KannalaBrandt{4}, src, REGION)
        kb = estimate(r)
        @test lens(kb) isa KannalaBrandt{4}
        @test maximum(abs.(collect(parameter_vector(lens(kb))) .- [-0.02, 0.0, 0.0, 0.0])) <= 1.0e-3
        @test abs(metadata(r).family.focal_scale - 1) <= 1.0e-3
        @test rms(r) < 0.05 && metadata(r).max_px < 0.5
        # a gauge K's principal point is fitted (changed on purpose, 2026-11-11: it was kept, the pixel imaging ẑ): GRID
        # is the fisheye's own spline, so it lands on the fisheye's (500, 500) to the spline's approximation
        @test maximum(abs, calibration_matrix(kb)[1:2, 3] - calibration_matrix(GRID)[1:2, 3]) < 1.0e-5
        @test maximum(abs, rotation(metadata(r).rotation) - I) < 1.0e-8
    end
    # the image samples carry the :image density only
    @test_throws ArgumentError fit_camera(KannalaBrandt{4}, GRID, REGION; weight = :solid_angle)
end

@testitem "fit_camera(SplineLens): the forward is fitted on the chart project evaluates it on" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: estimate, lens, SplineLens, default_device
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: fit_camera
    cam = estimate(fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3))
    q = VgtkEstimation._chart_quadrature(cam, REGION, default_device())
    @test q.chart == VgtkCore.forward_chart(SplineLens) == VgtkCore.equidistant_chart(SVector(0.0, 0.0, 1.0))
end

@testitem "fit_camera(SplineLens), fit_camera(NoncentralSplineModel): the source's batched rays on `device`" setup = [FitCameraFisheye] begin
    using Test
    using VisualGeometryToolkitCore: estimate, lens, SplineLens, NoncentralSplineModel, default_device
    using VisualGeometryToolkitEstimation: fit_camera, pspline_grid
    a = estimate(fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3, device = default_device()))
    b = estimate(fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3))
    @test lens(a).forward.controls == lens(b).forward.controls
    # the keyword reaches the batched backprojection (a backend is a KernelAbstractions.Backend)
    @test_throws TypeError fit_camera(SplineLens, GRID_BOUNDED, REGION; tolerance = 1.0e-3, device = :none)
    @test_throws TypeError fit_camera(NoncentralSplineModel, GRID_BOUNDED, REGION; tolerance = 1.0e-3, device = :none)
    @test_throws TypeError pspline_grid(SplineLens, GRID_BOUNDED, REGION, (12, 12); device = :none)
end

# =============================================================================
# CameraRegression, the camera-regression refine, the fit-then-refine recipe — one model-to-model regression
# over a region of the sensor
# =============================================================================
#
# A lens family's approximation (the fit-then-refine recipe: CameraRegression's linear start,
# then the camera-regression refine's chart LM) reproduces the profile regression it replaced
# (recorded), with the region's exact 1-D reduction; CameraRegression alone is one
# linear least squares, its residual the returned camera's; the forward spline of
# the synthetic fisheye's SplineLens camera meets its tolerance on held-out
# points, edge band included, its backward kept exactly, and its size bisection
# rests on a monotone error; a SplineLens backward fitted from a parametric model
# beats node interpolation and stays smooth outside the region.

@testitem "the fit-then-refine recipe: a lens family reproduces the recorded profile regressions" setup = [CameraRecipes, FitCameraLensCases] begin
    using Test
    using VisualGeometryToolkitCore: lens, parameter_vector
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, CameraFitMetadata
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    worst = 0.0
    for (name, (L, m, _, region)) in LENS_CASES
        name in MOVED && continue
        r = solve(EstimationProblem(CameraRegressionProblem(L, m, region)), CameraRegression(; refine = LevenbergMarquardt()))
        @test r.metadata isa CameraFitMetadata
        p, s, max_px = RECORDED[name]
        q = parameter_vector(lens(r.u))
        @test agrees(q, p)
        @test abs(r.metadata.family.focal_scale - s) <= 1.0e-6
        @test agrees_px(r.metadata.max_px, max_px)
        @test 0 <= r.rms <= r.metadata.max_px
        isempty(p) || (worst = max(worst, maximum(abs.(collect(q) .- collect(p)))))
    end
    @test worst < 1.0e-6                          # the recorded LM from the zero anchor against the refine from the linear start
end

@testitem "the fit-then-refine recipe: the recorded perspective-polynomial stalls are refused (changed on purpose, 2026-11-11)" setup = [CameraRecipes, FitCameraLensCases] begin
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # Lens{Perspective, Radial{Even, 3, 0, Float64}} over a 155° field: the recorded LM never left its zero start (925 672 px at the edge). A perspective
    # gnomonic family applies its distortion on the perspective plane, which ends at 90°: the linear fit, and so the
    # approximation, refuse the field, naming the plane and the region's extent, instead of fitting a clipped field
    for name in MOVED
        L, m, _, region = Dict(LENS_CASES)[name]
        @test_throws VgtkEstimation.ViewAngleLimitError solve(EstimationProblem(CameraRegressionProblem(L, m, region)), CameraRegression())
        @test_throws VgtkEstimation.ViewAngleLimitError solve(EstimationProblem(CameraRegressionProblem(L, m, region)), CameraRegression(; refine = LevenbergMarquardt()))
        @test_throws r"perspective plane" solve(EstimationProblem(CameraRegressionProblem(L, m, region)), CameraRegression())
    end
end

@testitem "CameraRegression: the ImageMeasure() density's dr/dθ is exact — the central difference it replaced to its truncation error" setup = [FitCameraLensCases] begin
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

@testitem "CameraRegression: the profile regression's pieces" setup = [FitCameraLensCases] begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: lens, parameters, Lens, LiftLens, KannalaBrandt, Division, Orthographic, Equidistant, Perspective, Radial, Even, DoubleSphere, Additive, Tangential, ThinPrism, radial_part, view_angle_limit, CameraModel
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    @testset "Clenshaw–Curtis integrates polynomials" begin
        θ, q = VgtkEstimation._clenshaw_curtis(32, 1.3)
        @test sum(q) ≈ 1.3 rtol = 1.0e-12
        @test sum(q .* θ .^ 2) ≈ 1.3^3 / 3 rtol = 1.0e-10
        @test sum(q .* θ .^ 5) ≈ 1.3^6 / 6 rtol = 1.0e-8
    end
    @testset "the family limits are VGC's own view-angle limits" begin
        @test VgtkEstimation._family_limit(Lens{Perspective, Nothing}) == view_angle_limit(Lens(Perspective())) == π / 2 - 1.0e-3
        @test VgtkEstimation._family_limit(KB4) == view_angle_limit(Lens(Equidistant())) == π - 1.0e-3
        @test VgtkEstimation._family_limit(typeof(Lens(Orthographic()))) == view_angle_limit(Lens(Orthographic()))
        @test VgtkEstimation._family_limit(typeof(Lens(DoubleSphere(0.5, 0.5)))) == π - 1.0e-3
    end
    @testset "a region past the target family's view-angle limit is refused" begin
        # refused: the perspective plane ends at 90° (changed on purpose, 2026-11-11: it was fitted over the clipped
        # field with a warning)
        @test_throws r"perspective plane \(θ < 90°\), but the region reaches 154.9°" solve(EstimationProblem(CameraRegressionProblem(Lens{Perspective, Radial{Even, 3, 0, Float64}}, mdiv, disc(mdiv, rcorner))), CameraRegression())
        @test_throws r"its view-angle limit" solve(EstimationProblem(CameraRegressionProblem(typeof(Lens(Orthographic())), mdiv, disc(mdiv, rcorner))), CameraRegression())
        # a typed refusal: its fields carry the measured angle and the family's limit (radians), read without the text
        e = try solve(EstimationProblem(CameraRegressionProblem(Lens{Perspective, Radial{Even, 3, 0, Float64}}, mdiv, disc(mdiv, rcorner))), CameraRegression()); nothing catch err; err end
        @test e isa VgtkEstimation.ViewAngleLimitError && e.family === Lens{Perspective, Radial{Even, 3, 0, Float64}}
        @test e.limit == view_angle_limit(Lens(Perspective())) && round(rad2deg(e.field_angle); digits = 1) == 154.9
        @test e.field_angle > e.limit
    end
    @testset "decentering: a target holds it at zero, a source's is stripped and warned" begin
        rd = solve(EstimationProblem(CameraRegressionProblem(typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0)))), mdiv, disc(mdiv, 1000.0))), CameraRegression())
        cd = parameters(lens(rd.u))
        @test cd.p1 == 0 && cd.p2 == 0 && cd.d1 != 0          # the rational factor's denominator is linear: fitted
        synthetic = Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)), Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))
        @test VgtkEstimation._decentering(VgtkEstimation.VgtkCore.Cameras.RadialSymmetry(synthetic), synthetic) ≈ hypot(1.0e-3, -5.0e-4, -5.5e-3, 7.0e-4, 2.2e-3, -1.2e-3)
        @test radial_part(synthetic) === Lens(Perspective(), Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)))
        src = CameraModel(; pp = (960.0, 540.0), focal = 900.0, lens = synthetic)
        @test_logs (:warn, r"non-radial terms") match_mode = :any solve(EstimationProblem(CameraRegressionProblem(Lens{Perspective, Radial{Even, 3, 0, Float64}}, src, disc(src, 500.0))), CameraRegression())
    end
    @testset "Division alone is the one-coefficient model; a type without its arities is refused" begin
        @test lens(solve(EstimationProblem(CameraRegressionProblem(Division, mdiv, disc(mdiv, rcorner))), CameraRegression()).u) == lens(solve(EstimationProblem(CameraRegressionProblem(Division{1}, mdiv, disc(mdiv, rcorner))), CameraRegression()).u)
        @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(LiftLens, mdiv, disc(mdiv, rcorner))), CameraRegression())
        @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt, mdiv, disc(mdiv, rcorner))), CameraRegression())
        @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, disc(mdiv, rcorner); weight = :pixels)), CameraRegression())
    end
end

@testitem "CameraRegression: the region's share of each radius" setup = [FitCameraLensCases] begin
    using Test
    using StaticArrays: SMatrix, SVector
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: lens, parameter_vector, calibration_matrix, Rect, WholePlane
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    K = SMatrix{3, 3, Float64}(calibration_matrix(mdiv))
    # the exit radius per azimuth of the normalized plane, as CameraRegression reads it: Core's farthest_radius along A·d
    pp, A = SVector(K[1, 3], K[2, 3]), SMatrix{2, 2}(K[1, 1], K[2, 1], K[1, 2], K[2, 2])
    exit_radii(region) = [VgtkCore.farthest_radius(region, pp, A * SVector(cospi(2 * (k - 0.5) / 720), sinpi(2 * (k - 0.5) / 720))) for k in 1:720]
    # a disc about the principal point: every azimuth leaves it at its radius
    e = exit_radii(disc(mdiv, 1000.0))
    @test maximum(abs.(e .- 1000.0 / 927.0)) < 1.0e-12
    # the frame: the exit radius along an azimuth is the distance to the frame's edge
    frame = Rect(0, 0, 4240, 2832)
    e = exit_radii(frame)
    ψ = [2π * (k - 0.5) / length(e) for k in eachindex(e)]
    edge = [min(2120.0 / abs(cos(a)), 1416.0 / abs(sin(a))) / 927.0 for a in ψ]
    @test maximum(abs.(e .- edge)) < 1.0e-9
    @test VgtkEstimation._coverage(e, 1000.0 / 927.0) == 1.0            # inside the frame on every azimuth
    @test 0 < VgtkEstimation._coverage(e, 2000.0 / 927.0) < 1           # past the frame's half height, not its half width
    @test VgtkEstimation._coverage(e, 1.01 * rcorner / 927.0) == 0.0
    # the frame against the disc through its corners: the same domain (to the azimuth
    # sampling, which reaches the corners within half an azimuth step), less weight
    # past the edges
    rf = solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, frame)), CameraRegression())
    rd = solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, disc(mdiv, rcorner))), CameraRegression())
    @test rf.metadata.family.θ_max ≈ rd.metadata.family.θ_max rtol = 1.0e-3
    @test parameter_vector(lens(rf.u)) != parameter_vector(lens(rd.u))
    @test rf.metadata.max_px_edge <= rf.metadata.max_px
    # a region that misses the principal point, the whole plane: refused
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, Rect(0, 0, 100, 100))), CameraRegression())
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(KB4, mdiv, WholePlane())), CameraRegression())
end

@testitem "CameraRegression: a region past the source's fold is refused (RegionBeyondSupport), never clipped" begin
    using Test
    using StaticArrays: SA
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Rect, RegionBeyondSupport, valid_domain, px
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # KB(−0.02) at f = 200 px images radii to r(π) ≈ 2.52 (504 px); the frame's corners lie at 707 px, where the source
    # has no view angle. Until 0.11.0 the fit took the source's own domain, clipped, with a warning; the owner's ruling
    # of 2026-11-13: refuse, never warn — the region reaches past the source's valid_domain (Core's guard, at entry)
    src = CameraModel(200.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    e = try solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, src, Rect(0, 0, 1000, 1000))), CameraRegression()); nothing catch err; err end
    @test e isa RegionBeyondSupport && e.who == "CameraRegression" && e.support == valid_domain(src)
end


@testitem "CameraRegression(SplineLens): a SplineLens source keeps its backward, on its optical axis, and gains a forward spline" setup = [FitCameraFisheye] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: ChartSpline, SplineLens, CameraModel, BoundedCamera, Size2, project, backproject, calibration_matrix, lens, rotation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, CameraFitMetadata, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-3))
    cam = r.u
    @test cam isa CameraModel && lens(cam) isa SplineLens && lens(cam).forward isa ChartSpline{2}
    @test r.metadata isa CameraFitMetadata
    @test retcode_symbol(r) === :converged
    # the backward is the source's on its optical axis (2026-11-11, changed on purpose: a gauge-K source is gauged
    # first, `OpticalAxis` then Core's `axis_gauged`): its controls turned by the fit's rotation, exactly, its K's
    # principal point moved to the axis's pixel. GRID is the fisheye's own spline, so the turn is its approximation
    # error alone. The axis is fitted over the source's bound, GRID_BOUNDED's whole 1000 × 1000 image (2026-11-11,
    # owner: the bound, never the fit's region): its corners lie past the disc, so the turn measured 1.3e-7
    R = rotation(r.metadata.rotation)
    @test lens(cam).backward.controls == map(c -> R * c, lens(GRID).backward.controls)
    @test maximum(abs, R - one(R)) < 1.0e-6
    # its K is the symmetric camera's (2026-11-11, owner: the re-gauge takes the fit's principal point and aspect):
    # its principal point GRID's own to the spline's approximation. Its focal is the fitted family's scale, a gauge
    # choice (any skew-0 K is an exact gauge of a SplineLens camera), traded against the radial terms over this
    # whole-image bound (corners to 164°): 299.99977 px at 6 terms, 300.00290 at the axis family's 4 (0.11.0), so the
    # focal is not GRID's to 1e-3 px and is not asserted here; the principal point moves by 4e-7 px between the two.
    K, KG = calibration_matrix(cam), calibration_matrix(GRID)
    @test K == calibration_matrix(solve(EstimationProblem(CameraRegressionProblem(GRID_BOUNDED)), OpticalAxis()).u)
    @test maximum(abs, K[1:2, 3] - KG[1:2, 3]) < 1.0e-3
    fam = r.metadata.family
    @test fam.λ === nothing && fam.grid_size == (40, 40) && fam.outer_iterations == 0
    @test r.metadata.max_px < 1.0e-9                                           # the kept backward against itself
    # the forward's held-out maximum, the edge band's included (the old forward map
    # it replaced measured 2026-11-04: 9.2e-4 px, in the band; rms 1.0e-4 px; 28 × 28 controls)
    @test fam.forward.max_px <= 1.0e-3
    @test fam.forward.max_px_edge <= 1.0e-3
    @test fam.forward.rms < 2.0e-4
    @test fam.forward.counts[1] == fam.forward.counts[2]                                        # a full disc: isotropic counts
    @test fam.tolerance == 1.0e-3 && r.stats.nsteps == 0 && fam.forward.outer_iterations > 1
    # the round trip at fresh points of the region, through the public verbs: the
    # camera projects through its forward spline
    pts = [SVector(500.0 + ρ * cos(a), 500.0 + ρ * sin(a)) for ρ in (0.0, 120.0, 300.0, 470.0) for a in range(0, 2π; length = 13)]
    @test maximum(p -> norm(project(cam, backproject(GRID, p)) - p), pts) < 1.0e-3
    # a ray far outside the fitted cone (~97°) is not imaged, wherever the continuation lands
    @test all(isnan, project(cam, SVector(sind(170.0), 0.0, cosd(170.0))))
    # forward = false: the backward alone, the same controls; no forward record
    rb = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-3, forward = false))
    @test lens(rb.u).forward === nothing && lens(rb.u).backward.controls == lens(cam).backward.controls
    @test rb.metadata.family.forward === nothing
    # refitting the fitted camera, which holds a forward: the forward is dropped and
    # refitted, the backward gauged again — already on its axis, so the second turn is the axis fit's own tolerance
    # (changed on purpose, 2026-11-11: no longer `==` the first fit's, which the second gauge turns by ~1e-10 rad)
    r2 = solve(EstimationProblem(CameraRegressionProblem(SplineLens, BoundedCamera(cam, Size2(1000, 1000)), REGION)), CameraRegression(; tolerance = 1.0e-3))
    R2 = rotation(r2.metadata.rotation)
    @test lens(r2.u).backward.controls == map(c -> R2 * c, lens(cam).backward.controls)
    @test maximum(abs, R2 - one(R2)) < 1.0e-8
    @test maximum(norm.(lens(r2.u).forward.controls .- lens(cam).forward.controls)) < 1.0e-6
    @test retcode_symbol(r2) === :converged && r2.metadata.family.λ === nothing && r2.metadata.family.forward.max_px <= 1.0e-3
    # a SplineLens source is resampled exactly (0.11.6): `grid_size` chooses no lattice, `lattice_spacing` does (next item)
    @test_throws r"lattice_spacing" solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-3, grid_size = (40, 40), forward = false))
    @test rb.metadata.family.subdivision == (1, 1) && !any(rb.metadata.family.extrapolated) && rb.metadata.family.extrapolated_fraction == 0
end

# 0.11.6: a SplineLens source resampled EXACTLY (Core 0.10.8's lattice operations): its lattice extended on its own
# spacing to every node the region needs (new nodes marked, Core's bending-energy continuation), subdivided by the
# integers giving the cell closest to `lattice_spacing`, turned onto its optical axis. Inside the source's domain the
# result is the source's function to rounding (≤ 1e-9 rad), on square and non-square cells (RC's 16 × 18 and 26 × 20
# lattices on its 4256 × 2848 sensor); a region past the source's domain is defined everywhere, its extrapolated nodes
# marked; Core's fold report is recorded.
@testitem "CameraRegression(SplineLens): a SplineLens source resampled exactly — extended, subdivided, turned" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SA, SMatrix, SVector
    using LinearAlgebra: norm, normalize, dot, cross
    using VisualGeometryToolkitCore: retcode_symbol, SplineLens, CameraModel, BoundedCamera, Size2, CameraCalibrationMat, Ellipse, Rect, backproject, calibration_matrix, lens, rotation, grid_size, valid_domain, lattice_spacing, KannalaBrandt, px
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    import VisualGeometryToolkitEstimation as VgtkEstimation
    angle(a, b) = atan(norm(cross(a, b)), dot(a, b))
    # the direct angle between the result and the turned source, over the pixels of `region` the source images
    function max_angle(src, r, region; stride = 7.0)
        R = SMatrix{3, 3, Float64}(rotation(r.metadata.rotation)); cam = r.u; ms = VgtkCore.model(src)
        W, H = VgtkCore.extent(src)
        a = 0.0
        for y in 3.0:stride:(H - 1), x in 3.0:stride:(W - 1)
            p = SVector(x, y); p ∈ region || continue
            s = backproject(ms, p); all(isfinite, s) || continue
            a = max(a, angle(SVector{3, Float64}(backproject(cam, p)), R * normalize(SVector{3, Float64}(s))))
        end
        return a
    end
    # 1. GRID (40 × 40, ~27 px cells) at a 9 px target: subdivided 3 × 3, the same function; nothing extrapolated
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-3, lattice_spacing = 9.0, forward = false))
    fam = r.metadata.family
    @test retcode_symbol(r) === :converged && fam.subdivision == (3, 3) && grid_size(r.u) == (3 * 37 + 3, 3 * 37 + 3)
    @test all(abs.(fam.lattice_spacing .- 1000 / 111) .< 1.0e-9) && !any(fam.extrapolated) && fam.extrapolated_fraction == 0
    @test max_angle(GRID_BOUNDED, r, REGION) < 1.0e-9 && r.metadata.max_px < 1.0e-6
    # 2. non-square cells: RC's lattice shapes on its sensor, one target spacing
    sensor = Size2(4256, 2848)
    lens14 = CameraModel(1727.0px, SA[2140.0px, 1415.0px]; lens = KannalaBrandt(0.02))
    for g in ((16, 18), (26, 20))
        src = BoundedCamera(CameraModel(SplineLens, lens14, sensor; grid_size = g), sensor)
        reg = Rect(0, 0, 4256, 2848)
        rs = solve(EstimationProblem(CameraRegressionProblem(SplineLens, src, reg)), CameraRegression(; tolerance = 1.0e-3, lattice_spacing = 60.0, forward = false))
        f = rs.metadata.family
        h = lattice_spacing(VgtkCore.model(src))
        # per axis an integer bracketing h / 60 (the nearer cell, or the squarer near tie), the cell h / m exactly
        @test all(max.(1, floor.(Int, h ./ 60)) .<= f.subdivision .<= ceil.(Int, h ./ 60)) && all(abs.(f.lattice_spacing .- h ./ f.subdivision) .< 1.0e-9)
        @test max_angle(src, rs, reg; stride = 37.0) < 1.0e-9
    end
    # 3. a region past the source's domain: a lattice over a calibrated area of 100 … 900 px, the region the disc of
    # 480 px; every node the region needs exists and is finite, the new ones marked, the region inside the result's
    # valid domain and imaged at every pixel, the fidelity inside the source's domain to rounding
    part = BoundedCamera(CameraModel(SplineLens, GRID, Size2(1000, 1000); grid_size = (30, 30), calib_area = Rect(100, 100, 800, 800)), Size2(1000, 1000))
    re = solve(EstimationProblem(CameraRegressionProblem(SplineLens, part, REGION)), CameraRegression(; tolerance = 1.0e-3, lattice_spacing = 20.0, forward = false))
    fe = re.metadata.family; ce = re.u
    @test retcode_symbol(re) === :converged && any(fe.extrapolated) && 0 < fe.extrapolated_fraction < 1
    @test all(c -> all(isfinite, c), lens(ce).backward.controls)
    @test REGION ⊆ valid_domain(ce)                                              # the region inside the result's domain
    @test all(p -> all(isfinite, backproject(ce, p)), (SVector(x, y) for x in 1.0:9.0:999.0, y in 1.0:9.0:999.0 if SVector(x, y) ∈ REGION))
    @test max_angle(part, re, REGION) < 1.0e-9                                   # where the source images a ray
    @test fe.folds isa NamedTuple && haskey(fe.folds, :folded)                    # Core's report, recorded
    # 4. a region within the source's domain: nothing extrapolated, nothing marked
    ri = solve(EstimationProblem(CameraRegressionProblem(SplineLens, part, Rect(200, 200, 600, 600))), CameraRegression(; tolerance = 1.0e-3, lattice_spacing = 20.0, forward = false))
    @test !any(ri.metadata.family.extrapolated) && ri.metadata.family.extrapolated_fraction == 0
    # a `support` the caller states is checked: the lattice is never extended past a re-gridded camera's calibration
    @test_throws VgtkCore.RegionBeyondSupport solve(EstimationProblem(CameraRegressionProblem(SplineLens, part, REGION; support = Rect(100, 100, 800, 800))), CameraRegression(; tolerance = 1.0e-3, lattice_spacing = 20.0, forward = false))
    # the factors: the cell aspect nearest the target's among near ties (here 20 × 10 px cells, not the squarer 20 × 10.5);
    # a pixel quantity taken, a non-positive or a length refused
    @test VgtkEstimation._resample_factors((100.0, 105.0), (20.0, 10.0)) == (5, 11)
    @test VgtkEstimation._resample_factors((27.0, 27.0), 9.0px) == (3, 3)
    @test_throws ArgumentError VgtkEstimation._resample_factors((27.0, 27.0), 0.0)
    @test_throws Exception VgtkEstimation._resample_factors((27.0, 27.0), 9.0VgtkCore.mm)
    # `lattice_spacing` is the spline source's: a parametric source takes `grid_size`
    @test_throws r"lattice_spacing" solve(EstimationProblem(CameraRegressionProblem(SplineLens, BoundedCamera(lens14, sensor), Rect(0, 0, 4256, 2848))), CameraRegression(; tolerance = 1.0e-3, lattice_spacing = 60.0))
end

@testitem "CameraRegression(SplineLens): a region on the source's lattice edge lies in the resample's domain (0.11.9)" begin
    using Test
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: SplineLens, CameraModel, BoundedCamera, Size2, Rect, Ellipse, lens, valid_domain, model, rotation, backproject
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # a Schöps central-generic source calibrated over the whole 640 × 480 sensor (7 × 6 nodes of 160 px; VisualGeometryToolkitIO's
    # reader builds it so): Core's constructor maps the area through K⁻¹ and back, so its domain's top edge is 2.8e-14 px below
    # the sensor's. The sensor is on its lattice edge (extend_lattice adds no ring), and the axis turn moved the area by K′⁻¹K
    # again; under 0.11.8's axis the resample's domain started 2.8e-14 px inside the sensor and `sensor ⊆ valid_domain` failed
    # (RadiometricCamera's exact start refused it: RegionBeyondSupport). Now each edge is the target's where a float lands on it,
    # else one output ulp outside: the domain holds the region and claims at most that rounding beyond it
    controls = [normalize(SVector(0.12 * (gx - 4), 0.12 * (gy - 3.5), 1.0)) for gy in 1:6, gx in 1:7]
    sensor = Rect(0.0, 0.0, 640.0, 480.0)
    src = BoundedCamera(CameraModel(SplineLens, controls, Size2(640, 480); calib_area = Rect(0, 0, 640, 480), unit_controls = true), Size2(640, 480))
    R(r) = SMatrix{3, 3, Float64}(rotation(r.metadata.rotation))
    edges(b) = SVector(minimum(b)..., maximum(b)...)
    # one step of K′'s map (fx·eps(x) ≈ 1340 · 2.8e-17 ≈ 3.7e-14 px: the edge found 2.8e-14 px outside 0), or one ulp of 640
    within_ulp(b, target) = maximum(abs.(edges(b) - edges(target))) <= 2 * eps(640.0)
    # (a coverage disc whose bounding box's far edge is a rounded sum, 0.1 + 639.9: the region's own bounds are Core's `⊆`'s)
    for (region, spacing) in ((sensor, nothing), (sensor, 80.0), (sensor ∩ Ellipse(SVector(320.0, 240.0), 380.0, 380.0, 0.0), nothing),
                              (sensor ∩ Ellipse(SVector(320.05, 240.0), 319.95, 400.0, 0.0), nothing))
        r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, src, region)), CameraRegression(; tolerance = Inf, forward = false, lattice_spacing = spacing))
        cam = r.u
        @test region ⊆ valid_domain(cam) && sensor ⊆ valid_domain(cam) && within_ulp(valid_domain(cam), sensor)
        # the region's edge imaged: its low corner and the corner pixels' centres back-project (the outermost area value)
        @test all(p -> all(isfinite, backproject(cam, p)), (SVector(0.0, 0.0), SVector(0.5, 0.5), SVector(639.5, 0.5), SVector(0.5, 479.5), SVector(639.5, 479.5)))
        # the source's nodes, turned, bit for bit (unsubdivided)
        spacing === nothing && @test lens(cam).backward.controls == map(c -> R(r) * c, lens(model(src)).backward.controls)
    end
    # a region strictly inside the lattice: the turn keeps the source's own domain (to an outward ulp), nothing moved onto the region
    inner = Rect(10.0, 10.0, 600.0, 440.0)
    vd = valid_domain(solve(EstimationProblem(CameraRegressionProblem(SplineLens, src, inner)), CameraRegression(; tolerance = Inf, forward = false)).u)
    @test valid_domain(model(src)) ⊆ vd && within_ulp(vd, valid_domain(model(src)))
end

@testitem "CameraRegression: the forward spline's error is non-increasing in its size" setup = [FitCameraFisheye] begin
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # the property the bisection relies on, at three spacings (the forward map it
    # replaced measured 2026-11-04: 5.1e-3, 2.7e-3, 5.4e-4 px at 8, 16, 32 cells)
    q = VgtkEstimation._chart_quadrature(GRID, REGION, VgtkEstimation.default_device())
    errs = [VgtkEstimation._forward_fit(q, c).errors.max for c in (8, 16, 32)]
    @test issorted(errs; rev = true)
    @test errs[3] < 1.0e-3 < errs[1]
end

@testitem "CameraRegression(SplineLens): region shapes" setup = [FitCameraFisheye] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SA
    using VisualGeometryToolkitCore: SplineLens, CameraModel, BoundedCamera, Size2, KannalaBrandt, Rect, px
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, ImageMeasure
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # the disc cut by the frame: a non-square region, anisotropic counts
    cut = Rect(0, 150, 1000, 700) ∩ DISC
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, cut)), CameraRegression(; tolerance = 1.0e-3))
    @test retcode_symbol(r) === :converged && r.metadata.family.forward.max_px <= 1.0e-3
    n = r.metadata.family.forward.counts
    @test n[1] > n[2]                                                           # wider than tall (measured 28 × 22)
    # the whole sensor of a parametric camera (corners at 81°): its backward lattice
    # and its forward, both to the tolerance
    wide = BoundedCamera(CameraModel(500.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), Size2(1000, 1000))
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, wide, Rect(0, 0, 1000, 1000))), CameraRegression(; tolerance = 1.0e-3))
    @test retcode_symbol(r) === :converged && r.metadata.max_px <= 1.0e-3 && r.metadata.family.forward.max_px <= 1.0e-3
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
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, Rect(0, 0, 1000, 1000))), CameraRegression(; tolerance = 1.0e-3))
    @test retcode_symbol(r) === :converged && r.metadata.family.forward.max_px <= 1.0e-3
    # a forward tolerance out of reach within the bracket: the fit says so and
    # reports what it reached, the kept backward notwithstanding
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-7))
    @test retcode_symbol(r) === :max_iterations && 1.0e-7 < r.metadata.family.forward.max_px < 1.0e-3 && r.metadata.max_px < 1.0e-9
    # the quadrature is on the image: a spline target takes no weight
    @test_throws MethodError solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION; weight = ImageMeasure())), CameraRegression(; tolerance = 1.0e-3))
end

@testitem "CameraRegression(SplineLens): a backward lattice from a parametric model" setup = [FitCameraFisheye] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: BoundedCamera, Size2, SplineLens, CameraModel, KannalaBrandt, Rect, Ellipse, angle_between, calibration_matrix, lens, grid_size, px
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-4, forward = false))
    g = r.u
    G = lens(g).backward.controls
    md = r.metadata
    # the tolerance chooses the lattice: the largest spacing that meets it (measured
    # 2026-11-04: 35 × 35 nodes, 9.6e-5 px; 40 × 40 reached 5.4e-5 px)
    @test retcode_symbol(r) === :converged && r.stats.nsteps == 0 && md.family.outer_iterations > 1
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
    rf = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-12, forward = false))
    @test retcode_symbol(rf) === :max_iterations && rf.metadata.max_px > 1.0e-9 && rf.metadata.family.grid_size == (169, 169)
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
    ru = solve(EstimationProblem(CameraRegressionProblem(SplineLens, small, Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0))), CameraRegression(; tolerance = 1.0e-6, forward = false))
    @test retcode_symbol(ru) === :max_iterations && ru.metadata.max_px > 1.0e-6 && ru.metadata.family.grid_size == (13, 13)
    # a given lattice: no search, the tolerance read off it
    rg = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-4, grid_size = (40, 40), forward = false))
    @test rg.metadata.family.grid_size == (40, 40) && rg.metadata.family.outer_iterations == 0 && retcode_symbol(rg) === :converged
    # a non-square one: given as (columns, rows) and reported in that order, Core's grid_size's
    rn = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-4, grid_size = (40, 30), forward = false))
    @test rn.metadata.family.grid_size == grid_size(rn.u) == (40, 30) && size(lens(rn.u).backward.controls) == (30, 40)
    # a bare camera model has no image for the full-image lattice
    @test_throws r"no image" solve(EstimationProblem(CameraRegressionProblem(SplineLens, FISHEYE, REGION)), CameraRegression(; tolerance = 0.01))
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 0.0))
    # fewer than 4 nodes on an axis: a named ArgumentError
    @test_throws r"at least 4 × 4 nodes" VgtkEstimation._penalized_solve(Tuple{SVector{16, Int}, SVector{16, Float64}}[], zeros(0, 2), 3, 9, Val(2))
end

@testitem "CameraRegression: the held-out errors read no back-projection Jacobian (0.11.6)" setup = [FitCameraFisheye] begin
    # Until 0.11.6 a held-out pixel where the source's Jacobian was not finite was dropped and counted: the error was a
    # pixel mapping through the Jacobian. Now the error is the angle to the source's ray, f·angle: a source whose
    # Jacobian is NaN everywhere has every held-out pixel's error, finite, and nothing dropped.
    using Test
    using StaticArrays: SMatrix, SVector
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: BoundedCamera, Size2, SplineLens, backproject_with_jacobian
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    struct NaNJacobian
        model::Any
    end
    VgtkCore.backproject_with_jacobian(s::NaNJacobian, u) = (first(backproject_with_jacobian(s.model, u)), fill(NaN, SMatrix{3, 2, Float64}))
    VgtkCore.backproject_with_jacobian(s::NaNJacobian, us::AbstractVector{<:SVector{2}}; device) = map(u -> backproject_with_jacobian(s, u), us)
    g = solve(EstimationProblem(CameraRegressionProblem(SplineLens, BoundedCamera(FISHEYE, Size2(1000, 1000)), REGION)), CameraRegression(; tolerance = 1.0e-3, forward = false)).u
    q = VgtkEstimation._image_quadrature(FISHEYE, REGION)
    held = VgtkEstimation._held_out(SplineLens, NaNJacobian(FISHEYE), q, g.intrinsics, VgtkCore.default_device())
    @test held.n_dropped == 0 && held.px == q.held_px
    e = VgtkEstimation._direction_errors(g, held)
    @test length(e) == length(q.held_px) && all(isfinite, e)
    @test e == VgtkEstimation._direction_errors(g, VgtkEstimation._held_out(SplineLens, FISHEYE, q, g.intrinsics, VgtkCore.default_device()))
end

@testitem "CameraRegression: a non-central model's held-out errors read no back-projection Jacobian (0.11.6)" setup = [FitCameraNoncentral] begin
    using Test
    using StaticArrays: SMatrix, SVector
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: NoncentralSplineModel, backproject_with_jacobian
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    struct NaNJacobian
        model::Any
    end
    VgtkCore.backproject_with_jacobian(s::NaNJacobian, u) = (first(backproject_with_jacobian(s.model, u)), fill(NaN, SMatrix{3, 2, Float64}))
    VgtkCore.backproject_with_jacobian(s::NaNJacobian, us::AbstractVector{<:SVector{2}}; device) = map(u -> backproject_with_jacobian(s, u), us)
    # the line offsets and direction errors at every held-out pixel
    fitted = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, SOURCE, REGION)), CameraRegression(; tolerance = 1.0e-3)).u
    ql = VgtkEstimation._image_quadrature(VgtkEstimation._lines_on(VgtkEstimation.default_device()), MODEL, REGION)
    hl = VgtkEstimation._held_out(NoncentralSplineModel, NaNJacobian(MODEL), ql, fitted.intrinsics, VgtkCore.default_device())
    direction, offsets = VgtkEstimation._line_errors(fitted, hl)
    @test hl.n_dropped == 0 && length(direction.values) == length(offsets.values) == length(ql.held_px)
end

@testitem "PSpline: the SplineLens backward's one solve on a given lattice — CameraRegression's at the lattice it chose" setup = [FitCameraFisheye] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using VisualGeometryToolkitCore: BoundedCamera, Size2, SplineLens, lens
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, PSpline, CameraFitMetadata
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-3))
    rb = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-3, forward = false))
    @test lens(r.u).backward.controls == lens(rb.u).backward.controls   # the forward does not move the backward
    gw, gh = r.metadata.family.grid_size
    p = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), PSpline(; grid_size = (gw, gh)))       # (columns, rows), Core's lattice convention
    @test retcode_symbol(p) === :solved && p.stats.nsteps == 0 && p.metadata isa CameraFitMetadata && lens(p.u).forward === nothing
    @test lens(p.u).backward.controls == lens(r.u).backward.controls && p.rms == r.rms && p.metadata.max_px == r.metadata.max_px
    @test p.metadata.family == (; λ = r.metadata.family.λ, grid_size = (gw, gh), stride = r.metadata.family.stride, n_held_out_dropped = 0)
    for (cols, rows) in ((35, 25), (105, 75))
        g = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), PSpline(; grid_size = (cols, rows)))
        G = lens(g.u).backward.controls
        @test size(G) == (rows, cols) && g.metadata.family.grid_size == (cols, rows) && retcode_symbol(g) === :solved
        @test isfinite(g.metadata.max_px) && all(c -> all(isfinite, c), G)
    end
    @test solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), PSpline(; grid_size = (105, 75))).metadata.max_px < solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), PSpline(; grid_size = (35, 25))).metadata.max_px
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(SplineLens, FISHEYE, REGION)), PSpline(; grid_size = (35, 25)))      # a bare model has no image
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), PSpline(; grid_size = (3, 9)))             # fewer than 4 nodes on an axis
end

@testitem "PSpline: the non-central model on a given lattice — CameraRegression's at the lattice it chose" setup = [FitCameraNoncentral] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using VisualGeometryToolkitCore: NoncentralSplineModel
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, PSpline
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    r = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, SOURCE, REGION)), CameraRegression(; tolerance = 1.0e-3))
    gw, gh = r.metadata.family.grid_size
    p = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, SOURCE, REGION)), PSpline(; grid_size = (gw, gh)))
    @test retcode_symbol(p) === :solved && p.u.directions.controls == r.u.directions.controls &&
        p.u.points.controls == r.u.points.controls
    @test p.metadata.max_px == r.metadata.max_px && p.metadata.family.max_offset == r.metadata.family.max_offset &&
        p.metadata.family.center == r.metadata.family.center && p.rms == r.rms
    for (cols, rows) in ((35, 25), (105, 75))
        g = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, SOURCE, REGION)), PSpline(; grid_size = (cols, rows)))
        @test size(g.u.directions.controls) == (rows, cols) && isfinite(g.metadata.max_px) && isfinite(g.metadata.family.max_offset)
    end
end

@testitem "CameraRegression(SplineLens): a function source goes through Core's source constructor, which gauges it" setup = [FitCameraFisheye] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using VisualGeometryToolkitCore: backproject, CameraModel, SplineLens, BoundedCamera, Size2, lens, rotation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # a function of the pixel has neither an image nor a K: the fit refuses it, and
    # the route is the SplineLens camera of the function (its K the lattice's gauge),
    # whose backward the fit keeps and whose forward it fits
    f = u -> backproject(FISHEYE, u)
    @test_throws r"no image" solve(EstimationProblem(CameraRegressionProblem(SplineLens, f, REGION)), CameraRegression(; tolerance = 1.0e-2))
    @test_throws r"CameraModel\(SplineLens, f, size" solve(EstimationProblem(CameraRegressionProblem(SplineLens, f, REGION)), CameraRegression(; tolerance = 1.0e-2))   # the refusal names the route
    cam = CameraModel(SplineLens, f, Size2(1000, 1000); grid_size = (40, 40), init = :quasi_interpolate)
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, BoundedCamera(cam, Size2(1000, 1000)), REGION)), CameraRegression(; tolerance = 1.0e-2))
    @test retcode_symbol(r) === :converged && r.metadata.family.forward.max_px <= 1.0e-2
    # the backward is the camera's on its optical axis (changed on purpose, 2026-11-11: a gauge-K source is gauged
    # first), its controls turned by the fit's rotation exactly
    R = rotation(r.metadata.rotation)
    @test lens(r.u).backward.controls == map(c -> R * c, lens(cam).backward.controls)
end

@testitem "CameraRegression, the fit-then-refine recipe: a lens refitted onto its own family has zero residuals" setup = [CameraRecipes] begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore, CameraModel, Lens, Perspective, Equidistant, Stereographic, Division, Scaramuzza, Ellipse
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # the regression's targets are the source's plain radii, so a source at its family's anchor is reproduced: by the
    # linear fit to its solve's rounding, and by the refine from it exactly or to the rounding of the reduction kernel
    # (on an accelerator its sin/cos/atan differ from the host's in the last bits, 2026-11-11)
    region = Ellipse(SVector(320.0, 240.0), 200.0, 200.0, 0.0)
    for l in (Lens(Perspective()), Lens(Equidistant()), Lens(Stereographic()), zero(Division{2}), zero(Scaramuzza{3}))
        src = CameraModel(; pp = (320.0, 240.0), focal = 300.0, lens = l)
        r = solve(EstimationProblem(CameraRegressionProblem(typeof(l), src, region)), CameraRegression())
        @test r.metadata.max_px < 1.0e-9 && r.rms < 1.0e-9
        rc = solve(EstimationProblem(CameraRegressionProblem(typeof(l), src, region; device = VisualGeometryToolkitCore.KA.CPU())), CameraRegression(; refine = LevenbergMarquardt()))
        @test rc.metadata.max_px < 1.0e-9 && rc.rms < 1.0e-9
        rd = solve(EstimationProblem(CameraRegressionProblem(typeof(l), src, region)), CameraRegression(; refine = LevenbergMarquardt()))
        @test rd.metadata.max_px < 1.0e-9 && rd.rms < 1.0e-9
    end
end

@testitem "CameraRegression: a non-central model from a non-central source" setup = [FitCameraNoncentral] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SVector
    using LinearAlgebra: norm, normalize, dot
    using VisualGeometryToolkitCore: NoncentralSplineModel, ChartSpline, backproject, NearestPoint, Point, solve, EstimationProblem, calibration_matrix
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # The direction tolerance bounds the DIRECTION error only (f·angle in pixels, 0.11.6); the line offset is reported
    # beside it and no tolerance bounds it (an offset tolerance is a follow-up). This item runs at a tighter direction
    # tolerance than the fit's 1e-3 px default use and asserts the directions against it.
    tol = 3.0e-4
    r = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, SOURCE, REGION)), CameraRegression(; tolerance = tol))
    g = r.u
    md = r.metadata
    # the tolerance chooses the lattice
    @test retcode_symbol(r) === :converged && r.stats.nsteps == 0 && md.family.outer_iterations > 1
    gw, gh = md.family.grid_size
    # Core's full-image lattice with the source's K
    @test g isa NoncentralSplineModel && g.intrinsics.K == MODEL.intrinsics.K && g.directions.area == MODEL.directions.area
    @test size(g.directions.controls) == size(g.points.controls) == (gh, gw)
    # the anchor: Core's nearest point of every quadrature line of the source
    q = VgtkEstimation._image_quadrature(VgtkEstimation._lines_on(VgtkEstimation.default_device()), MODEL, REGION)
    @test md.family.center ≈ solve(EstimationProblem(Point, [o => d for (o, d) in q.fit_rays]), NearestPoint()).u
    # held out, interior and edge band (measured 2026-11-04 at 26 × 26: direction
    # 9.2e-4 px max, 1.3e-4 px rms, the max at the edge; line offset 5.1e-8 m max
    # against a 3 mm field; at 40 × 40 7.1e-4 px and 7.9e-9 m) — and far better
    # than sampling the source's lines at the nodes
    @test md.max_px <= tol && r.rms < tol / 2 && md.max_px_edge <= md.max_px
    @test isfinite(md.family.max_offset) && isfinite(md.family.rms_offset)   # reported, not bounded
    h = 1000 / (gw - 3)
    lines = [backproject(MODEL, SVector(clamp((j - 2) * h, 0.0, 999.0), clamp((i - 2) * h, 0.0, 999.0))) for i in 1:gh, j in 1:gw]
    sampled = NoncentralSplineModel(
        g.intrinsics, ChartSpline(g.directions.area, [SVector{3, Float64}(l.direction) for l in lines]),
        ChartSpline(g.directions.area, [SVector{3, Float64}(l.origin) for l in lines])
    )
    e_s, o_s = VgtkEstimation._line_errors(sampled, VgtkEstimation._held_out(NoncentralSplineModel, MODEL, q, g.intrinsics, VgtkCore.default_device()))
    @test md.max_px < e_s.max / 100
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
    @test maximum(angle, grazing) < tol / VgtkEstimation._focal_px(g.intrinsics) * 1.0001   # the direction tolerance, as an angle
    @test all(isfinite, map(offset, grazing))
    # frame in = frame out: translating every source line by t translates the
    # fitted points and the anchor by t (absolute points, not relative to c*)
    t = SVector(0.5, -0.25, 2.0)
    rt = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, noncentral_source(t), REGION)), CameraRegression(; tolerance = tol))
    @test rt.metadata.family.grid_size == md.family.grid_size
    @test maximum(norm.(rt.u.points.controls .- g.points.controls .- Ref(t))) < 1.0e-9
    @test norm(rt.metadata.family.center - md.family.center - t) < 1.0e-12
    @test rt.u.directions.controls ≈ g.directions.controls
    # a tolerance out of reach at the finest lattice (169 × 169 for 1000 px)
    # returns what it reached (measured 4.3e-5 px), never throws
    ru = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, SOURCE, REGION)), CameraRegression(; tolerance = 1.0e-9))
    @test retcode_symbol(ru) === :max_iterations && 1.0e-9 < ru.metadata.max_px < 1.0e-3
    @test ru.metadata.family.grid_size == (169, 169)
    # the bare model has no image: bound it
    @test_throws r"no image" solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, MODEL, REGION)), CameraRegression(; tolerance = 1.0e-3))
end

@testitem "CameraRegression: a non-central model from a central source" setup = [FitCameraNoncentral] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using Test
    using StaticArrays: SA, SVector
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: NoncentralSplineModel, SplineLens, BoundedCamera, Size2, backproject, CameraModel, KannalaBrandt, Rect, Ellipse, px
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    r = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, bc, REGION)), CameraRegression(; tolerance = 1.0e-3))
    g = r.u
    @test retcode_symbol(r) === :converged
    # every line through the projection centre: c* is the centre and every
    # point is c* (offsets ≡ 0)
    @test r.metadata.family.center == SVector(0.0, 0.0, 0.0)
    @test all(iszero, g.points.controls)
    @test r.metadata.family.max_offset == 0
    # the directions as the SplineLens target fits its backward: the same lattice
    # (measured 21 × 21, 8.5e-4 px)
    @test r.metadata.max_px <= 1.0e-3
    @test r.metadata.family.grid_size == solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-3, forward = false)).metadata.family.grid_size
    # out of reach at the finest lattice of a 60 px image: said, not warned
    small = BoundedCamera(CameraModel(18.0px, SA[30.0px, 30.0px]; lens = KannalaBrandt(-0.02)), Size2(60, 60))
    ru = solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, small, Rect(0, 0, 60, 60) ∩ Ellipse(SVector(30.0, 30.0), 29.0, 29.0, 0.0))), CameraRegression(; tolerance = 1.0e-6))
    @test retcode_symbol(ru) === :max_iterations && ru.metadata.max_px > 1.0e-6
    # the batched central source lines are the scalar back-projection, normalized
    pix = [SVector(500.5, 400.5), SVector(20.5, 500.5), SVector(950.5, 980.5)]
    # (to the last bit: on an accelerator the batched backprojection's functions are the device's, 2026-11-11)
    @test all(map((a, b) -> a[1] == b[1] && maximum(abs, a[2] - b[2]) <= 4 * eps(), VgtkEstimation._source_lines(FISHEYE, pix, VgtkEstimation.default_device()),
        [(SVector(0.0, 0.0, 0.0), normalize(SVector{3, Float64}(backproject(FISHEYE, p)))) for p in pix]))
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, FISHEYE, REGION)), CameraRegression(; tolerance = 0.01))
end

@testitem "the fit-then-refine recipe: a parametric lens from a SplineLens camera, through the camera's rays (the exporters' route)" setup = [CameraRecipes, FitCameraFisheye] begin
    using Test
    using VisualGeometryToolkitCore: lens, parameter_vector, calibration_matrix, KannalaBrandt, rotation
    using LinearAlgebra: I
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, SolidAngleMeasure
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    # A SplineLens camera has no OpenCV/COLMAP form: the exporters' refusal (VisualGeometryToolkitIO) says to fit a
    # parametric lens to it first, `the linear fit then its refine` — the bounded camera or its model.
    # Its K is a gauge, so its lens has no radial profile of its own: the fit samples the camera's rays. GRID is the
    # 40 × 40 SplineLens camera of a Kannala–Brandt fisheye (k₁ = −0.02, f = 300 px, its K kept).
    for src in (GRID_BOUNDED, GRID)
        r = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, src, REGION)), CameraRegression(; refine = LevenbergMarquardt()))
        kb = r.u
        @test lens(kb) isa KannalaBrandt{4}
        @test maximum(abs.(collect(parameter_vector(lens(kb))) .- [-0.02, 0.0, 0.0, 0.0])) <= 1.0e-3
        @test abs(r.metadata.family.focal_scale - 1) <= 1.0e-3
        @test r.rms < 0.05 && r.metadata.max_px < 0.5
        # a gauge K's principal point is fitted (changed on purpose, 2026-11-11: it was kept, the pixel imaging ẑ): GRID
        # is the fisheye's own spline, so it lands on the fisheye's (500, 500) to the spline's approximation
        @test maximum(abs, calibration_matrix(kb)[1:2, 3] - calibration_matrix(GRID)[1:2, 3]) < 1.0e-5
        @test maximum(abs, rotation(r.metadata.rotation) - I) < 1.0e-8
    end
    # the linear fit alone reads the radii about the gauge's own principal point and the angles about its ẑ: GRID's are
    # the fisheye's, so it is already close; the refine frees the principal point, the aspect and the axis
    rl = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, GRID, REGION)), CameraRegression())
    @test maximum(abs.(collect(parameter_vector(lens(rl.u))) .- [-0.02, 0.0, 0.0, 0.0])) <= 1.0e-2
    @test rotation(rl.metadata.rotation) == one(rotation(rl.metadata.rotation))
    # the image samples carry the ImageMeasure() density only
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, GRID, REGION; weight = SolidAngleMeasure())), CameraRegression())
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, GRID, REGION; weight = SolidAngleMeasure())), CameraRegression(; refine = LevenbergMarquardt()))
end

@testitem "CameraRegression(SplineLens): the forward is fitted on the chart project evaluates it on" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: lens, SplineLens, default_device
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    cam = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-3)).u
    q = VgtkEstimation._chart_quadrature(cam, REGION, default_device())
    @test q.chart == VgtkCore.forward_chart(SplineLens) == VgtkCore.equidistant_chart(SVector(0.0, 0.0, 1.0))
end

@testitem "CameraRegression(SplineLens), CameraRegression(NoncentralSplineModel): the source's batched rays on `device`" setup = [FitCameraFisheye] begin
    using Test
    using VisualGeometryToolkitCore: lens, SplineLens, NoncentralSplineModel, default_device
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, PSpline
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    a = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION; device = default_device())), CameraRegression(; tolerance = 1.0e-3)).u
    b = solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION)), CameraRegression(; tolerance = 1.0e-3)).u
    @test lens(a).forward.controls == lens(b).forward.controls
    # the keyword reaches the batched backprojection (a backend is a KernelAbstractions.Backend)
    @test_throws TypeError solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION; device = :none)), CameraRegression(; tolerance = 1.0e-3))
    @test_throws TypeError solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, GRID_BOUNDED, REGION; device = :none)), CameraRegression(; tolerance = 1.0e-3))
    @test_throws TypeError solve(EstimationProblem(CameraRegressionProblem(SplineLens, GRID_BOUNDED, REGION; device = :none)), PSpline(; grid_size = (12, 12)))
end

@testitem "ForwardSpline: a SplineLens camera given its forward, its backward, K and frame kept" begin
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, KannalaBrandt, Size2, BoundedCamera, model, lens, project, calibration_matrix, retcode_symbol, imaged_region, backproject
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, CameraRegressionProblem, ForwardSpline
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    kb = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    cam = BoundedCamera(CameraModel(SplineLens, kb, Size2(1280, 960); grid_size = (12, 10)), Size2(1280, 960))
    @test lens(model(cam)).forward === nothing
    r = solve(EstimationProblem(CameraRegressionProblem(cam)), ForwardSpline(; tolerance = 1.0e-3))
    est = r.u
    @test retcode_symbol(r) === :converged || VC.successful_retcode(r)
    @test est isa BoundedCamera && lens(model(est)).forward !== nothing
    # everything but the forward is the camera's own, bit for bit
    @test lens(model(est)).backward.controls == lens(model(cam)).backward.controls && lens(model(est)).backward.area == lens(model(cam)).backward.area
    @test calibration_matrix(model(est)) == calibration_matrix(model(cam))
    @test r.metadata.max_px <= 1.0e-3
    # the forward's projection agrees with the backward's solve to the tolerance over the image
    back = model(cam)
    for u in (SVector(640.0, 480.0), SVector(100.0, 80.0), SVector(1200.0, 900.0), SVector(900.0, 300.0))
        X = backproject(back, u)
        @test norm(project(model(est), X) - project(back, X)) <= 1.0e-3
    end
    # a parametric camera has no forward spline to fit
    @test_throws ArgumentError solve(EstimationProblem(CameraRegressionProblem(BoundedCamera(kb, Size2(1280, 960)))), ForwardSpline(; tolerance = 1.0e-3))
end

# 0.11.6: the held-out direction error is the ANGLE to the source's ray, f·angle in pixels — never a pixel mapping
# through the source's back-projection Jacobian, whose pseudo-inverse blows up where the source's ray field degenerates
# (a Schöps source's domain edge reported 10⁵ px for a 17 px angle). The source here is GRID with its three left
# control columns set to the fourth: its rays stop varying along x at the left edge, so ∂d/∂u is singular there.
@testitem "CameraRegression(SplineLens): the held-out error is the direct angle, also where the source's Jacobian degenerates" setup = [FitCameraFisheye] begin
    using Test
    using StaticArrays: SMatrix, SVector
    using LinearAlgebra: normalize, dot, norm, cross, svdvals
    using VisualGeometryToolkitCore: SplineLens, BoundedCamera, Size2, backproject, backproject_with_jacobian, calibration_matrix, lens, rotation, with_directions
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    C = copy(lens(GRID).backward.controls)
    for c in 1:3
        C[:, c] .= C[:, 4]
    end
    src = BoundedCamera(with_directions(GRID, C), Size2(1000, 1000))
    # the degeneracy: the source's Jacobian has a (near) zero singular value at the left edge of the region
    _, J = backproject_with_jacobian(src, SVector(30.0, 500.0))
    @test minimum(svdvals(Matrix(J))) < 2.0e-2 * maximum(svdvals(Matrix(J)))   # (measured 1.0e-2)
    r = solve(EstimationProblem(CameraRegressionProblem(SplineLens, src, REGION)), CameraRegression(; tolerance = 1.0e-3, lattice_spacing = 13.5, forward = false))
    md = r.metadata; cam = r.u; R = SMatrix{3, 3, Float64}(rotation(md.rotation))
    f = calibration_matrix(GRID)[1, 1]
    angle(a, b) = atan(norm(cross(a, b)), dot(a, b))
    direct = [f * angle(SVector{3, Float64}(backproject(cam, p)), R * normalize(SVector{3, Float64}(backproject(src, p))))
              for p in (SVector(x, y) for x in 1.0:4.0:999.0, y in 1.0:4.0:999.0) if p ∈ REGION]
    filter!(isfinite, direct)
    # the reported maximum is the direct error's (at the held-out pixels, a subset of the region): never a Jacobian blow-up
    @test md.max_px <= 1.5 * maximum(direct) + 1.0e-9
    @test md.max_px < 1.0e-6   # an exact resample: rounding, never a Jacobian blow-up
end

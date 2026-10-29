@testitem "core_defaults" setup = [EpipolarScenes, HomographyScenes, PoseScenes, DivisionBoards, Contamination,
                                   CircleScene, EllipseScene, RotationScene] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# The core's defaults (K+ phase 4): every `find_*` scores with
# VisualGeometryToolkitCore's MSAC at its class's threshold convention
# (configs/default.toml `[pipelines.<class>] threshold`, read through `_dflt`;
# hand-eye's a fraction of the characteristic length) and refines with
# `Irls{TLSShape}()`, which follows MSAC's threshold. `threshold` is the one
# width keyword; `outlier_halfwidth` (a marginal scorer's) is refused, naming
# the way to say it; `threshold` with `scoring` is refused; the essential
# problem refuses a sub-detector threshold. And each pipeline converges on its
# synthetic scene under the defaults alone.
#
# Every `refine_*` is its `find_*`'s local-optimization step alone, from a given
# model: the same defaults, the shared keywords minus the sampling ones, and
# from the raw RANSAC incumbent it reaches `find_*`'s model under any strategy
# the problem hosts; a strategy the problem cannot host is refused as inside
# `find_*`.
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays, StructArrays
using StableRNGs
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: MsacScoring, TLSShape, converged, estimate, csponds, radial_transfer, RadialHomography,
    HomographyMat, inlier_threshold, rotation, translation
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: CauchyShape, GaussianShape, Circle
using VisualGeometryToolkitEstimation: _dflt, _dflt_lo, _default_scoring_at, _threshold_unit, RadialHomographyProblem,
    RadialCameraProblem, find_radial_homography, find_radial_camera, HandEyeProblem, NoLocalOptimization,
    RansacConfig, RansacWorkspace

thr(class) = Float64(_dflt("pipelines.$class", "threshold"))

# --- the scenes ---------------------------------------------------------------
function h_scene()
    src, dst = homography_pairs(
        StableRNG(11), H_PX; n_in = 120, n_out = 60, σ = 0.3,
        source = gaussian_source, target = rng -> randn(rng, 2) .* 400 .+ 300
    )
    return StructArray(src .=> dst), 120
end
function stereo_scene(cam; seed = 13)
    u₁, u₂ = stereo_pixel_pairs(
        StableRNG(seed), cam, cam, RotMatrix(RotXYZ(0.05, 0.12, -0.03)), SVector{3, Float64}(0.9, 0.05, 0.1);
        n = 140, σ₂ = 0.4, n_out = 60,
        point = rng -> SA[2.0 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), 4.0 + 3.0 * rand(rng)],
        outlier = rng -> (SA[960 * rand(rng), 720 * rand(rng)], SA[960 * rand(rng), 720 * rand(rng)])
    )
    return u₁, u₂, length(u₁) - 60
end
const PCAM = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
const R_PNP = RotMatrix(RotXYZ(0.15, -0.25, 0.1))
const T_PNP = Vec3{Float64}(0.3, -0.2, 0.6)
function pnp_scene()
    pairs, _ = contaminated_pairs(
        StableRNG(19), (rng, _) -> Pair(frustum_point(rng, PCAM, R_PNP, T_PNP)...), 120; σ = 0.5, n_out = 60,
        outlier = (rng, _) -> first(frustum_point(rng, PCAM, R_PNP, T_PNP)) => SVector(40.0 + 880.0 * rand(rng), 40.0 + 640.0 * rand(rng))
    )
    return StructArray(first.(pairs) .=> last.(pairs))
end
const RW, RH = 1200.0, 900.0
const M_RH = RadialHomography{Float64}(
    HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0))),
    -0.25 / hypot(RW, RH)^2, SVector(RW / 2, RH / 2)
)
function radial_homography_scene()
    function inlier(rng, _)
        while true
            s = SVector(200 * rand(rng), 150 * rand(rng))
            d = radial_transfer(M_RH, s)
            d === nothing && continue
            (0 < d[1] < RW && 0 < d[2] < RH) && return s => SVector(d[1], d[2])
        end
    end
    return contaminated_pairs(
        StableRNG(17), inlier, 70; σ = 0.3, n_out = 30, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng), 150 * rand(rng)) => SVector(RW * rand(rng), RH * rand(rng))
    )
end
const RC_CENTER = SVector(2000.0, 1500.0)
function radial_camera_scene()
    cam = board_camera(RotXYZ(0.35, -0.25, 0.15), SVector(40.0, -30.0, 600.0); f = 900.0, ks = (-2.0e-7,), center = RC_CENTER)
    return contaminated_pairs(
        StableRNG(19), board_pixels(cam; grid = -100.0:25.0:100.0); σ = 0.2, n_out = 20, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng) - 100, 200 * rand(rng) - 100) => SVector(4000 * rand(rng), 3000 * rand(rng))
    )
end
function hand_eye_scene()
    rng = StableRNG(29)
    sc = robot_world_hand_eye_scene(rng; N = 30, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1),
                                    A = (ang_max = π / 3, t_scale = 0.5))
    X, Y, A = only(sc.X), sc.Y, sc.A
    B = [perturb_pose(rng, marker_pose(Y, A[i], X); σ_R = 0.001, σ_t = 0.001) for i in eachindex(A)]
    return A .=> B, X
end

# The default run and the run with its default spelled out (MSAC at the class
# threshold in the problem's residual unit) are the same run.
function same_run(r_default, r_explicit)
    @test r_default !== nothing && converged(r_default)
    @test estimate(r_default) == estimate(r_explicit)
    @test r_default.inlier_mask == r_explicit.inlier_mask
    @test r_default.quality == r_explicit.quality
end

@testset "the class threshold conventions of record" begin
    @test (thr(:homography), thr(:fundmat), thr(:essential), thr(:absolute_pose)) == (3.0, 3.0, 1.0, 8.0)
    @test (thr(:radial_homography), thr(:radial_camera), thr(:hand_eye)) == (3.0, 3.0, 0.01)
    for class in (:homography, :fundmat, :essential, :absolute_pose, :radial_homography, :radial_camera, :hand_eye)
        @test _dflt_lo(class) == Irls{TLSShape}()                     # follows MSAC's threshold
    end
    # the default scorer, and the workspace a problem type is pre-allocated for
    cs, _ = h_scene()
    p = HomographyProblem(cs)
    @test _default_scoring_at(p, 3.0) === MsacScoring(p; threshold = 3.0)
    @test _threshold_unit(p) == 1.0
    ws = VgtkEstimation.RansacWorkspace(HomographyProblem{Float64}, length(cs))
    @test ws.buf.scoring_ws isa VgtkCore.ThresholdScoringWorkspace
end

@testset "find_* default scorer: MSAC at the class threshold" begin
    @testset "homography" begin
        cs, n_in = h_scene()
        p = HomographyProblem(cs)
        r = find_homography(cs; rng = StableRNG(1))
        same_run(r, find_homography(cs; scoring = MsacScoring(p; threshold = thr(:homography)), rng = StableRNG(1)))
        same_run(r, find_homography(cs; threshold = thr(:homography), rng = StableRNG(1)))
        @test count(r.inlier_mask[1:n_in]) > 0.9 * n_in && count(r.inlier_mask[(n_in + 1):end]) < 10
        @test r.lo_retcode !== :not_run                                     # the default LO ran
    end
    @testset "fundmat" begin
        u₁, u₂, n_in = stereo_scene(PCAM)
        cs = StructArray(u₁ .=> u₂)
        r = find_fundmat(cs; rng = StableRNG(2))
        same_run(r, find_fundmat(cs; scoring = MsacScoring(FundMatProblem(cs); threshold = thr(:fundmat)), rng = StableRNG(2)))
        @test count(r.inlier_mask[1:n_in]) > 0.75 * n_in && count(r.inlier_mask[(n_in + 1):end]) < 15
    end
    @testset "essential" begin
        u₁, u₂, n_in = stereo_scene(ECAM; seed = 14)
        cs = csponds(u₁, u₂)
        r = find_essential(ECAM, ECAM, cs; rng = StableRNG(3))
        p = EssentialProblem(ECAM, ECAM, cs)
        same_run(r, find_essential(ECAM, ECAM, cs; scoring = MsacScoring(p; threshold = thr(:essential)), rng = StableRNG(3)))
        @test count(r.inlier_mask[1:n_in]) > 0.75 * n_in && count(r.inlier_mask[(n_in + 1):end]) < 15
    end
    @testset "absolute pose" begin
        cs = pnp_scene()
        r = find_absolute_pose(PCAM, cs; rng = StableRNG(4))
        p = AbsolutePoseProblem(PCAM, cs)
        same_run(r, find_absolute_pose(PCAM, cs; scoring = MsacScoring(p; threshold = thr(:absolute_pose)), rng = StableRNG(4)))
        @test count(r.inlier_mask[1:120]) > 100
        @test rotation_error_deg(estimate(r), R_PNP) < 1.0 && translation_error(estimate(r), T_PNP) < 0.05
    end
    @testset "radial homography" begin
        pairs, gt = radial_homography_scene()
        c = (RW / 2, RH / 2)
        r = find_radial_homography(pairs; center = c, rng = StableRNG(5))
        p = RadialHomographyProblem(pairs; center = c)
        same_run(r, find_radial_homography(pairs; center = c, scoring = MsacScoring(p; threshold = thr(:radial_homography)),
                                           rng = StableRNG(5)))
        @test count(r.inlier_mask .& gt) >= count(gt) - 3 && count(r.inlier_mask .& .!gt) <= 2
    end
    @testset "radial camera" begin
        pairs, gt = radial_camera_scene()
        r = find_radial_camera(pairs; center = RC_CENTER, rng = StableRNG(6))
        p = RadialCameraProblem(pairs; center = RC_CENTER)
        same_run(r, find_radial_camera(pairs; center = RC_CENTER, scoring = MsacScoring(p; threshold = thr(:radial_camera)),
                                       rng = StableRNG(6)))
        @test count(r.inlier_mask .& gt) >= count(gt) - 2 && count(r.inlier_mask .& .!gt) == 0
    end
    @testset "hand-eye (a fraction of the characteristic length)" begin
        cs, X = hand_eye_scene()
        r = find_hand_eye(cs; rng = StableRNG(7))
        p = HandEyeProblem(cs; min_rel_angle = deg2rad(_dflt("robot_world_hand_eye", "min_rel_angle_deg")))
        @test _threshold_unit(p) == p.char_length
        same_run(r, find_hand_eye(cs; scoring = MsacScoring(p; threshold = thr(:hand_eye) * p.char_length), rng = StableRNG(7)))
        @test norm(translation(estimate(r)) - translation(X)) < 0.01
    end
end

@testset "the width keywords: threshold, never outlier_halfwidth" begin
    cs, _ = h_scene()
    p = HomographyProblem(cs)
    θ = estimate(find_homography(cs; rng = StableRNG(1)))
    lo = NoLocalOptimization()
    # `outlier_halfwidth` is refused on every core entry, naming the way to say it
    for call in (
            () -> find_homography(cs; outlier_halfwidth = 20.0),
            () -> locally_optimize(p, θ; outlier_halfwidth = 20.0, local_optimization = lo),
        )
        err = try
            call(); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("`outlier_halfwidth` is no keyword of the core entries", err.msg)
        @test occursin("MSAC at `threshold`", err.msg)
        @test occursin("StudentTUniformScoring", err.msg)               # the way to supply a marginal scorer
    end
    # `threshold` builds only the default scoring: with `scoring` it is refused
    sc = MsacScoring(p; threshold = 3.0)
    for call in (
            () -> find_homography(cs; threshold = 3.0, scoring = sc),
            () -> locally_optimize(p, θ; threshold = 3.0, scoring = sc, local_optimization = lo),
        )
        err = try
            call(); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("supply `threshold` or `scoring`, not both", err.msg)
    end
    # `locally_optimize` takes no class: a threshold or a scoring is required
    err = try
        locally_optimize(p, θ; local_optimization = lo); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("`threshold` is required", err.msg)
    r = locally_optimize(p, θ; threshold = 3.0, local_optimization = Irls{TLSShape}())
    @test converged(r)
    @test r.inlier_mask == locally_optimize(p, θ; scoring = sc, local_optimization = Irls{TLSShape}()).inlier_mask
    # a non-positive threshold is the scorer's refusal
    @test_throws ArgumentError find_homography(cs; threshold = 0.0)
end

@testset "essential: a sub-detector threshold is refused" begin
    u₁, u₂, _ = stereo_scene(ECAM; seed = 14)
    cs = csponds(u₁, u₂)
    p = EssentialProblem(ECAM, ECAM, cs)
    err = try
        find_essential(ECAM, ECAM, cs; threshold = 0.02); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("threshold 0.02 is below", err.msg)
    @test_throws ArgumentError find_essential(ECAM, ECAM, cs; scoring = MsacScoring(p; threshold = 0.05))
    @test inlier_threshold(MsacScoring(p; threshold = 0.06), nothing) == 0.06
    @test find_essential(ECAM, ECAM, cs; threshold = 0.06, rng = StableRNG(8)) !== nothing   # at the floor: accepted
end

@testset "refine_*: the keywords of the shared refine pipeline" begin
    kw(m) = Set(Base.kwarg_decl(m))
    shared = kw(only(methods(VgtkEstimation._refine_pipeline)))
    @test shared == Set([:threshold, :outlier_halfwidth, :local_optimization, :scoring, :workspace])
    @test shared == setdiff(kw(only(methods(VgtkEstimation._ransac_pipeline))), [:max_trials, :min_trials, :config, :rng])
    # each refine_* forwards the rest like its find_*, and declares only the
    # find_*'s model-specific keywords — minus those that choose hypotheses (the
    # radial centre comes with the model)
    sampling_only = [:center, :extra_check, :minimal]
    for (f, g) in (find_homography => refine_homography, find_fundmat => refine_fundmat,
                   find_essential => refine_essential, find_absolute_pose => refine_absolute_pose,
                   find_rotation => refine_rotation, find_circle => refine_circle, find_ellipse => refine_ellipse,
                   find_radial_homography => VgtkEstimation.refine_radial_homography,
                   find_radial_camera => VgtkEstimation.refine_radial_camera, find_hand_eye => refine_hand_eye)
        @test Set(kw.(methods(g))) == Set(setdiff(k, sampling_only) for k in kw.(methods(f)))
        @test all(m -> Symbol("kwargs...") in kw(m), methods(g))
    end
    # every shared keyword is accepted, with the meaning it has in find_*
    pts, _ = circle_scene(Xoshiro(33), 300; outl = 0.3)
    p = CircleProblem(pts)
    g = Circle(Point2(C₀ + SVector(1.0, -1.0)), R₀ + 1.0)
    for (k, v) in (:threshold => 2.0, :local_optimization => Irls{TLSShape}(), :scoring => MsacScoring(p; threshold = 1.0),
                   :workspace => RansacWorkspace(CircleProblem, 300))
        @test converged(refine_circle(g, pts; (; k => v)...))
    end
    @test_throws ArgumentError refine_circle(g, pts; outlier_halfwidth = 5.0)
    @test_throws ArgumentError refine_circle(g, pts; threshold = 2.0, scoring = MsacScoring(p; threshold = 1.0))
    for (k, v) in (:max_trials => 10, :min_trials => 1, :config => RansacConfig(), :rng => StableRNG(1))
        @test_throws MethodError refine_circle(g, pts; (; k => v)...)        # no sampling to configure
    end
    @test refine_circle(g, pts[1:2]) === nothing && refine_homography(H_PX, h_scene()[1][1:3]) === nothing
end

@testset "refine_*: from the raw incumbent to find_*'s model, under any strategy the problem hosts" begin
    # Each class's find_* scene; the guess is the RANSAC incumbent with NO local
    # optimization (a minimal-sample model), refined by refine_*; the reference
    # is find_* with the same strategy and draws. Same model to the noise level:
    # the residuals on find_*'s inliers within 0.05 px (0.05·L for hand-eye) and
    # the same consensus but for a few points.
    pts, _ = circle_scene(Xoshiro(5), 600)
    epts, _ = ellipse_scene(Xoshiro(6), 600)
    hcs, _ = h_scene()
    u₁, u₂, _ = stereo_scene(PCAM); fcs = StructArray(u₁ .=> u₂)
    e₁, e₂, _ = stereo_scene(ECAM; seed = 14); ecs = csponds(e₁, e₂)
    pcs = pnp_scene()
    rcs, _, _ = rotation_scene(Xoshiro(8), RotMatrix(RotXYZ(0.2, -0.3, 0.4)), 300; outl = 0.3)
    rh, _ = radial_homography_scene(); c = (RW / 2, RH / 2)
    rc, _ = radial_camera_scene()
    hecs, _ = hand_eye_scene()
    triggs = Triggs{CauchyShape}(scale_multiple = 2.5)
    cases = (
        (:circle, (; kw...) -> find_circle(pts; kw...), (θ; kw...) -> refine_circle(θ, pts; kw...), (triggs, Irls{TLSShape}())),
        (:ellipse, (; kw...) -> find_ellipse(epts; kw...), (θ; kw...) -> refine_ellipse(θ, epts; kw...), (triggs,)),
        (:homography, (; kw...) -> find_homography(hcs; kw...), (θ; kw...) -> refine_homography(θ, hcs; kw...), (triggs, Irls{TLSShape}())),
        (:fundmat, (; kw...) -> find_fundmat(fcs; kw...), (θ; kw...) -> refine_fundmat(θ, fcs; kw...), (triggs,)),
        (:essential, (; kw...) -> find_essential(ECAM, ECAM, ecs; kw...), (θ; kw...) -> refine_essential(θ, ECAM, ECAM, ecs; kw...), (triggs,)),
        (:absolute_pose, (; kw...) -> find_absolute_pose(PCAM, pcs; kw...), (θ; kw...) -> refine_absolute_pose(θ, PCAM, pcs; kw...), ()),
        (:rotation, (; kw...) -> find_rotation(FISHEYE, FISHEYE, rcs; kw...), (θ; kw...) -> refine_rotation(θ, FISHEYE, FISHEYE, rcs; kw...), (triggs,)),
        (:radial_homography, (; kw...) -> find_radial_homography(rh; center = c, kw...), (θ; kw...) -> VgtkEstimation.refine_radial_homography(θ, rh; kw...), (triggs,)),
        (:radial_camera, (; kw...) -> find_radial_camera(rc; center = RC_CENTER, kw...), (θ; kw...) -> VgtkEstimation.refine_radial_camera(θ, rc; kw...), ()),
        (:hand_eye, (; kw...) -> find_hand_eye(hecs; kw...), (θ; kw...) -> refine_hand_eye(θ, hecs; kw...), ()),
    )
    for (class, find, refine, strategies) in cases
        raw = estimate(find(; local_optimization = NoLocalOptimization(), rng = StableRNG(3)))
        for lo in (nothing, strategies...)
            kws = lo === nothing ? (;) : (; local_optimization = lo)
            f = find(; kws..., rng = StableRNG(3))
            r = refine(raw; kws...)
            @test converged(r) && r.lo_retcode !== :not_run
            m = f.inlier_mask
            tol = class === :hand_eye ? 0.05 * HandEyeProblem(hecs).char_length : 0.05
            @test maximum(abs, Float64.(r.residuals[m]) .- Float64.(f.residuals[m])) < tol
            @test count(r.inlier_mask .!= m) <= 3
            @test isapprox(r.rms, f.rms; rtol = 0.01)
        end
    end
    # the ellipse under Irls{TLSShape} at the 1 px threshold: the truncation is
    # path-dependent (here find_* keeps 311 points at rms 0.449 px), so the
    # refine is checked against the truth instead — to the noise level
    centre(e) = SVector(e.center[1], e.center[2])
    e = estimate(refine_ellipse(estimate(find_ellipse(epts; local_optimization = NoLocalOptimization(), rng = StableRNG(3))), epts;
                                local_optimization = Irls{TLSShape}()))
    @test norm(centre(e) - centre(E₀)) < 0.2 && abs(e.a - E₀.a) < 0.2 && abs(e.b - E₀.b) < 0.2 && abs(e.θ - E₀.θ) < 2.0e-3
    # a strategy the problem cannot host is refused as inside find_*
    err(f) = try
        f(); nothing
    catch e
        e
    end
    raw = estimate(find_hand_eye(hecs; local_optimization = NoLocalOptimization(), rng = StableRNG(3)))
    ef = err(() -> find_hand_eye(hecs; local_optimization = triggs, rng = StableRNG(3)))
    er = err(() -> refine_hand_eye(raw, hecs; local_optimization = triggs))
    @test ef isa ArgumentError && er isa ArgumentError && ef.msg == er.msg
end

@testset "refine_*: from a perturbed guess on contaminated data, and from the truth on clean data" begin
    centre(e) = SVector(e.center[1], e.center[2])
    pts, _ = circle_scene(Xoshiro(41), 800)
    g = Circle(Point2(C₀ + SVector(2.0, -1.5)), R₀ + 3.0)
    for lo in (nothing, Irls{TLSShape}())
        kws = lo === nothing ? (;) : (; local_optimization = lo, threshold = 5.0)   # TLS: the truncation's basin is the threshold
        cr = estimate(refine_circle(g, pts; kws...))
        @test norm(centre(cr) - C₀) < 0.1 && abs(cr.r - R₀) < 0.1
    end
    epts, _ = ellipse_scene(Xoshiro(42), 1000)
    ge = Ellipse(Point2(515.0, 399.0), 303.0, 177.0, 0.62)
    for lo in (nothing, Irls{TLSShape}())
        kws = lo === nothing ? (;) : (; local_optimization = lo, threshold = 8.0)
        e = estimate(refine_ellipse(ge, epts; kws...))
        @test norm(centre(e) - centre(E₀)) < 0.2 && abs(e.a - E₀.a) < 0.2 && abs(e.b - E₀.b) < 0.2 && abs(e.θ - E₀.θ) < 2.0e-3
    end
    # the exact model on clean data comes back
    clean, _ = circle_scene(Xoshiro(43), 200; σ = 0.0, outl = 0.0)
    c₀ = Circle(Point2(C₀), R₀)
    cr = estimate(refine_circle(c₀, clean))
    @test norm(centre(cr) - C₀) < 1.0e-8 && abs(cr.r - R₀) < 1.0e-8
    eclean, _ = ellipse_scene(Xoshiro(44), 200; σ = 0.0, outl = 0.0)
    e = estimate(refine_ellipse(E₀, eclean))
    @test norm(centre(e) - centre(E₀)) < 1.0e-8 && abs(e.a - E₀.a) < 1.0e-8 && abs(e.b - E₀.b) < 1.0e-8
    src, dst = homography_pairs(StableRNG(45), H_PX; n_in = 100, n_out = 0, σ = 0.0, source = gaussian_source,
                                target = rng -> randn(rng, 2) .* 400 .+ 300)
    H = SMatrix{3, 3, Float64, 9}(H_PX)
    for lo in (nothing, Triggs{CauchyShape}(scale_multiple = 2.5), Irls{GaussianShape}())
        kws = lo === nothing ? (;) : (; local_optimization = lo)
        Ĥ = SMatrix{3, 3, Float64, 9}(estimate(refine_homography(HomographyMat{Float64}(Tuple(H)), StructArray(src .=> dst); kws...)))
        @test norm(Ĥ / Ĥ[3, 3] - H / H[3, 3]) / norm(H / H[3, 3]) < 1.0e-8
    end
end
end

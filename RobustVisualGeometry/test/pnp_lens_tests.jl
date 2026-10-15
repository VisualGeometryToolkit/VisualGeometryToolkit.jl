@testitem "pnp_lens" setup = [PoseScenes, Contamination] begin
# =============================================================================
# pnp_lens_tests.jl — the absolute-pose problem on a DISTORTING camera: the
# residual currency is the raw pixel through the lens for scoring, the
# M-step and the chart rows alike. A board imaged by a strong
# division fisheye out to its rim, with noise and gross outliers, must be
# recovered from the RAW pixels — no undistortion anywhere — and the rim
# points must be inliers. The Pinhole path is untouched.
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays, StructArrays
using FiniteDiff: FiniteDiff
using Rotations: RotMatrix, RotXYZ
using VisualGeometryCore
using VisualGeometryCore: LeftRigidManifold6, project
using VisualGeometryCore: estimate, rotation, translation   # VGC accessors (public)
using RobustVisualGeometry
const RVGL = RobustVisualGeometry

# A 4240×2832 frame, f = 950 px, a division lens whose 90° pole sits ~1330 px
# from the centre (the hallway's Sigma 8 mm), and a board tilted towards the rim.
_lens_camera(lens) = CameraModel(; focal = 950.0, pp = (2120.0, 1416.0), lens)
const _CAMS_L = (
    "division2" => _lens_camera(Division(-0.3, -0.08)),
    "opencv5" => _lens_camera(Lens(Perspective(), Additive(Radial((-0.25, 0.06, -0.01)), Tangential(0.001, -0.0005)))),
)
# Where the board sits, per lens: the division fisheye is valid to its 90°
# pole, so its board goes far off-axis and its far edge reaches ~70° (the
# rim, > 1100 px); the OpenCV polynomial is only a model inside ~35°,
# so its board stays in that field (> 650 px counts as its rim).
const _SCENES = Dict(
    "division2" => (RotMatrix(RotXYZ(0.35, -0.35, 0.2)), Vec3{Float64}(850.0, 250.0, 350.0), 1100.0),
    "opencv5" => (RotMatrix(RotXYZ(0.35, -0.55, 0.2)), Vec3{Float64}(280.0, -60.0, 420.0), 650.0),
)

function _board_scene(rng, cam, R, t; σ = 0.3, n_out = 40)
    board = [Point3{Float64}(X, Y, 0.0) for X in -160.0:20.0:160.0 for Y in -120.0:20.0:120.0]
    inliers = [Xw => SVector{2, Float64}(project(cam, R * SVector(Xw) + t)) for Xw in board]
    pairs, gt = contaminated_pairs(
        rng, filter(c -> all(isfinite, c.second), inliers); σ, n_out, shuffle = true,
        outlier = (rng, _) -> Point3{Float64}(320 * rand(rng) - 160, 240 * rand(rng) - 120, 0.0) => SVector(4240 * rand(rng), 2832 * rand(rng))
    )
    return first.(pairs), last.(pairs), gt
end
@testset "absolute pose through a distorting lens — $name" for (name, cam) in _CAMS_L
    R_gt, t_gt, rim = _SCENES[name]
    rng = MersenneTwister(21)
    src, dst, gt = _board_scene(rng, cam, R_gt, t_gt)
    c = SVector(2120.0, 1416.0)
    radius = [norm(SVector{2, Float64}(d) - c) for d in dst]
    @test count(gt .& (radius .> rim)) >= 20                        # the scene really reaches its rim
    prob = AbsolutePoseProblem(cam, StructArray(src .=> dst))
    @test prob.context.model === cam
    # scoring: the raw-pixel reprojection through the lens — zero at the truth (up to noise)
    r = zeros(length(src)); RVGL.residuals!(r, prob, EuclideanMap(R_gt, t_gt))
    @test maximum(r[gt]) < 1.5 && minimum(r[.!gt]) > 5.0
    arms = (
        "Irls{TLS} (default)" => RVGL.Irls{RVGL.TLSShape}(),
        "Irls{Welsch}" => RVGL.Irls{RVGL.WelschShape}(scale_multiple = 1.5),
    )   # the Student-t + uniform IRLS and the Welsch HQ lift: LocalOptimizationDoneRight's suite
    @testset "$lo_name" for (lo_name, lo) in arms
        est = find_absolute_pose(cam, src .=> dst; threshold = 5.0, local_optimization = lo, rng = MersenneTwister(3))
        @test est !== nothing
        @test rotation_error_deg(estimate(est), R_gt) < 0.05
        @test translation_error(estimate(est), t_gt) < 1.0
        m = est.inlier_mask
        @test count(m .& gt) >= count(gt) - 2 && count(m .& .!gt) == 0
        @test count(m .& gt .& (radius .> rim)) >= count(gt .& (radius .> rim)) - 1    # the rim is kept
    end
    # the M-step routines against central differences of each other (central,
    # not ForwardDiff: the rows go through the lens's Float64 projection)
    n = length(src)
    R₀ = RotMatrix(RotXYZ(0.02, -0.03, 0.01)) * R_gt; t₀ = t_gt + Vec3{Float64}(-5.0, 5.0, 5.0)
    δ0 = SVector(0.01, -0.02, 0.005, 1.0, -2.0, 0.5)
    lm = RVGL._lm_state(prob, LeftRigidManifold6(R₀, t₀), ones(2n))
    @test VisualGeometryCore.check_rows_jacobian(lm, δ0, 2n; rtol = 1.0e-4, reference = (f, θ) -> FiniteDiff.finite_difference_jacobian(f, θ, Val(:central)))
    # the chart rows' refresh/Jacobian through the provider protocol (VGC's rows contract)
    seed = EuclideanMap(R₀, t₀)
    prov = VisualGeometryCore.rows_provider(prob)
    θ = VisualGeometryCore.lift_init(prov, seed)
    VisualGeometryCore.lift_refresh!(prov, θ)
    rr = zeros(n); RVGL.residuals!(rr, prob, VisualGeometryCore.lift_model(prov, θ))
    @test maximum(abs.(VisualGeometryCore.lift_residuals(prov)[1:n] .- rr)) < 1.0e-9   # one currency: the rows' norms ARE the scoring residuals
    VisualGeometryCore.lift_jacobian!(prov, θ)
    Jl = copy(VisualGeometryCore.lift_rows(prov).J[1:2n, :])
    lift_rows(v) = (VisualGeometryCore.lift_refresh!(prov, VisualGeometryCore.lift_retract(prov, θ, v)); VisualGeometryCore.lift_rows(prov).r[1:2n])
    Jc = FiniteDiff.finite_difference_jacobian(lift_rows, zero(SVector{6, Float64}), Val(:central))
    @test maximum(abs.(Jl - Jc)) < 1.0e-4 * max(1.0, maximum(abs.(Jc)))
end

@testset "absolute pose: a point the lens cannot place is an outlier, not a NaN" begin
    # A FOLDING lens (k₂ > 0): its lift stops increasing at a finite radius, so
    # there are directions it genuinely cannot image, and those are what this
    # test is about. It used to use `Division(-0.3, -0.08)`, whose k₂ < 0 means
    # NO fold and a field running to 180° — that lens images the points below
    # (166°-180° off axis) perfectly well, and the assertion only succeeded
    # because the residual was gated on `X_cam[3] > 0`, the gnomonic chart's
    # domain rather than the lens's.
    cam = _lens_camera(Division(-0.3, 0.08))
    src = [Point3{Float64}(0.0, 0.0, 0.0), Point3{Float64}(50.0, 0.0, 0.0), Point3{Float64}(0.0, 50.0, 0.0), Point3{Float64}(50.0, 50.0, 0.0)]
    dst = [Point2{Float64}(project(cam, SVector(s) + SVector(0.0, 0.0, 300.0))...) for s in src]
    prob = AbsolutePoseProblem(cam, StructArray(src .=> dst))
    r = zeros(4)
    RVGL.residuals!(r, prob, EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64})), Vec3{Float64}(0.0, 0.0, -300.0)))   # behind the camera
    @test all(==(VisualGeometryCore.OUT_OF_FIELD_RESIDUAL), r)
end
end

@testitem "contract_conformance" setup = [RadialCameraScene, HomographyScenes, EpipolarScenes, P3PScenes, PoseScenes, DivisionBoards, Contamination] begin
    import VisualGeometryToolkitCore as VgtkCore
# =============================================================================
# The K+ contracts, checked by their owner's conformance helpers
# (VisualGeometryToolkitCore's test extension): the scoring contract on the baseline
# scorers at a threshold, the constraint-problem protocol on every problem the
# suite builds scenes for, the local-optimization contract on the
# strategies this package hosts, and the chart-rows contract on the providers
# of the chart problems.
#
# `allocations = false` for the local optimizers for now: the zero-allocation
# claim is the next step, and flips to the helper's default once it holds.
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays, StructArrays
using StableRNGs
using Rotations: RotY, RotXYZ, RotMatrix
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: EuclideanMap, Correspondences
using VisualGeometryToolkitCore: test_scoring_contract, test_constraint_problem, test_local_optimization_contract,
    MsacScoring, RansacScoring, estimate, converged, correspondences, model_type, RadialCamera, RadialHomography,
    radial_transfer, WelschShape, TLSShape, data_size, codimension, minimal_sample_size, mahalanobis_distances!,
    residual_context, whitened_residual, rotation, translation, radial_rows, homography, division_coeff
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: RansacConfig, RelposeBundleProblem, BearingPoseProblem
using VisualGeometryToolkitCore: RowsWorkspace, rows_provider, lift_init, lift_refresh!, lift_jacobian!, lift_retract,
    lift_model, lift_valid, lift_rows, lift_residuals, lift_size, point_norms!
using VisualGeometryToolkitCore.Synthetic: projective_distance
using FiniteDiff: FiniteDiff

# The incumbent of a short seeded RANSAC run: a model of the problem's own type,
# near the scene's truth.
function fitted(p, threshold; seed = 1)
    r = ransac(p, MsacScoring(p; threshold); config = RansacConfig(max_trials = 300), rng = StableRNG(seed))
    @test converged(r)
    return estimate(r)
end

# --- the scenes --------------------------------------------------------------
function h_scene()
    p = HomographyProblem(correspondences(homography_pairs(StableRNG(7), H_TRUE; n_in = 80, n_out = 20, σ = 0.5)...))
    return p, fitted(p, 3.0)
end
function epipolar_scenes()
    u₁, u₂ = calibrated_pairs(StableRNG(11), RotY(0.25), normalize(SA[0.6, -0.15, 0.35]);
                              n = 110, σ = 0.001, outlier_every = 12)
    pe = ecs(u₁, u₂)                                              # E in ECAM's pixels
    pf = FundMatProblem(StructArray(u₁ .=> u₂))                   # F on the calibrated points
    return (pe, fitted(pe, 3.0)), (pf, fitted(pf, 0.005))
end
function pnp_scene()
    _, _, p, _ = p3p_scene(StableRNG(13), ECAM, 60; σ = 0.5)
    return p, fitted(p, 3.0)
end
function radial_homography_scene()
    W, H = 1200.0, 900.0
    m = RadialHomography{Float64}(
        HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0))),
        -0.25 / hypot(W, H)^2, SVector(W / 2, H / 2)
    )
    function inlier(rng, _)
        while true
            s = SVector(200 * rand(rng), 150 * rand(rng))
            d = radial_transfer(m, s)
            d === nothing && continue
            (0 < d[1] < W && 0 < d[2] < H) && return s => SVector(d[1], d[2])
        end
    end
    pairs, _ = contaminated_pairs(
        StableRNG(17), inlier, 70; σ = 0.3, n_out = 20, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng), 150 * rand(rng)) => SVector(W * rand(rng), H * rand(rng))
    )
    p = RadialHomographyProblem(pairs; center = (W / 2, H / 2))
    return p, fitted(p, 3.0)
end
function radial_camera_scene()
    p = RadialCameraProblem(first(radial_camera_pairs()); center = RC_CENTER)
    return p, fitted(p, 3.0)
end
function circle_scene()
    rng = StableRNG(29)
    pts = [Point2{Float64}(300.0 + 120cos(t) + 0.3randn(rng), 200.0 + 120sin(t) + 0.3randn(rng)) for t in 2π .* rand(rng, 80)]
    append!(pts, [Point2{Float64}(600rand(rng), 400rand(rng)) for _ in 1:20])
    p = CircleProblem(pts)
    return p, fitted(p, 1.0)
end
function ellipse_scene()
    rng = StableRNG(31)
    pts = [Point2{Float64}(300.0 + 150cos(t) * cos(0.5) - 90sin(t) * sin(0.5) + 0.3randn(rng),
                           200.0 + 150cos(t) * sin(0.5) + 90sin(t) * cos(0.5) + 0.3randn(rng)) for t in 2π .* rand(rng, 80)]
    append!(pts, [Point2{Float64}(600rand(rng), 400rand(rng)) for _ in 1:20])
    p = EllipseProblem(pts)
    return p, fitted(p, 1.0)
end
function line_scenes()
    rng = StableRNG(23)
    pts = [Point2{Float64}(10 * rand(rng) - 5, 0.0) for _ in 1:60]
    pts = [Point2{Float64}(q[1], 2q[1] + 1 + 0.05 * randn(rng)) for q in pts]
    append!(pts, [Point2{Float64}(30 * rand(rng) - 10, 30 * rand(rng) - 10) for _ in 1:20])
    pl = LineFittingProblem(pts)
    pih = InhomLineFittingProblem([SVector(q[1], q[2]) for q in pts])
    return (pl, fitted(pl, 0.25)), (pih, fitted(pih, 0.25))
end
function hand_eye_scene()
    rng = StableRNG(29)
    sc = robot_world_hand_eye_scene(rng; N = 30, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1),
                                    A = (ang_max = π / 3, t_scale = 0.5))
    X, Y, A = only(sc.X), sc.Y, sc.A
    B = [perturb_pose(rng, marker_pose(Y, A[i], X); σ_R = 0.001, σ_t = 0.001) for i in eachindex(A)]
    p = HandEyeProblem(A .=> B)
    r = find_hand_eye(A .=> B; rng = StableRNG(31))
    @test converged(r)
    return p, convert(model_type(p), estimate(r))
end

@testset "scoring contract: the baseline scorers at a threshold" begin
    rng = StableRNG(3)
    r = vcat(abs.(0.8 .* randn(rng, 150)), 40.0 .* rand(rng, 50))
    for sc in (MsacScoring(2; threshold = 3.0), RansacScoring(2; threshold = 3.0),
               MsacScoring(1; threshold = 1.5), RansacScoring(1; threshold = 1.5))
        test_scoring_contract(sc, r; d_g = VgtkCore.codimension(sc))
        test_scoring_contract(sc, r; d_g = VgtkCore.codimension(sc), sample = [1, 2, 3, 4])
    end
end

# The relative-pose bundle's scoring problem (its own residual context: the
# cheiral pose of θ, the optimally triangulated reprojection) on a noisy
# two-view scene, at the true essential matrix.
function relpose_bundle_scene()
    rng = StableRNG(37)
    K = SMatrix{3, 3, Float64, 9}(800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 320.0, 240.0, 1.0)
    R = SMatrix{3, 3, Float64, 9}(RotY(0.2)); t = normalize(SVector(0.5, -0.1, 0.2))
    Xs = [SVector(2rand(rng) - 1, 2rand(rng) - 1, 4 + 2rand(rng)) for _ in 1:60]
    px(X) = (x = K * X; SVector(x[1] / x[3], x[2] / x[3]))
    u1 = [px(X) + 0.5 * randn(rng, SVector{2, Float64}) for X in Xs]
    u2 = [px(R * X + t) + 0.5 * randn(rng, SVector{2, Float64}) for X in Xs]
    return RelposeBundleProblem(K, K, correspondences(u1, u2)), VgtkCore.EssentialMat(R, t)
end
# The bearing pose problem (central bearings, a 2-component tangent residual),
# at the true pose, with noisy bearings.
function bearing_pose_scene()
    rng = StableRNG(41)
    E = EuclideanMap(RotMatrix(RotXYZ(0.1, -0.2, 0.3)), SVector(0.1, -0.2, 3.0))
    Xs = [SVector(2rand(rng) - 1, 2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:50]
    fs = [normalize(E.R * X + E.t + 0.003 * randn(rng, SVector{3, Float64})) for X in Xs]
    return BearingPoseProblem(Correspondences(Xs, fs)), E
end

@testset "constraint-problem protocol: every problem with a scene" begin
    (pe, θe), (pf, θf) = epipolar_scenes()
    (pl, θl), (pih, θi) = line_scenes()
    # One rule for every problem, Core's: the batch rows and the per-point form
    # agree to rounding at the residuals' scale (E measured ≤ 5.7e-14 px against
    # residuals up to 271 px, 2026-10-09; PnP ≤ 1.3e-13 px; FMA contraction
    # differs between the vectorized loop and a single call).
    scenes = (h_scene(), radial_homography_scene(), radial_camera_scene(), circle_scene(), ellipse_scene(), (pl, θl), (pih, θi),
        hand_eye_scene(), pnp_scene(), (pe, θe), (pf, θf), relpose_bundle_scene(), bearing_pose_scene())
    for (p, θ) in scenes
        test_constraint_problem(p, θ)
    end
    # A scalar constraint's residual is SIGNED: on noisy data the points lie on
    # both sides of the model, so a residual whose sign was discarded (a
    # distance under the residual's name) shows as no negative entry.
    @testset "d_g = 1: the residual keeps its sign ($(nameof(typeof(p))))" for (p, θ) in [s for s in scenes if codimension(s[1]) == 1]
        ctx = residual_context(p, θ)
        w = [whitened_residual(p, ctx, i) for i in 1:data_size(p)]
        @test any(<(0), w) && any(>(0), w)
    end
end

# The core's strategies; the strategies of LocalOptimizationDoneRight run the
# same helper in its suite.
@testset "local-optimization contract: the hosted strategies (H scene)" begin
    p, θ = h_scene()
    msac = MsacScoring(p; threshold = 3.0)
    for (lo, sc) in (
            (nothing, msac),
            (Irls{WelschShape}(efficiency_constant = 1.5), msac),
            (Irls{TLSShape}(), msac),                              # follows the scorer's threshold
            (Irls{TLSShape}(), RansacScoring(p; threshold = 3.0)),
        )
        test_local_optimization_contract(lo, p, θ, sc; allocations = false)
    end
end

@testset "local-optimization contract: the hosted strategies (E scene)" begin
    (p, θ), _ = epipolar_scenes()
    msac = MsacScoring(p; threshold = 2.0)
    for (lo, sc) in (
            (nothing, msac),
            (Irls{WelschShape}(efficiency_constant = 1.5), msac),
            (Irls{TLSShape}(), msac),
        )
        test_local_optimization_contract(lo, p, θ, sc; allocations = false)
    end
end

# =============================================================================
# The chart-rows contract (VisualGeometryToolkitCore robust/rows.jl) on every chart
# problem's provider: dims, the residual currency, the rows' Jacobian against
# central differences of the provider's own refresh along its retraction, and
# the chart round trip.
# =============================================================================
# Models agree up to the representative: projectively for the matrix models,
# by rows for the radial camera, exactly (to rounding) for a pose, a circle and an ellipse.
same_model(a::VgtkCore.EuclideanMap, b) = norm(SMatrix{3, 3}(rotation(a)) - SMatrix{3, 3}(rotation(b))) < 1.0e-10 &&
    norm(translation(a) - translation(b)) < 1.0e-10 * max(1.0, norm(translation(b)))
same_model(a::RadialCamera, b::RadialCamera) = projective_distance(radial_rows(a), radial_rows(b)) < 1.0e-10
same_model(a::RadialHomography, b::RadialHomography) =
    projective_distance(SMatrix{3, 3}(homography(a)), SMatrix{3, 3}(homography(b))) < 1.0e-10 &&
    isapprox(division_coeff(a), division_coeff(b); rtol = 1.0e-10)
same_model(a::VgtkCore.Circle, b::VgtkCore.Circle) =
    norm(VgtkCore.ambient_coordinates(a) - VgtkCore.ambient_coordinates(b)) < 1.0e-10 * norm(VgtkCore.ambient_coordinates(b))
same_model(a::VgtkCore.Ellipse, b::VgtkCore.Ellipse) =
    norm(VgtkCore.ambient_coordinates(a) - VgtkCore.ambient_coordinates(b)) < 1.0e-10 * norm(VgtkCore.ambient_coordinates(b))
same_model(a, b) = projective_distance(SMatrix{3, 3, Float64}(a), SMatrix{3, 3, Float64}(b)) < 1.0e-10

rows_dims(::RowsWorkspace{D, C}) where {D, C} = (D, C)

# The rows' Jacobian by central differences of the provider's own refresh along
# its retraction, one chart direction at a time, the step sized so the rows
# move by ~1e-3 along it: the chart directions differ in scale by orders of
# magnitude (the radial homography's k against its H block), and one step for
# all of them is either truncation- or roundoff-bound on some.
function rows_jacobian_fd(prov, ch, J)
    rws = lift_rows(prov)
    D = size(J, 2)
    rows_at(v) = (lift_refresh!(prov, lift_retract(prov, ch, v)); copy(rws.r))
    Jc = similar(J, size(J)...)
    for k in 1:D
        h = 1.0e-3 / max(maximum(abs, view(J, :, k)), 1.0e-12)
        e = SVector{D, Float64}(ntuple(i -> i == k ? h : 0.0, D))
        Jc[:, k] = (rows_at(e) - rows_at(-e)) / 2h
    end
    lift_refresh!(prov, ch)
    return Jc
end

# `inliers`: the points the comparison with the scoring residual and with the
# differenced refresh is restricted to — every point, except for the ellipse,
# whose rows are a different (orthogonal) distance: equal to the scoring
# residual only to first order, so measured on its inliers. (Until 0.4 the
# homography's rows froze their transfer whitening at the anchor and needed the
# same allowance; they are now the exact rows of the scoring residual.)
function check_rows_provider(p, θ; inliers = trues(data_size(p)), currency_tol = 1.0e-10, jacobian_tol = 1.0e-5)
    n = data_size(p)
    prov = rows_provider(p)
    rws = lift_rows(prov)
    D, C = rows_dims(rws)
    @test lift_size(prov) == n
    @test C == codimension(p)
    @test size(rws.J) == (C * n, D) && length(rws.r) == C * n && length(rws.rn) == n
    @test length(rws.onev) == C * n && all(==(1.0), rws.onev)
    ch = lift_init(prov, θ)
    # the chart round trip: the chart at the model reads the model back
    @test same_model(lift_model(prov, ch), θ)
    lift_refresh!(prov, ch)
    # the residual currency: the rows themselves at codim 1, the per-point
    # norms of the C-row groups at codim 2 — the problem's scoring residual
    if C == 1
        @test lift_residuals(prov) === rws.r
    else
        @test lift_residuals(prov) === rws.rn
        @test rws.rn ≈ point_norms!(zeros(n), rws.r, n, C) rtol = 1.0e-14
    end
    r = zeros(n); mahalanobis_distances!(r, p, θ)
    res = lift_residuals(prov)
    @test maximum(i -> abs(abs(res[i]) - abs(r[i])) / max(abs(r[i]), 1.0), findall(inliers)) < currency_tol
    # the participation mask: every point, unless the problem masks some (PnP)
    valid = lift_valid(prov)
    @test valid === nothing || (valid isa AbstractVector{Bool} && length(valid) >= n)
    # the Jacobian at the anchor (its rows refreshed first) against the
    # differenced refresh
    lift_refresh!(prov, ch)
    lift_jacobian!(prov, ch)
    J = copy(rws.J)
    Jc = rows_jacobian_fd(prov, ch, J)
    rows = [C * (i - 1) + c for i in findall(inliers) for c in 1:C]
    @test maximum(abs.(J[rows, :] - Jc[rows, :])) / maximum(abs, Jc[rows, :]) < jacobian_tol
    return nothing
end

@testset "chart-rows contract: every chart problem's provider" begin
    # a lift runs on the MinimalSolveProblem family under its own name: the alias is retired (check_naming), no alias
    @test !isdefined(VisualGeometryToolkitEstimation, Symbol("Liftable" * "Problem"))
    (pe, θe), (pf, θf) = epipolar_scenes()
    ph, θh = h_scene()
    for (name, (p, θ)) in (
            ("homography", (ph, θh)),
            ("essential", (pe, θe)),
            ("fundmat", (pf, θf)),
            ("absolute_pose", pnp_scene()),
            ("radial_homography", radial_homography_scene()),
            ("radial_camera", radial_camera_scene()),
            ("circle", circle_scene()),
        )
        @testset "$name" begin
            check_rows_provider(p, θ)
        end
    end
    @testset "ellipse" begin
        # its rows are the ORTHOGONAL distances, the scoring's Sampson distance to
        # first order: the currency is compared on the inliers — their difference is second order, δ²·κ/2 with the curvature κ ≤ a/b²
        # (0.019 here): ≤ 0.01 at |δ| < 1 px, measured 5.1e-3
        pl, θl = ellipse_scene()
        rl = zeros(data_size(pl)); mahalanobis_distances!(rl, pl, θl)
        check_rows_provider(pl, θl; inliers = abs.(rl) .< 1, currency_tol = 1.0e-2)
    end
end
end

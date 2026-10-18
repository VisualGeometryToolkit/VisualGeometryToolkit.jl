@testitem "contract_conformance" setup = [HomographyScenes, EpipolarScenes, P3PScenes, PoseScenes, DivisionBoards, Contamination] begin
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
using Rotations: RotY, RotXYZ
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: test_scoring_contract, test_constraint_problem, test_local_optimization_contract,
    MsacScoring, RansacScoring, estimate, converged, csponds, model_type, RadialCamera, RadialHomography,
    radial_transfer, WelschShape, TLSShape, data_size, codimension, minimal_sample_size, residuals!,
    residual_context, whitened_residual, rotation, translation, radial_rows, homography, division_coeff
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: RansacConfig
using VisualGeometryToolkitCore: RowsWorkspace, rows_provider, lift_init, lift_refresh!, lift_jacobian!, lift_retract,
    lift_model, lift_valid, lift_rows, lift_residuals, lift_size, point_norms!
using VisualGeometryToolkitCore.Synthetic: projective_distance
using FiniteDiff: FiniteDiff

# The incumbent of a short seeded RANSAC run: a model of the problem's own type,
# near the scene's truth.
function fitted(p, threshold; seed = 1)
    r = ransac(p, MsacScoring(p; threshold); config = RansacConfig(max_trials = 300), rng = MersenneTwister(seed))
    @test converged(r)
    return estimate(r)
end

# --- the scenes --------------------------------------------------------------
function h_scene()
    p = HomographyProblem(csponds(homography_pairs(MersenneTwister(7), H_TRUE; n_in = 80, n_out = 20, σ = 0.5)...))
    return p, fitted(p, 3.0)
end
function epipolar_scenes()
    u₁, u₂ = calibrated_pairs(MersenneTwister(11), RotY(0.25), normalize(SA[0.6, -0.15, 0.35]);
                              n = 110, σ = 0.001, outlier_every = 12)
    pe = ecs(u₁, u₂)                                              # E in ECAM's pixels
    pf = FundMatProblem(StructArray(u₁ .=> u₂))                   # F on the calibrated points
    return (pe, fitted(pe, 3.0)), (pf, fitted(pf, 0.005))
end
function pnp_scene()
    _, _, p, _ = p3p_scene(MersenneTwister(13), ECAM, 60; σ = 0.5)
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
        MersenneTwister(17), inlier, 70; σ = 0.3, n_out = 20, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng), 150 * rand(rng)) => SVector(W * rand(rng), H * rand(rng))
    )
    p = RadialHomographyProblem(pairs; center = (W / 2, H / 2))
    return p, fitted(p, 3.0)
end
function radial_camera_scene()
    c = SVector(2000.0, 1500.0)
    cam = board_camera(RotXYZ(0.35, -0.25, 0.15), SVector(40.0, -30.0, 600.0); f = 900.0, ks = (-2.0e-7,), center = c)
    pairs, _ = contaminated_pairs(
        MersenneTwister(19), board_pixels(cam; grid = -100.0:25.0:100.0); σ = 0.2, n_out = 20, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng) - 100, 200 * rand(rng) - 100) => SVector(4000 * rand(rng), 3000 * rand(rng))
    )
    p = RadialCameraProblem(pairs; center = c)
    return p, fitted(p, 3.0)
end
function line_scenes()
    rng = MersenneTwister(23)
    pts = [Point2{Float64}(10 * rand(rng) - 5, 0.0) for _ in 1:60]
    pts = [Point2{Float64}(q[1], 2q[1] + 1 + 0.05 * randn(rng)) for q in pts]
    append!(pts, [Point2{Float64}(30 * rand(rng) - 10, 30 * rand(rng) - 10) for _ in 1:20])
    pl = LineFittingProblem(pts)
    pih = InhomLineFittingProblem([SVector(q[1], q[2]) for q in pts])
    return (pl, fitted(pl, 0.25)), (pih, fitted(pih, 0.25))
end
function hand_eye_scene()
    rng = MersenneTwister(29)
    sc = robot_world_hand_eye_scene(rng; N = 30, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1),
                                    A = (ang_max = π / 3, t_scale = 0.5))
    X, Y, A = only(sc.X), sc.Y, sc.A
    B = [perturb_pose(rng, marker_pose(Y, A[i], X); σ_R = 0.001, σ_t = 0.001) for i in eachindex(A)]
    p = HandEyeProblem(A .=> B)
    r = find_hand_eye(A .=> B; rng = MersenneTwister(31))
    @test converged(r)
    return p, convert(model_type(p), estimate(r))
end

@testset "scoring contract: the baseline scorers at a threshold" begin
    rng = MersenneTwister(3)
    r = vcat(abs.(0.8 .* randn(rng, 150)), 40.0 .* rand(rng, 50))
    for sc in (MsacScoring(2; threshold = 3.0), RansacScoring(2; threshold = 3.0),
               MsacScoring(1; threshold = 1.5), RansacScoring(1; threshold = 1.5))
        test_scoring_contract(sc, r; d_g = VgtkCore.codimension(sc))
        test_scoring_contract(sc, r; d_g = VgtkCore.codimension(sc), sample = [1, 2, 3, 4])
    end
end

@testset "constraint-problem protocol: every problem with a scene" begin
    (pe, θe), (pf, θf) = epipolar_scenes()
    (pl, θl), (pih, θi) = line_scenes()
    for (p, θ) in (h_scene(), radial_homography_scene(), radial_camera_scene(), (pl, θl), (pih, θi), hand_eye_scene(), pnp_scene())
        test_constraint_problem(p, θ)
    end
    # E and F: `residuals!` (the batch rows) and `whitened_residual` (the
    # per-point form) agree only to rounding, not bit for bit as the helper
    # demands — so the agreement is checked at the rounding of the residuals'
    # own scale. Measured 2026-10-09: E ≤ 5.7e-14 px against residuals up to
    # 271 px (a minimal-sample point, |r| ~ 4e-14, differs by 36% of itself:
    # rounding, which no relative tolerance on the point alone admits); F
    # ≤ 6.4e-17. Whether the two paths happen to agree bitwise depends on
    # code generation (under `--check-bounds=yes`, as `Pkg.test` runs, F agrees
    # bitwise and E does not), so bitwise equality is not tested here until
    # the two paths are one. Everything else of the protocol holds. (PnP's two
    # paths became one with the lens-family merge: it runs the full protocol
    # above.)
    for (p, θ) in ((pe, θe), (pf, θf))
        @testset "constraint problem (to rounding): $(nameof(typeof(p)))" begin
            n = data_size(p)
            @test n isa Int && codimension(p) isa Int && minimal_sample_size(p) isa Int
            r = Vector{Float64}(undef, n)
            @inferred residuals!(r, p, θ)
            @test (@allocated residuals!(r, p, θ)) == 0
            ctx = residual_context(p, θ)
            w = [Float64(whitened_residual(p, ctx, i)) for i in 1:n]
            scale = maximum(abs, r)
            @test all(i -> isapprox(w[i], r[i]; rtol = 1.0e-11, atol = 1.0e-14 * scale), 1:n)
        end
    end
end

# The core's strategies; the strategies of LocalOptimizationDoneRight run the
# same helper in its suite.
@testset "local-optimization contract: the hosted strategies (H scene)" begin
    p, θ = h_scene()
    msac = MsacScoring(p; threshold = 3.0)
    for (lo, sc) in (
            (NoLocalOptimization(), msac),
            (Irls{WelschShape}(scale_multiple = 1.5), msac),
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
            (NoLocalOptimization(), msac),
            (Irls{WelschShape}(scale_multiple = 1.5), msac),
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
# by rows for the radial camera, exactly (to rounding) for a pose.
same_model(a::VgtkCore.EuclideanMap, b) = norm(SMatrix{3, 3}(rotation(a)) - SMatrix{3, 3}(rotation(b))) < 1.0e-10 &&
    norm(translation(a) - translation(b)) < 1.0e-10 * max(1.0, norm(translation(b)))
same_model(a::RadialCamera, b::RadialCamera) = projective_distance(radial_rows(a), radial_rows(b)) < 1.0e-10
same_model(a::RadialHomography, b::RadialHomography) =
    projective_distance(SMatrix{3, 3}(homography(a)), SMatrix{3, 3}(homography(b))) < 1.0e-10 &&
    isapprox(division_coeff(a), division_coeff(b); rtol = 1.0e-10)
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
# differenced refresh is restricted to — every point, except for the
# homography, whose rows FREEZE their transfer whitening at the anchor: its
# currency is the scoring residual to first order, and a refresh at a moved
# anchor re-whitens, which the frozen-whitening Jacobian does not see — both
# differences scale with the residual, so they are measured on the inliers.
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
    r = zeros(n); residuals!(r, p, θ)
    res = lift_residuals(prov)
    @test maximum(i -> abs(abs(res[i]) - abs(r[i])) / max(abs(r[i]), 1.0), findall(inliers)) < currency_tol
    # the participation mask: every point, unless the problem masks some (PnP)
    valid = lift_valid(prov)
    @test valid === nothing || (valid isa AbstractVector{Bool} && length(valid) >= n)
    # the Jacobian at the anchor (its rows refreshed first: H freezes its
    # whitening there) against the differenced refresh
    lift_refresh!(prov, ch)
    lift_jacobian!(prov, ch)
    J = copy(rws.J)
    Jc = rows_jacobian_fd(prov, ch, J)
    rows = [C * (i - 1) + c for i in findall(inliers) for c in 1:C]
    @test maximum(abs.(J[rows, :] - Jc[rows, :])) / maximum(abs, Jc[rows, :]) < jacobian_tol
    return nothing
end

@testset "chart-rows contract: every chart problem's provider" begin
    (pe, θe), (pf, θf) = epipolar_scenes()
    ph, θh = h_scene()
    rh = zeros(data_size(ph)); residuals!(rh, ph, θh)
    @testset "homography" begin
        # measured on this scene's 80 inliers: currency 1.6e-4 relative, Jacobian 1.1e-3
        check_rows_provider(ph, θh; inliers = abs.(rh) .< 3, currency_tol = 1.0e-3, jacobian_tol = 1.0e-2)
    end
    for (name, (p, θ)) in (
            ("essential", (pe, θe)),
            ("fundmat", (pf, θf)),
            ("absolute_pose", pnp_scene()),
            ("radial_homography", radial_homography_scene()),
            ("radial_camera", radial_camera_scene()),
        )
        @testset "$name" begin
            check_rows_provider(p, θ)
        end
    end
end
end

@testitem "ransac_homography" setup = [HomographyScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using LinearAlgebra
using StaticArrays
using Random
using StableRNGs
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: Csponds, csponds, exact
using VisualGeometryToolkitCore: homography_4pt   # the positional minimal kernel (public, not exported)
using VisualGeometryToolkitCore: converged, estimate   # VGC accessors (public)
using VisualGeometryToolkitCore: MsacScoring, RansacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation: UniformSampler, refine!, inlier_ratio, sampler   # public, not exported
using VisualGeometryToolkitEstimation
using Rotations: RotMatrix, RotX, RotZ
using VisualGeometryToolkitEstimation: minimal_sample_size, data_size, model_type,
    ProsacSampler, reset!

# =============================================================================
# Test utilities
# =============================================================================

function make_homography_data(; n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42)
    source_pts, target_pts = homography_pairs(StableRNG(seed), H_TRUE; n_in = n_inliers, n_out = n_outliers, σ = noise)
    return source_pts, target_pts, H_TRUE, n_inliers
end

@testset "RANSAC Homography" begin

    @testset "_orientation_preserving — uniform-sign rule" begin
        # Unit square, counterclockwise in each frame's native labelling.
        square = (SA[0.0, 0.0], SA[1.0, 0.0], SA[1.0, 1.0], SA[0.0, 1.0])

        # Same handedness (image→image, y-down on both sides): accept.
        @test VgtkCore._orientation_preserving(square, square)

        # Uniform flip (e.g. world y-up board ↔ image y-down pixels).
        # Negating y turns every triangle sign simultaneously → all 4 σᵢ < 0.
        flipped_y = map(p -> SA[p[1], -p[2]], square)
        @test VgtkCore._orientation_preserving(square, flipped_y)

        # Uniform flip on a non-trivial (projective-ish) target: still accept.
        warped = (SA[0.0, 0.0], SA[2.0, 0.3], SA[1.8, 2.5], SA[-0.1, 2.0])
        warped_flipped = map(p -> SA[p[1], -p[2]], warped)
        @test VgtkCore._orientation_preserving(square, warped_flipped)

        # Bow-tie via adjacent swap: target positions 3↔4 swap makes edges
        # (p2→p3) and (p4→p1) cross. Triangle signs come out mixed
        # (+, +, −, −) against the CCW source → reject.
        bowtie = (square[1], square[2], square[4], square[3])
        @test !VgtkCore._orientation_preserving(square, bowtie)

        # Bow-tie via diagonal swap: target positions 2↔3 swap draws the two
        # diagonals of the quadrilateral, which cross at (0.5, 0.5). Triangle
        # signs come out mixed (−, +, +, −) → reject.
        bowtie_diag = (square[1], square[3], square[2], square[4])
        @test !VgtkCore._orientation_preserving(square, bowtie_diag)

        # Uniform reflection (positions 1↔3 swap): every target triangle
        # flips sign together. The uniform-sign rule accepts this — it is
        # a coordinate-label reflection, not a self-intersecting map.
        uniform_reflection = (square[3], square[2], square[1], square[4])
        @test VgtkCore._orientation_preserving(square, uniform_reflection)

        # Degenerate source (3 collinear points) — one σ is exactly 0, which
        # contributes no evidence; the rest are consistent → accept.
        collinear_src = (SA[0.0, 0.0], SA[1.0, 0.0], SA[2.0, 0.0], SA[0.0, 1.0])
        @test VgtkCore._orientation_preserving(collinear_src, square)
    end

    @testset "test_sample — object→image (y-up ↔ y-down) accepts uniform flip" begin
        # Board coordinates: world y-up (right-handed 2D in its native frame).
        board = [
            SA[-1.0, -1.0], SA[1.0, -1.0], SA[1.0, 1.0], SA[-1.0, 1.0],
            SA[0.0, 0.5], SA[0.5, 0.0],
        ]
        # Image coordinates: simulated y-down projection of the board
        # (uniform sign flip on y, plus a translation into pixel space).
        img = [SA[300.0 + 200.0 * b[1], 400.0 - 200.0 * b[2]] for b in board]

        p = HomographyProblem(csponds(board, img))
        # Every 4-subset should succeed at test_sample — none are bow-ties, all are
        # uniform-flip. Under the old rule every sample would be rejected.
        for idx in ([1, 2, 3, 4], [1, 3, 5, 6], [2, 3, 4, 5], [3, 4, 5, 6], [1, 2, 4, 6])
            @test VgtkEstimation.test_sample(p, idx)
        end

        # And a full RANSAC converges on the clean board→image data.
        result = ransac(
            p, MsacScoring(p; threshold = 2.0);
            config = RansacConfig(max_trials = 500, min_trials = 50),
            rng = StableRNG(126)
        )
        @test converged(result)
        @test sum(result.inlier_mask) == length(board)
    end

    @testset "HomographyProblem construction" begin
        src = [SA[1.0, 2.0], SA[3.0, 4.0], SA[5.0, 6.0], SA[7.0, 8.0]]
        dst = [SA[2.0, 3.0], SA[4.0, 5.0], SA[6.0, 7.0], SA[8.0, 9.0]]
        p = HomographyProblem(csponds(src, dst))
        @test minimal_sample_size(p) == 4
        @test data_size(p) == 4
        @test model_type(p) == HomographyMat{Float64}
        @test p isa HomographyProblem{Float64}

        # Too few correspondences
        @test_throws ArgumentError HomographyProblem(csponds(src[1:3], dst[1:3]))
    end

    # The problem's constraint triple — what scoring whitens — IS the owner's
    # (VGC `constraint_with_jacobians(H, s, d)`, whose derivatives VGC's own
    # tests verify) evaluated on the problem's data.
    @testset "constraint_with_jacobians — the owner's, on the problem's data" begin
        src_pts, dst_pts = homography_pairs(StableRNG(77), H_TRUE; n_in = 10, σ = 2.0)
        prob = HomographyProblem(src_pts .=> dst_pts)
        for i in 1:10
            g, G_x, G_θ = VgtkEstimation.constraint_with_jacobians(prob, H_TRUE, i)
            g₀, G_x₀, G_θ₀ = VgtkCore.constraint_with_jacobians(H_TRUE, src_pts[i], dst_pts[i])
            @test g == g₀ && G_x == G_x₀ && G_θ == G_θ₀
        end
    end

    @testset "4-point DLT solver" begin
        # Identity recovery on noiseless points
        src = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        p_id = HomographyProblem(csponds(src, copy(src)))
        models = solve(p_id, [1, 2, 3, 4])
        @test !isnothing(models)
        H_id_recovered = first(models)
        I3 = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        H_id = HomographyMat{Float64}(Tuple(I3 / norm(I3)))
        @test max_transfer_error(
            H_id_recovered, H_id,
            [SA[300.0, 400.0], SA[600.0, 300.0]]
        ) < 1.0e-10

        # Edge case: very small h33 (near-degenerate for h33=1 affine form)
        H_raw = SMatrix{3, 3, Float64, 9}(1.0, 0.0, 0.002, 0.0, 1.0, 0.001, 0.0, 0.0, 1.0e-4)
        H_raw = H_raw / norm(H_raw)
        H_raw[3, 3] < 0 && (H_raw = -H_raw)
        H_true = HomographyMat{Float64}(Tuple(H_raw))
        dst = homography_warp(H_true).(src)
        H = homography_4pt(
            src[1], src[2], src[3], src[4],
            dst[1], dst[2], dst[3], dst[4]
        )
        @test !isnothing(H)
        @test max_transfer_error(
            H, H_true,
            [SA[300.0, 400.0], SA[600.0, 300.0]]
        ) < 1.0e-6

        # Collinear source → returns nothing
        src_c = [SA[0.0, 0.0], SA[1.0, 1.0], SA[2.0, 2.0], SA[3.0, 3.0]]
        dst_ok = [SA[10.0, 20.0], SA[30.0, 10.0], SA[50.0, 40.0], SA[20.0, 60.0]]
        @test isnothing(
            homography_4pt(
                src_c[1], src_c[2], src_c[3], src_c[4],
                dst_ok[1], dst_ok[2], dst_ok[3], dst_ok[4]
            )
        )
    end

    @testset "RANSAC end-to-end" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        problem = HomographyProblem(csponds(source_pts, target_pts))

        result = ransac(
            problem, MsacScoring(problem; threshold = 3.0);
            config = RansacConfig(max_trials = 3000, min_trials = 200),
            rng = StableRNG(127)
        )

        @test converged(result)
        test_pts = [
            SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
            SA[200.0, 200.0], SA[700.0, 500.0],
        ]
        @test max_transfer_error(estimate(result), H_true, test_pts) < 2.0
        @test inlier_ratio(result) > 0.5
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

    @testset "fit — weighted DLT roundtrip" begin
        H_true = H_TRUE

        n = 20
        src, dst = homography_pairs(StableRNG(42), H_true; n_in = n)

        p = HomographyProblem(csponds(src, dst))
        # Algebraic no-θ fit needs the DLT workspace explicitly — the problem's
        # LO workspace is now the sl(3) LM buffer (NonlinLsq refit_kind).
        ws = VgtkEstimation.DltWorkspace{9, Float64}(
            VgtkEstimation.codimension(p) * VgtkEstimation.data_size(p)
        )

        H_rec = refine!(ws, p, nothing, ones(n))
        @test !isnothing(H_rec)
        @test max_transfer_error(
            H_rec, H_true,
            [
                SA[300.0, 400.0], SA[600.0, 300.0],
                SA[450.0, 550.0],
            ]
        ) < 1.0e-8

        # Too-few-inliers boundary: weights with only `min_sample_size` non-zero
        # entries fall short of `min_sample_size + 1` and the refit returns nothing.
        w_few = zeros(n); w_few[1:4] .= 1.0
        @test isnothing(refine!(ws, p, nothing, w_few))
    end

    # (The Student-t + uniform IRLS arm of this run is LocalOptimizationDoneRight's test.)
    @testset "LO-RANSAC — Irls{Welsch}" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        problem = HomographyProblem(csponds(source_pts, target_pts))

        result = ransac(
            problem, MsacScoring(problem; threshold = 3.0);
            local_optimization = Irls{WelschShape}(scale_multiple = 1.5),
            config = RansacConfig(max_trials = 3000, min_trials = 200),
            rng = StableRNG(128)
        )

        @test converged(result)
        test_pts = [
            SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
            SA[200.0, 200.0], SA[700.0, 500.0],
        ]
        @test max_transfer_error(estimate(result), H_true, test_pts) < 2.0
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

    @testset "CountedLO — instrumentation wrapper answers as the wrapped strategy" begin
        source_pts, target_pts, _, _ = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        problem = HomographyProblem(csponds(source_pts, target_pts))
        inner = Irls{WelschShape}(scale_multiple = 1.5)
        counted = VgtkEstimation.CountedLO(inner)
        run(lo) = ransac(
            problem, MsacScoring(problem; threshold = 3.0);
            local_optimization = lo,
            config = RansacConfig(max_trials = 3000, min_trials = 200),
            rng = StableRNG(128)
        )
        r_inner = run(inner)
        r_counted = run(counted)
        @test estimate(r_counted) == estimate(r_inner)
        @test counted.calls[] >= 1
        @test counted.iterations[] >= counted.calls[]
        @test VgtkCore.lo_width(counted) == VgtkCore.lo_width(inner)
        @test VgtkCore.lo_label(counted) == VgtkCore.lo_label(inner)
        @test VgtkCore.scale_interface(counted) == VgtkCore.scale_interface(inner)
        @test VgtkCore.optimization_path(counted) == VgtkCore.optimization_path(inner)
        @test VgtkCore.loss_family(counted) == VgtkCore.loss_family(inner)
    end

    @testset "PROSAC with scored correspondences" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )

        # High scores for inliers, low scores for outliers
        rng = StableRNG(99)
        n_total = length(source_pts)
        # The prior is the correspondences' `:score` attribute column.
        scores = [i <= n_inliers ? 0.8 + 0.2 * rand(rng) : 0.1 * rand(rng) for i in 1:n_total]
        problem = HomographyProblem(Csponds(source_pts, target_pts; score = scores))
        @test sampler(problem) isa ProsacSampler

        result = ransac(
            problem, MsacScoring(problem; threshold = 3.0);
            config = RansacConfig(max_trials = 3000, min_trials = 200),
            rng = StableRNG(131)
        )

        @test converged(result)
        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_transfer_error(estimate(result), H_true, test_pts) < 2.0
        # PROSAC's growth horizon is the run's budget, not a fixed constant.
        @test sampler(problem).T_N == 3000
    end

    @testset "ProsacSampler — pool expansion" begin
        sorted = collect(1:20)
        ps = ProsacSampler(sorted, 4; T_N = 1000)
        @test ps.n == 4
        @test ps.t == 0

        idx = Vector{Int}(undef, 4)
        for _ in 1:1000
            VgtkEstimation._draw_prosac!(Random.default_rng(), idx, ps)
        end
        @test ps.n == 20  # fully expanded by T_N draws

        reset!(ps)
        @test ps.t == 0
        @test ps.n == 4

        # reset!(ps, budget) re-horizons the schedule; a sampler with no
        # schedule ignores the budget.
        reset!(ps, 200)
        @test ps.T_N == 200 && ps.n == 4 && ps.t == 0
        for _ in 1:200
            VgtkEstimation._draw_prosac!(Random.default_rng(), idx, ps)
        end
        @test ps.n == 20
        @test_throws ArgumentError reset!(ps, 0)
        @test reset!(UniformSampler(10), 500) === nothing
        # The constructor's default horizon is the shipped [ransac] max_trials.
        @test ProsacSampler(sorted, 4).T_N == RansacConfig().max_trials
    end

    # -------------------------------------------------------------------------
    # MAPPED homography: pixels in, `K⁻¹`-style measurement maps, a model-frame
    # H out. The maps are anisotropic, differ per view and carry a ROLL, so a
    # consumer that applies the model-frame H to raw pixels (the LM refit, the
    # local optimizers, the consensus gate) is caught rather than accidentally agreeing.
    # -------------------------------------------------------------------------
    _roll(θ) = SMatrix{3, 3, Float64, 9}(RotZ(θ))
    _K(fx, fy, cx, cy) = SMatrix{3, 3, Float64, 9}(calibration_matrix(CameraModel(; focal = (fx, fy), pp = (cx, cy))))
    C₁ = _roll(deg2rad(70)) * inv(_K(820.0, 790.0, 480.0, 400.0))
    C₂ = _roll(deg2rad(-110)) * inv(_K(760.0, 805.0, 510.0, 380.0))
    # Model-frame H ↔ pixel H, by composition with the maps: H_px = inv(C₂) ∘ H ∘ C₁.
    _to_px(H) = inv(HomographyMap(C₂)) ∘ H ∘ HomographyMap(C₁)
    _to_model(H) = HomographyMap(C₂) ∘ H ∘ inv(HomographyMap(C₁))
    mapped_test_pts = [
        SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
        SA[200.0, 200.0], SA[700.0, 500.0],
    ]

    @testset "mapped H — test_consensus reads the model frame (rolled maps)" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 60, n_outliers = 0, noise = 0.0, seed = 7
        )
        pm = HomographyProblem(csponds(source_pts, target_pts); src_map = C₁, dst_map = C₂)
        @test VgtkEstimation.test_consensus(pm, _to_model(H_true), trues(n_inliers))
    end

    @testset "mapped H — LO refits in a consistent frame" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        cs = csponds(source_pts, target_pts)
        pm = HomographyProblem(cs; src_map = C₁, dst_map = C₂)
        pu = HomographyProblem(cs)
        # the core arms (the class default, Irls{TLS}, and the static Welsch IRLS);
        # the Student-t + uniform IRLS and the Welsch HQ lift on the mapped
        # problem are LocalOptimizationDoneRight's test
        lo_default = VgtkEstimation._dflt_lo(:homography)
        for lo in (Irls{WelschShape}(scale_multiple = 1.5), lo_default)
            cfg = RansacConfig(max_trials = 3000, min_trials = 200)
            rm = ransac(pm, MsacScoring(pm; threshold = 3.0); local_optimization = lo, config = cfg, rng = StableRNG(129))
            ru = ransac(pu, MsacScoring(pu; threshold = 3.0); local_optimization = lo, config = cfg, rng = StableRNG(129))
            @test converged(rm)
            @test estimate(rm) isa model_type(pm)
            @test sum(rm.inlier_mask[1:n_inliers]) >= 0.9 * n_inliers
            @test sum(rm.inlier_mask[(n_inliers + 1):end]) <= 3
            @test max_transfer_error(_to_px(estimate(rm)), H_true, mapped_test_pts) < 1.0
            # The LO objective is the same pixel objective mapped or not, so
            # the two runs land on the same homography.
            @test max_transfer_error(_to_px(estimate(rm)), estimate(ru), mapped_test_pts) < 0.5
        end
        # `locally_optimize` from a perturbed model-frame seed.
        Hs = _to_model(HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(H_true) .* (1 .+ 1.0e-3 .* SMatrix{3, 3, Float64, 9}(1, -2, 1, 2, -1, 1, -1, 1, 0)))))
        for lo in (Irls{WelschShape}(scale_multiple = 1.5), lo_default)
            r = VgtkEstimation.locally_optimize(pm, Hs; scoring = MsacScoring(pm; threshold = 3.0), local_optimization = lo)
            @test max_transfer_error(_to_px(estimate(r)), H_true, mapped_test_pts) < 1.0
        end
    end

    @testset "mapped H from Float32 pixels solves in the model eltype" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 60, n_outliers = 10, noise = 0.3, seed = 11
        )
        cs32 = csponds(SVector{2, Float32}.(source_pts), SVector{2, Float32}.(target_pts))
        pm = HomographyProblem(cs32; src_map = C₁, dst_map = C₂)
        @test model_type(pm) == HomographyMat{Float32}
        sols = VgtkEstimation.solve(pm, [1, 15, 30, 45])
        @test sols[1] isa model_type(pm)
        r = ransac(
            pm, MsacScoring(pm; threshold = 3.0);
            config = RansacConfig(max_trials = 2000, min_trials = 100),
            rng = StableRNG(130)
        )
        @test converged(r)
        @test max_transfer_error(_to_px(estimate(r)), H_true, mapped_test_pts) < 2.0
    end

    @testset "metre-unit board: the sample gate is scale-free" begin
        # A 5 cm board in METRES, seen at 0.3 m by an f = 800 camera: grid
        # spacing ~7 mm, so an absolute 0.1 coincidence tolerance on the raw
        # board coordinates would refuse every minimal sample.
        rng = StableRNG(5)
        # the board plane z = 0 seen by a pinhole camera: H = K·[r₁ r₂ t]
        H_true = HomographyMat(PosedCamera(CameraModel(; focal = 800.0, pp = (320.0, 240.0)), EuclideanMap(RotMatrix(RotX(0.25)), SA[-0.025, -0.025, 0.3])))
        warp = homography_warp(H_true)
        board = [SA[0.05 * i / 7, 0.05 * j / 7] for i in 0:7 for j in 0:7]
        pix = [warp(b) + SA[0.2 * randn(rng), 0.2 * randn(rng)] for b in board]
        for k in 1:8   # a few gross mismatches
            pix[k] = pix[k] + SA[40.0 * randn(rng), 40.0 * randn(rng)]
        end
        p = HomographyProblem(csponds(board, pix))
        @test VgtkEstimation.test_sample(p, [9, 16, 57, 64])
        # The board is EXACT (VGC conventions/measurement-noise.md): its residual
        # is then the transfer error in pixels, and the default 3 px MSAC
        # threshold means 3 px. As a PLAIN (unit isotropic) metre board the
        # residual is in metres — ~7e-5 for a 0.2 px inlier, ~0.015 for a 40 px
        # mismatch, every one under 3 — and the fit lands ~40 px off.
        result = find_homography(csponds(exact.(board), pix))
        @test converged(result)
        @test sum(result.inlier_mask[9:end]) >= 50
        @test !any(result.inlier_mask[1:8])
        @test max_transfer_error(estimate(result), H_true, [SA[0.01, 0.01], SA[0.04, 0.02], SA[0.025, 0.045]]) < 1.0
        plain = find_homography(csponds(board, pix))
        @test all(plain.inlier_mask[1:8])          # the unit error the rule removes
        # A genuinely coincident pair is still refused.
        @test !VgtkEstimation.test_sample(
            HomographyProblem(csponds([board[1], board[1], board[8], board[57], board[64]], [pix[1], pix[1], pix[8], pix[57], pix[64]])),
            [1, 2, 3, 5]
        )
    end

    @testset "a pipeline workspace serves any threshold scorer" begin
        # `RansacWorkspace(P, n)` builds MSAC's tables (at threshold 1.0); a
        # threshold scorer's workspace is width-free, so the same workspace
        # serves the class default, another threshold and RANSAC counting.
        source_pts, target_pts, _, _ = make_homography_data(n_inliers = 40, n_outliers = 10, seed = 3)
        cs = csponds(source_pts, target_pts)
        ws = VgtkEstimation.RansacWorkspace(HomographyProblem{Float64}, length(cs))
        p = HomographyProblem(cs)
        @test converged(find_homography(cs; workspace = ws, rng = StableRNG(132)))
        @test converged(find_homography(cs; workspace = ws, threshold = 2.0, rng = StableRNG(133)))
        @test converged(find_homography(cs; workspace = ws, scoring = RansacScoring(p; threshold = 3.0), rng = StableRNG(134)))
    end

end
end

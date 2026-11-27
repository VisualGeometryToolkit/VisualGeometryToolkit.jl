# R22: contract test — the homography refit step (`refine!`) and the external-refit contract
@testitem "ransac_homography" setup = [HomographyScenes] begin
    import VisualGeometryToolkitEstimation as VgtkEstimation; include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "thin_homography.jl"))   # `thin_homography_correspondences`
    using VisualGeometryToolkitCore: inlier_mask, noise_scale
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using LinearAlgebra
using StaticArrays
using Random
using StableRNGs
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: Correspondences, correspondences, exact
using VisualGeometryToolkitCore: homography_4pt

using VisualGeometryToolkitCore: MsacScoring, RansacScoring
using VisualGeometryToolkitEstimation: UniformSampler, inlier_ratio, sampler
using VisualGeometryToolkitCore: refine!
using VisualGeometryToolkitEstimation
using Rotations: RotMatrix, RotX, RotZ
using VisualGeometryToolkitEstimation: minimal_sample_size, data_size, model_type, ProsacSampler, reset!

# =============================================================================
# Test utilities
# =============================================================================

function make_homography_data(; n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42)
    source_pts, target_pts = homography_pairs(StableRNG(seed), H_TRUE; n_in = n_inliers, n_out = n_outliers, σ = noise)
    return source_pts, target_pts, H_TRUE, n_inliers
end

# An external strategy (the `ExternalRefit` family, as PoseLib's LM/LZ arms):
# its bracket records each step in its workspace, so a wrapper that does not
# forward a hook is visible.
struct _ExternalArm <: VgtkCore.AbstractOptimizationStrategy end
VgtkCore.RefitFamily(::_ExternalArm) = VgtkCore.ExternalRefit()
VgtkCore.external_workspace(::_ExternalArm, _problem, _frame, _scoring) = Ref(0)
VgtkCore.prepare_external!(ws, ::_ExternalArm, _problem, _θ, _width, _scoring) = (ws[] += 1; true)
VgtkCore.external_refit!(ws, ::_ExternalArm, _problem, θ, ::VgtkCore.LossWidth, _scoring) =
    (ws[] += 10; VgtkCore.FitResult(θ, VgtkCore.ReturnCode.Success, VgtkCore.ClosedForm, VgtkCore.SolveStats(), NaN, 0, nothing))

@testset "RANSAC Homography" begin

    @testset "test_sample — object→image (y-up ↔ y-down) accepts uniform flip" begin
        # Board coordinates: world y-up (right-handed 2D in its native frame).
        board = [
            SA[-1.0, -1.0], SA[1.0, -1.0], SA[1.0, 1.0], SA[-1.0, 1.0],
            SA[0.0, 0.5], SA[0.5, 0.0],
        ]
        # Image coordinates: simulated y-down projection of the board
        # (uniform sign flip on y, plus a translation into pixel space).
        img = [SA[300.0 + 200.0 * b[1], 400.0 - 200.0 * b[2]] for b in board]

        p = HomographyProblem(correspondences(board, img))
        # Every 4-subset should succeed at test_sample — none are bow-ties, all are
        # uniform-flip. Under the old rule every sample would be rejected.
        for idx in ([1, 2, 3, 4], [1, 3, 5, 6], [2, 3, 4, 5], [3, 4, 5, 6], [1, 2, 4, 6])
            @test VgtkEstimation.test_sample(p, idx)
        end

        # And a full RANSAC converges on the clean board→image data.
        result = solve(EstimationProblem(p), Ransac(; scoring = MsacScoring(p; threshold = 2.0), local_optimization_strategy = nothing, config = RansacConfig(max_trials = 500, min_trials = 50)); rng = StableRNG(126))
        @test successful_retcode(result)
        @test sum(inlier_mask(result)) == length(board)
    end

    @testset "HomographyProblem construction" begin
        src = [SA[1.0, 2.0], SA[3.0, 4.0], SA[5.0, 6.0], SA[7.0, 8.0]]
        dst = [SA[2.0, 3.0], SA[4.0, 5.0], SA[6.0, 7.0], SA[8.0, 9.0]]
        p = HomographyProblem(correspondences(src, dst))
        @test minimal_sample_size(p) == 4
        @test data_size(p) == 4
        @test model_type(p) == HomographyMat{Float64}
        @test p isa HomographyProblem{Float64}
        # without maps the problem is ONE concrete type, decided by the types (no run-time identity test of the maps)
        @test (@inferred HomographyProblem(correspondences(src, dst))) isa HomographyProblem{Float64}

        # Too few correspondences
        # fewer than a minimal sample is a problem too: its solve is the no-model result, not a refusal at construction
        @test VgtkEstimation.data_size(HomographyProblem(correspondences(src[1:3], dst[1:3]))) == 3
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
        p_id = HomographyProblem(correspondences(src, copy(src)))
        models = VgtkEstimation.sample_models(p_id, [1, 2, 3, 4])
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
        problem = HomographyProblem(correspondences(source_pts, target_pts))

        result = solve(EstimationProblem(problem), Ransac(; scoring = MsacScoring(problem; threshold = 3.0), local_optimization_strategy = nothing, config = RansacConfig(max_trials = 3000, min_trials = 200)); rng = StableRNG(127))

        @test successful_retcode(result)
        test_pts = [
            SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
            SA[200.0, 200.0], SA[700.0, 500.0],
        ]
        @test max_transfer_error(result.u, H_true, test_pts) < 2.0
        @test inlier_ratio(result) > 0.5
        @test sum(inlier_mask(result)) >= n_inliers * 0.7
    end

    @testset "fit — weighted DLT roundtrip" begin
        H_true = H_TRUE

        n = 20
        src, dst = homography_pairs(StableRNG(42), H_true; n_in = n)

        p = HomographyProblem(correspondences(src, dst))
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
        problem = HomographyProblem(correspondences(source_pts, target_pts))

        result = solve(EstimationProblem(problem), Ransac(; scoring = MsacScoring(problem; threshold = 3.0), local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5), config = RansacConfig(max_trials = 3000, min_trials = 200)); rng = StableRNG(128))

        @test successful_retcode(result)
        test_pts = [
            SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
            SA[200.0, 200.0], SA[700.0, 500.0],
        ]
        @test max_transfer_error(result.u, H_true, test_pts) < 2.0
        @test sum(inlier_mask(result)) >= n_inliers * 0.7
    end

    @testset "CountedLO — instrumentation wrapper answers as the wrapped strategy" begin
        source_pts, target_pts, _, _ = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        problem = HomographyProblem(correspondences(source_pts, target_pts))
        inner = Irls{WelschShape}(efficiency_constant = 1.5)
        counted = VgtkEstimation.CountedLO(inner)
        run(lo) = solve(EstimationProblem(problem), Ransac(; scoring = MsacScoring(problem; threshold = 3.0), local_optimization_strategy = lo, config = RansacConfig(max_trials = 3000, min_trials = 200)); rng = StableRNG(128))
        r_inner = run(inner)
        r_counted = run(counted)
        @test r_counted.u == r_inner.u
        @test counted.calls[] >= 1
        @test counted.iterations[] >= counted.calls[]
        # each call's record, beside the counters: its iterations, ReturnCode and StopRule
        recs = counted.call_records
        @test recs isa Vector{<:VgtkCore.FitResult{Nothing, Nothing, Nothing}}
        @test length(recs) == counted.calls[]
        @test sum(r -> r.stats.nsteps, recs) == counted.iterations[]
        # a seeded call's record is the termination record its result reports
        seeded = VgtkEstimation.CountedLO(inner)
        sc = MsacScoring(problem; threshold = 3.0)
        rs = solve(RefineProblem(problem, r_inner.u), LocalOptimization(seeded; scoring = sc))
        @test length(seeded.call_records) == 1
        rec = only(seeded.call_records)
        @test (rec.stats.nsteps, rec.retcode, rec.stop_rule) == (seeded.iterations[], rs.metadata.lo_retcode, rs.metadata.lo_stop_rule)
        @test VgtkCore.lo_width(counted) == VgtkCore.lo_width(inner)
        @test VgtkCore.lo_label(counted) == VgtkCore.lo_label(inner)
        @test VgtkCore.scale_interface(counted) == VgtkCore.scale_interface(inner)
        @test VgtkCore.optimization_path(counted) == VgtkCore.optimization_path(inner)
        @test VgtkCore.loss_family(counted) == VgtkCore.loss_family(inner)
        # rewidthing keeps the counters (the same Refs)
        @test VgtkCore.with_width(counted, VgtkCore.LossWidth(2.0)).calls === counted.calls
        @test VgtkCore.with_width(counted, VgtkCore.LossWidth(2.0)).iterations === counted.iterations
        @test VgtkCore.with_width(counted, VgtkCore.LossWidth(2.0)).call_records === counted.call_records
        # the loss identity and the fixed-width control are the inner's
        for lo in (inner, Irls{GemanMcclureShape}(efficiency_constant = 2.0))
            c = VgtkEstimation.CountedLO(lo)
            @test VgtkCore.strategy_loss(c) === VgtkCore.strategy_loss(lo)
            @test VgtkCore.fixed_width(c, 4.0) == VgtkCore.fixed_width(lo, 4.0)
            @test VgtkCore.with_width(c, VgtkCore.fixed_width(c, 4.0)).inner ==
                VgtkCore.with_width(lo, VgtkCore.fixed_width(lo, 4.0))
            t = VgtkEstimation.TerminalLO(lo)
            @test VgtkCore.strategy_loss(t) === VgtkCore.strategy_loss(lo)
            @test VgtkCore.fixed_width(VgtkEstimation.CountedLO(t), 4.0) == VgtkCore.fixed_width(lo, 4.0)
        end
    end

    @testset "CountedLO — one counter shared by threaded runs adds up exactly" begin
        problems = map(1:16) do seed
            src, tgt, _, _ = make_homography_data(n_inliers = 100, n_outliers = 30, noise = 0.5, seed = seed)
            HomographyProblem(correspondences(src, tgt))
        end
        inner = Irls{WelschShape}(efficiency_constant = 1.5)
        run(lo, j) = solve(EstimationProblem(problems[j]), Ransac(; scoring = MsacScoring(problems[j]; threshold = 3.0), local_optimization_strategy = lo, config = RansacConfig(max_trials = 500, min_trials = 200)); rng = StableRNG(j))
        serial = VgtkEstimation.CountedLO(inner)
        foreach(j -> run(serial, j), eachindex(problems))
        threaded = VgtkEstimation.CountedLO(inner)
        Threads.@threads for j in eachindex(problems)
            run(threaded, j)
        end
        @test serial.calls[] > length(problems)
        @test threaded.calls[] == serial.calls[]
        @test threaded.iterations[] == serial.iterations[]
        @test length(threaded.call_records) == threaded.calls[]
        by = r -> (r.stats.nsteps, r.retcode, r.stop_rule, r.rms, r.n)
        @test sort(threaded.call_records; by) == sort(serial.call_records; by)     # the same calls, in the threads' order
        # Contention the sweep above rarely reaches: four tasks hammer one counter
        # with near-instant calls (a plain `Ref` loses increments here).
        sc = MsacScoring(problems[1]; threshold = 3.0)
        θ = run(inner, 1).u
        hammered = VgtkEstimation.CountedLO(nothing)
        n_calls = 20_000
        tasks = map(1:4) do _
            Threads.@spawn begin
                ws = VgtkEstimation.RansacWorkspace(problems[1], sc, hammered)
                for _ in 1:n_calls
                    solve(RefineProblem(problems[1], θ), LocalOptimization(hammered; scoring = sc))
                end
            end
        end
        foreach(wait, tasks)
        @test hammered.calls[] == 4 * n_calls
        @test length(hammered.call_records) == 4 * n_calls                  # no record lost under contention
    end

    @testset "TerminalLO — the raw race, then one seeded LO of its incumbent" begin
        source_pts, target_pts, _, _ = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        problem = HomographyProblem(correspondences(source_pts, target_pts))
        sc = MsacScoring(problem; threshold = 3.0)
        cfg = RansacConfig(max_trials = 3000, min_trials = 200)
        inner = Irls{WelschShape}(efficiency_constant = 1.5)
        events(trace) = (trial, score, ws) -> push!(trace, (trial, score, ws.best_model, ws.best_lo_retcode, ws.best_lo_stop_rule))
        run(lo, trace = []; kw...) = solve(EstimationProblem(problem),
            Ransac(; scoring = sc, local_optimization_strategy = lo, config = cfg, on_improvement = events(trace), kw...);
            rng = StableRNG(128))

        # The recipe it standardizes: the race with no LO, then the seeded refine.
        trace0 = []
        r0 = run(nothing, trace0)
        seeded = solve(RefineProblem(problem, r0.u), LocalOptimization(inner; scoring = sc))
        trace = []
        rt = run(VgtkEstimation.TerminalLO(inner), trace)
        @test successful_retcode(rt)
        @test rt.u == seeded.u                 # bit-identical refined model
        @test rt.u != r0.u                     # the LO did move it
        @test rt.stats.nsteps == r0.stats.nsteps   # trials of the raw race
        @test rt.metadata.sample_acceptance_rate == r0.metadata.sample_acceptance_rate
        @test !isempty(trace0) && trace == trace0              # the raw incumbent's events
        @test rt.metadata.lo_retcode == seeded.metadata.lo_retcode && rt.metadata.lo_stop_rule == seeded.metadata.lo_stop_rule
        @test inlier_mask(rt) == inlier_mask(seeded)             # re-scored for the refined model
        @test rt.metadata.quality == seeded.metadata.quality
        @test rt.resid == seeded.resid
        @test noise_scale(rt) === noise_scale(seeded)

        # A reused workspace (the cache of `init`, solved twice) gives the same.
        c = init(EstimationProblem(problem), Ransac(; scoring = sc, local_optimization_strategy = VgtkEstimation.TerminalLO(inner), config = cfg);
                 rng = StableRNG(128))
        solve!(c)
        c.cache.rng = StableRNG(128)
        @test solve!(c).u == seeded.u

        # Placement is a static trait; a counter counts the terminal call in either nesting.
        @test VgtkEstimation.lo_placement(inner) === VgtkEstimation.InLoopPlacement()
        @test VgtkEstimation.lo_placement(VgtkEstimation.TerminalLO(inner)) === VgtkEstimation.TerminalPlacement()
        outer = VgtkEstimation.CountedLO(VgtkEstimation.TerminalLO(inner))
        @test VgtkEstimation.lo_placement(outer) === VgtkEstimation.TerminalPlacement()
        @test run(outer).u == seeded.u
        @test outer.calls[] == 1
        counted = VgtkEstimation.CountedLO(inner)
        @test run(VgtkEstimation.TerminalLO(counted)).u == seeded.u
        @test counted.calls[] == 1

        # The per-candidate guard has no candidate to guard.
        @test_throws ArgumentError run(VgtkEstimation.TerminalLO(inner); score_guard = true)

        # Every hook answers as the inner strategy; the label names the placement.
        t = VgtkEstimation.TerminalLO(inner)
        @test VgtkCore.lo_label(t) == VgtkCore.lo_label(inner) * " (terminal)"
        @test VgtkCore.lo_width(t) == VgtkCore.lo_width(inner)
        @test VgtkCore.scale_interface(t) == VgtkCore.scale_interface(inner)
        @test VgtkCore.optimization_path(t) == VgtkCore.optimization_path(inner)
        @test VgtkCore.loss_family(t) == VgtkCore.loss_family(inner)
        @test VgtkCore.lo_marker(t) == VgtkCore.lo_marker(inner)
        @test VgtkCore.with_width(t, VgtkCore.LossWidth(2.0)) ==
            VgtkEstimation.TerminalLO(VgtkCore.with_width(inner, VgtkCore.LossWidth(2.0)))
        @test VgtkCore.fixed_width(t, 4.0) == VgtkCore.fixed_width(inner, 4.0)
        # A seeded refine has no race: it runs the inner strategy.
        @test solve(RefineProblem(problem, r0.u), LocalOptimization(t; scoring = sc)).u ==
            seeded.u
    end

    @testset "the wrappers forward the refit family and the external-refit bracket" begin
        arm = _ExternalArm()
        θ = H_TRUE
        for w in (VgtkEstimation.CountedLO(arm), VgtkEstimation.TerminalLO(arm), VgtkEstimation.CountedLO(VgtkEstimation.TerminalLO(arm)))
            @test VgtkCore.RefitFamily(w) === VgtkCore.RefitFamily(arm) === VgtkCore.ExternalRefit()
            ws = VgtkCore.external_workspace(w, nothing, nothing, nothing)
            @test ws isa Base.RefValue{Int}                                      # the inner's workspace
            @test VgtkCore.prepare_external!(ws, w, nothing, θ, VgtkCore.LossWidth(2.0), nothing) && ws[] == 1
            @test VgtkCore.external_refit!(ws, w, nothing, θ, VgtkCore.LossWidth(2.0), nothing).u === θ && ws[] == 11
            @test VgtkCore.external_refit!(ws, w, nothing, θ, VgtkCore.FixedS2(4.0), nothing).u === θ && ws[] == 21
            # an external refit is not counted: the wrapper that ran the bracket counts nothing
            w isa VgtkEstimation.CountedLO && @test w.calls[] == 0 && w.iterations[] == 0 && isempty(w.call_records)
        end
        native = Irls{WelschShape}(efficiency_constant = 1.5)
        @test VgtkCore.RefitFamily(VgtkEstimation.CountedLO(native)) === VgtkCore.RefitFamily(native) === VgtkCore.NativeRefit()
        @test VgtkEstimation.lo_placement(VgtkEstimation.CountedLO(VgtkEstimation.TerminalLO(arm))) === VgtkEstimation.TerminalPlacement()
    end

    @testset "TerminalLO — a thin incumbent comes back unrefined (its refit's record InitialFailure)" begin
        # five exact correspondences of a homography among scattered outliers:
        # the incumbent's consensus is m + 1 = 5 points, no residual dof
        problem = HomographyProblem(thin_homography_correspondences())   # the suite's fixture file, shared with the golden records
        sc = MsacScoring(problem; threshold = 0.5)
        cfg = RansacConfig(max_trials = 5000, min_trials = 3000)
        r0 = solve(EstimationProblem(problem), Ransac(; scoring = sc, local_optimization_strategy = nothing, config = cfg); rng = StableRNG(1))
        rt = solve(EstimationProblem(problem), Ransac(; scoring = sc, local_optimization_strategy = VgtkEstimation.TerminalLO(Irls{WelschShape}(efficiency_constant = 1.5)), config = cfg); rng = StableRNG(1))
        @test sum(inlier_mask(rt)) == 5
        @test (rt.metadata.lo_retcode, rt.metadata.lo_stop_rule) == (VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule)   # the refit did not run
        @test rt.u == r0.u
        @test rt.retcode == r0.retcode && rt.stop_rule == r0.stop_rule   # the search's record
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
        problem = HomographyProblem(Correspondences(source_pts, target_pts; score = scores))
        @test sampler(problem) isa ProsacSampler

        result = solve(EstimationProblem(problem), Ransac(; scoring = MsacScoring(problem; threshold = 3.0), local_optimization_strategy = nothing, config = RansacConfig(max_trials = 3000, min_trials = 200)); rng = StableRNG(131))

        @test successful_retcode(result)
        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_transfer_error(result.u, H_true, test_pts) < 2.0
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
    # MAPPED homography: pixels in, `K⁻¹`-style observation maps, a model-frame
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
        pm = HomographyProblem(correspondences(source_pts, target_pts); src_map = C₁, dst_map = C₂)
        @test VgtkEstimation.test_consensus(pm, _to_model(H_true), trues(n_inliers))
    end

    @testset "mapped H — LO refits in a consistent frame" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        cs = correspondences(source_pts, target_pts)
        pm = HomographyProblem(cs; src_map = C₁, dst_map = C₂)
        pu = HomographyProblem(cs)
        # the core arms (the class default, Irls{TLS}, and the static Welsch IRLS);
        # the Student-t + uniform IRLS and the Welsch HQ lift on the mapped
        # problem are LocalOptimizationDoneRight's test
        lo_default = VgtkEstimation._dflt_lo(:homography)
        for lo in (Irls{WelschShape}(efficiency_constant = 1.5), lo_default)
            cfg = RansacConfig(max_trials = 3000, min_trials = 200)
            rm = solve(EstimationProblem(pm), Ransac(; scoring = MsacScoring(pm; threshold = 3.0), local_optimization_strategy = lo, config = cfg); rng = StableRNG(129))
            ru = solve(EstimationProblem(pu), Ransac(; scoring = MsacScoring(pu; threshold = 3.0), local_optimization_strategy = lo, config = cfg); rng = StableRNG(129))
            @test successful_retcode(rm)
            @test rm.u isa model_type(pm)
            @test sum(inlier_mask(rm)[1:n_inliers]) >= 0.9 * n_inliers
            @test sum(inlier_mask(rm)[(n_inliers + 1):end]) <= 3
            @test max_transfer_error(_to_px(rm.u), H_true, mapped_test_pts) < 1.0
            # The LO objective is the same pixel objective mapped or not, so
            # the two runs land on the same homography.
            @test max_transfer_error(_to_px(rm.u), ru.u, mapped_test_pts) < 0.5
        end
        # `LocalOptimization` from a perturbed model-frame seed.
        Hs = _to_model(HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(H_true) .* (1 .+ 1.0e-3 .* SMatrix{3, 3, Float64, 9}(1, -2, 1, 2, -1, 1, -1, 1, 0)))))
        for lo in (Irls{WelschShape}(efficiency_constant = 1.5), lo_default)
            r = solve(RefineProblem(pm, Hs), LocalOptimization(lo; scoring = MsacScoring(pm; threshold = 3.0)))
            @test max_transfer_error(_to_px(r.u), H_true, mapped_test_pts) < 1.0
        end
    end

    @testset "mapped H from Float32 pixels solves in the model eltype" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers = 60, n_outliers = 10, noise = 0.3, seed = 11
        )
        cs32 = correspondences(SVector{2, Float32}.(source_pts), SVector{2, Float32}.(target_pts))
        pm = HomographyProblem(cs32; src_map = C₁, dst_map = C₂)
        @test model_type(pm) == HomographyMat{Float32}
        sols = VgtkEstimation.sample_models(pm, [1, 15, 30, 45])
        @test sols[1] isa model_type(pm)
        r = solve(EstimationProblem(pm), Ransac(; scoring = MsacScoring(pm; threshold = 3.0), local_optimization_strategy = nothing, config = RansacConfig(max_trials = 2000, min_trials = 100)); rng = StableRNG(130))
        @test successful_retcode(r)
        @test max_transfer_error(_to_px(r.u), H_true, mapped_test_pts) < 2.0
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
        p = HomographyProblem(correspondences(board, pix))
        @test VgtkEstimation.test_sample(p, [9, 16, 57, 64])
        # The board is EXACT (VGC conventions/observation-noise.md): its residual
        # is then the transfer error in pixels, and the default 3 px MSAC
        # threshold means 3 px. As a PLAIN (unit isotropic) metre board the
        # residual is in metres — ~7e-5 for a 0.2 px inlier, ~0.015 for a 40 px
        # mismatch, every one under 3 — and the fit lands ~40 px off.
        result = solve(EstimationProblem(HomographyProblem(correspondences(exact.(board), pix))), Ransac())
        @test successful_retcode(result)
        @test sum(inlier_mask(result)[9:end]) >= 50
        @test !any(inlier_mask(result)[1:8])
        @test max_transfer_error(result.u, H_true, [SA[0.01, 0.01], SA[0.04, 0.02], SA[0.025, 0.045]]) < 1.0
        plain = solve(EstimationProblem(HomographyProblem(correspondences(board, pix))), Ransac())
        @test all(inlier_mask(plain)[1:8])          # the unit error the rule removes
        # A genuinely coincident pair is still refused.
        @test !VgtkEstimation.test_sample(
            HomographyProblem(correspondences([board[1], board[1], board[8], board[57], board[64]], [pix[1], pix[1], pix[8], pix[57], pix[64]])),
            [1, 2, 3, 5]
        )
    end

    @testset "a pipeline workspace serves any threshold scorer" begin
        # `RansacWorkspace(P, n)` builds MSAC's tables (at threshold 1.0); a
        # threshold scorer's workspace is width-free, so the same workspace
        # serves the class default, another threshold and RANSAC counting.
        source_pts, target_pts, _, _ = make_homography_data(n_inliers = 40, n_outliers = 10, seed = 3)
        cs = correspondences(source_pts, target_pts)
        ws = VgtkEstimation.RansacWorkspace(HomographyProblem{Float64}, length(cs))
        p = HomographyProblem(cs)
        @test successful_retcode(solve(EstimationProblem(HomographyProblem(cs)), Ransac(); rng = StableRNG(132)))
        @test successful_retcode(solve(EstimationProblem(HomographyProblem(cs)), Ransac(; scoring = MsacScoring(HomographyProblem; threshold = 2.0)); rng = StableRNG(133)))
        @test successful_retcode(solve(EstimationProblem(HomographyProblem(cs)), Ransac(; scoring = RansacScoring(p; threshold = 3.0)); rng = StableRNG(134)))
    end

end
end

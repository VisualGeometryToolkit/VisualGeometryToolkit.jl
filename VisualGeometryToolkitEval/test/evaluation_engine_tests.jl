# =============================================================================
# evaluation_engine_tests.jl — the composition evaluator (src/evaluation/
# engine/*: the pair geometry, the one refit layer, the post-LO matrices, the
# sample-budget engine, live RANSAC on pool pairs, the selection stream, the
# arms and the roster's gate-enforced readers), ported from
# LocalOptimizationDoneRight's test_engine.jl (K+, 2026-10-09) onto the core
# contracts alone: this package may not load the scorer or optimizer packages,
# even in tests, so every component here is VisualGeometryToolkitCore's,
# VisualGeometryToolkitEstimation's or test-local.
#
# Test-local components (a paper's would take their place):
#
#   HalfwidthMsac{DG}(a)   a scoring TEMPLATE with an outlier half-width, the
#                          refit-template contract (`outlier_halfwidth`,
#                          `with_width(·, OutlierHalfwidth(a))`) on core
#                          parts: at half-width `a` it IS MSAC at threshold `a`
#                          (`with_width` returns that `MsacScoring`, the refit
#                          scoring), and as a selection scorer it scores as that
#                          MSAC. A stand-in for a paper's Student-t marginal,
#                          whose outlier-half-width axis VGC and RVG do not ship.
#   WidthRefit(max_iter)   an ExternalRefit strategy on the external-refit
#                          contract: its workspace keeps the two-view frame it
#                          was built from; its refine is VisualGeometryToolkitEstimation's
#                          `Irls{TLSShape}` at the absolute loss width it is
#                          handed (through this package's `refit_model`); a
#                          fixed scale `FixedS2(s2)` reaches it as the loss
#                          width √s2 (the contract's default).
#   RefusingRefit, ThrowingRefit   an external refiner whose `prepare_external!`
#                          declines, and a native one whose refine throws.
#
# Fixtures: VisualGeometryToolkitEval's synthetic E pool pair (a light `PoolPair` with
# its residual matrix): `epair` is 30 noisy correspondences, 12 single-model
# minimal samples, σ̂² = 4e-6; `synthetic_e_pool_pair()` 25 samples, 27 models.
# =============================================================================

@testitem "evaluation_engine" begin
    import VisualGeometryToolkitEval as VgtkEval
    using Test, Random, LinearAlgebra, StaticArrays
    using Statistics: median
    using VisualGeometryToolkitEval
    using VisualGeometryToolkitEval.Evaluation
    using VisualGeometryToolkitEval.Evaluation: PairGeom, ScoreSelector, OracleSelector, update!, checkpoint, lo_error!,
        lo_iteration_budget, n_batches, n_los, batch_sample_counts, lo_sample_counts, deployed_ref
    using VisualGeometryToolkitEval.Pools: PoolResiduals, ScoringPool, synthetic_e_pool_pair, sample_starts, GT_CONVENTION
    using VisualGeometryToolkitEval.Tuning: tuning_spec, GridSpec, SpacedGrid, LinearCount, grid, save_tuning_artifact,
        tuning_table, ScaleMultipleWindow, PairUnit
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation, Irls, NoLocalOptimization, EssentialProblem, FundMatProblem,
        HomographyProblem, RelposeBundleProblem, fixed_width
    import VisualGeometryToolkitCore as VGC
    using VisualGeometryToolkitCore: AbstractScoring, AbstractLocalOptimization, ExternalRefit, NativeRefit, RefitFamily,
        MsacScoring, TLSShape, ScaleMultiple, LossWidth, OutlierHalfwidth, EssentialMat, FundMat,
        width_axis_name, with_width, outlier_halfwidth, outlier_halfwidth_to_loss_width, loss_width,
        select_on_grid!, stage_selection, selection_staging, best_pool_model, scoring_workspace
    using VisualGeometryToolkitCore.Pools: PoolPair, num_models, num_points, pool_imc

    # ------------------------------------------------------------------------
    # Test-local components
    # ------------------------------------------------------------------------
    struct HalfwidthMsac{DG} <: AbstractScoring
        outlier_halfwidth::Float64
    end
    HalfwidthMsac(dg::Integer, outlier_halfwidth::Real) = HalfwidthMsac{Int(dg)}(Float64(outlier_halfwidth))
    msac_of(s::HalfwidthMsac{DG}) where {DG} = MsacScoring(DG; threshold = s.outlier_halfwidth)
    VGC.outlier_halfwidth(s::HalfwidthMsac) = s.outlier_halfwidth
    VGC.width_axis(::HalfwidthMsac) = OutlierHalfwidth()
    VGC.with_width(::HalfwidthMsac{DG}, a::OutlierHalfwidth{Float64}) where {DG} = MsacScoring(DG; threshold = a.value)
    VGC.codimension(::HalfwidthMsac{DG}) where {DG} = DG
    VGC.scoring_workspace(s::HalfwidthMsac, n::Int, d_g::Int) = scoring_workspace(msac_of(s), n, d_g)
    VGC.score!(ws, s::HalfwidthMsac, scores, r, n; kw...) = VGC.score!(ws, msac_of(s), scores, r, n; kw...)

    # A threshold scorer on the loss-width axis that is no grid family's: at
    # width w it is MSAC at threshold w.
    struct ThresholdMsac{DG} <: AbstractScoring
        w::Float64
    end
    ThresholdMsac(dg::Integer, w::Real) = ThresholdMsac{Int(dg)}(Float64(w))
    VGC.width_axis(::ThresholdMsac) = LossWidth()
    VGC.with_width(::ThresholdMsac{DG}, w::LossWidth{NTuple{2, Float64}}) where {DG} = MsacScoring(DG; threshold = loss_width(w))

    struct WidthRefit <: AbstractLocalOptimization
        max_iter::Int
    end
    mutable struct WidthRefitWs
        frame::Any
        prepared::Int
        refined::Int
    end
    VGC.RefitFamily(::WidthRefit) = ExternalRefit()
    VGC.external_workspace(::WidthRefit, _problem, frame, _scoring) = WidthRefitWs(frame, 0, 0)
    VGC.prepare_external!(ws::WidthRefitWs, ::WidthRefit, _problem, _θ, _width, _scoring) =
        (ws.prepared += 1; true)
    function VGC.refine_external!(ws::WidthRefitWs, lo::WidthRefit, problem, θ, w::LossWidth, _scoring)
        ws.refined += 1
        lw = loss_width(w)
        return refit_model(problem, θ, Irls{TLSShape}(lo.max_iter, LossWidth(lw)),
                           MsacScoring(VGC.codimension(problem); threshold = lw))
    end

    struct RefusingRefit <: AbstractLocalOptimization end
    VGC.RefitFamily(::RefusingRefit) = ExternalRefit()
    VGC.prepare_external!(_ws, ::RefusingRefit, _problem, _θ, _width, _scoring) = false
    VGC.refine_external!(_ws, ::RefusingRefit, _problem, _θ, _width, _scoring) = error("never called")

    struct ThrowingRefit <: AbstractLocalOptimization end   # a native refiner with no locally_optimize!

    # ------------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------------
    const EPAIR_GT_VARIANCE = 4.0e-6
    epair(rng) = synthetic_e_pool_pair(rng; sibling_counts = fill(1, 12), noise = 0.002, δ_rot = 0.2,
                                       t_jitter = 0.1, gt_variance = EPAIR_GT_VARIANCE)
    const P = EssentialProblem
    template(a = 1.0) = HalfwidthMsac(1, a)
    irls() = Irls{TLSShape}()                               # follows the refit scoring's threshold
    irls_sm(m = 2.5) = Irls{TLSShape}(; scale_multiple = m)  # consensus-anchored

    # A one-scene ScoringPool of `pairs` stamped with `total` minimal samples:
    # what `sample_starts` certifies its boundaries against.
    starts_pool(pairs, total) = ScoringPool(pairs, :E, "synthetic"; scene_names = fill("synthetic", length(pairs)),
        stamps = (gt_convention = GT_CONVENTION, pool_rng = "LehmerRNG", julia_version = string(VERSION),
                  total_samples = total, total_solutions = sum(num_models, pairs), mean_solutions_per_sample = 1.0,
                  provenance = Dict{String,Any}()))
    # The pair restricted to the stored models `cols`.
    subpair(pair, cols) = PoolPair{FundMat{Float64}}(; (f => getfield(pair, f) for f in fieldnames(typeof(pair)))...,
        models = pair.models[cols], sample_indices = pair.sample_indices[:, cols],
        errors_r = pair.errors_r[cols], errors_t = pair.errors_t[cols])

    @testset "the test-local template is a refit template" begin
        t = template(0.02)
        @test outlier_halfwidth(t) == 0.02 && VGC.width_axis(t) === OutlierHalfwidth()
        @test with_width(t, OutlierHalfwidth(0.05)) == MsacScoring(1; threshold = 0.05)
        @test RefitFamily(WidthRefit(5)) isa ExternalRefit && RefitFamily(irls()) isa NativeRefit
    end

    @testset "problem_type and the protocol constants" begin
        @test problem_type(:E) === EssentialProblem && problem_type(:F) === FundMatProblem &&
              problem_type(:H) === HomographyProblem
        @test_throws ArgumentError problem_type(:X)
        @test OPERATING_SCALE_MULTIPLE === 2.5
        @test lo_iteration_budget() isa Int && lo_iteration_budget() > 0
    end

    @testset "build_pair_ctx: the per-pair geometry" begin
        pair, _ = epair(MersenneTwister(7))
        ctx = build_pair_ctx(P, pair; scoring = template())
        @test ctx isa PairCtx && ctx.pair === pair && ctx.problem_type === P
        @test ctx.scoring === template() && ctx.solo_scoring === ctx.scoring && ctx.bp === nothing
        @test ctx.ep isa EssentialProblem && isempty(ctx.lo_ws)
        @test ctx.geom isa PairGeom && length(ctx.geom.cs) == num_points(pair)
        @test build_pair_ctx(P, pair).scoring === nothing          # a selection-only context
        @test build_pair_ctx(P, pair; scoring = template(), solo_scoring = template(2.0)).solo_scoring === template(2.0)
        @test build_pair_ctx(P, pair; bundle = true).bp isa RelposeBundleProblem
        # one imc, dispatched on the model: a pool model, its metric seed, the
        # stored lower bound under it
        for j in 1:num_models(pair)
            seed = refit_seed(ctx, pair.models[j])
            @test seed isa EssentialMat
            @test model_imc(pair.models[j], ctx) ≈ model_imc(seed, ctx) atol = 1e-8
            @test pool_imc(pair, j) <= model_imc(pair.models[j], ctx.geom) + 1e-9
        end
        f1, f2 = essential_bearings(ctx)
        @test length(f1) == length(f2) == num_points(pair) && all(x -> norm(x) ≈ 1, f1)
        @test_throws ArgumentError essential_bearings(build_pair_ctx(FundMatProblem, pair))
        # the basin perturbation is reproducible from its rng
        R0, t0 = VGC.Pools.gt_pose(pair)
        @test perturbed_seed_E(MersenneTwister(1), R0, t0, 0.1) == perturbed_seed_E(MersenneTwister(1), R0, t0, 0.1)
    end

    @testset "refit_imc!: the native path" begin
        pair, _ = epair(MersenneTwister(7))
        ctx = build_pair_ctx(P, pair; scoring = template())
        seed = refit_seed(ctx, pair.models[3])
        a = 0.01
        e, ns = refit_imc!(ctx, irls(), seed, a)
        E = refit_model(ctx.ep, seed, irls(), MsacScoring(1; threshold = a))
        @test E !== nothing && E != seed
        @test e === model_imc(E, ctx) && ns > 0               # the template rebuilt at nat′ = a
        @test haskey(ctx.lo_ws, :ep_refit_Irls)                # one cached workspace per (problem, template, LO)
        @test refit_imc!(ctx, irls(), seed, a)[1] === e        # the cached workspace changes nothing
        @test refit_imc!(ctx, irls(), seed, 10a)[1] !== e      # the native moves the refit
        # the solo template: its own workspace key, the same arithmetic here
        @test refit_imc!(ctx, irls(), seed, a; scoring = ctx.solo_scoring)[1] === e
        @test refit_imc!(ctx, irls(), seed, a; scoring = template(7.0))[1] === e
        @test any(k -> startswith(String(k), "ep_sc"), keys(ctx.lo_ws))
        # s2: the strategy deployed at an absolute scale (its scale multiple at √s2)
        s2 = (0.004)^2
        e_s2 = first(refit_imc!(ctx, irls_sm(), seed, a; s2))
        @test e_s2 === model_imc(refit_model(ctx.ep, seed, irls_sm(), MsacScoring(1; threshold = a); s2), ctx)
        @test e_s2 === model_imc(refit_model(ctx.ep, seed, with_width(irls_sm(), fixed_width(irls_sm(), s2)),
                                             MsacScoring(1; threshold = a)), ctx)
        @test first(refit_imc!(ctx, irls_sm(), seed, a; s2 = 400 * s2)) !== e_s2
        # the guard keeps the refit or the seed
        @test first(refit_imc!(ctx, ScoreGuarded(irls()), seed, a)) in (e, model_imc(seed, ctx))
        @test VGC.optimization_path(ScoreGuarded(irls())) === VGC.optimization_path(irls())
        # fallbacks: a throwing refine → the seed's imc, or the caller's on_fail
        e_fail, _ = @test_logs (:warn,) match_mode = :any refit_imc!(ctx, ThrowingRefit(), seed, a)
        @test e_fail === model_imc(seed, ctx)
        @test isnan(first(@test_logs (:warn,) match_mode = :any refit_imc!(ctx, ThrowingRefit(), seed, a; on_fail = NaN)))
        # refusals: no template, a bundle refit without the bundle problem
        @test_throws ArgumentError refit_imc!(build_pair_ctx(P, pair), irls(), seed, a)
        @test_throws ErrorException refit_imc!(ctx, irls(), seed, a; bundle = true)
        # the refit row's guarded bracket
        @test guarded_refit(() -> nothing, identity, 7.0)[1] === 7.0
        @test guarded_refit(() -> 3, x -> NaN, 7.0)[1] === 7.0
        @test guarded_refit(() -> 3, x -> 2x, 7.0)[1] === 6.0
    end

    @testset "refit_imc!: the external path" begin
        pair, _ = epair(MersenneTwister(7))
        ctx = build_pair_ctx(P, pair; scoring = template())
        seed = refit_seed(ctx, pair.models[3])
        lo = WidthRefit(25)
        for w in (0.002, 0.005, 0.02)
            e_native, ns = refit_imc!(ctx, lo, seed, w; on_fail = NaN)
            e_override, _ = refit_imc!(ctx, lo, seed, 0.123; on_fail = NaN, s2 = w^2)
            @test isfinite(e_native) && ns > 0 && e_native === e_override   # the two ports: one width, bitwise
            E = refit_model(ctx.ep, seed, Irls{TLSShape}(25, LossWidth(w)), MsacScoring(1; threshold = w))
            @test e_native === model_imc(E, ctx)
        end
        ws = ctx.lo_ws[:ext_refit_WidthRefit]
        @test ws.prepared == ws.refined == 6                    # one workspace per (pair, template, refiner)
        # the frame the external workspace is built from: the pair's two-view geometry
        @test ws.frame.Kc1 == ctx.geom.Kc1 && ws.frame.Kc2 == ctx.geom.Kc2 && ws.frame.cs == ctx.geom.cs
        # a declined preparation is no refit and no call
        @test refit_imc!(ctx, RefusingRefit(), seed, 0.01) === (model_imc(seed, ctx), Int64(0))
        @test isnan(first(refit_imc!(ctx, RefusingRefit(), seed, 0.01; on_fail = NaN)))
        # the axes the two families consume, and the crossover between them
        @test consumed_axis(lo) === LossWidth() && consumed_axis(irls()) === OutlierHalfwidth()
        @test consumed_axis(ScoreGuarded(irls())) === OutlierHalfwidth()
        @test consumed_native(lo, LossWidth(), 0.2; d_g = 1) === 0.2
        @test consumed_native(irls(), OutlierHalfwidth(), 0.2; d_g = 1) === 0.2
        @test consumed_native(lo, OutlierHalfwidth(), 0.2; d_g = 1) ===
              outlier_halfwidth_to_loss_width(0.2; operating_scale_multiple = OPERATING_SCALE_MULTIPLE, d_g = 1)
        @test consumed_native(lo, OutlierHalfwidth(), 0.2; d_g = 2, operating_scale_multiple = 3.0) ===
              outlier_halfwidth_to_loss_width(0.2; operating_scale_multiple = 3.0, d_g = 2)
        @test consumed_native(irls(), LossWidth(), consumed_native(lo, OutlierHalfwidth(), 0.2; d_g = 1); d_g = 1) ≈ 0.2
        @test_throws MethodError consumed_native(lo, ScaleMultiple(), 0.2; d_g = 1)
        @test_throws UndefKeywordError consumed_native(lo, LossWidth(), 0.2)
    end

    @testset "LoRefitSpec: construction and refusals" begin
        idxs = ones(Int32, 2, 3)
        s = LoRefitSpec("x", irls(), OutlierHalfwidth(), 1:3, idxs)
        @test s.grid == [1.0, 2.0, 3.0] && s.grid isa Vector{Float64} && s.idxs == idxs && s.axis === OutlierHalfwidth()
        @test s.seedcols === s.natives === s.widths === s.fixed_s2s === s.columns === nothing && !s.bundle
        @test_throws "native axis" LoRefitSpec("x", NoLocalOptimization(), ScaleMultiple(), [1.0, 2.0, 3.0], idxs)
        @test_throws "mutually exclusive" LoRefitSpec("x", irls(), LossWidth(), [1.0, 2.0, 3.0], idxs;
                                                      natives = [1.0, 1.0, 1.0], widths = [1.0, 1.0, 1.0])
        @test_throws "mutually exclusive" LoRefitSpec("x", irls(), LossWidth(), [1.0, 2.0, 3.0], idxs;
                                                      widths = [1.0, 1.0, 1.0], fixed_s2s = [1.0, 1.0, 1.0])
        @test_throws "outside the grid" LoRefitSpec("x", irls(), LossWidth(), [1.0, 2.0, 3.0], idxs; columns = [0, 2])
        @test_throws "outside the grid" LoRefitSpec("x", irls(), LossWidth(), [1.0, 2.0, 3.0], idxs; columns = [4])
        s2 = LoRefitSpec("x", irls(), LossWidth(), [1.0, 2.0, 3.0], idxs; natives = [1, 2, 3], fixed_s2s = [1, 1, 1],
                         seedcols = [3, 2, 1], columns = [2])
        @test s2.natives == [1.0, 2.0, 3.0] && s2.fixed_s2s == [1.0, 1.0, 1.0] && s2.seedcols == [3, 2, 1] &&
              s2.columns == [2]
    end

    @testset "apply_lo_refits" begin
        rng = MersenneTwister(20261003)
        pairs = [first(epair(rng)) for _ in 1:3]
        hw = [0.005, 0.01, 0.02, 0.04, 0.08]
        idxs = Int32[j % 4 for j in 1:3, _ in 1:5]              # some 0 ⇒ no consensus
        t = template()
        ext = WidthRefit(25)
        specs = [LoRefitSpec("native", irls(), OutlierHalfwidth(), hw, idxs),
                 LoRefitSpec("external", ext, OutlierHalfwidth(), hw, idxs),
                 LoRefitSpec("sel", NoLocalOptimization(), OutlierHalfwidth(), hw, idxs),
                 LoRefitSpec("natives", irls(), OutlierHalfwidth(), hw, idxs; natives = 2 .* hw, seedcols = [5, 4, 3, 2, 1]),
                 LoRefitSpec("widths", irls_sm(1.0), OutlierHalfwidth(), hw, idxs; widths = hw ./ 5),
                 LoRefitSpec("widths_ext", ext, OutlierHalfwidth(), hw, idxs; widths = hw ./ 5),
                 LoRefitSpec("fixed", irls_sm(), OutlierHalfwidth(), hw, idxs; fixed_s2s = (hw ./ 10) .^ 2),
                 LoRefitSpec("columns", irls(), OutlierHalfwidth(), hw, idxs; columns = [2, 4])]
        M = apply_lo_refits(P, specs, pairs; scoring = t)
        @test Set(keys(M)) == Set(s.label for s in specs) && all(m -> size(m) == (3, 5), values(M))
        # the task space (pair × spec) cannot move a cell
        for sp in specs
            @test isequal(M[sp.label], only(values(apply_lo_refits(P, [sp], pairs; scoring = t))))
        end
        @test isempty(apply_lo_refits(P, LoRefitSpec[], pairs))
        @test_throws ErrorException apply_lo_refits(P, specs[1:1], pairs[1:2]; scoring = t)
        # every cell is the refit layer's call on that pair's selected model
        for p in 1:3
            ctx = build_pair_ctx(P, pairs[p]; scoring = t)
            seed(j) = refit_seed(ctx, pairs[p].models[j])
            for c in 1:5
                j = Int(idxs[p, c])
                if j == 0
                    @test all(isnan, (M[l][p, c] for l in ("native", "external", "sel", "widths", "fixed")))
                    continue
                end
                @test M["sel"][p, c] === model_imc(pairs[p].models[j], ctx.geom)
                @test M["native"][p, c] === first(refit_imc!(ctx, irls(), seed(j), hw[c]))
                @test M["external"][p, c] ===
                      first(refit_imc!(ctx, ext, seed(j), consumed_native(ext, OutlierHalfwidth(), hw[c]; d_g = 1)))
                # an absolute width: the solo delivery, bitwise, for either family
                for (lab, lo) in (("widths", irls_sm(1.0)), ("widths_ext", ext))
                    solo = SoloSpec("s", :refinement, LossWidth(), _ -> lo)
                    @test M[lab][p, c] === first(solo_refit_imc!(ctx, solo, seed(j), hw[c] / 5, hw[c]; on_fail = nothing))
                end
                @test M["widths"][p, c] ===
                      first(refit_imc!(ctx, irls_sm(1.0), seed(j), hw[c]; scoring = ctx.solo_scoring, s2 = (hw[c] / 5)^2))
                @test M["fixed"][p, c] === first(refit_imc!(ctx, irls_sm(), seed(j), hw[c]; s2 = (hw[c] / 10)^2))
                @test (c in (2, 4)) ? M["columns"][p, c] === M["native"][p, c] : isnan(M["columns"][p, c])
            end
            for c in 1:5
                j = Int(idxs[p, 6 - c])
                j == 0 ? (@test isnan(M["natives"][p, c])) :
                         (@test M["natives"][p, c] === first(refit_imc!(ctx, irls(), seed(j), 2 * hw[c])))
            end
        end
        # the ladder convention: a failed refit is NaN, not the seed (a seed
        # whose consensus is too thin for an LO step comes back as itself)
        fails = only(values(apply_lo_refits(P, [LoRefitSpec("t", ThrowingRefit(), OutlierHalfwidth(), hw, idxs)],
                                            pairs; scoring = t, on_fail = NaN)))
        @test any(isnan, fails[idxs .!= 0])
        for p in 1:3, c in 1:5
            idxs[p, c] == 0 && continue
            ctx = build_pair_ctx(P, pairs[p]; scoring = t)
            @test isequal(fails[p, c], first(refit_imc!(ctx, ThrowingRefit(), refit_seed(ctx, pairs[p].models[idxs[p, c]]),
                                                        hw[c]; on_fail = NaN)))
        end
    end

    @testset "sample-budget engine" begin
        pair, R = synthetic_e_pool_pair()      # 25 minimal samples: two carry 2 siblings, 23 carry 1
        cfg = SampleBudgetConfig(; sample_cap = 20, batch_size = 5, lo_every = 10, budget_unit = :models,
                                 lo_strategy = irls())
        @testset "config" begin
            @test n_batches(cfg) == 4 && n_los(cfg) == 3
            @test batch_sample_counts(cfg) == [5, 10, 15, 20] && lo_sample_counts(cfg) == [5, 10, 20]
            sbc(; kw...) = SampleBudgetConfig(; sample_cap = 20, batch_size = 5, lo_every = 10, lo_strategy = irls(), kw...)
            @test_throws ArgumentError sbc(budget_unit = :modles)
            @test_throws ArgumentError sbc(budget_unit = :models, nat_axis = ScaleMultiple())
        end
        methods = [SampleBudgetMethodSpec(:narrow, template(0.01)), SampleBudgetMethodSpec(:wide, template(0.03))]
        msac = [MsacSpec(:msac, 0.01, WidthRefit(25); refinement_loss_width = 0.02)]
        @test MsacSpec(:m, 1, WidthRefit(2)).refinement_loss_width === nothing
        @testset "per pair: shapes, selections, refits, the MSAC and oracle columns" begin
            res = evaluate_sample_budget(P, methods, [pair], [R]; pair_ids = ["synth-1"], msac_specs = msac, cfg)
            @test res.methods == [:narrow, :wide, :msac, :oracle] && res.pair_ids == ["synth-1"]
            @test res.sample_counts == [5, 10, 15, 20] && res.lo_counts == [5, 10, 20]
            @test size(res.err_batch) == (1, 4, 4) && size(res.err_lo) == (1, 4, 3)
            @test all(0 .<= res.best_idx_batch .<= num_models(pair)) && !any(isnan, res.err_batch)
            ctx = build_pair_ctx(P, pair; scoring = template(0.01))
            for (b, cap) in enumerate(res.sample_counts)
                q, Rq = subpair(pair, 1:cap), R[:, 1:cap]
                for (m, a) in ((1, 0.01), (2, 0.03), (3, 0.01))     # the running best is the prefix's argmax
                    j = best_pool_model(MsacScoring(1; threshold = a), q, Rq)
                    @test res.best_idx_batch[1, m, b] == j
                    @test res.err_batch[1, m, b] === Float32(model_imc(pair.models[j], ctx))
                end
                @test res.err_batch[1, 4, b] === Float32(minimum(j -> model_imc(pair.models[j], ctx), 1:cap))
            end
            for (l, cap) in enumerate(res.lo_counts)
                b = cap ÷ 5
                seed(m) = refit_seed(ctx, pair.models[res.best_idx_batch[1, m, b]])
                # the config's LO at the selector's width (no nat_ref); the oracle at the first method's
                @test res.err_lo[1, 1, l] === Float32(first(refit_imc!(ctx, irls(), seed(1), 0.01)))
                @test res.err_lo[1, 2, l] === Float32(first(refit_imc!(ctx, irls(), seed(2), 0.03)))
                @test res.err_lo[1, 4, l] === Float32(first(refit_imc!(ctx, irls(), seed(4), 0.01)))
                # the MSAC column: its own refiner at its refinement width
                @test res.err_lo[1, 3, l] === Float32(first(refit_imc!(ctx, WidthRefit(25), seed(3), 0.02)))
            end
            @test res.lo_calls_total[3] > 0 && res.lo_startup_ns[3] > 0
            @test [t.method for t in summarize_lo_timing(res)] == res.methods
            # nat_ref: every config-LO column refits at it
            cfg_ref = SampleBudgetConfig(; sample_cap = 20, batch_size = 5, lo_every = 10, budget_unit = :models,
                                         lo_strategy = irls(), nat_ref = 0.05)
            r2 = evaluate_sample_budget(P, methods, [pair], [R]; msac_specs = msac, cfg = cfg_ref)
            @test r2.best_idx_batch == res.best_idx_batch && r2.err_lo[:, 3, :] == res.err_lo[:, 3, :]
            seed1 = refit_seed(ctx, pair.models[r2.best_idx_batch[1, 1, end]])
            @test r2.err_lo[1, 1, end] === Float32(first(refit_imc!(ctx, irls(), seed1, 0.05)))
            # reuse: a rerun is identical, and a pair among others is scored as it is alone
            again = evaluate_sample_budget(P, methods, [pair], [R]; pair_ids = ["synth-1"], msac_specs = msac, cfg)
            @test again.err_batch == res.err_batch && again.best_idx_batch == res.best_idx_batch &&
                  again.err_lo == res.err_lo
            other, Ro = synthetic_e_pool_pair(MersenneTwister(99))
            both = evaluate_sample_budget(P, methods, [other, pair, other], [Ro, R, Ro]; msac_specs = msac, cfg)
            @test both.best_idx_batch[2:2, :, :] == res.best_idx_batch && both.err_lo[2:2, :, :] == res.err_lo
            @test both.best_idx_batch[1, :, :] == both.best_idx_batch[3, :, :] && both.err_lo[1, :, :] == both.err_lo[3, :, :]
        end
        @testset "refusals" begin
            # one refit template per evaluation
            @test_throws "different scoring templates" evaluate_sample_budget(P,
                [methods[1], SampleBudgetMethodSpec(:h, HalfwidthMsac(2, 0.01))], [pair], [R]; cfg)
            # a threshold scorer has no outlier half-width: it is an MsacSpec
            @test_throws "MsacSpec" evaluate_sample_budget(P,
                [SampleBudgetMethodSpec(:t, MsacScoring(1; threshold = 0.01))], [pair], [R]; cfg)
            @test_throws DimensionMismatch evaluate_sample_budget(P, methods, [pair], typeof(R)[]; cfg)
            @test_throws ErrorException evaluate_sample_budget(P, methods, [pair], [R]; pair_ids = ["a", "b"], cfg)
        end
        @testset "a :samples budget counts through sample_starts" begin
            cfg_s = SampleBudgetConfig(; sample_cap = 20, batch_size = 20, lo_every = 20, budget_unit = :samples,
                                       lo_strategy = irls())
            starts = sample_starts(starts_pool([pair], 25))
            res_s = evaluate_sample_budget(P, methods[1:1], [pair], [R]; msac_specs = msac, cfg = cfg_s, sample_starts = starts)
            # 20 samples reach model 22
            @test res_s.best_idx_batch[1, 2, end] == best_pool_model(MsacScoring(1; threshold = 0.01), subpair(pair, 1:22), R[:, 1:22])
            @test_throws ArgumentError evaluate_sample_budget(P, methods[1:1], [pair], [R]; cfg = cfg_s)
            @test_throws DimensionMismatch evaluate_sample_budget(P, methods[1:1], [pair], [R]; cfg = cfg_s,
                                                                  sample_starts = [starts[1], starts[1]])
        end
        @testset "a repeated draw counts as two samples" begin
            # The oracle-best model closes a stream in which sample 1 (2
            # siblings) was drawn twice in a row: A A B C, 4 samples in 6 columns.
            full = SampleBudgetConfig(; sample_cap = 27, batch_size = 27, lo_every = 27, budget_unit = :models,
                                      lo_strategy = NoLocalOptimization())
            ob = Int(evaluate_sample_budget(P, methods[1:1], [pair], [R]; cfg = full).best_idx_batch[1, end, 1])
            @test ob > 4                                            # a single-model sample (models 5:27)
            b = ob == 5 ? 6 : 5
            cols = [1, 2, 1, 2, b, ob]
            q, Rq = subpair(pair, cols), R[:, cols]
            starts = sample_starts(starts_pool([q], 4))
            @test starts == [[1, 3, 5, 6]]
            run(cap, unit; kw...) = evaluate_sample_budget(P, methods[1:1], [q], [Rq];
                cfg = SampleBudgetConfig(; sample_cap = cap, batch_size = cap, lo_every = cap, budget_unit = unit,
                                         lo_strategy = NoLocalOptimization()), kw...).best_idx_batch[1, :, 1]
            @test run(3, :samples; sample_starts = starts) == run(5, :models)   # 3 samples = columns 1:5
            @test run(3, :samples; sample_starts = starts)[end] != 6
            @test run(5, :models)[end] != run(6, :models)[end] == 6
            @test run(4, :samples; sample_starts = starts) == run(6, :models)
        end
    end

    @testset "the running-best selectors: refit at the refinement native; zero-allocation folds" begin
        pair, R = epair(MersenneTwister(7))
        n, M = num_points(pair), num_models(pair)
        ctx = build_pair_ctx(P, pair; scoring = template())
        sc = MsacScoring(1; threshold = 0.01)
        sel = ScoreSelector(:msac, sc, 0.01, scoring_workspace(sc, n, 1), Vector{Float64}(undef, n))
        cfg(nat_ref) = SampleBudgetConfig(; sample_cap = 12, batch_size = 12, lo_every = 12, budget_unit = :models,
                                          lo_strategy = irls(), nat_ref)
        @test checkpoint(sel, ctx.geom, pair) === (NaN32, Int32(0)) && lo_error!(ctx, sel, cfg(nothing)) === (NaN32, Int64(0))
        for j in 1:M
            update!(sel, ctx.geom, pair, R, j, view(pair.sample_indices, :, j), n)
        end
        @test sel.best_idx == best_pool_model(sc, pair, R)
        seed = refit_seed(ctx, pair.models[sel.best_idx])
        at_w = first(refit_imc!(ctx, irls(), seed, 0.04)); at_a = first(refit_imc!(ctx, irls(), seed, 0.01))
        @test at_w != at_a                                      # the fixture separates the two widths
        @test first(lo_error!(ctx, sel, cfg(0.04))) === Float32(at_w)
        @test first(lo_error!(ctx, sel, cfg(nothing))) === Float32(at_a)   # no refinement native: the selection's
        oracle = OracleSelector(:oracle, 0.01)
        for j in 1:M
            update!(oracle, ctx.geom, pair, R, j, view(pair.sample_indices, :, j), n)
        end
        @test oracle.best_imc == minimum(j -> model_imc(pair.models[j], ctx), 1:M)
        @test first(lo_error!(ctx, oracle, cfg(0.04))) === Float32(first(refit_imc!(ctx, irls(), refit_seed(ctx, pair.models[oracle.best_idx]), 0.04)))
        # the per-model fold and the checkpoint read allocate nothing once warm (G10)
        fold(s, geom, pair, R, n, M) = (a = 0; for j in 1:M; a += @allocated update!(s, geom, pair, R, j, view(pair.sample_indices, :, j), n); end; a)
        reads(s, geom, pair) = @allocated checkpoint(s, geom, pair)
        VgtkEval.Evaluation.reset!(sel)
        fold(sel, ctx.geom, pair, R, n, M); reads(sel, ctx.geom, pair)            # warm
        VgtkEval.Evaluation.reset!(sel)
        bytes_update = fold(sel, ctx.geom, pair, R, n, M)
        bytes_checkpoint = reads(sel, ctx.geom, pair)
        @info "ScoreSelector{MSAC} allocations once warm" bytes_update bytes_checkpoint
        @test bytes_update == 0
        @test bytes_checkpoint == 0
    end

    @testset "budget engine at the full budget ≡ the miscal evaluation at ζ = 1" begin
        # Every arm kind of a composed roster: the budget engine's LO at the last
        # checkpoint (every pool model inspected) reproduces, bitwise, the miscal
        # evaluation's ζ = 1 column (`miscal_evaluation_refit` → `apply_lo_refits`).
        rng = MersenneTwister(20261007)
        prs = [epair(rng) for _ in 1:4]
        pairs = first.(prs); Rs = last.(prs)
        halfwidths = [0.005, 0.01, 0.02, 0.04]; a = halfwidths[2]
        grids = Dict(:outlier_halfwidth => halfwidths, :scale_multiple => [1.0, 2.0, 3.0],
                     :loss_width => [0.001, 0.002, 0.004])
        idxs = Matrix{Int32}(undef, 4, 4)
        select_on_grid!(idxs, P, template(), halfwidths, stage_selection(selection_staging(template()), pairs, Rs))
        idxs_of = Dict(:hw => idxs)
        arms = [ComposedSpec("HW +Irls", :hw, OutlierHalfwidth(), OutlierHalfwidth(), _ -> irls()) => 0.03,
                ComposedSpec("HW +Irls@a*", :hw, OutlierHalfwidth(), OutlierHalfwidth(), _ -> irls();
                             source = SelectionWidth()) => nothing,
                ComposedSpec("HW +Irls(m)", :hw, OutlierHalfwidth(), ScaleMultiple(), m -> irls_sm(m)) => 2.0,
                ComposedSpec("HW +Ext", :hw, OutlierHalfwidth(), LossWidth(), _ -> WidthRefit(25)) => 0.002]
        budget = Dict{String,Vector{Float32}}()
        for (arm, tuned_width) in arms
            width = operating_width(arm, a, tuned_width)
            spec = miscal_evaluation_refit(arm, idxs_of, (selection = a, width = tuned_width), [1.0], grids; d_g = 1)
            miscal = only(values(apply_lo_refits(P, [spec], pairs; scoring = template())))
            ref = deployed_ref(arm.kind, a, width)
            nat = arm.kind isa ScaleMultiple ? (;) : (; nat_ref = ref, nat_axis = arm.kind)
            cfg = SampleBudgetConfig(; sample_cap = 12, batch_size = 4, lo_every = 4, budget_unit = :models,
                                     lo_strategy = arm.ctor(width), nat...)
            r = evaluate_sample_budget(P, [SampleBudgetMethodSpec(:hw, template(a))], pairs, Rs; cfg)
            @test r.best_idx_batch[:, 1, end] == idxs[:, 2]     # one seed: the full pool's a* selection
            budget[arm.base] = r.err_lo[:, 1, end]
            @test all(isfinite, miscal) && budget[arm.base] == Float32.(miscal[:, 1])
        end
        @test budget["HW +Irls"] != budget["HW +Irls@a*"]        # their refit widths differ
    end

    @testset "live runs: the caller's problem, the refit half-width in the loop" begin
        ctxs = [build_pair_ctx(P, first(epair(MersenneTwister(100 + i)))) for i in 1:8]
        a = 0.01
        sc = MsacScoring(1; threshold = a)
        calls = Threads.Atomic{Int}(0)
        problem_of = pair -> (Threads.atomic_add!(calls, 1); EssentialProblem(pair))
        run(lo; warmup = false) = live_runs(sc, lo, ctxs; problem_of, seed_of = j -> 7 + j, warmup)
        bare = run(irls())
        @test calls[] == 8                                       # one problem per pair, the caller's
        @test length(bare.err) == length(bare.trials) == length(bare.wall_s) == 8
        @test all(isfinite, bare.err) && all(>(0), bare.trials) && all(>(0), bare.wall_s)
        again = run(irls(); warmup = true)
        @test again.err == bare.err && again.trials == bare.trials   # seeded per pair: schedule-free
        @test calls[] == 8 + 8 + 1                               # the warm-up solves pair 1 once
        # the refit at the selection's own half-width is the bare arm, exactly
        at_a = run(RefitHalfwidth(irls(), a, template()))
        @test at_a.err == bare.err && at_a.trials == bare.trials
        @test run(RefitHalfwidth(irls(), a, _ -> MsacScoring(1; threshold = a))).err == bare.err
        @test_throws ArgumentError RefitHalfwidth(irls(), 0.0, template())
        @test_throws ArgumentError RefitHalfwidth(irls(), Inf, template())
        row = live_row(bare)
        @test 0 <= row.maa <= 1 && row.n_eligible == 8 && row.med == median(bare.err) && row.med_trials == median(bare.trials)
    end

    @testset "live runs: per-pair hooks leave the defaults' output untouched" begin
        ctxs = [build_pair_ctx(P, first(epair(MersenneTwister(100 + i)))) for i in 1:8]
        sc = MsacScoring(1; threshold = 0.01)
        problem_of = EssentialProblem
        run(; kw...) = live_runs(sc, irls(), ctxs; problem_of, seed_of = j -> 7 + j, warmup = false, kw...)
        base = run()
        @test isapprox(base.err, [3.585203625487833, 7.8158303757585434, 3.7568851311201112, 2.618818515365572, 3.0636424185959137, 0.668822182465071, 1.9706137868124836, 5.3988742939118275]; rtol = 1.0e-6)  && base.trials == [100, 100, 100, 100, 100, 100, 100, 100]   # recorded before the hooks existed; the IRLS errors agree to ~1e-8 across Julia 1.12 and 1.13
        @test propertynames(base) == (:err, :trials, :wall_s)
        # lo_of overriding with the same strategy, a callback and the guard off: same outputs
        fired = [Threads.Atomic{Int}(0) for _ in 1:8]
        cb_of(j) = (trial, score, ws) -> Threads.atomic_add!(fired[j], 1)
        hooked = run(; lo_of = j -> irls(), on_improvement_of = cb_of)
        @test hooked.err == base.err && hooked.trials == base.trials
        @test all(f -> f[] > 0, fired)                                 # the callback fires on every pair
        # the warm-up asks for callback 0 (scratch storage), never pair 1's
        events = [Int[] for _ in 0:8]                      # index j + 1
        rec_of(j) = (trial, score, ws) -> push!(events[j + 1], trial)
        live_runs(sc, irls(), ctxs; problem_of, seed_of = j -> 7 + j, warmup = true, on_improvement_of = rec_of)
        warm = deepcopy(events)
        empty!.(events)
        live_runs(sc, irls(), ctxs; problem_of, seed_of = j -> 7 + j, warmup = false, on_improvement_of = rec_of)
        @test !isempty(warm[1]) && isempty(events[1])      # j = 0 saw the warm-up only
        @test warm[2:end] == events[2:end]                 # the pairs' records are the warm-up-free ones
        # finalize: extras only when asked, one per pair, in the pair's slot
        fin = run(; finalize = (j, r) -> (j, VGC.converged(r)))
        @test fin.err == base.err && fin.extras == [(j, true) for j in 1:8]
        # the guard is forwarded to ransac and still yields a model per pair
        guarded = run(; score_guard = true)
        @test all(isfinite, guarded.err)
        # cancellation: polled before each pair; nothing runs
        calls = Threads.Atomic{Int}(0)
        none = run(; cancelled = () -> true, finalize = (j, r) -> Threads.atomic_add!(calls, 1))
        @test all(==(Inf), none.err) && all(==(0), none.trials) && all(isnan, none.wall_s)
        @test calls[] == 0 && all(isnothing, none.extras)
        # Experiments' kaimon hooks: false / no-op without Main.KaimonGate
        @test Experiments.cancelled() === false
        @test Experiments.progress("x") === nothing
    end

    @testset "stream_selections: the :pool staging, the generic select_on_grid!" begin
        rng = MersenneTwister(20261001)
        prs = [epair(rng) for _ in 1:3]
        pairs = first.(prs); Rs = last.(prs)
        scene(r) = ((pairs = pairs[r],), PoolResiduals("k", :tangent_sampson, Rs[r]))
        loaders = [() -> scene(1:2), () -> scene(3:3)]
        halfwidths = grid(1.0e-3, 0.1, LinearCount(7))
        loss_widths = grid(1.0e-3, 0.1, LinearCount(5))
        # Both grids are test-local scorers on the `:pool` staging, so the
        # generic fallback runs whatever else is loaded (VisualGeometryToolkitEstimation
        # loads RANSACScoringDoneRight, whose grid families stage `:rsdr`; VGC's
        # MSAC at a threshold is none of them).
        hwsc, msac = template(), ThresholdMsac(1, 1.0)
        @test selection_staging(hwsc) === selection_staging(msac) === :pool
        @test selection_staging(MsacScoring(1; threshold = 1.0)) === :pool
        refits_of = (idxs_of, _) -> [
            LoRefitSpec("HW sel", NoLocalOptimization(), OutlierHalfwidth(), halfwidths, idxs_of[:hw]),
            LoRefitSpec("MSAC sel", NoLocalOptimization(), LossWidth(), loss_widths, idxs_of[:msac]),
            LoRefitSpec("HW +Irls", irls(), OutlierHalfwidth(), halfwidths, idxs_of[:hw]; columns = [2, 5]),
        ]
        grids = [:hw => (halfwidths, hwsc), :msac => (loss_widths, msac)]
        M, s2, np = stream_selections(refits_of, Float64, P, loaders, grids, template(), nothing, 1)
        @test np == [2, 1] && s2 == [[EPAIR_GT_VARIANCE, EPAIR_GT_VARIANCE], [EPAIR_GT_VARIANCE]]
        @test Set(keys(M)) == Set(["HW sel", "MSAC sel", "HW +Irls"])
        M32, _, _ = stream_selections(refits_of, Float32, P, loaders, grids, template(), nothing, 3)
        for (si, r) in enumerate((1:2, 3:3))
            # the generic fallback: per column with_width, then the per-pair argmax
            want_idx = Dict(:hw => Int32[best_pool_model(with_width(hwsc, OutlierHalfwidth(w)), pairs[p], Rs[p])
                                         for p in r, w in halfwidths],
                            :msac => Int32[best_pool_model(with_width(msac, LossWidth(w)), pairs[p], Rs[p])
                                           for p in r, w in loss_widths])
            want = apply_lo_refits(P, refits_of(want_idx, si), pairs[r]; scoring = template())
            for l in keys(M)
                @test isequal(M[l][si], want[l])                 # batching is pair-local
                @test isequal(M32[l][si], Float32.(want[l]))     # in the stream's element type
            end
            @test all(isfinite, M["HW sel"][si])
        end
        # the generic method is the :pool staging's, column by column
        idxs = Matrix{Int32}(undef, 3, length(loss_widths))
        st = stage_selection(:pool, pairs, Rs)
        @test st == (; pairs, residuals = Rs)
        @test select_on_grid!(idxs, P, msac, loss_widths, st) === idxs
        @test idxs == Int32[best_pool_model(MsacScoring(1; threshold = w), pairs[p], Rs[p]) for p in 1:3, w in loss_widths]
    end

    @testset "arms: width sources, deliveries, the evaluation refit" begin
        bcd = _ -> irls()
        tuned = ComposedSpec("HW +Irls", :hw, OutlierHalfwidth(), OutlierHalfwidth(), bcd)
        matched = ComposedSpec("HW +Irls@a*", :hw, OutlierHalfwidth(), OutlierHalfwidth(), bcd; source = SelectionWidth())
        @test tuned.source === TunedWidth() && is_tuned(tuned) && !is_tuned(matched)
        @test_throws ArgumentError ComposedSpec("x", :hw, OutlierHalfwidth(), ScaleMultiple(), bcd; source = SelectionWidth())
        @test_throws ArgumentError ComposedSpec("x", :msac, LossWidth(), OutlierHalfwidth(), bcd; source = SelectionWidth())
        @test_throws ArgumentError ComposedSpec("x", :hw, ScaleMultiple(), ScaleMultiple(), bcd)
        a = 0.017; w = 0.023
        @test operating_width(tuned, a, w) === w && operating_width(matched, a, nothing) === a
        @test operating_width(matched, a, a) === a
        @test_throws ArgumentError operating_width(matched, a, w)
        @test_throws ArgumentError operating_width(tuned, a, nothing)
        # deliveries
        @test width_delivery(LossWidth(), irls(), 2.0, 2) == (; widths = [2.0, 2.0])
        @test width_delivery(OutlierHalfwidth(), irls(), 2.0, 2) == (; natives = [2.0, 2.0])
        @test width_delivery(ScaleMultiple(), irls(), 2.0, 2) == (;)
        @test_throws ArgumentError SoloSpec("x", :refinement, ScaleMultiple(), identity)
        @test_throws ArgumentError SoloSpec("x", :tuning, LossWidth(), identity)
        @test_throws ErrorException solo_refit_imc!(nothing, SoloSpec("x", :selection, LossWidth(), identity), nothing, 1.0, 1.0)
        # the evaluation refit: the matched arm is the tuned arm at width = a*, bitwise
        grids = Dict(:outlier_halfwidth => sort!([a * ζ for ζ in (0.25, 1.0, 4.0)]),
                     :scale_multiple => [1.0, 2.0, 3.0], :loss_width => [0.001, 0.002, 0.004])
        idxs_of = Dict(:hw => ones(Int32, 2, 3))
        zetas = [0.25, 1.0, 4.0]
        rm = miscal_evaluation_refit(matched, idxs_of, (selection = a, width = nothing), zetas, grids; d_g = 1)
        rt = miscal_evaluation_refit(tuned, idxs_of, (selection = a, width = a), zetas, grids; d_g = 1)
        @test rm.natives == rt.natives == [a * ζ for ζ in zetas] && rm.seedcols == rt.seedcols == [1, 2, 3]
        @test rm.label == "HW +Irls@a*|evaluation" && rm.axis === OutlierHalfwidth() && rm.grid == zetas
        @test_throws ArgumentError miscal_evaluation_refit(matched, idxs_of, (selection = a, width = w), zetas, grids; d_g = 1)
        # a scale-multiple arm refits at the selection native mapped onto the refiner's axis
        for (arm, sel_w) in ((ComposedSpec("HW +m", :hw, OutlierHalfwidth(), ScaleMultiple(), m -> irls_sm(m)), 2.0),
                             (ComposedSpec("HW +ext", :hw, OutlierHalfwidth(), LossWidth(), _ -> WidthRefit(25)), 0.002))
            ro = miscal_evaluation_refit(arm, idxs_of, (selection = a, width = sel_w), [1.0], grids; d_g = 1)
            @test only(ro.natives) == (arm.kind isa ScaleMultiple ? consumed_native(irls_sm(2.0), OutlierHalfwidth(), a; d_g = 1) : 0.002)
            @test ro.fixed_s2s === nothing
        end
        # a loss-width arm of a native refiner is an absolute width: basin-only
        lw_native = ComposedSpec("HW +Irls(w)", :hw, OutlierHalfwidth(), LossWidth(), _ -> irls())
        @test_throws "basin-only" miscal_evaluation_refit(lw_native, idxs_of, (selection = a, width = 0.002), zetas, grids; d_g = 1)
    end

    @testset "miscal evaluation: the fixed external scale port" begin
        pair, _ = epair(MersenneTwister(7))
        arm = ComposedSpec("HW +Irls(m)", :hw, OutlierHalfwidth(), ScaleMultiple(), m -> irls_sm(m))
        grids = Dict(:outlier_halfwidth => [0.005, 0.01, 0.02], :scale_multiple => [1.0, 2.0, 3.0],
                     :loss_width => [0.001, 0.002, 0.004])
        idxs_of = Dict(:hw => fill(Int32(3), 1, 3))
        tuned = (selection = grids[:outlier_halfwidth][2], width = grids[:scale_multiple][2])
        zetas = [0.25, 1.0, 4.0]
        s0 = 2.0e-3
        with = miscal_evaluation_refit(arm, idxs_of, tuned, zetas, grids; d_g = 1)
        without = miscal_evaluation_refit(arm, idxs_of, tuned, zetas, grids; d_g = 1, plugin_scale = s0)
        @test with.fixed_s2s === nothing && without.fixed_s2s == [(ζ * s0)^2 for ζ in zetas]
        @test without.natives == with.natives && without.seedcols == with.seedcols
        M_with = only(values(apply_lo_refits(P, [with], [pair]; scoring = template())))
        M_without = only(values(apply_lo_refits(P, [without], [pair]; scoring = template())))
        ctx = build_pair_ctx(P, pair; scoring = template())
        seed = refit_seed(ctx, pair.models[3])
        for j in eachindex(zetas)
            @test M_without[1, j] === first(refit_imc!(ctx, with.lo, seed, with.natives[j]; s2 = (zetas[j] * s0)^2))
            @test M_with[1, j] === first(refit_imc!(ctx, with.lo, seed, with.natives[j]))
        end
        lw = ComposedSpec("HW +Ext", :hw, OutlierHalfwidth(), LossWidth(), _ -> WidthRefit(25))
        @test miscal_evaluation_refit(lw, idxs_of, tuned, zetas, grids; d_g = 1).fixed_s2s === nothing
        @test_throws ArgumentError miscal_evaluation_refit(lw, idxs_of, tuned, zetas, grids; d_g = 1, plugin_scale = s0)
    end

    @testset "Roster: the gate-enforced readers" begin
        composed = p -> [
            ComposedSpec("HW +Irls",   :hw, OutlierHalfwidth(), OutlierHalfwidth(), _ -> irls()),
            ComposedSpec("HW +Irls(m)", :hw, OutlierHalfwidth(), ScaleMultiple(), m -> irls_sm(m)),
            ComposedSpec("MSAC +Ext",  :msac, LossWidth(), LossWidth(), _ -> WidthRefit(25)),
            ComposedSpec("HW +Irls@a*", :hw, OutlierHalfwidth(), OutlierHalfwidth(), _ -> irls(); source = SelectionWidth()),
        ]
        basin = p -> [
            ComposedSpec("HW +Irls(w)", :hw, OutlierHalfwidth(), LossWidth(), _ -> irls_sm(1.0)),
            ComposedSpec("HW +Ext",     :hw, OutlierHalfwidth(), LossWidth(), _ -> WidthRefit(25)),
        ]
        roster = Roster(composed, (p; role = :refinement) -> SoloSpec[], basin)
        spec = tuning_spec(Dict{String,Any}("tuning" => Dict{String,Any}("t" => Dict{String,Any}(
            "type" => "windowed_bootstrap", "scale_multiple_window" => Dict("lo" => 1.0, "hi" => 3.3),
            "cross_validation" => Dict{String,Any}("scene_split" => Dict{String,Any}(
                "type" => "registered", "name" => "phototourism"))))), "t")
        gs = Dict{Symbol,GridSpec}(:outlier_halfwidth => SpacedGrid(0.01, 0.1, LinearCount(5)),
                                   :loss_width => SpacedGrid(0.001, 0.01, LinearCount(5)),
                                   :scale_multiple => SpacedGrid(1.0, 3.3, LinearCount(5)))
        g = Dict(k => grid(v) for (k, v) in gs)
        window = ScaleMultipleWindow(1.0, 3.3)
        function write_sidecar(path, arms; mode, hits = Dict(a.base => String[] for a in arms))
            save_tuning_artifact(path, spec; grids = gs, unit = PairUnit(), code = (VisualGeometryToolkitEval,), mode,
                problem = "E", d_g = 1, s0 = 0.001, arms = [a.base for a in arms],
                tuned_selection = Dict(a.base => g[width_axis_name(a.selection)][3] for a in arms),
                tuned_width = Dict(a.base => g[width_axis_name(a.kind)][3] for a in arms),
                tuned_objective = Dict(a.base => 0.0 for a in arms), boundary_hits = hits,
                windows = Dict(a.base => tuning_table(window) for a in arms))
        end
        mktempdir() do dir
            cpath = joinpath(dir, "composed.jld2"); write_sidecar(cpath, composed(:E); mode = "miscal")
            bud = Dict(deployed_budget_settings(roster, cpath))
            @test bud["HW +Irls(m)"].ref == bud["HW +Irls(m)"].sel        # scale multiple: refit at the selection native
            @test bud["MSAC +Ext"].ref == g[:loss_width][3]                # a native width at its tuned value
            @test bud["HW +Irls"].ref == g[:outlier_halfwidth][3]
            @test bud["HW +Irls(m)"].kind === ScaleMultiple() && bud["MSAC +Ext"].kind === LossWidth()
            @test bud["HW +Irls(m)"].lo == irls_sm(g[:scale_multiple][3])
            @test bud["HW +Irls@a*"].ref == bud["HW +Irls@a*"].sel == g[:outlier_halfwidth][3]   # pinned at a*
            @test Dict(deployed_arm_settings(roster, cpath))["MSAC +Ext"][2] == g[:loss_width][3]
            @test @test_logs (:info,) match_mode = :any audit_operating_points(roster, cpath)
            @test_throws ErrorException deployed_basin_settings(roster, cpath)

            bpath = joinpath(dir, "basin.jld2"); write_sidecar(bpath, basin(:E); mode = "basin")
            dep = Dict(deployed_basin_settings(roster, bpath))
            @test dep["HW +Ext"].spec.axis === LossWidth() && dep["HW +Ext"].width == g[:loss_width][3]
            @test dep["HW +Irls(w)"].spec.role === :refinement
            @test audit_operating_points(roster, bpath)
            @test_throws ErrorException deployed_budget_settings(roster, bpath)

            epath = joinpath(dir, "clipped.jld2")
            write_sidecar(epath, composed(:E); mode = "miscal",
                          hits = Dict(a.base => (a.base == "MSAC +Ext" ? ["loss_width_hi"] : String[]) for a in composed(:E)))
            err = try deployed_budget_settings(roster, epath) catch e; e end
            @test occursin("MSAC +Ext", err.msg) && occursin("expand the window on the flagged side", err.msg)
            @test_throws ErrorException audit_operating_points(roster, epath)

            # a selection-width arm whose record carries another width is refused
            mpath = joinpath(dir, "mispinned.jld2")
            save_tuning_artifact(mpath, spec; grids = gs, unit = PairUnit(), code = (VisualGeometryToolkitEval,), mode = "miscal",
                problem = "E", d_g = 1, arms = ["HW +Irls@a*"],
                tuned_selection = Dict("HW +Irls@a*" => g[:outlier_halfwidth][3]),
                tuned_width = Dict("HW +Irls@a*" => g[:outlier_halfwidth][4]),
                tuned_objective = Dict("HW +Irls@a*" => 0.0), boundary_hits = Dict("HW +Irls@a*" => String[]))
            @test_throws ArgumentError deployed_budget_settings(roster, mpath)

            # a scale-multiple arm on the basin roster is refused
            bad = Roster(composed, roster.solo, p -> composed(p))
            @test_throws ErrorException deployed_basin_settings(bad, bpath)
            # a loss-width arm of a native refiner is basin-only: refused by the composed readers
            lwn = Roster(p -> basin(p), roster.solo, basin)
            write_sidecar(joinpath(dir, "lwn.jld2"), basin(:E); mode = "miscal")
            @test_throws "basin-only" deployed_budget_settings(lwn, joinpath(dir, "lwn.jld2"))
        end
    end
end

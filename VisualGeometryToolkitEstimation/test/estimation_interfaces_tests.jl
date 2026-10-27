@testitem "estimation_interfaces" setup = [HomographyScenes, Contamination] begin
    import VisualGeometryToolkitCore as VgtkCore
# =============================================================================
# estimation_interfaces_tests.jl — the VGC shared-contract surface as tests.
#
# Gates the estimation interfaces that RVG implements against
# VisualGeometryToolkitCore (robust/estimation.jl + its loss family):
#   1. ConsensusSet epoch-freshness — `check_fresh` refuses a stale set with
#      an ArgumentError (a real throw, not an elidable @assert); the fresh
#      handoff works; `inlier_count` reads the scorer's k*.
#   2. Weight-policy verb availability per adapter (update_nuisance!/
#      update_weights! methods exist for every concrete policy).
#   3. NormalizedWeightPolicy membership — the ρ(∞)=∞ boundary as a type:
#      mixture/posterior adapters are likelihood-grade, loss-derived ones not.
#   4. Loss trait table — liftable × bounded for the five unified losses;
#      Cauchy is the liftable-but-unbounded witness (the two traits are NOT
#      one axis).
#   5. RefitFamily routing — the VGC default + the constructor-extension
#      pattern eval relies on.
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays
using StableRNGs
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: csponds,
    AbstractWeightPolicy, NormalizedWeightPolicy,
    RefitFamily, NativeRefit, ExternalRefit,
    ConsensusSet, is_fresh, inlier_count, consensus_set, score_model!, MsacScoring,
    codimension, minimal_sample_size, lo_width, with_width, LossWidth, ScaleMultiple, sq_loss_width,
    liftable, bounded,
    GaussianShape, WelschShape, GemanMcclureShape, CauchyShape,
    SmoothTLSShape
using VisualGeometryToolkitCore: scale_s2, check_fresh   # VGC accessors (public)
using VisualGeometryToolkitEstimation: PosteriorLikelihood   # public, not exported
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: RansacWorkspace   # workspace type is unexported (eval imports it explicitly too)
using VisualGeometryToolkitCore: s_hat_sq   # the posterior-weight toolkit's accessor (VGC)
const VgtkEstimation = VisualGeometryToolkitEstimation

# Local strategy for the constructor-extension routing test (the pattern the
# eval refit layer uses: a foreign concrete type declares its family by
# extending VGC's abstract-type constructor).
struct _ContractTestLO <: AbstractLocalOptimization end
VgtkCore.RefitFamily(::_ContractTestLO) = ExternalRefit()

# Small clean-ish homography problem (the HomographyScenes scene).
_contract_problem(; n_in = 40, n_out = 10, noise = 0.3, seed = 7) =
    HomographyProblem(csponds(homography_pairs(StableRNG(seed), H_TRUE; n_in, n_out, σ = noise)...))

@testset "ConsensusSet epoch freshness + cached k*" begin
    problem = _contract_problem()
    scoring = MsacScoring(problem; threshold = 1.5)   # 5σ of the scene noise
    ws = RansacWorkspace(problem, scoring, Irls{WelschShape}(scale_multiple = 1.5))
    buf = ws.buf
    # A model from an all-inlier sample (the scene's first 40 points are its
    # inliers). One noisy quadruple can be conditioned badly enough to support
    # only itself, so the best supported of ten disjoint quadruples is used.
    samples = [collect(i:(i + 3)) for i in 1:4:37]
    models = solve(problem, first(samples))
    @test models !== nothing && !isempty(models)
    model = argmax(m -> score_model!(buf, scoring, problem, m)[2],
                   (m for idx in samples for m in solve(problem, idx)))

    # score_model! returns (S, k*) — the scorer's k*, equal to the mask popcount.
    S, k = score_model!(buf, scoring, problem, model)
    @test S isa Float64
    @test k == inlier_count(scoring, buf.scoring_ws) == count(buf.mask)
    @test k > minimal_sample_size(problem)

    # Fresh set: stamped with the current epoch, consumable by the LO init.
    fresh() = consensus_set(scoring, buf.scoring_ws, buf.residuals;
                            model_dof = minimal_sample_size(problem) + 1, d_g = codimension(problem), epoch = buf.epoch)
    cs = fresh()
    @test cs isa ConsensusSet
    @test inlier_count(cs) == k && is_fresh(cs, buf.epoch) && is_fresh(cs, buf)
    @test isfinite(scale_s2(cs)) && scale_s2(cs) > 0
    state = VgtkEstimation._seeded_irls_state!(ws.lo_ws, problem, model, cs, buf, scoring, 20.0)   # must not throw
    @test s_hat_sq(state.rws) > 0

    # Stale read: any subsequent score bumps the epoch and invalidates
    # previously issued ConsensusSets — the LO-init freshness check must refuse.
    stale = fresh()
    score_model!(buf, scoring, problem, model)            # epoch bump
    @test !is_fresh(stale, buf.epoch)
    @test_throws ArgumentError VgtkEstimation._seeded_irls_state!(ws.lo_ws, problem, model, stale, buf, scoring, 20.0)
    @test_throws ArgumentError check_fresh(stale, buf.epoch)
    # the fresh set's scale is what the score hands a consensus-anchored
    # strategy — through the owner, VGC's `loss_at` (here via `lo_loss`)
    cs = fresh()
    @test scale_s2(VgtkEstimation.lo_loss(Irls{WelschShape}(scale_multiple = 1.0), cs, scoring, buf)) == max(scale_s2(cs), eps())
end

@testset "weight-policy verbs available per adapter" begin
    for T in (
            PosteriorLikelihood,
            # One row: `update_nuisance!`/`update_weights!` are declared on the bare
            # `LossPlainWeights`, so the deployed-shape parameter is
            # dispatch-irrelevant and three spellings tested one method.
            VgtkEstimation.LossPlainWeights,
        )   # the moved strategies' policies: LocalOptimizationDoneRight's suite
        @test hasmethod(update_nuisance!, Tuple{T})
        @test hasmethod(update_weights!, Tuple{T})
    end
    # The verbs ARE the VGC verbs (one generic across the stack).
    @test VgtkEstimation.update_nuisance! === VgtkCore.update_nuisance!
    @test VgtkEstimation.update_weights! === VgtkCore.update_weights!
end

@testset "NormalizedWeightPolicy membership (the ρ(∞)=∞ boundary)" begin
    # Proper normalized probability models (t/uniform mixture family): yes.
    for T in (
            PosteriorLikelihood,
        )
        @test T <: NormalizedWeightPolicy
    end
    # Loss-derived reweighting rules (bounded potentials / frozen scale): no.
    for T in (
            VgtkEstimation.LossPlainWeights,
        )
        @test T <: AbstractWeightPolicy
        @test !(T <: NormalizedWeightPolicy)
    end
end

@testset "loss trait table (liftable × bounded)" begin
    #                    loss                 liftable  bounded
    table = [
        (GaussianShape(), false, false),
        (WelschShape(), true, true),
        (GemanMcclureShape(), true, true),
        (CauchyShape(), true, false),   # THE witness: liftable ≠ bounded
        (SmoothTLSShape(), true, true),
    ]
    for (shape, lft, bnd) in table
        @test liftable(shape) == lft
        @test bounded(shape) == bnd
    end
end

@testset "RefitFamily routing (VGC trait + constructor extension)" begin
    # VGC default: every strategy is NativeRefit unless it declares otherwise.
    @test RefitFamily(NoLocalOptimization()) isa NativeRefit
    @test RefitFamily(Irls{WelschShape}(scale_multiple = 1.5)) isa NativeRefit
    @test RefitFamily(Irls{VgtkCore.TLSShape}()) isa NativeRefit
    # Constructor-extension pattern (what eval's engines do): dispatches.
    @test RefitFamily(_ContractTestLO()) isa ExternalRefit
end

    # -------------------------------------------------------------------------
    # 6. with_width — the SET accessor paired with `lo_width`/`scale_interface`.
    #
    # One strategy per CONSTRUCTOR SHAPE (parametric, plain-keyword,
    # positional-then-keyword, validating), asserting the width mode moved and
    # everything else did not.
    # -------------------------------------------------------------------------
    @testset "with_width sets the width mode and preserves every other field" begin
        RVGK = VisualGeometryToolkitEstimation
        strategies = Any[
            RVGK.Irls{RVGK.WelschShape}(scale_multiple = 1.5),                                   # parametric
            RVGK.Irls{RVGK.GemanMcclureShape}(scale_multiple = 1.25, max_iter = 7),
        ]   # the moved strategies' constructor shapes: LocalOptimizationDoneRight's suite
        for lo in strategies
            @test lo_width(lo) isa ScaleMultiple
            @test VgtkCore.scale_interface(lo) == VgtkCore.ConsensusAnchored()
            out = with_width(lo, LossWidth(0.25))
            @test typeof(out).name === typeof(lo).name
            @test lo_width(out) == LossWidth(0.25)
            @test VgtkCore.scale_interface(out) == FixedScale()
            for f in fieldnames(typeof(lo))
                f === :width && continue
                @test isequal(getfield(out, f), getfield(lo, f))
            end
            # and back: the multiple restored is the strategy constructed with it
            @test with_width(out, lo_width(lo)) == lo
        end
        # A strategy with no width carries no width mode and implements no setter.
        @test lo_width(NoLocalOptimization()) === missing
        @test_throws MethodError with_width(NoLocalOptimization(), LossWidth(1.0))
    end


@testset "the consensus set of MSAC at a threshold (deployed TLS loss)" begin
    # MSAC's loss is truncated least squares, so MSAC at a threshold is
    # TLSShape deployed at that width. Plant 45 inliers near a known H and 15 uniform outliers.
    rng = Xoshiro(7)
    H = HomographyMat{Float64}(Tuple(@SMatrix [1.05 0.02 3.0; -0.01 0.98 -2.0; 1e-5 2e-5 1.0]))
    src = [uniform_source(rng) for _ in 1:60]
    pairs, _ = contaminated_pairs(
        rng, src[1:45] .=> homography_warp(H).(src[1:45]); σ = 0.5, n_out = 15,
        outlier = (rng, k) -> src[45 + k] => uniform_source(rng)
    )
    problem = HomographyProblem(pairs)
    scoring = MsacScoring(problem; threshold = 3.0)                # MSAC at loss width 3 px
    ws = RansacWorkspace(problem, scoring)
    buf = ws.buf
    score_model!(buf, scoring, problem, H)
    cs = consensus_set(scoring, buf.scoring_ws, buf.residuals;
                       model_dof = minimal_sample_size(problem) + 1, d_g = codimension(problem), epoch = buf.epoch)

    @test cs isa ConsensusSet
    @test inlier_count(cs) == 45                                   # the planted inliers, no outliers
    @test is_fresh(cs, buf.epoch)
    # a TLS strategy that follows the scorer deploys TLS at the scorer's threshold
    @test sq_loss_width(VgtkEstimation.lo_loss(Irls{VgtkCore.TLSShape}(), cs, scoring, buf)) == 9.0
end

end

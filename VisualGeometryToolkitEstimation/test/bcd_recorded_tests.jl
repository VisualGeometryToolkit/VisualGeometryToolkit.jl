@testitem "block_coordinate_descent: Recorded == bare, every state" setup = [DivisionBoards, EpipolarScenes] begin
# =============================================================================
# VGC's `Recorded` decorator on every block-coordinate state RVG runs: the
# posterior IRLS (the frozen-loss likelihood of `Irls{K}`; the KeepModel EM of
# `fit_residual_mixture`) and the division-camera IRLS. A recorded run must BE the bare run:
# the same result, and — truncated after every iteration k — the same state,
# so iterate for iterate. Each state is built as its `locally_optimize!` arm
# builds it (the RVG internals), on a fresh workspace per run.
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays
using StableRNGs
using Rotations: RotXYZ
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: CameraModel, Point2, Vec3, csponds, estimate, retcode, iterations, Attributed,
    WelschShape, GemanMcclureShape, HuberShape, block_coordinate_descent!, Recorded, current_cost, MsacScoring
using VisualGeometryToolkitEstimation
const VgtkEstimation = VisualGeometryToolkitEstimation
const VgtkCore = VisualGeometryToolkitCore


# The consensus set of `model`, as `_locally_optimize_seed!` hands it to the
# local optimizer (`_lo_on_partition!`), on a workspace composed for `lo`.
function seeded(p, sc, model, lo)
    ws = VgtkEstimation._entry_workspace(nothing, p, sc, lo)
    VgtkEstimation._prepare!(p)
    buf = ws.buf
    VgtkCore.score_model!(buf, sc, p, VgtkEstimation._seed_scored!(ws, p, sc, model))
    cs = VgtkCore.consensus_set(sc, buf.scoring_ws, buf.residuals;
                         model_dof = VgtkEstimation.minimal_sample_size(p) + 1, d_g = VgtkEstimation.codimension(p), epoch = buf.epoch)
    return ws, cs
end

# (state, likelihood, method, problem, θ₀, max_iter) of the core's LO arm, fresh:
# the static M-estimator `Irls{K}` on the posterior-weight iteration, built as
# its `locally_optimize!` builds it.
function arm(p, sc, model::M, lo::VgtkEstimation.Irls) where {M}
    ws, cs = seeded(p, sc, model, lo)
    buf = ws.buf
    n = VgtkEstimation.data_size(p)
    state = VgtkEstimation._seeded_irls_state!(ws.lo_ws, p, model, cs, buf, sc, something(VgtkCore.outlier_halfwidth(sc), 1.0))
    lh = VgtkEstimation.LossPlainWeights(state.rws, n, VgtkEstimation.lo_loss(lo, cs, sc, buf))
    VgtkEstimation._seed_policy_weights!(state, lh, n)
    return state, lh, VgtkEstimation.PosteriorIrlsMethod(ws.lo_ws.refine), p, model, lo.max_iter
end

# The division-camera IRLS, set up as `fit_division_camera` sets it up: the shared
# posterior-weight iteration under its table's scorer and EM (`_division_irls`).
function division_arm()
    cams = [board_camera(RotXYZ(a...), t) for (a, t) in zip(BOARD_ANGLES, BOARD_TS)]
    boards, _ = noisy_board_views(StableRNG(3), cams; σ = 0.4, ε = 0.08)
    ids, Hs, bs = VgtkEstimation._keyed_views(Dict(enumerate(VgtkEstimation.RadialCamera.(cams))), Dict(enumerate(boards)))
    prior = VgtkEstimation._point_weights(bs)
    N = VgtkEstimation._division_normalization(bs, VgtkEstimation._shared_center(Hs), prior)
    problem = VgtkEstimation.DivisionCameraProblem(Hs, bs, prior, 1, nothing, N, sum(length, bs), false)
    state, lh, method, state0 = VgtkEstimation._division_irls(problem, VgtkEstimation._division_scoring_default(),
                                                              VgtkEstimation._division_mixture_em_default(), 2500.0)
    return state, lh, method, problem, state0, 40
end

# The KeepModel EM of `fit_residual_mixture`, its posterior seeded at the
# residuals' median (the scorer-seeded start is the marginal's,
# LocalOptimizationDoneRight's test).
function mixture_arm()
    rng = StableRNG(11)
    r = vcat([0.8 * hypot(randn(rng), randn(rng)) for _ in 1:300],
             [hypot(40.0 * (2rand(rng) - 1), 40.0 * (2rand(rng) - 1)) for _ in 1:60])
    n = length(r)
    rws = VgtkEstimation.median_seeded_posterior(r .^ 2, n; outlier_halfwidth = 40.0, codimension = 2)
    state = VgtkEstimation.PosteriorIrlsState(rws, copy(r), copy(r), r, VgtkEstimation.mixture_nll(rws, r, n))
    return state, VgtkEstimation.PosteriorLikelihood(rws, n), VgtkEstimation.PosteriorIrlsMethod(nothing, VgtkEstimation.MixtureEmConfig().model_update),
        VgtkEstimation._FixedDistances(r), r, 30
end

# Recorded == bare, iterate for iterate: VisualGeometryToolkitCore's `test_recorded`.

problems = (
    (:H, VgtkEstimation.HomographyProblem(contaminated_pixel_pairs(Xoshiro(101), 300; planar = true))),
    (:F, VgtkEstimation.FundMatProblem(contaminated_pixel_pairs(Xoshiro(111), 300; planar = false))),
    (:E, VgtkEstimation.EssentialProblem(CameraModel(PIXEL_PAIRS_K), CameraModel(PIXEL_PAIRS_K), contaminated_pixel_pairs(Xoshiro(121), 300; planar = false))),
)
# The moved strategies' states (the Student-t + uniform and Barron-adaptive
# IRLS, StuTuBCD, the HQ lifts, GNC, the mixture lifts): LocalOptimizationDoneRight's suite.
arms(kind) = Any[VgtkEstimation.Irls{WelschShape}(efficiency_constant = 1.5), VgtkEstimation.Irls{GemanMcclureShape}(efficiency_constant = 1.9422174675375463)]   # GM: decision_width(GM, 1.25), the retired half-weight currency's 1.25
@testset "Recorded == bare: $kind $(nameof(typeof(lo)))" for (kind, p) in problems, lo in arms(kind)
    sc = MsacScoring(p; threshold = 3.0)   # 6σ of the scenes' 0.5 px noise
    θ = estimate(VgtkEstimation.ransac(p, sc; rng = Xoshiro(1), config = VgtkEstimation.RansacConfig(max_trials = 200)))
    @test VgtkCore.test_recorded(() -> arm(p, sc, θ, lo)) >= 1
end
@testset "Recorded == bare: division camera IRLS" begin
    @test VgtkCore.test_recorded(division_arm) >= 1
end
@testset "Recorded == bare: the KeepModel EM" begin
    @test VgtkCore.test_recorded(mixture_arm) >= 1
end
end

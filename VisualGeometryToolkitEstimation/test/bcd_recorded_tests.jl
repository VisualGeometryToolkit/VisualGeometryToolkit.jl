@testitem "block_coordinate_descent: Recorded == bare, every state" setup = [DivisionBoards] begin
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
using VisualGeometryToolkitCore: CameraModel, Point2, Vec3, csponds, estimate, retcode, iterations, Incumbent,
    WelschShape, GemanMcclureShape, HuberShape, block_coordinate_descent!, Recorded, current_cost, MsacScoring
using VisualGeometryToolkitEstimation
const VgtkEstimation = VisualGeometryToolkitEstimation
const VgtkCore = VisualGeometryToolkitCore

K = SA[800.0 0 320; 0 820.0 240; 0 0 1]
function two_view(rng, n; planar)
    Rm = SMatrix{3, 3, Float64, 9}(RotXYZ(0.05, -0.1, 0.03)); t = SA[1.0, 0.1, 0.05]
    u1 = Point2{Float64}[]; u2 = Point2{Float64}[]
    for _ in 1:n
        X = SA[2 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), planar ? 4.0 : 3.0 + 2 * rand(rng)]
        p1 = K * X; p2 = K * (Rm * X + t)
        push!(u1, Point2{Float64}(p1[1] / p1[3], p1[2] / p1[3]))
        push!(u2, rand(rng) < 0.4 ? Point2{Float64}(640 * rand(rng), 480 * rand(rng)) :
                  Point2{Float64}(p2[1] / p2[3] + 0.5 * randn(rng), p2[2] / p2[3] + 0.5 * randn(rng)))
    end
    return csponds(u1, u2)
end

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

# The division-camera IRLS, set up as `fit_division_camera` sets it up.
function division_arm()
    cams = [board_camera(RotXYZ(a...), t) for (a, t) in zip(BOARD_ANGLES, BOARD_TS)]
    boards, _ = noisy_board_views(StableRNG(3), cams; σ = 0.4, ε = 0.08)
    ids, Hs, bs = VgtkEstimation._keyed_views(Dict(enumerate(VgtkEstimation.RadialCamera.(cams))), Dict(enumerate(boards)))
    prior = VgtkEstimation._point_weights(bs)
    N = VgtkEstimation._division_normalization(bs, VgtkEstimation._shared_center(Hs), prior)
    state0 = VgtkEstimation._division_camera_step(N, Hs, bs; ncoef = 1, k = nothing, point_weights = prior,
                                     whiten_at = 2500.0, whitening = VgtkCore.SampsonWhitening())
    n = sum(length, bs)
    rws = VgtkEstimation.median_seeded_posterior(VgtkEstimation._focal_sq!(Vector{Float64}(undef, n), state0), n;
                                    outlier_halfwidth = 50.0, codimension = 2)
    method = VgtkEstimation.DivisionCameraIrls(Hs, bs, prior, 1, nothing, VgtkCore.SampsonWhitening(), N, false)
    return VgtkEstimation.DivisionCameraIrlsState(state0, rws, n, Inf, 0), VgtkEstimation.PosteriorLikelihood(rws, n), method,
        nothing, state0, 40
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
        VgtkEstimation._FixedResiduals(r), r, 30
end

# Everything a state holds that a run moves (numbers, arrays, incumbents, the
# GNC ladder), not the shared workspaces it points into.
snap(x::Union{Number, Symbol, Nothing}) = x
snap(x::AbstractArray{<:Number}) = collect(x)
snap(x::StaticArray) = Tuple(x)
snap(x::Incumbent) = (snap(x.θ), x.cost)
snap(x) = isbits(x) ? x : nameof(typeof(x)) in (:PosteriorWeightState, :_PositionalDivisionState, :MuLadderState) ?
    Tuple(snap(getfield(x, i)) for i in 1:fieldcount(typeof(x))) : :skipped
snap_state(st) = Tuple(snap(getfield(st, i)) for i in 1:fieldcount(typeof(st)))
fp(res) = (snap(estimate(res)), retcode(res), iterations(res))

function check_recorded(mk)
    # the full run, bare and recorded
    st, lh, m, p, θ₀, max_iter = mk()
    bare = block_coordinate_descent!(st, lh, m, p, θ₀; max_iter)
    bare_state = snap_state(st)
    st2, lh2, m2, p2, θ₀2, _ = mk()
    rec = Recorded(st2)
    @test isequal(fp(block_coordinate_descent!(rec, lh2, m2, p2, θ₀2; max_iter)), fp(bare))
    @test isequal(snap_state(st2), bare_state)
    @test rec.reason[] === retcode(bare)
    k_run = iterations(bare)
    k_rec = length(rec.iterates)
    @test k_rec == (retcode(bare) === :solve_failed ? k_run - 1 : k_run)
    @test length(rec.cost) == length(rec.s2) == k_rec
    # iterate for iterate: the bare run truncated after k is the recorded run's k-th entry
    for k in 1:k_rec
        stk, lhk, mk_, pk, θk, _ = mk()
        resk = block_coordinate_descent!(stk, lhk, mk_, pk, θk; max_iter = k)
        @test isequal(current_cost(stk), rec.cost[k])
        @test iterations(resk) == k
        k == k_rec && @test isequal(snap_state(stk), bare_state)
    end
    return k_rec
end

problems = (
    (:H, VgtkEstimation.HomographyProblem(two_view(Xoshiro(101), 300; planar = true))),
    (:F, VgtkEstimation.FundMatProblem(two_view(Xoshiro(111), 300; planar = false))),
    (:E, VgtkEstimation.EssentialProblem(CameraModel(K), CameraModel(K), two_view(Xoshiro(121), 300; planar = false))),
)
# The moved strategies' states (the Student-t + uniform and Barron-adaptive
# IRLS, StuTuBCD, the HQ lifts, GNC, the mixture lifts): LocalOptimizationDoneRight's suite.
arms(kind) = Any[VgtkEstimation.Irls{WelschShape}(scale_multiple = 1.5), VgtkEstimation.Irls{GemanMcclureShape}(scale_multiple = 1.25)]
@testset "Recorded == bare: $kind $(nameof(typeof(lo)))" for (kind, p) in problems, lo in arms(kind)
    sc = MsacScoring(p; threshold = 3.0)   # 6σ of the scenes' 0.5 px noise
    θ = estimate(VgtkEstimation.ransac(p, sc; rng = Xoshiro(1), config = VgtkEstimation.RansacConfig(max_trials = 200)))
    @test check_recorded(() -> arm(p, sc, θ, lo)) >= 1
end
@testset "Recorded == bare: division camera IRLS" begin
    @test check_recorded(division_arm) >= 1
end
@testset "Recorded == bare: the KeepModel EM" begin
    @test check_recorded(mixture_arm) >= 1
end
end

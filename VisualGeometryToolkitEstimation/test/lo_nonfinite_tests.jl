@testitem "lo_nonfinite" begin
# =============================================================================
# The local optimization's two weight rules (0.11.2)
# =============================================================================
#
# WHY THIS FILE EXISTS: `Irls{GaussianShape}` weighed EVERY datum 1, and a datum
# whose residual is not finite (an out-of-reach point; an infinite Sampson
# distance) made Σwᵢrᵢ² infinite at the seed and at every refit. `Inf < Inf` is
# false, so the local optimization never accepted a refit: it handed back its
# seed (`NoImprovement`) on any problem with one non-finite residual, and on
# every other problem it fitted the outliers at full weight. The rules:
#
#   1. a non-finite residual gets weight 0, under every shape and policy;
#   2. under the Gaussian shape the refit weighs only the scorer's consensus set
#      (the standard LO-RANSAC least squares on the inliers).
# =============================================================================

using Test
using StaticArrays
using StableRNGs
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: Point2, Line2D, GaussianShape, WelschShape, HuberShape, MsacScoring, PosteriorWeightState, JeffreysPrior, deploy, update_weights!, robust_weight, inlier_weight, is_inlier
using VisualGeometryToolkitEstimation: RansacWorkspace

# The line y = 0: 57 inliers (σ = 0.01), 8 finite outliers on one side (0.4 off,
# out of the scorer's 0.1 threshold) and, when `out_of_reach`, 3 points whose
# residual overflows the Float32 residual storage (Inf).
function scene(rng; out_of_reach::Bool)
    pts = [Point2{Float64}(Float64(i), 0.01 * randn(rng)) for i in 0:56]
    append!(pts, [Point2{Float64}(Float64(x), 0.4) for x in 5:7:54])
    out_of_reach && append!(pts, [Point2{Float64}(Float64(x), 1.0e200) for x in (10, 30, 50)])
    return LineFittingProblem(pts)
end
const SEED = Line2D{Float64}(SVector(0.0, 1.0, -0.03))   # y = 0.03: the inliers' consensus, the outliers' not
off_line(line) = abs(signed_distance(line, Point2(28.0, 0.0)))

@testset "Gaussian LO" begin
    @testset "improves its seed beside out-of-reach points" begin
        prob = scene(StableRNG(71); out_of_reach = true)
        r = solve(RefineProblem(prob, SEED), LocalOptimization(Irls{GaussianShape}(); scoring = MsacScoring(prob; threshold = 0.1)))
        @test r.u != SEED
        @test off_line(r.u) < 0.005
    end
    @testset "fits the consensus set, not the outliers" begin
        # Least squares over every point would sit 8·0.4/65 ≈ 0.05 off.
        prob = scene(StableRNG(72); out_of_reach = false)
        r = solve(RefineProblem(prob, SEED), LocalOptimization(Irls{GaussianShape}(); scoring = MsacScoring(prob; threshold = 0.1)))
        @test off_line(r.u) < 0.005
    end
end

@testset "a robust shape: only the non-finite rule" begin
    # Huber's weight at an infinite residual is already 0, and nothing else
    # changes: the finite outliers keep their own (nonzero) Huber weight — no
    # consensus restriction.
    prob = scene(StableRNG(73); out_of_reach = true)
    sc = MsacScoring(prob; threshold = 0.1)
    lo = Irls{HuberShape}(efficiency_constant = 1.345)
    ws = RansacWorkspace(prob, sc, lo)
    _, k = VgtkCore.score_model!(ws.buf, sc, prob, SEED)
    est = VgtkEstimation._lo_on_partition!(ws, prob, SEED, sc, k, lo)
    @test off_line(est.u) < off_line(SEED)
    w = ws.lo_ws.posterior_w
    @test all(isfinite, w)
    @test all(>(0), w[58:65])          # the finite outliers: weighed, out of the consensus set
    @test all(iszero, w[66:68])        # the out-of-reach points
end

@testset "the LO's posterior seeding zeroes a non-finite residual (Core's E-step)" begin
    # `consensus_posterior` seeds every posterior-weight LO from the score that just ran:
    # a NaN point's robust weight and inlier posterior are 0, so the seed cost Σwᵢrᵢ² is finite.
    pts = [Point2{Float64}(Float64(i), 0.01 * randn(StableRNG(74))) for i in 0:56]
    push!(pts, Point2{Float64}(NaN, NaN))
    prob = LineFittingProblem(pts)
    sc = MsacScoring(prob; threshold = 0.1)
    ws = RansacWorkspace(prob, sc, Irls{HuberShape}(efficiency_constant = 1.345))
    VgtkCore.score_model!(ws.buf, sc, prob, SEED)
    cs = VgtkCore.consensus_set(sc, ws.buf.scoring_ws, ws.buf.residuals; model_dof = 3, d_g = 1, epoch = ws.buf.epoch)
    rws = VgtkEstimation.consensus_posterior(ws.lo_ws, prob, cs, ws.buf, sc, 1.0)
    @test isnan(ws.buf.residuals[58])
    @test robust_weight(rws)[58] === 0.0 && inlier_weight(rws)[58] === 0.0
    @test isfinite(VgtkCore.weighted_rss(robust_weight(rws), VgtkCore.sq_distances(rws), 58))
end

@testset "the weight policies zero a non-finite residual" begin
    sq = [0.25, Inf, NaN, 4.0]
    state() = PosteriorWeightState(zeros(4), zeros(4), copy(sq), 5.0, 1.0, log(2 * 10.0), 1, JeffreysPrior())
    for loss in (deploy(WelschShape(), 1.0; efficiency_constant = 2.0), deploy(GaussianShape(), Inf; efficiency_constant = 1.0))
        rws = state()
        lh = VgtkEstimation.LossPlainWeights(rws, 4, loss)
        update_weights!(lh)
        w = robust_weight(rws)
        @test w[1] > 0 && w[4] > 0
        @test w[2] === 0.0 && w[3] === 0.0
        @test !is_inlier(lh, 2) && !is_inlier(lh, 3)
    end
    rws = state()
    lh = VgtkEstimation.PosteriorLikelihood(rws, 4)
    update_weights!(lh)
    @test robust_weight(rws)[1] > 0
    @test robust_weight(rws)[2] === 0.0 && robust_weight(rws)[3] === 0.0
    @test inlier_weight(rws)[3] === 0.0
end
end

@testitem "lo_nonfinite: Triggs" begin
# =============================================================================
# The same two rules for the robustified chart LM, Triggs{K} (0.11.3)
# =============================================================================
#
# WHY THIS ITEM EXISTS: `Triggs{GaussianShape}` ran one chart LM over EVERY datum
# at unit weight — least squares on the outliers, and an out-of-reach point's
# row (its residual beyond the storage range) made the LM's cost infinite at its
# start, so the local optimization handed back its seed. The rules, as 0.11.2's
# `Irls{K}`: a datum whose seed residual is not finite weighs 0 under every
# shape; under the Gaussian shape the LM weighs only the scorer's consensus set.
# =============================================================================

using Test
using LinearAlgebra
using StaticArrays
using StableRNGs
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: Point2, Circle, Triggs, GaussianShape, CauchyShape, MsacScoring
using VisualGeometryToolkitEstimation: CircleProblem
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitCore as VgtkCore

# The circle (100, 80), radius 50: 60 inliers (σ = 0.1); `outliers`: 8 points 3 px
# outside it on one arc (out of the scorer's 1 px threshold); `out_of_reach`: 3
# points 1e200 out. (The circle's Sampson distance is degenerate there — Core's
# rule — so their residuals stay finite and their rows are zero rows: these fits
# check that such points move nothing; the non-finite rule itself is checked on
# the LM's weights, "the Triggs weights" below.)
function circle_points(rng; outliers::Bool, out_of_reach::Bool)
    pts = [Point2{Float64}(100 + 50cos(t) + 0.1randn(rng), 80 + 50sin(t) + 0.1randn(rng)) for t in range(0, 2π, length = 61)[1:60]]
    outliers && append!(pts, [Point2{Float64}(100 + 53cos(t), 80 + 53sin(t)) for t in range(0.0, 0.7, length = 8)])
    out_of_reach && append!(pts, [Point2{Float64}(s * 1.0e200, 1.0e200) for s in (1.0, -1.0, 2.0)])
    return pts
end
const SEED = Circle(Point2(100.2, 79.9), 50.15)
circle_error(c) = max(norm(SVector(c.center...) - SVector(100.0, 80.0)), abs(c.r - 50))
fit(pts, lo) = (p = CircleProblem(pts); solve(RefineProblem(p, SEED), LocalOptimization(lo; scoring = MsacScoring(p; threshold = 1.0))).u)

@testset "Gaussian Triggs" begin
    @testset "improves its seed beside out-of-reach points" begin
        c = fit(circle_points(StableRNG(81); outliers = true, out_of_reach = true), Triggs{GaussianShape}())
        @test c != SEED
        @test circle_error(c) < 0.05
    end
    @testset "fits the consensus set, not the outliers" begin
        # least squares over every point sits ~0.3 px off
        c = fit(circle_points(StableRNG(82); outliers = true, out_of_reach = false), Triggs{GaussianShape}())
        @test circle_error(c) < 0.05
    end
end

@testset "a robust shape: only the non-finite rule" begin
    lo = Triggs{CauchyShape}(efficiency_constant = 2.5)
    with = fit(circle_points(StableRNG(83); outliers = true, out_of_reach = true), lo)
    without = fit(circle_points(StableRNG(83); outliers = true, out_of_reach = false), lo)
    v(c) = SVector(c.center..., c.r)
    # the out-of-reach rows weigh 0: the fit is the one without them, up to the LM's
    # stop settings (scaled by the row count; measured 2e-3 px)
    @test norm(v(with) - v(without)) < 0.01
    # no consensus restriction for a robust shape: only the Gaussian Triggs takes the scorer's set
    @test VgtkEstimation._lo_consensus(lo, nothing, nothing, nothing) === nothing
    @test hasmethod(VgtkEstimation._lo_consensus,
                    Tuple{Triggs{GaussianShape}, VgtkCore.ConsensusSet, VgtkCore.ScoreBuffers, Any})
end

@testset "the Triggs weights" begin
    # The circle's out-of-reach rows are already zero rows (Core's Sampson rows give a
    # degenerate element a zero row), so the rule is checked on the weights the LM gets.
    w = fill(NaN, 5)
    r = Float32[0.5, Inf32, NaN32, 2.0, 0.1]
    @test VgtkEstimation._triggs_weights!(w, nothing, Float32[0.5, 1.0, 2.0, 0.1, 3.0], 5) === nothing   # every row at unit weight
    @test collect(VgtkEstimation._triggs_weights!(w, nothing, r, 5)) == [1.0, 0.0, 0.0, 1.0, 1.0]
    @test collect(VgtkEstimation._triggs_weights!(w, BitVector([true, true, true, false, true]), r, 5)) == [1.0, 0.0, 0.0, 0.0, 1.0]
end
end

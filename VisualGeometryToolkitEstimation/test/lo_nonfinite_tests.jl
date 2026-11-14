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
using VisualGeometryToolkitCore: Point2, Line2D, GaussianShape, WelschShape, HuberShape, MsacScoring, estimate,
    PosteriorWeightState, JeffreysPrior, deploy, update_weights!, robust_weight, inlier_weight, is_inlier
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
        r = locally_optimize(prob, SEED; local_optimization_strategy = Irls{GaussianShape}(), scoring = MsacScoring(prob; threshold = 0.1))
        @test estimate(r) != SEED
        @test off_line(estimate(r)) < 0.005
    end
    @testset "fits the consensus set, not the outliers" begin
        # Least squares over every point would sit 8·0.4/65 ≈ 0.05 off.
        prob = scene(StableRNG(72); out_of_reach = false)
        r = locally_optimize(prob, SEED; local_optimization_strategy = Irls{GaussianShape}(), scoring = MsacScoring(prob; threshold = 0.1))
        @test off_line(estimate(r)) < 0.005
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
    @test off_line(estimate(est)) < off_line(SEED)
    w = ws.lo_ws.posterior_w
    @test all(isfinite, w)
    @test all(>(0), w[58:65])          # the finite outliers: weighed, out of the consensus set
    @test all(iszero, w[66:68])        # the out-of-reach points
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

@testitem "line_fitting" begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# fit_line + LineFittingProblem
# =============================================================================
#
# WHY THIS FILE EXISTS: `src/fitting/line.jl` (both `fit_line` methods, the
# scatter accumulator, the parameter covariance) had zero coverage — the
# suite tested `InhomLineFittingProblem` but never the plain `LineFittingProblem`
# LO refit, which shares `fit_line`'s covariance-weighted step. That step must
# honour the LO's SOFT weights (it once used them only as a > 0 mask, so every
# soft-weighting LO refit the outliers at full weight), and the posterior-IRLS
# LO must run until its cost stops falling (it once stopped after two cycles).
# =============================================================================

using Test
using LinearAlgebra
using Statistics: cov
using StaticArrays
using Random
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: Point2, Uncertain, CovMat, UnlabeledFrame, Line2D, HuberShape
using VisualGeometryToolkitCore: coeffs, converged, estimate, geom, iterations, retcode   # VGC accessors (public)
using VisualGeometryToolkitCore: data_size, minimal_sample_size   # the constraint-problem contract (VGC)
using VisualGeometryToolkitCore: MsacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: fit_line, RansacWorkspace
const RVG = VisualGeometryToolkitEstimation



@testset "line fitting" begin

    @testset "fit_line — heteroscedastic (Uncertain points)" begin
        rng = MersenneTwister(29)
        # Half the points are 100× noisier; the weighted fit must ignore them.
        pts = Uncertain{Point2{Float64}, CovMat{2, Float64, 4, UnlabeledFrame{2}}}[]
        for i in 1:60
            σ = isodd(i) ? 0.01 : 1.0
            y = 2.0i + 1.0 + σ * randn(rng)
            Σ = SMatrix{2, 2, Float64}(1.0e-8, 0.0, 0.0, σ^2)
            push!(pts, Uncertain(Point2{Float64}(Float64(i), y), CovMat(Σ)))
        end
        res = fit_line(pts)
        # Precise points lie close to the recovered line.
        errs = [abs(distance(geom(estimate(res)), geom(pts[i]))) for i in 1:2:59]
        @test maximum(errs) < 0.1
        @test res.ν == 58
    end

    @testset "LineFittingProblem — buffer type and LO refit" begin
        rng = MersenneTwister(31)
        pts = [Point2{Float64}(Float64(i), 2.0i + 1.0 + 0.02 * randn(rng)) for i in 1:40]
        prob = LineFittingProblem(pts)

        @test data_size(prob) == 40
        @test minimal_sample_size(prob) == 2
        @test model_type(prob) === Line2D{Float64}

        # `refine!` (θ unread) over all-inlier weights recovers the generating line.
        line = VgtkEstimation.refine!(nothing, prob, nothing, ones(40))
        @test line !== nothing
        for p in pts
            @test abs(distance(line, p)) < 0.2
        end

        # Excluded points carry weight 0 and must not move the fit: zero out the
        # second half and replace it with garbage first.
        pts2 = copy(pts)
        for i in 21:40
            pts2[i] = Point2{Float64}(Float64(i), -500.0 * rand(rng))
        end
        prob2 = LineFittingProblem(pts2)
        w = [i <= 20 ? 1.0 : 0.0 for i in 1:40]
        line2 = VgtkEstimation.refine!(nothing, prob2, nothing, w)
        @test line2 !== nothing
        for i in 1:20
            @test abs(distance(line2, pts2[i])) < 0.2
        end

        # Fewer than 3 surviving weights is a `nothing`, not a throw.
        @test VgtkEstimation.refine!(nothing, prob, nothing, [i <= 2 ? 1.0 : 0.0 for i in 1:40]) === nothing

        # SOFT weights are weights, not a mask: a point at weight 1e-6 barely
        # moves the line. As a mask it would pull it by 0.5·3/60 = 0.025.
        pts3 = [Point2{Float64}(Float64(i), 0.0) for i in 0:56]
        append!(pts3, [Point2{Float64}(Float64(x), 0.5) for x in (10, 30, 50)])
        w3 = [i <= 57 ? 1.0 : 1.0e-6 for i in 1:60]
        line3 = VgtkEstimation.refine!(nothing, LineFittingProblem(pts3), nothing, w3)
        @test abs(distance(line3, Point2(0.0, 0.0))) < 1.0e-4
        @test abs(distance(line3, Point2(56.0, 0.0))) < 1.0e-4
    end

    # 57 inliers on y = 0 (σ = 0.01) and 3 outliers at y = 0.5, seeded at the
    # contaminated mean line.
    function _contaminated_line(rng)
        pts = [Point2{Float64}(Float64(i), 0.01 * randn(rng)) for i in 0:56]
        append!(pts, [Point2{Float64}(Float64(x), 0.5) for x in (10, 30, 50)])
        return LineFittingProblem(pts), Line2D{Float64}(SVector(0.0, 1.0, -0.025))
    end

    # (The Student-t + uniform IRLS version is LocalOptimizationDoneRight's test.)
    @testset "Irls{Welsch} recovers the line through its soft weights" begin
        prob, θ₀ = _contaminated_line(MersenneTwister(57))
        r = locally_optimize(prob, θ₀; local_optimization = Irls{WelschShape}(scale_multiple = 1.5), threshold = 0.1)   # 10σ; the outliers sit 0.475 off the seed
        line = estimate(r)
        @test abs(distance(line, Point2(0.0, 0.0))) < 0.006
        @test abs(distance(line, Point2(56.0, 0.0))) < 0.006
        @test count(r.inlier_mask[1:57]) >= 55 && !any(r.inlier_mask[58:60])
    end

    @testset "posterior IRLS runs until its cost stops falling" begin
        # Huber keeps a bounded pull on the outliers, so its IRLS approaches the
        # robust line geometrically: the cost falls for several cycles. The stop
        # rule once compared the cost against an incumbent it had just set equal
        # to it, and every run stopped after two cycles.
        prob, _ = _contaminated_line(MersenneTwister(58))
        θ₀ = Line2D{Float64}(SVector(0.0, 1.0, -0.1))
        sc = MsacScoring(prob; threshold = 0.2)   # the seed's consensus: the inliers (0.1 off), not the outliers (0.4 off)
        lo = Irls{HuberShape}(scale_multiple = 1.345)
        ws = RansacWorkspace(prob, sc, lo)
        _, k = VgtkCore.score_model!(ws.buf, sc, prob, θ₀)
        est = RVG._lo_on_partition!(ws, prob, θ₀, sc, k, lo)
        @test iterations(est) > 2
        @test retcode(est) in (:converged, :max_iterations)
        @test abs(distance(estimate(est), Point2(28.0, 0.0))) < abs(distance(θ₀, Point2(28.0, 0.0)))
    end

    @testset "Irls survives a failed refit" begin
        # The inhomogeneous line has no weighted refit (`refine!` is `nothing`):
        # the local optimizer must keep its incumbent, not hand `nothing` to the
        # scorer. (StuTuBCD's version: LocalOptimizationDoneRight's suite.)
        rng = MersenneTwister(59)
        pts = [SVector(Float64(i), 1.0 + 0.5i + 0.01 * randn(rng)) for i in 1:40]
        prob = InhomLineFittingProblem(pts)
        r = ransac(prob, MsacScoring(prob; threshold = 0.05); local_optimization = Irls{WelschShape}(scale_multiple = 1.5), rng = MersenneTwister(60))
        @test converged(r)
        @test r.lo_retcode === :solve_failed
    end

end
end

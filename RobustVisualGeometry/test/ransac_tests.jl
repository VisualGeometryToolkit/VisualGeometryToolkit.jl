@testitem "ransac" begin
using Test
using LinearAlgebra
using StaticArrays
using Random
using VisualGeometryCore: Attributed
using VisualGeometryCore: consensus_order, inlier_threshold, ThresholdScoringWorkspace   # the scoring contract (VGC)
using VisualGeometryCore: converged, estimate, iterations, retcode   # VGC accessors (public)
using VisualGeometryCore: FitResult, FixedModels   # owned and exported by VisualGeometryCore
using VisualGeometryCore: MsacScoring, RansacScoring, TLSShape   # owned and exported by VisualGeometryCore
using RobustVisualGeometry: AbstractRansacProblem, GaussianSampler, inlier_ratio   # public, not exported
using RobustVisualGeometry
using RobustVisualGeometry: minimal_sample_size, LineFittingProblem
using VisualGeometryCore: Line2D, Point2, coeffs
using RobustVisualGeometry: RansacWorkspace, draw_sample!
using RobustVisualGeometry: InhomLineFittingProblem, HandEyeProblem, RelposeBundleProblem, UniformSampler, Rigid3
using VisualGeometryCore: EssentialMat, model_type
using VisualGeometryCore: VisualGeometryCore   # the scoring contract the test scorer below extends

# The problem under the generic loop: the package's own 2D line problem
# (`LineFittingProblem`, points with identity covariances — the residual is the
# signed distance, the model a `Line2D`).
_line_coeffs(r) = coeffs(estimate(r))

# A scorer with a workspace of its own type (only what the entry check reads).
struct _WorkspaceTestScoring <: VisualGeometryCore.AbstractScoring end
struct _WorkspaceTestWs end
VisualGeometryCore.codimension(::_WorkspaceTestScoring) = 1
VisualGeometryCore.scoring_workspace(::_WorkspaceTestScoring, _n::Int, _d_g::Int) = _WorkspaceTestWs()
VisualGeometryCore.inlier_count(::_WorkspaceTestScoring, ::_WorkspaceTestWs) = 0

# =============================================================================
# Test data — line y = 2x + 1 with optional outliers
# =============================================================================

function make_line_data(; n_inliers = 100, n_outliers = 50, noise = 0.1, seed = 42)
    rng = MersenneTwister(seed)
    a, b, c = 2.0, -1.0, 1.0
    len = sqrt(a^2 + b^2)
    a, b, c = a / len, b / len, c / len

    inliers = Point2{Float64}[]
    for _ in 1:n_inliers
        x = 10 * rand(rng) - 5
        y = 2x + 1 + noise * randn(rng)
        push!(inliers, SA[x, y])
    end
    outliers = Point2{Float64}[]
    for _ in 1:n_outliers
        x = 30 * rand(rng) - 10
        y = 30 * rand(rng) - 10
        push!(outliers, SA[x, y])
    end
    return vcat(inliers, outliers), SA[a, b, c], n_inliers
end

@testset "RANSAC" begin

    @testset "RansacConfig defaults" begin
        config = RansacConfig()
        @test config.success_prob == 0.99
        @test config.max_trials == 10_000
        @test config.min_trials == 100
    end

    @testset "RansacWorkspace construction" begin
        ws = RansacWorkspace(LineFittingProblem{Float64}, 100)
        @test length(ws.buf.residuals) == 100
        @test length(ws.buf.scores) == 100
        @test length(ws.buf.mask) == 100
        @test length(ws.sample_indices) == minimal_sample_size(LineFittingProblem{Float64}) == 2
        @test ws.has_best == false
        # The default scorer's workspace (MSAC's, at threshold 1.0) is
        # allocated by the problem-type ctor.
        @test ws.buf.scoring_ws isa ThresholdScoringWorkspace
        @test length(ws.buf.scoring_ws.order) == 100
    end

    @testset "draw_sample! uniqueness" begin
        problem = LineFittingProblem(
            [Point2{Float64}(i, i) for i in 1:20]
        )
        indices = Vector{Int}(undef, 5)
        rng = Xoshiro(0xd4a7)

        for _ in 1:100
            draw_sample!(rng, indices, problem)
            @test length(unique(indices)) == 5
            @test all(1 .<= indices .<= 20)
        end
    end

    @testset "seeded rng — bit-reproducible runs" begin
        # High outlier rate so the draw sequence genuinely decides the run:
        # identical results under one seed are then evidence that EVERY draw
        # routes through `rng`, not luck.
        points, _, _ = make_line_data(n_inliers = 60, n_outliers = 180, seed = 99)
        problem = LineFittingProblem(points)
        scoring = MsacScoring(problem; threshold = 0.5)
        cfg = RansacConfig(max_trials = 500, min_trials = 200)
        run_seeded(seed) = ransac(problem, scoring; config = cfg, rng = Xoshiro(seed))
        r1 = run_seeded(7)
        r2 = run_seeded(7)
        @test estimate(r1) == estimate(r2)                    # bitwise-identical model
        @test r1.inlier_mask == r2.inlier_mask
        # Ambient RNG state must be irrelevant when `rng` is supplied.
        Random.seed!(0xDEAD)
        r3 = run_seeded(7)
        @test estimate(r3) == estimate(r1)
        @test r3.inlier_mask == r1.inlier_mask
        # Sampler protocol: rng-leading draw is deterministic per seed.
        s = RobustVisualGeometry.UniformSampler(50)
        i1 = Vector{Int}(undef, 5)
        i2 = Vector{Int}(undef, 5)
        RobustVisualGeometry.draw!(Xoshiro(3), i1, s)
        RobustVisualGeometry.draw!(Xoshiro(3), i2, s)
        @test i1 == i2
    end

    @testset "GaussianSampler draw matches partialsortperm" begin
        # `_draw_gaussian!` calls the exact `partialsortperm!` routine
        # (Base.Sort._sort! + ScratchQuickSort on a Perm ordering) with
        # sampler-owned scratch. This pins the per-draw output bitwise
        # against the public-API reference — if a Base upgrade changes the
        # routine's call shape or semantics, this fails loudly.
        rng = Xoshiro(0x5eed)
        snn = [0.4 + 0.5 * (i % 7) / 7 for i in 1:257]   # deterministic, mixed
        gs = GaussianSampler(snn; snn_sigma = 1 / 8)
        idx = Vector{Int}(undef, 5)
        for _ in 1:200
            RobustVisualGeometry.draw!(rng, idx, gs)
            # keys_buf still holds this draw's keys — reference selection:
            ref = partialsortperm(gs.keys_buf, 1:5; rev = true)
            @test idx == collect(ref)
            @test length(unique(idx)) == 5
        end
        # The hot-loop draw must not allocate (40 KB/draw regression guard).
        RobustVisualGeometry.draw!(rng, idx, gs)   # warm
        @test (@allocated RobustVisualGeometry.draw!(rng, idx, gs)) == 0
        # Cached inverse weights are exactly 1/w (bitwise key parity).
        @test gs.inv_weights == 1.0 ./ gs.weights
    end

    @testset "GaussianSampler draws vary when the SNN mass is concentrated" begin
        # FEWER dominant matches than the sample size (3 at SNN 0, normalized
        # weight ≈ 1/3) among 300 poor ones on the ε floor (≈ 3e-7 each). Every
        # 5-subset holds the 3 dominant matches plus 2 poor ones, drawn in
        # proportion to their (equal) weights. The old key U^(1/w) = U^(3e6)
        # underflowed to 0.0 for nearly every poor match, so those 2 slots were
        # a tie among zeros broken by index: 34 distinct subsets in 500 draws,
        # against 497 with log-space keys. (With MORE dominant matches than the
        # sample size the top-k never reaches the underflowed keys: old and new
        # keys then draw alike, ≈ C(7,5) = 21 subsets for 7 dominant matches.)
        rng = Xoshiro(7)
        snn = vcat(fill(0.0, 3), fill(0.85, 300))
        gs = GaussianSampler(snn; snn_sigma = 1 / 8)
        idx = Vector{Int}(undef, 5)
        draws = Set{Vector{Int}}()
        for _ in 1:500
            RobustVisualGeometry.draw!(rng, idx, gs)
            @test length(unique(idx)) == 5
            push!(draws, sort(idx))
        end
        @test length(draws) > 400
        @test length(union(draws...)) > 250            # the poor matches are all reachable
        @test all(isfinite, gs.keys_buf) && all(<(0), gs.keys_buf)
    end

    @testset "GaussianSampler first pick follows the weights" begin
        # log(U)/w keys are exponential races (−E/w, E ~ Exp(1)): the top key
        # is index i with probability w_i, and the k-draw is successive
        # sampling without replacement. 7 matches at SNN 0.3, 300 on the floor.
        rng = Xoshiro(1)
        gs = GaussianSampler(vcat(fill(0.3, 7), fill(0.85, 300)); snn_sigma = 1 / 8)
        idx = Vector{Int}(undef, 5)
        first_pick = zeros(Int, length(gs.weights))
        M = 100_000
        for _ in 1:M
            RobustVisualGeometry.draw!(rng, idx, gs)
            first_pick[idx[1]] += 1
        end
        @test isapprox(sum(first_pick[1:7]) / M, sum(gs.weights[1:7]); atol = 5.0e-4)
        @test all(i -> isapprox(first_pick[i] / M, gs.weights[i]; atol = 5.0e-3), 1:7)
    end

    @testset "threshold scorers under the loop" begin
        ms = MsacScoring(2; threshold = 0.5)
        @test inlier_threshold(ms, nothing) == 0.5
        @test_throws ArgumentError MsacScoring(2; threshold = -1.0)
        # RANSAC counting runs the same loop to the same line.
        points, true_model, _ = make_line_data(n_inliers = 200, n_outliers = 50)
        problem = LineFittingProblem(points)
        r = ransac(problem, RansacScoring(problem; threshold = 0.5); local_optimization = Irls{TLSShape}(),
                   config = RansacConfig(max_trials = 2000, min_trials = 100), rng = MersenneTwister(144))
        @test converged(r)
        model = _line_coeffs(r)
        dot(model, true_model) < 0 && (model = -model)
        @test model ≈ true_model atol = 0.1
    end

    @testset "MSAC line fitting end-to-end" begin
        points, true_model, n_inliers = make_line_data(n_inliers = 200, n_outliers = 50)
        problem = LineFittingProblem(points)

        scoring = MsacScoring(problem; threshold = 0.5)
        result = ransac(
            problem, scoring;
            config = RansacConfig(max_trials = 2000, min_trials = 100),
            rng = MersenneTwister(145)
        )

        @test converged(result)
        model = _line_coeffs(result)
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol = 0.1
        @test model[2] ≈ true_model[2] atol = 0.1
        @test model[3] ≈ true_model[3] atol = 0.2
        @test inlier_ratio(result) > 0.5
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

    @testset "MSAC high outlier rate" begin
        points, true_model, _ = make_line_data(n_inliers = 50, n_outliers = 200, seed = 456)
        problem = LineFittingProblem(points)

        scoring = MsacScoring(problem; threshold = 0.5)
        result = ransac(
            problem, scoring;
            config = RansacConfig(max_trials = 10_000, min_trials = 500),
            rng = MersenneTwister(146)
        )

        @test converged(result)
        model = _line_coeffs(result)
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol = 0.2
        @test model[2] ≈ true_model[2] atol = 0.2
    end

    @testset "FitResult integration" begin
        points, _, _ = make_line_data()
        problem = LineFittingProblem(points)

        scoring = MsacScoring(problem; threshold = 0.5)
        result = ransac(
            problem, scoring;
            config = RansacConfig(max_trials = 500),
            rng = MersenneTwister(147)
        )

        @test result isa FitResult
        @test estimate(result) isa Line2D{Float64}
        @test retcode(result) === :converged
        @test converged(result) == true
        @test result.inlier_mask isa BitVector
        # Default workspace storage is Float32: per-point vectors come back
        # narrow; the scalar statistics (quality, scale) are always Float64.
        @test result.residuals isa Vector{Float32}
        @test result.weights isa Vector{Float32}
        @test result.quality isa Float64
        @test result.sigma isa Float64
        @test iterations(result) isa Int
        @test iterations(result) > 0
    end

    @testset "dof and scale fields" begin
        points, _, _ = make_line_data(n_inliers = 200, n_outliers = 50, noise = 0.1)
        problem = LineFittingProblem(points)
        p = minimal_sample_size(problem)

        scoring = MsacScoring(problem; threshold = 0.5)
        result = ransac(
            problem, scoring;
            config = RansacConfig(max_trials = 2000, min_trials = 100),
            rng = MersenneTwister(148)
        )
        @test converged(result)
        @test result.dof > 0
        @test result.dof == sum(result.inlier_mask) - p

        # scale is s = √(RSS/ν), the unbiased estimate
        n_in = sum(result.inlier_mask)
        RSS = sum(
            result.residuals[i]^2 for i in 1:length(result.residuals)
                if result.inlier_mask[i]
        )
        nu = n_in - p
        @test result.sigma ≈ sqrt(RSS / nu)
    end

    @testset "adaptive trial count at the edges" begin
        at(mask, k, cfg) = RobustVisualGeometry._adaptive_trials(count(mask), length(mask), k, cfg)
        # success_prob = 1 needs every trial: the budget, not an InexactError.
        cfg1 = RansacConfig(success_prob = 1.0, max_trials = 700, min_trials = 10)
        @test at(BitVector([trues(50); falses(50)]), 2, cfg1) == 700
        # p_failure rounds to 1 (8 inliers of 1000, a 7-point sample): budget.
        cfg = RansacConfig(max_trials = 900, min_trials = 10)
        @test at(BitVector([trues(8); falses(992)]), 7, cfg) == 900
        # An ordinary case is still the hypergeometric count, clamped.
        n = at(BitVector([trues(50); falses(50)]), 2, cfg)
        @test 10 <= n < 900
    end

    @testset "a threshold scorer's workspace grows and shrinks with the problem" begin
        # MSAC's tables are width- and size-free (a consensus order), so a
        # workspace built for a small problem serves a larger one — the same run
        # as on a fresh workspace — and a smaller problem reuses a larger one.
        small, _, _ = make_line_data(n_inliers = 20, n_outliers = 5)
        large, _, _ = make_line_data(n_inliers = 200, n_outliers = 50)
        sc = MsacScoring(2; threshold = 0.5)
        lo = Irls{TLSShape}()
        ws = RansacWorkspace(LineFittingProblem(small), sc, lo)
        r_ws = ransac(LineFittingProblem(large), sc; local_optimization = lo, workspace = ws, rng = MersenneTwister(149))
        r_fresh = ransac(LineFittingProblem(large), sc; local_optimization = lo, rng = MersenneTwister(149))
        @test converged(r_ws) && estimate(r_ws) == estimate(r_fresh) && r_ws.inlier_mask == r_fresh.inlier_mask
        wl = RansacWorkspace(LineFittingProblem(large), sc)
        @test converged(ransac(LineFittingProblem(small), sc; workspace = wl, rng = MersenneTwister(150)))
    end

    @testset "another scorer's workspace is refused at entry" begin
        # Handed MSAC's workspace, a scorer with a workspace of its own type is
        # refused once at entry, by name — not a MethodError deep in `score!`.
        p = LineFittingProblem(first(make_line_data(n_inliers = 20, n_outliers = 5)))
        ws = RansacWorkspace(p, MsacScoring(p; threshold = 0.5), Irls{TLSShape}())
        err = try
            ransac(p, _WorkspaceTestScoring(); workspace = ws)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("built for MsacScoring", err.msg) && occursin("for _WorkspaceTestScoring", err.msg)
    end

    # The traits a pre-allocated workspace reads are defined on the TYPE, so
    # `RansacWorkspace(P, n)` works for every problem, not only the
    # MinimalSolveProblem family.
    @testset "RansacWorkspace(P, n) for every problem type" begin
        for (P, M) in (
                (LineFittingProblem{Float64}, Line2D{Float64}),
                (InhomLineFittingProblem{Float64}, SVector{2, Float64}),
                (HandEyeProblem{UniformSampler}, Rigid3),
                (RelposeBundleProblem{Float64, Matrix{Float64}}, EssentialMat{Float64}),
            )
            @test model_type(P) === M
            ws = RansacWorkspace(P, 40)
            @test ws isa RansacWorkspace
        end
    end

end
end

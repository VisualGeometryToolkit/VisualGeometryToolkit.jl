@testitem "inhom_line" begin
    import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using LinearAlgebra
using StaticArrays
using StructArrays
using Random
using StableRNGs
using ForwardDiff
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: estimate, metadata   # VGC accessors (public)
using VisualGeometryToolkitCore: codimension, solver_covariance, scoring_workspace   # the constraint-problem and scoring contracts (VGC)
using VisualGeometryToolkitCore: MsacScoring, ThresholdScoringWorkspace   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation: RansacMetadata   # public, not exported
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: inlier_threshold   # the scoring contract (VGC)
using VisualGeometryToolkitEstimation: RansacConfig, ransac,
    RansacWorkspace,
    data_size, minimal_sample_size, model_type, constraint_with_jacobians
using VisualGeometryToolkitCore: whiten

# =============================================================================
# The owner's minimal solve and constraint, as functions ForwardDiff can
# differentiate: the problem rebuilt on Dual data.
# =============================================================================

# (x₁, y₁, x₂, y₂) → (a, b) of the line y = a + b·x through the two points.
_solved(x) = first(solve(InhomLineFittingProblem([SVector(x[1], x[2]), SVector(x[3], x[4])]), [1, 2]))
# the constraint g of point i, as a function of the model and of that point's y
_g_of_θ(pts, θ, i) = constraint_with_jacobians(InhomLineFittingProblem([SVector{2, eltype(θ)}(p) for p in pts]), SVector{2}(θ[1], θ[2]), i)[1]
_g_of_y(pts, model, y, i) = constraint_with_jacobians(
    InhomLineFittingProblem([j == i ? SVector(typeof(y)(p[1]), y) : SVector{2, typeof(y)}(p) for (j, p) in enumerate(pts)]),
    SVector{2, typeof(y)}(model), i
)[1]

# =============================================================================
# Tests
# =============================================================================

@testset "Inhomogeneous Line Fitting" begin

    @testset "solver_covariance — ForwardDiff verification (y-only)" begin
        rng = StableRNG(42)

        for trial in 1:10
            # Random two points
            p1 = SA[randn(rng), randn(rng)]
            p2 = SA[randn(rng), randn(rng)]
            while abs(p2[1] - p1[1]) < 0.1
                p2 = SA[randn(rng), randn(rng)]
            end

            pts = [p1, p2, SA[0.0, 0.0]]  # 3rd point unused
            prob = InhomLineFittingProblem(pts)
            idx = [1, 2]

            models = solve(prob, idx)
            @test !isnothing(models)
            model = first(models)

            Σ_θ = solver_covariance(prob, model, idx)
            @test !isnothing(Σ_θ)
            @test Σ_θ isa SMatrix{2, 2, Float64}

            # ForwardDiff reference: full 2×4, then extract y-columns [2, 4]
            x0 = [p1[1], p1[2], p2[1], p2[2]]
            J_ad_full = ForwardDiff.jacobian(_solved, x0)
            J_ad_y = J_ad_full[:, [2, 4]]  # y-only columns
            Σ_ad = J_ad_y * J_ad_y'

            @test maximum(abs.(Matrix(Σ_θ) - Σ_ad)) < 1.0e-10
        end
    end

    @testset "MSAC — end-to-end" begin
        rng = StableRNG(999)

        # True line: y = 5 + 2x
        a_true, b_true = 5.0, 2.0
        n_inliers = 50
        n_outliers = 20
        noise = 0.5

        # Inliers
        x_in = range(-10, 10, length = n_inliers)
        pts_in = [SA[x, a_true + b_true * x + noise * randn(rng)] for x in x_in]

        # Outliers
        pts_out = [SA[20.0 * rand(rng) - 10.0, 50.0 * rand(rng) - 25.0] for _ in 1:n_outliers]

        pts = vcat(pts_in, pts_out)
        n = length(pts)

        prob = InhomLineFittingProblem(pts)

        scoring = MsacScoring(prob; threshold = 3noise)   # 3σ of the inlier noise
        config = RansacConfig(; max_trials = 500, success_prob = 0.999)

        result = ransac(prob, scoring; config, rng = StableRNG(171))

        # Check model accuracy
        model = estimate(result)
        @test abs(model[1] - a_true) < 1.0  # intercept
        @test abs(model[2] - b_true) < 0.5  # slope

        attrs = metadata(result)
        @test attrs isa RansacMetadata

        # Inlier ratio should be reasonable
        ir = sum(attrs.inlier_mask) / n
        @test ir > 0.5
        @test ir < 0.95  # shouldn't include all outliers

        # Most inliers should be in the first n_inliers points
        n_correct_inliers = sum(attrs.inlier_mask[1:n_inliers])
        @test n_correct_inliers >= 0.8 * n_inliers

        # Few outliers should be classified as inliers
        n_false_inliers = sum(attrs.inlier_mask[(n_inliers + 1):end])
        @test n_false_inliers <= 10

        # Scale estimate should be reasonable
        @test attrs.sigma > 0
        @test attrs.sigma < 2.0 * noise
    end

    @testset "Problem-aware constructors" begin
        pts = [SA[Float64(i), 2.0 + 3.0 * i] for i in 1:50]
        prob = InhomLineFittingProblem(pts)

        # MsacScoring(problem; threshold) reads the problem's codimension.
        # Scoring is spec-only; the workspace is allocated separately by the
        # contract's `scoring_workspace(scoring, n, d_g)`.
        mq = MsacScoring(prob; threshold = 1.5)
        mq_manual = MsacScoring(1; threshold = 1.5)
        @test mq === mq_manual
        @test inlier_threshold(mq, nothing) == 1.5
        @test codimension(mq) == codimension(prob) == 1
        mq_ws = scoring_workspace(mq, data_size(prob), codimension(prob))
        @test mq_ws isa ThresholdScoringWorkspace
        @test length(mq_ws.order) == 50

        # Homography problem has different codimension
        src = [SA[1.0, 2.0], SA[5.0, 6.0], SA[9.0, 10.0], SA[13.0, 14.0], SA[17.0, 18.0]]
        dst = [SA[3.0, 4.0], SA[7.0, 8.0], SA[11.0, 12.0], SA[15.0, 16.0], SA[19.0, 20.0]]
        hprob = HomographyProblem(StructArray(src .=> dst))
        hmq = MsacScoring(hprob; threshold = 3.0)
        @test codimension(hmq) == codimension(hprob) == 2
        @test minimal_sample_size(hprob) == 4
        hmq_ws = scoring_workspace(hmq, data_size(hprob), codimension(hprob))
        @test length(hmq_ws.order) == 5
    end

    @testset "constraint_with_jacobians — return verification" begin
        pts = [SA[1.0, 5.0], SA[3.0, -1.0], SA[0.5, 3.5]]
        prob = InhomLineFittingProblem(pts)
        model = SA[7.0, -2.0]  # y = 7 - 2x

        for i in 1:3
            g, G_x, G_θ = constraint_with_jacobians(prob, model, i)

            # g is the problem's scoring residual of point i
            r = zeros(3)
            VgtkEstimation.residuals!(r, prob, model)
            @test g ≈ r[i] atol = 1.0e-14

            # G_x = ∂g/∂yᵢ and G_θ = ∂g/∂(a, b), as ForwardDiff sees g
            @test G_x ≈ SA[ForwardDiff.derivative(y -> _g_of_y(pts, model, y, i), pts[i][2])] atol = 1.0e-14
            @test G_θ ≈ ForwardDiff.gradient(θ -> _g_of_θ(pts, θ, i), Vector(model)) atol = 1.0e-14

            # whiten should be identity (c = 1)
            r, J_w, ℓ = whiten(g, G_x, G_θ)
            @test r ≈ g atol = 1.0e-14
            @test J_w ≈ G_θ atol = 1.0e-14
            @test ℓ ≈ 0.0 atol = 1.0e-14
        end
    end

end
end

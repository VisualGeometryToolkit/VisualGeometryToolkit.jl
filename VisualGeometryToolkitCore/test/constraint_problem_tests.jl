# The constraint-problem protocol (`AbstractConstraintProblem`): the default
# `residuals!` whitens each constraint through `whiten`, so a DEGENERATE
# measurement follows the stack's one Sampson degeneracy rule — residual 0 when
# the constraint vanishes (at the constraint's rounding scale, when it supplies
# one), ±Inf otherwise — and `whitened_residual` is the per-point extension
# point, SIGNED for a scalar constraint.
#
# The restatement it replaced (`_mahalanobis(g, ∂ₓg, Σ̃_x)`) returned the RAW,
# unwhitened `g` for a scalar constraint whenever Σ̃_g ≤ eps — at an exact
# degeneracy, and also for a legitimately small (rescaled) model — and solved
# `Σ̃_g \ g` for a vector one with no degeneracy test.

@testitem "constraint_problem" begin
    using Test, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: AbstractConstraintProblem, _mahalanobis

    # A problem that IS its list of constraint tuples (g, ∂ₓg, ∂_θg[, Σ̃_x]).
    struct _CDProblem{C} <: AbstractConstraintProblem
        cds::Vector{C}
    end
    VgtkCore.data_size(p::_CDProblem) = length(p.cds)
    VgtkCore.codimension(::_CDProblem) = 1
    VgtkCore.minimal_sample_size(::_CDProblem) = 2
    VgtkCore.constraint_with_jacobians(p::_CDProblem, _model, i::Int) = p.cds[i]

    _scalar(g, gx) = (g, SVector{2,Float64}(gx...), SVector(1.0, 2.0))

    # A problem with a per-model context: θ ↦ (θ, 1/(2θ)), built once per model
    # (counted), each point's residual (x − θ)·(1/(2θ)).
    struct _CtxProblem <: AbstractConstraintProblem
        x::Vector{Float64}
        n_contexts::Base.RefValue{Int}
    end
    VgtkCore.data_size(p::_CtxProblem) = length(p.x)
    VgtkCore.codimension(::_CtxProblem) = 1
    VgtkCore.minimal_sample_size(::_CtxProblem) = 1
    VgtkCore.residual_context(p::_CtxProblem, θ::Float64) = (p.n_contexts[] += 1; (θ, 1 / (2θ)))
    VgtkCore.whitened_residual(p::_CtxProblem, ctx::Tuple{Float64,Float64}, i::Int) =
        (p.x[i] - ctx[1]) * ctx[2]

    # The epipolar constraint of FundMat at each pair, as a 5-tuple with the
    # model's rounding scale.
    struct _F5Problem <: AbstractConstraintProblem
        cs::Vector{Pair{SVector{2,Float64},SVector{2,Float64}}}
    end
    VgtkCore.data_size(p::_F5Problem) = length(p.cs)
    VgtkCore.codimension(::_F5Problem) = 1
    VgtkCore.minimal_sample_size(::_F5Problem) = 7
    function VgtkCore.constraint_with_jacobians(p::_F5Problem, F, i::Int)
        u, v = p.cs[i]
        g, Gx, Gθ = VgtkCore.constraint_with_jacobians(F, u, v)
        return (g, Gx, Gθ, I, VgtkCore.constraint_rounding_scale(F, u, v))
    end

    @testset "whitened residuals: whiten, and its degeneracy rule" begin
        @testset "scalar constraint (d_g = 1)" begin
            # regular: g/√c, bit for bit
            @test _mahalanobis(_scalar(0.3, (0.6, 0.8))) === 0.3 / sqrt(0.6^2 + 0.8^2)
            # a legitimately SMALL gradient (c = 1e-18 < eps) is whitened — the
            # residual is scale-free — where the old floor returned the raw g
            @test _mahalanobis(_scalar(3e-10, (6e-10, 8e-10))) ≈ 0.3 rtol = 1e-12
            # exactly vanishing gradient: ±Inf unless the constraint vanishes too
            @test _mahalanobis(_scalar(0.5, (0.0, 0.0))) === Inf
            @test _mahalanobis(_scalar(-0.5, (0.0, 0.0))) === -Inf
            @test _mahalanobis(_scalar(0.0, (0.0, 0.0))) === 0.0
            @test isnan(_mahalanobis(_scalar(0.5, (NaN, 0.0))))
        end

        @testset "vector constraint (d_g = 2)" begin
            g = SVector(0.3, -0.1)
            Gx = SMatrix{2,4,Float64,8}(1.0, 0.2, -0.3, 0.9, 0.5, 0.1, 0.0, -0.4)
            Gθ = SMatrix{2,3,Float64,6}(1.0, 0.0, 0.0, 1.0, 0.5, 0.5)
            Σg = Gx * Gx'
            @test _mahalanobis((g, Gx, Gθ)) ≈ sqrt(dot(g, Σg \ g)) rtol = 1e-12
            @test _mahalanobis((g, Gx, Gθ)) == norm(first(VgtkCore.whiten(g, Gx, Gθ)))
            # heteroscedastic 4-tuple: Σ̃_x enters
            Σx = Diagonal(SVector(4.0, 1.0, 1.0, 0.25))
            @test _mahalanobis((g, Gx, Gθ, Σx)) ≈ sqrt(dot(g, (Gx * Σx * Gx') \ g)) rtol = 1e-12
            # rank-deficient Σ̃_g (parallel rows): an outlier, not a solve through a
            # singular matrix
            Gx1 = SMatrix{2,4,Float64,8}(1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0)
            @test _mahalanobis((g, Gx1, Gθ)) === Inf
            @test _mahalanobis((zero(g), Gx1, Gθ)) === 0.0
        end

        @testset "residuals! reads a degenerate point as an outlier" begin
            n = 12
            cds = [_scalar(0.01 * i, (0.6, 0.8)) for i in 1:n]
            cds[5] = _scalar(0.02, (0.0, 0.0))              # degenerate, unsatisfied
            cds[n] = _scalar(3.0, (0.6, 0.8))               # an ordinary outlier
            p = _CDProblem(cds)
            r = zeros(n); VgtkCore.residuals!(r, p, nothing)
            @test r == [_mahalanobis(cd) for cd in cds]
            @test r[5] === Inf
        end

        @testset "whitened_residual: the per-point default, one signed contract" begin
            n = 8
            cds = [_scalar((-1.0)^i * 0.05 * i, (0.6, 0.8)) for i in 1:n]
            p = _CDProblem(cds)
            r = zeros(n); VgtkCore.residuals!(r, p, nothing)
            @test r == [VgtkCore.whitened_residual(p, nothing, i) for i in 1:n]
            @test any(<(0), r)                               # signed, not a norm
            # vector constraints: the norm
            g = SVector(0.3, -0.1)
            Gx = SMatrix{2,4,Float64,8}(1.0, 0.2, -0.3, 0.9, 0.5, 0.1, 0.0, -0.4)
            Gθ = SMatrix{2,3,Float64,6}(1.0, 0.0, 0.0, 1.0, 0.5, 0.5)
            pv = _CDProblem([(g, Gx, Gθ)])
            @test VgtkCore.whitened_residual(pv, nothing, 1) == norm(first(VgtkCore.whiten(g, Gx, Gθ)))
        end

        @testset "residual_context: per-model work once, per-point residual on it" begin
            # The default context is the model itself.
            @test VgtkCore.residual_context(_CDProblem([_scalar(0.1, (0.6, 0.8))]), :θ) === :θ
            n = 5
            p = _CtxProblem(collect(1.0:n), Ref(0))
            r = zeros(n); VgtkCore.residuals!(r, p, 2.0)
            @test p.n_contexts[] == 1
            @test r == [(x - 2.0) / 4.0 for x in 1.0:n]
        end

        @testset "a 5-tuple constraint: the rounding scale reaches whiten" begin
            # A FundMat problem returning (g, ∂ₓg, ∂_θg, I, ρ), ρ its
            # `constraint_rounding_scale`: residuals! must agree with
            # `sampson_distance` on degenerate and rounding-level pairs, which
            # only the rounding scale decides.
            S2 = SVector{2,Float64}
            # F₁ = diag(1, 0, 1): the point gradient vanishes at x₁ = x₂ = 0 while
            # g = x₁x₂ + 1 does not — an epipolar line at infinity.
            F1 = VgtkCore.FundMat{Float64}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            # F₂ = [e₃]ₓ: both epipoles at the origin, g = x₁y₂ − y₁x₂.
            F2 = VgtkCore.FundMat{Float64}(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            cases = (
                (F1, [S2(0.0, 5.0) => S2(0.0, -3.0),       # gradient exactly 0, g ≠ 0: Inf
                      S2(1e-18, 5.0) => S2(0.0, -3.0),     # gradient at rounding, g ≠ 0: Inf
                      S2(0.7, 2.0) => S2(-0.4, 1.0)]),     # regular
                (F2, [S2(0.0, 0.0) => S2(0.0, 0.0),        # both at the epipoles: 0
                      S2(1e-17, 0.0) => S2(0.0, 1e-17),    # at rounding, g at rounding: 0
                      S2(0.3, -0.2) => S2(0.5, 0.9)]))     # regular
            for (F, cs) in cases
                p = _F5Problem(cs)
                @test length(VgtkCore.constraint_with_jacobians(p, F, 1)) == 5
                n = length(cs)
                r = zeros(n); VgtkCore.residuals!(r, p, F)
                d = [VgtkCore.sampson_distance(F, c) for c in cs]
                @test abs.(r) == d
                @test r == [VgtkCore.signed_sampson_distance(F, c.first, c.second) for c in cs]
            end
            @test VgtkCore.residuals!(zeros(3), _F5Problem(cases[1][2]), F1)[1:2] == [Inf, Inf]
            @test VgtkCore.residuals!(zeros(3), _F5Problem(cases[2][2]), F2)[1:2] == [0.0, 0.0]
        end
    end

    @testset "solver_covariance: a non-provider refuses, naming itself" begin
        p = _CDProblem([_scalar(0.1, (0.6, 0.8))])
        @test !VgtkCore.provides_solver_covariance(typeof(p))
        err = try
            VgtkCore.solver_covariance(p, nothing, [1, 2]); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("_CDProblem", err.msg)
    end

    # The conformance helper every problem's package runs.
    VgtkCore.test_constraint_problem(_CDProblem([_scalar((-1.0)^i * 0.05 * i, (0.6, 0.8)) for i in 1:8]), nothing)
    VgtkCore.test_constraint_problem(_CtxProblem(collect(1.0:5), Ref(0)), 2.0)
end

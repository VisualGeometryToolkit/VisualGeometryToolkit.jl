# The constraint-problem protocol (`AbstractConstraintProblem`): the default
# `whitened_residual` whitens each constraint through the Sampson layer, so a DEGENERATE
# observation follows the stack's one Sampson degeneracy rule — residual 0 when
# the constraint vanishes (at the constraint's rounding scale, when it supplies
# one), ±Inf otherwise — SIGNED and in the constraint's shape;
# `mahalanobis_distance` (its norm, ≥ 0) is the per-point extension point
# `mahalanobis_distances!` loops over, and `whitened_residuals!` stacks the rows.
#
# The restatement it replaced (`distance_of(g, ∂ₓg, Σ̃_x)`) returned the RAW,
# unwhitened `g` for a scalar constraint whenever Σ̃_g ≤ eps — at an exact
# degeneracy, and also for a legitimately small (rescaled) model — and solved
# `Σ̃_g \ g` for a vector one with no degeneracy test.

@testitem "constraint_problem" setup = [ConstraintTuples] begin
    using Test, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: AbstractConstraintProblem
    # `CDProblem`: a problem that IS its list of constraint tuples (g, ∂ₓg, ∂_θg),
    # its points' noise and rounding scales optional (the ConstraintTuples setup).

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

    # The epipolar constraint of FundMat at each pair, and the model's rounding
    # scale at it (the problem's `constraint_rounding_scale`).
    struct _F5Problem <: AbstractConstraintProblem
        cs::Vector{Pair{SVector{2,Float64},SVector{2,Float64}}}
    end
    VgtkCore.data_size(p::_F5Problem) = length(p.cs)
    VgtkCore.codimension(::_F5Problem) = 1
    VgtkCore.minimal_sample_size(::_F5Problem) = 7
    VgtkCore.constraint_with_jacobians(p::_F5Problem, F, i::Int) = VgtkCore.constraint_with_jacobians(F, p.cs[i]...)
    VgtkCore.constraint_rounding_scale(p::_F5Problem, F, i::Int) = VgtkCore.constraint_rounding_scale(F, p.cs[i]...)

    @testset "whitened residuals: the Sampson layer, and its degeneracy rule" begin
        @testset "scalar constraint (d_g = 1)" begin
            # regular: g/√c, bit for bit
            @test whitened_of(_scalar(0.3, (0.6, 0.8))) === 0.3 / sqrt(0.6^2 + 0.8^2)
            @test whitened_of(_scalar(-0.3, (0.6, 0.8))) === -0.3 / sqrt(0.6^2 + 0.8^2)
            @test distance_of(_scalar(-0.3, (0.6, 0.8))) === 0.3 / sqrt(0.6^2 + 0.8^2)   # the distance: |r|
            # a legitimately SMALL gradient (c = 1e-18 < eps) is whitened — the
            # residual is scale-free — where the old floor returned the raw g
            @test distance_of(_scalar(3e-10, (6e-10, 8e-10))) ≈ 0.3 rtol = 1e-12
            # exactly vanishing gradient: ±Inf unless the constraint vanishes too
            @test whitened_of(_scalar(0.5, (0.0, 0.0))) === Inf
            @test whitened_of(_scalar(-0.5, (0.0, 0.0))) === -Inf
            @test distance_of(_scalar(-0.5, (0.0, 0.0))) === Inf
            @test distance_of(_scalar(0.0, (0.0, 0.0))) === 0.0
            @test isnan(distance_of(_scalar(0.5, (NaN, 0.0))))
        end

        @testset "vector constraint (d_g = 2)" begin
            g = SVector(0.3, -0.1)
            Gx = SMatrix{2,4,Float64,8}(1.0, 0.2, -0.3, 0.9, 0.5, 0.1, 0.0, -0.4)
            Gθ = SMatrix{2,3,Float64,6}(1.0, 0.0, 0.0, 1.0, 0.5, 0.5)
            Σg = Gx * Gx'
            @test distance_of((g, Gx, Gθ)) ≈ sqrt(dot(g, Σg \ g)) rtol = 1e-12
            @test distance_of((g, Gx, Gθ)) ≈ norm(cholesky(Symmetric(Gx * Gx')).L \ g) rtol = 1e-14
            # heteroscedastic: the point's Σ̃_x (observation_covariance) enters
            Σx = Diagonal(SVector(4.0, 1.0, 1.0, 0.25))
            @test distance_of((g, Gx, Gθ); Σ = Σx) ≈ sqrt(dot(g, (Gx * Σx * Gx') \ g)) rtol = 1e-12
            # rank-deficient Σ̃_g (parallel rows): an outlier, not a solve through a
            # singular matrix
            Gx1 = SMatrix{2,4,Float64,8}(1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0)
            @test distance_of((g, Gx1, Gθ)) === Inf
            @test distance_of((zero(g), Gx1, Gθ)) === 0.0
        end

        @testset "mahalanobis_distances! reads a degenerate point as an outlier" begin
            n = 12
            cds = [_scalar(0.01 * i, (0.6, 0.8)) for i in 1:n]
            cds[5] = _scalar(0.02, (0.0, 0.0))              # degenerate, unsatisfied
            cds[n] = _scalar(3.0, (0.6, 0.8))               # an ordinary outlier
            p = CDProblem(cds)
            r = zeros(n); VgtkCore.mahalanobis_distances!(r, p, nothing)
            @test r == [distance_of(cd) for cd in cds]
            @test r[5] === Inf
        end

        @testset "whitened_residual signed, mahalanobis_distance its norm" begin
            n = 8
            cds = [_scalar((-1.0)^i * 0.05 * i, (0.6, 0.8)) for i in 1:n]
            p = CDProblem(cds)
            d = zeros(n); VgtkCore.mahalanobis_distances!(d, p, nothing)
            w = [VgtkCore.whitened_residual(p, nothing, i) for i in 1:n]
            @test any(<(0), w)                               # the residual: signed
            @test d == abs.(w)                               # the distance: its norm, bit for bit
            @test d == [VgtkCore.mahalanobis_distance(p, nothing, i) for i in 1:n]
            # the stacked rows of a scalar constraint are the residuals
            @test VgtkCore.whitened_residuals!(zeros(n), p, nothing) == w
            # vector constraints: the d_g components, and the distance their norm
            g = SVector(0.3, -0.1)
            Gx = SMatrix{2,4,Float64,8}(1.0, 0.2, -0.3, 0.9, 0.5, 0.1, 0.0, -0.4)
            Gθ = SMatrix{2,3,Float64,6}(1.0, 0.0, 0.0, 1.0, 0.5, 0.5)
            pv = CDProblem([(g, Gx, Gθ), (-g, Gx, Gθ)])
            wv = VgtkCore.whitened_residual(pv, nothing, 1)
            @test wv isa SVector{2, Float64}
            @test wv ≈ cholesky(Symmetric(Gx * Gx')).L \ g rtol = 1e-14
            @test VgtkCore.mahalanobis_distance(pv, nothing, 1) === norm(wv)
            @test VgtkCore.whitened_residuals!(zeros(4), pv, nothing) ≈ vcat(wv, VgtkCore.whitened_residual(pv, nothing, 2)) rtol = 4eps()
            @test_throws DimensionMismatch VgtkCore.whitened_residuals!(zeros(3), pv, nothing)   # not whole points
            @test_throws DimensionMismatch VgtkCore.whitened_residuals!(zeros(2), pv, nothing)   # not every point
            # a model that is itself a vector: the problem-level rows, not the model-level form
            @test VgtkCore.whitened_residuals!(zeros(4), pv, [1.0, 2.0]) == VgtkCore.whitened_residuals!(zeros(4), pv, nothing)
        end

        @testset "residual_context: per-model work once, per-point residual on it" begin
            # The default context is the model itself.
            @test VgtkCore.residual_context(CDProblem([_scalar(0.1, (0.6, 0.8))]), :θ) === :θ
            n = 5
            p = _CtxProblem(collect(1.0:n), Ref(0))
            r = zeros(n); VgtkCore.mahalanobis_distances!(r, p, 2.0)
            @test p.n_contexts[] == 1
            @test r == [abs((x - 2.0) / 4.0) for x in 1.0:n]
            @test VgtkCore.whitened_residuals!(zeros(n), p, 2.0) == [(x - 2.0) / 4.0 for x in 1.0:n]
            @test p.n_contexts[] == 2
        end

        @testset "a problem's rounding scale reaches the Sampson layer" begin
            # A FundMat problem answering its model's `constraint_rounding_scale`
            # per point: mahalanobis_distances! must agree with
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
                @test VgtkCore.constraint_rounding_scale(p, F, 1) == VgtkCore.constraint_rounding_scale(F, cs[1]...)
                n = length(cs)
                r = zeros(n); VgtkCore.mahalanobis_distances!(r, p, F)
                d = [VgtkCore.sampson_distance(F, c) for c in cs]
                @test r == d
                @test VgtkCore.whitened_residuals!(zeros(n), p, F) == [only(VgtkCore.sampson_residual(F, c.first, c.second)) for c in cs]
            end
            @test VgtkCore.mahalanobis_distances!(zeros(3), _F5Problem(cases[1][2]), F1)[1:2] == [Inf, Inf]
            @test VgtkCore.mahalanobis_distances!(zeros(3), _F5Problem(cases[2][2]), F2)[1:2] == [0.0, 0.0]
        end
    end

    @testset "solver_covariance: a non-provider refuses, naming itself" begin
        p = CDProblem([_scalar(0.1, (0.6, 0.8))])
        @test !VgtkCore.provides_solver_covariance(typeof(p))
        err = try
            VgtkCore.solver_covariance(p, nothing, [1, 2]); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("CDProblem", err.msg)
    end

    @testset "a 4- or 5-tuple constraint is refused, not truncated" begin
        g, Gx = 0.3, SVector(0.6, 0.8)
        p5 = CDProblem([(g, Gx, SVector(1.0, 2.0), I, nothing)])
        @test_throws ArgumentError VgtkCore.whitened_residual(p5, nothing, 1)
        @test_throws ArgumentError VgtkCore.mahalanobis_distance(p5, nothing, 1)
        @test_throws ArgumentError VgtkCore.whitened_constraint_rows(p5, nothing, 1)
    end

    # A `whitened_residual` override written before 0.7, when that hook was the
    # distance: a scalar at d_g = 2. The conformance helper refuses its shape.
    struct _ScalarOverrideProblem <: AbstractConstraintProblem end
    VgtkCore.data_size(::_ScalarOverrideProblem) = 3
    VgtkCore.codimension(::_ScalarOverrideProblem) = 2
    VgtkCore.minimal_sample_size(::_ScalarOverrideProblem) = 1
    VgtkCore.whitened_residual(::_ScalarOverrideProblem, _ctx, i::Int) = 0.1 * i
    # The helper's verdict, captured rather than recorded (its testset fails by
    # design). A nested `@testset` takes its parent's type, so the helper's own
    # testsets are captures too, and hand themselves to the enclosing capture.
    mutable struct _Capture <: Test.AbstractTestSet
        results::Vector{Any}
    end
    _Capture(_desc; _kw...) = _Capture(Any[])
    Test.record(ts::_Capture, t) = (push!(ts.results, t); t)
    function Test.finish(ts::_Capture)
        Test.get_testset_depth() > 0 && (parent = Test.get_testset()) isa _Capture && Test.record(parent, ts)
        return ts
    end
    _fails(ts::_Capture) = sum(t -> t isa _Capture ? _fails(t) : Int(t isa Union{Test.Fail, Test.Error}), ts.results; init = 0)
    @testset "a scalar whitened_residual at d_g = 2 is refused" begin
        ts = Test.@testset _Capture "captured" begin
            VgtkCore.test_constraint_problem(_ScalarOverrideProblem(), nothing)
        end
        @test _fails(ts) == 1                             # the shape, and only the shape
        # the stacked rows read d_g = codimension(problem) components per point and
        # every point: the scalar override fails loudly, never reads past the data
        @test_throws DimensionMismatch VgtkCore.whitened_residuals!(zeros(6), _ScalarOverrideProblem(), nothing)
        @test_throws DimensionMismatch VgtkCore.whitened_residuals!(zeros(4), _ScalarOverrideProblem(), nothing)
    end

    # The conformance helper every problem's package runs.
    VgtkCore.test_constraint_problem(CDProblem([_scalar((-1.0)^i * 0.05 * i, (0.6, 0.8)) for i in 1:8]), nothing)
    VgtkCore.test_constraint_problem(_CtxProblem(collect(1.0:5), Ref(0)), 2.0)
end

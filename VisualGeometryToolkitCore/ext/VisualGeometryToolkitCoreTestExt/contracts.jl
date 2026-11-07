# =============================================================================
# The contract conformance helpers (stubs in src/quality/contract_helpers.jl).
# =============================================================================

import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: scoring_workspace, score!, inlier_count, consensus_order, consensus_set,
    consensus_s2, inlier_threshold, has_inlier_threshold, score_bound, ScoreBuffers, score_model!,
    lo_workspace, locally_optimize!, LoInfo, FitResult, estimate, mahalanobis_distances!, data_size,
    codimension, minimal_sample_size, whitened_residual, mahalanobis_distance, residual_context, model_type,
    check_workspace

using VisualGeometryToolkitCore: Manifold, EssentialManifold, tangent_size, zero_tangent, recenter, retract,
    retract_jacobian, ambient_coordinates
using StaticArrays: SVector, MVector, StaticVector
using LinearAlgebra: norm

function test_manifold_contract(m::Manifold; atol::Real = 1.0e-12, allocations::Bool = true)
    return Test.@testset "manifold contract: $(nameof(typeof(m)))" begin
        D = tangent_size(m)
        z = Test.@inferred zero_tangent(m)                        # the size is a constant of the type
        Test.@test z isa StaticVector && length(z) == D
        if D > 0
            δ = SVector{D}(ntuple(i -> oftype(float(eltype(z))(1), 0.01i), D))
            for v in (δ, MVector(δ))
                Test.@test (Test.@inferred recenter(m, v)) isa typeof(m)   # what a solver's variable holds
            end
            Test.@test recenter(m, collect(δ)) isa typeof(m)               # any AbstractVector, a plain Vector too
            # a known allocation (`allocations = false`) is recorded broken, so its fix shows
            allocations ? Test.@test((@allocated recenter(m, δ)) == 0) : Test.@test_broken((@allocated recenter(m, δ)) == 0)
            fd(v) = ForwardDiff.jacobian(w -> ambient_coordinates(retract(m, w)), v)
            Test.@test isapprox(retract_jacobian(m), fd(z); atol)
            Test.@test isapprox(retract_jacobian(m, δ), fd(δ); atol)
            # a plain Vector is the static tangent: the chart's own method answers, bit for bit
            Test.@test isequal(retract_jacobian(m, collect(δ)), retract_jacobian(m, δ))
            _test_chart_invariants(m, δ)
        end
    end
end
# A chart's own invariants after a step: none by default; the essential chart's
# translation stays on the unit sphere (its retraction normalizes it).
_test_chart_invariants(m::Manifold, δ) = nothing
_test_chart_invariants(m::EssentialManifold, δ) = Test.@test abs(norm(recenter(m, δ).t) - 1) <= 4 * eps(Float64)

# A workspace no scorer builds: every scorer's `check_workspace` must refuse it.
struct _ForeignWorkspace end

function test_scoring_contract(scoring, residuals::AbstractVector; n::Integer = length(residuals),
                               d_g::Integer = 1, sample = nothing)
    return Test.@testset "scoring contract: $(nameof(typeof(scoring)))" begin
        W = scoring_workspace(scoring, n, d_g)
        sc = Vector{Float64}(undef, n)
        S = Test.@inferred score!(W, scoring, sc, residuals, n; sample)
        Test.@test S isa Float64
        score!(W, scoring, sc, residuals, n; sample)
        Test.@test (@allocated score!(W, scoring, sc, residuals, n; sample)) == 0
        Test.@test score!(W, scoring, sc, residuals, n; sample) === S          # deterministic
        k = Test.@inferred inlier_count(scoring, W)
        Test.@test 0 <= k <= n
        # its own workspace is accepted at entry, another scorer's refused
        Test.@test check_workspace(scoring, W) === nothing
        Test.@test_throws ArgumentError check_workspace(scoring, _ForeignWorkspace())
        order = consensus_order(scoring, W)
        Test.@test sort!(collect(order[1:n])) == 1:n
        cs = Test.@inferred consensus_set(scoring, W, residuals; model_dof = 1, d_g, epoch = 1)
        Test.@test VgtkCore.inlier_count(cs) == k
        Test.@test (Test.@inferred consensus_s2(scoring, residuals, order, k; model_dof = 1, d_g)) isa Float64
        if has_inlier_threshold(scoring)
            buf = ScoreBuffers{eltype(residuals)}(n, W)
            Test.@test inlier_threshold(scoring, buf) isa Float64
        end
        b = score_bound(scoring)
        if b !== nothing && isfinite(S)
            # a bound the score beats or ties never stops it early
            Test.@test score!(W, scoring, sc, residuals, n; sample, bound = S) === S
            Test.@test score!(W, scoring, sc, residuals, n; sample, bound = prevfloat(S)) === S
            # an unreachable bound may stop it, at or below the bound
            Test.@test score!(W, scoring, sc, residuals, n; sample, bound = S + n * b + 1) <= S + n * b + 1
            Test.@test score!(W, scoring, sc, residuals, n; sample) === S
        end
    end
end

# The element type is the optional leading type argument (conventions/signatures.md,
# slot 4), never a keyword.
test_local_optimization_contract(lo, problem, θ, scoring; allocations::Bool = true) =
    test_local_optimization_contract(Float64, lo, problem, θ, scoring; allocations)
function test_local_optimization_contract(::Type{T}, lo, problem, θ, scoring; allocations::Bool = true) where {T}
    return Test.@testset "local-optimization contract: $(nameof(typeof(lo))) on $(nameof(typeof(problem)))" begin
        n = data_size(problem)
        d_g = codimension(problem)
        W = scoring_workspace(scoring, n, d_g)
        buf = ScoreBuffers{T}(n, W)
        L = lo_workspace(typeof(problem), T, lo, n)
        run() = begin
            score_model!(buf, scoring, problem, θ)
            cs = consensus_set(scoring, W, buf.residuals; model_dof = minimal_sample_size(problem) + 1, d_g, epoch = buf.epoch)
            locally_optimize!(L, lo, problem, θ, cs, buf, scoring)
        end
        r = run()
        Test.@test r isa FitResult{<:Any, LoInfo}
        Test.@test estimate(r) isa model_type(problem)
        if r.residuals_current
            rr = similar(buf.residuals)
            mahalanobis_distances!(rr, problem, estimate(r))
            Test.@test rr == buf.residuals
        end
        score_model!(buf, scoring, problem, θ)
        cs = consensus_set(scoring, W, buf.residuals; model_dof = minimal_sample_size(problem) + 1, d_g, epoch = buf.epoch)
        Test.@inferred locally_optimize!(L, lo, problem, θ, cs, buf, scoring)
        if allocations
            run()
            Test.@test (@allocated run()) == 0
        end
    end
end

function test_constraint_problem(problem, θ)
    return Test.@testset "constraint problem: $(nameof(typeof(problem)))" begin
        n = data_size(problem)
        Test.@test n isa Int && codimension(problem) isa Int && minimal_sample_size(problem) isa Int
        d = Vector{Float64}(undef, n)
        Test.@inferred mahalanobis_distances!(d, problem, θ)
        Test.@test (@allocated mahalanobis_distances!(d, problem, θ)) == 0
        Test.@test !any(<(0), d)                       # a distance: ≥ 0 (or NaN)
        # to rounding, at the point's or the distances' scale (see the docstring:
        # FMA contraction differs between the loop and a single call)
        ctx = residual_context(problem, θ)
        m = [Float64(mahalanobis_distance(problem, ctx, i)) for i in 1:n]
        finite = filter(isfinite, d)
        scale = isempty(finite) ? 0.0 : maximum(finite)
        Test.@test all(i -> isequal(m[i], d[i]) || isapprox(m[i], d[i]; rtol = 1.0e-11, atol = 1.0e-14 * scale), 1:n)
        # the signed residual has the constraint's shape: a scalar at d_g = 1, the
        # d_g components otherwise — a scalar there is a distance under the
        # residual's name (a `whitened_residual` override written before 0.7,
        # when that hook was today's `mahalanobis_distance`), refused
        d_g = codimension(problem)
        Test.@testset "whitened_residual has the constraint's shape (d_g = $d_g)" begin
            Test.@test all(i -> _has_constraint_shape(whitened_residual(problem, ctx, i), d_g), 1:n)
        end
        # …and its norm is the distance (a gate's non-finite distance a non-finite
        # residual): the two hooks describe one point
        Test.@testset "norm(whitened_residual) is mahalanobis_distance" begin
            Test.@test all(i -> _norm_matches(whitened_residual(problem, ctx, i), m[i], scale), 1:n)
        end
    end
end
_has_constraint_shape(r, d_g::Int) = r isa Union{Real, StaticVector} && length(r) == d_g
function _norm_matches(r, d, scale)
    q = Float64(r isa Real ? abs(r) : sqrt(sum(abs2, r)))
    isfinite(q) || return !isfinite(d)
    return isfinite(d) && (q == d || isapprox(q, d; rtol = 1.0e-11, atol = 1.0e-14 * scale))
end

# =============================================================================
# FitResult — type-stable by construction (the 2026-09-02 hard requirement)
# =============================================================================

@testitem "fit_result" begin
    using VisualGeometryToolkitCore: FitResult, BundleMetadata,
        estimate, retcode, converged, iterations, rms, metadata, weights, dof,
        _reject_shadowing, _FITRESULT_CORE, _FITRESULT_PROPS

    # An IRLS-style metadata (scale, ν, per-point weights), forwarded as properties.
    struct _IrlsMeta
        scale::Float64
        ν::Float64
        weights::Vector{Float64}
    end

    @testset "FitResult" begin
        lean = FitResult(3.14, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 7, 0.21, 100, nothing)
        @test @inferred(estimate(lean)) == 3.14
        @test @inferred(retcode(lean)) === VisualGeometryToolkitCore.ReturnCode.Success
        @test @inferred(converged(lean))
        @test @inferred(iterations(lean)) == 7
        @test @inferred(rms(lean)) == 0.21
        @test metadata(lean) === nothing
        # A lean result has no weights — an ERROR, never a silent `nothing`.
        @test_throws Exception weights(lean)
        @test_throws Exception lean.weights

        ir = FitResult([1.0, 2.0], VisualGeometryToolkitCore.ReturnCode.MaxIters, VisualGeometryToolkitCore.IterationCap, 50, 0.5, 3, _IrlsMeta(1.2, 8.0, [0.9, 1.0, 0.1]))
        @test !converged(ir)
        @test @inferred(weights(ir)) == [0.9, 1.0, 0.1]
        @test ir.metadata.scale == 1.2 && ir.metadata.ν == 8.0

        bu = FitResult(
            :cal, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 12, 0.2, 500, BundleMetadata(
                [1.0], [0.4], [1, 3], 488, 1.1,
                (rms = 0.31, median = 0.25, p90 = 0.62), (rms = 0.2, median = 0.18, p90 = 0.4);
                scale = 0.7, scale_source = VisualGeometryToolkitCore.ScaleGiven
            )
        )
        # the bundle's scale record: read by noise_scale/scale_source, required (no default)
        @test VisualGeometryToolkitCore.noise_scale(bu) === 0.7 && VisualGeometryToolkitCore.scale_source(bu) === VisualGeometryToolkitCore.ScaleGiven
        @test_throws UndefKeywordError BundleMetadata([1.0], [0.4], [1], 1, NaN, bu.before, bu.after)
        @test @inferred(dof(bu)) == 488
        @test bu.metadata.after.median == 0.18 && bu.metadata.before.p90 == 0.62
        @test rms(bu) == bu.metadata.after.rms == 0.2
        # one residual per weight — one per observation
        @test_throws ArgumentError BundleMetadata([1.0], [0.4, 0.5], [1], 1, NaN, bu.before, bu.after;
                                                  scale = NaN, scale_source = VisualGeometryToolkitCore.ScaleNone)
        # concrete types all the way down — no Union fields anywhere
        @test isconcretetype(typeof(ir.metadata)) && isconcretetype(typeof(bu.metadata))
        @test isconcretetype(typeof(ir)) && isconcretetype(typeof(lean))

        @testset "property forwarding into metadata" begin
            # The core fields read as properties …
            @test ir.estimate == [1.0, 2.0] && ir.retcode === VisualGeometryToolkitCore.ReturnCode.MaxIters && ir.stop_rule === VisualGeometryToolkitCore.IterationCap
            @test ir.iterations == 50 && ir.rms == 0.5 && ir.n == 3
            # … the derived `converged` is a property too — uniform with the rest,
            # computed from retcode on access, never stored …
            _c(r) = r.converged
            @test @inferred(_c(lean)) === true && @inferred(_c(ir)) === false && _c(bu) === true
            @test ir.converged === converged(ir)
            # … and so do the metadata fields, through the same `@generated`
            # path `Attributed` uses — type-stable, resolved at compile time.
            # (`@inferred` needs a call, so property access goes through a closure)
            _w(r) = r.weights; _s(r) = r.scale; _ν(r) = r.ν; _d(r) = r.dof
            @test @inferred(_w(ir)) == [0.9, 1.0, 0.1]
            @test @inferred(_s(ir)) == 1.2
            @test @inferred(_ν(ir)) == 8.0
            @test @inferred(_d(bu)) == 488
            @test bu.after.rms == 0.2 && bu.fixed_frames == [1, 3]
            @test bu.residuals == [0.4] && bu.loss_width == 1.1
            # The forwarded property IS the accessor — one value, two spellings.
            @test ir.weights === weights(ir)
            @test bu.dof === dof(bu)
            # propertynames advertises core + metadata, in that order.
            @test propertynames(lean) == (:estimate, :retcode, :stop_rule, :iterations, :rms, :n, :metadata, :converged)
            @test propertynames(ir) == (
                :estimate, :retcode, :stop_rule, :iterations, :rms, :n, :metadata, :converged,
                :scale, :ν, :weights,
            )
            # An unknown name is an error on every variant — never `nothing`.
            @test_throws Exception lean.no_such_field
            @test_throws Exception ir.no_such_field
            @test_throws Exception bu.inlier_mask
        end

        @testset "metadata may not shadow a core field" begin
            # A metadata type whose field would collide with the core is refused
            # when the result is BUILT — silently resolving `r.n` to one of two
            # different things is exactly the ambiguity this type exists to kill.
            struct _ShadowingMeta
                n::Int
                extra::Float64
            end
            @test_throws ErrorException FitResult(1.0, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 1, 0.0, 10, _ShadowingMeta(99, 2.0))
            # the derived property's name is reserved just like a stored core field
            struct _ShadowsConverged
                converged::Bool
            end
            @test_throws ErrorException FitResult(1.0, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 1, 0.0, 10, _ShadowsConverged(true))
            # and the check is a compile-time no-op for a valid metadata type
            @test _reject_shadowing(_IrlsMeta) === nothing
            # the core list the forwarder derives from IS the struct's field order
            @test fieldnames(FitResult) == _FITRESULT_CORE
            @test _FITRESULT_PROPS == (_FITRESULT_CORE..., :converged)
        end
    end
end

@testitem "fit_result nested metadata" begin
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: FitResult, nested_metadata, _reject_shadowing

    struct _NestInner
        b::Float64
        c::Vector{Int}
    end
    struct _NestOuter
        a::Int
        inner::_NestInner
    end
    VgtkCore.nested_metadata(::Type{_NestOuter}) = (:inner,)

    struct _NestClashInner
        a::Int
    end
    struct _NestClashOuter   # nested field `a` equals an outer field
        a::Int
        inner::_NestClashInner
    end
    VgtkCore.nested_metadata(::Type{_NestClashOuter}) = (:inner,)

    struct _NestCoreInner
        rms::Float64
    end
    struct _NestClashCore    # nested field `rms` equals a core field
        inner::_NestCoreInner
    end
    VgtkCore.nested_metadata(::Type{_NestClashCore}) = (:inner,)

    struct _NestTwo          # two nested fields declare `b`
        p::_NestInner
        q::_NestInner
    end
    VgtkCore.nested_metadata(::Type{_NestTwo}) = (:p, :q)

    struct _NestNotField
        a::Int
    end
    VgtkCore.nested_metadata(::Type{_NestNotField}) = (:nope,)

    @test nested_metadata(Nothing) === ()
    @test nested_metadata(_NestInner) === ()

    m = _NestOuter(7, _NestInner(2.5, [1, 2]))
    r = FitResult(:est, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 3, 0.1, 9, m)
    @test r.b == 2.5 && r.c == [1, 2] && r.a == 7
    @test r.inner === m.inner
    @test r.estimate === :est && r.converged
    @test :b in propertynames(r) && :c in propertynames(r) && :inner in propertynames(r)
    _b(x) = x.b; _c(x) = x.c; _a(x) = x.a
    @test @inferred(_b(r)) == 2.5
    @test @inferred(_c(r)) == [1, 2]
    @test @inferred(_a(r)) == 7
    @test_throws Exception r.nothing_here

    @test_throws ErrorException FitResult(:e, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 1, 0.0, 1, _NestClashOuter(1, _NestClashInner(2)))
    @test_throws ErrorException FitResult(:e, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 1, 0.0, 1, _NestClashCore(_NestCoreInner(2.0)))
    @test_throws ErrorException FitResult(:e, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 1, 0.0, 1, _NestTwo(_NestInner(1.0, [1]), _NestInner(2.0, [2])))
    @test_throws ErrorException FitResult(:e, VisualGeometryToolkitCore.ReturnCode.Success, VisualGeometryToolkitCore.CostDecrease, 1, 0.0, 1, _NestNotField(1))
    @test _reject_shadowing(_NestOuter) === nothing
end

# The one RMS computation (base/fit_result.jl): over n points, a point's stacked rows its squared norm.
@testitem "rms(residuals, n): the fits' root-mean-square residual" begin
    using Test
    using VisualGeometryToolkitCore: rms
    e = [3.0, 4.0, 12.0]
    @test rms(e) === sqrt((9.0 + 16.0 + 144.0) / 3)
    rows = [3.0, 4.0, 6.0, 8.0]                     # two points' (x, y) rows: norms 5 and 10
    @test rms(rows, 2) === sqrt(sum(abs2, rows) / 2) && rms(rows, 2) ≈ sqrt((25 + 100) / 2)
end

# A RANSAC result's estimate is `Union{M, Nothing}`: `map_estimate` infers `f`'s image of `M` (not of the union, which
# a generic `f` with no `nothing` method widens to `Any`), present or absent.
@testitem "map_estimate: the union's present type mapped, inferred" begin
    using Test
    using VisualGeometryToolkitCore: FitResult, map_estimate, estimate, ReturnCode, NoRule, CostDecrease
    double(x::Float64) = 2x
    double(x) = Base.inferencebarrier(x)                   # a generic fallback, uninferable: what `nothing` would reach
    present = FitResult{Union{Float64, Nothing}, Nothing}(1.5, ReturnCode.Success, CostDecrease, 3, 0.1, 10, nothing)
    absent = FitResult{Union{Float64, Nothing}, Nothing}(nothing, ReturnCode.Failure, NoRule, 0, NaN, 10, nothing)
    @test map_estimate(double, present) isa FitResult{Union{Float64, Nothing}, Nothing}
    @test estimate(map_estimate(double, present)) == 3.0
    @test estimate(map_estimate(double, absent)) === nothing && map_estimate(double, absent) isa FitResult{Union{Float64, Nothing}}
    @test map_estimate(string, present) isa FitResult{Union{String, Nothing}, Nothing}
end

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
        lean = FitResult(3.14, :converged, 7, 0.21, 100, nothing)
        @test @inferred(estimate(lean)) == 3.14
        @test @inferred(retcode(lean)) === :converged
        @test @inferred(converged(lean))
        @test @inferred(iterations(lean)) == 7
        @test @inferred(rms(lean)) == 0.21
        @test metadata(lean) === nothing
        # A lean result has no weights — an ERROR, never a silent `nothing`.
        @test_throws Exception weights(lean)
        @test_throws Exception lean.weights

        ir = FitResult([1.0, 2.0], :max_iterations, 50, 0.5, 3, _IrlsMeta(1.2, 8.0, [0.9, 1.0, 0.1]))
        @test !converged(ir)
        @test @inferred(weights(ir)) == [0.9, 1.0, 0.1]
        @test ir.metadata.scale == 1.2 && ir.metadata.ν == 8.0

        bu = FitResult(
            :cal, :converged, 12, 0.2, 500, BundleMetadata(
                [1.0], [0.4], [1, 3], 488, 1.1,
                (rms = 0.31, median = 0.25, p90 = 0.62), (rms = 0.2, median = 0.18, p90 = 0.4)
            )
        )
        @test @inferred(dof(bu)) == 488
        @test bu.metadata.after.median == 0.18 && bu.metadata.before.p90 == 0.62
        @test rms(bu) == bu.metadata.after.rms == 0.2
        # one residual per weight — one per observation
        @test_throws ArgumentError BundleMetadata([1.0], [0.4, 0.5], [1], 1, NaN, bu.before, bu.after)
        # concrete types all the way down — no Union fields anywhere
        @test isconcretetype(typeof(ir.metadata)) && isconcretetype(typeof(bu.metadata))
        @test isconcretetype(typeof(ir)) && isconcretetype(typeof(lean))

        @testset "property forwarding into metadata" begin
            # The core fields read as properties …
            @test ir.estimate == [1.0, 2.0] && ir.retcode === :max_iterations
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
            @test propertynames(lean) == (:estimate, :retcode, :iterations, :rms, :n, :metadata, :converged)
            @test propertynames(ir) == (
                :estimate, :retcode, :iterations, :rms, :n, :metadata, :converged,
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
            @test_throws ErrorException FitResult(1.0, :converged, 1, 0.0, 10, _ShadowingMeta(99, 2.0))
            # the derived property's name is reserved just like a stored core field
            struct _ShadowsConverged
                converged::Bool
            end
            @test_throws ErrorException FitResult(1.0, :converged, 1, 0.0, 10, _ShadowsConverged(true))
            # and the check is a compile-time no-op for a valid metadata type
            @test _reject_shadowing(_IrlsMeta) === nothing
            # the core list the forwarder derives from IS the struct's field order
            @test fieldnames(FitResult) == _FITRESULT_CORE
            @test _FITRESULT_PROPS == (_FITRESULT_CORE..., :converged)
        end
    end
end

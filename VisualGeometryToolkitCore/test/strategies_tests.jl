# =============================================================================
# The core optimization strategies Irls{K} and Triggs{K}, their loss and
# strategy registries and their defaults file (robust/strategies.jl,
# configs/default.toml). The TOML reader that builds a strategy from a table
# (`optimization_strategy_from_toml`) is VisualGeometryToolkitEstimation's and
# is tested there; this item holds what Core owns: construction, the width
# modes, the registries and the defaults Core reads.
# =============================================================================

@testitem "strategies" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Irls, Triggs, LOSS_TAGS, STRATEGY_TAGS, tag_of, lo_tag, lo_defaults, lo_construct,
        lo_width, with_width, fixed_width, strategy_loss, optimization_path, scale_interface, lo_label, lo_marker,
        EfficiencyConstant, LossWidth, FollowsScorer, ConsensusAnchored, FixedScale, NoScale,
        GaussianShape, WelschShape, GemanMcclureShape, CauchyShape, SmoothTLSShape, TLSShape, HuberShape, BarronShape

    @testset "the defaults file: Core's configs/default.toml" begin
        table = VgtkCore._CORE_DEFAULTS["optimization_strategy"]
        @test table["irls"]["max_iter"] == 25
        @test lo_defaults(Irls{WelschShape}) === table                       # the registry reads that table
        @test Irls{WelschShape}().max_iter == 25                             # an omitted field is its default
        @test Irls{WelschShape}(max_iter = 7).max_iter == 7                  # a keyword overrides it
        @test lo_construct(Irls{WelschShape}; defaults = Dict{String, Any}("irls" => Dict{String, Any}("max_iter" => 3))).max_iter == 3
    end

    @testset "the registries: STRATEGY_TAGS and LOSS_TAGS round trip" begin
        @test STRATEGY_TAGS["irls"] === Irls && STRATEGY_TAGS["triggs"] === Triggs
        @test tag_of(STRATEGY_TAGS, Irls) == "irls" && tag_of(STRATEGY_TAGS, Triggs) == "triggs"
        @test lo_tag(Irls{TLSShape}()) == "irls" && lo_tag(Triggs{WelschShape}(efficiency_constant = 2.5)) == "triggs"
        @test_throws ArgumentError STRATEGY_TAGS["no_such_strategy"]
        for K in (GaussianShape, WelschShape, GemanMcclureShape, CauchyShape, SmoothTLSShape, TLSShape, HuberShape, BarronShape)
            @test LOSS_TAGS[tag_of(LOSS_TAGS, K)] === K                      # tag → shape → tag
        end
        @test LOSS_TAGS["geman_mcclure"] === GemanMcclureShape && tag_of(LOSS_TAGS, SmoothTLSShape) == "smooth_tls"
        @test_throws ArgumentError LOSS_TAGS["no_such_loss"]
    end

    @testset "construction and inference" begin
        lo = @inferred Irls{WelschShape}(25, EfficiencyConstant(1.5))       # the positional constructors infer
        @test lo isa Irls{WelschShape, EfficiencyConstant{Float64}}
        @test (@inferred Triggs{CauchyShape}(EfficiencyConstant(2.5))) isa Triggs{CauchyShape, EfficiencyConstant{Float64}}
        @test (@inferred Irls{GaussianShape}(25, nothing)) isa Irls{GaussianShape, Nothing}
        # the keyword constructors (`lo_construct`, through a Dict of keywords) build the same concrete types
        @test Irls{WelschShape}(efficiency_constant = 1.5) === lo
        @test typeof(Triggs{CauchyShape}(efficiency_constant = 2.5)) === Triggs{CauchyShape, EfficiencyConstant{Float64}}
        @test strategy_loss(lo) === WelschShape && strategy_loss(Triggs{HuberShape}) === HuberShape
        @test optimization_path(lo) === :irls && optimization_path(Triggs{TLSShape}(efficiency_constant = 2.5)) === :triggs
        # refusals at construction: an unknown keyword, both widths, a shape that carries its α
        @test_throws r"no keyword `iters`" Irls{WelschShape}(iters = 3)
        @test_throws r"not both" Irls{WelschShape}(efficiency_constant = 1.5, loss_width = 2.0)
        @test_throws r"zero-argument shape tag" Irls{BarronShape}()
        @test_throws ArgumentError Triggs{BarronShape}(efficiency_constant = 2.5)
        @test_throws r"plain least squares" Irls{GaussianShape}(efficiency_constant = 1.5)
    end

    @testset "width modes: efficiency_constant, loss_width, none" begin
        # Irls: a multiple of the consensus scale, an absolute width, or none (follow the scorer)
        anchored = Irls{WelschShape}(efficiency_constant = 1.5)
        @test lo_width(anchored) == EfficiencyConstant(1.5) && scale_interface(anchored) == ConsensusAnchored()
        absolute = Irls{WelschShape}(loss_width = 2.0)
        @test lo_width(absolute) == LossWidth(2.0) && scale_interface(absolute) == FixedScale()
        follows = Irls{WelschShape}()
        @test lo_width(follows) === nothing && scale_interface(follows) == FollowsScorer()
        @test scale_interface(Irls{GaussianShape}()) == NoScale()            # least squares consumes no width
        @test with_width(follows, EfficiencyConstant(1.5)) === anchored
        @test with_width(anchored, nothing) === follows
        @test_throws ArgumentError with_width(anchored, 1.5)                 # a bare number is no width mode
        @test_throws ArgumentError EfficiencyConstant(0.0)
        @test_throws ArgumentError EfficiencyConstant(-1.0)
    end

    # Moved from VisualGeometryToolkitEstimation's triggs_tests (the strategy's identity is Core's)
    @testset "Triggs: identity, the required multiple, fixed_width's freeze" begin
        lo = Triggs{CauchyShape}(efficiency_constant = 2.5)
        @test lo_width(lo) == EfficiencyConstant(2.5) && optimization_path(lo) == :triggs && scale_interface(lo) == ConsensusAnchored()
        # its own path: the Loss-Path label and the path's marker, beside IRLS's
        @test endswith(lo_label(lo), "-Triggs") && lo_marker(lo) === :square
        @test with_width(lo, EfficiencyConstant(3.0)) == Triggs{CauchyShape}(efficiency_constant = 3.0)
        # the width is always an efficiency constant; none stated (a refine's efficiency_constant(K, d), a RANSAC's
        # scorer boundary) is `nothing`
        @test Triggs{CauchyShape}().width === nothing
        @test_throws ArgumentError Triggs{CauchyShape}(loss_width = 2.0)
        @test Triggs{GaussianShape}().width === nothing                      # the plain chart LM: no width
        # fixed_width's freeze is the one absolute width it takes: the consensus
        # scale held at s2, the anchored strategy's loss, and back
        frozen = with_width(lo, fixed_width(lo, 4.0))
        @test frozen isa Triggs{CauchyShape} && lo_width(frozen) == fixed_width(lo, 4.0)
        @test lo_width(frozen) isa LossWidth && scale_interface(frozen) == FixedScale()
        @test with_width(frozen, lo_width(lo)) == lo
        @test_throws ArgumentError with_width(Triggs{GaussianShape}(), fixed_width(Irls{CauchyShape}(efficiency_constant = 2.5), 4.0))
    end
end

# A strategy with no width follows the scorer (`loss_at`): a shape with a decision width at the scorer's decision point,
# unchanged; one without (Huber) at the table's efficiency_constant(K, d) times the consensus scale (the owner's
# ruling, 2026-11-12); a shape with neither is refused by name, never a MethodError.
@testitem "strategies: a no-width strategy follows the scorer" begin
    using Test
    struct NeitherShape <: VisualGeometryToolkitCore.AbstractLossShape end   # no decision width, no efficiency constant
    # A no-width strategy of a shape that carries a parameter (LODR's BarronHQ, whose lift arm calls
    # `loss_at(lo, BarronShape(lo.α), cs, scoring, buf)`): Irls/Triggs take zero-argument shape tags only.
    struct ScorerFollowing <: VisualGeometryToolkitCore.AbstractOptimizationStrategy end
    VisualGeometryToolkitCore.lo_width(::ScorerFollowing) = nothing
    using VisualGeometryToolkitCore: Irls, Triggs, loss_at, ConsensusSet, MsacScoring, HuberShape, WelschShape, TLSShape,
        CauchyShape, BarronShape, sq_loss_width, efficiency_constant, decision_width, decision_boundary
    cs = ConsensusSet(; count = 50, s2 = 0.25, epoch = 0)          # ŝ = 0.5
    sc = MsacScoring(2; threshold = 3.0)                            # codimension 2, a 3 px threshold
    for lo in (Irls{HuberShape}(), Triggs{HuberShape}())
        loss = @inferred loss_at(lo, HuberShape(), cs, sc, nothing)
        @test sqrt(sq_loss_width(loss)) ≈ efficiency_constant(HuberShape, 2) * 0.5
    end
    # the decision-width shapes: the scorer's decision point, as before
    for K in (WelschShape, CauchyShape, TLSShape)
        loss = loss_at(Irls{K}(), K(), cs, sc, nothing)
        @test sqrt(sq_loss_width(loss)) ≈ decision_width(K(), decision_boundary(sc, nothing))
    end
    # Barron has a decision width (its half-weight point): followed as before, at every α, from a no-width strategy
    # called as LODR's Barron lift arms call it
    for α in (-2.0, 0.0, 1.0)
        loss = @inferred loss_at(ScorerFollowing(), BarronShape(α), cs, sc, nothing)
        @test sqrt(sq_loss_width(loss)) ≈ decision_width(BarronShape(α), decision_boundary(sc, nothing))
        @test decision_boundary(BarronShape(α), sqrt(sq_loss_width(loss))) ≈ 3.0
    end
    # every concrete shape with a decision_width method follows the scorer — the trait is read off the methods, so no
    # hand-kept list can drop one; Huber (none) runs at its efficiency constant
    shapes = (WelschShape(), VisualGeometryToolkitCore.GemanMcclureShape(), CauchyShape(), VisualGeometryToolkitCore.SmoothTLSShape(),
              TLSShape(), BarronShape(0.5), HuberShape())
    for sh in shapes
        has = hasmethod(decision_width, Tuple{typeof(sh), Float64})
        @test VisualGeometryToolkitCore._has_decision_width(sh) == has
        has && @test sqrt(sq_loss_width(loss_at(ScorerFollowing(), sh, cs, sc, nothing))) ≈
                     decision_width(sh, decision_boundary(sc, nothing))
    end
    @test count(sh -> VisualGeometryToolkitCore._has_decision_width(sh), shapes) == 6
    # a shape with neither a decision width nor an efficiency constant (a test-local one): refused, naming it and the
    # remedy, never a MethodError
    e = try
        loss_at(ScorerFollowing(), NeitherShape(), cs, sc, nothing); nothing
    catch err
        err
    end
    @test e isa ArgumentError && occursin("NeitherShape", e.msg) && occursin("efficiency_constant", e.msg)
end

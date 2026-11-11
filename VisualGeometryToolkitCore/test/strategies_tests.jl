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
        table = VgtkCore._STRATEGY_DEFAULTS["optimization_strategy"]
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

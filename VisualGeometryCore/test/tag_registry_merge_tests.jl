# =============================================================================
# merge(::TagRegistry, ::TagRegistry...) (quality/tag_registry.jl): one family's
# tags across packages — the core's and a downstream package's.
# =============================================================================
@testitem "tag registry merge" begin
    using Test
    using VisualGeometryCore: TagRegistry, tag_of, tag_family, tag_paths, retired_tags
    abstract type Lo end
    struct KeepModel <: Lo end
    struct WeightedRefine{N} <: Lo end
    struct Downstream <: Lo end            # a downstream package's strategy
    struct TLSShape end

    err(f) = try
        f()
        nothing
    catch e
        e
    end
    core = TagRegistry(
        :model_update, "", "keep_model" => KeepModel, "weighted_refine" => WeightedRefine, "none" => nothing;
        retired = ("keep" => "keep_model",), paths = [r"^lo\.model_update$"]
    )
    down = TagRegistry(:model_update, "", "downstream" => Downstream;
                       retired = ("down" => "downstream",), paths = [r"^lo\.model_update$", r"^arms\[\d+\]\.type$"])
    m = merge(core, down)

    @testset "the union: current tags in order, retired tags, paths once each" begin
        @test collect(m) == [collect(core); "downstream" => Downstream]
        @test m["downstream"] === Downstream && m["keep_model"] === KeepModel && m["none"] === nothing
        @test retired_tags(m) == ["keep" => "keep_model", "down" => "downstream"]
        @test tag_paths(m) == [r"^lo\.model_update$", r"^arms\[\d+\]\.type$"]
        @test tag_family(m) === :model_update
        @test tag_of(m, Downstream()) == "downstream" && tag_of(m, WeightedRefine{2}) == "weighted_refine"
        # the operands are untouched; one registry merges to an equal one; three merge
        @test length(core) == 3 && !haskey(core, "downstream")
        @test collect(merge(core)) == collect(core)
        @test merge(core, down, TagRegistry(:model_update, "", "extra" => TLSShape))["extra"] === TLSShape
    end

    @testset "retired tags carried, each naming its successor" begin
        e = err(() -> m["down"])
        @test e isa ArgumentError && e.msg == "tag \"down\" was renamed \"downstream\""
        e = err(() -> m["keep"])
        @test e isa ArgumentError && e.msg == "tag \"keep\" was renamed \"keep_model\""
        e = err(() -> m["nope"])
        @test e isa ArgumentError && occursin("\"downstream\"", e.msg)
    end

    @testset "duplicates refused" begin
        # declared by two registries
        @test_throws ArgumentError merge(core, TagRegistry(:model_update, "", "keep_model" => Downstream))
        # current in one, retired in another (both orders)
        @test_throws ArgumentError merge(core, TagRegistry(:model_update, "", "keep" => Downstream))
        @test_throws ArgumentError merge(core, TagRegistry(:model_update, "", "x" => Downstream; retired = ("none" => "x",)))
    end

    @testset "families must agree" begin
        e = err(() -> merge(core, TagRegistry(:loss, "", "tls" => TLSShape)))
        @test e isa ArgumentError && occursin("different families", e.msg)
        @test_throws ArgumentError merge(core, TagRegistry(:model_update, "Shape", "x" => Downstream))   # suffix too
    end
end

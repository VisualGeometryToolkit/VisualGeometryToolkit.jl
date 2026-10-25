@testitem "lo_registry" begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# The local-optimization strategy machinery as a public, hook-driven API
# =============================================================================
#
# What this pins: this package's own strategies construct, write and read back
# through the public names exactly as through their keyword constructors and
# `local_optimization_from_toml`; a strategy family registered with
# `@lo_registry` in ANOTHER module (this test item's) gets the same keyword
# constructor, reader and writer, its hooks honoured, and every method it adds
# lands on a declared contract; an unknown loss is an `ArgumentError` that
# lists the loss registry.
# =============================================================================

using Test
using VisualGeometryToolkitCore: AbstractLocalOptimization, AbstractLossShape, TLSShape, WelschShape,
    GaussianShape, BarronShape, ScaleMultiple, LossWidth, NoScale, optimization_path, lo_label,
    loss_family, scale_interface, contract_generic_violations
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: Irls, NoLocalOptimization, LO_TAGS, LOSS_TAGS, @lo_registry,
    lo_construct, lo_from_toml, lo_config_dict, config_dict, local_optimization_from_toml, lo_tag,
    lo_defaults, lo_admits_shape

# --- a downstream strategy family, registered here --------------------------
# A plain strategy with a Greek field, a width, a default read from another
# key (`lo_key`), a cross-field check (`lo_validate`) and a string scale
# multiple it resolves itself (`lo_resolve_scale_multiple`); and a
# loss-parametric family with its own admission (`lo_admits`).
struct ToyArm{W} <: AbstractLocalOptimization
    max_iter::Int
    μ::Float64
    width::W
end
struct ToyFamily{K <: AbstractLossShape, W} <: AbstractLocalOptimization
    max_iter::Int
    width::W
end
ToyFamily{K}(max_iter::Integer, width::W) where {K, W} = ToyFamily{K, W}(Int(max_iter), width)
VgtkEstimation.strategy_loss(::Type{<:ToyFamily{K}}) where {K} = K
# A wrapper family whose parameter is a strategy, named in TOML by `inner` +
# `loss` (the shape of LocalOptimizationDoneRight's `Gnc{Irls{K}}`).
struct ToyWrap{S <: AbstractLocalOptimization, W} <: AbstractLocalOptimization
    factor::Float64
    width::W
end
ToyWrap{S}(factor::Real, width::W) where {S, W} = ToyWrap{S, W}(Float64(factor), width)
VgtkEstimation.strategy_loss(::Type{<:ToyWrap{S}}) where {S} = VgtkEstimation.strategy_loss(S)

const TOY_DEFAULTS = Dict{String, Any}(
    "toy_arm" => Dict{String, Any}("toy_iters" => 4, "mu" => 0.5),
    "toy_family" => Dict{String, Any}("max_iter" => 9),
    "toy_wrap" => Dict{String, Any}("factor" => 1.4),
)
#   type        tag           path   scale      family  label        shape        parametric
const TOY_ROWS = (
    (ToyArm,    "toy_arm",    :irls, nothing,   nothing, raw"Toy",    WelschShape, false),
    (ToyFamily, "toy_family", :hq,   NoScale(), nothing, nothing,     nothing,     true),
    (ToyWrap,   "toy_wrap",   :gnc,  nothing,   nothing, nothing,     nothing,     true),
)
const TOY_TAGS = @lo_registry TOY_ROWS defaults = TOY_DEFAULTS retired = ("toy" => "toy_arm",)

VgtkEstimation.lo_key(::Type{<:ToyArm}, ::Val{:max_iter}) = "toy_iters"
VgtkEstimation.lo_validate(lo::ToyArm) =
    lo.μ > 0 ? lo : throw(ArgumentError("ToyArm: μ must be positive, got $(lo.μ)"))
VgtkEstimation.lo_resolve_scale_multiple(::Type{<:ToyArm}, ::Type{K}, v::AbstractString; class::Symbol) where {K} =
    v == "tuned" ? (class === :homography && K === WelschShape ? 3.0 : 4.0) :
    throw(ArgumentError("toy: unknown scale multiple $(repr(v))"))
VgtkEstimation.lo_admits(::Type{<:ToyFamily}, ::Type{K}) where {K} =
    K === GaussianShape ? throw(ArgumentError("ToyFamily{GaussianShape}: no")) : lo_admits_shape(ToyFamily{K}, K)

# The wrapper admits an `Irls` inner strategy only, is concrete at its inner
# strategy (resolved through the registry the table was read with), and
# writes its `inner` key.
VgtkEstimation.lo_admits(::Type{<:ToyWrap}, ::Type{S}) where {S} =
    S <: Irls ? lo_admits_shape(ToyWrap{S}, VgtkEstimation.strategy_loss(S)) :
    throw(ArgumentError("ToyWrap{$S}: the inner strategy must be an Irls"))
function VgtkEstimation.lo_concrete(::Type{ToyWrap}, d::AbstractDict, sec::AbstractString; tags)
    inner = tags[get(d, "inner", "irls")]
    C, extra = VgtkEstimation.lo_concrete(inner, d, sec; tags)
    return ToyWrap{C}, ("inner", extra...)
end
function VgtkEstimation.config_dict(lo::ToyWrap)
    d = lo_config_dict(lo)
    d["inner"] = "irls"
    return d
end

@testset "LO-strategy machinery" begin
    @testset "this package's strategies through the public API" begin
        @test lo_construct(Irls{TLSShape}) == Irls{TLSShape}()
        @test lo_construct(Irls{WelschShape}; scale_multiple = 2) == Irls{WelschShape}(scale_multiple = 2.0)
        @test lo_construct(Irls{WelschShape}; defaults = Dict("irls" => Dict("max_iter" => 3))).max_iter == 3
        @test lo_tag(Irls{TLSShape}) == "irls"
        @test lo_tag(NoLocalOptimization) == "no_local_optimization"
        @test lo_defaults(Irls) === VgtkEstimation._TOML_DEFAULTS["local_optimization"]
        # The writer, recorded before the machinery went public (byte-identical tables).
        @test lo_config_dict(Irls{TLSShape}()) == Dict{String, Any}("type" => "irls", "loss" => "tls", "max_iter" => 25)
        @test lo_config_dict(Irls{WelschShape}(scale_multiple = 2.0)) ==
            Dict{String, Any}("type" => "irls", "loss" => "welsch", "max_iter" => 25, "scale_multiple" => 2.0)
        @test lo_config_dict(Irls{WelschShape}(loss_width = 2)) ==
            Dict{String, Any}("type" => "irls", "loss" => "welsch", "max_iter" => 25, "loss_width" => 2.0)
        @test lo_config_dict(NoLocalOptimization()) == Dict{String, Any}("type" => "no_local_optimization")
        @test config_dict(Irls{TLSShape}()) == lo_config_dict(Irls{TLSShape}())

        # The terminal placement: a nested `inner` table, read through the same registry.
        term = Dict{String, Any}("type" => "terminal_lo",
                                 "inner" => Dict{String, Any}("type" => "irls", "loss" => "welsch", "scale_multiple" => 1.5))
        t = local_optimization_from_toml(term; class = :essential)
        @test t == VgtkEstimation.TerminalLO(Irls{WelschShape}(scale_multiple = 1.5))
        @test lo_tag(t) == "terminal_lo"
        @test lo_config_dict(t) == config_dict(t) ==
            Dict{String, Any}("type" => "terminal_lo", "inner" => lo_config_dict(Irls{WelschShape}(scale_multiple = 1.5)))
        @test local_optimization_from_toml(config_dict(t); class = :essential) == t
        @test VgtkEstimation.TerminalLO(; inner = Irls{TLSShape}()) == VgtkEstimation.TerminalLO(Irls{TLSShape}())
        # The config-tag gate reads the nested strategy's tag too, at any depth.
        mktempdir() do dir
            write(joinpath(dir, "nested.toml"), """
                [ransac.local_optimization]
                type = "terminal_lo"
                [ransac.local_optimization.inner]
                type = "none"
                [other.local_optimization]
                type = "terminal_lo"
                [other.local_optimization.inner]
                type = "terminal_lo"
                [other.local_optimization.inner.inner]
                type = "irls_typo"
                """)
            found = VgtkCore.config_tag_violations([LO_TAGS], [dir])
            @test length(found) == 2
            @test any(f -> occursin("ransac.local_optimization.inner.type", f) && occursin("no_local_optimization", f), found)
            @test any(f -> occursin("other.local_optimization.inner.inner.type", f) && occursin("irls_typo", f), found)
        end
        @test_throws ArgumentError local_optimization_from_toml(Dict("type" => "terminal_lo"); class = :essential)
        @test_throws ArgumentError local_optimization_from_toml(
            Dict("type" => "terminal_lo", "inner" => "irls"); class = :essential)
        @test_throws ArgumentError local_optimization_from_toml(
            Dict("type" => "terminal_lo", "inner" => Dict("type" => "irls", "loss" => "tls"), "max_iter" => 3); class = :essential)
        # The reader, called on the registry entry directly.
        @test lo_from_toml(Irls, Dict("type" => "irls", "loss" => "huber", "max_iter" => 7); class = :homography) ==
            Irls{VgtkCore.HuberShape}(max_iter = 7)
        @test lo_from_toml(NoLocalOptimization, Dict("type" => "no_local_optimization"); class = :homography) ==
            NoLocalOptimization()
        # The admission check the parametric families share.
        @test_throws ArgumentError Irls{BarronShape}()
        @test_throws ArgumentError lo_admits_shape(Irls{BarronShape}, BarronShape)
        @test lo_admits_shape(Irls{TLSShape}, TLSShape) === nothing
        # Least squares takes no scale multiple, by keyword or by key.
        @test_throws ArgumentError Irls{GaussianShape}(scale_multiple = 1.0)
        @test_throws ArgumentError local_optimization_from_toml(
            Dict("type" => "irls", "loss" => "gaussian", "scale_multiple" => 1.0); class = :homography)
        # The core resolves no string scale multiple.
        @test_throws ArgumentError local_optimization_from_toml(
            Dict("type" => "irls", "loss" => "tls", "scale_multiple" => "cross_validated"); class = :homography)
    end

    @testset "an unknown loss is refused, naming every LOSS_TAGS key" begin
        msg = try
            local_optimization_from_toml(Dict("type" => "irls", "loss" => "welch"); class = :homography)
            ""
        catch e
            e isa ArgumentError ? sprint(showerror, e) : "not an ArgumentError: $e"
        end
        @test occursin("unknown loss \"welch\"", msg)
        for tag in keys(LOSS_TAGS)
            @test occursin(repr(tag), msg)
        end
        @test_throws ArgumentError lo_from_toml(Irls, Dict("type" => "irls", "loss" => "welch"); class = :homography)
    end

    @testset "a strategy registered downstream with @lo_registry" begin
        # The registry and the row facts.
        @test TOY_TAGS["toy_arm"] === ToyArm
        @test TOY_TAGS["toy_family"] === ToyFamily
        @test_throws ArgumentError TOY_TAGS["toy"]          # retired, naming its successor
        @test lo_tag(ToyFamily{TLSShape}) == "toy_family"
        @test lo_defaults(ToyArm) === TOY_DEFAULTS
        @test optimization_path(ToyArm(; μ = 1.0)) === :irls
        @test lo_label(ToyArm(; μ = 1.0)) == "Toy"
        @test loss_family(ToyArm(; μ = 1.0)) === loss_family(WelschShape)   # the shape column
        @test scale_interface(ToyFamily{TLSShape}()) === NoScale()
        @test VgtkCore.toml_skip_fields(ToyArm) == (:width,)

        # The keyword constructor: defaults from the owner's table (one read
        # from another key), the width as its two keywords, the hooks honoured.
        a = ToyArm()
        @test (a.max_iter, a.μ) == (4, 0.5)
        @test a.width === nothing
        @test ToyArm(; scale_multiple = 2).width === ScaleMultiple(2.0)
        @test ToyArm(; loss_width = 2).width isa LossWidth
        @test_throws ArgumentError ToyArm(; μ = -1.0)                         # lo_validate
        @test_throws ArgumentError ToyArm(; scale_multiple = 1, loss_width = 1)
        @test_throws ArgumentError ToyArm(; iters = 3)                        # unknown keyword
        @test ToyFamily{TLSShape}().max_iter == 9
        @test_throws ArgumentError ToyFamily{GaussianShape}()                 # lo_admits
        @test_throws ArgumentError ToyFamily{BarronShape}()                   # lo_admits_shape

        # The reader through the merged registry, and the writer back.
        tags = merge(LO_TAGS, TOY_TAGS)
        load(d; class = :homography) = local_optimization_from_toml(d; class, tags)
        t = load(Dict("type" => "toy_arm", "mu" => 0.25, "scale_multiple" => "tuned"))
        @test t == ToyArm(; μ = 0.25, scale_multiple = 3.0)                   # lo_resolve_scale_multiple at its shape
        @test load(Dict("type" => "toy_arm", "scale_multiple" => "tuned"); class = :essential).width === ScaleMultiple(4.0)
        @test load(Dict("type" => "toy_arm", "scale_multiple" => 1.5)).width === ScaleMultiple(1.5)
        @test lo_config_dict(t) == Dict{String, Any}("type" => "toy_arm", "max_iter" => 4, "mu" => 0.25, "scale_multiple" => 3.0)
        @test load(lo_config_dict(t)) == t
        f = load(Dict("type" => "toy_family", "loss" => "welsch", "loss_width" => 2.0))
        @test f == ToyFamily{WelschShape}(; loss_width = 2.0)
        @test config_dict(f) == Dict{String, Any}("type" => "toy_family", "loss" => "welsch", "max_iter" => 9, "loss_width" => 2.0)
        @test load(config_dict(f)) == f
        @test_throws ArgumentError load(Dict("type" => "toy_family"))                       # no loss
        @test_throws ArgumentError load(Dict("type" => "toy_family", "loss" => "nope"))     # unknown loss
        @test_throws ArgumentError load(Dict("type" => "toy_arm", "mu" => 1.0, "nu" => 2))  # unknown key
        @test_throws ArgumentError load(Dict("type" => "toy_arm", "mu" => -1.0))            # lo_validate

        # A wrapper family: its parameter is admitted, its table names the inner strategy.
        @test ToyWrap{Irls{WelschShape}}().factor == 1.4
        @test_throws ArgumentError ToyWrap{NoLocalOptimization}()
        @test_throws ArgumentError ToyWrap{Irls{BarronShape}}()
        w = load(Dict("type" => "toy_wrap", "inner" => "irls", "loss" => "welsch", "factor" => 2.0, "scale_multiple" => 1.5))
        @test w == ToyWrap{Irls{WelschShape}}(; factor = 2.0, scale_multiple = 1.5)
        @test config_dict(w) == Dict{String, Any}("type" => "toy_wrap", "inner" => "irls", "loss" => "welsch",
                                                  "factor" => 2.0, "scale_multiple" => 1.5)
        @test load(config_dict(w)) == w
        @test_throws ArgumentError load(Dict("type" => "toy_wrap", "loss" => "welsch", "max_iter" => 3))   # unknown key

        # Every method the registration added lands on a declared contract.
        @test isempty(contract_generic_violations(@__MODULE__))
    end
end
end

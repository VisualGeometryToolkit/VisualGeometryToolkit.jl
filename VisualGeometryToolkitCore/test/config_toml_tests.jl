@testitem "config_toml" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using TOML: TOML
    using VisualGeometryToolkitCore: present_kwargs, check_keys, toml_skip_fields, toml_key, toml_value,
        toml_field, toml_table, allowed_keys, convert_field, toml_default, write_toml
    import VisualGeometryToolkitCore

    # A config type with every non-mechanical case the packages hit: a renamed
    # key, a skipped field, a nested struct, a sentinel, a symbol leaf.
    Base.@kwdef struct _Inner
        scale::Float64 = 1.5
        mode::Symbol = :fast
    end
    Base.@kwdef struct _Outer
        inner::_Inner = _Inner()
        μ::Float64 = 0.1
        max_n::Int = typemax(Int)
        camera::Any = nothing        # data, supplied at load — never emitted
        flag::Bool = true
    end
    VgtkCore.toml_skip_fields(::Type{<:_Outer}) = (:camera,)
    VgtkCore.toml_key(::Type{<:_Outer}, f::Symbol) = f === :μ ? "mu" : String(f)
    VgtkCore.toml_field(c::_Outer, ::Val{:inner}) = toml_table(c.inner)
    VgtkCore.toml_field(c::_Outer, ::Val{:max_n}) = c.max_n == typemax(Int) ? "unbounded" : c.max_n

    @testset "TOML config skeleton" begin
        @testset "toml_table derives from fieldnames, hooks handle the rest" begin
            d = toml_table(_Outer())
            @test Set(keys(d)) == Set(["inner", "mu", "max_n", "flag"])   # camera skipped, μ renamed
            @test d["inner"] == Dict("scale" => 1.5, "mode" => "fast")     # symbol → string
            @test d["max_n"] == "unbounded"                                 # sentinel
            @test d["flag"] === true
            # allowed keys come from the SAME walk, plus extras
            @test sort(allowed_keys(_Outer)) == sort(["inner", "mu", "max_n", "flag"])
            @test "type" in allowed_keys(_Inner, :type)
            @test toml_skip_fields(_Inner) == ()
            @test toml_key(_Inner(), :scale) == "scale"
            @test toml_value((1, :a)) == Any[1, "a"]
            @test toml_value(Dict(:k => :v)) == Dict("k" => "v")
        end
        @testset "present_kwargs — only present keys, fallback, field renaming" begin
            d = Dict("mu" => 2, "flag" => false)
            kw = present_kwargs(d, "mu" => Float64, "flag" => Bool, "max_n" => Int; field_of = k -> k == "mu" ? :μ : Symbol(k))
            @test kw == [:μ => 2.0, :flag => false]           # max_n absent → not emitted
            kw = present_kwargs(d, "max_n" => Int; fallback = Dict("max_n" => 7))
            @test kw == [:max_n => 7]
            @test convert_field(Symbol, "fast") === :fast
            @test convert_field(Float64, 1) === 1.0
        end
        @testset "check_keys refuses typos, tolerates retired keys" begin
            @test check_keys(Dict("scale" => 1), allowed_keys(_Inner), "inner") === nothing
            @test_throws ArgumentError check_keys(Dict("scael" => 1), allowed_keys(_Inner), "inner")
            err = try
                check_keys(Dict("scael" => 1), ["scale"], "inner"); nothing
            catch e
                e
            end
            @test occursin("scael", sprint(showerror, err)) && occursin("scale", sprint(showerror, err))
            # retired is (section, key)-scoped
            @test_logs (:warn, r"retired") check_keys(Dict("init" => 1), ["scale"], "inner"; retired = Set([("inner", "init")]))
            @test_throws ArgumentError check_keys(Dict("init" => 1), ["scale"], "other"; retired = Set([("inner", "init")]))
        end
        @testset "toml_default — dotted path, loud on a missing key" begin
            defaults = Dict("ransac" => Dict("k" => 3, "lo" => Dict("iters" => 5)), "top" => 1)
            @test toml_default(defaults, "ransac", "k") == 3
            @test toml_default(defaults, "ransac.lo", "iters") == 5
            @test toml_default(defaults, "", "top") == 1
            @test_throws KeyError toml_default(defaults, "ransac", "nope")
            @test_throws KeyError toml_default(defaults, "nope", "k")
        end
        @testset "write_toml round-trips through TOML.parsefile" begin
            path = joinpath(mktempdir(), "sub", "c.toml")
            body = Dict("outer" => toml_table(_Outer()), "experiment" => Dict("name" => "x"))
            @test write_toml(path, body; header = "# generated") == path
            @test startswith(read(path, String), "# generated\n")
            back = TOML.parsefile(path)
            @test back["outer"]["mu"] == 0.1 && back["outer"]["inner"]["mode"] == "fast"
            @test back["experiment"]["name"] == "x"
        end
    end
end

@testitem "config_versions: migrate_config and Core's own CONFIG_VERSIONS" begin
    using Test, TOML
    using VisualGeometryToolkitCore: ConfigVersions, ConfigMigration, migrate_config, current_config_version,
        CONFIG_VERSION_KEY, CONFIG_VERSIONS, parse_defaults

    # Core's converter: version 2 (the LM stops in [levenberg_marquardt.stops], the device LM's in
    # [static_levenberg_marquardt.stops]); the stored 0.13 file (no key, [lm]) reads as the shipped one
    @test current_config_version(CONFIG_VERSIONS) == 2
    path = joinpath(pkgdir(VisualGeometryToolkitCore), "configs", "default.toml")
    shipped = TOML.parsefile(path)
    @test shipped[CONFIG_VERSION_KEY] == 2
    @test parse_defaults(path; versions = CONFIG_VERSIONS) == shipped
    stored = TOML.parsefile(joinpath(@__DIR__, "fixtures", "stored_configs", "core_default_v1.toml"))
    @test migrate_config(stored, CONFIG_VERSIONS) == shipped
    @test_throws ArgumentError migrate_config(Dict{String, Any}(CONFIG_VERSION_KEY => 3), CONFIG_VERSIONS)
    # a key moves with its value (the device LM's keys out of [lm])
    @test migrate_config(stored, CONFIG_VERSIONS)["static_levenberg_marquardt"]["stops"]["max_iter"] == stored["lm"]["device_max_iter"]
    @test_throws ArgumentError migrate_config(Dict{String, Any}(CONFIG_VERSION_KEY => 2, "lm" => Dict{String, Any}()), CONFIG_VERSIONS)

    # a move that empties its parent table takes the parent with it ([schemes.a] → [a] leaves no [schemes])
    vp = ConfigVersions("Pkg", ConfigMigration(; tables = ["schemes"], renames = ["schemes.a" => "a"]); tables = ["a"])
    @test migrate_config(Dict{String, Any}("schemes" => Dict{String, Any}("a" => Dict{String, Any}("k" => 1))), vp) ==
          Dict{String, Any}(CONFIG_VERSION_KEY => 2, "a" => Dict{String, Any}("k" => 1))
    vq = ConfigVersions("Pkg", ConfigMigration(; tables = ["s"], renames = ["s.x.a" => "a"]); tables = ["a", "s"])
    @test migrate_config(Dict{String, Any}("s" => Dict{String, Any}("x" => Dict{String, Any}("a" => 1), "y" => 2)), vq) ==
          Dict{String, Any}(CONFIG_VERSION_KEY => 2, "a" => 1, "s" => Dict{String, Any}("y" => 2))   # a parent still holding keys stays

    # the mechanism over two migrations: renames move whole tables (into new parents), versions chain, refusals name the key
    v = ConfigVersions("Pkg",
        ConfigMigration(; tables = ["a", "b"], renames = ["a" => "x.a"]),
        ConfigMigration(; tables = ["x", "b"], renames = ["b" => "c"]); tables = ["x", "c"])
    @test current_config_version(v) == 3
    d1 = Dict{String, Any}("a" => Dict{String, Any}("k" => 1, "t" => Dict{String, Any}("u" => 2)), "b" => Dict{String, Any}("m" => 3))
    @test migrate_config(d1, v) == Dict{String, Any}(CONFIG_VERSION_KEY => 3,
        "x" => Dict{String, Any}("a" => Dict{String, Any}("k" => 1, "t" => Dict{String, Any}("u" => 2))), "c" => Dict{String, Any}("m" => 3))
    @test haskey(d1, "a")                                                     # the input untouched
    d2 = Dict{String, Any}(CONFIG_VERSION_KEY => 2, "b" => Dict{String, Any}())
    @test migrate_config(d2, v) == Dict{String, Any}(CONFIG_VERSION_KEY => 3, "c" => Dict{String, Any}())
    msg(d) = try
        migrate_config(d, v; source = "f.toml"); ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("[q] is not a table of Pkg's configuration version 1", msg(Dict{String, Any}("q" => Dict{String, Any}())))
    @test occursin("[b] is Pkg's configuration version 2's name; version 3 calls it [c]",
        msg(Dict{String, Any}(CONFIG_VERSION_KEY => 3, "b" => Dict{String, Any}())))
    @test occursin("newer", msg(Dict{String, Any}(CONFIG_VERSION_KEY => 4)))
    @test occursin("positive integer", msg(Dict{String, Any}(CONFIG_VERSION_KEY => 0)))
    @test occursin("[q] is not a table of Pkg's configuration version 3", msg(Dict{String, Any}(CONFIG_VERSION_KEY => 3, "q" => Dict{String, Any}())))
    clash = ConfigVersions("Pkg", ConfigMigration(; tables = ["a", "b"], renames = ["a" => "b"]); tables = ["b"])
    e = try
        migrate_config(Dict{String, Any}("a" => Dict{String, Any}(), "b" => Dict{String, Any}()), clash); nothing
    catch err
        err
    end
    @test e isa ArgumentError && occursin("[a] moves to [b]", e.msg) && occursin("already there", e.msg)
end

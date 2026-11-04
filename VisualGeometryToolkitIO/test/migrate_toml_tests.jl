# =============================================================================
# vgio_migrate on a TOML record: the schema owner's step, the generic file form
# with its verified read-back
# =============================================================================

@testitem "migrate_toml" begin
    import VisualGeometryToolkitIO as VgtkIO
    using TOML, Dates

    VgtkIO.migrate_schema(::Val{Symbol("vgiotest/config/v1")}, t::Dict{String, Any}) =
        merge(Dict{String, Any}(k => v for (k, v) in t if k != "old_name"),
            Dict{String, Any}("schema" => "vgiotest/config/v2", "name" => t["old_name"]))

    fixture = joinpath(@__DIR__, "fixtures", "migrate", "config_v1.toml")
    orig = read(fixture)
    mktempdir() do dir
        tmp = joinpath(dir, "out.toml")
        r = @test_logs (:warn, r"comments") VgtkIO.vgio_migrate(fixture; out = tmp)
        @test r.comments_dropped
        @test r.from == "vgiotest/config/v1" && r.to == "vgiotest/config/v2"
        @test read(fixture) == orig
        a, b = TOML.parsefile(fixture), TOML.parsefile(tmp)
        @test b["schema"] == "vgiotest/config/v2"
        @test b["name"] == "run-7" && !haskey(b, "old_name")
        for k in ("count", "scale", "ratio", "taken", "limits", "solver", "boards")
            @test b[k] == a[k]
            @test typeof(b[k]) === typeof(a[k])
        end
        @test b["count"] isa Int && b["scale"] isa Float64 && b["taken"] isa Date
        @test b["limits"]["lo"] isa Int && b["limits"]["hi"] isa Float64
        @test b["boards"][1]["size"] isa Float64 && b["boards"][2]["id"] isa Int
        @test b["solver"]["inner"]["names"] == ["a", "b"]

        # an existing out is refused
        @test_throws ArgumentError VgtkIO.vgio_migrate(fixture; out = tmp)

        # a dry run writes nothing
        none = joinpath(dir, "none.toml")
        d = @test_logs (:warn, r"comments") VgtkIO.vgio_migrate(fixture; out = none, dry_run = true)
        @test d.out === nothing && !ispath(none)
        @test sort(readdir(dir)) == ["out.toml"]

        # in place
        inplace = joinpath(dir, "cfg.toml")
        cp(fixture, inplace)
        @test_logs (:warn, r"comments") VgtkIO.vgio_migrate(inplace; out = inplace)
        @test TOML.parsefile(inplace)["schema"] == "vgiotest/config/v2"

        # no comments, no flag
        plain = joinpath(dir, "plain.toml")
        write(plain, "schema = \"vgiotest/config/v1\"\nold_name = \"x\"\n")
        r2 = @test_logs (:warn, r"comments") VgtkIO.vgio_migrate(plain; out = joinpath(dir, "plain2.toml"))
        @test !r2.comments_dropped

        # no schema
        noschema = joinpath(dir, "ns.toml")
        write(noschema, "a = 1\n")
        @test_throws ArgumentError("$(noschema): not a vgio file (no schema)") VgtkIO.vgio_migrate(noschema)

        # the default out
        write(joinpath(dir, "c.toml"), read(fixture))
        @test_logs (:warn, r"comments") VgtkIO.vgio_migrate(joinpath(dir, "c.toml"))
        @test isfile(joinpath(dir, "c.migrated.toml"))
    end
end

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

        # no comments: no flag, no warning
        plain = joinpath(dir, "plain.toml")
        write(plain, "schema = \"vgiotest/config/v1\"\nold_name = \"x\"\n")
        r2 = @test_logs min_level = Base.CoreLogging.Warn VgtkIO.vgio_migrate(plain; out = joinpath(dir, "plain2.toml"))
        @test !r2.comments_dropped

        # a trailing comment only: flagged and warned
        trailing = joinpath(dir, "trailing.toml")
        write(trailing, "schema = \"vgiotest/config/v1\"\nold_name = \"x\"  # note\n")
        r3 = @test_logs (:warn, r"comments") VgtkIO.vgio_migrate(trailing; out = joinpath(dir, "trailing2.toml"))
        @test r3.comments_dropped

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

@testitem "migrate_toml refuses what TOML.print cannot reproduce" begin
    import VisualGeometryToolkitIO as VgtkIO
    using TOML

    # a Float32 prints, and reads back as a Float64 (a different type); a Symbol makes TOML.print throw
    VgtkIO.migrate_schema(::Val{Symbol("vgiotest/lossy32/v1")}, t::Dict{String, Any}) =
        Dict{String, Any}("schema" => "vgiotest/lossy32/v2", "x" => Float32(0.5))
    VgtkIO.migrate_schema(::Val{Symbol("vgiotest/lossysym/v1")}, t::Dict{String, Any}) =
        Dict{String, Any}("schema" => "vgiotest/lossysym/v2", "x" => :sym)

    mktempdir() do dir
        for kind in ("lossy32", "lossysym")
            src = joinpath(dir, "$(kind).toml")
            write(src, "schema = \"vgiotest/$(kind)/v1\"\n")
            before = sort(readdir(dir))
            out = joinpath(dir, "$(kind).out.toml")
            for kw in ((; out), (; out, dry_run = true), (; out = src))
                e = try
                    Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                        VgtkIO.vgio_migrate(src; kw...)
                    end
                    nothing
                catch err
                    err
                end
                @test e isa ArgumentError
                @test occursin(src, e.msg)
                @test sort(readdir(dir)) == before      # no out, no staging directory left
                @test read(src, String) == "schema = \"vgiotest/$(kind)/v1\"\n"
            end
        end
    end
end

@testitem "migrate_toml: a malformed file and a step that drops the schema are ArgumentErrors naming what is wrong" begin
    import VisualGeometryToolkitIO as VgtkIO

    VgtkIO.migrate_schema(::Val{Symbol("vgiotest/noschema/v1")}, t::Dict{String, Any}) = Dict{String, Any}("x" => 1)

    mktempdir() do dir
        bad = joinpath(dir, "bad.toml")
        write(bad, "schema = \"vgiotest/config/v1\"\nold_name = = 3\n")
        e = try VgtkIO.vgio_migrate(bad; out = joinpath(dir, "o.toml")) catch err; err end
        @test e isa ArgumentError && occursin(bad, e.msg) && occursin("not valid TOML", e.msg)

        dropped = joinpath(dir, "dropped.toml")
        write(dropped, "schema = \"vgiotest/noschema/v1\"\n")
        out = joinpath(dir, "dropped.out.toml")
        e = try VgtkIO.vgio_migrate(dropped; out) catch err; err end
        @test e isa ArgumentError && occursin("not a later version", e.msg)
        @test !ispath(out)
    end
end

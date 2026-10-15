# configs/default.toml — THE defaults of record, read through `_config_default`
# (src/config_defaults.jl). Completeness both ways: every key the file
# declares is read somewhere in src/ (a key nothing reads is a dead knob that
# only looks authoritative), and every literal `_config_default(table, key)` in src/
# names a key the file has (a missing one throws at load or at call).

@testitem "config_defaults" begin
    using Test, TOML
    using VisualGeometryEval
    const RE = VisualGeometryEval

    src = normpath(joinpath(pkgdir(RE), "src"))
    text = join((read(joinpath(d, f), String) for (d, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl")), "\n")
    # every `_config_default(<table expr>, "<key>")` call site
    calls = [(m[1], m[2]) for m in eachmatch(r"_config_default\(([^,()]+),\s*\"([A-Za-z0-9_]+)\"\)", text)]
    @test !isempty(calls)
    literal(t) = (m = match(r"^\"([A-Za-z0-9_.]+)\"$", strip(t))) === nothing ? nothing : m[1]

    # the leaves of the file: (dotted table, key) of every non-table value
    leaves = Tuple{String,String}[]
    walk(d, path) = for (k, v) in d
        v isa AbstractDict ? walk(v, isempty(path) ? k : "$path.$k") : push!(leaves, (path, k))
    end
    walk(RE._TOML_DEFAULTS, "")

    @testset "every literal _config_default call names a declared key" begin
        for (t, k) in calls
            tbl = literal(t)
            tbl === nothing && continue          # a computed table: covered by the runtime checks below
            @test (RE._config_default(tbl, k); true)
        end
    end

    @testset "every declared key is read" begin
        literal_pairs = Set((literal(t), k) for (t, k) in calls if literal(t) !== nothing)
        computed_keys = Set(k for (t, k) in calls if literal(t) === nothing)
        for (tbl, k) in leaves
            read_literally = (tbl, k) in literal_pairs
            # a table reached through a computed path: its key is read by a
            # computed call and its table is named in src
            read_computed = k in computed_keys && occursin("\"$tbl\"", text)
            @test read_literally || read_computed
            read_literally || read_computed || @info "unread default" table = tbl key = k
        end
    end

    @testset "the readers resolve against the file" begin
        @test_throws KeyError RE._config_default("pool.heb", "no_such_key")
        @test_throws KeyError RE._config_default("no_such_table", "target")
    end
end

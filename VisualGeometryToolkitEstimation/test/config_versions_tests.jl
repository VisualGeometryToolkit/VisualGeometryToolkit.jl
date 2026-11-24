@testitem "ESTIMATION_CONFIG_VERSIONS: a stored version-1 configuration is read at version 2, table for table" begin
    using Test, TOML
    using VisualGeometryToolkitCore: migrate_config, current_config_version, CONFIG_VERSION_KEY
    using VisualGeometryToolkitEstimation: ESTIMATION_CONFIG_VERSIONS, default_config_path

    @test current_config_version(ESTIMATION_CONFIG_VERSIONS) == 2
    shipped = TOML.parsefile(default_config_path())
    @test shipped[CONFIG_VERSION_KEY] == 2
    @test migrate_config(shipped, ESTIMATION_CONFIG_VERSIONS) == shipped   # the current file reads as itself

    # the shipped 0.12 file (version 1, no version key): every value lands where version 2 keeps it, nothing else moves
    v1 = TOML.parsefile(joinpath(@__DIR__, "fixtures", "stored_configs", "estimation_default_v1.toml"))
    @test !haskey(v1, CONFIG_VERSION_KEY)
    @test migrate_config(v1, ESTIMATION_CONFIG_VERSIONS; source = "estimation_default_v1.toml") == shipped
    @test !haskey(v1, "division_camera_fit")   # the input is not modified

    # each old table to its new path, and back by the inverse renames: the move is exact both ways
    mig = only(ESTIMATION_CONFIG_VERSIONS.migrations)
    for (old, new) in mig.renames
        t = Dict{String, Any}("k" => 1.5, "sub" => Dict{String, Any}("x" => "y"))
        out = migrate_config(Dict{String, Any}(old => t), ESTIMATION_CONFIG_VERSIONS)
        node = out
        for seg in split(new, '.')
            node = node[seg]
        end
        @test node == t
        back = Dict{String, Any}(k => v for (k, v) in out if k != CONFIG_VERSION_KEY)
        for (o, n) in mig.renames   # the inverse: the new path's table back under its old name
            segs = split(n, '.')
            parent = foldl((d, s) -> get(d, s, nothing), segs[1:(end - 1)]; init = back)
            parent isa AbstractDict && haskey(parent, segs[end]) || continue
            back[o] = pop!(parent, segs[end])
            length(segs) > 1 && isempty(back[segs[1]]) && delete!(back, segs[1])
        end
        @test back == Dict{String, Any}(old => t)
    end
    # a written and re-read migrated file is the same configuration
    io = IOBuffer(); TOML.print(io, migrate_config(v1, ESTIMATION_CONFIG_VERSIONS)); @test TOML.parse(String(take!(io))) == shipped

    # refusals, each naming the key
    err(d) = try
        migrate_config(d, ESTIMATION_CONFIG_VERSIONS; source = "run.toml"); nothing
    catch e
        e
    end
    e = err(Dict{String, Any}("refine_widget" => Dict{String, Any}()))   # not a table of version 1
    @test e isa ArgumentError && occursin("[refine_widget] is not a table", e.msg) && occursin("version 1", e.msg)
    e = err(Dict{String, Any}(CONFIG_VERSION_KEY => 2, "fit_rotation" => Dict{String, Any}()))   # an old name in a current file
    @test e isa ArgumentError && occursin("[fit_rotation]", e.msg) && occursin("[reweighted_least_squares.rotation]", e.msg)
    e = err(Dict{String, Any}(CONFIG_VERSION_KEY => 3))
    @test e isa ArgumentError && occursin("newer", e.msg)
    e = err(Dict{String, Any}(CONFIG_VERSION_KEY => "2"))
    @test e isa ArgumentError && occursin("config_version", e.msg)
    e = err(Dict{String, Any}("fit_rotation" => Dict{String, Any}(), "reweighted_least_squares" => Dict{String, Any}("rotation" => Dict{String, Any}())))
    @test e isa ArgumentError && occursin("reweighted_least_squares", e.msg)   # a version-1 file has no such table
end

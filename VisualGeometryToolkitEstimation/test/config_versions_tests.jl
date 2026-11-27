@testitem "ESTIMATION_CONFIG_VERSIONS: stored version-1 and version-2 configurations are read at version 3, table for table" begin
    using Test, TOML
    using VisualGeometryToolkitCore: migrate_config, current_config_version, CONFIG_VERSION_KEY
    using VisualGeometryToolkitEstimation: ESTIMATION_CONFIG_VERSIONS, default_config_path

    @test current_config_version(ESTIMATION_CONFIG_VERSIONS) == 3
    shipped = TOML.parsefile(default_config_path())
    @test shipped[CONFIG_VERSION_KEY] == 3
    @test migrate_config(shipped, ESTIMATION_CONFIG_VERSIONS) == shipped   # the current file reads as itself

    # the shipped files of 0.12 (version 1, no version key) and 0.13 (version 2): every value lands where version 3
    # keeps it, nothing else moves
    stored(v) = TOML.parsefile(joinpath(@__DIR__, "fixtures", "stored_configs", "estimation_default_v$v.toml"))
    v1, v2 = stored(1), stored(2)
    @test !haskey(v1, CONFIG_VERSION_KEY) && v2[CONFIG_VERSION_KEY] == 2
    @test migrate_config(v1, ESTIMATION_CONFIG_VERSIONS; source = "estimation_default_v1.toml") == shipped
    @test migrate_config(v2, ESTIMATION_CONFIG_VERSIONS; source = "estimation_default_v2.toml") == shipped
    @test !haskey(v1, "division_camera_fit") && haskey(v2, "pipelines")   # the inputs are not modified

    # a dotted path's value in a nested table, and the path's tables built around one value
    at(d, path) = foldl((t, s) -> t isa AbstractDict ? get(t, s, nothing) : nothing, split(path, '.'); init = d)
    function nested(path, x)
        segs = split(path, '.')
        return foldr((s, t) -> Dict{String, Any}(String(s) => t), segs[1:(end - 1)]; init = Dict{String, Any}(String(segs[end]) => x))
    end
    # each old path to its new one, version by version: the move is exact, and nothing else is left behind
    for (k, mig) in enumerate(ESTIMATION_CONFIG_VERSIONS.migrations), (old, new) in mig.renames
        t = Dict{String, Any}("k" => 1.5, "sub" => Dict{String, Any}("x" => "y"))
        d = nested(old, t)
        k > 1 && (d[CONFIG_VERSION_KEY] = k)
        out = migrate_config(d, ESTIMATION_CONFIG_VERSIONS)
        # a later version may move the path again: follow it
        dest = new
        for later in ESTIMATION_CONFIG_VERSIONS.migrations[(k + 1):end], (o, n) in later.renames
            dest == o && (dest = n)
        end
        @test at(out, dest) == t
        @test Set(keys(out)) == Set([CONFIG_VERSION_KEY, first(split(dest, '.'))])
    end
    # a written and re-read migrated file is the same configuration
    io = IOBuffer(); TOML.print(io, migrate_config(v2, ESTIMATION_CONFIG_VERSIONS)); @test TOML.parse(String(take!(io))) == shipped

    # refusals, each naming the key
    err(d) = try
        migrate_config(d, ESTIMATION_CONFIG_VERSIONS; source = "run.toml"); nothing
    catch e
        e
    end
    e = err(Dict{String, Any}("refine_widget" => Dict{String, Any}()))   # not a table of version 1
    @test e isa ArgumentError && occursin("[refine_widget] is not a table", e.msg) && occursin("version 1", e.msg)
    e = err(Dict{String, Any}(CONFIG_VERSION_KEY => 3, "fit_rotation" => Dict{String, Any}()))   # a version-1 name in a current file
    @test e isa ArgumentError && occursin("[fit_rotation]", e.msg) && occursin("[reweighted_least_squares.rotation]", e.msg)
    e = err(Dict{String, Any}(CONFIG_VERSION_KEY => 3, "simfitpp" => Dict{String, Any}()))   # a version-2 name in a current file
    @test e isa ArgumentError && occursin("[simfitpp]", e.msg) && occursin("[sim_fit_pp]", e.msg)
    e = err(nested("pipelines.homography", Dict{String, Any}()) |> d -> (d[CONFIG_VERSION_KEY] = 3; d))
    @test e isa ArgumentError && occursin("[pipelines", e.msg)
    e = err(Dict{String, Any}(CONFIG_VERSION_KEY => 4))
    @test e isa ArgumentError && occursin("newer", e.msg)
    e = err(Dict{String, Any}(CONFIG_VERSION_KEY => "2"))
    @test e isa ArgumentError && occursin("config_version", e.msg)
    e = err(Dict{String, Any}("fit_rotation" => Dict{String, Any}(), "reweighted_least_squares" => Dict{String, Any}("rotation" => Dict{String, Any}())))
    @test e isa ArgumentError && occursin("reweighted_least_squares", e.msg)   # a version-1 file has no such table
    # version 2's `[pipelines.hand_eye] motion_pairs` is the hand-eye problem's own default, not its Ransac's
    out = migrate_config(Dict{String, Any}(CONFIG_VERSION_KEY => 2, "pipelines" => Dict{String, Any}("hand_eye" =>
        Dict{String, Any}("motion_pairs" => "all", "scoring" => Dict{String, Any}("type" => "msac", "threshold" => 0.02)))),
        ESTIMATION_CONFIG_VERSIONS)
    @test out["hand_eye_problem"]["motion_pairs"] == "all" && out["hand_eye_problem"]["ransac"] ==
          Dict{String, Any}("scoring" => Dict{String, Any}("type" => "msac", "threshold" => 0.02))
end

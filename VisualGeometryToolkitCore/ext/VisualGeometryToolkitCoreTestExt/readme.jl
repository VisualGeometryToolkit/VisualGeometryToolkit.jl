# =============================================================================
# The README's examples, run as a reader runs them (stubs and docstrings:
# src/quality/test_helpers.jl). Every stack package's readme test calls these;
# they restated them before (the monorepo's packages, RANSACScoringDoneRight,
# LocalOptimizationDoneRight, PanoramaStitching; BlobBoards' and CaliBlobs'
# install-row runners).
# =============================================================================

function readme_examples(root::AbstractString, pages; langs = ("julia", "python", "bash"))
    out = ReadmeExample[]
    fence = Regex("^```(" * join(langs, "|") * ")\\n(.*?)^```", "ms")
    for page in pages
        text = read(joinpath(root, page), String)
        for m in eachmatch(fence, text)
            code = String(m[2])
            lines = split(code, '\n')
            illustrative = startswith(lines[1], "# Illustrative")
            header = illustrative && length(lines) > 1 ? lines[2] : lines[1]
            row = match(r"^# Install row: (.+)$", header)
            push!(out, ReadmeExample(String(page), String(m[1]), row === nothing ? "" : String(row[1]), illustrative, code))
        end
    end
    return out
end

function install_table(path::AbstractString, tasks)
    rows = Dict{String, String}()
    for line in eachline(path)
        m = match(r"^\| ([^|]+?) \| (.+?) \| [^|]+ \|$", line)
        m === nothing && continue
        m[1] in tasks || continue
        rows[String(m[1])] = String(m[2])
    end
    return rows
end

function install_packages(entry::AbstractString)
    m = match(r"`pkg> add ([^`]+)`", entry)
    return m === nothing ? String[] : String.(split(m[1]))
end

function loaded_packages(code::AbstractString)
    loaded = Set{String}()
    for m in eachmatch(r"^\s*(?:using|import)\s+([^:\n]+)"m, code)
        for part in split(replace(m[1], r"#.*" => ""), ',')   # a trailing comment is not a package
            name = first(split(strip(part), r"[.\s]"))
            isempty(name) || push!(loaded, String(name))
        end
    end
    return Set(filter(name -> !isdir(joinpath(Sys.STDLIB, name)), loaded))
end

function run_readme_child(env::AbstractString, script::AbstractString)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$env $script`
    proc = withenv("JULIA_LOAD_PATH" => "@:@stdlib", "JULIA_PROJECT" => nothing) do   # --project overrides JULIA_PROJECT
        run(ignorestatus(cmd))
    end
    return proc.exitcode
end

_package_name(root) = TOML.parsefile(joinpath(root, "Project.toml"))["name"]

function readme_environment(root::AbstractString, packages; developed = Dict{String, String}(), setup::AbstractString = "")
    local_paths = Dict{String, String}(_package_name(root) => String(root))
    for (name, path) in developed
        local_paths[String(name)] = String(path)
    end
    by_path = unique([local_paths[p] for p in packages if haskey(local_paths, p)])
    isempty(by_path) && push!(by_path, String(root))
    registered = String[p for p in packages if !haskey(local_paths, p)]
    env = mktempdir()
    # The adds do not precompile on their own (Pkg.add's automatic precompilation
    # cannot always identify an indirect dependency from the environment, Julia
    # 1.13): the environment is precompiled once, after the adds.
    write(joinpath(env, "build.jl"), """
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    using Pkg
    Pkg.develop([PackageSpec(path = p) for p in $(repr(by_path))]; io = devnull)
    isempty($(repr(registered))) || Pkg.add($(repr(registered)); io = devnull)
    $setup
    Pkg.precompile(; io = devnull)
    """)
    Test.@test run_readme_child(env, joinpath(env, "build.jl")) == 0
    return env
end

function test_readme_row(env::AbstractString, task::AbstractString, blocks; after::AbstractString = "")
    Test.@test !isempty(blocks)
    write(joinpath(env, "run.jl"), """
    using Test
    mods = Module[]
    @testset $(repr("examples of: " * task)) begin
        for (i, code) in enumerate($(repr(collect(String, blocks))))
            m = Module(Symbol("Example", i))
            push!(mods, m)
            value = include_string(m, code, "example_\$i.jl")
            if endswith(strip(last(split(strip(code), '\\n'))), "# true")
                @test value === true
            end
        end
        $after
    end
    """)
    Test.@test run_readme_child(env, joinpath(env, "run.jl")) == 0
    return nothing
end

function test_readme_example(root::AbstractString; index::AbstractString = joinpath("docs", "src", "index.md"),
                             install_pages = (), developed = Dict{String, String}(), checks::AbstractString = "")
    julia_blocks(text) = [String(m[1]) for m in eachmatch(r"^```julia\n(.*?)^```"ms, text)]
    readme = read(joinpath(root, "README.md"), String)
    example = julia_blocks(readme)
    Test.@test length(example) == 1
    Test.@test julia_blocks(read(joinpath(root, index), String)) == example       # one example, shown twice
    install = [String(m[1]) for m in eachmatch(r"^pkg> add (.+)$"m, readme)]
    Test.@test length(install) == 1
    (length(example) == 1 && length(install) == 1) || return nothing
    line = only(install)
    packages = String.(split(line))
    Test.@test loaded_packages(only(example)) == Set(packages)       # the line adds exactly what the example loads
    for page in (index, install_pages...)
        Test.@test occursin("pkg> add $line", read(joinpath(root, page), String))
    end
    env = readme_environment(root, packages; developed)
    nclaims = count(r"#\s*true$"m, only(example))
    write(joinpath(env, "readme_example.jl"), only(example))
    write(joinpath(env, "run.jl"), """
    using Test
    readme = Module(:ReadmeSandbox)
    code = read($(repr(joinpath(env, "readme_example.jl"))), String)
    @testset "README example" begin
        # expression by expression, so an expression whose source ends `# true` is a claim
        pos = firstindex(code); nclaims = 0
        while pos <= ncodeunits(code)
            ex, next = Meta.parse(code, pos; greedy = true, raise = true)
            chunk = code[pos:prevind(code, next)]
            pos = next
            ex === nothing && continue
            value = Core.eval(readme, ex)
            if occursin(r"#\\s*true\\z", rstrip(chunk))
                nclaims += 1
                @test value === true
            end
        end
        @test nclaims == $nclaims                           # every claim was asserted
        $checks
    end
    """)
    Test.@test run_readme_child(env, joinpath(env, "run.jl")) == 0
    return nothing
end

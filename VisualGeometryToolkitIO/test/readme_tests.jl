# readme_tests.jl — every ```julia block of the README runs as written.
#
# A reviewer's first contact with the package is the README. Its one example
# once used a `pose` it never defined, and nothing caught the UndefVarError
# (General registration review, 2026-10). This item runs every ```julia block,
# in order, in one fresh module, so later blocks see what earlier ones define.
#
# The installation block is not run against the network: `Pkg` in the sandbox
# is a stub that records what `Pkg.add` was asked for. The item then checks that
# every package another block loads is installed by that block, so the
# instructions cannot fall behind the examples.

@testitem "README: every julia block runs" begin
    using VisualGeometryToolkitIO

    readme = read(joinpath(pkgdir(VisualGeometryToolkitIO), "README.md"), String)
    blocks = [String(m[1]) for m in eachmatch(r"^```julia\n(.*?)^```"ms, readme)]
    @test !isempty(blocks)

    sandbox = Module(:ReadmeSandbox)
    include_string(sandbox, """
        module Pkg
            const added = String[]
            add(name::AbstractString) = (push!(added, name); nothing)
            add(names::AbstractVector{<:AbstractString}) = (append!(added, names); nothing)
        end
        """, "Pkg stub")

    # The loaded packages, from every `using`/`import` line outside the
    # installation block.
    loaded = Set{String}()
    for (i, code) in enumerate(blocks)
        lines = split(code, '\n')
        is_install = any(l -> occursin("Pkg.add", l), lines)
        if !is_install
            for l in lines
                m = match(r"^\s*(?:using|import)\s+(.*)$", l)
                m === nothing && continue
                for item in split(m[1], ',')
                    name = first(split(strip(item), (' ', ':', '.')))
                    isempty(name) || push!(loaded, String(name))
                end
            end
        end
        # `using Pkg` would bring in the real package manager; the stub stands in.
        runnable = join((l for l in lines if strip(l) != "using Pkg"), '\n')
        @test (include_string(sandbox, runnable, "README.md (julia block $(i))"); true)
    end

    added = Set(Core.eval(sandbox, :(Pkg.added)))
    @test !isempty(added)
    stdlib = Set(["Pkg", "LinearAlgebra", "Random", "Statistics", "Dates", "Test"])
    @test setdiff(loaded, added, stdlib) == Set{String}()
end

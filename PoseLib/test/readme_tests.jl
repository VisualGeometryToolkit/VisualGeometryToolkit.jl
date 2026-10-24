# readme_tests.jl: every julia block of the README runs as written. A reader's
# first contact with the package is those blocks, and a sibling package's
# Quick Start once called a name that was public but not exported; nothing
# caught the UndefVarError (General registration review, 2026-10-17).
#
# Each block runs in a fresh sandbox module. The Installation block is the one
# exception: it would add packages over the network, so it is checked instead
# to add every package that the other blocks load.

@testset "README" begin
    readme = read(joinpath(pkgdir(PoseLib), "README.md"), String)
    blocks = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, readme)]
    @test length(blocks) >= 2

    is_install(code) = occursin("Pkg.add", code)
    install = filter(is_install, blocks)
    @test length(install) == 1

    # The packages of every `using`/`import` line: `using A, B`, `using A: f`,
    # `import A as B` and `using A.Sub` all name the package A.
    function loaded(code)
        pkgs = Set{String}()
        for line in split(code, '\n')
            m = match(r"^\s*(?:using|import)\s+([^:]*)", line)
            m === nothing && continue
            for item in split(m[1], ',')
                name = first(split(strip(item), r"[\s.]"))
                isempty(name) || push!(pkgs, name)
            end
        end
        return pkgs
    end
    added = Set(m[1] for m in eachmatch(r"\"([A-Za-z0-9_]+)\"", only(install)))

    for (i, code) in enumerate(blocks)
        is_install(code) && continue
        @testset "block $i" begin
            # Every package the block loads is one the Installation block adds
            # (standard libraries excepted: they need no installation).
            for pkg in loaded(code)
                pkg in ("Pkg", "LinearAlgebra", "Random", "Statistics") && continue
                @test pkg in added
            end
            sandbox = Module(Symbol("ReadmeBlock", i))
            @test (include_string(sandbox, code, "README.md (block $i)"); true)
        end
    end
end

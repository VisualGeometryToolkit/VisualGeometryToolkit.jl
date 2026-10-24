# readme_tests.jl — every ```julia block of the README runs as written. A
# reviewer's first contact with the package is that file; the General registry
# review of the sibling packages (2026-10) found README blocks with undefined
# variables and imports that the installation did not add.
#
# The installation block (the one that calls `Pkg.add`) is not run: it needs
# the network. Instead, every package the other blocks import must be added
# there or be a standard library. The other blocks run in order in one sandbox
# module, as a reader would paste them into one session.

@testitem "readme" begin
    using Test, VisualGeometryToolkitEval

    readme = read(joinpath(pkgdir(VisualGeometryToolkitEval), "README.md"), String)
    blocks = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, readme)]
    install = filter(b -> occursin("Pkg.add", b), blocks)
    examples = filter(b -> !occursin("Pkg.add", b), blocks)
    @test length(install) == 1
    @test !isempty(examples)

    # The packages the examples import: the first name after `using`/`import`.
    imported = Set{String}()
    for b in examples, m in eachmatch(r"^\s*(?:using|import)\s+([A-Za-z_][A-Za-z0-9_]*)"m, b)
        push!(imported, m[1])
    end
    added = Set(m[1] for m in eachmatch(r"\"([A-Za-z_][A-Za-z0-9_]*)\"", only(install)))
    is_stdlib(name) = isdir(joinpath(Sys.STDLIB, name))
    not_added = sort!([p for p in imported if !(p in added) && !is_stdlib(p)])
    isempty(not_added) || @error "README imports packages its installation does not add" not_added
    @test isempty(not_added)

    sandbox = Module(:ReadmeExamples)
    for (i, code) in enumerate(examples)
        ran = try
            include_string(sandbox, code, "README.md (julia block $i)")
            true
        catch err
            @error "README julia block $i failed" exception = (err, catch_backtrace())
            false
        end
        @test ran
    end
end

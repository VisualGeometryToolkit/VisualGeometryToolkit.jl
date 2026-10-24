# readme_tests.jl — every ```julia block of the README runs as written. A
# reader's first contact with the package is that code, and the General
# registration review (2026-10) found that none of it ran. The blocks run in
# order in one sandbox module, so a later block may use an earlier one's names.
# The installation block is skipped: it calls Pkg.add, and the test environment
# already holds what it adds. A statement whose last line ends in `# true` is a
# claim the README makes, and is tested to be `true`.

@testitem "README julia blocks run" begin
    using Test
    readme = read(joinpath(pkgdir(VisualGeometryToolkitFeatures), "README.md"), String)
    blocks = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, readme)]
    runnable = filter(b -> !occursin("Pkg.add", b), blocks)
    @test length(blocks) >= 2           # the installation and the quick start
    @test length(runnable) >= 1
    sandbox = Module(:ReadmeSandbox)
    nclaims = 0
    for (k, code) in enumerate(runnable)
        pos = firstindex(code)
        while pos <= ncodeunits(code)
            ex, next = Meta.parse(code, pos; greedy = true, raise = true)
            chunk = code[pos:prevind(code, next)]
            pos = next
            ex === nothing && continue
            value = Core.eval(sandbox, ex)
            if occursin(r"#\s*true\z", rstrip(chunk))
                nclaims += 1
                @test value === true
                value === true || @error "README claim is not true" block = k code = strip(chunk)
            end
        end
    end
    @test nclaims >= 1
end

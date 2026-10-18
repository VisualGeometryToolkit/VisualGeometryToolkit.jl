@testitem "agent docs" begin
    # CLAUDE.md, AGENTS.md and anything tracked under .claude/ ship with the
    # package; they carry its conventions and nothing private. Release tooling
    # adds its own patterns through AGENT_DOC_PATTERNS (a file, one per line).
    using VisualGeometryToolkitCore: agent_doc_violations
    violations = agent_doc_violations(pkgdir(VisualGeometryToolkitCore); extra = get(ENV, "AGENT_DOC_PATTERNS", nothing))
    isempty(violations) || @error "agent docs carry private content" violations
    @test isempty(violations)
end

@testitem "agent docs: the check" begin
    using VisualGeometryToolkitCore: agent_doc_violations, AGENT_DOC_FORBIDDEN
    root = mktempdir()
    @test agent_doc_violations(root) == ["CLAUDE.md: missing"]
    write(joinpath(root, "CLAUDE.md"), "Conventions.\nData under /home/someone/data.\nA Widget is fine.\n")
    mkpath(joinpath(root, ".claude"))
    write(joinpath(root, ".claude", "notes.md"), "api_key = \"abcdefghijkl\"\n")
    write(joinpath(root, "README.md"), "/home/someone is not an agent doc\n")
    v = agent_doc_violations(root)
    @test length(v) == 2
    @test any(startswith("CLAUDE.md:2 matches"), v)
    @test any(startswith(joinpath(".claude", "notes.md") * ":1 matches"), v)
    # extra patterns: a regex, a collection, or a file of one per line
    @test length(agent_doc_violations(root; extra = r"widget"i)) == 3
    @test length(agent_doc_violations(root; extra = [r"Widget", r"Conventions"])) == 4
    pats = joinpath(mktempdir(), "patterns.txt")
    write(pats, "# private names\n\n(?i)widget\n")
    @test length(agent_doc_violations(root; extra = pats)) == 3
    @test length(agent_doc_violations(root; extra = "")) == 2
    @test isempty(agent_doc_violations(root; forbidden = ()))
end

@testitem "agent docs: the one copy of the conventions" begin
    # The stack conventions live once, in conventions/ (shipped with the
    # package): CLAUDE.md imports every file, the docs' Conventions page is
    # generated from them, and no CLAUDE.md carries a shared-section copy.
    using VisualGeometryToolkitCore: conventions_dir, convention_files
    root = pkgdir(VisualGeometryToolkitCore)
    dir = conventions_dir()
    @test dir == joinpath(root, "conventions")
    @test isdir(dir)
    files = convention_files()
    @test all(isfile, files)
    @test allunique(files)
    names = basename.(files)
    @test Set(filter(endswith(".md"), readdir(dir))) == Set([names; "README.md"])
    readme = read(joinpath(dir, "README.md"), String)
    @test all(n -> occursin("`$n`", readme), names)

    claude = read(joinpath(root, "CLAUDE.md"), String)
    imports = [m[1] for m in eachmatch(r"^@conventions/(\S+)$"m, claude)]
    @test imports == names                   # every file, in reading order
    @test !occursin("<!-- shared:", claude)  # no copy of the old shared blocks
    @test !occursin("Frame in = frame out.", claude)

    # The docs page is built from the same files, never kept as a third copy.
    make = read(joinpath(root, "docs", "make.jl"), String)
    @test occursin("VisualGeometryToolkitCore.convention_files()", make)
    @test occursin("\"Conventions\" => \"guide/conventions.md\"", make)
    @test "docs/src/guide/conventions.md" in strip.(readlines(joinpath(root, ".gitignore")))
end

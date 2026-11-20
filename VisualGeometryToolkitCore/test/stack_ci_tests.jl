# stack_ci_tests.jl — the CI every stack workspace calls (src/quality/stack_ci.jl), on a
# workspace built in a temporary directory: two packages (Beta depends on Alpha), a tooling
# member and a neutral directory.

@testsetup module StackCIWorkspace
export make_workspace, package_project

const MIT = "MIT License\n\nCopyright (c) the authors\n"

package_project(name, uuid, version; deps = "", compat = "") = """
    name = "$name"
    uuid = "$uuid"
    version = "$version"

    [deps]
    $deps
    [compat]
    $(compat)julia = "1.13"
    """

function write_package(root, name, uuid, version; deps = "", compat = "")
    dir = joinpath(root, name)
    mkpath(joinpath(dir, "test"))
    write(joinpath(dir, "Project.toml"), package_project(name, uuid, version; deps, compat))
    write(joinpath(dir, "LICENSE"), MIT)
    write(joinpath(dir, "test", "Project.toml"), """
        [deps]
        $name = "$uuid"

        [sources]
        $name = {path = ".."}

        [compat]
        julia = "1.13"
        """)
    return dir
end

# A git checkout holding the workspace; `settings` is its WORKSPACE.toml ("" for none).
function make_workspace(; settings = "", beta_version = "0.1.0")
    root = mktempdir()
    write(joinpath(root, "Project.toml"), """
        [workspace]
        projects = ["AlphaWidgets", "BetaWidgets", "ci"]
        """)
    isempty(settings) || write(joinpath(root, "WORKSPACE.toml"), settings)
    a = "5a1c1d3e-0b7f-4e5a-9d0c-1f2e3d4c5b01"
    b = "5a1c1d3e-0b7f-4e5a-9d0c-1f2e3d4c5b02"
    write_package(root, "AlphaWidgets", a, "0.1.0")
    write_package(root, "BetaWidgets", b, beta_version; deps = "AlphaWidgets = \"$a\"\n", compat = "AlphaWidgets = \"0.1\"\n")
    mkpath(joinpath(root, "ci"))
    write(joinpath(root, "ci", "Project.toml"), "[compat]\njulia = \"1.13\"\n")
    write(joinpath(root, "CLAUDE.md"), "Conventions.\n")
    # the stack's allow-list rows of these two packages (the `extra` of stack_dependency_violations)
    write(joinpath(root, "rows.toml"), "AlphaWidgets = []\nBetaWidgets = [\"AlphaWidgets\"]\n")
    run(`git -C $root init -q`)
    run(`git -C $root add .`)
    run(`git -C $root -c user.name=t -c user.email=t@example.org commit -q -m workspace`)
    return root
end
end

@testitem "stack_ci: a workspace's packages, settings and the packages a change affects" setup = [StackCIWorkspace] begin
    using VisualGeometryToolkitCore: workspace_packages, workspace_settings, affected_packages, stack_ci

    root = make_workspace()
    @test workspace_packages(root) == ["AlphaWidgets", "BetaWidgets"]
    s = workspace_settings(root)
    @test s.tooling == ["ci"] && isempty(s.neutral) && isempty(s.manual_merge)
    @test s.placeholder_version === nothing && s.stack_rows === nothing

    # a package and its dependents; a root file or a tooling directory: every package
    @test affected_packages(root, ["BetaWidgets/src/x.jl"]) == ["BetaWidgets"]
    @test affected_packages(root, ["AlphaWidgets/src/x.jl"]) == ["AlphaWidgets", "BetaWidgets"]
    @test affected_packages(root, ["README.md"]) == ["AlphaWidgets", "BetaWidgets"]
    @test affected_packages(root, ["BetaWidgets/a.jl", "ci/x.jl"]) == ["AlphaWidgets", "BetaWidgets"]
    @test isempty(affected_packages(root, String[]))
    # without a neutral list, site/ is a workspace-level change
    @test affected_packages(root, ["site/index.html"]) == ["AlphaWidgets", "BetaWidgets"]

    settings = """
        neutral = ["site", "docs"]
        placeholder_version = "0.0.1"
        stack_rows = "rows.toml"
        [manual_merge]
        BetaWidgets = "close to another name"
        """
    root = make_workspace(; settings)
    s = workspace_settings(root)
    @test s.neutral == ["site", "docs"] && s.placeholder_version == v"0.0.1"
    @test s.stack_rows == joinpath(root, "rows.toml")
    @test s.manual_merge == Dict("BetaWidgets" => "close to another name")
    @test isempty(affected_packages(root, ["site/index.html", "docs/design/notes.md"]))
    @test affected_packages(root, ["site/index.html", "BetaWidgets/a.jl"]) == ["BetaWidgets"]
    write(joinpath(root, "WORKSPACE.toml"), "neutrals = []\n")
    @test_throws ArgumentError workspace_settings(root)

    # the command line: JSON on io, the same list in $GITHUB_OUTPUT
    root = make_workspace()
    write(joinpath(root, "BetaWidgets", "LICENSE"), "MIT License\n\nmore\n")
    run(`git -C $root -c user.name=t -c user.email=t@example.org commit -q -am change`)
    out = IOBuffer()
    github = tempname()
    code = withenv(() -> stack_ci(["affected", "HEAD~1", "HEAD"]; root, io = out), "GITHUB_OUTPUT" => github)
    @test code == 0
    @test String(take!(out)) == "[\"BetaWidgets\"]\n"
    @test read(github, String) == "packages=[\"BetaWidgets\"]\n"
    out = IOBuffer()
    @test withenv(() -> stack_ci(["packages"]; root, io = out), "GITHUB_OUTPUT" => nothing) == 0
    @test String(take!(out)) == "[\"AlphaWidgets\",\"BetaWidgets\"]\n"
    @test_throws ArgumentError stack_ci(["plan"]; root)
    @test_throws ArgumentError stack_ci(String[]; root)
end

@testitem "stack_ci: the file stands alone for a planning job" begin
    # A CI planning job includes src/quality/stack_ci.jl by path, without loading the
    # package: standard libraries only at load time.
    script = joinpath(pkgdir(VisualGeometryToolkitCore), "src", "quality", "stack_ci.jl")
    m = Module(:StackCIByPath)
    Base.include(m, script)
    @test isdefined(m, :stack_ci) && isdefined(m, :affected_packages)
    root = mktempdir()
    write(joinpath(root, "Project.toml"), "[workspace]\nprojects = [\"AlphaWidgets\", \"ci\"]\n")
    @test Base.invokelatest(m.workspace_packages, root) == ["AlphaWidgets"]
    @test Base.invokelatest(m.affected_packages, root, ["x.md"]) == ["AlphaWidgets"]
    @test occursin("- name: AlphaWidgets", Base.invokelatest(m.codecov_config, ["AlphaWidgets"]))
end

@testitem "stack_ci: the registration rules" setup = [StackCIWorkspace] begin
    using VisualGeometryToolkitCore: registration_violations, codecov_config, workspace_packages, stack_ci

    rows = "stack_rows = \"rows.toml\"\n"
    errors_of(root; kwargs...) = registration_violations(root; kwargs...).errors
    # the General checks need the registry; every other finding is independent of it
    structural(found) = filter(e -> !occursin("General", e), found)

    root = make_workspace(; settings = rows)
    @test isempty(structural(errors_of(root)))
    @test isempty(structural(errors_of(root; initial = false)))

    # the version rule of a first registration, and its placeholder
    root = make_workspace(; settings = rows, beta_version = "0.0.1")
    @test any(==("ERROR  BetaWidgets: version 0.0.1; the first registration is 0.1.0"), errors_of(root))
    @test !any(contains("first registration"), errors_of(root; initial = false))
    root = make_workspace(; settings = rows * "placeholder_version = \"0.0.1\"\n", beta_version = "0.0.1")
    found = registration_violations(root)
    @test !any(contains("first registration is"), structural(found.errors))
    @test any(startswith("WARN   BetaWidgets: version 0.0.1 is the placeholder"), found.warnings)

    # compat: present, bounded, a sibling's series
    root = make_workspace(; settings = rows)
    project = joinpath(root, "BetaWidgets", "Project.toml")
    text = read(project, String)
    write(project, replace(text, "AlphaWidgets = \"0.1\"" => "AlphaWidgets = \"0.2\""))
    @test any(contains("stack compat AlphaWidgets = \"0.2\", expected AlphaWidgets's series \"0.1\""), errors_of(root))
    write(project, replace(text, "AlphaWidgets = \"0.1\"" => "AlphaWidgets = \"0.1.3\""))
    @test isempty(structural(errors_of(root)))                     # a floor inside the series
    write(project, replace(text, "AlphaWidgets = \"0.1\"\n" => ""))
    @test any(contains("[deps] AlphaWidgets has no [compat] entry"), errors_of(root))
    write(project, replace(text, "julia = \"1.13\"" => "julia = \">= 1.13\""))
    @test any(contains("compat julia = \">= 1.13\" is unbounded"), errors_of(root))
    write(project, text * "\n[extras]\nTest = \"8dfed614-e22c-5e08-85e1-65c5234f0b40\"\n")
    @test any(==("ERROR  BetaWidgets: registered Project.toml has [extras]"), errors_of(root))
    write(project, text)

    # the license, the test project's source, the allow-list
    rm(joinpath(root, "AlphaWidgets", "LICENSE"))
    @test any(==("ERROR  AlphaWidgets: no LICENSE in the package directory"), errors_of(root))
    root = make_workspace(; settings = rows)
    test_project = joinpath(root, "AlphaWidgets", "test", "Project.toml")
    write(test_project, replace(read(test_project, String), "{path = \"..\"}" => "{path = \"../..\"}"))
    @test any(contains("test/Project.toml: [sources] AlphaWidgets = {path = \"..\"} missing"), errors_of(root))
    root = make_workspace()                                        # no rows: the packages are not in the allow-list
    @test any(contains("stack allow-list"), errors_of(root))

    # the workspace: no tracked Manifest, a generated codecov.yml
    root = make_workspace(; settings = rows)
    write(joinpath(root, "Manifest.toml"), "")
    run(`git -C $root add Manifest.toml`)
    @test any(==("ERROR  <root>: tracked Manifests: Manifest.toml"), errors_of(root))
    root = make_workspace(; settings = rows)
    config = codecov_config(workspace_packages(root))
    @test occursin("    - name: BetaWidgets\n      paths:\n        - BetaWidgets/src/\n        - BetaWidgets/ext/\n", config)
    write(joinpath(root, "codecov.yml"), config)
    @test isempty(structural(errors_of(root)))
    write(joinpath(root, "codecov.yml"), replace(config, "BetaWidgets/ext/" => "BetaWidgets/extensions/"))
    @test any(startswith("ERROR  <root>: codecov.yml is not the workspace's codecov_config"), errors_of(root))
    out = IOBuffer()
    @test stack_ci(["codecov"]; root, io = out) == 0
    @test String(take!(out)) == config

    # the command line: findings on io, the summary on log, the exit code
    out, log = IOBuffer(), IOBuffer()
    code = stack_ci(["registration-lint", "--subsequent"]; root, io = out, log)
    @test code == 1
    @test occursin("ERROR  <root>: codecov.yml", String(take!(out)))
    @test startswith(String(take!(log)), "registration_lint: 2 packages, ")
end

@testitem "stack_ci: the agent-docs policy over a checkout" setup = [StackCIWorkspace] begin
    using VisualGeometryToolkitCore: workspace_agent_doc_violations, stack_ci

    root = make_workspace()
    @test isempty(workspace_agent_doc_violations(root; extra = nothing))
    @test_throws ArgumentError workspace_agent_doc_violations(mktempdir(); extra = nothing)

    write(joinpath(root, "CLAUDE.md"), "Conventions.\nData under /home/someone/data.\n")
    write(joinpath(root, "AlphaWidgets", "notes.md"), "The Sprocket project.\n")
    run(`git -C $root add .`)
    @test length(workspace_agent_doc_violations(root; extra = nothing)) == 1
    patterns = joinpath(mktempdir(), "patterns.txt")
    write(patterns, "# private names\n\nSprocket\n")
    # the private patterns: over the agent docs alone, then over every tracked text file
    @test length(workspace_agent_doc_violations(root; extra = patterns)) == 1
    found = workspace_agent_doc_violations(root; extra = patterns, every_file = true)
    @test length(found) == 2
    @test joinpath("AlphaWidgets", "notes.md") * ":1 matches extra pattern #1" in found

    out, log = IOBuffer(), IOBuffer()
    code = withenv(() -> stack_ci(["agent-docs", "--all"]; root, io = out, log), "AGENT_DOC_PATTERNS" => patterns)
    @test code == 1
    @test length(split(strip(String(take!(out))), '\n')) == 2
    @test String(take!(log)) == "agent_docs: 2 finding(s) [agent docs + all tracked files; private patterns: from AGENT_DOC_PATTERNS]\n"
end

@testitem "stack_ci: a downstream project and the docs inventory" setup = [StackCIWorkspace] begin
    using VisualGeometryToolkitCore: source_downstream, docs_inventory, docs_local, docs_linkcheck_ignore
    using TOML

    root = make_workspace()
    down = mktempdir()
    write(joinpath(down, "Project.toml"), """
        name = "Downstream"
        [deps]
        BetaWidgets = "5a1c1d3e-0b7f-4e5a-9d0c-1f2e3d4c5b02"
        Sprockets = "5a1c1d3e-0b7f-4e5a-9d0c-1f2e3d4c5b03"
        [compat]
        BetaWidgets = "0.0.9"
        """)
    @test source_downstream(down, root) == ["BetaWidgets"]
    project = TOML.parsefile(joinpath(down, "Project.toml"))
    @test project["sources"] == Dict("BetaWidgets" => Dict("path" => joinpath(root, "BetaWidgets")))
    @test project["compat"]["BetaWidgets"] == "0.0.9"
    other = mktempdir()
    @test source_downstream(down, root; compat = true, extra = Dict("Sprockets" => other, "Unnamed" => other)) ==
        ["BetaWidgets", "Sprockets"]
    project = TOML.parsefile(joinpath(down, "Project.toml"))
    @test project["sources"]["Sprockets"] == Dict("path" => other)
    @test project["compat"]["BetaWidgets"] == "0.1"

    site = "https://widgets.example.org"
    url = "https://widgets.example.org/AlphaWidgets/dev/"
    @test docs_inventory(site, "AlphaWidgets") == url
    withenv("WIDGETS_DOCS_LOCAL" => nothing) do
        @test !docs_local("WIDGETS_DOCS_LOCAL")
        @test isempty(docs_linkcheck_ignore(site, "WIDGETS_DOCS_LOCAL"))
        @test docs_inventory(site, "AlphaWidgets"; root, variable = "WIDGETS_DOCS_LOCAL") == url
    end
    withenv("WIDGETS_DOCS_LOCAL" => "1") do
        @test docs_local("WIDGETS_DOCS_LOCAL")
        ignore = docs_linkcheck_ignore(site, "WIDGETS_DOCS_LOCAL")
        @test length(ignore) == 1 && occursin(only(ignore), url) && !occursin(only(ignore), "https://widgetsXexample.org/")
        # no local build of the package yet: the published inventory
        @test docs_inventory(site, "AlphaWidgets"; root, variable = "WIDGETS_DOCS_LOCAL") == url
        inventory = joinpath(root, "AlphaWidgets", "docs", "build", "objects.inv")
        mkpath(dirname(inventory))
        write(inventory, "")
        @test docs_inventory(site, "AlphaWidgets"; root, variable = "WIDGETS_DOCS_LOCAL") == (url, inventory)
    end
end

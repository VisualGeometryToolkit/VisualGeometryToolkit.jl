# =============================================================================
# template.jl — the paper-project template: DrWatson's `initialize_project`
# with the paper-project folder template, then the generated files of
# templates/paper_project/ (placeholders {{NAME}}, {{FIGDIR}}, …) — a project
# that satisfies `paper_project_violations` once its example stage has run.
# =============================================================================

"""
    PAPER_PROJECT_TEMPLATE

The folder template [`new_paper_project`](@ref) hands DrWatson's
`initialize_project`: `configs/`, `scripts/`, `src/`, `test/`, `paper/figures/`
(for the default paper), `data/{exp_raw,exp_pro,sims}` (the inputs, the
results, the smoke runs — Experiments `data_root()`/`results_root()`) and
`plots/`.
"""
const PAPER_PROJECT_TEMPLATE = ["configs", "scripts", "src", "test", "plots",
                                "data" => ["exp_raw", "exp_pro", "sims"]]

const _PKG_ROOT = dirname(dirname(@__DIR__))
const _TEMPLATE_DIR = joinpath(_PKG_ROOT, "templates", "paper_project")

"""
    new_paper_project(path; name = basename(path), papers = ["paper/main.tex"],
                      authors = nothing, git = true, force = false) -> String

Create a paper project at `path` that follows the stack's paper-project
conventions: DrWatson's `initialize_project` with
[`PAPER_PROJECT_TEMPLATE`](@ref) (plus each paper's `figures/` directory), then

- `Project.toml` (DrWatson, VisualGeometryToolkitEval with compat `"1"`, StableRNGs,
  TOML, Test), `README.md`, `CLAUDE.md` (the conventions pointer), `.gitignore`
  (the data bindings, local plots, LaTeX products; the Manifest is tracked);
- `REPRODUCE.md` with the asset-index table, `configs/example.toml`,
  `scripts/example_stage.jl` (stage runner, config of record, run key,
  provenance; it writes a figure, its caption and a macro file), `src/<name>.jl`
  (the prelude), `test/runtests.jl` (calls [`paper_project_violations`](@ref));
- the papers (the first reads the example's assets), `Artifacts.toml` and the
  data catalog stub `configs/data_artifacts.toml`, `data/README.md`.

`initialize_project` activates the new project and adds DrWatson to it; the
previously active project is restored afterwards. `git = false` for a project
inside an existing repository (accv2026 lives in LocalOptimizationDoneRight).
Then: `Pkg.instantiate()` in the project, run `scripts/example_stage.jl`, and
`test/runtests.jl` reports zero violations.
"""
function new_paper_project(path::AbstractString; name::AbstractString = basename(rstrip(abspath(path), '/')),
                           papers::AbstractVector{<:AbstractString} = ["paper/main.tex"],
                           authors = nothing, git::Bool = true, force::Bool = false)
    isempty(papers) && throw(ArgumentError("new_paper_project: a paper project has at least one paper"))
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", name) ||
        throw(ArgumentError("new_paper_project: name $(repr(name)) is not a Julia identifier (it names src/<name>.jl and the DrWatson project)"))
    path = abspath(path)
    figdirs = unique(joinpath(dirname(p), "figures") for p in papers)
    template = Any[PAPER_PROJECT_TEMPLATE..., figdirs...]
    prev = Base.active_project()
    try
        initialize_project(path, name; template, git, force, authors, readme = false,
                           add_test = false, add_docs = false, placeholder = false,
                           folders_to_gitignore = String[])
    finally
        prev === nothing ? Pkg.activate() : Pkg.activate(prev)
    end
    # DrWatson's generic files give way to the paper-project ones.
    for f in ("scripts/intro.jl", "src/dummy_src_file.jl", "Manifest.toml")
        rm(joinpath(path, f); force = true)
    end
    _populate!(path; name, papers, authors)
    git && _commit_all(path, "Paper project setup (VisualGeometryToolkitEval new_paper_project)")
    return path
end

# The generated files of a paper project at `path` (the directories exist or
# are created). Kept apart from DrWatson's step so the tests can build fixture
# projects without touching Pkg.
function _populate!(path::AbstractString; name::AbstractString,
                    papers::AbstractVector{<:AbstractString} = ["paper/main.tex"], authors = nothing)
    first_paper = first(papers)
    figdir = joinpath(dirname(first_paper), "figures")
    subst = [
        "{{NAME}}" => name,
        "{{FIGDIR}}" => figdir,
        "{{FIGDIR_ARGS}}" => join(repr.(splitpath(figdir)), ", "),
        "{{PAPERDIR}}" => dirname(first_paper),
        "{{PAPERS_JL}}" => "[" * join(repr.(papers), ", ") * "]",
    ]
    render(src) = replace(read(joinpath(_TEMPLATE_DIR, src), String), subst...)
    put(dst, text) = (p = joinpath(path, dst); mkpath(dirname(p)); write(p, text))

    for (src, dst) in ("README.md.tmpl" => "README.md", "REPRODUCE.md.tmpl" => "REPRODUCE.md",
                       "CLAUDE.md.tmpl" => "CLAUDE.md", "gitignore.tmpl" => ".gitignore",
                       "Artifacts.toml.tmpl" => "Artifacts.toml",
                       "configs/example.toml.tmpl" => "configs/example.toml",
                       "configs/data_artifacts.toml.tmpl" => "configs/data_artifacts.toml",
                       "scripts/example_stage.jl.tmpl" => "scripts/example_stage.jl",
                       "src/NAME.jl.tmpl" => "src/$name.jl",
                       "test/runtests.jl.tmpl" => "test/runtests.jl",
                       "data/README.md.tmpl" => "data/README.md")
        put(dst, render(src))
    end
    put(first_paper, render("paper/main.tex.tmpl"))
    for p in papers[2:end]
        put(p, render("paper/other.tex.tmpl"))
    end
    for d in unique(joinpath(dirname(p), "figures") for p in papers)
        mkpath(joinpath(path, d))
    end
    for d in ("plots", "data/exp_raw", "data/exp_pro", "data/sims")
        mkpath(joinpath(path, d))
    end
    put("Project.toml", _project_toml(name, authors))
    return path
end

# The project's environment: its deps' UUIDs are this package's own (and
# itself), the compat of VisualGeometryToolkitEval is its major.
function _project_toml(name, authors)
    own = TOML.parsefile(joinpath(_PKG_ROOT, "Project.toml"))
    test_toml = joinpath(_PKG_ROOT, "test", "Project.toml")
    test_deps = isfile(test_toml) ? get(TOML.parsefile(test_toml), "deps", Dict{String,Any}()) : Dict{String,Any}()
    uuids = merge(own["deps"], get(own, "extras", Dict{String,Any}()), test_deps, Dict(own["name"] => own["uuid"]))
    deps = ("DrWatson", "StableRNGs", "TOML", "Test", "VisualGeometryToolkitEval")
    v = VersionNumber(own["version"])
    major = v.major == 0 ? "0.$(v.minor)" : string(v.major)   # the semver-breaking prefix
    compat = Dict{String,Any}("VisualGeometryToolkitEval" => major, "julia" => own["compat"]["julia"])
    for d in ("DrWatson", "StableRNGs", "TOML", "Test")
        haskey(own["compat"], d) && (compat[d] = own["compat"][d])
    end
    io = IOBuffer()
    println(io, "# A paper project (VisualGeometryToolkitEval PaperProjects, new_paper_project): a DrWatson")
    println(io, "# project, not a package. Its Manifest.toml is tracked: the environment of record.")
    println(io, "name = ", repr(name))
    authors === nothing || println(io, "authors = ", repr(authors isa AbstractString ? [authors] : collect(authors)))
    println(io)
    TOML.print(io, Dict("deps" => Dict(d => uuids[d] for d in deps)); sorted = true)
    println(io)
    TOML.print(io, Dict("compat" => compat); sorted = true)
    return String(take!(io))
end

function _commit_all(path, msg)
    repo = LibGit2.GitRepo(path)
    LibGit2.add!(repo, ".")
    sig = try
        LibGit2.Signature(LibGit2.default_signature(repo))   # the user's git identity
    catch
        LibGit2.Signature("VisualGeometryToolkitEval", "no@mail")
    end
    LibGit2.commit(repo, msg; author = sig, committer = sig)
    return nothing
end

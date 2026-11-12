# =============================================================================
# reproduce/reproduce_script.jl — the runner's bootstrap a project carries, `scripts/reproduce/reproduce.sh`, generated
# from the one template shipped here (templates/reproduce/reproduce.sh): a project never writes its own copy.
# =============================================================================

"""
    REPRODUCE_SCRIPT_TEMPLATE

The template of a project's runner bootstrap, `templates/reproduce/reproduce.sh` in VisualGeometryToolkitEval: a bare
`julia` finds the package's sources (fetching them on a fresh checkout), includes `src/cli.jl` and
`src/reproduce/runner.jl` and runs [`reproduce_main`](@ref) on the project's graph. [`reproduce_script`](@ref) fills it.
"""
const REPRODUCE_SCRIPT_TEMPLATE = normpath(joinpath(@__DIR__, "..", "..", "templates", "reproduce", "reproduce.sh"))

const _SCRIPT_PATH = r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$"

"""
    reproduce_script(; julia_project = ".", graph = "<julia_project>/configs/reproduce.toml",
                     path = "<julia_project>/scripts/reproduce/reproduce.sh") -> String

The text of a project's runner bootstrap ([`REPRODUCE_SCRIPT_TEMPLATE`](@ref) filled): run from anywhere in the
repository, it changes to the repository root and runs the runner with `--project=<julia_project>` on `graph`, every
path relative to the repository root. A project's suite checks its copy against it (`read(path, String) ==
reproduce_script(; …)`), and [`write_reproduce_script`](@ref) writes it.
"""
function reproduce_script(; julia_project::AbstractString = ".",
                          graph::AbstractString = normpath(joinpath(julia_project, "configs", "reproduce.toml")),
                          path::AbstractString = normpath(joinpath(julia_project, "scripts", "reproduce", "reproduce.sh")))
    for (k, v) in ("julia_project" => julia_project, "graph" => graph, "path" => path)
        occursin(_SCRIPT_PATH, v) && !isabspath(v) ||
            throw(ArgumentError("reproduce_script: $k = $(repr(v)) is not a path relative to the repository root (letters, digits, _ . / -)"))
    end
    text = read(REPRODUCE_SCRIPT_TEMPLATE, String)
    for (k, v) in ("@@JULIA_PROJECT@@" => julia_project, "@@GRAPH@@" => graph, "@@SCRIPT@@" => path)
        text = replace(text, k => v)
    end
    return text
end

"""
    write_reproduce_script(root; julia_project = ".", graph, path) -> String

Write the runner bootstrap ([`reproduce_script`](@ref)) of the project in the repository `root` to `<root>/<path>`
(by default `<julia_project>/scripts/reproduce/reproduce.sh`), executable. Returns its path.
"""
function write_reproduce_script(root::AbstractString; julia_project::AbstractString = ".",
                                graph::AbstractString = normpath(joinpath(julia_project, "configs", "reproduce.toml")),
                                path::AbstractString = normpath(joinpath(julia_project, "scripts", "reproduce", "reproduce.sh")))
    text = reproduce_script(; julia_project, graph, path)
    out = joinpath(root, path)
    mkpath(dirname(out))
    write(out, text)
    chmod(out, 0o755)
    return out
end

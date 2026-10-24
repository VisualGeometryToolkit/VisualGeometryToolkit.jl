# Build the VisualGeometryToolkitEval documentation, from the repository root:
#
#   julia --project=VisualGeometryToolkitEval/docs --startup-file=no VisualGeometryToolkitEval/docs/make.jl
#
# The site is written to VisualGeometryToolkitEval/docs/build/.

using Documenter
using VisualGeometryToolkitEval

# The guide pages cite the names of every submodule unqualified, as in
# [`stable_seed`](@ref). Documenter resolves such a reference in Main, so Main
# imports each submodule's public names. A name that two submodules share is
# imported from the first; a page cites the other one qualified.
const SUBMODULES = (VisualGeometryToolkitEval.Stats, VisualGeometryToolkitEval.Tuning,
    VisualGeometryToolkitEval.Evaluation, VisualGeometryToolkitEval.Experiments, VisualGeometryToolkitEval.Render,
    VisualGeometryToolkitEval.Pools, VisualGeometryToolkitEval.Datasets, VisualGeometryToolkitEval.DataArtifacts,
    VisualGeometryToolkitEval.Reproduce, VisualGeometryToolkitEval.PaperProjects)
for m in SUBMODULES, n in names(m)
    (n === nameof(m) || isdefined(Main, n)) && continue
    Core.eval(Main, Expr(:import, Expr(:(:), Expr(:., fullname(m)...), Expr(:., n))))
end

makedocs(;
    modules = [VisualGeometryToolkitEval],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "VisualGeometryToolkitEval",
    format = Documenter.HTML(; canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEval/",
                             edit_link = "main",
                             repolink = "https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl",
                             prettyurls = get(ENV, "CI", "false") == "true",
                             # api/evaluation.html is 110 KiB (2026-10-24): the warning
                             # threshold is raised from 100 KiB; the 200 KiB error stays.
                             size_threshold_warn = 150 * 2^10),
    remotes = nothing,
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Guide" => [
            "The experimental protocol" => "protocol.md",
            "The [tuning] schema" => "tuning.md",
            "The composition evaluator" => "evaluator.md",
            "Reproducible figures" => "reproduce.md",
            "Paper projects" => "paper-projects.md",
            "Defaults and file formats" => "configuration.md",
        ],
        "API" => [
            "The top module" => "api/toplevel.md",
            "Stats" => "api/stats.md",
            "Tuning" => "api/tuning.md",
            "Evaluation" => "api/evaluation.md",
            "Experiments" => "api/experiments.md",
            "Render" => "api/render.md",
            "Pools" => "api/pools.md",
            "Datasets" => "api/datasets.md",
            "DataArtifacts" => "api/data_artifacts.md",
            "Reproduce" => "api/reproduce.md",
            "PaperProjects" => "api/paper_projects.md",
        ],
        "Developer notes" => "developer.md",
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "VisualGeometryToolkitEval",
    tag_prefix = "VisualGeometryToolkitEval-",
    devbranch = "main",
    push_preview = false,
)

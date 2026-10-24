# The documentation site. Build locally (no deployment without CI):
#
#   julia --project=docs --startup-file=no docs/make.jl     # → docs/build/index.html

using Documenter
using PoseLib

makedocs(;
    modules = [PoseLib],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "PoseLib.jl",
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "VisualGeometryToolkit.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/PoseLib/",
        edit_link = "main",
        size_threshold = nothing,
    ),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Getting started" => "guide/getting-started.md",
        "Guide" => [
            "Minimal solvers" => "guide/solvers.md",
            "Refiners" => "guide/refiners.md",
            "Platforms and binaries" => "guide/binaries.md",
        ],
        "Developer notes" => "developer.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "PoseLib",
    tag_prefix = "PoseLib-",
    devbranch = "main",
    push_preview = false,
)

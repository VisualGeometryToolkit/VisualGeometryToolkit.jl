# Build the VisualGeometryEval documentation (a local build; no deployment yet):
#
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#   julia --project=docs docs/make.jl            # → docs/build/index.html

using Documenter
using VisualGeometryEval

makedocs(;
    modules = [VisualGeometryEval],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "VisualGeometryEval",
    # One API page (every module's @autodocs) is well over Documenter's default
    # HTML size threshold; it is one page by design, so the threshold is lifted.
    format = Documenter.HTML(; canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryEval/", edit_link = "main", prettyurls = get(ENV, "CI", "false") == "true", size_threshold = nothing),
    remotes = nothing,
    warnonly = [:missing_docs, :cross_references, :docs_block],
    pages = [
        "Home" => "index.md",
        "The experimental protocol" => "protocol.md",
        "The [tuning] schema" => "tuning.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "VisualGeometryEval",
    tag_prefix = "VisualGeometryEval-",
    devbranch = "main",
    push_preview = false,
)

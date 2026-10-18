# Build the VisualGeometryToolkitEval documentation (a local build; no deployment yet):
#
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#   julia --project=docs docs/make.jl            # → docs/build/index.html

using Documenter
using VisualGeometryToolkitEval

makedocs(;
    modules = [VisualGeometryToolkitEval],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "VisualGeometryToolkitEval",
    # One API page (every module's @autodocs) is well over Documenter's default
    # HTML size threshold; it is one page by design, so the threshold is lifted.
    format = Documenter.HTML(; canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEval/", edit_link = "main", prettyurls = get(ENV, "CI", "false") == "true", size_threshold = nothing),
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
    dirname = "VisualGeometryToolkitEval",
    tag_prefix = "VisualGeometryToolkitEval-",
    devbranch = "main",
    push_preview = false,
)

# The documentation site: the README as the index, the guide pages under
# docs/src/guide/, the API reference under docs/src/api/ and a developer page. Build from the
# repository root (no deployment without CI):
#
#   julia --project=VisualGeometryToolkitFeatures/docs --startup-file=no VisualGeometryToolkitFeatures/docs/make.jl

using Documenter
using VisualGeometryToolkitFeatures

# The index is GENERATED from the package README (docs/src/index.md is ignored
# by git), so the two never drift.
let src = joinpath(@__DIR__, "src")
    mkpath(src)
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    meta = join(["```@meta", "EditURL = " * repr("../../README.md"), "```", "", ""], "\n")
    write(joinpath(src, "index.md"), meta * readme)
end

makedocs(;
    modules = [VisualGeometryToolkitFeatures],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "VisualGeometryToolkitFeatures.jl",
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "VisualGeometryToolkit.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/",
        edit_link = "main",
        size_threshold = nothing,
    ),
    # README links point into the repository, not at documented names.
    warnonly = [:cross_references],
    # every exported and every `public` name needs a docstring; the build fails otherwise
    checkdocs = :public,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Detection" => "guide/detection.md",
            "Matching" => "guide/matching.md",
        ],
        "API Reference" => [
            "Detection" => "api/detection.md",
            "Matching" => "api/matching.md",
            "Cache and Visualization" => "api/cache.md",
        ],
        "Developer Notes" => "developer.md",
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "VisualGeometryToolkitFeatures",
    tag_prefix = "VisualGeometryToolkitFeatures-",
    devbranch = "main",
    push_preview = false,
)

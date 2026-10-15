# The minimal documentation site (DESIGN D6): the README as the index, one
# @autodocs API page. Build locally (no deployment without CI):
#
#   julia --project=docs --startup-file=no docs/make.jl     # → docs/build/index.html

using Documenter
using PoseLib

# The index is GENERATED from the package README (docs/src/index.md is ignored
# by git), so the two never drift.
let src = joinpath(@__DIR__, "src")
    mkpath(src)
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    meta = join(["```@meta", "EditURL = " * repr("../../README.md"), "```", "", ""], "\n")
    write(joinpath(src, "index.md"), meta * readme)
end

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
    # README links point into the repository, not at documented names.
    warnonly = [:cross_references, :missing_docs],
    checkdocs = :exports,
    pages = ["Home" => "index.md", "API" => "api.md"],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "PoseLib",
    tag_prefix = "PoseLib-",
    devbranch = "main",
    push_preview = false,
)

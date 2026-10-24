# The documentation site: a hand-written manual (docs/src/index.md and
# docs/src/guide/) and an @autodocs API reference (docs/src/api/). Build locally:
#
#   julia --project=docs --startup-file=no docs/make.jl     # → docs/build/index.html
#
# `checkdocs = :public` with no `warnonly`: a public name without a docstring
# on an API page, a broken cross-reference or a failing doctest stops the build.

using Documenter
using VisualGeometryToolkitIO

DocMeta.setdocmeta!(
    VisualGeometryToolkitIO, :DocTestSetup,
    :(using VisualGeometryToolkitCore, VisualGeometryToolkitIO; import VisualGeometryToolkitIO as VgtkIO);
    recursive = true,
)

makedocs(;
    modules = [VisualGeometryToolkitIO],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "VisualGeometryToolkitIO.jl",
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "VisualGeometryToolkit.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/",
        edit_link = "main",
        size_threshold = nothing,
    ),
    checkdocs = :public,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Schema Rules" => "guide/schemas.md",
            "Kinds and the I/O Surface" => "guide/kinds.md",
            "Conventions on Disk" => "guide/conventions.md",
            "Third-Party Calibrations" => "guide/third-party.md",
            "Related Formats and Exporters" => "guide/related-formats.md",
            "Layers and Design" => "guide/design.md",
            "Testing and Verification" => "guide/verification.md",
        ],
        "API Reference" => [
            "Native Kinds" => "api/native.md",
            "COLMAP" => "api/colmap.md",
            "Third-Party Calibrations" => "api/third-party.md",
            "Exporters" => "api/exporters.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "VisualGeometryToolkitIO",
    tag_prefix = "VisualGeometryToolkitIO-",
    devbranch = "main",
    push_preview = false,
)

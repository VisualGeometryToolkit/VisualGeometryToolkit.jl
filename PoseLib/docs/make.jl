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
    ),
    # The docs gates (General review, #171997, #173031, #173032): every public
    # docstring is in the manual, the doctests run, every external link
    # resolves, and nothing is warnonly — a warning fails the build. That
    # every public name HAS a docstring is test/docs_tests.jl's job (the
    # missing-docs check only sees docstrings that exist).
    checkdocs = :public,
    doctest = true,
    linkcheck = true,
    # github.com rate-limits a crawler (429 on CITATION.cff in
    # VisualGeometryToolkitCore's build, 2026-11-04); lychee
    # (.github/workflows/Links.yml) checks its links with a token.
    linkcheck_ignore = [r"^https://github\.com/"],
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

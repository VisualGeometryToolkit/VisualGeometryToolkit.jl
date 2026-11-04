# The documentation site: the landing page (docs/src/index.md), the guide pages
# under docs/src/guide/, the API reference under docs/src/api/ and a developer
# page. Build from the repository root (no deployment without CI):
#
#   julia --project=VisualGeometryToolkitFeatures/docs --startup-file=no VisualGeometryToolkitFeatures/docs/make.jl

using Documenter
using VisualGeometryToolkitFeatures
using DocumenterCitations

# Literature: `[Key](@cite)` against docs/src/refs.bib, listed on the References page.
const BIB = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

# The landing page shows the README's install line and example verbatim;
# test/readme_tests.jl checks that the two agree and runs the example.

makedocs(;
    modules = [VisualGeometryToolkitFeatures],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "VisualGeometryToolkitFeatures.jl",
    plugins = [BIB],
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "VisualGeometryToolkit.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/",
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
    # Rate-limited or bot-blocking hosts that answer a crawler with 403/429
    # (github.com gave 429 on Core's build, 2026-11-04); lychee
    # (.github/workflows/Links.yml) checks them with a token.
    linkcheck_ignore = [r"^https://github\.com/", r"^https://doi\.org/", r"^https://(www\.)?sciencedirect\.com/", r"^https://ieeexplore\.ieee\.org/", r"^https://dl\.acm\.org/"],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Detection" => "guide/detection.md",
            "The Förstner operator" => "guide/foerstner.md",
            "Matching" => "guide/matching.md",
        ],
        "API Reference" => [
            "Detection" => "api/detection.md",
            "Matching" => "api/matching.md",
            "Cache and Visualization" => "api/cache.md",
        ],
        "Developer Notes" => "developer.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "VisualGeometryToolkitFeatures",
    tag_prefix = "VisualGeometryToolkitFeatures-",
    devbranch = "main",
    push_preview = false,
)

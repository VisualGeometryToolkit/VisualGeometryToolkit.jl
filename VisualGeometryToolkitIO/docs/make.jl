# The documentation site: a hand-written manual (docs/src/index.md and
# docs/src/guide/) and an @autodocs API reference (docs/src/api/). Build locally:
#
#   julia --project=docs --startup-file=no docs/make.jl     # → docs/build/index.html
#
# `checkdocs = :public` with no `warnonly`: a public docstring missing from an
# API page, a broken cross-reference, a dead link or a failing doctest stops
# the build.

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
        # The native-kinds API page is docstring-dense: api/native.html
        # measured 118 KiB on 2026-11-11, over the 100 KiB default warning.
        # The hard 200 KiB error threshold is kept; split the page before it
        # reaches it.
        size_threshold_warn = 150 * 2^10,
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
    # (github.com gave 429 on CITATION.cff, 2026-11-04); lychee
    # (.github/workflows/Links.yml) checks them with a token.
    linkcheck_ignore = [r"^https://github\.com/"],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Schema Rules" => "guide/schemas.md",
            "Kinds and the I/O Surface" => "guide/kinds.md",
            "Conventions on Disk" => "guide/conventions.md",
            "Quantities, Frames and Detections" => "guide/serialization.md",
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

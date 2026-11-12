using Documenter
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore

using DocumenterCitations, DocumenterInterLinks

# Literature: `[Key](@cite)` / `@citet` / `@citep` against docs/src/refs.bib,
# listed on the References page.
const BIB = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)
# The other stack packages' docs: `[text](@extref <Package> ...)` resolves against the inventory (objects.inv) their
# published `dev` docs carry; with VGTK_DOCS_LOCAL=1 (ci/slurm/submit_docs.sh) against this checkout's own build of
# that package where one exists, so a release's docs build before its dependencies' docs are published.
include(joinpath(@__DIR__, "..", "..", "ci", "docs_inventory.jl"))   # vgtk_inventory(p)
const LINKS = InterLinks((p => vgtk_inventory(p) for p in (
    "VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitFeatures",
    "VisualGeometryToolkitEstimation", "VisualGeometryToolkitEval", "PoseLib"))...)

makedocs(;
    plugins=[BIB, LINKS],
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "VisualGeometryToolkit.jl"),
    modules=[VisualGeometryToolkitEstimation],
    authors="James Pritts <jbpritts@gmail.com>",
    sitename="VisualGeometryToolkitEstimation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/",
        edit_link = "main",
        assets=String[],
        # The RANSAC API page is docstring-dense: api/ransac.html measured
        # 142 KiB on 2026-11-04, over the 100 KiB default warning. The hard
        # 200 KiB error threshold is kept; split the page before it reaches it.
        size_threshold_warn = 150 * 2^10,
        # The covariance note's notation (guide/covariance.md).
        mathengine=Documenter.KaTeX(Dict(:macros => Dict(
            "\\bx" => "\\mathbf{x}", "\\bg" => "\\mathbf{g}", "\\br" => "\\mathbf{r}",
            "\\bW" => "\\mathbf{W}", "\\bSigma" => "\\boldsymbol{\\Sigma}",
            "\\btSigma" => "\\widetilde{\\boldsymbol{\\Sigma}}",
            "\\cS" => "\\mathcal{S}", "\\cI" => "\\mathcal{I}",
            "\\vect" => "\\operatorname{vec}",
        ))),
        repolink = "https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl",
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
    # VGTK_DOCS_LOCAL=1: an @extref into a sibling package built here points at anchors not yet published
    linkcheck_ignore = [(get(ENV, "VGTK_DOCS_LOCAL", "") == "1" ? [r"^https://visualgeometrytoolkit\.github\.io/"] : Regex[])..., r"^https://github\.com/", r"^https://doi\.org/", r"^https://(www\.)?sciencedirect\.com/", r"^https://ieeexplore\.ieee\.org/", r"^https://dl\.acm\.org/"],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Architecture" => "guide/architecture.md",
            "Scoring Functions" => "guide/scoring.md",
            "RANSAC" => "guide/ransac.md",
            "Fitting Algorithms" => "guide/fitting.md",
            "Covariance and Gauge Invariance" => "guide/covariance.md",
            "Ray Density Correction" => "guide/ray-density.md",
            "Homography Decomposition" => "guide/homography-decomposition.md",
            "Homography Cheirality" => "guide/cheirality.md",
            "Rig-to-Pattern Hand–Eye Calibration" => "guide/hand-eye.md",
            "Robot–World Hand–Eye Calibration" => "guide/robot-world-hand-eye.md",
            "Extending" => "guide/extending.md",
        ],
        "API Reference" => [
            "IRLS & Posterior Weighting" => "api/irls.md",
            "Scoring & LO Contracts" => "api/scoring.md",
            "RANSAC" => "api/ransac.md",
            "Line Fitting" => "api/line.md",
            "Circle Fitting" => "api/circle.md",
            "Homography" => "api/homography.md",
            "Essential Matrix" => "api/essential.md",
            "Fundamental Matrix" => "api/fundmat.md",
            "P3P" => "api/p3p.md",
            "Hand-Eye Calibration" => "api/hand_eye.md",
            "Radial and Lens Calibration" => "api/radial.md",
            "Ellipse Fitting" => "api/ellipse.md",
            "Pose and Rotation from Directions" => "api/bearings.md",
            "View and Observation Graphs" => "api/view_graph.md",
        ],
        "References" => "references.md",
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "VisualGeometryToolkitEstimation",
    tag_prefix = "VisualGeometryToolkitEstimation-",
    devbranch = "main",
    push_preview = false,
)

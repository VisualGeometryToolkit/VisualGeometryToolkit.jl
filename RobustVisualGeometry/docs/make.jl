using Documenter
using RobustVisualGeometry
# VGC is in `modules` ONLY so that `@docs` blocks can resolve the docstrings of
# the contracts and toolkit names the loop is built on (VGC owns them). Its own
# doc COVERAGE is not this package's business — hence `checkdocs = :none`.
using VisualGeometryCore

makedocs(;
    modules=[RobustVisualGeometry, VisualGeometryCore],
    checkdocs=:none,
    authors="James Pritts <jbpritts@gmail.com>",
    sitename="RobustVisualGeometry.jl",
    remotes=nothing,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/RobustVisualGeometry/",
        edit_link = "main",
        assets=String[],
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
    warnonly=[:cross_references],
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
            "Homography" => "api/homography.md",
            "Essential Matrix" => "api/essential.md",
            "Fundamental Matrix" => "api/fundmat.md",
            "P3P" => "api/p3p.md",
            "Hand-Eye Calibration" => "api/hand_eye.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "RobustVisualGeometry",
    tag_prefix = "RobustVisualGeometry-",
    devbranch = "main",
    push_preview = false,
)

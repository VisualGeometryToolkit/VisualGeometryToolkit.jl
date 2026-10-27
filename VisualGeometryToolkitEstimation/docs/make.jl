using Documenter
using VisualGeometryToolkitEstimation
# VGC is in `modules` ONLY so that `@docs` blocks can resolve the docstrings of
# the contracts and toolkit names the loop is built on (VGC owns them). Its own
# doc coverage is checked by its own docs build. Documenter's `checkdocs` would
# check every module in `modules`, and `checkdocs_ignored_modules` skips only
# submodules, so `checkdocs = :none` and this package's coverage is checked
# below instead.
using VisualGeometryToolkitCore

# Every public name this package owns has a docstring and appears in an `@docs`
# block of the manual. A name another package owns (an alias, a method added
# to a contract function) is documented by its owner.
function check_public_docs(M::Module, src::AbstractString)
    listed = Set{Symbol}()
    entry = Regex("^\\s*(?:" * string(nameof(M)) * "\\.)?([^\\s.(]+)\\s*(?:\\(|\$)")
    for (root, _, files) in walkdir(src), f in files
        endswith(f, ".md") || continue
        for m in eachmatch(r"^```@docs\n(.*?)^```"ms, read(joinpath(root, f), String))
            for line in split(m[1], '\n')
                name = match(entry, line)
                name === nothing || push!(listed, Symbol(name[1]))
            end
        end
    end
    owned(n) = (v = getfield(M, n); !(v isa Module) && (!(v isa Function || v isa Type) || parentmodule(v) === M))
    names_ = filter(n -> n !== nameof(M) && Base.ispublic(M, n) && isdefined(M, n) && owned(n), names(M; all = true))
    undocumented = sort(filter(n -> !Docs.hasdoc(M, n), names_))
    unlisted = sort(filter(n -> Docs.hasdoc(M, n) && n ∉ listed, names_))
    isempty(undocumented) && isempty(unlisted) && return nothing
    error("public names of $M without a docstring: $(join(undocumented, ", ")); " *
          "with a docstring but in no @docs block: $(join(unlisted, ", "))")
end
check_public_docs(VisualGeometryToolkitEstimation, joinpath(@__DIR__, "src"))

using DocumenterCitations, DocumenterInterLinks

# Literature: `[Key](@cite)` / `@citet` / `@citep` against docs/src/refs.bib,
# listed on the References page.
const BIB = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)
# The other stack packages' docs: `[text](@extref <Package> ...)` resolves
# against the inventory (objects.inv) their published `dev` docs carry.
const LINKS = InterLinks((p => "https://visualgeometrytoolkit.github.io/$p/dev/" for p in (
    "VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitFeatures",
    "VisualGeometryToolkitEstimation", "VisualGeometryToolkitEval", "PoseLib"))...)

makedocs(;
    plugins=[BIB, LINKS],
    modules=[VisualGeometryToolkitEstimation, VisualGeometryToolkitCore],
    checkdocs=:none,
    authors="James Pritts <jbpritts@gmail.com>",
    sitename="VisualGeometryToolkitEstimation.jl",
    remotes=nothing,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/",
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
            "Radial and Lens Calibration" => "api/radial.md",
            "Ellipse Fitting" => "api/ellipse.md",
            "Pose and Rotation from Directions" => "api/bearings.md",
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

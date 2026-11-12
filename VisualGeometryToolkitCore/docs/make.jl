using Documenter
using VisualGeometryToolkitCore

# Load Makie to activate the plotting extension (GLMakie needs a display,
# which headless docs builds / CI don't have)
using Makie

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

# The docs-build-only docstring shims that used to live here (prec, cov,
# consensus_set, lo_marker, test_cheirality, is_peak, and the Cameras-scope
# import of epipolar_map) landed in
# src in 0.14 — the docstrings attach at their definition sites and the
# scope imports live in cameras.jl.

# The Conventions page is GENERATED from conventions/*.md, the one copy of the
# stack conventions (conventions/README.md says how to edit them), in the order
# of `convention_files()`; docs/src/guide/conventions.md is ignored by git. A
# guide-page link there is written relative to conventions/ and rewritten here.
# The `@meta` block is required: the files' `@ref` links resolve public,
# not-exported names (`CspondNormalization`, `test_model`, ...) in this module.
let page = joinpath(@__DIR__, "src", "guide", "conventions.md")
    body = join((rstrip(read(f, String)) for f in VisualGeometryToolkitCore.convention_files()), "\n\n")
    body = replace(body, "](../docs/src/guide/" => "](")
    write(page, """
        ```@meta
        CurrentModule = VisualGeometryToolkitCore
        EditURL = "../../../conventions/README.md"
        ```

        """ * body * """


        ## See also

        - [Coordinate Frames](frames.md) — the `Frame{N}` hierarchy and the tags
        - [Robust Estimation](estimators.md) — which estimator for which task
        - [Epipolar Forms](forms.md) — model forms and their composition
        - [Design: `Attributed{V,A}`](design.md) — composition and attribute columns
        """)
end

makedocs(;
    # VGTK_DOCS_INVENTORY_ONLY=1 (ci/slurm/docs.sbatch, only after a failed strict build): every check warns, so the build still
    # writes objects.inv for the packages that link to Core; the strict build's failure stands as the gate's result.
    warnonly = get(ENV, "VGTK_DOCS_INVENTORY_ONLY", "") == "1",
    plugins = [BIB, LINKS],
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "VisualGeometryToolkit.jl"),
    modules = [VisualGeometryToolkitCore],
    authors = "James Pritts <jbpritts@gmail.com>",
    sitename = "VisualGeometryToolkitCore.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/",
        edit_link = "main",
        assets = String[],
        # The API reference is docstring-dense, so the warn threshold is raised
        # from the 100 KiB default. The hard 200 KiB error threshold is kept.
        # Measured at the 2026-09-15 build: api/cameras.html 187 KiB, 13 KiB
        # short of the error. Split on 2026-09-21: calibration.jl and the lens regression to
        # api/calibration.md, the grid models to api/spline_lens.md, and the
        # manifolds + confidence regions off api/estimators (184 KiB) to
        # api/manifolds.md. Split on 2026-10-09 (api/estimators.html had grown
        # to 365 KiB): the losses and width axes to api/robust_losses.md, the
        # statistics toolkit, block coordinate descent and the LM core to
        # api/robust_statistics.md, the tag registry and stack contracts to
        # api/stack_contracts.md. Split on 2026-10-19 (api/solvers.html reached
        # 201.7 KiB with the citation links): the epipolar forms to
        # api/epipolar_forms.md. Split on 2026-11-11 (with every public
        # docstring in the manual, api/cameras.html measured 207 KiB and
        # api/estimators.html 246 KiB): the intrinsics and sensors to
        # api/intrinsics.md, the pool selection and its grid kernels to
        # api/pool_selection.md, the Student-t + uniform scorer to
        # api/student_t_scoring.md.
        size_threshold_warn = 150 * 2^10,
        # The search index crossed the 500 KiB default with the 0.14
        # docstring additions (the shim-block docstrings moved into src plus
        # the DoubleSphere/B-spline/ray-plane surfaces); same rationale as
        # size_threshold_warn above. Measured 1.15 MiB on 2026-11-04, once every
        # public docstring was in the manual (checkdocs = :public); 1.38 MiB on
        # 2026-11-11, with the 19 docstrings whose files were in no page added.
        search_size_threshold_warn = 1600 * 2^10,
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
    # VGTK_DOCS_LOCAL=1: an @extref into a sibling package built here points at anchors not yet published
    linkcheck_ignore = [(get(ENV, "VGTK_DOCS_LOCAL", "") == "1" ? [r"^https://visualgeometrytoolkit\.github\.io/"] : Regex[])..., r"^https://github\.com/", r"^https://doi\.org/", r"^https://(www\.)?sciencedirect\.com/", r"^https://ieeexplore\.ieee\.org/", r"^https://dl\.acm\.org/"],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Conventions" => "guide/conventions.md",
            "Design: Attributed{V,A}" => "guide/design.md",
            "Coordinate Frames" => "guide/frames.md",
            "Epipolar Forms" => "guide/forms.md",
            "Stereo Rigs and Uncertainty" => "guide/stereo.md",
            "Cameras" => "guide/cameras.md",
            "The Framed Imaging Chain" => "guide/imaging-chain.md",
            "Ray Maps and Frame Transitions" => "guide/sphere.md",
            "Lens Families" => "guide/lenses.md",
            "Back-projection and Undistortion" => "guide/undistortion.md",
            "Batches" => "guide/batched.md",
            "Geometric Primitives" => "guide/primitives.md",
            "Transforms" => "guide/transforms.md",
            "Robust Estimation" => "guide/estimators.md",
            "Features & Detection" => "guide/features.md",
            "Visualization" => "guide/visualization.md",
        ],
        "API Reference" => [
            "Frames" => "api/frames.md",
            "Transforms" => "api/transforms.md",
            "Warping" => "api/warping.md",
            "Batched" => "api/batched.md",
            "Cameras" => "api/cameras.md",
            "Intrinsics & Sensors" => "api/intrinsics.md",
            "Spline Lens" => "api/spline_lens.md",
            "Calibration" => "api/calibration.md",
            "Lenses" => "api/lenses.md",
            # Split on 2026-11-04 (api/imaging.html 226 KiB, api/core.html
            # 206 KiB, both over the 200 KiB error): the rigs, stereo and
            # captures to api/rigs.md, the synthetic data to api/synthetic.md.
            "Imaging Chain & Targets" => "api/imaging.md",
            "Rigs, Stereo & Captures" => "api/rigs.md",
            "Primitives" => "api/primitives.md",
            "Solvers" => "api/solvers.md",
            "Epipolar Forms" => "api/epipolar_forms.md",
            "Correspondences" => "api/matching.md",
            "Robust Estimation" => "api/estimators.md",
            "Student-t + Uniform Scoring" => "api/student_t_scoring.md",
            "Pool Selection" => "api/pool_selection.md",
            "Robust Losses" => "api/robust_losses.md",
            "Robust Statistics & Solvers" => "api/robust_statistics.md",
            "Configuration Tags & Stack Contracts" => "api/stack_contracts.md",
            "Manifolds & Uncertainty" => "api/manifolds.md",
            "Core Types" => "api/types.md",
            "Core" => "api/core.md",
            "Synthetic Data" => "api/synthetic.md",
        ],
        "References" => "references.md",
    ],
)

deploydocs(;
    repo = "github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl.git",
    dirname = "VisualGeometryToolkitCore",
    tag_prefix = "VisualGeometryToolkitCore-",
    devbranch = "main",
    push_preview = false,
)

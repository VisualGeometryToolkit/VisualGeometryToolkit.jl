# =============================================================================
# asset_stages.jl — the last two stages of a paper-project driver whose
# renders are a keyed product's files: VERIFY each against the paper's copy and
# DEPLOY it into the paper (verify_assets / deploy_assets, paper_assets.jl),
# the deploy under the config's sign-off. One pair for every project's drivers
# (hoisted 2026-11-21 from the paper projects' own copies, RANSACScoringDoneRight's
# neurips2026/src/assets.jl). Not dependency-free: it reads the keyed products
# (Experiments) and the bound project's paper directory.
# =============================================================================

"""
    paper_figures_dir(paper = "paper/main.tex"; root = project_root()) -> String

The directory the paper `paper` (its main .tex, relative to the project) reads
its generated assets from: `figures/` beside it, under the project `root`
(Experiments' `project_root()`, the bound project). Where a driver's
deploy stage puts its renders ([`deploy_assets`](@ref), [`asset_stages`](@ref))
and where the template's papers keep them (PaperProjects' `new_paper_project`);
`root = ""` gives it relative to the project.
"""
paper_figures_dir(paper::AbstractString = joinpath("paper", "main.tex"); root::AbstractString = project_root()) =
    joinpath(root, dirname(paper), "figures")

"""
    deploy_signoff(cfg) -> Union{Bool, Vector{String}}

The author's sign-off of a driver's config on replacing submitted assets, its
`[deploy] replace_submitted`: `true` for every asset, the names of the
signed-off assets, or `false` (the default, also without a `[deploy]` table)
— the `replace_submitted` of [`deploy_assets`](@ref). Any other value is an
`ArgumentError`.
"""
function deploy_signoff(cfg::AbstractDict)
    v = get(get(cfg, "deploy", Dict{String, Any}()), "replace_submitted", false)
    v isa Bool && return v
    v isa AbstractVector && all(x -> x isa AbstractString, v) && return String.(v)
    throw(ArgumentError("[deploy] replace_submitted = $(repr(v)): true, false or the names of the signed-off assets"))
end

"""
    asset_stages(product, cfg, assets; paper = "paper/main.tex") -> Vector{Stage}

The verify and deploy check stages (Experiments' `Stage`, for `run_stages`) of
a driver whose renders `assets` (file names) are files of the keyed product
`product()` — a function, so the product is built when its stage runs:

- `:verify`, each render against the paper's copy in
  [`paper_figures_dir`](@ref)`(paper)` ([`verify_assets`](@ref): bytes, and a
  `.tex` fragment's numbers), the report beside the renders;
- `:deploy`, into that directory where nothing changes, or where the config's
  `[deploy] replace_submitted` signs the difference off
  ([`deploy_assets`](@ref), [`deploy_signoff`](@ref)).

Each refuses a product that is not produced under its current key
(Experiments' `produced_dir`: a commit re-keys it): its producing stage runs
first. A driver lists them after its producing stages:

```julia
const STAGES = [Stage(:figure, …; items = () -> [figure_product()], …),
                asset_stages(figure_product, CFG, ASSETS)...]
```
"""
asset_stages(product::Function, cfg::AbstractDict, assets; paper::AbstractString = joinpath("paper", "main.tex")) = [
    Stage(:verify, "each render against the paper's copy (bytes, coordinates)";
          check = () -> (verify_assets(produced_dir(product()), paper_figures_dir(paper), assets); nothing)),
    Stage(:deploy, "into the paper's figures where nothing changes (or signed off)";
          check = () -> (deploy_assets(produced_dir(product()), paper_figures_dir(paper), assets;
                                       replace_submitted = deploy_signoff(cfg)); nothing)),
]

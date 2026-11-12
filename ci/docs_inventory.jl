# docs_inventory.jl — included by the docs/make.jl of the packages that link to their siblings' docs (Core, Estimation,
# PoseLib): the inventory of a stack package `p` for DocumenterInterLinks. The published `dev` docs' objects.inv; with
# VGTK_DOCS_LOCAL=1 (ci/slurm/submit_docs.sh) this checkout's own build of `p` where one exists, so a release's docs build
# before its dependencies' docs are published.
vgtk_inventory(p) = (url = "https://visualgeometrytoolkit.github.io/$p/dev/";
    inv = joinpath(@__DIR__, "..", p, "docs", "build", "objects.inv");
    get(ENV, "VGTK_DOCS_LOCAL", "") == "1" && isfile(inv) ? (url, inv) : url)

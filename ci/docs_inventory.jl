# docs_inventory.jl — included by the docs/make.jl of the packages that link to their siblings' docs (Core, Estimation,
# PoseLib): the inventory of a stack package `p` for DocumenterInterLinks. The published `dev` docs' objects.inv; with
# VGTK_DOCS_LOCAL=1 (ci/slurm/submit_docs.sh) this checkout's own build of `p` where one exists, so a release's docs build
# before its dependencies' docs are published.
# `vgtk_docs_local()` is the one reader of VGTK_DOCS_LOCAL, and `vgtk_local_linkcheck_ignore()` its linkcheck rule: in a
# local build an @extref into a sibling package built here points at anchors not yet published, so the published docs'
# URLs are not link-checked.
vgtk_docs_local() = get(ENV, "VGTK_DOCS_LOCAL", "") == "1"
vgtk_local_linkcheck_ignore() = vgtk_docs_local() ? [r"^https://visualgeometrytoolkit\.github\.io/"] : Regex[]
vgtk_inventory(p) = (url = "https://visualgeometrytoolkit.github.io/$p/dev/";
    inv = joinpath(@__DIR__, "..", p, "docs", "build", "objects.inv");
    vgtk_docs_local() && isfile(inv) ? (url, inv) : url)

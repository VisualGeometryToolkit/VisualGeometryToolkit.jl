# docs_inventory.jl — included by the docs/make.jl of the packages that link to their siblings' docs (Core, Estimation,
# PoseLib): the inventory of a stack package `p` for DocumenterInterLinks. The published `dev` docs' objects.inv; with
# VGTK_DOCS_LOCAL=1 (ci/slurm/submit_docs.sh) this checkout's own build of `p` where one exists, so a release's docs build
# before its dependencies' docs are published.
# `vgtk_docs_local()` is the one reader of VGTK_DOCS_LOCAL, and `vgtk_local_linkcheck_ignore()` its linkcheck rule: in a
# local build an @extref into a sibling package built here points at anchors not yet published, so the published docs'
# URLs are not link-checked.
# The implementation is VisualGeometryToolkitCore's (`docs_local`, `docs_linkcheck_ignore`, `docs_inventory`); this file
# holds this workspace's site and variable.
import VisualGeometryToolkitCore as VgtkCore
const VGTK_DOCS_SITE = "https://visualgeometrytoolkit.github.io"
vgtk_docs_local() = VgtkCore.docs_local("VGTK_DOCS_LOCAL")
vgtk_local_linkcheck_ignore() = VgtkCore.docs_linkcheck_ignore(VGTK_DOCS_SITE, "VGTK_DOCS_LOCAL")
vgtk_inventory(p) = VgtkCore.docs_inventory(VGTK_DOCS_SITE, p; root = dirname(@__DIR__), variable = "VGTK_DOCS_LOCAL")

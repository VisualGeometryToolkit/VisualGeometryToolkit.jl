# packages.sh — the monorepo's packages, in dependency order: the one list every script of this directory reads
# (common.sh for the launches, summarize.sh for the table's order). Sourced; defines VGTK_ALL_PACKAGES only.
VGTK_ALL_PACKAGES=(VisualGeometryToolkitCore VisualGeometryToolkitIO VisualGeometryToolkitEstimation
                   VisualGeometryToolkitFeatures VisualGeometryToolkitEval PoseLib)

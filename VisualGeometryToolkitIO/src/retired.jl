# =============================================================================
# Retired entry points (VisualGeometryToolkitIO 0.11.0): the kinds that moved
# to their owners
# =============================================================================
#
# The vignetting, environment-light and radial-profile kinds are
# VisualRadiometryToolkitIO's, with the types they hold
# (VisualRadiometryToolkitCore's); a panorama's results are PanoramaStitching's.
# Each keeps its schema name, so every file written before the move reads with
# its owner (validate.jl, `_MOVED_KINDS`: a file of one handed to `validate`,
# `vgio_migrate` or a reader here is refused by name with that owner).
#
# `load(AbstractVignetting, path)`, `save(path, vignetting)` and their light and
# radial-profile forms are METHODS of this package's two verbs, which
# VisualRadiometryToolkitIO adds: a caller who loads it keeps the same calls.
# The panorama's two functions had names of their own, refused here by name:
# no alias, no forwarding, not public.
# =============================================================================

"""
    MOVED_NAMES

The names this package retired by moving them to another package
(`save_panorama` and `load_panorama`: PanoramaStitching's), each a refusal
here. The stack's duplication check reads it (VisualGeometryToolkitCore's
`MOVED_NAMES` convention): the package that took a name over is not reported
for defining it.
"""
const MOVED_NAMES = (:save_panorama, :load_panorama)

_moved_to_panorama_stitching(name::Symbol) = throw(ArgumentError(
    "VisualGeometryToolkitIO.$name is retired (0.11.0): $(_moved_kind_message("panorama"))"))

"Retired (VisualGeometryToolkitIO 0.11.0): the panorama kind is PanoramaStitching's."
save_panorama(args...; kwargs...) = _moved_to_panorama_stitching(:save_panorama)
"Retired (VisualGeometryToolkitIO 0.11.0): the panorama kind is PanoramaStitching's."
load_panorama(args...; kwargs...) = _moved_to_panorama_stitching(:load_panorama)

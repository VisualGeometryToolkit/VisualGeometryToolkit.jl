# =============================================================================
# The stack conventions: one copy, shipped with the package
# =============================================================================
#
#   * `conventions_dir`   — the `conventions/` folder of this installed copy.
#   * `convention_files`  — its convention files, in reading order.
#
# The folder is the one canonical text of the conventions every stack package
# follows. VisualGeometryCore's CLAUDE.md imports the files, every other
# package's CLAUDE.md points at the folder, and docs/make.jl builds the
# Conventions page from `convention_files()`.
# =============================================================================

# Reading order; conventions/README.md is the folder's index and not part of it.
const _CONVENTION_FILES = ("overview.md", "naming.md", "signatures.md", "data-arguments.md",
    "frames.md", "pixel-origins.md", "bearings.md", "measurement-noise.md", "persisted-geometry.md", "jacobians.md", "units.md",
    "stack-contracts.md")

"""
    conventions_dir() -> String

The `conventions/` folder of this VisualGeometryCore installation: the one
canonical copy of the conventions every stack package follows (naming,
signatures, data arguments, coordinate frames and normalization, pixel origins,
bearings, measurement noise, Jacobians, units, stack contracts). A downstream
package finds the installed text here rather than carrying a copy.
"""
conventions_dir() = joinpath(pkgdir(@__MODULE__), "conventions")

"""
    convention_files() -> Vector{String}

The absolute paths of the convention files in [`conventions_dir`](@ref), in
reading order (the order of the docs' Conventions page).
"""
convention_files() = [joinpath(conventions_dir(), f) for f in _CONVENTION_FILES]

# =============================================================================
# VISUALIZATION STUBS (implemented by Makie extension)
# =============================================================================
#
# These functions are defined as stubs here so they can be imported for LSP
# symbol indexing. The implementations are in the Makie extension
# (ext/VisualGeometryToolkitCoreMakieExt.jl, code in src/plotting/), which loads with
# any Makie backend; the GLMakie extension adds only `window!` and `closeall`.
#
# =============================================================================

# Image display
export CampixelImshow, campixel_imshow, campixel_imshow!
export imshow, imshow!
export plot_rectification

# 2D overlays (poly! not exported to avoid conflicts with Makie)
export matches, matches!
export ellipses!
# VisualGeometryToolkitFeatures EXTENDS matches/matches! for its feature types (the old
# cspond!/matchlines names are gone from both packages).

# 3D visualization
export LScene3D, camera!
export cameraplot!, frameplot!
export Frustum
# The recipes' non-mutating forms and the frame spec: reached qualified (the guides use the `!` forms)
public cameraplot, frameplot, FrameSpec

# Utilities
export window!, closeall

# -----------------------------------------------------------------------------
# Image Display Functions
# -----------------------------------------------------------------------------

"Create an LScene spec with campixel! camera for SpecApi composition. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function CampixelImshow end

"Display an image using an LScene with campixel! camera. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function campixel_imshow end

"Add an image to an existing LScene with campixel! camera. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function campixel_imshow! end

"""Batched elliptical-boundary overlay — see the Makie extension."""
function ellipses! end

"Display an image with correct orientation using a standard Makie Axis. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function imshow end

"Add an image to an existing Axis. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function imshow! end

"Raw | rectified panels through one camera, or one panel per camera for comparison. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function plot_rectification end

# -----------------------------------------------------------------------------
# Overlay Functions
# -----------------------------------------------------------------------------

"Visualize point correspondences on a single axis. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function matches end
function matches! end

"Add a polygon/shape to an LScene. Not exported to avoid Makie conflicts. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function poly! end

# -----------------------------------------------------------------------------
# 3D Visualization Functions
# -----------------------------------------------------------------------------

"Create a 3D scene with Camera3D controls. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function LScene3D end

"Visualize a coordinate frame with RGB axes. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function frameplot end
function frameplot! end

"Visualize a camera with center marker, axes, and frustum. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function cameraplot end
function cameraplot! end

"Add camera visualization to an LScene using SpecApi. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function camera! end

# -----------------------------------------------------------------------------
# Declarative PlotSpec Constructors
# -----------------------------------------------------------------------------

"Create a coordinate frame as a Lines PlotSpec. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function FrameSpec end

"Create a camera frustum wireframe as a Lines PlotSpec. Requires a Makie backend (implementation lives in VisualGeometryToolkitCoreMakieExt)."
function Frustum end

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

"Display a Figure in a new named GLMakie window. Requires GLMakie (implementation lives in VisualGeometryToolkitCoreGLMakieExt)."
function window! end

"Close all GLMakie screen windows. Requires GLMakie (implementation lives in VisualGeometryToolkitCoreGLMakieExt)."
function closeall end

module VisualGeometryToolkitCoreMakieExt

# =============================================================================
# Makie Extension for VisualGeometryToolkitCore
# =============================================================================
#
# This extension provides plotting functionality that works with any Makie
# backend (CairoMakie, GLMakie, WGLMakie, RPRMakie).  It activates as soon as
# Makie is loaded — no GLMakie required.  The GLMakie window helpers
# (`window!`, `closeall`) live in `VisualGeometryToolkitCoreGLMakieExt`.
#
# Source structure (mirrors the original layout):
#   src/plotting/geometry.jl       - Pure geometry computation (source of truth)
#   src/plotting/image.jl          - Image display (CampixelImshow, imshow, campixel_imshow)
#   src/plotting/rectification.jl  - plot_rectification
#   src/plotting/overlays.jl       - 2D overlays (poly!)
#   src/plotting/specs3d.jl        - 3D PlotSpecs (FrameSpec, Frustum, camera!, LScene3D)
#   src/plotting/convert.jl        - convert_arguments for images/blobs
#   src/plotting/*_recipe.jl       - Makie recipes (matches!, cameraplot!, frameplot!)
#
# =============================================================================

using VisualGeometryToolkitCore

# Makie API — sufficient for everything below; GLMakie-specific calls live in
# the GLMakie extension only.
using Makie: Makie, @recipe, AbstractPlot, Attributes, Plot, Poly,
    camera, convert_arguments, lines!, mesh!, scatter!,
    campixel!, PlotSpec, automatic, cam3d!, DataAspect
import Makie: Fixed as MakieFixed
import Makie.SpecApi as MakieSpec

# Additional dependencies needed by plotting code
using Colors: Colors, Colorant
using GeometryBasics: GeometryBasics, Point2, Point2f, Point3f, Vec3f, Rect2,
    coordinates, widths   # `widths` — plotting/rectification.jl
using StaticArrays: StaticArrays, SVector
using StructArrays: StructArrays
using LinearAlgebra: LinearAlgebra, I

# VisualGeometryToolkitCore types
using VisualGeometryToolkitCore: AbstractDetection, Ellipse, AbstractConic3D, Circle3D
using VisualGeometryToolkitCore: Line2D, clip_to_rect
using VisualGeometryToolkitCore: LineConfidenceBand, band_boundary
using VisualGeometryToolkitCore: canonical_basis, centroid
using VisualGeometryToolkitCore: FeatureFrame
using VisualGeometryToolkitCore.Cameras: PosedCamera, CameraModel, unproject, pose
using VisualGeometryToolkitCore: EuclideanMap, has_units   # a physical pose or camera is drawn in mm (plotting/geometry.jl)
using VisualGeometryToolkitCore: is_posed   # cameraplot! draws a posed camera only (plotting/camera_recipe.jl)
using VisualGeometryToolkitCore: DEFAULT_MAX_FOV   # plot_rectification's field-of-view default (plotting/rectification.jl)

# Stub functions to extend
import VisualGeometryToolkitCore: CampixelImshow, campixel_imshow, campixel_imshow!
import VisualGeometryToolkitCore: imshow, imshow!
import VisualGeometryToolkitCore: plot_rectification   # extended in plotting/rectification.jl
using VisualGeometryToolkitCore: FramedImage   # self-placing imshow forms (plotting/image.jl)
import VisualGeometryToolkitCore: matches, matches!
import VisualGeometryToolkitCore: poly!
import VisualGeometryToolkitCore: LScene3D, camera!
import VisualGeometryToolkitCore: cameraplot, cameraplot!
import VisualGeometryToolkitCore: frameplot, frameplot!
import VisualGeometryToolkitCore: FrameSpec, Frustum

# =============================================================================
# Include Plotting Code
# =============================================================================
include("../src/plotting/geometry.jl")
include("../src/plotting/image.jl")
include("../src/plotting/rectification.jl")
include("../src/plotting/matches_recipe.jl")
include("../src/plotting/overlays.jl")
include("../src/plotting/specs3d.jl")
include("../src/plotting/convert.jl")
include("../src/plotting/camera_recipe.jl")
include("../src/plotting/frame_recipe.jl")

end # module VisualGeometryToolkitCoreMakieExt

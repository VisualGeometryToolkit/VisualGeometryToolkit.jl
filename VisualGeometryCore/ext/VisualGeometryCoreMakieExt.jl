module VisualGeometryCoreMakieExt

# =============================================================================
# Makie Extension for VisualGeometryCore
# =============================================================================
#
# This extension provides plotting functionality that works with any Makie
# backend (CairoMakie, GLMakie, WGLMakie, RPRMakie).  It activates as soon as
# Makie is loaded — no GLMakie required.  The GLMakie window helpers
# (`window!`, `closeall`) live in `VisualGeometryCoreGLMakieExt`.
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

using VisualGeometryCore

# Makie API — sufficient for everything below; GLMakie-specific calls live in
# the GLMakie extension only.
using Makie: Makie, @recipe, AbstractPlot, Attributes, Plot, Poly,
    camera, convert_arguments, image, lines!, mesh!, scatter!,
    campixel!, PlotSpec, automatic, cam3d!, DataAspect
import Makie: Fixed as MakieFixed
import Makie.SpecApi as MakieSpec

# Additional dependencies needed by plotting code
using Colors: Colors, Colorant
using GeometryBasics: GeometryBasics, Point2, Point2f, Point3f, Vec2f, Vec3f, Rect2,
    coordinates, height, origin, width, widths   # `widths` — plotting/rectification.jl
using StaticArrays: StaticArrays, SVector, StaticVector
using StructArrays: StructArrays
using LinearAlgebra: LinearAlgebra, I

# VisualGeometryCore types
using VisualGeometryCore: AbstractDetection, Disc, Ellipse, AbstractConic3D, Ellipse3D, Circle3D
using VisualGeometryCore: Line2D, clip_to_rect
using VisualGeometryCore: LineConfidenceBand, band_boundary
using VisualGeometryCore: canonical_basis, centroid
using VisualGeometryCore: FeatureFrame
using VisualGeometryCore.Cameras: Camera, CameraModel, unproject, pose
using VisualGeometryCore: EuclideanMap
using VisualGeometryCore: DEFAULT_MAX_FOV   # plot_rectification's field-of-view default (plotting/rectification.jl)

# Stub functions to extend
import VisualGeometryCore: CampixelImshow, campixel_imshow, campixel_imshow!
import VisualGeometryCore: imshow, imshow!
import VisualGeometryCore: plot_rectification   # extended in plotting/rectification.jl
using VisualGeometryCore: FramedImage   # self-placing imshow forms (plotting/image.jl)
import VisualGeometryCore: matches, matches!
import VisualGeometryCore: poly!
import VisualGeometryCore: LScene3D, camera!
import VisualGeometryCore: cameraplot, cameraplot!
import VisualGeometryCore: frameplot, frameplot!
import VisualGeometryCore: FrameSpec, Frustum

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

end # module VisualGeometryCoreMakieExt

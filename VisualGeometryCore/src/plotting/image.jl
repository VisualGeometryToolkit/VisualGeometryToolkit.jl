# =============================================================================
# Image Display Functions
# =============================================================================
#
# This module provides image display for both SpecApi and non-SpecApi workflows:
#
# SpecApi (declarative):
#   - CampixelImshow(img) -> LScene spec for composition
#
# Non-SpecApi (immediate):
#   - imshow(img) -> (Figure, Axis, Plot)
#   - campixel_imshow(img) -> (Figure, LScene, Plot)
#
# All use COLMAP coordinate convention (pixel centers at 0.5).
# =============================================================================

import Makie.SpecApi as MakieSpec

# =============================================================================
# SpecApi: CampixelImshow
# =============================================================================

# Dispatch: bounds (ranges, tuples, intervals) - forwarded directly to Spec.Image
"""
    CampixelImshow(img; width=nothing, height=nothing, ...) -> LScene spec
    CampixelImshow(x_bounds, y_bounds, img; kwargs...) -> LScene spec

Create an LScene spec with campixel! camera for displaying an image with proper orientation.

This is for **SpecApi composition** - returns a spec, not actual Makie objects.
For standard plotting that returns (Figure, Axis, Plot), use `imshow()` instead.

Uses COLMAP coordinate convention (pixel corners at integers, pixel centers at 0.5).
The first form uses default bounds (0, width) × (0, height).
The second form takes explicit bounds (ranges, tuples, or intervals).

# Arguments
- `img`: Image matrix (AbstractMatrix{<:Colorant})
- `x_bounds`: X-axis bounds (range, tuple, interval, or any type Makie.Image accepts)
- `y_bounds`: Y-axis bounds (range, tuple, interval, or any type Makie.Image accepts)

# Keyword Arguments
- `width::Union{Nothing,Int}=nothing`: LScene width. If nothing, uses image width.
- `height::Union{Nothing,Int}=nothing`: LScene height. If nothing, uses image height.
- `scenekw::NamedTuple=(camera=campixel!,)`: Scene keyword arguments.
- `imagekw::NamedTuple=(interpolate=false, rasterize=true)`: Image keyword arguments.

# Examples
```julia
using VisualGeometryCore

# Basic usage (COLMAP convention)
lscene = CampixelImshow(img)

# Add overlays using SpecApi
push!(lscene.plots, Makie.SpecApi.Scatter(x, y; color=:red))

# Compose in GridLayout
import Makie.SpecApi as S
layout = S.GridLayout([(1,1) => CampixelImshow(img1), (1,2) => CampixelImshow(img2)])
fig = plot(layout)
```

See also: `campixel_imshow` for non-SpecApi usage, `imshow` for Axis-based display.
"""
function CampixelImshow(
        x_bounds, y_bounds, img;
        width = nothing, height = nothing,
        scenekw = (camera = campixel!,),
        imagekw = (interpolate = false, rasterize = true),
        plots = PlotSpec[]
    )
    # Scene dimensions: use provided or image dimensions (width, height)
    img_width, img_height = reverse(size(img))
    scene_width = something(width, img_width)
    scene_height = something(height, img_height)

    # Forward bounds and imagekw directly to Spec.Image
    image_spec = MakieSpec.Image(x_bounds, y_bounds, transpose(img); imagekw...)

    # Use then_funcs to apply y-flip transformation after scene is created
    function flip_y_axis(lscene_block)
        scene = lscene_block.scene
        scene.transformation.scale[] = (1.0, -1.0, 1.0)
        scene.transformation.translation[] = (0.0, Float64(scene_height), 0.0)
        notify(scene.camera.projectionview)
        return
    end

    lscene = MakieSpec.LScene(
        plots = vcat([image_spec], plots),
        show_axis = false,
        width = Makie.Fixed(scene_width),
        height = Makie.Fixed(scene_height),
        tellwidth = true,
        tellheight = true,
        scenekw = scenekw
    )

    push!(lscene.then_funcs, flip_y_axis)
    return lscene
end

# Dispatch: no bounds (image fills scene) - always uses COLMAP convention
function CampixelImshow(
        img;
        width = nothing, height = nothing, scale = 1.0,
        scenekw = (camera = campixel!,),
        imagekw = (interpolate = false, rasterize = true),
        plots = PlotSpec[]
    )
    # Scene dimensions: use provided or image dimensions (width, height)
    img_width, img_height = reverse(size(img))
    scene_width = something(width, img_width)
    scene_height = something(height, img_height)

    # COLMAP convention: pixel corners at integers, bounds (0, width) × (0, height)
    x_bounds = (0, img_width)
    y_bounds = (0, img_height)

    # Create image spec with computed bounds and forward imagekw
    image_spec = MakieSpec.Image(x_bounds, y_bounds, transpose(img); imagekw...)

    # Use then_funcs to apply y-flip transformation after scene is created
    function flip_y_axis(lscene_block)
        scene = lscene_block.scene
        scene.transformation.scale[] = (scale, -scale, 1.0)
        scene.transformation.translation[] = (0.0, scale * Float64(scene_height), 0.0)
        notify(scene.camera.projectionview)
        return
    end

    lscene = MakieSpec.LScene(
        plots = vcat([image_spec], plots),
        show_axis = false,
        width = Makie.Fixed(scale * scene_width),
        height = Makie.Fixed(scale * scene_height),
        tellwidth = true,
        tellheight = true,
        scenekw = scenekw
    )

    push!(lscene.then_funcs, flip_y_axis)
    return lscene
end

"""
    CampixelImshow(position::GridPosition, args...; kwargs...)

Create an image display at a specific grid position for GridLayout composition.
"""
function CampixelImshow(position::Makie.GridPosition, args...; kwargs...)
    lscene = CampixelImshow(args...; kwargs...)
    row_span = position.span.rows
    col_span = position.span.cols
    return MakieSpec.GridPosition(row_span, col_span, lscene)
end

# =============================================================================
# Non-SpecApi: Axis-based (imshow)
# =============================================================================

"""
    imshow(img; interpolate=false, kwargs...) -> (Figure, Axis, Plot)
    imshow(position, img; interpolate=false, kwargs...) -> (Axis, Plot)

Display an image with proper orientation using a standard Axis.

Uses COLMAP coordinate convention (pixel corners at integers, pixel centers at 0.5).
Automatically handles y-axis reversal, image transpose, and aspect ratio.

# Arguments
- `img`: Image matrix (AbstractMatrix)
- `position`: GridPosition (e.g., `fig[1,1]`) for the second form

# Keyword Arguments
- `interpolate=false`: Whether to interpolate pixels
- `kwargs...`: Additional arguments forwarded to Axis (e.g., title, xlabel)

# Returns
- First form: `(Figure, Axis, Plot)` tuple
- Second form: `(Axis, Plot)` tuple

# Examples
```julia
using GLMakie
using VisualGeometryCore

# Basic usage - creates a new Figure
fig, ax, plt = imshow(img)

# Add overlays using standard Makie functions
scatter!(ax, [100, 200], [150, 250]; color=:red, markersize=10)
```
"""
function imshow(img::AbstractMatrix; interpolate = false, kwargs...)
    fig = Makie.Figure()
    ax, plt = imshow(fig[1, 1], img; interpolate = interpolate, kwargs...)
    return fig, ax, plt
end

function imshow(position, img::AbstractMatrix; interpolate = false, kwargs...)
    ax = Makie.Axis(position; aspect = DataAspect(), yreversed = true, kwargs...)
    plt = Makie.image!(ax, transpose(img); interpolate = interpolate)
    return ax, plt
end

"""
    imshow(position, img, rect; interpolate=false, kwargs...) -> (Axis, Plot)
    imshow!(ax, img, rect; interpolate=false) -> Plot

Display `img` placed at `rect` — the window of the plotting frame that the
image covers (COLMAP convention: `rect`'s origin is the image's top-left
corner, its widths the image's footprint), so images produced in a common
pixel frame at different scales and offsets share ONE set of axes. The
canonical use is an undistorted image: `undistort(camera, img, ext; scale)`
renders the window `ext = undistorted_rect(camera, size)` of the camera's
pixel frame at reduced resolution; `imshow(pos, u, ext)` draws it where it
belongs, next to the raw frame at `Rect(0, 0, W, H)`, principal point and all.
"""
function imshow(
        position, img::AbstractMatrix, rect::GeometryBasics.HyperRectangle{2};
        interpolate = false, kwargs...
    )
    ax = Makie.Axis(position; aspect = DataAspect(), yreversed = true, kwargs...)
    return ax, imshow!(ax, img, rect; interpolate)
end

# Self-placing forms: a FramedImage carries its own window of the pixel
# frame, so nothing travels beside the pixels.
imshow(position, ci::FramedImage; interpolate = false, kwargs...) =
    imshow(position, ci.data, ci.rect; interpolate, kwargs...)
imshow!(ax::Makie.Axis, ci::FramedImage; interpolate = false) =
    imshow!(ax, ci.data, ci.rect; interpolate)

function imshow!(
        ax::Makie.Axis, img::AbstractMatrix, rect::GeometryBasics.HyperRectangle{2};
        interpolate = false
    )
    ax.yreversed[] = true
    ax.aspect[] = DataAspect()
    o = GeometryBasics.origin(rect)
    w = GeometryBasics.widths(rect)
    return Makie.image!(
        ax, (Float64(o[1]), Float64(o[1] + w[1])), (Float64(o[2]), Float64(o[2] + w[2])),
        transpose(img); interpolate
    )
end

"""
    imshow!(ax, img; interpolate=false) -> Plot

Add an image to an existing Axis with proper orientation.
Sets `yreversed=true` and `aspect=DataAspect()` on the axis.
"""
function imshow!(ax::Makie.Axis, img::AbstractMatrix; interpolate = false)
    ax.yreversed[] = true
    ax.aspect[] = DataAspect()
    return Makie.image!(ax, transpose(img); interpolate = interpolate)
end

function imshow!(ax::Makie.Axis, img_obs::Makie.Observable{<:AbstractMatrix}; interpolate = false)
    ax.yreversed[] = true
    ax.aspect[] = DataAspect()
    transposed = map(transpose, img_obs)
    return Makie.image!(ax, transposed; interpolate = interpolate)
end

# =============================================================================
# Non-SpecApi: LScene-based (campixel_imshow)
# =============================================================================

"""
    campixel_imshow(img; kwargs...) -> (Figure, LScene, Plot)
    campixel_imshow(position, img; kwargs...) -> (LScene, Plot)

Display an image using an LScene with campixel! camera.

Unlike `imshow` (which uses an Axis), this creates an LScene with pixel-perfect
camera alignment. Use this when you need precise pixel coordinates for overlays.

Uses COLMAP coordinate convention (pixel corners at integers, pixel centers at 0.5).

# Keyword Arguments
- `scale=1.0`: Scale factor for display size
- `interpolate=false`: Whether to interpolate pixels

# Examples
```julia
fig, lscene, plt = campixel_imshow(img)
scatter!(lscene.scene, [100, 200], [150, 250]; color=:red)
```

See also: `CampixelImshow` for SpecApi composition, `imshow` for Axis-based display.
"""
function campixel_imshow(img::AbstractMatrix; scale = 1.0, interpolate = false)
    fig = Makie.Figure()
    lscene, plt = campixel_imshow(fig[1, 1], img; scale = scale, interpolate = interpolate)
    return fig, lscene, plt
end

function campixel_imshow(position, img::AbstractMatrix; scale = 1.0, interpolate = false)
    h, w = size(img)

    # COLMAP convention: pixel corners at integers
    x_bounds = (0, w)
    y_bounds = (0, h)

    # Create LScene with campixel camera and fixed size
    lscene = Makie.LScene(
        position;
        show_axis = false,
        width = Makie.Fixed(scale * w),
        height = Makie.Fixed(scale * h),
        scenekw = (camera = campixel!,)
    )

    # Apply y-flip transformation
    scene = lscene.scene
    scene.transformation.scale[] = (scale, -scale, 1.0)
    scene.transformation.translation[] = (0.0, scale * Float64(h), 0.0)

    # Add image
    plt = Makie.image!(scene, x_bounds, y_bounds, transpose(img); interpolate = interpolate)

    return lscene, plt
end

"""
    campixel_imshow!(lscene, img; interpolate=false) -> Plot

Add an image to an existing LScene with campixel! camera.

Note: The LScene should already have the y-flip transformation applied.
"""
function campixel_imshow!(lscene::Makie.LScene, img::AbstractMatrix; interpolate = false)
    h, w = size(img)
    x_bounds = (0, w)
    y_bounds = (0, h)
    return Makie.image!(lscene.scene, x_bounds, y_bounds, transpose(img); interpolate = interpolate)
end

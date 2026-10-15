# =============================================================================
# 2D/3D Overlay Functions — Geometric Primitives
# =============================================================================
#
# poly!(ax, shape) — add Circle, Ellipse, etc. to a Makie Axis/LScene
# Correspondence plotting is `matches!`/`matches` (matches_recipe.jl);
# VisualGeometryFeatures extends them for its feature types.

import Makie.SpecApi as MakieSpec

# =============================================================================
# 2D Geometric Primitives (poly!)
# =============================================================================

"""
    poly!(lscene, shape; kwargs...)
    poly!(lscene, shapes::AbstractVector; kwargs...)

Add geometric primitive(s) to an LScene as a Poly plot.

# Arguments
- `lscene`: The LScene to add the plot to
- `shape` or `shapes`: Geometric primitive(s) (Circle, Ellipse, Rect, etc.)

# Common Keyword Arguments
- `color=:transparent`: Fill color
- `strokecolor`: Stroke color
- `strokewidth=2.0`: Stroke width

# Examples
```julia
fig, ax, plt = imshow(img)
poly!(ax, circle; strokecolor=:red, strokewidth=3.0)
poly!(ax, [circle1, circle2]; strokecolor=:blue)
```
The target is a Makie `Axis`/`LScene` (this forwards to `Makie.poly!`). For a
SpecApi scene from [`CampixelImshow`](@ref), push a `SpecApi.Poly` onto
`scene.plots` instead.
"""
function poly!(
        ax, shape;
        color = :transparent,
        strokewidth = 2.0,
        linestyle = nothing,
        kwargs...
    )
    ls_arg = isnothing(linestyle) ? NamedTuple() : (linestyle = Makie.Linestyle(linestyle),)
    Makie.poly!(ax, shape; color, strokewidth, ls_arg..., kwargs...)
    return nothing
end

function poly!(
        ax, shapes::AbstractVector;
        color = :transparent,
        strokewidth = 2.0,
        linestyle = nothing,
        kwargs...
    )
    ls_arg = isnothing(linestyle) ? NamedTuple() : (linestyle = Makie.Linestyle(linestyle),)
    Makie.poly!(ax, shapes; color, strokewidth, ls_arg..., kwargs...)
    return nothing
end

# =============================================================================
# 3D Geometric Primitives (poly! for AbstractConic3D: Ellipse3D, Circle3D)
# =============================================================================

"""
    poly!(lscene, c::AbstractConic3D; n=64, kwargs...)
    poly!(lscene, conics::AbstractVector{<:AbstractConic3D}; n=64, kwargs...)

Add 3D conic(s) (Ellipse3D, Circle3D) to an LScene3D as line loops.
"""
poly!(lscene, c::AbstractConic3D; kwargs...) = poly!(lscene, [c]; kwargs...)

function poly!(
        lscene, conics::AbstractVector{<:AbstractConic3D};
        n::Int = 64,
        strokecolor = :blue,
        strokewidth = 2.0,
        kwargs...
    )
    for c in conics
        coords = coordinates(c, n)
        push!(lscene.plots, MakieSpec.Lines(Point3f.(coords); color = strokecolor, linewidth = strokewidth, kwargs...))
    end
    return nothing
end

# =============================================================================
# ellipses! — batched elliptical-boundary overlay (THE frame/blob inspector)
# =============================================================================

"""
    ellipses!(ax, shapes; cutoff=1.0, color=nothing, colormap=:viridis,
              strokecolor=:cyan, strokewidth=1.5, kwargs...)

Draw a vector of elliptical shapes as one batched boundary overlay.

`shapes` may hold `Disc`, `EllipticFrame`, `Ellipse`, or `Circle` (mixed is
fine); `Disc`/`EllipticFrame` are converted via `region(shape, cutoff)`,
`Ellipse`/`Circle` are drawn as-is. One `poly!` call regardless of count —
never loop `poly!` per shape.

Color-by-value: supply `color = values::AbstractVector{<:Real}` (one per shape,
e.g. scale ratios or responses) to stroke each ellipse through `colormap`;
otherwise every ellipse uses `strokecolor`.

Canonical inspection primitive: `VisualGeometryFeatures.detections!` and
BlobBoards' pipeline `inspect` panels compose this.
"""
function VisualGeometryCore.ellipses!(
        ax, shapes::AbstractVector;
        cutoff::Real = 1.0,
        color = nothing,
        colormap = :viridis,
        strokecolor = :cyan,
        strokewidth = 1.5,
        kwargs...
    )
    isempty(shapes) && return nothing
    to_b(s::Union{Ellipse, GeometryBasics.Circle}) = s
    to_b(s) = region(s, cutoff)
    bs = [to_b(s) for s in shapes]
    if color === nothing
        Makie.poly!(ax, bs; color = :transparent, strokecolor, strokewidth, kwargs...)
    else
        Makie.poly!(
            ax, bs; color = :transparent, strokecolor = collect(color),
            colormap, strokewidth, kwargs...
        )
    end
    return nothing
end

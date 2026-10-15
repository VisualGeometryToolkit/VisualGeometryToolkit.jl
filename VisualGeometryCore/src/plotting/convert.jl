# =============================================================================
# Makie convert_arguments for VisualGeometryCore Types
# =============================================================================
#
# Provides convert_arguments methods so Makie can directly plot:
#   - Image + blobs: plot(img, blobs) → the image in COLMAP coordinates with blob overlays
#   - Blobs: plot(blobs) → circles at blob locations
#   - Ellipses: poly!(ax, ellipse) → polygon approximation
#
# =============================================================================

import Makie.SpecApi as MakieSpec

# =============================================================================
# Type Union for Blob-like Objects
# =============================================================================

# BlobLike covers both Detection-based (AbstractDetection, including detections
# of an Uncertain frame) and FeatureFrame-based types (Disc). This allows
# plotting code to work with either type; `Circle` reads each one's frame.
const BlobLike = Union{AbstractDetection, FeatureFrame}

# =============================================================================
# Image convert_arguments
# =============================================================================

# No conversion for a bare image matrix: `convert_arguments(::Type{<:AbstractPlot},
# ::AbstractMatrix{<:Colorant})` extended a Makie function on only foreign types,
# so loading this extension changed `plot(img)` for every Makie user. Images go
# through the owned entry points — `imshow`, `CampixelImshow` — or carry blobs
# (owned types) below.

# =============================================================================
# Image + Blobs convert_arguments
# =============================================================================

Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}, ::Vector{<:BlobLike}) =
    (:interpolate, :color, :extent, :linewidth)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, img, blobs; kwargs...)

Convert image and blobs to composed GridLayout with overlay.

# Examples
```julia
img = testimage("cameraman")
blobs = [Disc(Point2(100.0, 100.0), 20.0)]
fig, ax, pl = plot(img, blobs; color=:red, extent=3.0)
```
"""
function Makie.convert_arguments(
        ::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant},
        blobs::Vector{<:BlobLike};
        interpolate = false, color = :green, extent::Float64 = 3.0,
        linewidth::Float64 = 1.0
    )
    lscene = CampixelImshow(img; imagekw = (interpolate = interpolate, rasterize = true))
    circles = Circle.(blobs, Ref(extent))
    blob_spec = MakieSpec.Poly(circles, color = :transparent, strokecolor = color, strokewidth = linewidth)
    push!(lscene.plots, blob_spec)
    return MakieSpec.GridLayout([lscene]; rowgaps = Makie.Fixed(0), colgaps = Makie.Fixed(0))
end

# =============================================================================
# Image + Detections + Ground Truth convert_arguments
# =============================================================================

Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}, ::Vector{<:BlobLike}, ::Vector{<:BlobLike}) =
    (:interpolate, :detection_color, :ground_truth_color, :extent, :linewidth)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, img, detections, ground_truth; kwargs...)

Convert image, detected blobs, and ground truth blobs to composed GridLayout.

# Examples
```julia
fig, ax, pl = plot(img, detections, ground_truth; extent=3.0)
```
"""
function Makie.convert_arguments(
        ::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant},
        detections::Vector{<:BlobLike}, ground_truth::Vector{<:BlobLike};
        interpolate = false,
        detection_color = :blue, ground_truth_color = :green,
        extent::Float64 = 3.0,
        linewidth::Float64 = 1.0
    )
    lscene = CampixelImshow(img; imagekw = (interpolate = interpolate, rasterize = true))

    # Add ground truth blobs
    gt_circles = Circle.(ground_truth, Ref(extent))
    gt_spec = MakieSpec.Poly(
        gt_circles, color = :transparent,
        strokecolor = ground_truth_color,
        strokewidth = linewidth
    )
    push!(lscene.plots, gt_spec)

    # Add detection blobs
    det_circles = Circle.(detections, Ref(extent))
    det_spec = MakieSpec.Poly(
        det_circles, color = :transparent,
        strokecolor = detection_color,
        strokewidth = linewidth
    )
    push!(lscene.plots, det_spec)

    return MakieSpec.GridLayout([lscene]; rowgaps = Makie.Fixed(0), colgaps = Makie.Fixed(0))
end

# =============================================================================
# Blobs-only convert_arguments
# =============================================================================

Makie.used_attributes(::Type{<:Plot}, ::Vector{<:BlobLike}) =
    (:color, :extent, :linewidth)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, blobs; kwargs...)

Convert blobs to PlotSpec for Makie recipe integration.
Allows: plot(blobs; color=:red, extent=2.0)
"""
function Makie.convert_arguments(
        ::Type{<:AbstractPlot}, blobs::Vector{<:BlobLike};
        color = :green, extent::Float64 = 3.0,
        linewidth::Float64 = 1.0
    )
    circles = Circle.(blobs, Ref(extent))
    return MakieSpec.Poly(
        circles, color = :transparent,
        strokecolor = color, strokewidth = linewidth
    )
end

# DiscDetection-specific converters moved to VisualGeometryFeatures

# =============================================================================
# Ellipse convert_arguments (for poly!)
# =============================================================================

"""
    Makie.convert_arguments(::Type{<:Poly}, ellipse::Ellipse)

Convert Ellipse to polygon representation for Makie's poly plotting.

```julia
e = Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)
poly!(ax, e, color=:red, strokewidth=2)
```
"""
function Makie.convert_arguments(P::Type{<:Poly}, ellipse::Ellipse)
    return convert_arguments(P, GeometryBasics.Polygon(GeometryBasics.coordinates(ellipse, 64)))
end

"""
    Makie.convert_arguments(::Type{<:Poly}, ellipses::Vector{<:Ellipse})

Convert vector of Ellipses to polygon representations.

```julia
ellipses = [Ellipse(Point2(i*3.0, 0.0), 2.0, 1.0, π/6*i) for i in 1:5]
poly!(ax, ellipses, color=:transparent, strokecolor=1:5, strokewidth=2, strokecolormap=:viridis)
```
"""
function Makie.convert_arguments(P::Type{<:Poly}, ellipses::Vector{<:Ellipse})
    polys = [GeometryBasics.Polygon(GeometryBasics.coordinates(e, 64)) for e in ellipses]
    return convert_arguments(P, polys)
end

# =============================================================================
# Line2D + Rect2: clip to rect, plot as polyline / NaN-separated segments
# =============================================================================

"""
    Makie.convert_arguments(::Type{<:Lines}, l::Line2D, rect::Rect2)

Clip the homogeneous 2D line `l` to `rect` (Liang-Barsky) and return the
segment as a two-element `Point2f` vector for `lines!`. Empty vector if
the line misses the rect.

```julia
lines!(ax, F(u1), img2_rect; color=:red)   # one epipolar line
```
"""
function Makie.convert_arguments(::Type{<:Makie.Lines}, l::Line2D, rect::Rect2)
    seg = clip_to_rect(l, rect)
    pts = seg === nothing ? Point2f[] : Point2f[seg.points[1], seg.points[2]]
    return (pts,)
end

"""
    Makie.convert_arguments(::Type{<:Lines}, ls, rect::Rect2)

Clip each line in `ls` to `rect` and join the resulting segments with
`Point2f(NaN,NaN)` separators so a single `lines!` call draws them all.
Lines that miss the rect are omitted.

```julia
lines!(ax, F.(u1_points), img2_rect; color=:red)   # many epipolar lines
```
"""
function Makie.convert_arguments(
        ::Type{<:Makie.Lines},
        ls::AbstractVector{<:Line2D}, rect::Rect2
    )
    pts = Point2f[]
    nan = Point2f(NaN, NaN)
    for l in ls
        seg = clip_to_rect(l, rect)
        seg === nothing && continue
        isempty(pts) || push!(pts, nan)
        push!(pts, seg.points[1])
        push!(pts, seg.points[2])
    end
    return (pts,)
end

# =============================================================================
# LineConfidenceBand + Rect2: epipolar confidence band as two boundary branches
# =============================================================================

"""
    Makie.convert_arguments(::Type{<:Lines}, b::LineConfidenceBand, rect::Rect2)

Sample the two boundary branches of the epipolar confidence band inside
`rect` (`band_boundary`) and join them with `Point2f(NaN,NaN)` separators so
one `lines!` call draws the whole band outline. Unbounded sections (the
near-degeneracy blow-up) break the polyline rather than clamping.

```julia
band = epipolar_band(F, Σθ, u1; level=0.95)
lines!(ax, F(u1), img2_rect; color=:black)      # nominal epipolar line
lines!(ax, band, img2_rect; color=:crimson)     # 95% band boundary
```
"""
function Makie.convert_arguments(
        ::Type{<:Makie.Lines},
        b::LineConfidenceBand, rect::Rect2
    )
    upper, lower = band_boundary(b, rect)
    pts = Point2f[]
    append!(pts, Point2f.(upper))
    isempty(pts) || push!(pts, Point2f(NaN, NaN))
    append!(pts, Point2f.(lower))
    return (pts,)
end

# =============================================================================
# 3D conics as Lines (sampled boundary loop)
# =============================================================================

"""
    Makie.convert_arguments(::Type{<:Lines}, c::AbstractConic3D; n=64)

Plot a 3D conic (`Circle3D`, `Ellipse3D`) directly as a closed line loop.

```julia
lines!(scene, circle3d; color=:blue)
lines!(scene, ellipse3d)
```
"""
function Makie.convert_arguments(::Type{<:Makie.Lines}, c::AbstractConic3D; n::Int = 64)
    return (Point3f.(coordinates(c, n)),)
end

# =============================================================================
# Arrows as (origin, direction) and as 3D-conic normal
# =============================================================================

# No (Point, Vec) arrow conversion of our own: it extended Makie on only foreign
# types (GeometryBasics' Point/Vec). Makie's own `arrows2d!`/`arrows3d!` accept a
# single origin/direction pair as well as vectors of them.

"""
    Makie.convert_arguments(::Type{<:Arrows3D}, c::Circle3D)

Plot a 3D circle's normal as an arrow at the centroid, with length equal
to the circle's radius. For a custom length, use the explicit
`(origin, vec)` form: `arrows3d!(scene, c.center, len * normal(c))`.

```julia
arrows3d!(scene, circle3d; color=:magenta)
```
"""
function Makie.convert_arguments(::Type{<:Makie.Arrows3D}, c::Circle3D)
    return ([Point3f(c.center...)], [Vec3f((c.radius * normal(c))...)])
end

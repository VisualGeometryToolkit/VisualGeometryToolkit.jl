# ==============================================================================
# Matches Recipe
# ==============================================================================
#
# Makie recipe for visualizing point correspondences on a single axis.
# Unifies weighted-alpha overlays (pose verification) and colormap-indexed
# match visualization (feature matching) into one recipe.
#
#   matches!(ax, src_pts, tgt_pts; weights=...)    # alpha-weighted
#   matches!(ax, src_pts, tgt_pts; colormap=:hsv)  # per-match color
#   matches!(ax, src_pts, tgt_pts; colors=...)      # explicit colors
#
# ==============================================================================

"""
    matches(src_pts, tgt_pts; kwargs...)
    matches!(ax, src_pts, tgt_pts; kwargs...)

Visualize point correspondences on a single axis.

Source and target points are drawn as markers connected by line segments.
Three coloring modes are supported (checked in priority order):

1. `colors`: Explicit per-match color vector — each match (source marker,
   target marker, line) uses the same color for visual correspondence.
2. `colormap`: Per-match indexed colors from a colormap — same visual
   grouping as `colors`, auto-generated from the map.
3. `weights`: Alpha-weighted fixed colors — source/target/line use distinct
   fixed colors (`src_color`, `tgt_color`, `line_color`) with per-match
   alpha from `weights` (e.g., posterior inlier probabilities).
4. None set: Full opacity fixed colors.

# Arguments
- `src_pts::AbstractVector{<:Point2}`: Source (e.g., projected 3D) points
- `tgt_pts::AbstractVector{<:Point2}`: Target (e.g., detected 2D) points

# Attributes
- `weights=nothing`: Per-correspondence α ∈ [0,1] (mode 3)
- `colormap=nothing`: Colormap symbol for per-match colors (mode 2)
- `colors=nothing`: Explicit per-match color vector (mode 1)
- `src_color=:cyan`: Source marker color (weight mode only)
- `tgt_color=:lime`: Target marker color (weight mode only)
- `line_color=:yellow`: Line color (weight mode only)
- `src_marker=:cross`: Source marker shape
- `tgt_marker=:circle`: Target marker shape
- `src_markersize=12`: Source marker size
- `tgt_markersize=10`: Target marker size
- `linewidth=0.5`: Connection line width
- `show_lines=true`: Show connection lines
- `show_markers=true`: Show markers

# Examples
```julia
# Alpha-weighted (pose verification — inlier probabilities)
matches!(ax, proj_pts, det_pts; weights)

# Colormap-indexed (feature matching — identify correspondences)
matches!(ax, src_pts, tgt_pts; colormap=:hsv)

# Explicit colors (e.g., green=inlier, red=outlier)
colors = [m ? :green : :red for m in inlier_mask]
matches!(ax, src_pts, tgt_pts; colors)
```
"""
@recipe(Matches, src_pts, tgt_pts) do scene
    Attributes(
        weights = nothing,
        colormap = nothing,
        colors = nothing,
        src_color = :cyan,
        tgt_color = :lime,
        line_color = :yellow,
        src_marker = :cross,
        tgt_marker = :circle,
        src_markersize = 12,
        tgt_markersize = 10,
        linewidth = 0.5,
        show_lines = true,
        show_markers = true,
    )
end

function Makie.plot!(p::Matches)
    src = p.src_pts[]
    tgt = p.tgt_pts[]
    n = length(src)

    cmap = p.colormap[]
    match_colors = p.colors[]
    weights = p.weights[]

    if !isnothing(match_colors)
        # Mode 1: Explicit per-match colors
        _matches_colored!(p, src, tgt, match_colors, n)
    elseif !isnothing(cmap)
        # Mode 2: Colormap-indexed per-match colors
        cs = [Makie.cgrad(cmap, n, categorical = true)[i] for i in 1:n]
        _matches_colored!(p, src, tgt, cs, n)
    else
        # Mode 3/4: Alpha-weighted fixed colors
        α = something(weights, ones(Float32, n))

        if p.show_markers[]
            Makie.scatter!(
                p, src;
                color = [(p.src_color[], α[i]) for i in 1:n],
                markersize = p.src_markersize, marker = p.src_marker,
                visible = p.show_markers
            )
            Makie.scatter!(
                p, tgt;
                color = [(p.tgt_color[], α[i]) for i in 1:n],
                markersize = p.tgt_markersize, marker = p.tgt_marker,
                visible = p.show_markers
            )
        end

        if p.show_lines[]
            segments = [pt for st in zip(src, tgt) for pt in st]
            seg_colors = [(p.line_color[], α[i]) for i in 1:n for _ in 1:2]
            Makie.linesegments!(
                p, segments; color = seg_colors,
                linewidth = p.linewidth, visible = p.show_lines
            )
        end
    end

    return p
end

# Shared: all elements (src marker, tgt marker, line) share the same color per match
function _matches_colored!(p, src, tgt, colors, n)
    if p.show_markers[]
        Makie.scatter!(
            p, src;
            color = colors,
            markersize = p.src_markersize, marker = p.src_marker,
            visible = p.show_markers
        )
        Makie.scatter!(
            p, tgt;
            color = colors,
            markersize = p.tgt_markersize, marker = p.tgt_marker,
            visible = p.show_markers
        )
    end

    if p.show_lines[]
        segments = [pt for st in zip(src, tgt) for pt in st]
        seg_colors = [colors[i] for i in 1:n for _ in 1:2]
        Makie.linesegments!(
            p, segments; color = seg_colors,
            linewidth = p.linewidth, visible = p.show_lines
        )
    end

    return nothing
end

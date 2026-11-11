# =============================================================================
# The station fan of a floor plan as TikZ: one annular wedge per cell of
# `station_cells` (station_cells.jl), coloured by one of the cell's statistics.
# BlobBoards bmvc2026 (scripts/plots/error_distributions.jl,
# `write_floorplan_tikz`) and CaliBlobs icra2026 (src/floorplan_tikz.jl,
# `write_floorplan_fan`, a verbatim copy) each wrote it.
# =============================================================================

"""
    write_station_fan(colormap, io::IO, cells; value, span, logarithmic = false, reverse = false,
                      comment = "station fan", ring_labels = true) -> io
    write_station_fan(colormap, path, cells; <the same keywords>) -> path

A floor plan's station fan as a bare TikZ picture (no title): per cell of
`cells` ([`station_cells`](@ref): `angle` in degrees, `dist` in mm, `n` the
placed count) an annular wedge 10° wide (0° up, clockwise) from `dist - 300`
to `dist + 300` mm, drawn in metres, its geometry scaled by the `\\fpscale`
macro (`\\providecommand`ed to 1; fonts are unaffected), the picture anchored
at its top (`baseline=(current bounding box.north)`, level with a colorbar
beside it), the rig a black square at the origin, and with `ring_labels` the
rings labelled 1, 2 and 3 m (`\\scriptsize`, not scaled: drop them below
about `\\fpscale` 0.3).

A cell with `n == 0` is grey (`gray!30`): a hole in the fan means no
detection. Any other cell is filled with `colormap(t)` — a colour with fields
`r`, `g`, `b` in [0, 1], written at 4 decimals (Makie's or PlotUtils'
`t -> get(cgrad(:RdYlGn), t)`) — where `t` is the cell's statistic `value` (a
field name: `:rate`, `:med`, `:medr`) placed in `span = (lo, hi)` (in the
statistic's units) and clamped to [0, 1], on a log10 axis when `logarithmic`,
and `1 - t` when `reverse` (a statistic that is better low, an error).
`comment` is the file's first line, after `% `. Requires, in the document,
`\\usepackage{tikz}`.
"""
function write_station_fan(colormap, io::IO, cells; value::Symbol, span::Tuple{Real, Real},
        logarithmic::Bool = false, reverse::Bool = false, comment::AbstractString = "station fan",
        ring_labels::Bool = true)
    lo, hi = logarithmic ? (log10(span[1]), log10(span[2])) : (Float64(span[1]), Float64(span[2]))
    hi > lo || throw(ArgumentError("write_station_fan: span $(span) is empty"))
    placed(v) = clamp(((logarithmic ? clamp(log10(v), lo, hi) : v) - lo) / (hi - lo), 0, 1)
    rgb(t) = let c = colormap(t)
        "{rgb,1:red,$(round(c.r; digits = 4));green,$(round(c.g; digits = 4));blue,$(round(c.b; digits = 4))}"
    end
    println(io, "% ", comment)
    println(io, "\\providecommand{\\fpscale}{1}")
    println(io, "\\begin{tikzpicture}[x=\\fpscale cm, y=\\fpscale cm, baseline=(current bounding box.north)]")
    for c in cells
        r1, r2 = (c.dist - 300) / 1000, (c.dist + 300) / 1000
        φ1, φ2 = 95 - c.angle, 85 - c.angle      # 0° up, clockwise → TikZ's mathematical angles
        t = c.n == 0 ? nothing : placed(getproperty(c, value))
        fill = t === nothing ? "gray!30" : rgb(reverse ? 1 - t : t)
        println(io, "  \\fill[fill=$(fill), draw=black!30, line width=0.2pt] " *
                    "($(round(φ1; digits = 2)):$(round(r1; digits = 3))) arc[start angle=$(round(φ1; digits = 2)), end angle=$(round(φ2; digits = 2)), radius=$(round(r1; digits = 3))] -- " *
                    "($(round(φ2; digits = 2)):$(round(r2; digits = 3))) arc[start angle=$(round(φ2; digits = 2)), end angle=$(round(φ1; digits = 2)), radius=$(round(r2; digits = 3))] -- cycle;")
    end
    println(io, "  \\fill[black] (-0.09, -0.09) rectangle (0.09, 0.09);")
    if ring_labels
        for metres in (1, 2, 3)
            println(io, "  \\node[font=\\scriptsize\\bfseries, text=black!60] at ($(metres), -0.22) {$(metres)\\,m};")
        end
    end
    println(io, "\\end{tikzpicture}")
    return io
end

function write_station_fan(colormap, path::AbstractString, cells; kwargs...)
    text = sprint(io -> write_station_fan(colormap, io, cells; kwargs...))
    write(path, text)
    return path
end

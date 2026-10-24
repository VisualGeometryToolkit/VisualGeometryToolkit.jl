# =============================================================================
# src/render/faceted.jl — one pgfplots faceted-groupplot emitter.
#
# Consolidates the four hand-rolled, byte-similar emission blocks of the
# faceted renderers (render_basin_faceted / render_miscal_faceted /
# render_sb_faceted / render_anchoring_paired): the groupplot preamble, the
# `legend to name` spec, the per-arm `\addplot` line and the legend-image
# line. The axis blocks of a paper's panels are its `PaperStyle`'s `axes`.
#
# BYTE-IDENTITY CONTRACT: these helpers reproduce the historical emitters'
# output EXACTLY (the committed paper panels are byte-compared) — change the
# grammar here only together with a re-render of every faceted panel.
# =============================================================================

"""
    open_faceted_groupplot(io; cols, rows, width, height, opts,
                           hsep = "6pt", vsep = "12pt", edge_labels = true)

Emit the shared groupplot preamble:

    \\begin{groupplot}[
      group style={group size=<cols> by <rows>, horizontal sep=<hsep>[, vertical sep=<vsep>,
        xticklabels at=edge bottom, xlabels at=edge bottom]},
      scale only axis, width=<width>, height=<height>,
      <opts[1]>,
      ...
      <opts[end]>]

`opts` are the remaining option LINES (each printed at 2-space indent; the
last gets the closing `]`). `edge_labels=false` drops the edge-tick lines AND
the vertical sep (the 1-row paired layout).
"""
function open_faceted_groupplot(io::IO; cols::Integer, rows::Integer,
                                width::AbstractString, height::AbstractString,
                                opts::AbstractVector{<:AbstractString},
                                hsep::AbstractString = "6pt",
                                vsep::AbstractString = "12pt",
                                edge_labels::Bool = true)
    println(io, raw"\begin{groupplot}[")
    if edge_labels
        println(io, "  group style={group size=", cols, " by ", rows,
                ", horizontal sep=", hsep, ", vertical sep=", vsep, ",")
        println(io, raw"    xticklabels at=edge bottom, xlabels at=edge bottom},")
    else
        println(io, "  group style={group size=", cols, " by ", rows,
                ", horizontal sep=", hsep, "},")
    end
    println(io, "  scale only axis, width=", width, ", height=", height, ",")
    for (i, l) in enumerate(opts)
        println(io, "  ", l, i == length(opts) ? "]" : ",")
    end
    return io
end

"""
    faceted_legend_opts(name; columns = 4) -> String

The shared `legend to name` axis-option suffix (prepend to the first cell's
`\\nextgroupplot[...]` options): one shared legend drawn by the host via
`\\pgfplotslegendfromname{<name>}`.
"""
faceted_legend_opts(name::AbstractString; columns::Integer = 4) =
    ", legend to name=$name, legend columns=$columns, legend cell align=left, " *
    raw"legend style={draw=none, font=\footnotesize, /tikz/every even column/.append style={column sep=8pt}}"

"""
    arm_plot_opts(color, marker; dashed = false, line_width = "0.8pt",
                  mark_size = "1.6pt") -> String

The shared per-arm plot options: `color=…, mark=…, mark
options={solid,fill=…}, line width=…, mark size=…[, dashed]`.
"""
arm_plot_opts(color::AbstractString, marker::AbstractString; dashed::Bool = false,
              line_width::AbstractString = "0.8pt",
              mark_size::AbstractString = "1.6pt") =
    "color=$color, mark=$marker, mark options={solid,fill=$color}, " *
    "line width=$line_width, mark size=$mark_size" * (dashed ? ", dashed" : "")

"""
    arm_addplot(io, color, marker, dashed, coords; line_width = "0.8pt")

One per-arm curve: `\\addplot[<arm_plot_opts>, forget plot] coordinates
{<coords>};` (legends are image-only — see `arm_legendimage`).
"""
arm_addplot(io::IO, color, marker, dashed::Bool, coords::AbstractString;
            line_width::AbstractString = "0.8pt") =
    println(io, "\\addplot[", arm_plot_opts(color, marker; dashed, line_width),
            ", forget plot] coordinates {", coords, "};")

"""
    arm_legendimage(io, color, marker, dashed, label;
                    line_width = "0.8pt", mark_size = nothing)

One legend entry: `\\addlegendimage{color=…, mark=…, mark
options={solid,fill=…}, line width=…[, mark size=…][, dashed]}\\addlegendentry{…}`.
The faceted 3×2 panels omit `mark size` in legend images (historical bytes);
the paired layout forwards its plot options (`mark_size = "1.6pt"`).
"""
function arm_legendimage(io::IO, color, marker, dashed::Bool, label;
                         line_width::AbstractString = "0.8pt",
                         mark_size::Union{Nothing,AbstractString} = nothing)
    opts = mark_size === nothing ?
        "color=$color, mark=$marker, mark options={solid,fill=$color}, line width=$line_width" *
        (dashed ? ", dashed" : "") :
        arm_plot_opts(color, marker; dashed, line_width, mark_size)
    println(io, "\\addlegendimage{", opts, "}\\addlegendentry{", label, "}")
end

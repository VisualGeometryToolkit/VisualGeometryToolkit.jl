# =============================================================================
# src/render/paper_style.jl — PaperStyle: a paper's styling as ONE explicit
# value, given to every renderer that draws arms or paper colours.
#
# It replaces the registries a paper used to mutate at include time (the
# palette, the summary-panel STYLE rows, the faceted ARM_REGISTRY, the selector
# label tables, the frozen axis blocks): a renderer reads only the style it is
# handed, so two papers (or two variants of one) render side by side in one
# session, and nothing is styled by load order.
# =============================================================================

"""
    PaperStyle(; colors = Pair{String,String}[], rows = [], arms = Dict(),
               grid, zeta_x, zeta_med_y, zeta_maa_y, med_ylabel, maa_ylabel)

A paper's figure style, built once by the paper (accv2026 builds it from its
palette, arm table and figure-style blocks) and given as the first argument
of every renderer that draws arms or paper colours.

Fields:
- `colors::Dict{String,String}`: colour NAME => hex `"RRGGBB"` — the paper
  palette ([`default_palette`](@ref) is the house base) plus every arm colour
  whose record carries a hex; emitted by [`paper_color_defs`](@ref).
- `rows::Vector{Tuple{String,String,String,String,Bool}}`: the summary-panel
  rows of [`build_axis`](@ref), in legend order: `(method key, display label,
  colour name, mark, dashed)`.
- `arms::Dict{String,NamedTuple}`: the faceted-arm style records,
  [`arm_style_records`](@ref) — family, path, scale, label, colour, hex,
  marker, dashed, shade.
- `axes::NamedTuple`: the paper's axis blocks of the faceted emitters —
  `grid` (the grid options spliced into the budget/basin/time panels; `""` is
  the house rule, no grid), `zeta_x` (x block of the ζ panels), `zeta_med_y`
  (a function `(ymin; ymax, ytick_spacing) -> String`, the median row's y
  block), `zeta_maa_y` (the mAA row's y block), `med_ylabel` and `maa_ylabel`
  (the two rows' y labels).

A paper whose figures were frozen at an earlier style (the ACCV panels are
byte-compared) supplies those exact blocks here; nothing in this package
defaults to them.
"""
struct PaperStyle
    colors::Dict{String,String}
    rows::Vector{Tuple{String,String,String,String,Bool}}
    arms::Dict{String,NamedTuple}
    axes::@NamedTuple{grid::String, zeta_x::String, zeta_med_y::Function, zeta_maa_y::String,
                      med_ylabel::String, maa_ylabel::String}
end

function PaperStyle(; colors = Pair{String,String}[], rows = Tuple{String,String,String,String,Bool}[],
                    arms::AbstractDict = Dict{String,NamedTuple}(),
                    grid::AbstractString, zeta_x::AbstractString, zeta_med_y::Function,
                    zeta_maa_y::AbstractString, med_ylabel::AbstractString, maa_ylabel::AbstractString)
    cols = Dict{String,String}(String(n) => String(h) for (n, h) in colors)
    for r in values(arms)
        r.hex === nothing && continue
        h = get(cols, r.color, r.hex)
        h == r.hex || throw(ArgumentError("PaperStyle: colour $(r.color) is $h in the palette and $(r.hex) on an arm"))
        cols[r.color] = r.hex
    end
    return PaperStyle(cols, Tuple{String,String,String,String,Bool}[Tuple(r) for r in rows],
                      Dict{String,NamedTuple}(String(k) => v for (k, v) in arms),
                      (; grid = String(grid), zeta_x = String(zeta_x), zeta_med_y,
                       zeta_maa_y = String(zeta_maa_y), med_ylabel = String(med_ylabel),
                       maa_ylabel = String(maa_ylabel)))
end

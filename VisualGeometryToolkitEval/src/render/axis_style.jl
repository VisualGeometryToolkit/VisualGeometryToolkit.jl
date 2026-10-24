# =============================================================================
# src/render/axis_style.jl — the house axis-style source.
#
# Every figure in every paper renders through this module; hand-styled axis
# options in paper scripts are a DEFECT (enforced by the contract rules,
# Experiments `contract_violations`). Paper directories supply COLORS and ARM
# ROWS (figure_style.jl grammar, in their PaperStyle); new figures define no
# axis styling of their own.
#
# A paper whose panels were frozen at an earlier style (the ACCV faceted ζ
# blocks, byte-compared) carries those blocks in its own `PaperStyle`; this
# file holds the HOUSE conventions (maintainer-conformed 2026-07-19) for all
# NEW figures (those after ACCV 2026):
#       - NO grid lines of any kind;
#       - NO scientific notation on tick labels (log axes keep log spacing,
#         plain-number labels via `log ticks with fixed point`);
#       - error panels LINEAR y with a truncated display range (curves may
#         exit the top — the truncation is stated in the CAPTION, never
#         inside the plot);
#       - NO text inside the axes — markers only; generators emit a
#         companion caption snippet next to the figure fragment;
#       - one color + one legend name per arm.
#
# DEPENDENCY-FREE + include-guarded: external-arm generators that cannot load
# the eval package (e.g. a PythonCall environment) `include` THIS
# file directly — same file, same symbols, one source. Docstrings inside the
# guard are attached with `@doc` (a bare string inside an `if` documents
# nothing).
# =============================================================================
if !@isdefined(HOUSE_AXIS_STYLE_LOADED)

const HOUSE_AXIS_COMMON = "grid=none, log ticks with fixed point, " *
    "tick label style={font=\\scriptsize}, label style={font=\\small}, " *
    "legend cell align=left, legend style={font=\\scriptsize, draw=none, fill=none}"
@doc """
    HOUSE_AXIS_COMMON

The common axis-option string every NEW (post-ACCV) panel starts from:
no grids, plain-number tick labels on log axes, house fonts and legend
style. Panel-specific options (labels, limits, modes) are appended after.
""" HOUSE_AXIS_COMMON

const HOUSE_LINE_WIDTH = "thick"
@doc "House default line width for curve plots." HOUSE_LINE_WIDTH

const HOUSE_MARK_SIZE = "1.6pt"
@doc "House default marker size for curve plots." HOUSE_MARK_SIZE

color_defs(pairs; command::AbstractString = "definecolor") =
    join(["\\$(command){$(n)}{HTML}{$(h)}" for (n, h) in pairs], "\n")
@doc """
    color_defs(pairs; command = "definecolor") -> String

The colour-definition emitter: one `\\<command>{<name>}{HTML}{<hex>}` line per
`name => "RRGGBB"` pair, newline-separated (no trailing newline). `command` is
`"definecolor"` or `"providecolor"` (idempotent; a host's own definition wins).
""" color_defs

house_colors(pairs) = color_defs(pairs) * "\n"
@doc """
    house_colors(pairs::Vector{<:Pair}) -> String

`\\definecolor` block ([`color_defs`](@ref)) from `name => "RRGGBB"` pairs,
newline-terminated — the one way a house figure loads its color registry
(names come from the paper's arm/legend vocabulary; one color per arm).
""" house_colors

const HOUSE_AXIS_STYLE_LOADED = true
end

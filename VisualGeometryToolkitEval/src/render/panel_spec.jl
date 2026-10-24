# =============================================================================
# panel_spec.jl — PanelSpec: The display-option record for the figure renderers.
#
# Every cosmetic/display choice that used to be threaded as parallel kwarg
# lists through render_bundle → _write_lo_panels → build_axis lives here as
# one field. Compute knobs (nboot, xgrid, design, panel_name) stay
# as function arguments — a PanelSpec describes how a panel LOOKS, never what
# it computes.
#
# Adding a display knob = one field here + one use-site in the renderer.
# The `PanelSpec(base; overrides...)` copy constructor lets call sites derive
# variants (e.g. render_bundle resolving the mode-default xlabel) without
# mutation.
#
# Field defaults are the historical kwarg defaults of build_axis — call sites
# that need a different default (e.g. _write_lo_panels' legend_name="lolegend")
# set it explicitly in their backward-compat wrappers.
#
# Fields are deliberately loosely typed (Union{Nothing,Real}, AbstractVector):
# values must flow into the .tex emitters EXACTLY as given — an Int 0 prints
# `ymin=0`, a Float64 0.0 prints `ymin=0.0`, and the committed artifacts are
# byte-compared. This is a config record, never hot-loop data.
# =============================================================================

Base.@kwdef struct PanelSpec
    # --- geometry -------------------------------------------------------------
    width::AbstractString  = raw"\linewidth"
    height::AbstractString = raw"0.618\linewidth"     # golden ratio 1/phi
    # --- y axis ---------------------------------------------------------------
    ymin::Union{Nothing,Real}       = nothing         # nothing ⇒ pgfplots auto
    ymax::Union{Nothing,Real}       = nothing
    ytick_spacing::Union{Nothing,Real} = nothing         # nothing ⇒ panel default + adaptive refine
    y_unit_degree::Bool             = true            # degree on tick numbers (house convention)
    ylog::Bool                      = false           # log-y variant of the panel
    show_ci::Bool                   = true            # fillbetween CI bands
    # --- x axis ---------------------------------------------------------------
    xlabel::Union{Nothing,AbstractString}     = nothing  # nothing ⇒ renderer context default
    xmode::AbstractString                     = "log"
    xtick::Union{Nothing,AbstractVector}      = nothing  # ticks AT the checkpoints
    xticklabels::Union{Nothing,AbstractVector} = nothing
    xticks_fixed::Bool                        = false    # plain decimals on log x
    tight_x::Bool                             = false    # clamp axis to the data span
    # --- decoration -----------------------------------------------------------
    title::Union{Nothing,AbstractString}        = nothing
    oracle_floor::Union{Nothing,Real}           = nothing  # dashed gray selection floor
    # --- legend ---------------------------------------------------------------
    legend::Bool                                  = false
    legend_name::AbstractString                   = "paperlegend"  # `legend to name`
    legend_cols::Union{Nothing,Integer}           = nothing        # nothing ⇒ one per curve
    legend_strip::Union{Nothing,AbstractString}   = nothing        # suffix to strip from entries
end

"""
    PanelSpec(base::PanelSpec; overrides...) -> PanelSpec

Copy `base` with the given fields replaced (non-mutating derivation, e.g.
`PanelSpec(spec; xlabel = x_label)`).
"""
PanelSpec(base::PanelSpec; kwargs...) =
    PanelSpec(; (f => getfield(base, f) for f in fieldnames(PanelSpec))..., kwargs...)

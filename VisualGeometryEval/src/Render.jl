# =============================================================================
# Render — figure aggregation + emission: the PanelSpec display record, the
# summary-panel build_axis stack, the faceted-arm grammar, the house axis
# style, the shared faceted-groupplot emitter and the generic panel/table
# emitters.
#
# Paper-owned data (palette, summary rows, arm records, frozen axis blocks) is
# ONE explicit `PaperStyle` value (paper_style.jl) the paper builds and hands
# to every renderer that draws arms or paper colours — never a registry.
# =============================================================================
module Render

using PGFPlotsX: PGFPlotsX, @pgf, Axis, LegendEntry, Plot, Table
using JLD2: JLD2
using DataFrames: DataFrame, SubDataFrame, groupby
import Printf
using Printf: @sprintf
import .._config_default   # configs/default.toml (config_defaults.jl)
using ..Stats: finite_median, eligible_maa, pair_outcomes, scene_mean_median
using ..Tuning: load_tuning_artifact, MeanAverageAccuracy, tuning_criterion_label

export
    # panel_spec.jl
    PanelSpec,
    # paper_style.jl — a paper's styling, one value
    PaperStyle,
    # plot_core.jl — load_results + the PanelSpec build_axis + colours
    load_results, paper_color_name, paper_color_defs, default_palette, PANELS,
    write_article_preview, build_axis,
    # figure_style.jl — the faceted-arm grammar
    arm_style_records, FAMILY_ORDER, family_rank, PATH_RANK, path_rank, arm_order,
    has_arm, arm_record, arm_style, family_of, panel_title,
    # faceted.jl — THE shared faceted-groupplot emitter
    open_faceted_groupplot, faceted_legend_opts, arm_plot_opts, arm_addplot, arm_legendimage,
    # axis_style.jl — THE house axis-style source (all NEW figures)
    HOUSE_AXIS_COMMON, HOUSE_LINE_WIDTH, HOUSE_MARK_SIZE, color_defs, house_colors,
    # emitters.jl — the generic panel/table emission bodies
    pgf_coords, evaluation_curve,
    emit_miscal_faceted, emit_miscal_faceted_curves, emit_basin_faceted,
    budget_curves, emit_budget_faceted_curves, emit_budget_time_panel,
    lo_us_per_call, emit_lo_timing_table, miscal_summary_rows, emit_miscal_summary_table,
    # tuning_table.jl — a coarse-fine tuning artifact's curve or surface
    tuning_slice, emit_tuning_table

include("render/panel_spec.jl")
include("render/axis_style.jl")
include("render/paper_style.jl")
include("render/plot_core.jl")
include("render/figure_style.jl")
include("render/faceted.jl")
include("render/emitters.jl")
include("render/tuning_table.jl")

end # module Render

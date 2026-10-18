# =============================================================================
# src/render/plot_core.jl — the reusable half of the historical plot_val.jl:
# load_results, the PanelSpec-driven build_axis, the article-preview compiler,
# and the colour helpers.
#
# PAPER-OWNED data (the palette, the summary-panel rows) is a `PaperStyle`
# value the caller supplies (paper_style.jl). The consumer of this stack is
# LocalOptimizationDoneRight's plot_lo_sensitivity.jl (load_results, build_axis,
# write_article_preview, PANELS).
#
# (Pruned 2026-09-21: the paper-mode panel/group/legend emitters
# (build_axis_paper, render_panel*, render_sensitivity_legend), the keyword
# build_axis wrapper, the run-mode provenance helpers (_timestamp, _eval_stamp,
# _dump_config_toml, _parse_eval_name), the caption label tables, and the
# write-only RESULTS_ROOT/RESULTS_PROJECT. Their drivers — the paper-side
# plot_val.jl — no longer exist anywhere, and build_axis_paper threw
# UndefVarError (ytick_spacing, ymin_data, ymax_data) on every call.)
# =============================================================================

"""
    load_results(path) -> Dict{Tuple{String,String}, SubDataFrame}

Load a results JLD2 carrying a `summary` DataFrame (the rows
`Evaluation.compute_method_rows` emits — LocalOptimizationDoneRight's
plot_lo_sensitivity writes it as `summary.jld2`) and bucket that `summary`
into one `SubDataFrame` per `(method_key, tuning_criterion)` for
the PGFPlotsX renderer.  Method key strips a trailing `(median|mAA)` suffix
so the style rows' bare keys (`"RANSAC"`) match.

Each sub-DataFrame is sorted by `n` so downstream rendering can index
column vectors directly (`s.nn`, `s.mean_med`, `s.lo95_mean_med`, ...).
"""
function load_results(path::String)
    keys_in_jld2 = JLD2.jldopen(path, "r") do f; collect(keys(f)); end
    "summary" in keys_in_jld2 ||
        error("load_results: no `summary` key in $path (keys: $keys_in_jld2)")
    summary::DataFrame = JLD2.load(path, "summary")
    df = copy(summary)
    # Strip a trailing `(median|mAA)` suffix from the persisted method name.
    # Case-insensitive: artifacts written by the retired `_crit_tag` spell it
    # `(maa)` while `tuning_criterion` is `"mAA"`, and both must still read.
    df.method_key = [replace(strip(String(m)), r"\s*\((?:median|mAA)\)$"i => "")
                     for m in df.method]
    df.tune_key   = String.(df.tuning_criterion)
    df.nn         = Float64.(df.n)
    sort!(df, :nn)
    data = Dict{Tuple{String,String}, SubDataFrame}()
    for sub in groupby(df, [:method_key, :tune_key])
        key = (String(sub.method_key[1]), String(sub.tune_key[1]))
        data[key] = sub
    end
    data
end

# =============================================================================
# Colours
# =============================================================================

"""
    paper_color_name(key) -> String

The LaTeX colour name of a method key: `"Color"` + the key's alphanumerics
(ALL others stripped, so keys with spaces/`+` such as `"IG  sel"`, `"MSAC +LO"`
yield valid names).
"""
paper_color_name(k::AbstractString) = "Color" * replace(k, r"[^A-Za-z0-9]" => "")

"""
    paper_color_defs(style::PaperStyle) -> String

The `\\providecolor{<name>}{HTML}{<HEX>}` block of every colour of `style`
(sorted by colour name, hex upper-cased), through the one colour emitter
[`color_defs`](@ref) — what a standalone preview prepends so it is
self-contained for any roster (`\\providecolor` is idempotent and the paper's
own definitions win).
"""
paper_color_defs(style::PaperStyle) =
    color_defs([n => uppercase(h) for (n, h) in sort(collect(style.colors); by = first)];
               command = "providecolor")


"""
    default_palette() -> Vector{Pair{String,String}}

THE house base palette (`name => hex`, ordered): the standing selector/
reference colors every paper starts from — RGB-primary "ours" hues
(Jeffreys red `e6194b`, IG-Robust green `3cb44b`, IG-MLE blue `4363d8`),
baselines chosen so none competes with Jeffreys red (RANSAC pastel pink
`ffb6c1`, MSAC yellow `ffe119`, GaU cyan `42d4f4`, MAGSAC++ orange
`f58231`), and the medium-gray Oracle `808080`. Per-arm FAMILY shades
(Welsch greens / TLS oranges / GM purples — the faceted grammar's hue-=
family rule) are paper-assigned on top of this base in the paper's
[`PaperStyle`](@ref); papers override or extend freely — this table is the
documented default, not a constraint.
"""
default_palette() = [
    "RANSAC"    => "ffb6c1",
    "MSAC"      => "ffe119",
    "GaU"       => "42d4f4",
    "MAGSAC"    => "f58231",
    "Jeffreys"  => "e6194b",
    "IG-Robust" => "3cb44b",
    "IG-MLE"    => "4363d8",
    "Oracle"    => "808080",
]

# A render-only exclusion is the `exclude` keyword of the emitters
# (emitters.jl `emit_*`, `miscal_summary_rows`) — the ONE owner. `build_axis`
# below plots the style rows it is given, so excluding a method from it means
# leaving its row out of the style.

"""
    PANELS

The summary panels, keyed by id: `(id, yfn, ylabel, log_y, y_spacing,
y_minor_subdiv)`. `y_spacing` is the major-tick (labeled) gradation on a
linear y-axis (NaN = off); `y_minor_subdiv` the number of minor ticks BETWEEN
majors (0 = none). `eR_mean` is a legacy id: for E and F the metric is the imc
POSE error `max(Δrot,Δtrans)`; a homography's imc is the decomposition's
rotation alone (VGC `pool_imc` on an H pool), so an H panel overrides the
label (`build_axis(...; ylabel)`).
"""
const PANELS = (
    eR_mean  = (id="eR_mean",  yfn = s -> (s.mean_med, s.lo95_mean_med, s.hi95_mean_med),
                ylabel=raw"median $\max(\Delta \mathbf{R},\Delta \mathbf{t})$ (deg)",  log_y=false, y_spacing=1.0, y_minor_subdiv=1),
    mAA_mean = (id="mAA_mean", yfn = s -> (s.mean_mAA, s.lo95_mean_mAA, s.hi95_mean_mAA),
                ylabel=raw"$\mathbb{E}[\mathrm{mAA}]$", log_y=false, y_spacing=0.025, y_minor_subdiv=0),
)

"""
    write_article_preview(out_dir, panel_base, colordefs; legend_ref = "")

Write an `article`-class `preview.tex` in `out_dir` that `\\input`s the panel
fragment `panel_base` (its filename WITHOUT .tex), then compile it to PDF with
latexmk when it is installed. `colordefs` is the `\\providecolor` block for the
panel's colours; `legend_ref` names a legend to `\\ref` below the panel.
"""
function write_article_preview(out_dir::AbstractString, panel_base::AbstractString,
                               colordefs::AbstractString; legend_ref::AbstractString="")
    legend_line = isempty(legend_ref) ? "" : "\\\\[6pt]\n\\ref{$legend_ref}"
    write(joinpath(out_dir, "preview.tex"), """
\\documentclass[10pt]{article}
\\usepackage[a4paper,margin=2cm]{geometry}
\\usepackage{amsmath,amssymb}
\\usepackage{pgfplots}\\pgfplotsset{compat=1.18}
\\usepgfplotslibrary{fillbetween,groupplots}
\\usepackage{xcolor}
$colordefs
\\pagestyle{empty}
\\begin{document}\\centering
\\input{$panel_base}$legend_line
\\end{document}
""")
    ml = Sys.which("latexmk")
    if ml === nothing
        @info "latexmk not found; preview.tex written, PDF skipped" out_dir
        return nothing
    end
    try
        # latexmk auto-reruns to resolve the forward \ref to the legend.
        run(pipeline(Cmd(`latexmk -pdf -interaction=nonstopmode preview.tex`; dir = out_dir);
                     stdout = devnull, stderr = devnull))
    catch err
        @warn "latexmk failed (preview.tex still written)" out_dir err
    end
    return nothing
end
"""
    build_axis(style::PaperStyle, panel, data, tune::String, spec::PanelSpec; ylabel = "")

The PGFPlots axis of one ([`PANELS`](@ref) entry, tuning-criterion label
`tune`): one curve per row of `style.rows` present in `data`
([`load_results`](@ref)). Every display option comes from `spec`;
`spec.xlabel === nothing` is "Validation set size". `ylabel` overrides the
panel's.
"""
function build_axis(style::PaperStyle, panel, data, tune::String, spec::PanelSpec;
                    ylabel::AbstractString="")
    xlabel        = spec.xlabel === nothing ? "Validation set size" : spec.xlabel
    xmode         = spec.xmode
    legend        = spec.legend
    legend_name   = spec.legend_name
    xticks_fixed  = spec.xticks_fixed
    tight_x       = spec.tight_x
    xtick         = spec.xtick
    xticklabels   = spec.xticklabels
    height        = spec.height
    width         = spec.width
    y_unit_degree = spec.y_unit_degree
    ytick_spacing    = spec.ytick_spacing
    title         = spec.title
    show_ci       = spec.show_ci
    legend_cols   = spec.legend_cols
    plots = Any[]
    # Track data-driven x/y ranges (x for `tight_x`; y for y_spacing gradation).
    ymin_data =  Inf;  ymax_data = -Inf
    xmin_data =  Inf;  xmax_data = -Inf
    n_plotted = 0                                  # # curves actually drawn (legend columns)

    for (idx, (k, lbl, color, mark, dashed)) in enumerate(style.rows)
        dk = (k, tune)
        haskey(data, dk) || continue
        s = data[dk]
        y, ylo, yhi = panel.yfn(s)

        # Filter out rows whose central `y` is non-finite (pgfplots writes
        # `nan` literally and that fails as `Missing number`). For rows
        # where only the CI band (ylo/yhi) is NaN — e.g., n_bootstrap=1
        # where std is undefined — we KEEP the central curve but collapse
        # the band to a zero-width strip by substituting NaN→y, so the
        # band silently disappears instead of crashing pgflatex.
        keep = [isfinite(yi) for yi in y]
        any(keep) || continue
        nn_f  = s.nn[keep]
        y_f   = y[keep]
        ylo_f = [isfinite(ylo[i]) ? ylo[i] : y[i] for i in eachindex(y) if keep[i]]
        yhi_f = [isfinite(yhi[i]) ? yhi[i] : y[i] for i in eachindex(y) if keep[i]]

        # Accumulate range across all plotted series (the kept finite values).
        ylo_s, yhi_s = extrema(Iterators.flatten((y_f, ylo_f, yhi_f)))
        xlo_s, xhi_s = extrema(nn_f)
        ymin_data = min(ymin_data, ylo_s); ymax_data = max(ymax_data, yhi_s)
        xmin_data = min(xmin_data, xlo_s); xmax_data = max(xmax_data, xhi_s)

        upper_name = "U$(panel.id)_$(tune)_$(idx)"
        lower_name = "L$(panel.id)_$(tune)_$(idx)"

        if show_ci
            push!(plots, @pgf Plot({"name path" = upper_name,
                                    draw = "none", mark = "none",
                                    "forget plot"},
                                   Table([nn_f, yhi_f])))
            push!(plots, @pgf Plot({"name path" = lower_name,
                                    draw = "none", mark = "none",
                                    "forget plot"},
                                   Table([nn_f, ylo_f])))
            push!(plots, @pgf Plot({fill = color, "fill opacity" = 0.18,
                                    draw = "none", mark = "none",
                                    "forget plot"},
                                   raw"fill between [of=" * upper_name *
                                   raw" and " * lower_name * raw"]"))
        end
        opts = @pgf {color = color, mark = mark,
                     "mark options" = "{solid,fill=" * color * "}",
                     "line width" = "0.8pt", "mark size" = "1.8pt"}
        if dashed
            opts["dashed"] = nothing
        end
        push!(plots, Plot(opts, Table([nn_f, y_f])))
        n_plotted += 1
        legend && push!(plots, LegendEntry(lbl))
    end

    isempty(plots) && return nothing

    axopts = @pgf {
        width = width,
        height = height,                # golden ratio 1/phi by default
        xmode = xmode, "log basis x" = 2,   # "log basis x" is ignored when xmode=normal
        xlabel = xlabel,
        ylabel = (isempty(ylabel) ? panel.ylabel : ylabel),   # "" ⇒ panel default (Sensitivity)
        xmajorgrids, ymajorgrids,
        "grid style" = "{gray!20}",
        "label style" = "{font=\\footnotesize}",
        "tick label style" = "{font=\\scriptsize}",
    }
    if legend
        # Store a horizontal legend to a name; the document \ref's it BELOW the
        # panel (see write_article_preview). One row, `n_plotted` columns.
        axopts["legend to name"]    = legend_name
        axopts["legend columns"]    = (legend_cols === nothing ? n_plotted : legend_cols)
        axopts["legend cell align"] = "left"
        axopts["legend style"]      = "{draw=none, font=\\footnotesize, /tikz/every even column/.append style={column sep=8pt}}"
    end
    if xticks_fixed
        # Label the log x-axis with plain decimals (0.0625, …, 1, …, 128), not 2^k.
        axopts["log ticks with fixed point"] = nothing
    end
    if tight_x && isfinite(xmin_data) && isfinite(xmax_data)
        # Clamp the axis to the data span (e.g. n: 2…2048) — no auto-range padding.
        axopts["xmin"] = xmin_data
        axopts["xmax"] = xmax_data
    end
    xtick === nothing || (axopts["xtick"] = "{" * join(xtick, ",") * "}")   # ticks AT the checkpoints
    xticklabels === nothing || (axopts["xticklabels"] = "{" * join(xticklabels, ",") * "}")
    if panel.log_y
        axopts["ymode"] = "log"
        # Plain decimal labels on y (e.g. 0.1, 1, 10) instead of 10^k.
        axopts["log ticks with fixed point"] = nothing
        # `log ticks with fixed point` also affects x; restore x-axis to 2^k.
        axopts["xticklabel"] =
            raw"\ensuremath{2^{\pgfmathparse{int(\tick)}\pgfmathprintnumber{\pgfmathresult}}}"
    else
        # Explicit ytick_spacing overrides the panel default AND the adaptive
        # refinement (a codified per-figure choice is never second-guessed).
        spacing = ytick_spacing === nothing ? panel.y_spacing : Float64(ytick_spacing)
        # ADAPTIVE tick density: panel y_spacing is degree-scale (1.0); on
        # sub-degree axes (e.g. ETH3D, range ≲2 steps) it leaves ≤1 visible
        # tick — refine by 5× until ≥3 ticks span the data.
        if ytick_spacing === nothing && isfinite(spacing) && isfinite(ymax_data) && isfinite(ymin_data)
            while (ymax_data - ymin_data) < 2.5 * spacing
                spacing /= 5
            end
        end
        if isfinite(spacing)
            axopts["ytick distance"] = spacing
            if panel.y_minor_subdiv > 0
                axopts["minor y tick num"] = panel.y_minor_subdiv
                axopts["yminorgrids"] = nothing
            end
            # Disable pgfplots' auto-scaled tick labels (e.g. "·10^{-2}"
            # multiplier) and print each tick literally with precision
            # inferred from spacing (e.g. 0.025 -> 3, 0.05 -> 2, 1.0 -> 0).
            s = @sprintf("%g", spacing)
            prec = (i = findfirst('.', s)) === nothing ? 0 : length(s) - i
            axopts["scaled y ticks"] = "false"
            axopts["yticklabel style"] =
                "{/pgf/number format/.cd, fixed, precision=$prec}"
        end
    end
    title === nothing || (axopts["title"] = title)
    # HOUSE CONVENTION: error axes carry the degree on the TICK NUMBERS
    # (2°, 4°, …) — never "(deg)" in the ylabel. Set last so it applies in
    # both the linear and log-y paths.
    y_unit_degree && (axopts["yticklabel"] =
        raw"$\pgfmathprintnumber{\tick}^\circ$")
    return Axis(axopts, plots...)
end

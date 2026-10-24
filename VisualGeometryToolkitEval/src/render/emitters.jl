# =============================================================================
# src/render/emitters.jl — the GENERIC emission bodies of the paper panel
# renderers (promoted from papers/accv2026/render_miscal_faceted.jl,
# render_basin_faceted.jl, render_sb_faceted.jl and supp/emit_timing.jl).
#
# PROMOTION BOUNDARY (the 'promote what a future paper uses unchanged' rule):
# everything paper-specific stays in the thin paper callers: env/ARGS parsing,
# default artifact paths, the PANELS tables (facet order + data getters),
# `exclude` defaults, legend names, oracle constants, label tables. These
# functions hold the data/axis/emission logic only, over the shared faceted
# grammar (faceted.jl) and the caller's `PaperStyle` (paper_style.jl): its arm
# records and its axis blocks.
#
# BYTE-IDENTITY CONTRACT: the emitted panels are byte-compared against the
# committed paper artifacts — change emission logic here only together with a
# re-render of every affected panel.
# =============================================================================

# -----------------------------------------------------------------------------
# The shared skeleton of the faceted emitters: row validation, the facet walk
# of the legend, the per-cell y-label/title fragments, the oracle reference
# line, the arm curves + legend images, the groupplot close and the write-out.
# -----------------------------------------------------------------------------
const _ORACLE_STYLE  = "black, line width=0.5pt"
const _FACET_WIDTH   = raw"0.265\linewidth"
const _FACET_HEIGHT  = raw"0.23\linewidth"       # ζ-miscalibration and basin panels
const _BUDGET_HEIGHT = raw"0.205\linewidth"      # accuracy-vs-budget panels
const _TITLE_STYLE   = raw"title style={font=\footnotesize, align=center}"
const _FONT_OPTS     = raw"label style={font=\footnotesize}, tick label style={font=\scriptsize}"
const _PENDING       = "\\node[gray] at (rel axis cs:0.5,0.6) {\\scriptsize\\itshape pending};"

# Row label of a standalone panel: the dataset + model-class symbol stacked
# above the statistic in the first facet, exactly as the paper's grids print
# it; empty (the default) keeps the statistic alone.
_row_ylabel(row_label::AbstractString, stat::AbstractString) =
    isempty(row_label) ? "ylabel={$(stat)}" :
    "ylabel={$(row_label)\\\\$(stat)}, ylabel style={align=center}"

# The y-label fragment of facet column `c`: the (row-labelled) statistic on the
# first column, suppressed tick labels on the others.
_cell_ylabel(c::Integer, row_label, stat) =
    c == 1 ? ", " * _row_ylabel(row_label, stat) : ", yticklabels={}"

# The title fragment: facet titles on the first row only.
_cell_title(r::Integer, name) = r == 1 ? ", title={$name}" : ""

function _check_rows(rows, allowed::Tuple)
    isempty(rows) && error("rows must name at least one of $(join(map(repr, allowed), ", "))")
    all(k -> k in allowed, rows) ||
        error("rows must be a subset of ($(join(map(repr, allowed), ", "))), got $rows")
    return length(rows)
end

# The arms of one facet in the within-facet order, and the legend walk over all
# facets (facet order, then arm order).
_facet_arms(style, arms, family) =
    sort([a for a in arms if family_of(style, a) === family], by = a -> arm_order(style, a))
_legend_order(style, arms, panels) = String[a for (fam, _) in panels for a in _facet_arms(style, arms, fam)]

# Axis-option fragments joined with ", ", an empty fragment (a style without
# grid options) dropped.
_join_opts(parts...) = join(filter(!isempty, collect(parts)), ", ")

_oracle_addplot(io::IO, coords::AbstractString) =
    println(io, "\\addplot[$(_ORACLE_STYLE), mark=none, forget plot] coordinates {", coords, "};")
_oracle_legend(io::IO, label) =
    println(io, "\\addlegendimage{$(_ORACLE_STYLE)}\\addlegendentry{$label}")

_arm_curve(io::IO, style, arm, coords::AbstractString) =
    (s = arm_style(style, arm); arm_addplot(io, s.color, s.marker, s.dashed, coords))
_arm_legend(io::IO, style, arms) =
    foreach(a -> (s = arm_style(style, a); arm_legendimage(io, s.color, s.marker, s.dashed, s.label)), arms)

# Close the groupplot: inside a host tikzpicture (`deploy`, the host draws the
# legend via \pgfplotslegendfromname{<legend_name>}) or with the legend placed
# below the last row.
function _close_groupplot(io::IO, deploy::Bool, nrows::Integer, legend_name::AbstractString)
    println(io, raw"\end{groupplot}")
    if deploy
        println(io, raw"\end{tikzpicture}")
    else
        println(io, "\\path (group c1r$(nrows).south west) -- (group c3r$(nrows).south east)")
        println(io, "  node[midway, below=9mm] {\\pgfplotslegendfromname{$legend_name}};")
    end
end

_write_tex(out::AbstractString, io::IOBuffer) =
    (mkpath(dirname(out)); open(f -> write(f, String(take!(io))), out, "w"); println("WROTE ", out); out)

"""
    pgf_coords(xs, ys; fmt = "(%.6f,%.4f)") -> String

The pgfplots `coordinates {…}` body of a curve: one `fmt`-formatted `(x,y)`
per point with a FINITE `y` (a non-finite point is dropped — pgfplots cannot
draw it), space-separated. `fmt` is a Printf format of the two coordinates
(`"(%d,%.4f)"` for an integer budget axis).
"""
function pgf_coords(xs, ys; fmt::AbstractString = "(%.6f,%.4f)")
    f = Printf.Format(fmt)
    return join([Printf.format(f, xs[j], ys[j]) for j in eachindex(xs, ys) if isfinite(ys[j])], " ")
end

"""
    evaluation_curve(evaluation, arm, stat) -> Vector{Float64}

One arm's curve over the columns of a deployed evaluation (`evaluation[arm]`, pairs
× ζ): `stat` of each column — `finite_median` for the median row,
`eligible_maa` for the mAA row (the eligibility rule: every stored pair is
eligible, a pair without a model scores 0).
"""
function evaluation_curve(evaluation, arm, stat)
    m = Float64.(evaluation[arm])
    return [stat(view(m, :, j)) for j in axes(m, 2)]
end

# -----------------------------------------------------------------------------
# Objective-faceted composed-miscalibration groupplot (3 loss panels × 2 rows:
# median pose error / mAA@10° vs scale corruption ζ), from a miscalibration sidecar.
# -----------------------------------------------------------------------------
"""
    emit_miscal_faceted(style::PaperStyle, sidecar, out; row_label = "", oracle, oracle_maa,
                        oracle_class = "E", ymin, deploy, ymax = 6, ytick_spacing = 1,
                        exclude = Set{String}(), legend_name = "miscallegend",
                        panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                        arm_filter = a -> true, rows = [:med, :maa])

Emit the ζ-miscalibration faceted groupplot body to `out` from the tuning
artifact `sidecar` (`zetas` + `evaluation`): the per-arm [`evaluation_curve`](@ref)s through
[`emit_miscal_faceted_curves`](@ref). Rows: median pose error (oracle FLOOR
`oracle`) and mAA@10° (oracle CEILING `oracle_maa`); the median row's y block
is `style.axes.zeta_med_y(ymin; ymax, ytick_spacing)`, `ymin` its y floor
(string, TeX-interpolated), `ymax`/`ytick_spacing` its ceiling and tick
spacing (default 6°/1°). `deploy` wraps in a tikzpicture and leaves the legend
to the host (`\\pgfplotslegendfromname{<legend_name>}`); `exclude` drops arm
keys from THIS render only; `arm_filter` restricts the roster (e.g. one
scoring selector).
"""
function emit_miscal_faceted(style::PaperStyle, sidecar::AbstractString, out::AbstractString;
                             row_label::AbstractString = "",
                             oracle::Real, oracle_maa::Real,
                             oracle_class::AbstractString = "E",
                             ymin::AbstractString, deploy::Bool,
                             ymax::Real = 6, ytick_spacing::Real = 1,
                             exclude = Set{String}(),
                             legend_name::AbstractString = "miscallegend",
                             panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                             arm_filter = a -> true,
                             rows::AbstractVector{Symbol} = [:med, :maa])
    d  = load_tuning_artifact(sidecar)
    ro = d["evaluation"]
    present = [a for a in keys(ro) if has_arm(style, a) && arm_filter(a) && a ∉ exclude]
    emit_miscal_faceted_curves(style, out; row_label, zetas = Float64.(d["zetas"]),
        med = Dict(a => evaluation_curve(ro, a, finite_median) for a in present),
        maa = Dict(a => evaluation_curve(ro, a, eligible_maa) for a in present),
        deploy, legend_name, oracle, oracle_maa, oracle_class,
        med_axis = style.axes.zeta_med_y(ymin; ymax, ytick_spacing), panels, rows)
    isempty(exclude) || println("  excluded (render-only): ", join(sort(collect(exclude)), ", "))
    return out
end

"""
    emit_miscal_faceted_curves(style::PaperStyle, out; zetas, med, maa, deploy, legend_name, rows,
                               oracle = nothing, oracle_maa = nothing,
                               oracle_class = "",
                               x_axis = style.axes.zeta_x,
                               med_axis = style.axes.zeta_med_y("0.4"),
                               maa_axis = style.axes.zeta_maa_y,
                               med_ylabel = style.axes.med_ylabel,
                               maa_ylabel = style.axes.maa_ylabel,
                               panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                               flat = Set{String}())

The objective-faceted ζ groupplot from PRECOMPUTED per-arm curves: `med[arm]`
and `maa[arm]` are vectors aligned with `zetas` (NaN = masked point); arms are
keys of `style.arms`, so family, mechanism order, label, color and marker all
derive from the style. Arms in `flat` carry a single value that is drawn as
a constant line across `zetas` (width-free controls). The oracle line spans
`zetas` and is drawn only when `oracle` is given. Axis blocks default to the
style's; supply other blocks for other error scales.
"""
function emit_miscal_faceted_curves(style::PaperStyle, out::AbstractString; row_label::AbstractString = "",
                                    zetas::AbstractVector{<:Real},
                                    med::AbstractDict, maa::AbstractDict,
                                    deploy::Bool, legend_name::AbstractString,
                                    oracle = nothing, oracle_maa = nothing,
                                    oracle_class::AbstractString = "",
                                    x_axis::AbstractString = style.axes.zeta_x,
                                    med_axis::AbstractString = style.axes.zeta_med_y("0.4"),
                                    maa_axis::AbstractString = style.axes.zeta_maa_y,
                                    med_ylabel::AbstractString = style.axes.med_ylabel,
                                    maa_ylabel::AbstractString = style.axes.maa_ylabel,
                                    panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                                    rows::AbstractVector{Symbol} = [:med, :maa],
                                    flat = Set{String}())
    nrows = _check_rows(rows, (:med, :maa))
    xs = Float64.(zetas)
    series(d, a) = a in flat ? fill(Float64(first(d[a])), length(xs)) : Float64.(d[a])
    arms = [a for a in keys(med) if has_arm(style, a)]
    io = IOBuffer()
    deploy && println(io, raw"\begin{tikzpicture}")
    open_faceted_groupplot(io; cols = 3, rows = nrows, width = _FACET_WIDTH, height = _FACET_HEIGHT,
                           opts = [x_axis, _TITLE_STYLE])
    for (ri, kind) in enumerate(rows), (c, (obj, name)) in enumerate(panels)
        ismed = kind === :med              # row KIND selects the statistic; ri its position
        leg = (ri == 1 && c == 1) ? faceted_legend_opts(legend_name) : ""
        println(io, "\\nextgroupplot[", ismed ? med_axis : maa_axis,
                _cell_ylabel(c, row_label, ismed ? med_ylabel : maa_ylabel), _cell_title(ri, name), leg, "]")
        # oracle: median row = pool-best selection FLOOR; mAA row = pool-best CEILING
        orc = ismed ? oracle : oracle_maa
        orc === nothing || _oracle_addplot(io, "($(xs[1]),$orc)($(xs[end]),$orc)")
        pa = _facet_arms(style, arms, obj)
        foreach(a -> _arm_curve(io, style, a, pgf_coords(xs, series(ismed ? med : maa, a))), pa)
        isempty(pa) && println(io, _PENDING)
        if ri == 1 && c == 1               # complete legend once (image-only)
            oracle === nothing || _oracle_legend(io, "oracle (pool-best \$$(oracle_class)\$)")
            _arm_legend(io, style, _legend_order(style, arms, panels))
        end
    end
    _close_groupplot(io, deploy, nrows, legend_name)
    _write_tex(out, io)
    for (obj, name) in panels
        pa = [arm_style(style, a).label for a in _facet_arms(style, arms, obj)]
        println("  ", name, ": ", isempty(pa) ? "(empty)" : join(pa, ", "))
    end
    return out
end

"""
    emit_budget_faceted_curves(style::PaperStyle, out; row_label = "", budgets, med, maa,
                               med_ylabel, legend_name, deploy, xticks = nothing,
                               maa_ylabel = style.axes.maa_ylabel,
                               panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                               xlabel = raw"sample budget", xlog = true, rows = [:med, :maa],
                               oracle_med = nothing, oracle_maa = nothing,
                               oracle_label = "oracle")

The objective-faceted accuracy-vs-budget groupplot, from PRECOMPUTED per-arm
curves ([`budget_curves`](@ref) for a budget cell's scene files; the PnP
budget's own aggregation): `med[arm]`
(median error, row label `med_ylabel`) and `maa[arm]` (mAA@10°) are vectors
aligned with `budgets`; arms are keys of `style.arms`, and the grid options are
`style.axes.grid`. `xlog` draws a log₂ budget axis with plain-number
ticks at the given budgets (small budget grids such as 5…2000); the y ranges
are derived from the curves. `oracle_med`/`oracle_maa` (aligned with `budgets`)
draw the oracle reference line when given.
"""
function emit_budget_faceted_curves(style::PaperStyle, out::AbstractString; row_label::AbstractString = "",
                                    budgets::AbstractVector{<:Integer},
                                    med::AbstractDict, maa::AbstractDict, med_ylabel::AbstractString,
                                    legend_name::AbstractString, deploy::Bool,
                                    xticks::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                    maa_ylabel::AbstractString = style.axes.maa_ylabel,
                                    panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                                    xlabel::AbstractString = raw"sample budget",
                                    xlog::Bool = true,
                                    rows::AbstractVector{Symbol} = [:med, :maa],
                                    oracle_med = nothing, oracle_maa = nothing,
                                    oracle_label::AbstractString = "oracle")
    nrows = _check_rows(rows, (:med, :maa))
    B = Int.(budgets)
    arms = [a for a in keys(med) if has_arm(style, a)]
    maav = filter(isfinite, reduce(vcat, [Float64.(maa[a]) for a in arms]))
    medv = filter(isfinite, reduce(vcat, [Float64.(med[a]) for a in arms]))
    amin = floor(minimum(maav) / 0.05) * 0.05; amax = min(1.0, ceil(maximum(maav) / 0.05) * 0.05)
    atick = join([@sprintf("%.2f", y) for y in amin:0.05:amax], ",")
    emin = floor(minimum(medv) / 0.5) * 0.5; emax = ceil(maximum(medv) / 0.5) * 0.5
    etick = join([@sprintf("%.1f", y) for y in emin:0.5:emax], ",")
    coords(ys) = pgf_coords(B, Float64.(ys); fmt = "(%d,%.4f)")
    tk = xticks === nothing ? B : Int.(xticks)   # dense grids supply a thinned tick set
    xopts = xlog ?
        "xmode={log}, log basis x={2}, xmin={$(B[1])}, xmax={$(B[end])}, xtick={$(join(tk, ","))}, xticklabels={$(join(tk, ","))}" :
        "xmin={$(B[1])}, xmax={$(B[end])}, xtick={$(join(tk, ","))}"
    io = IOBuffer()
    deploy && println(io, raw"\begin{tikzpicture}")
    open_faceted_groupplot(io; cols = 3, rows = nrows, width = _FACET_WIDTH, height = _BUDGET_HEIGHT,
                           opts = filter(!isempty, [xopts, style.axes.grid, _FONT_OPTS, "xlabel={$xlabel}", _TITLE_STYLE]))
    for (ri, kind) in enumerate(rows), (ci, (obj, title)) in enumerate(panels)
        ismed = kind === :med              # row KIND selects the statistic; ri its position
        yopt = ismed ?
            "ymin={$(@sprintf("%.1f",emin))}, ymax={$(@sprintf("%.1f",emax))}, ytick={$etick}, minor y tick num={1}, yminorgrids, " *
            raw"yticklabel style={text width=2.4em, align=right, /pgf/number format/.cd, fixed, precision=1}, yticklabel={$\pgfmathprintnumber{\tick}^\circ$}" :
            "ymin={$(@sprintf("%.2f",amin))}, ymax={$(@sprintf("%.2f",amax))}, ytick={$atick}, minor y tick num={0}, yminorgrids, " *
            raw"yticklabel style={text width=2.4em, align=right}"
        leg = (ri == 1 && ci == 1) ? faceted_legend_opts(legend_name) : ""
        println(io, "\\nextgroupplot[", yopt, _cell_title(ri, title),
                _cell_ylabel(ci, row_label, ismed ? med_ylabel : maa_ylabel), leg, "]")
        orc = ismed ? oracle_med : oracle_maa
        orc === nothing || _oracle_addplot(io, coords(orc))
        pa = _facet_arms(style, arms, obj)
        foreach(a -> _arm_curve(io, style, a, coords((ismed ? med : maa)[a])), pa)
        isempty(pa) && println(io, _PENDING)
        if ri == 1 && ci == 1
            (oracle_med === nothing && oracle_maa === nothing) || _oracle_legend(io, oracle_label)
            _arm_legend(io, style, _legend_order(style, arms, panels))
        end
    end
    _close_groupplot(io, deploy, nrows, legend_name)
    return _write_tex(out, io)
end

# -----------------------------------------------------------------------------
# Basin/travel mechanism matrix (3 loss panels × 2 rows: success rate
# P(e≤succ°) / conditional mean final error vs pose perturbation δ), from a
# ladder bundle.
# -----------------------------------------------------------------------------
"""
    emit_basin_faceted(style::PaperStyle, bundle, out; row_label = "", succ, deploy,
                       exclude = Set{String}(), legend_name = "basinlegend",
                       panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                       minsucc = [render] min_success_for_cond_mean, seed_key = "seed",
                       rows = [:succ, :cond])

Emit the δ-perturbation basin faceted groupplot body to `out` from the ladder
`bundle` (`errs` [pairs × nδ × nmethod], `methods`, `deltas`, `refinable`).
`rows` selects the statistic per row, in order: `:succ` — success rate
P(e≤succ°); `:maa` — mAA@succ° under the eligibility rule (`eligible_maa`:
the refinable pairs are the eligible ones, a failed refit — stored as NaN —
scores 0); `:cond` — conditional mean final error given success, masked below
success rate `minsucc`. Success is recomputed at `succ` degrees; the
`seed_key` method draws as the dashed reference (its success rate, mAA, or
unconditional median per row). `deploy`/`exclude`/`legend_name` as in
[`emit_miscal_faceted`](@ref).
"""
function emit_basin_faceted(style::PaperStyle, bundle::AbstractString, out::AbstractString;
                            row_label::AbstractString = "",
                            succ::Real, deploy::Bool,
                            exclude = Set{String}(),
                            legend_name::AbstractString = "basinlegend",
                            panels = [(f, panel_title(f)) for f in FAMILY_ORDER],
                            minsucc::Real = _config_default("render", "min_success_for_cond_mean"),
                            seed_key::AbstractString = "seed",
                            rows::AbstractVector{Symbol} = [:succ, :cond])
    nrows   = _check_rows(rows, (:succ, :maa, :cond))
    d       = JLD2.load(bundle)
    errs    = d["errs"]                                  # [pairs × nδ × nmethod]
    methods = d["methods"]
    deltas  = rad2deg.(Float64.(d["deltas"]))            # δ in degrees
    refin   = d["refinable"]                             # per-pair refinable mask
    mask    = refin isa AbstractVector{Bool} ? refin : trues(size(errs, 1))
    midx    = Dict(m => i for (i, m) in enumerate(methods))

    col(mi, j) = @view errs[:, j, mi]
    succ_rate(mi, j) = (v = col(mi, j); c = 0; n = 0;
        for p in eachindex(v); mask[p] || continue; n += 1; (isfinite(v[p]) && v[p] <= succ) && (c += 1); end;
        n == 0 ? NaN : c / n)
    cond_mean(mi, j) = (v = col(mi, j); s = 0.0; c = 0;
        for p in eachindex(v); mask[p] || continue; (isfinite(v[p]) && v[p] <= succ) && (s += v[p]; c += 1); end;
        c == 0 ? NaN : s / c)
    # Conditional-mean masking: when P(e≤succ°) < minsucc too few seeds succeed
    # for the survivor mean to be meaningful (it collapses onto a lucky few).
    cond_mean_m(mi, j) = (succ_rate(mi, j) < minsucc ? NaN : cond_mean(mi, j))
    maa_rate(mi, j) = eligible_maa(col(mi, j), mask; T = succ)
    # Seed reference: UNCONDITIONAL median error over all refinable pairs. Unlike
    # the survivor mean it is defined at every δ (there are always refinable
    # pairs), so the dashed reference spans the full axis.
    seed_med(mi, j) = finite_median(view(col(mi, j), mask))

    present = [m for m in methods if has_arm(style, m) && !(m in exclude)]
    coords(ys) = pgf_coords(deltas, ys; fmt = "(%.4f,%.4f)")

    # x-axis in degrees
    xt = join([@sprintf("%.4f", x) for x in deltas], ",")
    xl = join([@sprintf("\$%.0f^\\circ\$", round(x)) for x in deltas], ",")
    AXX = _join_opts("xmode={log}, log basis x={2}, xmin={$(deltas[1]*0.85)}, xmax={$(deltas[end]*1.15)}, " *
                     "xtick={$xt}, xticklabels={$xl}",
                     style.axes.grid, _FONT_OPTS, raw"xlabel={pose perturbation $\delta$}")
    SUCCY = raw"ymin={0}, ymax={1}, ytick={0,0.2,0.4,0.6,0.8,1}, minor y tick num={1}, yminorgrids"
    DESTY = "ymin={0}, ymax={$(succ)}, ytick={0,2,4,6,8,$(Int(succ))}, minor y tick num={1}, yminorgrids, " *
      raw"yticklabel style={/pgf/number format/.cd, fixed, precision=0}, yticklabel={$\pgfmathprintnumber{\tick}^\circ$}"
    SUCCLAB = "\$P(e_{\\mathrm{pose}}\\le $(Int(succ))^\\circ)\$"
    DESTLAB = raw"$\mathbb{E}[e_{\mathrm{pose}}\mid \mathrm{success}]$"
    MAALAB  = "mAA@\$$(Int(succ))^\\circ\$"
    rowspec(kind)         = kind === :succ ? (SUCCY, SUCCLAB) : kind === :maa ? (SUCCY, MAALAB) : (DESTY, DESTLAB)
    rowstat(kind, mi, j)  = kind === :succ ? succ_rate(mi, j) : kind === :maa ? maa_rate(mi, j) : cond_mean_m(mi, j)
    seedstat(kind, mi, j) = kind === :succ ? succ_rate(mi, j) : kind === :maa ? maa_rate(mi, j) : seed_med(mi, j)

    io = IOBuffer()
    deploy && println(io, raw"\begin{tikzpicture}")
    open_faceted_groupplot(io; cols = 3, rows = nrows, width = _FACET_WIDTH, height = _FACET_HEIGHT,
                           opts = [AXX, _TITLE_STYLE])
    for (r, kind) in enumerate(rows), (c, (obj, name)) in enumerate(panels)
        yopt, ylab = rowspec(kind)
        leg = (r == 1 && c == 1) ? faceted_legend_opts(legend_name) : ""
        println(io, "\\nextgroupplot[", yopt, _cell_ylabel(c, row_label, ylab), _cell_title(r, name), leg, "]")
        # perturbed-seed reference (no LO), repeated per panel. The seed is a
        # reference, not an evaluated arm, so it is exempt from the minsucc
        # mask: we draw its raw curve across the full range.
        if haskey(midx, seed_key)
            sj = midx[seed_key]
            println(io, "\\addplot[black, thin, dashed, forget plot] coordinates {",
                    coords([seedstat(kind, sj, j) for j in eachindex(deltas)]), "};")
        end
        for m in _facet_arms(style, present, obj)
            _arm_curve(io, style, m, coords([rowstat(kind, midx[m], j) for j in eachindex(deltas)]))
        end
        if r == 1 && c == 1
            println(io, "\\addlegendimage{black, thin, dashed}\\addlegendentry{perturbed seed}")
            _arm_legend(io, style, _legend_order(style, present, panels))
        end
    end
    _close_groupplot(io, deploy, nrows, legend_name)
    _write_tex(out, io)
    for (obj, name) in panels
        pa = [arm_style(style, m).label for m in present if family_of(style, m) === obj]
        println("  ", replace(name, "\\\\" => " "), ": ", join(pa, ", "))
    end
    return out
end

# -----------------------------------------------------------------------------
# Sample-budget panels from per-scene budget JLD2s — one path for every budget
# cell: `budget_curves` (the statistics) → `emit_budget_faceted_curves` (the
# faceted panel, also the PnP budget's) and, where a cell draws one,
# `emit_budget_time_panel` (mAA vs cumulative LO time).
# -----------------------------------------------------------------------------
# The budget scene schema (accv2026 src/cell_stages.jl `budget_cell_stages`):
# `err_lo[arm]` (P × L, the scoring column, at the LO checkpoints `lo_counts`)
# and `oracle_err_batch` (P × the dense batch grid `sample_counts`).
_budget_err(d, arm, j) = view(d["err_lo"][arm], :, j)
_budget_oracle(d, j) = view(d["oracle_err_batch"], :, j)

"""
    budget_curves(files; arms) -> (; budgets, med, maa, oracle_med, oracle_maa, outcomes)

The statistics of a budget cell from its per-scene budget `files` (the scene
schema above), for each of `arms` and the oracle, at every LO checkpoint
(`budgets` = the files' `lo_counts`; the oracle read off its dense batch grid
at those budgets):

  * `med` — the mean over scenes of the per-scene finite median
    (Shekhovtsov's mean-median; a scene without a finite error is skipped);
  * `maa` — the eligibility-ruled mAA (`eligible_maa`) POOLED over every pair
    of every scene — every stored pool pair is eligible, a pair whose run left
    no model (non-finite error) scores 0.

Every file must carry the same count axes and every one of `arms`.
`outcomes[arm]` is `pair_outcomes` at the last checkpoint (why runs fail).
"""
function budget_curves(files::AbstractVector{<:AbstractString}; arms::AbstractVector{<:AbstractString})
    isempty(files) && throw(ArgumentError("budget_curves: no scene files"))
    ds = [JLD2.load(f) for f in files]
    B = Int.(ds[1]["lo_counts"]); sc = Int.(ds[1]["sample_counts"])
    for (f, d) in zip(files, ds)
        Int.(d["lo_counts"]) == B && Int.(d["sample_counts"]) == sc ||
            throw(ArgumentError("budget_curves: $f has other count axes than $(files[1])"))
        missing_arms = setdiff(arms, keys(d["err_lo"]))
        isempty(missing_arms) || throw(ArgumentError("budget_curves: $f lacks arms $missing_arms"))
    end
    bi = [findfirst(==(c), sc) for c in B]
    any(isnothing, bi) && throw(ArgumentError("budget_curves: an LO count is not on the batch axis $sc"))
    pooled(get) = reduce(vcat, [Float64.(get(d)) for d in ds])
    med_of(get) = scene_mean_median(get(d) for d in ds)
    med = Dict{String,Vector{Float64}}(a => [med_of(d -> _budget_err(d, a, j)) for j in eachindex(B)] for a in arms)
    maa = Dict{String,Vector{Float64}}(a => [eligible_maa(pooled(d -> _budget_err(d, a, j))) for j in eachindex(B)] for a in arms)
    oracle_med = [med_of(d -> _budget_oracle(d, k)) for k in bi]
    oracle_maa = [eligible_maa(pooled(d -> _budget_oracle(d, k))) for k in bi]
    outcomes = Dict(a => pair_outcomes(pooled(d -> _budget_err(d, a, length(B)))) for a in arms)
    outcomes["oracle"] = pair_outcomes(pooled(d -> _budget_oracle(d, bi[end])))
    return (; budgets = B, med, maa, oracle_med, oracle_maa, outcomes)
end

"""
    emit_budget_time_panel(style::PaperStyle, out; row_label = "", budgets, maa, percall_us, legend_name,
                           xmax, xticks, maamax, panels = [(f, panel_title(f)) for f in FAMILY_ORDER])

The COMPUTE-EFFICIENCY companion of a budget panel: mAA (`maa[arm]`, aligned
with `budgets`, from [`budget_curves`](@ref)) against the cumulative LO time
per pair [ms] — at checkpoint j, j × the arm's deterministic single-thread
per-call time `percall_us[arm]` [µs] (one LO per pair per checkpoint). LO cost
only: scoring, sampling and termination are shared and excluded (the caption
must keep that scoping); the oracle runs no LO and has no curve. `xmax` /
`xticks` / `maamax` are the panel's fixed axis constants. Every arm of `maa`
must have a per-call time.
"""
function emit_budget_time_panel(style::PaperStyle, out::AbstractString; row_label::AbstractString = "",
                                budgets::AbstractVector{<:Integer}, maa::AbstractDict,
                                percall_us::AbstractDict, legend_name::AbstractString,
                                xmax::Real, xticks::AbstractString, maamax::Real,
                                panels = [(f, panel_title(f)) for f in FAMILY_ORDER])
    arms = [a for a in keys(maa) if has_arm(style, a)]
    untimed = setdiff(arms, keys(percall_us))
    isempty(untimed) || throw(ArgumentError("emit_budget_time_panel: no per-call time for $untimed"))
    L = length(budgets)
    cum = Dict(a => [j * percall_us[a] / 1e3 for j in 1:L] for a in arms)
    amin = floor(minimum(filter(isfinite, reduce(vcat, [maa[a] for a in arms]))) / 0.1) * 0.1
    atick = join([@sprintf("%.1f", y) for y in amin:0.1:maamax], ",")
    io = IOBuffer()
    println(io, raw"\begin{tikzpicture}")
    open_faceted_groupplot(io; cols = 3, rows = 1, width = _FACET_WIDTH, height = raw"0.22\linewidth",
        opts = filter(!isempty, ["xmin={0}, xmax={$(@sprintf("%g", xmax))}, xtick={$xticks}",
                                 style.axes.grid, _FONT_OPTS, _TITLE_STYLE]))
    for (ci, (obj, title)) in enumerate(panels)
        yopt = "ymin={$(@sprintf("%.2f",amin))}, ymax={$(@sprintf("%.2f",maamax))}, ytick={$atick}, minor y tick num={0}, yminorgrids, " *
               raw"yticklabel style={text width=2.4em, align=right}"
        xlab = ci == 2 ? raw", xlabel={cum.\ LO time per pair [ms]}" : ""
        leg  = ci == 1 ? faceted_legend_opts(legend_name) : ""
        println(io, "\\nextgroupplot[$yopt, title={$title}", _cell_ylabel(ci, row_label, style.axes.maa_ylabel), xlab, leg, "]")
        foreach(a -> _arm_curve(io, style, a, pgf_coords(cum[a], maa[a]; fmt = "(%.3f,%.4f)")),
                _facet_arms(style, arms, obj))
        ci == 1 && _arm_legend(io, style, _legend_order(style, arms, panels))
    end
    _close_groupplot(io, true, 1, legend_name)
    return _write_tex(out, io)
end

# -----------------------------------------------------------------------------
# The miscalibration summary (rows at three ζ, and their table).
# -----------------------------------------------------------------------------
"""
    miscal_summary_rows(style::PaperStyle, sidecar; stat = :med, exclude = Set{String}(),
                        arm_filter = a -> true)
        -> (arms, rows, zetas)

The deployed evaluation of a miscalibration sidecar summarized at three
points of its ζ ladder — the deflation extreme, calibration (ζ nearest 1) and
the inflation extreme — per arm of `style`, in the faceted order (facet, then
ramp step). `stat` is `:med` (finite median of the imc error, deg) or `:maa`
(the eligibility rule, `eligible_maa`). `rows[arm] = (deflation, calibration, inflation)`;
`zetas` the three ζ values.
"""
function miscal_summary_rows(style::PaperStyle, sidecar::AbstractString; stat::Symbol = :med,
                             exclude = Set{String}(), arm_filter = a -> true)
    d = load_tuning_artifact(sidecar)
    zetas = Float64.(d["zetas"]); ro = d["evaluation"]
    f = stat === :med ? finite_median : stat === :maa ? eligible_maa :
        error("miscal_summary_rows: stat must be :med or :maa, got $stat")
    jlo, jcal, jhi = argmin(zetas), argmin(abs.(zetas .- 1.0)), argmax(zetas)
    arms = sort([String(a) for a in keys(ro) if has_arm(style, a) && arm_filter(a) && !(a in exclude)];
                by = a -> (family_rank(family_of(style, a)), arm_order(style, a)...))
    rows = Dict(a => (c = evaluation_curve(ro, a, f); (c[jlo], c[jcal], c[jhi])) for a in arms)
    return arms, rows, (zetas[jlo], zetas[jcal], zetas[jhi])
end

"""
    emit_miscal_summary_table(style::PaperStyle, path; tracks, stat = :med, exclude = Set{String}(),
                              arm_filter = a -> true, label = a -> arm_record(style, a).label)

The miscalibration summary table of a set of cells, from their sidecars (the
same evaluations the faceted panels draw, so table and figure agree by
construction). `tracks = [(title, [(column_name, sidecar), …]), …]`: one
booktabs block per track, rows = the union of the track's arms of `style` in
the faceted order, per dataset three columns — calibration, the deflation
extreme and the inflation extreme of the ladder — with the best value per
column bold (min for `:med`, max for `:maa`) and `--` where a cell lacks the
arm. Returns `path`.
"""
function emit_miscal_summary_table(style::PaperStyle, path::AbstractString; tracks::AbstractVector,
                                   stat::Symbol = :med, exclude = Set{String}(), arm_filter = a -> true,
                                   label = a -> arm_record(style, a).label)
    fmt(v) = stat === :maa ? @sprintf("%.3f", v) : @sprintf("%.2f", v)
    best = stat === :maa ? maximum : minimum
    io = IOBuffer()
    println(io, "% Auto-generated (Render.emit_miscal_summary_table) from the product-tuning sidecars. DO NOT EDIT.")
    println(io, stat === :maa ? "% Values are mAA@10° at ζ = 1 (calibration), the deflation extreme and the inflation extreme." :
                                "% Values are med. max(ΔR,Δt) in degrees at ζ = 1 (calibration), the deflation extreme and the inflation extreme.")
    for (title, cells) in tracks
        data = [(cn, miscal_summary_rows(style, sc; stat, exclude, arm_filter)) for (cn, sc) in cells if isfile(sc)]
        isempty(data) && continue
        # row order: the union of the cells' arms in the faceted order (first occurrence wins)
        arms = String[]
        for (_, (as, _, _)) in data, a in as
            a in arms || push!(arms, a)
        end
        zl, zc, zh = data[1][2][3]
        ztex(z) = z ≈ round(z) ? @sprintf("%d", round(Int, z)) : @sprintf("\\tfrac{1}{%d}", round(Int, 1 / z))
        println(io, "\\begin{center}\\small\\setlength{\\tabcolsep}{3.5pt}\\renewcommand{\\arraystretch}{1.1}")
        println(io, "\\textbf{$title}\\\\[3pt]")
        println(io, "\\begin{tabularx}{\\linewidth}{@{}X" * "rrr"^length(data) * "@{}}")
        println(io, "\\toprule")
        println(io, "Pipeline & " * join(["\\multicolumn{3}{c}{$(cn)}" for (cn, _) in data], " & ") * " \\\\")
        for i in eachindex(data)
            c0 = 2 + 3(i - 1)
            print(io, "\\cmidrule(lr){$(c0)-$(c0+2)}")
        end
        println(io)
        sub = "\\multicolumn{1}{c}{\$\\zeta{=}$(ztex(zc))\$} & \\multicolumn{1}{c}{\$$(ztex(zl))\$} & \\multicolumn{1}{c}{\$$(ztex(zh))\$}"
        println(io, " & " * join(fill(sub, length(data)), " & ") * " \\\\")
        println(io, "\\midrule")
        for a in arms
            cells_tex = String[]
            for (_, (as, rows, _)) in data
                if !haskey(rows, a)
                    append!(cells_tex, ["--", "--", "--"]); continue
                end
                lo, cal, hi = rows[a]
                bcal = best(r[2] for r in values(rows)); blo = best(r[1] for r in values(rows)); bhi = best(r[3] for r in values(rows))
                b(v, bv) = v ≈ bv ? "\\textbf{$(fmt(v))}" : fmt(v)
                append!(cells_tex, [b(cal, bcal), b(lo, blo), b(hi, bhi)])
            end
            println(io, label(a) * " & " * join(cells_tex, " & ") * " \\\\")
        end
        println(io, "\\bottomrule\\end{tabularx}\\end{center}")
        println(io, "\\vspace{4pt}")
    end
    mkpath(dirname(path))
    write(path, String(take!(io)))
    println("rendered  ", basename(path))
    return path
end

# -----------------------------------------------------------------------------
# LO per-call timing table (µs/call) from lo_timing JLD2s.
# -----------------------------------------------------------------------------
"""
    lo_us_per_call(res, m) -> Float64

The mean warm LO wall clock per call [µs] of method column `m` of a budget
result (anything carrying `lo_ns_total` / `lo_calls_total` per method — a
budget engine's result or a run record read back from its sidecar); `NaN` when
the column made no LO call. A pure accessor on persisted data.
"""
lo_us_per_call(res, m::Integer) =
    res.lo_calls_total[m] == 0 ? NaN : res.lo_ns_total[m] / res.lo_calls_total[m] / 1e3

"""
    emit_lo_timing_table(figdir; timedir, arm_rows, dataset_names,
                         rider_labels, filename = "table_lo_timing.tex")

Emit the LO per-call wall-clock table (`<figdir>/<filename>`) from the
`lo_timing_*.jld2` files under `timedir`. `arm_rows` is the ordered
`[(display_label, run_tag), …]` row table; `dataset_names` maps dataset tag →
display name; `rider_labels` maps a rider method Symbol (a non-`:ig` column
of a run) → its arm label, `nothing` to skip. Each run's `:ig` column is
keyed by the run tag itself (the timing driver's tags ARE the base names, so
a roster extension cannot silently KeyError this emitter).
"""
function emit_lo_timing_table(figdir::AbstractString;
                              timedir::AbstractString,
                              arm_rows::AbstractVector,
                              dataset_names::AbstractDict,
                              rider_labels::AbstractDict,
                              filename::AbstractString = "table_lo_timing.tex")
    files = filter(f -> startswith(basename(f), "lo_timing_") && endswith(f, ".jld2"),
                   readdir(timedir; join = true))
    isempty(files) && (@warn "no lo_timing files" timedir; return)
    # collect: dataset tag -> arm -> µs/call
    got = Dict{String,Dict{String,Float64}}()
    for f in files
        d = JLD2.load(f); tag = d["tag"]
        col = get!(got, tag, Dict{String,Float64}())
        for r in d["runs"]
            for (mi, m) in enumerate(r.methods)
                us = lo_us_per_call(r, mi)
                if m === :ig
                    col[r.tag] = us                      # this run's strategy, keyed by tag
                elseif haskey(rider_labels, m) && rider_labels[m] !== nothing
                    col[rider_labels[m]] = us            # riders (e.g. MSAC/Le-Zach on the Stu-t run)
                end
            end
        end
    end
    tags = sort(collect(keys(got)))
    io = IOBuffer()
    println(io, "% Auto-generated.")
    println(io, "\\begin{tabular}{@{}l" * "r"^length(tags) * "@{}}")
    println(io, "\\toprule")
    hdr = [get(dataset_names, t, replace(t, "_" => "\\_")) for t in tags]
    n = length(tags)
    println(io, "Local optimizer & \\multicolumn{$n}{c}{Per-call wall-clock [\$\\mu\$s]} \\\\")
    println(io, "\\cmidrule(lr){2-$(n+1)}")
    println(io, " & " * join(["\\multicolumn{1}{c}{$h}" for h in hdr], " & ") * " \\\\")
    println(io, "\\midrule")
    for (disp, runtag) in arm_rows
        vals = [haskey(got[t], runtag) ? @sprintf("%.0f", got[t][runtag]) : "--" for t in tags]
        println(io, "$disp & " * join(vals, " & ") * " \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write(joinpath(figdir, filename), String(take!(io)))
    println("rendered  ", filename, "   (datasets: ", join(tags, ", "), ")")
    return joinpath(figdir, filename)
end

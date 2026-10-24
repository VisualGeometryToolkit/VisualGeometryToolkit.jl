# =============================================================================
# src/render/figure_style.jl — the objective-faceted figure-style GRAMMAR.
#
# The visual grammar every faceted panel obeys (moved from
# eval/accv2026/figure_style.jl; the concrete arm rows — probe instances +
# paper-owned colors/scale-interface — are paper config, built by
# `arm_style_records` into the paper's `PaperStyle`):
#
#   * Color HUE  = loss family:  Welsch → greens, TLS → oranges, GM → purples.
#   * SHADE      = optimization path within a family (light→dark):
#                    static / direct  →  HQ lift  →  GNC.
#   * MARKER     = optimization path:  o IRLS / triangle HQ lift / diamond GNC / oplus BCD /
#                    square truncated LM|LZ (PoseLib fixed-scale).
#   * LINE STYLE = scale interface: dashed only for fixed-EXTERNAL-scale arms
#                    (TLS-LM, TLS-LZ); solid for everything else.
#   * REFERENCES: oracle = solid black; perturbed-seed / gray refs = thin
#                    dashed black!55.
#
# DERIVATION (trait rewrite, 2026-07-18; two-axis rewrite 2026-09-06): label / marker / family / path / scale
# DERIVE from the RVG strategy types via the VGC identity traits (lo_label /
# lo_marker / loss_family / optimization_path / scale_interface) probed on one instance per deployed
# arm — the same ctors the paper's roster deploys, so legends cannot drift
# from the production strategies. Colors and the scale interface
# stay PAPER-OWNED (fig_colors.tex names), keyed PER ARM.
# =============================================================================

using VisualGeometryToolkitCore: lo_label, lo_marker, loss_family, optimization_path, scale_interface,
    FixedScale, UnspecifiedScale

"""
    arm_style_records(probes, table) -> Dict{String,NamedTuple}

The faceted-arm style records of a paper, keyed by arm: one record
`(family, path, scale, label, color, hex, marker, dashed, shade)` per
`arm_key => probe_instance` pair of `probes` (one LO instance per deployed arm;
ctor kwargs immaterial, the identity traits are type/variant-level), with
`table[arm_key]` the paper-owned part, `(color = …, [hex = …], [shade = …],
[scale = …])`: the colour NAME, optionally its hex (a [`PaperStyle`](@ref)
defines it under that name), the arm's step on its family's light→dark colour
ramp (`shade`, 1 = lightest; it fixes the within-facet legend/plot order) and
an optional scale tag, which must agree with the strategy's `scale_interface`.
`family`, `path`, `label`, `marker` and `dashed` (`FixedScale` ⇒ dashed) DERIVE
from the probe's identity traits. Errors on a probe with unspecified identity
traits (registry hygiene).
"""
function arm_style_records(probes, table)
    out = Dict{String,NamedTuple}()
    for (k, p) in probes
        row = table[k]
        iface = scale_interface(p)
        r = (family    = loss_family(p),
             path      = optimization_path(p),
             scale     = iface,
             label     = lo_label(p),
             color     = String(row.color),
             hex       = haskey(row, :hex) ? String(row.hex) : nothing,
             marker    = String(lo_marker(p)),
             dashed    = iface isa FixedScale,
             shade     = get(row, :shade, nothing))
        (r.family === :unspecified || r.path === :unspecified || iface isa UnspecifiedScale) &&
            error("figure_style: arm $k probe has unspecified loss_family/optimization_path/scale_interface traits")
        if haskey(row, :scale)
            (row.scale === :fixed_ext) == (iface isa FixedScale) ||
                error("figure_style: arm $k paper scale tag $(row.scale) disagrees with scale_interface $(iface)")
        end
        out[String(k)] = r
    end
    return out
end

# Loss-family facet order (left→right in every groupplot).
# Facet order of every faceted figure: the truncated / uniform-mixture
# baselines on the left, the two M-estimators side by side (maintainer
# decision 2026-09-07). Legends follow the same walk (see arm_order).
"The loss-family facet order, left to right, of every faceted figure and legend."
const FAMILY_ORDER = (:tls, :welsch, :gm)

# Mechanism sort rank WITHIN a family (legend + plot order): static → truncated
# → HQ → GNC, matching the light→dark shade progression. :bcd (exact
# score-objective coordinate ascent — StuTuBCD) sorts after :gnc; classical LO last: the
# uniform-mixtures column's second parameter-free row.
# Within a facet the TLS-objective arms stay contiguous (IRLS, HQ, GNC, then the
# classical fixed-loss-width LO); the Stu-tU BCD control closes the facet.
"The within-family sort rank of each optimization path (`arm_style_records`' `path`)."
const PATH_RANK = Dict(:seed => 0, :irls => 1, :hq => 2, :gnc => 3, :classical_lo => 4, :bcd => 5)
"Within-family optimization-path sort rank of arm `key` of `style`."
path_rank(style, key) = PATH_RANK[style.arms[key].path]
# The within-facet arm order (curves and legend entries): the paper's ramp step
# (`shade`, so a legend reads light→dark exactly as the curves do) when the
# paper declares one, else the optimization path; then label — deterministic,
# so every figure's legend lists arms identically.
"""
    arm_order(style, key) -> Tuple

The within-facet sort key of arm `key` of `style`: its ramp step (`shade`)
when the paper declares one, else its optimization path's rank; then its label.
"""
function arm_order(style, key)
    r = style.arms[key]
    return (r.shade === nothing ? path_rank(style, key) : Int(r.shade), r.label)
end


# ── accessors ────────────────────────────────────────────────────────────────
"Is `key` a faceted arm of `style`?"
has_arm(style, key) = haskey(style.arms, key)

"Full style record (NamedTuple) of arm `key` of `style`."
arm_record(style, key) = style.arms[key]

"Style tuple `(family, label, color, marker, dashed)` of arm `key` of `style`."
function arm_style(style, key)
    r = style.arms[key]
    return (family=r.family, label=r.label, color=r.color, marker=r.marker, dashed=r.dashed)
end

"Loss family (`:welsch`/`:tls`/`:gm`) of arm `key` of `style`."
family_of(style, key) = style.arms[key].family

"""
    family_rank(family) -> Int

The facet position of a loss `family` in [`FAMILY_ORDER`](@ref) (1 = leftmost);
a family outside it ranks after every facet. The facet-walk key of tables and
vocabularies that list arms in figure order.
"""
family_rank(family) = something(findfirst(==(family), FAMILY_ORDER), length(FAMILY_ORDER) + 1)

"Two-line panel title (TeX) for a loss `family`."
function panel_title(family)
    family === :welsch && return "Welsch"
    family === :tls    && return raw"Uniform mixtures\\(TLS, Stu-$t$U)"
    family === :gm     && return raw"Geman--McClure\\(GM)"
    error("no panel title for family $family")
end

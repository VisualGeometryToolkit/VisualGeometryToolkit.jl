# =============================================================================
# StyleFixtures — a PaperStyle for the render items, without an estimator
# package: probe LO types that declare VisualGeometryToolkitCore's identity traits
# (what a paper's roster strategies declare), and a style built from them with
# placeholder axis blocks.
# =============================================================================

@testsetup module StyleFixtures
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEval.Render: PaperStyle, arm_style_records

    export ProbeLO, probe_style

    "A probe strategy: its loss family, optimization path, label and scale interface."
    struct ProbeLO <: VgtkCore.AbstractLocalOptimization
        family::Symbol
        path::Symbol
        label::String
        fixed::Bool
    end
    ProbeLO(family, path, label) = ProbeLO(family, path, label, false)
    VgtkCore.loss_family(p::ProbeLO) = p.family
    VgtkCore.optimization_path(p::ProbeLO) = p.path
    VgtkCore.lo_label(p::ProbeLO) = p.label
    VgtkCore.scale_interface(p::ProbeLO) = p.fixed ? VgtkCore.FixedScale() : VgtkCore.ConsensusAnchored()

    "A PaperStyle over `probes` (`arm => ProbeLO`) and their `table` rows."
    probe_style(probes, table; colors = Pair{String,String}[], grid = "xmajorgrids, ymajorgrids") =
        PaperStyle(; colors, arms = arm_style_records(probes, table), grid,
                   zeta_x = "xmode={log}", zeta_med_y = (ymin; ymax = 6, ytick_spacing = 1) -> "ymin={$ymin}, ymax={$ymax}",
                   zeta_maa_y = "ymin={0.2}, ymax={1}", med_ylabel = "med", maa_ylabel = "mAA")
end

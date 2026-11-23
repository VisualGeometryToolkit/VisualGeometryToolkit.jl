# =============================================================================
# TuningFixtures — a `[tuning.<name>]` spec for the items that build one from a
# partial table: every spec names its data (a registered scene split, there is
# no default), so a table without a `cross_validation` gets the registered
# PhotoTourism split with every unit of each side.
# =============================================================================

@testsetup module TuningFixtures
    using VisualGeometryToolkitEval.Tuning: tuning_spec

    export fixture_spec, with_cross_validation, coarse_fine_axis

    const FIXTURE_CROSS_VALIDATION = Dict{String,Any}(
        "scene_split" => Dict{String,Any}("type" => "registered", "name" => "phototourism"))

    "`table` with the fixture cross-validation unless it states its own."
    with_cross_validation(table::AbstractDict) =
        haskey(table, "cross_validation") ? Dict{String,Any}(table) :
            merge(Dict{String,Any}(table), Dict{String,Any}("cross_validation" => FIXTURE_CROSS_VALIDATION))

    "The spec `[tuning.<name>]` = `table` (with the fixture cross-validation)."
    fixture_spec(table::AbstractDict; name::AbstractString = "t") =
        tuning_spec(Dict{String,Any}("tuning" => Dict{String,Any}(name => with_cross_validation(table))), name)

    "A joint coarse-fine axis table: a spaced coarse grid on the fixed window [lo, hi], its fine rule, and (given) its `expensive` flag."
    function coarse_fine_axis(lo, hi, spacing, fine; expensive = nothing)
        ax = Dict{String,Any}("coarse" => Dict{String,Any}("type" => "spaced",
            "window" => Dict{String,Any}("type" => "fixed", "lo" => lo, "hi" => hi), "spacing" => spacing), "fine" => fine)
        expensive === nothing || (ax["expensive"] = expensive)
        return ax
    end
end

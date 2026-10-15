# =============================================================================
# TuningFixtures — a `[tuning.<name>]` spec for the items that build one from a
# partial table: every spec names its data (a registered scene split, there is
# no default), so a table without a `cross_validation` gets the registered
# PhotoTourism split with every unit of each side.
# =============================================================================

@testsetup module TuningFixtures
    using VisualGeometryEval.Tuning: tuning_spec

    export fixture_spec, with_cross_validation

    const FIXTURE_CROSS_VALIDATION = Dict{String,Any}(
        "scene_split" => Dict{String,Any}("type" => "registered", "name" => "phototourism"))

    "`table` with the fixture cross-validation unless it states its own."
    with_cross_validation(table::AbstractDict) =
        haskey(table, "cross_validation") ? Dict{String,Any}(table) :
            merge(Dict{String,Any}(table), Dict{String,Any}("cross_validation" => FIXTURE_CROSS_VALIDATION))

    "The spec `[tuning.<name>]` = `table` (with the fixture cross-validation)."
    fixture_spec(table::AbstractDict; name::AbstractString = "t") =
        tuning_spec(Dict{String,Any}("tuning" => Dict{String,Any}(name => with_cross_validation(table))), name)
end

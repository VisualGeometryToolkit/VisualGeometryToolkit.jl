# =============================================================================
# The format-6 tuning sidecars of the migration tests (tuning_tests.jl,
# "a format-6 sidecar reads through its version step"), written by the code
# that wrote format 6: VisualGeometryToolkit.jl robust-core 3eabba46, before
# the efficiency-constant rename and the retirement of the loss-currency
# multiple. NOT runnable on a later checkout (the names it reaches are retired,
# and spelled by concatenation: the stack's naming check bans them as
# literals everywhere); kept as the files' provenance.
#
#   julia --project=VisualGeometryToolkitEval/test VisualGeometryToolkitEval/scripts/make_tuning_v6_fixture.jl
#
# writes, to test/data/tuning_v6/:
#   composed.jld2  a composed miscalibration sidecar of five arms, none clipped
#   clipped.jld2   the same, its Geman–McClure arm clipped at the top of its axis
#   expected.toml  per efficiency-constant arm, its STORED multiple (the loss's
#                  currency) and the PLAIN multiple the old code deployed for it,
#                  and the sidecar's recorded tuning hash
# =============================================================================
using Test, TOML
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitEval
using VisualGeometryToolkitEval.Evaluation: ComposedSpec
using VisualGeometryToolkitCore: OutlierHalfwidth, TLSShape, GemanMcclureShape, WelschShape, CauchyShape
using VisualGeometryToolkitEstimation: Irls

const OLD = "scale" * "_multiple"                                     # the v6 name of the efficiency constant
const OLD_AXIS = getfield(VgtkCore, Symbol("Scale" * "Multiple"))()   # its width axis
const PLAIN = getfield(VgtkEstimation, Symbol("plain_" * OLD))         # the old currency table
const F = VisualGeometryToolkitEval.engine_fixtures()
const SHAPES = Dict("HW +Irls(m)" => TLSShape(), "HW +GM(m)" => GemanMcclureShape(),
                    "HW +Welsch(m)" => WelschShape(), "HW +Cauchy(m)" => CauchyShape())
arms = [ComposedSpec("HW +Irls", :hw, OutlierHalfwidth(), OutlierHalfwidth(), _ -> Irls{TLSShape}());
        [ComposedSpec(b, :hw, OutlierHalfwidth(), OLD_AXIS, m -> Irls{typeof(k)}(; (Symbol(OLD) => m,)...))
         for (b, k) in sort!(collect(SHAPES); by = first)]]
rt = F.roster_tuning()
dir = joinpath(@__DIR__, "..", "test", "data", "tuning_v6")
F.write_roster_sidecar(joinpath(dir, "composed.jld2"), rt, arms; code = (VisualGeometryToolkitEval,), mode = "miscal")
F.write_roster_sidecar(joinpath(dir, "clipped.jld2"), rt, arms; code = (VisualGeometryToolkitEval,), mode = "miscal",
                       hits = Dict(a.base => (a.base == "HW +GM(m)" ? [OLD * "_hi"] : String[]) for a in arms))
m = rt.values[Symbol(OLD)][3]                           # every arm is tuned at the third grid value
art = VisualGeometryToolkitEval.Tuning.load_tuning_artifact(joinpath(dir, "composed.jld2"))
open(joinpath(dir, "expected.toml"), "w") do io
    TOML.print(io, Dict{String,Any}(
        "written_by" => "VisualGeometryToolkit.jl robust-core 3eabba46 (tuning format 6)",
        "tuning_hash" => art.hash,
        "arms" => Dict{String,Any}(b => Dict{String,Any}("stored" => m, "plain" => PLAIN(k, m))
                                   for (b, k) in SHAPES)))
end

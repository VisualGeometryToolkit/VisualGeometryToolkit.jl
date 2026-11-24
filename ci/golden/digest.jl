# The SHA-256 of a replacement's whole record for the given cases — the `new` value of a whole-case intended change
# (intended_changes.toml, `field = "*"`), computed exactly as replay.jl's `whole_case_change` does:
#
#   JULIA_LOAD_PATH="@:$PWD:@stdlib" julia -t 1 --startup-file=no --project=VisualGeometryToolkitEstimation/test \
#       ci/golden/digest.jl <group> <case id> [<case id> …]
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "bits.jl"))
include(joinpath(@__DIR__, "cases.jl"))
using SHA: SHA
using VisualGeometryToolkitCore
g = ARGS[1]
cases = ReplayCase[]
mod = golden_module(g)
Base.include(mod, joinpath(@__DIR__, "fixtures", g * ".jl"))
Base.include(mod, joinpath(@__DIR__, "replay", g * ".jl"))
Base.invokelatest(Base.invokelatest(getglobal, mod, Symbol("replay_", g)), cases)
for id in ARGS[2:end]
    c = only(filter(c -> c.id == id, cases))
    rec = outcome_record(() -> Base.invokelatest(c.run, c.fixture))
    println(id, "\tnew\t", _record_digest(rec))
    println(id, "\told\t", _record_digest(read_records(g)[id]))
end

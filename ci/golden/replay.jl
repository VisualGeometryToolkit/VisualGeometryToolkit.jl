# Replay the golden bit records on the migration branch: every recorded case of every group is solved by its
# replacement (`replay/<group>.jl`) on the same fixture (`fixtures/<group>.jl`), and the result must agree with
# the record on every leaf, `===` bit for bit, under the case's explicit field mapping. A recorded case without a
# replacement, a replacement without a record, and any differing leaf fail the gate.
#
#   JULIA_LOAD_PATH="@:$PWD:@stdlib" julia -t 1 --startup-file=no \
#       --project=VisualGeometryToolkitEstimation/test ci/golden/replay.jl [group ...]
#
# A deliberate behaviour change (a review fix that alters a recorded value) is never a loosened comparison: it is
# an entry of `intended_changes.toml` — group, case, recorded field, its exact old and new bits, and the reason —
# and the leaf must match both. An entry that matches no leaf is itself a failure (a stale exception).
#
# Exit status 0 only when every group replays 100 % (listed changes included).

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("replay.jl runs with one thread: julia -t 1")

include(joinpath(@__DIR__, "bits.jl"))
include(joinpath(@__DIR__, "cases.jl"))
using TOML: TOML
using SHA: SHA
# Main sees what it saw when the records were made (record.jl's `using VisualGeometryToolkitCore`): an error message that
# prints a type at its throw prints it as Main resolves it, so the two processes must agree on Main's bindings.
using VisualGeometryToolkitCore: VisualGeometryToolkitCore
using VisualGeometryToolkitCore

# The intended changes, by (group, case) => recorded path => entry.
const INTENDED = let d = Dict{Tuple{String, String}, Dict{String, Any}}()
    for c in get(TOML.parsefile(joinpath(@__DIR__, "intended_changes.toml")), "change", Any[])
        get!(d, (c["group"], c["case"]), Dict{String, Any}())[c["field"]] = c
    end
    d
end

# A whole-case intended change (`field = "*"`): a case whose OUTCOME changed kind — an exception that is now a value, or
# a value that is now an exception — so no leaf maps onto a recorded one. Exact on both sides: each side is
# `{ type = "exception", value = "<exception type>" }` or `{ type = "record", value = "<SHA-256 of the whole record>" }`
# (every leaf, `path\ttype\tvalue` lines, in walk order), the recorded side checked against the record, the new side
# against the replacement's outcome.
_outcome_matches(rec, side) =
    side["type"] == "exception" ? any(((p, t, v),) -> p == "exception" && v == side["value"], rec) :
    side["type"] == "record" && !any(((p, t, v),) -> p == "exception", rec) && _record_digest(rec) == side["value"]
function whole_case_change(old, new_record, c, used)
    _outcome_matches(old, c["old"]) && _outcome_matches(new_record, c["new"]) && (push!(used, "*"); return String[])
    return ["`*`: whole-case intended change does not match — recorded digest $(_record_digest(old)), new digest $(_record_digest(new_record)); listed old $(c["old"]), new $(c["new"])"]
end

const GROUPS = isempty(ARGS) ? sort!([splitext(f)[1] for f in readdir(RECORDS_DIR) if endswith(f, ".tsv")]) : ARGS

failed = false
total = 0
for g in GROUPS
    global failed, total
    recorded = read_records(g)
    cases = ReplayCase[]
    mod = golden_module(g)   # the group's own module (record.jl states why)
    Base.include(mod, joinpath(@__DIR__, "fixtures", g * ".jl"))
    Base.include(mod, joinpath(@__DIR__, "replay", g * ".jl"))
    Base.invokelatest(Base.invokelatest(getglobal, mod, Symbol("replay_", g)), cases)
    ids = Set(c.id for c in cases)
    want = Set(k for k in keys(recorded) if !endswith(k, "#fixture"))
    bad = String[]
    for (gg, id) in keys(INTENDED)
        gg == g && !(replace(id, r"#fixture$" => "") in ids) && push!(bad, "$id: an intended change is listed for a case not replayed")
    end
    for id in sort!(collect(setdiff(want, ids)))
        push!(bad, "$id: recorded, not replayed")
    end
    for id in sort!(collect(setdiff(ids, want)))
        push!(bad, "$id: replayed, not recorded")
    end
    n_ok = 0
    for c in cases
        c.id in want || continue
        # a fixture that holds an upstream result (a find feeding a refine) moves with it: its leaves may be listed as
        # intended changes of the case `<id>#fixture`, exact on both sides like any other
        fx_intended = get(INTENDED, (g, c.id * "#fixture"), Dict{String, Any}())
        fx_used = Set{String}()
        fx = compare_record(recorded[c.id * "#fixture"], bit_record(c.fixture_view(c.fixture)); map = c.fixture_map,
                            intended = fx_intended, used = fx_used)
        for p in setdiff(keys(fx_intended), fx_used)
            push!(fx, "intended change of `$p` is listed but matched no leaf (stale entry)")
        end
        isempty(fx) || (append!(bad, ["$(c.id) FIXTURE: $d" for d in fx]); continue)
        intended = get(INTENDED, (g, c.id), Dict{String, Any}())
        used = Set{String}()
        new_record = outcome_record(() -> Base.invokelatest(c.run, c.fixture))
        d = haskey(intended, "*") ? whole_case_change(recorded[c.id], new_record, intended["*"], used) :
            compare_record(recorded[c.id], new_record; map = c.map, intended, used)
        for p in setdiff(keys(intended), used)
            push!(d, "intended change of `$p` is listed but matched no leaf (stale entry)")
        end
        isempty(d) ? (n_ok += 1) : append!(bad, ["$(c.id): $x" for x in d])
    end
    total += n_ok
    println(rpad(g, 28), lpad(n_ok, 6), " / ", length(want), " cases bit-identical")
    if !isempty(bad)
        failed = true
        foreach(b -> println("    ", b), first(bad, 200))
        length(bad) > 200 && println("    … ", length(bad) - 200, " more")
    end
end
println("total ", total, " cases bit-identical", failed ? "; FAILED" : "")
exit(failed ? 1 : 0)

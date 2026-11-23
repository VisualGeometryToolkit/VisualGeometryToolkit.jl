# Replay the golden bit records on the migration branch: every recorded case of every group is solved by its
# replacement (`replay/<group>.jl`) on the same fixture (`fixtures/<group>.jl`), and the result must agree with
# the record on every leaf, `===` bit for bit, under the case's explicit field mapping. A recorded case without a
# replacement, a replacement without a record, and any differing leaf fail the gate.
#
#   JULIA_LOAD_PATH="@:$PWD:@stdlib" julia -t 1 --startup-file=no \
#       --project=VisualGeometryToolkitEstimation/test ci/golden/replay.jl [group ...]
#
# Exit status 0 only when every group replays 100 %.

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("replay.jl runs with one thread: julia -t 1")

include(joinpath(@__DIR__, "bits.jl"))
include(joinpath(@__DIR__, "cases.jl"))

const GROUPS = isempty(ARGS) ? sort!([splitext(f)[1] for f in readdir(RECORDS_DIR) if endswith(f, ".tsv")]) : ARGS

failed = false
total = 0
for g in GROUPS
    global failed, total
    recorded = read_records(g)
    cases = ReplayCase[]
    Base.include(Main, joinpath(@__DIR__, "fixtures", g * ".jl"))
    Base.include(Main, joinpath(@__DIR__, "replay", g * ".jl"))
    Base.invokelatest(Base.invokelatest(getglobal, Main, Symbol("replay_", g)), cases)
    ids = Set(c.id for c in cases)
    want = Set(k for k in keys(recorded) if !endswith(k, "#fixture"))
    bad = String[]
    for id in sort!(collect(setdiff(want, ids)))
        push!(bad, "$id: recorded, not replayed")
    end
    for id in sort!(collect(setdiff(ids, want)))
        push!(bad, "$id: replayed, not recorded")
    end
    n_ok = 0
    for c in cases
        c.id in want || continue
        fx = compare_record(recorded[c.id * "#fixture"], bit_record(c.fixture); map = c.fixture_map)
        isempty(fx) || (append!(bad, ["$(c.id) FIXTURE: $d" for d in fx]); continue)
        d = compare_record(recorded[c.id], outcome_record(() -> Base.invokelatest(c.run, c.fixture)); map = c.map)
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

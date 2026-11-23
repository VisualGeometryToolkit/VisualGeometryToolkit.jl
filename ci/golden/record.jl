# Record the golden bit records of every public estimator the Core 0.14 migration deletes, at the
# pre-migration main (3d8a06da). Run ONCE, from a checkout of that commit, never on the migration branch:
#
#   cd <checkout at 3d8a06da>
#   JULIA_LOAD_PATH="@:$PWD:@stdlib" julia -t 1 --startup-file=no \
#       --project=VisualGeometryToolkitEstimation/test <branch>/ci/golden/record.jl [group ...]
#
# (the Estimation test environment carries StableRNGs and the test fixtures' packages; the workspace root on the
# load path adds PoseLib, Eval, IO and Features from the same Manifest). Each group's fixtures are built by
# `fixtures/<group>.jl` — the SAME file the replay builds them with, so both sides solve identical inputs, and
# each fixture's own record is checked by the replay before any result is compared — and solved by the old
# estimators in `record/<group>.jl`. This file and `record/` call the retired names by design: they are the
# frozen old side, outside every package's `check_naming` scope (ci/ is no package).

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("record.jl runs with one thread: julia -t 1")

include(joinpath(@__DIR__, "bits.jl"))
include(joinpath(@__DIR__, "cases.jl"))

const GROUPS = isempty(ARGS) ? sort!([splitext(f)[1] for f in readdir(joinpath(@__DIR__, "record")) if endswith(f, ".jl")]) : ARGS

function _head(dir)
    return try
        readchomp(`git -C $dir rev-parse HEAD`)
    catch
        "unknown"
    end
end

using VisualGeometryToolkitCore
const SOURCE_COMMIT = _head(pkgdir(VisualGeometryToolkitCore))

for g in GROUPS
    cases = Case[]
    Base.include(Main, joinpath(@__DIR__, "fixtures", g * ".jl"))
    Base.include(Main, joinpath(@__DIR__, "record", g * ".jl"))
    Base.invokelatest(Base.invokelatest(getglobal, Main, Symbol("record_", g)), cases)
    entries = Pair{String, Vector{Tuple{String, String, String}}}[]
    t = time()
    for c in cases
        push!(entries, c.id * "#fixture" => bit_record(c.fixture))
        push!(entries, c.id => outcome_record(() -> Base.invokelatest(c.run, c.fixture)))
    end
    path = write_records(g, entries)
    open(joinpath(RECORDS_DIR, g * ".source"), "w") do io
        println(io, "commit\t", SOURCE_COMMIT)
        println(io, "julia\t", VERSION)
        println(io, "cases\t", length(cases))
        println(io, "leaves\t", sum(length(last(e)) for e in entries; init = 0))
    end
    println(rpad(g, 28), lpad(length(cases), 6), " cases  ", round(time() - t; digits = 1), " s  → ", path)
end

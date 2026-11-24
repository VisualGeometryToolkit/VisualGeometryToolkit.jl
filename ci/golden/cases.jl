# A golden case: an id, the fixture it solves and the solve. Shared by record.jl (the old estimator) and
# replay.jl (its replacement), which agree on ids and fixtures.

"""
    Case(id, fixture, run)

`run(fixture)` is the solve whose outcome is recorded (or replayed). `fixture` is everything the solve reads
that is DATA (correspondences, a guess, a camera); run settings (seeds, strategies, keywords) live in `run`.
The id names the family, the fixture, the setting and the seed, and never an estimator's function name (the
old and the new side share it).
"""
struct Case
    id::String
    fixture::Any
    run::Any
end

"""
    case!(cases, id, fixture, run)

Add a case; a repeated id is an error.
"""
function case!(cases, id::AbstractString, fixture, run)
    any(c -> c.id == id, cases) && error("golden case id `$id` is defined twice")
    push!(cases, Case(String(id), fixture, run))
    return cases
end

"""
    outcome_record(f) -> record

The bit record of `f()`'s result, or — when it throws — of the exception: its type and message under the
path `exception`, so a replacement must throw the same type (its message is compared too, unless the replay
maps it).
"""
function outcome_record(f)
    return try
        bit_record(f())
    catch err
        err isa InterruptException && rethrow()
        [("exception", "type", _tstr(typeof(err))), ("exception.msg", "String", escape_string(_portable_message(err)))]
    end
end

# An exception's message as recorded: its first line, without file paths or line numbers (a MethodError's
# "Closest candidates" list names the checkout's paths, which neither belong in a record nor survive the move).
function _portable_message(err)
    line = first(split(sprint(showerror, err; context = :module => _GoldenNoImports), '\n'))   # types fully qualified, whatever the process has loaded
    return replace(line, r"(@ \S+ )?(?<![\w.])/(?:[^\s/:]+/)+[^\s/:]+(:\d+)?" => "<path>")
end

const SEEDS = (1, 2, 3)

"""
    ReplayCase(id, fixture, run, map, fixture_map, fixture_view)

The replacement's side of a recorded case: `run(fixture)` calls the NEW entry, and `map` (a `FieldMap`) states
how its result's fields map onto the recorded ones. `fixture_map` does the same for the fixture, which differs
from the recorded one only where a fixture holds an upstream result (a find's result seeding a refine).
`fixture_view` is what the fixture is COMPARED as (`run` still gets the fixture itself): `identity`, or — for a
fixture holding an ARRAY of upstream results, which a record stores as one digest of its leaves in the old field
order, so no field mapping can reach it — a function that rebuilds the OLD leaf sequence from the new values (each
result as a tuple of its old fields, in their old order), so the array's digest is compared bit for bit. A view
never drops or loosens a leaf: every old leaf must be rebuilt, from the value that replaces it.
"""
struct ReplayCase
    id::String
    fixture::Any
    run::Any
    map::FieldMap
    fixture_map::FieldMap
    fixture_view::Any
end

"""
    replay!(cases, id, fixture, run; map = IDENTITY_MAP, fixture_map = IDENTITY_MAP, fixture_view = identity)

Add the replacement of the recorded case `id`; a repeated id is an error (`fixture_view`: [`ReplayCase`](@ref)).
"""
function replay!(cases, id::AbstractString, fixture, run; map::FieldMap = IDENTITY_MAP, fixture_map::FieldMap = IDENTITY_MAP,
                 fixture_view = identity)
    any(c -> c.id == id, cases) && error("replay case id `$id` is defined twice")
    push!(cases, ReplayCase(String(id), fixture, run, map, fixture_map, fixture_view))
    return cases
end

"""
    golden_module(g) -> Module

A fresh module for the group `g`, into which record.jl and replay.jl include its fixture file and its record or
replay file: the group sees the framework names imported here and its own bindings, never another group's, whatever
groups the process ran before it (record.jl states the defect this prevents).
"""
function golden_module(g)
    mod = Module(Symbol("golden_", g))
    Core.eval(mod, :(include(path) = Base.include($mod, path)))   # a bare Module has no `include` of its own
    Core.eval(mod, :(using Main: case!, replay!, SEEDS, FieldMap, IDENTITY_MAP, bit_record, Case, ReplayCase))
    return mod
end

# The SHA-256 of a whole record (`path\ttype\tvalue` lines in walk order): a whole-case intended change's value
# (replay.jl `whole_case_change`, digest.jl).
_record_digest(rec) = bytes2hex(SHA.sha256(join(("$p\t$t\t$v" for (p, t, v) in rec), "\n")))

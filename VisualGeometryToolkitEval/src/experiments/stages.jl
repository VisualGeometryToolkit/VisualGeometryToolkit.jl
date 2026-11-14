# =============================================================================
# src/experiments/stages.jl — the experiment stage runner, dependency-free.
#
# One experiment file = one deliverable; each file declares its DATA INPUTS
# (`require_input` — hard error with a provenance message) and its STAGES
# (`Stage(name, desc; steps, run)` — the stage's keyed steps, each produced or
# loaded by `produce`; `Stage(name, desc; check)` — a check run every time),
# and ends with `run_stages(name, STAGES)`.
#
# CLI (uniform across every experiment file):
#   --plan          print the stage plan (done/pending) and exit — the
#                   structural check gate before any sbatch launch
#   --stage=NAME    run exactly one stage, or a comma-separated subset
#   --force         produce the selected stages' steps again
#
# DEPENDENCY-FREE BY CONTRACT (Dates is a stdlib; `cli_option` comes from the
# includer — src/cli.jl): the Experiments module includes it, AND the
# papers' external-arm environments (PythonCall — they cannot load this
# package) include it BY PATH, like the house style source axis_style.jl and
# the single CV path cv_probe.jl. One copy for both.
# =============================================================================
using Dates: Dates

"""
    Stage(name::Symbol, desc::String; steps, run)
    Stage(name::Symbol, desc::String; check)

One stage of a codified experiment. A stage that PRODUCES declares its keyed steps: `steps()` returns them (an experiment's
`Step`s, cheap to build: no GPU, no heavy loads) and `run(step)` produces one — it calls `produce`, which loads a produced
step and produces a pending one (DrWatson's `produce_or_load`), so `run` is called for every step of the stage, and a stage
whose steps are all produced costs their checks only. A stage that only CHECKS (its inputs present, a refusal) declares
`check()`, run every time. Declared in figure/table order and executed by [`run_stages`](@ref).

The done-closure form `Stage(name, desc, done, run)` is retired: a step's own record says whether it is produced.
"""
struct Stage
    name::Symbol
    desc::String
    steps::Union{Nothing, Function}   # () -> the stage's steps (a producing stage); nothing for a check
    run::Function                     # step -> nothing (produce it), or () -> nothing (a check)
    function Stage(name::Symbol, desc::AbstractString; steps = nothing, run = nothing, check = nothing)
        check === nothing && steps isa Function && run isa Function && return new(name, String(desc), steps, run)
        check isa Function && steps === nothing && run === nothing && return new(name, String(desc), nothing, check)
        throw(ArgumentError("Stage $name: give `steps` and `run` (a stage that produces keyed steps), or `check` alone"))
    end
end
Stage(name::Symbol, desc::AbstractString, done, run) =
    throw(ArgumentError("Stage(name, desc, done, run) is retired: a stage declares its keyed steps, " *
                        "Stage(name, desc; steps = () -> [Step(…), …], run = step -> produce(…)), or Stage(name, desc; check)"))

# Whether a step is produced, and the unmarking of a step that --force reruns: methods for the experiments' Step
# (steps.jl), so this file stays dependency-free.
function _step_produced end
function _step_unmark! end

# A stage's status for the plan: "check", or "<produced>/<steps> produced"; a steps() that throws is "?" (the plan must print).
function _stage_status(st::Stage)
    st.steps === nothing && return "check"
    try
        xs = st.steps()
        return "$(count(_step_produced, xs))/$(length(xs)) produced"
    catch err
        println("   [$(st.name)] steps ERROR: $(sprint(showerror, err))")
        return "?"
    end
end

"""
    run_stages(experiment::AbstractString, stages; argv = ARGS)

The uniform experiment entrypoint: `--plan` prints the stage table (each stage's produced steps of its steps, or `check`) and
exits (the structural check); otherwise runs the stages in declaration order (`--stage=NAME` restricts to one, or to a
comma-separated subset — declaration order is still honored). A producing stage calls its `run` on EVERY one of its steps,
which loads a produced step and produces a pending one; its steps are listed when the stage runs, so a step an earlier stage
re-keyed is the new one. `--force` unmarks the selected stages' steps first (their provenance removed), so each is produced
again. A stage failure aborts the run (later stages depend on earlier ones).
"""
function run_stages(experiment::AbstractString, stages::Vector{Stage}; argv = ARGS)
    plan  = "--plan" in argv
    force = "--force" in argv
    # A subset lets one experiment file be split across sbatch shapes — e.g.
    # the GPU stage on a gpu box, then the CPU-only stages on a big-core box.
    only  = let s = cli_option(argv, "stage", ""); isempty(s) ? String[] : split(s, ',') end
    println("== experiment $experiment  ($(length(stages)) stages)  " *
            Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    width = maximum(st -> length(String(st.name)), stages; init = 0)
    for st in stages   # the name, its status and its description, always separated (a long name never runs into its status)
        println("   ", rpad(String(st.name), width), "  ", rpad(_stage_status(st), 16), " ", st.desc)
    end
    plan && (println("== plan only (--plan) — nothing run"); return nothing)
    for st in stages
        isempty(only) || String(st.name) in only || continue
        println("== run  $(st.name)  ", Dates.format(Dates.now(), "HH:MM:SS"))
        flush(stdout)
        if st.steps === nothing
            st.run()
        else
            xs = st.steps()                                   # listed now: an earlier stage of this run may have re-keyed them
            force && foreach(_step_unmark!, xs)
            foreach(st.run, xs)
        end
        println("== done $(st.name)  ", Dates.format(Dates.now(), "HH:MM:SS"))
        flush(stdout)
    end
    return nothing
end

"""
    require_input(path; what, provenance, sha256 = nothing)

Declared data input of an experiment: hard error (with the provenance recipe)
when absent. Inputs are declared at the top of each experiment file so the
dependency graph is readable without running anything. The check is the
runner's and `verify_inputs`' one, `inspect_input` (src/code_identity.jl): with
`sha256` (the input's content hash of record, 64 hex digits) another content is
refused (`ArgumentError`: what, the path, expected and actual) — a file's
`file_sha256`, a directory's `tree_sha256` — and so is a scratch run's output,
unless this run is one ([`scratch_run`](@ref)).
"""
function require_input(path::AbstractString; what::AbstractString,
                       provenance::AbstractString, sha256 = nothing)
    p = rstrip(abspath(path), '/')
    t = Dict{String, Any}("path" => basename(p))
    sha256 === nothing || (t["sha256"] = String(sha256))
    x = inspect_input(what, t, dirname(p); allow_scratch = scratch_run())          # the one check (src/code_identity.jl)
    x.state === :missing && error("MISSING INPUT [$what]: $path\n  provenance: $provenance")
    x.state === :refused && throw(ArgumentError("refused: input [$what] $path is not the data of record: $(x.problem)"))
    return path
end

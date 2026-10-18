# =============================================================================
# src/experiments/stages.jl — THE experiment stage runner, dependency-free.
#
# One experiment file = one deliverable; each file declares its DATA INPUTS
# (`require_input` — hard error with a provenance message) and its STAGES
# (`Stage(name, desc, done, run)` — idempotent: a stage whose `done()` holds is
# skipped unless --force), and ends with `run_stages(name, STAGES)`.
#
# CLI (uniform across every experiment file):
#   --plan          print the stage plan (done/pending) and exit — THE
#                   structural check gate before any sbatch launch
#   --stage=NAME    run exactly one stage, or a comma-separated subset
#                   (still skip-if-done; add --force)
#   --force         re-run stages even when their done-predicate holds
#
# DEPENDENCY-FREE BY CONTRACT (Dates is a stdlib; `cli_option` comes from the
# includer — src/cli.jl): the Experiments module includes it, AND the
# papers' external-arm environments (PythonCall — they cannot load this
# package) include it BY PATH, like the house style source axis_style.jl and
# the single CV path cv_probe.jl. One copy for both.
# =============================================================================
using Dates: Dates

"""
    Stage(name::Symbol, desc::String, done, run)

One idempotent step of a codified experiment: `done()` is a CHEAP predicate
(no GPU, no heavy loads) — a stage whose `done()` holds is skipped unless
`--force`; `run()` performs the work (often a `run_driver` shell-out) and may
assume every earlier stage completed. Declared in figure/table order and
executed by [`run_stages`](@ref).
"""
struct Stage
    name::Symbol
    desc::String
    done::Function     # () -> Bool   (cheap; no GPU, no heavy loads)
    run::Function      # () -> nothing
end

# A done-check that throws is pending, never fatal (the plan must still print).
function _stage_done(st::Stage)
    try
        return st.done()
    catch err
        println("   [$(st.name)] done-check ERROR: $(sprint(showerror, err)) — treated as pending")
        return false
    end
end

"""
    run_stages(experiment::AbstractString, stages; argv = ARGS)

The uniform experiment entrypoint: `--plan` prints the stage table and exits
(the structural check); otherwise runs every pending stage in declaration
order (`--stage=NAME` restricts to one, or to a comma-separated subset —
declaration order is still honored; `--force` re-runs done stages). The
printed table is a snapshot; each stage's `done()` is RE-EVALUATED
immediately before it would be skipped, because an earlier stage of the same
run may have invalidated it. A stage failure aborts the run (later stages
depend on earlier ones).
"""
function run_stages(experiment::AbstractString, stages::Vector{Stage}; argv = ARGS)
    plan  = "--plan" in argv
    force = "--force" in argv
    # A subset lets one experiment file be split across sbatch shapes — e.g.
    # the GPU CV stage on a gpu box, then the CPU-only stages (with --force)
    # on a big-core box that has no GPU.
    only  = let s = cli_option(argv, "stage", ""); isempty(s) ? String[] : split(s, ',') end
    println("== experiment $experiment  ($(length(stages)) stages)  " *
            Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    for st in stages
        println("   ", rpad(String(st.name), 16), _stage_done(st) ? "DONE     " : "pending  ", st.desc)
    end
    plan && (println("== plan only (--plan) — nothing run"); return nothing)
    for st in stages
        isempty(only) || String(st.name) in only || continue
        # RE-CHECK at execution time, not the plan snapshot above: a stage's
        # input is often (re)written by an earlier stage of the SAME run, so a
        # cached "DONE" silently skips work that has just been invalidated
        # (2026-09-09: timing_table's emit/design_table were skipped after
        # time_pt/time_scannet rewrote the measurements they read).
        if !force && _stage_done(st)
            println("== skip $(st.name) (done; --force to re-run)")
            continue
        end
        println("== run  $(st.name)  ", Dates.format(Dates.now(), "HH:MM:SS"))
        flush(stdout)
        st.run()
        println("== done $(st.name)  ", Dates.format(Dates.now(), "HH:MM:SS"))
        flush(stdout)
    end
    return nothing
end

"""
    require_input(path; what, provenance)

Declared data input of an experiment: hard error (with the provenance recipe)
when absent. Inputs are declared at the top of each experiment file so the
dependency graph is readable without running anything.
"""
function require_input(path::AbstractString; what::AbstractString,
                       provenance::AbstractString)
    isfile(path) || isdir(path) ||
        error("MISSING INPUT [$what]: $path\n  provenance: $provenance")
    return path
end

"""
    fresher(out, inputs...) -> Bool

The standard [`Stage`](@ref) done-predicate: `out` exists, every input
exists, and `out` is at least as new as each input — a stale panel from a
missing/renewed input is NOT done.
"""
fresher(out::AbstractString, inputs::AbstractString...) =
    isfile(out) && all(i -> ispath(i) && mtime(out) >= mtime(i), inputs)

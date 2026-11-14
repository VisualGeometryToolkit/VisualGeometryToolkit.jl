# =============================================================================
# src/experiments/stages.jl — the experiment stage runner, dependency-free.
#
# One experiment file = one deliverable; each file declares its DATA INPUTS
# (`require_input` — hard error with a provenance message) and its STAGES
# (`Stage(name, desc; items, product, run)` — each item's keyed product, produced or
# loaded by `produce`; `Stage(name, desc; check)` — a check run every time),
# and ends with `run_stages(name, STAGES)`.
#
# CLI (uniform across every experiment file):
#   --plan          print the stage plan (done/pending) and exit — the
#                   structural check gate before any sbatch launch
#   --stage=NAME    run exactly one stage, or a comma-separated subset
#   --force         produce the selected stages' products again
#
# DEPENDENCY-FREE BY CONTRACT (Dates is a stdlib; `cli_option` comes from the
# includer — src/cli.jl): the Experiments module includes it, AND the
# papers' external-arm environments (PythonCall — they cannot load this
# package) include it BY PATH, like the house style source axis_style.jl and
# the single CV path cv_probe.jl. One copy for both.
# =============================================================================
using Dates: Dates

"""
    Stage(name::Symbol, desc::String; items, product, run)
    Stage(name::Symbol, desc::String; check)

One stage of a codified experiment. A stage that PRODUCES declares its items (`items()`: the cases, sets or names it treats —
cheap: no GPU, no heavy loads), each item's keyed product (`product(item)`, an experiment's `Product`) and its runner (`run(item)`,
which calls `produce`: a produced product loads, a pending one is produced — DrWatson's `produce_or_load`), so `run` is called for
every item of the stage and a stage whose products are all produced costs their checks only. A stage that only CHECKS (its inputs
present, a refusal, a deploy) declares `check()`, run every time. Declared in figure/table order and executed by
[`run_stages`](@ref).

The done-closure form `Stage(name, desc, done, run)` is retired: a product's own record says whether it is produced.
"""
struct Stage
    name::Symbol
    desc::String
    items::Union{Nothing, Function}     # () -> the stage's items (a producing stage); nothing for a check
    product::Union{Nothing, Function}   # item -> its Product
    run::Function                       # item -> nothing (produce its product), or () -> nothing (a check)
    function Stage(name::Symbol, desc::AbstractString; items = nothing, product = nothing, run = nothing, check = nothing)
        check === nothing && items isa Function && product isa Function && run isa Function && return new(name, String(desc), items, product, run)
        check isa Function && items === nothing && product === nothing && run === nothing && return new(name, String(desc), nothing, nothing, check)
        throw(ArgumentError("Stage $name: give `items`, `product` and `run` (a stage that produces keyed products), or `check` alone"))
    end
end
Stage(name::Symbol, desc::AbstractString, done, run) =
    throw(ArgumentError("Stage(name, desc, done, run) is retired: a stage declares its keyed products, " *
                        "Stage(name, desc; items = () -> […], product = item -> Product(…), run = item -> produce(…)), or Stage(name, desc; check)"))

# Whether a product is produced, and the unmarking of a product that --force reruns: methods for the experiments' Product
# (products.jl), so this file stays dependency-free.
function _product_produced end
function _product_unmark! end

# A stage's status for the plan: "check", or "<produced>/<products> produced"; an items() that throws is "?" (the plan must print).
function _stage_status(st::Stage)
    st.items === nothing && return "check"
    try
        xs = st.items()
        return "$(count(x -> _product_produced(st.product(x)), xs))/$(length(xs)) produced"
    catch err
        println("   [$(st.name)] products ERROR: $(sprint(showerror, err))")
        return "?"
    end
end

"""
    run_stages(experiment::AbstractString, stages; argv = ARGS)

The uniform experiment entrypoint: `--plan` prints the stage table (each stage's produced products of its items, or `check`) and
exits (the structural check); otherwise runs the stages in declaration order (`--stage=NAME` restricts to one, or to a
comma-separated subset — declaration order is still honored). A producing stage calls its `run` on EVERY one of its items, which
loads a produced product and produces a pending one; its items are listed when the stage runs, so a product an earlier stage
re-keyed is the new one. `--force` unmarks each product of the selected stages (its provenance removed) just before its item
runs, so each is produced again and a failure leaves the later ones as they were. A stage failure aborts the run (later stages depend on earlier ones).
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
    for st in stages   # the name, (under --plan) its status, and its description, always separated
        println("   ", rpad(String(st.name), width), "  ", plan ? rpad(_stage_status(st), 16) * " " : "", st.desc)
    end
    plan && (println("== plan only (--plan) — nothing run"); return nothing)
    for st in stages
        isempty(only) || String(st.name) in only || continue
        println("== run  $(st.name)  ", Dates.format(Dates.now(), "HH:MM:SS"))
        flush(stdout)
        if st.items === nothing
            st.run()
        else
            xs = st.items()                                   # listed now: an earlier stage of this run may have re-keyed them
            for x in xs                                       # --force unmarks each product just before its own run
                force && _product_unmark!(st.product(x))
                st.run(x)
            end
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

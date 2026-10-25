# =============================================================================
# Experiments — the paper-generic experiment-codification framework (promoted
# from papers/accv2026/experiments/common.jl; the paper file is now a thin
# shim that sets the driver context and keeps the paper constants).
#
# One experiment file = one figure/table; each file declares
#
#   * its DATA INPUTS  (require_input — hard error with provenance message),
#   * its STAGES       (Stage(name, desc, done, run) — idempotent: a stage
#                       whose `done()` is true is skipped unless --force),
#   * and ends by RENDERING its artifact into the paper repo's figures/.
#
# CLI (uniform across every experiment file):
#   --plan          print the stage plan (done/pending) and exit — the
#                   structural check gate before any sbatch launch
#   --stage=NAME    run exactly one stage (still skip-if-done; add --force)
#   --force         re-run stages even when their done-predicate holds
#
# CV DISCIPLINE: no experiment file implements a sweep/argmin. All tuning is
# The single CV path (package Evaluation cv_probe.jl, with boundary_hits
# detection + read-time refusal); every sidecar is audited for interior
# argmins BEFORE consumption.
#
# The codification CONTRACT itself (what an experiment file may/must contain)
# is executable: `contract_violations(file)` backs test/contracts_tests.jl and
# the paper repos' contract suites.
#
# The stage runner (Stage / run_stages / require_input / fresher) lives in the
# dependency-free experiments/stages.jl, which the papers' external-arm
# environments include by path; this module includes the same file.
# =============================================================================
"""
    Experiments

The paper-generic experiment-codification framework: the [`Stage`](@ref)
table + [`run_stages`](@ref) entrypoint (`--plan`/`--stage`/`--force`), the
[`run_driver`](@ref) shell-out, [`require_input`](@ref)/[`fresher`](@ref)
declarations, the run records (the configuration-of-record layer), the
executable codification contract ([`contract_violations`](@ref)) and the one
data layer of every paper project: the root ladder and artifact view
([`set_project_root!`](@ref), [`data_root`](@ref), [`ArtifactView`](@ref)),
the input write guard ([`assert_not_input`](@ref)), the provenance record
([`write_provenance`](@ref), [`read_provenance`](@ref)), the keyed cache
([`cached_result`](@ref)) and the data policy ([`data_policy_violations`](@ref)).

This module holds what every paper uses unchanged: the runner mechanics and
the contract rules. A paper's own decisions stay in that paper's project: its
stage rosters and input paths, its driver wiring (set with
[`set_driver_context!`](@ref)), its artifact destinations, splits and oracle
constants.
"""
module Experiments

using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VGC   # check_keys (run_records.jl)
using ..Tuning: resolve_tuning!, _toml_hash
import ..cli_option   # src/cli.jl: stages.jl reads `--stage=` through it

export cancelled, progress, Stage, run_stages, run_driver, set_driver_context!, require_input,
       fresher, contract_violations,
       # run records (experiments/run_records.jl): the configuration-of-record layer
       load_run_config, run_key, RUN_KEY_EXCLUDED, RUN_CONFIG_KEYS, experiment_dir,
       write_run_records, oracle_floors_path,
       # the data/results binding (experiments/data.jl)
       set_project_root!, project_root, data_root, results_root, is_primary_copy,
       assert_primary_copy, assert_not_input, ArtifactView, install_artifact, link_tree!,
       artifact_view!, is_artifact_view,
       # the provenance record (experiments/provenance.jl)
       write_provenance, read_provenance, provenance_inputs, commit_drift, warn_commit_drift,
       run_complete, legacy_hash_key, cached_result,
       # the data policy (experiments/data_policy.jl)
       DataPolicyAllow, data_policy_violations,
       # re-keying (experiments/rekey.jl): the old → new key map of a tag rename
       TagRename, rename_tags, rekey_map, write_rekey_map, read_rekey_map

# Cooperative hooks of a kaimon warm-REPL run: no dependency on kaimon; they act
# only when the session has defined `Main.KaimonGate`.
"""
    cancelled() -> Bool

Whether a kaimon `cancel_eval` was requested for the running evaluation
(`Main.KaimonGate.is_cancelled()`); `false` outside a kaimon session. Poll it from
long loops, e.g. `live_runs(...; cancelled = Experiments.cancelled)`.
"""
cancelled() = isdefined(Main, :KaimonGate) && Base.invokelatest(Main.KaimonGate.is_cancelled)::Bool

"""
    progress(msg) -> Bool

Report `msg` as the running evaluation's progress (`Main.KaimonGate.progress(msg)`)
and return `true`; a no-op returning `false` outside a kaimon session.
"""
progress(msg::AbstractString) =
    isdefined(Main, :KaimonGate) && (Base.invokelatest(Main.KaimonGate.progress, msg); true)

# The stage runner — Stage, run_stages, require_input, fresher.
include("experiments/stages.jl")

# -----------------------------------------------------------------------------
# Driver shell-out. The paper config supplies the driver directory + project
# (set_driver_context! or per-call kwargs); with neither, both default to the
# CALLING project's directory — so an external paper repo that runs its
# drivers out of its own project root needs no configuration at all.
# -----------------------------------------------------------------------------
const _DRIVER_DIR     = Ref{String}("")
const _DRIVER_PROJECT = Ref{String}("")

"""
    set_driver_context!(driver_dir; project = driver_dir)

Register the default script directory (and Julia project) `run_driver`
resolves against — called once by a paper's experiments/common.jl shim.
"""
function set_driver_context!(driver_dir::AbstractString;
                             project::AbstractString = driver_dir)
    _DRIVER_DIR[]     = String(driver_dir)
    _DRIVER_PROJECT[] = String(project)
    return nothing
end

_active_project_dir() = dirname(something(Base.active_project(), pwd()))
default_driver_dir()     = isempty(_DRIVER_DIR[])     ? _active_project_dir() : _DRIVER_DIR[]
default_driver_project() = isempty(_DRIVER_PROJECT[]) ? _active_project_dir() : _DRIVER_PROJECT[]

"""
    run_driver(script, args...; driver_dir = default_driver_dir(),
               project = default_driver_project(),
               env = Dict(), threads = get(ENV, "SLURM_CPUS_PER_TASK", "auto"))

Shell out to a paper driver in a fresh Julia process (the drivers parse ARGS
at include time, so in-process include is not composable). A relative
`script` resolves against `driver_dir`; the child runs `--project=<project>`.
Errors propagate (`run` throws on nonzero exit).
"""
function run_driver(script::AbstractString, args...;
                    driver_dir::AbstractString = default_driver_dir(),
                    project::AbstractString = default_driver_project(),
                    env::AbstractDict = Dict{String,String}(),
                    threads::Union{AbstractString,Int} = get(ENV, "SLURM_CPUS_PER_TASK", "auto"))
    jl = joinpath(Sys.BINDIR, "julia")
    path = isabspath(script) ? String(script) : joinpath(driver_dir, script)
    cmd = Cmd([jl, "-t", string(threads), "--project=$(project)",
               path, string.(args)...])
    println(">> ", script, " ", join(string.(args), " "),
            isempty(env) ? "" : "   env: " * join(["$k=$v" for (k, v) in env], " "))
    flush(stdout)
    run(setenv(cmd, merge(copy(ENV), Dict(String(k) => String(v) for (k, v) in env))))
    return nothing
end

# =============================================================================
# The experiment-codification CONTRACT, as a checkable API (test/contracts_tests.jl
# asserts each rule; the paper projects' contract suites run it on their files).
#
# FORBIDDEN in experiment files:
#   F1  grid options of any kind in emitted axis styling — the house style is
#       gridless and lives in src/render/axis_style.jl.
#   F2  scientific-notation axis options — plain tick numbers only.
#   F3  hand-rolled CV sweeps (LinRange() — every tune goes through the
#       package coarse-fine search (coarse_fine_search and its scalar forms).
#   F4  local axis-style definitions (fonts belong to the package style module).
# REQUIRED:
#   R1  PGFPlots emitters consume the package style source (HOUSE_AXIS_COMMON)
#       and emit a caption snippet.
#   R2  scalar tuning goes through the single CV path.
#   R3  a provenance block + the stage runner (support/worker files exempt).
# =============================================================================

# Strip line comments conservatively (keeps strings intact enough for greps —
# the forbidden tokens never legitimately appear in code, commented or not,
# EXCEPT in prose; so scan code+strings but drop full-line comments).
_strip_comment_lines(src) =
    join([l for l in split(src, '\n') if !startswith(lstrip(l), "#")], '\n')

"""
    contract_violations(file) -> Vector{String}

The codification-contract check for one experiment file: returns a
human-readable violation per broken rule (empty ⇒ compliant). Support files
(registry.jl and `*common.jl`, e.g. a paper's shared external-arm plumbing)
are exempt from R3; `*_worker.jl` subroutines are
exempt from its stage-runner requirement (they are driven by their parent
experiment) but still carry a provenance line.
"""
function contract_violations(file::AbstractString)
    src_full = read(file, String)
    src = _strip_comment_lines(src_full)
    emits_axes = occursin("begin{groupplot}", src) || occursin("begin{axis}", src)
    support = basename(file) == "registry.jl" || endswith(basename(file), "common.jl")
    worker  = endswith(basename(file), "_worker.jl")
    v = String[]
    # F1 — no grid options anywhere in emitted styling.
    # pgfplots' `grid=major|minor|both|none` (a Julia binding such as `grid = Dict(...)` is not styling)
    occursin(r"\bgrid\s*=\s*(major|minor|both|none)\b", src) && push!(v, "F1: grid= axis option")
    occursin("xmajorgrids", src) && push!(v, "F1: xmajorgrids")
    occursin("ymajorgrids", src) && push!(v, "F1: ymajorgrids")
    occursin("yminorgrids", src) && push!(v, "F1: yminorgrids")
    occursin("grid style=", src) && push!(v, "F1: grid style=")
    # F2 — no scientific-notation axis options.
    occursin(r"scaled\s+(x |y )?ticks\s*=\s*\{?true", src) && push!(v, "F2: scaled ticks=true")
    occursin("number format/sci", src) && push!(v, "F2: number format/sci")
    # F3 — no hand-rolled CV grids.
    occursin("LinRange(", src) && push!(v, "F3: LinRange( sweep")
    # F4 — no local axis-style definitions (fonts belong to the package style
    # module; generators splice HOUSE_AXIS_COMMON).
    occursin("tick label style={font", src) && push!(v, "F4: local tick label font")
    occursin("label style={font", src) && push!(v, "F4: local label font")
    if emits_axes
        # R1 — package style consumption + caption snippet.
        occursin("HOUSE_AXIS_COMMON", src) || push!(v, "R1: emits axes without HOUSE_AXIS_COMMON")
        occursin("_caption.tex", src) || occursin("caption", lowercase(src)) ||
            push!(v, "R1: emits axes without a caption snippet")
    end
    if occursin("tuning_criterion", src) || occursin("x_star", src)
        # R2 — the single CV path only.
        occursin("coarse_fine_search(", src) || occursin("coarse_fine_cv(", src) || occursin("probe_curve(", src) ||
            occursin("fine_stage(", src) || occursin("deployed_", src) ||
            occursin("thr_cv", src) ||   # readers of CV sidecars, not tuners
            push!(v, "R2: tunes a scalar off the single CV path")
    end
    # R3 — provenance + stage runner (comments count: provenance IS prose).
    # Support files (registry / *common) and *_worker.jl subroutines are driven
    # by their parent experiments.
    if !support
        occursin(r"provenance"i, src_full) || push!(v, "R3: no provenance block")
    end
    if !support && !worker
        occursin("run_stages(", src) || push!(v, "R3: no run_stages( entrypoint")
    end
    return v
end

# The configuration-of-record layer (run keys, data/results roots, provenance records).
include("experiments/run_records.jl")
include("experiments/data_policy.jl")

# Re-keying: the old-key → new-key map of a tag rename (computed before it lands).
include("experiments/rekey.jl")

end # module Experiments

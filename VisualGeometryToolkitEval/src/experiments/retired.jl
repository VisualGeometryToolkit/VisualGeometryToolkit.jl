# =============================================================================
# retired.jl — the experiment caching that the keyed step (steps.jl) replaced
# (2026-11-14, the owner's rule: every step through DrWatson's produce_or_load,
# keyed by only what it reads). Each name is refused BY NAME with its
# replacement — never an alias, never a forwarding wrapper.
# =============================================================================

_retired(old::AbstractString, new::AbstractString) =
    throw(ArgumentError("Experiments.$old is retired (VisualGeometryToolkitEval 0.9): $new"))

"Retired: a stage's done-predicate is its steps' own record — `Stage(name, desc; steps, run)`, `produced(step)`."
fresher(args...; kwargs...) = _retired("fresher", "declare the stage's keyed steps, Stage(name, desc; steps, run); a step is produced(step)")

"Retired: `produced(step)` (a `Step`'s output, its provenance and its code trees)."
run_complete(args...; kwargs...) = _retired("run_complete", "a Step's reuse is produced(step); produce(f, step) loads or produces it")

"Retired: `produce(f, step)` — DrWatson's produce_or_load on a keyed `Step`, its record saved through VisualGeometryToolkitIO."
cached_result(args...; kwargs...) = _retired("cached_result", "produce(f, Step(experiment, name, record; output)) — DrWatson's produce_or_load, its record saved and loaded through VisualGeometryToolkitIO (load_output)")

"Retired: every key is `run_key` (SHA-256 of the canonical JSON), a `Step`'s."
legacy_hash_key(args...; kwargs...) = _retired("legacy_hash_key", "run_key(record), a Step's key")

"Retired: `step_dir(Step(experiment, name, record; output))` — DrWatson's datadir(\"exp_pro\", experiment, name, key)."
keyed_cell_dir(args...; kwargs...) = _retired("keyed_cell_dir", "step_dir(Step(experiment, name, record; output, files, upstream, parts)), DrWatson's datadir(\"exp_pro\", experiment, name, key)")

"Retired: a producer's step is a `Step` under the results root, made an input of record by `promote_run(step_dir(s), tier)`."
keyed_data_dir(args...; kwargs...) = _retired("keyed_data_dir", "produce the Step (under the results root), then promote_run(step_dir(step), tier)")

"Retired: a project's deeper layout is a `Step`'s experiment path, `Step(joinpath(dataset, matcher, problem, experiment), name, record; output)`."
experiment_dir(args...; kwargs...) = _retired("experiment_dir", "a Step's experiment may be a relative path, Step(joinpath(<layout>…, experiment), name, record; output)")

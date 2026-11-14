# =============================================================================
# retired.jl — the experiment caching that the keyed product (products.jl) replaced
# (2026-11-14, the owner's rule: every product through DrWatson's produce_or_load,
# keyed by only what it reads). Each name is refused BY NAME with its
# replacement — never an alias, never a forwarding wrapper.
# =============================================================================

_retired(old::AbstractString, new::AbstractString) =
    throw(ArgumentError("Experiments.$old is retired (VisualGeometryToolkitEval 0.9): $new"))

"Retired: a stage's done-predicate is its products' own record — `Stage(name, desc; items, product, run)`, `produced(product)`."
fresher(args...; kwargs...) = _retired("fresher", "declare the stage's keyed products, Stage(name, desc; items, product, run); a product is produced(product)")

"Retired: `produced(product)` (a `Product`'s output, its provenance and its code trees)."
run_complete(args...; kwargs...) = _retired("run_complete", "a Product's reuse is produced(product); produce(f, product) loads or produces it")

"Retired: `produce(f, product)` — DrWatson's produce_or_load on a keyed `Product`, its record saved through VisualGeometryToolkitIO."
cached_result(args...; kwargs...) = _retired("cached_result", "produce(f, Product(experiment, name, record; output)) — DrWatson's produce_or_load, its record saved and loaded through VisualGeometryToolkitIO (load_output)")

"Retired: every key is `run_key` (SHA-256 of the canonical JSON), a `Product`'s."
legacy_hash_key(args...; kwargs...) = _retired("legacy_hash_key", "run_key(record), a Product's key")

"Retired: `product_dir(Product(experiment, name, record; output))` — DrWatson's datadir(\"exp_pro\", experiment, name, key)."
keyed_cell_dir(args...; kwargs...) = _retired("keyed_cell_dir", "product_dir(Product(experiment, name, record; output, files, upstream, parts)), DrWatson's datadir(\"exp_pro\", experiment, name, key)")

"Retired: a producer's product is a `Product` under the results root, made an input of record by `promote_run(product_dir(s), tier)`."
keyed_data_dir(args...; kwargs...) = _retired("keyed_data_dir", "produce the Product (under the results root), then promote_run(product_dir(product), tier)")

"Retired: a project's deeper layout is a `Product`'s experiment path, `Product(joinpath(dataset, matcher, problem, experiment), name, record; output)`."
experiment_dir(args...; kwargs...) = _retired("experiment_dir", "a Product's experiment may be a relative path, Product(joinpath(<layout>…, experiment), name, record; output)")

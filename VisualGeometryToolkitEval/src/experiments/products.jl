# =============================================================================
# products.jl — the ONE cache of a codified experiment: a keyed product, produced or
# loaded by DrWatson's `produce_or_load`.
#
# A product of an experiment (decode, detect, index, a start state, a refinement,
# an evaluation, a figure) is keyed by ONLY what it reads — its own config
# section, its inputs, the keys of the products it reads (`upstream`) — through
# `run_key`, so a later product's setting never recomputes an earlier product. Its
# directory is DrWatson's processed-data layout:
#
#   datadir("exp_pro", <experiment>, <product>, <key>)      (results_root() is datadir("exp_pro"))
#
# holding the product's output, its `_config.toml` (the record it is keyed by) and
# its `_provenance.toml` (DrWatson's tagged dict + the code trees), written
# LAST: the completion mark. `produce` calls `produce_or_load` with
# `loadfile = false`; a product that returns a record has it saved by
# produce_or_load through `DrWatson._wsave(file, ::Produced)`, which is
# `VisualGeometryToolkitIO.save`, and read back with `load_output`
# (`VisualGeometryToolkitIO.load`). A product that writes its own outputs (a child
# process, a directory of outputs) returns `nothing`: its unit is the whole
# directory and its `output` the file it writes last.
#
# The key is the dependency tracking (`Product`: the record, the upstream keys,
# the pinned inputs, the code trees); produce_or_load produces or loads the
# keyed output. Justified extension, applied around the call: a save that
# fails is an error (produce_or_load warns and returns, saving_files.jl:107-116).
# =============================================================================

"""
    Product(experiment, name, record; output, files = nothing, upstream = nothing, parts = nothing,
            code = current_code_trees(project_root()))

The keyed product of one step of a pipeline of `experiment` (a name, or a relative path of segments for a project whose results are laid out deeper:
`"ETH/sift/homography/fig4"`): its `name` (`"prepare"`, `"start"`, `"refine"`, …), the `record` it is keyed by (ONLY the
config fields and inputs the product reads), its `key` and its `output`, the file under the product's directory
([`product_dir`](@ref)) that holds its result — the record [`produce`](@ref) saves, or the file a product that writes its own
outputs writes last.

The key is THE dependency tracking of a pipeline (`run_key(record; files, upstream, parts, code)`): the record; the KEYS of the
products it reads (`upstream`: a `Product`, whose key is folded in, or a key — a Merkle chain, never a re-hash of their contents);
the content of the files it reads (`files`; a config's pinned inputs belong in its record, by their `sha256`); a choice made on
a command line (`parts`); and the CODE it runs (`code`): the source-tree hash of every stack package its environment loads and
the project's own `src` ([`current_code_trees`](@ref): a path checkout's git trees of `src`, `ext`, `Project.toml`, a registered
version's Manifest tree hash). A change of any of them is another key, so another directory, produced anew; an unchanged one is
the same product, loaded. A change of a package re-keys every product that loads it (conservative). A result of an earlier key
stays where it is.
"""
struct Product
    experiment::String
    name::String
    record::Dict{String, Any}
    key::String
    output::String
end

function Product(experiment::AbstractString, name::AbstractString, record::AbstractDict; output::AbstractString,
                 files = nothing, upstream = nothing, parts = nothing, code = current_code_trees(project_root()))
    _segments_ok(experiment) || throw(ArgumentError("Product: experiment $(repr(experiment)) must be a relative path of segments (no `.`, `..`)"))
    (isempty(name) || occursin('/', name) || name in (".", "..")) && throw(ArgumentError("Product: name $(repr(name)) must be one path segment"))
    (isempty(output) || isabspath(output) || any(==(".."), splitpath(output))) &&
        throw(ArgumentError("Product: output $(repr(output)) must be a relative path inside the product's directory"))
    up = upstream === nothing ? nothing :
         Dict{String, String}(String(k) => (v isa Product ? v.key : String(v)) for (k, v) in (upstream isa Union{AbstractDict, NamedTuple} ? pairs(upstream) : upstream))
    rec = Dict{String, Any}(String(k) => v for (k, v) in record)
    return Product(String(experiment), String(name), rec, run_key(rec; files, upstream = up, parts, code), String(output))
end

_segments_ok(p::AbstractString) = !isempty(p) && !isabspath(p) && !occursin('\\', p) &&
    all(seg -> !isempty(seg) && !(seg in (".", "..")), split(p, '/'))

Base.show(io::IO, s::Product) = print(io, "Product(", s.experiment, "/", s.name, "/", s.key, ": ", s.output, ")")

"""
    product_dir(s::Product) -> String

The product's directory: `<results_root()>/<experiment>/<name>/<key>`, DrWatson's `datadir("exp_pro", experiment, name, key)`
(`results_root()` is the project's `data/exp_pro`, or its binding `VGTK_RESULTS_ROOT`).
"""
product_dir(s::Product) = joinpath(results_root(), s.experiment, s.name, s.key)

"The file of the product's result: `<product_dir(s)>/<output>`."
product_output(s::Product) = joinpath(product_dir(s), s.output)

"""
    produced(s::Product) -> Bool

Whether the product is produced and reused: its output and its `_provenance.toml` with its key are there (the key holds the code
it runs), its scratch flag is this run's ([`scratch_run`](@ref)) and no code tree it recorded was uncommitted (a scratch run's
`+uncommitted` mark names HEAD, not the edits: never reused).
"""
function produced(s::Product)
    dir = product_dir(s)
    isfile(product_output(s)) && isfile(joinpath(dir, PROVENANCE_FILE)) || return false
    prov = try read_provenance(dir) catch; return false end
    get(prov, "run_key", "") == s.key || return false
    get(prov, "scratch", false) === scratch_run() || return false
    trees = get(prov, "code_trees", Dict{String, Any}())
    uncommitted = [k for (k, t) in trees if _is_uncommitted_tree(string(t))]
    isempty(uncommitted) && return true
    @info "producing $(s.experiment)/$(s.name)/$(s.key) again: it was computed from uncommitted code" trees = join(sort!(uncommitted), ", ")
    return false
end

"""
    Produced(record = nothing; inputs = String[], kwargs...)

What a product's `f` returns to [`produce`](@ref) when it says more than its record: the `record` (saved by produce_or_load through
`DrWatson._wsave(file, ::Produced)`, which is `VisualGeometryToolkitIO.save(file, record; kwargs...)` — a camera's `name`, a
`length_unit`, a `provenance`), or `nothing` when `f` wrote the product's output itself; and the `inputs` it read (paths, recorded
in its provenance beside `produce`'s own `inputs`). A bare record returned is `Produced(record)`.
"""
struct Produced{T, K}
    record::T
    inputs::Vector{String}
    kwargs::K
end
Produced(record = nothing; inputs = String[], kwargs...) = Produced(record, String[String(x) for x in inputs], values(kwargs))
Base.copy(x::Produced) = x                                            # produce_or_load saves `copy(data)`
DrWatson._wsave(file, x::Produced) =
    x.record === nothing ? (isfile(file) || error("the product wrote no $file")) : VgtkIO.save(file, x.record; x.kwargs...)

"""
    produce(f, s::Product; driver, config_source, inputs = String[], producer = false, fields = Dict(),
            force = false, gitpath = project_root()) -> String

The product `s` produced or loaded — DrWatson's `produce_or_load` on the product's output (`loadfile = false`). When it is
[`produced`](@ref) (and not `force`) nothing runs; otherwise its old provenance is removed (a failed product never reads as
produced), `f(dir)` runs in the product's directory and returns the product's record — saved by produce_or_load through
`VisualGeometryToolkitIO.save` — or `nothing` when `f` wrote `s.output` itself, or a [`Produced`](@ref) (a record or `nothing`,
with the inputs `f` read and the save's keywords); the product's records are written last ([`write_run_records`](@ref):
`_config.toml`, the record it is keyed by; `_provenance.toml` with `run_key`, `driver`, `config_source`, the declared `inputs` and
`f`'s, `fields`; `producer` for a product that writes under the data root). A save that fails is an error. Returns the product's directory; read its record with [`load_output`](@ref).
"""
function produce(f, s::Product; driver::AbstractString, config_source::AbstractString, inputs::AbstractVector = String[],
                 producer::Bool = false, fields::AbstractDict = Dict{String, Any}(), force::Bool = false,
                 gitpath::AbstractString = project_root())
    dir = product_dir(s)
    reuse = !force && produced(s)
    if !reuse
        assert_not_input(dir; producer)
        _recorded_code_trees(gitpath, dir)                            # a run of record of uncommitted code is refused before it computes
        rm(joinpath(dir, PROVENANCE_FILE); force = true)
        rm(product_output(s); force = true)                          # a save that fails never leaves an earlier output standing
    end
    read = String[]                                                   # the inputs f says it read
    DrWatson.produce_or_load(s.record, dirname(dir); filename = _ -> joinpath(s.key, s.output), prefix = "", suffix = "",
                             tag = false, loadfile = false, force = !reuse, verbose = false) do _
        mkpath(dir)
        out = f(dir)
        out = out isa Produced ? out : Produced(out)
        append!(read, out.inputs)
        out
    end
    reuse && return dir
    isfile(product_output(s)) || error("$(s.experiment)/$(s.name): $(product_output(s)) was not saved (the error is printed above)")
    write_run_records(dir, s.record; driver, config_source, key = s.key, inputs = String[String.(inputs); read], producer, fields,
                      gitpath)
    return dir
end

"""
    load_output(T, s::Product; kwargs...)

The product's record: `VisualGeometryToolkitIO.load(T, <product_dir(s)>/<output>; kwargs...)`.
"""
load_output(::Type{T}, s::Product; kwargs...) where {T} = VgtkIO.load(T, product_output(s); kwargs...)

# The stage runner's view of a product (stages.jl): produced, and unmarked by --force.
_product_produced(s::Product) = produced(s)
_product_unmark!(s::Product) = (rm(joinpath(product_dir(s), PROVENANCE_FILE); force = true); nothing)

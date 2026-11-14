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
# produce_or_load through `DrWatson._wsave(file, ::Saved)`, which is
# `VisualGeometryToolkitIO.save`, and read back with `load_output`
# (`VisualGeometryToolkitIO.load`). A product that writes its own outputs (a child
# process, a directory of outputs) returns `nothing`: its unit is the whole
# directory and its `output` the file it writes last.
#
# Justified extensions of produce_or_load, both applied around the call: a
# stored output is reused only while the code trees its provenance records are
# the current ones (`produced`: the one reuse rule), and a save that fails is
# an error (produce_or_load warns and returns, saving_files.jl:107-116).
# =============================================================================

"""
    Product(experiment, name, record; output, files = nothing, upstream = nothing, parts = nothing)

The keyed product of one step of a pipeline of `experiment` (a name, or a relative path of segments for a project whose results are laid out deeper:
`"ETH/sift/homography/fig4"`): its `name` (`"prepare"`, `"start"`, `"refine"`, …), the `record` it is keyed by (ONLY the
config fields and inputs the product reads), its `key` (`run_key(record; files, upstream, parts)`; an `upstream` value is a
`Product`, whose key is folded in, or a key) and its `output`, the file under the product's directory ([`product_dir`](@ref)) that
holds its result — the record [`produce`](@ref) saves, or the file a product that writes its own outputs writes last. A changed
record (or input file, upstream key or part) is another key, so another directory; an unchanged one is the same product.
"""
struct Product
    experiment::String
    name::String
    record::Dict{String, Any}
    key::String
    output::String
end

function Product(experiment::AbstractString, name::AbstractString, record::AbstractDict; output::AbstractString,
              files = nothing, upstream = nothing, parts = nothing)
    _segments_ok(experiment) || throw(ArgumentError("Product: experiment $(repr(experiment)) must be a relative path of segments (no `.`, `..`)"))
    (isempty(name) || occursin('/', name) || name in (".", "..")) && throw(ArgumentError("Product: name $(repr(name)) must be one path segment"))
    (isempty(output) || isabspath(output) || any(==(".."), splitpath(output))) &&
        throw(ArgumentError("Product: output $(repr(output)) must be a relative path inside the product's directory"))
    up = upstream === nothing ? nothing :
         Dict{String, String}(String(k) => (v isa Product ? v.key : String(v)) for (k, v) in (upstream isa Union{AbstractDict, NamedTuple} ? pairs(upstream) : upstream))
    rec = Dict{String, Any}(String(k) => v for (k, v) in record)
    return Product(String(experiment), String(name), rec, run_key(rec; files, upstream = up, parts), String(output))
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
    produced(s::Product; gitpath = project_root()) -> Bool

Whether the product is produced and reused: its output and its `_provenance.toml` are there, the code trees that record keeps are
the current ones and its scratch flag is this run's ([`scratch_run`](@ref)). A changed tree is not produced: the product runs
again, and an `@info` names each tree that changed. A record from before the trees were kept has none and runs once more.
"""
function produced(s::Product; gitpath::AbstractString = project_root())
    dir = product_dir(s)
    isfile(product_output(s)) && isfile(joinpath(dir, PROVENANCE_FILE)) || return false
    prov = try read_provenance(dir) catch; return false end
    get(prov, "run_key", "") == s.key || return false
    changes = _reuse_changes(prov, current_code_trees(gitpath), scratch_run())
    isempty(changes) && return true
    @info "producing $(s.experiment)/$(s.name)/$(s.key) again: its code changed" changes = join(changes, "; ")
    return false
end

"""
    Saved(record; kwargs...)

A product's result as `produce_or_load` saves it: `DrWatson._wsave(file, ::Saved)` is `VisualGeometryToolkitIO.save(file, record;
kwargs...)` (a camera's `name`, a `length_unit`, a `provenance`). [`produce`](@ref) wraps what a product returns in it.
"""
struct Saved{T, K}
    record::T
    kwargs::K
end
Saved(record; kwargs...) = Saved(record, values(kwargs))
Base.copy(x::Saved) = x                                               # produce_or_load saves `copy(data)`
DrWatson._wsave(file, x::Saved) = VgtkIO.save(file, x.record; x.kwargs...)

# A product that wrote its own output: nothing to save, the file must be there.
struct _Written end
Base.copy(x::_Written) = x
DrWatson._wsave(file, ::_Written) = isfile(file) || error("the product wrote no $file")

"""
    produce(f, s::Product; driver, config_source, inputs = String[], producer = false, fields = Dict(),
            force = false, gitpath = project_root()) -> String

The product `s` produced or loaded — DrWatson's `produce_or_load` on the product's output (`loadfile = false`). When it is
[`produced`](@ref) (and not `force`) nothing runs; otherwise its old provenance is removed (a failed product never reads as
produced), `f(dir)` runs in the product's directory and returns the product's record — saved by produce_or_load through
`VisualGeometryToolkitIO.save` (a [`Saved`](@ref) keeps its keywords) — or `nothing` when `f` wrote `s.output` itself, and the
product's records are written last ([`write_run_records`](@ref): `_config.toml`, the record it is keyed by; `_provenance.toml`
with `run_key`, `driver`, `config_source`, the declared `inputs` — paths, or a function returning them, called once the product
ran (a produced product checks none) — `fields`; `producer` for a product that writes under the data root). A save that fails is an error. Returns the product's directory; read its record with [`load_output`](@ref).
"""
function produce(f, s::Product; driver::AbstractString, config_source::AbstractString, inputs = String[],
                 producer::Bool = false, fields::AbstractDict = Dict{String, Any}(), force::Bool = false,
                 gitpath::AbstractString = project_root())
    dir = product_dir(s)
    reuse = !force && produced(s; gitpath)
    if !reuse
        assert_not_input(dir; producer)
        _recorded_code_trees(gitpath, dir)                            # a run of record of uncommitted code is refused before it computes
        rm(joinpath(dir, PROVENANCE_FILE); force = true)
    end
    DrWatson.produce_or_load(s.record, dirname(dir); filename = _ -> joinpath(s.key, s.output), prefix = "", suffix = "",
                             tag = false, loadfile = false, force = !reuse, verbose = false) do _
        mkpath(dir)
        out = f(dir)
        out === nothing ? _Written() : out isa Saved ? out : Saved(out)
    end
    reuse && return dir
    isfile(product_output(s)) || error("$(s.experiment)/$(s.name): $(product_output(s)) was not saved (the error is printed above)")
    ins = inputs isa Function ? inputs() : inputs                     # a function: the inputs the product read, listed once it ran
    write_run_records(dir, s.record; driver, config_source, key = s.key, inputs = String[String(x) for x in ins], producer, fields)
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

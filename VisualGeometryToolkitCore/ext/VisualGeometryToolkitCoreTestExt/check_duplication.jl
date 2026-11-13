# =============================================================================
# check_duplication — one implementation per concept, checked mechanically
# =============================================================================
#
# Name ownership (a loaded module's bindings against its upstream stack
# packages'), token-shingle similarity of function bodies, whole-file identity
# and a retired-pattern table (the `check_raw_decoding` mechanism). The entry
# point and its result type are in src/quality/duplication.jl. Like
# `check_naming`, the retired patterns are BUILT from fragments, so this file
# spells none of the code it flags.
# =============================================================================

const _DUPLICATION_FILE = "DUPLICATION.toml"
const _STACK_PATHS_ENV = "VGTK_STACK_PATHS"
const _DEFAULT_SIMILARITY = ((0.6, 60), (0.85, 25))
# A shingle in more definitions than this is an idiom, not evidence. The definitions counted are this repository's
# only (a label without a `<Repository>:` prefix), never the stack repositories' (`VGTK_STACK_PATHS`): the idioms are
# then the same whether or not a run reads the stack, so a pair inside the repository is found, or not, alike in CI
# (no stack) and in a release gate (the stack). Counted over the whole corpus, the stack's extra definitions pushed a
# test fixture's shingles over the cap in the gate only (wave 113: two fixture pairs CI found and the gate did not).
# A pair ACROSS repositories (one side a stack repository's, which only the gate reads) is an idiom when its shingle is one
# in ANY repository of the run: a language idiom one repository writes forty times (Core's one-field wrapper constructor
# `S{T}(x::X) = S{T, X}(x)`) is an idiom for a downstream package's single use of it too (PanoramaStitching 0.7), with no
# per-repository exception file. An own pair still reads the own count, so CI and the gate find the same own pairs.
const _MAX_POSTINGS = 40

# --- the retired patterns ----------------------------------------------------

# Each `(use, pattern, owner, home)`: `home` is `(package, files)`, the files of
# that package (relative paths, or a Regex over them) where the pattern is the owner's own.
const _RETIRED_PATTERNS = (
    # THE LM GATE: a direct call of an LM primitive outside the one adapter. The primitives are NLLSsolver's solver API
    # (its optimize, problem, cache and options constructors) and Core's internals of the adapter (the retired knob map
    # and its private successor, the termination decode and stop-code decoder, the engine) and of the device LM (its
    # increment solve). The entry points are not primitives: `lmsolve_chart!`, `LmProblem` (with `LmStops`) and
    # `static_lm` (with `StaticLmStops`) may be called anywhere. Every own file is scanned — src/, ext/, test/,
    # benchmark/, scripts — the home alone excepted: lm_core.jl and the adapter's own tests.
    (:lm_primitive, Regex("\\bNLLS" * "Options\\s*\\(|\\bNLLSsolver\\.(optimize!|NLLS" * "Problem|Optimize" * "Cache)\\s*\\(|" *
            "(?<![\\p{L}\\p{N}_!])(_?nlls_" * "options|nlls_" * "termination|nlls_" * "stop_code|_lm_" * "run!|_static_lm_" * "increment)\\s*\\("),
        "VisualGeometryToolkitCore's one LM adapter (src/robust/lm_core.jl): `lmsolve_chart!` on a chart state or an `LmProblem`, under `LmStops`; `static_lm` under `StaticLmStops` on a device",
        ("VisualGeometryToolkitCore", ("src/robust/lm_core.jl", "test/lm_driver_tests.jl", "test/static_lm_tests.jl"))),
    (:rotation_angle, Regex("\\ba" * "cos\\s*\\(\\s*clamp\\s*\\(\\s*\\(\\s*(LinearAlgebra\\.)?(tr|sum)\\s*\\("),
        "Rotations' `rotation_angle` (as VisualGeometryToolkitCore's stereo.jl): exact near 0 and π", ("", ())),
    (:srgb_transfer, Regex("\\b0\\.00" * "31308\\b|\\b0\\.0" * "4045\\b"),
        "VisualGeometryToolkitFeatures' sRGB transfer (src/linear_image.jl, `srgb_to_linear`)",
        ("VisualGeometryToolkitFeatures", ("src/linear_image.jl", "test/linear_image_tests.jl"))),
    (:tex_macro_writer, Regex("\\b(print|println|write)\\s*\\(.*\\\\{1,2}" * "new" * "command"),
        "VisualGeometryToolkitEval's number macros (src/reproduce/number_macros.jl: `PaperNumbers`, `add_number!`, `write_numbers`)",
        ("VisualGeometryToolkitEval", ("src/reproduce/number_macros.jl", r"^test/"))),
    # the sensor model (owner rulings D1–D13, 2026-11-13): a consumer never constructs a component, a characterisation or
    # a model (the absent `SensorModel()` aside); a test describes one through the one fixture helper (D13)
    (:sensor_component,
        Regex("\\b(?:File" * "Black|Channel" * "Black|Da" * "rk|No" * "Noise|Read" * "Noise|Poisson" * "Gaussian|Relative" * "Noise|No" *
              "Defects|Defect" * "List|SensorCharac" * "terisation)(?:\\{[^}]*\\})?\\s*\\(|\\bSensor" * "Model(?:\\{[^}]*\\})?\\s*\\((?!\\s*\\))"),
        "VisualGeometryToolkitCore's sensor model (src/raw/sensor_model.jl): `sensor_model(characterisation, capture, t)`, " *
            "`load(SensorCharacterisation, path)` (VisualGeometryToolkitIO); in tests `sensor_fixture` (D13)",
        ("VisualGeometryToolkitCore", ("src/raw/sensor_model.jl", "src/raw/sensor_estimators.jl",
                                       "ext/VisualGeometryToolkitCoreTestExt/sensor_fixture.jl"))),
    # one variance law of a pixel value: LinearRaw's kernel under a PoissonGaussian (or RelativeNoise), its uncertain image
    # (D11); `α·x + σ²` or `x/K + σ²` written by hand restates it (the Anscombe transform's `α x + 3α²/8 + σ²` does not match)
    (:value_variance_law,
        Regex("(?:\\b(?:α|al" * "pha)\\w*\\s*\\.?\\*\\s*[\\w.\\[\\]]+|[\\w.\\[\\]]+\\s*\\.?/\\s*(?:K|gain|electrons_per" * "_dn)\\b)" *
              "\\s*\\.?\\+\\s*(?:σ|sig" * "ma)\\w*(?:²|\\s*\\.?\\^\\s*2)"),
        "VisualGeometryToolkitCore's LinearRaw under a PoissonGaussian: each value's variance in its uncertain image (`L.image`'s `attributes`)",
        ("VisualGeometryToolkitCore", ("src/raw/sensor_model.jl", "src/raw/linear_raw.jl", "test/sensor_model_tests.jl"))),
    # one black subtraction: LinearRaw / linearview under the frame's SensorModel (its black slot); a hand-indexed
    # per-channel black (`mosaic.black[c + 1]`) restates it
    (:black_subtraction, Regex("\\.bla" * "ck\\s*\\["),
        "VisualGeometryToolkitCore's one linearisation: `LinearRaw(mosaic; mode, sensor_model)`, `linearview(mosaic; sensor_model)`",
        ("VisualGeometryToolkitCore", (r"^src/raw/", "test/raw_mosaic_tests.jl"))),
)

_in_home(f, home) = any(h -> h isa Regex ? occursin(h, f) : f == h, home)

# --- the repository and the stack around a directory ----------------------------

# The git checkout holding `dir` (its `.git` is a directory, or a file in a worktree), else `dir`.
function _repo_root(dir::AbstractString)
    d = abspath(dir)
    while true
        ispath(joinpath(d, ".git")) && return String(rstrip(d, '/'))
        p = dirname(d)
        p == d && return String(rstrip(abspath(dir), '/'))
        d = p
    end
end

# A repository's label in a finding: its root Project.toml name, else its directory's name.
function _repo_label(root::AbstractString)
    name = _package_name(root)
    return isempty(name) ? basename(rstrip(root, '/')) : name
end

# Another repository's path in a finding or an exception: `<Repository>:<path>` (a `::` is a function's).
const _EXTERNAL_ID = r"^([^/:]+:)(?!:)"

# The trees of a repository, each `(directory, label prefix)`: a workspace root that is
# not itself a package (a monorepo) is its member projects, each scanned in its own
# scope as its own quality item scans it; any other repository is its root. Another
# repository's trees are labelled `<Repository>:`, a monorepo's members by their own
# package names (`<Member>:`), so a label never depends on where it is checked out;
# this repository's members are paths from its root (`<member>/`).
function _repo_trees(root::AbstractString; external::Bool)
    p = joinpath(root, "Project.toml")
    project = isfile(p) ? TOML.parsefile(p) : Dict{String, Any}()
    members = get(get(project, "workspace", Dict{String, Any}()), "projects", String[])
    (haskey(project, "name") || isempty(members)) && return [(String(root), external ? _repo_label(root) * ":" : "")]
    trees = [(joinpath(root, m), m) for m in members if isdir(joinpath(root, m))]
    return [(d, external ? _repo_label(d) * ":" : m * "/") for (d, m) in trees]
end

_stack_from_env() = String[p for p in split(get(ENV, _STACK_PATHS_ENV, ""), Sys.iswindows() ? ';' : ':') if !isempty(p)]

# --- tokens and definitions --------------------------------------------------------

# Every function and macro definition (outermost: a closure is its enclosing definition's).
function _definitions!(out, node)
    k = _JS.kind(node)
    (k == _JS.K"function" || k == _JS.K"macro") && return push!(out, node)
    cs = _JS.children(node)
    cs === nothing || foreach(c -> _definitions!(out, c), cs)
    return out
end

# A definition's name: its callee's source text (`f`, `Base.show`, `(::T)`).
function _definition_name(node)
    cs = _JS.children(node)
    (cs === nothing || isempty(cs)) && return "?"
    sig = cs[1]
    while _JS.kind(sig) in (_JS.K"where", _JS.K"::") && !isempty(something(_JS.children(sig), ()))
        sig = _JS.children(sig)[1]
    end
    target = _JS.kind(sig) == _JS.K"call" ? _JS.children(sig)[1] : sig
    return replace(String(_JS.sourcetext(target)), r"\s+" => "")
end

# One definition: where it is, its token count and its shingle set (sorted hashes).
struct _Definition
    file::String       # the finding's path: repository-relative, `<Repository>:<path>` elsewhere
    own::Bool          # in the package being checked
    line::Int
    name::String
    ntokens::Int
    shingles::Vector{UInt64}
end

function _shingles(ids::AbstractVector{Int}, k::Int)
    length(ids) < k && return UInt64[]
    out = Vector{UInt64}(undef, length(ids) - k + 1)
    for i in eachindex(out)
        h = UInt64(k)
        for j in i:(i + k - 1)
            h = hash(ids[j], h)
        end
        out[i] = h
    end
    return unique!(sort!(out))
end

# The definitions and the whole-file token hash of one Julia file.
function _scan_julia!(defs, files, path, label, own, ids, k)
    src = read(path, String)
    toks = _code_tokens(src)
    push!(files, (label, own, hash(toks.raw), length(toks.raw)))
    tree = try
        _JS.parseall(_JS.SyntaxNode, src; filename = path, ignore_errors = true)
    catch
        return
    end
    for node in _definitions!(Any[], tree)
        lo = searchsortedfirst(toks.starts, _JS.first_byte(node))
        hi = searchsortedlast(toks.starts, _JS.last_byte(node))
        hi < lo && continue
        norm = [get!(ids, toks.normalized[i], length(ids) + 1) for i in lo:hi]
        push!(defs, _Definition(label, own, _JS.source_line(node), _definition_name(node), length(norm), _shingles(norm, k)))
    end
    return
end

# --- exceptions ---------------------------------------------------------------------

const _EXCEPTION_KEYS = Dict(
    "name" => ("module", "name", "reason"), "similar" => ("functions", "reason"),
    "file" => ("files", "reason"), "pattern" => ("use", "file", "reason"),
    "backend" => ("file", "function", "reason"), "twin" => ("names", "reason"),
    "splat" => ("file", "function", "reason"),   # check_splat's (check_splat.jl)
)

# The entries of the root's DUPLICATION.toml, by kind (each a Dict); throws on a defect.
function _duplication_exceptions(root::AbstractString)
    out = Dict(kind => Dict{String, Any}[] for kind in keys(_EXCEPTION_KEYS))
    file = joinpath(root, _DUPLICATION_FILE)
    isfile(file) || return out
    errors = String[]
    decl = TOML.parsefile(file)
    for (kind, entries) in decl
        haskey(_EXCEPTION_KEYS, kind) && entries isa AbstractVector ||
            (push!(errors, "`$kind` is not a [[name]], [[similar]], [[file]], [[pattern]], [[backend]], [[twin]] or [[splat]] entry"); continue)
        for e in entries
            want = _EXCEPTION_KEYS[kind]
            lacking = filter(k -> !haskey(e, k) || isempty(e[k]), collect(want))
            unknown = setdiff(keys(e), want)
            isempty(lacking) || push!(errors, "[[$kind]] $(repr(e)) lacks $(join(lacking, ", "))")
            isempty(unknown) || push!(errors, "[[$kind]] $(repr(e)) has unknown keys $(join(sort!(collect(unknown)), ", "))")
            kind in ("similar", "file", "twin") && haskey(e, want[1]) && !(e[want[1]] isa AbstractVector && length(e[want[1]]) == 2) &&
                push!(errors, "[[$kind]] $(repr(e)): `$(want[1])` is a list of two")
            isempty(lacking) && isempty(unknown) && push!(out[kind], e)
        end
    end
    isempty(errors) || throw(ArgumentError("$_DUPLICATION_FILE:\n  " * join(errors, "\n  ")))
    return out
end

_entry_text(kind, e) = "[[$kind]] " * join((string(k, " = ", repr(e[k])) for k in _EXCEPTION_KEYS[kind] if k != "reason"), ", ")

# --- backend branches and CPU/GPU twins ---------------------------------------------------
#
# The owner's policy (2026-11-11): a GPU branch's CPU branch is the SAME
# KernelAbstractions kernel on the CPU backend. A definition that tests the
# backend, `get_backend`'s answer or a GPU array type to run other code, a
# dispatch pair on a CPU and a GPU type, and two names that differ only by a
# CPU/GPU marker are each a duplicate. The transfer and allocation helpers
# (an upload or download, a zero array, an allocation, a synchronization, a
# device query: its names, free bytes, pool reclaim, default device and a
# kernel's resident threads) are the backends' own and are left alone, by their WHOLE names:
# every `_`-separated token of the name (its leading `_` and trailing `!`
# dropped) is a word of `_HELPER_TOKENS`. A fragment is not enough —
# `fill_sqrt_weights!`, `transfer_ellipse`, `_copy_sq_distances!` and
# `cones` compute, so their backend tests are checked.
# Like the retired patterns, the tests are built from fragments.

const _KA = "(KA\\.|KernelAbstractions\\.)?"
const _GPU_ARRAYS = "(CUDA\\.)?(Cu" * "Array|CuDevice" * "Array|ROC" * "Array|Mtl" * "Array|one" * "Array|AbstractGPU" * "Array|CUDA" * "Backend)"
const _BACKEND_TEST = Regex("\\bisa\\s+" * _KA * "(CPU|GPU)\\b|get_backend\\s*\\([^)]*\\)\\s*(isa|===?|!==?)\\s|" *
                            "\\bisa\\s+" * _GPU_ARRAYS * "\\b|(===?|!==?)\\s*" * _KA * "CPU\\(\\)")
const _CPU_SIGNATURE = Regex("::\\s*" * _KA * "CPU\\b")
const _GPU_SIGNATURE = Regex("::\\s*" * _KA * "(GPU|Backend)\\b|::\\s*" * _GPU_ARRAYS * "\\b")
const _HELPER_TOKENS = Set(["upload", "download", "to", "on", "host", "device", "devices", "zeros", "ones", "fill", "allocate",
                            "synchronize", "synchronise", "adapt", "structure", "storage", "copy", "copyto", "transfer",
                            "names", "free", "bytes", "reclaim", "default", "buffer", "buffers", "ka", "memory",
                            "resident", "threads"])   # a kernel's resident threads: a device query, no shared kernel computes it
_is_helper_name(name) = all(in(_HELPER_TOKENS), split(rstrip(lstrip(_bare_name(name), '_'), '!'), '_'; keepempty = false))
# a CPU/GPU marker: the suffix pairs and the prefix pair, and `_cuda` beside the plain name
const _TWIN_SUFFIXES = (("_cpu", "_gpu"), ("_host", "_device"))
const _TWIN_PREFIXES = (("cpu_", "gpu_"),)

# Every function definition of a parsed file, long form and short form, with its name and its signature's text.
function _all_definitions!(out, node)
    k = _JS.kind(node)
    cs = _JS.children(node)
    if k == _JS.K"function" || (k == _JS.K"=" && cs !== nothing && !isempty(cs) && _is_syntax_signature(cs[1]))
        push!(out, node)
    end
    cs === nothing || foreach(c -> _all_definitions!(out, c), cs)
    return out
end
function _is_syntax_signature(sig)   # a JuliaSyntax node (check_naming's `_is_signature` reads an Expr)
    while _JS.kind(sig) in (_JS.K"where", _JS.K"::") && !isempty(something(_JS.children(sig), ()))
        sig = _JS.children(sig)[1]
    end
    return _JS.kind(sig) == _JS.K"call"
end
_signature_text(node) = String(_JS.sourcetext(_JS.children(node)[1]))
_bare_name(name) = replace(last(split(name, '.')), "!" => "")

function _backend_violations(files, entries)
    out = DuplicationViolation[]
    backend_used = falses(length(entries["backend"])); twin_used = falses(length(entries["twin"]))
    allowed_backend(file, name) = (i = findfirst(e -> e["file"] == file && e["function"] == name, entries["backend"]);
                                   i === nothing ? false : (backend_used[i] = true; true))
    names = Dict{String, Tuple{String, Int}}()                  # bare name => where first defined
    sigs = Dict{String, Vector{Tuple{String, Int, Symbol}}}()   # name => (file, line, :cpu | :gpu)
    for (label, path) in files
        src = read(path, String)
        lines = _julia_code_lines(src)
        tree = try
            _JS.parseall(_JS.SyntaxNode, src; filename = path, ignore_errors = true)
        catch
            continue
        end
        for node in _all_definitions!(Any[], tree)
            name = _definition_name(node)
            first_line = _JS.source_line(node)
            last_line = first_line + count(==('\n'), _JS.sourcetext(node))
            get!(names, _bare_name(name), (label, first_line))
            _is_helper_name(name) && continue
            sig = _signature_text(node)
            occursin(_CPU_SIGNATURE, sig) && push!(get!(sigs, name, Tuple{String, Int, Symbol}[]), (label, first_line, :cpu))
            occursin(_GPU_SIGNATURE, sig) && push!(get!(sigs, name, Tuple{String, Int, Symbol}[]), (label, first_line, :gpu))
            for n in first_line:min(last_line, length(lines))
                occursin(_BACKEND_TEST, lines[n]) || continue
                allowed_backend(label, name) && break
                push!(out, DuplicationViolation(:backend_branch, label, n, string(name, ": ", strip(lines[n])),
                    "the same KernelAbstractions kernel on the CPU backend"))
                break
            end
        end
    end
    # a dispatch pair on a CPU and a GPU type
    for (name, s) in sigs
        any(x -> x[3] === :cpu, s) && any(x -> x[3] === :gpu, s) || continue
        cpu = s[findfirst(x -> x[3] === :cpu, s)]; gpu = s[findfirst(x -> x[3] === :gpu, s)]
        allowed_backend(cpu[1], name) && continue
        push!(out, DuplicationViolation(:backend_branch, cpu[1], cpu[2], string(name, " (a method per backend)"),
            string(gpu[1], ":", gpu[2], " ", name)))
    end
    # twin names
    twins = Tuple{String, String}[]
    for n in keys(names)
        _is_helper_name(n) && continue
        # by characters, not bytes: a name may be non-ASCII (`θ_cpu`)
        for (a, b) in _TWIN_SUFFIXES
            endswith(n, a) && haskey(names, chop(n; tail = length(a)) * b) && push!(twins, (n, chop(n; tail = length(a)) * b))
        end
        for (a, b) in _TWIN_PREFIXES
            startswith(n, a) && haskey(names, b * chop(n; head = length(a), tail = 0)) &&
                push!(twins, (n, b * chop(n; head = length(a), tail = 0)))
        end
        endswith(n, "_cuda") && haskey(names, chop(n; tail = 5)) && push!(twins, (String(chop(n; tail = 5)), n))
    end
    for (a, b) in twins
        i = findfirst(e -> Set(String.(e["names"])) == Set([a, b]), entries["twin"])
        i === nothing || (twin_used[i] = true; continue)
        (file, line) = names[a]
        push!(out, DuplicationViolation(:twin_name, file, line, a, b))
    end
    # stale entries: a backend entry only when its file was scanned here (a package's own run reads the monorepo's
    # file, whose entries about another package's files it cannot match)
    scanned = Set(first.(files))
    stale(kind, list, used, mine) = [DuplicationViolation(:stale_exception, _DUPLICATION_FILE, 0, _entry_text(kind, e), "matches nothing")
        for (e, u) in zip(list, used) if !u && mine(e)]
    append!(out, stale("backend", entries["backend"], backend_used, e -> e["file"] in scanned))
    append!(out, stale("twin", entries["twin"], twin_used, _ -> true))
    return out
end

# --- name ownership ---------------------------------------------------------------------

# A module and its submodules (a submodule defined in it, not one it imports).
function _module_tree(M::Module)
    out = Module[M]
    i = 1
    while i <= length(out)
        S = out[i]
        for n in names(S; all = true)
            isdefined(S, n) || continue
            v = getfield(S, n)
            v isa Module && v !== S && parentmodule(v) === S && !(v in out) && push!(out, v)
        end
        i += 1
    end
    return out
end

# Julia's per-module names, and the stack's per-package conventions: every package declares
# its own `CONTRACT_GENERICS` (the functions others may add methods to; `contract_generic_violations`).
const _INFRASTRUCTURE_NAMES = (:eval, :include, :__init__, :CONTRACT_GENERICS)

# A function, data type or module whose own module is not `S` (a Union has none: its binding decides).
function _defined_elsewhere(v, S)
    t = v isa Type ? Base.unwrap_unionall(v) : v
    (t isa Function || t isa DataType || t isa Module) || return false
    return parentmodule(t) !== S
end

# The names a module tree owns, each `name => the (sub)module that owns it`.
function _owned_names(M::Module)
    out = Dict{Symbol, Module}()
    for S in _module_tree(M), n in names(S; all = true)
        s = string(n)
        (startswith(s, '#') || n in _INFRASTRUCTURE_NAMES || n === nameof(S)) && continue
        isdefined(S, n) || continue
        Base.binding_module(S, n) === S || continue          # imported or re-exported: its owner's
        v = getfield(S, n)
        _defined_elsewhere(v, S) && continue                  # an alias of another module's function or type
        get!(out, n, S)
    end
    return out
end

# The loaded modules of a package's Project.toml `[deps]`.
function _loaded_deps(M::Module)
    dir = pkgdir(M)
    dir === nothing && return Module[]
    p = joinpath(dir, "Project.toml")
    isfile(p) || return Module[]
    deps = get(TOML.parsefile(p), "deps", Dict{String, Any}())
    out = Module[]
    for (name, uuid) in deps
        id = Base.PkgId(Base.UUID(uuid), name)
        haskey(Base.loaded_modules, id) && push!(out, Base.loaded_modules[id])
    end
    return out
end

function _dependency_closure(M::Module)
    seen = Module[]
    todo = _loaded_deps(M)
    while !isempty(todo)
        U = pop!(todo)
        U in seen && continue
        push!(seen, U)
        append!(todo, _loaded_deps(U))
    end
    return seen
end

# The stack packages upstream of M: Core, and every dependency that depends on Core.
function _stack_upstream(M::Module)
    core = VisualGeometryToolkitCore
    return [U for U in _dependency_closure(M) if U === core || core in _dependency_closure(U)]
end

function _name_location(v, root)
    v isa Function || return ("", 0)
    ms = collect(methods(v))
    isempty(ms) && return ("", 0)
    file, line = String(ms[1].file), ms[1].line
    file = startswith(file, root * "/") ? relpath(file, root) : file
    return (file, line)
end

function _name_violations(M::Module, upstream_modules, root, entries)
    out = DuplicationViolation[]
    upstream = Dict{Symbol, Module}()
    for U in sort(upstream_modules; by = string), (n, S) in _owned_names(U)
        get!(upstream, n, S)
    end
    used = falses(length(entries))
    for (n, S) in sort!(collect(_owned_names(M)); by = first)
        haskey(upstream, n) || continue
        hit = findfirst(e -> e["module"] == string(nameof(M)) && e["name"] == string(n), entries)
        hit === nothing || (used[hit] = true; continue)
        file, line = _name_location(getfield(S, n), root)
        push!(out, DuplicationViolation(:name, file, line, string(S === M ? "" : string(nameof(S), "."), n),
            "owned by " * string(upstream[n])))
    end
    for (e, u) in zip(entries, used)
        u || e["module"] != string(nameof(M)) || push!(out, DuplicationViolation(:stale_exception, _DUPLICATION_FILE, 0, _entry_text("name", e), "matches nothing"))
    end
    return out
end

# --- the static checks ----------------------------------------------------------------

# `(label, path)` of every in-scope file of a tree, its label the prefix and its relative path.
_scope_files(dir, prefix) = [(prefix * f, joinpath(dir, f)) for f in naming_scope(dir).files]

function _similar_pairs(defs, similarity)
    minimum_tokens = minimum(last, similarity)
    keep = [i for (i, d) in enumerate(defs) if d.ntokens >= minimum_tokens && !isempty(d.shingles)]
    postings = Dict{UInt64, Vector{Int}}()
    in_repo = Dict{UInt64, Int}()            # the definitions of this repository holding a shingle (the idiom count)
    by_repo = Dict{Tuple{UInt64, String}, Int}()   # per repository (its label prefix, "" for this one)
    in_any = Dict{UInt64, Int}()             # the largest of those counts: the idiom count of a cross-repository pair
    for i in keep, h in defs[i].shingles
        push!(get!(postings, h, Int[]), i)
        m = match(_EXTERNAL_ID, defs[i].file)
        m === nothing && (in_repo[h] = get(in_repo, h, 0) + 1)
        k = (h, m === nothing ? "" : String(m.captures[1]))
        by_repo[k] = get(by_repo, k, 0) + 1
        in_any[h] = max(get(in_any, h, 0), by_repo[k])
    end
    pairs = Tuple{Int, Int, Float64}[]
    count = zeros(Int, length(defs))
    touched = Int[]
    for a in keep
        defs[a].own || continue
        for h in defs[a].shingles
            l = postings[h]
            own_idiom = get(in_repo, h, 0) > _MAX_POSTINGS
            any_idiom = get(in_any, h, 0) > _MAX_POSTINGS
            own_idiom && continue
            for b in l
                (b == a || (defs[b].own && b < a)) && continue          # an own pair counted once
                any_idiom && match(_EXTERNAL_ID, defs[b].file) !== nothing && continue   # across repositories: an idiom anywhere
                count[b] == 0 && push!(touched, b)
                count[b] += 1
            end
        end
        for b in touched
            A, B = defs[a], defs[b]
            c = count[b]; count[b] = 0
            A.file == B.file && continue
            j = c / (length(A.shingles) + length(B.shingles) - c)
            n = min(A.ntokens, B.ntokens)
            any(((jmin, nmin),) -> j >= jmin && n >= nmin, similarity) && push!(pairs, (a, b, j))
        end
        empty!(touched)
    end
    return pairs
end

function _static_violations(dir, root, stack, entries; similarity, shingle, min_file_tokens)
    own_prefix = relpath(abspath(dir), root) == "." ? "" : relpath(abspath(dir), root) * "/"
    defs = _Definition[]; files = Tuple{String, Bool, UInt64, Int}[]; ids = Dict{String, Int}()
    scanned = Set{String}()               # the labels of the trees compared: `""` (this repository) and each `<Repository>:`
    trees = Tuple{String, String, Bool}[(dir, own_prefix, true)]
    for (tree, prefix) in _repo_trees(root; external = false)        # a monorepo's other packages
        prefix == own_prefix || rstrip(abspath(tree), '/') == rstrip(abspath(dir), '/') || push!(trees, (tree, prefix, false))
    end
    push!(scanned, "")
    for s in stack
        if !isdir(s)
            @info "check_duplication: the stack repository $s is absent; its comparison is skipped"
            continue
        end
        rstrip(abspath(s), '/') == root && continue
        for (t, p) in _repo_trees(rstrip(abspath(s), '/'); external = true)
            push!(scanned, p)
            push!(trees, (t, p, false))
        end
    end
    code = Tuple{String, String}[]          # the own files for the pattern table
    # Each file is scanned once, by its real path: `dir` may lie inside a tree that is also
    # scanned (a subdirectory of a single-package repository — a paper project — is inside the
    # repository's own tree), and a file compared with itself is every finding `X ⇔ X`. The own
    # tree is first in `trees`, so a file keeps its own side.
    seen = Set{String}()
    for (tree, prefix, own) in trees, (label, path) in _scope_files(tree, prefix)
        rp = realpath(path)
        rp in seen && continue
        push!(seen, rp)
        endswith(path, ".jl") && _scan_julia!(defs, files, path, label, own, ids, shingle)
        own && any(e -> endswith(path, e), _RAW_CODE_EXTENSIONS) && push!(code, (label, path))
    end
    out = DuplicationViolation[]
    side_scanned(id) = (m = match(_EXTERNAL_ID, id); m === nothing || m.captures[1] in scanned)
    stale(kind, ids_of) = [DuplicationViolation(:stale_exception, _DUPLICATION_FILE, 0, _entry_text(kind, e), "matches nothing")
        for (e, u) in ids_of if !u]

    # similarity
    sim_entries = [(Set(String.(e["functions"])), e) for e in entries["similar"]]
    sim_used = falses(length(sim_entries))
    for (a, b, j) in _similar_pairs(defs, similarity)
        A, B = defs[a], defs[b]
        key = Set([A.file * "::" * A.name, B.file * "::" * B.name])
        hit = findfirst(((s, _),) -> s == key, sim_entries)
        hit === nothing || (sim_used[hit] = true; continue)
        push!(out, DuplicationViolation(:similar, A.file, A.line,
            string(A.name, " (", A.ntokens, " tokens; Jaccard ", round(j; digits = 2), ")"),
            string(B.file, ":", B.line, " ", B.name, " (", B.ntokens, " tokens)")))
    end
    # whole files
    file_entries = [(Set(String.(e["files"])), e) for e in entries["file"]]
    file_used = falses(length(file_entries))
    by_hash = Dict{UInt64, Vector{Int}}()
    for (i, (_, _, h, n)) in enumerate(files)
        n >= min_file_tokens && push!(get!(by_hash, h, Int[]), i)
    end
    for group in values(by_hash), a in group, b in group
        A, B = files[a], files[b]
        (A[2] && (!B[2] || a < b)) || continue
        hit = findfirst(((s, _),) -> s == Set([A[1], B[1]]), file_entries)
        hit === nothing || (file_used[hit] = true; continue)
        push!(out, DuplicationViolation(:file, A[1], 0, string(A[4], " tokens"), B[1]))
    end
    # retired patterns
    pattern_used = falses(length(entries["pattern"]))
    package = _package_name(dir)
    for (label, path) in code
        rel = label[(length(own_prefix) + 1):end]
        src = read(path, String)
        lines = endswith(path, ".jl") ? _julia_code_lines(src) : _hash_code_lines(src)
        for (use, re, owner, (home_package, home)) in _RETIRED_PATTERNS
            package == home_package && _in_home(rel, home) && continue
            for (n, line) in enumerate(lines)
                occursin(re, line) || continue
                hit = findfirst(e -> e["use"] == string(use) && e["file"] == label, entries["pattern"])
                hit === nothing || (pattern_used[hit] = true; continue)
                push!(out, DuplicationViolation(:pattern, label, n, string("[", use, "] ", strip(line)), owner))
            end
        end
    end
    # backend branches and CPU/GPU twins, over this package's Julia files
    append!(out, _backend_violations([(l, p) for (l, p) in code if endswith(p, ".jl")], entries))
    # stale entries: those about this package, every side of which was compared
    about_own(ids) = any(id -> !occursin(_EXTERNAL_ID, id) && startswith(id, own_prefix), ids) && all(side_scanned, ids)
    for (kind, list, used) in (("similar", sim_entries, sim_used), ("file", file_entries, file_used))
        append!(out, stale(kind, [(e, u) for ((s, e), u) in zip(list, used) if about_own(s)]))
    end
    append!(out, stale("pattern", [(e, u) for (e, u) in zip(entries["pattern"], pattern_used) if startswith(e["file"], own_prefix)]))
    return out
end

function check_duplication(dir::AbstractString; stack = _stack_from_env(), similarity = _DEFAULT_SIMILARITY,
        shingle::Integer = 8, min_file_tokens::Integer = 25)
    root = _repo_root(dir)
    entries = _duplication_exceptions(root)
    out = _static_violations(dir, root, stack, entries; similarity, shingle = Int(shingle), min_file_tokens)
    return sort!(out; by = v -> (v.check, v.file, v.line, v.text))
end

function check_duplication(M::Module; kwargs...)
    dir = pkgdir(M)
    root = _repo_root(dir)
    entries = _duplication_exceptions(root)
    names_found = _name_violations(M, _stack_upstream(M), root, entries["name"])
    return sort!(vcat(names_found, check_duplication(dir; kwargs...)); by = v -> (v.check, v.file, v.line, v.text))
end

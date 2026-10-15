# =============================================================================
# Tag registries — the one map from a TOML tag to a type
# =============================================================================
#
# A configuration file names a type by a short string, its TAG (`type =
# "welsch_hq_newton"`). Every such map in the stack is a `TagRegistry`: one
# structure, one spelling rule, one error for a renamed tag and one for an
# unknown one, and a reverse lookup for the writer.
#
#   * `snake_case`          — the spelling rule's one function.
#   * `TagRegistry`         — tag → type (or `nothing`, spelled "none"), the
#                             retired tags and the TOML key paths of the family.
#   * `tag_of`              — the writer side: a type (or value) → its tag.
#   * `tag_rule_violations` — the pairs whose tag is not the rule's.
#   * `tag_registries`      — every registry constant of a module tree.
#   * `tag_table_violations`, `config_tag_violations` — the source and config
#                             scans behind the quality gates.
#
# Construction never checks the rule (a package's load must not fail on a
# spelling): the quality gate does, through `tag_rule_violations`.
# =============================================================================

"""
    snake_case(name::Symbol) -> String
    snake_case(name::AbstractString) -> String

The tag spelling of a CamelCase name, by exactly two rules, then lowercase:

1. an `_` goes before an uppercase letter that follows a lowercase letter or a digit;
2. an `_` goes before the last uppercase letter of an uppercase run when a
   lowercase letter follows it.

A digit stays with its letters.

```julia
snake_case(:FixedS2)              # "fixed_s2"
snake_case(:SmoothTLS)            # "smooth_tls"
snake_case(:WelschHQNewton)       # "welsch_hq_newton"
snake_case(:StudentTUniform)      # "student_t_uniform"
snake_case(:NoLocalOptimization)  # "no_local_optimization"
```
"""
snake_case(name::Symbol) = snake_case(String(name))
function snake_case(name::AbstractString)
    c = collect(name)
    out = Char[]
    for i in eachindex(c)
        if i > 1 && isuppercase(c[i])
            prev = c[i - 1]
            after_lower = islowercase(prev) || isdigit(prev)
            run_end = isuppercase(prev) && i < length(c) && islowercase(c[i + 1])
            (after_lower || run_end) && push!(out, '_')
        end
        push!(out, lowercase(c[i]))
    end
    return String(out)
end

"""
    TagRegistry(family::Symbol, suffix::AbstractString, pairs::Pair{<:AbstractString}...;
                retired = (), paths = Regex[])

The ONLY structure a configuration loader uses to map a TOML tag to a type (or,
rarely, a value). `registry[tag]` returns the type; a retired tag throws
`ArgumentError("tag \\"keep\\" was renamed \\"keep_model\\"")`, and an unknown
tag throws an `ArgumentError` that lists the known tags.

- `family` names the family (`:local_optimization`, `:loss`) in messages.
- `suffix` is the family's common type-name suffix (`"Shape"` for
  `TLSShape`), stripped before the spelling rule; `""` for none.
- `pairs` are `tag => type`. The one value that is not a type is `nothing`,
  whose tag is `"none"`.
- `retired` are `old => new` renames: the old tag no longer loads, and the
  error names its successor.
- `paths` are the TOML key paths whose string values are tags of this family,
  matched whole against the dotted path of a value (`"lo.arms[2].type"`; an
  array element is `[i]`, counted from 1). The config gate reads them.

**The tag rule**: a type's tag is [`snake_case`](@ref) of its name with
`suffix` stripped from its end when present (the whole name otherwise), and
`nothing` is `"none"`. Construction does not check it — a package's load never
fails on a spelling — [`tag_rule_violations`](@ref) does, and the quality gate
runs it over every registry constant. A duplicate tag, or a retired tag that is
also current, is a construction error.

`keys`, `values`, `haskey`, `get`, `length` and iteration (`tag => type`
pairs, in declaration order) cover the current tags only; [`tag_of`](@ref) is
the reverse lookup.
"""
struct TagRegistry{V}
    family::Symbol
    suffix::String
    tags::Vector{String}
    entries::Vector{V}
    index::Dict{String, Int}
    retired::Vector{Pair{String, String}}
    paths::Vector{Regex}
end

function TagRegistry(
        family::Symbol, suffix::AbstractString, pairs::Pair{<:AbstractString}...;
        retired = (), paths = Regex[]
    )
    tags = String[String(first(p)) for p in pairs]
    entries = [last(p) for p in pairs]
    index = Dict{String, Int}()
    for (i, t) in enumerate(tags)
        haskey(index, t) && throw(ArgumentError("$family registry: tag \"$t\" is declared twice"))
        index[t] = i
    end
    ret = Pair{String, String}[String(first(p)) => String(last(p)) for p in retired]
    for (old, _) in ret
        haskey(index, old) && throw(ArgumentError("$family registry: tag \"$old\" is both current and retired"))
    end
    V = isempty(entries) ? Any : mapreduce(typeof, typejoin, entries)
    return TagRegistry{V}(family, String(suffix), tags, collect(V, entries), index, ret, collect(Regex, paths))
end

"""
    tag_family(registry::TagRegistry) -> Symbol

The family a [`TagRegistry`](@ref) names.
"""
tag_family(r::TagRegistry) = r.family

"""
    tag_paths(registry::TagRegistry) -> Vector{Regex}

The TOML key paths whose string values are tags of `registry`'s family.
"""
tag_paths(r::TagRegistry) = r.paths

"""
    retired_tags(registry::TagRegistry) -> Vector{Pair{String, String}}

The `old => new` renames of a [`TagRegistry`](@ref).
"""
retired_tags(r::TagRegistry) = r.retired

_known_tags(r::TagRegistry) = join(("\"$t\"" for t in r.tags), ", ")

function Base.getindex(r::TagRegistry, tag::AbstractString)
    i = get(r.index, tag, 0)
    i > 0 && return r.entries[i]
    for (old, new) in r.retired
        old == tag && throw(ArgumentError("tag \"$tag\" was renamed \"$new\""))
    end
    throw(ArgumentError("unknown $(r.family) tag \"$tag\"; the known tags are $(_known_tags(r))"))
end

Base.haskey(r::TagRegistry, tag::AbstractString) = haskey(r.index, tag)
Base.get(r::TagRegistry, tag::AbstractString, default) = (i = get(r.index, tag, 0); i > 0 ? r.entries[i] : default)
Base.keys(r::TagRegistry) = r.tags
Base.values(r::TagRegistry) = r.entries
Base.length(r::TagRegistry) = length(r.tags)
Base.eltype(::Type{TagRegistry{V}}) where {V} = Pair{String, V}
Base.iterate(r::TagRegistry, i::Int = 1) = i > length(r.tags) ? nothing : (r.tags[i] => r.entries[i], i + 1)

"""
    merge(a::TagRegistry, b::TagRegistry...) -> TagRegistry

The registry of one family that two packages share — the core's tags and a
downstream package's (`merge(RobustVisualGeometry.LO_TAGS,
LocalOptimizationDoneRight.LO_TAGS)`): every current and retired tag of each,
and their key paths. The family and its suffix must agree, and a tag declared
by two of them (or current in one and retired in another) is an
`ArgumentError`, as within one registry. There is no global registry: a caller
that loads several packages' tags merges them explicitly.
"""
function Base.merge(a::TagRegistry, bs::TagRegistry...)
    for b in bs
        (b.family == a.family && b.suffix == a.suffix) || throw(ArgumentError(
            "merge: registries of different families ($(a.family), $(b.family)) do not merge"))
    end
    rs = (a, bs...)
    pairs = [t => e for r in rs for (t, e) in zip(r.tags, r.entries)]
    retired = [p for r in rs for p in r.retired]
    paths = unique([p for r in rs for p in r.paths])
    return TagRegistry(a.family, a.suffix, pairs...; retired, paths)
end

function Base.show(io::IO, r::TagRegistry)
    return print(io, "TagRegistry(:", r.family, ", ", length(r.tags), " tags: ", join(r.tags, ", "), ")")
end

"""
    tag_of(registry::TagRegistry, x) -> String

The writer side of a [`TagRegistry`](@ref): the tag of `x`. `x` is a registered
value (`nothing` → `"none"`), a type registered as such, a subtype of a
registered type (`Irls{2}` under a registered `Irls`; the most specific
registered supertype wins) or an instance of one. An `ArgumentError` when
nothing matches.
"""
function tag_of(r::TagRegistry, x)
    for (t, v) in r
        v === x && return t
    end
    T = x isa Type ? x : typeof(x)
    cands = [(t, v) for (t, v) in r if v isa Type && T <: v]
    isempty(cands) && throw(ArgumentError("no $(r.family) tag for $x; the known tags are $(_known_tags(r))"))
    best = filter(c -> all(o -> c[2] <: o[2], cands), cands)
    length(best) == 1 || throw(ArgumentError("$(r.family) tags $(join(first.(cands), ", ")) all match $x"))
    return first(only(best))
end

# The type name the tag rule spells: a type's own name, a function's name, an
# instance's type name.
function _tag_source_name(v)::String
    v isa Type && v !== Union{} && !(Base.unwrap_unionall(v) isa Union) && return string(nameof(v))
    v isa Function && return string(nameof(v))
    return string(nameof(typeof(v)))
end

"""
    expected_tag(registry::TagRegistry, value) -> String

The tag the rule gives `value` in `registry`: `"none"` for `nothing`, otherwise
[`snake_case`](@ref) of its type name with the registry's suffix stripped from
the end when present.
"""
function expected_tag(r::TagRegistry, v)
    v === nothing && return "none"
    name::String = _tag_source_name(v)
    if !isempty(r.suffix) && endswith(name, r.suffix) && length(name) > length(r.suffix)
        name = name[1:(end - length(r.suffix))]
    end
    return snake_case(name)
end

"""
    tag_rule_violations(registry::TagRegistry) -> Vector{String}

Every way `registry` breaks the tag rule (see [`TagRegistry`](@ref)): a tag
that is not its value's [`expected_tag`](@ref), and a retired tag whose
successor is not a current tag. Empty when the registry obeys it.
"""
function tag_rule_violations(r::TagRegistry)
    out = String[]
    for (t, v) in r
        e = expected_tag(r, v)
        t == e || push!(out, "$(r.family): tag \"$t\" for $v should be \"$e\"")
    end
    for (old, new) in r.retired
        haskey(r, new) || push!(out, "$(r.family): retired tag \"$old\" names \"$new\", which is not a current tag")
    end
    return out
end

# Every module of a package: M and its submodules, depth first.
function _module_tree(m::Module, acc = Module[])
    push!(acc, m)
    for n in names(m; all = true)
        isdefined(m, n) && !Base.isdeprecated(m, n) || continue
        x = getfield(m, n)
        x isa Module && x !== m && parentmodule(x) === m && !(x in acc) && _module_tree(x, acc)
    end
    return acc
end

"""
    tag_registries(M::Module) -> Vector{Pair{String, TagRegistry}}

Every [`TagRegistry`](@ref) constant of `M` and its submodules, as
`"Module.NAME" => registry`.
"""
function tag_registries(M::Module)
    out = Pair{String, TagRegistry}[]
    for m in _module_tree(M), n in names(m; all = true)
        isdefined(m, n) && isconst(m, n) || continue
        v = getfield(m, n)
        v isa TagRegistry && push!(out, "$m.$n" => v)
    end
    return out
end

# The tag-table scan's two patterns: a Dict keyed by a string or symbol whose
# values are types, and a string literal paired with a CamelCase identifier
# that is not called (a type, not a constructor call).
const _TYPE_TABLE = r"Dict\{\s*(?:String|Symbol|AbstractString)\s*,\s*(?:<:\s*)?(?:Type|DataType|UnionAll)\b"
const _TYPE_PAIR = r"\"[A-Za-z0-9_]*\"\s*=>\s*[A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*(?:\{[^{}]*\})?(?=\s*(?:[,)\]]|$|#))"m

# A pair whose value is a Base or Core type (`"cutoff" => Float32`) is a
# key-converter spec, not a tag table: only a package's own types count.
function _is_tag_pair(m::RegexMatch)
    name = Symbol(first(split(strip(last(split(m.match, "=>"))), '{')))
    return !(isdefined(Core, name) || isdefined(Base, name))
end

# Blank out (keeping newlines, so line numbers stand) every `TagRegistry(...)`
# call, every triple-quoted string and every comment — and, with `strings`,
# the contents of every ordinary string literal too. A lexer of the cheap kind:
# ordinary strings are tracked so a `#` inside one is not a comment, and a
# character literal is skipped whole.
function _blank_spans(src::AbstractString; strings::Bool = false)
    c = collect(src)
    n = length(c)
    keep = trues(n)
    at(i, s) = i + length(s) - 1 <= n && all(k -> c[i + k - 1] == s[k], 1:length(s))
    i, instr = 1, false
    while i <= n
        if instr
            c[i] == '"' && (instr = false)
            strings && instr && (keep[i] = false; c[i] == '\\' && i < n && (keep[i + 1] = false))
            i += c[i] == '\\' ? 2 : 1
        elseif at(i, "\"\"\"")
            j = i + 3
            while j <= n && !at(j, "\"\"\"")
                j += c[j] == '\\' ? 2 : 1
            end
            stop = min(j + 2, n)
            keep[i:stop] .= false
            i = stop + 1
        elseif c[i] == '"'
            instr = true
            i += 1
        elseif c[i] == '\'' && i + 2 <= n && (c[i + 2] == '\'' || (c[i + 1] == '\\' && i + 3 <= n && c[i + 3] == '\''))
            i += c[i + 1] == '\\' ? 4 : 3
        elseif c[i] == '#'
            j = i
            while j <= n && c[j] != '\n'
                j += 1
            end
            keep[i:(j - 1)] .= false
            i = j
        elseif at(i, "TagRegistry(") && (i == 1 || !(isletter(c[i - 1]) || isdigit(c[i - 1]) || c[i - 1] == '_'))
            j, depth = i + length("TagRegistry("), 1
            while j <= n && depth > 0
                c[j] == '(' && (depth += 1)
                c[j] == ')' && (depth -= 1)
                j += 1
            end
            keep[i:(j - 1)] .= false
            i = j
        else
            i += 1
        end
    end
    return String([keep[k] || c[k] == '\n' ? c[k] : ' ' for k in 1:n])
end

"""
    tag_table_violations(dir) -> Vector{String}

A heuristic source scan of the `.jl` files under `dir` (a package's `src/`) for
tag tables built outside a [`TagRegistry`](@ref): a `Dict{String, <:Type}`-style
table (keyed by a string or symbol, valued by `Type`, `DataType` or
`UnionAll`), or a `"tag" => SomeType` pair — a string literal paired with a
CamelCase name that is not called and is not a Base or Core type (`"cutoff" =>
Float32` is a key's converter, not a tag). `TagRegistry(...)` calls, comments and
triple-quoted strings are not scanned. Each finding is `"file:line: text"`,
relative to `dir`.
"""
function tag_table_violations(dir::AbstractString)
    out = String[]
    isdir(dir) || return out
    for (root, _, files) in walkdir(dir), f in sort(files)
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        src = _blank_spans(read(path, String))
        for (k, line) in enumerate(split(src, '\n'))
            (occursin(_TYPE_TABLE, line) || any(_is_tag_pair, eachmatch(_TYPE_PAIR, line))) &&
                push!(out, "$(relpath(path, dir)):$k: $(strip(line))")
        end
    end
    return out
end

# Every leaf of a parsed TOML document as `"dotted.path[i]" => value`.
function _toml_leaves(x, prefix::String, out::Vector{Pair{String, Any}})
    if x isa AbstractDict
        for k in sort!(collect(keys(x)))
            _toml_leaves(x[k], isempty(prefix) ? String(k) : "$prefix.$k", out)
        end
    elseif x isa AbstractVector
        for (i, v) in enumerate(x)
            _toml_leaves(v, "$prefix[$i]", out)
        end
    else
        push!(out, prefix => x)
    end
    return out
end

_whole_match(r::Regex, s::AbstractString) = (m = match(r, s); m !== nothing && m.match == s)

"""
    config_tag_violations(registries, config_dirs) -> Vector{String}

Parse every `.toml` under `config_dirs` (recursively); every string value at a
key path one of the [`TagRegistry`](@ref)s claims ([`tag_paths`](@ref),
matched whole) must be a current tag of that registry — not a retired tag and
not an unknown one. Each finding is `"file:path: message"`. A file that does
not parse is a finding. `registries` is a collection of registries or of
`name => registry` pairs (the form [`tag_registries`](@ref) returns).
"""
function config_tag_violations(registries, config_dirs)
    regs = TagRegistry[r isa Pair ? last(r) : r for r in registries]
    out = String[]
    for dir in config_dirs
        isdir(dir) || continue
        for (root, _, files) in walkdir(dir), f in sort(files)
            endswith(f, ".toml") || continue
            path = joinpath(root, f)
            doc = try
                TOML.parsefile(path)
            catch e
                push!(out, "$path: does not parse ($(sprint(showerror, e)))")
                continue
            end
            for (p, v) in _toml_leaves(doc, "", Pair{String, Any}[]), r in regs
                v isa AbstractString && any(rx -> _whole_match(rx, p), r.paths) || continue
                haskey(r, v) && continue
                msg = try
                    r[v]
                    ""
                catch e
                    e isa ArgumentError ? e.msg : rethrow()
                end
                push!(out, "$path:$p: $msg")
            end
        end
    end
    return out
end

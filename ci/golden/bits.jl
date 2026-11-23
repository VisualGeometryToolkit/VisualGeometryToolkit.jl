# The golden bit records of the Core 0.14 migration: the walker, the record format and the comparator.
#
# A record is the bit pattern of every leaf of a result, by PATH:
#   - a primitive scalar (a float, an integer, a Bool, a Char, an enum) is its bits, hex, with its type;
#   - an array is its type, its size and the SHA-256 of the bits of its leaves, in index order;
#   - a struct, tuple or named tuple is its fields (`a.b.c`; a tuple's fields `a.1`); a Dict its entries,
#     sorted by key (`a[key]`); `nothing`, `missing`, a String, a Symbol, a type and a function are recorded as text.
# Two results are bit-identical, `===` leaf by leaf, exactly when their records agree on every path: a
# float's bits distinguish -0.0 from 0.0 and every NaN payload, as `===` does.
#
# The format is TSV, one line per leaf: `id \t path \t type \t value`. A run fixes the process, not the
# algorithms: one Julia thread and one BLAS thread (record.jl and replay.jl both set them).

using SHA: SHA256_CTX, update!, digest!

const GOLDEN_DIR = @__DIR__
const RECORDS_DIR = joinpath(GOLDEN_DIR, "records")

# ---------------------------------------------------------------------------------------------------------------
# Leaves: a primitive scalar's bytes
# ---------------------------------------------------------------------------------------------------------------

_is_primitive(::Type{T}) where {T} = isprimitivetype(T)

# A type's name, fully qualified, whatever the running script has loaded: shown from a module that imports
# nothing (an exported name and a `public` one print the same, so a changed export does not move a record).
baremodule _GoldenNoImports end
# A function or type a golden group defines (in Main, or in its own module `Main.golden_<group>`, cases.jl) is
# named without that module path, so a record does not depend on where the group was evaluated.
_tstr(T) = replace(sprint(io -> show(IOContext(io, :module => _GoldenNoImports), T)), r"\bMain\.(?:golden_\w+\.)?" => "")
_hex(x) = bytes2hex(reinterpret(UInt8, [x]))

# Every leaf's bytes of `x`, in field and index order, into the hash context: the array digest. A leaf
# that is no plain bits (a String, a Symbol, nothing) enters as its text, prefixed by its kind.
function _feed!(ctx, x)
    T = typeof(x)
    if _is_primitive(T)
        update!(ctx, reinterpret(UInt8, [x]))
    elseif x isa Union{AbstractString, Symbol}
        update!(ctx, codeunits(string("s:", x, "\0")))
    elseif x === nothing || x === missing
        update!(ctx, codeunits(string("n:", x, "\0")))
    elseif x isa Union{Type, Function, Module}   # named as the walker names them (`_walk!`)
        update!(ctx, codeunits(string("t:", x isa Type ? _tstr(x) : x isa Function ? nameof(x) : x, "\0")))
    elseif x isa AbstractArray
        update!(ctx, codeunits(string("a:", join(size(x), "x"), "\0")))
        for v in x
            _feed!(ctx, v)
        end
    elseif x isa AbstractDict
        update!(ctx, codeunits(string("d:", length(x), "\0")))
        for k in sort!(collect(keys(x)); by = repr)
            _feed!(ctx, k); _feed!(ctx, x[k])
        end
    else
        for i in 1:nfields(x)
            isdefined(x, i) ? _feed!(ctx, getfield(x, i)) : update!(ctx, codeunits("u:\0"))
        end
    end
    return ctx
end

function _digest(x)
    ctx = SHA256_CTX()
    _feed!(ctx, x)
    return bytes2hex(digest!(ctx))
end

# ---------------------------------------------------------------------------------------------------------------
# The walker: a result → path => (type, value)
# ---------------------------------------------------------------------------------------------------------------

"""
    bit_record(x) -> Vector{Tuple{String, String, String}}

Every leaf of `x` as `(path, type, value)`, in walk order. The root's path is `""`; a field's `parent.name`.
"""
function bit_record(x)
    out = Tuple{String, String, String}[]
    _walk!(out, "", x, IdDict{Any, Nothing}(), 0)
    return out
end

_join(p, s) = isempty(p) ? s : string(p, ".", s)

function _walk!(out, path, x, seen, depth)
    depth > 40 && error("bit_record: nesting deeper than 40 at `$path`")
    T = typeof(x)
    if _is_primitive(T)
        push!(out, (path, _tstr(T), _hex(x)))
    elseif x === nothing || x === missing
        push!(out, (path, _tstr(T), string(x)))
    elseif x isa Union{AbstractString, Symbol}
        push!(out, (path, _tstr(T), escape_string(string(x))))
    elseif x isa Type
        push!(out, (path, "type", _tstr(x)))              # fully qualified, whatever the process has loaded
    elseif x isa Function
        push!(out, (path, "type", string(nameof(x))))     # its name, not its module path
    elseif x isa Module
        push!(out, (path, "type", string(x)))
    elseif x isa AbstractArray
        push!(out, (path, string("array ", _tstr(typeof(x)), " [", join(size(x), "x"), "]"), _digest(x)))
    elseif x isa AbstractDict
        push!(out, (path, string("dict[", length(x), "]"), ""))
        for k in sort!(collect(keys(x)); by = repr)
            _walk!(out, string(path, "[", repr(k), "]"), x[k], seen, depth + 1)
        end
    else
        if ismutable(x)
            haskey(seen, x) && (push!(out, (path, "cycle", _tstr(T))); return out)
            seen[x] = nothing
        end
        nfields(x) == 0 && (push!(out, (path, "singleton", _tstr(T))); return out)
        push!(out, (path, "struct", _tstr(T)))
        for (i, name) in enumerate(fieldnames(T))
            sub = _join(path, x isa Tuple ? string(i) : string(name))
            isdefined(x, i) ? _walk!(out, sub, getfield(x, i), seen, depth + 1) : push!(out, (sub, "undef", ""))
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------------------------------
# Record files
# ---------------------------------------------------------------------------------------------------------------

_tsv_field(s) = replace(s, '\t' => "\\t", '\n' => "\\n")

"""
    write_records(group, entries)

Write `entries` (`id => record`, in order) to `records/<group>.tsv`, one line per leaf.
"""
function write_records(group::AbstractString, entries)
    mkpath(RECORDS_DIR)
    open(joinpath(RECORDS_DIR, group * ".tsv"), "w") do io
        println(io, "# golden bit records, group ", group, ": id\tpath\ttype\tvalue")
        for (id, rec) in entries, (p, t, v) in rec
            println(io, _tsv_field(id), '\t', _tsv_field(p), '\t', _tsv_field(t), '\t', _tsv_field(v))
        end
    end
    return joinpath(RECORDS_DIR, group * ".tsv")
end

"""
    read_records(group) -> Dict{String, Vector{Tuple{String, String, String}}}

The records of `group`, by id, each in its recorded order.
"""
function read_records(group::AbstractString)
    d = Dict{String, Vector{Tuple{String, String, String}}}()
    for line in eachline(joinpath(RECORDS_DIR, group * ".tsv"))
        startswith(line, "#") && continue
        id, p, t, v = split(line, '\t'; limit = 4)
        push!(get!(d, String(id), Tuple{String, String, String}[]), (String(p), String(t), String(v)))
    end
    return d
end

# ---------------------------------------------------------------------------------------------------------------
# Comparison under an explicit field mapping
# ---------------------------------------------------------------------------------------------------------------

"""
    FieldMap(renames, new_only, retyped, ignored)

How a replacement's record maps onto the recorded one. `renames` is a vector of `new_prefix => old_prefix`
pairs, applied to each NEW path, the longest matching prefix first (a prefix matches a whole path segment).
`new_only` lists the new paths (or prefixes) that have no recorded counterpart (`prob`, `alg`, `original`, a
counter the old result did not keep): they must be listed, never silently dropped. `retyped` lists the (old)
paths whose TYPE the migration changes (a result's own type, a renamed metadata type, an array of one): a
struct's fields are still compared one by one, an array's leaf bits as a whole. Every recorded path must be matched by exactly one new path, with the
same leaf type and the same bits. `ignored` lists recorded paths that are not compared at all — only an
exception's message (`exception.msg`), whose wording names the old keywords; its type is always compared.
`intended` holds a case's deliberate behaviour changes (`intended_changes.toml`, by recorded path): such a
leaf must equal the listed old bits in the record AND the listed new bits in the replacement, exactly; each one
matched is pushed to `used`, so the replay can refuse an entry that matched nothing.
"""
struct FieldMap
    renames::Vector{Pair{String, String}}
    new_only::Vector{String}
    retyped::Vector{String}
    ignored::Vector{String}
end
FieldMap(; renames = Pair{String, String}[], new_only = String[], retyped = String[], ignored = String[]) =
    FieldMap(collect(renames), collect(new_only), collect(retyped), collect(ignored))
const IDENTITY_MAP = FieldMap()

_has_prefix(path, pre) = path == pre || startswith(path, pre * ".") || startswith(path, pre * "[")

function _map_path(m::FieldMap, path)
    best = nothing
    for (new, old) in m.renames
        _has_prefix(path, new) || continue
        (best === nothing || length(new) > length(first(best))) && (best = new => old)
    end
    best === nothing && return path
    new, old = best
    rest = path[(lastindex(new) + 1):end]
    return isempty(old) ? lstrip(rest, '.') : string(old, rest)
end

_is_new_only(m::FieldMap, path) = any(pre -> _has_prefix(path, pre), m.new_only)

"""
    compare_record(old, new_record; map = IDENTITY_MAP) -> Vector{String}

The differences between the recorded leaves `old` and the replacement's `new_record` (a `bit_record`) under
`map`: every recorded path missing from the new result, every new path neither mapped onto a recorded one
nor declared `new_only`, and every leaf whose type or bits differ. Empty means bit-identical.
"""
function compare_record(old, new_record; map::FieldMap = IDENTITY_MAP, intended = Dict{String, Any}(), used = Set{String}())
    diffs = String[]
    want = Dict(p => (t, v) for (p, t, v) in old)
    got = Dict{String, Tuple{String, String}}()
    for (p, t, v) in new_record
        _is_new_only(map, p) && continue
        q = _map_path(map, p)
        haskey(got, q) && push!(diffs, "two new paths map onto `$q`")
        got[q] = (t, v)
    end
    for p in map.ignored
        p == "exception.msg" || error("FieldMap: only `exception.msg` may be ignored, got `$p`")
        delete!(want, p); delete!(got, p)
    end
    for (p, (t, v)) in want
        if haskey(intended, p) && haskey(got, p)
            c = intended[p]   # a deliberate behaviour change (intended_changes.toml): exact old and new bits, or a failure
            if (t, v) == (c["old"]["type"], c["old"]["value"]) && got[p] == (c["new"]["type"], c["new"]["value"])
                push!(used, p)
            else
                push!(diffs, "`$p`: intended change does not match — recorded $t $v, new $(got[p][1]) $(got[p][2]); listed old $(c["old"]["type"]) $(c["old"]["value"]), new $(c["new"]["type"]) $(c["new"]["value"])")
            end
        elseif !haskey(got, p)
            push!(diffs, "recorded `$p` ($t) has no counterpart")
        elseif p in map.retyped && (t == "struct") == (got[p][1] == "struct") && v == got[p][2]
            continue   # a declared type change: the struct's fields, or the array's leaf bits, still compared
        elseif got[p] != (t, v)
            push!(diffs, "`$p`: recorded $t $v, new $(got[p][1]) $(got[p][2])")
        end
    end
    for p in keys(got)
        haskey(want, p) || push!(diffs, "new `$p` maps onto no recorded path and is not declared new_only")
    end
    return sort!(diffs)
end

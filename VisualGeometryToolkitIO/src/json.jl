# =============================================================================
# The JSON file boundary — one reader, one writer
# =============================================================================
#
# Every vgio schema file crosses this boundary through these functions, so the
# on-disk shape of a vgio file is decided in one place rather than re-agreed per
# schema, and a reader cannot pick up a variant spelling by being written later
# than its siblings. JSON.jl (1.x) does the encoding and the parsing.
#
# The shape: pretty-printed UTF-8 with a four-space indent, and no trailing
# newline, so every vgio file ends `}`. These files are configuration-adjacent
# and get diffed on recalibration, where a missing final newline shows up as
# `\ No newline at end of file`. Adding one is a one-line change here and
# nowhere else, but it changes the bytes of every vgio file, so it belongs in a
# release that says so.
#
# Numbers are written as JSON.jl writes them: a Float64 keeps its `.0`
# (`1.0`) and an integer is bare. The JSON3-based writer of earlier releases
# printed an integral float as an integer (`1`) and an empty array over two
# lines. Every reader accepts both, and a content identity (canonical.jl) is
# the same for both.
#
# `_parse_json` deliberately does not narrow field types. A parsed object is a
# `JSON.Object{String, Any}` (an ordered `AbstractDict` with dot access, which
# also accepts Symbol keys) and an array is a `Vector{Any}`. The narrowing
# belongs at each use — `d.frame::String`, `d.poses::Vector{Any}` — where the
# expected type is a documented property of that schema field and the assert
# doubles as validation. A blanket narrowing here would only move the `Any`
# one call deeper.

_json(x; allownan::Bool = false) = JSON.json(x; allownan)

# `allownan` reads NaN / Infinity as the value it names. It also reads EVERY
# number as a Float64 (an integer field included), so it is used only to
# locate a non-finite number the strict parse refused, and name its field.
_parse_json(s::AbstractString; allownan::Bool = false) = JSON.parse(s; allownan)

_write_json(path::AbstractString, body) = open(path, "w") do io
    JSON.json(io, body; pretty = 4)
end
# The text `_write_json` writes, as a string.
_json_text(body) = JSON.json(body; pretty = 4)

_read_json(path::AbstractString; allownan::Bool = false) = _parse_json(read(path, String); allownan)

# A vgio file, parsed strictly; a NaN or ±Infinity in it is refused naming its
# field (JSON has no such number, and every vgio number is finite).
function _read_vgio_json(path::AbstractString)
    text = read(path, String)
    return try
        _parse_json(text)
    catch e
        e isa ArgumentError || rethrow()
        doc = try
            _parse_json(text; allownan = true)
        catch
            throw(e)                    # not a non-finite number: JSON's own refusal
        end
        _refuse_nonfinite(path, doc)
        throw(e)
    end
end

# The first non-finite number of a parsed document, refused by its path.
function _refuse_nonfinite(file, doc)
    at = _first_nonfinite(doc, "")
    at === nothing && return doc
    throw(ArgumentError("$(file): $(isempty(at) ? "top level" : at): expected a finite number"))
end
_first_nonfinite(x::Real, at) = (x isa Bool || isfinite(Float64(x))) ? nothing : at
function _first_nonfinite(x::AbstractDict, at)
    for (k, v) in x
        r = _first_nonfinite(v, isempty(at) ? String(k) : "$(at).$(k)")
        r === nothing || return r
    end
    return nothing
end
function _first_nonfinite(x::AbstractVector, at)
    for (i, v) in enumerate(x)
        r = _first_nonfinite(v, "$(at)[$(i)]")
        r === nothing || return r
    end
    return nothing
end
_first_nonfinite(x, at) = nothing

# The same search over a value about to be WRITTEN, before it is serialized:
# the document `_json(x)` would hold, walked in the order JSON.jl writes it
# (a NamedTuple's or Dict's entries in iteration order, a vector's or tuple's
# elements), so the field named is the one a parse of the written text would
# name. A value of any other type (one JSON.jl lowers on its own) is
# serialized and parsed on its own, as a whole document was before.
function _refuse_nonfinite_body(file, body)
    found = _nonfinite_in_body(body)
    found === nothing && return body
    at = _body_path(found)
    throw(ArgumentError("$(file): $(isempty(at) ? "top level" : at): expected a finite number"))
end
# `nothing`, or the path to the first non-finite number as its parts from the
# leaf up (a key or a 1-based index); the path text is built only on a hit.
_nonfinite_in_body(x::Real) = _first_nonfinite(x, "") === nothing ? nothing : Any[]
_nonfinite_in_body(::Union{AbstractString, Symbol, Nothing}) = nothing
_nonfinite_in_body(::AbstractVector{<:Union{AbstractString, Symbol}}) = nothing
_nonfinite_in_body(x::AbstractVector{<:AbstractFloat}) = all(isfinite, x) ? nothing : Any[findfirst(!isfinite, x)]
function _nonfinite_in_body(x::Union{NamedTuple, AbstractDict})
    for (k, v) in pairs(x)
        r = _nonfinite_in_body(v)
        r === nothing || return push!(r, string(k))
    end
    return nothing
end
function _nonfinite_in_body(x::Union{AbstractVector, Tuple})
    for (i, v) in enumerate(x)
        r = _nonfinite_in_body(v)
        r === nothing || return push!(r, i)
    end
    return nothing
end
function _nonfinite_in_body(x)
    at = _first_nonfinite(_parse_json(_json(x; allownan = true); allownan = true), "")
    return at === nothing ? nothing : Any[_LoweredPath(at)]   # a lowered value: its own path, already text
end
# The path `_first_nonfinite` found inside a value JSON.jl lowered ("" for the value itself).
struct _LoweredPath
    at::String
end
_join_path(at, index::Integer) = "$(at)[$(index)]"
_join_path(at, key::AbstractString) = isempty(at) ? String(key) : "$(at).$(key)"
_join_path(at, inner::_LoweredPath) =
    isempty(inner.at) ? at : (isempty(at) || startswith(inner.at, "[")) ? at * inner.at : "$(at).$(inner.at)"
_body_path(parts) = foldl(_join_path, Iterators.reverse(parts); init = "")


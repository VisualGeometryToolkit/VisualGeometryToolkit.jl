# =============================================================================
# Canonical JSON — the byte string content identities are minted from
# =============================================================================
#
# A hash over a JSON encoding is only stable if the encoding is: keys sorted
# bytewise, no whitespace, integers bare, integral floats printed as integers,
# other floats via Base's shortest round-trip printing (Ryu). Written here
# rather than delegated to a JSON library so a serializer upgrade can never
# re-mint every identity in a dataset (BlobBoards' board uids are the first
# three hex digits of exactly this hash; hoisted from there, 2026-09).
#
# The INPUT is an already-lowered tree: `nothing`, `Bool`, `Int64`, `Float64`,
# strings, vectors and string-keyed dicts. Lowering a struct into that tree
# (StructTypes, NamedTuples, units) is the caller's business — it is where the
# schema lives.
# =============================================================================

using SHA: sha256

function _canonical_string(io::IO, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\b'
            print(io, "\\b")
        elseif c == '\f'
            print(io, "\\f")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif (c::AbstractChar) < ' '   # narrowed: an AbstractString iterates `Any`
            print(io, "\\u", lpad(string(UInt16(c), base = 16), 4, '0'))
        else
            print(io, c)
        end
    end
    return print(io, '"')
end

"""
    canonical_json(io::IO, tree)
    canonical_json(tree) -> String

The canonical encoding of a lowered JSON tree (see the file header): sorted
keys, no whitespace, `Int64` bare, integral `Float64` printed as an integer,
other floats shortest-round-trip. Throws on a non-finite number and on any
value outside the lowered vocabulary — a config that cannot be encoded must
fail loudly, never hash to something.
"""
function canonical_json(io::IO, x)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Int64
        print(io, x)
    elseif x isa Float64
        isfinite(x) || throw(ArgumentError("non-finite number in canonical tree"))
        if isinteger(x) && abs(x) <= 9.007199254740992e15
            print(io, Int64(x))
        else
            print(io, x)   # Base shortest round-trip (Ryu)
        end
    elseif x isa AbstractString
        _canonical_string(io, x)
    elseif x isa AbstractVector
        print(io, '[')
        for (i, v) in enumerate(x)
            i > 1 && print(io, ',')
            canonical_json(io, v)
        end
        print(io, ']')
    elseif x isa AbstractDict
        print(io, '{')
        for (i, k) in enumerate(sort!(collect(String, keys(x))))
            i > 1 && print(io, ',')
            _canonical_string(io, k)
            print(io, ':')
            canonical_json(io, x[k])
        end
        print(io, '}')
    else
        throw(ArgumentError("unexpected $(typeof(x)) in canonical tree"))
    end
    return nothing
end
canonical_json(x) = sprint(canonical_json, x)

"""
    sha256_hex(s::AbstractString) -> String

Lower-case hex SHA-256 of the string's bytes — the one hash behind every
content identity in the family (`sha256_hex(canonical_json(tree))`).
"""
sha256_hex(s::AbstractString) = bytes2hex(sha256(codeunits(s)))

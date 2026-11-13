# =============================================================================
# Canonical JSON — RFC 8785 (the JSON Canonicalization Scheme, JCS)
# =============================================================================
#
# A hash over a JSON encoding is only stable if the encoding is, and only
# reproducible outside Julia if the encoding is a published standard: content
# identities (`content_id`) and BlobBoards' board uids are minted from RFC 8785
# (JCS) — object members sorted by their names' UTF-16 code units, no
# whitespace, strings escaped minimally (`\"`, `\\`, `\b \f \n \r \t`, other
# control characters as `\u00xx`, lowercase hex), and numbers in ECMAScript's
# `Number.prototype.toString` form of the IEEE double (ECMA-262 §6.1.6.1.20):
# the shortest digits that round-trip, written without an exponent for a
# decimal exponent in −6 < n ≤ 21 (`1e-05` is `0.00001`, `1.0` is `1`, `-0.0`
# is `0`) and as `d.ddde±x` outside it. Any JCS implementation (Python's, a
# browser's) recomputes the same bytes; written here, not delegated to a JSON
# library, so a serializer upgrade can never re-mint an identity.
#
# The INPUT is an already-lowered tree: `nothing`, `Bool`, `Int64` (exactly
# representable as a double: |n| ≤ 2^53), `Float64`, strings, vectors and
# string-keyed dicts. Lowering a value into that tree is the caller's business
# — it is where the schema lives.
# =============================================================================

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

# ECMAScript's Number::toString of a finite double (RFC 8785 §3.2.2.3): its
# shortest round-trip digits (Julia's printing, Ryu) placed by the decimal
# exponent n (the value is 0.d₁d₂…d_k × 10^n).
function _jcs_number(io::IO, x::Float64)
    isfinite(x) || throw(ArgumentError("non-finite number in canonical tree"))
    iszero(x) && return print(io, '0')            # ±0
    x < 0 && print(io, '-')
    r = repr(abs(x))                              # shortest round-trip: "123.45", "0.001", "1.0e-5", "1.234e21"
    i = findfirst(==('e'), r)
    mant, e = i === nothing ? (r, 0) : (r[1:(i - 1)], parse(Int, r[(i + 1):end]))
    ip, fp = split(mant, '.')
    all_digits = ip * fp                          # the value is 0.(all_digits) × 10^(length(ip) + e)
    stripped = lstrip(all_digits, '0')
    n = length(ip) + e - (length(all_digits) - length(stripped))
    digits = rstrip(stripped, '0')
    k = length(digits)
    if k <= n <= 21
        print(io, digits, '0'^(n - k))
    elseif 0 < n <= 21
        print(io, digits[1:n], '.', digits[(n + 1):end])
    elseif -6 < n <= 0
        print(io, "0.", '0'^(-n), digits)
    else
        print(io, digits[1], k > 1 ? "." * digits[2:end] : "", 'e', n - 1 >= 0 ? '+' : '-', abs(n - 1))
    end
    return nothing
end

"""
    canonical_json(io::IO, tree)
    canonical_json(tree) -> String

The RFC 8785 (JCS) serialization of a lowered JSON tree (see the file header):
members sorted by their names' UTF-16 code units, no whitespace, minimal string
escapes, every number in ECMAScript's form of its double (`1.0` is `1`, `1e-5`
is `0.00001`, `1.0e21` is `1e+21`). Throws on a non-finite number, an `Int64`
past ±2^53 (not exactly a double) and any value outside the lowered vocabulary
— a tree that cannot be encoded must fail loudly, never hash to something.
`Core.sha256_hex(canonical_json(tree))` is a content identity.
"""
function canonical_json(io::IO, x)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Int64
        abs(x) <= 2^53 || throw(ArgumentError("integer $(x) in canonical tree is not exactly a double (|n| ≤ 2^53)"))
        print(io, x)
    elseif x isa Float64
        _jcs_number(io, x)
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
        for (i, k) in enumerate(sort!(collect(String, keys(x)); by = k -> transcode(UInt16, k)))
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

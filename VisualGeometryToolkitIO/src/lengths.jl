# =============================================================================
# Lengths in files: decimal-exact conversion
# =============================================================================
#
# A file holds a length in a human unit (62.8 mm); memory holds metres. For a
# power-of-ten unit the conversion is a SHIFT OF THE DECIMAL EXPONENT, not a
# multiplication: `62.8` read as millimetres is the metre value nearest the
# decimal 0.0628, and writing that value back prints `62.8`. Multiplying by
# the scale (`62.8 * 0.001`) lands one ulp away for a few percent of inputs,
# and the round trip then prints `62.800000000000004`.
#
# `read_length` and `write_length` are the ONE pair every schema reads and
# writes a file length through. Other units (inch, ft, pt) have no decimal
# shift: reading multiplies by the unit's scale, and writing prints the
# fewest significant digits that read back to the identical quantity.

# `parse(Float64, mantissa e (exp + k))` — `repr(x)` may already hold an exponent.
function _shift_exponent(x::Float64, k::Integer)
    isfinite(x) || return x * exp10(k)
    s = repr(x)
    i = findfirst(==('e'), s)
    mantissa, exponent = i === nothing ? (s, 0) : (s[1:(i - 1)], parse(Int, s[(i + 1):end]))
    return parse(Float64, string(mantissa, "e", exponent + k))
end

# The k of a unit whose scale is exactly 10^k metres, or `nothing`.
function _decimal_exponent(scale::Real)
    scale > 0 || return nothing
    k = round(Int, log10(scale))
    return scale == parse(Float64, string("1e", k)) ? k : nothing
end

function _length_scale(u::Units)
    dimension(u) == dimension(m) ||
        throw(ArgumentError("a file length needs a length unit, got `$(unit_string(u))`"))
    return Float64(uscale(u))
end

"""
    read_length(x::Real, u::Units) -> Quantity

The file number `x`, written in the length unit `u`, as a base-form quantity in
metres. For a power-of-ten unit (`mm`, `cm`, `μm`, `m`, `km`) the conversion
shifts the decimal exponent of `x`, so `read_length(62.8, mm)` is the metre
value nearest the decimal 0.0628 and `write_length` returns `62.8` exactly;
other units (`inch`, `ft`, `pt`) multiply by the unit's scale. An `ArgumentError`
for a unit that is not a length.
"""
function read_length(x::Real, u::Units)
    scale = _length_scale(u)
    k = _decimal_exponent(scale)
    v = Float64(x)
    metres = k === nothing ? v * scale : _shift_exponent(v, k)
    return metres * m
end

"""
    write_length(q::Quantity, u::Units) -> Float64

The number a file holds for the length `q` in the unit `u`, the inverse of
[`read_length`](@ref): `write_length(read_length(x, u), u)` prints as `x` for
every `x` of at most 15 significant digits. A power-of-ten unit shifts the
decimal exponent of the metre value; any other unit takes the fewest
significant digits (1 to 17) that `read_length` maps back to the identical `q`.
An `ArgumentError` for a unit that is not a length.
"""
function write_length(q::Quantity, u::Units)
    scale = _length_scale(u)
    dimension(q) == dimension(m) ||
        throw(ArgumentError("a file length needs a length quantity, got dimension $(dimension(q))"))
    metres = Float64(ustrip(m, q))
    k = _decimal_exponent(scale)
    k === nothing || return _shift_exponent(metres, -k)
    target = metres
    value = metres / scale
    for d in 1:17
        candidate = round(value; sigdigits = d)
        Float64(ustrip(m, read_length(candidate, u))) === target && return candidate
    end
    return value
end

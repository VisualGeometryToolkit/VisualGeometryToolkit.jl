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
# writes a file length through; a covariance entry, a length squared or a
# length times an angle, goes through the same shift raised to its power
# (`_read_power` / `_write_power`). Other units (inch, ft, pt) have no decimal
# shift: reading multiplies by the unit's scale, and writing prints the
# fewest significant digits that read back to the identical quantity (a
# computed value the scale's grid skips is written within an ulp).
#
# What is exact, and what cannot be. A file number of at most 15 significant
# digits reads and writes back to its own text, so a file written by a person
# or by the stack from such numbers is rewritten byte for byte. A full
# precision (17 digit) file number need not be: a millimetre double and a
# metre double are on different grids, and about a quarter of 17-digit
# millimetre numbers share their nearest metre value with a neighbour, so one
# of the two cannot come back. Its first rewrite moves it by at most 2 ulp;
# every rewrite after that is byte-identical, because `write_length` writes
# the number that reads back to the identical quantity whenever one exists.
#
# Every Float64 crosses: negatives, ±0 (the sign kept), integers, subnormal
# and huge values, NaN and ±Inf (NaN stays NaN, an infinity keeps its sign).
# Whether a FILE may hold a non-finite value is its kind's rule, not this
# pair's: every vgio kind refuses one by name (validate.jl), JSON having no
# such number; the text exporters write what the value is.

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

# A plain metre value as FlexUnits' base-form length (the sign of a zero kept:
# `x * m` would add an offset and lose it).
_metres(x::Real) = Quantity(Float64(x), dimension(m))

# The file number `x` of a quantity of length dimension `n` (0, 1 or 2: a
# ratio, a length, an area), written in `u`ⁿ, as plain base units (mⁿ).
function _read_power(x::Real, u::Units, n::Int)
    v = Float64(x)
    n == 0 && return v
    scale = _length_scale(u)
    k = _decimal_exponent(scale)
    return k === nothing ? v * scale^n : _shift_exponent(v, n * k)
end

# The inverse: the number a file holds in `u`ⁿ for the plain base value `y`
# (mⁿ) — the one whose `_read_power` is `y` itself whenever there is one.
function _write_power(y::Float64, u::Units, n::Int)
    n == 0 && return y
    scale = _length_scale(u)
    k = _decimal_exponent(scale)
    back(c) = _read_power(c, u, n) === y
    if k !== nothing
        c = _shift_exponent(y, -n * k)
        back(c) && return c
        # a value the shift does not reproduce (a computed, full-precision
        # one): the neighbouring file numbers, nearest first
        for j in 1:3, c2 in (prevfloat(c, j), nextfloat(c, j))
            back(c2) && return c2
        end
        return c          # no file number reads back to `y`: the nearest (within an ulp)
    end
    value = y / scale^n
    for d in 1:17
        candidate = round(value; sigdigits = d)
        back(candidate) && return candidate
    end
    for j in 1:3, c2 in (prevfloat(value, j), nextfloat(value, j))
        back(c2) && return c2
    end
    return value          # no file number reads back to `y` (the scale's grid skips it): within an ulp
end

"""
    read_length(x::Real, u::Units) -> Quantity

The file number `x`, written in the length unit `u`, as a base-form quantity in
metres. For a power-of-ten unit (`mm`, `cm`, `μm`, `m`, `km`) the conversion
shifts the decimal exponent of `x`, so `read_length(62.8, mm)` is the metre
value nearest the decimal 0.0628 and `write_length` returns `62.8` exactly;
other units (`inch`, `ft`, `pt`) multiply by the unit's scale. Any Float64
crosses — negative, ±0 with its sign, NaN, ±Inf; an integer is read as its
Float64. An `ArgumentError` for a unit that is not a length.
"""
read_length(x::Real, u::Units) = _metres(_read_power(x, u, 1))

"""
    write_length(q::Quantity, u::Units) -> Float64

The number a file holds for the length `q` in the unit `u`, the inverse of
[`read_length`](@ref): `write_length(read_length(x, u), u)` prints as `x` for
every `x` of at most 15 significant digits, and `read_length` of the result is
`q` itself whenever any file number reads back to it (a full-precision metre
value may have none in millimetres; the nearest is written, within an ulp). A
power-of-ten unit shifts the decimal exponent of the metre value; any other
unit takes the fewest significant digits (1 to 17) that `read_length` maps back
to the identical `q`. NaN and ±Inf are written as they are. An `ArgumentError`
for a unit that is not a length, or a quantity that is not one.
"""
function write_length(q::Quantity, u::Units)
    _length_scale(u)
    dimension(q) == dimension(m) ||
        throw(ArgumentError("a file length needs a length quantity, got dimension $(dimension(q))"))
    return _write_power(Float64(ustrip(m, q)), u, 1)
end

# -----------------------------------------------------------------------------
# The schemas' use of the pair: vectors of lengths, and their covariances
# -----------------------------------------------------------------------------

# A plain value where a physical one is meant: refused, naming the fix.
_refuse_plain(what) = throw(ArgumentError("$(what) holds plain numbers; attach its unit, e.g. t .* m"))

# A vector of lengths as the file's numbers in `u`; plain numbers are refused.
function _lengths_json(v::AbstractVector, u::Units, what)
    (!isempty(v) && !has_units(v)) && _refuse_plain(what)
    return Float64[write_length(q, u) for q in v]
end

# The file's numbers in `u` as a static vector of lengths (metres).
_lengths_from_json(::Val{N}, v::AbstractVector, u::Units) where {N} =
    SVector{N, Quantity{Float64, Dimensions{FixRat32}}}(ntuple(i -> read_length(v[i]::Real, u), Val(N)))

# A covariance in base units (entry (i, j) of length dimension
# `dims[i] + dims[j]`, each `dims` 0 or 1) as the wire's rows in `u`, and back.
_length_cov_json(Σ::StaticMatrix, u::Units, dims) =
    [Float64[_write_power(Float64(Σ[i, j]), u, dims[i] + dims[j]) for j in axes(Σ, 2)] for i in axes(Σ, 1)]
_length_cov_from_json(c, n::Int, u::Units, dims) = SMatrix{n, n, Float64}(
    ntuple(k -> (i = (k - 1) % n + 1; j = (k - 1) ÷ n + 1; _read_power(c[i][j]::Real, u, dims[i] + dims[j])), n * n)
)
# a pose's 6×6 over [δω; δt] (rotation first: rad, then the length), a point's d×d
const _POSE_COV_DIMS = (0, 0, 0, 1, 1, 1)
_point_cov_dims(n::Int) = ntuple(_ -> 1, n)

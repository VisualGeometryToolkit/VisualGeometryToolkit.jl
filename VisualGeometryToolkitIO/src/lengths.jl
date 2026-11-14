# =============================================================================
# Lengths in files: decimal-exact conversion
# =============================================================================
#
# A file holds a length in a human unit (62.8 mm); memory holds metres. For a
# power-of-ten unit the conversion is a SHIFT OF THE DECIMAL EXPONENT, not a
# multiplication: `62.8` read as millimetres is the metre value nearest the
# decimal 0.0628, and writing that value back prints `62.8`. FlexUnits'
# conversion of a float file number (`62.8 * mm`) multiplies by the Float64 of
# the unit's exact rational scale (`62.8 * 0.001`) and lands one ulp away for
# about one decimal in seven; the round trip then prints `62.800000000000004`.
#
# The two halves have different futures. READING is the one hand-rolled
# conversion of the stack (rule R11 names it): it goes once FlexUnits applies a
# rational scale to a float as `x * num / den`, rounded once (the rule asked
# for in Deduction42/FlexUnits.jl#131). WRITING stays a search under any
# rounding of FlexUnits': a correctly rounded `ustrip(mm, q)` of a value read
# from a decimal file number gives the number back for only about three
# quarters of them (measured: 1 - 50 552/200 000 at 15 digits), because the
# metre and millimetre grids differ, so `write_length` returns the number that
# READS BACK to the stored value, not the converted one.
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
# precision value is written as the number that reads back to it exactly
# whenever one exists; when none does (a millimetre double and a metre double
# are on different grids: about one computed metre value in eight), the writer
# writes the decimal of its PERSISTED value (`_persisted`: the metre value at 15
# significant digits, the value content_id hashes), which reads back as exactly
# that value (IO 0.10.1). So a value, the file written from it and the value
# loaded back always have one identity, and every rewrite is byte-identical. A
# 17-digit number a person wrote in a file that shares its metre value with a
# neighbour moves on its first rewrite, by at most 2 ulp or to its persisted
# form, and stays fixed after that.
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
    fast = _shift_exponent_exact(s, k)
    fast === nothing || return fast
    i = findfirst(==('e'), s)
    mantissa, exponent = i === nothing ? (s, 0) : (s[1:(i - 1)], parse(Int, s[(i + 1):end]))
    return parse(Float64, string(mantissa, "e", exponent + k))
end

# The common case of that `parse` without building or parsing a string: `repr`'s
# digits D (as an integer) times 10^p, p its exponent plus `k` less its
# fraction digits. When D ≤ 2^53 and |p| ≤ 22, D and 10^|p| are exact doubles,
# so ONE IEEE multiply or divide is the correctly rounded D·10^p — the very
# value `parse` returns, which is correctly rounded too (Clinger's fast path).
# `nothing` otherwise, and the string path decides.
const _EXACT_POWERS_OF_TEN = Tuple(parse(Float64, string("1e", p)) for p in 0:22)   # each exact
function _shift_exponent_exact(s::String, k::Integer)
    D = UInt64(0)
    fraction_digits = 0
    in_fraction = false
    negative = false
    exponent = 0
    cu = codeunits(s)
    i = 1
    n = length(cu)
    @inbounds while i <= n
        c = cu[i]
        if c == UInt8('-') && i == 1
            negative = true
        elseif c == UInt8('.')
            in_fraction = true
        elseif UInt8('0') <= c <= UInt8('9')
            D = 10D + (c - UInt8('0'))
            D > UInt64(2)^53 && return nothing
            fraction_digits += in_fraction
        elseif c == UInt8('e')
            exponent = parse(Int, view(s, (i + 1):n))
            break
        else
            return nothing
        end
        i += 1
    end
    p = exponent + Int(k) - fraction_digits
    abs(p) <= 22 || return nothing
    v = p >= 0 ? Float64(D) * _EXACT_POWERS_OF_TEN[p + 1] : Float64(D) / _EXACT_POWERS_OF_TEN[1 - p]
    return negative ? -v : v
end

# The k of a unit whose scale is exactly 10^k metres, or `nothing` (the
# registry's exact 1//1000 converts to the double nearest 10^-3, as `_ten_to`).
function _decimal_exponent(scale::Real)
    scale > 0 || return nothing
    k = round(Int, log10(scale))
    return scale == _ten_to(k) ? k : nothing
end
# `parse(Float64, string("1e", k))`, the nearest double to 10^k, from a table built by
# that very parse for the exponents a length unit has.
const _TEN_TO = Tuple(parse(Float64, string("1e", k)) for k in -30:30)
_ten_to(k::Int) = -30 <= k <= 30 ? _TEN_TO[k + 31] : parse(Float64, string("1e", k))

function _length_scale(u::Units)
    dimension(u) == dimension(m) ||
        throw(ArgumentError("a file length needs a length unit, got `$(unit_string(u))`"))
    return Float64(uscale(u))
end

# A plain metre value as FlexUnits' base-form length (the sign of a zero kept:
# `x * m` would add an offset and lose it).
_metre_quantity(x::Real) = Quantity(Float64(x), dimension(m))

# The file number `x` of a quantity of length dimension `n` (0, 1 or 2: a
# ratio, a length, an area), written in `u`ⁿ, as plain base units (mⁿ).
function _read_power(x::Real, u::Units, n::Int)
    v = Float64(x)
    n == 0 && return v
    scale = _length_scale(u)
    return _read_scaled(v, scale, _decimal_exponent(scale), n)
end
# `_read_power` once the unit's scale and decimal exponent `k` are known.
_read_scaled(v::Float64, scale::Float64, k, n::Int) = k === nothing ? v * scale^n : _shift_exponent(v, n * k)

# The inverse: the number a file holds in `u`ⁿ for the plain base value `y`
# (mⁿ) — the one whose `_read_power` is `y` itself whenever there is one.
function _write_power(y::Float64, u::Units, n::Int)
    n == 0 && return y
    return _write_scaled(y, _length_scale(u), n)
end
# Set (only) by the reproductions of IO 0.9's identities (migrate_v010.jl, `_v09`): their millimetre numbers are the
# ones 0.9's `_persisted` read back, i.e. 0.10.0's writer, whose last resort was the nearest file number.
const _WRITE_UNROUNDED = Base.ScopedValues.ScopedValue(false)
function _write_scaled(y::Float64, scale::Float64, n::Int)
    k = _decimal_exponent(scale)
    back(c) = _read_scaled(c, scale, k, n) === y    # `_read_power(c, u, n) === y`
    # The last resort, when no file number reads back to `y` itself (a computed, full-precision value; about one in
    # eight in mm): a file number whose value has `y`'s IDENTITY, its persisted form (`_persisted`, content_id's
    # length leaf: the metre value at 15 significant digits), so the value, the file written from it and the value
    # loaded back have one identity and a rewrite is byte-stable (IO 0.10.1). 0.10.0 wrote the nearest file number,
    # which read back an ulp off, on the other side of a 15th-digit rounding boundary for about one computed value in
    # ninety: two identities for one rig.
    same_identity(c) = _persisted(_read_scaled(c, scale, k, n), n) === _persisted(y, n)
    if k !== nothing
        # the common path: the shift, and one read-back
        c = _shift_exponent(y, -n * k)
        back(c) && return c
        # a value the shift does not reproduce (a computed, full-precision
        # one): the neighbouring file numbers, nearest first
        for j in 1:3, c2 in (prevfloat(c, j), nextfloat(c, j))
            back(c2) && return c2
        end
        _WRITE_UNROUNDED[] && return c   # 0.10.0's: the nearest (within an ulp)
        # the persisted value's own decimal: it reads back as exactly the persisted value
        return _shift_exponent(_persisted(y, n), -n * k)
    end
    # The fewest significant digits that read back. The search starts at one
    # digit, which is where a person's number (2.5 inch) is found; a guessed
    # starting count would have to prove every count below it fails, which is
    # the same search.
    value = y / scale^n
    for d in 1:17
        candidate = round(value; sigdigits = d)
        back(candidate) && return candidate
    end
    for j in 1:3, c2 in (prevfloat(value, j), nextfloat(value, j))
        back(c2) && return c2
    end
    _WRITE_UNROUNDED[] && return value   # 0.10.0's: within an ulp
    # the scale's grid skips `y`: the fewest digits whose value has `y`'s identity
    for d in 1:17
        candidate = round(value; sigdigits = d)
        same_identity(candidate) && return candidate
    end
    for j in 1:3, c2 in (prevfloat(value, j), nextfloat(value, j))
        same_identity(c2) && return c2
    end
    return value          # within an ulp (no number of this unit has `y`'s identity: not seen)
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

This is the stack's one hand-rolled unit conversion: FlexUnits' own
`x * u` of a float `x` multiplies by the Float64 of the unit's exact
rational scale, one ulp off the decimal for about one number in seven. It is
removed — `read_length(x, u)` becoming `base_quantity(x * u)` — when
FlexUnits applies a rational scale to a float as `x * num / den`, rounded
once (Deduction42/FlexUnits.jl#131).
"""
read_length(x::Real, u::Units) = _metre_quantity(_read_power(x, u, 1))

"""
    write_length(q::Quantity, u::Units) -> Float64

The number a file holds for the length `q` in the unit `u`, the inverse of
[`read_length`](@ref): `write_length(read_length(x, u), u)` prints as `x` for
every `x` of at most 15 significant digits, and `read_length` of the result is
`q` itself whenever any file number reads back to it. When none does (a
full-precision metre value computed in code, about one in eight in
millimetres), the number written reads back as `q`'s PERSISTED value, its metre
value at 15 significant digits (the value [`content_id`](@ref) hashes): either
way the length, the file written from it and the length loaded back have one
identity, and a rewrite is byte for byte (IO 0.10.1; 0.10.0 wrote the nearest
number, an ulp off, whose identity could differ). A power-of-ten unit shifts
the decimal exponent of the metre value; any other unit takes the fewest
significant digits (1 to 17) that `read_length` maps back to the identical `q`.
NaN and ±Inf are written as they are. An `ArgumentError` for a unit that is not
a length, or a quantity that is not one.

The search is not a conversion and stays: FlexUnits' `ustrip(u, q)`, even
correctly rounded, gives back the file's own number for only about three
quarters of decimal file numbers, since the stored metre value is on a
different grid; `write_length` returns the number that reads back to `q`.
"""
function write_length(q::Quantity, u::Units)
    scale = _length_scale(u)
    dimension(q) == dimension(m) ||
        throw(ArgumentError("a file length needs a length quantity, got dimension $(dimension(q))"))
    return _write_scaled(Float64(ustrip(m, q)), scale, 1)
end

# The PERSISTED form of a plain base-unit number (a length in metres, a
# covariance entry in m^n): the Float64 nearest to it rounded to 15 significant
# decimal digits. The identity tree (`content_id`) hashes this form. Every file
# number of at most 15 significant digits, in any decimal unit, reads back
# (decimal-exact, `read_length`) as exactly such a value, so a value built in
# code (`v * mm`, one ulp off the nearest decimal of `v`), the file written from
# it and the value loaded back have one identity; it moves a value by at most
# one rounding at the 15th digit and is idempotent. No unit is chosen: the
# normalisation is of the metre value itself.
_persisted(y::Real, n::Int = 1) = (x = Float64(y); isfinite(x) && !iszero(x) ? parse(Float64, @sprintf("%.14e", x)) : x)

# Where a writer puts a length leaf: a file's unit (`Units`), or `_PERSISTED`,
# the identity tree's (`content_id`) persisted metre value. The target, pose and
# covariance writers take either, so the identity is built by the writer's own
# walk and converts at the leaves only.
struct _Persisted end
const _PERSISTED = _Persisted()
const _LeafUnit = Union{Units, _Persisted}
_leaf_power(y::Float64, u::Units, n::Int) = _write_power(y, u, n)
_leaf_power(y::Float64, ::_Persisted, n::Int) = _persisted(y, n)
_leaf_length(q, u::Units) = write_length(q, u)
_leaf_length(q, ::_Persisted) = _persisted(ustrip(m, q))

# -----------------------------------------------------------------------------
# The schemas' use of the pair: vectors of lengths, and their covariances
# -----------------------------------------------------------------------------

# A plain value where a physical one is meant: refused, naming the fix.
_refuse_plain(what) = throw(ArgumentError("$(what) holds plain numbers; attach its unit, e.g. t .* m"))

# A vector of lengths as the file's numbers in `u`; plain numbers are refused.
function _lengths_json(v::AbstractVector, u::_LeafUnit, what)
    (!isempty(v) && !has_units(v)) && _refuse_plain(what)
    return Float64[_leaf_length(q, u) for q in v]
end

# The file's numbers in `u` as a static vector of lengths (metres).
_lengths_from_json(::Val{N}, v::AbstractVector, u::Units) where {N} =
    SVector{N, BaseQuantity}(ntuple(i -> read_length(v[i]::Real, u), Val(N)))

# A covariance in base units (entry (i, j) of length dimension
# `dims[i] + dims[j]`, each `dims` 0 or 1, or false or true as in a
# calibration's `length_mask`) as the wire's rows in `u`, and back.
# `_cov_rows` applies `f(entry, dimension)` to every entry, row by row.
_cov_rows(f, Σ::AbstractMatrix, dims) = [Float64[f(Float64(Σ[i, j]), dims[i] + dims[j]) for j in axes(Σ, 2)] for i in axes(Σ, 1)]

# The wire form states the storage the covariance holds (VisualGeometryToolkitCore's
# CovMat: an SMatrix, SDiagonal or ScaledIdentity): rows; `{"diagonal": [d]}`;
# `{"isotropic": λ}`. An isotropic one over entries of different length
# dimensions (a pose's angles and lengths) has no single number in the file's
# units, so it is written as its diagonal. `f(entry, dimension)` per entry.
_cov_wire(f, Σ::CovMat, dims) = _cov_wire(f, parent(Σ), dims)
_cov_wire(f, Σ::AbstractMatrix, dims) = _cov_rows(f, Σ, dims)
_cov_wire(f, D::Diagonal{<:Any, <:StaticVector}, dims) =
    Dict{String, Any}("diagonal" => Float64[f(Float64(D.diag[i]), dims[i] + dims[i]) for i in eachindex(D.diag)])
_cov_wire(f, S::ScaledIdentity, dims) = all(==(first(dims)), dims) ?
    Dict{String, Any}("isotropic" => f(Float64(S.λ), first(dims) + first(dims))) : _cov_wire(f, Diagonal(S), dims)
# … and back, `g(number, dimension)` per entry: the storage the wire form states.
_cov_from_wire(g, c::AbstractVector, n::Int, dims) = SMatrix{n, n, Float64}(
    ntuple(k -> (i = (k - 1) % n + 1; j = (k - 1) ÷ n + 1; g(c[i][j]::Real, dims[i] + dims[j])), n * n)
)
function _cov_from_wire(g, c::AbstractDict, n::Int, dims)
    haskey(c, "diagonal") && return Diagonal(SVector{n, Float64}(ntuple(i -> g(c["diagonal"][i]::Real, dims[i] + dims[i]), n)))
    haskey(c, "isotropic") && return ScaledIdentity{n}(Float64(g(c["isotropic"]::Real, first(dims) + first(dims))))
    throw(ArgumentError("a covariance object is {\"diagonal\": […]} or {\"isotropic\": λ}, got the keys $(collect(keys(c)))"))
end
_length_cov_json(Σ::AbstractMatrix, u::_LeafUnit, dims) = _cov_wire((x, n) -> _leaf_power(x, u, n), Σ, dims)
_length_cov_from_json(c, n::Int, u::Units, dims) = _cov_from_wire((x, k) -> _read_power(x, u, k), c, n, dims)
# a pose's 6×6 over [δω; δt] (rotation first: rad, then the length), a point's d×d
const _POSE_COV_DIMS = (0, 0, 0, 1, 1, 1)
_point_cov_dims(n::Int) = ntuple(_ -> 1, n)

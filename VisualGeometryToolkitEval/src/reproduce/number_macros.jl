# =============================================================================
# number_macros.jl — the ONE writer of a paper's number macros. Every number a
# performance change could move is a LaTeX macro: a number of the prose, a
# cell of a table row, a value a figure (its caption, its axis) reads. A driver
# collects them in a `PaperNumbers` and writes one fragment beside its results
# (`write_numbers`); `gather_numbers` (paper_assets.jl) gathers the fragments
# into the paper's numbers file, and `deploy_assets` puts that into the paper
# only with sign-off.
#
# One macro per number, never a row of `a & b & c`: the gatherer and its checks
# work on a macro's value as one number (`compare_fragments`' same numbers,
# `macro_literal_violations`' re-typed literals, `uncited_macros`), and the
# prose cites a table's number by the same macro the table prints.
#
# Dependency-free by contract (Base only), like reproduce_index.jl, which it
# follows; paper_assets.jl follows it (`gather_numbers` keeps the provenance
# comments `write_numbers` writes).
# =============================================================================

# ---------------------------------------------------------------------------
# Typesetting a number
# ---------------------------------------------------------------------------

# An integer `q` scaled by 10^-decimals as a decimal string: "-12.30" for
# (-1230, 2). A zero never carries a sign.
function _decimal_string(q::Integer, decimals::Integer)
    s = string(abs(q))
    if decimals > 0
        s = lpad(s, decimals + 1, '0')
        s = s[1:(end - decimals)] * "." * s[(end - decimals + 1):end]
    end
    return (q < 0 ? "-" : "") * s
end

_pow10(e::Integer) = e >= 0 ? big(10)^e // 1 : 1 // big(10)^(-e)

# The exact value `r` at `digits` decimals, rounded to nearest, ties to even:
# Printf's `%.<digits>f` of the same value.
_fixed(r::Rational{BigInt}, digits::Integer) = _decimal_string(round(BigInt, r * _pow10(digits), RoundNearest), digits)

# The exact value `r` at `sigdigits` significant digits, trailing zeros dropped,
# never an exponent.
function _significant(r::Rational{BigInt}, sigdigits::Integer)
    iszero(r) && return "0"
    a = abs(r)
    e = floor(Int, log10(Float64(a)))
    while a < _pow10(e)
        e -= 1
    end
    while a >= _pow10(e + 1)
        e += 1
    end
    decimals = sigdigits - 1 - e
    q = round(BigInt, a * _pow10(decimals), RoundNearest)
    if q == big(10)^sigdigits          # rounded up to the next power of ten
        decimals -= 1
        q = round(BigInt, a * _pow10(decimals), RoundNearest)
    end
    s = decimals >= 0 ? _decimal_string(q, decimals) : string(q * big(10)^(-decimals))
    occursin('.', s) && (s = rstrip(rstrip(s, '0'), '.'))
    return (r < 0 ? "-" : "") * s
end

# Thin-space groups of three in the integer part, from `group_from` digits on (0: never).
function _grouped(s::AbstractString, group_from::Integer)
    sign, body = startswith(s, "-") ? ("-", s[2:end]) : ("", s)
    int, frac = occursin('.', body) ? split(body, '.'; limit = 2) : (body, nothing)
    if group_from > 0 && length(int) >= group_from
        int = join(reverse([reverse(join(g)) for g in Iterators.partition(reverse(int), 3)]), "\\,")
    end
    return sign * int * (frac === nothing ? "" : "." * frac)
end

function _typeset(x::Real, digits, sigdigits, percent::Bool, group_from::Integer)
    isfinite(x) || throw(ArgumentError("tex_number: $x is not a number the paper can print"))
    digits === nothing || sigdigits === nothing ||
        throw(ArgumentError("tex_number: give digits or sigdigits, not both"))
    digits === nothing || digits >= 0 || throw(ArgumentError("tex_number: digits = $digits is negative"))
    sigdigits === nothing || sigdigits >= 1 || throw(ArgumentError("tex_number: sigdigits = $sigdigits is below 1"))
    group_from >= 0 || throw(ArgumentError("tex_number: group_from = $group_from is negative"))
    r = Rational{BigInt}(x) * (percent ? 100 : 1)
    s = digits !== nothing ? _fixed(r, digits) :
        sigdigits !== nothing ? _significant(r, sigdigits) :
        isinteger(r) ? string(numerator(r)) :
        throw(ArgumentError("tex_number: $x is not an integer; give digits or sigdigits"))
    return _grouped(s, group_from) * (percent ? "\\%" : "")
end

"""
    tex_number(x; digits = nothing, sigdigits = nothing, percent = false, group_from = 0,
               unit = "") -> String
    tex_number(lo, hi; range_separator = "--", <the same keywords>) -> String

A number as the paper prints it, the body of its macro. `digits` fixed
decimals (Printf's `%.<digits>f`: the exact value rounded to nearest, ties to
even); or `sigdigits` significant digits with trailing zeros dropped, never an
exponent; an integer value (`3`, `3.0`) needs neither. A zero never carries a
sign (`-0.001` at 2 digits is `0.00`). `percent` prints `100x` followed by
`\\%`; `group_from` groups the integer part in threes at thin spaces (`\\,`)
from that many digits on (`4`: `1\\,234`; `0`: never); `unit` follows the
number after a thin space (`1.5\\,mm`). Two values are a range, `lo` and `hi`
each typeset so, joined by `range_separator` (LaTeX's en dash by default; in
math mode, `"\\text{--}"`), the unit once after `hi`.

A NaN or an infinity is an `ArgumentError`: a missing evaluation fails its
stage, it never reaches the manuscript. So is a non-integer `x` without
`digits` or `sigdigits`, or both given.
"""
function tex_number(x::Real; digits::Union{Nothing, Integer} = nothing, sigdigits::Union{Nothing, Integer} = nothing,
        percent::Bool = false, group_from::Integer = 0, unit::AbstractString = "")
    return _typeset(x, digits, sigdigits, percent, group_from) * _unit_suffix(unit)
end

function tex_number(lo::Real, hi::Real; range_separator::AbstractString = "--",
        digits::Union{Nothing, Integer} = nothing, sigdigits::Union{Nothing, Integer} = nothing,
        percent::Bool = false, group_from::Integer = 0, unit::AbstractString = "")
    _check_tex_text(range_separator, "range_separator")
    return _typeset(lo, digits, sigdigits, percent, group_from) * range_separator *
           _typeset(hi, digits, sigdigits, percent, group_from) * _unit_suffix(unit)
end

_unit_suffix(unit::AbstractString) = isempty(unit) ? "" : "\\," * _check_tex_text(unit, "unit")

# ---------------------------------------------------------------------------
# Macro names and bodies
# ---------------------------------------------------------------------------

# A name `\newcommand` accepts and the gatherer reads: letters only, never
# `end…` (LaTeX refuses those: `\endX` closes environment X).
function _check_macro_name(name::AbstractString)
    occursin(r"^[A-Za-z]+$", name) ||
        throw(ArgumentError("a number macro name is letters only (digits spelled out, see number_macro_name): $(repr(name))"))
    startswith(name, "end") && throw(ArgumentError("a macro name may not begin with \"end\" (LaTeX refuses it): $(repr(name))"))
    return String(name)
end

# A macro body (or a piece of one): one line, braces balanced, and none of the
# characters that would make it more than one value: an unescaped `%` (the
# rest is a comment), `#` (an argument), `&` (a second table cell), `\\` (a row break).
function _check_tex_text(s::AbstractString, what::AbstractString)
    bad(msg) = throw(ArgumentError("$what $(repr(s)): $msg"))
    isempty(s) && bad("is empty")
    occursin(r"[\n\r]", s) && bad("is not one line")
    occursin("\\\\", s) && bad("holds a row break \\\\")
    depth, prev = 0, ' '
    for c in s
        if prev != '\\'
            c == '{' && (depth += 1)
            c == '}' && (depth -= 1)
            depth < 0 && bad("closes a brace it never opened")
            c in ('%', '#', '&') && bad("holds an unescaped $c")
        end
        prev = c
    end
    depth == 0 || bad("leaves a brace open")
    return String(s)
end

const _DIGIT_WORDS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")

"""
    number_macro_name(prefix, parts...) -> String

The name of a number macro (without its backslash) from a driver's `prefix`
and name `parts`, the stack's naming rule: each part (a string, symbol or
number) split into words at `_`, `-`, `.`, `/`, `×` and spaces, each word
capitalised, each digit spelled out as a word of its own.
`number_macro_name("chromeTable", "kb_start", "median")` is
`"chromeTableKbStartMedian"`; `number_macro_name("pose", "fx3_14mm")` is
`"poseFxThreeOneFourMm"`. The prefix is letters only; a part holding any other
character (a Greek letter, `%`, a comma) is an `ArgumentError`, as is a name
LaTeX refuses (one beginning with `end`).
"""
function number_macro_name(prefix::AbstractString, parts...)
    occursin(r"^[A-Za-z]+$", prefix) || throw(ArgumentError("a macro prefix is letters only: $(repr(prefix))"))
    words = String[]
    for p in parts
        s = string(p)
        occursin(r"^[A-Za-z0-9_\-./× ]*$", s) ||
            throw(ArgumentError("a macro name part is ASCII letters, digits and the separators _ - . / × space: $(repr(s))"))
        for w in split(s, r"[_\-./×\s]+"; keepempty = false), m in eachmatch(r"[A-Za-z]+|\d", w)
            push!(words, isdigit(m.match[1]) ? _DIGIT_WORDS[parse(Int, m.match) + 1] : uppercasefirst(m.match))
        end
    end
    return _check_macro_name(prefix * join(words))
end

# ---------------------------------------------------------------------------
# A driver's numbers
# ---------------------------------------------------------------------------

const _PROVENANCE_FIELDS = (:driver, :config, :run_key, :commit)
const _Provenance = NamedTuple{_PROVENANCE_FIELDS, NTuple{4, String}}

function _provenance(p::NamedTuple)
    extra = setdiff(keys(p), _PROVENANCE_FIELDS)
    isempty(extra) || throw(ArgumentError("a number's provenance is driver, config, run_key and commit; not $(collect(extra))"))
    out = _Provenance(Tuple(String(p[k]) for k in _PROVENANCE_FIELDS))
    for k in _PROVENANCE_FIELDS
        v = out[k]
        (isempty(v) || occursin(r"\s", v)) && throw(ArgumentError("provenance $k $(repr(v)): empty or holds a space"))
    end
    for k in (:driver, :config)
        isabspath(out[k]) && throw(ArgumentError("provenance $k $(repr(out[k])) is a machine path; give it project-relative"))
    end
    return out
end

struct _NumberMacro
    value::String
    provenance::_Provenance
    label::String
end

"""
    PaperNumbers(; driver, config, run_key, commit)

The number macros one driver writes, empty, each carrying its provenance: the
`driver` and its `config` of record (project-relative paths), the `run_key` of
the run the numbers are read from and the `commit` that computed it (the run's
recorded `gitcommit`, not the checkout's: the commit moves only when the run
does). A number added with its own `provenance` (one read from another run)
overrides any of the four. Fill it with [`add_number!`](@ref) and
[`add_table_row!`](@ref); write it with [`write_numbers`](@ref).
"""
struct PaperNumbers
    provenance::_Provenance
    macros::Dict{String, _NumberMacro}
end

PaperNumbers(; driver::AbstractString, config::AbstractString, run_key::AbstractString, commit::AbstractString) =
    PaperNumbers(_provenance((; driver, config, run_key, commit)), Dict{String, _NumberMacro}())

Base.length(nums::PaperNumbers) = length(nums.macros)

function _add!(nums::PaperNumbers, name::AbstractString, body::AbstractString, label::AbstractString, provenance::NamedTuple)
    _check_macro_name(name)
    haskey(nums.macros, name) && throw(ArgumentError("\\$name is already one of this driver's numbers"))
    isempty(label) || _check_tex_text(label, "label")
    nums.macros[name] = _NumberMacro(body, _provenance(merge(nums.provenance, provenance)), label)
    return nums
end

_body(value::AbstractString; format...) =
    isempty(format) ? _check_tex_text(value, "the value") :
    throw(ArgumentError("a typeset value $(repr(value)) takes no format keywords ($(join(keys(format), ", ")))"))
_body(value::Real; format...) = tex_number(value; format...)
_body(value::Tuple{Real, Real}; format...) = tex_number(value...; format...)

"""
    add_number!(nums, name, value; label = "", provenance = (;), format...) -> nums

Add the number macro `\\<name>` to `nums`. A `value` that is a number is
typeset by [`tex_number`](@ref) with the `format` keywords (`digits`,
`sigdigits`, `percent`, `group_from`, `unit`); a pair `(lo, hi)` is a range; a
string is its body as typeset (a label, `"---"` for a missing cell, `10^{-6}`)
and takes no format. `label` names what reads the number when it is not the
prose: a figure's `fig:…` (its caption, its axis), a table's `tab:…`. It goes
into the macro's provenance comment. `provenance` overrides fields of the
driver's ([`PaperNumbers`](@ref)) for this number.

`name` is letters only, never beginning with `end` ([`number_macro_name`](@ref)
builds one); a name `nums` already holds, a body of more than one value (an
unescaped `%`, `#` or `&`, a row break, unbalanced braces) and a NaN are each
an `ArgumentError`.
"""
function add_number!(nums::PaperNumbers, name::AbstractString, value::Union{Real, Tuple{Real, Real}, AbstractString};
        label::AbstractString = "", provenance::NamedTuple = (;), format...)
    return _add!(nums, name, _body(value; format...), label, provenance)
end

"""
    add_table_row!(nums, row, cells; label = "", provenance = (;), format...) -> String

Add a table row's numbers to `nums`, one macro per cell: `cells` is a
collection of `column => value` pairs in the table's column order, and the
cell's macro is `\\<row><Column>` (`row` and each `column` letters only, the
column capitalised). Each value is typeset as by [`add_number!`](@ref), the
`format` keywords shared by the row; a cell of another format is given as its
[`tex_number`](@ref) string. Returns the row as the table's tex reads it,
`"\\<row><A> & \\<row><B> & …"`, without the row break.

One macro per cell, not one per row: the prose cites a table's number by the
macro the table prints, and the gatherer's checks
([`compare_fragments`](@ref), [`macro_literal_violations`](@ref),
[`uncited_macros`](@ref)) see each number.
"""
function add_table_row!(nums::PaperNumbers, row::AbstractString, cells;
        label::AbstractString = "", provenance::NamedTuple = (;), format...)
    _check_macro_name(row)
    isempty(cells) && throw(ArgumentError("table row $(repr(row)) has no cells"))
    isempty(label) || _check_tex_text(label, "label")
    _provenance(merge(nums.provenance, provenance))
    staged = map(collect(cells)) do (col, value)       # every cell checked before any is added
        column = string(col)
        occursin(r"^[A-Za-z]+$", column) || throw(ArgumentError("a table column is letters only: $(repr(column))"))
        name = _check_macro_name(row * uppercasefirst(column))
        haskey(nums.macros, name) && throw(ArgumentError("\\$name is already one of this driver's numbers"))
        name => _body(value; format...)
    end
    allunique(first.(staged)) || throw(ArgumentError("table row $(repr(row)) repeats a column"))
    foreach(((name, body),) -> _add!(nums, name, body, label, provenance), staged)
    return join(("\\" * first(c) for c in staged), " & ")
end

# The provenance comment of a macro, the line before its \newcommand.
_provenance_comment(name::AbstractString, m::_NumberMacro) =
    "% provenance \\" * name * ": driver=" * m.provenance.driver * "; config=" * m.provenance.config *
    "; run=" * m.provenance.run_key * "; commit=" * m.provenance.commit * (isempty(m.label) ? "" : "; label=" * m.label)

# The provenance comment of each macro of a tex text, as `name => comment line`.
function _macro_provenance(text::AbstractString)
    out = Dict{String, String}()
    for m in eachmatch(r"^%\s*provenance\s+\\([A-Za-z]+):.*$"m, text)
        out[String(_cap(m, 1))] = String(m.match)
    end
    return out
end

"""
    write_numbers(path, nums) -> path

Write `nums` as a number-macro fragment at `path` (a file beside the driver's
results, never in the paper): a header naming the driver, then per macro,
sorted by name, its provenance comment (`% provenance \\<name>: driver=…;
config=…; run=…; commit=…[; label=…]`) and its `\\newcommand{\\<name>}{<body>}`.
The same numbers write the same bytes. [`gather_numbers`](@ref) gathers the
fragments, comments included, into the paper's numbers file;
[`tex_macro_values`](@ref) reads the macros back.
"""
function write_numbers(path::AbstractString, nums::PaperNumbers)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "% generated by ", nums.provenance.driver, " — do not edit; gathered by gather_numbers")
        for name in sort!(collect(keys(nums.macros)))
            m = nums.macros[name]
            println(io, _provenance_comment(name, m))
            println(io, "\\newcommand{\\", name, "}{", m.value, "}")
        end
    end
    return path
end

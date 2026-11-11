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

# A macro body (or a piece of one): one line, braces and math shifts balanced,
# no trailing backslash (it would escape the closing brace), and none of the
# characters that would make it more than one value: an unescaped `%` (the
# rest is a comment), `#` (an argument), `&` (a second table cell), `\\` (a row break).
function _check_tex_text(s::AbstractString, what::AbstractString)
    bad(msg) = throw(ArgumentError("$what $(repr(s)): $msg"))
    isempty(s) && bad("is empty")
    occursin(r"[\n\r]", s) && bad("is not one line")
    occursin("\\\\", s) && bad("holds a row break \\\\")
    depth, shifts, prev = 0, 0, ' '
    for c in s
        if prev != '\\'
            c == '{' && (depth += 1)
            c == '}' && (depth -= 1)
            c == '$' && (shifts += 1)
            depth < 0 && bad("closes a brace it never opened")
            c in ('%', '#', '&') && bad("holds an unescaped $c")
        end
        prev = c
    end
    prev == '\\' && bad("ends in a backslash")
    depth == 0 || bad("leaves a brace open")
    iseven(shifts) || bad("leaves math mode open (an odd number of \$)")
    return String(s)
end

# A label (`fig:…`, `tab:…`): tex text without the provenance comment's separator `;`.
function _check_label(label::AbstractString)
    isempty(label) && return ""
    occursin(';', label) && throw(ArgumentError("label $(repr(label)) holds a `;` (the provenance comment's separator)"))
    return _check_tex_text(label, "label")
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
        (isempty(v) || occursin(r"[\s;]", v)) &&
            throw(ArgumentError("provenance $k $(repr(v)): empty, or holds a space or a `;` (the comment's separator)"))
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
    better::Union{Nothing, Symbol}
    arity::Int                     # 0: a number; 1–9: a macro with parameters (add_macro!)
end

# The direction in which a number improves: `:lower` (an error, a time),
# `:higher` (a rate, a score) or `:none` (a count, a setting), a Symbol or its
# String; `nothing` states none. The results comparison (results_branches.jl)
# reads it from the provenance comment.
const _DIRECTIONS = (:lower, :higher, :none)

const _Direction = Union{Nothing, Symbol, AbstractString}

function _check_better(better)
    better === nothing && return nothing
    b = better isa AbstractString ? Symbol(better) : better
    b isa Symbol && b in _DIRECTIONS ||
        throw(ArgumentError("better = $(repr(better)): a number improves to :lower, :higher or :none"))
    return b
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
    added::Vector{String}          # the names in the order they were added (write_numbers' `order = :added`)
end

PaperNumbers(; driver::AbstractString, config::AbstractString, run_key::AbstractString, commit::AbstractString) =
    PaperNumbers(_provenance((; driver, config, run_key, commit)), Dict{String, _NumberMacro}(), String[])

Base.length(nums::PaperNumbers) = length(nums.macros)

function _add!(nums::PaperNumbers, name::AbstractString, body::AbstractString, label::AbstractString, provenance::NamedTuple,
        better::Union{Nothing, Symbol}, arity::Integer = 0)
    _check_macro_name(name)
    haskey(nums.macros, name) && throw(ArgumentError("\\$name is already one of this driver's numbers"))
    _check_label(label)
    nums.macros[name] = _NumberMacro(body, _provenance(merge(nums.provenance, provenance)), label, _check_better(better), arity)
    push!(nums.added, String(name))
    return nums
end

_body(value::AbstractString; format...) =
    isempty(format) ? _check_tex_text(value, "the value") :
    throw(ArgumentError("a typeset value $(repr(value)) takes no format keywords ($(join(keys(format), ", ")))"))
_body(value::Real; format...) = tex_number(value; format...)
_body(value::Bool; format...) = throw(ArgumentError("$value is a Bool, not a number the paper prints; give its text"))
_body(value::Tuple{Real, Real}; format...) = tex_number(value...; format...)

"""
    add_number!(nums, name, value; label = "", provenance = (;), better = nothing, format...) -> nums

Add the number macro `\\<name>` to `nums`. A `value` that is a number is
typeset by [`tex_number`](@ref) with the `format` keywords (`digits`,
`sigdigits`, `percent`, `group_from`, `unit`); a pair `(lo, hi)` is a range; a
string is its body as typeset (a label, `"---"` for a missing cell, `10^{-6}`)
and takes no format. `label` names what reads the number when it is not the
prose: a figure's `fig:…` (its caption, its axis), a table's `tab:…`. It goes
into the macro's provenance comment. `provenance` overrides fields of the
driver's ([`PaperNumbers`](@ref)) for this number. `better` is the direction
in which the number improves, `:lower` (an error, a time), `:higher` (a rate, a
score) or `:none` (a count, a setting), as a Symbol or a String; it goes into the provenance comment
(`; better=lower`), from which a results comparison
([`results_branch`](@ref)) calls a change IMPROVED or DEGRADED. Without it a
change is reported CHANGED, with no verdict.

`name` is letters only, never beginning with `end` ([`number_macro_name`](@ref)
builds one); a name `nums` already holds, a body of more than one value (an
unescaped `%`, `#` or `&`, a row break, unbalanced braces or math shifts, a
trailing backslash), a `;` in the label or provenance (the comment's
separator), a `Bool` and a NaN are each an `ArgumentError`.
"""
function add_number!(nums::PaperNumbers, name::AbstractString, value::Union{Real, Tuple{Real, Real}, AbstractString};
        label::AbstractString = "", provenance::NamedTuple = (;), better::_Direction = nothing, format...)
    return _add!(nums, name, _body(value; format...), label, provenance, _check_better(better))
end

"""
    add_table_row!(nums, row, cells; label = "", provenance = (;), better = nothing, format...) -> String

Add a table row's numbers to `nums`, one macro per cell: `cells` is an
ordered collection (a vector, a tuple, an `OrderedDict`) of `column => value`
pairs in the table's column order; a `Dict` or a `Set` has no column order and
is an `ArgumentError`. The
cell's macro is `\\<row><Column>` (`row` and each `column` letters only, the
column capitalised). Each value is typeset as by [`add_number!`](@ref), the
`format` keywords shared by the row; `better` is one direction for every cell,
or a collection of `column => direction` pairs for the cells that state one (a
column the row lacks is an `ArgumentError`); a cell of another format is given
as its [`tex_number`](@ref) string. Returns the row as the table's tex reads it,
`"\\<row><A> & \\<row><B> & …"`, without the row break.

One macro per cell, not one per row: the prose cites a table's number by the
macro the table prints, and the gatherer's checks
([`compare_fragments`](@ref), [`macro_literal_violations`](@ref),
[`uncited_macros`](@ref)) see each number.
"""
function add_table_row!(nums::PaperNumbers, row::AbstractString, cells;
        label::AbstractString = "", provenance::NamedTuple = (;), better = nothing, format...)
    _check_macro_name(row)
    cells isa Union{Dict, IdDict, WeakKeyDict, Set} &&             # unordered; an OrderedDict keeps its order
        throw(ArgumentError("table row $(repr(row)): a $(nameof(typeof(cells))) has no column order; give the cells as an ordered vector or tuple of column => value pairs"))
    isempty(cells) && throw(ArgumentError("table row $(repr(row)) has no cells"))
    _check_label(label)
    _provenance(merge(nums.provenance, provenance))
    directions = better isa _Direction ? nothing :
        Dict(string(first(p)) => _check_better(last(p)) for p in better if p isa Pair || throw(ArgumentError(
            "table row $(repr(row)): better is a direction or column => direction pairs, not $(repr(better))")))
    staged = map(collect(cells)) do (col, value)       # every cell checked before any is added
        column = string(col)
        occursin(r"^[A-Za-z]+$", column) || throw(ArgumentError("a table column is letters only: $(repr(column))"))
        name = _check_macro_name(row * uppercasefirst(column))
        haskey(nums.macros, name) && throw(ArgumentError("\\$name is already one of this driver's numbers"))
        (name, _body(value; format...), directions === nothing ? _check_better(better) : get(directions, column, nothing))
    end
    allunique(first.(staged)) || throw(ArgumentError("table row $(repr(row)) repeats a column"))
    directions === nothing || issubset(keys(directions), Set(string(c) for (c, _) in cells)) ||
        throw(ArgumentError("table row $(repr(row)): better names a column the row does not have"))
    foreach(((name, body, b),) -> _add!(nums, name, body, label, provenance, b), staged)
    return join(("\\" * first(c) for c in staged), " & ")
end

"""
    add_macro!(nums, name, body; arity, label = "", provenance = (;)) -> nums

Add the macro `\\<name>` with `arity` parameters (1 to 9) to `nums`: its
`body` is TeX in which `#1` … `#<arity>` stand for the arguments (a figure's
equal-height panel cell, `\\parbox[c][\\panelht][c]{#1}{\\centering #2}`). Written
by [`write_numbers`](@ref) as `\\newcommand{\\<name>}[<arity>]{<body>}` after its
provenance comment, gathered by [`gather_numbers`](@ref) as it is; it is not a
value, so [`tex_macro_values`](@ref) does not return it. A parameter beyond the
arity, an arity outside 1–9, unbalanced braces, an unescaped `%` and a name
`nums` already holds are each an `ArgumentError`.
"""
function add_macro!(nums::PaperNumbers, name::AbstractString, body::AbstractString; arity::Integer,
        label::AbstractString = "", provenance::NamedTuple = (;))
    1 <= arity <= 9 || throw(ArgumentError("\\$name: a macro takes 1 to 9 parameters, not $arity"))
    used = [parse(Int, _cap(m, 1)) for m in eachmatch(r"(?<!\\)#(\d)", body)]
    all(k -> 1 <= k <= arity, used) ||
        throw(ArgumentError("\\$name: its body reads parameter(s) $(sort(unique(used))) of an arity-$arity macro"))
    _check_tex_text(replace(body, r"(?<!\\)#\d" => "x"), "\\$name's body")
    return _add!(nums, name, body, label, provenance, nothing, arity)
end

# The one writer of a macro definition line, `\newcommand{\<name>}[<arity>]{<body>}`
# (`write_numbers` and `gather_numbers` both write through it; no `[0]` for a value).
_write_macro(io::IO, name::AbstractString, body, arity::Integer = 0) =
    println(io, "\\newcommand{\\", name, "}", arity > 0 ? "[$arity]" : "", "{", body, "}")

# The provenance comment of a macro, the line before its \newcommand.
_provenance_comment(name::AbstractString, m::_NumberMacro) =
    "% provenance \\" * name * ": driver=" * m.provenance.driver * "; config=" * m.provenance.config *
    "; run=" * m.provenance.run_key * "; commit=" * m.provenance.commit * (isempty(m.label) ? "" : "; label=" * m.label) *
    (m.better === nothing ? "" : "; better=" * String(m.better))

# A provenance comment line, `% provenance \<name>: …`, capturing the name.
const _PROVENANCE_LINE = r"^%\s*provenance\s+\\([A-Za-z]+):"

# The provenance comment of each macro of a tex text, as `name => comment line`.
function _macro_provenance(text::AbstractString)
    out = Dict{String, String}()
    for line in eachsplit(text, '\n')
        m = match(_PROVENANCE_LINE, line)
        m === nothing || (out[String(_cap(m, 1))] = String(line))
    end
    return out
end

# A tex text without its provenance comment lines: what must be byte-equal for
# two numbers files to differ in their provenance alone.
_without_provenance(text::AbstractString) = [l for l in eachsplit(text, '\n') if !occursin(_PROVENANCE_LINE, l)]

# The definitions a numbers file of this writer holds, one per line: `name => (arity, body)`.
const _DEFINITION_LINE = r"^\\newcommand\{\\([A-Za-z]+)\}(?:\[(\d)\])?\{(.*)\}$"
_macro_definitions(text::AbstractString) =
    [String(_cap(m, 1)) => (isempty(_cap(m, 2)) ? 0 : parse(Int, _cap(m, 2)), String(_cap(m, 3)))
     for m in (match(_DEFINITION_LINE, l) for l in eachsplit(text, '\n')) if m !== nothing]

# A driver's section of a numbers file: its header, then per macro its provenance comment and definition.
_header(driver) = "% generated by " * driver * " — do not edit; gathered by gather_numbers"
const _HEADER_LINE = r"^% generated by (\S+) — do not edit; gathered by gather_numbers$"


# The sections of a shared numbers file, `driver => text`, each from its header line to the next.
function _sections(text::AbstractString, path)
    out = Pair{String, String}[]
    cur, buf = nothing, IOBuffer()
    for line in eachsplit(text, '\n'; keepempty = true)
        m = match(_HEADER_LINE, line)
        if m !== nothing
            cur === nothing || push!(out, cur => String(take!(buf)))
            cur = String(_cap(m, 1))
        elseif cur === nothing
            isempty(line) || throw(ArgumentError("write_numbers: $path holds text before its first driver's header"))
            continue
        end
        isempty(line) && continue
        println(buf, line)
    end
    cur === nothing || push!(out, cur => String(take!(buf)))
    allunique(first.(out)) || throw(ArgumentError("write_numbers: $path holds a driver's section twice"))
    return out
end

"""
    write_numbers(io::IO, nums; order = :name) -> io

Write `nums` to `io` exactly as [`write_numbers`](@ref)`(path, nums; order)`
writes its file (one implementation): the header naming the driver, then per
macro its provenance comment and its `\\newcommand`, sorted by name or in the
order they were added. The macros a figure defines in its own body (an
[`add_macro!`](@ref) cell) are inlined this way:
`sprint(io -> write_numbers(io, nums; order = :added))`. A file several
drivers share is the path form's (`shared = true`).
"""
function write_numbers(io::IO, nums::PaperNumbers; order::Symbol = :name)
    order in (:name, :added) || throw(ArgumentError("write_numbers: order is :name or :added, not $(repr(order))"))
    println(io, _header(nums.provenance.driver))
    for name in (order === :name ? sort(nums.added) : nums.added)
        m = nums.macros[name]
        println(io, _provenance_comment(name, m))
        _write_macro(io, name, m.value, m.arity)
    end
    return io
end

"""
    write_numbers(path, nums; shared = false, order = :name) -> path

Write `nums` as a number-macro fragment at `path` (a file beside the driver's
results, never in the paper): a header naming the driver, then per macro its
provenance comment (`% provenance \\<name>: driver=…; config=…; run=…;
commit=…[; label=…][; better=…]`) and its `\\newcommand{\\<name>}{<body>}`
(`[<arity>]` for a macro with parameters, [`add_macro!`](@ref)), sorted by name
(`order = :name`) or in the order they were added (`order = :added`). The same
numbers write the same bytes; the file is written whole (a temporary file
renamed into place).

`shared = true`: `path` is ONE file several drivers write (CaliBlobs icra2026's
trajectory macros). Each driver's entries are its own section, under its
header; this call rewrites the driver's section alone and keeps every other
section as it is, the sections ordered by driver, so the file is the same
whatever order the drivers ran in. A macro another section defines is an
`ArgumentError` (nothing is written). [`gather_numbers`](@ref) gathers the
fragments, comments included, into the paper's numbers file;
[`tex_macro_values`](@ref) reads the macros back.
"""
function write_numbers(path::AbstractString, nums::PaperNumbers; shared::Bool = false, order::Symbol = :name)
    mine = sprint(io -> write_numbers(io, nums; order))
    text = mine
    if shared && isfile(path)
        sections = filter(p -> first(p) != nums.provenance.driver, _sections(read(path, String), path))
        for (driver, other) in sections, (name, _) in _macro_definitions(other)
            haskey(nums.macros, name) && throw(ArgumentError(
                "write_numbers: \\$name of $(nums.provenance.driver) is defined by $driver's section of $path"))
        end
        push!(sections, nums.provenance.driver => mine)
        text = join(last.(sort!(sections; by = first)))
    end
    isdir(path) && throw(ArgumentError("write_numbers: $path is a directory"))   # mv(…; force) would delete it
    _write_atomic(io -> write(io, text), abspath(path))   # written whole, then renamed: never a fragment
    return path
end

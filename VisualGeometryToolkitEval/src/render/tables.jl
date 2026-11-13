# =============================================================================
# render/tables.jl — the plain-text table helpers of the report cells: the
# three-decimal number cell, and the tab-separated table file with its
# `#` comment lines (written and read back).
# =============================================================================

"""
    fmt3(x; nonfinite = nothing) -> String

A number cell: `x` with three decimals (`"%.3f"`). A non-finite `x` reads as
Julia's own spelling (`"NaN"`, `"Inf"`, `"-Inf"`) by default, or as
`nonfinite` (a `String`) when one is given: a report that collapses every
non-finite value to one text gives it as `nonfinite`.
"""
function fmt3(x::Real; nonfinite::Union{Nothing,AbstractString} = nothing)
    isfinite(x) && return @sprintf("%.3f", x)
    return nonfinite === nothing ? string(x) : String(nonfinite)
end

"""
    write_tsv(path, header, rows; comments = String[]) -> path

Write a tab-separated table: each of `comments` as a `# `-prefixed line, then
the `header` cells, then one line per row of `rows` (each cell through
`string`), lines ended by `\\n`. The parent directory is created.
[`read_tsv`](@ref) reads it back.
"""
function write_tsv(path::AbstractString, header, rows; comments = String[])
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        for c in comments
            println(io, "# ", c)
        end
        println(io, join(string.(header), '\t'))
        for r in rows
            println(io, join(string.(r), '\t'))
        end
    end
    return path
end

"""
    write_tsv(path, rows::AbstractVector{<:NamedTuple}; comments = String[]) -> path

The same table with its header the rows' field names, in their order, and one
line per row. Every row has the same field names (an `ArgumentError` names the
first row that differs). The element type may be abstract (`NamedTuple[]`, a
table built by `push!`): the header is read from the rows. A table without rows
writes the header when its element type names the fields
(`Vector{@NamedTuple{a::Int}}`), else an empty header line; either reads back
([`read_tsv`](@ref)) as no rows.
"""
function write_tsv(path::AbstractString, rows::AbstractVector{<:NamedTuple}; comments = String[])
    if isempty(rows)    # the header from the element type when it names the fields; else an empty header line
        return write_tsv(path, isconcretetype(eltype(rows)) ? fieldnames(eltype(rows)) : (), rows; comments)
    end
    header = keys(first(rows))
    i = findfirst(r -> keys(r) != header, rows)
    i === nothing || throw(ArgumentError("write_tsv: row $i has the fields $(keys(rows[i])), the first row $(header): one header per table"))
    return write_tsv(path, header, (values(r) for r in rows); comments)
end

"""
    read_tsv(path) -> Vector{Dict{String,String}}

The rows of a tab-separated table as dictionaries keyed by the header cells.
Blank lines and `#` comment lines are skipped; the first remaining line is the
header, and cells are strings (a row shorter than the header lacks the
trailing keys).
"""
function read_tsv(path::AbstractString)
    lines = filter(l -> !isempty(strip(l)) && !startswith(l, "#"), readlines(path))
    isempty(lines) && return Dict{String,String}[]           # no header: an empty, untyped table (`write_tsv`)
    hdr = split(lines[1], '\t')
    return [Dict(String(h) => String(v) for (h, v) in zip(hdr, split(l, '\t'))) for l in lines[2:end]]
end

# =============================================================================
# render/tables.jl — the plain-text table helpers of the report cells: the
# three-decimal number cell, and the tab-separated table file with its
# `#` comment lines (written and read back).
# =============================================================================

"""
    fmt3(x; nonfinite = "NaN") -> String

A number cell: `x` with three decimals (`"%.3f"`). A non-finite `x` reads as
`nonfinite` (a `String`), or as Julia's own spelling (`"NaN"`, `"Inf"`,
`"-Inf"`) when `nonfinite = nothing`. A report whose reader distinguishes the
infinities passes `nothing`; one that collapses them to a missing cell passes
its own text.
"""
function fmt3(x::Real; nonfinite::Union{Nothing,AbstractString} = "NaN")
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
    read_tsv(path) -> Vector{Dict{String,String}}

The rows of a tab-separated table as dictionaries keyed by the header cells.
Blank lines and `#` comment lines are skipped; the first remaining line is the
header, and cells are strings (a row shorter than the header lacks the
trailing keys).
"""
function read_tsv(path::AbstractString)
    lines = filter(l -> !isempty(strip(l)) && !startswith(l, "#"), readlines(path))
    hdr = split(lines[1], '\t')
    return [Dict(String(h) => String(v) for (h, v) in zip(hdr, split(l, '\t'))) for l in lines[2:end]]
end

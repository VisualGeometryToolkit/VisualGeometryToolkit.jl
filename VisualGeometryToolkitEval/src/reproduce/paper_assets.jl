# =============================================================================
# paper_assets.jl — the last stages of a paper's regeneration: VERIFY each
# regenerated asset against the submitted copy the paper reads, DEPLOY it into
# the paper only when that changes nothing or the author signed off, GATHER
# the drivers' number-macro fragments into the paper's numbers file, and check
# that the paper cites every number it defines.
#
# A generator renders into its results (never into the paper). The rule is one
# for every asset — a figure, a pgfplots fragment, the numbers file: a
# submitted asset is never silently replaced. `deploy_assets` copies a render
# when the paper lacks it or holds the identical bytes, and refuses a
# different one unless `replace_submitted = true`, naming the verify report it
# has just written.
#
# Dependency-free by contract (Base only), like reproduce_index.jl, which it
# follows (it reads macros with `_macro_values` and walks the paper with
# `paper_asset_graph`).
# =============================================================================

# ---------------------------------------------------------------------------
# Comparing a regenerated asset with the submitted one
# ---------------------------------------------------------------------------

"""
    fragment_coordinates(text) -> Vector{Vector{NTuple{2,Float64}}}

Every `coordinates {…}` block of a pgfplots fragment, in order, parsed to
numbers: the quantitative content of a panel, for comparing a regenerated
fragment with the submitted one independently of its formatting.
"""
function fragment_coordinates(text::AbstractString)
    blocks = Vector{NTuple{2, Float64}}[]
    for m in eachmatch(r"coordinates\s*\{([^}]*)\}", text)
        pts = NTuple{2, Float64}[]
        for p in eachmatch(r"\(\s*([^,()\s]+)\s*,\s*([^,()\s]+)\s*\)", _cap(m, 1))
            push!(pts, (parse(Float64, _cap(p, 1)), parse(Float64, _cap(p, 2))))
        end
        push!(blocks, pts)
    end
    return blocks
end

"""
    compare_fragments(new, old) -> NamedTuple

How a regenerated tex fragment `new` differs from the submitted `old` (texts):
`identical` (byte for byte); `same_numbers`, every `coordinates` block equal in
order ([`fragment_coordinates`](@ref)) and every argument-free macro
definition equal (a numbers file); the block counts `blocks_new` and
`blocks_old`; and `max_abs_diff`, the largest absolute coordinate difference
over blocks of equal length (`Inf` when the block structure differs).
"""
function compare_fragments(new::AbstractString, old::AbstractString)
    a, b = fragment_coordinates(new), fragment_coordinates(old)
    same_shape = length(a) == length(b) && all(length.(a) .== length.(b))
    maxdiff = same_shape ?
        maximum((abs(p[i] - q[i]) for (x, y) in zip(a, b) for (p, q) in zip(x, y) for i in 1:2); init = 0.0) : Inf
    same_macros = sort(_macro_values(new)) == sort(_macro_values(old))
    return (identical = new == old, same_numbers = same_shape && a == b && same_macros,
            blocks_new = length(a), blocks_old = length(b), max_abs_diff = maxdiff)
end

# ---------------------------------------------------------------------------
# Verify and deploy
# ---------------------------------------------------------------------------

"""
    VERIFY_REPORT

The file name of the verify report [`verify_assets`](@ref) writes beside the
renders: `"verify.tsv"`.
"""
const VERIFY_REPORT = "verify.tsv"

_is_tex(name::AbstractString) = lowercase(splitext(name)[2]) == ".tex"

function _verify_row(outdir::AbstractString, paperdir::AbstractString, name::AbstractString)
    new = joinpath(outdir, name)
    isfile(new) || error("verify: no render $new; run the stage that generates it")
    old = joinpath(paperdir, name)
    submitted = isfile(old)
    bytes = (bytes_new = filesize(new), bytes_old = submitted ? filesize(old) : 0)
    if !submitted
        nblocks = _is_tex(name) ? length(fragment_coordinates(read(new, String))) : 0
        return (; asset = String(name), submitted, identical = false, same_numbers = false,
                blocks_new = nblocks, blocks_old = 0, max_abs_diff = NaN, bytes...)
    elseif _is_tex(name)
        return (; asset = String(name), submitted, compare_fragments(read(new, String), read(old, String))..., bytes...)
    else   # an image or any other binary asset: its bytes are its content
        identical = read(new) == read(old)
        return (; asset = String(name), submitted, identical, same_numbers = identical,
                blocks_new = 0, blocks_old = 0, max_abs_diff = identical ? 0.0 : NaN, bytes...)
    end
end

_verdict(r) = !r.submitted ? "NEW (no submitted copy)" : r.identical ? "identical" :
    r.same_numbers ? "same numbers, formatting differs" :
    r.blocks_new + r.blocks_old > 0 ? "DIFFERS (max |Δ| = $(r.max_abs_diff), blocks $(r.blocks_new) vs $(r.blocks_old))" :
    "DIFFERS ($(r.bytes_new) vs $(r.bytes_old) bytes)"

"""
    verify_assets(outdir, paperdir, names) -> Vector{NamedTuple}

Compare each rendered `outdir/<name>` with the submitted `paperdir/<name>` and
write the report `outdir/verify.tsv` ([`VERIFY_REPORT`](@ref)): one row per
asset with `asset`, `submitted`, `identical`, `same_numbers`, `blocks_new`,
`blocks_old`, `max_abs_diff`, `bytes_new` and `bytes_old`. A `.tex` asset is
compared by its numbers as well as its bytes ([`compare_fragments`](@ref)); any
other asset by its bytes. Each verdict is printed. A missing render is an
error; a missing submitted copy is reported (`submitted = false`), not an
error. Returns the rows.
"""
function verify_assets(outdir::AbstractString, paperdir::AbstractString, names)
    rows = [_verify_row(outdir, paperdir, n) for n in names]
    cols = (:asset, :submitted, :identical, :same_numbers, :blocks_new, :blocks_old, :max_abs_diff, :bytes_new, :bytes_old)
    open(joinpath(outdir, VERIFY_REPORT), "w") do io
        println(io, join(cols, '\t'))
        for r in rows
            println(io, join((getfield(r, c) for c in cols), '\t'))
        end
    end
    width = maximum(r -> length(r.asset), rows; init = 0)
    for r in rows
        println("   ", rpad(r.asset, width), "  ", _verdict(r))
    end
    return rows
end

"""
    assets_deployed(outdir, paperdir, names) -> Bool

Whether every rendered `outdir/<name>` is byte-identical to the paper's
`paperdir/<name>`: a cheap done predicate of a deploy stage.
"""
assets_deployed(outdir::AbstractString, paperdir::AbstractString, names) =
    all(n -> isfile(joinpath(outdir, n)) && isfile(joinpath(paperdir, n)) &&
        read(joinpath(outdir, n)) == read(joinpath(paperdir, n)), names)

"""
    deploy_assets(outdir, paperdir, names; replace_submitted = false) -> Vector{String}

Copy each rendered `outdir/<name>` into `paperdir` when the paper has no copy
or holds a different one, after verifying them all ([`verify_assets`](@ref)
writes a fresh `outdir/verify.tsv` first). A submitted copy that differs is
replaced only with `replace_submitted = true`, the author's sign-off on the
difference the report states; without it the deploy is refused, as an error
naming each differing asset and the report, and copies nothing. Returns the
assets copied (an identical one is not copied).
"""
function deploy_assets(outdir::AbstractString, paperdir::AbstractString, names; replace_submitted::Bool = false)
    rows = verify_assets(outdir, paperdir, names)
    report = joinpath(outdir, VERIFY_REPORT)
    refused = [r for r in rows if r.submitted && !r.identical]
    if !replace_submitted && !isempty(refused)
        error("deploy: the regenerated asset(s) differ from the submitted one(s) in $paperdir:\n" *
              join(("  " * r.asset * ": " * _verdict(r) for r in refused), "\n") *
              "\nsee $report. Deploy with replace_submitted = true once the difference is signed off.")
    end
    deployed = String[]
    for r in rows
        r.identical && continue
        mkpath(paperdir)
        cp(joinpath(outdir, r.asset), joinpath(paperdir, r.asset); force = true)
        push!(deployed, r.asset)
        println("   deployed ", r.asset)
    end
    return deployed
end

# ---------------------------------------------------------------------------
# The numbers file
# ---------------------------------------------------------------------------

# The macros a tex text defines WITH arguments (`\newcommand{\x}[1]{…}`, `\def\x#1{…}`), comments ignored.
function _macros_with_arguments(source::AbstractString)
    text = _strip_comments(source)
    out = String[]
    for m in eachmatch(r"\\(?:re|provide)?newcommand\*?\s*\{?\\([A-Za-z@]+)\}?\s*\[\d\]", text)
        push!(out, String(_cap(m, 1)))
    end
    for m in eachmatch(r"\\def\s*\\([A-Za-z@]+)\s*#", text)
        push!(out, String(_cap(m, 1)))
    end
    return out
end

"""
    gather_numbers(fragments, out; generator, root, root_label = "<results_root()>") -> out

Write the paper's numbers file `out` from the generators' number-macro
`fragments` (tex files): every argument-free macro of every fragment, one
`\\newcommand{\\<name>}{<value>}` per line, sorted by name, under a header
naming the `generator` and each fragment as `<root_label>/<path relative to
root>` (no machine path). Never `\\providecommand`: a macro the manuscript
also defines must clash. A macro that two fragments define, and a fragment
macro with arguments (a number is a value, not a template), are an
`ArgumentError`.

Write `out` beside the generator's results, not into the paper: a submitted
numbers file is replaced under the same sign-off as a figure, by
[`deploy_assets`](@ref).
"""
function gather_numbers(fragments, out::AbstractString; generator::AbstractString, root::AbstractString,
        root_label::AbstractString = "<results_root()>")
    numbers = Dict{String, String}()
    owner = Dict{String, String}()
    for f in fragments
        text = read(f, String)
        args = _macros_with_arguments(text)
        isempty(args) || throw(ArgumentError("$f defines macros with arguments (" * join("\\" .* args, ", ") *
            "): a number fragment defines values only"))
        for (name, value) in _macro_values(text)
            haskey(owner, name) && throw(ArgumentError("\\$name is defined by both $(owner[name]) and $f"))
            numbers[name], owner[name] = value, f
        end
    end
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "% generated by ", generator, " — do not edit; gathered from:")
        for f in fragments
            println(io, "%   ", root_label, "/", relpath(f, root))
        end
        for name in sort!(collect(keys(numbers)))
            println(io, "\\newcommand{\\", name, "}{", numbers[name], "}")
        end
    end
    return out
end

"""
    uncited_macros(numbers_path, papers; root = dirname(first(papers)), search_dirs = String[],
                   allow = Dict{String,String}()) -> Vector{String}

The macros the numbers file `numbers_path` defines that the paper does not
cite: the tex files the paper reads are the ones [`paper_asset_graph`](@ref)
visits from the top-level `papers` (with `root` and `search_dirs` as there),
comments ignored, the numbers file itself excluded; a citation is the macro
name not followed by a letter (`\\nA` is not cited by `\\nAb`). The `allow`
table (`macro name without backslash => reason`) exempts a macro; an `allow`
entry naming no macro of the file is returned as `stale allow: <name>`. Sorted.
A paper that does not read `numbers_path` is an `ArgumentError`.
"""
function uncited_macros(numbers_path::AbstractString, papers::AbstractVector{<:AbstractString};
        root::AbstractString = dirname(abspath(first(papers))),
        search_dirs::AbstractVector{<:AbstractString} = String[], allow::AbstractDict = Dict{String, String}())
    graph = paper_asset_graph(papers; root, search_dirs)
    files = [joinpath(graph.root, f) for f in graph.tex_files]
    numbers = abspath(numbers_path)
    numbers in abspath.(files) ||
        throw(ArgumentError("the paper ($(join(papers, ", "))) does not read $numbers_path"))
    text = join((_strip_comments(read(f, String)) for f in files if abspath(f) != numbers), '\n')
    names = first.(tex_macro_values(numbers_path))
    out = [n for n in names if !haskey(allow, n) && !occursin(Regex("\\\\" * n * "(?![A-Za-z@])"), text)]
    append!(out, ["stale allow: " * String(a) for a in keys(allow) if !(a in names)])
    return sort!(out)
end

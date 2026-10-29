# =============================================================================
# reproduce_index.jl — the reproducible-figures contract of a paper project.
#
# Policy (every camera-ready paper): every figure, table and number macro the
# paper reads is produced by tracked code from declared data, indexed in the
# project's REPRODUCE.md (label → asset → kind → generator → data), and the
# project's suite enforces it by calling `reproduce_index_violations`. A
# hand-made asset is allowed only when the index declares it `static` with a
# provenance note.
#
# Dependency-free by contract (Base only): an environment that cannot load
# VisualGeometryToolkitEval includes this file by path, like experiments/stages.jl.
# =============================================================================

"""
    PaperAsset

One file the paper reads: `path` (relative to the project directory), `kind`
(`:tex` for `\\input`/`\\include`, `:graphics` for `\\includegraphics`, `:data`
for a pgfplots/tikz data file), the referencing `source` file and `line`, the
`\\label`s of the enclosing float (inherited through `\\input`), and whether
the file exists.
"""
struct PaperAsset
    path::String
    kind::Symbol
    source::String
    line::Int
    labels::Vector{String}
    exists::Bool
end

"""
    PaperAssetGraph

What a set of paper sources reads: the project directory `root`, the visited
tex files (project-relative), every [`PaperAsset`](@ref), every `\\label` key,
and the references whose target could not be resolved statically (a macro or
`#1` in the path) as `(source, line, text)`.
"""
struct PaperAssetGraph
    root::String
    tex_files::Vector{String}
    assets::Vector{PaperAsset}
    labels::Set{String}
    unresolved::Vector{Tuple{String, Int, String}}
end

"""
    IndexEntry

One row of a REPRODUCE.md asset index: the paper `label` it serves, the
`asset` (project-relative path or glob), its `kind` (`:generated`, `:static`,
or `:invalid` for an unrecognised kind cell), the `generators` (project-relative
paths), the `configs` of record they read (project-relative; from an optional
`config` column), the `data` it is generated from, the `provenance` note
(required for `:static`), and the index `line`.
"""
struct IndexEntry
    label::String
    asset::String
    kind::Symbol
    generators::Vector{String}
    configs::Vector{String}
    data::String
    provenance::String
    line::Int
end

"""
    ReproduceViolation

One breach of the reproducible-figures contract: a `kind` (see
[`reproduce_index_violations`](@ref)), the `subject` (an asset, generator or
macro), a human-readable `message` and a `location` (`file:line`).
"""
struct ReproduceViolation
    kind::Symbol
    subject::String
    message::String
    location::String
end

Base.show(io::IO, v::ReproduceViolation) =
    print(io, v.location, ": [", v.kind, "] ", v.subject, " — ", v.message)

# ---------------------------------------------------------------------------
# TeX scanning
# ---------------------------------------------------------------------------

const _GRAPHICS_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".PNG", ".JPG")
const _FLOAT_ENVS = r"^(figure|table|wrapfigure|wraptable|subfigure|subtable)\*?$"

# Remove comments (an unescaped `%` to end of line), keeping line structure.
function _strip_comments(text::AbstractString)
    out = IOBuffer()
    for (i, line) in enumerate(split(text, '\n'))
        i > 1 && write(out, '\n')
        cut = lastindex(line) + 1
        prev = ' '
        for (j, c) in pairs(line)
            if c == '%' && prev != '\\'
                cut = j
                break
            end
            prev = c
        end
        write(out, SubString(line, 1, prevind(line, cut)))
    end
    return String(take!(out))
end

# A regex capture as a string ("" for an unmatched optional group): keeps the
# Union{Nothing, SubString} of `captures` out of every caller.
_cap(m::RegexMatch, i::Integer) = something(m.captures[i], "")

_line_of(text::AbstractString, idx::Integer) = count(==('\n'), SubString(text, 1, prevind(text, idx))) + 1

# The brace-balanced argument starting at text[i] == '{'; returns (arg, index after).
function _braced(text::AbstractString, i::Integer)
    text[i] == '{' || return nothing
    depth, j = 0, i
    while j <= lastindex(text)
        c = text[j]
        c == '{' && (depth += 1)
        if c == '}'
            depth -= 1
            depth == 0 && return (SubString(text, nextind(text, i), prevind(text, j)), nextind(text, j))
        end
        j = nextind(text, j)
    end
    return nothing
end

# The float spans of a (comment-stripped) text, as (start, stop, labels).
function _float_spans(text::AbstractString)
    spans = Tuple{Int, Int, Vector{String}}[]
    stack = Tuple{String, Int}[]
    for m in eachmatch(r"\\(begin|end)\{([^}]+)\}", text)
        env = _cap(m, 2)
        occursin(_FLOAT_ENVS, env) || continue
        if _cap(m, 1) == "begin"
            push!(stack, (env, m.offset))
        elseif !isempty(stack) && first(last(stack)) == env
            _, start = pop!(stack)
            stop = m.offset
            labels = [String(_cap(l, 1)) for l in eachmatch(r"\\label\{([^}]+)\}", SubString(text, start, stop))]
            push!(spans, (start, stop, labels))
        end
    end
    return spans
end

function _labels_at(spans, idx::Integer)
    out = String[]
    for (a, b, labels) in spans
        a <= idx <= b && append!(out, labels)
    end
    return out
end

_is_static_path(s::AbstractString) = !occursin('\\', s) && !occursin('#', s) && !isempty(strip(s))

# A pgfplots `table {…}` argument names a file only when it looks like a path
# (inline data has whitespace or row separators).
_looks_like_data_file(s::AbstractString) =
    _is_static_path(s) && !occursin(r"\s", strip(s)) && occursin(r"(\.[A-Za-z0-9]+$)|/", strip(s))

function _resolve(base::AbstractString, ref::AbstractString, exts, search_dirs)
    ref = strip(ref)
    cands = String[]
    for d in ("", search_dirs...)
        p = normpath(joinpath(base, d, ref))
        push!(cands, p)
        isempty(splitext(ref)[2]) && append!(cands, p .* collect(exts))
    end
    for c in cands
        isfile(c) && return (c, true)
    end
    return (first(cands) * (isempty(splitext(ref)[2]) ? first(exts) : ""), false)
end

"""
    paper_asset_graph(tex_files; root = dirname(first(tex_files)), search_dirs = String[]) -> PaperAssetGraph

Every file a paper reads, found by following `\\input`/`\\include`/
`\\subfile`/`\\includestandalone` (recursively), `\\includegraphics` (with the
`\\graphicspath` directories and the default extensions tried in LaTeX's
order), and pgfplots/tikz data files (`\\addplot table {file}`,
`\\pgfplotstableread{file}`, `\\pgfplotstabletypeset{file}`). Paths resolve
against the directory of the top-level file, then against each of
`search_dirs` in order (the compile directory and the TEXINPUTS of a build run
elsewhere, e.g. `["paper"]` for a `main.tex` in `paper/src/` built from
`paper/` that writes `\\input{src/sec/intro}`; a relative entry is relative to
`root`), and are reported relative to `root` (the project directory). Graphics
try the `\\graphicspath` directories before `search_dirs`. Comments are ignored; a
reference whose path holds a macro or a `#` parameter is recorded in
`unresolved`, not guessed.
"""
function paper_asset_graph(tex_files::AbstractVector{<:AbstractString};
        root::AbstractString = dirname(abspath(first(tex_files))),
        search_dirs::AbstractVector{<:AbstractString} = String[])
    root = abspath(root)
    search = String[abspath(root, d) for d in search_dirs]
    rel(p) = relpath(abspath(p), root)
    assets = PaperAsset[]
    visited = String[]
    labels = Set{String}()
    unresolved = Tuple{String, Int, String}[]
    for top in tex_files
        top = abspath(top)
        base = dirname(top)
        # \graphicspath is a preamble setting in practice: once seen, it
        # applies to every later \includegraphics of this document
        graphicspath = String[]
        _walk_tex(top, base, String[], visited, assets, labels, unresolved, graphicspath, search, rel)
    end
    # A file reached from several places is one asset per reference; keep them
    # all (the locations matter for messages) but dedupe exact repeats.
    unique!(a -> (a.path, a.source, a.line), assets)
    return PaperAssetGraph(root, unique!(visited), assets, labels, unresolved)
end

function _walk_tex(file, base, inherited, visited, assets, labels, unresolved, graphicspath, search, rel)
    rel(file) in visited && return nothing
    push!(visited, rel(file))
    text = _strip_comments(read(file, String))
    for m in eachmatch(r"\\label\{([^}]+)\}", text)
        push!(labels, String(_cap(m, 1)))
    end
    for m in eachmatch(r"\\graphicspath\s*\{", text)
        arg = _braced(text, m.offset + length(m.match) - 1)
        arg === nothing && continue
        for g in eachmatch(r"\{([^}]*)\}", first(arg))
            push!(graphicspath, String(_cap(g, 1)))
        end
    end
    spans = _float_spans(text)
    src = rel(file)
    labs(idx) = unique!(vcat(inherited, _labels_at(spans, idx)))
    # \input / \include / \subfile / \includestandalone
    for m in eachmatch(r"\\(input|include|subfile|includestandalone)\s*(\[[^\]]*\])?\s*\{([^}]*)\}", text)
        ref, line = _cap(m, 3), _line_of(text, m.offset)
        if !_is_static_path(ref)
            push!(unresolved, (src, line, String(m.match)))
            continue
        end
        exts = _cap(m, 1) == "includestandalone" ? (".tex", ".pdf") : (".tex",)
        path, ok = _resolve(base, ref, exts, search)
        push!(assets, PaperAsset(rel(path), :tex, src, line, labs(m.offset), ok))
        ok && endswith(path, ".tex") &&
            _walk_tex(path, base, labs(m.offset), visited, assets, labels, unresolved, graphicspath, search, rel)
    end
    # \includegraphics
    for m in eachmatch(r"\\includegraphics\*?\s*(\[[^\]]*\])?\s*\{([^}]*)\}", text)
        ref, line = _cap(m, 2), _line_of(text, m.offset)
        if !_is_static_path(ref)
            push!(unresolved, (src, line, String(m.match)))
            continue
        end
        path, ok = _resolve(base, ref, _GRAPHICS_EXTS, vcat(graphicspath, search))
        push!(assets, PaperAsset(rel(path), :graphics, src, line, labs(m.offset), ok))
    end
    # pgfplots / tikz data files
    for re in (
            r"\\addplot3?\+?\s*(?:\[[^\]]*\])?\s*table\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}",
            r"\\pgfplotstable(?:read|typeset)\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}",
        )
        for m in eachmatch(re, text)
            ref = _cap(m, 1)
            _looks_like_data_file(ref) || continue
            path, ok = _resolve(base, ref, ("",), search)
            push!(assets, PaperAsset(rel(path), :data, src, _line_of(text, m.offset), labs(m.offset), ok))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

function _cells(line::AbstractString)
    s = strip(line)
    startswith(s, '|') && (s = s[2:end])
    endswith(s, '|') && (s = s[1:(end - 1)])
    return [strip(replace(c, "\\|" => "|")) for c in split(s, r"(?<!\\)\|")]
end

_is_separator(line) = occursin(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", line)

_backticked(cell) = [String(_cap(m, 1)) for m in eachmatch(r"`([^`]+)`", cell)]

_is_dash(cell) = strip(cell) in ("", "-", "—", "–", "n/a", "none")

"""
    read_reproduce_index(path) -> Vector{IndexEntry}

The asset index of a REPRODUCE.md: every markdown table whose header has the
columns `asset`, `kind` and `generator` (case-insensitive; `label`, `config`,
`data` and `provenance` are read when present, any other column is ignored). A cell's
backticked tokens are its values (several assets in one row expand to one
entry each); a row without backticks uses the cell text. `kind` is the first
word of its cell: `generated` or `static`.
"""
function read_reproduce_index(path::AbstractString)
    lines = readlines(path)
    entries = IndexEntry[]
    i = 1
    while i < length(lines)
        hdr = lowercase.(_cells(lines[i]))
        if occursin('|', lines[i]) && _is_separator(lines[i + 1]) &&
                all(c -> c in hdr, ("asset", "kind", "generator"))
            col(name) = findfirst(==(name), hdr)
            cl, ca, ck, cg, cd, cp = col("label"), col("asset"), col("kind"), col("generator"), col("data"), col("provenance")
            cc = col("config")
            i += 2
            while i <= length(lines) && occursin(r"^\s*\|", lines[i])
                cells = _cells(lines[i])
                get_cell(c) = (c === nothing || c > length(cells)) ? "" : String(cells[c])
                kindword = lowercase(first(split(get_cell(ck) * " ")))
                kind = kindword == "generated" ? :generated : kindword == "static" ? :static : :invalid
                assets = _backticked(get_cell(ca))
                isempty(assets) && (assets = [get_cell(ca)])
                gcell = get_cell(cg)
                gens = filter(g -> occursin(r"\.(jl|py|sh|tex|m)$|Makefile$", g), _backticked(gcell))
                cfgs = filter(c -> occursin(r"\.(toml|json|ya?ml)$", c), _backticked(get_cell(cc)))
                label = join(_backticked(get_cell(cl)), ", ")
                isempty(label) && (label = get_cell(cl))
                for a in assets
                    push!(entries, IndexEntry(label, a, kind, gens, cfgs, get_cell(cd), get_cell(cp), i))
                end
                i += 1
            end
        else
            i += 1
        end
    end
    return entries
end

# glob → anchored regex: `**` any path, `*` within a path segment, `?` one char.
function _glob_regex(g::AbstractString)
    io = IOBuffer()
    write(io, '^')
    i = firstindex(g)
    while i <= lastindex(g)
        c = g[i]
        if c == '*'
            j = nextind(g, i)
            if j <= lastindex(g) && g[j] == '*'
                write(io, ".*")
                i = nextind(g, j)
                continue
            end
            write(io, "[^/]*")
        elseif c == '?'
            write(io, "[^/]")
        elseif c in raw".+()[]{}^$|\\"
            write(io, '\\', c)
        else
            write(io, c)
        end
        i = nextind(g, i)
    end
    write(io, '$')
    return Regex(String(take!(io)))
end

_matches(pattern::AbstractString, path::AbstractString) =
    occursin(r"[*?]", pattern) ? occursin(_glob_regex(normpath(pattern)), path) : normpath(pattern) == path

"""
    declared_outputs(generator_path) -> Vector{String}

The outputs a generator declares, one per `reproduce: writes <path-or-glob>`
line in its source (in any comment syntax), project-relative.
"""
declared_outputs(path::AbstractString) =
    [String(_cap(m, 1)) for m in eachmatch(r"reproduce:\s*writes\s+(\S+)", read(path, String))]

"""
    declared_inputs(generator_path) -> Vector{String}

The configs / inputs a generator declares, one per `reproduce: reads <path>`
line in its source (in any comment syntax), project-relative.
"""
declared_inputs(path::AbstractString) =
    [String(_cap(m, 1)) for m in eachmatch(r"reproduce:\s*reads\s+(\S+)", read(path, String))]

function _generator_reads(genpath::AbstractString, config::AbstractString)
    any(c -> normpath(c) == normpath(config), declared_inputs(genpath)) && return true
    return occursin(basename(config), read(genpath, String))
end

function _generator_writes(genpath::AbstractString, asset::AbstractString)
    any(g -> _matches(g, asset), declared_outputs(genpath)) && return true
    return occursin(basename(asset), read(genpath, String))
end

# ---------------------------------------------------------------------------
# Macro literals
# ---------------------------------------------------------------------------

"""
    tex_macro_values(path) -> Vector{Pair{String,String}}

The argument-free macros a tex file defines (`\\newcommand`, `\\renewcommand`,
`\\providecommand`, `\\def`), as `name => value`.
"""
tex_macro_values(path::AbstractString) = _macro_values(read(path, String))

# The argument-free macro definitions of a tex text (comments ignored), as `name => value`.
function _macro_values(source::AbstractString)
    text = _strip_comments(source)
    out = Pair{String, String}[]
    for m in eachmatch(r"\\(?:re|provide)?newcommand\*?\s*\{?\\([A-Za-z@]+)\}?\s*(\[\d\])?\s*\{", text)
        m.captures[2] === nothing || continue
        arg = _braced(text, m.offset + length(m.match) - 1)
        arg === nothing || push!(out, String(_cap(m, 1)) => String(first(arg)))
    end
    for m in eachmatch(r"\\def\s*\\([A-Za-z@]+)\s*\{", text)
        arg = _braced(text, m.offset + length(m.match) - 1)
        arg === nothing || push!(out, String(_cap(m, 1)) => String(first(arg)))
    end
    return out
end

# The rendered form a literal comparison works on: math shifts, thin spaces,
# grouping braces and ties removed.
_render_norm(s::AbstractString) =
    strip(replace(s, r"\\[,;:! ]" => " ", "\$" => "", "{" => "", "}" => "", "~" => " ", r"\s+" => " "))

# A macro value worth looking for in prose: it carries a number, and a bare
# integer has at least `min_integer_digits` digits (small integers — counts,
# percentages, occlusion levels — coincide everywhere).
function _literal_candidate(v::AbstractString, min_integer_digits::Integer)
    occursin(r"\d", v) || return false
    occursin('\\', v) && return false
    occursin(r"^\d+$", v) && return length(v) >= min_integer_digits
    return length(v) >= 2
end

# Prose text of a line: arguments that are never rendered numbers in prose
# (cross references, citations, labels, graphics, options) removed.
_prose(line) = replace(line,
    r"\\(?:c?ref|Cref|eqref|cite[tp]?|label|includegraphics|input|include|url|href|begin|end|externaldocument|suppref|Suppref)\*?(?:\[[^\]]*\])?\{[^}]*\}" => " ",
    r"[0-9.]+\\(?:line|text|column)width" => " ",
    r"\[[^\]]*\]" => " ")

function _literal_hits(norm_line::AbstractString, value::AbstractString)
    pat = Regex("(?<![0-9A-Za-z.])" * replace(value, r"([.+*?()\[\]{}^$|\\])" => s"\\\1") * "(?![0-9]|\\.[0-9])")
    return occursin(pat, norm_line)
end

"""
    macro_literal_violations(prose_files, macro_files; root, ignore = String[],
                             min_integer_digits = 3) -> Vector{ReproduceViolation}

Best-effort check that a number a generated macro file defines is not re-typed
as a literal in the paper's prose: every argument-free macro of `macro_files`
whose rendered value carries a number is searched for in the comment-stripped
prose of `prose_files`, with number boundaries. A bare integer value is
searched only from `min_integer_digits` digits (the default 3 skips the
two-digit counts and percentages that coincide everywhere; 2 is strict and
needs `literal-ok` marks). A line holding `reproduce: literal-ok` (in a
comment) is exempt, as is any macro named in `ignore`.
"""
function macro_literal_violations(prose_files, macro_files; root::AbstractString, ignore = String[],
        min_integer_digits::Integer = 3)
    values = Dict{String, Vector{String}}()   # rendered value => macro names
    for f in macro_files, (name, val) in tex_macro_values(f)
        name in ignore && continue
        v = _render_norm(val)
        _literal_candidate(v, min_integer_digits) || continue
        push!(get!(values, v, String[]), name)
    end
    out = ReproduceViolation[]
    for f in prose_files
        raw = split(read(f, String), '\n')
        stripped = split(_strip_comments(join(raw, '\n')), '\n')
        for (i, line) in enumerate(stripped)
            occursin("reproduce: literal-ok", raw[i]) && continue
            occursin(r"\\(?:re|provide)?newcommand|\\def\b", line) && continue
            norm = _render_norm(_prose(line))
            for (v, names) in values
                _literal_hits(norm, v) || continue
                push!(out, ReproduceViolation(:macro_literal, v,
                    "the literal $(repr(v)) re-types the generated macro " * join("\\" .* names, " / ") *
                        "; use the macro (or mark the line `% reproduce: literal-ok` if the coincidence is not that number)",
                    "$(relpath(f, root)):$i"))
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

"""
    reproduce_index_violations(project_dir; papers, index = "REPRODUCE.md", check_literals = true,
                               ignore_macros = String[], min_integer_digits = 3,
                               require_config = false, search_dirs = String[]) -> Vector{ReproduceViolation}

The reproducible-figures contract of a paper project, as a list of violations
(empty = compliant). `papers` are the top-level tex files and `index` the
REPRODUCE.md, both relative to `project_dir`; every asset path in the index is
project-relative. `search_dirs` (project-relative) are the directories a
reference that does not resolve against a paper's own directory is looked up
in ([`paper_asset_graph`](@ref)): the compile directory of a build run outside
the paper's directory. Violation kinds:

- `:unindexed_asset` — the paper reads a file no index entry covers
- `:missing_asset` — the paper reads a file that does not exist
- `:unresolved_reference` — a reference whose path is not static (macro, `#1`)
- `:unreferenced_entry` — an index entry the paper does not read (stale row)
- `:invalid_kind` — a kind other than `generated` / `static`
- `:no_generator` — a generated entry naming no generator
- `:missing_generator` — a named generator does not exist
- `:generator_does_not_write` — no named generator writes the asset: none
  declares it (`reproduce: writes <glob>`) and none mentions its file name
- `:static_without_provenance` — a static entry with no provenance note
- `:no_config` — with `require_config = true`, a generated entry naming no
  config of record (a `config` column cell)
- `:missing_config` — a named config does not exist
- `:generator_does_not_read_config` — no named generator reads the config:
  none declares it (`reproduce: reads <path>`) and none names its file
- `:label_mismatch` — the entry's label is a `\\label` of the paper, but not of
  the float the asset sits in
- `:macro_literal` — a number a generated macro file defines, re-typed as a
  literal in the prose ([`macro_literal_violations`](@ref), with
  `ignore_macros` and `min_integer_digits`); off with `check_literals = false`
"""
function reproduce_index_violations(project_dir::AbstractString;
        papers::AbstractVector{<:AbstractString}, index::AbstractString = "REPRODUCE.md",
        check_literals::Bool = true, ignore_macros = String[], min_integer_digits::Integer = 3,
        require_config::Bool = false, search_dirs::AbstractVector{<:AbstractString} = String[])
    root = abspath(project_dir)
    out = ReproduceViolation[]
    idxpath = joinpath(root, index)
    if !isfile(idxpath)
        push!(out, ReproduceViolation(:missing_index, index, "no asset index", index))
        return out
    end
    graph = paper_asset_graph(joinpath.(root, papers); root, search_dirs)
    entries = read_reproduce_index(idxpath)
    loc(e::IndexEntry) = "$index:$(e.line)"
    loc(a::PaperAsset) = "$(a.source):$(a.line)"

    for (src, line, text) in graph.unresolved
        push!(out, ReproduceViolation(:unresolved_reference, text,
            "the path is not static; index what it expands to and reference it literally", "$src:$line"))
    end
    for a in graph.assets
        a.exists || push!(out, ReproduceViolation(:missing_asset, a.path, "referenced but not on disk", loc(a)))
        any(e -> _matches(e.asset, a.path), entries) ||
            push!(out, ReproduceViolation(:unindexed_asset, a.path,
                "the paper reads it but $index does not index it", loc(a)))
    end
    for e in entries
        hits = filter(a -> _matches(e.asset, a.path), graph.assets)
        isempty(hits) && push!(out, ReproduceViolation(:unreferenced_entry, e.asset,
            "indexed but the paper does not read it", loc(e)))
        if e.kind === :invalid
            push!(out, ReproduceViolation(:invalid_kind, e.asset, "kind must be `generated` or `static`", loc(e)))
        elseif e.kind === :static
            _is_dash(e.provenance) && push!(out, ReproduceViolation(:static_without_provenance, e.asset,
                "a static asset needs a provenance note (who made it, how, from what)", loc(e)))
        else
            if isempty(e.generators)
                push!(out, ReproduceViolation(:no_generator, e.asset, "a generated asset names no generator", loc(e)))
            end
            gens = filter(g -> isfile(joinpath(root, g)), e.generators)
            for g in setdiff(e.generators, gens)
                push!(out, ReproduceViolation(:missing_generator, g, "the generator does not exist", loc(e)))
            end
            if require_config && isempty(e.configs)
                push!(out, ReproduceViolation(:no_config, e.asset, "a generated asset names no config of record", loc(e)))
            end
            for c in e.configs
                if !isfile(joinpath(root, c))
                    push!(out, ReproduceViolation(:missing_config, c, "the config of record does not exist", loc(e)))
                elseif !isempty(gens) && !any(g -> _generator_reads(joinpath(root, g), c), gens)
                    push!(out, ReproduceViolation(:generator_does_not_read_config, c,
                        "none of " * join(gens, ", ") * " declares (`reproduce: reads`) or names it", loc(e)))
                end
            end
            if !isempty(gens)
                targets = isempty(hits) ? (occursin(r"[*?]", e.asset) ? String[] : [normpath(e.asset)]) :
                    unique(getfield.(hits, :path))
                for t in targets
                    any(g -> _generator_writes(joinpath(root, g), t), gens) ||
                        push!(out, ReproduceViolation(:generator_does_not_write, t,
                            "none of " * join(gens, ", ") * " declares (`reproduce: writes`) or names it", loc(e)))
                end
            end
        end
        lbls = strip.(split(e.label, ','))
        known = filter(in(graph.labels), lbls)
        for a in hits
            (isempty(known) || isempty(a.labels)) && continue
            isempty(intersect(known, a.labels)) &&
                push!(out, ReproduceViolation(:label_mismatch, a.path,
                    "indexed under $(join(known, ", ")) but it sits in $(join(a.labels, ", "))", loc(e)))
        end
    end
    if check_literals
        generated_tex = unique([a.path for a in graph.assets if a.kind === :tex && a.exists &&
            any(e -> e.kind === :generated && _matches(e.asset, a.path), entries)])
        macro_files = filter(p -> !isempty(tex_macro_values(joinpath(root, p))), generated_tex)
        prose = setdiff(graph.tex_files, generated_tex)
        append!(out, macro_literal_violations(joinpath.(root, prose), joinpath.(root, macro_files);
            root, ignore = ignore_macros, min_integer_digits))
    end
    return out
end

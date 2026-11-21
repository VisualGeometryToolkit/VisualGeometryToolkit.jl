# =============================================================================
# violations.jl — the paper-project check: every rule of the conventions, one
# call, one structured list. A paper project's suite holds it in one line:
#
#   @test isempty(paper_project_violations(projectdir(); papers = ["paper/main.tex"]))
#
# Checks (each a `check` symbol; `checks =` selects):
#
#   :layout       L1 a required path is missing (Project.toml, README.md,
#                    configs/, scripts/, src/, test/runtests.jl)
#                 L2 Project.toml does not depend on DrWatson (the layout and
#                    paths) and VisualGeometryToolkitEval (the conventions)
#                 L3 git does not ignore data/exp_raw and data/exp_pro (the
#                    per-machine bindings of the input and results trees)
#                 L4 test/runtests.jl does not call paper_project_violations
#                 L5 scripts/fetch_data.jl is missing or does not call fetch_data!
#                    (DataArtifacts: the one way the inputs reach a machine)
#                 L6 git tracks data under data/ (anything but data/README.md)
#   :data_policy  D1–D5 and STALE (Experiments `data_policy_violations`):
#                 machine paths (code, tests, examples, configs, docs,
#                 provenance), writes into the input tree, unkeyed output
#                 directories, `artifact"…"` in code, an artifact or download
#                 API outside DataArtifacts
#   :reproduce    the REPRODUCE.md asset index against what the paper reads
#                 (Reproduce `reproduce_index_violations`; rule = its kind)
#   :configs      C1 a producing script reads no config of record
#                 C2 a producing script holds a data constant (a literal seed,
#                    scene/frame list, data path or numeric table)
#                 C3 a script names a config under configs/ that does not exist
#   :provenance   P1 a producing script writes no provenance record through the
#                    shared writer (write_provenance / write_run_records /
#                    produce), itself or in what it includes or launches
#                 P2 a producing script keys no output by a run key (run_key)
#                 S1 a generator of the index has no stage runner (run_stages)
#   :catalog      A1 the DataArtifacts catalog disagrees with Artifacts.toml
#                    (DataArtifacts `catalog_violations`), checked once either
#                    file binds an artifact
#   :storage      A2 a config's input (`[inputs.<id>]`) names no tier the
#                    catalog lists: every input is catalogued
#                 A3 an artifact name bound to another tree elsewhere in the
#                    stack (another project's Artifacts.toml, `stack_artifacts`,
#                    or the artifact store's name record): one name, one tree
#                 A4 a link of the artifact view (data/exp_raw) resolves outside
#                    the artifact store (a tree in the depot, a moved store)
#   :release      (opt-in, before a release) R1 a record not yet published (no
#                    Zenodo id, no Hugging Face revision), R2 a Hugging Face
#                    record not stated `public = true`, R3 a binding whose URL
#                    is a placeholder or a local file
#   :contract     (opt-in) the codification contract of every stage file
#                 (Experiments `contract_violations`; rule = its code)
#
# A PRODUCING script is a .jl file under scripts/ that runs stages
# (`run_stages(`), declares an output (`reproduce: writes …`), is a generator
# of the index, or calls a result writer. The configs/provenance scan is
# static and conservative: a necessary condition, not proof.
#
# Opt-out: `# paper-project: allow <RULE> <reason>` — on the offending line or
# the line above it for a line rule (C2, C3), anywhere in the file for a file
# rule (C1, P1, P2, S1). An opt-out without a reason is itself a violation
# (X1). A project's deferred violations of ANY rule go in `allow`
# (Experiments `DataPolicyAllow(rule, file, match)`, `match` a substring of the
# violation's text); an entry that matches nothing is reported STALE.
# =============================================================================

"""
    PaperProjectViolation(check, rule, file, line, message)

One breach of the paper-project conventions: the `check` it belongs to
(`:layout`, `:data_policy`, `:reproduce`, `:configs`, `:provenance`,
`:catalog`, `:storage`, `:release`, `:contract`, `:allow`), the `rule` (`"L1"`, `"D2"`,
`"missing_asset"`, `"C2"`, …; see [`paper_project_violations`](@ref)), the
project-relative `file` and `line` (0 when not line-specific) and a
human-readable `message`.
"""
struct PaperProjectViolation
    check::Symbol
    rule::String
    file::String
    line::Int
    message::String
end

Base.show(io::IO, v::PaperProjectViolation) =
    print(io, v.file, v.line > 0 ? ":$(v.line)" : "", ": [", v.check, "/", v.rule, "] ", v.message)

"The checks [`paper_project_violations`](@ref) runs by default (`:contract` and `:release` are opt-in)."
const PAPER_PROJECT_CHECKS = (:layout, :data_policy, :reproduce, :configs, :provenance, :catalog, :storage)

# ---- source scanning ---------------------------------------------------------------

# The code lines of a Julia file: (line number, text) with full-line comments,
# `#= … =#` blocks and triple-quoted strings (docstrings, here-documents)
# blanked, and a trailing `# …` comment cut when it follows the last quote.
function _code_lines(src::AbstractString)
    out = Tuple{Int,String}[]
    in_triple = false
    in_block = false
    for (i, line) in enumerate(split(src, '\n'))
        s = String(line)
        if in_block
            occursin("=#", s) && (in_block = false)
            continue
        end
        if in_triple
            isodd(length(collect(eachmatch(r"\"\"\"", s)))) && (in_triple = false)
            continue
        end
        t = lstrip(s)
        if startswith(t, "#=")
            occursin("=#", t[3:end]) || (in_block = true)
            continue
        end
        startswith(t, "#") && continue
        if isodd(length(collect(eachmatch(r"\"\"\"", s))))
            in_triple = true
            s = split(s, "\"\"\"")[1]
        end
        push!(out, (i, _cut_trailing_comment(s)))
    end
    return out
end

# Cut `# …` that sits outside every string literal.
function _cut_trailing_comment(s::AbstractString)
    inq = false
    prev = ' '
    for (k, c) in pairs(s)
        if c == '"' && prev != '\\'
            inq = !inq
        elseif c == '#' && !inq && prev != '$'
            return s[1:prevind(s, k)]
        end
        prev = c
    end
    return s
end

_code(src) = join((l for (_, l) in _code_lines(src)), '\n')

# ---- opt-outs ------------------------------------------------------------------------

const _ALLOW_MARK = r"#\s*paper-project:\s*allow\s+([A-Z]\d)\b[ \t:—–-]*(.*)$"

# The opt-outs of a file: rule => line numbers, and the reasonless ones.
function _optouts(src::AbstractString)
    marks = Dict{String,Vector{Int}}()
    bare = Tuple{Int,String}[]
    for (i, line) in enumerate(split(src, '\n'))
        m = match(_ALLOW_MARK, line)
        m === nothing && continue
        rule, reason = String(something(m.captures[1])), strip(something(m.captures[2]))
        push!(get!(marks, rule, Int[]), i)
        isempty(reason) && push!(bare, (i, rule))
    end
    return marks, bare
end

_opted_line(marks, rule, i) = any(j -> j == i || j == i - 1, get(marks, rule, Int[]))

# ---- producers -------------------------------------------------------------------------

const _RESULT_WRITERS = r"\b(write_provenance|write_run_records|produce|produce_or_load|wsave|tagsave|safesave|jldsave)\("
const _PROVENANCE_WRITERS = r"\b(write_provenance|write_run_records|produce)\("
const _RUN_KEYS = r"\b(run_key|write_run_records|produce|Product)\("

_scripts(root::String, scripts::AbstractString) = (d = joinpath(root, scripts); isdir(d) ?
    sort!([joinpath(r, f) for (r, _, fs) in walkdir(d) for f in fs if endswith(f, ".jl")]) : String[])

function _is_producer(src, code, rel, generators)
    rel in generators && return true
    occursin("run_stages(", code) && return true
    occursin(r"reproduce:\s*writes\s+\S", src) && return true
    return occursin(_RESULT_WRITERS, code)
end

# The project files a script reaches by a static `include(…)` or launches by a
# static `run_driver("…")`: the transitive closure, the script included.
function _reached_files(root::AbstractString, file::AbstractString, scripts::AbstractString)
    seen = String[]
    todo = [abspath(file)]
    while !isempty(todo)
        f = pop!(todo)
        (f in seen || !isfile(f)) && continue
        push!(seen, f)
        code = _code(read(f, String))
        for m in eachmatch(r"\binclude\(\s*(srcdir|scriptsdir|projectdir)\(([^)]*)\)\s*\)", code)
            parts = [String(something(p.captures[1])) for p in eachmatch(r"\"([^\"]+)\"", something(m.captures[2]))]
            isempty(parts) && continue
            base = m.captures[1] == "srcdir" ? joinpath(root, "src") :
                   m.captures[1] == "scriptsdir" ? joinpath(root, "scripts") : root
            push!(todo, normpath(foldl(joinpath, parts; init = base)))
        end
        for m in eachmatch(r"\binclude\(\s*joinpath\(\s*@__DIR__\s*,([^)]*)\)\s*\)", code)
            parts = [String(something(p.captures[1])) for p in eachmatch(r"\"([^\"]+)\"", something(m.captures[1]))]
            isempty(parts) || push!(todo, normpath(foldl(joinpath, parts; init = dirname(f))))
        end
        for m in eachmatch(r"\binclude\(\s*\"([^\"]+)\"\s*\)", code)
            push!(todo, normpath(joinpath(dirname(f), something(m.captures[1]))))
        end
        for m in eachmatch(r"\brun_driver\(\s*\"([^\"]+\.jl)\"", code)
            p = String(something(m.captures[1]))
            push!(todo, isabspath(p) ? p : normpath(joinpath(root, scripts, p)))
        end
    end
    return seen
end

# ---- C2: data constants ----------------------------------------------------------------

# Each pattern: what it is, and the offending text of a code line or `nothing`.
_c2_match(re::Regex) = line -> (m = match(re, line); m === nothing ? nothing : String(m.match))

# A numeric range whose bounds are all literals, and either fractional (a sweep)
# or past single digits (a frame or index list): `0.1:0.1:2`, `1:200`.
function _c2_range(line)
    for m in eachmatch(r"(?<![\w.\]\)\$])(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?:(\d+(?:\.\d+)?)(?![\w.(\[])", line)
        nums = String[c for c in m.captures if c !== nothing]
        any(x -> occursin('.', x) || parse(Float64, x) >= 10, nums) && return String(m.match)
    end
    return nothing
end

const _C2_PATTERNS = [
    ("a literal seed", _c2_match(r"(?i)\b\w*seed\w*\s*=\s*\d+\b|\b(MersenneTwister|StableRNG|Xoshiro|seed!)\(\s*\d+")),
    ("a literal list of names (scenes, frames, sequences)",
        _c2_match(r"\[\s*\"[^\"]*\"\s*(,\s*\"[^\"]*\"\s*){2,},?\s*\]")),
    ("a literal numeric table", _c2_match(r"\[\s*-?\d[\d_.eE+-]*(\s*[,; ]\s*-?\d[\d_.eE+-]*){3,}\s*\]")),
    ("a literal numeric range", _c2_range),
    # a path with a directory and a file extension (configs and code excepted),
    # or an absolute / home / explicitly relative one
    ("a literal data path",
        _c2_match(r"\"(?:[^\"\s$]*/[^\"\s$]*\.(?!toml\"|jl\")[A-Za-z0-9]+|(?:/|~/|\./|\.\./)[^\"\s$]+)\"")),
]
# Lines whose literals are messages or structure, not data.
const _C2_SKIP = r"^\s*(using|import|include|export|@info|@warn|@error|@debug|println|print|error|throw|@test|@assert)\b|\brequire_input\(|\bprovenance\s*="

# ---- C3: configs named in code -----------------------------------------------------------

function _named_configs(code::AbstractString, configs::AbstractString)
    out = Tuple{String,Int}[]
    lines = split(code, '\n')
    for (i, line) in enumerate(lines)
        for m in eachmatch(Regex("\"((?:\\./)?$(configs)/[^\"\\s\$]+\\.toml)\""), line)
            push!(out, (normpath(String(something(m.captures[1]))), i))
        end
        for m in eachmatch(Regex("projectdir\\(\\s*\"$(configs)\"\\s*,\\s*\"([^\"\$]+\\.toml)\"\\s*\\)"), line)
            push!(out, (joinpath(configs, String(something(m.captures[1]))), i))
        end
    end
    return out
end

_takes_config(configs) = Regex("\"$(configs)\"|\\b$(configs)/|\\bload_run_config\\(|\\bTOML\\.parsefile\\(|\\bparsefile\\(|--config|cli_option\\(\\s*(?:[A-Za-z_][A-Za-z0-9_.]*\\s*,\\s*)?\"config\"")

# A string literal that is the name of a config of record (`"pt_e_budget"` for
# configs/pt_e_budget.toml): the driver hands it to a project builder that loads it.
_names_config(code, names) = any(m -> something(m.captures[1]) in names, eachmatch(r"\"([\w.-]+)\"", code))

# ---- the check -------------------------------------------------------------------------

"""
    paper_project_violations(dir; papers = ["paper/main.tex"], index = "REPRODUCE.md",
                             checks = PAPER_PROJECT_CHECKS, scripts = "scripts",
                             configs = "configs", catalog = "configs/data_artifacts.toml",
                             catalog_kinds = String[], data_policy = (;), reproduce = (;),
                             stack_artifacts = String[], allow = DataPolicyAllow[]) -> Vector{PaperProjectViolation}

Every breach of the paper-project conventions in project `dir`, empty when
compliant — the one call a paper project's suite asserts empty. `papers` (the
top-level tex files) and `index` are relative to `dir`; `scripts`, `configs`
and `catalog` name the project's directories and its DataArtifacts catalog.

The checks (`checks` selects; [`PAPER_PROJECT_CHECKS`](@ref) by default,
`:contract` opt-in) and their rules:

- `:layout` — `L1` a required path is missing; `L2` Project.toml lacks
  DrWatson or VisualGeometryToolkitEval; `L3` git does not ignore `data/exp_raw` and
  `data/exp_pro`; `L4` test/runtests.jl does not call this check; `L5`
  `scripts/fetch_data.jl` is missing or does not call DataArtifacts'
  `fetch_data!`; `L6` git tracks data under `data/` (anything but `data/README.md`).
- `:data_policy` — Experiments `data_policy_violations` (`D1` machine paths in
  code, tests, examples, configs, docs and provenance, `D2` writes into the
  input tree, `D3` unkeyed outputs, `D4` `artifact"…"` in code, `D5` an
  artifact or download API outside DataArtifacts); `data_policy` forwards its
  keywords (`dirs`, `exts`).
- `:reproduce` — Reproduce `reproduce_index_violations` (rule = its kind);
  `reproduce` forwards its keywords (`search_dirs` for a paper built outside
  its own directory, `check_literals`, `ignore_macros`, …).
- `:configs` — `C1` a producing script reads no config of record; `C2` it
  holds a data constant (a literal seed, name list, numeric table or range,
  data path); `C3` a script names a config under `configs/` that does not exist.
- `:provenance` — `P1` a producing script (with what it includes and
  launches) writes no provenance through the shared writer; `P2` it keys no
  output by a run key; `S1` a generator of the index has no stage runner.
- `:catalog` — `A1` DataArtifacts `catalog_violations`, once the catalog or
  Artifacts.toml binds an artifact (`catalog_kinds` are the project's own
  kinds); and a produced entry whose `producer` is no stage of
  `configs/reproduce.toml`, or reads an input that is neither a published tier
  nor covered by another produced entry (or a cycle of them). A producer of
  another project, `"<project>:<stage>"`, must be a stage of that project's
  graph, the project one of `stack_artifacts` (its graph beside its
  Artifacts.toml, in `configs/reproduce.toml`).
- `:storage` — `A2` a config's `[inputs.<id>]` names no `tier` the catalog
  lists and no produced entry (`kind = "produced"`) covers its path (every
  input is catalogued); an input naming another project's product
  (`project`, `product`, `key`, …: Experiments' `input_path`) is accepted when
  well-formed, any `tier` it names catalogued; `A3` a name of the project's
  Artifacts.toml bound to another tree in one of `stack_artifacts` (other
  projects' Artifacts.toml files) or in the artifact store's name record
  (VisualGeometryToolkitCore's `store_name_clashes`): one name, one tree
  stack-wide; `A4` a link of the artifact view `data/exp_raw` that resolves
  outside the artifact store (VisualGeometryToolkitCore's `artifact_store`).
- `:release` (opt-in: before a release) — `R1` a catalog record not yet
  published (a Zenodo record without `id`, a Hugging Face record without
  `revision`), `R2` a Hugging Face record that does not state `public = true`,
  `R3` an Artifacts.toml binding whose URL is a placeholder or a local file.
- `:contract` — Experiments `contract_violations` of every stage file.

A producing script is a `.jl` under `scripts/` that runs stages, declares an
output (`reproduce: writes`), is a generator of the index or calls a result
writer. Opt out of one rule where the constant is justified with
`# paper-project: allow <RULE> <reason>` (on the line or the line above for
`C2`/`C3`; anywhere in the file for `C1`, `P1`, `P2`, `S1`); a reasonless
opt-out is `X1`. Deferred violations of any rule go in `allow`
(`DataPolicyAllow(rule, file, match)`, `match` a substring of the violation's
text); an entry matching nothing is reported `STALE`.
"""
function paper_project_violations(dir::AbstractString;
        papers::AbstractVector{<:AbstractString} = ["paper/main.tex"],
        index::AbstractString = "REPRODUCE.md",
        checks = PAPER_PROJECT_CHECKS,
        scripts::AbstractString = "scripts", configs::AbstractString = "configs",
        catalog::AbstractString = _PROJECT_CATALOG,
        catalog_kinds = String[], data_policy = (;), reproduce = (;),
        stack_artifacts::AbstractVector{<:AbstractString} = String[],
        allow::AbstractVector{DataPolicyAllow} = DataPolicyAllow[])
    root = String(abspath(dir))
    out = PaperProjectViolation[]
    add!(check, rule, file, line, msg) = push!(out, PaperProjectViolation(check, String(rule), String(file), line, String(msg)))
    dp_allow = filter(a -> occursin(r"^D\d$", a.rule), allow)
    own_allow = filter(a -> !occursin(r"^D\d$", a.rule), allow)

    :layout in checks && _layout!(add!, root, scripts, configs)

    if :data_policy in checks
        for v in data_policy_violations(root; allow = dp_allow, data_policy...)
            add!(:data_policy, v.rule, v.file, v.line, v.text)
        end
    end

    entries = isfile(joinpath(root, index)) ? read_reproduce_index(joinpath(root, index)) : IndexEntry[]
    if :reproduce in checks
        for v in reproduce_index_violations(root; papers, index, reproduce...)
            file, line = _split_location(v.location)
            add!(:reproduce, string(v.kind), file, line, "$(v.subject) — $(v.message)")
        end
    end

    generators = Set(normpath(g) for e in entries if e.kind === :generated for g in e.generators)
    (:configs in checks || :provenance in checks || :contract in checks) &&
        _scripts!(add!, root, scripts, configs, generators, checks)

    :catalog in checks && _catalog!(add!, root, configs, catalog, catalog_kinds, stack_artifacts)
    :storage in checks && _storage!(add!, root, configs, catalog, stack_artifacts)
    :release in checks && _release!(add!, root, catalog)

    return _apply_allow(out, own_allow)
end

_split_location(loc) = (m = match(r"^(.*):(\d+)$", loc); m === nothing ? (loc, 0) :
                        (String(something(m.captures[1])), parse(Int, something(m.captures[2]))))

function _apply_allow(found, allow)
    used = falses(length(allow))
    keep = filter(found) do v
        k = findfirst(a -> a.rule == v.rule && a.file == v.file && occursin(a.match, v.message), allow)
        k === nothing && return true
        used[k] = true
        return false
    end
    for (k, a) in enumerate(allow)
        used[k] || push!(keep, PaperProjectViolation(:allow, "STALE", a.file, 0,
            "allowlist entry $(a.rule) \"$(a.match)\" matches nothing"))
    end
    return keep
end

# ---- :layout ---------------------------------------------------------------------------

function _layout!(add!, root::String, scripts::AbstractString, configs::AbstractString)
    for p in ("Project.toml", "README.md", configs, scripts, "src", joinpath("test", "runtests.jl"))
        ispath(joinpath(root, p)) || add!(:layout, "L1", p, 0, "missing — the paper-project layout requires it")
    end
    proj = joinpath(root, "Project.toml")
    if isfile(proj)
        deps = get(TOML.parsefile(proj), "deps", Dict{String,Any}())
        for d in ("DrWatson", "VisualGeometryToolkitEval")
            haskey(deps, d) || add!(:layout, "L2", "Project.toml", 0, "[deps] lacks $d")
        end
    end
    for d in ("exp_raw", "exp_pro")
        p = joinpath("data", d)
        _git_ignored(root, p) || add!(:layout, "L3", ".gitignore", 0,
            "$p is not ignored — it is this machine's binding of the $(d == "exp_raw" ? "input" : "results") tree, never tracked")
    end
    rt = joinpath(root, "test", "runtests.jl")
    if isfile(rt)
        reached = _reached_files(root, rt, "test")
        any(f -> occursin("paper_project_violations(", _code(read(f, String))), reached) ||
            add!(:layout, "L4", "test/runtests.jl", 0, "the suite does not call paper_project_violations")
    end
    fd = joinpath(scripts, "fetch_data.jl")
    if !isfile(joinpath(root, fd))
        add!(:layout, "L5", fd, 0, "missing — the new-machine step that installs the published inputs (DataArtifacts fetch_data!)")
    elseif !occursin("fetch_data!(", _code(read(joinpath(root, fd), String)))
        add!(:layout, "L5", fd, 0, "does not call DataArtifacts' fetch_data! (the one way the inputs reach a machine)")
    end
    for f in _git_tracked(root, "data")
        f == joinpath("data", "README.md") && continue
        add!(:layout, "L6", f, 0, "tracked data under data/: inputs are published DataArtifacts tiers (in the artifact store), results are never tracked")
    end
    return nothing
end

# The files git tracks under `rel` (relative to `root`); none outside a git work tree.
_git_tracked(root::String, rel::String) =
    sort!(filter(!isempty, split(something(_git_line(root, ["ls-files", "-z", "--", rel]), ""), '\0')))

# Whether git ignores `rel` (a path under `root`, existing or not). Outside a
# git work tree, the project's own .gitignore is read: a line naming the path
# or one of its ancestors.
function _git_ignored(root::String, rel::String)
    probe = joinpath(rel, ".paper-project-probe")
    intree = try
        success(pipeline(`git -C $root rev-parse --is-inside-work-tree`; stdout = devnull, stderr = devnull))
    catch
        false
    end
    if intree
        # the path itself (a link is a file to git), else a file under it (a
        # directory pattern such as `data/` matches only through its contents)
        ign(p) = success(pipeline(`git -C $root check-ignore -q --no-index $p`; stdout = devnull, stderr = devnull))
        return ign(rel) || ign(probe)
    end
    gi = joinpath(root, ".gitignore")
    isfile(gi) || return false
    parts = splitpath(rel)
    anc = [foldl(joinpath, parts[1:k]) for k in eachindex(parts)]
    for l in eachline(gi)
        s = strip(l)
        (isempty(s) || startswith(s, "#")) && continue
        s = rstrip(lstrip(s, '/'), '/')
        s in anc && return true
    end
    return false
end

# ---- :configs / :provenance / :contract --------------------------------------------------

function _scripts!(add!, root::String, scripts::AbstractString, configs::AbstractString, generators, checks)
    takes_config = _takes_config(configs)
    cdir = joinpath(root, configs)
    config_names = isdir(cdir) ? Set(first(splitext(f)) for f in readdir(cdir) if endswith(f, ".toml")) : Set{String}()
    for path in _scripts(root, scripts)
        rel = relpath(path, root)
        src = read(path, String)
        lines = _code_lines(src)
        code = join((l for (_, l) in lines), '\n')
        marks, bare = _optouts(src)
        for (i, r) in bare
            add!(:allow, "X1", rel, i, "opt-out of $r gives no reason (`# paper-project: allow $r <reason>`)")
        end
        if :configs in checks
            for (i, line) in lines, (c, _) in _named_configs(line, configs)
                isfile(joinpath(root, c)) || _opted_line(marks, "C3", i) ||
                    add!(:configs, "C3", rel, i, "names $c, which does not exist")
            end
        end
        producer = _is_producer(src, code, rel, generators)
        producer || continue
        if :configs in checks
            (occursin(takes_config, code) || _names_config(code, config_names)) || haskey(marks, "C1") ||
                add!(:configs, "C1", rel, 0, "produces results or paper assets but reads no config of record under $configs/")
            for (i, line) in lines
                occursin(_C2_SKIP, line) && continue
                _opted_line(marks, "C2", i) && continue
                for (what, f) in _C2_PATTERNS
                    hit = f(line)
                    hit === nothing && continue
                    add!(:configs, "C2", rel, i, "$what in a producing script ($(strip(hit))) — it belongs in the config of record")
                    break
                end
            end
        end
        if :provenance in checks
            reached = _reached_files(root, path, scripts)
            reached_code = join((_code(read(f, String)) for f in reached), '\n')
            occursin(_PROVENANCE_WRITERS, reached_code) || haskey(marks, "P1") ||
                add!(:provenance, "P1", rel, 0,
                     "writes no provenance record through the shared writer (write_provenance / write_run_records / produce)")
            occursin(_RUN_KEYS, reached_code) || haskey(marks, "P2") ||
                add!(:provenance, "P2", rel, 0, "keys no output directory by the run key of its config (run_key)")
            rel in generators && !occursin("run_stages(", code) && !haskey(marks, "S1") &&
                add!(:provenance, "S1", rel, 0, "a generator of the asset index without the stage runner (run_stages)")
        end
        if :contract in checks && occursin("run_stages(", code)
            for v in contract_violations(path)
                add!(:contract, first(split(v, ':')), rel, 0, v)
            end
        end
    end
    return nothing
end

# ---- :storage / :release ---------------------------------------------------------------

# An Artifacts.toml entry's bindings: one table, or a platform-specific entry's tables (`[[name]]`).
_entries(b) = b isa AbstractVector ? b : [b]
# An entry's git-tree-sha1s, one per binding (sorted: a platform entry's are a set).
_tree_hashes(b) = sort!([string(get(e, "git-tree-sha1", "")) for e in _entries(b)])

# What accounts for a config input `t` of `tier`: `:published` (a tier the catalog lists, `names`), the produced entry
# that covers its path, or `nothing`.
function _input_cover(cat, names, t, tier)
    # another project's product: its run of record, reproducible there (a `tier` it names publishes it here)
    is_product_input(t) && return isempty(tier) || tier in names ? :product : nothing
    tier in names && return :published
    (cat === nothing || !haskey(t, "path")) && return nothing
    return _produced_cover(cat, string(t["path"]), tier)
end

# The stage graphs of the stack projects of `stack_artifacts` (each another project's Artifacts.toml, its graph at
# configs/reproduce.toml beside it), by project name; a project without a readable graph is left out.
function _stack_graphs(stack_artifacts)
    out = Dict{String, Any}()
    for a in stack_artifacts
        gp = joinpath(dirname(abspath(a)), "configs", "reproduce.toml")
        isfile(gp) || continue
        g = try load_stage_graph(gp) catch; continue end
        out[g.name] = g
    end
    return out
end

# The project's catalog, or `nothing` without one.
_project_catalog(root, catalog) = (path = joinpath(root, catalog); isfile(path) ? load_catalog(path) : nothing)

# The catalog's entry names and their parts (every name a config's tier may name).
function _catalogued(cat)
    entries = [n for (n, a) in cat.artifacts if !_is_produced(a)]
    names = Set{String}(entries)
    for n in entries
        union!(names, try artifact_parts(cat, n) catch; String[] end)
    end
    return names
end

function _storage!(add!, root::String, configs::AbstractString, catalog::AbstractString, stack_artifacts)
    # A2: every config input is catalogued: a tier it names, or a produced entry that covers it
    cat = _project_catalog(root, catalog)
    names = cat === nothing ? Set{String}() : _catalogued(cat)
    cdir = joinpath(root, configs)
    for f in (isdir(cdir) ? sort(readdir(cdir)) : String[])
        endswith(f, ".toml") || continue
        rel = joinpath(configs, f)
        rel == catalog && continue
        d = try TOML.parsefile(joinpath(cdir, f)) catch; continue end
        ins = get(d, "inputs", nothing)
        ins isa AbstractDict || continue
        for (id, t) in sort(collect(ins); by = first)
            t isa AbstractDict || continue
            tier = string(get(t, "tier", ""))
            if is_product_input(t)                     # another project's product: well-formed, any tier it names catalogued
                bad = try product_input_path(t); nothing catch err; sprint(showerror, err) end
                bad === nothing || (add!(:storage, "A2", rel, 0, "[inputs.$id]: $bad"); continue)
                isempty(tier) || (cat !== nothing && haskey(cat.artifacts, tier) && get(cat.artifacts[tier], "mount", "") == "results") ||
                    add!(:storage, "A2", rel, 0, "[inputs.$id] reads $(t["project"])'s product through tier $(repr(tier)), which " *
                                                 "$catalog does not list as a results-mount tier (mount = \"results\")")
                continue
            end
            isempty(tier) && (add!(:storage, "A2", rel, 0, "[inputs.$id] names no tier: an input is a catalogued tier's file ($catalog)"); continue)
            _input_cover(cat, names, t, tier) === nothing && add!(:storage, "A2", rel, 0, "[inputs.$id] reads tier $(repr(tier)), which $catalog does not list " *
                                             "(nor does a produced entry cover its path)")
        end
    end
    # A3: one name, one tree, stack-wide
    arts = joinpath(root, "Artifacts.toml")
    own = isfile(arts) ? TOML.parsefile(arts) : Dict{String, Any}()
    for other in stack_artifacts
        isfile(other) || (add!(:storage, "A3", "Artifacts.toml", 0, "stack_artifacts: no file $other"); continue)
        theirs = TOML.parsefile(other)
        for (n, b) in sort(collect(own); by = first)
            haskey(theirs, n) || continue
            h, g = _tree_hashes(b), _tree_hashes(theirs[n])
            h == g || add!(:storage, "A3", "Artifacts.toml", 0, "$n is the tree $(join(h, ", ")) here and $(join(g, ", ")) in $other: one name, one tree")
        end
    end
    view = joinpath(root, "data", "exp_raw")
    has_view = isdir(view) && is_artifact_view(view)
    (isempty(own) && !has_view) && return nothing                      # nothing published, nothing installed
    store = try artifact_store() catch err; err end
    if store isa Exception
        add!(:storage, "A4", "data/exp_raw", 0, sprint(showerror, store))
        return nothing
    end
    # the depot store: Pkg finds (and links) a tree in any depot of DEPOT_PATH
    tops = store.source === :depot ? [joinpath(d, "artifacts") for d in DEPOT_PATH] : [store.path]
    for (n, hs) in store_name_clashes(store)
        haskey(own, n) && add!(:storage, "A3", "Artifacts.toml", 0,
                               "the artifact store ($(store.path)) holds $(length(hs)) trees under the name $n ($(join(hs, ", "))): one name, one tree")
    end
    # A4: the artifact view's links resolve into the store
    if has_view
        for (r, _, fs) in walkdir(view), f in sort(fs)
            p = joinpath(r, f)
            islink(p) || continue
            t = ispath(p) ? realpath(p) : readlink(p)
            any(s -> _is_under(t, s), tops) || add!(:storage, "A4", relpath(p, root), 0,
                                       "links to $t, outside the artifact store $(store.path) ($(store.source)): remove data/exp_raw " *
                                       "and rerun scripts/fetch_data.jl (VisualGeometryToolkitCore.migrate_artifacts moves a depot's trees)")
        end
    end
    return nothing
end

function _release!(add!, root::String, catalog::AbstractString)
    path = joinpath(root, catalog)
    if isfile(path)
        cat = load_catalog(path)
        for (g, r) in sort(collect(cat.records); by = first)
            host = get(r, "host", "zenodo")
            if host == "zenodo"
                haskey(r, "id") || add!(:release, "R1", catalog, 0, "record $g: no Zenodo id (not published)")
            elseif host == "huggingface"
                haskey(r, "revision") || add!(:release, "R1", catalog, 0, "record $g: no Hugging Face revision (not uploaded)")
                get(r, "public", false) === true ||
                    add!(:release, "R2", catalog, 0, "record $g: a Hugging Face dataset must be public before a release (`public = true` once it is)")
            end
        end
    end
    arts = joinpath(root, "Artifacts.toml")
    for (n, b) in sort(collect(isfile(arts) ? TOML.parsefile(arts) : Dict{String, Any}()); by = first)
        for d in Iterators.flatten(get(e, "download", []) for e in _entries(b))
            u = get(d, "url", "")
            (!is_published_url(u) || startswith(u, "file:")) &&
                add!(:release, "R3", "Artifacts.toml", 0, "$n downloads from $(repr(u)): not a public host")
        end
    end
    return nothing
end

# ---- :catalog --------------------------------------------------------------------------

function _catalog!(add!, root::String, configs::AbstractString, catalog::AbstractString, kinds, stack_artifacts = String[])
    arts = joinpath(root, "Artifacts.toml")
    bound = isfile(arts) ? TOML.parsefile(arts) : Dict{String,Any}()
    cat = _project_catalog(root, catalog)
    if cat === nothing
        isempty(bound) || add!(:catalog, "A1", catalog, 0, "Artifacts.toml binds $(join(sort!(collect(keys(bound))), ", ")) but there is no catalog")
        return nothing
    end
    isempty(cat.artifacts) && isempty(bound) && return nothing   # nothing published yet
    for v in catalog_violations(cat; kinds)
        add!(:catalog, "A1", catalog, 0, v)
    end
    for v in _producer_violations(root, configs, cat; stack_artifacts)
        add!(:catalog, "A1", catalog, 0, v)
    end
    return nothing
end

# A produced entry's producer: a stage of the project's graph (`<configs>/reproduce.toml`) whose configs of record (every
# array item's: no environment selects a subset) have inputs that are each a published tier of the catalog or covered
# by another produced entry, with no cycle through the produced entries. A stage that cannot be read is a finding.
function _producer_violations(root::String, configs::AbstractString, cat; stack_artifacts = String[])
    produced = sort([n for (n, a) in cat.artifacts if _is_produced(a)])
    isempty(produced) && return String[]
    # a producer of ANOTHER project, `<project>:<stage>`: a stage of that project's graph, the project found among the
    # stack's other projects (`stack_artifacts`: their Artifacts.toml, the graph beside it in configs/reproduce.toml);
    # its own inputs are that project's check
    v = String[]
    others = occursin.(':', [string(get(cat.artifacts[n], "producer", "")) for n in produced])
    graphs = any(others) ? _stack_graphs(stack_artifacts) : Dict{String, Any}()
    for n in produced[others]
        stage = get(cat.artifacts[n], "producer", "")
        proj, st = split(stage, ':'; limit = 2)
        g = get(graphs, proj, nothing)
        g === nothing ? push!(v, "$n: producer $(repr(stage)): no project $(repr(proj)) among stack_artifacts (its Artifacts.toml, with configs/reproduce.toml beside it)") :
            _stage_index(g, st) === nothing && push!(v, "$n: producer $(repr(stage)): $(repr(st)) is no stage of $proj's graph ($(g.path))")
    end
    produced = produced[.!others]
    isempty(produced) && return v
    grel = joinpath(configs, "reproduce.toml")
    gpath = joinpath(root, grel)
    isfile(gpath) || return [v; ["$n: produced by stage $(repr(get(cat.artifacts[n], "producer", ""))), but there is no stage graph ($grel)" for n in produced]]
    g = try load_stage_graph(gpath) catch err; return [v; "the stage graph: $(sprint(showerror, err))"] end
    deps = Dict{String, Vector{String}}()           # produced entry → the produced entries its producer reads
    published = _catalogued(cat)
    for n in produced
        stage = get(cat.artifacts[n], "producer", "")
        k = _stage_index(g, stage)
        if k === nothing
            push!(v, "$n: producer $(repr(stage)) is no stage of $grel")
            continue
        end
        spec = g.stages[k]
        deps[n] = String[]
        try
            cfgs = _stage_configs(g, spec; env = Dict{String, String}(),
                                  missing! = (p, _, _) -> push!(v, "$n: its producer's config $p does not exist"))
            isempty(cfgs) && push!(v, "$n: producer $(repr(stage)) has no config of record (its inputs are read from one)")
            for (cfg, _) in cfgs
                p = joinpath(g.workdir, cfg)
                isfile(p) || (push!(v, "$n: its producer's config $cfg does not exist"); continue)
                for (id, t, tier) in sort(_config_inputs(p); by = first)
                    c = _input_cover(cat, published, t, tier)
                    c in (:published, :product) && continue
                    c === nothing ? push!(v, "$n: its producer reads [inputs.$id] of $cfg (tier $(repr(tier))), which is neither published nor produced") :
                                    (c in deps[n] || push!(deps[n], c))
                end
            end
        catch err
            push!(v, "$n: its producer $(repr(stage)) cannot be read: $(sprint(showerror, err))")
        end
    end
    # no cycle: a produced entry is never (transitively) an input of its own producer
    state = Dict{String, Int}()                     # 1 on the path, 2 done
    function visit(n, path)
        get(state, n, 0) == 2 && return
        get(state, n, 0) == 1 && (push!(v, "produced entries form a cycle: $(join([path; n], " → "))"); return)
        state[n] = 1
        foreach(m -> visit(m, [path; n]), get(deps, n, String[]))
        state[n] = 2
    end
    foreach(n -> visit(n, String[]), produced)
    return unique!(v)
end

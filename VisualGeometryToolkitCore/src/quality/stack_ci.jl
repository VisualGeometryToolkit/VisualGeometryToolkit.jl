# =============================================================================
# The CI of a stack repository: one implementation, called by every workspace
# =============================================================================
#
#   * `stack_ci`                        — the command line of everything below
#                                         (`affected`, `packages`, `registration-lint`,
#                                         `agent-docs`, `downstream`, `codecov`).
#   * `workspace_packages`,
#     `workspace_settings`              — a workspace's packages (its root Project.toml)
#                                         and its own parameters (its WORKSPACE.toml).
#   * `affected_packages`               — the packages a change affects, with their
#                                         dependents.
#   * `registration_violations`         — the General/AutoMerge rules that can be
#                                         checked before registering.
#   * `workspace_agent_doc_violations`  — the agent-docs policy over a whole checkout.
#   * `source_downstream`               — point a downstream project at a workspace.
#   * `codecov_config`                  — a workspace's codecov.yml.
#   * `docs_inventory`, `docs_local`,
#     `docs_linkcheck_ignore`           — a sibling package's Documenter inventory.
#
# Why here: a stack repository (the VisualGeometryToolkit monorepo, a toolkit
# built on it) is one Pkg workspace with the same CI. Each kept a copy of these
# scripts in its `ci/`; a copy drifts. They are functions of the repository
# root, and what differs between repositories is data: the package list (the
# root Project.toml's `[workspace]`) and a few parameters (WORKSPACE.toml).
#
# LOADABLE BY PATH (standard libraries only at load time: Pkg, TOML): a CI
# planning job includes this file without loading VisualGeometryToolkitCore,
# which would compile it first. `stack_ci`'s `affected`, `packages`,
# `downstream` and `codecov` commands need nothing else; `registration-lint`
# and `agent-docs` call `stack_dependency_violations` and
# `agent_doc_violations`, which exist where this file is the package's.
# =============================================================================

import Pkg, TOML

const _WORKSPACE_KEYS = ("tooling", "neutral", "placeholder_version", "stack_rows", "manual_merge")
const _INITIAL_VERSION = v"0.1.0"

"""
    workspace_settings(root) -> NamedTuple{(:tooling, :neutral, :placeholder_version, :stack_rows, :manual_merge)}

A workspace's own parameters of the stack's CI, read from `WORKSPACE.toml` at
the repository root `root` (every key optional; no file gives the defaults):

- `tooling` (default `["ci"]`): the `[workspace]` members that are not packages.
- `neutral` (default `[]`): top-level directories that hold no package code and
  no package tooling (a landing page, design notes); a change under one affects
  no package ([`affected_packages`](@ref)).
- `placeholder_version` (default: none): the version of a package under
  construction, which is not registered and must not be
  ([`registration_violations`](@ref)).
- `stack_rows` (default: none): the path, relative to `root`, of this
  repository's rows of the stack's dependency allow-list (the `extra` of
  [`stack_dependency_violations`](@ref)); returned absolute.
- `[manual_merge]` (default: empty): `Package = "reason"` for a package whose
  name AutoMerge's similarity guideline refuses, so a General maintainer merges
  its first registration by hand.

An unknown key throws.
"""
function workspace_settings(root::AbstractString)
    path = joinpath(root, "WORKSPACE.toml")
    t = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    unknown = sort!(String[k for k in keys(t) if !(k in _WORKSPACE_KEYS)])
    isempty(unknown) || throw(ArgumentError("workspace_settings: $path has unknown key(s) $(join(unknown, ", ")); " *
        "the keys are $(join(_WORKSPACE_KEYS, ", "))"))
    version = get(t, "placeholder_version", nothing)
    rows = get(t, "stack_rows", nothing)
    return (;
        tooling = String[x for x in get(t, "tooling", ["ci"])],
        neutral = String[x for x in get(t, "neutral", String[])],
        placeholder_version = version === nothing ? nothing : VersionNumber(version),
        stack_rows = rows === nothing ? nothing : joinpath(root, rows),
        manual_merge = Dict{String, String}(k => v for (k, v) in get(t, "manual_merge", Dict{String, Any}())),
    )
end

"""
    workspace_packages(root) -> Vector{String}

The packages of the Pkg workspace at the repository root `root`: the
`[workspace] projects` of its Project.toml, in the file's order, without the
tooling members ([`workspace_settings`](@ref)).
"""
function workspace_packages(root::AbstractString)
    tooling = workspace_settings(root).tooling
    projects = TOML.parsefile(joinpath(root, "Project.toml"))["workspace"]["projects"]
    return String[p for p in projects if !(p in tooling)]
end

# package → the workspace packages its environments name
function _workspace_edges(root, packages)
    out = Dict{String, Set{String}}()
    for p in packages
        s = Set{String}()
        for env in ("", "test", "docs")
            f = joinpath(root, p, env, "Project.toml")
            isfile(f) || continue
            t = TOML.parsefile(f)
            for sec in ("deps", "weakdeps"), n in keys(get(t, sec, Dict{String, Any}()))
                n in packages && n != p && push!(s, n)
            end
        end
        out[p] = s
    end
    return out
end

"""
    affected_packages(root, paths) -> Vector{String}

The packages of the workspace at `root` that a change to `paths` (relative to
`root`) affects, sorted: a path under `<Package>/` affects that package, a path
under a neutral directory ([`workspace_settings`](@ref)) affects none, and any
other path (a root file, `ci/`, `.github/`) affects every package. The result
is closed under the workspace's own dependency edges — `[deps]`, `[weakdeps]`
and the test and docs environments of each package — so a change to a package
also runs every package built on it.
"""
function affected_packages(root::AbstractString, paths)
    packages = workspace_packages(root)
    neutral = workspace_settings(root).neutral
    direct = Set{String}()
    for path in paths
        top = first(splitpath(path))
        if top in packages
            push!(direct, top)
        elseif top in neutral
            continue
        else
            return sort(packages)        # workspace-level change
        end
    end
    deps = _workspace_edges(root, packages)
    closure = copy(direct)
    changed = true
    while changed
        changed = false
        for (p, ds) in deps
            p in closure && continue
            isempty(intersect(ds, closure)) && continue
            push!(closure, p)
            changed = true
        end
    end
    return sort!(collect(closure))
end

# A package list as GitHub Actions reads it: JSON on `io`, and `packages=<json>` in $GITHUB_OUTPUT.
function _emit_packages(io::IO, packages)
    json = "[" * join(("\"$p\"" for p in packages), ",") * "]"
    println(io, json)
    out = get(ENV, "GITHUB_OUTPUT", "")
    isempty(out) || open(f -> println(f, "packages=", json), out, "a")
    return 0
end

# -----------------------------------------------------------------------------
# The registration rules
# -----------------------------------------------------------------------------

# A package's breaking series, from its version in the workspace: "0.m" below 1.0, "M" from 1.0 on.
_series(v::VersionNumber) = v.major == 0 ? "0.$(v.minor)" : "$(v.major)"
# The series itself, or a floor inside it: "0.5" or "0.5.N"
_in_series(c, s) = occursin(Regex("^" * replace(s, "." => "\\.") * raw"(\.\d+)*$"), c)

# optimal string alignment distance, case-insensitive
function _dl_distance(a::AbstractString, b::AbstractString)
    x, y = collect(lowercase(a)), collect(lowercase(b))
    m, n = length(x), length(y)
    d = zeros(Int, m + 1, n + 1)
    d[:, 1] = 0:m
    d[1, :] = 0:n
    for i in 1:m, j in 1:n
        c = x[i] == y[j] ? 0 : 1
        d[i + 1, j + 1] = min(d[i, j + 1] + 1, d[i + 1, j] + 1, d[i, j] + c)
        if i > 1 && j > 1 && x[i] == y[j - 1] && x[i - 1] == y[j]
            d[i + 1, j + 1] = min(d[i + 1, j + 1], d[i - 1, j - 1] + 1)
        end
    end
    return d[m + 1, n + 1]
end

function _general_registry()
    for r in Pkg.Registry.reachable_registries()
        r.name == "General" && return r
    end
    return nothing
end

# What one run of the rules reads and writes.
struct _RegistrationLint
    root::String
    packages::Vector{String}
    series::Dict{String, String}
    stdlibs::Set{String}
    placeholder_version::Union{Nothing, VersionNumber}
    stack_rows::Union{Nothing, String}
    manual_merge::Dict{String, String}
    errors::Vector{String}
    warnings::Vector{String}
end

_lint_error!(lint::_RegistrationLint, pkg, msg) = push!(lint.errors, "ERROR  $pkg: $msg")
_lint_warning!(lint::_RegistrationLint, pkg, msg) = push!(lint.warnings, "WARN   $pkg: $msg")

function _check_compat!(lint::_RegistrationLint, pkg, label, proj; skip = ())
    compat = get(proj, "compat", Dict{String, Any}())
    haskey(compat, "julia") || _lint_error!(lint, pkg, "$label: no julia compat")
    for sec in ("deps", "weakdeps"), (n, u) in get(proj, sec, Dict{String, Any}())
        n in skip && continue
        haskey(compat, n) ||
            _lint_error!(lint, pkg, "$label: [$sec] $n has no [compat] entry$(u in lint.stdlibs ? " (stdlib)" : "")")
    end
    for (n, c) in compat
        c isa AbstractString || (_lint_error!(lint, pkg, "$label: compat $n is not a string"); continue)
        (occursin('*', c) || occursin(">=", c) || occursin('≥', c)) &&
            _lint_error!(lint, pkg, "$label: compat $n = \"$c\" is unbounded")
        try
            Pkg.Versions.semver_spec(c)
        catch e
            _lint_error!(lint, pkg, "$label: compat $n = \"$c\" does not parse ($(sprint(showerror, e)))")
        end
        if n in lint.packages && !_in_series(c, lint.series[n])
            s = lint.series[n]
            _lint_error!(lint, pkg, "$label: stack compat $n = \"$c\", expected $n's series \"$s\" or a floor inside it (\"$s.N\")")
        end
    end
    return nothing
end

function _check_allow_list!(lint::_RegistrationLint, pkg, label, path; self_ok = false)
    found = try
        stack_dependency_violations(path, pkg; extra = lint.stack_rows)
    catch e
        [sprint(showerror, e)]
    end
    self_ok && filter!(f -> !endswith(f, ": $pkg"), found)
    foreach(f -> _lint_error!(lint, pkg, "$label: stack allow-list: $f"), found)
    return nothing
end

function _check_package!(lint::_RegistrationLint, pkg, general, initial)
    dir = joinpath(lint.root, pkg)
    path = joinpath(dir, "Project.toml")
    isfile(path) || return _lint_error!(lint, pkg, "no Project.toml")
    proj = TOML.parsefile(path)
    uuid = get(proj, "uuid", nothing)
    # a package General holds under its UUID is past its first registration
    general !== nothing && any(((u, e),) -> e.name == pkg && string(u) == uuid, general.pkgs) && (initial = false)
    name = get(proj, "name", nothing)
    name == pkg || _lint_error!(lint, pkg, "name = $(repr(name)), expected the directory name")
    try
        Base.UUID(uuid)
    catch
        _lint_error!(lint, pkg, "uuid $(repr(uuid)) is not a UUID")
    end
    v = try
        VersionNumber(get(proj, "version", ""))
    catch
        _lint_error!(lint, pkg, "version $(repr(get(proj, "version", nothing))) is not a version")
        nothing
    end
    placeholder = v !== nothing && v == lint.placeholder_version
    placeholder && _lint_warning!(lint, pkg, "version $v is the placeholder: under construction, not registered; " *
        "its first registration is $_INITIAL_VERSION")
    initial && !placeholder && v !== nothing && v != _INITIAL_VERSION &&
        _lint_error!(lint, pkg, "version $v; the first registration is $_INITIAL_VERSION")
    # name rules
    occursin(r"^[A-Z][A-Za-z0-9]*$", pkg) || _lint_error!(lint, pkg, "name must be ASCII letters/digits with an uppercase start")
    length(pkg) >= 5 || _lint_error!(lint, pkg, "name shorter than 5 characters")
    occursin(r"julia"i, pkg) && _lint_error!(lint, pkg, "name contains \"julia\"")
    endswith(lowercase(pkg), "jl") && _lint_error!(lint, pkg, "name ends in \"jl\"")
    for sec in ("sources", "extras", "targets")
        haskey(proj, sec) && _lint_error!(lint, pkg, "registered Project.toml has [$sec]")
    end
    _check_compat!(lint, pkg, "Project.toml", proj)
    _check_allow_list!(lint, pkg, "Project.toml", path)
    lic = joinpath(dir, "LICENSE")
    if isfile(lic)
        occursin("MIT License", read(lic, String)) || _lint_error!(lint, pkg, "LICENSE is not the MIT text")
    else
        _lint_error!(lint, pkg, "no LICENSE in the package directory")
    end
    # test and docs environments
    for env in ("test", "docs")
        tp = joinpath(dir, env, "Project.toml")
        if !isfile(tp)
            env == "test" ? _lint_error!(lint, pkg, "no test/Project.toml") : _lint_warning!(lint, pkg, "no docs/Project.toml")
            continue
        end
        t = TOML.parsefile(tp)
        sources = get(t, "sources", Dict{String, Any}())
        src = get(sources, pkg, nothing)
        src isa AbstractDict && get(src, "path", nothing) == ".." ||
            _lint_error!(lint, pkg, "$env/Project.toml: [sources] $pkg = {path = \"..\"} missing")
        for n in keys(sources)
            n == pkg && continue
            _lint_error!(lint, pkg, "$env/Project.toml: [sources] $n (only the package itself may be sourced)")
        end
        _check_compat!(lint, pkg, "$env/Project.toml", t; skip = (pkg,))
        _check_allow_list!(lint, pkg, "$env/Project.toml", tp; self_ok = true)
    end
    # General
    if general !== nothing
        for (u, e) in general.pkgs
            if e.name == pkg
                if string(u) != uuid
                    _lint_error!(lint, pkg, "the name is registered in General under another UUID ($u)")
                elseif initial || placeholder
                    _lint_error!(lint, pkg, "already registered in General under this UUID")
                end
            elseif string(u) == uuid
                _lint_error!(lint, pkg, "the UUID is registered in General as $(e.name)")
            end
        end
        if initial
            close = sort!([(_dl_distance(pkg, e.name), e.name) for e in values(general.pkgs)
                           if e.name != pkg && _dl_distance(pkg, e.name) <= 2])
            if !isempty(close)
                msg = "name within DL distance 2 of General: $(join(("$n ($d)" for (d, n) in close), ", "))"
                haskey(lint.manual_merge, pkg) ?
                    _lint_warning!(lint, pkg, msg * " — manual merge: " * lint.manual_merge[pkg]) :
                    _lint_error!(lint, pkg, msg)
            end
        end
    else
        _lint_warning!(lint, pkg, "General is not reachable from this depot: name/UUID checks skipped")
    end
    return nothing
end

"""
    registration_violations(root; initial = true) -> NamedTuple{(:errors, :warnings)}

The General/AutoMerge rules that can be checked before registering, over the
workspace at the repository root `root` (a git checkout): one line per finding,
`"ERROR  <package>: …"` in `errors` and `"WARN   <package>: …"` in `warnings`.

With `initial` the rules of a package's FIRST registration apply to every
package General does not hold yet: its version is the standard initial version
0.1.0 and its name is not within Damerau–Levenshtein distance 2 of another's
(a package named in the workspace's `[manual_merge]` gets a warning instead).
A package General holds under its UUID is never a first registration, so those
rules never apply to it; `initial = false` drops them for every package.

A package at the workspace's `placeholder_version`
([`workspace_settings`](@ref)) is under construction: every structural rule
still applies to it, the version rule is replaced by a warning, and General
holding it is an error.

Per package directory:
- Project.toml: `name` is the directory, a UUID, a version; the name rules
  (ASCII letters and digits, an uppercase start, at least 5 characters, no
  "julia", no "jl" suffix).
- `[compat]` for every `[deps]` and `[weakdeps]` entry, standard libraries and
  julia included; each entry parses and is bounded (no `*`, no `>=`); a compat
  on a package of the same workspace names that package's own series — its
  current breaking series (`"0.5"` for 0.5.x, `"1"` for 1.x) or a raised floor
  inside it (`"0.5.2"`).
- No `[sources]`, `[extras]` or `[targets]` in the registered file; the test
  and docs projects source the package itself as `{path = ".."}` and nothing else.
- A LICENSE holding the MIT text inside the directory (the registered tarball
  is the directory only).
- The stack's dependency allow-list ([`stack_dependency_violations`](@ref),
  with the workspace's `stack_rows`) for the package, test and docs projects.

Workspace:
- No tracked `Manifest*.toml` anywhere; the root Project.toml is not a package.
- A root `codecov.yml`, where there is one, is [`codecov_config`](@ref) of the
  workspace's packages.

The General checks need the General registry in the depot; without it they are
skipped with a warning.
"""
function registration_violations(root::AbstractString; initial::Bool = true)
    settings = workspace_settings(root)
    packages = workspace_packages(root)
    root_project = TOML.parsefile(joinpath(root, "Project.toml"))
    series = Dict{String, String}()
    for p in packages
        f = joinpath(root, p, "Project.toml")
        isfile(f) && (series[p] = _series(VersionNumber(TOML.parsefile(f)["version"])))
    end
    stdlibs = Set{String}(string(u) for u in keys(Pkg.Types.stdlibs()))
    lint = _RegistrationLint(String(root), packages, series, stdlibs, settings.placeholder_version, settings.stack_rows,
        settings.manual_merge, String[], String[])
    (haskey(root_project, "name") || haskey(root_project, "uuid")) &&
        _lint_error!(lint, "<root>", "the workspace root must not be a package")
    tracked = readlines(`git -C $root ls-files -- 'Manifest*.toml' '**/Manifest*.toml' 'JuliaManifest*.toml' '**/JuliaManifest*.toml'`)
    isempty(tracked) || _lint_error!(lint, "<root>", "tracked Manifests: $(join(tracked, ", "))")
    codecov = joinpath(root, "codecov.yml")
    isfile(codecov) && read(codecov, String) != codecov_config(packages) &&
        _lint_error!(lint, "<root>", "codecov.yml is not the workspace's codecov_config (the `codecov` command of stack_ci prints it)")
    for d in packages
        isdir(joinpath(root, d)) || _lint_error!(lint, d, "listed in [workspace] but missing")
    end
    general = _general_registry()
    for pkg in packages
        _check_package!(lint, pkg, general, initial)
    end
    return (; errors = lint.errors, warnings = lint.warnings)
end

# -----------------------------------------------------------------------------
# The agent-docs policy over a checkout
# -----------------------------------------------------------------------------

function _tracked_text_files(root)
    files = readlines(`git -C $root ls-files`)
    return filter(files) do f
        p = joinpath(root, f)
        isfile(p) && !islink(p) || return false
        filesize(p) < 4 * 2^20 || return false
        bytes = read(p, min(filesize(p), 8192))
        return !(0x00 in bytes)          # skip binaries
    end
end

_pattern_lines(path) = Regex[Regex(strip(l)) for l in eachline(path) if !isempty(strip(l)) && !startswith(strip(l), '#')]

"""
    workspace_agent_doc_violations(root; every_file = false, extra = get(ENV, "AGENT_DOC_PATTERNS", nothing)) -> Vector{String}

The agent-docs policy over the whole checkout at `root`, once:
[`agent_doc_violations`](@ref) of the repository root, where `git ls-files`
sees every package (inside a package directory there is no `.git`, and that
check would walk the tree and pick up untracked per-machine files).

`extra` is the path of a private pattern file (one regular expression per
line; release tooling keeps it outside the repository). With `every_file`, its
patterns are also matched against EVERY tracked text file, one
`"file:line matches extra pattern #n"` per finding.
"""
function workspace_agent_doc_violations(root::AbstractString; every_file::Bool = false,
        extra = get(ENV, "AGENT_DOC_PATTERNS", nothing))
    ispath(joinpath(root, ".git")) ||
        throw(ArgumentError("workspace_agent_doc_violations: $root is not a git checkout (the check needs `git ls-files`)"))
    found = agent_doc_violations(root; extra)
    if every_file && extra !== nothing && !isempty(extra)
        pats = _pattern_lines(extra)
        for f in _tracked_text_files(root), (i, line) in enumerate(eachline(joinpath(root, f))), (k, re) in enumerate(pats)
            occursin(re, line) && push!(found, "$f:$i matches extra pattern #$k")
        end
    end
    return found
end

# -----------------------------------------------------------------------------
# A downstream project against this workspace
# -----------------------------------------------------------------------------

"""
    source_downstream(dir, root; compat = false, extra = Dict{String, String}()) -> Vector{String}

Point the downstream project in `dir` at the packages of the workspace at
`root`: sets `[sources] <Package> = {path = "<its directory in the workspace>"}`
in `dir`'s Project.toml for every workspace package it names in `[deps]`,
`[weakdeps]` or `[extras]`, plus each `name => path` of `extra` (another
downstream checkout). An existing entry is replaced. With `compat`, the
project's compat entries on workspace packages are set to `"0.1"`. Returns the
packages sourced.

It REWRITES the file through `TOML.print` (comments are lost): for a throwaway
checkout only (a nightly downstream job, a dry run).
"""
function source_downstream(dir::AbstractString, root::AbstractString; compat::Bool = false,
        extra::AbstractDict = Dict{String, String}())
    packages = workspace_packages(root)
    path = joinpath(dir, "Project.toml")
    proj = TOML.parsefile(path)
    named = Set{String}()
    for sec in ("deps", "weakdeps", "extras")
        union!(named, keys(get(proj, sec, Dict{String, Any}())))
    end
    self = get(proj, "name", "")
    srcs = get!(proj, "sources", Dict{String, Any}())
    done = String[]
    for p in packages
        p in named && p != self || continue
        # an existing entry (a sibling checkout, a url) is replaced: this workspace wins
        srcs[p] = Dict("path" => joinpath(root, p))
        push!(done, p)
    end
    for (k, v) in extra
        k in named && k != self || continue
        srcs[k] = Dict("path" => abspath(v))
        push!(done, k)
    end
    isempty(srcs) && delete!(proj, "sources")
    if compat
        c = get!(proj, "compat", Dict{String, Any}())
        for p in packages
            haskey(c, p) && (c[p] = "0.1")
        end
    end
    open(io -> TOML.print(io, proj; sorted = true), path, "w")
    return sort!(done)
end

# -----------------------------------------------------------------------------
# codecov.yml
# -----------------------------------------------------------------------------

"""
    codecov_config(packages) -> String

The text of a workspace's root `codecov.yml`: informational project and patch
statuses and one carried-forward flag per package of `packages` over its `src/`
and `ext/`. CI tests only the packages a change affects, so each flag carries
its last report forward until its package is tested again. The file is
generated, never edited: [`registration_violations`](@ref) reports a
`codecov.yml` that differs.
"""
function codecov_config(packages)
    io = IOBuffer()
    print(io, """
        # One flag per package: CI tests only the packages a change affects, so each
        # flag carries its last report forward until its package is tested again.
        coverage:
          status:
            project:
              default:
                informational: true
            patch:
              default:
                informational: true
        flag_management:
          default_rules:
            carryforward: true
          individual_flags:
        """)
    for p in packages
        print(io, "    - name: $p\n      paths:\n        - $p/src/\n        - $p/ext/\n")
    end
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# A sibling package's Documenter inventory
# -----------------------------------------------------------------------------

"""
    docs_local(variable) -> Bool

Whether the environment variable `variable` asks for a local docs build (its
value is `"1"`): a release's docs then build against this checkout's own builds
of their sibling packages, before those are published
([`docs_inventory`](@ref), [`docs_linkcheck_ignore`](@ref)).
"""
docs_local(variable::AbstractString) = get(ENV, variable, "") == "1"

"""
    docs_inventory(site, package; root = nothing, variable = nothing)

The inventory of the stack package `package` for DocumenterInterLinks: the URL
of its published `dev` docs under `site` (`"<site>/<package>/dev/"`, whose
`objects.inv` DocumenterInterLinks reads). With `root` (a workspace checkout)
and `variable` naming a set local-build variable ([`docs_local`](@ref)), the
pair `(url, path)` of that URL and the checkout's own
`<package>/docs/build/objects.inv`, where one exists.
"""
function docs_inventory(site::AbstractString, package::AbstractString; root = nothing, variable = nothing)
    url = "$site/$package/dev/"
    (root === nothing || variable === nothing || !docs_local(variable)) && return url
    inv = joinpath(root, package, "docs", "build", "objects.inv")
    return isfile(inv) ? (url, inv) : url
end

"""
    docs_linkcheck_ignore(site, variable) -> Vector{Regex}

The Documenter `linkcheck_ignore` rule of a local docs build
([`docs_local`](@ref)`(variable)`): every URL under `site`, since an `@extref`
into a sibling package built in the same checkout points at anchors not yet
published. Empty otherwise.
"""
docs_linkcheck_ignore(site::AbstractString, variable::AbstractString) =
    docs_local(variable) ? Regex[Regex("^" * replace(site, "." => "\\.") * "/")] : Regex[]

# -----------------------------------------------------------------------------
# The command line
# -----------------------------------------------------------------------------

"""
    stack_ci(args; root = pwd(), io = stdout, log = stderr) -> Int

The command line of a stack repository's CI over the workspace at `root`:
`args[1]` names the command, the rest are its arguments. Findings and results
print to `io`, the one-line summary to `log`; returns the exit code (0, or 1
when a check found something), so a caller ends with `exit(stack_ci(ARGS))`.

- `packages`: the workspace's packages ([`workspace_packages`](@ref)) as a JSON
  list; in GitHub Actions also `packages=<list>` in `\$GITHUB_OUTPUT`.
- `affected [<base> [<head>]]` (default `origin/main`, `HEAD`): the packages
  the change `<base>...<head>` affects ([`affected_packages`](@ref)), printed
  the same way.
- `registration-lint [--subsequent]`: [`registration_violations`](@ref), the
  warnings then the errors; `--subsequent` is `initial = false`.
- `agent-docs [--all]`: [`workspace_agent_doc_violations`](@ref); `--all` is
  `every_file = true`.
- `downstream <project dir> [--compat] [Name=path ...]`:
  [`source_downstream`](@ref).
- `codecov`: [`codecov_config`](@ref) of the workspace's packages.

An unknown command throws.
"""
function stack_ci(args; root::AbstractString = pwd(), io::IO = stdout, log::IO = stderr)
    isempty(args) && throw(ArgumentError("stack_ci: no command; one of packages, affected, registration-lint, agent-docs, downstream, codecov"))
    command, rest = args[1], args[2:end]
    if command == "packages"
        return _emit_packages(io, workspace_packages(root))
    elseif command == "affected"
        base = get(rest, 1, "origin/main")
        head = get(rest, 2, "HEAD")
        paths = readlines(`git -C $root diff --name-only $base...$head`)
        return _emit_packages(io, affected_packages(root, paths))
    elseif command == "registration-lint"
        found = registration_violations(root; initial = !("--subsequent" in rest))
        foreach(l -> println(io, l), found.warnings)
        foreach(l -> println(io, l), found.errors)
        println(log, "registration_lint: $(length(workspace_packages(root))) packages, $(length(found.errors)) error(s), $(length(found.warnings)) warning(s)")
        return isempty(found.errors) ? 0 : 1
    elseif command == "agent-docs"
        every_file = "--all" in rest
        extra = get(ENV, "AGENT_DOC_PATTERNS", nothing)
        found = workspace_agent_doc_violations(root; every_file, extra)
        foreach(l -> println(io, l), found)
        scope = every_file ? "agent docs + all tracked files" : "agent docs"
        println(log, "agent_docs: $(length(found)) finding(s) [$scope; private patterns: $(extra === nothing ? "none" : "from AGENT_DOC_PATTERNS")]")
        return isempty(found) ? 0 : 1
    elseif command == "downstream"
        isempty(rest) && throw(ArgumentError("stack_ci: downstream needs the project directory"))
        dir = abspath(rest[1])
        compat = "--compat" in rest
        extra = Dict{String, String}()
        for a in rest[2:end]
            occursin('=', a) || continue
            k, v = split(a, '='; limit = 2)
            extra[k] = abspath(v)
        end
        done = source_downstream(dir, root; compat, extra)
        println(log, "downstream: $(relpath(joinpath(dir, "Project.toml"), dirname(dir))) sources → $(join(done, ", "))",
            compat ? " (stack compat → 0.1)" : "")
        return 0
    elseif command == "codecov"
        print(io, codecov_config(workspace_packages(root)))
        return 0
    end
    throw(ArgumentError("stack_ci: unknown command $(repr(command)); one of packages, affected, registration-lint, agent-docs, downstream, codecov"))
end

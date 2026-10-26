# The General/AutoMerge rules that can be checked before registering.
#
#   julia --project=ci ci/registration_lint.jl [--subsequent]
#
# By default the rules of the FIRST registration apply: every version is the
# standard initial version 0.1.0 and no package's name or UUID is in General
# yet. After wave 1, CI passes --subsequent (drop those two rules).
#
# Per package directory:
#   - Project.toml: name = directory, a UUID, a version; name rules (ASCII
#     letters/digits, uppercase start, ≥ 5 characters, no "julia", no "jl"
#     suffix, Damerau–Levenshtein distance > 2 from every General name unless
#     the package is in MANUAL_MERGE)
#   - [compat] for every [deps] and [weakdeps] entry, stdlibs and julia included;
#     each entry parses and is bounded (no "*", no ">="); stack packages in the
#     workspace's stack series: STACK_COMPAT itself, or a raised floor inside it
#     ("0.1.3") when a package needs a sibling's newer feature (the Downgrade
#     workflow tests every package at its compat floors)
#   - no [sources], [extras] or [targets] in the registered file (test deps live
#     in test/Project.toml, which [sources]-links the package as path "..")
#   - a LICENSE (OSI: MIT) inside the directory — the registered tarball is
#     the subdir only
#   - the stack dependency allow-list (VisualGeometryToolkitCore's
#     stack_dependency_violations) for the package, test and docs projects
# Workspace:
#   - no tracked Manifest*.toml anywhere
#   - the root Project.toml is not a package (no name/uuid)
#
# Prints one line per finding; exits 1 when there is an error.

using Pkg, TOML
using VisualGeometryToolkitCore: stack_dependency_violations

const ROOT = dirname(@__DIR__)
const ROOT_PROJECT = TOML.parsefile(joinpath(ROOT, "Project.toml"))
const PACKAGES = filter(!=("ci"), ROOT_PROJECT["workspace"]["projects"])
const INITIAL_VERSION = v"0.1.0"
const STACK_COMPAT = "0.2"
# STACK_COMPAT, or a floor inside the same series: "0.2" or "0.2.N"
const STACK_SERIES = Regex("^" * replace(STACK_COMPAT, "." => "\\.") * raw"(\.\d+)?$")
# AutoMerge's name-similarity guideline would fail these; each needs a manual
# merge by a General maintainer, with the reason recorded here (DESIGN D9).
const MANUAL_MERGE = Dict("PoseLib" => "DL distance 2 from PDELib; the name of the upstream library and of PoseLib_jll")

const errors = String[]
const warnings = String[]
err(pkg, msg) = push!(errors, "ERROR  $pkg: $msg")
warn(pkg, msg) = push!(warnings, "WARN   $pkg: $msg")

# optimal string alignment distance, case-insensitive
function dl_distance(a::AbstractString, b::AbstractString)
    a, b = collect(lowercase(a)), collect(lowercase(b))
    m, n = length(a), length(b)
    d = zeros(Int, m + 1, n + 1)
    d[:, 1] = 0:m; d[1, :] = 0:n
    for i in 1:m, j in 1:n
        c = a[i] == b[j] ? 0 : 1
        d[i+1, j+1] = min(d[i, j+1] + 1, d[i+1, j] + 1, d[i, j] + c)
        if i > 1 && j > 1 && a[i] == b[j-1] && a[i-1] == b[j]
            d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + 1)
        end
    end
    return d[m+1, n+1]
end

function general_registry()
    for r in Pkg.Registry.reachable_registries()
        r.name == "General" && return r
    end
    return nothing
end

const STDLIB_UUIDS = Set(string(u) for u in keys(Pkg.Types.stdlibs()))
is_stdlib(uuid) = uuid in STDLIB_UUIDS

function check_compat(pkg, label, proj; stack_ok = true, skip = ())
    compat = get(proj, "compat", Dict{String, Any}())
    haskey(compat, "julia") || err(pkg, "$label: no julia compat")
    for sec in ("deps", "weakdeps"), (n, u) in get(proj, sec, Dict{String, Any}())
        n in skip && continue
        if !haskey(compat, n)
            err(pkg, "$label: [$sec] $n has no [compat] entry$(is_stdlib(u) ? " (stdlib)" : "")")
            continue
        end
    end
    for (n, c) in compat
        c isa AbstractString || (err(pkg, "$label: compat $n is not a string"); continue)
        (occursin('*', c) || occursin(">=", c) || occursin('≥', c)) && err(pkg, "$label: compat $n = \"$c\" is unbounded")
        try
            Pkg.Versions.semver_spec(c)
        catch e
            err(pkg, "$label: compat $n = \"$c\" does not parse ($(sprint(showerror, e)))")
        end
        if stack_ok && n in PACKAGES && !occursin(STACK_SERIES, c)
            err(pkg, "$label: stack compat $n = \"$c\", expected \"$STACK_COMPAT\" or a floor inside it (\"$STACK_COMPAT.N\")")
        end
    end
end

function check_allow_list(pkg, label, path; self_ok = false)
    found = try
        stack_dependency_violations(path, pkg)
    catch e
        [sprint(showerror, e)]
    end
    self_ok && filter!(f -> !endswith(f, ": $pkg"), found)
    foreach(f -> err(pkg, "$label: stack allow-list: $f"), found)
end

function check_package(pkg, general, initial)
    dir = joinpath(ROOT, pkg)
    path = joinpath(dir, "Project.toml")
    isfile(path) || return err(pkg, "no Project.toml")
    proj = TOML.parsefile(path)
    name = get(proj, "name", nothing)
    name == pkg || err(pkg, "name = $(repr(name)), expected the directory name")
    uuid = get(proj, "uuid", nothing)
    try
        Base.UUID(uuid)
    catch
        err(pkg, "uuid $(repr(uuid)) is not a UUID")
    end
    v = try
        VersionNumber(get(proj, "version", ""))
    catch
        err(pkg, "version $(repr(get(proj, "version", nothing))) is not a version"); nothing
    end
    initial && v !== nothing && v != INITIAL_VERSION && err(pkg, "version $v; the first registration is $INITIAL_VERSION")
    # name rules
    occursin(r"^[A-Z][A-Za-z0-9]*$", pkg) || err(pkg, "name must be ASCII letters/digits with an uppercase start")
    length(pkg) >= 5 || err(pkg, "name shorter than 5 characters")
    occursin(r"julia"i, pkg) && err(pkg, "name contains \"julia\"")
    endswith(lowercase(pkg), "jl") && err(pkg, "name ends in \"jl\"")
    for sec in ("sources", "extras", "targets")
        haskey(proj, sec) && err(pkg, "registered Project.toml has [$sec]")
    end
    check_compat(pkg, "Project.toml", proj)
    check_allow_list(pkg, "Project.toml", path)
    lic = joinpath(dir, "LICENSE")
    isfile(lic) ? (occursin("MIT License", read(lic, String)) || err(pkg, "LICENSE is not the MIT text")) :
        err(pkg, "no LICENSE in the package directory")
    # test and docs environments
    for env in ("test", "docs")
        tp = joinpath(dir, env, "Project.toml")
        if !isfile(tp)
            env == "test" ? err(pkg, "no test/Project.toml") : warn(pkg, "no docs/Project.toml")
            continue
        end
        t = TOML.parsefile(tp)
        src = get(get(t, "sources", Dict()), pkg, nothing)
        src isa AbstractDict && get(src, "path", nothing) == ".." ||
            err(pkg, "$env/Project.toml: [sources] $pkg = {path = \"..\"} missing")
        for (n, s) in get(t, "sources", Dict())
            n == pkg && continue
            err(pkg, "$env/Project.toml: [sources] $n (only the package itself may be sourced)")
        end
        check_compat(pkg, "$env/Project.toml", t; skip = (pkg,))
        check_allow_list(pkg, "$env/Project.toml", tp; self_ok = true)
    end
    # General
    if general !== nothing
        for (u, e) in general.pkgs
            if e.name == pkg
                string(u) == uuid ? (initial ? err(pkg, "already registered in General under this UUID") : nothing) :
                    err(pkg, "the name is registered in General under another UUID ($u)")
            elseif string(u) == uuid
                err(pkg, "the UUID is registered in General as $(e.name)")
            end
        end
        if initial
            close = sort!([(dl_distance(pkg, e.name), e.name) for e in values(general.pkgs) if e.name != pkg && dl_distance(pkg, e.name) <= 2])
            if !isempty(close)
                msg = "name within DL distance 2 of General: $(join(("$n ($d)" for (d, n) in close), ", "))"
                haskey(MANUAL_MERGE, pkg) ? warn(pkg, msg * " — manual merge: " * MANUAL_MERGE[pkg]) : err(pkg, msg)
            end
        end
    else
        warn(pkg, "General is not reachable from this depot: name/UUID checks skipped")
    end
end

function main(args)
    initial = !("--subsequent" in args)   # the default until the first wave merges
    (haskey(ROOT_PROJECT, "name") || haskey(ROOT_PROJECT, "uuid")) && err("<root>", "the workspace root must not be a package")
    tracked = readlines(`git -C $ROOT ls-files -- 'Manifest*.toml' '**/Manifest*.toml' 'JuliaManifest*.toml' '**/JuliaManifest*.toml'`)
    isempty(tracked) || err("<root>", "tracked Manifests: $(join(tracked, ", "))")
    for d in PACKAGES
        isdir(joinpath(ROOT, d)) || err(d, "listed in [workspace] but missing")
    end
    general = general_registry()
    for pkg in PACKAGES
        check_package(pkg, general, initial)
    end
    foreach(println, warnings)
    foreach(println, errors)
    println(stderr, "registration_lint: $(length(PACKAGES)) packages, $(length(errors)) error(s), $(length(warnings)) warning(s)")
    exit(isempty(errors) ? 0 : 1)
end

main(ARGS)

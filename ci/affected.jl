# The packages a change affects, with their dependents in the workspace.
#
#   julia --startup-file=no ci/affected.jl <base> [<head>]     # → JSON list on stdout
#
# A path under <Package>/ affects that package; anything else (root files,
# ci/, .github/, .githooks/) affects every package. The closure follows the
# workspace's own dependency edges: [deps], [weakdeps] and the test and docs
# environments of each package, so a VisualGeometryToolkitCore change runs all six and
# a VisualGeometryToolkitEstimation change runs it and VisualGeometryToolkitEval. Stdlib only.
# In GitHub Actions the list is also written to $GITHUB_OUTPUT as `packages`.

using TOML

const ROOT = dirname(@__DIR__)
const PACKAGES = filter(p -> p != "ci", TOML.parsefile(joinpath(ROOT, "Project.toml"))["workspace"]["projects"])

# package → the stack packages its environments name
function edges()
    out = Dict{String, Set{String}}()
    for p in PACKAGES
        s = Set{String}()
        for env in ("", "test", "docs")
            f = joinpath(ROOT, p, env, "Project.toml"); isfile(f) || continue
            t = TOML.parsefile(f)
            for sec in ("deps", "weakdeps"), n in keys(get(t, sec, Dict()))
                n in PACKAGES && n != p && push!(s, n)
            end
        end
        out[p] = s
    end
    return out
end

function affected(paths)
    direct = Set{String}()
    for path in paths
        top = first(splitpath(path))
        if top in PACKAGES
            push!(direct, top)
        else
            return sort(PACKAGES)        # workspace-level change
        end
    end
    deps = edges()
    closure = copy(direct)
    changed = true
    while changed
        changed = false
        for (p, ds) in deps
            p in closure && continue
            isempty(intersect(ds, closure)) && continue
            push!(closure, p); changed = true
        end
    end
    return sort!(collect(closure))
end

function main(args)
    base = get(args, 1, "origin/main"); head = get(args, 2, "HEAD")
    paths = readlines(`git -C $ROOT diff --name-only $base...$head`)
    pkgs = affected(paths)
    json = "[" * join(("\"$p\"" for p in pkgs), ",") * "]"
    println(json)
    out = get(ENV, "GITHUB_OUTPUT", "")
    isempty(out) || open(io -> println(io, "packages=", json), out, "a")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__()) && main(ARGS)

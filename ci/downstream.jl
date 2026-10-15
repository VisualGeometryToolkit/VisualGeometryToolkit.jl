# Point a downstream project at this workspace's packages.
#
#   julia --startup-file=no ci/downstream.jl <project dir> [--compat] [Name=path ...]
#
# Sets `[sources] <Package> = {path = "<abs path in this workspace>"}` in the
# Project.toml in <project dir> for every workspace package it names in
# [deps], [weakdeps] or [extras], plus each extra `Name=path` pair (another
# downstream checkout, e.g. RansacScoringDoneRight for BlobBoards). With
# --compat the stack compat entries are set to the workspace's ("0.1"), for a
# downstream repository that has not yet taken the 0.1 compat (DESIGN §6.4).
#
# It REWRITES the file through TOML.print (comments are lost): run it on a
# throwaway checkout only (the Nightly downstream job, the dry run). Stdlib only.

using TOML

const ROOT = dirname(@__DIR__)
const PACKAGES = filter(!=("ci"), TOML.parsefile(joinpath(ROOT, "Project.toml"))["workspace"]["projects"])

function main(args)
    dir = abspath(args[1])
    compat = "--compat" in args
    extra = Dict{String, String}()
    for a in args[2:end]
        occursin('=', a) || continue
        k, v = split(a, '='; limit = 2); extra[k] = abspath(v)
    end
    path = joinpath(dir, "Project.toml")
    proj = TOML.parsefile(path)
    named = Set{String}()
    for sec in ("deps", "weakdeps", "extras")
        union!(named, keys(get(proj, sec, Dict{String, Any}())))
    end
    srcs = get!(proj, "sources", Dict{String, Any}())
    done = String[]
    for p in PACKAGES
        p in named && p != get(proj, "name", "") || continue
        # an existing entry (a sibling checkout, a url) is replaced: this workspace wins
        srcs[p] = Dict("path" => joinpath(ROOT, p)); push!(done, p)
    end
    for (k, v) in extra
        k in named && k != get(proj, "name", "") || continue
        srcs[k] = Dict("path" => v); push!(done, k)
    end
    isempty(srcs) && delete!(proj, "sources")
    if compat
        c = get!(proj, "compat", Dict{String, Any}())
        for p in PACKAGES
            haskey(c, p) && (c[p] = "0.1")
        end
    end
    open(io -> TOML.print(io, proj; sorted = true), path, "w")
    println(stderr, "downstream: $(relpath(path, dirname(dir))) sources → $(join(sort(done), ", "))", compat ? " (stack compat → 0.1)" : "")
end

main(ARGS)

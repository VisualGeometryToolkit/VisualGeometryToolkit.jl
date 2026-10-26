#!/usr/bin/env julia
# =============================================================================
# migrate_pools.jl — every format-4 light pool under a directory tree, carried
# to format 5 IN PLACE, each by `Pools.migrate_pool`'s guarded pattern
# =============================================================================
#
#     julia --project=<VisualGeometryToolkitEval> scripts/migrate_pools.jl DIR            # dry run
#     julia --project=<VisualGeometryToolkitEval> scripts/migrate_pools.jl DIR --apply    # migrate
#
# A DRY RUN unless --apply: every format-4 pool is migrated to a temporary
# directory, read back through `load_pool` and compared bit for bit with the
# original (`migrate_pool`), and nothing is written. With --apply each verified
# format-5 file is staged in a fresh directory beside the original (the same
# filesystem) and renamed over it; the original is replaced only once its
# migration is verified, and only by that rename. A pool's residuals file is
# left as it is: it is bound by `pool_key`, which the migration keeps.
#
# One line per file: `verified`, `migrated`, `current` (already format 5),
# `skipped` (a file that opens and is not a light pool: a residuals file, a
# sample-starts file, another artifact) or `FAILED` with the reason (a file
# that does not open is FAILED, not skipped); a failure does not stop the others,
# and the exit code is 1 when any file failed. `migrate_pool` writes its
# verification copy under the system's temporary directory, so that directory
# needs room for the largest pool.

using JLD2: JLD2
using VisualGeometryToolkitEval.Pools: migrate_pool

# The light pool's format stamp; `nothing` for a file that opens and is not a
# light pool. A file that does not open (corrupt, truncated, unreadable) is an
# error, reported FAILED by the caller — never counted as skipped.
function _pool_format(path)
    endswith(path, ".residuals.jld2") && return nothing
    return JLD2.jldopen(f -> haskey(f, "pool_format") && haskey(f, "pairs") ? Int(f["pool_format"]) : nothing, path, "r")
end

"""
    migrate_pools(root; apply = false, io = stdout) -> NamedTuple

Every `.jld2` light pool under `root`: a format-4 one migrated to format 5 in
place (verified first, by `migrate_pool`; with `apply` the verified file is
renamed over the original), a format-5 one left as it is. Logs one line per
file to `io`; returns the counts `(; migrated, verified, current, skipped, failed)`.
"""
function migrate_pools(root::AbstractString; apply::Bool = false, io::IO = stdout)
    isdir(root) || throw(ArgumentError("$root: no such directory"))
    n = (migrated = 0, verified = 0, current = 0, skipped = 0, failed = 0)
    bump(n, k) = merge(n, NamedTuple{(k,)}((getfield(n, k) + 1,)))
    for (dir, _, files) in walkdir(root), f in sort(files)
        endswith(f, ".jld2") || continue
        path = joinpath(dir, f)
        v = try
            _pool_format(path)
        catch e
            println(io, "FAILED    ", path, ": does not open as a JLD2 file (", sprint(showerror, e), ")")
            n = bump(n, :failed)
            continue
        end
        if v === nothing
            println(io, "skipped   ", path, ": not a light pool")
            n = bump(n, :skipped)
            continue
        elseif v == 5
            println(io, "current   ", path)
            n = bump(n, :current)
            continue
        end
        try
            if !apply
                r = migrate_pool(path; out = joinpath(dir, ".pool_v5_dry_run"), dry_run = true)
                println(io, "verified  ", path, ": format $v → 5, ", r.n_pairs, " pairs (dry run; --apply to migrate)")
                n = bump(n, :verified)
                continue
            end
            stage = mktempdir(dir; prefix = ".pool_v5_")      # beside the original: one filesystem, so a rename
            try
                r = migrate_pool(path; out = stage)
                mv(r.path, path; force = true)                 # the verified format-5 file replaces the original
                println(io, "migrated  ", path, ": format $v → 5, ", r.n_pairs, " pairs")
                n = bump(n, :migrated)
            finally
                rm(stage; recursive = true, force = true)      # the staged residuals copy, and anything left on failure
            end
        catch e
            e isa ArgumentError || rethrow()
            println(io, "FAILED    ", path, ": ", e.msg)
            n = bump(n, :failed)
        end
    end
    println(io, "pools: ", n.migrated, " migrated, ", n.verified, " verified, ", n.current, " current, ",
        n.skipped, " skipped, ", n.failed, " failed", apply ? "" : " (dry run)")
    return n
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = filter(a -> a != "--apply", ARGS)
    (length(args) == 1 && !startswith(only(args), "--")) ||
        (println(stderr, "usage: julia --project=<VisualGeometryToolkitEval> scripts/migrate_pools.jl DIR [--apply]"); exit(2))
    n = migrate_pools(only(args); apply = "--apply" in ARGS)
    exit(n.failed == 0 ? 0 : 1)
end

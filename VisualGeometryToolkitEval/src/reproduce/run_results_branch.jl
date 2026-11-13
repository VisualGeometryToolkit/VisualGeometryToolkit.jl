# =============================================================================
# reproduce/run_results_branch.jl — the last stage of a launch of the runner (reproduce/runner.jl): the launch's
# paper-facing renders made a results branch of the paper (results_branches.jl), its RUN.toml carrying what the run
# manifest records. Runs in the project's environment (it loads this package), so it is not in the dependency-free
# runner file.
# =============================================================================

"""
    results_branch_assets(g::StageGraph) -> Vector{Pair{String,String}}

The renders of a launch as `results_branch` assets, `render => path in the paper project`, from the graph's
`[results_branch] assets`: every file of the directory of `from` (relative to the paper project `project`) whose
name matches its glob goes to the directory `to` under its own name; with `submitted_only = true` only those whose submitted copy is already there.
"""
function results_branch_assets(g::StageGraph)
    rb = g.results_branch
    proj = normpath(joinpath(g.workdir, get(rb, "project", ".")))
    out = Pair{String, String}[]
    for a in rb["assets"]
        for k in keys(a)
            k in ("from", "to", "submitted_only") || throw(ArgumentError("[results_branch] assets: unknown key $k (known: from, to, submitted_only)"))
        end
        from = normpath(a["from"])
        base = joinpath(proj, dirname(from))
        isdir(base) || continue
        for f in sort(readdir(base))
            rel = joinpath(dirname(from), f)
            (isfile(joinpath(proj, rel)) && _matches(from, rel)) || continue
            to = joinpath(a["to"], f)
            get(a, "submitted_only", false) && !isfile(joinpath(proj, to)) && continue
            push!(out, joinpath(proj, rel) => to)
        end
    end
    return out
end

"""
    results_branch_of_run(manifest; scratch_root = nothing) -> NamedTuple

The results branch of the launch a run manifest records ([`results_branch`](@ref)): the renders of the graph's
`[results_branch] assets` ([`results_branch_assets`](@ref)), the commit the launch ran (`-dirty` when it was), every
stage's run key, the data tiers (the git-tree-sha1 of each entry of `[results_branch] artifacts`, an Artifacts.toml),
the site as the machine, the GPUs this job sees, and every Slurm job of the launch. The branch name is added to the manifest (`results_branch`), which the runner keeps when it rewrites it.
"""
function results_branch_of_run(manifest::AbstractString; scratch_root = nothing)
    m = TOML.parsefile(manifest)
    g = load_stage_graph(joinpath(m["workdir"], m["graph"]))
    assets = results_branch_assets(g)
    isempty(assets) && throw(ArgumentError("results_branch_of_run: no render matches [results_branch] assets of $(g.path)"))
    recs = m["stages"]
    tiers = Pair{String, String}[]
    arts = get(g.results_branch, "artifacts", "")
    if !isempty(arts)
        for (name, e) in TOML.parsefile(joinpath(g.workdir, get(g.results_branch, "project", "."), arts))
            e isa AbstractDict && haskey(e, "git-tree-sha1") && push!(tiers, name => e["git-tree-sha1"])
        end
    end
    jobs = sort(unique(String[r["job_id"] for (_, r) in recs if !isempty(get(r, "job_id", ""))]))
    commit = m["commit"] * (m["dirty"] ? "-dirty" : "")
    proj = normpath(joinpath(g.workdir, get(g.results_branch, "project", ".")))
    r = results_branch(proj, assets; commit, run_keys = sort([String(rec["key"]) for (_, rec) in recs]), tiers,
                       machine = m["site"], slurm_jobs = jobs, scratch_root)
    m["results_branch"] = r.branch
    _write_atomic(io -> TOML.print(io, m; sorted = true), manifest)      # as the runner writes its manifests
    println("results branch ", r.branch, " (worktree ", r.worktree, ")")
    return r
end

# =============================================================================
# results_branches.jl — every full reproduction of a paper's results becomes a
# branch of the paper's repository, reviewed before anything reaches the paper:
#
#   results_branch(dir, assets; …)        the branch results/<UTC stamp>-<commit>,
#                                         in a worktree of its own: the paper-facing
#                                         assets at the paper's paths, RUN.toml (the
#                                         run's manifest) and COMPARISON.md (every
#                                         number against the base branch)
#   accept_results(dir, branch; …)        the assets into the base branch as one
#                                         merge commit (deploy_assets' sign-off),
#                                         tagged accepted/<stamp>
#   remove_results_worktree(dir, branch)  the worktree removed; the branch kept, or
#                                         deleted on request (a rejected run)
#
# The user's checkout is never touched by a run; acceptance moves the base
# branch, and a checkout of it fast-forwards to the merge. Heavy results stay
# in the results root, referenced by run key: only macros, tables and figures
# are committed.
#
# Git through its command line: libgit2 (LibGit2) has no worktrees. Not part of
# the dependency-free files (it reads TOML and hashes with Core's sha256_hex).
# =============================================================================

"""
    RESULTS_MANIFEST

The manifest a results branch carries at the paper project's root:
`"RUN.toml"` ([`results_branch`](@ref)).
"""
const RESULTS_MANIFEST = "RUN.toml"

"""
    RESULTS_COMPARISON

The comparison report a results branch carries at the paper project's root:
`"COMPARISON.md"` ([`results_branch`](@ref)).
"""
const RESULTS_COMPARISON = "COMPARISON.md"

const _RESULTS_PREFIX = "results/"
const _BASE_FIGURES = joinpath("comparison", "base")     # the base's copy of each changed PNG, for the side-by-side

# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

# git in `dir`; its stdout (raw bytes with `bytes = true`, else stripped text).
# A failure is an error naming the command and git's own message.
function _git(dir::AbstractString, args...; bytes::Bool = false)
    out, err = IOBuffer(), IOBuffer()
    ok = success(pipeline(`git -C $dir $args`; stdout = out, stderr = err))
    ok || error("git $(join(args, ' ')) (in $dir): $(strip(String(take!(err))))")
    return bytes ? take!(out) : String(strip(String(take!(out))))
end

# git that commits (commit, merge): an identity of its own when the repository lacks a name or an email.
_has_config(dir, key) = success(pipeline(`git -C $dir config $key`; stdout = devnull, stderr = devnull))
_git_commit(dir::AbstractString, args...) =
    _has_config(dir, "user.name") && _has_config(dir, "user.email") ? _git(dir, args...) :
    _git(dir, "-c", "user.name=VisualGeometryToolkitEval", "-c", "user.email=vgtk@localhost", args...)

# A cleanup step after a failure or at the end: its own failure is a warning,
# never an error that would hide the one being handled.
function _quietly(f, what::AbstractString)
    try
        f()
    catch err
        @warn "results branches: $what failed" exception = err
    end
    return nothing
end

# The worktree that has `branch` checked out, or nothing.
function _branch_worktree(top::AbstractString, branch::AbstractString)
    path = nothing
    for line in eachsplit(_git(top, "worktree", "list", "--porcelain"), '\n')
        startswith(line, "worktree ") && (path = line[10:end])
        line == "branch refs/heads/" * branch && return path
    end
    return nothing
end

# The repository's top level and the project directory relative to it ("" at the top).
function _repository(dir::AbstractString)
    top = realpath(_git(dir, "rev-parse", "--show-toplevel"))
    rel = relpath(realpath(dir), top)
    return top, rel == "." ? "" : rel
end

_in_project(prefix, path) = isempty(prefix) ? path : joinpath(prefix, path)

function _check_results_branch(branch::AbstractString)
    startswith(branch, _RESULTS_PREFIX) ||
        throw(ArgumentError("$(repr(branch)) is not a results branch ($(_RESULTS_PREFIX)<stamp>-<commit>)"))
    return branch
end

# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

# A data tier's hash: an artifact's git-tree-sha1 (40 hex digits) as given; a
# directory's TIER.toml, MANIFEST.tsv or MANIFEST (the first it holds) by its
# sha256; a file by its sha256.
function _tier_hash(name::AbstractString, source::AbstractString)
    occursin(r"^[0-9a-f]{40}$", source) && return Dict("hash" => String(source), "of" => "artifact git-tree-sha1")
    file = isdir(source) ? something(findfirst(isfile, joinpath.(source, ("TIER.toml", "MANIFEST.tsv", "MANIFEST"))), 0) : 0
    isdir(source) && file == 0 &&
        throw(ArgumentError("data tier $name: $source holds no TIER.toml, MANIFEST.tsv or MANIFEST"))
    path = file > 0 ? joinpath(source, ("TIER.toml", "MANIFEST.tsv", "MANIFEST")[file]) : source
    isfile(path) || throw(ArgumentError("data tier $name: $(repr(source)) is no git-tree-sha1, directory or file"))
    return Dict("hash" => open(sha256_hex, path), "of" => "$(basename(path)) sha256")
end

# The GPUs this process sees, by name: nvidia-smi's (restricted to
# CUDA_VISIBLE_DEVICES when that lists indices), else CUDA.jl's when it is
# loaded; none when neither is there. Never required.
function _detected_gpus()
    smi = Sys.which("nvidia-smi")
    if smi !== nothing
        out = try readchomp(pipeline(`$smi --query-gpu=name --format=csv,noheader`; stderr = devnull)) catch; "" end
        names = String[strip(l) for l in split(out, '\n') if !isempty(strip(l))]
        visible = split(get(ENV, "CUDA_VISIBLE_DEVICES", ""), ','; keepempty = false)
        idx = tryparse.(Int, visible)
        if !isempty(names)
            (isempty(idx) || any(isnothing, idx) || !all(i -> 0 <= i < length(names), idx)) && return names
            return names[idx .+ 1]
        end
    end
    cuda = get(Base.loaded_modules, Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"), nothing)
    cuda === nothing && return String[]
    return try
        String[string(Base.invokelatest(getfield(cuda, :name), d)) for d in Base.invokelatest(getfield(cuda, :devices))]
    catch
        String[]
    end
end

# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

# A typeset number (tex_number's forms: a sign, thin-space thousands, decimals,
# `\%`, a `\,unit`) as its exact value, its decimals and its suffix (the
# percent sign and the unit); nothing for any other body (a range, a label, a
# math expression).
function _typeset_value(body::AbstractString)
    m = match(r"^(-?)(\d{1,3}(?:\\,\d{3})+|\d+)(?:\.(\d+))?((?:\\%)?(?:\\,(?!\d).*)?)$", body)
    m === nothing && return nothing
    frac = _cap(m, 3)
    q = parse(BigInt, replace(_cap(m, 2), "\\," => "") * frac)
    return ((isempty(_cap(m, 1)) ? q : -q) // big(10)^length(frac), length(frac), _cap(m, 4))
end

# The direction each macro of a tex text records in its provenance comment (`; better=…`).
_directions(text::AbstractString) =
    Dict(name => _cap(m, 1) for (name, line) in _macro_provenance(text)
         for m in (match(r";\s*better=(\w+)", line),) if m !== nothing)

const _VERDICT_ORDER = ("DEGRADED", "IMPROVED", "CHANGED", "NEW", "REMOVED", "UNCHANGED")

function _number_row(name, file, old, new, better)
    row(delta, verdict) = (; name, file, base = something(old, ""), new = something(new, ""), delta, better, verdict)
    old === nothing && return row("", "NEW")
    new === nothing && return row("", "REMOVED")
    a, b = _typeset_value(old), _typeset_value(new)
    if a === nothing || b === nothing || a[3] != b[3]          # not numbers, or a unit or percent sign changed
        return row("", old == new ? "UNCHANGED" : "CHANGED")
    end
    d = b[1] - a[1]
    decimals = max(a[2], b[2])
    iszero(d) && return row("0", "UNCHANGED")
    delta = (d > 0 ? "+" : "") * _decimal_string(round(BigInt, d * big(10)^decimals, RoundNearest), decimals)
    verdict = better == "lower" ? (d < 0 ? "IMPROVED" : "DEGRADED") :
              better == "higher" ? (d > 0 ? "IMPROVED" : "DEGRADED") : "CHANGED"
    return row(delta, verdict)
end

# Every macro of a numbers file, its base value against its new one. The
# direction is the new file's provenance comment's, else the base's (reported
# as the base's); one the comparison does not know (a hand-written
# `better=up`) gives no verdict and is reported as unknown.
function _compare_numbers(file::AbstractString, old_text::AbstractString, new_text::AbstractString)
    old, new = Dict(_macro_values(old_text)), Dict(_macro_values(new_text))
    old_dirs, new_dirs = _directions(old_text), _directions(new_text)
    return map(sort!(collect(union(keys(old), keys(new))))) do n
        b = get(new_dirs, n, nothing)
        from_base = b === nothing && haskey(old_dirs, n)
        from_base && (b = old_dirs[n])
        known = b in ("lower", "higher", "none")
        shown = b === nothing ? "" : b * (known ? "" : " (unknown)") * (from_base ? " (the base's)" : "")
        r = _number_row(n, file, get(old, n, nothing), get(new, n, nothing), known ? b : "")
        merge(r, (; better = shown))
    end
end

_md(s) = replace(s, "|" => "\\|")
_code(s) = isempty(s) ? "" : "`" * _md(s) * "`"

function _write_comparison(path, run, rows, numbers, pngs)
    n_of(v) = count(r -> r.verdict == v, numbers)
    open(path, "w") do io
        println(io, "# Results ", run.stamp, " of ", run.commit[1:7], " against `", run.base, "` at ", run.base_commit[1:7])
        println(io)
        println(io, "Compared against `", run.base, "` at ", run.base_commit, ". If `", run.base, "` has moved on since,",
                " this comparison is stale: `accept_results` warns and names the accepted assets the base changed.")
        println(io)
        println(io, "Branch `", run.branch, "`. Every number of the paper's number macros against the base branch;",
                " a number states the direction in which it improves (`add_number!(…; better)`), else a change is CHANGED.",
                " Accept with `accept_results`; remove the worktree with `remove_results_worktree`.")
        println(io)
        println(io, join(("$v $(n_of(v))" for v in _VERDICT_ORDER), " · "))
        println(io)
        println(io, "## Numbers")
        println(io)
        if isempty(numbers)
            println(io, "No number macros.")
        else
            println(io, "| macro | file | base | new | Δ | better | verdict |")
            println(io, "|---|---|---|---|---|---|---|")
            for r in sort(numbers; by = r -> (findfirst(==(r.verdict), _VERDICT_ORDER), r.file, r.name))
                println(io, "| `\\", r.name, "` | ", _md(r.file), " | ", _code(r.base), " | ", _code(r.new), " | ", r.delta,
                        " | ", r.better, " | ", r.verdict, " |")
            end
        end
        println(io)
        println(io, "## Files")
        println(io)
        println(io, "| asset | against the base |")
        println(io, "|---|---|")
        for r in rows
            println(io, "| ", _md(r.asset), " | ", _md(_verdict(r)), " |")
        end
        isempty(pngs) && return
        println(io)
        println(io, "## Figures side by side")
        for p in pngs
            println(io)
            println(io, "### ", p)
            println(io)
            println(io, "| base | new |")
            println(io, "|---|---|")
            println(io, "| ![base](", joinpath(_BASE_FIGURES, p), ") | ![new](", p, ") |")
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# A run's branch
# ---------------------------------------------------------------------------

_default_scratch(top) = get(ENV, "VGTK_RESULTS_WORKTREES", joinpath(dirname(top), basename(top) * "-results"))

"""
    results_branch(dir, assets; base = <the checked-out branch>, commit = <HEAD>,
                   scratch_root = ENV["VGTK_RESULTS_WORKTREES"] or "<repository>-results" beside it,
                   stamp = now(UTC), run_keys = String[], tiers = Pair[], machine = gethostname(),
                   gpus = <the GPUs seen: nvidia-smi, else CUDA.jl>, slurm_jobs = [ENV["SLURM_JOB_ID"]] or String[],
                   wall_times = Pair[]) -> NamedTuple

Make a full reproduction of the paper project at `dir` (a git repository or a
directory inside one) a branch of its repository for review:
`results/<UTC yyyymmdd_HHMMSS>-<short commit>`, `commit` the repository HEAD
the run evaluated (or any revision, resolved; DrWatson's `<sha>-dirty` record
marks the run dirty), created from `base` in a worktree of its own under
`scratch_root`. The user's checkout is not touched. A failure after the
worktree is created removes the worktree and the branch, and the error is
rethrown.

`assets` are the reproduction's paper-facing renders, `source => path` with
`path` where the paper reads it, relative to `dir` (the number macro files
[`write_numbers`](@ref) and [`gather_numbers`](@ref) write, the generated
tables, the figures). The branch's one commit holds them and, at `dir`'s place
in the repository:

- [`RESULTS_MANIFEST`](@ref) (`RUN.toml`): the branch, the stamp, the code
  `commit` (and whether the checkout was `dirty`), the `base` and its commit,
  the `run_keys` (the heavy results stay in the results root under them), each
  data tier's hash (`tiers`, `name => source`: an artifact's git-tree-sha1, a
  tier directory's `TIER.toml`, `MANIFEST.tsv` or `MANIFEST` sha256, or a
  file's sha256), the `machine`, `gpus`, `slurm_jobs`, `wall_times`
  (`name => seconds`) and the assets, and `missing`: which of `tiers`, `gpus`
  and `wall_times` the run did not give (each a warning);
- [`RESULTS_COMPARISON`](@ref) (`COMPARISON.md`): every macro of every number
  file against the base branch's copy (read by [`tex_macro_values`](@ref)'s
  parser) with its delta and IMPROVED, DEGRADED or UNCHANGED by the direction
  its provenance comment records (`add_number!(…; better)`), CHANGED without
  one, NEW or REMOVED; every asset's verdict against the base
  ([`verify_assets`](@ref)' rows); each changed PNG beside the base's copy
  (kept under `comparison/base/`).

Returns `(; branch, worktree, commit, numbers, assets)`: `commit` the branch's,
`numbers` the comparison's rows (`name`, `file`, `base`, `new`, `delta`,
`better`, `verdict`), `assets` the verify rows. [`accept_results`](@ref)
brings the assets into the base; [`remove_results_worktree`](@ref) removes the
worktree.
"""
function results_branch(dir::AbstractString, assets::AbstractVector{<:Pair};
        base::Union{Nothing, AbstractString} = nothing, commit::Union{Nothing, AbstractString} = nothing,
        scratch_root::Union{Nothing, AbstractString} = nothing, stamp::DateTime = now(UTC),
        run_keys::AbstractVector{<:AbstractString} = String[], tiers::AbstractVector{<:Pair} = Pair{String, String}[],
        machine::AbstractString = gethostname(), gpus::AbstractVector{<:AbstractString} = _detected_gpus(),
        slurm_jobs::AbstractVector{<:AbstractString} = haskey(ENV, "SLURM_JOB_ID") ? [ENV["SLURM_JOB_ID"]] : String[],
        wall_times::AbstractVector{<:Pair} = Pair{String, Float64}[])
    isempty(assets) && throw(ArgumentError("results_branch: no assets"))
    top, prefix = _repository(dir)
    paths = [String(p) for (_, p) in assets]
    for ((src, p), q) in zip(assets, paths)
        isfile(src) || throw(ArgumentError("results_branch: no render $src"))
        (isabspath(q) || startswith(normpath(q), "..")) && throw(ArgumentError("results_branch: $q is not a path inside $dir"))
    end
    allunique(paths) || throw(ArgumentError("results_branch: an asset path is given twice"))
    # everything the manifest records is checked before anything is created
    tier_hashes = Dict(String(n) => _tier_hash(String(n), String(s)) for (n, s) in tiers)
    wall = Dict(String(k) => (v isa Real ? Float64(v) :
                throw(ArgumentError("results_branch: wall time $k = $(repr(v)) is not a number of seconds"))) for (k, v) in wall_times)
    lacking = [k for (k, given) in ("tiers" => tiers, "gpus" => gpus, "wall_times" => wall_times) if isempty(given)]
    isempty(lacking) ||
        @warn "results_branch: RUN.toml records no $(join(lacking, ", ")): the run cannot be tied to its data and machine later"
    base = something(base, _git(top, "symbolic-ref", "--short", "HEAD"))
    base_commit = _git(top, "rev-parse", "--verify", "refs/heads/" * base * "^{commit}")
    given = something(commit, "HEAD")
    commit = _git(top, "rev-parse", "--verify", chopsuffix(given, "-dirty") * "^{commit}")
    dirty = endswith(given, "-dirty") ||
            (commit == _git(top, "rev-parse", "HEAD") && !isempty(_git(top, "status", "--porcelain", "--untracked-files=no")))
    stamp_s = Dates.format(stamp, "yyyymmdd_HHMMSS")
    branch = _RESULTS_PREFIX * stamp_s * "-" * commit[1:7]
    root = abspath(something(scratch_root, _default_scratch(top)))
    worktree = joinpath(mkpath(root), replace(branch, "/" => "-"))
    ispath(worktree) && throw(ArgumentError("results_branch: $worktree exists"))
    staged = mktempdir()                             # the renders at the paper's paths, beside the base's copies
    try
        _git(top, "worktree", "add", "-q", "-b", branch, worktree, base_commit)
    catch
        rm(staged; recursive = true, force = true)
        rethrow()
    end
    try
        return _fill_results_branch(top, prefix, worktree, staged, assets, paths;
            run = (; branch, stamp = stamp_s, commit, dirty, base, base_commit),
            manifest = Dict{String, Any}("run_keys" => collect(String, run_keys), "tiers" => tier_hashes,
                "machine" => String(machine), "gpus" => collect(String, gpus), "slurm_jobs" => collect(String, slurm_jobs),
                "wall_times" => wall, "missing" => lacking, "assets" => [_in_project(prefix, p) for p in paths]))
    catch
        _quietly(() -> _git(top, "worktree", "remove", "--force", worktree), "removing the half-made worktree $worktree")
        _quietly(() -> _git(top, "branch", "-q", "-D", branch), "deleting the half-made branch $branch")
        rethrow()
    finally
        rm(staged; recursive = true, force = true)
    end
end

# The branch's one commit: the renders, the manifest and the comparison.
function _fill_results_branch(top::AbstractString, prefix::AbstractString, worktree::AbstractString, staged::AbstractString,
                              assets, paths; run, manifest)
    proj = joinpath(worktree, prefix)
    for (src, p) in zip(first.(assets), paths)
        cp(src, joinpath(mkpath(dirname(joinpath(staged, p))), basename(p)); force = true)
    end
    rows = [_verify_row(staged, proj, p) for p in paths]
    numbers = NamedTuple[]
    pngs = String[]
    for (p, r) in zip(paths, rows)
        old, new = joinpath(proj, p), joinpath(staged, p)
        if _is_tex(p)
            old_text = isfile(old) ? read(old, String) : ""
            new_text = read(new, String)
            isempty(_macro_values(old_text)) && isempty(_macro_values(new_text)) ||
                append!(numbers, _compare_numbers(p, old_text, new_text))
        elseif r.submitted && !r.identical && lowercase(splitext(p)[2]) == ".png"
            cp(old, joinpath(mkpath(dirname(joinpath(proj, _BASE_FIGURES, p))), basename(p)))
            push!(pngs, p)
        end
        cp(new, joinpath(mkpath(dirname(old)), basename(p)); force = true)
    end

    merge!(manifest, Dict{String, Any}(String(k) => v for (k, v) in pairs(run)))
    open(io -> TOML.print(io, manifest; sorted = true), joinpath(proj, RESULTS_MANIFEST), "w")
    _write_comparison(joinpath(proj, RESULTS_COMPARISON), run, rows, numbers, pngs)

    counts = join(("$(count(r -> r.verdict == v, numbers)) $(lowercase(v))" for v in _VERDICT_ORDER[1:3]), ", ")
    _git(worktree, "add", "-f", "--", (_in_project(prefix, p) for p in [paths; RESULTS_MANIFEST; RESULTS_COMPARISON])...)
    isempty(pngs) || _git(worktree, "add", "-f", "--", _in_project(prefix, _BASE_FIGURES))
    _git_commit(worktree, "commit", "-q", "--no-verify", "-m",
                "Results $(run.stamp) of $(run.commit[1:7]) against $(run.base): $counts\n\n" *
                "See $(_in_project(prefix, RESULTS_COMPARISON)).")
    println("results branch ", run.branch, " in ", worktree, " (", counts, ")")
    return (; run.branch, worktree, commit = _git(worktree, "rev-parse", "HEAD"), numbers, assets = rows)
end

# ---------------------------------------------------------------------------
# Acceptance and removal
# ---------------------------------------------------------------------------

"""
    accept_results(dir, branch; replace_submitted, scratch_root = <results_branch's>) -> NamedTuple

Bring the assets of the results `branch` ([`results_branch`](@ref)) of the
paper project at `dir` into its base branch as one merge commit, and tag it
`accepted/<stamp>`. The assets go through [`deploy_assets`](@ref): an asset
the base holds a different copy of is replaced only with the sign-off
`replace_submitted` (`true` for every asset, or a collection of the assets'
paths relative to `dir`); without it the acceptance is refused and nothing
changes. The merge is made in a worktree of its own under `scratch_root`
(removed at the end, whatever happens); the tag is made on it, then the base
moves to it: a checkout of the base fast-forwards (one with local changes to an
accepted asset refuses: the tag is deleted again and the base stays; an
unrelated local change survives). That fast-forward is a `git merge` in the
user's checkout, so it runs the checkout's own `post-merge` hook; the commits
`accept_results` makes skip hooks (`--no-verify`). A base no checkout holds moves by
`update-ref`, so a checkout on another branch is untouched. Only the assets are
merged, never the branch's manifest or comparison. A base that moved since the
run (the manifest's `base_commit`) is a warning naming the accepted assets it
changed: the branch's comparison is against the older base.

Returns `(; commit, tag, deployed)`: the merge commit, the tag and the assets
copied (repository paths).
"""
function accept_results(dir::AbstractString, branch::AbstractString;
        replace_submitted::Union{Bool, AbstractVector{<:AbstractString}, AbstractSet{<:AbstractString}},
        scratch_root::Union{Nothing, AbstractString} = nothing)
    _check_results_branch(branch)
    top, prefix = _repository(dir)
    manifest = TOML.parse(String(_git(top, "show", branch * ":" * _in_project(prefix, RESULTS_MANIFEST); bytes = true)))
    base, tag = manifest["base"], "accepted/" * manifest["stamp"]
    isempty(_git(top, "tag", "--list", tag)) || error("accept_results: the tag $tag exists; $branch is accepted")
    old = _git(top, "rev-parse", "--verify", "refs/heads/" * base)
    assets = String.(manifest["assets"])
    if old != manifest["base_commit"]
        changed = split(_git(top, "diff", "--name-only", manifest["base_commit"], old, "--", assets...), '\n'; keepempty = false)
        @warn "accept_results: $base moved since the run ($(manifest["base_commit"][1:7]) to $(old[1:7])); the comparison " *
              "on $branch is against the older base; the base changed " * (isempty(changed) ? "no accepted asset" : join(changed, ", "))
    end
    signed = replace_submitted isa Bool ? replace_submitted : Set(_in_project(prefix, p) for p in replace_submitted)
    root = mkpath(abspath(something(scratch_root, _default_scratch(top))))
    merging = joinpath(root, "accept-" * manifest["stamp"])
    renders = merging * "-renders"                   # the branch's assets, by repository path
    for p in (merging, renders)
        ispath(p) && throw(ArgumentError("accept_results: $p exists (a previous acceptance's; remove it)"))
    end
    deployed = String[]
    try
        for a in assets
            write(joinpath(mkpath(dirname(joinpath(renders, a))), basename(a)), _git(top, "show", branch * ":" * a; bytes = true))
        end
        _git(top, "worktree", "add", "-q", "--detach", merging, old)
        # a merge of the branch that keeps the base's tree; then the assets alone, through the sign-off
        _git_commit(merging, "merge", "-q", "--no-ff", "--no-commit", "-s", "ours", branch)
        for (d, group) in Iterators.map(d -> d => filter(a -> dirname(a) == d, assets), unique(dirname.(assets)))
            names = basename.(group)
            sign = signed isa Bool ? signed : [basename(a) for a in group if a in signed]
            for n in deploy_assets(joinpath(renders, d), joinpath(merging, d), names; replace_submitted = sign)
                push!(deployed, joinpath(d, n))
            end
        end
        _git(merging, "add", "-f", "--", assets...)
        _git_commit(merging, "commit", "-q", "--no-verify", "-m",
                    "Accept $branch into $base\n\nThe results of $(manifest["commit"][1:7]) at $(manifest["stamp"]) " *
                    "($(_in_project(prefix, RESULTS_COMPARISON)) on the branch).")
        merged = _git(merging, "rev-parse", "HEAD")
        _git(top, "tag", tag, merged)                # the tag first: a base that cannot move takes it back
        try
            checkout = _branch_worktree(top, base)
            checkout === nothing ? _git(top, "update-ref", "refs/heads/" * base, merged, old) :
                                   _git(checkout, "merge", "-q", "--ff-only", merged)
        catch
            _quietly(() -> _git(top, "tag", "-d", tag), "deleting the tag $tag again")
            rethrow()
        end
        println("accepted ", branch, " into ", base, " as ", merged[1:7], " (", tag, ")")
        return (; commit = merged, tag, deployed)
    finally
        ispath(merging) && _quietly(() -> _git(top, "worktree", "remove", "--force", merging), "removing the worktree $merging")
        _quietly(() -> rm(renders; recursive = true, force = true), "removing $renders")
    end
end

"""
    remove_results_worktree(dir, branch; delete_branch = false) -> Nothing

Remove the worktree of the results `branch` ([`results_branch`](@ref)) of the
paper project at `dir`, after its acceptance ([`accept_results`](@ref)) or as
its rejection. The branch is kept, so a rejected run stays reviewable, unless
`delete_branch = true`. A worktree with uncommitted changes is refused. Any
branch not named `results/…` is an `ArgumentError`.
"""
function remove_results_worktree(dir::AbstractString, branch::AbstractString; delete_branch::Bool = false)
    _check_results_branch(branch)
    top, _ = _repository(dir)
    worktree = _branch_worktree(top, branch)
    worktree === nothing || _git(top, "worktree", "remove", worktree)
    delete_branch && _git(top, "branch", "-q", "-D", branch)
    return nothing
end

# =============================================================================
# data_policy.jl — the data policy as a checkable API, for each project's tests
# in one line:
#
#   @test isempty(data_policy_violations(projectdir(); allow = DATA_POLICY_ALLOW))
#
#   D1  no machine paths (/raid, /home, /mnt, /cluster, ~/…, $HOME/…, a host
#       name such as dgx01 outside a --site= argument, …) in code, launchers,
#       tests, examples, configs, docs, tracked provenance or the Markdown of
#       the root and data/ (README.md, REPRODUCE.md): every path derives
#       from DrWatson's projectdir/datadir, data_root(), results_root() or an
#       environment variable, and a machine's own paths live in
#       VisualGeometryToolkitEval's site profiles (configs/sites/, VGTK_SITE)
#       and nowhere else.
#   D2  no writes into an input root: data_root() (and a project's flat data
#       roots) are read-only; outputs go under results_root(). The static scan
#       flags a write call or an output directory spelled under the data root;
#       the writers' runtime guard (`assert_not_input`) catches the rest.
#   D3  keyed output directories: an output directory is named by the run key
#       of its effective config — never a wall-clock stamp, a hand-bumped
#       version literal (`…-v5`), a `-smoke` sibling of a keyed directory, or
#       an unkeyed file every run rewrites.
#   D4  no `artifact"…"` in code: published data reach the drivers through
#       data_root() (the artifact view, DataArtifacts `fetch_artifacts!`),
#       never through Pkg's string macro, which ties code to one binding and
#       bypasses the author's primary copy.
#   D5  no artifact or download API outside DataArtifacts: published inputs and
#       model weights are installed by VisualGeometryToolkitCore's artifact
#       store (DataArtifacts `fetch_data!`/`fetch_artifacts!`, packed by
#       `pack_artifacts`), never by a project's own `install_artifact`,
#       `ensure_artifact_installed`, `download_artifact`, `create_artifact`,
#       `bind_artifact!`, `with_artifacts_directory`, `Downloads.download`,
#       `HTTP.get`, `LazyArtifacts` or the store's own primitives: one path,
#       hash-verified, into the store, never the depot.
#
# The scan is textual and conservative (full-line comments are skipped; a
# launcher's `#SBATCH` lines are not comments). A project's known, deferred
# violations go in an EXPLICIT allowlist; an allowlist entry that matches
# nothing is itself a violation (`STALE`), so the list only shrinks.
# =============================================================================

"""
    DataPolicyAllow(rule, file, match)

One allowlisted (deferred) violation: rule `"D1"`…`"D5"`, `file`
relative to the scanned project root, and a substring `match` of the
offending line. Each entry should name what fixes it, in a comment beside it.
"""
struct DataPolicyAllow
    rule::String
    file::String
    match::String
end

# D1: a machine root (`/raid/…`, `/cluster/…`), a home-relative path (`~/src/…`, `$HOME/…`, `"$HOME"/…`, `${HOME}/…`,
# `$(HOME)/…`, `homedir()`, `ENV["HOME"]`), or a host name (`dgx01`) other than the value of a site selection
# (`--site=dgx01`, `--site dgx01`, `VGTK_SITE=dgx01`, `site = "dgx01"`, quoted or not).
const _D1 = r"(?<![\w.$}])/(raid|home|mnt|scratch|Users|nfs|net|cluster|srv|gpfs|lustre)/|(?<![\w.$}/~])~/[\w.]|\$\{?HOME\}?\"?/|\$\(HOME\)/|\bhomedir\(\)|ENV\[\"HOME\"\]|(?<!--site=)(?<!--site=\")(?<!--site=')(?<!--site )(?<!SITE=)(?<!SITE=\")(?<!site = \")(?<!site=\")(?<![\w.\-/])dgx[0-9]+\b"
const _D2_WRITE = r"\b(mkpath|mkdir|write|open|jldsave|wsave|tagsave|safesave|save|cp|mv|symlink|touch)\(\s*joinpath\(\s*(data_root\(\)|EXP_DATA_ROOT|datadir\(\s*\"exp_raw\")"
const _D4 = r"(?<![\w\"])(artifact\"|@artifact_str\b)"
const _D5 = r"\b(install_artifact|install_built_artifact|ensure_artifact_installed|download_artifact|create_artifact|bind_artifact!|archive_artifact|with_artifacts_directory|with_artifact_store|store_file|download_verified|migrate_artifacts)\(|\bDownloads\.download\(|\bHTTP\.(get|request)\(|\bLazyArtifacts\b|\bPkg\.Artifacts\b"
const _D2_OUTDIR = r"joinpath\(\s*(data_root\(\)|EXP_DATA_ROOT|datadir\(\s*\"exp_raw\"\s*\))\s*,\s*\"results\""
# `stamp`, `stamps` or `stamped` as a whole identifier token (`_`-separated,
# CamelCase split, an uppercase run ending where a capitalized word begins):
# `STAMP`, `eval_stamp`, `stamps`, `stamped`, `TimeStamp`, `UTCStamp` and
# `stampDir` match; `timestamp` and `timestamps` do not.
const _D3_STAMP = r"(?:(?<![A-Za-z0-9])[Ss]tamp(?:s|ed)?|(?<=[A-Za-z0-9])Stamp(?:s|ed)?)(?![a-z0-9])|(?<![A-Za-z0-9])STAMP(?:S|ED)?(?![A-Za-z0-9])"
# D3: a wall-clock stamp in an output path; a hand-bumped version literal; a
# `-smoke` sibling of a keyed directory (a smoke run belongs in DrWatson's
# datadir("sims")); an unkeyed file every run rewrites at the top of the
# results tree (oracle_floors_path() is the package's one such file: its unkeyed forms, the bare call and
# `oracle_floors_path(results_root())`, both `<results root>/oracle_floors.jld2`; `oracle_floors_path(keyed_dir)` is
# the documented keyed use).
_d3(line) =
    (occursin(r"Dates\.now\(\)|\bnow\(\)", line) && (occursin(_D3_STAMP, line) || occursin(r"(?i)joinpath|outdir", line))) ||
    occursin(r"\"[\w.-]*[-_]v\d+\"", line) ||
    (occursin(r"[\w)\]}\"]-smoke\b", line) && !occursin("sims", line)) ||
    occursin(r"\boracle_floors_path\(\s*(results_root\(\s*\))?\s*\)", line) ||
    occursin(r"joinpath\(\s*results_root\(\)\s*,\s*\"[^\"/]+\.(jld2|toml|tsv|csv|json)\"\s*\)", line)

_policy_comment(line, ext) = (l = lstrip(line); startswith(l, "#") && !(ext in (".sh", ".sbatch") && startswith(l, "#SBATCH")))

# THE layout of a codified experiment project: each folder (relative to the project root; "" is the root itself) and
# what the data policy reads in it, `scan`:
#   :code      every rule, full-line comments skipped (`data_policy_violations`' default `dirs`);
#   :text      D1 on every line of its code, launcher, config and text files — every comment and paragraph included,
#              since a path in a config's comment or a README is the next run's machine path (a site profile, a TOML
#              file with a `[site]` table, the one place a machine's paths belong, is not read);
#   :markdown  D1 in the Markdown files directly in it (README.md, REPRODUCE.md, data/README.md: the instructions a
#              reader copies);
#   :none      not read: the renders a driver publishes (generated, never tracked).
# `template` marks the folders PaperProjects' template creates (`PAPER_PROJECT_TEMPLATE`), `sub` their subfolders
# (data/: the inputs, the results, the smoke runs). The scan's folders and the template both derive from this one list,
# so a folder of the layout is never in one and missing from the other.
const _PROJECT_LAYOUT = [
    (dir = "src",        scan = :code,     template = true,  sub = String[]),
    (dir = "scripts",    scan = :code,     template = true,  sub = String[]),
    (dir = "test",       scan = :text,     template = true,  sub = String[]),
    (dir = "examples",   scan = :text,     template = false, sub = String[]),
    (dir = "configs",    scan = :text,     template = true,  sub = String[]),
    (dir = "docs",       scan = :text,     template = false, sub = String[]),
    (dir = "provenance", scan = :text,     template = false, sub = String[]),
    (dir = "",           scan = :markdown, template = false, sub = String[]),
    (dir = "data",       scan = :markdown, template = true,  sub = ["exp_raw", "exp_pro", "sims"]),
    (dir = "plots",      scan = :none,     template = true,  sub = String[]),
]
_layout_dirs(scan::Symbol) = [l.dir for l in _PROJECT_LAYOUT if l.scan === scan]

# Where D1 also looks for a machine path, beyond the code (`dirs`), and in which files.
const _MACHINE_PATH_DIRS = _layout_dirs(:text)
const _MACHINE_PATH_EXTS = [".jl", ".sh", ".sbatch", ".py", ".toml", ".md", ".json", ".yml", ".yaml", ".txt", ".tex"]
# … and the Markdown files directly in the layout's :markdown folders (with `git`, the ones git does not ignore).
function _root_markdown(root, git::Bool)
    md = String[]
    for d in (joinpath(root, l) for l in _layout_dirs(:markdown))
        isdir(d) || continue
        append!(md, filter(isfile, [joinpath(d, f) for f in sort(readdir(d)) if endswith(f, ".md")]))
    end
    (git && !isempty(md)) || return md
    keep = Set(normpath.(_git_listing(root, relpath.(md, root))))
    return filter(p -> normpath(p) in keep, md)
end

# The files of directory `d` of `root` the scan reads: with `git` (the project's files are git's: `_git_listing` lists
# some) the ones git does not ignore, tracked or new (an ignored build or output tree, docs/build, is never walked);
# without (no work tree, or an enclosing repository that ignores the whole project) every file under it, so the scan
# never goes quiet for want of git.
_policy_files(root, d, git::Bool) = git ? filter(isfile, _git_listing(root, [d])) :
    [joinpath(r, f) for (r, _, fs) in walkdir(joinpath(root, d)) for f in sort(fs)]

# The files git does not ignore within the pathspecs `spec` of `root` (tracked, or new and not ignored), sorted; none
# outside a work tree.
function _git_listing(root, spec)
    out = _git_line(root, ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", spec...])
    return out === nothing ? String[] : sort!([joinpath(root, f) for f in unique(split(out, '\0'; keepempty = false))])
end

_is_site_profile(path) = endswith(path, ".toml") && haskey(try TOML.parsefile(path) catch; Dict{String, Any}() end, "site")

"""
    data_policy_violations(root; dirs = ["src", "scripts"], exts = [".jl", ".sh", ".sbatch"],
                           allow = DataPolicyAllow[])
        -> Vector{NamedTuple{(:rule, :file, :line, :text)}}

The data-policy findings (rules D1–D5, see the file header) in the `dirs` of
project `root` (code, the `exts` files: every rule, comments skipped; their
other text files: D1), and D1 also in its tests,
examples, configs, docs and provenance (every line of their code, config and
text files; a site profile, which holds a machine's paths by design, excepted)
and in every Markdown file of the root and of `data/` (README.md, REPRODUCE.md,
data/README.md, …). In a git work tree the files git ignores are not read.
The result is minus the `allow`ed ones, plus a `STALE` finding for every allow entry that
matched nothing. Empty ⇔ compliant.
"""
function data_policy_violations(root::AbstractString; dirs = _layout_dirs(:code),
                                exts = [".jl", ".sh", ".sbatch"],
                                allow::AbstractVector{DataPolicyAllow} = DataPolicyAllow[])
    found = NamedTuple{(:rule, :file, :line, :text),Tuple{String,String,Int,String}}[]
    seen = Set{String}()
    git = !isempty(_git_listing(root, ["."]))
    walked = ((d, path) for d in filter(d -> isdir(joinpath(root, d)), unique([dirs; _MACHINE_PATH_DIRS]))
              for path in _policy_files(root, d, git))
    for (d, path) in Iterators.flatten((walked, (("", p) for p in _root_markdown(root, git))))
        ext = splitext(path)[2]
        rel = relpath(path, root)
        code = d in dirs && ext in exts
        (code || ext in _MACHINE_PATH_EXTS) && !(rel in seen) || continue
        push!(seen, rel)                                   # a code dir nested in a machine dir is read once
        code || !_is_site_profile(path) || continue
        for (i, line) in enumerate(eachline(path))
            hit(rule) = push!(found, (rule = rule, file = rel, line = i, text = strip(line)))
            if !code
                occursin(_D1, line) && hit("D1")
                continue
            end
            _policy_comment(line, ext) && continue
            occursin(_D1, line) && hit("D1")
            (occursin(_D2_WRITE, line) || occursin(_D2_OUTDIR, line)) && hit("D2")
            _d3(line) && hit("D3")
            occursin(_D4, line) && hit("D4")
            occursin(_D5, line) && hit("D5")
        end
    end
    used = falses(length(allow))
    keep = filter(found) do v
        k = findfirst(a -> a.rule == v.rule && a.file == v.file && occursin(a.match, v.text), allow)
        k === nothing && return true
        used[k] = true
        return false
    end
    for (k, a) in enumerate(allow)
        used[k] || push!(keep, (rule = "STALE", file = a.file, line = 0, text = "allowlist entry $(a.rule) \"$(a.match)\" matches nothing"))
    end
    return keep
end

# =============================================================================
# data_policy.jl — the data policy as a checkable API, for each project's tests
# in one line:
#
#   @test isempty(data_policy_violations(projectdir(); allow = DATA_POLICY_ALLOW))
#
#   D1  no machine paths (/raid, /home, /mnt, /cluster, …) in code, launchers,
#       tests, examples, configs, docs or tracked provenance: every path derives
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

const _D1 = r"(?<![\w.$}])/(raid|home|mnt|scratch|Users|nfs|net|cluster|srv|gpfs|lustre)/"
const _D2_WRITE = r"\b(mkpath|mkdir|write|open|jldsave|wsave|tagsave|safesave|save|cp|mv|symlink|touch)\(\s*joinpath\(\s*(data_root\(\)|EXP_DATA_ROOT|datadir\(\s*\"exp_raw\")"
const _D4 = r"(?<![\w\"])(artifact\"|@artifact_str\b)"
const _D5 = r"\b(install_artifact|ensure_artifact_installed|download_artifact|create_artifact|bind_artifact!|archive_artifact|with_artifacts_directory|with_artifact_store|store_file|download_verified|migrate_artifacts)\(|\bDownloads\.download\(|\bHTTP\.(get|request)\(|\bLazyArtifacts\b|\bPkg\.Artifacts\b"
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

"""
    MACHINE_PATH_DIRS

Where D1 also looks for a machine path, beyond the code (`dirs`): a project's
tests, examples, configs, docs and tracked provenance, in their
[`MACHINE_PATH_EXTS`](@ref) files — every comment and paragraph included, since
a path in a config's comment or a README is the next run's machine path.
"""
const MACHINE_PATH_DIRS = ["test", "examples", "configs", "docs", "provenance"]

"""
    MACHINE_PATH_EXTS

The files of [`MACHINE_PATH_DIRS`](@ref) D1 reads: code, launchers, configs and
text (`.jl`, `.sh`, `.sbatch`, `.toml`, `.md`, `.json`, `.yml`, `.yaml`,
`.txt`, `.tex`).
"""
const MACHINE_PATH_EXTS = [".jl", ".sh", ".sbatch", ".toml", ".md", ".json", ".yml", ".yaml", ".txt", ".tex"]

"""
    data_policy_violations(root; dirs = ["src", "scripts"], exts = [".jl", ".sh", ".sbatch"],
                           machine_dirs = MACHINE_PATH_DIRS, machine_exts = MACHINE_PATH_EXTS,
                           allow = DataPolicyAllow[])
        -> Vector{NamedTuple{(:rule, :file, :line, :text)}}

The data-policy findings (rules D1–D5, see the file header) in the `dirs` of
project `root` (code: every rule, comments skipped), and D1 also in its
`machine_dirs` (every line of their `machine_exts` files: [`MACHINE_PATH_DIRS`](@ref)),
minus the `allow`ed ones, plus a `STALE` finding for every allow entry that
matched nothing. Empty ⇔ compliant.
"""
function data_policy_violations(root::AbstractString; dirs = ["src", "scripts"],
                                exts = [".jl", ".sh", ".sbatch"],
                                machine_dirs = MACHINE_PATH_DIRS, machine_exts = MACHINE_PATH_EXTS,
                                allow::AbstractVector{DataPolicyAllow} = DataPolicyAllow[])
    found = NamedTuple{(:rule, :file, :line, :text),Tuple{String,String,Int,String}}[]
    scanned = Set{String}()
    for d in filter(d -> isdir(joinpath(root, d)), dirs), (r, _, fs) in walkdir(joinpath(root, d)), f in sort(fs)
        ext = splitext(f)[2]
        ext in exts || continue
        path = joinpath(r, f)
        rel = relpath(path, root)
        push!(scanned, rel)
        for (i, line) in enumerate(eachline(path))
            _policy_comment(line, ext) && continue
            hit(rule) = push!(found, (rule = rule, file = rel, line = i, text = strip(line)))
            occursin(_D1, line) && hit("D1")
            (occursin(_D2_WRITE, line) || occursin(_D2_OUTDIR, line)) && hit("D2")
            _d3(line) && hit("D3")
            occursin(_D4, line) && hit("D4")
            occursin(_D5, line) && hit("D5")
        end
    end
    for d in filter(d -> isdir(joinpath(root, d)), machine_dirs), (r, _, fs) in walkdir(joinpath(root, d)), f in sort(fs)
        splitext(f)[2] in machine_exts || continue
        path = joinpath(r, f)
        rel = relpath(path, root)
        rel in scanned && continue
        for (i, line) in enumerate(eachline(path))
            occursin(_D1, line) && push!(found, (rule = "D1", file = rel, line = i, text = strip(line)))
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

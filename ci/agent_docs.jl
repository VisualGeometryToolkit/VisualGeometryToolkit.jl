# The agent-docs policy over the whole workspace, once.
#
#   julia --project=ci ci/agent_docs.jl            # CLAUDE.md, AGENTS.md, .claude/** that git tracks
#   julia --project=ci ci/agent_docs.jl --all      # ... and the extra patterns over EVERY tracked text file
#
# VisualGeometryCore's `agent_doc_violations` holds the public patterns (home
# paths, keys, tokens). Release tooling adds its private list through the
# environment, one regex per line in a file that never enters this repository:
#
#   AGENT_DOC_PATTERNS=<path to the private pattern file> julia --project=ci ci/agent_docs.jl --all
#
# It runs at the repository root, where `git ls-files` sees every package:
# inside a package directory there is no `.git`, and the function would fall
# back to walking the tree and pick up untracked per-machine files.
# Findings print as "file:line matches <pattern>"; the exit code is their count
# (capped at 1).

using VisualGeometryCore: agent_doc_violations

const ROOT = dirname(@__DIR__)

function text_files(root)
    files = readlines(`git -C $root ls-files`)
    return filter(files) do f
        p = joinpath(root, f)
        isfile(p) && !islink(p) || return false
        filesize(p) < 4 * 2^20 || return false
        bytes = read(p, min(filesize(p), 8192))
        return !(0x00 in bytes)          # skip binaries
    end
end

_patterns(path) = [Regex(strip(l)) for l in eachline(path) if !isempty(strip(l)) && !startswith(strip(l), '#')]

function main(args)
    isdir(joinpath(ROOT, ".git")) || error("$ROOT is not a git checkout; agent_docs.jl needs `git ls-files`")
    extra = get(ENV, "AGENT_DOC_PATTERNS", nothing)
    found = agent_doc_violations(ROOT; extra)
    nagent = length(found)
    if "--all" in args && extra !== nothing && !isempty(extra)
        pats = _patterns(extra)
        for f in text_files(ROOT), (i, line) in enumerate(eachline(joinpath(ROOT, f))), re in pats
            occursin(re, line) && push!(found, "$f:$i matches extra pattern #$(findfirst(==(re), pats))")
        end
    end
    foreach(println, found)
    scope = "--all" in args ? "agent docs + all tracked files" : "agent docs"
    println(stderr, "agent_docs: $(length(found)) finding(s) [$scope; private patterns: $(extra === nothing ? "none" : "from AGENT_DOC_PATTERNS")]")
    exit(isempty(found) ? 0 : 1)
end

main(ARGS)

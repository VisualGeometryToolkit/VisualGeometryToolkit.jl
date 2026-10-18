# =============================================================================
# Agent instructions carry nothing private
# =============================================================================
#
#   * `AGENT_DOC_FORBIDDEN`  — the public patterns: machine paths and secrets.
#   * `agent_doc_violations` — a package's CLAUDE.md, AGENTS.md and tracked
#                              .claude/ files checked against them.
#
# Every stack package runs `agent_doc_violations` in its test suite; release
# tooling adds its own patterns through `extra`.
# =============================================================================

const AGENT_DOC_FORBIDDEN = (
    # absolute paths into a user's home directory
    r"/home/[^/\s]", r"/Users/[^/\s]", r"\b[A-Za-z]:[\\/]Users[\\/]",
    # private keys and access tokens
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\bgh[pousr]_[A-Za-z0-9]{30,}", r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bAKIA[0-9A-Z]{16}\b", r"\bsk-[A-Za-z0-9_-]{20,}", r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    r"\b(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[=:]\s*[\"'][^\"'\s]{8,}[\"']"i,
)

@doc """
    AGENT_DOC_FORBIDDEN

The patterns no agent instruction file of a stack package may match
([`agent_doc_violations`](@ref)): absolute paths into a user's home directory,
private keys and access tokens.
""" AGENT_DOC_FORBIDDEN

_is_agent_doc(path) = basename(path) in ("CLAUDE.md", "AGENTS.md") || ".claude" in splitpath(path)

function _agent_doc_files(root)
    git = Sys.which("git")
    if git !== nothing && ispath(joinpath(root, ".git"))
        return filter(_is_agent_doc, readlines(`$git -C $root ls-files`))
    end
    files = String[]
    for (dir, dirs, names) in walkdir(root)
        filter!(!=(".git"), dirs)
        append!(files, filter(_is_agent_doc, relpath.(joinpath.(dir, names), root)))
    end
    return files
end

_agent_doc_patterns(::Nothing) = Regex[]
_agent_doc_patterns(re::Regex) = Regex[re]
function _agent_doc_patterns(path::AbstractString)
    isempty(path) && return Regex[]
    lines = strip.(readlines(path))
    return Regex[Regex(l) for l in lines if !isempty(l) && !startswith(l, '#')]
end
_agent_doc_patterns(res) = Regex[re for re in res]

"""
    agent_doc_violations(root; forbidden = AGENT_DOC_FORBIDDEN, extra = nothing) -> Vector{String}

The lines of the agent instruction files under the package directory `root` —
CLAUDE.md, AGENTS.md and anything under `.claude/`, the files git tracks when
`root` is a checkout — that match a pattern of `forbidden` or `extra`, one
`"file:line matches pattern"` per finding, plus `"CLAUDE.md: missing"` when
the package has none. These files ship with the package: they carry its
conventions and nothing private.

`extra` adds patterns: a `Regex`, a collection of them, or the path to a text
file of one regular expression per line (blank lines and `#` comments skipped;
an empty path adds none). The stack's tests call it with
`extra = get(ENV, "AGENT_DOC_PATTERNS", nothing)`, so release tooling supplies
its own list by setting that variable.
"""
function agent_doc_violations(root::AbstractString; forbidden = AGENT_DOC_FORBIDDEN, extra = nothing)
    patterns = vcat(_agent_doc_patterns(forbidden), _agent_doc_patterns(extra))
    files = _agent_doc_files(root)
    out = String[]
    "CLAUDE.md" in files || push!(out, "CLAUDE.md: missing")
    for f in files, (i, line) in enumerate(eachline(joinpath(root, f))), re in patterns
        occursin(re, line) && push!(out, "$f:$i matches $re")
    end
    return out
end

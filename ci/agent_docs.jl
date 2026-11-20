# The agent-docs policy over the whole workspace, once.
#
#   julia --project=ci ci/agent_docs.jl            # CLAUDE.md, AGENTS.md, .claude/** that git tracks
#   julia --project=ci ci/agent_docs.jl --all      # ... and the extra patterns over EVERY tracked text file
#
# VisualGeometryToolkitCore's `agent_doc_violations` holds the public patterns (home
# paths, keys, tokens). Release tooling adds its private list through the
# environment, one regex per line in a file that never enters this repository:
#
#   AGENT_DOC_PATTERNS=<path to the private pattern file> julia --project=ci ci/agent_docs.jl --all
#
# Findings print as "file:line matches <pattern>"; the exit code is their count
# (capped at 1). The implementation is VisualGeometryToolkitCore's `stack_ci`
# (the `agent-docs` command; `workspace_agent_doc_violations`).

using VisualGeometryToolkitCore: stack_ci

exit(stack_ci(["agent-docs"; ARGS]; root = dirname(@__DIR__)))

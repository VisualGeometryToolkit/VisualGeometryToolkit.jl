# =============================================================================
# JSONL append — one record per line, safe under threads and processes
# =============================================================================
#
# The primitive every run log hand-rolls (and drifts): BlobBoards carried two
# copies with different safety stories — one relied on single-write append
# atomicity across processes, the other on a per-channel lock across threads.
# This is the superset of both, owned once:
#
#   * the line is built COMPLETELY first and written with ONE `write` call on
#     an O_APPEND handle, so concurrent appends from other PROCESSES
#     interleave at line boundaries;
#   * a per-path lock (module registry) serializes appends across THREADS of
#     this process.

const _JSONL_LOCKS = Dict{String, ReentrantLock}()
const _JSONL_LOCKS_GUARD = ReentrantLock()

_jsonl_lock(path::String) = lock(_JSONL_LOCKS_GUARD) do
    get!(ReentrantLock, _JSONL_LOCKS, path)
end

"""
    append_jsonl(path, row)

Append `row` (anything `JSON3.write` serializes — a `NamedTuple`, `Dict`, …)
to `path` as one JSON line. Thread-safe within the process (per-path lock)
and process-append-safe (the line is written with a single `write` call on an
append-mode handle, so concurrent writers interleave at line boundaries).
Read the file back with `JSON3.read` per line.
"""
function append_jsonl(path::AbstractString, row)
    line = JSON3.write(row) * "\n"
    lock(_jsonl_lock(String(path))) do
        open(io -> write(io, line), path, "a")
    end
    return nothing
end

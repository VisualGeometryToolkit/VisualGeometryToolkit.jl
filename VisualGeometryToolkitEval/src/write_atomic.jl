# =============================================================================
# src/write_atomic.jl — the one atomic file write of the package's dependency-free files: the runner's manifests and
# completion markers (reproduce/runner.jl, through src/cli.jl) and a driver's number macros
# (reproduce/number_macros.jl). Base only. Each includer includes it only when its module has no `_write_atomic` yet,
# so a file included by path from a bare `julia` brings it along, and the package defines it once.
# =============================================================================

# `f(io)` writes the content; it goes to a temporary file in `path`'s directory, renamed onto `path` only when `f`
# returned: a reader sees the old file or the new one, never a part of one, and a failure leaves the old one.
function _write_atomic(f::Function, path::AbstractString)
    tmp, io = mktemp(mkpath(dirname(path)); cleanup = false)
    try
        f(io)
        close(io)
        chmod(tmp, 0o644)                  # mktemp's file is private; a marker or manifest is read by others
        mv(tmp, path; force = true)
    catch
        close(io)
        rm(tmp; force = true)
        rethrow()
    end
    return path
end

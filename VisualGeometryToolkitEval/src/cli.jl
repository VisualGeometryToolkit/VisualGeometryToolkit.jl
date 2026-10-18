# =============================================================================
# src/cli.jl — THE shared CLI + output scaffolding of every driver.
#
# CLI dialect is `--key=value` (the RoMa/producer convention).
# =============================================================================

# -----------------------------------------------------------------------------
# Periodic stdout/stderr flushing (sbatch/nohup-safe: buffered output otherwise
# vanishes on an OOM/timeout kill).
#
# A Timer, NOT a logger: GPUCompiler queries the active logger's
# `min_enabled_level` through an invoke in a FROZEN world, where methods of a
# logger type defined after that world are invisible — so a custom flushing
# logger dies inside every CUDA driver (observed 2026-08-31). A timer touches no
# logging machinery.
# -----------------------------------------------------------------------------
const _FLUSH_TIMER = Ref{Union{Nothing,Timer}}(nothing)

"""
    start_output_flush!(; interval = 2) -> Timer

Flush `stdout` and `stderr` every `interval` seconds for the rest of the
session (idempotent: a second call returns the running timer). THE one
flusher of every driver, GPU drivers included — it installs no logger type.
"""
function start_output_flush!(; interval::Real = 2)
    t = _FLUSH_TIMER[]
    (t !== nothing && isopen(t)) && return t
    _FLUSH_TIMER[] = Timer(_ -> (flush(stderr); flush(stdout)), 1; interval)
    return _FLUSH_TIMER[]
end

# -----------------------------------------------------------------------------
# `--key=value` lookup
# -----------------------------------------------------------------------------
"""
    cli_option(args, key[, default])

The string value of `--key=value` in `args`. With `default`, the default when
the option is absent; without it, the option is REQUIRED and its absence
throws.
"""
function cli_option(args, key, default)
    for a in args
        startswith(a, "--$key=") && return a[length(key)+4:end]
    end
    return default
end
function cli_option(args, key)
    v = cli_option(args, key, nothing)
    v === nothing && error("missing required argument --$key=…")
    return v
end

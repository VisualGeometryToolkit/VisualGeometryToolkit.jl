# =============================================================================
# batched_rows — the one chunking skeleton behind every describe backend
# =============================================================================

"""
    batched_rows(run_chunk, N::Int, max_batch::Int; retry_single = false) -> Matrix{Float32}

Run `run_chunk(offset, n)` over `1:N` in chunks of at most `max_batch` rows
(`max_batch <= 0` means one chunk) and assemble the N × D result: the row
width D is read off the FIRST chunk's output (any shape with `n * D`
elements), the matrix is allocated lazily then, and each chunk is assigned
into its contiguous row slice. With `retry_single`, a chunk that throws is
retried one row at a time (the batch size drops to 1 for the rest of the
run) — the recovery an inference session wants when a batch overflows its
arena. `N == 0` returns an empty `0 × 0` matrix.

Hoisted from BlobBoards, where the ONNX host, ORT device-tensor and two Lux
describe backends had each drifted their own copy (2026-09).
"""
function batched_rows(run_chunk, N::Int, max_batch::Int; retry_single::Bool = false)
    N == 0 && return Matrix{Float32}(undef, 0, 0)
    B = max_batch > 0 ? min(max_batch, N) : N
    rows = nothing
    offset = 0
    while offset < N
        n = min(B, N - offset)
        out = if retry_single && n > 1
            try
                run_chunk(offset, n)
            catch e
                e isa InterruptException && rethrow()   # Ctrl-C is not an arena overflow
                B = 1
                continue
            end
        else
            run_chunk(offset, n)
        end
        if rows === nothing
            rows = Matrix{Float32}(undef, N, div(length(out), n))
        end
        rows[(offset + 1):(offset + n), :] .= reshape(out, n, :)
        offset += n
    end
    return rows
end

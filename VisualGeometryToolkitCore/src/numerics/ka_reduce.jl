# =============================================================================
# KernelAbstractions workgroup reductions
# =============================================================================

using KernelAbstractions: @context, @synchronize

"""
    tree_sum!(@context, smem, tid, nt, half = Int32(128))

Workgroup tree reduction: sums `smem[1:nt]` into `smem[1]`, thread `tid` of
`nt` folding `smem[tid + s]` into `smem[tid]` for `s = half, half/2, …, 1`
with a barrier per level (left to right within a level, so the sum is the one
every 256-thread reduction of the stack forms). `half` is HALF the workgroup
size (the default serves 256-thread groups). A helper, not a kernel — it takes
the kernel's `@context` so it may `@synchronize`; every thread in the group
must call it, after a barrier that follows its write of `smem[tid]` (the
barriers are convergent only then).

On KernelAbstractions' CPU backend a barrier must be written in the kernel
body itself (its CPU transform splits the body at a syntactic `@synchronize`;
one inside a called function is an error), so a CPU kernel calls this with a
group of one, `nt == 1` — the stack's CPU group size for its per-blob kernels —
where there is nothing to fold and no barrier is taken: `smem[1]` is the sum.
"""
@inline function tree_sum!(@context, smem, tid, nt, half::Int32 = Int32(128))
    nt == 1 && return nothing          # a group of one: the sum is smem[1], and no barrier (the CPU backend's group)
    s = half
    while s >= Int32(1)
        if tid <= s && tid + s <= nt
            @inbounds smem[tid] += smem[tid + s]
        end
        @synchronize()
        s >>= Int32(1)
    end
    return nothing
end

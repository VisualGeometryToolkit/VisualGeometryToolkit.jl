# =============================================================================
# KernelAbstractions workgroup reductions
# =============================================================================

using KernelAbstractions: @context, @synchronize

"""
    tree_sum!(@context, smem, tid, nt, half = Int32(128))

Workgroup tree reduction: sums `smem[1:nt]` into `smem[1]`, thread `tid` of
`nt` folding `smem[tid + s]` into `smem[tid]` for `s = half, half/2, …, 1`
with a barrier per level. `half` is HALF the workgroup size (the default
serves 256-thread groups). A helper, not a kernel — it takes the kernel's
`@context` so it may `@synchronize`; every thread in the group must call it
(the barriers are convergent only then). Hoisted verbatim from BlobBoards'
adaptation kernels (2026-09).
"""
@inline function tree_sum!(@context, smem, tid, nt, half::Int32 = Int32(128))
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

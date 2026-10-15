# =============================================================================
# Stacked multi-gallery KA matching — hoisted from BlobBoards (2026-08-10)
# =============================================================================
#
# Match ONE feature set against MANY descriptor galleries with a single
# stacked GEMM plus the matcher's portable kernels, on any KA backend. This
# replaces per-gallery upload→GEMM→download round trips (measured in
# BlobBoards: 50 galleries at ~560 ms/call latency-bound, vs ~40 ms fused on
# GPU). It is the one matcher of matching_ka.jl with the galleries as blocks of
# its rows; `match` is the same matcher with a single block.
#
# `MutualNN` is evaluated PER GALLERY BLOCK: a feature's reverse best is
# computed within each gallery's row block, so the result for gallery `k`
# is identical to running `nn_match(gallery_k, features, MutualNN())` on
# its own — galleries never compete with each other here (arbitration
# between them is the caller's concern).
#
# The `StackedDescriptorIndex` is the reusable device-resident gallery
# stack: build once per gallery set and supply it as `index` — rebuilding it
# per call is the dominant allocation (~85 MB host staging for a 60K-row
# stack).

"""
    StackedDescriptorIndex(backend, galleries::AbstractDict)

Device-resident stack of descriptor galleries: all galleries' descriptor
rows in one matrix, plus per-gallery offsets/sizes for the block-local
kernels. `galleries` maps keys to descriptor matrices (rows = features) or
to anything `descriptor_matrix` accepts. The stack keeps the galleries' float
type (Float32 galleries stack as Float32, Float64 as Float64). Build once,
reuse across `nn_match_stacked` calls.
"""
struct StackedDescriptorIndex{T, K}
    d_all::T                # backend array — stacked descriptors (M_total × D)
    keys::Vector{K}         # gallery keys in stacking order
    offsets::Vector{Int}    # cumulative row offsets per gallery
    sizes::Vector{Int}      # rows per gallery
end

_descmat(x::AbstractMatrix) = x
_descmat(x) = descriptor_matrix(x)

function StackedDescriptorIndex(backend::KA.Backend, galleries::AbstractDict)
    ks = collect(keys(galleries))
    # ONE staging copy: fill each gallery's row block of one preallocated
    # matrix directly (descriptor_matrix is already a Float32 view).
    descs = [_descmat(galleries[k]) for k in ks]
    sizes = [size(d, 1) for d in descs]
    offsets = cumsum([0; sizes[1:(end - 1)]])
    T = isempty(descs) ? Float32 : _score_eltype(descs...)
    all_desc = Matrix{T}(undef, sum(sizes), isempty(descs) ? 0 : size(first(descs), 2))
    for (i, d) in enumerate(descs)
        copyto!(view(all_desc, (offsets[i] + 1):(offsets[i] + sizes[i]), :), d)
    end
    d_all = KA.allocate(backend, T, size(all_desc)...)
    copyto!(d_all, all_desc)
    # The block kernels take each chunk's block offsets/sizes, uploaded per
    # chunk by `_nn_select`; whole-stack device copies here were never read.
    return StackedDescriptorIndex(d_all, ks, offsets, sizes)
end

"""
    nn_match_stacked(strategy, galleries::AbstractDict, features;
                     backend, index = nothing, min_similarity = 0.6, chunk_bytes = 0)
        -> Dict{K, Vector{NNMatch}}

Match `features` against every gallery in `galleries` with one stacked
GEMM. Per gallery the result is identical to matching that gallery alone
with `strategy` (`MutualNN` reverse checks are block-local; `KnnMatch(k)`
takes any k): it is the matcher `match` runs, with the galleries as blocks of
its rows. Matches use LOCAL indices: `m.first` indexes into gallery `k`'s
rows, `m.second` into `features`. Supply a prebuilt
[`StackedDescriptorIndex`](@ref) as `index` to skip the per-call stack/upload.

When `index` is given it — not `galleries` — defines what gets matched:
the returned dictionary has one entry per `index.keys`, and `galleries`
supplies only the key type `K`. The two key sets therefore need NOT
agree, and callers rely on that: a caller narrowing its gallery set
across calls can keep ONE index built over the full set rather than
restaging descriptors every call, and simply ignore the extra entries.
Build the index from the widest set you will ever match against.

The score matrix is built a row range at a time. `chunk_bytes > 0` caps each
range's score block (rows × features) at that many bytes; the default `0`
sizes the ranges from free device memory. The result does not depend on the
chunking. Scores are cosine similarities in the wider of the index's and the
features' float type.
"""
function nn_match_stacked(
        strategy::NNStrategy, galleries::AbstractDict{K}, features;
        backend::KA.Backend, index::Union{Nothing, StackedDescriptorIndex} = nothing,
        min_similarity = 0.6, chunk_bytes::Integer = 0
    ) where {K}
    idx = index === nothing ? StackedDescriptorIndex(backend, galleries) : index
    fd = _descmat(features)
    T = _score_eltype(idx.d_all, fd)
    metric = CosineSimilarity()
    top_idx, top_sc, rev_idx = _nn_select(
        strategy, metric, backend, idx.offsets, idx.sizes;
        gal = idx.d_all, feat = _ka_upload(backend, T, fd), chunk_bytes
    )
    N_feat = size(fd, 1)
    match_dict = Dict{K, Vector{NNMatch}}()
    for (b, key) in enumerate(idx.keys)
        match_dict[key] = _block_matches(
            strategy, metric, top_idx, top_sc, rev_idx, b, idx.offsets[b], idx.sizes[b], N_feat;
            min_similarity
        )
    end
    return match_dict
end

# =============================================================================
# KA descriptor matching — the one matcher, on the CPU and any GPU backend
# =============================================================================
#
# Every nearest-neighbour match runs here: `match` on `KA.CPU()`, `nn_match` on
# the backend it is handed, both with the gallery as one block of rows, and
# `nn_match_stacked` with many galleries as many blocks (matching_stacked_ka.jl
# holds only its index and entry point). The expensive part is a GEMM (`d1 * d2'`), generic
# array code that hits BLAS on `Array` and the vendor BLAS (cuBLAS/rocBLAS/…) on
# device arrays. The selection is two portable kernels: the top k feature columns
# per gallery row, and MutualNN's reverse check, the best row per feature column
# within each gallery block. The thresholds, Lowe ratio and ordering are applied
# on the host to the k selected per row.
#
# There is no host twin: KernelAbstractions runs the kernels on the CPU. The
# host implementation this replaced (a partialsortperm per row and per column)
# was 8–27× slower on the CPU from 50×200 descriptors up, and at 20×24 the
# kernels are no slower. Descriptors keep their float type throughout.

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: mul!

# =============================================================================
# Selection kernels
# =============================================================================

# Scores are selected LARGEST first. A distance is negated on the way in, which
# is exact, so one kernel serves both kinds of metric; `_score` undoes it.
@inline _signed(v, ::Val{true}) = v
@inline _signed(v, ::Val{false}) = -v
_maximize(::DescriptorSimilarity) = Val(true)
_maximize(::DescriptorDistance) = Val(false)

# Insert score `v` of column `j` into a row's best-first top-k tuple, after every
# entry ≥ v: of equal scores the first column seen keeps the higher rank, the
# strict `>` rule of every kernel here.
@inline function _topk_insert(s::NTuple{K, T}, ix::NTuple{K, Int32}, v::T, j::Int32) where {K, T}
    p = 1 + sum(map(x -> Int(x >= v), s))
    s′ = ntuple(r -> r < p ? s[r] : r == p ? v : s[max(r - 1, 1)], Val(K))
    ix′ = ntuple(r -> r < p ? ix[r] : r == p ? j : ix[max(r - 1, 1)], Val(K))
    return s′, ix′
end

@inline function _topk_store!(top_idx, top_sc, row, s::NTuple{K}, ix::NTuple{K}) where {K}
    ntuple(Val(K)) do r
        @inbounds top_sc[row, r] = s[r]
        @inbounds top_idx[row, r] = ix[r]
        nothing
    end
    return nothing
end

# Per gallery row, its K best feature columns, best first: `top_idx[row, r]` and
# the signed score `top_sc[row, r]` (index 0 and −Inf past the last column).
# `init` is K copies of −Inf in the score type — the top-k lives in registers,
# a tuple whose length is K.
@kernel function _ka_topk_rows!(top_idx, top_sc, @Const(dm), maximize, init)
    row = @index(Global)
    @inbounds if row <= size(dm, 1)
        s = init
        ix = map(_ -> Int32(0), init)
        for j in 1:size(dm, 2)
            v = _signed(dm[row, j], maximize)
            if v > last(s)
                s, ix = _topk_insert(s, ix, v, Int32(j))
            end
        end
        _topk_store!(top_idx, top_sc, row, s, ix)
    end
end

# Per (feature column, gallery block), the best row of the block within this
# chunk's rows, merged with the running best from the block's earlier chunks —
# the reverse direction of MutualNN's check, so galleries never compete. `bases[blk]`
# is where the chunk's rows of that block start in the block's own row
# numbering, so `best_idx` is block-local whatever the chunking; strict `>` keeps
# the first maximum in row order, chunked or not.
@kernel function _ka_top1_cols_blocked!(
        best_idx, best_sc, @Const(dm), @Const(offsets), @Const(sizes), @Const(bases), maximize
    )
    col, blk = @index(Global, NTuple)
    @inbounds if col <= size(dm, 2) && blk <= length(sizes)
        o = offsets[blk]
        max1 = best_sc[col, blk]
        best = best_idx[col, blk]
        for li in 1:sizes[blk]
            v = _signed(dm[o + li, col], maximize)
            if v > max1
                max1 = v
                best = bases[blk] + Int32(li)
            end
        end
        best_idx[col, blk] = best
        best_sc[col, blk] = max1
    end
end

# =============================================================================
# Uploads and the pairwise matrix
# =============================================================================

# The float type descriptors are compared in: their own, never narrowed —
# Float32 stays Float32, Float64 stays Float64, integers and fixed-point widen.
_score_eltype(As...) = float(promote_type(map(eltype, As)...))

# A `T` copy of `A` on `dev`. On the CPU that is one copy: allocating and then
# `copyto!(d, T.(A))` made two.
_ka_upload(::KA.CPU, ::Type{T}, A::AbstractMatrix) where {T} = Matrix{T}(A)
function _ka_upload(dev, ::Type{T}, A::AbstractMatrix) where {T}
    d = KA.allocate(dev, T, size(A)...)
    copyto!(d, T.(A))
    return d
end

_to_host(A::Array) = A
_to_host(A) = Array(A)

# A precomputed score matrix as the kernels read it: used as is when it already
# lives on `dev` in a float type, otherwise copied there in its own float type.
_on_backend(dev, A::AbstractMatrix) =
    eltype(A) <: AbstractFloat && KA.get_backend(A) == dev ? A : _ka_upload(dev, float(eltype(A)), A)

# The scores of gallery rows r1:r2 against every feature, into `blk`. Generic
# array code: the GEMM is BLAS on `Array` and the vendor BLAS on device arrays,
# in `blk`'s precision (a gallery stored narrower is widened a chunk at a time,
# never the features narrowed). `norms` is `_norms(metric, gal, feat)`.
function _score_rows!(blk, ::CosineSimilarity, gal, feat, norms, r1::Int, r2::Int)
    mul!(blk, _rows_as(eltype(blk), gal, r1, r2), feat')
    return blk
end
function _score_rows!(blk, ::L2, gal, feat, norms, r1::Int, r2::Int)
    # ‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b: exact for any descriptors. The unit-norm
    # shortcut √(2 − 2a·b) the host used once gave other distances than a device
    # run whenever the input was not normalized.
    mul!(blk, _rows_as(eltype(blk), gal, r1, r2), feat')
    blk .= sqrt.(max.(view(norms.gal, r1:r2) .+ norms.feat' .- 2 .* blk, zero(eltype(blk))))
    return blk
end

_rows_as(::Type{T}, gal::AbstractMatrix{T}, r1, r2) where {T} = view(gal, r1:r2, :)
_rows_as(::Type{T}, gal, r1, r2) where {T} = T.(view(gal, r1:r2, :))

# The squared row norms L2 needs, once per call rather than once per chunk.
_norms(::CosineSimilarity, gal, feat) = nothing
_norms(::L2, gal, feat) = (gal = vec(sum(abs2, gal; dims = 2)), feat = vec(sum(abs2, feat; dims = 2)))

"""
    pairwise_matrix(metric, desc1, desc2, device = KA.CPU()) -> Matrix

The N1×N2 pairwise comparison matrix: `a·b` for `CosineSimilarity()`,
`‖a − b‖` for `L2()`, in the descriptors' float type (Float32 descriptors give
Float32, Float64 give Float64). The GEMM runs on `device`, any
KernelAbstractions backend (`CUDABackend()`, …); the matrix is returned on the
host. One implementation serves every backend: there is no separate host method.
"""
function pairwise_matrix(
        metric::DescriptorMetric, desc1::AbstractMatrix, desc2::AbstractMatrix,
        device::KA.Backend = KA.CPU()
    )
    T = _score_eltype(desc1, desc2)
    d1 = _ka_upload(device, T, desc1); d2 = _ka_upload(device, T, desc2)
    S = KA.allocate(device, T, size(d1, 1), size(d2, 1))
    return _to_host(_score_rows!(S, metric, d1, d2, _norms(metric, d1, d2), 1, size(d1, 1)))
end

# =============================================================================
# The matcher
# =============================================================================

# Columns of the forward top-k: the best two give NN's and MutualNN's Lowe
# ratio; KnnMatch keeps its k.
_topk_width(::Union{NN, MutualNN}) = 2
_topk_width(s::KnnMatch) = s.k

# Fraction of free device memory the score matrix may claim. Deliberately
# conservative: the matcher runs with the detector pyramids and the descriptor
# session's arena already resident, so "free" is what is left after them.
const _SCORE_BUDGET = 0.25

"""
    _row_chunks(backend, M, N, T, chunk_bytes) -> Vector{Tuple{Int, Int}}

Consecutive ROW ranges of an `M`-row gallery whose score block (`rows × N`
of `T`) fits the budget. `chunk_bytes > 0` is honoured verbatim; otherwise the
budget is `_SCORE_BUDGET` of `device_free_bytes`, which returns `typemax` on
backends exposing no query — yielding a single chunk, the historical behaviour.

Rows, not whole galleries: one calibration target with thousands of blobs
is ONE gallery whose block alone can exceed the card (9.3 GiB against a
40k-feature frame, 2026-09-15), so a gallery may span several chunks. That
is still exact — the forward sweep is per row, and the reverse sweep merges a
gallery's per-column best across its chunks (`_ka_top1_cols_blocked!`).
"""
function _row_chunks(backend, M::Int, N::Int, ::Type{T}, chunk_bytes::Integer) where {T}
    M == 0 && return Tuple{Int, Int}[]
    budget = chunk_bytes > 0 ? Int(chunk_bytes) :
        let f = device_free_bytes(backend)
            f == typemax(Int) ? typemax(Int) : floor(Int, _SCORE_BUDGET * f)
    end
    row_bytes = max(N * sizeof(T), 1)
    max_rows = budget == typemax(Int) ? typemax(Int) : max(1, fld(budget, row_bytes))
    chunks = Tuple{Int, Int}[]
    r1 = 1
    while r1 <= M
        r2 = max_rows == typemax(Int) ? M : min(M, r1 + max_rows - 1)
        push!(chunks, (r1, r2))
        r1 = r2 + 1
    end
    return chunks
end

# The blocks a row range touches, and each one's rows within it: the block
# offset in the chunk, the block size, and the block-local base of the rows.
function _chunk_blocks(offsets::Vector{Int}, sizes::Vector{Int}, r1::Int, r2::Int)
    b1 = searchsortedlast(offsets, r1 - 1)
    b2 = searchsortedlast(offsets, r2 - 1)
    coff = Int32[]; csz = Int32[]; cbase = Int32[]
    for b in b1:b2
        g1 = offsets[b] + 1
        g2 = offsets[b] + sizes[b]
        lo = max(g1, r1); hi = min(g2, r2)
        push!(coff, Int32(lo - r1)); push!(csz, Int32(hi - lo + 1)); push!(cbase, Int32(lo - g1))
    end
    return b1, b2, coff, csz, cbase
end

function _upload_i32(dev, v::Vector{Int32})
    d = KA.allocate(dev, Int32, length(v))
    copyto!(d, v)
    return d
end

"""
    _nn_select(strategy, metric, dev, offsets, sizes; gal, feat, S, chunk_bytes)
        -> (top_idx, top_sc, rev_idx)

The selection behind every match, on backend `dev`. The gallery is the rows,
cut into blocks (`offsets`, 0-based, and `sizes`); the scores are either
computed from descriptors already on `dev` (`gal`, rows × D, and `feat`,
features × D, in the score type) a row range at a time, or read from a
precomputed matrix `S` (rows × features). Returns host arrays: per row the top
k feature indices and their signed scores (rows × k), and for MutualNN each
feature's best row within each block (features × blocks, block-local), else
`nothing`.

The score block is the pipeline's largest single allocation (M × N: 3.09 GiB for
a 60k gallery against 13.7k detections, 9.79 GiB against 43.5k). Materialising
it whole is what OOMs on cluttered frames, so it is built `_row_chunks` at a
time and each block released eagerly via `KA.unsafe_free!` rather than waiting
for a finalizer. The result does not depend on the chunking.
"""
function _nn_select(
        strategy::NNStrategy, metric::DescriptorMetric, dev::KA.Backend,
        offsets::Vector{Int}, sizes::Vector{Int};
        gal = nothing, feat = nothing, S = nothing, chunk_bytes::Integer = 0
    )
    M, N, T = S === nothing ? (size(gal, 1), size(feat, 1), eltype(feat)) : (size(S)..., eltype(S))
    K = _topk_width(strategy)
    mx = _maximize(metric)
    norms = S === nothing ? _norms(metric, gal, feat) : nothing
    top_idx = KA.allocate(dev, Int32, M, K)
    top_sc = KA.allocate(dev, T, M, K)
    # MutualNN's reverse sweep: per (feature, block) the running best row and
    # score, merged across chunks. One `nothing` for both buffers, so a check on
    # it narrows both (two separately-guarded buffers left JET a Union each).
    rev = if strategy isa MutualNN
        (
            idx = fill!(KA.allocate(dev, Int32, N, length(sizes)), Int32(0)),
            sc = fill!(KA.allocate(dev, T, N, length(sizes)), T(-Inf)),
        )
    else
        nothing
    end
    topk! = _ka_topk_rows!(dev, (256,))
    init = ntuple(_ -> T(-Inf), K)
    chunks = S === nothing ? _row_chunks(dev, M, N, T, chunk_bytes) : [(1, M)]
    for (r1, r2) in chunks
        blk = if S === nothing
            _score_rows!(KA.allocate(dev, T, r2 - r1 + 1, N), metric, gal, feat, norms, r1, r2)
        else
            view(S, r1:r2, :)
        end
        topk!(view(top_idx, r1:r2, :), view(top_sc, r1:r2, :), blk, mx, init; ndrange = r2 - r1 + 1)
        if rev !== nothing
            b1, b2, hoff, hsz, hbase = _chunk_blocks(offsets, sizes, r1, r2)
            coff = _upload_i32(dev, hoff); csz = _upload_i32(dev, hsz); cbase = _upload_i32(dev, hbase)
            _ka_top1_cols_blocked!(dev, (256, 1))(
                view(rev.idx, :, b1:b2), view(rev.sc, :, b1:b2), blk, coff, csz, cbase, mx;
                ndrange = (N, b2 - b1 + 1)
            )
            KA.synchronize(dev)
            KA.unsafe_free!(coff); KA.unsafe_free!(csz); KA.unsafe_free!(cbase)
        else
            KA.synchronize(dev)
        end
        S === nothing && KA.unsafe_free!(blk)
    end
    KA.synchronize(dev)
    return Array(top_idx), Array(top_sc), rev === nothing ? nothing : Array(rev.idx)
end

# A selected score back in the metric's own sense, as the NNMatch stores it.
_score(v, ::DescriptorSimilarity) = Float32(v)
_score(v, ::DescriptorDistance) = Float32(-v)

# The score threshold that applies to the metric: a floor on similarities, a
# ceiling on distances.
_keep(s, ::DescriptorSimilarity, min_similarity, max_distance) =
    min_similarity === nothing || s >= min_similarity
_keep(s, ::DescriptorDistance, min_similarity, max_distance) =
    max_distance === nothing || s <= max_distance

# The matches of gallery block `b` (rows o+1:o+M of the selection, reported
# block-local), best first. The Lowe ratio is `_ratio` of the two best scores
# whenever a second neighbour exists, `Inf32` otherwise — and `Inf32` too when a
# similarity's best score is non-positive, where the ratio is undefined (see
# `_ratio`). `ratio_threshold` is strict.
function _block_matches(
        strategy::Union{NN, MutualNN}, metric, top_idx, top_sc, rev_idx, b::Int, o::Int, M::Int,
        N_feat::Int; ratio_threshold = nothing, min_similarity = nothing, max_distance = nothing
    )
    ms = NNMatch[]
    for i in 1:M
        g = o + i
        j = Int(top_idx[g, 1])
        j == 0 && continue
        rev_idx === nothing || rev_idx[j, b] == i || continue
        s = _score(top_sc[g, 1], metric)
        _keep(s, metric, min_similarity, max_distance) || continue
        ratio = N_feat >= 2 ? _ratio(s, _score(top_sc[g, 2], metric), metric) : Inf32
        (ratio_threshold === nothing || ratio < ratio_threshold) || continue
        push!(ms, NNMatch(i, j, s, ratio))
    end
    return sort!(ms; by = m -> m.score, rev = _sort_rev(metric))
end

# KnnMatch: each row's k best (k capped at the feature count), sharing the row's
# ratio of its two best.
function _block_matches(
        strategy::KnnMatch, metric, top_idx, top_sc, rev_idx, b::Int, o::Int, M::Int,
        N_feat::Int; ratio_threshold = nothing, min_similarity = nothing, max_distance = nothing
    )
    k = min(strategy.k, N_feat)
    ms = NNMatch[]
    for i in 1:M
        g = o + i
        s1 = _score(top_sc[g, 1], metric)
        ratio = k >= 2 ? _ratio(s1, _score(top_sc[g, 2], metric), metric) : Inf32
        (ratio_threshold === nothing || ratio < ratio_threshold) || continue
        for r in 1:k
            s = _score(top_sc[g, r], metric)
            _keep(s, metric, min_similarity, max_distance) || continue
            push!(ms, NNMatch(i, Int(top_idx[g, r]), s, ratio))
        end
    end
    return sort!(ms; by = m -> m.score, rev = _sort_rev(metric))
end

# `match` and `nn_match` on backend `dev`: one gallery block.
function _match(
        desc1::AbstractMatrix, desc2::AbstractMatrix, strategy::NNStrategy, dev::KA.Backend;
        metric::DescriptorMetric = CosineSimilarity(), dm = nothing,
        ratio_threshold = nothing, min_similarity = nothing, max_distance = nothing
    )
    N1, D1 = size(desc1)
    N2, D2 = size(desc2)
    D1 == D2 || throw(DimensionMismatch("Descriptor dimensions must match: $D1 ≠ $D2"))
    (N1 < 1 || N2 < 1) && return NNMatch[]
    top_idx, top_sc, rev_idx = if dm === nothing
        T = _score_eltype(desc1, desc2)
        _nn_select(
            strategy, metric, dev, [0], [N1];
            gal = _ka_upload(dev, T, desc1), feat = _ka_upload(dev, T, desc2)
        )
    else
        size(dm) == (N1, N2) ||
            throw(DimensionMismatch("dm is $(size(dm)) for $N1 × $N2 descriptors"))
        _nn_select(strategy, metric, dev, [0], [N1]; S = _on_backend(dev, dm))
    end
    return _block_matches(
        strategy, metric, top_idx, top_sc, rev_idx, 1, 0, N1, N2;
        ratio_threshold, min_similarity, max_distance
    )
end

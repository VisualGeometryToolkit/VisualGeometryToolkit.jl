# =============================================================================
# spread_strongest — N detections spread over the image: SSC on KernelAbstractions
# =============================================================================
#
# Suppression via Square Covering (SSC) of Bailo, Rameau, Joo, Park, Bogdan and
# Kweon, "Efficient adaptive non-maximal suppression algorithms for homogeneous
# spatial keypoint distribution", Pattern Recognition Letters 106 (2018), ported
# from the authors' C++ (`ssc` in C++/include/anms.h of
# https://github.com/BAILOOL/ANMS-Codes, MIT License, Copyright (c) 2018
# Oleksandr Bailo): its integer binary search over the square's width, its
# initial range (the positive root of the grid-capacity quadratic, and
# ⌊√(M / N)⌋ at least 1), its tolerance band and its cell grid of side width / 2.
#
# The reference covers sequentially: in descending score order, a point whose
# cell is not yet covered is kept and covers the cells within `reach =
# ⌊width / c⌋ = 2` cells (Chebyshev) of its own. Two observations make it
# parallel without changing its result:
#
#   1. Only a cell's strongest point can be kept (keeping it covers its own
#      cell), so the sweep reduces to one point per occupied cell, found by two
#      atomic reductions (the largest score, then the smallest index at that score:
#      the order of the reference's stable descending sort, exactly, with no
#      sort at all).
#   2. A cell is kept iff no STRONGER kept cell lies within `reach`: the
#      reference's result is the greedy (lexicographically first) maximal
#      independent set of the occupied cells under "within reach", in score
#      order. That set is computed in synchronous rounds (Blelloch, Fineman and
#      Shun, SPAA 2012): an undecided cell next to a kept one is removed; an
#      undecided cell stronger than every undecided neighbour is kept. Each
#      round reads the previous round's states and writes the next (double
#      buffered, no race), and every cell decided is decided as the sequential
#      sweep decides it, so the kept set is IDENTICAL to the reference's, ties
#      included.
#
# Per binary-search step: one kernel for the cells, two for the reductions, the
# rounds (each the strongest undecided cell is decided, so they end; on the
# Tamron frames a handful), one count. The search itself is a few scalars on
# the host.
# =============================================================================

import Atomix

const _SSC_UNDECIDED = 0x00
const _SSC_KEPT = 0x01
const _SSC_REMOVED = 0x02
const _SSC_NO_POINT = typemax(Int32)

# The cell of each point on the grid of side c (the reference's `floor(pt / c)`,
# clamped onto the grid), as a linear index into the (ncols + 1) × (nrows + 1) grid.
@kernel function _ssc_cells!(cell, @Const(x), @Const(y), c::Float64, ncols::Int32, nrows::Int32)
    i = @index(Global)
    @inbounds begin
        col = clamp(unsafe_trunc(Int32, floor(x[i] / c)), Int32(0), ncols)
        row = clamp(unsafe_trunc(Int32, floor(y[i] / c)), Int32(0), nrows)
        cell[i] = row * (ncols + Int32(1)) + col + Int32(1)
    end
end

# Each cell's largest score key (0: an empty cell).
@kernel function _ssc_cell_max!(best, @Const(cell), @Const(key))
    i = @index(Global)
    @inbounds Atomix.@atomic max(best[cell[i]], key[i])
end

# Each cell's smallest point index at its largest score: the reference's first point there.
@kernel function _ssc_cell_first!(first, @Const(best), @Const(cell), @Const(key))
    i = @index(Global)
    @inbounds if key[i] == best[cell[i]]
        Atomix.@atomic min(first[cell[i]], Int32(i))
    end
end

# The grid back to empty, cell by occupied cell (O(points), not O(cells)).
@kernel function _ssc_clear!(best, first, @Const(cell))
    i = @index(Global)
    @inbounds begin
        best[cell[i]] = zero(eltype(best))
        first[cell[i]] = _SSC_NO_POINT
    end
end

# One round of the greedy maximal independent set over the occupied cells: point i,
# its cell's first point, removed next to a kept cell, kept when stronger than every
# undecided neighbour, else undecided (and `pending` raised).
@kernel function _ssc_round!(snew, @Const(sold), pending, @Const(first), @Const(cell), @Const(key),
                             ncols::Int32, nrows::Int32, reach::Int32)
    i = @index(Global)
    @inbounds begin
        s = sold[i]
        if s == _SSC_UNDECIDED && first[cell[i]] == i
            l = cell[i] - Int32(1)
            row = l ÷ (ncols + Int32(1)); col = l - row * (ncols + Int32(1))
            ki = key[i]
            removed = false; blocked = false
            for r in max(row - reach, Int32(0)):min(row + reach, nrows), cc in max(col - reach, Int32(0)):min(col + reach, ncols)
                q = first[r * (ncols + Int32(1)) + cc + Int32(1)]
                (q == _SSC_NO_POINT || q == i) && continue
                sq = sold[q]
                if sq == _SSC_KEPT
                    removed = true
                elseif sq == _SSC_UNDECIDED && (key[q] > ki || (key[q] == ki && q < i))
                    blocked = true
                end
            end
            s = removed ? _SSC_REMOVED : blocked ? _SSC_UNDECIDED : _SSC_KEPT
            s == _SSC_UNDECIDED && (pending[1] = Int32(1))
        end
        snew[i] = s
    end
end

@kernel function _ssc_count!(n, @Const(state))
    i = @index(Global)
    @inbounds state[i] == _SSC_KEPT && Atomix.@atomic n[1] += Int32(1)
end

# The order of a score as an unsigned key: |s|'s bits (monotone for nonnegative
# floats) plus one, so that 0 is the empty cell.
_ssc_key(s::Real) = reinterpret(UInt64, Float64(abs(s))) + UInt64(1)

# The device buffers of one selection, the grid grown as the search narrows the width.
mutable struct _SSCBuffers{V, U, I, B}
    x::V; y::V; key::U; cell::I; s0::B; s1::B; pending::I; count::I
    best::U; first::I
end

# One covering at `width`: the kept count, the states (KEPT on the kept points) in `b.s0`.
function _ssc_cover!(b::_SSCBuffers, dev, width::Real, cols::Int, rows::Int)
    M = length(b.key)
    c = width / 2.0
    ncols = floor(Int, cols / c); nrows = floor(Int, rows / c)
    reach = floor(Int, width / c)
    ncell = (ncols + 1) * (nrows + 1)
    if length(b.best) < ncell                  # a narrower square than any before: a larger grid, empty
        b.best = KA.zeros(dev, UInt64, ncell)
        b.first = KA.allocate(dev, Int32, ncell); fill!(b.first, _SSC_NO_POINT)
    end
    # each launch through `_launch!` (foerstner.jl), completed before the next: behind its untyped barrier the
    # kernel is not a Union over an abstract backend's kernels (no JET report)
    _launch!(_ssc_cells!, dev, b.cell, b.x, b.y, c, Int32(ncols), Int32(nrows); ndrange = M)
    _launch!(_ssc_cell_max!, dev, b.best, b.cell, b.key; ndrange = M)
    _launch!(_ssc_cell_first!, dev, b.first, b.best, b.cell, b.key; ndrange = M)
    fill!(b.s0, _SSC_UNDECIDED)
    while true
        fill!(b.pending, Int32(0))
        _launch!(_ssc_round!, dev, b.s1, b.s0, b.pending, b.first, b.cell, b.key, Int32(ncols), Int32(nrows), Int32(reach); ndrange = M)
        b.s0, b.s1 = b.s1, b.s0
        Array(b.pending)[1] == 0 && break
    end
    _launch!(_ssc_clear!, dev, b.best, b.first, b.cell; ndrange = M)
    fill!(b.count, Int32(0))
    _launch!(_ssc_count!, dev, b.count, b.s0; ndrange = M)
    return Int(Array(b.count)[1])
end

# The indices (into `x`, `y`, `scores`) SSC keeps, ascending.
function _ssc_select(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, scores::AbstractVector{<:Real},
                     cols::Int, rows::Int, n::Int, tolerance::Real, dev::KA.Backend)
    M = length(scores)
    all(isfinite, scores) || throw(ArgumentError("spread_strongest: every score must be finite"))
    # the reference's initial range: the positive root of the grid-capacity quadratic, and ⌊√(M / N)⌋ at least 1
    exp1 = rows + cols + 2n
    exp2 = 4 * Int64(cols) + 4 * Int64(n) + 4 * Int64(rows) * n + Int64(rows)^2 + Int64(cols)^2 -
           2 * Int64(rows) * cols + 4 * Int64(rows) * cols * n
    exp3 = sqrt(Float64(exp2)); exp4 = Float64(n - 1)
    sol1 = -round((exp1 + exp3) / exp4); sol2 = -round((exp1 - exp3) / exp4)
    high = Int(max(sol1, sol2))
    low = max(1, floor(Int, sqrt(M / n)))
    # the tolerance band as the reference computes it (single precision, rounded half away from zero)
    K = Float32(n); t = Float32(tolerance)
    kmin = round(Int, K - K * t, RoundNearestTiesAway); kmax = round(Int, K + K * t, RoundNearestTiesAway)
    b = _SSCBuffers(on_device(Float64, dev, x), on_device(Float64, dev, y), on_device(UInt64, dev, _ssc_key.(scores)),
                    KA.allocate(dev, Int32, M), KA.allocate(dev, UInt8, M), KA.allocate(dev, UInt8, M),
                    KA.allocate(dev, Int32, 1), KA.allocate(dev, Int32, 1), KA.zeros(dev, UInt64, 0), KA.allocate(dev, Int32, 0))
    # the reference's search over integer widths; the coverings either side of the band are kept
    prev = -1
    more = less = nothing                       # (width, states) of the widest covering above the band, the narrowest below
    while true
        width = low + (high - low) ÷ 2
        (width == prev || low > high) && break
        k = _ssc_cover!(b, dev, width, cols, rows)
        kmin <= k <= kmax && return findall(==(_SSC_KEPT), Array(b.s0))
        if k < kmin
            high = width - 1; less = (width, Array(b.s0))
        else
            low = width + 1; more = (width, Array(b.s0))
        end
        prev = width
    end
    # No integer width lands in the band (the reference would return its last covering): the width between the two
    # integers that bracket the band, by bisection (the count falls as the square widens)
    if more !== nothing && less !== nothing
        lo, hi = Float64(more[1]), Float64(less[1])
        for _ in 1:_SSC_BISECTIONS
            w = (lo + hi) / 2
            k = _ssc_cover!(b, dev, w, cols, rows)
            kmin <= k <= kmax && return findall(==(_SSC_KEPT), Array(b.s0))
            k < kmin ? (hi = w) : (lo = w; more = (w, Array(b.s0)))
        end
    end
    # still none: the covering above the band nearest it, less its weakest, down to the band's top; else the covering below it
    more === nothing && return less === nothing ? Int[] : findall(==(_SSC_KEPT), less[2])
    kept = findall(==(_SSC_KEPT), more[2])
    order = sortperm(kept; by = i -> (_ssc_key(scores[i]), -i), rev = true)
    return sort!(kept[order[1:kmax]])
end

# Real-valued bisections of the width after the reference's integer search misses the band: 2⁻³⁰ px.
const _SSC_BISECTIONS = 30

"""
    spread_strongest(detections, image_size, n; tolerance = 0.1, device = default_device()) -> detections

About `n` of the strongest detections spread over the image: Suppression via
Square Covering (SSC; Bailo et al., Pattern Recognition Letters 106, 2018,
ported from the authors' code, https://github.com/BAILOOL/ANMS-Codes, MIT).
In descending `|response|` (ties: the earlier detection first), a detection is
kept unless a kept one lies in the square of side `w` about it (on a grid of
cells of side `w / 2`); the side `w` is an integer number of pixels, found by
the authors' binary search until the kept count is within
`round(n ∓ n·tolerance)`. The kept detections are returned in detector order.

Where the global top `n` by `|response|` would sit (on a frame with a
bright window and a dim interior) nearly all in the window, this keeps
the strongest of each neighbourhood, at whatever contrast the neighbourhood has.

`image_size` is `(H, W)` or a channelview's `(C, H, W)`; the detections'
centroids are in its COLMAP pixel frame (`[0, W] × [0, H]`). With `n` or fewer
detections, `detections` itself is returned; `n = 1` is the single strongest.
Where no integer width lands in the band (the reference then returns its last
covering, outside the band), the width is refined between the two bracketing
integers by bisection, and if the count still jumps over the band, the covering
above it loses its weakest detections down to the band's top; whenever the
reference's search lands in the band, the result is the reference's. The count
is below the band only when every covering is (fewer detections than the band
can hold).

One KernelAbstractions implementation on the CPU or any GPU backend (`device`):
the covering is computed in parallel (a per-cell maximum, then synchronous
rounds of the greedy independent set of the cells in score order), and its
result is the reference's sequential sweep's exactly, ties included; every
backend returns the same set, run to run.
"""
function spread_strongest(detections, image_size, n::Integer; tolerance::Real = 0.1,
                          device::KA.Backend = default_device())
    n >= 0 || throw(ArgumentError("spread_strongest: n must be nonnegative, got $n"))
    0 <= tolerance < 1 || throw(ArgumentError("spread_strongest: the tolerance must be in [0, 1), got $tolerance"))
    length(detections) <= n && return detections
    n == 0 && return detections[Int[]]
    scores = abs.(getproperty.(detections, :response))
    if n == 1                                   # one detection: the strongest (the reference's quadratic is singular there)
        return detections[[argmax(scores)]]
    end
    H, W = _image_hw(image_size)
    cs = centroid.(detections)
    keep = _ssc_select(first.(cs), last.(cs), scores, Int(W), Int(H), Int(n), tolerance, device)
    return detections[keep]
end

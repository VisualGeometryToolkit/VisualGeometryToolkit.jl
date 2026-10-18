# =============================================================================
# src/tuning/grids.jl — the persisted GRID SPEC of a tuning axis.
#
# A run persists the SPEC of every grid it tuned on, never only the realized
# values: `grid(spec)` rebuilds the grid, and the resume, merge and pin checks
# compare specs. An expand-on-boundary extension is part of the spec (the
# points added on each side), so an extended grid is reproducible too.
# =============================================================================

"""
    GridSpec

The persisted, reproducible description of one realized tuning grid:
[`SpacedGrid`](@ref) (a window, a spacing and its extension) or
[`ListedGrid`](@ref) (explicit values). `grid(spec)` realizes it.
"""
abstract type GridSpec end

"""
    SpacedGrid(lo, hi, spacing, n_lo = 0, n_hi = 0)

The grid `grid(lo, hi, spacing)` extended by `n_lo` points below `lo` and
`n_hi` above `hi` at the same step (log or linear). TOML tag `"spaced"`.
"""
struct SpacedGrid <: GridSpec
    lo::Float64
    hi::Float64
    spacing::GridSpacing
    n_lo::Int
    n_hi::Int
    function SpacedGrid(lo::Real, hi::Real, spacing::GridSpacing, n_lo::Integer, n_hi::Integer)
        _check_window(Float64(lo), Float64(hi), spacing)
        n_lo >= 0 && n_hi >= 0 || throw(ArgumentError("SpacedGrid: extension counts must be ≥ 0, got ($n_lo, $n_hi)"))
        return new(Float64(lo), Float64(hi), spacing, Int(n_lo), Int(n_hi))
    end
end
SpacedGrid(lo::Real, hi::Real, spacing::GridSpacing) = SpacedGrid(lo, hi, spacing, 0, 0)

"""
    ListedGrid(values)

A grid given by its values (finite, strictly increasing): the
factorial tunings whose axes are hand-chosen settings. It has no spacing, so
it cannot be extended. TOML tag `"listed"`.
"""
struct ListedGrid <: GridSpec
    values::Vector{Float64}
    function ListedGrid(values::AbstractVector{<:Real})
        v = collect(Float64, values)
        length(v) >= 1 || throw(ArgumentError("ListedGrid: no values"))
        all(isfinite, v) && issorted(v; lt = <=) ||
            throw(ArgumentError("ListedGrid: values must be finite and strictly increasing, got $v"))
        return new(v)
    end
end

Base.:(==)(a::SpacedGrid, b::SpacedGrid) =
    a.lo === b.lo && a.hi === b.hi && a.spacing == b.spacing && a.n_lo == b.n_lo && a.n_hi == b.n_hi
Base.:(==)(a::ListedGrid, b::ListedGrid) = a.values == b.values
Base.hash(g::SpacedGrid, h::UInt) = hash((:SpacedGrid, g.lo, g.hi, g.spacing, g.n_lo, g.n_hi), h)
Base.hash(g::ListedGrid, h::UInt) = hash((:ListedGrid, g.values), h)

# One range over the extended window: the base step continued `n_lo` steps
# below `lo` and `n_hi` above `hi` (a log spacing's step is a ratio).
function grid(g::SpacedGrid)
    n = npoints(g.lo, g.hi, g.spacing)
    m = n + g.n_lo + g.n_hi
    g.spacing isa LinearCount || return logrange(g.lo / (g.hi / g.lo)^(g.n_lo / (n - 1)),
                                                 g.hi * (g.hi / g.lo)^(g.n_hi / (n - 1)), m)
    Δ = (g.hi - g.lo) / (n - 1)
    return range(g.lo - g.n_lo * Δ, g.hi + g.n_hi * Δ; length = m)
end
grid(g::ListedGrid) = copy(g.values)

"""
    extend(spec::SpacedGrid, sides; npts) -> (spec′, n_lo)

THE expand-on-boundary extension of a grid: `npts` more points on each side in
`sides ⊆ (:lo, :hi)`, continuing the base step. Every point of `grid(spec)`
reappears (to rounding) in `grid(spec′)` at index `+ n_lo`, the count
prepended, so already-computed columns stay valid by index (see
[`extension_offset`](@ref)). A [`ListedGrid`](@ref) has no spacing to continue
and refuses.
"""
function extend(g::SpacedGrid, sides; npts::Integer)
    npts >= 1 || throw(ArgumentError("extend: npts = $npts must be ≥ 1"))
    issubset(sides, (:lo, :hi)) || throw(ArgumentError("extend: sides must be ⊆ (:lo, :hi), got $(collect(sides))"))
    n_lo = :lo in sides ? Int(npts) : 0
    n_hi = :hi in sides ? Int(npts) : 0
    return SpacedGrid(g.lo, g.hi, g.spacing, g.n_lo + n_lo, g.n_hi + n_hi), n_lo
end
extend(g::ListedGrid, sides; npts::Integer) =
    throw(ArgumentError("extend: a listed grid has no spacing to continue ($(g.values))"))

"""
    extension_offset(base::GridSpec, sup::GridSpec) -> Int | nothing

The offset `n_lo ≥ 0` at which `grid(base)` sits inside `grid(sup)` when `sup`
is `base` or an extension of it — same window and spacing, at least as many
points on each side — else `nothing`. THE compatibility test of a resumed
checkpoint, a merged sidecar and a pinned grid: specs are compared, never
realized values.
"""
function extension_offset(base::SpacedGrid, sup::SpacedGrid)
    (base.lo === sup.lo && base.hi === sup.hi && base.spacing == sup.spacing) || return nothing
    (sup.n_lo >= base.n_lo && sup.n_hi >= base.n_hi) || return nothing
    return sup.n_lo - base.n_lo
end
extension_offset(base::ListedGrid, sup::ListedGrid) = base == sup ? 0 : nothing
extension_offset(::GridSpec, ::GridSpec) = nothing

"""
    nearest_column(grid, x) -> Int

Index of the grid column nearest `x` in LOG space — the deployment evaluation
helper (e.g. the column of the ζ-scaled tuned native). Evaluation only; tuning
is always a windowed free argmin, never nearest-column selection.
"""
nearest_column(grid::AbstractVector{<:Real}, x::Real) =
    argmin(abs.(log.(grid) .- log(Float64(x))))

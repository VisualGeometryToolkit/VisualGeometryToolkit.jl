# =============================================================================
# src/tuning/spacing.jl — grid spacing strategies and THE grid constructor.
#
# A spacing is a strategy TYPE whose fields are its complete parameter set;
# `grid(lo, hi, spacing)` is the one constructor of a tuning grid over the
# window [lo, hi]: a Base `range` (linear) or `logrange` (log), its endpoints
# exactly `lo` and `hi`, at least two points.
# =============================================================================

"""
    GridSpacing

How the points of a tuning grid are spaced over a window `[lo, hi]`:
[`LogPerDecade`](@ref), [`LogPerOctave`](@ref), [`LogCount`](@ref),
[`LinearCount`](@ref). [`grid`](@ref) builds the grid; `extend`
continues a spaced grid past either end at the same step.
"""
abstract type GridSpacing end

"""
    LogPerDecade(per_decade)

Log spacing at a DENSITY: `d = log10(hi / lo)` decades hold `d·per_decade`
intervals, so the grid has `round(Int, d·per_decade) + 1` points (at least 2)
and its log10 step is `d / (n - 1)` — `1 / per_decade` when `d·per_decade`
is an integer. (`d·per_decade` POINTS would space them coarser
than advertised: the fencepost.) TOML tag `"log_per_decade"`.
"""
struct LogPerDecade <: GridSpacing
    per_decade::Int
    function LogPerDecade(per_decade::Integer)
        per_decade >= 1 || throw(ArgumentError("LogPerDecade: per_decade = $per_decade must be ≥ 1"))
        return new(Int(per_decade))
    end
end

"""
    LogPerOctave(per_octave)

Log spacing at a density per factor of two: `round(Int, log2(hi / lo)·per_octave) + 1`
points (at least 2) — the half-octave ladders (`per_octave = 2`) of the PnP
and detector tunings. TOML tag `"log_per_octave"`.
"""
struct LogPerOctave <: GridSpacing
    per_octave::Int
    function LogPerOctave(per_octave::Integer)
        per_octave >= 1 || throw(ArgumentError("LogPerOctave: per_octave = $per_octave must be ≥ 1"))
        return new(Int(per_octave))
    end
end

"""
    LogCount(n)

`n ≥ 2` log-spaced points over the window (a COUNT, whatever the window's
width in decades). TOML tag `"log_count"`.
"""
struct LogCount <: GridSpacing
    n::Int
    function LogCount(n::Integer)
        n >= 2 || throw(ArgumentError("LogCount: n = $n must be ≥ 2"))
        return new(Int(n))
    end
end

"""
    LinearCount(n)

`n ≥ 2` linearly spaced points over the window. TOML tag `"linear_count"`.
"""
struct LinearCount <: GridSpacing
    n::Int
    function LinearCount(n::Integer)
        n >= 2 || throw(ArgumentError("LinearCount: n = $n must be ≥ 2"))
        return new(Int(n))
    end
end

"The number of points `grid(lo, hi, s)` has."
npoints(lo::Real, hi::Real, s::LogPerDecade) = max(2, round(Int, log10(hi / lo) * s.per_decade) + 1)
npoints(lo::Real, hi::Real, s::LogPerOctave) = max(2, round(Int, log2(hi / lo) * s.per_octave) + 1)
npoints(::Real, ::Real, s::Union{LogCount,LinearCount}) = s.n

_is_log(::GridSpacing) = true
_is_log(::LinearCount) = false

function _check_window(lo::Float64, hi::Float64, s::GridSpacing)
    (isfinite(lo) && isfinite(hi) && lo < hi) ||
        throw(ArgumentError("grid: need finite lo < hi, got [$lo, $hi]"))
    _is_log(s) && lo <= 0 && throw(ArgumentError("grid: a log spacing needs 0 < lo, got [$lo, $hi]"))
    return nothing
end

"""
    grid(lo, hi, spacing::GridSpacing) -> AbstractRange{Float64}
    grid(spec::GridSpec) -> AbstractVector{Float64}

THE tuning-grid constructor: `npoints(lo, hi, spacing)` points of `spacing`
over `[lo, hi]` — Base `range(lo, hi; length)` for a linear spacing, Base
`logrange(lo, hi, n)` for a log one (both hit `lo` and `hi` exactly). The
second form realizes a persisted [`GridSpec`](@ref), its extension included.
"""
grid(lo::Real, hi::Real, s::GridSpacing) = grid(SpacedGrid(lo, hi, s))   # grids.jl

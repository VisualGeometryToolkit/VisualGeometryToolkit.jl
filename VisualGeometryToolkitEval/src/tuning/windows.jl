# =============================================================================
# src/tuning/windows.jl — the efficiency-constant window and the WINDOW rules of a
# tuning axis.
#
# A window rule computes an axis's window `(lo, hi)`; the spacing then builds
# the grid over it. Separating the two is what makes a data-dependent window
# (the coverage rule reads the validation pool's σ̂² quantiles) reproducible:
# the realized window is persisted in the axis's GridSpec, and the rule with
# its parameters in the tuning spec.
# =============================================================================

"""
    EfficiencyConstantWindow(lo, hi)

The standardized window `[lo, hi]` of efficiency constants (a width as a multiple
of the estimated inlier scale, σ-units) of a tuning run: the dimensionless
window every windowed free argmin searches, mapped to a native axis at the
supplied scale (`window_columns`), and the window of the efficiency-constant axis
itself ([`StandardizedWindow`](@ref)). It carries no grid: the grids are
built by the spacings.
"""
struct EfficiencyConstantWindow
    lo::Float64
    hi::Float64
    function EfficiencyConstantWindow(lo::Real, hi::Real)
        0 < lo < hi || throw(ArgumentError("EfficiencyConstantWindow: need 0 < lo < hi, got [$lo, $hi]"))
        return new(Float64(lo), Float64(hi))
    end
end

"""
    extend(w::EfficiencyConstantWindow, sides; factor) -> EfficiencyConstantWindow

The expand-on-boundary widening of a window ([`WindowExtension`](@ref)): its
`lo` divided by `factor` if `:lo in sides`, its `hi` multiplied if `:hi in
sides` (`sides ⊆ (:lo, :hi)`, `factor > 1`); the grid's twin is
`extend(::SpacedGrid, sides; npts)`.
"""
function extend(w::EfficiencyConstantWindow, sides; factor::Real)
    factor > 1 || throw(ArgumentError("extend: factor = $factor must be > 1"))
    issubset(sides, (:lo, :hi)) || throw(ArgumentError("extend: sides must be ⊆ (:lo, :hi), got $(collect(sides))"))
    return EfficiencyConstantWindow(:lo in sides ? w.lo / factor : w.lo, :hi in sides ? w.hi * factor : w.hi)
end

"""
    WindowRule

How an axis's window `(lo, hi)` is found: [`FixedWindow`](@ref),
[`StandardizedWindow`](@ref), [`CoverageWindow`](@ref),
[`CenteredWindow`](@ref). `window_bounds(rule; context...)` evaluates it; each rule
takes only the context it needs and refuses a missing one.

Factored, a window is STATED or PLACED, and the placed ones differ only in
their ANCHOR and how their endpoints leave it:

| rule | anchor | endpoints |
|---|---|---|
| `FixedWindow` | — | its own two native numbers |
| `StandardizedWindow` | — | the efficiency-constant window verbatim |
| `CoverageWindow` | `(ŝ_min/ζ, ŝ_max·ζ)`, per side | the window through `native`, ± margin |
| `CenteredWindow` | a supplied native point | ÷/× factor |

Note where ζ sits: it scales the ANCHOR, never the endpoints. A window whose
anchor spreads is asserting that the scale could be anywhere in that spread —
right for a ζ ladder or a budget sweep, wrong for a tune, which has one
reference scale. Anchor spread is bought with the grid's resolution.
"""
abstract type WindowRule end

"""
    FixedWindow(lo, hi)

A window stated in the axis's native units. TOML tag `"fixed"`.
"""
struct FixedWindow <: WindowRule
    lo::Float64
    hi::Float64
    function FixedWindow(lo::Real, hi::Real)
        lo < hi || throw(ArgumentError("FixedWindow: need lo < hi, got [$lo, $hi]"))
        return new(Float64(lo), Float64(hi))
    end
end

"""
    StandardizedWindow()

The run's [`EfficiencyConstantWindow`](@ref) itself: the window of the
dimensionless efficiency-constant axis. TOML tag `"standardized"`.
"""
struct StandardizedWindow <: WindowRule end

"""
    CoverageWindow(q_lo, q_hi, log_margin, zeta_octaves)

The COVERAGE window of a native axis (outlier half-width, loss width): every
native the efficiency-constant window can request at any reference scale the
validation pool can produce, `[native(lo, ŝ_min/ζ)·e^{-m}, native(hi, ŝ_max·ζ)·e^{m}]`
with `[lo, hi]` the efficiency-constant window, `(ŝ_min, ŝ_max) = scale_bounds(s2,
(q_lo, q_hi))`, `ζ = 2^zeta_octaves` (the scale corruptions the run reads out
at, 0 for none) and `m = log_margin` (log headroom so a request at the bounds
sits strictly inside). TOML tag `"coverage"`.

The name is the GUARANTEE, not the mechanism: the realized grid contains every
native the window can request at any scale the pool produces, so a windowed
free argmin provably cannot leave its grid. That is what the validation-budget
mode needs, where 3500 draws share one sweep and each has its own `ŝ(n)`.

`q_lo == q_hi` is legal and is the ONE-ANCHOR case: both sides anchor on the
same `√quantile(σ̂², q)`, and at `q = 0.5` that is `reference_scale(s2)`, so
`coverage { q_lo = 0.5, q_hi = 0.5, zeta_octaves = 0.0 }` is the WINDOW-MATCHED
grid — the native image of the efficiency-constant window and nothing more. A tune
has one reference scale, so this is the tuning case; a spread band is for the
modes that have many. (It replaced a separate `StandardizedNativeWindow` rule,
which was this rule with its band collapsed and was a distinct type only
because the constructor forbade `q_lo == q_hi`.)
"""
struct CoverageWindow <: WindowRule
    q_lo::Float64
    q_hi::Float64
    log_margin::Float64
    zeta_octaves::Float64
    function CoverageWindow(q_lo::Real, q_hi::Real, log_margin::Real, zeta_octaves::Real)
        0 <= q_lo <= q_hi <= 1 ||
            throw(ArgumentError("CoverageWindow: need 0 ≤ q_lo ≤ q_hi ≤ 1, got ($q_lo, $q_hi)"))
        log_margin >= 0 && zeta_octaves >= 0 ||
            throw(ArgumentError("CoverageWindow: log_margin and zeta_octaves must be ≥ 0"))
        return new(Float64(q_lo), Float64(q_hi), Float64(log_margin), Float64(zeta_octaves))
    end
end

"""
    CenteredWindow(factor)

`[center / factor, center · factor]` about a center the caller supplies: a
transferred operating point centres on itself. TOML tag `"centered"`.
"""
struct CenteredWindow <: WindowRule
    factor::Float64
    function CenteredWindow(factor::Real)
        factor > 1 || throw(ArgumentError("CenteredWindow: factor = $factor must be > 1"))
        return new(Float64(factor))
    end
end

_required(x, what, rule) = x === nothing ?
    throw(ArgumentError("window_bounds: the $(nameof(typeof(rule))) rule needs `$what`")) : x

"""
    window_bounds(rule::WindowRule; efficiency_constant_window = nothing, native = nothing,
           s2 = nothing, center = nothing) -> (lo, hi)

Evaluate a window rule. `efficiency_constant_window` is the run's
[`EfficiencyConstantWindow`](@ref), `native(efficiency_constant, scale)` the axis's
native map (the caller's `native_map(axis; d_g)`), `s2` the validation pool's
per-pair `σ̂²`, `center` the axis's centre; a rule refuses a context it needs
and is not given.
"""
window_bounds(r::FixedWindow; _...) = (r.lo, r.hi)
window_bounds(r::StandardizedWindow; efficiency_constant_window = nothing, _...) =
    (w = _required(efficiency_constant_window, :efficiency_constant_window, r); (w.lo, w.hi))
function window_bounds(r::CoverageWindow; efficiency_constant_window = nothing, native = nothing, s2 = nothing, _...)
    w = _required(efficiency_constant_window, :efficiency_constant_window, r)
    f = _required(native, :native, r); v = _required(s2, :s2, r)
    scale_min, scale_max = scale_bounds(v, (r.q_lo, r.q_hi))
    ζ = 2.0^r.zeta_octaves
    return (exp(log(Float64(f(w.lo, scale_min / ζ))) - r.log_margin),
            exp(log(Float64(f(w.hi, scale_max * ζ))) + r.log_margin))
end
window_bounds(r::CenteredWindow; center = nothing, _...) =
    (x = Float64(_required(center, :center, r)); (x / r.factor, x * r.factor))

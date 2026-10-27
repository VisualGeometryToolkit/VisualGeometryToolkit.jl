# =============================================================================
# lm_core.jl — the stack's Levenberg–Marquardt: NLLSsolver on the chart-LM row
# protocol.
#
# Every nonlinear least-squares fit of the stack runs NLLSsolver. A refit on a
# chart — the √w-weighted rows of one linearization state, `residual_rows!` /
# `residual_jacobian_rows!` below — runs through ONE adapter, `lmsolve_chart!`:
# the iterate is the N-dimensional chart update x, started at zero (a chart is
# local, centred at the base model) and returned as an `SVector{N}`; the chart
# is fixed for the run and the caller re-centres it once, at the solution.
# VisualGeometryToolkitEstimation's LO refits (H: N = 8, F: 7, E: 5, PnP and hand-eye: 6,
# the radial homography: 9, the radial camera: 5), its `fit_lens` and its
# `refine_ellipse` run the chart LM; its block-structured fits (the
# relative-pose bundle, `refine_robot_world_hand_eye`) build their own
# NLLSsolver problems and share the adapter's knob map (`nlls_options`) and
# stop decoding (`nlls_retcode`, the stall postcondition `is_stationary`).
#
# The adapter presents a linearization state as an NLLSsolver problem: one
# variable, the chart vector (an `SVector{N}`, NLLSsolver's `EuclideanVector`),
# and one residual block of all the rows, whose cost, gradient and Gauss–Newton
# Hessian it supplies through NLLSsolver's `computecostgradhess` hook as static
# arrays (Jᵀr and JᵀJ accumulated in one sweep over the rows: measured
# 2026-10-27 on 317 captured E refits, 0.84× FastLevenbergMarquardt's time to
# convergence, against 0.97× with NLLSsolver forming JᵀJ itself). The problem
# is built once per (workspace, linearization-state type) and kept in the
# workspace with NLLSsolver's `OptimizeCache`, so a refit reuses the problem
# and the solver's internal data.
#
# The fork's additions (github.com/prittjam/NLLSsolver.jl, branch `stack`:
# upstream pull requests #60–#63 and the static retained blocks) are DETECTED,
# never assumed: `OptimizeCache` here, the Schur-complement linear system in
# VisualGeometryToolkitEstimation's bundle (`nlls_features`). On the registered
# NLLSsolver every fit takes the upstream path — the same results, slower.
# =============================================================================

"""
    nlls_features() -> (; cache::Bool, schur::Bool)

Which of the additions of the stack's NLLSsolver fork the loaded NLLSsolver
has: `cache`, `NLLSsolver.OptimizeCache` (reuse of a problem's internal data
across `optimize!` calls; upstream pull request #62); `schur`, the
Schur-complement linear system for eliminable variables
(`NLLSsolver.MultiVariateLSschur`, `NLLSsolver.iseliminable`; #60). Every fit
of the stack is correct without them and uses them when they are there; a
checkout of the stack pins the fork, a registry install does not (see the
README's "Recovering the full speed of the LM driver").
"""
nlls_features() = (; cache = _NLLS_HAS_CACHE, schur = _NLLS_HAS_SCHUR)
const _NLLS_HAS_CACHE = isdefined(NLLSsolver, :OptimizeCache)
const _NLLS_HAS_SCHUR = isdefined(NLLSsolver, :MultiVariateLSschur) && isdefined(NLLSsolver, :iseliminable)

# NLLSsolver's consecutive-failure cap (its default, `maxfails = 3`) and no time
# cap (NLLSsolver's default is 30 s; 1e6 s never fires, and a fit that hits it
# would read as an iteration cap).
const _NLLS_MAXFAILS = 3
const _NLLS_NO_TIME_CAP = 1.0e6

"""
    nlls_options(; max_iter, ftol, xtol, iterator = :levenbergmarquardt_dtd) -> NLLSsolver.NLLSOptions

THE map of the stack's documented LM knobs onto NLLSsolver's options, the one
place they meet:

| stack knob | meaning | NLLSsolver |
|---|---|---|
| `max_iter` (`maxit`) | the LM iteration cap | `maxiters`: outer iterations, one accepted step each (FastLevenbergMarquardt counted rejected trials too) |
| `ftol` | relative cost tolerance: stop once a step lowers the cost by less than `ftol`·F | `reldcost` |
| `xtol` | step-size tolerance (FastLevenbergMarquardt's default `1e-10`) | `dstep`: the step's max-norm (chart units) |
| — | absolute cost decrease | `absdcost = 0`: the stack has no such knob |
| — | consecutive failed iterations | `maxfails = 3`, NLLSsolver's default |
| — | time cap | none (1e6 s) |
| `iterator` | the damping | `:levenbergmarquardt_dtd`, λ·DᵀD with DᵀD the running maximum of diag(JᵀJ) — Marquardt's scaling, which every former FastLevenbergMarquardt fit used — or `:levenbergmarquardt`, λ·I |

Single-threaded (`numthreads = 1`): a fit runs inside the caller's threads.
FastLevenbergMarquardt's `gtol` (`‖Jᵀf‖∞ < eps`) has no counterpart: a zero
gradient gives a zero step, which the step test stops and the stall
postcondition reads as stationary. Its initial damping, `1e-6`, is
NLLSsolver's λ₀ for `:levenbergmarquardt_dtd`; neither is a stack knob.
"""
nlls_options(; max_iter::Integer, ftol::Real, xtol::Real, iterator::Symbol = :levenbergmarquardt_dtd) =
    _nlls_options(NLLSsolver.static(iterator), max_iter, ftol, xtol)
# The map itself, on a static iterator: the chart LM calls it per refit with the
# constant `_DTD`, so the options are inferred and allocate nothing (a runtime
# Symbol made `static(iterator)` a dynamic call, three allocations per refit).
@inline _nlls_options(iterator, max_iter::Integer, ftol::Real, xtol::Real) =
    NLLSsolver.NLLSOptions(
        Float64(ftol), 0.0, Float64(xtol), _NLLS_MAXFAILS, Int(max_iter), UInt64(round(_NLLS_NO_TIME_CAP * 1.0e9)),
        iterator, NLLSsolver.static(1)
    )
const _DTD = NLLSsolver.static(:levenbergmarquardt_dtd)

"""
    ManifoldLmWorkspace{N}
    ManifoldLmWorkspace(n, N)

The buffers of the chart LM ([`lmsolve_chart!`](@ref)) at `n` residual rows
and an `N`-dimensional chart: the per-row √weights the caller's refit fills,
the residual and `n × N` Jacobian buffers the rows are evaluated into, the
stall postcondition's scratch, and the NLLSsolver problem the adapter builds
on first use (and keeps, with NLLSsolver's internal data, for every later
refit on a linearization state of the same type) — allocated once per problem
instead of per LM iteration. `N` is the model manifold's tangent size
(essential 5, fundamental 7, homography 8 in VisualGeometryToolkitEstimation). The
residual SEMANTICS come entirely from the caller's linearization state —
Sampson distance for E/F, whitened transfer error for H, calibrated
reprojection for P3P — which is why the type is named for the manifold, not
the distance.
"""
mutable struct ManifoldLmWorkspace{N}
    const sqrtw::Vector{Float64}   # per-row √weights — filled by VisualGeometryToolkitEstimation's LO refits, not read here
    const r::Vector{Float64}       # the rows at the iterate
    const J::Matrix{Float64}       # their n × N Jacobian
    const g::MVector{N, Float64}   # the stall postcondition's Jᵀr …
    const w::MVector{N, Float64}   # … and squared column norms
    run::Any                       # the `_ChartRun` of the last linearization-state type, or `nothing`
end
"The per-row √weight buffer of a [`ManifoldLmWorkspace`](@ref), filled by the caller's refit."
sqrt_weights(ws::ManifoldLmWorkspace) = ws.sqrtw

# The concrete workspace type at N unknowns (a caller that type-asserts a
# memoized workspace names this).
lm_workspace_type(::Val{N}) where {N} = ManifoldLmWorkspace{N}

ManifoldLmWorkspace(n::Integer, N::Integer) = ManifoldLmWorkspace(Int(n), Val(Int(N)))
ManifoldLmWorkspace(n::Int, ::Val{N}) where {N} =
    ManifoldLmWorkspace{N}(zeros(n), zeros(n), zeros(n, N), zeros(MVector{N, Float64}), zeros(MVector{N, Float64}), nothing)

# =============================================================================
# The chart-LM protocol — the rows a caller's linearization state fills
# =============================================================================

"""
    residual_rows!(r, lm, θ) -> r
    residual_jacobian_rows!(J, lm, θ) -> J

THE CHART-LM PROTOCOL — one pair for every model: the residual and Jacobian
rows the chart LM, NLLSsolver, minimizes in [`lmsolve_chart!`](@ref).

`lm` is the per-call LINEARIZATION STATE a refit builds once: the anchor the
chart is centred on, the whitening frozen there, the data views and the
√weights. `θ` is the CHART vector, not a model. The rows are √w-WEIGHTED and
live in chart coordinates — a different currency from a RANSAC problem's
UNWEIGHTED `residuals!(r, problem, model)` in data units. Two names, because
one name with two currencies is what the frame-in = frame-out rule exists to
prevent.

Ceres analogue: together they are what a `CostFunction::Evaluate` fills —
`residual_rows!` the `residuals` buffer, `residual_jacobian_rows!` the
`jacobians` — with the manifold's `retract_jacobian` (Ceres' `PlusJacobian`)
already folded into the chart directions. GTSAM's nearest equivalent is a
factor's `linearize(values)`, which returns the same rows and Jacobian as a
`JacobianFactor` at the linearization point rather than filling caller buffers.

Both write into caller-owned buffers and return them; neither allocates.
Methods live with the models (VisualGeometryToolkitEstimation's problems).
"""
function residual_rows! end

@doc (@doc residual_rows!)
function residual_jacobian_rows! end

# A linearization state that implements no rows names itself, rather than a
# bare MethodError from inside the LM's callbacks.
residual_rows!(r, lm, θ) = throw(ArgumentError("$(typeof(lm)) implements no `residual_rows!` method"))

# The Jacobian's default: ForwardDiff through `residual_rows!` (which must then be
# eltype-generic, so Duals flow through it), into `J`. A state with a closed form
# — or with structure the dense AD cannot see (per-point blocks) — overrides it.
# Untyped, so a state's own method (typed on `lm`) is always the more specific.
residual_jacobian_rows!(J, lm, θ) =
    J .= ForwardDiff.jacobian(z -> residual_rows!(Vector{eltype(z)}(undef, size(J, 1)), lm, z), θ)

# √wᵢ for each point's `rows` residual rows (2 for a 2-vector residual, 1 for a
# scalar one) — the one fill every LM refit's row scaling uses. The weights obey
# the one weights contract (`check_weights`: finite, non-negative), checked here.
@inline function fill_sqrt_weights!(sw::AbstractVector, weights::AbstractVector, n::Int, rows::Int = 1)
    check_weights(view(weights, 1:n))
    @inbounds for i in 1:n
        s = sqrt(Float64(weights[i]))
        for j in 1:rows
            sw[rows * (i - 1) + j] = s
        end
    end
    return sw
end

# =============================================================================
# The adapter — a linearization state as an NLLSsolver problem
# =============================================================================

# The linearization state of the current refit: the problem's one residual
# block reads it, and each refit replaces it (the problem itself is kept).
mutable struct _ChartState{L}
    lm::L
end

# The one residual block: every row of the state at the chart vector x,
# evaluated into the workspace's buffers.
struct _ChartRows{N, L} <: NLLSsolver.AbstractResidual
    state::_ChartState{L}
    r::Vector{Float64}
    J::Matrix{Float64}
end
NLLSsolver.ndeps(::_ChartRows) = NLLSsolver.static(1)
NLLSsolver.nres(c::_ChartRows) = length(c.r)
NLLSsolver.varindices(::_ChartRows) = SVector(1)
NLLSsolver.getvars(::_ChartRows{N}, vars::Vector) where {N} = (vars[1]::SVector{N, Float64},)
Base.eltype(::_ChartRows) = Float64
NLLSsolver.computeresidual(c::_ChartRows, x) = residual_rows!(c.r, c.state.lm, x)

# The cost ½‖r‖², the one sum both callbacks below take (same order, so the
# cost NLLSsolver reads at a trial point and at a linearization agree to the
# bit). A non-finite cost is +Inf: a candidate step whose rows are NaN (a step
# past a NaN or discontinuity wall — the radial-camera, radial-homography and
# lens rows have them) is a REJECTED step, as it was for FastLevenbergMarquardt,
# which compared F(x + p) < F(x); NLLSsolver accepts any trial whose cost is
# not greater than the best, which NaN is not, and would stop there.
@inline _finite_cost(c::Float64) = isfinite(c) ? c : Inf
function NLLSsolver.computecost(c::_ChartRows, x)
    r = residual_rows!(c.r, c.state.lm, x)
    s = 0.0
    @inbounds for i in eachindex(r)
        s += r[i] * r[i]
    end
    return _finite_cost(0.5 * s)
end

# The cost, the gradient Jᵀr and the Gauss–Newton Hessian JᵀJ in one sweep over
# the rows, as static arrays: NLLSsolver's own `jac' * jac` on a dynamic n × N
# Jacobian was the one NLLSsolver-specific cost of a refit (10% of its
# samples), and this layout is the measured 0.84× (lm_core.jl header).
function NLLSsolver.computecostgradhess(_varflags, c::_ChartRows{N}, x) where {N}
    lm = c.state.lm
    r = residual_rows!(c.r, lm, x)
    J = residual_jacobian_rows!(c.J, lm, x)
    g = zero(MVector{N, Float64})
    H = zero(MMatrix{N, N, Float64})
    s = 0.0
    @inbounds for i in eachindex(r)
        ri = r[i]
        s += ri * ri
        for a in 1:N
            Jia = J[i, a]
            g[a] += Jia * ri
            for b in 1:a
                H[a, b] += Jia * J[i, b]
            end
        end
    end
    @inbounds for a in 1:N, b in (a + 1):N
        H[a, b] = H[b, a]
    end
    return _finite_cost(0.5 * s), SVector(g), SMatrix(H)
end

const _NllsCache = _NLLS_HAS_CACHE ? NLLSsolver.OptimizeCache : Nothing
_nlls_cache() = _NLLS_HAS_CACHE ? NLLSsolver.OptimizeCache() : nothing

# The kept NLLSsolver problem of one linearization-state type, its state slot
# and its internal-data cache.
struct _ChartRun{N, L}
    state::_ChartState{L}
    problem::NLLSsolver.NLLSProblem{SVector{N, Float64}, _ChartRows{N, L}}
    cache::_NllsCache
end
function _ChartRun(ews::ManifoldLmWorkspace{N}, lm::L) where {N, L}
    state = _ChartState{L}(lm)
    problem = NLLSsolver.NLLSProblem(SVector{N, Float64}, _ChartRows{N, L})
    NLLSsolver.addvariable!(problem, zero(SVector{N, Float64}))
    NLLSsolver.addcost!(problem, _ChartRows{N, L}(state, ews.r, ews.J))
    return _ChartRun{N, L}(state, problem, _nlls_cache())
end

# The workspace's run for this state type: the kept one, or a new one kept in
# its place (a workspace serves one state type at a time; the LO's never
# changes type).
@inline function _chart_run(ews::ManifoldLmWorkspace{N}, lm::L) where {N, L}
    run = ews.run
    run isa _ChartRun{N, L} && return run
    new = _ChartRun(ews, lm)
    ews.run = new
    return new
end

# =============================================================================
# The chart LM — NLLSsolver on a linearization state, and its stop codes
# =============================================================================

# The chart LM's stop code from NLLSsolver's termination bits (`nlls_retcode`
# lists them), in the codes FastLevenbergMarquardt used and `lm_retcode`
# reads: a relative or absolute cost-decrease stop (bits 2, 3) is `1`, the
# analogue of FLM's ftol stop, converged; a step-size stop (6) is `2`, FLM's
# xtol stop; the iteration or time cap (8, 9) is `-1`; every other stop — a
# non-finite cost or step (0, 1, 4, 5), the consecutive-failure cap (7), a
# cost disagreement (10) — is `4`, a stop without a tolerance. A tolerance
# stop outranks a cap, as in FLM's loop (it tested its tolerances first). `2`
# and `4` prove nothing: the stall postcondition decides them.
function _lm_code(bits::Integer)
    bits & ((1 << 2) | (1 << 3)) != 0 && return 1
    bits & (1 << 6) != 0 && return 2
    bits & ((1 << 8) | (1 << 9)) != 0 && return -1
    return 4
end

"""
    _LM_STALLED

The code [`lmsolve_chart!`](@ref) returns in place of a step-size or
tolerance-free stop (`2`, `4`) when the returned point fails the stationarity
test: `-2`. [`lm_retcode`](@ref) maps it to `:stalled`.
"""
const _LM_STALLED = -2

"""
    lmsolve_chart!(ews::ManifoldLmWorkspace{N}, lm; maxit = 1000, ftol = eps(), xtol = 1e-10, stationarity_tol, cache = true)
        -> (x::SVector{N}, F, converged, iterations)

THE chart LM: NLLSsolver's Levenberg–Marquardt (Marquardt's DᵀD damping)
minimizing `‖r(x)‖²`, with `r` the [`residual_rows!`](@ref) of the
linearization state `lm` and its Jacobian [`residual_jacobian_rows!`](@ref),
from the chart origin `x = 0`. `x` is the solution chart vector, `F` the final
cost `‖r(x)‖²`, [`lm_retcode`](@ref)`(converged)` the estimator's retcode and
`iterations` NLLSsolver's iteration count (one accepted step each; 0 never: a
start at a stationary point takes one zero step).

`maxit`, `ftol` and `xtol` are the stack's LM knobs, mapped once by
[`nlls_options`](@ref); their defaults are FastLevenbergMarquardt's, which the
stack documented its configurations against. `stationarity_tol` is the stall
postcondition's tolerance — required: the caller's configuration owns it.
`cache = false` takes the upstream path where the fork's `OptimizeCache` is
loaded (`nlls_features`; the fallback test runs it): the same iterates, the
solver's internal data rebuilt per call.

THE STALL POSTCONDITION. A stop on the step size (`2`) or without a tolerance
(`4`) does not prove convergence: a run whose damping grew until no step could
reduce the cost (a wrong-sign Jacobian; a NaN wall the minimizer lies behind)
stops there, at a non-stationary point. So for those stops — and only those,
the extra Jacobian evaluation is paid by them alone — stationarity is TESTED
at the returned `x` (`_lm_stationary`, with `ftol` as the exact-fit test),
and `converged` is `_LM_STALLED` when it fails. A cost-decrease stop (`1`) is
converged, as FastLevenbergMarquardt's ftol stop was.
"""
function lmsolve_chart!(
        ews::ManifoldLmWorkspace{N}, lm; maxit::Integer = 1000, ftol::Real = eps(Float64), xtol::Real = 1.0e-10,
        stationarity_tol::Real, cache::Bool = true
    ) where {N}
    run = _chart_run(ews, lm)
    run.state.lm = lm
    problem = run.problem
    problem.variables[1] = zero(SVector{N, Float64})
    options = _nlls_options(_DTD, maxit, ftol, xtol)
    c = run.cache
    result = if cache && c !== nothing
        NLLSsolver.optimize!(problem, options, 1, NLLSsolver.nullcallback, c)
    else
        NLLSsolver.optimize!(problem, options, 1)
    end
    x = problem.variables[1]
    F = 2 * result.bestcost
    converged = _lm_code(result.termination)
    if converged == 2 || converged == 4
        _lm_stationary(ews, lm, x, stationarity_tol, ftol) || (converged = _LM_STALLED)
    end
    return x, F, converged, result.niterations
end

"""
    _lm_stationary(ews::ManifoldLmWorkspace, lm, x, tol, ftol) -> Bool

Whether `x` is a stationary point of `F = ‖r(x)‖²`. Two ways it is:

1. `F < ftol` — the cost is ZERO by the cost tolerance's own scale. A zero-
   residual fit ends with `r` at its rounding floor, which points in no
   particular direction, so the cosine below says nothing there: measured
   2026-10-01, exact fits (robot-world, lens, relative-pose, essential) ended
   at `F ≤ 4e-23` with cosines up to 0.62.
2. Every column `Jⱼ` of the Jacobian is (numerically) orthogonal to the
   residual `r`:

       max_j |Jⱼᵀ r| / (‖Jⱼ‖ ‖r‖) ≤ tol,

   a cosine — invariant to the scale of the residuals and of each unknown, so
   one `tol` serves every problem. An all-zero column (an unknown no row
   depends on) is skipped; a NaN anywhere fails.

It evaluates `r` and `J` at `x` once, into the workspace's buffers, and the
cosine on the workspace's scratch: allocation-free.
"""
function _lm_stationary(ews::ManifoldLmWorkspace, lm, x, tol::Real, ftol::Real)
    r = residual_rows!(ews.r, lm, x)
    sum(abs2, r) < ftol && return true
    J = residual_jacobian_rows!(ews.J, lm, x)
    return _residual_cosine!(ews.g, ews.w, J, r) <= tol
end

# max_j |Jⱼᵀ f| / (‖Jⱼ‖ ‖f‖), with `g` and `w` scratch of length size(J, 2)
# (0 when ‖f‖ = 0; an all-zero column is skipped; NaN propagates, and
# NaN ≤ tol is false).
function _residual_cosine!(g, w, J, f)
    fn = norm(f)
    iszero(fn) && return zero(fn)
    LinearAlgebra.mul!(g, J', f)
    sum!(abs2, w', J)
    c = zero(fn)
    @inbounds for j in eachindex(g, w)
        iszero(w[j]) && continue
        c = max(c, abs(g[j]) / (sqrt(w[j]) * fn))
    end
    return c
end

"""
    is_stationary(J::AbstractMatrix, f::AbstractVector; tol, ftol) -> Bool

The stall postcondition of [`lmsolve_chart!`](@ref) for a fit the chart LM
does not run (a block-structured NLLSsolver problem, [`nlls_retcode`](@ref)):
whether the point at which the residual `f` and its Jacobian `J` were
evaluated is a stationary point of `F = ‖f‖²` — `F < ftol` (an exact fit,
whose residual points in no particular direction), or every column `Jⱼ`
numerically orthogonal to `f`, `max_j |Jⱼᵀ f| / (‖Jⱼ‖ ‖f‖) ≤ tol`
(scale-invariant in the residuals and in each unknown; an all-zero column is
skipped, a NaN fails). A fit whose solver stopped without this holding stopped
where its damping grew, not at a minimizer: its retcode is `:stalled`.
"""
function is_stationary(J::AbstractMatrix, f::AbstractVector; tol::Real, ftol::Real)
    sum(abs2, f) < ftol && return true
    n = size(J, 2)
    return _residual_cosine!(similar(f, n), similar(f, n), J, f) <= tol
end

"""
    lm_retcode(converged::Int) -> Symbol

The retcode of a chart-LM refit from the code [`lmsolve_chart!`](@ref)
returns: `:max_iterations` when it stopped at `maxit` (`-1`); `:stalled` when
it stopped on its step size, or without a tolerance, at a point that is NOT
stationary (`_LM_STALLED`: the damping grew until no step reduced the cost —
the estimate is wherever the run stopped, not a minimizer); `:converged` for
the tolerance stops (`1` a cost decrease below ftol·F; `2` a step below xtol
and `4` a stop without a tolerance, once the stationarity postcondition
holds).
"""
lm_retcode(converged::Int) =
    converged == -1 ? :max_iterations : converged == _LM_STALLED ? :stalled : :converged

"""
    nlls_retcode(result, J::AbstractMatrix, r::AbstractVector; tol, ftol) -> Symbol

The retcode of a fit run by NLLSsolver, from its `NLLSsolver.NLLSResult` and the
Jacobian `J` and residual `r` at the returned estimate. NLLSsolver's
termination bits (`result.termination`; NLLSsolver 5.x, unchanged since 0.1,
`optimizeloop!` in src/optimize.jl): 0 the cost is infinite, 1 the cost is
NaN, 2 the relative decrease in cost is too small, 3 the absolute decrease in
cost is too small, 4 an infinity in the step, 5 a NaN in the step, 6 the max
of the step size is too small, 7 the maximum number of consecutive failed
iterations (iterations whose cost is higher than the current best,
`maxfails`) is reached, 8 the maximum number of iterations is reached, 9 the
maximum amount of time is exceeded, 10 the cost computations do not agree, 16
terminated by the user callback. Checked in this order: any of bits 0, 1, 4,
5, 10 is `:solve_failed`; bit 9 is `:max_time`; bit 8 is `:max_iterations`.
Every other stop — a small cost decrease (2, 3) or step (6), the
consecutive-failure cap (7), the callback (16) — does not prove convergence:
the stall postcondition does, so it is `:converged` when
[`is_stationary`](@ref)`(J, r; tol, ftol)` holds and `:stalled` otherwise — the
same postcondition the chart LM applies to its step-size stops
([`lm_retcode`](@ref)).
"""
function nlls_retcode(result::NLLSsolver.NLLSResult, J::AbstractMatrix, r::AbstractVector; tol::Real, ftol::Real)
    bits = result.termination
    for bit in (0, 1, 4, 5, 10)
        bits & (1 << bit) != 0 && return :solve_failed
    end
    bits & (1 << 9) != 0 && return :max_time
    bits & (1 << 8) != 0 && return :max_iterations
    return is_stationary(J, r; tol, ftol) ? :converged : :stalled
end

"""
    loss_robustifier(shape::AbstractLossShape, loss_width::Real)

NLLSsolver's robustifier (an `NLLSsolver.AbstractRobustifier`) for the loss
`shape` at `loss_width`. NLLSsolver hands a robustifier the squared residual
norm `cost` and reads `robustify(k, cost) = ρ` and
`robustifydcost(k, cost) = (ρ, ρ′, ρ″)`; here ρ is Core's [`ρ`](@ref)`(shape,
cost, loss_width²)` in the stack's convention `ρ′(0) = w_max` (no factor ½), ρ′
its [`weight`](@ref) and ρ″ the derivative of that weight (Core has no closed ρ″
per shape; ForwardDiff differentiates the weight).
"""
loss_robustifier(shape::AbstractLossShape, loss_width::Real) = LossRobustifier(shape, Float64(loss_width)^2)

struct LossRobustifier{S <: AbstractLossShape, T <: Real} <: NLLSsolver.AbstractRobustifier
    shape::S
    sq_loss_width::T
end
NLLSsolver.robustify(k::LossRobustifier, cost) = ρ(k.shape, cost, k.sq_loss_width)
function NLLSsolver.robustifydcost(k::LossRobustifier, cost)
    ρ′ = weight(k.shape, cost, k.sq_loss_width)
    ρ″ = ForwardDiff.derivative(s -> weight(k.shape, s, k.sq_loss_width), cost)
    return ρ(k.shape, cost, k.sq_loss_width), ρ′, ρ″
end

# =============================================================================
# Manifolds as NLLSsolver variables
# =============================================================================
#
# NLLSsolver's variable protocol on Core's `Manifold` (base/manifold.jl): a
# variable's size is the manifold's `tangent_size` (a constant of its type, so
# static) and an update re-anchors the manifold at the stepped point
# (`recenter`), reading the variable's `tangent_part` of the solver's
# increment. NLLSsolver's automatic Jacobian differentiates `update`, so
# `recenter` accepts a Dual increment and returns a Dual-valued manifold; a
# residual on manifold variables then needs only `computeresidual`, written
# generic in the variables' element type. A residual may still supply
# `computeresjac` when a hand-written Jacobian is faster.
NLLSsolver.nvars(m::Manifold) = NLLSsolver.static(tangent_size(m))
NLLSsolver.update(m::Manifold, δ, start = 1) = recenter(m, tangent_part(m, δ, start - 1))

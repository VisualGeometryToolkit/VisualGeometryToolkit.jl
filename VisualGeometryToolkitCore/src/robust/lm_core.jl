# =============================================================================
# lm_core.jl — the stack's Levenberg–Marquardt: FastLevenbergMarquardt's
# in-place `lmsolve!` on the chart-LM row protocol.
#
# Every nonlinear least-squares refit of the stack runs FastLevenbergMarquardt
# (FLM) `lmsolve!` on the √w-weighted rows of one linearization state —
# `residual_rows!` / `residual_jacobian_rows!` below — through ONE call site,
# `lmsolve_rows!`. The chart LM `lmsolve_chart!` is its static-N form: the
# iterate is the N-dimensional chart update x, started at zero (a chart is
# local, centred at the base model) and returned as an `SVector{N}`; the chart
# is fixed for the run and the caller re-centres it once, at the solution.
# VisualGeometryToolkitEstimation's LO refits (H: N = 8, F: 7, E: 5, PnP and hand-eye: 6,
# the radial homography: 9, the radial camera: 5), its `fit_lens` and its
# `refine_robot_world_hand_eye` (N = 6(M + 1)) run the chart LM; its relative-pose
# bundle runs `lmsolve_rows!` with its own Schur-complement solver.
# A step-size stop is checked for stationarity before it is called converged
# (`_lm_stationary`; retcode `:stalled` otherwise).
#
# The buffers — FLM's `LMWorkspace` around an `MVector{N}` iterate, and its
# Cholesky solver — live in a `ManifoldLmWorkspace` allocated once per (task,
# size), so a refit allocates nothing (a fresh residual vector + n×N Jacobian
# per LM iteration was ~1.1 MB of GC pressure per `locally_optimize` call).
# =============================================================================

# The chart LM's normal-equation solver at N unknowns. Up to N = 12 FLM solves
# with its unrolled static Cholesky on an `MMatrix` (its `CholeskySolver`
# limit); above, an `MMatrix` would take StaticArrays' generic factorization —
# compiled per N, allocating past N = 24 — so the solver is a `Matrix` and FLM
# factors it with LAPACK. Measured 2026-10-01 on `refine_robot_world_hand_eye`'s LM
# (median per solve, MVector iterate; MMatrix vs Matrix solver):
#   N = 12: 1.80 vs 1.72 ms, both 0 allocations
#   N = 18: 3.70 ms + 105 allocations vs 3.81 ms, 0
#   N = 24: 5.88 ms + 111 allocations vs 5.69 ms, 0   (first call 2.6 vs 0.3 s)
#   N = 30: 8.83 ms + 252 allocations (431 KB) vs 8.67 ms, 0   (first call 5.5 vs 0.3 s)
# and at the LO sizes the MMatrix is the faster (PnP N = 6, n = 500: 159 vs
# 175 µs; E N = 5, n = 1000: 855 vs 939 µs).
const _LM_STATIC_SOLVE_MAX = 12
_lm_solver_type(::Val{N}) where {N} =
    N <= _LM_STATIC_SOLVE_MAX ? FLM.CholeskySolver{MMatrix{N, N, Float64, N * N}} : FLM.CholeskySolver{Matrix{Float64}}
_lm_solver(::Val{N}) where {N} =
    N <= _LM_STATIC_SOLVE_MAX ? FLM.CholeskySolver(zeros(MMatrix{N, N, Float64}), zeros(MMatrix{N, N, Float64})) :
    FLM.CholeskySolver(N)

# FLM's workspace around an `MVector{N}` iterate (its p, g, x̃ and D are
# `similar` to it) with heap residual and Jacobian buffers.
const _LmBuffers{N} = FLM.LMWorkspace{
    MVector{N, Float64}, Vector{Float64}, Matrix{Float64},
    MVector{N, Float64}, MVector{N, Float64}, MVector{N, Float64}, MVector{N, Float64},
}

"""
    ManifoldLmWorkspace{N, S}
    ManifoldLmWorkspace(n, N)

The buffers of the chart LM ([`lmsolve_chart!`](@ref)) at `n` residual rows
and an `N`-dimensional chart: the per-row √weights the caller's refit fills,
FastLevenbergMarquardt's `LMWorkspace` (an `MVector{N}` iterate, the residual
and candidate-residual vectors, the `n × N` Jacobian) and its Cholesky solver
`S` (static up to N = 12, LAPACK above) — allocated once per problem instead of
per LM iteration. `N` is the model manifold's tangent size (essential 5,
fundamental 7, homography 8 in VisualGeometryToolkitEstimation). The residual SEMANTICS
come entirely from the caller's linearization state — Sampson distance for
E/F, whitened transfer error for H, calibrated reprojection for P3P — which is
why the type is named for the manifold, not the distance.
"""
struct ManifoldLmWorkspace{N, S <: FLM.CholeskySolver}
    sqrtw::Vector{Float64}   # per-row √weights — filled by VisualGeometryToolkitEstimation's LO refits, not read here
    buffers::_LmBuffers{N}   # FLM's iterate, residual, candidate and Jacobian buffers
    solver::S                # FLM's normal-equation solver (`_lm_solver`)
end
"The per-row √weight buffer of a [`ManifoldLmWorkspace`](@ref), filled by the caller's refit."
sqrt_weights(ws::ManifoldLmWorkspace) = ws.sqrtw

# The concrete workspace type at N unknowns (a caller that type-asserts a
# memoized workspace names this, not the partial `ManifoldLmWorkspace{N}`).
lm_workspace_type(v::Val{N}) where {N} = ManifoldLmWorkspace{N, _lm_solver_type(v)}

ManifoldLmWorkspace(n::Integer, N::Integer) = ManifoldLmWorkspace(Int(n), Val(Int(N)))
function ManifoldLmWorkspace(n::Int, v::Val{N}) where {N}
    buffers = FLM.LMWorkspace(zeros(MVector{N, Float64}), zeros(n), zeros(n, N))
    return lm_workspace_type(v)(zeros(n), buffers, _lm_solver(v))
end

# =============================================================================
# The chart-LM protocol — the rows a caller's linearization state fills
# =============================================================================

"""
    residual_rows!(r, lm, θ) -> r
    residual_jacobian_rows!(J, lm, θ) -> J

THE CHART-LM PROTOCOL — one pair for every model: the residual and Jacobian
rows FastLevenbergMarquardt minimizes in [`lmsolve_rows!`](@ref).

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
# The LM run — FastLevenbergMarquardt's `lmsolve!` on a linearization state
# =============================================================================

# FLM's callbacks are `fun!(f, x, data)` / `jac!(J, x, data)`; the protocol's
# are `residual_rows!(r, lm, θ)` / `residual_jacobian_rows!(J, lm, θ)`. A static
# iterate reaches the rows as an `SVector` (FLM's is an `MVector` it mutates).
_flm_rows!(f, x, lm) = residual_rows!(f, lm, _flm_point(x))
_flm_jacobian!(J, x, lm) = residual_jacobian_rows!(J, lm, _flm_point(x))
@inline _flm_point(x::StaticVector) = SVector(x)
@inline _flm_point(x) = x

"""
    lmsolve_rows!(buffers::FastLevenbergMarquardt.LMWorkspace, solver, lm; stationarity_tol, kw...)
        -> (x, F, converged, iterations)

THE stack's LM run: FastLevenbergMarquardt's in-place `lmsolve!` minimizing
`‖r(x)‖²` with `r` the [`residual_rows!`](@ref) of the linearization state `lm`
and its Jacobian [`residual_jacobian_rows!`](@ref), from `buffers.x`, with
FLM's normal-equation `solver` (a `CholeskySolver`, or a caller's
`FastLevenbergMarquardt.AbstractSolver`). `stationarity_tol` is the stall
postcondition's tolerance (`_lm_stationary`) — required: the caller's
configuration owns it. The other keywords are FLM's (`maxit`, `ftol`, `xtol`,
`gtol`, …; FLM's defaults). `x` IS `buffers.x`, overwritten by the next run on
these buffers; `converged` is FLM's code, or `_LM_STALLED` when the
postcondition fails ([`lm_retcode`](@ref) reads either), and `iterations` its
LM iteration count (accepted and rejected steps alike; 0 when the start
already meets a tolerance).

THE STALL POSTCONDITION. FLM's step-size stop does not prove convergence: in
FLM 0.1.0 its test runs on REJECTED steps too, so a run whose damping grew
until no step could reduce the cost (a wrong-sign Jacobian; a NaN wall the
minimizer lies behind) stops with code `2` at a non-stationary point, and the
upstream fix (FastLevenbergMarquardt.jl issue #4 / PR #5) reports that case as
code `4`. So when the code is `2` or `4` — and only then, the one extra
Jacobian evaluation is paid by those runs alone — stationarity is TESTED at
the returned `x` (`_lm_stationary`, with FLM's `ftol`).
"""
function lmsolve_rows!(buffers::FLM.LMWorkspace, solver, lm; stationarity_tol::Real, kw...)
    x, F, converged, iterations, _, _ = FLM.lmsolve!(_flm_rows!, _flm_jacobian!, buffers, lm; solver, kw...)
    if converged == 2 || converged == 4
        ftol = get(kw, :ftol, eps(eltype(buffers.x)))     # FLM's own default
        _lm_stationary(buffers, lm, stationarity_tol, ftol) || (converged = _LM_STALLED)
    end
    return x, F, converged, iterations
end

"""
    _LM_STALLED

The code [`lmsolve_rows!`](@ref) returns in place of FastLevenbergMarquardt's
step-size stop (`2`, or upstream's `4`) when the returned point fails the
stationarity test: `-2`, outside FLM's codes (`1`–`4`, `-1`).
[`lm_retcode`](@ref) maps it to `:stalled`.
"""
const _LM_STALLED = -2

"""
    _lm_stationary(buffers::FastLevenbergMarquardt.LMWorkspace, lm, tol, ftol) -> Bool

Whether `buffers.x` is a stationary point of `F = ‖r(x)‖²`. Two ways it is:

1. `F < ftol` — the cost is ZERO by FastLevenbergMarquardt's own test (its
   early exit: a run started there stops at once with code `1`). A zero-
   residual fit ends with `f` at its rounding floor, which points in no
   particular direction, so the cosine below says nothing there: measured
   2026-10-01, exact fits (robot-world, lens, relative-pose, essential) ended
   at `F ≤ 4e-23` with cosines up to 0.62.
2. Every column `Jⱼ` of the Jacobian is (numerically) orthogonal to the
   residual `f`:

       max_j |Jⱼᵀ f| / (‖Jⱼ‖ ‖f‖) ≤ tol,

   a cosine — invariant to the scale of the residuals and of each unknown, so
   one `tol` serves every problem. An all-zero column (an unknown no row
   depends on) is skipped; a NaN anywhere fails.

It evaluates `J` at `x` once, into `buffers.J`, and uses FLM's own scratch
(`buffers.g` for `Jᵀf`, `buffers.w` for the column norms) with the two
operations FLM itself applies to `J` (`mul!(g, J', f)`, `sum!(abs2, w', J)`),
so it is allocation-free for every Jacobian representation FLM accepts
(dense, sparse, static). `buffers.f` is FLM's residual at `x` (its documented
guarantee).
"""
_lm_stationary(buffers::FLM.LMWorkspace, lm, tol::Real, ftol::Real) =
    sum(abs2, buffers.f) < ftol || _lm_residual_cosine(buffers, lm) <= tol

# The left side of test 2: max_j |Jⱼᵀ f| / (‖Jⱼ‖ ‖f‖) at `buffers.x` (0 when
# ‖f‖ = 0; NaN propagates, and NaN ≤ tol is false).
function _lm_residual_cosine(buffers::FLM.LMWorkspace, lm)
    x, f, g, w = buffers.x, buffers.f, buffers.g, buffers.w
    fn = norm(f)
    iszero(fn) && return zero(fn)
    J = _flm_jacobian!(buffers.J, x, lm)
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
    lmsolve_chart!(ews::ManifoldLmWorkspace{N}, lm; kw...) -> (x::SVector{N}, F, converged, iterations)

The chart LM: [`lmsolve_rows!`](@ref) on the linearization state `lm` in the
workspace's buffers, from the chart origin `x = 0`. `x` is the solution chart
vector (a copy — the buffers are reused), `F` the final cost `‖r(x)‖²`,
[`lm_retcode`](@ref)`(converged)` the estimator's retcode and `iterations` the
LM iteration count. Keywords are `lmsolve_rows!`'s: the required
`stationarity_tol` and FastLevenbergMarquardt's (`maxit`, …).
"""
function lmsolve_chart!(ews::ManifoldLmWorkspace{N}, lm; kw...) where {N}
    fill!(ews.buffers.x, 0.0)
    x, F, converged, iterations = lmsolve_rows!(ews.buffers, ews.solver, lm; kw...)
    return SVector{N, Float64}(x), F, converged, iterations
end

"""
    lm_retcode(converged::Int) -> Symbol

The retcode of an LM refit from the code [`lmsolve_rows!`](@ref) returns:
`:max_iterations` when it stopped at `maxit` (FastLevenbergMarquardt's `-1`);
`:stalled` when FLM stopped on its step size at a point that is NOT stationary
(`_LM_STALLED`: the damping grew until no step reduced the cost — the
estimate is wherever the run stopped, not a minimizer); `:converged` for FLM's
tolerance stops (`1` ftol: actual and predicted reductions below it; `3` gtol:
the gradient below it; `2` xtol and upstream's `4`, a step below it, once the
stationarity postcondition holds).
"""
lm_retcode(converged::Int) =
    converged == -1 ? :max_iterations : converged == _LM_STALLED ? :stalled : :converged

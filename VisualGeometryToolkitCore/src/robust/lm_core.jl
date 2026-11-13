# =============================================================================
# lm_core.jl — the stack's Levenberg–Marquardt: NLLSsolver on the chart-LM row
# protocol.
#
# Every nonlinear least-squares fit of the stack runs through this file: ONE host
# adapter over NLLSsolver, `lmsolve_chart!`, under ONE owner of the stop settings,
# `LmStops` (configs/default.toml [lm]), with ONE success rule (the stationarity
# postcondition at the returned point, at every stop that is not a failure or a
# cap); and ONE device LM, `static_lm` (a KernelAbstractions kernel's fixed-N solve,
# which NLLSsolver's heap problems cannot be), under `StaticLmStops`, owned with the
# host's and judged by the same rule. No package calls NLLSsolver's solver API or the
# adapter's internals directly: `check_duplication`'s `:lm_primitive` pattern fails
# any such call outside this file (and its own tests).
#
# The host adapter has three forms on one engine (`_lm_run!`): a refit on a chart —
# the √w-weighted rows of one linearization state, `residual_rows!` /
# `residual_jacobian_rows!` below — whose iterate is the N-dimensional chart update x,
# started at zero (a chart is local, centred at the base model) and returned as an
# `SVector{N}`, the caller re-centring the chart once, at the solution; the same
# robustified per datum; and a multi-block problem (`LmProblem`: heterogeneous
# variables and residual blocks, each with its own robustifier — the bundles).
# VisualGeometryToolkitEstimation's LO refits (H: N = 8, F: 7, E: 5, PnP and hand-eye: 6,
# the radial homography: 9, the radial camera: 5), its `fit_camera` and its
# `refine_ellipse` run the chart LM; its block-structured fits (the relative-pose
# bundle, the camera-and-poses refines, `refine_robot_world_hand_eye`) the
# multi-block one.
#
# The adapter presents a linearization state as an NLLSsolver problem: one
# variable, the chart vector (an `SVector{N}`, NLLSsolver's `EuclideanVector`),
# and one residual block of all the rows, whose cost, gradient and Gauss–Newton
# Hessian it supplies through NLLSsolver's `computecostgradhess` hook as static
# arrays (Jᵀr and JᵀJ accumulated in one sweep over the rows: 13% faster to
# convergence than NLLSsolver forming JᵀJ itself, measured 2026-10-27 on 317
# captured E refits; CHANGELOG.md, the LM driver). The problem
# is built once per (workspace, linearization-state type) and kept in the
# workspace with NLLSsolver's `OptimizeCache`, so a refit reuses the problem
# and the solver's internal data.
#
# NLLSsolver ≥ 5.3.3 has every upstream pull request the stack relies on: the
# typed `OptimizeCache` (#62), the linear system reused at an accepted trial
# (#63), a NaN trial rejected (#64), a non-finite start point stopped (#65) and
# a non-finite step or gradient at an accepted point stopped, the damping loop
# bounded (#68: termination bit 12).
# One addition is still open upstream and DETECTED, never assumed: the
# Schur-complement linear system VisualGeometryToolkitEstimation's bundle uses
# (`nlls_features`).
# =============================================================================

"""
    nlls_features() -> (; schur::Bool)

Whether the loaded NLLSsolver has the Schur-complement linear system for
eliminable variables (`NLLSsolver.MultiVariateLSschur`,
`NLLSsolver.iseliminable`; upstream pull request #60, open). Until #60 merges a
registered NLLSsolver lacks it and the bundle takes NLLSsolver's sparse LDL
over every unknown — the same optimum, slower; the stack's fork has it.
"""
nlls_features() = (; schur = _NLLS_HAS_SCHUR)
const _NLLS_HAS_SCHUR = isdefined(NLLSsolver, :MultiVariateLSschur) && isdefined(NLLSsolver, :iseliminable)

# NLLSsolver's consecutive-failure cap (its default, `maxfails = 3`) and no time
# cap (NLLSsolver's default is 30 s; 1e6 s never fires, and a fit that hits it
# would read as an iteration cap).
const _NLLS_MAXFAILS = 3
const _NLLS_NO_TIME_CAP = 1.0e6

# The stop settings of record (configs/default.toml [lm]): the one place their values live.
const _LM_MAX_ITER = toml_default(_CORE_DEFAULTS, "lm", "max_iter")::Int
const _LM_FTOL = toml_default(_CORE_DEFAULTS, "lm", "ftol")::Float64
const _LM_XTOL = toml_default(_CORE_DEFAULTS, "lm", "xtol")::Float64
const _LM_STATIONARITY_TOL = toml_default(_CORE_DEFAULTS, "lm", "stationarity_tol")::Float64
const _STATIC_LM_MAX_ITER = toml_default(_CORE_DEFAULTS, "lm", "device_max_iter")::Int

"""
    LmStops(; max_iter, ftol, xtol, stationarity_tol, max_time = nothing)

THE stop settings of every host Levenberg–Marquardt fit of the stack — the one owner of its knobs, which
[`lmsolve_chart!`](@ref) maps onto NLLSsolver's options in one place. Every keyword defaults to Core's
configuration of record (configs/default.toml `[lm]`); a fit overrides one only from its own configuration table.

| knob | meaning | NLLSsolver |
|---|---|---|
| `max_iter` | the iteration cap (`[lm]` 1000) | `maxiters`: outer iterations, one ACCEPTED step each; an iteration's damping loop may reject ~8–10 candidate steps first, so the cap bounds accepted steps, not work |
| `ftol` | relative cost tolerance (`[lm]` 1e-10): stop once a step lowers the cost by less than `ftol`·F; also the stationarity postcondition's exact-fit test, F < `ftol` | `reldcost` |
| `xtol` | step tolerance (`[lm]` 1e-10) | `dstep`: ABSOLUTE, the step's max-norm in chart units (max\\|pᵢ\\| < xtol) |
| `stationarity_tol` | the stationarity postcondition's column cosine (`[lm]` 0.05) | — (the stack's own test, [`is_stationary`](@ref)) |
| `max_time` | wall-clock cap in seconds; `nothing` (the default) or `Inf`: none | `maxtime` |
| — | absolute cost decrease | `absdcost = 0`: the stack has no such knob |
| — | consecutive failed iterations | `maxfails = 3`, NLLSsolver's default |

The damping is Marquardt's, λ·DᵀD with DᵀD the running maximum of diag(JᵀJ), its initial λ NLLSsolver's `1e-6`;
neither is a knob. Single-threaded (`numthreads = 1`): a fit runs inside the caller's threads. There is no gradient
tolerance: success is the stationarity postcondition's ([`lmsolve_chart!`](@ref)).
"""
struct LmStops
    max_iter::Int
    ftol::Float64
    xtol::Float64
    stationarity_tol::Float64
    max_time::Float64
end
function LmStops(; max_iter::Integer = _LM_MAX_ITER, ftol::Real = _LM_FTOL, xtol::Real = _LM_XTOL,
        stationarity_tol::Real = _LM_STATIONARITY_TOL, max_time::Union{Nothing, Real} = nothing)
    max_iter >= 0 || throw(ArgumentError("LmStops: max_iter is a count of accepted iterations, ≥ 0; got $max_iter"))
    (ftol >= 0 && xtol >= 0) || throw(ArgumentError("LmStops: ftol and xtol are tolerances, ≥ 0; got ftol = $ftol, xtol = $xtol"))
    0 < stationarity_tol <= 1 || throw(ArgumentError("LmStops: stationarity_tol is a cosine in (0, 1]; got $stationarity_tol"))
    return LmStops(Int(max_iter), Float64(ftol), Float64(xtol), Float64(stationarity_tol), _time_cap(max_time))
end
# `max_time` in seconds: `nothing` or `Inf` (or anything past the no-cap value) is
# no cap; a negative or NaN cap is no cap anyone meant.
_time_cap(::Nothing) = _NLLS_NO_TIME_CAP
function _time_cap(t::Real)
    (isnan(t) || t < 0) && throw(ArgumentError("LmStops: max_time is a time in seconds, ≥ 0 (Inf or nothing: no cap); got $t"))
    return min(Float64(t), _NLLS_NO_TIME_CAP)
end
# The map onto NLLSsolver's options, on the static iterator `_DTD` (Marquardt's damping), so the options are inferred
# and allocate nothing (a runtime Symbol made `static(iterator)` a dynamic call, three allocations per refit).
@inline _nlls_options(s::LmStops) =
    NLLSsolver.NLLSOptions(
        s.ftol, 0.0, s.xtol, _NLLS_MAXFAILS, s.max_iter, UInt64(round(s.max_time * 1.0e9)),
        _DTD, NLLSsolver.static(1), NLLSsolver.static(false)   # NLLSsolver ≥ 5.3: no callback of ours changes the problem's state
    )
const _DTD = NLLSsolver.static(:levenbergmarquardt_dtd)
const _NLLS_OPTIONS_TYPE = typeof(_nlls_options(LmStops(1, 1.0, 1.0, 1.0, _NLLS_NO_TIME_CAP)))

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
    run::Any                       # the chart LM's kept `LmProblem` (plain or robust) of the last state type, or `nothing`
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
UNWEIGHTED `whitened_residuals!(r, problem, model)` in data units. Two names, because
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

"""
    residual_cost!(r, lm, θ) -> ½‖r(θ)‖²
    residual_normal_equations!(r, J, lm, θ::SVector{N}) -> (½‖r‖², Jᵀr::SVector{N}, JᵀJ::SMatrix{N, N})

THE CHART LM'S TWO REDUCTIONS of the rows: what [`lmsolve_chart!`](@ref) reads.
- `residual_cost!` gives the cost at a trial point.
- `residual_normal_equations!` gives the cost, the gradient and the Gauss–Newton Hessian at a linearization. It also
  feeds the stall postcondition, whose cosine `|Jⱼᵀr| / (‖Jⱼ‖‖r‖)` reads `g` and `diag(H)`.

The defaults evaluate [`residual_rows!`](@ref) and [`residual_jacobian_rows!`](@ref) into the caller's buffers `r`
and `J`, and reduce on the host.

A state whose rows live on a DEVICE overrides both with one kernel that evaluates every row and its Jacobian and
reduces them where they are. The rows never reach the host; `r` and `J` go unused. Only the `N × N` step is the
host's.
"""
function residual_cost! end

@doc (@doc residual_cost!)
function residual_normal_equations! end

function residual_cost!(r, lm, θ)
    r = residual_rows!(r, lm, θ)
    s = 0.0
    @inbounds for i in eachindex(r)
        s += r[i] * r[i]
    end
    return 0.5 * s
end

# The cost, the gradient Jᵀr and the Gauss–Newton Hessian JᵀJ in one sweep over
# the rows, as static arrays: NLLSsolver's own `jac' * jac` on a dynamic n × N
# Jacobian was the one NLLSsolver-specific cost of a refit (10% of its
# samples), and this layout is the measured 0.84× (lm_core.jl header).
function residual_normal_equations!(r, J, lm, x::SVector{N}) where {N}
    r = residual_rows!(r, lm, x)
    J = residual_jacobian_rows!(J, lm, x)
    # Immutable accumulators: an `MMatrix` here escaped to the heap under
    # `--check-bounds=yes` on Julia 1.12 (22 allocations per refit), the
    # static values stay on the stack in every mode.
    g = zero(SVector{N, Float64})
    H = zero(SMatrix{N, N, Float64})
    s = 0.0
    @inbounds for i in eachindex(r)
        ri = r[i]
        s += ri * ri
        Ji = SVector{N, Float64}(ntuple(a -> J[i, a], Val(N)))
        g += Ji * ri
        H += Ji * Ji'
    end
    return 0.5 * s, g, H
end

"""
    reduce_normal_equations(partials::AbstractMatrix, ::Val{N}) -> (½‖r‖², Jᵀr::SVector{N}, JᵀJ::SMatrix{N, N})

[`residual_normal_equations!`](@ref)'s value for a device state, from its per-work-item partial sums. Column `w` of
`partials` is work item `w`'s share over its rows, `1 + N + N²` entries: `Σ ‖rᵢ‖²`, then the `N` of `Σ Jᵢᵀrᵢ`, then the
`N × N` of `Σ JᵢᵀJᵢ`, column-major. The columns are summed where they are, on the array's backend
(`sum(partials; dims = 2)`), and only the `1 + N + N²` totals come to the host. The result is isbits. The cost alone is
[`reduce_cost`](@ref).
"""
function reduce_normal_equations(partials::AbstractMatrix, ::Val{N}) where {N}
    size(partials, 1) == 1 + N + N * N ||
        throw(DimensionMismatch("reduce_normal_equations: $(size(partials, 1)) rows of partial sums for N = $N (1 + N + N² = $(1 + N + N * N))"))
    t = Array(sum(partials; dims = 2))
    return 0.5 * t[1], SVector{N, Float64}(ntuple(a -> t[1 + a], Val(N))), SMatrix{N, N, Float64}(ntuple(k -> t[1 + N + k], Val(N * N)))
end

"""
    reduce_cost(partials::AbstractMatrix) -> ½‖r‖²

[`residual_cost!`](@ref)'s value for a device state: half the sum of row 1 of its per-work-item partial sums (the
layout of [`reduce_normal_equations`](@ref); only row 1 is read, so a cost kernel may write a `1 × W` matrix), summed
on the array's backend.
"""
reduce_cost(partials::AbstractMatrix) = 0.5 * sum(view(partials, 1, :))

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
# bit). A non-finite cost is +Inf. NLLSsolver (≥ 5.3.1) rejects a NaN trial
# (#64) and stops at a non-finite start (#65), but it still ACCEPTS a trial at
# a step below `xtol` whatever its cost, and its per-iteration check
# `cost == cost_` (optimize.jl) is false for NaN: a step whose rows are NaN (a
# step past a NaN or discontinuity wall — the radial-camera, radial-homography
# and lens rows have them) would throw an AssertionError there. +Inf compares
# equal, and is rejected like NaN everywhere else.
"""
    finite_cost(c::Float64) -> Float64

A cost as the stack's NLLSsolver fits read it: `c` when finite, `+Inf`
otherwise. NLLSsolver (≥ 5.3.1) accepts a trial at a step below `xtol` whatever
its cost and asserts `cost == cost_` per iteration, which NaN fails; +Inf
compares equal and is rejected like NaN everywhere else. Every NLLSsolver
residual block of the stack returns its cost through it.
"""
@inline finite_cost(c::Float64) = isfinite(c) ? c : Inf

# The cost and the normal equations NLLSsolver reads: the state's two reductions (`residual_cost!`,
# `residual_normal_equations!`), so a device state's rows never reach the host.
NLLSsolver.computecost(c::_ChartRows, x) = finite_cost(residual_cost!(c.r, c.state.lm, x))
NLLSsolver.computecostgradhess(_varflags, c::_ChartRows{N}, x) where {N} = residual_normal_equations!(c.r, c.J, c.state.lm, x)

# =============================================================================
# THE engine — an NLLSsolver problem kept with its cache, solved under the stop settings
# =============================================================================

"""
    LmProblem(variables::AbstractVector, blocks::AbstractVector)

An NLLSsolver problem as the stack's one LM adapter ([`lmsolve_chart!`](@ref)) solves it: `variables` (any NLLSsolver
variables — Core [`Manifold`](@ref)s, `SVector`s, [`Eliminable`](@ref)s — of one type or several) and the residual
`blocks` (NLLSsolver `AbstractResidual`s, of one type or several, each with its own `varindices` and its
`robustkernel`: `NLLSsolver.NoRobust()`, or a frozen loss through [`loss_robustifier`](@ref)). The problem is built
once, with NLLSsolver's typed `OptimizeCache` (the linear system's layout — the Schur system where the variables
allow it), and kept: a later solve on the same structure (a RANSAC refit's bundle) rebuilds neither.

`p.variables` is the problem's variable vector: set the start in place before a solve, read the solution after it
(NLLSsolver swaps its buffers, so read it through `p`, never through a saved reference). `p.blocks` are the blocks;
the residual statistics of a solve are [`nlls_rows_and_jacobian`](@ref)`(p.blocks, p.variables)`.
"""
struct LmProblem{P <: NLLSsolver.NLLSProblem, C, B <: AbstractVector, U}
    problem::P
    cache::C
    blocks::B
    unfixed::U            # NLLSsolver's free variables: `1` (a chart: one variable), or `nothing` (every one)
end
function LmProblem(variables::AbstractVector, blocks::AbstractVector)
    isempty(variables) && throw(ArgumentError("LmProblem: no variables"))
    isempty(blocks) && throw(ArgumentError("LmProblem: no residual blocks"))
    VT = _element_union(variables)
    BT = _element_union(blocks)
    problem = NLLSsolver.NLLSProblem(VT, BT)
    for v in variables
        NLLSsolver.addvariable!(problem, v)
    end
    bl = Vector{BT}(blocks)
    for b in bl
        NLLSsolver.addcost!(problem, b)
    end
    return _lm_problem(problem, bl, nothing)
end
# The element type a vector's values need: its own when concrete, else the union of the values' types (a camera
# chart and pose charts; a pose and N points).
_element_union(xs::AbstractVector) = isconcretetype(eltype(xs)) ? eltype(xs) : Union{unique(typeof.(xs))...}
_lm_problem(problem, blocks, unfixed) =
    LmProblem(problem, NLLSsolver.OptimizeCache(problem, _nlls_options(LmStops()), unfixed), blocks, unfixed)
# A chart LM's run, of its concrete type `R` (the typed cache converted to the inferred cache type, as `_kept_run` asserts)
_chart_lm_problem(::Type{R}, problem, blocks) where {R} =
    R(problem, NLLSsolver.OptimizeCache(problem, _nlls_options(LmStops()), 1), blocks, 1)
Base.getproperty(p::LmProblem, s::Symbol) = s === :variables ? getfield(p, :problem).variables : getfield(p, s)
Base.propertynames(::LmProblem) = (:problem, :cache, :blocks, :unfixed, :variables)

# The concrete LmProblem of a problem type `P` with blocks `B` and one free variable — the chart LMs' runs, whose
# typed cache's type is the constructor's inferred return type (a constant of the types).
_chart_problem_type(::Type{P}, ::Type{B}) where {P, B} =
    LmProblem{P, Core.Compiler.return_type(NLLSsolver.OptimizeCache, Tuple{P, _NLLS_OPTIONS_TYPE, Int}), Vector{B}, Int}

# THE one caching path of the chart LMs, plain and robust: the workspace's kept run when it is of type `R`, else a new
# one (`build()`) kept in its place — a workspace serves one run type at a time (the LO's never changes type).
@inline function _kept_run(build::F, ews, ::Type{R}) where {F, R}
    run = ews.run
    run isa R && return run
    new = build()::R
    ews.run = new
    return new
end

# THE solve, every form's: NLLSsolver under the stop settings on the kept problem and cache, its stop code
# (`nlls_stop_code`) and termination record (`nlls_termination`) under THE success rule — the stationarity
# postcondition (`postcondition()`, at the returned point) decides every stop that is not a failure or a cap: the
# cost-decrease stop, the step-size stop and the rejection cap alike. A stop it fails is `_LM_STALLED`.
@inline function _lm_run!(p::LmProblem, stops::LmStops, callback::F, postcondition::G) where {F, G}
    result = NLLSsolver.optimize!(p.problem, _nlls_options(stops), p.unfixed, callback, p.cache)
    code = nlls_stop_code(result.termination)
    stationary = (code == 1 || code == 2 || code == 4) ? postcondition()::Bool : true
    stationary || (code = _LM_STALLED)
    return result, code, nlls_termination(result, stationary)
end

# The linear-algebra failures NLLSsolver's step solve may throw (an indefinite or singular normal system): a
# multi-block solve maps them to a failed solve, `(ReturnCode.Failure, NonFinite)`. Anything else is a bug and
# propagates.
const _LM_LINEAR_FAILURES = Union{
    LinearAlgebra.PosDefException, LinearAlgebra.SingularException, LinearAlgebra.ZeroPivotException,
    LinearAlgebra.LAPACKException,
}

# =============================================================================
# The chart LM — NLLSsolver on a linearization state, and its stop codes
# =============================================================================

"""
    nlls_stop_code(bits::Integer) -> Int

THE decoder of NLLSsolver's termination bits (`result.termination`, listed at
[`nlls_stop_reasons`](@ref)) into one stop code — the one both the chart LM's code
(`lmsolve_chart!`'s `converged`) and `nlls_termination` are views
of, and the one a caller running its own `NLLSsolver.optimize!` reads (with
[`nlls_stop_reasons`](@ref) for every bit that is set). One order, the first
that applies:

1. a failure — a non-finite cost (0, 1) or step (4, 5), a cost disagreement
   (10), a start point whose cost or gradient is not finite (11), a gradient
   that is not finite at an accepted point (12) — WITHOUT a
   step-size stop: `-3`, `:solve_failed`;
2. the step-size stop (6): `2`, the stall postcondition decides it (checked
   before the cost tolerance: a tiny last step at a wall also shows a tiny
   cost decrease);
3. a relative or absolute cost-decrease stop (2, 3): `1`, the postcondition decides it (a
   tolerance outranks a cap);
4. the consecutive-failure cap (7): `4`, the postcondition decides it;
5. the time cap (9): `-4`, `:max_time`; then the iteration cap (8): `-1`,
   `:max_iterations`;
6. anything else (16, a user callback): `4`.

A non-finite TRIAL at a step-size stop (bits 0 or 1 with 6: the chart LM's
NaN wall, its non-finite cost read as +Inf) is a step-size stop, not a failure.
"""
function nlls_stop_code(bits::Integer)
    bit(k) = bits & (1 << k) != 0
    any(bit, (0, 1, 4, 5, 10, 11, 12)) && !bit(6) && return _LM_FAILED
    bit(6) && return 2
    (bit(2) || bit(3)) && return 1
    bit(7) && return 4
    bit(9) && return _LM_MAX_TIME
    bit(8) && return -1
    return 4
end

# NLLSsolver's termination bits by name (NLLSsolver 5.x, `optimizeloop!`).
const _NLLS_STOP_BITS = (
    (0, :cost_infinite), (1, :cost_nan), (2, :relative_cost_decrease), (3, :absolute_cost_decrease),
    (4, :delta_infinite), (5, :delta_nan), (6, :delta_size), (7, :max_consecutive_rejections),
    (8, :max_iterations), (9, :max_time), (10, :cost_disagreement), (11, :nonfinite_start), (12, :nonfinite_gradient),
    (16, :callback),
)

"""
    nlls_stop_reasons(bits::Integer) -> Vector{Symbol}

Every reason NLLSsolver's termination bits (`result.termination`) state, in
bit order: `:cost_infinite` (0), `:cost_nan` (1), `:relative_cost_decrease`
(2, below `reldcost`), `:absolute_cost_decrease` (3, below `absdcost`),
`:delta_infinite` (4), `:delta_nan` (5), `:delta_size` (6, the LM update δ below `dstep`),
`:max_consecutive_rejections` (7, `maxfails` iterations whose cost exceeded the
best), `:max_iterations` (8), `:max_time` (9), `:cost_disagreement` (10, the
residual and the residual-Jacobian evaluations disagree), `:nonfinite_start`
(11, the start point's cost or gradient is not finite: no step is tried),
`:nonfinite_gradient` (12, the gradient at an accepted point is not finite: the
solve stops there, its best point kept), `:callback` (16).
A report of the stop; `nlls_stop_code` is the one decision read from
the same bits.
"""
nlls_stop_reasons(bits::Integer) = Symbol[name for (b, name) in _NLLS_STOP_BITS if (bits >> b) & 1 == 1]

"""
    nlls_termination(result::NLLSsolver.NLLSResult, postcondition) -> (ReturnCode.T, StopRule)
    nlls_termination(bits::Integer, postcondition) -> (ReturnCode.T, StopRule)

The termination record of an NLLSsolver solve — its [`ReturnCode`](@ref) and
[`StopRule`](@ref) — from its termination bits (`result.termination`, NLLSsolver
5.x `optimizeloop!`; the bits as [`nlls_stop_reasons`](@ref) names them),
decoded in `nlls_stop_code`'s one order:

| bits (the first that applies) | `StopRule` | `ReturnCode` |
|:--|:--|:--|
| 11, a non-finite start point (no step tried) | `NonFinite` | `InitialFailure` |
| 0, 1, 4, 5, 10, 12 without a step-size stop | `NonFinite` | `Failure` |
| 6, the step size | `IncrementSize` | `Success` if the postcondition holds, else `Stalled` |
| 2, 3, a cost decrease | `CostDecrease` | `Success` if the postcondition holds, else `Stalled` |
| 7, the consecutive-rejection cap | `RejectionCap` | `Success` if the postcondition holds, else `Stalled` |
| 16 and above only, a user callback | `Callback` | `Terminated` |
| 9, the time cap | `TimeCap` | `MaxTime` |
| 8, the iteration cap | `IterationCap` | `MaxIters` |

`postcondition` is the stall postcondition at the returned point — a `Bool`, or a
callable `() -> Bool` that runs only for the stops that need it (a cost-decrease,
step-size or rejection stop: [`is_stationary`](@ref), or the chart LM's own test) — THE success
rule: no stop is a success without it. A tolerance
met on the last iteration outranks the cap, so `MaxIters` means no tolerance
fired.
"""
nlls_termination(result::NLLSsolver.NLLSResult, postcondition) = nlls_termination(result.termination, postcondition)
function nlls_termination(bits::Integer, postcondition)
    bits & (1 << 11) != 0 && bits & (1 << 6) == 0 && return (ReturnCode.InitialFailure, NonFinite)
    code = nlls_stop_code(bits)
    code == _LM_FAILED && return (ReturnCode.Failure, NonFinite)
    code == _LM_MAX_TIME && return (ReturnCode.MaxTime, TimeCap)
    code == -1 && return (ReturnCode.MaxIters, IterationCap)
    # the stops whose verdict is the postcondition's (a cost decrease, a step-size stop, the rejection cap), and a callback's
    rule = code == 1 ? CostDecrease : code == 2 ? IncrementSize : bits & (1 << 7) != 0 ? RejectionCap : bits >> 16 != 0 ? Callback : NoRule
    rule == Callback && return (ReturnCode.Terminated, Callback)
    return (_holds(postcondition) ? ReturnCode.Success : ReturnCode.Stalled, rule)
end
_holds(p::Bool) = p
_holds(p) = p()::Bool

# The decoder's two codes beyond the stop codes 1, 2, 4 and −1: a failed solve
# and the time cap.
const _LM_FAILED = -3
const _LM_MAX_TIME = -4

"""
    _LM_STALLED

The code [`lmsolve_chart!`](@ref) returns in place of a cost-decrease, step-size
or tolerance-free stop (`1`, `2`, `4`) when the returned point fails the
stationarity test: `-2`; its record is `ReturnCode.Stalled`.
"""
const _LM_STALLED = -2

"""
    lmsolve_chart!(ews::ManifoldLmWorkspace{N}, lm, stops::LmStops = LmStops())
        -> (x::SVector{N}, F, converged, iterations, (code, rule))
    lmsolve_chart!(ews::ManifoldLmWorkspace{N}, lm, robustifier, ::Val{C}, stops::LmStops = LmStops())
        -> (x::SVector{N}, F, converged, iterations, (code, rule))
    lmsolve_chart!(p::LmProblem, stops::LmStops = LmStops(); callback = nothing)
        -> (; code::ReturnCode.T, rule::StopRule, iterations::Int, cost::Float64)

THE stack's host Levenberg–Marquardt: NLLSsolver (Marquardt's DᵀD damping) under the stop settings `stops`
([`LmStops`](@ref), the one owner of the knobs), on a kept problem and NLLSsolver cache, with ONE success rule.

1. THE CHART LM minimizes `‖r(x)‖²`, `r` the [`residual_rows!`](@ref) of the linearization state `lm` (its Jacobian
   [`residual_jacobian_rows!`](@ref), both read through the state's two reductions), from the chart origin `x = 0`.
2. THE ROBUSTIFIED CHART LM minimizes `Σᵢ ρ(‖rᵢ(x)‖²)` from the chart origin, `rᵢ` datum `i`'s `C` consecutive rows
   (`C` the codimension) and `ρ` the NLLSsolver `robustifier` — a stack loss through
   [`loss_robustifier`](@ref)`(shape, loss_width)`, applied per datum (NLLSsolver's robustified Hessian), so one solve
   is the robust M-estimate. `F` is `Σᵢ ρ(‖rᵢ‖²)` at `x`; on return the workspace's `r` holds the unweighted rows at `x`.
3. THE MULTI-BLOCK LM solves an [`LmProblem`](@ref) — heterogeneous variables and residual blocks, each block with its
   own robustifier — in place in `p.variables`; `cost` is `Σₖ ρ(‖rₖ‖²)` (‖r‖² without robustifiers), `callback` an
   NLLSsolver iteration callback (`nothing`: none). An indefinite or singular normal system in NLLSsolver's step solve
   (`PosDefException`, `SingularException`, `ZeroPivotException`, `LAPACKException`) is a failed solve,
   `(ReturnCode.Failure, NonFinite)`, the variables as the solver left them.

The chart forms keep the problem and NLLSsolver's `OptimizeCache` in the workspace (one caching path, plain and
robust alike), so a refit rebuilds neither. `x` is the solution chart vector, `F` the final cost, `converged` the stop
code (`nlls_stop_code`'s, or `-2` for a stop the postcondition fails), `iterations` NLLSsolver's count (one
accepted step each; a start at a stationary point takes one zero step, and only a start whose cost or gradient is not
finite stops at 0, `:solve_failed`), and `(code, rule)` the termination record, a [`ReturnCode`](@ref) and a
[`StopRule`](@ref) (`nlls_termination`'s decode).

THE SUCCESS RULE. A stop is a success only at a STATIONARY point: for the cost-decrease stop, the step-size stop and
the rejection cap alike, stationarity is TESTED at the returned point, and a stop that fails it is
`ReturnCode.Stalled` (`converged = -2`) with the rule that fired. A tiny cost decrease or a tiny step proves nothing: a
run whose damping grew until no step could lower the cost (a wrong-sign Jacobian; a NaN wall the minimizer lies
behind) stops there. Stationary means the cost is zero by the exact-fit test, `F < stops.ftol`, or every column of
the Jacobian is numerically orthogonal to the residual, `max_j |Jⱼᵀr| / (‖Jⱼ‖ ‖r‖) ≤ stops.stationarity_tol`
(√ρ′-weighted per datum or block when robust: [`is_stationary`](@ref)). The iteration and time caps are `MaxIters`
and `MaxTime`, a failed solve `Failure`.
"""
function lmsolve_chart!(ews::ManifoldLmWorkspace{N}, lm::L, stops::LmStops = LmStops()) where {N, L}
    R = _chart_problem_type(NLLSsolver.NLLSProblem{SVector{N, Float64}, _ChartRows{N, L}}, _ChartRows{N, L})
    run = _kept_run(ews, R) do
        problem = NLLSsolver.NLLSProblem(SVector{N, Float64}, _ChartRows{N, L})
        NLLSsolver.addvariable!(problem, zero(SVector{N, Float64}))
        blocks = [_ChartRows{N, L}(_ChartState{L}(lm), ews.r, ews.J)]
        NLLSsolver.addcost!(problem, blocks[1])
        _chart_lm_problem(R, problem, blocks)
    end
    run.blocks[1].state.lm = lm
    run.variables[1] = zero(SVector{N, Float64})
    result, converged, record = _lm_run!(run, stops, NLLSsolver.nullcallback,
        () -> _lm_stationary(ews, lm, run.variables[1], stops.stationarity_tol, stops.ftol))
    return run.variables[1], 2 * result.bestcost, converged, result.niterations, record
end

# =============================================================================
# The ROBUSTIFIED chart LM — the same rows, one NLLSsolver residual per datum
# =============================================================================
#
# The chart LM above minimizes ‖r‖² with the rows as ONE residual block. A
# robust loss acts per datum — on the squared norm of that datum's `C` rows —
# so the robustified LM hands NLLSsolver one residual per datum carrying the
# loss as its robustifier (`loss_robustifier`). NLLSsolver then forms each
# datum's cost ρ(‖rᵢ‖²), its gradient ρ′Jᵢᵀrᵢ and its robustified Hessian
# (`computerescostgradhess`, NLLSsolver's residual.jl): nothing of the robust
# Newton update is restated here.
#
# The rows are still evaluated by the state's ONE `residual_rows!` /
# `residual_jacobian_rows!` sweep: the first datum NLLSsolver asks for at a
# chart vector fills the rows (and, for a linearization, the Jacobian) of all
# of them into the workspace, and every other datum at the same vector reads
# its slice — NLLSsolver evaluates every residual at one vector before moving.

mutable struct _RobustChartState{N, L, K}
    lm::L
    kernel::K
    x::SVector{N, Float64}       # the chart vector `r` (and `J` when `jac`) hold the rows at
    rows::Bool
    jac::Bool
    const r::Vector{Float64}
    const J::Matrix{Float64}
end

@inline function _robust_rows!(s::_RobustChartState, x)
    (s.rows && x == s.x) && return s.r
    residual_rows!(s.r, s.lm, x)
    s.x = x; s.rows = true; s.jac = false
    return s.r
end
@inline function _robust_jacobian!(s::_RobustChartState, x)
    _robust_rows!(s, x)
    s.jac && return s.J
    residual_jacobian_rows!(s.J, s.lm, x)
    s.jac = true
    return s.J
end

# One datum: its `C` rows of the state, robustified by the state's kernel.
struct _DatumRows{N, C, S} <: NLLSsolver.AbstractResidual
    state::S
    i::Int
end
NLLSsolver.ndeps(::_DatumRows) = NLLSsolver.static(1)
NLLSsolver.nres(::_DatumRows{N, C}) where {N, C} = NLLSsolver.static(C)
NLLSsolver.varindices(::_DatumRows) = SVector(1)
NLLSsolver.getvars(::_DatumRows{N}, vars::Vector) where {N} = (vars[1]::SVector{N, Float64},)
Base.eltype(::_DatumRows) = Float64
NLLSsolver.robustkernel(d::_DatumRows) = d.state.kernel
@inline _datum_rows(r, ::Val{C}, i) where {C} = SVector{C, Float64}(ntuple(k -> @inbounds(r[C * (i - 1) + k]), Val(C)))
NLLSsolver.computeresidual(d::_DatumRows{N, C}, x) where {N, C} = _datum_rows(_robust_rows!(d.state, x), Val(C), d.i)
function NLLSsolver.computeresjac(_varflags, d::_DatumRows{N, C}, x) where {N, C}
    J = _robust_jacobian!(d.state, x)
    o = C * (d.i - 1)
    Ji = SMatrix{C, N, Float64}(ntuple(k -> @inbounds(J[o + (k - 1) % C + 1, (k - 1) ÷ C + 1]), Val(C * N)))
    return _datum_rows(d.state.r, Val(C), d.i), Ji
end

function lmsolve_chart!(
        ews::ManifoldLmWorkspace{N}, lm::L, robustifier::K, ::Val{C}, stops::LmStops = LmStops()
    ) where {N, L, K, C}
    rows = length(ews.r)
    rows % C == 0 || throw(ArgumentError("lmsolve_chart!: $rows rows are not a whole number of $C-row data"))
    S = _RobustChartState{N, L, K}
    B = _DatumRows{N, C, S}
    R = _chart_problem_type(NLLSsolver.NLLSProblem{SVector{N, Float64}, B}, B)
    run = _kept_run(ews, R) do
        state = S(lm, robustifier, zero(SVector{N, Float64}), false, false, ews.r, ews.J)
        problem = NLLSsolver.NLLSProblem(SVector{N, Float64}, B)
        NLLSsolver.addvariable!(problem, zero(SVector{N, Float64}))
        blocks = [B(state, i) for i in 1:(rows ÷ C)]
        foreach(b -> NLLSsolver.addcost!(problem, b), blocks)
        _chart_lm_problem(R, problem, blocks)
    end
    state = run.blocks[1].state
    state.lm = lm; state.kernel = robustifier; state.rows = false; state.jac = false
    run.variables[1] = zero(SVector{N, Float64})
    result, converged, record = _lm_run!(run, stops, NLLSsolver.nullcallback,
        () -> _blocks_stationary!(ews.g, ews.w, _CHART_OFFSETS, run.blocks, run.variables, _ONE_FREE, stops.stationarity_tol, stops.ftol))
    x = run.variables[1]
    _robust_rows!(state, x)            # the solver's last evaluation may have been a rejected trial
    return x, 2 * result.bestcost, converged, result.niterations, record
end
const _ONE_FREE = (true,)      # the chart LMs' one variable, free
const _CHART_OFFSETS = (0,)   # its columns start at 1

function lmsolve_chart!(p::LmProblem, stops::LmStops = LmStops(); callback = nothing)
    cb = callback === nothing ? NLLSsolver.nullcallback : callback
    out = try
        _lm_run!(p, stops, cb, () -> is_stationary(p.blocks, p.variables, trues(length(p.variables));
            tol = stops.stationarity_tol, ftol = stops.ftol))
    catch e
        e isa _LM_LINEAR_FAILURES || rethrow()
        nothing
    end
    out === nothing && return (; code = ReturnCode.Failure, rule = NonFinite, iterations = 0, cost = NaN)
    result, _, (code, rule) = out
    return (; code, rule, iterations = result.niterations, cost = 2 * result.bestcost)
end

# THE decision of the robust stall postcondition from the √ρ′-weighted gradient `g = Σ ρ′ Jᵀr`, column norms
# `w = Σ ρ′ ‖Jⱼ‖²`, weighted cost `f2 = Σ ρ′ ‖r‖²` (ρ′ clamped at 0) and whether any datum carried weight — the chart
# LM's and the multi-block LM's (`_blocks_stationary!`), after the unweighted
# exact-fit test: a plateau (every ρ′ = 0) fails, a weighted exact fit holds, else the column cosine at `tol`.
function _robust_stationary_decision(g, w, f2, carried::Bool, tol::Real, ftol::Real)
    carried || return false
    f2 < ftol && return true
    fn = sqrt(f2)
    isnan(fn) && return false
    c = 0.0
    @inbounds for j in eachindex(g, w)
        iszero(w[j]) && continue
        c = max(c, abs(g[j]) / (sqrt(w[j]) * fn))
    end
    return c <= tol
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

It reads the state's two reductions at `x` (`residual_cost!`, then, past the exact-fit test,
`residual_normal_equations!`: `Jⱼᵀr = gⱼ`, `‖Jⱼ‖² = Hⱼⱼ`), so a device state's rows stay where they are.
"""
function _lm_stationary(ews::ManifoldLmWorkspace, lm, x, tol::Real, ftol::Real)
    cost = residual_cost!(ews.r, lm, x)
    2 * cost < ftol && return true
    _, g, H = residual_normal_equations!(ews.r, ews.J, lm, x)
    return _column_cosine(g, LinearAlgebra.diag(H), sqrt(2 * cost)) <= tol
end

# max_j |Jⱼᵀ f| / (‖Jⱼ‖ ‖f‖), with `g` and `w` scratch of length size(J, 2)
# (0 when ‖f‖ = 0; an all-zero column is skipped; NaN propagates, and
# NaN ≤ tol is false).
function _residual_cosine!(g, w, J, f)
    fn = norm(f)
    iszero(fn) && return zero(fn)
    LinearAlgebra.mul!(g, J', f)
    sum!(abs2, w', J)
    return _column_cosine(g, w, fn)
end
# THE cosine of the stall postcondition from `g = Jᵀf`, the squared column norms `w` and ‖f‖ = `fn`.
function _column_cosine(g, w, fn)
    iszero(fn) && return zero(fn)
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
does not run (a block-structured NLLSsolver problem, `nlls_termination`'s postcondition):
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
    is_stationary(blocks, variables, free::AbstractVector{Bool}; tol, ftol) -> Bool

The same stall postcondition for an NLLSsolver problem's residual `blocks` at
its `variables`, in the chart coordinates of the variables `free` marks (a held
variable's columns are left out, as the solver held them): the cost
NLLSsolver minimizes, `F = Σₖ ρ(‖rₖ‖²)` (`ρ` each block's `robustkernel`,
the identity for a plain block), is stationary when `Σₖ ‖rₖ‖² < ftol` (or `Σₖ ρ′ₖ‖rₖ‖² < ftol` with some ρ′ₖ > 0), or
when every free column of the √ρ′-weighted system is numerically orthogonal to
its rows, `max_j |gⱼ| / (√wⱼ ‖√ρ′ r‖) ≤ tol` with `gⱼ = Σₖ ρ′ₖ Jₖⱼᵀ rₖ` and
`wⱼ = Σₖ ρ′ₖ ‖Jₖⱼ‖²` — the test the matrix method above makes of the stacked
rows `√ρ′ₖ rₖ` and Jacobian `√ρ′ₖ Jₖ`, and exactly the robust chart LM's
(`lmsolve_chart!`): ρ′ clamped at 0, the unweighted cost below `ftol` an exact
fit, and a plateau (every block past a redescending loss's cutoff, ρ′ = 0
throughout) NOT stationary. The rows and their Jacobian
(`NLLSsolver.computeresjac` of each block with all its variables, as
[`nlls_rows_and_jacobian`](@ref) evaluates them, a held variable's columns skipped) are accumulated block by block
into `g`, `w` and `F`: the problem's Jacobian is never held.
"""
function is_stationary(blocks, variables::AbstractVector, free::AbstractVector{Bool}; tol::Real, ftol::Real)
    length(free) == length(variables) ||
        throw(DimensionMismatch("is_stationary: $(length(variables)) variables, $(length(free)) free flags"))
    offsets = cumsum([0; [free[v] ? Int(NLLSsolver.dynamic(NLLSsolver.nvars(variables[v]))) : 0 for v in eachindex(variables)]])
    return _blocks_stationary!(zeros(offsets[end]), zeros(offsets[end]), offsets, blocks, variables, free, tol, ftol)
end
# The test itself, into the caller's scratch `g`, `w` (one entry per free chart coordinate): the multi-block LM's and
# the robust chart LM's (its workspace's `g`, `w`: a refit allocates nothing). Free variable `v`'s columns are
# `offsets[v] + 1 : offsets[v] + nvars`.
function _blocks_stationary!(g, w, offsets, blocks, variables, free, tol::Real, ftol::Real)
    fill!(g, 0.0); fill!(w, 0.0)
    F = 0.0; f2 = 0.0; carried = false
    for c in blocks
        vi = NLLSsolver.varindices(c)
        # every variable of the block differentiated (static flags, as `nlls_rows_and_jacobian`); a held one's columns skipped
        r, J = NLLSsolver.computeresjac(NLLSsolver.static(2^length(vi) - 1), c, NLLSsolver.getvars(c, variables)...)
        sq = sum(abs2, r)
        ρ′ = max(NLLSsolver.robustifydcost(NLLSsolver.robustkernel(c), sq)[2], 0.0)
        F += sq; f2 += ρ′ * sq; carried |= ρ′ > 0
        col = 0
        for v in vi
            nv = Int(NLLSsolver.dynamic(NLLSsolver.nvars(variables[v])))
            if free[v]
                o = offsets[v]
                @inbounds for k in 1:nv
                    q = o + k; jc = col + k
                    jr = 0.0; jj = 0.0
                    for i in eachindex(r)
                        jr += J[i, jc] * r[i]; jj += J[i, jc]^2
                    end
                    g[q] += ρ′ * jr; w[q] += ρ′ * jj
                end
            end
            col += nv
        end
    end
    F < ftol && return true
    return _robust_stationary_decision(g, w, f2, carried, tol, ftol)
end

"""
    nlls_rows_and_jacobian(blocks, variables) -> (r, J::SparseMatrixCSC)

The rows of NLLSsolver residual `blocks` at `variables` (an NLLSsolver problem's
variable vector) and their sparse Jacobian in the variables' chart coordinates
— `NLLSsolver.computeresjac` of every block with all its variables free —
the rows in block order, each variable's `NLLSsolver.nvars` columns in variable
order (a run-time size included). What `nlls_termination`'s stall
postcondition reads, and a covariance or a GCV linearization of the same fit.
"""
function nlls_rows_and_jacobian(blocks, variables::AbstractVector)
    offsets = cumsum([0; [Int(NLLSsolver.dynamic(NLLSsolver.nvars(v))) for v in variables]])
    rows = Float64[]; Ii = Int[]; Jj = Int[]; Vv = Float64[]
    for c in blocks
        vi = NLLSsolver.varindices(c)
        rk, Jk = NLLSsolver.computeresjac(NLLSsolver.static(2^length(vi) - 1), c, NLLSsolver.getvars(c, variables)...)
        o = length(rows)
        append!(rows, rk)
        col = 0
        for v in vi, q in (offsets[v] + 1):offsets[v + 1]
            col += 1
            for i in eachindex(rk)
                x = Jk[i, col]
                iszero(x) && continue
                push!(Ii, o + i); push!(Jj, q); push!(Vv, x)
            end
        end
    end
    return rows, sparse(Ii, Jj, Vv, length(rows), offsets[end])
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

"""
    Eliminable{E}(value)

An NLLSsolver variable — a Core [`Manifold`](@ref), an `SVector`, any value
with NLLSsolver's `nvars`/`update` — that the solver may ELIMINATE (`E =
true`): the Schur-complement linear system eliminates every such variable
(a bundle's structure points, a rotation bundle's track directions) before it
solves for the rest. `nvars` and `update` forward to `value` (an update
re-wraps the updated value); `iseliminable` is declared only where the
installed NLLSsolver has the Schur system ([`nlls_features`](@ref)`().schur`),
and `Eliminable{false}` is the same variable left in the full sparse LDL — what
a solver without the Schur system takes, and a fallback test forces.
"""
struct Eliminable{E, V}
    value::V
end
Eliminable{E}(value::V) where {E, V} = Eliminable{E, V}(value)
NLLSsolver.nvars(e::Eliminable) = NLLSsolver.nvars(e.value)
NLLSsolver.update(e::Eliminable{E}, δ, start = 1) where {E} = Eliminable{E}(NLLSsolver.update(e.value, δ, start))
if _NLLS_HAS_SCHUR
    NLLSsolver.iseliminable(::Type{<:Eliminable{true}}) = NLLSsolver.static(true)
end
# The units an area is read in by `_spline_lm_box`: the reference's pixel constants
# are kept at its 1000-px calibrated area.
const _LM_AREA_UNITS = 1000

# =============================================================================
# THE device LM — `static_lm`, at static N inside a KernelAbstractions kernel
# =============================================================================
#
# THE two-unknown LM of the spline models: `fj(p) -> (r, ∂r/∂p)`, the residual and
# its Jacobian from ONE evaluation at `p` (a spline's basis weights once per point:
# every trial point's Jacobian is the next iteration's when the step is accepted),
# from `x₀`, the point bounded to the box `[lb, ub]` (a step is clipped at the
# bound), for at most `max_iter` LM iterations; converged means a STATIONARY point (the
# host LM's success rule, its exact-fit floor and column cosine) whose cost is below the
# acceptance `tol2`. Returns the point, whether it converged and the Jacobian at the point
# (the callers that need only the first two destructure them; the lens inverse
# reads its determinant). `s` is the problem's unit per axis, as the reference's
# pixel: its pixel constants are read in it — the step tolerance `ϵx = 1e-10 / s`,
# the DᵀD clamp `[1e-12, 1e16]·s²`. Device-safe — static arrays only, no
# allocation, no Union return — so the batched kernels run it per ray.
# Generic in the float type `T` of its arguments (Float32 in a device
# kernel; at Float64 the reference's operations and constants exactly). Its stops
# are precision-aware: the step tolerance is the reference's in ulps of T
# (`1e-10·eps(T)/eps(Float64)`), and the absolute residual and gradient stops are
# the reference's `eps(Float64)` or what T resolves of a residual of scale `fs`
# (`(16 eps(T) fs)²`), whichever is larger — so a Float32 solve stops where
# Float32 can resolve it, and at Float64 with `fs` of order one (the SplineLens
# residual, a difference of unit directions) every stop is the reference's
# exactly. The ray–board root of a MappedSurface calls it with `fs` its chart's
# diagonal (mapped_surface.jl `_board_root`). `bound` limits every step: `nothing`
# (the reference's: no limit), or a relative radius `b`, a step longer than
# `b·max(‖x‖, 1)` shortened to it along its direction — COLMAP's
# IterativeUndistortion clipping (`b = 0.1`), which the parametric lens inverse
# reads (lens_rays.jl `_inverse_lm`). It returns `(x, converged, J, e)`, `e` the
# evaluation `fj(x)` itself (anything `fj` returns past `(r, J)` with it: the
# board root's surface point). `cost_floor` is the absolute stops' floor: 0 by
# default, the stops at what T resolves (a root that must be resolved to rounding:
# the lens inverse, whose seed with a residual of 1e-9 is not its root); the
# spline lenses give the reference's `eps(Float64)` (`_SPLINE_REFERENCE_COST_FLOOR`),
# a residual of order one resolved to √eps.
#
# It is the ONE device-side exception to the stack's one LM driver: NLLSsolver
# builds heap problems and cannot run inside a KernelAbstractions kernel, so
# this kernel is the stated single instance the retired-names rule allows
# (`check_naming`'s RETIRED_NAMES: the hand-rolled LM cores stay retired).
#
# Levenberg's and Marquardt's damped Gauss–Newton (Levenberg 1944; Marquardt
# 1963) with Marquardt's diagonal scaling, at any static N (N = 2: the spline lenses'
# inverse, the MappedSurface root; N = 3: VisualGeometryToolkitFeatures' continuous
# DoG-extremum refinement) and derived
# from a static bounded `lmsolve` (Copyright (c) 2023 kamesy, MIT License),
# with its defaults and operations in its order — the damping (λ₀ = 1e-6,
# ×1/13 on ρ > 0.75, ×3 on ρ < 1e-3 or a rejected step, clamped to
# [1e-28, 1e32]), the running-maximum DᵀD clamped to [1e-12, 1e16], the
# unrolled N × N Cholesky-type solve (`_static_lm_increment`), the ftol = gtol = eps and xtol = 1e-10 stops (at
# s = 1). NOT the same bits as a run of that routine: its products are the
# explicit fused ones (`_fma_dot`, `_fma_mul`, `_fma_tmul`: base/fused_products.jl; the increment's
# `fma`s), the same in every compiled copy, where the routine's fuse as LLVM decides.
# Measured against a recorded run of it on 1000 seeded problems
# (test/fixtures/pixel_lm), at s = 1: at most 5.5e-12 px (Julia 1.13) /
# 4.5e-8 px (1.12) apart, 1.8e-6 px on the review's 20 000-problem probe,
# converged flags identical; gated at 2e-6 px (pixel_lm_reference_tests.jl).
"""
    StaticLmStops(scale::SVector{N, T}; acceptance, residual_scale = 1, cost_floor = 0, max_iter, stationarity_tol)

THE stop settings of the device LM [`static_lm`](@ref) (isbits: a kernel takes it as an argument), owned with the
host LM's ([`LmStops`](@ref)): `max_iter` defaults to configs/default.toml `[lm] device_max_iter`, `stationarity_tol`
to `[lm] stationarity_tol` — the ONE success rule's cosine, the host's.

- `scale`: the problem's unit per coordinate, in which the step tolerance (`1e-10·(eps(T)/eps(Float64)) ./ scale`,
  RELATIVE to the coordinate) and the DᵀD clamp (`[1e-12, 1e16]·scale²`) are read.
- `acceptance`: the cost a solution must be below, F < `acceptance` — a root-finding caller's residual bound.
- `residual_scale`: the residual's scale `fs`, so the absolute cost and gradient floors are what `T` resolves of it,
  `(16 eps(T) fs)²`.
- `cost_floor`: the absolute floor under those (`0`: resolved to rounding; the spline lenses' reference routine
  stops at `eps(Float64)`).
"""
struct StaticLmStops{N, T <: AbstractFloat}
    scale::SVector{N, T}
    acceptance::T
    residual_scale::T
    cost_floor::T
    stationarity_tol::T
    max_iter::Int
end
@inline StaticLmStops(
    scale::SVector{N, T}; acceptance::Real, residual_scale::Real = one(T), cost_floor::Real = zero(T),
    max_iter::Integer = _STATIC_LM_MAX_ITER, stationarity_tol::Real = _LM_STATIONARITY_TOL
) where {N, T <: AbstractFloat} =
    StaticLmStops{N, T}(scale, T(acceptance), T(residual_scale), T(cost_floor), T(stationarity_tol), Int(max_iter))

"""
    static_lm(fj, x₀::SVector{N, T}, lb, ub, stops::StaticLmStops{N, T}[, bound]) -> (x, converged, J, e, rule::StopRule)

The stack's ONE device-side Levenberg–Marquardt (the spline lenses' inverse at
N = 2, any static N): Marquardt-scaled, bounded to the box `[lb, ub]` (a step is
clipped to it), with an unrolled Cholesky-type solve and explicit fused products,
so it allocates nothing, never throws, and runs inside a KernelAbstractions kernel
with the same bits in every compiled copy. `fj(x)` returns `(r, J, …)` (the residual
`SVector`, its `SMatrix` Jacobian, anything else returned with them as `e`); `stops`
its stop settings ([`StaticLmStops`](@ref)); `bound` a relative step radius (`nothing`: none).
`converged` is THE success rule of the stack's LMs — the point is STATIONARY: zero to what
the stops resolve (the cost below the exact-fit floor, the gradient below its floor, or the
undamped Gauss–Newton step from the point below the step tolerance: the device form of the
host's exact-fit test), or `max_j |Jⱼᵀr| / (‖Jⱼ‖ ‖r‖) ≤ stops.stationarity_tol`, the host
LM's cosine; on the box the gradient of a coordinate a bound holds (its descent leaves the box)
is dropped, the bounded problem's stationarity — and its cost is below `stops.acceptance`.
`rule` is the stop that fired: `ExactFit` (the cost floor),
`ToleranceMet` (the gradient floor), `IncrementSize` (the step), `CostDecrease`
(the relative cost decrease) or `IterationCap`; [`static_lm_termination`](@ref)
gives the §14 record. NLLSsolver ([`lmsolve_chart!`](@ref)) is every host fit's LM.
"""
@inline function static_lm(
        fj::FJ, x₀::SVector{N, T}, lb::SVector{N, T}, ub::SVector{N, T}, stops::StaticLmStops{N, T}, bound::B = nothing
    ) where {FJ, N, T <: AbstractFloat, B <: Union{Nothing, T}}
    s, fs, cost_floor, maxiter = stops.scale, stops.residual_scale, stops.cost_floor, stops.max_iter
    ϵ = eps(T)
    ϵx = T(1.0e-10 * (eps(T) / eps(Float64))) ./ s   # the reference's 1e-10 at Float64, the same in ulps of T
    # the absolute stops: the cost floor (the reference's eps(Float64)), or what T resolves of a residual of scale fs
    ϵF = max(cost_floor, (16 * ϵ * fs)^2)
    ϵG = max(cost_floor, (16 * ϵ * fs)^2 * maximum(s) / _LM_AREA_UNITS)
    λ = T(1.0e-6)
    x = min.(max.(x₀, lb), ub)
    ex = fj(x)
    f, Jx = ex[1], ex[2]
    F = sum(abs2, f)
    F < ϵF && return x, F < stops.acceptance, Jx, ex, ExactFit
    g = _fma_tmul(Jx, f)
    norm(g, Inf) < ϵG && return x, F < stops.acceptance, Jx, ex, ToleranceMet
    JtJ = _fma_tmul(Jx, Jx)
    DtD = clamp.(LinearAlgebra.diag(JtJ), T(1.0e-12) .* s .^ 2, T(Inf))
    DtD_max = T(1.0e16) .* s .^ 2
    iter = 0
    rule = IterationCap
    while iter < maxiter
        iter += 1
        accepted = false
        ac = zero(T)
        pr = zero(T)
        ρ = zero(T)
        xk = x
        ek = ex
        fk = f
        Jk = Jx
        Fk = F
        p, info = _static_lm_increment(JtJ, λ, DtD, g)
        # a step clipped to zero by either bound is rejected, as a failed solve is
        clipped = false
        if info == 0
            p = _bounded_increment(bound, p, x)
            p = min.(x .- lb, p)
            clipped = all(iszero, p)
            if !clipped
                p = max.(x .- ub, p)
                clipped = all(iszero, p)
            end
        end
        if info == 0 && !clipped
            xk = x .- p
            ek = fj(xk)
            fk, Jk = ek[1], ek[2]
            Fk = sum(abs2, fk)
            if Fk < F
                ac = F - Fk
                pr = _fma_dot(p, (@. λ * DtD * p + g))
                ρ = ac / pr
                accepted = true
            end
            if all(>(0), (@. abs(p) < ϵx * (ϵx + abs(x))))
                if accepted
                    x, f, F, Jx, ex = xk, fk, Fk, Jk, ek
                end
                rule = IncrementSize
                break
            end
        end
        if accepted
            x = xk
            f = fk
            Fold = F
            F = Fk
            Jx = Jk
            ex = ek
            # the reference routine's ftol test reads its swapped buffers: `Fk` is then the old F
            abs(ac) < ϵ * Fold && pr < ϵ * Fold && ρ < 2 && (rule = CostDecrease; break)
            g = _fma_tmul(Jx, f)
            norm(g, Inf) < ϵG && (rule = ToleranceMet; break)
            JtJ = _fma_tmul(Jx, Jx)
            DtD = clamp.(LinearAlgebra.diag(JtJ), DtD, DtD_max)
            if ρ > T(0.75)
                λ *= T(1 / 13)
            elseif ρ < T(1.0e-3)
                λ *= 3
            end
        else
            λ *= 3
        end
        λ = clamp(λ, T(1.0e-28), T(1.0e32))
    end
    # THE success rule at the returned point: STATIONARY — the cost zero to what the stops resolve (the exact-fit and
    # gradient floors, or: the undamped Gauss–Newton step from x, (JᵀJ)⁻¹Jᵀr, is below the step tolerance, the
    # IncrementSize test — a root resolved to the step tolerance, or a minimizer whose gradient no step can follow),
    # or the host LM's column cosine (`_column_cosine`, of g = Jᵀr and the squared column norms diag JᵀJ) — and accepted.
    # The step-tolerance test is the device form of the host's exact-fit test (F < ftol: zero by the cost tolerance):
    # at a root of a square system the rows' rounding points in no direction, so the cosine says nothing there.
    # On the box, stationarity is the bounded problem's (Karush–Kuhn–Tucker): a coordinate held at a bound by a
    # gradient pointing out of the box (the descent −g leaves it) is not free, its gradient component dropped.
    g = _fma_tmul(Jx, f)
    held = @. ((x <= lb) & (g > 0)) | ((x >= ub) & (g < 0))
    g = ifelse.(held, zero(T), g)
    JtJ = _fma_tmul(Jx, Jx)
    pgn, info = _static_lm_increment(JtJ, zero(T), LinearAlgebra.diag(JtJ), g)
    resolved = info == 0 && all(>(0), (@. abs(pgn) < ϵx * (ϵx + abs(x))))
    stationary = F < ϵF || norm(g, Inf) < ϵG || resolved || _column_cosine(g, LinearAlgebra.diag(JtJ), sqrt(F)) <= stops.stationarity_tol
    return x, stationary & (F < stops.acceptance), Jx, ex, rule
end


"""
    static_lm_termination(converged::Bool, rule::StopRule) -> (ReturnCode.T, StopRule)

The §14 termination record of a [`static_lm`](@ref) solve: `ReturnCode.Success`
when it converged; otherwise `ReturnCode.MaxIters` at the iteration cap and
`ReturnCode.Stalled` at any other stop (a step, cost or gradient stop short of
the tolerance).
"""
static_lm_termination(converged::Bool, rule::StopRule) =
    (converged ? ReturnCode.Success : rule == IterationCap ? ReturnCode.MaxIters : ReturnCode.Stalled, rule)

# The step `p` (the LM subtracts it) within the relative radius `b·max(‖x‖, 1)`.
@inline _bounded_increment(::Nothing, p, x) = p
@inline function _bounded_increment(b::T, p::SVector{N, T}, x::SVector{N, T}) where {N, T}
    r2 = b * b * max(_plain_sumsq(x), one(T))
    n2 = _plain_sumsq(p)
    return ifelse(n2 > r2, p * sqrt(r2 / n2), p)
end
# x₁² + x₂² + … left to right in plain (never contracted) arithmetic: at N = 2 the
# expression `x[1] * x[1] + x[2] * x[2]` exactly.
@generated function _plain_sumsq(x::SVector{N}) where {N}
    ex = :(x[1] * x[1])
    for k in 2:N
        ex = :($ex + x[$k] * x[$k])
    end
    return :(@inbounds $ex)
end

# The unrolled solve of (JᵀJ + λ·DᵀD) p = g at static N: the damped diagonals
# checked first, in order (`info` = the first negative one's index: a rejected
# step), then symmetric Gaussian elimination on the lower triangle and back
# substitution, every product-sum an explicit `fma`. At N = 2 it is the reference
# routine's 2 × 2 operations in its order exactly:
#   A11 = fma(λ, D1, J11); A22 = fma(λ, D2, J22); A22 = fma(−A21, A21/A11, A22);
#   x2 = fma(−x1, A21/A11, x2); x2 /= A22; x1 = fma(−x2, A21, x1); x1 /= A11.
@generated function _static_lm_increment(JtJ::StaticMatrix{N, N}, λ, DtD, g) where {N}
    A(i, j) = Symbol(:A, i, :_, j)
    X(i) = Symbol(:x, i)
    body = Expr[]
    for i in 1:N
        push!(body, :($(X(i)) = g[$i]))
    end
    for i in 1:N
        for j in 1:(i - 1)
            push!(body, :($(A(i, j)) = JtJ[$i, $j]))
        end
        push!(body, :($(A(i, i)) = fma(λ, DtD[$i], JtJ[$i, $i])))
        push!(body, :($(A(i, i)) >= zero($(A(i, i))) || return zero(SVector{$N, typeof($(A(i, i)))}), $i))
    end
    for k in 1:(N - 1), i in (k + 1):N
        for j in (k + 1):i
            push!(body, :($(A(i, j)) = fma(-$(A(i, k)), $(A(j, k)) / $(A(k, k)), $(A(i, j)))))
        end
        push!(body, :($(X(i)) = fma(-$(X(k)), $(A(i, k)) / $(A(k, k)), $(X(i)))))
    end
    for i in N:-1:1
        for j in (i + 1):N
            push!(body, :($(X(i)) = fma(-$(X(j)), $(A(j, i)), $(X(i)))))
        end
        push!(body, :($(X(i)) /= $(A(i, i))))
    end
    return quote
        @inbounds begin
            $(body...)
        end
        return SVector($([X(i) for i in 1:N]...)), 0
    end
end


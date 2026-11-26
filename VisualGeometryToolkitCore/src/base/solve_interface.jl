# =============================================================================
# The estimation interface — SciML's problem + algorithm, CommonSolve.jl's verbs
# =============================================================================
#
#     solve(problem, algorithm; settings...)        init(problem, algorithm; settings...) → solve!(cache)
#     solve(problem; settings...)                   the problem's declared default_algorithm
#
# The design record is docs/design/sciml-mapping.md (the SciML study and the owner's rulings of 2026-11-23).
#
# - A PROBLEM says what is estimated, from which data and in which context. Its `f` is the problem's interface
#   object (SciML's `f`): a VisualGeometryToolkitEstimation family problem (`HomographyProblem(cs)`), which
#   implements the chart-LM row protocol and the RANSAC problem contract for its own type; a residual closure
#   for the device LM; or, for a closed form, the data themselves beside the model type. Its context is `p`,
#   a TYPE PARAMETER (SciML's parameters; `NullParameters()` when there is none). Run settings (rng, scale or
#   width, stops) are solve keywords, never problem fields.
# - An ALGORITHM is a typed value saying how (`DLT()`, `LevenbergMarquardt()`, a minimal solver, Estimation's
#   `Ransac(…)`); its configuration is its fields.
# - Core owns `solve` and `init` for every problem (SciML's front end, CommonSolve's rule that an extension owns
#   its first argument's type): the kind law, then `__solve`/`__init`, the extension points a package adds
#   methods to for ITS OWN algorithm types. The front end records the problem and the algorithm on a
#   `FitResult` (`with_problem`), the one place they are set.
# =============================================================================

import CommonSolve: solve, init, solve!

"""
    AbstractEstimationProblem

The parent of every estimation problem: [`EstimationProblem`](@ref) (from scratch),
[`RefineProblem`](@ref) (from a required guess, SciML's `NonlinearLeastSquaresProblem`) and
[`MinimalProblem`](@ref) (every root of a minimal sample). Solved by `solve(problem, algorithm)`.
"""
abstract type AbstractEstimationProblem end

"""
    AbstractEstimationAlgorithm

The parent of every estimation algorithm: a typed value saying how a problem is solved
([`DLT`](@ref), [`LevenbergMarquardt`](@ref), [`StaticLevenbergMarquardt`](@ref), every
[`MinimalSolver`](@ref), VisualGeometryToolkitEstimation's `Ransac`). Its kind is
[`estimator_kind`](@ref)`(algorithm)`; a package that defines one adds `estimator_kind` and its
`__solve` (or `__init` + `solve!`) methods.
"""
abstract type AbstractEstimationAlgorithm end

"""
    NullParameters()

The context `p` of a problem that has none (SciMLBase's name, mirrored: the design record's D1). Indexing or
iterating it is an error that says so.
"""
struct NullParameters end
Base.getindex(::NullParameters, i...) = _no_context()
Base.iterate(::NullParameters) = _no_context()
@noinline _no_context() =
    throw(ArgumentError("the problem has no context `p` (NullParameters()): give one as the problem's `p` argument"))

"""
    ClassDefault()

An algorithm field left at its problem class's default of record (the configuration of record,
`configs/default.toml`), resolved by the method that knows the class: [`LevenbergMarquardt`](@ref)'s
`optimization_strategy`, VisualGeometryToolkitEstimation's `Ransac`'s `local_optimization_strategy` (the design
record's E7).
"""
struct ClassDefault end

# -----------------------------------------------------------------------------
# Kinds — the naming convention's table, as types
# -----------------------------------------------------------------------------

"""
    EstimatorKind

The kind of an estimation algorithm, which fixes how it can fail (conventions/naming.md):
[`MinimalKind`](@ref) (every root of a minimal sample; a degenerate sample has none),
[`ClosedFormKind`](@ref) (one deterministic algebraic step; can only be degenerate), [`FitKind`](@ref)
(iterative, seeds itself; can fail to converge), [`FindKind`](@ref) (sample and verify; decides the
inliers) and [`RefineKind`](@ref) (needs a guess; can end in a local minimum). Read by
[`needs_initial_model`](@ref) and [`decides_inliers`](@ref); randomness is [`supports_solve_rng`](@ref),
a trait of its own.
"""
abstract type EstimatorKind end
"`MinimalKind()`: the [`EstimatorKind`](@ref) of a minimal solver — every root of a minimal sample."
struct MinimalKind <: EstimatorKind end
"`ClosedFormKind()`: the [`EstimatorKind`](@ref) of a one-step algorithm (DLT, Kabsch) — deterministic, can only be degenerate."
struct ClosedFormKind <: EstimatorKind end
"`FitKind()`: the [`EstimatorKind`](@ref) of an iterative estimator that seeds itself — can fail to converge."
struct FitKind <: EstimatorKind end
"`FindKind()`: the [`EstimatorKind`](@ref) of a sample-and-verify search (RANSAC) — decides the inliers."
struct FindKind <: EstimatorKind end
"`RefineKind()`: the [`EstimatorKind`](@ref) of a refinement from a required guess — can end in a local minimum."
struct RefineKind <: EstimatorKind end

"""
    estimator_kind(algorithm) -> EstimatorKind

The kind of `algorithm` ([`EstimatorKind`](@ref)). Every concrete [`AbstractEstimationAlgorithm`](@ref) defines
one method (a contract generic: a package defines it for its own algorithm types); the kind law of
[`solve`](@ref) reads it.
"""
function estimator_kind end

"""
    needs_initial_model(algorithm) -> Bool

Whether `algorithm` refines a guess (a [`RefineKind`](@ref)): it solves only a [`RefineProblem`](@ref), which
cannot be built without one.
"""
needs_initial_model(alg) = estimator_kind(alg) isa RefineKind

"""
    decides_inliers(algorithm) -> Bool

Whether `algorithm` decides an inlier set (a [`FindKind`](@ref)); every other kind takes its data as inliers (at
most soft-weighted).
"""
decides_inliers(alg) = estimator_kind(alg) isa FindKind

"""
    supports_solve_rng(problem, algorithm) -> Bool

Whether `solve(problem, algorithm; rng)` takes an `rng` and draws from it (SciMLBase's trait, on the pair). Derived
from the kind — a [`FindKind`](@ref) draws samples — and overridden by an algorithm that is randomized without
being a find (VisualGeometryToolkitEstimation's `SimFitPP`). Only a solve whose pair answers `true` has an `rng`
keyword (check_naming); every other refuses one.
"""
supports_solve_rng(::AbstractEstimationProblem, alg) = estimator_kind(alg) isa FindKind

# -----------------------------------------------------------------------------
# The problems
# -----------------------------------------------------------------------------

"""
    read_context(p, required::Tuple{Vararg{Symbol}}, optional::NamedTuple, who) -> NamedTuple

THE reading of a solve's keyed context `p`: the `required` keys and the `optional` ones (a NamedTuple of their
defaults), in any order; the one `ArgumentError`, naming the missing and the unknown keys (`who` the algorithm), for
any other (`NullParameters()` holds none). Returns the context with every optional key filled.
"""
function read_context end
@inline read_context(::NullParameters, required::Tuple{Vararg{Symbol}}, optional::NamedTuple, who) =
    read_context((;), required, optional, who)
@inline function read_context(p::NamedTuple, required::Tuple{Vararg{Symbol}}, optional::NamedTuple, who)
    _has_keys(p, required) && _known_keys(keys(p), required, optional) || _context_error(p, required, optional, who)
    return merge(optional, p)
end
# The two checks as tuple recursions (no closure over `p`, which boxed a context of static matrices on every read).
@inline _has_keys(p::NamedTuple, ks::Tuple) = isempty(ks) || (haskey(p, first(ks)) && _has_keys(p, Base.tail(ks)))
@inline _known_keys(ks::Tuple, required, optional) =
    isempty(ks) || ((first(ks) in required || haskey(optional, first(ks))) && _known_keys(Base.tail(ks), required, optional))
@noinline function _context_error(p, required, optional, who)
    missing_keys = Tuple(k for k in required if !haskey(p, k))
    unknown = Tuple(k for k in keys(p) if !(k in required || haskey(optional, k)))
    throw(ArgumentError(string(who, ": the context `p` ",
        isempty(missing_keys) ? "" : string("lacks ", join(missing_keys, ", ")), isempty(missing_keys) || isempty(unknown) ? "" : " and ",
        isempty(unknown) ? "" : string("holds the unknown key(s) ", join(unknown, ", ")),
        " (it reads ", join((required..., keys(optional)...), ", "), isempty(optional) ? "" : string("; optional: ", join(keys(optional), ", ")), ")")))
end

"""
    EstimationProblem(f, p = NullParameters())
    EstimationProblem(M::Type, data, p = NullParameters())

Estimate a model from scratch: no guess. Solved by a [`ClosedFormKind`](@ref), [`FitKind`](@ref) or
[`FindKind`](@ref) algorithm.

- `EstimationProblem(f, p)`: `f` is a problem-interface object whose [`model_type`](@ref) is the model estimated
  — a VisualGeometryToolkitEstimation family problem, `solve(EstimationProblem(HomographyProblem(cs)), Ransac();
  rng)`.
- `EstimationProblem(M, data, p)`: a model type `M` and the data a closed form or a fit reads directly,
  `solve(EstimationProblem(HomographyMat, cs), DLT())`.

`p` is the problem's context (an extent, a camera pair, a centre; [`NullParameters`](@ref) when there is
none), a type parameter: `EstimationProblem{M, F, P}`.
"""
struct EstimationProblem{M, F, P} <: AbstractEstimationProblem
    f::F
    p::P
end
EstimationProblem(f::F, p::P = NullParameters()) where {F, P} = EstimationProblem{model_type(f), F, P}(f, p)
EstimationProblem(::Type{M}, data::D, p::P = NullParameters()) where {M, D, P} = EstimationProblem{M, D, P}(data, p)

"""
    RefineProblem(f, u0, p = NullParameters(); lb = nothing, ub = nothing)

Refine the guess `u0` (SciMLBase's `NonlinearLeastSquaresProblem(f, u0, p; lb, ub)`): there is no constructor
without one. `f` is the residual interface — a VisualGeometryToolkitEstimation family problem, whose chart-LM
rows and constraint it implements (`RefineProblem(HomographyProblem(cs), H₀)`), or, for
[`StaticLevenbergMarquardt`](@ref), the residual function `f(x) -> (r, J, …)`; `lb`/`ub` a box on `u0`'s
coordinates (`nothing`: unbounded). Solved only by a [`RefineKind`](@ref) algorithm; the default of a constraint problem's refine
is [`LevenbergMarquardt`](@ref)`()`, and a residual closure has none (the device LM is named).
"""
struct RefineProblem{F, U, P, LB, UB} <: AbstractEstimationProblem
    f::F
    u0::U
    p::P
    lb::LB
    ub::UB
end
@inline RefineProblem(f::F, u0::U, p::P = NullParameters(); lb::LB = nothing, ub::UB = nothing) where {F, U, P, LB, UB} =
    RefineProblem{F, U, P, LB, UB}(f, u0, p, lb, ub)

# The refusal of a box by an unbounded host algorithm (its name `alg`): a RefineProblem's `lb`/`ub` is the device LM's.
_refuse_box(prob::RefineProblem, alg::Symbol) = (prob.lb === nothing && prob.ub === nothing) || throw(ArgumentError(
    string(alg, " is unbounded: a RefineProblem's `lb`/`ub` is StaticLevenbergMarquardt's (the device LM)")))

# THE refusal of a contract generic `g` that the type of `x` implements no method of, and what needs it (`need`).
@noinline _no_method(x, g, need::AbstractString) =
    throw(ArgumentError(string("a ", nameof(typeof(x)), " implements no `", nameof(g), "` method: ", need)))

"""
    MinimalProblem(M::Type, data, sample, p = NullParameters())

The minimal sample `sample` (indices into the data) from which a model is solved, EVERY root returned as a
[`FixedModels`](@ref) (no root on a degenerate sample). Solved only by a [`MinimalSolver`](@ref) of the model
type: `solve(MinimalProblem(HomographyMat, cs, idx), Homography4pt())`; `solve(MinimalProblem(M, cs, idx))` runs
the model's one minimal solver ([`default_algorithm`](@ref)). `data` is any container of the
[`sources`](@ref)/[`targets`](@ref) protocol the solver reads (its field is `f`, the problem's interface).
"""
struct MinimalProblem{M, F, I, P} <: AbstractEstimationProblem
    f::F
    sample::I
    p::P
end
@inline MinimalProblem(::Type{M}, data::D, sample::I, p::P = NullParameters()) where {M, D, I, P} =
    MinimalProblem{M, D, I, P}(data, sample, p)

# A minimal sample without a context: a minimal solver that reads none takes only these, so a stray `p` is refused (no
# method) rather than ignored. (An EstimationProblem's algorithms say `EstimationProblem{M, D, NullParameters}`.)
const _MinimalNoContext = MinimalProblem{<:Any, <:Any, <:Any, NullParameters}

"""
    model_type(x) -> Type

The model type `x` estimates or produces, one function across the estimation interface: a problem's (below), a
[`MinimalSolver`](@ref)'s, and a problem-interface object's — the `f` of [`EstimationProblem`](@ref)`(f, p)`, which reads
it to type the problem (VisualGeometryToolkitEstimation's `HomographyProblem(cs)` estimates a `HomographyMat`). A
package defines it for its own problem-interface types: it is a contract generic.
"""
function model_type end

"""
    model_type(problem::AbstractEstimationProblem) -> Type

The model type a problem estimates: the `M` of an [`EstimationProblem`](@ref) or a [`MinimalProblem`](@ref),
the guess's type for a [`RefineProblem`](@ref).
"""
model_type(::EstimationProblem{M}) where {M} = M
model_type(::MinimalProblem{M}) where {M} = M
model_type(prob::RefineProblem) = typeof(prob.u0)

# -----------------------------------------------------------------------------
# The front end — the kind law, the default algorithm, the extension points
# -----------------------------------------------------------------------------

"""
    __solve(problem, algorithm; settings...)
    __init(problem, algorithm; settings...)

The extension points of [`solve`](@ref) and [`init`](@ref) (SciMLBase's names): a package adds methods for its
OWN algorithm types, on Core's problems, and declares nothing else. `solve` checks the kind law first and records
the problem and algorithm on the result; a method neither repeats the check nor sets them. An algorithm that holds
workspaces defines `__init` (returning its cache) and `solve!(cache)`; `__solve` then defaults to
`solve!(__init(…))`, CommonSolve's identity.
"""
function __solve end
@doc (@doc __solve) function __init end
__solve(prob::AbstractEstimationProblem, alg::AbstractEstimationAlgorithm; kwargs...) = solve!(__init(prob, alg; kwargs...))
# A pair with neither method: the MethodError of the pair itself, stated (the generic route above lands here).
__init(prob::AbstractEstimationProblem, alg::AbstractEstimationAlgorithm; kwargs...) = throw(MethodError(__solve, (prob, alg)))

"""
    solve(problem, algorithm; settings...) -> FitResult (or FixedModels for a MinimalProblem)
    solve(problem; settings...)

Solve an estimation problem with an algorithm (CommonSolve's `solve`, shared with SciML). The keywords are run
settings only — an `rng` for an algorithm that draws ([`supports_solve_rng`](@ref)), a `scale` or `width` for a
robust refine, the `stops` of an LM — and an algorithm refuses one it does not take (`MethodError`). Without an
algorithm, the problem's declared [`default_algorithm`](@ref) runs; a problem with none is an `ArgumentError`
listing the algorithms that solve it.

The kind law is checked first, before any method runs (`ArgumentError`): a [`RefineProblem`](@ref) only by a
[`RefineKind`](@ref), a [`MinimalProblem`](@ref) only by a [`MinimalSolver`](@ref) of its model type, an
[`EstimationProblem`](@ref) by neither.
"""
@inline function solve(prob::AbstractEstimationProblem, alg::AbstractEstimationAlgorithm; kwargs...)
    check_kind(prob, alg)
    return with_problem(__solve(prob, alg; kwargs...), prob, alg)
end
solve(prob::AbstractEstimationProblem; kwargs...) = solve(prob, default_algorithm(prob); kwargs...)

"""
    init(problem, algorithm; settings...) -> cache

The cache of an algorithm that holds workspaces (CommonSolve's `init`), after the kind law: an
[`EstimationCache`](@ref) of the algorithm's own (`__init`'s) with the problem and the algorithm. `solve!(cache)` runs
it and records them on the result; `reinit!(cache[, u0]; f, p, alg, settings...)` points it at a new guess, context,
data, algorithm configuration or run setting. The only way a workspace is reused across solves.
"""
function init(prob::AbstractEstimationProblem, alg::AbstractEstimationAlgorithm; kwargs...)
    check_kind(prob, alg)
    return EstimationCache(__init(prob, alg; kwargs...), prob, alg)
end

"""
    EstimationCache(cache, prob, alg)

What [`init`](@ref) returns: the algorithm's own `cache` (its `__init`'s) with the problem and the algorithm, so that
`solve!(c)` records them on the result as `solve` does ([`with_problem`](@ref): ONE place for both paths). Its
`solve!` runs the cache's owner's `solve!(c.cache)`; its `reinit!(c[, u0]; f, p, alg, settings...)` forwards to the
owner's `reinit!(c.cache[, u0]; f, p[, alg][, settings...])` with BOTH of the new problem's (the caller's as given,
each kept from the cache's problem where the call names none), the new algorithm and run settings only where the call
names them, and records the problem and the algorithm it now solves.

A result of `solve!(c)` may ALIAS the cache's buffers (its owner documents which: a RANSAC result's `resid` and
weights), valid until the next `solve!` of the same cache, as SciML's caches: a caller that keeps a result across
solves copies it.
"""
mutable struct EstimationCache{C, Prob <: AbstractEstimationProblem, A <: AbstractEstimationAlgorithm}
    const cache::C
    prob::Prob
    alg::A
end
solve!(c::EstimationCache) = with_problem(solve!(c.cache), c.prob, c.alg)
function reinit!(c::EstimationCache{C, Prob, A}, u0...; alg = nothing, f = c.prob.f, p = c.prob.p,
                 settings...) where {C, Prob, A}
    # every refusal first, so a refused call leaves both sides as they were: an algorithm of another concrete type
    # (the cache, its workspaces and every result type are built for A), then a problem of another type
    alg === nothing || alg isa A || throw(ArgumentError(string("reinit!: the cache was initialized for a ", A,
        "; the new algorithm is a ", typeof(alg), " (init a new cache for an algorithm of another type)")))
    prob = _reprob(c.prob, u0...; f, p)
    prob isa Prob || throw(ArgumentError(string("reinit!: the cache was initialized for a ", Prob, "; the new ",
        "guess, context or data make a ", typeof(prob), " (init a new cache for a problem of another type)")))
    # the owner sees the whole new problem (the caller's f and p as given), and a new algorithm or run setting only
    # where the call names one
    alg === nothing ? reinit!(c.cache, u0...; f = prob.f, p = prob.p, settings...) :
        reinit!(c.cache, u0...; f = prob.f, p = prob.p, alg, settings...)
    c.prob = prob
    alg === nothing || (c.alg = alg)
    return c
end
# The problem with a new guess `u0`, context `p` or interface `f`, every other field as it was.
_reprob(prob::EstimationProblem{M}; f = prob.f, p = prob.p) where {M} = EstimationProblem{M, typeof(f), typeof(p)}(f, p)
_reprob(prob::RefineProblem, u0 = prob.u0; f = prob.f, p = prob.p) = RefineProblem(f, u0, p; lb = prob.lb, ub = prob.ub)
_reprob(prob::MinimalProblem{M}; f = prob.f, p = prob.p) where {M} = MinimalProblem(M, f, prob.sample, p)
init(prob::AbstractEstimationProblem; kwargs...) = init(prob, default_algorithm(prob); kwargs...)

"""
    reinit!(cache[, u0]; f, p, alg, settings...) -> cache

Point a solve's cache at a new guess `u0`, context `p` or problem interface `f` (new data: a new family problem),
keeping its workspaces (SciMLBase's `reinit!(cache[, u0]; p)`, extended by `f` because the stack's data live in
`f`, the design record's D8). `alg` is a new configuration of the SAME algorithm: a value of the cache's algorithm's
concrete type (another type is an `ArgumentError`, refused before anything changes), which the owner re-resolves
against the kept workspaces (a scorer at a new width); a run setting (`rng`) replaces the cache's. The cache's owner
adds the methods; a keyword it does not take is a `MethodError`.

What the next `solve!` returns may alias the cache's buffers until the solve after it ([`EstimationCache`](@ref)).
"""
function reinit! end

# -----------------------------------------------------------------------------
# The kind law
# -----------------------------------------------------------------------------

"""
    check_kind(problem, algorithm) -> nothing

The kind law: an `ArgumentError` naming the problem `algorithm` solves when `problem` is of another kind, before
any method runs. Resolved at compile time for a concrete pair (the checks are on types).
"""
@inline check_kind(prob::AbstractEstimationProblem, alg) = _check_kind(prob, estimator_kind(alg), alg)

@inline _check_kind(::RefineProblem, ::RefineKind, alg) = nothing
@inline _check_kind(::RefineProblem, ::EstimatorKind, alg) = throw(ArgumentError(
    "$(nameof(typeof(alg))) estimates from scratch: it solves EstimationProblem(…), and a RefineProblem's guess would be ignored"))
@inline _check_kind(::EstimationProblem, ::RefineKind, alg) = throw(ArgumentError(
    "$(nameof(typeof(alg))) refines a guess: it solves RefineProblem(f, u0)"))
@inline _check_kind(::EstimationProblem, ::MinimalKind, alg) = _refuse_minimal(alg)
@inline _check_kind(::RefineProblem, ::MinimalKind, alg) = _refuse_minimal(alg)
@inline _check_kind(::EstimationProblem, ::EstimatorKind, alg) = nothing
@inline _check_kind(prob::MinimalProblem, ::MinimalKind, alg) = _check_minimal_model(prob, alg)
@inline _check_kind(::MinimalProblem, ::EstimatorKind, alg) = throw(ArgumentError(
    "$(nameof(typeof(alg))) does not solve a minimal sample: a MinimalProblem is solved by a minimal solver"))

@noinline _refuse_minimal(alg) = throw(ArgumentError(
    "$(nameof(typeof(alg))) is a minimal solver: it solves MinimalProblem(M, data, sample)"))

# A minimal solver solves the model type it produces (`model_type(solver)`): either side may be the narrower
# (`HomographyMat` against a `HomographyMat{Float64}` sample's problem).
@inline function _check_minimal_model(::MinimalProblem{M}, alg) where {M}
    N = model_type(alg)
    (M <: N || N <: M) && return nothing
    throw(ArgumentError("$(nameof(typeof(alg))) solves a $(N), not a $(M)"))
end

# --- the default algorithm -----------------------------------------------------

"""
    default_algorithm(problem) -> AbstractEstimationAlgorithm

The algorithm `solve(problem)` runs, DECLARED per problem type and deterministic and guess-free (decision 7 of
2026-11-23; the design record's D3): a closed form over a fit, [`LevenbergMarquardt`](@ref)`()` for a
[`RefineProblem`](@ref) of a constraint problem, the model's one minimal solver for a [`MinimalProblem`](@ref). A problem with no
qualifying algorithm declares none — robustness is always named — and `solve(problem)` is then an
`ArgumentError` listing the algorithms that solve it, generated from the `__solve`/`__init` methods that apply.
A contract generic: a package declares the defaults of its own problem types.
"""
default_algorithm(prob::AbstractEstimationProblem) = throw(ArgumentError(_no_default_message(prob)))

@noinline function _no_default_message(prob)
    algs = String[]
    for f in (__solve, __init), m in methods(f)
        sig = Base.unwrap_unionall(m.sig)
        length(sig.parameters) >= 3 || continue
        P, A = sig.parameters[2], sig.parameters[3]
        A isa TypeVar && (A = A.ub)
        A = Base.unwrap_unionall(A)
        (A isa DataType && A !== AbstractEstimationAlgorithm) || continue   # the generic __solve → __init route
        typeof(prob) <: Base.rewrap_unionall(P, m.sig) || continue
        push!(algs, string(nameof(A)))
    end
    sort!(unique!(algs))
    return string("a ", nameof(typeof(prob)), " of a ", model_type(prob), " has no default algorithm: name one — ",
        isempty(algs) ? "no loaded algorithm solves it" : join(algs, ", "), " (solve(problem, algorithm))")
end

# =============================================================================
# Solver layer — CspondNormalization, the MinimalSolver protocol, model registry
# =============================================================================
#
# The verb half of the solver/correspondence layering (#21):
#
#   CspondNormalization          normalization as a value: forward AND inverse maps
#                         computed once; models move between the data frame
#                         and the normalized frame by `∘` with its two sides
#   MinimalSolver{M}      exactly-determined solvers — `solve(s, cs, idx)`
#                         returns FixedModels (count = 0 on failure)
#   minimal_solvers /     the induced-model registry: which models a
#   induced_models        correspondence element-type pair supports
#
# The positional minimal kernels (homography_4pt, fundamental_7pt, p3p,
# essential_5pt) survive underneath as computational kernels; their
# Union{Nothing,…} returns are an internal detail that stops at this layer.
# The overdetermined one-step solvers are called by their algorithm names
# (`homography_dlt(cs)`, `fundamental_dlt(cs)`): there is no estimator verb.

# NOTE: CspondNormalization lives in homography_solvers.jl
# (they must be defined before the DLT kernels that take a `frame` kwarg).

# -----------------------------------------------------------------------------
# MinimalSolver — the protocol
# -----------------------------------------------------------------------------

"""
    MinimalSolver{M}

Abstract parent for exactly-determined minimal solvers producing models of
type `M`. Each algorithm is its own singleton subtype (two solvers may
share a model type but require different correspondence types — p3p vs.
circle relative pose both yield poses).

Interface (all required):
- `solve(s, cs, idx) -> FixedModels{max_solutions(s), M}` — ALWAYS a
  `FixedModels`; failure is `count == 0`, never `nothing`
- `sample_size(s)` — minimal sample cardinality
- `max_solutions(s)` — solution-count bound (the `N` in `FixedModels`)
- `model_type(s) == M`

Solvers with pluggable numeric backends carry a
`backend::AbstractSolverBackend` field (default `NativeSolver()`).
"""
abstract type MinimalSolver{M} end

"""
    model_type(s) -> Type

The model type produced by a [`MinimalSolver`](@ref).
"""
model_type(::MinimalSolver{M}) where {M} = M

"""
    sample_size(s::MinimalSolver) -> Int

Number of correspondences in the solver's minimal sample.
"""
function sample_size end

"""
    max_solutions(s::MinimalSolver) -> Int

Upper bound on the number of candidate models one minimal sample can
produce (the `N` of the returned `FixedModels{N,M}`).
"""
function max_solutions end

"""
    solve(solver::MinimalSolver, cs, idx) -> FixedModels

Run a minimal solver on the sample `idx` drawn from the correspondence
container `cs` (any container satisfying the [`sources`](@ref)/
[`targets`](@ref) protocol). ALWAYS returns `FixedModels`; a degenerate
sample yields `count == 0`. This is the single public solve protocol —
the positional kernels underneath are implementation detail.
"""
function solve end

# Scalar type of the correspondence coordinates (drives model eltype).
_coordtype(cs) =
    promote_type(eltype(eltype(sources(cs))), eltype(eltype(targets(cs))))

# Adapter giving backend methods with the legacy `(cs, idx)` duck-typed
# convention (`.first`/`.second`) a protocol-backed view.
struct _PairColumns{VS, VT}
    first::VS
    second::VT
end
_pair_columns(cs) = _PairColumns(sources(cs), targets(cs))

@inline _wrap1(::Type{M}, ::Nothing) where {M} = FixedModels{1, M}(0, (zero(M),))
@inline function _wrap1(::Type{M}, m) where {M}
    is_valid_solution(m) || return FixedModels{1, M}(0, (zero(M),))
    return FixedModels{1, M}(1, (M(Tuple(m)),))
end

# --- Homography4pt ------------------------------------------------------------

"""
    Homography4pt(backend = NativeSolver())

Minimal 4-point homography solver ([`solve`](@ref) verb over the
`homography_4pt` kernel). `sample_size` 4, `max_solutions` 1.
"""
struct Homography4pt{B <: AbstractSolverBackend} <: MinimalSolver{HomographyMat}
    backend::B
end
Homography4pt() = Homography4pt(NativeSolver())
sample_size(::Homography4pt) = 4
max_solutions(::Homography4pt) = 1

function solve(s::Homography4pt, cs, idx::AbstractVector{<:Integer})
    M = HomographyMat{_coordtype(cs)}
    return _wrap1(M, _h4(s.backend, cs, idx))
end

@inline function _h4(::NativeSolver, cs, idx)
    u = sources(cs); v = targets(cs)
    return @inbounds homography_4pt(
        u[idx[1]], u[idx[2]], u[idx[3]], u[idx[4]],
        v[idx[1]], v[idx[2]], v[idx[3]], v[idx[4]]
    )
end
_h4(b::AbstractSolverBackend, cs, idx) = homography_4pt(b, _pair_columns(cs), idx)

# --- Homography5ptRadial ------------------------------------------------------

"""
    Homography5ptRadial(; backend = NativeSolver(), extra_check = false)

Minimal 5-point radially distorted homography solver — one-sided
one-parameter division model on the TARGET side ([`solve`](@ref) verb over
the [`homography_5pt_radial`](@ref) kernel; [Wadenback2026](@citet)).
`sample_size` 5, `max_solutions` 3 (at most 1 with `extra_check`, which
keeps only the root most consistent with the redundant equation — the
reference enables it inside its RANSAC wrapper).

Deliberately NOT entered in the [`minimal_solvers`](@ref) registry: the
division coefficient `k` is only meaningful in roughly unit-scale target
coordinates, a convention generic pixel correspondences don't carry (cf.
the calibrated convention `Essential5pt` presumes). Opt in explicitly.
"""
struct Homography5ptRadial{B <: AbstractSolverBackend} <: MinimalSolver{RadialHomography}
    backend::B
    extra_check::Bool
    # No external backend ships a radial 5-point kernel: refuse at
    # construction, not inside the RANSAC loop.
    function Homography5ptRadial(backend::B, extra_check::Bool) where {B <: AbstractSolverBackend}
        B === NativeSolver || throw(ArgumentError("Homography5ptRadial has no $B kernel; use NativeSolver()"))
        return new{B}(backend, extra_check)
    end
end
Homography5ptRadial(; backend::AbstractSolverBackend = NativeSolver(), extra_check::Bool = false) =
    Homography5ptRadial(backend, extra_check)
sample_size(::Homography5ptRadial) = 5
max_solutions(::Homography5ptRadial) = 3

solve(s::Homography5ptRadial, cs, idx::AbstractVector{<:Integer}) =
    _h5r(s.backend, cs, idx, s.extra_check)

@inline function _h5r(::NativeSolver, cs, idx, extra_check::Bool)
    u = sources(cs); v = targets(cs)
    return @inbounds homography_5pt_radial(
        u[idx[1]], u[idx[2]], u[idx[3]], u[idx[4]], u[idx[5]],
        v[idx[1]], v[idx[2]], v[idx[3]], v[idx[4]], v[idx[5]];
        extra_check = extra_check
    )
end

# --- Fundamental7pt -----------------------------------------------------------

"""
    Fundamental7pt(backend = NativeSolver())

Minimal 7-point fundamental-matrix solver ([`solve`](@ref) verb over the
`fundamental_7pt` kernel). `sample_size` 7, `max_solutions` 3.
"""
struct Fundamental7pt{B <: AbstractSolverBackend} <: MinimalSolver{FundMat}
    backend::B
end
Fundamental7pt() = Fundamental7pt(NativeSolver())
sample_size(::Fundamental7pt) = 7
max_solutions(::Fundamental7pt) = 3

function solve(s::Fundamental7pt, cs, idx::AbstractVector{<:Integer})
    T = _coordtype(cs)
    M = FundMat{T}
    r = _f7(s.backend, cs, idx, T)
    r === nothing && return FixedModels{3, M}(0, ntuple(_ -> zero(M), Val(3)))
    return r
end

@inline function _f7(::NativeSolver, cs, idx, ::Type{T}) where {T}
    u = sources(cs); v = targets(cs)
    return @inbounds fundamental_7pt(
        u[idx[1]], u[idx[2]], u[idx[3]], u[idx[4]], u[idx[5]], u[idx[6]], u[idx[7]],
        v[idx[1]], v[idx[2]], v[idx[3]], v[idx[4]], v[idx[5]], v[idx[6]], v[idx[7]]
    )
end
_f7(b::AbstractSolverBackend, cs, idx, ::Type{T}) where {T} =
    fundamental_7pt(b, _pair_columns(cs), idx)

# --- Essential5pt -------------------------------------------------------------

"""
    Essential5pt(backend = NativeSolver())

Minimal 5-point essential-matrix solver (Nistér; [`solve`](@ref) verb over
the `essential_5pt` kernel). `sample_size` 5, `max_solutions` 10.

Correspondence endpoints are **bearing vectors** (`StaticVector{3}`), and only
those. A calibrated 2D point is a `MethodError`, not an input.

# Why a 2-vector is refused rather than lifted

Until 0.29 a calibrated 2-vector was accepted and lifted to `(x, y, 1)`. That
lift places the ray on the z = 1 plane, which **cannot represent a ray at or
past 90° from the optical axis at all** — so it silently excluded every
fisheye and omnidirectional camera, which are the cameras this stack exists
for. The failure was invisible: the lift always succeeded and returned a
plausible `EssentialMat`, computed from a chart that could not hold the data.

Refusing by TYPE is the point. A caller holding pixels builds bearings with
[`backproject`](@ref) — or [`backproject_with_jacobian`](@ref) when the pixel
Jacobian is wanted too, which is what keeps a residual threshold in pixels.

The kernel computes in `Float64`; returned models are
`EssentialMat{Float64}`. The epipolar constraint is scale-invariant per
endpoint, so the algebra does not require unit norm — but unit bearings are
the stack's convention and what `backproject` returns.

# What the type refusal does NOT catch

It catches a 2-vector. It cannot catch a 3-vector that is *still on the z = 1
plane* — `(x, y, 1)` is a perfectly good `StaticVector{3}`, and because the
epipolar algebra is scale-invariant per endpoint, such a point even produces
the right answer for any ray the chart can represent. The exclusion of rays at
and past 90° survives that lift unchanged; it has simply stopped being visible
in the type.

So this signature is a NECESSARY condition, not a guarantee, and it should not
be read as one. The guarantee has to come from where the vectors are BUILT —
`backproject(model, u)`, which is defined for every ray the camera can see —
not from where they are consumed. A caller that reaches a 3-vector by dividing
through by z has moved the same defect one layer up and past this gate.
"""
struct Essential5pt{B <: AbstractSolverBackend} <: MinimalSolver{EssentialMat}
    backend::B
end
Essential5pt() = Essential5pt(NativeSolver())
sample_size(::Essential5pt) = 5
max_solutions(::Essential5pt) = 10

# Bearings only. There is deliberately NO `StaticVector{2}` method: the z = 1
# lift it would perform is the affine-plane assumption the bearing form exists
# to remove (see the `Essential5pt` docstring), so a 2-vector endpoint raises a
# MethodError here instead of being silently promoted. Do not add one back.
@inline _bearing(p::StaticVector{3}) = SVector{3, Float64}(p)

function solve(s::Essential5pt, cs, idx::AbstractVector{<:Integer})
    u = sources(cs); v = targets(cs)
    x1 = SVector{5}(@inbounds ntuple(k -> _bearing(u[idx[k]]), Val(5)))
    x2 = SVector{5}(@inbounds ntuple(k -> _bearing(v[idx[k]]), Val(5)))
    return essential_5pt(s.backend, x1, x2)
end

# --- P3P ----------------------------------------------------------------------

"""
    P3P(backend = NativeSolver())

Perspective-3-point absolute-pose solver ([`solve`](@ref) verb over the
`p3p` kernel). `sample_size` 3, `max_solutions` 4.

Correspondence contract: sources are 3D world points, targets are **unit**
bearing vectors (pixel targets are lifted to bearings beforehand, e.g.
`lift(cs; target = u -> backproject(model, u))`). Models are `Extrinsics3` with
`x_cam = R·X + t`.
"""
struct P3P{B <: AbstractSolverBackend} <: MinimalSolver{Extrinsics3}
    backend::B
end
P3P() = P3P(NativeSolver())
sample_size(::P3P) = 3
max_solutions(::P3P) = 4

function solve(s::P3P, cs, idx::AbstractVector{<:Integer})
    u = sources(cs); v = targets(cs)
    X = SVector{3}(@inbounds ntuple(k -> SVector{3, Float64}(u[idx[k]]), Val(3)))
    x = SVector{3}(@inbounds ntuple(k -> SVector{3, Float64}(v[idx[k]]), Val(3)))
    return p3p(s.backend, x, X)
end

# -----------------------------------------------------------------------------
# Induced-model registry
# -----------------------------------------------------------------------------

"""
    minimal_solvers(::Type{S}, ::Type{T}) -> Tuple

The minimal solvers applicable to correspondences with source element type
`S` and target element type `T`. The registry is extended by adding a
method — registering a new solver family requires NO edits to the verb
layer or any problem struct.

Registered pairs:
- **2D ↔ 2D points**: `Homography4pt`, `Fundamental7pt`. Both are
  frame-covariant — a homography relates two planes, a fundamental matrix two
  images, in whatever 2D frame the caller holds — so here the element types
  really do determine that the solver applies.
- **3D ↔ 3D**: `P3P` and `Essential5pt`.

# The 3D pair is ambiguous, and the registry says so rather than choosing

`P3P` and `Essential5pt` are MUTUALLY EXCLUSIVE and **cannot be told apart by
element type**: P3P's source is a world POINT and its target a unit BEARING,
while both of `Essential5pt`'s endpoints are bearings. A bare
`StaticVector{3}` is all three of those things, and no dispatch can separate
them. What actually distinguishes them is the frame the caller built the data
in, which lives in the frame tags — not in the element type.

So the two queries answer differently, on purpose:

- [`minimal_solver`](@ref)`(M, S, T)` disambiguates by MODEL type and is
  exact: ask for `EssentialMat` and you get `Essential5pt`, ask for
  `Extrinsics3` and you get `P3P`. This is the lookup consumers use.
- [`induced_models`](@ref)`(S, T)` returns BOTH, because "what is estimable
  from a pair of 3-vectors" honestly has two answers and the types do not
  say which. It is not a claim that both apply to your data at once.

# Why `Essential5pt` is not at (2D, 2D) any more

Until 0.29 it was, carrying the note that it "presumes calibrated coordinates
— the element types cannot express that; it is the caller's frame
convention". That was a claim about the CALLER'S DATA written down as a
property of the code, and it was false in the way that matters: the lift it
relied on sent a calibrated 2-vector to the z = 1 plane, which cannot
represent a ray at or past 90°, silently excluding every fisheye and
omnidirectional camera. Registering it there also made `induced_models` OFFER
an essential matrix for plane data. The lift is deleted, the registration
moved, and a 2-vector endpoint is now a `MethodError` at the solver.

The ambiguity that remains at (3D, 3D) is of a different kind: under-determined
by the element type, but correct whenever the caller's data matches the
solver's stated contract. The old (2D, 2D) entry was not under-determined —
it was wrong.
"""
minimal_solvers(::Type, ::Type) = ()
minimal_solvers(::Type{<:StaticVector{2}}, ::Type{<:StaticVector{2}}) =
    (Homography4pt(), Fundamental7pt())
minimal_solvers(::Type{<:StaticVector{3}}, ::Type{<:StaticVector{3}}) =
    (P3P(), Essential5pt())

"""
    induced_models(S, T)  /  induced_models(cs) -> Tuple{Vararg{Type}}

The model types estimable from correspondences with element types `(S, T)`
(or from the container `cs` directly): `model_type.(minimal_solvers(S, T))`.

At `(3D, 3D)` this returns `(Extrinsics3, EssentialMat)`, which are MUTUALLY
EXCLUSIVE — a bare `StaticVector{3}` is both a world point and a bearing, and
no dispatch can tell them apart. That is reported rather than resolved here;
[`minimal_solver`](@ref) disambiguates by model type. See
[`minimal_solvers`](@ref) for why the registry is built this way.
"""
induced_models(::Type{S}, ::Type{T}) where {S, T} =
    map(model_type, minimal_solvers(S, T))
induced_models(cs::AbstractVector) =
    induced_models(eltype(sources(cs)), eltype(targets(cs)))

"""
    minimal_solver(::Type{M}, ::Type{S}, ::Type{T}) -> MinimalSolver{<:M}

Registry lookup: the minimal solver producing models of type `M` from
`(S, T)` correspondences. Errors if the registry has no such solver.
"""
function minimal_solver(::Type{M}, ::Type{S}, ::Type{T}) where {M, S, T}
    for s in minimal_solvers(S, T)
        model_type(s) <: M && return s
    end
    throw(
        ArgumentError(
            "no registered minimal solver for model $(M) on Cspond{$(S), $(T)} " *
                "(registered: $(join(string.(induced_models(S, T)), ", ")))"
        )
    )
end


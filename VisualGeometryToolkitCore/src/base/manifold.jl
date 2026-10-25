# =============================================================================
# Manifold — THE CONCEPT: local parameterizations (retractions) of a model
# manifold. This file is the contract; the concrete manifolds that implement
# it live one-per-file in geometry/manifolds/.
#
# The split follows GTSAM, which keeps the concept in gtsam/base/ (Manifold.h,
# Lie.h, VectorSpace.h) and the concrete types one-per-file in gtsam/geometry/;
# Ceres draws the same line (manifold.h vs sphere_manifold.h, line_manifold.h).
# It is also a real dependency boundary: the contract needs nothing but
# StaticArrays and pinv, so it loads with base/, while the concrete manifolds
# need EuclideanMap and the model matrix types and load with geometry/.
#
# A Manifold is a LOCAL COORDINATE SYSTEM anchored at a model iterate:
# `retract(ch, v)` evaluates the ambient model at local coordinates
# v ∈ R^D, D = tangent_size(ch) (v = 0 is the anchor), `recenter(ch, v)` re-anchors the manifold
# there. Engines use manifolds in two modes, both first-class:
#   • re-anchor-per-step (the essential GN/lift engines): every iteration
#     solves for a step Δ in the CURRENT manifold and re-centers;
#   • iterate-in-manifold (the Bartoli–Sturm F refit): the manifold stays anchored
#     at the seed and the optimizer iterates the coordinate vector.
#
# COVARIANCE TRANSPORT (the paper-3 requirement — first-order seed-covariance
# propagation in the same manifold the LO optimizes in):
#   `retract_jacobian(ch, v = 0)` — T(v) = ∂vec(retract(ch, v))/∂v, the
#   retraction differential at the anchor or at any coordinates an
#   iterate-in-manifold optimizer holds (computed by ForwardDiff through the
#   closed-form `retract`, so every manifold's transport is derivation-risk-free
#   and pinned against finite differences in the tests);
#   `transform_covariance(from, to, Σ)` moves a covariance between charts of one
#   model — the ambient space is a chart too (`EuclideanManifold(ch)`):
#   to it, T·Σv·Tᵀ (manifold → ambient); from it, T⁺·Σm·T⁺ᵀ (ambient →
#   manifold, via pinv — gauge-deficient manifolds project out their gauge).
#
# BIT-STABILITY: the `recenter` kernels that replace engine step code are
# @noinline ON PURPOSE — StaticArrays 3×3 matmul lowers through muladd
# (LLVM fmuladd, may-fuse) and inlining into different callers makes
# context-dependent fma choices (1–2 ulp pose drift between source-identical
# engines; measured during the LO-engine refactor). Compiled once,
# standalone, the arithmetic is context-independent and reproduces the
# historical inlined loops bit-for-bit (gated per consumer).
# =============================================================================

using ForwardDiff: ForwardDiff
using LinearAlgebra: pinv   # each manifold file imports what src/VisualGeometryToolkitCore.jl's `using LinearAlgebra` does not

"""
    Manifold

Local coordinate system anchored at a model iterate. Contract:
- `tangent_size(ch)::Int` — manifold dimension D;
- `model(ch)` — the ambient model at the anchor (= `retract(ch, 0)`);
- `retract(ch, v)` — ambient model at local coordinates `v::SVector{D}`;
- `recenter(ch, v)` — the same manifold family re-anchored at `retract(ch, v)`,
  with ONE normalization policy: a rotation part re-projected onto SO(3)
  (`_reproject_so3`), the homography chart's anchor its unit-Frobenius
  representative of the same model (a first-order retraction lets the scale
  drift); `retract` itself never normalizes. Its element type is
  `promote_type` of the manifold's and `v`'s, as arithmetic's is: a step of
  the manifold's own float type returns the manifold's own type (what a solver
  stepping a manifold variable holds), and a step of ForwardDiff Duals returns
  a Dual-valued manifold, so automatic differentiation through `recenter`
  works (NLLSsolver's, through `update`);
- `retract_jacobian(ch, v = 0)` — T(v) = ∂vec(retract(ch, v))/∂v (9 × D for
  3×3 models): at the anchor by default, at any `v` for an optimizer that
  iterates in the manifold instead of re-anchoring. Named after the function
  it differentiates, as every `<thing>_jacobian` is: it is Ceres'
  `Manifold::PlusJacobian` and GTSAM's derivative of `retract`. The inverse
  form, if one is ever needed, is `local_coordinates_jacobian` (Ceres'
  `MinusJacobian`);
- `action_jacobian(ch, x, v = 0)` — ∂(retract(ch, v)(x))/∂v, the manifold's
  action on a point or ray differentiated (at the anchor by default, at any `v`
  for an optimizer that iterates in the manifold): generically the chain rule of
  the model's own `transform_deriv_params` through T, closed form where a
  manifold has one;
- `local_coordinates(ch, target)` — the inverse retraction (closed form for
  the rigid and lens manifolds, Gauss–Newton for the projective ones);
- covariance transport through the retraction, first order:
  `transform_covariance(ch, EuclideanManifold(ch), Σv)` to the ambient space and
  back, or to another chart of the same model (see header).

NAMES, against the established libraries: `retract` and `local_coordinates`
are GTSAM's `retract` / `localCoordinates`, which are Ceres' `Manifold::Plus`
and `Manifold::Minus`. `action_jacobian` is the optional Jacobian GTSAM's
`transformFrom` fills. The two constraint verbs have NO counterpart in either
— Ceres and GTSAM both frame a problem as residuals of an explicit cost, while
`constraint_with_jacobians` and `constraint_jacobians` are the IMPLICIT
constraint `g(θ, x) = 0` with Förstner's `A = ∂g/∂θ` and `B = ∂g/∂x`, which is
what an implicit-function covariance and a Gauss-Helmert step need.

A manifold is the MODEL's (`minimal_manifold(model)`), and the free/fixed choice a
fitter makes is a manifold operation ([`restrict`](@ref)) — so a residual
composed with a manifold is differentiated once, here, not per consumer.
"""
abstract type Manifold end

function tangent_size end
function retract end
function recenter end

# `recenter`'s element-type rule: the manifold's element type `S` promoted with
# the step's (`promote_type(S, eltype(v))`), and a static storage type `A` rebuilt
# in it (a `RotMatrix` keeps its type: its method is beside the rigid charts).
_storage_with_eltype(::Type{A}, ::Type{T}) where {A <: StaticArray, T} = similar_type(A, T)

"""
    retract_jacobian(ch::Manifold) -> SMatrix
    retract_jacobian(ch::Manifold, v::StaticVector) -> SMatrix

The RETRACTION DIFFERENTIAL: `∂vec(retract(ch, v))/∂v`, the ambient model's
derivative with respect to the chart coordinates — at the anchor (`v = 0`) in
the one-argument form, at any `v` for an optimizer that iterates in the
manifold instead of re-anchoring.

Named after the function it differentiates, as every `<thing>_jacobian` is.
Against the established libraries it is Ceres' `Manifold::PlusJacobian` and
GTSAM's derivative of `retract`; the inverse form, if one is ever needed, is
`local_coordinates_jacobian` (Ceres' `MinusJacobian`).

This is the ∂ambient/∂v factor of [`action_jacobian`](@ref), which chains it
with the model's own `transform_deriv_params`. The default for a chart that
does not override it is `_retract_jacobian_fd`, ForwardDiff through `retract`
— exact to first order with no hand-derivation risk, and every closed-form
override is pinned against it in the manifold contract tests. A linear chart
(`LensManifold`) returns the identity. The default needs only `retract` and the
model's `_ambient_coordinates`.

Public, not exported; VisualGeometryToolkitEstimation builds its LM and covariance steps
on it.
"""
retract_jacobian(ch::Manifold, v::StaticVector = zero_tangent(ch)) = _retract_jacobian_fd(ch, v)

function local_coordinates end

"""
    zero_tangent(ch::Manifold) -> SVector{tangent_size(ch)}

The chart coordinates of the anchor, `v = 0`, in the chart's float type
(`Float64` unless the chart stores another): what [`model`](@ref)`(ch)` and the
one-argument [`retract_jacobian`](@ref) evaluate at. A chart generic in its
float type overrides it.
"""
zero_tangent(ch::Manifold) = zero(SVector{tangent_size(ch), Float64})

"""
    model(ch::Manifold)

The ambient model at the chart's anchor: `retract(ch, zero_tangent(ch))`. A
chart that stores its anchor model overrides it with a read.
"""
model(ch::Manifold) = retract(ch, zero_tangent(ch))

"""
    action_jacobian(ch::Manifold, x) -> SMatrix
    action_jacobian(ch::Manifold, x, v) -> SMatrix

The Jacobian of the manifold's ACTION on `x` — `∂(retract(ch, v)(x))/∂v` at
`v = 0`, or at the chart coordinates `v` — what every residual through a posed
or parameterized model needs. Generic: the chain rule of the model's own
parameter derivative (`transform_deriv_params(retract(ch, v), x)`,
∂model(x)/∂ambient) through the retraction differential
(`retract_jacobian(ch, v)`, ∂ambient/∂v). A manifold with a closed form
overrides it (`RigidManifold6`: `[−4·skew(R·x)  I₃]` at the anchor); every
override is pinned against this default in the manifold tests.
"""
action_jacobian(ch::Manifold, x) = transform_deriv_params(model(ch), x) * retract_jacobian(ch)
action_jacobian(ch::Manifold, x, v::StaticVector) = transform_deriv_params(retract(ch, v), x) * retract_jacobian(ch, v)

"""
    retract_directions(ch::Manifold, v[, Val(Q)]) -> NTuple{Q, SMatrix{3, 3}}

The first `Q` (default: all) columns of [`retract_jacobian`](@ref)`(ch, v)` —
`∂M/∂vq` for a chart whose ambient model is a 3×3 matrix (its first nine
ambient coordinates, column-major) — each reshaped to a 3×3 matrix: the `dM`
tuple the row Jacobians take (`sampson_jacobian_rows!`,
`transfer_jacobian_rows!`, `radial_transfer_jacobian_rows!`). For a radial
homography's `[vec H; k]` chart, `Val(8)` gives the eight sl(3) directions of
`H`. In the Jacobian's element type.
"""
@inline function retract_directions(ch::Manifold, v::StaticVector{D}, ::Val{Q} = Val(D)) where {D, Q}
    A = retract_jacobian(ch, v)
    return ntuple(q -> SMatrix{3, 3, eltype(A), 9}(A[SVector{9}(1:9), q]), Val(Q))
end

# Generic ForwardDiff retraction differential at local coordinates `v`, through
# the model's ambient coordinates: exact first-order transport with zero
# hand-derivation risk (pinned against central differences in the manifold
# contract tests). The default of `retract_jacobian`.
_retract_jacobian_fd(ch::Manifold, v::StaticVector) =
    ForwardDiff.jacobian(w -> _ambient_coordinates(retract(ch, w)), v)

"""
    transform_covariance(from::Manifold, to::Manifold, Σ) -> Matrix

A covariance in `from`'s coordinates re-expressed in `to`'s, the two charts
anchored at the same model and sharing its ambient vectorization, to first order:
through the ambient space, `T_to⁺·(T_from·Σ·T_fromᵀ)·T_to⁺ᵀ` (`T` the charts'
`retract_jacobian`s). The ambient space is a chart as well,
`EuclideanManifold(ch)`: TO it the covariance is pushed, `T·Σ·Tᵀ`; FROM
it pulled, `T⁺·Σ·T⁺ᵀ` (a gauge-deficient chart projects its gauge out). Exact
where `to` is full rank. A rigid pose's covariance between the MRP
(`RigidManifold6`), left-perturbation (`LeftRigidManifold6`) and decoupled
(`RigidExpManifold6`) charts, for instance — the MRP's factor 4 at the anchor and
the translation coupling done by the charts themselves, not by hand. A change of
coordinates at one model, not a parallel transport between points.

GTSAM equivalent: `TransformCovariance<T>` (a group-frame change), and the
`Local`/`Retract` Jacobians a chart change composes.
"""
transform_covariance(from::Manifold, to::Manifold, Σ::AbstractMatrix) =
    propagate(pinv(retract_jacobian(to)), propagate(retract_jacobian(from), Σ))

# The ambient coordinates of a model — the vector its charts' `retract_jacobian`
# differentiates, the anchor of its ambient chart `EuclideanManifold(ch)`. Every
# model a chart retracts to adds its method beside its chart.
_ambient_coordinates(M::StaticMatrix) = SVector{length(M)}(Tuple(M))
_ambient_coordinates(x::StaticVector) = x
_ambient_coordinates(parts::Tuple) = vcat(map(_ambient_coordinates, parts)...)

"""
    restrict(ch::Manifold, free::SVector{D, Bool}) -> SubsetManifold
    restrict(ch::Manifold, idx::SVector{M, Int})

The manifold `ch` on the coordinates `free` marks (or the ascending, distinct
indices `idx`): `tangent_size` is the number of free coordinates, and `retract`,
`recenter`, `retract_jacobian` and `action_jacobian` act through the parent
with the fixed coordinates at zero. `restrict(ch, ())` — no coordinate — is
the pinned model. See [`SubsetManifold`](@ref); a lens manifold also restricts
by coefficient name.
"""
function restrict end

"""
    tangent_part(ch::Manifold, v, offset = 0) -> SVector{tangent_size(ch)}

The coordinates of `ch` in a longer tangent vector `v`: the `tangent_size(ch)`
entries after the first `offset`, statically sized and in `v`'s element type
(ForwardDiff Duals included), since a manifold's `tangent_size` is a constant of
its type. A product manifold cuts its parts' coordinates with it, and so does a
solver that stacks several manifolds' coordinates in one step vector.
"""
tangent_part(ch::Manifold, v, offset = 0) =
    SVector{tangent_size(ch), eltype(v)}(ntuple(j -> v[offset + j], Val(tangent_size(ch))))

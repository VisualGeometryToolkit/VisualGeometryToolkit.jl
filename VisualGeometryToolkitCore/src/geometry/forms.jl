# =============================================================================
# Slot-labeled epipolar forms — the forms layer (0.8)
# =============================================================================
#
# Maps have ENDPOINTS; forms have SLOTS; rank lives on the manifold.
#
# A `HomographyMap{Src, Dst}` is a collineation — it consumes `Src`-frame
# points and produces `Dst`-frame points, so its frame tags name the two
# endpoints of a map. The epipolar matrix is a different citizen: a bilinear
# form with two contravariant point arguments, `x₂ᵀ M x₁ = 0`. It maps
# neither argument anywhere — it CONSTRAINS the pair — so its frame tags
# label the form's two argument SLOTS ("slot" in the MTW sense, used as a
# mnemonic throughout this file: the tag names which frame's points an
# argument position accepts, not an endpoint of a map).
#
# The associated point ↦ line action is a CORRELATION (point to hyperplane;
# contrast "collineation", point to point), and a rank-2 one — hence one-way.
# See `epipolar_map` and the deliberate absence of `inv` below.
#
# Layering: `FundMat`/`EssentialMat` stay frame-free algebra — rank and
# manifold constraints live there, not here. `EpipolarForm` adds ONLY the
# slot labels, declared at the Mat → form lift, exactly as `HomographyMat` →
# `HomographyMap` declares endpoints at the Mat → Map lift.

"""
    EpipolarForm{S1<:Frame{2}, S2<:Frame{2}, M<:AbstractFundMat}

Slot-labeled epipolar form: a bilinear form with two contravariant point
arguments, wrapping a frame-free [`FundMat`](@ref) or [`EssentialMat`](@ref).

**Slot convention (pinned here; every other method defers to this):** slot 1
takes points of frame `S1` (view 1), slot 2 takes points of frame `S2`
(view 2), and the wrapped matrix `M` satisfies

    x₂ᵀ M x₁ = 0

for corresponding `x₁ ∈ S1`, `x₂ ∈ S2` (homogeneous). Slot 1 is the RIGHT
argument of the matrix, matching the `u₂ᵀ F u₁` convention used throughout
the solver layer. In slot-monogram terms the form is `M_{S2,S1}` — first
index left slot, second index right slot.

The frame tags label the slots of a form, not the endpoints of a map: an
`EpipolarForm` is not a `Transformation`, does not compose with `∘`, and has
**no `inv`** (deliberately — see the note below). Its point ↦ line action is
[`epipolar_map`](@ref), and a frame change is composition on that map:
`EpipolarForm(H₂ ∘ epipolar_map(f) ∘ inv(H₁))` (see `∘` on
[`EpipolarMap`](@ref)).

Construction is the labeled Mat → form lift; unlabeled lifts are allowed
for gradual adoption (both slots `UnlabeledFrame{2}`):

```julia
form = EpipolarForm{ImageFrame{:a}, ImageFrame{:b}}(F)   # labeled lift
form = EpipolarForm(F)                                 # unlabeled lift
```

Evaluating the form applies the slot convention operationally:
`form(x₁, x₂)` returns the scalar `x̃₂ᵀ M x̃₁` (zero for an exact
correspondence). The wrapper does NOT enforce the rank-2 (or E-manifold)
constraint — that discipline belongs to the wrapped Mat type and the
solvers that produce it.

# Why there is no `inv`

The epipolar correlation is rank 2: its kernel is the epipole `e₁`
(`M e₁ = 0`), every point on one epipolar line maps to the SAME line in the
other view, and the image is not all lines but the 1-parameter pencil
through the other epipole `e₂`. A 2-parameter point domain collapsing onto a
1-parameter family of lines has no inverse — `inv(::EpipolarForm)` is
deliberately left undefined (a `MethodError`), and [`transpose`](@ref
transpose(::EpipolarForm)) (swap the slots) is the only involution the form
supports. Use [`epipoles`](@ref) to make the degeneracy explicit.

See also [`epipolar_map`](@ref), [`EpipolarMap`](@ref), [`epipoles`](@ref),
[`sampson_distance`](@ref), [`ImageFrame`](@ref).
"""
struct EpipolarForm{S1 <: Frame{2}, S2 <: Frame{2}, M <: AbstractFundMat}
    mat::M
end

# Unlabeled lift (gradual-typing default, as for the Maps).
EpipolarForm(mat::AbstractFundMat) =
    EpipolarForm{UnlabeledFrame{2}, UnlabeledFrame{2}, typeof(mat)}(mat)

# Labeled Mat → form lift.
(::Type{EpipolarForm{S1, S2}})(mat::AbstractFundMat) where {S1 <: Frame{2}, S2 <: Frame{2}} =
    EpipolarForm{S1, S2, typeof(mat)}(mat)

EpipolarForm(f::EpipolarForm) = f

Base.parent(f::EpipolarForm) = f.mat

# Two-slot evaluation — the slot convention of the struct docstring, made
# executable: slot 1 (frame S1) is the right argument of the matrix.
"""
    (f::EpipolarForm)(x₁, x₂) -> scalar

Evaluate the bilinear form on a point per slot: `x̃₂ᵀ M x̃₁` with
`x̃ = [x; 1]`. Zero (up to noise) iff `(x₁, x₂)` satisfy the epipolar
constraint. `x₁` feeds slot 1 (frame `S1`), `x₂` slot 2 (frame `S2`) — the
convention pinned at [`EpipolarForm`](@ref).
"""
@inline function (f::EpipolarForm)(x₁::StaticVector{2}, x₂::StaticVector{2})
    return dot(to_affine(x₂), f.mat * to_affine(x₁))
end

# Frame labeling escape hatches — the EpipolarForm methods of the house
# relabel/unlabel verbs. For a form the two tags are SLOTS, not endpoints;
# semantics are otherwise identical: deliberate, unchecked, zero-cost.
@inline relabel(
    ::Type{S1}, ::Type{S2}, f::EpipolarForm
) where {S1 <: Frame{2}, S2 <: Frame{2}} = EpipolarForm{S1, S2, typeof(f.mat)}(f.mat)

@inline unlabel(f::EpipolarForm) = relabel(UnlabeledFrame{2}, UnlabeledFrame{2}, f)

"""
    transpose(f::EpipolarForm{S1, S2}) -> EpipolarForm{S2, S1}

Swap the slots of the form: wraps `Mᵀ`, so the constraint reads
`x₁ᵀ Mᵀ x₂ = 0` — the same incidence relation with the argument order
exchanged. An involution (`transpose(transpose(f))` recovers `f`), and the
ONLY inversion-like operation the form supports — the rank-2 correlation has
no `inv` (see [`EpipolarForm`](@ref)). The wrapped Mat type is preserved:
the transpose of a fundamental matrix is fundamental, and `Eᵀ` remains on
the essential manifold.

`adjoint` (`f'`) is the same operation (real coefficients).
"""
function Base.transpose(f::EpipolarForm{S1, S2, M}) where {S1, S2, M}
    T = eltype(f.mat)
    mt = M(Tuple(SMatrix{3, 3, T, 9}(transpose(SMatrix{3, 3, T, 9}(f.mat)))))
    return EpipolarForm{S2, S1, M}(mt)
end

Base.adjoint(f::EpipolarForm) = transpose(f)

# NOTE: no `Base.inv(::EpipolarForm)` method — deliberate; see the
# struct docstring. The absence is pinned by a MethodError test.

# -----------------------------------------------------------------------------
# epipolar_map — the honest point ↦ line correlation
# -----------------------------------------------------------------------------

"""
    EpipolarMap{S1, S2, M}

Typed callable returned by [`epipolar_map`](@ref): the rank-2 correlation of
an [`EpipolarForm`](@ref), taking a point in frame `S1` to its epipolar line
in frame `S2`; `EpipolarMap(F)` / `EpipolarMap{S1, S2}(F)` lift a Mat directly.
Not a `Transformation` — it has no inverse (see the rank-2 note at
`EpipolarForm`) — but it composes with collineations on either side (`∘`
below), which is how it changes frame.
"""
struct EpipolarMap{S1 <: Frame{2}, S2 <: Frame{2}, M <: AbstractFundMat}
    mat::M
end

"""
    epipolar_map(f::EpipolarForm{S1, S2}) -> EpipolarMap{S1, S2}

The honest epipolar correlation: a callable taking a point `x₁` in slot 1's
frame `S1` and returning its epipolar LINE in slot 2's frame `S2`,

    ℓ₂ = M x̃₁

as a `Line2D` — VGC's 2D `Hyperplane`, i.e. the covector of the incidence
constraint `ℓ₂ᵀ x̃₂ = 0`. The returned line is itself a one-slot form (a
linear form on `S2`-frame points), which is why a correlation lands on
`Line2D` and not on a point type: point ↦ line, not point ↦ point.

`epipolar_map(transpose(f))` goes the other way (`S2` points to `S1`
lines, via `Mᵀ`). There is no inverse map — the correlation is rank 2 (see
[`EpipolarForm`](@ref)); the two degenerate directions are exposed by
[`epipoles`](@ref).

!!! note
    Since 0.8 this name refers ONLY to the form action. The former
    `epipolar_map(source::PosedCamera, target::PosedCamera, extrinsics)` — a
    unit-depth point-TRANSFER closure, not an epipolar correlation — is
    renamed [`depth_transfer`](@ref).
"""
epipolar_map(f::EpipolarForm{S1, S2, M}) where {S1, S2, M} = EpipolarMap{S1, S2, M}(f.mat)

@inline function (em::EpipolarMap)(x::StaticVector{2})
    l = em.mat * to_affine(x)
    return Line2D{eltype(l)}(SVector{3}(l))
end

# The Mat → map lift, unlabeled and labeled (as for the forms and the Maps), and
# the map back to its form: the same matrix, the same two frames.
EpipolarMap(mat::AbstractFundMat) = EpipolarMap{UnlabeledFrame{2}, UnlabeledFrame{2}, typeof(mat)}(mat)
(::Type{EpipolarMap{S1, S2}})(mat::AbstractFundMat) where {S1 <: Frame{2}, S2 <: Frame{2}} =
    EpipolarMap{S1, S2, typeof(mat)}(mat)
EpipolarForm(em::EpipolarMap{S1, S2, M}) where {S1, S2, M} = EpipolarForm{S1, S2, M}(em.mat)
Base.parent(em::EpipolarMap) = em.mat

# -----------------------------------------------------------------------------
# Composition — the correlation's frame change is `∘`
# -----------------------------------------------------------------------------

"""
    h ∘ em          (h::HomographyMap{3}, em::EpipolarMap) -> EpipolarMap
    em ∘ h          (em::EpipolarMap, h::HomographyMap{3}) -> EpipolarMap

A frame change of the epipolar correlation is composition, as for every map.
On the RIGHT a collineation moves the points fed in: `em ∘ h` is
`x ↦ em(h(x))`, the matrix `M·H`. On the LEFT it moves the returned line by
the line law `ℓ′ = H⁻ᵀ ℓ` (the [`Hyperplane`](@ref) action of a
`HomographyMap`): `h ∘ em` is the matrix `H⁻ᵀ·M`. Together, with `Hᵢ` moving
slot `i`'s points into their new frame,

    H₂ ∘ em ∘ inv(H₁)   has   M′ = H₂⁻ᵀ · M · H₁⁻¹,

the form's transformation law (`x₂′ᵀ M′ x₁′ = x₂ᵀ M x₁`). With
`Hᵢ = HomographyMap{GnomonicFrame{i}, ImageFrame{i}}(Kᵢ)` (the calibrated →
pixel collineations) `H₂ ∘ em ∘ inv(H₁)` is exactly `F = K₂⁻ᵀ E K₁⁻¹`, and a
correspondence normalization moves a model by
`destination_map(N) ∘ em ∘ inv(source_map(N))` (and back by
`inv(destination_map(N)) ∘ em ∘ source_map(N)`). Each junction is frame-checked
under the standing wildcard truth table (see `Base.:∘` at `EuclideanMap`): the
correlation's slot-1 frame meets the right map's `Dst`, its slot-2 frame the
left map's `Src`; a labeled mismatch is a `MethodError`. A CoordinateTransformations
affine (`AffineMap`, `LinearMap`, `Translation`) composes as its `HomographyMap`.

The result always wraps a [`FundMat`](@ref): an arbitrary collineation
preserves rank 2 but NOT the essential-manifold constraint (σ₁ = σ₂), so an
`EssentialMat` wrapper kept through a general frame change would be a silent
lie. Rewrap via `EssentialMat(F, K₁, K₂)` at a trusted boundary if the
destination frames are calibrated. [`EpipolarForm`](@ref)`(em)` is the form
of the result.
"""
function Base.:∘(h::HomographyMap{3, T, M, Src, Dst}, em::EpipolarMap{S1, S2}) where {T, M, Src, Dst, S1, S2}
    _check_junction(Src, S2, ∘, (h, em))
    Te = eltype(em.mat)
    F = transpose(SMatrix{3, 3, T, 9}(h.inverse)) * SMatrix{3, 3, Te, 9}(em.mat)
    return EpipolarMap{S1, Dst}(FundMat{eltype(F)}(Tuple(F)))
end
function Base.:∘(em::EpipolarMap{S1, S2}, h::HomographyMap{3, T, M, Src, Dst}) where {S1, S2, T, M, Src, Dst}
    _check_junction(S1, Dst, ∘, (em, h))
    Te = eltype(em.mat)
    F = SMatrix{3, 3, Te, 9}(em.mat) * SMatrix{3, 3, T, 9}(h.forward)
    return EpipolarMap{Src, S2}(FundMat{eltype(F)}(Tuple(F)))
end
Base.:∘(a::_CTLeafTransformation, em::EpipolarMap) = HomographyMap(a) ∘ em
Base.:∘(em::EpipolarMap, a::_CTLeafTransformation) = em ∘ HomographyMap(a)

"""
    a ∘ H ∘ b       (H::HomographyMat)     -> HomographyMat
    a ∘ F ∘ b       (F::AbstractFundMat)   -> FundMat

A frame change of a model MATRIX is composition with collineations `a`, `b`
(a `HomographyMap{3}` or a CoordinateTransformations affine), returning the
model type, as `∘` on a [`RadialHomography`](@ref) or a `RadialCamera` does:

- collineation, `H′ = A·H·B`;
- correlation, `F′ = A⁻ᵀ·F·B` (the laws of `∘` on a `HomographyMap` and on an
  [`EpipolarMap`](@ref); `A⁻¹` is `a`'s stored or analytic inverse, never a
  generic 3×3 inversion).

A matrix carries no frames: it is the unlabeled wildcard at both junctions,
and the maps keep their own labels. The epipolar result is always a
[`FundMat`](@ref) — a general collineation keeps rank 2 but not the essential
constraint (see `∘` on an `EpipolarMap`). A normalization's two sides move a
model into its normalized frame by `destination_map(N) ∘ m ∘ inv(source_map(N))`
and back by `inv(destination_map(N)) ∘ m ∘ source_map(N)`. The composition is exact algebra only: the
caller's policy (the `sign_normalize` gauge, `enforce_rank_two`, the element
type) is a separate call. A matrix composed with a matrix, or with anything
that is not a collineation, is an `ArgumentError` naming the lift
(`HomographyMap(H)`, `EpipolarMap(F)`).
"""
function Base.:∘(a::Union{HomographyMap{3}, _CTLeafTransformation}, H::HomographyMat)
    H′ = SMatrix{3, 3}(to_homogeneous(a)) * SMatrix{3, 3}(H)
    return HomographyMat{eltype(H′)}(Tuple(H′))
end
function Base.:∘(H::HomographyMat, b::Union{HomographyMap{3}, _CTLeafTransformation})
    H′ = SMatrix{3, 3}(H) * SMatrix{3, 3}(to_homogeneous(b))
    return HomographyMat{eltype(H′)}(Tuple(H′))
end
function Base.:∘(a::Union{HomographyMap{3}, _CTLeafTransformation}, F::AbstractFundMat)
    F′ = transpose(SMatrix{3, 3}(to_homogeneous(inv(a)))) * SMatrix{3, 3}(F)
    return FundMat{eltype(F′)}(Tuple(F′))
end
function Base.:∘(F::AbstractFundMat, b::Union{HomographyMap{3}, _CTLeafTransformation})
    F′ = SMatrix{3, 3}(F) * SMatrix{3, 3}(to_homogeneous(b))
    return FundMat{eltype(F′)}(Tuple(F′))
end

# Anything else meeting a bare model matrix would fall to Base's generic `∘` and
# build a lazy `ComposedFunction` whose failure surfaces far from here: an
# `ArgumentError` naming the lift instead.
const _ModelMat = Union{HomographyMat, AbstractFundMat}
_lift_name(::HomographyMat) = "HomographyMap(H)"
_lift_name(::AbstractFundMat) = "EpipolarMap(F)"
@noinline _unlifted(m) = throw(
    ArgumentError(
        "∘ with a bare $(nameof(typeof(m))): a model matrix composes only with a collineation " *
            "(a HomographyMap or an affine); lift it with $(_lift_name(m)) to compose it as a map"
    )
)
Base.:∘(m::_ModelMat, b) = _unlifted(m)
Base.:∘(a, m::_ModelMat) = _unlifted(m)
Base.:∘(m::_ModelMat, ::_ModelMat) = _unlifted(m)

"""
    compose_jacobian(outer, m, inner) -> SMatrix{9, 9}

The derivative of the composition `outer ∘ m ∘ inner` in the model's matrix,
`∂vec(M′)/∂vec(M)` (column-major `vec`), for `m` an [`EpipolarMap`](@ref) or a
`HomographyMap{3}` and `outer`/`inner` collineations (a `HomographyMap{3}` or a
CoordinateTransformations affine). The composition is linear in `M`, so this is
the constant Kronecker factor of its law (`_conjugation_jacobian`, the one
`∂vec(A·M·B)/∂vec(M)`), `B` = `inner`'s matrix and `A` = `outer`'s:

- correlation, `M′ = A⁻ᵀ·M·B`: `kron(Bᵀ, A⁻ᵀ)`;
- collineation, `H′ = A·H·B`: `kron(Bᵀ, A)`.

What a constraint evaluated on the composed model needs to pull its model
gradient back into the model's own coordinates: `∂g/∂vec(M) = Jᵀ·∂g/∂vec(M′)`
for a scalar constraint's 9-vector, `G_θ·J` for a 2×9. Frame-checked like `∘`;
in the promoted element type of the model and the maps.
"""
compose_jacobian(outer, m, inner) = _compose_jacobian(HomographyMap(outer), m, HomographyMap(inner))
function _compose_jacobian(
        A::HomographyMap{3, TA, MA, SrcA, DstA}, em::EpipolarMap{S1, S2}, B::HomographyMap{3, TB, MB, SrcB, DstB}
    ) where {TA, MA, SrcA, DstA, S1, S2, TB, MB, SrcB, DstB}
    _check_junction(SrcA, S2, ∘, (A, em))
    _check_junction(S1, DstB, ∘, (em, B))
    T = float(promote_type(eltype(em.mat), TA, TB))
    return _conjugation_jacobian(transpose(SMatrix{3, 3, T, 9}(A.inverse)), SMatrix{3, 3, T, 9}(B.forward))
end
function _compose_jacobian(
        A::HomographyMap{3, TA, MA, SrcA, DstA}, h::HomographyMap{3, T0, M0, Src, Dst}, B::HomographyMap{3, TB, MB, SrcB, DstB}
    ) where {TA, MA, SrcA, DstA, T0, M0, Src, Dst, TB, MB, SrcB, DstB}
    _check_junction(SrcA, Dst, ∘, (A, h))
    _check_junction(Src, DstB, ∘, (h, B))
    T = float(promote_type(T0, TA, TB))
    return _conjugation_jacobian(SMatrix{3, 3, T, 9}(A.forward), SMatrix{3, 3, T, 9}(B.forward))
end

# -----------------------------------------------------------------------------
# epipoles — the degeneracy, made discoverable
# -----------------------------------------------------------------------------

"""
    epipoles(f::EpipolarForm{S1, S2}) -> (e₁, e₂)

The two kernels of the rank-2 form, labeled by their frames: `e₁` is the
epipole in slot 1's frame `S1` (right kernel, `M e₁ ≈ 0`) and `e₂` the
epipole in slot 2's frame `S2` (left kernel, `Mᵀ e₂ ≈ 0`), returned as a
NamedTuple `(e₁ = ..., e₂ = ...)` of homogeneous `SVector{3}` (unit norm,
from the SVD null vectors — homogeneous because an epipole can lie at
infinity; dehomogenize with `to_euclidean` when `e[3] ≠ 0`).

These are the degenerate directions that make the correlation one-way:
[`epipolar_map`](@ref) sends `e₁` to the zero "line", and every returned
line goes through `e₂` (the pencil property). Computed by SVD of the
wrapped matrix; for a noisy or rank-3 matrix the returned vectors are the
nearest-kernel directions.
"""
function epipoles(f::EpipolarForm)
    S = svd(SMatrix{3, 3, Float64, 9}(f.mat))
    e₁ = SVector{3, Float64}(S.V[1, 3], S.V[2, 3], S.V[3, 3])
    e₂ = SVector{3, Float64}(S.U[1, 3], S.U[2, 3], S.U[3, 3])
    return (e₁ = e₁, e₂ = e₂)
end

# -----------------------------------------------------------------------------
# sampson_distance — checked overloads delegating to the Mat machinery
# -----------------------------------------------------------------------------

# Measurement-frame accessor for the checked delegation below: an `Uncertain`
# whose statistical form is framed (0.9) exposes its frame; plain points and
# unframed measurements are the unlabeled wildcard.
@inline _meas_frame(::Any) = UnlabeledFrame{2}
@inline _meas_frame(u::Uncertain{V, <:Union{CovMat{2}, PrecMat{2}}}) where {V} =
    _frame(getfield(u, :attributes))


"""
    sampson_distance(f::EpipolarForm, x₁, x₂)
    sampson_distance(f::EpipolarForm, c::Pair)

Form-level Sampson distance: delegates to the [`sampson_distance`](@ref)
machinery on the wrapped Mat, with the slot convention of
[`EpipolarForm`](@ref) — `x₁` feeds slot 1 (frame `S1`), `x₂` slot 2
(frame `S2`). Accepts everything the Mat surface accepts (plain points or
`Uncertain` measurements).

Since 0.9 the delegation is CHECKED where a frame is available: an
`Uncertain` measurement whose `CovMat`/`PrecMat` carries a frame label must
meet the slot it feeds (standing wildcard truth table — a labeled mismatch
is a `MethodError`; plain points and unlabeled matrices are the wildcard).
"""
@inline function sampson_distance(f::EpipolarForm{S1, S2}, x₁, x₂) where {S1, S2}
    (_junction_ok(S1, _meas_frame(x₁)) && _junction_ok(S2, _meas_frame(x₂))) ||
        throw(MethodError(sampson_distance, (f, x₁, x₂)))
    return sampson_distance(f.mat, x₁, x₂)
end
@inline sampson_distance(f::EpipolarForm, c::Pair) = sampson_distance(f, c.first, c.second)

# NOTE: no `transfer` overload for EpipolarForm — `transfer` is the
# point ↦ point depth-transport verb (camera models + depth). A correlation
# returns a LINE; forcing it under `transfer` would blur exactly the
# maps-vs-forms distinction this file exists to draw.

# =============================================================================
# The transfer form of the homography constraint
# =============================================================================
#
# Unlike `EpipolarForm`, whose tags label SLOTS, a `TransferForm`'s tags are
# genuine ENDPOINTS: the transfer constraint g = π(H·ũ₁) − u₂ measures a
# residual in the DESTINATION frame, so `Src` is the frame `u₁` is given in
# and `Dst` the frame `u₂` and the residual live in. That is why it wears the
# map's `{Src, Dst}` vocabulary rather than the form's slot vocabulary, even
# though it is spelled as a form: it exists so the transfer constraint is a
# METHOD of `constraint_with_jacobians` rather than a second name for the
# same question.

"""
    TransferForm{Src, Dst}(H::HomographyMat)
    TransferForm(H::HomographyMat)

The TRANSFER form of a homography constraint: the residual is measured in the
image, `g = π(H·ũ₁) − u₂`, as against the ALGEBRAIC (cross-multiplied) form
that `constraint_with_jacobians(H, u₁, u₂)` returns and the DLT minimizes.

The two tags are ENDPOINTS, not slots (see the note above): `Src` is `u₁`'s
frame, `Dst` is the frame of `u₂` and of the residual. The bare constructor
lifts to `UnlabeledFrame{2}` endpoints, the gradual-typing default used by the
Maps and by [`EpipolarForm`](@ref).

Public, not exported.
"""
struct TransferForm{Src <: Frame{2}, Dst <: Frame{2}, M <: HomographyMat}
    mat::M
end

# Unlabeled lift (gradual-typing default, as for the Maps and EpipolarForm).
TransferForm(mat::HomographyMat) =
    TransferForm{UnlabeledFrame{2}, UnlabeledFrame{2}, typeof(mat)}(mat)

# Labeled Mat → form lift.
(::Type{TransferForm{Src, Dst}})(mat::HomographyMat) where {Src <: Frame{2}, Dst <: Frame{2}} =
    TransferForm{Src, Dst, typeof(mat)}(mat)

TransferForm(f::TransferForm) = f

Base.parent(f::TransferForm) = f.mat

"""
    constraint_with_jacobians(f::TransferForm, u₁, u₂) -> (g, G_x, G_θ)

The transfer constraint of the wrapped homography and its Jacobians:
`g = π(H·ũ₁) − u₂`, the transfer error in the destination image, with
`G_x = [Jπ·H₁₂  −I₂]` (2×4, over `(x₁, y₁, x₂, y₂)`) and
`G_θ = Jπ·(ũ₁ᵀ ⊗ I₃)` (2×9, ∂g/∂vec(H) column-major), `Jπ` the perspective
derivative at `H·ũ₁` and `H₁₂` the first two columns of `H`.

The ALGEBRAIC counterpart is `constraint_with_jacobians(H, u₁, u₂)` on the
bare Mat — same verb, same return shape, different residual. Which one you
get is decided by whether the model is wrapped in a `TransferForm`, not by a
second function name. Generic in the element type; device-safe. Image points
only: `u₁`, `u₂` are 2-vectors, and a 3-vector (a bearing) is a `MethodError`.
"""
@inline function constraint_with_jacobians(f::TransferForm, u₁::StaticVector{2}, u₂::StaticVector{2})
    H = f.mat
    T = float(promote_type(eltype(H), eltype(u₁), eltype(u₂)))
    Hm = SMatrix{3, 3, T, 9}(H)
    x1 = _lift_image_point(u₁, T)
    h = Hm * x1
    Jπ = transform_deriv(PerspectiveMap(), h)
    g = SVector{2, T}(PerspectiveMap()(h) - SVector{2, T}(u₂[1], u₂[2]))
    H₁₂ = SMatrix{3, 2, T, 6}(Hm[1, 1], Hm[2, 1], Hm[3, 1], Hm[1, 2], Hm[2, 2], Hm[3, 2])
    G_x = hcat(Jπ * H₁₂, -SMatrix{2, 2, T, 4}(I))
    return g, G_x, _transfer_params_jacobian(Jπ, x1)
end

# The transfer constraint on UNIT BEARINGS — the infinite homography of a
# calibrated camera (`H = R` under a pure rotation, `K₂ R K₁⁻¹` in pixels) on
# rays, where the z = 1 chart and its perspective divide do not exist past 90°.
# The parts every bearing-transfer route reads, in ONE arithmetic order: the
# transferred ray `ĥ = H f₁ / ‖H f₁‖`, the residual `g = P·ĥ` in `f₂`'s tangent
# plane (`P` the 2×3 orthonormal basis of `s2_tangent_basis(f₂)`, so `P f₂ = 0`),
# the normalization's derivative `Jn = (I − ĥĥᵀ)/‖H f₁‖` and `D = P·Jn`, the
# derivative of `g` along the transferred point `H f₁` — what the pixel form's
# `Jπ` is to its perspective divide. `cheiral`: the transferred ray is in `f₂`'s
# hemisphere (`ĥ·f₂ > 0`); an antipodal pair has a small `g` too, so a scorer
# reads this verdict rather than the residual alone.
@inline function _bearing_transfer_parts(H, f₁::StaticVector{3}, f₂::StaticVector{3})
    T = float(promote_type(eltype(H), eltype(f₁), eltype(f₂)))
    Hm = SMatrix{3, 3, T, 9}(H)
    s = SVector{3, T}(f₁[1], f₁[2], f₁[3])
    d = SVector{3, T}(f₂[1], f₂[2], f₂[3])
    h = Hm * s
    nh = norm(h)
    ĥ = h / nh
    b1, b2 = s2_tangent_basis(d)
    P = SMatrix{2, 3, T, 6}(b1[1], b2[1], b1[2], b2[2], b1[3], b2[3])
    D = P * ((SMatrix{3, 3, T, 9}(I) - ĥ * ĥ') / nh)
    return (; g = P * ĥ, P, D, s, Hm, cheiral = dot(ĥ, d) > 0)
end

"""
    constraint_with_jacobians(f::TransferForm, f₁, f₂; M₁, M₂) -> (g, G_x, G_θ)

The transfer constraint of the wrapped homography on UNIT BEARINGS — the
infinite homography of a calibrated camera (under a pure rotation `H = R`):
`g = P·(H f₁/‖H f₁‖)`, the transferred ray in the tangent plane of `f₂` (`P` the
2×3 orthonormal basis of [`s2_tangent_basis`](@ref)`(f₂)`), zero when `f₂ ∥ H f₁`
— the constraint `f₂ × H f₁ = 0` at codimension 2. Its Jacobians:

- `G_x = [D·H·M₁  −P·M₂]` (2×4) in PIXEL coordinates, `M₁`/`M₂` the 3×2
  back-projection Jacobians of [`backproject_with_jacobian`](@ref) and
  `D = P·(I − ĥĥᵀ)/‖H f₁‖`; both images noisy, so `G_x Gₓᵀ =
  P(H̄M₁M₁ᵀH̄ᵀ + M₂M₂ᵀ)Pᵀ` (`H̄` the normalized transfer, `R` for a rotation) is the
  two-sided Sampson covariance in `f₂`'s tangent plane. The `f₂` block is the
  first-order derivative at `ĥ = f₂` (Sampson's linearization at the
  measurement), exact there.
- `G_θ = D·(f₁ᵀ ⊗ I₃)` (2×9, ∂g/∂vec(H) column-major) — the pixel form's
  `_transfer_params_jacobian` with `D` in the place of `Jπ`.

The pixel sibling `constraint_with_jacobians(f, u₁, u₂)` is the same constraint in
the z = 1 chart. An antipodal pair (`ĥ·f₂ < 0`) also gives a small `g`; a
scorer must refuse it separately (the hemisphere of the transferred ray).
"""
@inline function constraint_with_jacobians(
        f::TransferForm, f₁::StaticVector{3}, f₂::StaticVector{3};
        M₁::StaticMatrix{3, 2}, M₂::StaticMatrix{3, 2}
    )
    p = _bearing_transfer_parts(f.mat, f₁, f₂)
    T = eltype(p.g)
    G_x = hcat(p.D * p.Hm * SMatrix{3, 2, T, 6}(M₁), -(p.P * SMatrix{3, 2, T, 6}(M₂)))
    return p.g, G_x, _transfer_params_jacobian(p.D, p.s)
end

# -----------------------------------------------------------------------------
# TransferForm ↔ HomographyMap — the frame change is `∘` on the map
# -----------------------------------------------------------------------------

"""
    HomographyMap(f::TransferForm{Src, Dst}) -> HomographyMap{Src, Dst}
    TransferForm(h::HomographyMap{3, T, M, Src, Dst}) -> TransferForm{Src, Dst}

A transfer form's tags are endpoints, so its homography IS a framed
[`HomographyMap`](@ref) with the same two frames, and its frame change is that
map's composition (frame-checked by the map's own junction rule): with `D₁`,
`D₂` carrying the current endpoints into the new ones,

    TransferForm(D₂ ∘ HomographyMap(f) ∘ inv(D₁))   has   H′ = D₂ · H · D₁⁻¹.

An estimator that solves in a model frame holds MEASUREMENT maps `Cᵢ` carrying
data-frame points into the model frame (`x̃ᵢ = Cᵢ xᵢ`; for calibrated homography
`Cᵢ = Kᵢ⁻¹`), so moving the solved model back to the data frame is
`inv(C₂) ∘ HomographyMap(f) ∘ C₁`. `inv` on a `HomographyMap` is an O(1) swap
of its two stored matrices.

A residual is EQUIVARIANT, not invariant: `g = π(H x̃₁) − x₂` is a displacement
in the destination image, so a frame change moves it by that frame's local
Jacobian — exactly (not to first order) for an affine `D₂`, whose Jacobian is
constant. What IS frame-independent is the ZERO SET (an exact correspondence
stays exact in every frame). Measured over 2000 random draws with `κ(H)`,
`κ(D₁)`, `κ(D₂) ≤ 100` (pinned, seed and all, in
`test/transfer_composition_tests.jl`), as a relative error against
`max(|value|, 1)`:

| quantity | relation under the composition | measured max |
|---|---|---|
| transferred point `π(H x̃₁)` | commutes: `π(H′ b̃₁) = D₂(π(H ã₁))` | 8.2e-14 |
| algebraic `c = x̃₂ × H x̃₁` | `c′ = (λ₁λ₂ det D₂)·D₂⁻ᵀ c` | 3.5e-14 |
| transfer residual `g = π(H x̃₁) − x₂`, affine `dst` | `g′ = A₂ g` (`A₂` = `D₂`'s linear part) | 3.2e-13 |

(The same grid puts the WRONG direction — the measurement maps composed
uninverted — no closer than 4.5e-2 on the first row.)

Public, not exported (as `TransferForm` is).
"""
HomographyMap(f::TransferForm{Src, Dst}) where {Src, Dst} = relabel(Src, Dst, HomographyMap(f.mat))
TransferForm(h::HomographyMap{3, T, M, Src, Dst}) where {T, M, Src, Dst} =
    TransferForm{Src, Dst}(HomographyMat{T}(Tuple(h.forward)))

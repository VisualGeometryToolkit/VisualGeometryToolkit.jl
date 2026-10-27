# =============================================================================
# Lens families — two lens kinds, eight projections, five terms, two combinators
# =============================================================================
#
#   Lens(projection, distortion)   forward:  x_d = distortion(projection(X))
#   LiftLens(radial)               backward: the image point lifts to the ray
#                                  (x_d·N(ρ), D(ρ)); projection is a root on the ray
#
# A projection maps a ray onto a frame (perspective, equidistant, …, or a
# parametric one such as the double sphere); a distortion is a map in that
# frame, built from terms: `Radial` (a rational radial factor), `Tangential`,
# `ThinPrism`, `Tilt`, `FieldOfView`, combined by `Additive` (every term at the
# same point, summed: OpenCV/COLMAP) and `Chain` (in sequence).
#
# Projections and terms are LEAVES: structs of numbers and tuples of numbers,
# whose number type is their last type parameter. Lenses and combinators are
# NODES, whose type parameters are their fields' types. Everything a lens is
# asked — `parameter_vector`, `with_parameters`, `parameter_names`, the zero
# lens, the v3 record, the manifold, every Jacobian — is derived from the
# fields by one generic walk; no leaf states anything of its own.

import ForwardDiff

"""
    AbstractLensModel

Base type for all lens models.

A lens model determines how 3D rays through the optical center map to points
on the sensor, including both the projection geometry and any aberrations.

# Interface — the lens contract is on rays
A lens is a map between RAYS and NORMALIZED image points, and the pair
[`project`](@ref)`(lens, X)` (a ray → its image point) and
[`backproject`](@ref)`(lens, xd)` (an image point → its unit ray) is the whole
contract (`geometry/cameras/lens_rays.jl`): NaN where the lens does not image,
generic in the number type, allocation-free. There are two kinds of lens: a
forward [`Lens`](@ref) — a projection onto a frame followed by a distortion in
that frame — and a backward [`LiftLens`](@ref), whose image point lifts to its
ray in closed form (lenses.jl). Everything else is derived from the pair,
for every lens: `transform_deriv(lens, X)`, `transform_deriv_params(lens, X)`,
`project_with_jacobians`, the radial profile ([`projected_radius`](@ref) /
[`view_angle`](@ref)), and the perspective views [`distort_normalized`](@ref) /
[`undistort_normalized`](@ref) with the functor `lens(x)`.

A lens is a `CoordinateTransformations.Transformation`, so its derivative is
that package's `transform_deriv(lens, x)` (ForwardDiff through the perspective
view), and its back-projection is a map too: `inv(lens)` is an
[`InverseLens`](@ref) whose call is `undistort_normalized` and whose
`transform_deriv` is ForwardDiff through it.
Compositions inherit both — `transform_deriv(K ∘ inv(lens) ∘ K⁻¹, x)` is the
chain rule. The other generic defaults are `imaging_transform(intrinsics,
lens)` and the capability trait `has_distortion(lens)`. Every lens model here
is a central projection (single center of projection); finite-aperture models
are downstream.

Subtyping `CoordinateTransformations.Transformation` makes every imaging
chain a `ComposedTransformation` regardless of lens (the Pinhole and
distorted paths used to produce DIFFERENT composition types, breaking
structural pattern-matches like `pnp_pixel_whitening`).
"""
abstract type AbstractLensModel <: CoordinateTransformations.Transformation end

"""
    AbstractProjection

A ray → frame-point projection: the first stage of a forward [`Lens`](@ref).
The five fixed projections ([`Perspective`](@ref), [`Equidistant`](@ref),
[`Stereographic`](@ref), [`Equisolid`](@ref), [`Orthographic`](@ref)) each
land on a retinal frame ([`projection_frame`](@ref)); the parametric ones
([`DoubleSphere`](@ref), [`Unified`](@ref), [`ExtendedUnified`](@ref)) carry
coefficients and land on a frame their coefficients define.
"""
abstract type AbstractProjection end

"Perspective projection `x = X₁₂/X₃` (`r = tan θ`), onto the [`GnomonicFrame`](@ref); `θ < 90°`."
struct Perspective <: AbstractProjection end
"Equidistant projection `r = θ`, onto the [`EquidistantFrame`](@ref); `θ < 180°`."
struct Equidistant <: AbstractProjection end
"Stereographic projection `r = 2 tan(θ/2)`, onto the [`StereographicFrame`](@ref); `θ < 180°`."
struct Stereographic <: AbstractProjection end
"Equisolid-angle projection `r = 2 sin(θ/2)`, onto the [`EquisolidFrame`](@ref); `θ ≤ 180°`."
struct Equisolid <: AbstractProjection end
"Orthographic projection `r = sin θ`, onto the [`OrthographicFrame`](@ref); `θ ≤ 90°`."
struct Orthographic <: AbstractProjection end

"""
    DoubleSphere(xi, alpha) <: AbstractProjection

Usenko, Demmel & Cremers' double-sphere projection (3DV 2018): with
`d₁ = ‖X‖`, `z_s = ξ·d₁ + X₃` and `d₂ = √(X₁² + X₂² + z_s²)`,
`x = X₁₂ / (α·d₂ + (1 − α)·z_s)`. `ξ = α = 0` is the perspective projection.
Valid for rays inside the paper's field-of-view bound; the closed-form
inverse has the paper's domain. `α` moves through its logit in
[`LensManifold`](@ref).
"""
struct DoubleSphere{T <: Real} <: AbstractProjection
    xi::T
    alpha::T
end

"""
    Unified(xi) <: AbstractProjection

The unified camera model (Geyer & Daniilidis; Mei & Rives 2007; Kalibr's
`omni`, OpenCV's `omnidir`): `x = X₁₂ / (X₃ + ξ·‖X‖)`.
"""
struct Unified{T <: Real} <: AbstractProjection
    xi::T
end

"""
    ExtendedUnified(alpha, beta) <: AbstractProjection

The enhanced unified camera model of [Khomutenko2016](@citet) (Kalibr's `eucm`):
with `d = √(β(X₁² + X₂²) + X₃²)`, `x = X₁₂ / (α·d + (1 − α)·X₃)`.
"""
struct ExtendedUnified{T <: Real} <: AbstractProjection
    alpha::T
    beta::T
end

DoubleSphere(xi::Real, alpha::Real) = DoubleSphere(promote(float(xi), float(alpha))...)
Unified(xi::Integer) = Unified(float(xi))
ExtendedUnified(alpha::Real, beta::Real) = ExtendedUnified(promote(float(alpha), float(beta))...)

# -----------------------------------------------------------------------------
# Terms
# -----------------------------------------------------------------------------

"""
    AbstractDistortionTerm

A distortion term in a projection's frame: a MAP `x ↦ f(x)` ([`Radial`](@ref),
[`Tilt`](@ref), [`FieldOfView`](@ref)) or a DELTA `x ↦ Δ(x)` added to the
point ([`Tangential`](@ref), [`ThinPrism`](@ref)) — see [`Additive`](@ref).
"""
abstract type AbstractDistortionTerm end

"The power basis of a [`Radial`](@ref) factor: `Even` — `ρ², ρ⁴, …` (the default); `Full` — `ρ², ρ³, ρ⁴, …`."
struct Even end
@doc (@doc Even) struct Full end

"""
    Radial{B, P, Q, T}(k::NTuple{P}, d::NTuple{Q})
    Radial(k, d = ())                  # B = Even
    Radial{Full}(k, d)

The radial factor `F(ρ) = N(ρ)/D(ρ)` with `N = 1 + Σ kᵢ φᵢ(ρ)` and
`D = 1 + Σ dⱼ φⱼ(ρ)`, where the basis `B` sets `φᵢ`: `Even` — `ρ²ⁱ`; `Full`
— `ρⁱ⁺¹` (no linear term: the factor is smooth on the axis).

In a [`Lens`](@ref) it is the map `x ↦ F(|x|)·x` in the projection's frame —
OpenCV's rational radial factor (`k₁…k₃ / k₄…k₆`) in the perspective frame,
Kannala–Brandt's θ-polynomial in the equidistant frame. In a
[`LiftLens`](@ref) it is the undistortion of the image point, `x_u = F(ρ_d)·x_d`,
represented as the ray `(x_d·N, D)` — the division model, BabelCalib's rational
model, Scaramuzza's polynomial.

Coefficient names: `k1…kP` then `d1…dQ`.

The constructor also stores where the factor's imaged branch ends, in the basis
variable `u` (`ρ²` for `Even`): `u_fold`, the first positive root of the
profile's derivative (the fold), and `u_end`, the smaller of the fold and the
pole (the first positive root of `D`); `Inf` when there is none. Both follow
from the coefficients, so equal coefficients give equal factors; they are not
parameters, and a per-point branch test is one comparison against them.
"""
struct Radial{B, P, Q, T <: Real} <: AbstractDistortionTerm
    k::NTuple{P, T}
    d::NTuple{Q, T}
    u_fold::T
    u_end::T
    function Radial{B, P, Q, T}(k::Tuple, d::Tuple) where {B, P, Q, T <: Real}
        kT, dT = convert(NTuple{P, T}, k), convert(NTuple{Q, T}, d)
        return new{B, P, Q, T}(kT, dT, _radial_branch_ends(B, T, kT, dT)...)
    end
    # the ends carried over, not recomputed: only `_with_own_parameters`, whose
    # coefficients have the primal values the ends were computed from
    Radial{B, P, Q, T}(k::NTuple{P, T}, d::NTuple{Q, T}, u_fold::T, u_end::T, ::Val{:carried}) where {B, P, Q, T <: Real} =
        new{B, P, Q, T}(k, d, u_fold, u_end)
end
Base.show(io::IO, r::Radial) = (show(io, typeof(r)); print(io, "("); show(io, r.k); print(io, ", "); show(io, r.d); print(io, ")"))

"""
    Tangential(p1, p2)

Brown's decentering delta in OpenCV's convention:
`Δ = (2p₁uv + p₂(r² + 2u²), 2p₂uv + p₁(r² + 2v²))`.
"""
struct Tangential{T <: Real} <: AbstractDistortionTerm
    p1::T
    p2::T
end

"""
    ThinPrism{N, T}(sx::NTuple{N}, sy::NTuple{N})

The thin-prism delta `Δ = (Σ sxᵢ r²ⁱ, Σ syᵢ r²ⁱ)`: OpenCV's `(s₁, s₂; s₃, s₄)`
is `N = 2`, COLMAP's `(sx₁; sy₁)` is `N = 1`.
"""
struct ThinPrism{N, T <: Real} <: AbstractDistortionTerm
    sx::NTuple{N, T}
    sy::NTuple{N, T}
end

"""
    Tilt(tx, ty)

OpenCV's tilted-sensor map (`CALIB_TILTED_MODEL`, `τx`, `τy`): the projective
map of `computeTiltProjectionMatrix(τx, τy)` applied to `(x, 1)`.
"""
struct Tilt{T <: Real} <: AbstractDistortionTerm
    tx::T
    ty::T
end

"""
    FieldOfView(omega)

The field-of-view model of [Devernay2001](@citet) (COLMAP `FOV`, Kalibr `fov`):
`x ↦ x·atan(2r·tan(ω/2)) / (r·ω)`.
"""
struct FieldOfView{T <: Real} <: AbstractDistortionTerm
    omega::T
end

function Radial{B}(k::Tuple, d::Tuple = ()) where {B}
    T = float(promote_type(map(typeof, (k..., d..., 0))...))
    return Radial{B, length(k), length(d), T}(map(T, k), map(T, d))
end
Radial(k::Tuple, d::Tuple = ()) = Radial{Even}(k, d)
Tangential(p1::Real, p2::Real) = Tangential(promote(float(p1), float(p2))...)
function ThinPrism(sx::Tuple, sy::Tuple)
    length(sx) == length(sy) || throw(ArgumentError("ThinPrism: sx and sy need the same number of coefficients; got $(length(sx)) and $(length(sy))"))
    T = float(promote_type(map(typeof, (sx..., sy..., 0))...))
    return ThinPrism{length(sx), T}(map(T, sx), map(T, sy))
end
Tilt(tx::Real, ty::Real) = Tilt(promote(float(tx), float(ty))...)
FieldOfView(omega::Integer) = FieldOfView(float(omega))

"""
    isdelta(term) -> Bool

Is the term a delta (added to the point by [`Additive`](@ref)) rather than a map?
"""
isdelta(::AbstractDistortionTerm) = false
isdelta(::Union{Tangential, ThinPrism}) = true

# -----------------------------------------------------------------------------
# Combinators
# -----------------------------------------------------------------------------

"""
    Additive(m, Δ₁, …, Δₙ)
    Additive(Δ₁, …, Δₙ)

Terms evaluated at the SAME point and summed: `m(x) + (Δ₁(x) + … + Δₙ(x))`,
the deltas left to right; without a leading map, `x + (Δ₁(x) + …)`. OpenCV's
`projectPoints` and COLMAP's `Distortion()` are of this form.
"""
struct Additive{D <: Tuple} <: AbstractDistortionTerm
    terms::D
end

"""
    Chain(D₁, D₂, …)

Distortions applied in sequence, in the projection's frame: `Dₙ(…D₁(x))`.
Each stage is a map term or an [`Additive`](@ref). Aria's Fisheye624 (COLMAP
`RAD_TAN_THIN_PRISM_FISHEYE`) is a radial factor followed by tangential and
thin-prism deltas of the radially distorted point.
"""
struct Chain{D <: Tuple} <: AbstractDistortionTerm
    stages::D
end

function Additive(ts::AbstractDistortionTerm...)
    isempty(ts) && throw(ArgumentError("Additive needs at least one term"))
    for (i, t) in enumerate(ts)
        (t isa Additive || t isa Chain) && throw(ArgumentError("Additive: terms are leaves, not $(nameof(typeof(t)))"))
        i > 1 && !isdelta(t) && throw(ArgumentError("Additive: only the first term may be a map; term $i is a $(nameof(typeof(t)))"))
    end
    return _checked_names(Additive(ts))
end

# An Additive's head (its leading map, or `nothing`) and its deltas; a lone delta
# is the Additive of one.
@inline _head(a::Additive) = isdelta(first(a.terms)) ? nothing : first(a.terms)
@inline _head(_) = nothing
@inline _deltas(a::Additive) = isdelta(first(a.terms)) ? a.terms : Base.tail(a.terms)
@inline _deltas(t) = (t,)

function Chain(ds::AbstractDistortionTerm...)
    length(ds) >= 2 || throw(ArgumentError("Chain needs at least two stages"))
    for (i, d) in enumerate(ds)
        (isdelta(d) || d isa Chain) && throw(ArgumentError("Chain: stage $i must be a map term or an Additive, not a $(nameof(typeof(d)))"))
    end
    return _checked_names(Chain(ds))
end

# -----------------------------------------------------------------------------
# Lenses
# -----------------------------------------------------------------------------

"""
    Lens(projection[, distortion]) <: AbstractLensModel

A forward lens: the ray is projected onto `projection`'s frame and the frame
point is distorted by `distortion` (`nothing`, a term, an [`Additive`](@ref)
or a [`Chain`](@ref)). Projection is closed form; back-projection inverts the
distortion (`backproject`).
"""
struct Lens{F <: AbstractProjection, D} <: AbstractLensModel
    projection::F
    distortion::D
end
Lens(p::AbstractProjection) = Lens(p, nothing)
function Lens(p::AbstractProjection, d::AbstractDistortionTerm)
    # anywhere in the distortion: a Full-basis factor has no forward map in the frame
    any(t -> t isa Radial{Full}, _leaves(d)) &&
        throw(ArgumentError("Lens: a forward radial factor uses the Even basis; the Full basis is a LiftLens's"))
    return _checked_names(Lens{typeof(p), typeof(d)}(p, d))
end

"""
    LiftLens(radial::Radial) <: AbstractLensModel

A backward lens: the image point `x_d` lifts to the ray `(x_d·N(ρ), D(ρ))`,
`ρ = |x_d|` — closed-form back-projection; projection solves the image radius
as a root on the ray. `D` crossing zero is the 90° circle and `D < 0` the field
behind it, so the lens images past 90°.
"""
struct LiftLens{R <: Radial} <: AbstractLensModel
    radial::R
end

# --- the named models -----------------------------------------------------------

# The aliases bound the number type as the structs do: without the bound,
# `Division{1}` is a UnionAll over EVERY T, so not a subtype of `LiftLens`, and
# a method on `Type{<:LiftLens}` never sees it.

"The ideal perspective lens, `Lens(Perspective())`."
const Pinhole = Lens{Perspective, Nothing}
Pinhole() = Lens(Perspective())

"""
    KannalaBrandt(k₁, …, k_N)

Kannala & Brandt's fisheye (2006) in OpenCV's/COLMAP's form,
`θ_d = θ·(1 + k₁θ² + … + k_Nθ²ᴺ)`: `Lens(Equidistant(), Radial((k₁, …, k_N)))`.
"""
const KannalaBrandt{N, T <: Real} = Lens{Equidistant, Radial{Even, N, 0, T}}
KannalaBrandt(k::Real...) = Lens(Equidistant(), Radial(k))

"""
    Division(k₁, …, k_Q)

The division model (Fitzgibbon 2001): `x_u = x_d / (1 + k₁ρ² + … + k_Qρ²Q)`,
the lift `(x_d, 1 + k₁ρ² + …)`: `LiftLens(Radial((), (k₁, …)))`.
"""
const Division{Q, T <: Real} = LiftLens{Radial{Even, 0, Q, T}}
Division(k::Real...) = LiftLens(Radial((), k))

"""
    Scaramuzza(a₂, …, a_{N+1})

Scaramuzza's omnidirectional model (Scaramuzza, Martinelli & Siegwart 2006;
MATLAB's `fisheyeIntrinsics`), normalized: the lift
`(x_d, 1 + a₂ρ² + a₃ρ³ + …)` — `LiftLens(Radial{Full}((), (a₂, …)))`. The
coefficients are named `d1, d2, …` for `a₂, a₃, …`.
"""
const Scaramuzza{N, T <: Real} = LiftLens{Radial{Full, 0, N, T}}
Scaramuzza(a::Real...) = LiftLens(Radial{Full}((), a))

# -----------------------------------------------------------------------------
# The walk: everything derived from the fields
# -----------------------------------------------------------------------------
#
# A leaf's coefficients are its fields in order, a tuple field contributing its
# elements; a coefficient is named by its field, a tuple field `k` as `k1, k2, …`.
# The leaves of a lens are its projection, then its distortion depth-first, left
# to right — parameter order.

# A leaf's parameters are its first `_param_fields(X)` fields: a Radial's branch
# ends after them are derived from its coefficients, never parameters.
_param_fields(::Type{X}) where {X} = length(fieldnames(X))
_param_fields(::Type{<:Radial}) = 2
@generated _fields(x) = Expr(:tuple, (:(getfield(x, $i)) for i in 1:_param_fields(x))...)
@generated _flatmap(f, t::Tuple) = Expr(:tuple, (Expr(:..., :(f(t[$i]))) for i in 1:fieldcount(t))...)

@inline _leaves(::Nothing) = ()
@inline _leaves(t::Tuple) = _flatmap(_leaves, t)
@inline _leaves(x::Union{AbstractLensModel, Additive, Chain}) = _leaves(_fields(x))
@inline _leaves(leaf) = (leaf,)

@inline _coefficients(leaf) = _flatmap(f -> f isa Tuple ? f : (f,), _fields(leaf))
@inline _coefficients_of(x) = _flatmap(_coefficients, _leaves(x))
@generated function coefficient_names(::Type{X}) where {X}
    names = Symbol[]
    for (n, F) in zip(fieldnames(X)[1:_param_fields(X)], fieldtypes(X))
        F <: Tuple ? append!(names, (Symbol(n, i) for i in 1:fieldcount(F))) : push!(names, n)
    end
    return Tuple(names)
end
_coefficient_names_of(x) = _flatmap(leaf -> coefficient_names(typeof(leaf)), _leaves(x))

# The coefficient number type — `Union{}` for a lens without coefficients, so it
# never widens a `promote_type`.
@inline _coeff_type(x) = promote_type(map(leaf -> eltype(typeof(_coefficients(leaf))), _leaves(x))...)

# Rebuild a lens (or any part of one) of type X from flat coefficients, as code
# generated once per type: every offset and every result type is static. A leaf
# takes the number type T of its new coefficients (`nothing`: keeps its own); a
# node takes its children's types. `coef(i, T)` is the expression of coefficient i.
# With `src`, the expression of the part being rebuilt, a Radial keeps that part's
# branch ends instead of sweeping for them: only for coefficients whose primal
# values are the part's own.
_is_node(X) = X <: Union{AbstractLensModel, Additive, Chain}
function _rebuild_expr(X, T, coef, o = Ref(0), src = nothing)
    X === Nothing && return :nothing, Nothing
    if X <: Tuple || _is_node(X)
        sub(j) = src === nothing ? nothing : :(getfield($src, $j))
        parts = [_rebuild_expr(F, T, coef, o, sub(j)) for (j, F) in enumerate(fieldtypes(X))]
        X <: Tuple && return Expr(:tuple, first.(parts)...), Tuple{last.(parts)...}
        Y = X.name.wrapper{last.(parts)...}               # a node's parameters are its fields' types
        return :($Y($(first.(parts)...))), Y
    end
    fieldcount(X) == 0 && return :($X()), X
    S = T === nothing ? X.parameters[end] : T
    Y = X.name.wrapper{X.parameters[1:(end - 1)]..., S}
    num() = coef(o[] += 1, S)
    args = [F <: Tuple ? Expr(:tuple, (num() for _ in 1:fieldcount(F))...) : num() for F in fieldtypes(X)[1:_param_fields(X)]]
    (src !== nothing && X <: Radial) &&
        push!(args, :($S(_primal($src.u_fold))), :($S(_primal($src.u_end))), :(Val(:carried)))
    return :($Y($(args...))), Y
end
_number_type_of(C) = (T = C <: Tuple ? promote_type(fieldtypes(C)...) : eltype(C); T === Union{} ? Float64 : float(T))
@generated _with_parameters(x, c) = first(_rebuild_expr(x, _number_type_of(c), (i, S) -> :($S(c[$i]))))
# `x` with coefficients `c` whose primal values are x's own (its Duals, or its
# primal): its radial factors keep their branch ends — no sweep per evaluation.
@generated _with_own_parameters(x, c) = first(_rebuild_expr(x, _number_type_of(c), (i, S) -> :($S(c[$i])), Ref(0), :x))

# Names must be unique within a lens, so a coefficient is addressable by name.
function _checked_names(x)
    names = _coefficient_names_of(x)
    allunique(names) || throw(
        ArgumentError("$(nameof(typeof(x))): coefficient names repeat ($(join(String.(names), ", "))) — two terms of one kind in one lens")
    )
    return x
end

# A leaf's record tag: its name in snake case (`DoubleSphere` → `double_sphere`).
_tag(::Type{X}) where {X} = lowercase(replace(string(nameof(X)), r"(?<=[a-z])(?=[A-Z])" => "_"))
_tag(::Type{<:Radial{Full}}) = "radial_full"

"""
    zero(L::Type{<:AbstractLensModel}) -> L

The lens of type `L` with every coefficient zero: `zero(KannalaBrandt{4})`,
`zero(Division{2, Float64})`, `zero(typeof(lens))` — the anchor a fitter starts
from. The arities must be in the type; the number type defaults to Float64.
"""
function Base.zero(::Type{L}) where {L <: AbstractLensModel}
    refuse() = throw(ArgumentError("zero($L): the type leaves an arity free — give it, e.g. KannalaBrandt{4} or Division{2}"))
    C = L isa UnionAll ? (try L{Float64} catch; L end) : L       # a free number type takes Float64
    isconcretetype(C) || refuse()
    return try _zero(C) catch; refuse() end
end
@generated _zero(::Type{X}) where {X} = first(_rebuild_expr(X, nothing, (i, S) -> :(zero($S))))

"""
    radial(lens) -> Radial | nothing

The lens's radially symmetric factor: a lone `Radial` distortion, the leading
`Radial` of an `Additive`, the lift of a `LiftLens`; `nothing` otherwise. Its
folds and poles bound what the lens images (`monotone_radius`,
`view_angle_limit`), and it is what a regression onto a radially symmetric
family fits.
"""
radial(l::LiftLens) = l.radial
radial(l::Lens) = (h = l.distortion isa Additive ? _head(l.distortion) : l.distortion; h isa Radial ? h : nothing)

"""
    projection_frame(projection) -> Type{<:RetinalFrame} | nothing

The retinal frame a fixed projection lands on; `nothing` for a parametric
projection, whose frame its coefficients define.
"""
projection_frame(::Perspective) = GnomonicFrame
projection_frame(::Equidistant) = EquidistantFrame
projection_frame(::Stereographic) = StereographicFrame
projection_frame(::Equisolid) = EquisolidFrame
projection_frame(::Orthographic) = OrthographicFrame
projection_frame(::AbstractProjection) = nothing

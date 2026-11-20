# =============================================================================
# Blur — one Gaussian blur of an imaging model, by its covariance
# =============================================================================
#
# A point imaged through a camera is spread by several blurs at once — the defocus (a radiometric model's aperture
# integral), the camera's in-focus blur (diffraction, the sensor's anti-aliasing filter, demosaicing), a print's own blur.
# Their composition is the point spread function. A point spread function is a probability density: where a photon
# from a point source lands. A Gaussian blur is that density N(0, Σ), so its parameter IS a covariance, held as a
# `CovMat{2}` (`cov`), its frame tag the plane the blur lives on (a sensor's pixel frame, a target's frame on a print).
#
# The storage states the structure, as for every statistical form: `ScaledIdentity` for a round blur, `SDiagonal` for
# an axis-aligned one, a symmetric `SMatrix` for a general one. The algebra is the covariance's: blurs in sequence are
# a sum of independent displacements, so their covariances add (`convolve`); a linear map J of the plane pushes the
# covariance forward, Σ ↦ J Σ Jᵀ (`propagate`, the stack's one push-forward), full whatever the input's storage.
# The covariance's unit is the owner's: px² on a sensor, m² (FlexUnits quantities) on a print.
# =============================================================================

"""
    AbstractBlur

A blur kernel of an imaging model on a plane (a sensor, a print). One of
several that compose into a point spread; [`GaussianBlur`](@ref) is the one
implementation.
"""
abstract type AbstractBlur end

struct _Projected end   # the tag of GaussianBlur's internal constructor (a covariance `_psd_projected` built)

"""
    GaussianBlur(σ)                       # round: Σ = σ² I
    GaussianBlur(σx, σy)                  # axis-aligned: Σ = diag(σx², σy²)
    GaussianBlur(Σ::CovMat{2})            # by its covariance, its frame kept
    GaussianBlur(Σ::ScaledIdentity{2})    # round
    GaussianBlur(Σ::SDiagonal{2})         # axis-aligned
    GaussianBlur(Σ::SMatrix{2, 2})        # general: symmetric, positive semidefinite

A Gaussian blur: the point spread density `N(0, Σ)`, where a photon from a
point source lands relative to its ideal image. Its parameter is the
covariance `Σ`, a [`CovMat`](@ref)`{2}` whose frame tag names
the plane the blur lives on: px² for a blur on a sensor, m² (FlexUnits
quantities) for one on a print. `σ`, `σx`, `σy` are standard deviations. The
storage is the structure: a round blur is stored as a `ScaledIdentity` (one
number, unchanged by a rotation), an axis-aligned one as an `SDiagonal`, a
general one as a full symmetric `SMatrix`. Isbits.

The accessors are Statistics.jl's: `cov(b)` is `Σ` as stored (its frame kept),
`var(b)` the per-axis variances `(Σ₁₁, Σ₂₂)` and `std(b)` the per-axis
standard deviations `(√Σ₁₁, √Σ₂₂)`. A print's blur holds m² quantities
(`GaussianBlur(0.02mm)`); `ustrip(u, b)` is its plain form in `u²` and
`with_unit(b, u)` reads a plain one as in `u²`.

Blurs in sequence add their covariances ([`convolve`](@ref)); a linear map of
the plane pushes the covariance forward ([`propagate`](@ref)).
"""
struct GaussianBlur{C <: CovMat{2}} <: AbstractBlur
    covariance::C
    function GaussianBlur(Σ::CovMat{2})
        _check_blur_covariance(parent(Σ))
        return new{typeof(Σ)}(Σ)
    end
    # A covariance this file computed and projected onto the PSD cone (`_psd_projected`): its exact check would
    # refuse a rank-deficient result over a rounded determinant, so it is built as it is.
    GaussianBlur(Σ::CovMat{2}, ::_Projected) = new{typeof(Σ)}(Σ)
end
GaussianBlur(Σ::_FormStorage{2}) = GaussianBlur(CovMat(Σ))
GaussianBlur(σ::Number) = GaussianBlur(ScaledIdentity{2}(_checked_sigma(σ)^2))
GaussianBlur(σx::Number, σy::Number) = GaussianBlur(LinearAlgebra.Diagonal(StaticArrays.SVector(_checked_sigma(σx)^2, _checked_sigma(σy)^2)))
# a standard deviation: ≥ 0 (a zero blur is the identity), a float (no Rational reaches the storage)
_checked_sigma(σ) = σ >= zero(σ) ? float(σ) : throw(ArgumentError("GaussianBlur: a standard deviation must be ≥ 0, got $σ"))

_check_blur_covariance(S::ScaledIdentity{2}) =
    S.λ >= zero(S.λ) || throw(ArgumentError("GaussianBlur: the variance must be ≥ 0, got $(S.λ)"))
_check_blur_covariance(D::LinearAlgebra.Diagonal) =
    all(>=(zero(eltype(D.diag))), D.diag) || throw(ArgumentError("GaussianBlur: the variances must be ≥ 0, got $(D.diag)"))
_check_blur_covariance(M::StaticArrays.SMatrix{2, 2}) =
    (M[1, 2] == M[2, 1] && M[1, 1] >= zero(M[1, 1]) && M[2, 2] >= zero(M[2, 2]) && M[1, 1] * M[2, 2] >= M[1, 2]^2) ||
    throw(ArgumentError("GaussianBlur: the covariance must be symmetric positive semidefinite, got $M"))

# The Statistics.jl / Distributions.jl accessors of the density N(0, Σ) (stated in the GaussianBlur docstring, not
# here: a docstring on `cov` would be listed twice, by the `cov` entry of the manual and by this file's page):
# the covariance as stored (its frame kept), the per-axis variances (Σ₁₁, Σ₂₂) and standard deviations (√Σ₁₁, √Σ₂₂).
Statistics.cov(b::GaussianBlur) = b.covariance
Statistics.var(b::GaussianBlur) = (Σ = b.covariance; StaticArrays.SVector(Σ[1, 1], Σ[2, 2]))
Statistics.std(b::GaussianBlur) = sqrt.(Statistics.var(b))

"""
    convolve(a::GaussianBlur, b::GaussianBlur) -> GaussianBlur

The blur of `a` followed by `b` (convolution commutes): the sum of two
independent displacements, so the covariances add, diagonal when both are. The
two must live on one plane (their frames meet; an unlabeled one takes the
other's).
"""
function convolve(a::GaussianBlur, b::GaussianBlur)
    A, B = _frame(a.covariance), _frame(b.covariance)
    _check_junction(A, B, convolve, (a, b))
    s = parent(a.covariance) + parent(b.covariance)
    return GaussianBlur(CovMat{2, eltype(s), typeof(s), A <: UnlabeledFrame ? B : A}(s))
end

"""
    propagate(m, b::GaussianBlur) -> GaussianBlur

The blur `b` seen through a linear map of its plane (a pixel footprint onto a
board, a resampling): its covariance pushed forward, `Σ ↦ J Σ Jᵀ`, by
[`propagate`](@ref)'s own law for `m` — a static Jacobian `J` (the raw layer:
a bare Jacobian carries no frames, so the result is unlabeled whatever the
blur's frame), or a framed map with a constant Jacobian (`EuclideanMap`,
`SimilarityMap`, a camera's `FramedAffine`: the blur's frame checked against
the map's source, the result in its destination). Stored full, symmetrized; a
singular or near-singular `J` gives a rank-deficient covariance whose rounding
can leave an eigenvalue a hair below zero, which is clamped to zero (one
clearly negative is an `ArgumentError`).
"""
propagate(m, b::GaussianBlur) = _projected_blur(_psd_projected(propagate(m, b.covariance)))
_projected_blur(Σ::CovMat{2}) = GaussianBlur(Σ, _Projected())             # a framed map's: its frame
_projected_blur(M::StaticArrays.SMatrix{2, 2}) = GaussianBlur(CovMat(M), _Projected())   # the raw layer's: unlabeled
_psd_projected(Σ::CovMat{2, T, <:StaticArrays.SMatrix, F}) where {T, F} = relabel(F, CovMat(_psd_projected(parent(Σ))))
function _psd_projected(M::StaticArrays.SMatrix{2, 2})
    S = (M + M') / 2
    ok = S[1, 1] >= zero(S[1, 1]) && S[2, 2] >= zero(S[2, 2]) && S[1, 1] * S[2, 2] >= S[1, 2]^2
    ok && return S
    # the eigen-clamp of the plain numbers, in the entries' unit (m² for a print's blur, 1 for plain ones)
    u = oneunit(S[1, 1])
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(StaticArrays.SMatrix{2, 2}(map(x -> Float64(x / u), S))))
    λ, V = E.values, E.vectors
    tol = 64 * eps(Float64) * maximum(abs, λ)
    all(l -> l >= -tol, λ) || throw(ArgumentError("GaussianBlur: the pushed-forward covariance is not positive semidefinite (eigenvalues $(Tuple(λ)))"))
    P = V * LinearAlgebra.Diagonal(max.(λ, 0.0)) * V'
    return StaticArrays.SMatrix{2, 2}(P[1, 1] * u, P[1, 2] * u, P[1, 2] * u, P[2, 2] * u)   # exactly symmetric
end

convert_eltype(::Type{T}, b::GaussianBlur) where {T <: Real} = GaussianBlur(CovMat{2, T}(b.covariance))

# The parameter protocol: a blur's coefficients are the free entries of its
# covariance's storage — a round blur its variance `(:variance,)`, an
# axis-aligned one `(:variance_x, :variance_y)`, a general one
# `(:variance_x, :covariance_xy, :variance_y)` — so a fit keeps the structure
# it was given. A variance is bounded below by 0 (open: a
# `ParameterManifold`'s chart is then `σ² = σ₀²·exp(v)`, the log chart); the
# off-diagonal is unbounded, and a general covariance stepped past positive
# semidefiniteness is refused by `with_parameters`, as by the constructor. The
# frame of the covariance is kept. A print's blur (m² quantities) is stripped
# first (`ustrip(u, b)`).
parameter_vector(b::GaussianBlur) = (_plain_blur(b, parameter_vector); _blur_parameters(parent(b.covariance)))
_blur_parameters(S::ScaledIdentity{2}) = StaticArrays.SVector(S.λ)
_blur_parameters(D::LinearAlgebra.Diagonal) = StaticArrays.SVector(D.diag[1], D.diag[2])
_blur_parameters(M::StaticArrays.SMatrix{2, 2}) = StaticArrays.SVector(M[1, 1], M[1, 2], M[2, 2])
parameter_names(b::GaussianBlur) = _blur_names(parent(b.covariance))
_blur_names(::ScaledIdentity{2}) = (:variance,)
_blur_names(::LinearAlgebra.Diagonal) = (:variance_x, :variance_y)
_blur_names(::StaticArrays.SMatrix{2, 2}) = (:variance_x, :covariance_xy, :variance_y)
function parameter_bounds(b::GaussianBlur)
    θ = parameter_vector(b)
    T = float(eltype(θ))
    lo = map(n -> n === :covariance_xy ? T(-Inf) : zero(T), parameter_names(b))
    return (StaticArrays.SVector(lo), StaticArrays.SVector(map(_ -> T(Inf), lo)))
end
function with_parameters(b::GaussianBlur, θ::AbstractVector)
    _plain_blur(b, with_parameters)
    S = parent(b.covariance)
    n = length(_blur_names(S))
    length(θ) == n || throw(ArgumentError("GaussianBlur takes $n parameters ($(join(_blur_names(S), ", "))); got $(length(θ))"))
    Σ = _blur_storage(S, θ)
    return GaussianBlur(CovMat{2, eltype(Σ), typeof(Σ), _frame(b.covariance)}(Σ))
end
_blur_storage(::ScaledIdentity{2}, θ) = ScaledIdentity{2}(θ[1])
_blur_storage(::LinearAlgebra.Diagonal, θ) = LinearAlgebra.Diagonal(StaticArrays.SVector(θ[1], θ[2]))
_blur_storage(::StaticArrays.SMatrix{2, 2}, θ) = StaticArrays.SMatrix{2, 2}(θ[1], θ[2], θ[2], θ[3])
_plain_blur(b::GaussianBlur, f) = has_units(b) ?
    throw(ArgumentError("$(nameof(f)): the blur's covariance holds quantities; strip it first, `ustrip(u, b)`")) : b
# The imaging models with the parameter protocol (vignetting.jl, lighting.jl, here) on their ParameterManifold.
local_coordinates(ch::ParameterManifold, x::Union{AbstractVignetting, EnvironmentLight, GaussianBlur}) = _parameter_offsets(ch, x)

# Units: a print's blur is in m² (quantities), its plain form in a unit `u` in u² — the whole value's `ustrip(u, b)`.
has_units(b::GaussianBlur) = has_units(b.covariance)
FlexUnits.ustrip(u::Units, b::GaussianBlur) = has_units(b) ? GaussianBlur(_map_entries(x -> ustrip(u^2, x), b.covariance)) :
    throw(ArgumentError("ustrip: the blur's covariance is plain numbers already; `with_unit` attaches a unit"))
with_unit(b::GaussianBlur, u::Units) = has_units(b) ? throw(ArgumentError("with_unit: the blur's covariance already has units")) :
    GaussianBlur(_map_entries(x -> base_quantity(x * u^2), b.covariance))
# a covariance's stored entries mapped by `f`, its storage and frame kept
_map_entries(f, Σ::CovMat{2, T, M, F}) where {T, M, F} = relabel(F, CovMat(_map_entries(f, parent(Σ))))
_map_entries(f, S::ScaledIdentity{2}) = ScaledIdentity{2}(f(S.λ))
_map_entries(f, D::LinearAlgebra.Diagonal) = LinearAlgebra.Diagonal(map(f, D.diag))
_map_entries(f, M::StaticArrays.SMatrix{2, 2}) = map(f, M)

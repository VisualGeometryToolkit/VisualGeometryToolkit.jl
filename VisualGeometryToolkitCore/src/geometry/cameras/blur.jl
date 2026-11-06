# =============================================================================
# Blur — one Gaussian blur kernel of an imaging model, by its second-moment matrix
# =============================================================================
#
# A point imaged through a camera is spread by several blurs at once — the defocus (a radiometric model's aperture
# integral), the camera's in-focus blur (diffraction, the sensor's anti-aliasing filter, demosaicing), a print's own blur.
# Their composition is the point spread function; each of them is a blur, and a Gaussian one is described by its
# second-moment matrix M (`moment_matrix`): the kernel is ∝ exp(−½ uᵀ M⁻¹ u). It is a shape, not a statistical
# covariance (a deterministic kernel, applied by convolution), so it is not a `CovMat`.
#
# The storage states the structure, as for the statistical forms: `ScaledIdentity` for a round blur, `SDiagonal` for an
# axis-aligned one, a symmetric `SMatrix` for a general one. Convolving two Gaussians adds their moments (`convolve`); a linear map J of the
# plane takes M to J M Jᵀ (`transform_blur`), full whatever the input's storage. The moments' unit is the owner's: px²
# on a sensor, m² (quantities) on a print.
# =============================================================================

"""
    AbstractBlur

A blur kernel of an imaging model on a plane (a sensor, a print). One of
several that compose into a point spread; [`GaussianBlur`](@ref) is the one
implementation.
"""
abstract type AbstractBlur end

"""
    GaussianBlur(σ)                       # round: M = σ² I
    GaussianBlur(σx, σy)                  # axis-aligned: M = diag(σx², σy²)
    GaussianBlur(M::ScaledIdentity{2})    # round, by its moments
    GaussianBlur(M::SDiagonal{2})         # axis-aligned, by its moments
    GaussianBlur(M::SMatrix{2, 2})        # general: symmetric, positive semidefinite

A Gaussian blur kernel `k(u) ∝ exp(−½ uᵀ M⁻¹ u)` with the second-moment matrix `M`
([`moment_matrix`](@ref)): the kernel normalized to unit integral has
`∫ u uᵀ k(u) du = M` (its covariance as a density; a PSF's second moments, as
optics states a PSF's width). px² for a blur on a sensor, m² for one on a print.
The storage is the structure: a round blur is stored as a `ScaledIdentity`
(one number, unchanged by a rotation), an axis-aligned one as an `SDiagonal`,
a general one as a full symmetric `SMatrix`. Isbits.
"""
struct GaussianBlur{M <: _FormStorage{2}} <: AbstractBlur   # the three storages of a CovMat (base/types.jl)
    moments::M
    function GaussianBlur(S::ScaledIdentity{2})
        S.λ >= zero(S.λ) || throw(ArgumentError("GaussianBlur: the moment must be ≥ 0, got $(S.λ)"))
        return new{typeof(S)}(S)
    end
    function GaussianBlur(M::StaticArrays.SMatrix{2, 2})
        (M[1, 2] == M[2, 1] && M[1, 1] >= zero(M[1, 1]) && M[2, 2] >= zero(M[2, 2]) && M[1, 1] * M[2, 2] >= M[1, 2]^2) ||
            throw(ArgumentError("GaussianBlur: the moment matrix must be symmetric positive semidefinite, got $M"))
        return new{typeof(M)}(M)
    end
    function GaussianBlur(D::LinearAlgebra.Diagonal{<:Any, <:StaticArrays.SVector{2}})
        all(>=(zero(eltype(D))), D.diag) || throw(ArgumentError("GaussianBlur: the moments must be ≥ 0, got $(D.diag)"))
        return new{typeof(D)}(D)
    end
end
GaussianBlur(σ) = GaussianBlur(ScaledIdentity{2}(σ^2))
GaussianBlur(σx, σy) = GaussianBlur(LinearAlgebra.Diagonal(StaticArrays.SVector(σx^2, σy^2)))

moment_matrix(b::GaussianBlur) = b.moments

"""
    convolve(a::GaussianBlur, b::GaussianBlur) -> GaussianBlur

The blur of `a` followed by `b` (convolution commutes): its moments are the
sum, diagonal when both are.
"""
convolve(a::GaussianBlur, b::GaussianBlur) = GaussianBlur(a.moments + b.moments)

"""
    transform_blur(J::StaticMatrix{2, 2}, b::GaussianBlur) -> GaussianBlur

The blur `b` seen through the linear map `J` of its plane (a pixel footprint
onto a board, a resampling): moments `J M Jᵀ`, stored full.
"""
transform_blur(J::StaticArrays.StaticMatrix{2, 2}, b::GaussianBlur) = GaussianBlur(_symmetrized(StaticArrays.SMatrix{2, 2}(J * b.moments * J')))
@inline _symmetrized(M::StaticArrays.SMatrix{2, 2}) = (M + M') / 2

convert_eltype(::Type{T}, b::GaussianBlur{<:ScaledIdentity}) where {T <: Real} = GaussianBlur(ScaledIdentity{2}(T(b.moments.λ)))
convert_eltype(::Type{T}, b::GaussianBlur{<:LinearAlgebra.Diagonal}) where {T <: Real} =
    GaussianBlur(LinearAlgebra.Diagonal(StaticArrays.SVector{2, T}(b.moments.diag)))
convert_eltype(::Type{T}, b::GaussianBlur{<:StaticArrays.SMatrix}) where {T <: Real} = GaussianBlur(StaticArrays.SMatrix{2, 2, T}(b.moments))

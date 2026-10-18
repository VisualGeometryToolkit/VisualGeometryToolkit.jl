# =============================================================================
# Measurement noise — the covariance of a correspondence's endpoints
# =============================================================================
#
# THE RULE (conventions/measurement-noise.md). Noise lives on the MEASUREMENTS, never in a
# problem type:
#
#   - a PLAIN point (no covariance) is unit isotropic noise in its own frame,
#     Σ = I — every image-to-image match, every pool pair, every F/E/H as it
#     always was;
#   - `Uncertain(p, CovMat(Σ))` (or a `PrecMat`) supplies its covariance;
#   - `exact(p)` is a ZERO covariance: a known point (a board in metres, a
#     world point).
#
# A correspondence's constraint covariance is then the first-order propagation
# of both endpoints, Σ̃ = G₁Σ₁G₁ᵀ + G₂Σ₂G₂ᵀ (G_x = [G₁ G₂]), whitened r = L⁻¹g,
# LLᵀ = Σ̃. Both endpoints exact leaves nothing to whiten (Σ̃ = 0): refused.
#
# With Σ₁ = 0 and Σ₂ = I the homography's algebraic constraint g = h₁₂ᵀs̃ −
# x′·h₃ᵀs̃ has G₂ = −c·I (c = h₃ᵀs̃), so Σ̃ = c²·I and r = g/|c| — the TRANSFER
# error in the target image, in the target's units. The isotropic Σ = I₄ of two
# plain points divides instead by the board→image scale, which is why a board
# in metres scored in metres.
#
# STORAGE. A correspondence set carries the covariances as two attribute
# columns, `Σ₁` (sources) and `Σ₂` (targets), static 2×2 matrices in the
# caller's frame; an ABSENT column is unit isotropic noise, so a plain set has
# no columns and every reader takes its unchanged `Σ = I` path, bit for bit.
# [`noise_columns`](@ref) builds them from `Uncertain` endpoints (an estimator's
# constructor calls it once); [`pair_covariance`](@ref) and the transfer
# whitening read them. A normalization moves them with the points
# (`normalize(N, cs)`: Σ′ = A·Σ·Aᵀ, `A` the side's linear part).

"""
    exact(p::StaticVector) -> Uncertain

The measurement `p` known EXACTLY: `p` with a ZERO covariance, the third case of
the measurement-noise rule (a plain point is unit isotropic noise, an
[`Uncertain`](@ref) carries its own covariance). For data with no measurement
error in its frame — a calibration board's points in metres, a world point.
Board → image correspondences built as `exact.(board) .=> pixels` score the
transfer error in the image's pixels, whatever the board's units.

A correspondence needs one noisy endpoint: an estimator refuses a
correspondence whose two endpoints are both exact (nothing to whiten).

The zero covariance is an ordinary `CovMat` — no second representation. It is
read once, where an estimator's constructor turns endpoints into its covariance
columns ([`noise_columns`](@ref)); the per-point matrices cost 4 numbers each at
that boundary and nothing in the inner loop, where the columns are what is read.
"""
exact(p::StaticVector{N, T}) where {N, T <: Real} =
    Uncertain(p, CovMat{N, float(T), N * N}(ntuple(_ -> zero(float(T)), Val(N * N))))

# One endpoint's covariance as a static matrix: `I` (unit isotropic) for a plain
# point, the stored or inverted form of an `Uncertain`.
@inline _point_covariance(::Any) = I
@inline _point_covariance(u::Uncertain{V, <:CovMat{N}}) where {V, N} =
    SMatrix{N, N}(getfield(u, :attributes))
@inline _point_covariance(u::Uncertain{V, <:PrecMat{N}}) where {V, N} =
    SMatrix{N, N}(cov(u))

# The 2×2 block of one side: `I` stays the identity in the other side's eltype.
@inline _block2(Σ::UniformScaling, ::Type{T}) where {T} = SMatrix{2, 2, T, 4}(Σ.λ, 0, 0, Σ.λ)
@inline _block2(Σ::StaticMatrix{2, 2}, ::Type{T}) where {T} = SMatrix{2, 2, T, 4}(Σ)
_cov_eltype(::UniformScaling) = Bool
_cov_eltype(Σ::AbstractMatrix) = eltype(Σ)

"""
    pair_covariance(Σ₁, Σ₂)
    pair_covariance(cs, i)

The 4×4 measurement covariance of a correspondence over `(x₁, y₁, x₂, y₂)`: the
block diagonal of its endpoints' 2×2 covariances, `I` (the `UniformScaling`)
when both are unit isotropic — the shape [`whiten`](@ref) and
[`whitening_factor`](@ref) take as `Σ_x`. With a correspondence set `cs`, the
endpoints' covariances are its `Σ₁`/`Σ₂` columns ([`noise_columns`](@ref)), an
absent column unit isotropic; a plain set returns `I` for every `i`.
"""
@inline pair_covariance(::UniformScaling, ::UniformScaling) = I
@inline function pair_covariance(Σ₁, Σ₂)
    T = float(promote_type(_cov_eltype(Σ₁), _cov_eltype(Σ₂)))
    A = _block2(Σ₁, T); B = _block2(Σ₂, T); z = zero(T)
    return SMatrix{4, 4, T, 16}(
        A[1, 1], A[2, 1], z, z,
        A[1, 2], A[2, 2], z, z,
        z, z, B[1, 1], B[2, 1],
        z, z, B[1, 2], B[2, 2],
    )
end
@inline pair_covariance(cs, i::Integer) =
    pair_covariance(_side_covariance(attribute_column(cs, :Σ₁), i), _side_covariance(attribute_column(cs, :Σ₂), i))

# One side's covariance of correspondence `i`: unit isotropic without a column.
@inline _side_covariance(::Nothing, i) = I
@inline _side_covariance(col::AbstractVector, i) = @inbounds col[i]

# The two endpoints' covariance for a Sampson distance of two points (plain,
# `Uncertain` or `exact`): the pair's block diagonal, refused when both are exact.
@inline function _pair_covariance(u₁, u₂)
    Σ₁ = _point_covariance(u₁); Σ₂ = _point_covariance(u₂)
    _both_exact(Σ₁, Σ₂) && throw(ArgumentError(_BOTH_EXACT))
    return pair_covariance(Σ₁, Σ₂)
end
@inline _both_exact(_, _) = false
@inline _both_exact(Σ₁::AbstractMatrix, Σ₂::AbstractMatrix) = iszero(Σ₁) && iszero(Σ₂)

const _BOTH_EXACT =
    "both endpoints of a correspondence are exact (zero covariance): its constraint " *
    "has no measurement noise to whiten. At most one side may be `exact`."

"""
    noise_columns(sources, targets) -> NamedTuple

The covariance columns of a correspondence set whose endpoints may be
[`Uncertain`](@ref) (or [`exact`](@ref)): `(; Σ₁, Σ₂)`, each a vector of static
2×2 covariances in the endpoints' own frame, with a side whose endpoints are
all plain LEFT OUT (an absent column is unit isotropic noise) — so a plain set
gives `(;)` and its estimator runs exactly as without. Merge them into the
geometry-only set with `withattrs(cs; noise_columns(s, t)...)`; the endpoints
themselves are `geom` of these.

Throws `ArgumentError` for a correspondence whose two endpoints are both exact,
or a covariance that is not finite.
"""
function noise_columns(s::AbstractVector, t::AbstractVector)
    length(s) == length(t) || throw(DimensionMismatch("$(length(s)) sources for $(length(t)) targets"))
    Σ₁ = _noise_column(s); Σ₂ = _noise_column(t)
    if Σ₁ !== nothing && Σ₂ !== nothing
        for i in eachindex(Σ₁, Σ₂)
            (iszero(Σ₁[i]) && iszero(Σ₂[i])) && throw(ArgumentError(_BOTH_EXACT * " (correspondence $i)"))
        end
    end
    return _present_columns(Σ₁, Σ₂)
end
_present_columns(::Nothing, ::Nothing) = (;)
_present_columns(Σ₁, ::Nothing) = (; Σ₁)
_present_columns(::Nothing, Σ₂) = (; Σ₂)
_present_columns(Σ₁, Σ₂) = (; Σ₁, Σ₂)

# One side's column: `nothing` when no endpoint carries a covariance; otherwise
# every endpoint's 2×2 covariance (plain ones unit isotropic), Float64.
function _noise_column(x::AbstractVector)
    _may_carry_noise(eltype(x)) || return nothing
    any(p -> p isa Uncertain, x) || return nothing
    col = Vector{SMatrix{2, 2, Float64, 4}}(undef, length(x))
    for (k, i) in enumerate(eachindex(x))
        Σ = _block2(_point_covariance(x[i]), Float64)
        all(isfinite, Σ) || throw(ArgumentError("the covariance of endpoint $k is not finite: $Σ"))
        col[k] = Σ
    end
    return col
end

# A concrete plain eltype carries no covariance: no scan of the column.
_may_carry_noise(::Type{E}) where {E} = !isconcretetype(E) || E <: Uncertain

# A side's covariance column carried through an affine map with linear part A
# (Σ′ = A·Σ·Aᵀ) — what a normalization does to its points' noise. Every other
# attribute column is forwarded unchanged.
_transport_noise(attrs::NamedTuple, A₁, A₂) =
    _with_column(_with_column(attrs, Val(:Σ₁), A₁), Val(:Σ₂), A₂)
function _with_column(attrs::NamedTuple, ::Val{k}, A) where {k}
    haskey(attrs, k) || return attrs
    M = SMatrix{2, 2, Float64, 4}(A)
    col = [SMatrix{2, 2, Float64, 4}(M * Σ * M') for Σ in attrs[k]]
    return merge(attrs, NamedTuple{(k,)}((col,)))
end

# =============================================================================
# Per-datum weights and their kind (the robust-estimation contract, owner ruling Q3)
# =============================================================================
#
# Every robust result carries per-datum weights in [0, 1] in its METADATA,
# read by `weights(result)` as a plain vector. What a weight MEANS — and so
# where its inlier cut lies — is the weight KIND, a trait decided by the
# producer's type, never a field:
#
#   BinaryWeights     a threshold scorer's decision (1/0), and every L2 result (all ones)
#   PosteriorWeights  a posterior inlier probability (the Student-t + uniform scorer, an EM fit)
#   IrlsWeights       an M-estimator's loss weight ρ′ (Irls, Triggs), normalized to w_max = 1
#
# A metadata type answers `weight_kind(::Type{M})`, from its producer's type
# parameter where it has one (`RansacMetadata{T, S}` reads its scorer `S`).
# `inlier_mask(result)` dispatches on the kind: binary w > 0, posterior the MAP
# rule (> ½), IRLS refuses without an explicit cut — a loss weight's ½ means
# nothing.
# =============================================================================

"""
    WeightKind

What a robust result's per-datum [`weights`](@ref) mean, and so where their
inlier cut lies: [`BinaryWeights`](@ref), [`PosteriorWeights`](@ref) or
[`IrlsWeights`](@ref). A trait of the producer's type, read by
[`weight_kind`](@ref); [`inlier_mask`](@ref) dispatches on it.
"""
abstract type WeightKind end

"""
    BinaryWeights()

A [`WeightKind`](@ref): weights in {0, 1} — a threshold scorer's consensus
decision (`RansacScoring`, `MsacScoring`), or an L2 result's all ones. The
inlier mask is `w > 0`.
"""
struct BinaryWeights <: WeightKind end

"""
    PosteriorWeights()

A [`WeightKind`](@ref): posterior inlier probabilities — the Student-t +
uniform scorer's, or an EM fit's E-step. The inlier mask is the MAP rule,
[`is_map_inlier`](@ref) (posterior > ½).
"""
struct PosteriorWeights <: WeightKind end

"""
    IrlsWeights()

A [`WeightKind`](@ref): an M-estimator's loss weights `ρ′`, normalized to a
maximum of 1 (`Irls`, `Triggs`). They have no inlier cut of their own:
[`inlier_mask`](@ref) needs one given explicitly.
"""
struct IrlsWeights <: WeightKind end

"""
    weight_kind(result) -> WeightKind
    weight_kind(::Type{P}) -> WeightKind

The [`WeightKind`](@ref) of a robust result's [`weights`](@ref): a trait of its
metadata's TYPE, which answers from its producer's type (a scorer, a local
optimization). A contract generic: a producer or metadata type of another
package adds its method on the type.
"""
function weight_kind end
weight_kind(r::FitResult) = weight_kind(typeof(r.metadata))
weight_kind(::Type{<:AbstractThresholdScoring}) = BinaryWeights()
weight_kind(::Type{<:StudentTUniformScoring}) = PosteriorWeights()
weight_kind(::Type{Nothing}) = throw(ArgumentError("a result without metadata has no weights"))

"""
    inlier_mask(result) -> BitVector
    inlier_mask(result, cut::Real) -> BitVector

The inliers of a robust result, from its [`weights`](@ref) by their
[`weight_kind`](@ref): `w > 0` for [`BinaryWeights`](@ref), the MAP rule
[`is_map_inlier`](@ref) for [`PosteriorWeights`](@ref), and an
`ArgumentError` for [`IrlsWeights`](@ref), whose loss weights have no cut of
their own. With `cut`, `w > cut` for any kind — the cut stated by the caller.
"""
inlier_mask(r::FitResult) = inlier_mask(weight_kind(r), weights(r))
inlier_mask(r::FitResult, cut::Real) = BitVector(w > cut for w in weights(r))
inlier_mask(::BinaryWeights, w::AbstractVector) = BitVector(x > 0 for x in w)
inlier_mask(::PosteriorWeights, w::AbstractVector) = BitVector(is_map_inlier(x) for x in w)
inlier_mask(::IrlsWeights, ::AbstractVector) = throw(ArgumentError(
    "an IRLS loss weight has no inlier cut of its own; give one: inlier_mask(result, cut)"))

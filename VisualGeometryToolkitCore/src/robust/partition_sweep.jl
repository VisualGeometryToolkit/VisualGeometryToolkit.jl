# =============================================================================
# The partition sweep of the Student-t + uniform marginal score (Algorithm 1 of
# "RANSAC Scoring Done Right"; moved from RANSACScoringDoneRight's sweep.jl and
# extreme_value.jl, 2026-11-11, not rewritten).
#
#   label priors        the prior of a partition's labels, a table over the
#                       inlier count k: flat γ = ½, γ profiled at k/N, or γ
#                       integrated out under Beta(α, β)
#   outlier models      what a partition's complement pays: the supplied
#                       uniform half-width (`FixedHalfwidth`) here; a model that
#                       ESTIMATES the support (RANSACScoringDoneRight's
#                       extreme-value and generalized Pareto models) implements
#                       the four hooks of the estimated-support sweep below
#   ScoringWorkspace    the sweep's scratch (sort permutation, radix buffers,
#                       log Γ table, label-prior table, the launch buffers)
#   partition_score     the ONE per-partition formula of every sweep
#   ig_sweep_kernel!    the ONE prefix sweep, a KernelAbstractions kernel: a
#                       pool grid launches it per (pair, model, width) on any
#                       backend, `sweep!` as one work item on the CPU backend
#   sweep!              the driver: the sort (an O(n) radix), the gather, the
#                       launch; the exact parametric sweep for heterogeneous
#                       per-point penalties (the predictive score)
#
# The scale priors (`IGPrior`, `JeffreysPrior`) are robust/scale.jl's. The scorer
# that composes all of this is robust/student_t_uniform_scoring.jl.
# =============================================================================

# -----------------------------------------------------------------------------
# Label priors
# -----------------------------------------------------------------------------

"""
    AbstractLabelPrior

The prior of a partition's labels in [`StudentTUniformScoring`](@ref): it adds
a term that depends on the partition only through its inlier count `k`, held
in the workspace's label-prior table ([`label_prior_table`](@ref)), so the
sort-and-sweep stays exact. Its TOML tag is [`LABEL_PRIOR_TAGS`](@ref)'.
"""
abstract type AbstractLabelPrior end

"""
    FlatLabelPrior()

The inlier probability fixed at γ = ½: every partition has prior weight
`2^{-N}`, so the label prior adds nothing (the profile-marginal score's choice).
"""
struct FlatLabelPrior <: AbstractLabelPrior end

"""
    ProfiledLabelPrior()

The inlier probability profiled at γ̂ = k/N: the partition gains
`k log(k/N) + (N-k) log((N-k)/N)`.
"""
struct ProfiledLabelPrior <: AbstractLabelPrior end

"""
    BetaLabelPrior(alpha, beta)

The inlier probability integrated out under its conjugate prior
`Beta(alpha, beta)` (`BetaLabelPrior(0.5, 0.5)` is Jeffreys'): the partition
gains `log B(k + alpha, N - k + beta) - log B(alpha, beta)`. Both parameters
are the caller's (a configuration's `alpha` and `beta` keys).
"""
struct BetaLabelPrior <: AbstractLabelPrior
    alpha::Float64
    beta::Float64
    function BetaLabelPrior(alpha::Real, beta::Real)
        (alpha > 0 && beta > 0) ||
            throw(ArgumentError("BetaLabelPrior: alpha and beta must be positive, got ($alpha, $beta)"))
        return new(Float64(alpha), Float64(beta))
    end
end

"""
    LABEL_PRIOR_TAGS

The label priors' [`TagRegistry`](@ref) (family suffix `LabelPrior`):
`"flat"` ([`FlatLabelPrior`](@ref)), `"profiled"` ([`ProfiledLabelPrior`](@ref))
and `"beta"` ([`BetaLabelPrior`](@ref)), the values of a Student-t scorer
table's `label_prior` key.
"""
const LABEL_PRIOR_TAGS = TagRegistry(:label_prior, "LabelPrior",
    "flat" => FlatLabelPrior, "profiled" => ProfiledLabelPrior, "beta" => BetaLabelPrior;
    paths = [r"^(.*\.)?label_prior$"])

"""
    label_prior_table(n, label_prior) -> Vector{Float64}

The label prior's term for `k = 1..n` inliers of `n` points (a workspace's
table; empty for the flat prior, which adds nothing).

The profiled prior's term is the log of `γ̂^{k}(1−γ̂)^{n−k}` at `γ̂ = k/n`,
`k·log(k/n) + (n−k)·log((n−k)/n)` (`−n·H(k/n)`, `H` the binary entropy;
`0·log 0 = 0`, so the `k = n` term is 0); the Beta prior's
`log B(k + α, n − k + β) − log B(α, β)`.
"""
label_prior_table(n::Int, ::FlatLabelPrior) = Float64[]
function label_prior_table(n::Int, ::ProfiledLabelPrior)
    pt = Vector{Float64}(undef, n)
    n_f = Float64(n)
    @inbounds for k in 1:n-1
        k_f = Float64(k)
        pt[k] = k_f * log(k_f / n_f) + (n_f - k_f) * log((n_f - k_f) / n_f)
    end
    if n >= 1
        pt[n] = 0.0  # all inliers: n·log(1) + 0·log(0) = 0
    end
    return pt
end
function label_prior_table(n::Int, p::BetaLabelPrior)
    c = logbeta(p.alpha, p.beta)
    return [logbeta(k + p.alpha, n - k + p.beta) - c for k in 1:n]
end

# -----------------------------------------------------------------------------
# Outlier models
# -----------------------------------------------------------------------------

"""
    AbstractOutlierModel

The outlier component of [`StudentTUniformScoring`](@ref): what a partition's
complement pays. Core's is the supplied uniform half-width,
[`FixedHalfwidth`](@ref). A model that ESTIMATES the support from the
residuals (RANSACScoringDoneRight's extreme-value and generalized Pareto
models) is scored by the estimated-support sweep, model-certain at codimension
1, through its methods of [`outlier_cost`](@ref) and
[`scored_outlier_halfwidth`](@ref) (and optionally [`outlier_context!`](@ref)).
"""
abstract type AbstractOutlierModel end

"""
    FixedHalfwidth(outlier_halfwidth)

The uniform outlier density at a supplied half-width per axis (the
profile-marginal score): the complement pays `log(2·outlier_halfwidth)` per
point and axis.
"""
struct FixedHalfwidth <: AbstractOutlierModel
    outlier_halfwidth::Float64
    function FixedHalfwidth(outlier_halfwidth::Real)
        outlier_halfwidth > 0 ||
            throw(ArgumentError("outlier_halfwidth must be positive, got $outlier_halfwidth"))
        return new(Float64(outlier_halfwidth))
    end
end

"""
    estimates_outlier_support(model::AbstractOutlierModel) -> Bool

Whether `model` estimates the outlier support from the residuals (`true` for
every model but the supplied half-width, [`FixedHalfwidth`](@ref)): its
workspace then carries the estimated-support sweep's buffers (the sorted
distances and their two suffix sums).
"""
estimates_outlier_support(::AbstractOutlierModel) = true
estimates_outlier_support(::FixedHalfwidth) = false

"""
    outlier_context!(model, z, T1, T2, n) -> context

What an outlier model that estimates the support precomputes once per
hypothesis, from the sorted distances `z[1:n]` and their suffix sums
([`suffix_sums!`](@ref): `T1`, `T2`, each of length `n + 1`, `T1` free to be
overwritten as scratch); passed to every [`outlier_cost`](@ref). `nothing` by
default.
"""
outlier_context!(::AbstractOutlierModel, z, T1, T2, n) = nothing

"""
    outlier_cost(model, context, z, T1, T2, k, n) -> Float64

The complement's cost (`-log` density) of the partition of the `k` smallest of
the sorted distances `z[1:n]` (`k < n`), under an outlier model that estimates
the support: the hook the estimated-support sweep calls per prefix. No
default: a model implements it.
"""
function outlier_cost end

"""
    scored_outlier_halfwidth(model, context, z, T1, T2, k, n) -> Float64

The uniform-equivalent outlier half-width the selected partition (`k` of `n`)
was scored with, recorded on the workspace (what the scorer's
`inlier_threshold` reads); `NaN` for a model that states none. No default.
"""
function scored_outlier_halfwidth end

"""
    suffix_sums!(T1, T2, z) -> (T1, T2)

The suffix sums of the sorted values `z` (length `n`) into `T1` and `T2`
(length `n + 1`, `T·[n+1] = 0`): `T1[g] = Σ_{l≥g} z[l]`,
`T2[g] = Σ_{l≥g} (n-l+1)·z[l]`, ranks counted from the top, so a short tail's
moments never come from a difference of two large sums.
"""
function suffix_sums!(T1, T2, z)
    n = length(z)
    T1[n+1] = 0.0; T2[n+1] = 0.0
    @inbounds for l in n:-1:1
        T1[l] = T1[l+1] + z[l]
        T2[l] = T2[l+1] + (n - l + 1) * z[l]
    end
    return T1, T2
end

# =============================================================================
# Loggamma Table
# =============================================================================

"""
    log_gamma_table(n, d_g = 1; alpha0 = 0.0) -> Vector{Float64}

`log Γ(k·d_g/2 + α₀)` for `k = 1..n` (SpecialFunctions' `loggamma`): the
sweep's table, one path for Jeffreys (`α₀ = 0`) and the inverse-gamma prior
(`α₀ > 0`). A sweep reads entry `k − m` (the inliers' `ν/2 = d_g·(k − m)/2`).

Cost: ~5000 host-side `loggamma` calls per probe (~0.5 ms total), read-only on
the device for the entire sweep. (A log recurrence specialised on Jeffreys was
retired for this one path: uniform Float64 values, no recurrence drift.)
"""
function log_gamma_table(n::Int, d_g::Int = 1; alpha0::Float64 = 0.0)
    lg = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        lg[k] = loggamma(k * d_g / 2 + alpha0)
    end
    return lg
end

# =============================================================================
# Radix perm-sort — allocation-free stable sortperm for the sweep fast path
# =============================================================================
#
# The single-sort fast path of `sweep!` spends nearly all of its time in
# `sortperm!` over the per-point scores (~72 of ~87 µs per scored candidate at
# n = 2500 in the e2e RANSAC loop — the dominant cost of the whole loop).
# `Base.sortperm!` documents a STABILITY GUARANTEE ("the permutation is
# guaranteed to be stable ... indices of equal elements appear in ascending
# order"), so ANY stable ascending-by-`isless` permutation sort produces the
# identical `perm` — ties included. LSD radix sort over a monotone unsigned key
# is such a sort, runs in O(n) with small constants, and needs no comparisons.
#
# WHY A HAND-ROLLED RADIX AND NOT A LIBRARY (measured 2026-07-26, and please
# do not re-litigate without re-measuring). Sorting 715-residual columns over
# 4000 hypotheses of the PT-E british_museum pool, µs per hypothesis:
#
#     Base sortperm!(F64, reusable scratch)            19.70
#     Base sortperm!(F32 input)                        19.69   ← no F32 benefit
#     Base sort! on packed (key<<32 | index) UInt64    17.39
#     SortingAlgorithms.jl RadixSort, same packing     17.31
#     SortingAlgorithms.jl RadixSort via sortperm!     24.18
#     this implementation, UInt64 keys                 10.12
#     this implementation, UInt32 keys (default)        6.15
#
# All variants emit identical permutations; the gap is structural, not
# algorithmic cleverness: this version OWNS its scratch (the scoring loop
# calls it ~3·10⁵ times per scene — per-call allocation dominates the
# library paths), fuses every digit histogram into ONE scan over the keys,
# and skips digits on which all keys agree, which is most digits when
# residuals span a narrow range. A library sort cannot do the first or the
# third through its public interface. If a future Base/SortingAlgorithms
# release closes that gap, delete this and use it — after re-running
# RANSACScoringDoneRight's neurips2026/scripts/studies/f32_radix_probe.jl.
#
# Key transform (`_radix_key`): the standard IEEE-754 total-order flip
# (positive: set sign bit; negative: bitwise NOT), with every NaN collapsed to
# the maximum key. This reproduces `isless` exactly: -Inf < ... < -0.0 < +0.0
# < ... < +Inf < NaN, all NaNs mutually equal (payload/sign ignored — the
# collapse plus LSD stability preserves their original index order, exactly
# like a stable comparison sort under `isless`, which treats NaNs as
# incomparable-equal).
# =============================================================================

"""
    RadixPermBuffers{K}(n)
    RadixPermBuffers(n)          # K = UInt32 (Float32 key precision) — THE default

Scratch buffers for [`radix_sortperm!`](@ref) — the allocation-free stable
LSD radix `sortperm` used by the `sweep!` single-sort fast path. Owned by
[`ScoringWorkspace`](@ref); sized to the data count and resized with the other
sweep buffers (`resize!(sws, n)`).

### Key width `K` — the sort-precision knob (2026-07-26)

The radix does `sizeof(K)` byte iterations, each scattering a `K` key plus an
`Int` perm entry, so the key width sets both the iteration count and the scatter
bandwidth. `K = UInt32` (sort keys at Float32 precision) is the **default**:
measured 1.65× faster than `UInt64` on real PhotoTourism residual columns
(10.12 → 6.15 µs/hypothesis; ≈1.16× on the whole marginal per-pair scorer),
and it loses nothing, because the scored residuals are Float32 to begin with —
a pool's residual matrix is a `Matrix{Float32}`, so the Float64 sweep
buffer holds an exactly-representable widening and the round trip back to
Float32 is lossless. Verified on that pool: the emitted permutation is
BYTE-IDENTICAL to the `UInt64` path in 150/150 hypotheses (worst residual
delta exactly 0.0).

`K = UInt64` remains available and is the right choice for a caller whose
residuals carry genuine Float64 information (they are then ordered at full
precision). Both widths run the SAME generic code below — the width is a
type parameter, not a second implementation.

Ordering semantics at `K = UInt32`: the key is monotone but not injective on
Float64 (values within one Float32 ulp collapse to equal keys). Stability
then orders such a tie group by index, exactly as a stable comparison sort
under an `isless` that cannot separate them. Every sweep prefix boundary
inside such a group is between residuals equal to Float32 precision.
"""
struct RadixPermBuffers{K<:Unsigned}
    keys::Vector{K}          # monotone-transformed sort keys
    keys2::Vector{K}         # ping-pong partner of `keys`
    perm2::Vector{Int}       # ping-pong partner of the caller's perm
    counts::Matrix{Int}      # 256 × sizeof(K) per-digit histograms (one fused scan)
end
RadixPermBuffers{K}(n::Int) where {K<:Unsigned} = RadixPermBuffers{K}(
    Vector{K}(undef, n), Vector{K}(undef, n),
    Vector{Int}(undef, n), Matrix{Int}(undef, 256, sizeof(K)))
RadixPermBuffers(n::Int) = RadixPermBuffers{UInt32}(n)

"""
    radix_key(x, ::Type{K}) -> K

Monotone key: `radix_key(a, K) < radix_key(b, K)` ⟹ `isless(a, b)`.
For `K = UInt64` the converse holds too — `radix_key(a) == radix_key(b)` ⟺
neither `isless(a, b)` nor `isless(b, a)` (every NaN maps to the single
maximum key). Total-order equivalent of `isless` on all Float64 values,
including ±0.0 (distinct, -0.0 < +0.0 — exactly `isless`'s ordering) and
±Inf. `radix_key(x)` is the `UInt64` (full-Float64) width.

For `K = UInt32` a Float64 is first rounded to Float32 (the precision the
scored residuals actually carry); the map stays monotone non-decreasing, so
the emitted order is a valid ascending order, with ties broken by index
through LSD stability. A Float32 at `UInt32` keys is the identity precision
(exact `isless` order), at `UInt64` keys a lossless widening.
"""
@inline function radix_key(x::Float64, ::Type{UInt64})
    isnan(x) && return 0xffffffffffffffff
    u = reinterpret(UInt64, x)
    return (u & 0x8000000000000000) == zero(UInt64) ? (u | 0x8000000000000000) : ~u
end
@inline function radix_key(x::Float64, ::Type{UInt32})
    f = Float32(x)
    isnan(f) && return 0xffffffff
    u = reinterpret(UInt32, f)
    return (u & 0x80000000) == zero(UInt32) ? (u | 0x80000000) : ~u
end
@inline radix_key(x::Float64) = radix_key(x, UInt64)
@inline function radix_key(x::Float32, ::Type{UInt32})
    isnan(x) && return 0xffffffff
    u = reinterpret(UInt32, x)
    return (u & 0x80000000) == zero(UInt32) ? (u | 0x80000000) : ~u
end
@inline radix_key(x::Float32, ::Type{UInt64}) = radix_key(Float64(x), UInt64)

"""
    radix_sortperm!(perm, target, rb::RadixPermBuffers, n) -> nothing

Fill `perm[1:n]` with the stable ascending-`isless` permutation of
`target[1:n]` — bitwise identical to `sortperm!(perm, target)` for every
input (see the stability argument in the header comment). O(n): one fused
histogram scan, then ≤ 8 stable counting iterations over byte digits, skipping
digits on which all keys agree (a stable no-op). Allocation-free; buffers
may have capacity ≥ n.
"""
function radix_sortperm!(perm::Vector{Int}, target::AbstractVector{<:Union{Float32,Float64}},
                         rb::RadixPermBuffers{K}, n::Int) where {K<:Unsigned}
    keys = rb.keys; keys2 = rb.keys2; perm2 = rb.perm2; counts = rb.counts
    ndig = sizeof(K)                       # byte iterations: 4 for UInt32, 8 for UInt64
    @inbounds for i in 1:n
        keys[i] = radix_key(target[i], K)
        perm[i] = i
    end
    fill!(counts, 0)
    # Fused histogram: one scan over the keys accumulating every byte digit.
    # `ndig` is a compile-time constant per K, so the loop fully unrolls.
    @inbounds for i in 1:n
        k = keys[i]
        for d in 1:ndig
            counts[Int((k >> (8 * (d - 1))) & 0xff) + 1, d] += 1
        end
    end
    src_k, dst_k = keys, keys2
    src_p, dst_p = perm, perm2
    @inbounds for d in 1:ndig
        # Trivial digit (all n keys in one bucket): the stable iteration would be
        # an identity copy — skip it.
        b1 = 1
        while counts[b1, d] == 0
            b1 += 1
        end
        counts[b1, d] == n && continue
        # Prefix sums: counts[b,d] becomes the 0-based start offset of bucket b.
        off = 0
        for b in 1:256
            c = counts[b, d]
            counts[b, d] = off
            off += c
        end
        shift = 8 * (d - 1)
        for i in 1:n
            k = src_k[i]
            b = Int((k >> shift) & 0xff) + 1
            j = counts[b, d] + 1
            counts[b, d] = j
            dst_k[j] = k
            dst_p[j] = src_p[i]
        end
        src_k, dst_k = dst_k, src_k
        src_p, dst_p = dst_p, src_p
    end
    if src_p !== perm
        copyto!(perm, 1, src_p, 1, n)
    end
    return nothing
end

# =============================================================================
# The launch buffers of the one-work-item sweep
# =============================================================================

# What `sweep!` hands `ig_sweep_kernel!` besides the sorted column: the result
# cells and the one-pair "pool" layout (offset 0, `n` residuals, one model, one
# width). Owned by the workspace, so a scored hypothesis allocates nothing; the
# kernel is instantiated with a STATIC ndrange and workgroup of one on the CPU
# backend, whose launch is allocation-free (a dynamic ndrange's is not).
struct _SweepLaunch
    best::Array{Float64, 3}
    best_k::Array{Int32, 3}
    pair_offset::Vector{Int64}
    count::Vector{Int32}
    models::Vector{Int32}
    log_two_outlier_halfwidth::Vector{Float64}
    sorted::Vector{Float64}             # the gathered sorted column (capacity n)
    sorted_penalties::Vector{Float64}   # its per-point penalties (capacity n)
end
_SweepLaunch(n::Int) = _SweepLaunch(Array{Float64, 3}(undef, 1, 1, 1), zeros(Int32, 1, 1, 1), Int64[0], Int32[0],
                                    Int32[1], [NaN], Vector{Float64}(undef, n), Vector{Float64}(undef, n))
function _resize_launch!(L::_SweepLaunch, n::Int)
    resize!(L.sorted, n); resize!(L.sorted_penalties, n)
    return L
end

# =============================================================================
# ScoringWorkspace — Pre-allocated buffers for Algorithm 1 sweep
# =============================================================================

"""
    ScoringWorkspace

Pre-allocated scratch buffers for the scale-free marginal score (Algorithm 1),
the scoring contract's `scoring_workspace` of a [`StudentTUniformScoring`](@ref):
the scorer carries only the immutable spec, the mutable workspace lives with
the caller (a RANSAC's workspace holds one per thread).

### Prior consistency

The log Γ table is built at allocation time with a specific `α₀` — once a
`ScoringWorkspace` is constructed it is **committed to that scale prior**, and
its label-prior table to that label prior. [`scale_prior`](@ref)`(sws)` is the
single source of truth for which prior the buffers are configured for; the
bare `sweep!(sws, …)` reads it from here rather than accepting one, which
would risk pairing a Jeffreys table with an inverse-gamma score.

Read it through its accessors: [`sweep_order`](@ref) (the order of the last
sweep; its first `k*` entries are the consensus set), [`penalties`](@ref) and
[`sort_key`](@ref) (the predictive score's per-point penalty and ordering key,
filled before a sweep), [`lg_table`](@ref), [`scale_prior`](@ref). Build it
with `scoring_workspace(scoring, n, d_g)` (= [`alloc_scoring_workspace`](@ref))
or [`default_scoring_workspace`](@ref)`(n; dg, scale_prior)`.
"""
mutable struct ScoringWorkspace{K<:Unsigned}
    const perm::Vector{Int}
    const scratch::Vector{Int}
    const lg_table::Vector{Float64}
    const penalties::Vector{Float64}
    const sort_key::Vector{Float64}
    const prior_table::Vector{Float64}
    const scale_prior::IGPrior
    const radix::RadixPermBuffers{K}   # K = sort-key width; see RadixPermBuffers
    const label_prior::AbstractLabelPrior
    const tail::Vector{Float64}        # the estimated-support sweep's buffers; empty = supplied half-width
    const launch::_SweepLaunch         # the one-work-item kernel launch's buffers
    k::Int                             # k* of the last `score!` (the scoring contract's `inlier_count`)
    outlier_halfwidth::Float64         # the half-width the last score used at k*
end

# Derive the radix and launch scratch from the perm buffer's length, at the
# default key width — the form both workspace builders call. The `prior_table`
# must be `label_prior_table(n, label_prior)`, and `tail` holds the
# estimated-support buffers (none for a supplied width).
ScoringWorkspace(perm::Vector{Int}, scratch::Vector{Int}, lg_table::Vector{Float64},
                 penalties::Vector{Float64}, sort_key::Vector{Float64},
                 prior_table::Vector{Float64}, scale_prior::IGPrior;
                 label_prior::AbstractLabelPrior, tail::Vector{Float64} = Float64[]) =
    ScoringWorkspace(perm, scratch, lg_table, penalties, sort_key, prior_table,
                     scale_prior, RadixPermBuffers(length(perm)), label_prior, tail,
                     _SweepLaunch(length(perm)), 0, NaN)

"""
    sweep_order(sws::ScoringWorkspace) -> Vector{Int}

The order the last `sweep!` visited the points in: `sweep_order(sws)[1:k]` is
the support of the partition it returned `k` for (the scoring contract's
`consensus_order`).
"""
sweep_order(sws::ScoringWorkspace) = sws.perm

"""
    penalties(sws::ScoringWorkspace) -> Vector{Float64}

Per-point log-determinant penalties `ℓᵢ = log|Vᵢ|` of predictive scoring
(`StudentTUniformScoring{Predictive}`), filled before a sweep.
"""
penalties(sws::ScoringWorkspace) = sws.penalties

"""
    sort_key(sws::ScoringWorkspace) -> Vector{Float64}

The per-point sweep ordering key of predictive scoring (the unit-scale
Mahalanobis `q`), filled before a sweep.
"""
sort_key(sws::ScoringWorkspace) = sws.sort_key

"""
    lg_table(sws::ScoringWorkspace) -> Vector{Float64}

The `log Γ(k·dg/2 + α₀)` table the workspace was built with
([`log_gamma_table`](@ref)); its length is the largest `n` the workspace can
sweep (see `resize!`).
"""
lg_table(sws::ScoringWorkspace) = sws.lg_table

"""
    scale_prior(sws::ScoringWorkspace) -> IGPrior

The scale prior the workspace is committed to (that `sweep!(sws, …)` scores
under); a `StudentTUniformScoring` carries its own (the scoring contract's
`scale_prior`).
"""
scale_prior(sws::ScoringWorkspace) = sws.scale_prior

"""
    resize!(sws::ScoringWorkspace, n) -> sws

Size every per-point buffer of `sws` for a problem of `n` points: the sort
permutation and its scratch, the predictive `penalties` (zeroed, as a fresh
workspace has them) and `sort_key`, the radix perm-sort scratch — keeping the
exact-size radix fast path of `sweep!` reachable — and the launch's sorted
column. A label-prior table is REBUILT for `n` from the workspace's label
prior, since its terms depend on `N`. The estimated-support buffers, when
present, are resized with the rest.

The log Γ table is not resized: its entries depend on `k` alone, so a table of
capacity ≥ `n` serves any `n`. It cannot grow — the workspace does not record
the codimension it was built for — so `n > length(lg_table(sws))` throws;
build a larger workspace instead.
"""
function Base.resize!(sws::ScoringWorkspace, n::Integer)
    n >= 0 || throw(ArgumentError("resize!: n must be non-negative, got $n"))
    n <= length(sws.lg_table) || throw(ArgumentError(
        "resize!: this ScoringWorkspace's lg_table covers $(length(sws.lg_table)) " *
        "points, not $n; build a workspace for the larger problem"))
    resize!(sws.perm, n)
    resize!(sws.scratch, n)
    fill!(resize!(sws.penalties, n), 0.0)
    resize!(sws.sort_key, n)
    t = label_prior_table(Int(n), sws.label_prior)
    copyto!(resize!(sws.prior_table, length(t)), t)
    isempty(sws.tail) || resize!(sws.tail, 3n + 2)
    resize!(sws.radix.keys, n)
    resize!(sws.radix.keys2, n)
    resize!(sws.radix.perm2, n)
    _resize_launch!(sws.launch, Int(n))
    return sws
end

"""
    default_scoring_workspace(n; dg = 1, scale_prior) -> ScoringWorkspace

A scoring workspace with the flat label prior, for callers that sweep without
a `StudentTUniformScoring` in hand (a per-blob adaptation, a host-side seed
loop): length-`n` buffers for a problem of `n` points with codimension `dg`,
committed to `scale_prior` (the log Γ table's `α₀`; required — the stack's
default prior is a configuration's, not this function's).
"""
function default_scoring_workspace(n::Int; dg::Int = 1, scale_prior::IGPrior)
    ScoringWorkspace(
        Vector{Int}(undef, n),
        Vector{Int}(undef, n),
        log_gamma_table(n, dg; alpha0 = scale_prior.alpha0),
        zeros(Float64, n),
        Vector{Float64}(undef, n),
        Float64[],
        scale_prior;
        label_prior = FlatLabelPrior(),
    )
end

# =============================================================================
# Per-partition score (single formula; subsumes Jeffreys at α₀=β₀=0)
# =============================================================================

"""
    partition_score(prior::IGPrior, lg_half_dof, half_dof, RSS, outlier_cost) -> T

The marginal score of one inlier/outlier partition: the log evidence of its
inliers under the Gaussian with the scale integrated out against `prior` (the
inverse-gamma `IGPrior(α₀, β₀)`; Jeffreys at `IGPrior(0, 0)`), minus the outliers'
cost,

    S = lg_half_dof − (ν/2)·log 2π − (ν/2 + α₀)·log(RSS/2 + β₀) − outlier_cost,

with `half_dof = ν/2` (ν = d_g·(k − m), the inliers' residual dof),
`lg_half_dof = log Γ(ν/2 + α₀)` (a sweep's log Γ table entry, or a device
lgamma's value) and `RSS` the inliers' squared residual sum, floored by the
caller. Every sweep of the stack scores its partitions with this one formula —
`sweep!`, the parametric and estimated-support sweeps, the device grid kernel
(`ig_sweep_kernel!`) and a downstream device sweep (BlobBoards' per-blob
sweep). Plain arithmetic in `RSS`'s precision `T` (Float32 on a device),
callable inside a KernelAbstractions kernel.
"""
@inline function partition_score(p::IGPrior, lg_half_dof, half_dof, RSS::T, outlier_cost) where T
    lg_half_dof - half_dof * log(2 * T(π)) - (half_dof + T(p.alpha0)) * log(RSS / 2 + T(p.beta0)) - outlier_cost
end

# The score S(k) of the top-k partition of n points, from its running RSS: the
# ONE statement of the per-partition arithmetic every sweep shares — the
# parametric sweep's `_prefix_score` and the device grid kernel
# (`ig_sweep_kernel!`). The RSS floor clamps a COPY, never the running sum;
# ν = dg·(k − m) residual dof, and the n − k outliers pay (2·outlier_halfwidth)^{−dg} each.
# Plain Float64 arithmetic, callable inside a KernelAbstractions kernel.
@inline _table_partition_score(prior::IGPrior, lg_table, RSS::Float64, k::Integer, model_dof::Integer,
                               dg_f::Float64, n::Integer, log_two_outlier_halfwidth::Float64) =
    _table_partition_score_cost(prior, lg_table, RSS, k, model_dof, dg_f,
                                (n - k) * dg_f * log_two_outlier_halfwidth)

# The same score with the complement's cost given (an outlier model that
# estimates its support).
@inline _table_partition_score_cost(prior::IGPrior, lg_table, RSS::Float64, k::Integer, model_dof::Integer,
                                    dg_f::Float64, outlier_cost::Float64) =
    partition_score(prior, lg_table[k - model_dof], dg_f * (k - model_dof) / 2,
                    max(RSS, eps(Float64)), outlier_cost)

# =============================================================================
# The one prefix sweep — a KernelAbstractions kernel
# =============================================================================

"""
    ig_sweep_kernel!(backend[, workgroupsize[, ndrange]])(scores_out, sorted_sq_flat, pair_offset, K_per_pair,
                     M_per_pair, lg_table, log_two_outlier_halfwidth_grid, model_dof, dg, prior, p_offset,
                     sorted_penalties, prior_table, best_k_out; ndrange = (n, M_max, T))

One work-item per (pair, model, grid-point) of a flat pool layout (per pair
its models' sorted squared residuals, `K_per_pair[p]` each, from
`pair_offset[p]`; Core's `exact_pm_score_kernel!` layout): ONE sequential sweep
over the model's `K` sorted squared residuals, writing the best partition score
`max_k S(k)` to `scores_out[p_local, m, t]` (Float64). `S(k)` is the one
per-partition formula ([`partition_score`](@ref), indexed by the partition
size) at the grid point's `log(2·outlier_halfwidth)`, less half the running
sum of `sorted_penalties` (per point, in the same order; `nothing`: none) and
plus `prior_table[k]` (`nothing`: none). RSS and the penalty sum accumulate in
Float64. With `best_k_out` of nonzero size, the first k at the best score goes
to `best_k_out[p_local, m, t]` (0 when no partition scores above −Inf). This
is the stack's ONE prefix sweep: a pool grid (RANSACScoringDoneRight's
marginal grid, penalty- and prior-free, no k) and [`sweep!`](@ref) (one work
item on the CPU backend over its sorted column) both launch it. Work-items
beyond a pair's model count write nothing.
"""
@kernel function ig_sweep_kernel!(
        scores_out,              # (chunk, M_max, T) Float64
        @Const(sorted_sq_flat),
        @Const(pair_offset),
        @Const(K_per_pair),
        @Const(M_per_pair),
        @Const(lg_table),
        @Const(log_two_outlier_halfwidth_grid),   # (T,) log(2·outlier_halfwidth_t)
        model_dof::Int32, dg::Int32, prior::IGPrior, p_offset::Int32,
        sorted_penalties,        # per sorted point, as sorted_sq_flat (`nothing`: none)
        prior_table,             # (K,) added to S(k) (`nothing`: none)
        best_k_out)              # (chunk, M_max, T) Int32 the best k (0-size: not written)
    p_local, m, t = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)
    inbounds_thread = p_local <= size(scores_out, 1) && t <= size(scores_out, 3) &&
                      p <= length(M_per_pair)
    M = inbounds_thread ? Int(M_per_pair[p]) : 0
    if m <= M
        K     = Int(K_per_pair[p])
        base  = pair_offset[p] + (Int64(m) - Int64(1)) * Int64(K)
        log_two_outlier_halfwidth = log_two_outlier_halfwidth_grid[t]
        dg_f  = Float64(dg)
        RSS = 0.0
        pen_sum = 0.0
        bS  = -Inf
        bk  = 0
        @inbounds for k in 1:K
            RSS += Float64(sorted_sq_flat[base + k])
            sorted_penalties === nothing || (pen_sum += Float64(sorted_penalties[base + k]))
            if k > model_dof
                S = _table_partition_score(prior, lg_table, RSS, k, model_dof, dg_f, K, log_two_outlier_halfwidth)
                sorted_penalties === nothing || (S -= pen_sum / 2)
                prior_table === nothing || (S += prior_table[k])
                if S > bS
                    bS = S
                    bk = k
                end
            end
        end
        @inbounds scores_out[p_local, m, t] = bS
        length(best_k_out) > 0 && (@inbounds best_k_out[p_local, m, t] = Int32(bk))
    end
end

# =============================================================================
# Sort-and-Sweep
# =============================================================================

# The sweep's ascending sort of the first `n` points by `sort_target`: the
# stable Θ(n) radix perm-sort on the exact-size path when radix buffers are in
# hand (bitwise identical to `sortperm!`), else `sortperm!`; the partial-buffer
# path sorts views of the first-n prefix (see `sweep!`, Sizing).
function _sort_perm!(perm::Vector{Int}, sort_target_full, n::Int;
                     scratch::Union{Nothing,Vector{Int}} = nothing,
                     radix::Union{Nothing,RadixPermBuffers} = nothing)
    if length(perm) == n && length(sort_target_full) == n
        # Exact-size fast path (RANSAC scoring). With radix buffers in hand
        # the stable O(n) radix perm-sort replaces the comparison sortperm! —
        # the output permutation is bitwise identical (sortperm!'s documented
        # stability guarantee + the isless-exact key transform).
        if radix !== nothing && sort_target_full isa Union{Vector{Float64},Vector{Float32}} &&
           length(radix.keys) >= n && length(radix.keys2) >= n && length(radix.perm2) >= n
            radix_sortperm!(perm, sort_target_full, radix, n)
        elseif !isnothing(scratch) && length(scratch) >= n
            sortperm!(perm, sort_target_full; scratch)
        else
            sortperm!(perm, sort_target_full)
        end
    else
        # Partial path (a per-blob adaptation, a host seed loop): sort views
        # of the first-n prefix of each oversized buffer.
        perm_v = @view perm[1:n]
        tgt_v = length(sort_target_full) == n ? sort_target_full : @view(sort_target_full[1:n])
        if !isnothing(scratch) && length(scratch) >= n
            sortperm!(perm_v, tgt_v; scratch)
        else
            sortperm!(perm_v, tgt_v)
        end
    end
    return perm
end

"""
    sweep!(perm, lg_table, model_dof, log_two_outlier_halfwidth, dg, scores, n;
           scratch, penalties, sort_key, prior_table, scale_prior, radix, launch) -> (S*, k*)

The sort-and-sweep of the scale-free marginal score (Algorithm 1): sorts the
points (ascending by `scores`, or `sort_key` if given), then sweeps the prefixes
`k = m+1, …, n` scoring each partition by [`partition_score`](@ref) under
`scale_prior` (required: the stack's default prior is a configuration's).
Returns the best score `S*` and its inlier count `k*`; `perm[1:k*]` is the
selected support. Mutates `perm`.

### The per-hypothesis path — the one KernelAbstractions kernel

The prefix sweep is ONE KernelAbstractions kernel, [`ig_sweep_kernel!`](@ref):
a pool grid launches it per (pair, model, width), and this function launches
it as one work item on the CPU backend over the hypothesis's sorted column
(the owner's ruling on the 2026-11-11 twin audit, V2: the host's own scan is
gone). This function is the DRIVER around it: the sort (the radix below), the
gather of the sorted column and its per-point penalties, the label prior's
table, and the exact parametric sweep for heterogeneous gains (a different
enumeration, no prefix scan). With the workspace's `launch` buffers (every
`sweep!` on a [`ScoringWorkspace`](@ref)) the hypothesis allocates nothing;
without them the call allocates its column and launch state.

### Complexity — depends on which sort path is taken (see Sizing)

- **Θ(n)** on the exact-size fast path: [`radix_sortperm!`](@ref) is an LSD
  radix over monotone unsigned keys — one fused histogram scan plus at most
  `sizeof(K)` stable counting iterations (4 for the default `UInt32`), with trivial
  digits skipped. The sweep itself is a single Θ(n) scan.
- **Θ(n log n)** on the partial-buffer path, which falls back to the
  comparison-based `Base.sortperm!`.

This is not a constant-factor distinction: an oversized workspace silently
changes the complexity class (measured on PT-E-SIFT-4K, 2026-07-26: 51.5 µs per
hypothesis through the partial path, 21.7 through the radix path).
`resize!(sws, n)` resizes every sortable buffer to `n` for exactly this reason.

### Sizing

- **Exact-size fast path** (`length(perm) == n` and `length(sort_target) == n`):
  sorts via [`radix_sortperm!`](@ref) when radix buffers of capacity ≥ n are
  present, else via `sortperm!` directly.
- **Partial-buffer path** (`length(perm) > n` or `length(sort_target) > n`):
  sorts views of the first-n prefix through `sortperm!` (one workspace sized to
  the largest `n`, per-call `n` varying).

### Heterogeneous penalties — the parametric sweep

When per-point gains `g_i = -penalties_i/2` VARY, the optimal support is a
top-k set of `t_i(μ) = g_i - μ·q_i` at the support's own profiled
`μ_I = ν_k/(2 Q_I)` (the generalized sweep lemma), enumerated EXACTLY by the
parametric sweep. When all gains are equal (the model-certain score, constant
penalties) there are no crossings and the single sort is exact; it is taken
untouched, bit-identically.
"""
function sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                model_dof::Int, log_two_outlier_halfwidth::Float64, dg::Int,
                scores::AbstractVector{T}, n::Int;
                scratch::Union{Nothing,Vector{Int}}=nothing,
                penalties::Union{Nothing,AbstractVector{Float64}}=nothing,
                sort_key::Union{Nothing,AbstractVector{Float64}}=nothing,
                prior_table::Union{Nothing,AbstractVector{Float64}}=nothing,
                scale_prior::IGPrior,
                radix::Union{Nothing,RadixPermBuffers}=nothing,
                launch::Union{Nothing,_SweepLaunch}=nothing) where T
    if penalties !== nothing && n > 1
        pmin = penalties[1]; pmax = penalties[1]
        @inbounds for i in 2:n
            pmin = min(pmin, penalties[i]); pmax = max(pmax, penalties[i])
        end
        if pmax > pmin   # heterogeneous gains ⇒ exact parametric enumeration
            return _parametric_sweep!(perm, lg_table, model_dof, log_two_outlier_halfwidth, dg,
                                      scores, penalties, prior_table,
                                      scale_prior, n)
        end
    end
    _sort_perm!(perm, isnothing(sort_key) ? scores : sort_key, n; scratch, radix)
    # The prefix sweep is the stack's one kernel (`ig_sweep_kernel!`), one work item on the CPU backend over the sorted
    # column. It accumulates in Float64 whatever the scores' eltype (the running RSS decides k*), and clamps a copy of
    # the RSS. The column is gathered in Float64 (the kernel's own conversion, done once here).
    L = launch === nothing ? _SweepLaunch(n) : launch
    length(L.sorted) >= n || _resize_launch!(L, n)
    sorted = L.sorted
    @inbounds for k in 1:n
        sorted[k] = Float64(scores[perm[k]])
    end
    sorted_pen = penalties === nothing ? nothing : L.sorted_penalties
    if sorted_pen !== nothing
        @inbounds for k in 1:n
            sorted_pen[k] = Float64(penalties[perm[k]])
        end
    end
    L.count[1] = Int32(n); L.log_two_outlier_halfwidth[1] = log_two_outlier_halfwidth
    ig_sweep_kernel!(KernelAbstractions.CPU(), (1, 1, 1), (1, 1, 1))(
        L.best, sorted, L.pair_offset, L.count, L.models, lg_table, L.log_two_outlier_halfwidth,
        Int32(model_dof), Int32(dg), scale_prior, Int32(0), sorted_pen, prior_table, L.best_k)
    return L.best[1, 1, 1], Int(L.best_k[1, 1, 1])
end

"""
    sweep!(sws::ScoringWorkspace, sq_distances, n; model_dof, log_two_outlier_halfwidth, codimension = 1) -> (S*, k*)

The sort-and-sweep on a bare `ScoringWorkspace`, for callers without a
`StudentTUniformScoring` (a per-blob posterior adaptation, a host-side seed
loop). The scale prior is the workspace's, committed at allocation time
([`default_scoring_workspace`](@ref)`(n; dg, scale_prior)`). A forwarder: no
sweep loop of its own.
"""
sweep!(sws::ScoringWorkspace, sq_distances::AbstractVector{T}, n::Int;
       model_dof::Int, log_two_outlier_halfwidth::Float64,
       codimension::Int=1) where T =
    sweep!(sws.perm, sws.lg_table, model_dof, log_two_outlier_halfwidth, codimension,
           sq_distances, n; scratch=sws.scratch, scale_prior=sws.scale_prior,
           radix=sws.radix, launch=sws.launch)

# =============================================================================
# Parametric sweep — the exact enumeration of the generalized sweep lemma
# (uncertain_evidence 35ae25c, §"The generalized sweep"). Per-point gains
# g_i = -penalties_i/2 (the covariance penalty -1/2·log|V_i|; per-point
# prior/floor terms drop into the SAME vector when they exist), unit-scale
# residuals q_i = scores_i. The key t_i(μ) = g_i - μ·q_i is linear in μ, so
# the descending-t ordering is CONSTANT on each open interval between
# pairwise crossings μ_ij = (g_i - g_j)/(q_i - q_j) (positive only for
# CONFLICTED pairs: larger gain together with larger residual). Enumerating
# one representative μ per interval — below the first crossing, between
# consecutive distinct crossings, above the last — visits every top-k set
# attainable at any μ > 0; the optimal support is one of them (Lemma
# "Generalized sweep": it is a top-k set at its own self-consistent
# μ_I = ν_k/(2Q_I); if μ_I lands exactly ON a crossing, the gradient
# inequality makes the tie-partner support optimal too, and that one is
# realized on the adjacent open interval — interval interiors suffice).
#
# KINETIC enumeration (2026-09-22; it replaced a re-sort per interval). The
# ordering changes only AT a crossing, and a lone crossing (i, j) swaps one
# ADJACENT pair: the prefix sets below and above that position are unchanged,
# so only the one prefix at k = min(pos) is a new top-k set and is scored. The
# walk visits the crossings in ascending μ:
#   * the first interval is sorted and every prefix scored (Θ(n log n));
#   * a group of crossings that coincide (within 1e-13 relative of the group's
#     first), or a lone crossing whose pair is not adjacent (floating-point
#     disagreement between the crossing and the ordering), re-sorts at the
#     next interval's representative μ and re-scores every prefix — what the
#     retired version did at EVERY interval;
#   * every other crossing is one swap and one scored prefix, O(1).
# Total O(n² log n) for the crossings plus O(n log n) per re-sort, against the
# retired O(C·n log n). Measured (BenchmarkTools minimum, predictive-like
# instances): n = 60, 2681 → 36.2 µs (74×); n = 250, 139.0 → 0.716 ms (194×).
# Prefix sums are carried forward across a swap rather than re-summed, so S*
# agrees with a re-summing enumeration to summation order (≲ 1e-15 relative).
#
# A plain generic function, not a KernelAbstractions kernel: it is an ordered
# event walk over one hypothesis — sequential by construction — and it runs on
# the latency-bound per-hypothesis path.
# =============================================================================
function _parametric_sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                            model_dof::Int, log_two_outlier_halfwidth::Float64, dg::Int,
                            scores::AbstractVector{T}, penalties::AbstractVector{Float64},
                            prior_table::Union{Nothing,AbstractVector{Float64}},
                            scale_prior::IGPrior, n::Int) where T
    # all positive pairwise crossings (conflicted pairs only), with their pair
    cross_μ = Float64[]; cross_i = Int[]; cross_j = Int[]
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        dq = Float64(scores[i]) - Float64(scores[j])
        dq == 0.0 && continue
        μ = ((penalties[j] - penalties[i]) / 2) / dq     # (g_i - g_j)/(q_i - q_j)
        if μ > 0.0
            push!(cross_μ, μ); push!(cross_i, i); push!(cross_j, j)
        end
    end
    order = sortperm(cross_μ)
    nc = length(order)
    # groups of coinciding crossings: `group_start[g]` indexes `order`
    group_start = Int[]
    if nc > 0
        push!(group_start, 1)
        prev = cross_μ[order[1]]
        @inbounds for r in 2:nc
            c = cross_μ[order[r]]
            if c > prev * (1 + 1e-13) + 1e-300
                push!(group_start, r); prev = c
            end
        end
    end
    ng = length(group_start)

    ord = Vector{Int}(undef, n)          # the current descending-t(μ) order
    pos = Vector{Int}(undef, n)          # its inverse
    rss = Vector{Float64}(undef, n)      # prefix sums of q (Float64 regardless of T)
    pen = Vector{Float64}(undef, n)      # prefix sums of the penalties
    bestord = Vector{Int}(undef, n)
    best = _sweep_all_prefixes!(ord, pos, rss, pen, bestord, (-Inf, 0),
                                _interval_mu(cross_μ, order, group_start, 1),
                                lg_table, model_dof, log_two_outlier_halfwidth, dg, scores, penalties,
                                prior_table, scale_prior, n)
    @inbounds for g in 1:ng
        r0 = group_start[g]
        r1 = g == ng ? nc : group_start[g + 1] - 1
        if r0 == r1                                       # a lone crossing
            a = cross_i[order[r0]]; b = cross_j[order[r0]]
            pa = pos[a]; pb = pos[b]
            if abs(pa - pb) == 1                          # ... of an adjacent pair: swap
                p = min(pa, pb)
                x = ord[p]; y = ord[p + 1]
                ord[p] = y; ord[p + 1] = x; pos[y] = p; pos[x] = p + 1
                rss[p] = (p == 1 ? 0.0 : rss[p - 1]) + Float64(scores[y])
                pen[p] = (p == 1 ? 0.0 : pen[p - 1]) + Float64(penalties[y])
                if p > model_dof                          # the one new prefix set
                    S = _prefix_score(rss, pen, p, lg_table, model_dof, log_two_outlier_halfwidth, dg,
                                      prior_table, scale_prior, n)
                    if S > best[1]
                        best = (S, p)
                        copyto!(bestord, ord)
                    end
                end
                continue
            end
        end
        # coinciding crossings, or a non-adjacent pair: re-sort the next interval
        best = _sweep_all_prefixes!(ord, pos, rss, pen, bestord, best,
                                    _interval_mu(cross_μ, order, group_start, g + 1),
                                    lg_table, model_dof, log_two_outlier_halfwidth, dg, scores, penalties,
                                    prior_table, scale_prior, n)
    end
    # perm[1:k*] IS the selected support (ConsensusSet/mask contract)
    if length(perm) == n
        copyto!(perm, bestord)
    else
        @inbounds for k in 1:n; perm[k] = bestord[k]; end
    end
    return best
end

# The representative μ of ordering interval g: below the first crossing (g = 1),
# the midpoint of consecutive groups' first crossings, above the last (g = ng+1).
function _interval_mu(cross_μ, order, group_start, g)
    nc = length(order); ng = length(group_start)
    nc == 0 && return 1.0
    g == 1 && return cross_μ[order[1]] / 2
    g == ng + 1 && return cross_μ[order[nc]] * 2 + 1.0
    return (cross_μ[order[group_start[g - 1]]] + cross_μ[order[group_start[g]]]) / 2
end

# The partition score of the prefix k of the current order (`sweep!`'s formula,
# with the per-point penalties and the label prior's table).
@inline function _prefix_score(rss, pen, k, lg_table, model_dof, log_two_outlier_halfwidth, dg,
                               prior_table, scale_prior, n)
    S = _table_partition_score(scale_prior, lg_table, Float64(rss[k]), k, model_dof, Float64(dg),
                               n, log_two_outlier_halfwidth) - pen[k] / 2
    prior_table !== nothing && (S += Float64(prior_table[k]))
    return S
end

# Sort by descending t(μ) — ties by (ascending q, index), deterministic — and
# score every prefix; returns the running best (S, k), `bestord` its order.
function _sweep_all_prefixes!(ord, pos, rss, pen, bestord, best, μ, lg_table, model_dof,
                              log_two_outlier_halfwidth, dg, scores, penalties, prior_table, scale_prior, n)
    sortperm!(ord, 1:n;
              by = i -> (-(-penalties[i] / 2 - μ * Float64(scores[i])), Float64(scores[i]), i))
    r = 0.0; pl = 0.0
    @inbounds for k in 1:n
        idx = ord[k]; pos[idx] = k
        r += Float64(scores[idx]); pl += Float64(penalties[idx])
        rss[k] = r; pen[k] = pl
        k <= model_dof && continue
        S = _prefix_score(rss, pen, k, lg_table, model_dof, log_two_outlier_halfwidth, dg, prior_table, scale_prior, n)
        if S > best[1]
            best = (S, k)
            copyto!(bestord, ord)
        end
    end
    return best
end

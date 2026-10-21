# =============================================================================
# RANSAC Samplers — Uniform and PROSAC sampling strategies
# =============================================================================

# -----------------------------------------------------------------------------
# Abstract Sampler Type
# -----------------------------------------------------------------------------

"""
    AbstractSampler

Abstract type for RANSAC sampling strategies.

Subtypes implement `draw!` and `reset!` to provide different sampling behaviors
(uniform, PROSAC, NAPSAC, etc.). Problems store a sampler as a type-parameterized
field for compile-time dispatch.

# Required Methods
- `draw!(rng::Random.AbstractRNG, indices::AbstractVector{Int}, sampler)` —
  fill indices with a random sample drawn from `rng`. Every random decision a
  sampler makes MUST go through `rng` — this is what makes a seeded
  `ransac(...; rng = Xoshiro(seed))` run bit-reproducible. There is no
  two-argument form: the rng is named at every call site, so reaching for
  ambient task-local state stays visible in the source.
- `reset!(sampler)` — reset mutable state for reuse across `ransac()` calls

# Optional
- `reset!(sampler, budget::Int)` — reset, and take the run's trial budget
  (`ransac` supplies `config.max_trials`). The default ignores the budget;
  PROSAC schedules its pool growth over it.
- `trial_budget(sampler, config)` / `adapts(sampler)` — the run's schedule:
  how many trials `ransac` allows, and whether the incumbent may shorten it.
  The defaults are `config.max_trials` and `true` (adaptive RANSAC).

# Built-in Samplers
- `UniformSampler`: Stateless uniform random sampling
- `ProsacSampler`: Progressive sampling from high-quality matches
- `GaussianSampler`: Weighted sampling without replacement
- `ExhaustiveSampler`: Every minimal sample exactly once, no adaptive stop
"""
abstract type AbstractSampler end

"""
    UniformSampler <: AbstractSampler

Stateless uniform random sampling without replacement.

Stores the data size `n` for sampling from `1:n`.
"""
struct UniformSampler <: AbstractSampler
    n::Int
end

draw!(rng::Random.AbstractRNG, indices::AbstractVector{Int}, s::UniformSampler) =
    (_draw_uniform!(rng, indices, s.n); nothing)
reset!(::UniformSampler) = nothing

# A sampler whose schedule does not depend on the run's length ignores it.
reset!(s::AbstractSampler, ::Int) = reset!(s)

"""
    _draw_uniform!(rng, indices, n, k=length(indices))

Fill `indices[1:k]` with a uniform random sample without replacement from 1:n,
drawn from `rng`.
"""
function _draw_uniform!(
        rng::Random.AbstractRNG, indices::AbstractVector{Int},
        n::Int, k::Int = length(indices)
    )
    @inbounds for i in 1:k
        while true
            j = rand(rng, 1:n)
            duplicate = false
            for q in 1:(i - 1)
                indices[q] == j && (duplicate = true; break)
            end
            if !duplicate
                indices[i] = j
                break
            end
        end
    end
    return nothing
end

# =============================================================================
# PROSAC Sampler (Chum & Matas, CVPR 2005)
# =============================================================================
#
# Progressive Sample Consensus: samples high-quality correspondences first,
# then gradually expands the pool to include all data. Degenerates to standard
# RANSAC after T_N trials.
#
# Reference: Chum & Matas, "Matching with PROSAC — Progressive Sample Consensus"
#

"""
    ProsacSampler

Mutable state for PROSAC progressive sampling.

Maintains a pool of correspondences sorted by quality score. Sampling starts
from the top-m highest-scored matches and progressively expands the pool.

# Fields
- `sorted_indices::AbstractVector{Int}` — data indices sorted by score (best first)
- `m::Int` — sample size (constant, e.g. 4 for homography)
- `N::Int` — total data size (constant)
- `T_N::Int` — the trial budget the pool grows over; `ransac` sets it to
  `config.max_trials` when it prepares the sampler
- `n::Int` — current pool size (grows from m to N)
- `t::Int` — current trial count
- `T_n::Float64` — current T_n value (recurrence relation)
- `T_n_prime::Int` — discretized cumulative threshold

See also: `reset!`, `_draw_prosac!`
"""
mutable struct ProsacSampler{V <: AbstractVector{Int}} <: AbstractSampler
    sorted_indices::V
    m::Int
    N::Int
    T_N::Int
    n::Int
    t::Int
    T_n::Float64
    T_n_prime::Int
end

# The initial PROSAC T_n is `T_N · C(m,m)/C(N,m)`: the probability that a
# sample of m drawn from all N lies in the top m — the hypergeometric
# all-inlier probability the adaptive trial count reads (`_p_all_inliers`,
# loop.jl) at m "inliers" of N — times the budget.

"""
    ProsacSampler(sorted_indices, m; T_N = max_trials)

Create a PROSAC sampler from pre-sorted indices.

`sorted_indices` must be ordered by decreasing quality score.
`m` is the minimal sample size.
`T_N` is the trial budget over which the pool grows from the top `m` to
all `N` points (Chum & Matas's `T_N`). `ransac` replaces it with the run's
`config.max_trials` (`reset!(ps, budget)`), so the schedule spans the
budget the run actually has; the constructor default is the shipped
`[ransac] max_trials`, for a sampler drawn from directly.
"""
function ProsacSampler(
        sorted_indices::AbstractVector{Int}, m::Int;
        T_N::Int = _dflt("ransac", "max_trials")
    )
    N = length(sorted_indices)
    _holds_minimal_sample(m, N) || throw(ArgumentError("Need at least $m data points for PROSAC, got $N"))
    V = typeof(sorted_indices)
    return ProsacSampler{V}(sorted_indices, m, N, T_N, m, 0, _p_all_inliers(m, N, m) * T_N, 1)
end

"""
    reset!(ps::ProsacSampler)

Reset PROSAC sampler state for reuse (e.g., across multiple `ransac()` calls).
"""
function reset!(ps::ProsacSampler)
    ps.n = ps.m
    ps.t = 0
    ps.T_n = _p_all_inliers(ps.m, ps.N, ps.m) * ps.T_N
    ps.T_n_prime = 1
    return nothing
end

# The run's budget becomes the growth horizon. Was a hard-coded 200_000 until
# 2026-09-21, so at the default 10_000 trials the pool had grown only to
# where a 200_000-trial schedule would be after 10_000 trials.
function reset!(ps::ProsacSampler, budget::Int)
    budget >= 1 || throw(ArgumentError("PROSAC budget must be ≥ 1, got $budget"))
    ps.T_N = budget
    return reset!(ps)
end

"""
    _draw_prosac!(rng, indices, ps::ProsacSampler)

Draw a PROSAC sample into `indices`, drawing from `rng`.

Implements the Chum & Matas growth function:
1. Increment t, expand pool n while t > T'_n
2. If in PROSAC phase: sample m-1 from U_{n-1}, force index n
3. If in RANSAC phase: sample m from U_n (uniform)
"""
function _draw_prosac!(
        rng::Random.AbstractRNG, indices::AbstractVector{Int},
        ps::ProsacSampler
    )
    ps.t += 1
    m = ps.m

    # Expand pool: grow n while t exceeds T'_n threshold
    while ps.n < ps.N && ps.t > ps.T_n_prime
        # Recurrence: T_{n+1} = T_n * (n+1) / (n+1-m)
        ps.T_n *= (ps.n + 1) / (ps.n + 1 - m)
        ps.n += 1
        ps.T_n_prime = floor(Int, ps.T_n)
    end

    n = ps.n

    if ps.t > ps.T_n_prime
        # RANSAC mode: uniform sample of m from top-n
        _draw_uniform!(rng, indices, n)
        # Map to actual data indices
        @inbounds for i in eachindex(indices)
            indices[i] = ps.sorted_indices[indices[i]]
        end
    else
        # PROSAC mode: sample m-1 from U_{n-1}, force u_n
        if m > 1
            _draw_uniform!(rng, indices, n - 1, m - 1)
            @inbounds for i in 1:(m - 1)
                indices[i] = ps.sorted_indices[indices[i]]
            end
        end
        @inbounds indices[m] = ps.sorted_indices[n]
    end
    return nothing
end

draw!(rng::Random.AbstractRNG, indices::AbstractVector{Int}, s::ProsacSampler) =
    (_draw_prosac!(rng, indices, s); nothing)

# =============================================================================
# Gaussian Sampler  (Shekhovtsov-style weighted sampling via Gumbel-Max top-k)
# =============================================================================
#
# Samples size-k subsets without replacement from a categorical distribution
# whose weights are a Gaussian kernel in per-correspondence SNN ratios:
#
#     w_i = exp(-SNN_i^2 / (2 σ²)) + ε
#
# with σ = 1/8 by default (matches `model_H.py:280` in Shekhovtsov's code),
# and ε = 1e-6 a floor that keeps every correspondence eligible.  Sampling
# uses the Gumbel-Max trick:
#
#     key_i = log(U_i)/w_i,   U_i ~ Uniform(0,1);   argmax-k(key) → sample
#
# (the order of Shekhovtsov's U_i^{1/w_i}, `functional.py:71-90`, without its
# underflow; see `_draw_gaussian!`).  O(N) per draw for
# the key computation plus O(N log k) for the partial sort.
#

"""
    GaussianSampler <: AbstractSampler

Weighted sampling where each correspondence's weight is a Gaussian in its
SNN score.  Stateless (no progressive expansion).  Matches Shekhovtsov's
`sample_subsets` (RANSAC-Scoring `functional.py:71`).

# Fields
- `weights::Vector{Float64}` — precomputed normalized weights
- `inv_weights::Vector{Float64}` — cached `1.0 ./ weights` (the Gumbel-Max
  exponent; caching skips n divisions per draw — bitwise identical keys)
- `keys_buf::Vector{Float64}` — scratch for Gumbel-Max keys (draw-local)
- `perm_buf::Vector{Int}` — scratch index vector for the top-k partial sort
- `sort_scratch::Vector{Int}` — quicksort scratch for the top-k partial sort

The three scratch buffers make `draw!` allocation-free (the hot RANSAC loop
draws once per hypothesis); a `GaussianSampler` is therefore NOT safe to
share across concurrent `ransac()` calls (one sampler per problem instance,
which is how every constructor site builds it).
"""
struct GaussianSampler <: AbstractSampler
    weights::Vector{Float64}
    inv_weights::Vector{Float64}
    keys_buf::Vector{Float64}
    perm_buf::Vector{Int}
    sort_scratch::Vector{Int}
end

# Back-compat positional form (pre-scratch struct layout): derive the cached
# inverse weights and scratch buffers.
GaussianSampler(w::Vector{Float64}, keys_buf::Vector{Float64}) =
    GaussianSampler(
    w, 1.0 ./ w, keys_buf,
    Vector{Int}(undef, length(w)), Vector{Int}(undef, length(w))
)

"""
    GaussianSampler(snn; snn_sigma=1/8, floor=1e-6)

Build a `GaussianSampler` from per-correspondence SNN ratios.

`snn_sigma` is the standard deviation of the Gaussian SAMPLING PRIOR over SNN
ratios, `w ∝ exp(−SNN²/2σ²)` — it lives in SNN units and has NOTHING to do
with the robust loss width `scale_multiple·√s2` or with the residual noise σ. It is a
linear sigma on purpose: this is a prior over a dimensionless ratio, not the
robust layer's scale port (which speaks `s2`). `floor` is the
per-correspondence ε floor.
"""
function GaussianSampler(
        snn::AbstractVector{<:Real};
        snn_sigma::Real = 1 / 8, floor::Real = 1.0e-6
    )
    σ² = Float64(snn_sigma)^2
    w = [exp(-Float64(s)^2 / (2 * σ²)) + Float64(floor) for s in snn]
    # Normalized: `weights` is the sampling distribution (the log-space keys'
    # order is scale-invariant in the weights, so this changes no draw).
    s = sum(w)
    w .= w ./ s
    return GaussianSampler(w, similar(w))
end

reset!(::GaussianSampler) = nothing

"""
    _draw_gaussian!(rng, indices, s::GaussianSampler)

Gumbel-Max top-k draw without replacement:
    keys_i = log(U_i)/w_i  (the order of U_i^(1/w_i), without its underflow),
    argmax-k over i.

Allocation-free: the key computation writes into `s.keys_buf` and the
descending partial sort runs through the exact `partialsortperm!` routine
(`Base.Sort._sort!` with `ScratchQuickSort(1:k)` on a `Perm` ordering — the
same call `Base.partialsortperm` makes) with the sampler-owned index vector
and quicksort scratch, so per-draw results are bitwise identical to
`partialsortperm(keys, 1:k; rev = true)` while the hot loop allocates
nothing. Pinned against the public-API reference in
test/ransac_tests.jl ("GaussianSampler draw matches partialsortperm").
"""
function _draw_gaussian!(
        rng::Random.AbstractRNG, indices::AbstractVector{Int},
        s::GaussianSampler
    )
    k = length(indices)
    n = length(s.weights)
    # Keys in LOG space, log(U)/w: the same order as U^(1/w) (log is
    # monotone) without its underflow — U^(1/w) is exactly 0.0 for every
    # w ≲ 1e-3 (normalized), so on a pair whose SNN mass sits on fewer matches
    # than the sample size the rest of the top-k was a tie among zeros, broken
    # by index: the same degenerate sample every trial, no model in 10 000 trials (7% of
    # PhotoTourism test pairs end to end, 2026-10-04).
    @inbounds for i in 1:n
        s.keys_buf[i] = log(rand(rng)) * s.inv_weights[i]
    end
    # Descending partial sort: indices of the top-k largest keys.
    ix = s.perm_buf
    @inbounds for i in 1:n
        ix[i] = i
    end
    Base.Sort._sort!(
        ix, Base.Sort.InitialOptimizations(Base.Sort.ScratchQuickSort(1:k)),
        Base.Order.Perm(
            Base.Order.ord(isless, identity, true, Base.Order.Forward),
            s.keys_buf
        ),
        (; scratch = s.sort_scratch)
    )
    @inbounds for i in 1:k
        indices[i] = ix[i]
    end
    return nothing
end

draw!(rng::Random.AbstractRNG, indices::AbstractVector{Int}, s::GaussianSampler) =
    (_draw_gaussian!(rng, indices, s); nothing)

# =============================================================================
# Exhaustive Sampler — every minimal sample exactly once
# =============================================================================
#
# For small data (a handful of lights, markers, …) every minimal sample is
# cheap, so all of them are tried and adaptive termination — a bet that a good
# sample has been seen — has nothing to win. The sampler owns that decision
# through the schedule traits below; `ransac` reads them.

"""
    ExhaustiveSampler(n, k)

Every `k`-subset of `1:n` exactly once, in lexicographic order:
`binomial(n, k)` draws, after which `draw!` throws an `ArgumentError`.
Deterministic: the rng a `draw!` is handed is not used, so a run is the same
under every seed.

Under `ransac` it draws exactly `binomial(n, k)` samples whatever the
`RansacConfig` ([`trial_budget`](@ref)), and an incumbent never shortens the
run ([`adapts`](@ref)): `success_prob`, `max_trials` and `min_trials` do not
apply.
"""
mutable struct ExhaustiveSampler <: AbstractSampler
    n::Int
    k::Int
    next::Vector{Int}   # the subset the next draw! returns
    done::Bool
end

function ExhaustiveSampler(n::Integer, k::Integer)
    k >= 1 || throw(ArgumentError("sample size k must be ≥ 1, got $k"))
    n >= 0 || throw(ArgumentError("data size n must be ≥ 0, got $n"))
    return ExhaustiveSampler(Int(n), Int(k), collect(1:k), n < k)
end

function draw!(::Random.AbstractRNG, indices::AbstractVector{Int}, s::ExhaustiveSampler)
    s.done && throw(ArgumentError("ExhaustiveSampler: all binomial($(s.n), $(s.k)) samples drawn"))
    copyto!(indices, s.next)
    s.done = !_advance_combination!(s.next, s.n)
    return nothing
end

reset!(s::ExhaustiveSampler) = (s.next .= 1:s.k; s.done = s.n < s.k; nothing)

# The lexicographic successor of the k-subset `c` of 1:n, in place; `false`
# (and `c` unchanged) past the last one.
function _advance_combination!(c::AbstractVector{Int}, n::Int)
    k = length(c)
    i = k
    @inbounds while i >= 1 && c[i] == n - k + i
        i -= 1
    end
    i == 0 && return false
    @inbounds c[i] += 1
    @inbounds for j in (i + 1):k
        c[j] = c[j - 1] + 1
    end
    return true
end

# -----------------------------------------------------------------------------
# The schedule traits: how many trials a run makes, and whether it adapts
# -----------------------------------------------------------------------------

"""
    trial_budget(sampler, config::RansacConfig) -> Int

The number of trials `ransac` starts out allowing. `config.max_trials` for a
random sampler (the default); `binomial(n, k)`, every sample, for the
[`ExhaustiveSampler`](@ref), whatever the config.
"""
trial_budget(::AbstractSampler, config) = config.max_trials
trial_budget(s::ExhaustiveSampler, _config) = s.n < s.k ? 0 : binomial(s.n, s.k)

"""
    adapts(sampler) -> Bool

Whether `ransac` may shorten the run from the incumbent's consensus (the
hypergeometric adaptive trial count). `true` for a random sampler (the
default); `false` for the [`ExhaustiveSampler`](@ref), whose run is every
sample.
"""
adapts(::AbstractSampler) = true
adapts(::ExhaustiveSampler) = false

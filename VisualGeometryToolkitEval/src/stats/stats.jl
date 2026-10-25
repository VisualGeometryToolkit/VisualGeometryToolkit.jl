# =============================================================================
# src/stats/stats.jl — the shared summary statistics.
# =============================================================================

"""
    finite_median(v; empty = NaN) -> Float64

Median over the finite entries of `v`. `empty` is returned when no finite
entries exist — use the default `NaN` for REPORTING (propagates as missing)
and `Inf` for ARGMIN OBJECTIVES (an all-NaN column must lose the argmin,
never win it).
"""
finite_median(v; empty::Float64 = NaN) =
    (w = filter(isfinite, Float64.(v)); isempty(w) ? empty : median(w))

"""
    lower_median(v) -> eltype(v)

The `⌈n/2⌉`-th order statistic of `v` under `isless`: for an even `n` the
LOWER of the two middle values, never their mean, so the result is always an
element of `v`. `NaN` sorts last under `isless` — filter first when it should
not count. Empty `v` → `NaN`. The statistic of record of the PnP supplement
tables and the pool-size audits (an observed value, not an interpolation).
"""
lower_median(v) = isempty(v) ? NaN : partialsort(collect(v), cld(length(v), 2))

"""
    reference_scale(s2) -> Float64

The validation reference scale `s0 = √median σ̂²` over the finite entries of
the per-pair inlier variances `s2` — the scale every tuning window is
positioned at. Throws on an input with no finite entry: a window cannot be
placed without a scale, and a `NaN` scale would silently select nothing.
"""
function reference_scale(s2)
    m = finite_median(s2)
    isnan(m) && throw(ArgumentError("reference_scale: no finite inlier variance among $(length(s2)) entries"))
    return sqrt(m)
end

# -----------------------------------------------------------------------------
# The mAA rule. A pair with ground truth and enough valid correspondences to run
# (at least one minimal sample) is ELIGIBLE: its run counts, and a run that
# yields no model (error non-finite — `Inf` or `NaN`) scores accuracy 0 in the
# denominator. A pair that cannot be processed (no ground truth, or fewer valid
# correspondences than a minimal sample) is SKIPPED: it leaves the
# denominator. Every figure path computes mAA through `eligible_maa` and reports
# `pair_outcomes` beside it, so the failure counts are visible.
# -----------------------------------------------------------------------------

_eligibility(errors, ::Nothing) = trues(length(errors))
function _eligibility(errors, eligible::AbstractVector{Bool})
    length(eligible) == length(errors) ||
        throw(DimensionMismatch("eligible mask has $(length(eligible)) entries for $(length(errors)) errors"))
    return eligible
end

"""
    maa_threshold() -> Float64

The mAA threshold of the reported statistics, in degrees: configs/default.toml
`[metrics] maa_threshold_deg` (the 10 of mAA@10°). Every reader of that
threshold calls this; none spells the number.
"""
maa_threshold() = Float64(_config_default("metrics", "maa_threshold_deg"))

"""
    eligible_maa(errors, eligible = nothing; T = maa_threshold()) -> Float64

The mAA@`T` of a set of pairs under the eligibility rule: the continuous AUC
(VGC `mean_average_accuracy`) over the ELIGIBLE pairs, where a non-finite
error (a run that yielded no model) scores 0 and stays in the denominator,
and an ineligible pair (`eligible[i] == false`: no ground truth, or fewer
valid correspondences than a minimal sample) is excluded. `eligible =
nothing` treats every pair as eligible. `NaN` when no pair is eligible. `T`
defaults to [`maa_threshold`](@ref).
"""
function eligible_maa(errors, eligible = nothing; T::Real = maa_threshold())
    m = _eligibility(errors, eligible)
    return mean_average_accuracy(view(collect(errors), m); T)
end

"""
    pair_outcomes(errors, eligible = nothing) -> (; n_eligible, n_failed, n_skipped)

The counts behind [`eligible_maa`](@ref): eligible pairs, eligible pairs whose
run yielded no model (non-finite error), and skipped (ineligible) pairs —
reported beside every mAA so a reader sees why runs fail.
"""
function pair_outcomes(errors, eligible = nothing)
    m = _eligibility(errors, eligible)
    n_eligible = count(m)
    n_failed = count(i -> m[i] && !isfinite(errors[i]), eachindex(errors))
    return (; n_eligible, n_failed, n_skipped = length(m) - n_eligible)
end

"""
    bootstrap_ci(stat, rng, x; n_resamples, alpha) -> (lo, hi)

BCa (bias-corrected and accelerated) bootstrap CI of `stat(x)` via
Bootstrap.jl — Shekhovtsov's `scipy.stats.bootstrap(..., method='BCa')`
protocol. For a REAL sample the non-finite entries are dropped first and a
constant sample returns its value as a degenerate interval. Any other
element type is resampled as given (e.g. a scene-cluster bootstrap over
per-scene error matrices, with `stat` reducing the resampled scenes). Fewer
than 3 elements → `(NaN, NaN)` (BCa's jackknife needs them).

`rng` seeds the global Random stream before the Bootstrap.jl call (the
library resamples from it and takes no rng), so a fresh seed per call keeps a
run reproducible on one Julia version: the resampling itself is Julia's
task-local default stream, whose algorithm Julia does not fix across versions
(record `julia_version` beside a reported CI). `n_resamples` and `alpha`
default to configs/default.toml `[metrics]` (`bootstrap_resamples`,
`ci_alpha`).
"""
function bootstrap_ci(stat::Function, rng::AbstractRNG, x::AbstractVector;
                      n_resamples::Int = _config_default("metrics", "bootstrap_resamples"),
                      alpha::Float64 = _config_default("metrics", "ci_alpha"))
    xs = _bootstrap_sample(x)
    length(xs) < 3 && return (NaN, NaN)
    _degenerate(xs) && return (first(xs), first(xs))
    Random.seed!(rand(rng, UInt))
    try
        bs = bootstrap(stat, xs, BasicSampling(n_resamples))
        _, lo, hi = confint(bs, BCaConfInt(1 - alpha))[1]
        return (lo, hi)
    catch
        return (NaN, NaN)
    end
end
_bootstrap_sample(x::AbstractVector{<:Real}) = filter(isfinite, x)
_bootstrap_sample(x::AbstractVector) = x
# A zero-variance jackknife makes the BCa acceleration NaN.
_degenerate(x::AbstractVector{<:Real}) = isapprox(minimum(x), maximum(x))
_degenerate(::AbstractVector) = false

"""
    scene_mean(stat, scene_errors) -> Float64

The equal-weight MEAN over scenes of each scene's `stat` (`scene_errors`
iterates the scenes' value vectors); a scene whose statistic is not finite is
skipped, and the result is `NaN` when no scene has one.

The scene-level aggregation: a per-scene statistic weights scenes, not pairs,
which is what a cross-validation split BY SCENE is asking for. `isfinite` is the
right skip predicate for either statistic this takes — [`finite_median`](@ref)
filters non-finite values first and so returns a finite value or `NaN`, while
[`lower_median`](@ref) does not and can return `±Inf`.
"""
function scene_mean(stat, scene_errors)
    ms = filter(isfinite, [Float64(stat(v)) for v in scene_errors])
    return isempty(ms) ? NaN : mean(ms)
end

"""
    scene_mean_median(scene_errors) -> Float64

Shekhovtsov's "mean median": [`scene_mean`](@ref) of each scene's
[`finite_median`](@ref).
"""
scene_mean_median(scene_errors) = scene_mean(finite_median, scene_errors)

"""
    scene_ranges(sizes) -> Vector{UnitRange{Int}}

Contiguous per-scene index ranges partitioning `1:sum(sizes)` (pairs are laid
out scene-by-scene). `sizes[i]` = number of pairs in scene i.
"""
function scene_ranges(sizes)
    off = 0; rs = UnitRange{Int}[]
    for s in sizes
        push!(rs, (off + 1):(off + Int(s))); off += Int(s)
    end
    return rs
end

"""
    stable_seed(seed, keys...) -> UInt64

The seed derivation of every random sample in the evaluation stack: the first
eight bytes (big-endian) of the SHA-256 of `seed` and `keys` printed as text,
separated by the unit separator `\x1f`. A fixed algorithm, unlike `Base.hash`
(which changes across Julia versions), and keyed by NAME, unlike `seed + i`: a
scene's draw depends on that scene's name only, so adding, removing or
reordering other scenes leaves it unchanged.
"""
stable_seed(seed::Integer, keys...) =
    reduce((acc, b) -> (acc << 8) | UInt64(b), sha256(join((string(seed), string.(keys)...), '\x1f'))[1:8];
           init = zero(UInt64))

"""
    POOL_RNG

The RNG algorithm of every pool stream (a reader's per-scene pair subsample
and the per-pair minimal-sample draws of `write_scene_pool`): StableRNGs'
`StableRNG`, whose streams are fixed across Julia versions, seeded by
[`stable_seed`](@ref). Recorded in each pool file (`pool_rng`).
"""
const POOL_RNG = StableRNG

"""
    scene_mean_median_rows!(rows, method, xs, cols, scene_ranges; seed, ci_min = [metrics] ci_min_scenes)

Shekhovtsov's "Mean median": per `x`, split the per-pair error view `cols[j]`
by `scene_ranges`: point = [`scene_mean_median`](@ref), CI = scene-clustered [`bootstrap_ci`](@ref)`(mean, …)`
over the per-scene medians, resampled under `StableRNG(stable_seed(seed,
"scene_mean_median", j))` (the mAA's with key `"scene_mean_maa"`). Companion statistic: the per-scene
[`eligible_maa`](@ref) (every stored pair is eligible — the pool only holds
processable pairs), averaged over scenes. Schema `(method, tuning_criterion,
n, mean_med, lo95_mean_med, hi95_mean_med, mean_mAA, lo95_mean_mAA,
hi95_mean_mAA)`; NOT pooled over all pairs.
"""
function scene_mean_median_rows!(rows::AbstractVector, method::AbstractString,
                                 xs, cols, scene_ranges; seed::Integer,
                                 ci_min::Integer = _config_default("metrics", "ci_min_scenes"))
    for (j, x) in enumerate(xs)
        v = Float64.(cols[j])
        scenes = [view(v, rng) for rng in scene_ranges]
        sm = filter(!isnan, finite_median.(scenes))     # the per-scene medians the CI resamples
        sa = [eligible_maa(vr) for vr in scenes if !isempty(vr)]
        pt = isempty(sm) ? NaN : mean(sm)                # = scene_mean_median(scenes)
        lo, hi = length(sm) >= ci_min ?
            bootstrap_ci(mean, StableRNG(stable_seed(seed, "scene_mean_median", j)), sm) : (pt, pt)
        pa = isempty(sa) ? NaN : mean(sa)
        loa, hia = length(sa) >= ci_min ?
            bootstrap_ci(mean, StableRNG(stable_seed(seed, "scene_mean_maa", j)), sa) : (pa, pa)
        push!(rows, (method = String(method), tuning_criterion = "median",
                     n = Float64(x), mean_med = pt,
                     lo95_mean_med = lo, hi95_mean_med = hi,
                     mean_mAA = pa, lo95_mean_mAA = loa, hi95_mean_mAA = hia))
    end
    return rows
end

"""
    scale_bounds(s2, q) -> (ŝ_min, ŝ_max)

Robust scale extremes of a validation pool: √ of the `q = (q_lo, q_hi)`
quantiles of the per-pair noise variances `s2` (`σ̂²`, e.g. `gt_variance`;
non-finite entries are dropped). Restricting drawable validation pairs to
`σ̂² ∈ [ŝ_min², ŝ_max²]` makes the plug-in `ŝ(n) = √median σ̂²` over ANY drawn
subset lie in `[ŝ_min, ŝ_max]`, for every `n`. The coverage window rule
(`Tuning.CoverageWindow`, which carries `q`) and the validation-eligible mask
are built from these bounds, so a windowed tuner's window provably stays on
its grid. `q` has no default: it is a parameter of the tuning spec.
"""
function scale_bounds(s2::AbstractVector{<:Real}, q::Tuple{Real,Real})
    v = sort!(filter(isfinite, Float64.(s2)))
    isempty(v) && throw(ArgumentError("scale_bounds: no finite σ̂² in the validation pool"))
    0 <= q[1] <= q[2] <= 1 || throw(ArgumentError("scale_bounds: need 0 ≤ q_lo ≤ q_hi ≤ 1, got $q"))
    return (sqrt(quantile(v, q[1])), sqrt(quantile(v, q[2])))
end

"""
    quantile_or(v, p; empty = NaN) -> Real

`quantile(v, p)`, or `empty` when `v` has no element. The report cells'
quantile: a sample can be empty (every pair failed), and the caller chooses
what that reads as.
"""
quantile_or(v, p; empty = NaN) = isempty(v) ? empty : quantile(v, p)

"""
    ImprovementTrace(offsets, t_ns, hyp, err)

The persisted improvement events of `n` pairs in flat columns: pair `i` owns
the events `offsets[i]:offsets[i+1]-1` (so `length(offsets) == n + 1`), each
with the elapsed time `t_ns` (nanoseconds), the hypothesis count `hyp` (number
of hypotheses drawn so far) and the ground-truth error `err` of the best model
so far. Events of a pair are in time order.
"""
struct ImprovementTrace{O<:AbstractVector{<:Integer},T<:AbstractVector,H<:AbstractVector,E<:AbstractVector}
    offsets::O
    t_ns::T
    hyp::H
    err::E
    function ImprovementTrace(offsets::O, t_ns::T, hyp::H, err::E) where {O<:AbstractVector{<:Integer},T<:AbstractVector,H<:AbstractVector,E<:AbstractVector}
        isempty(offsets) && throw(ArgumentError("ImprovementTrace: offsets must hold n + 1 >= 1 entries"))
        first(offsets) == 1 ||
            throw(ArgumentError("ImprovementTrace: offsets are 1-based (first(offsets) == 1), got $(first(offsets))"))
        issorted(offsets) || throw(ArgumentError("ImprovementTrace: offsets must be non-decreasing"))
        n_events = last(offsets) - first(offsets)
        (length(t_ns) == length(hyp) == length(err) == n_events) ||
            throw(DimensionMismatch("ImprovementTrace: $n_events events by offsets, but t_ns, hyp, err have " *
                                    "$(length(t_ns)), $(length(hyp)), $(length(err))"))
        return new{O,T,H,E}(offsets, t_ns, hyp, err)
    end
end

"Number of pairs of a trace."
Base.length(tr::ImprovementTrace) = length(tr.offsets) - 1

"""
    trace_error_at(tr, i; t_ns = nothing, hyp = nothing) -> Float64

The ground-truth error of pair `i`'s best model once the budget is spent: the
error of its LAST event whose time is at most `t_ns` and whose hypothesis
count is at most `hyp` (a budget left `nothing` does not bind). `Inf` when no
event is within the budget: a pair with no model yet counts as a failure.
"""
function trace_error_at(tr::ImprovementTrace, i::Integer; t_ns = nothing, hyp = nothing)
    e = Inf
    for k in tr.offsets[i]:(tr.offsets[i+1]-1)
        t_ns !== nothing && tr.t_ns[k] > t_ns && break
        hyp !== nothing && tr.hyp[k] > hyp && break
        e = tr.err[k]
    end
    return e
end

"""
    anytime_curve(tr, tgrid; stat = eligible_maa) -> Vector

The accuracy-versus-time curve of a trace: for each budget `t` (seconds) of
`tgrid`, `stat` of the per-pair errors [`trace_error_at`](@ref)`(tr, i; t_ns =
1e9t)`. Pairs with no event by `t` enter as `Inf`, so a failure counts in
`stat` (`eligible_maa` scores it 0; `finite_median` drops it; `e -> count(isfinite,
e)` is the number of pairs reached).
"""
function anytime_curve(tr::ImprovementTrace, tgrid::AbstractVector{<:Real}; stat = eligible_maa)
    n = length(tr)
    e_at = Vector{Float64}(undef, n)
    return map(tgrid) do t
        tns = t * 1e9
        for i in 1:n
            e_at[i] = trace_error_at(tr, i; t_ns = tns)
        end
        stat(copy(e_at))   # a copy: a `stat` that keeps its argument must not alias the buffer
    end
end

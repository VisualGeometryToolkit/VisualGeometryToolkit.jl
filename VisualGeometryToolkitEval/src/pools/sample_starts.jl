# =============================================================================
# src/pools/sample_starts.jl — the minimal-sample boundaries of a pool: which
# stored models came from which minimal sample. The one owner of that
# reconstruction; no other code in the stack may rebuild sample boundaries
# from `sample_indices`.
#
# A format-4 pool stores no per-model sample id. A draw yields its sibling
# models in adjacent columns sharing one `sample_indices` column, and a
# REPEATED draw of the same sample (the samplers draw with replacement across
# trials) appends an identical column run. "A new sample where the column
# changes" therefore merges repeated draws, so a budget of B samples consumes
# more than B. The boundaries are recovered here, exactly, and certified
# against the pool's `total_samples` stamp.
# =============================================================================

# Runs of identical adjacent `sample_indices` columns: (first column, length).
function _column_runs(S::AbstractMatrix{<:Integer})
    m = size(S, 2)
    runs = Tuple{Int,Int}[]
    j = 1
    @inbounds while j <= m
        L = 1
        while j + L <= m && view(S, :, j + L) == view(S, :, j)
            L += 1
        end
        push!(runs, (j, L))
        j += L
    end
    return runs
end

# Whether the models of run (j, L) are `L ÷ k` repeats of one block of k: what
# r = L ÷ k repeated draws of one sample store (a solve is deterministic, so
# every draw of the sample yields the same sibling block).
_periodic(models, j, L, k) = L % k == 0 && all(t -> models[j + t] == models[j + t - k], k:L-1)

# The structural boundaries of one pair: per run, the SMALLEST block that
# repeats — the most draws the record is consistent with. Exact unless one solve
# returned bitwise-identical siblings in a repeating order (a duplicated root);
# `sample_starts` certifies the total against the stamp.
function _structural_starts(pair::PoolPair)
    S, models = pair.sample_indices, pair.models
    st = Int[]
    for (j, L) in _column_runs(S)
        k = findfirst(k -> _periodic(models, j, L, k), 1:L)
        append!(st, j:k:j+L-1)
    end
    return st
end

# Is `st` a sample partition the record admits: starts at 1, increasing, a
# boundary at every column change, and within each run equal blocks that repeat.
function _admissible(pair::PoolPair, st::AbstractVector{<:Integer})
    m = num_models(pair)
    m == 0 && return isempty(st)
    (isempty(st) || first(st) != 1 || !issorted(st; lt = <=) || last(st) > m) && return false
    stset = Set(st)
    for (j, L) in _column_runs(pair.sample_indices)
        j in stset || return false
        inner = [s for s in j:j+L-1 if s in stset]
        k = L ÷ length(inner)
        inner == collect(j:k:j+L-1) && _periodic(pair.models, j, L, k) || return false
    end
    return true
end

function _certify(pool::ScoringPool, starts, how::AbstractString)
    total = sum(length, starts; init = 0)
    total == pool.stamps.total_samples || throw(ArgumentError(
        "sample_starts($(pool.scene)): $how gives $total minimal samples, the pool's total_samples stamp is " *
        "$(pool.stamps.total_samples) — the boundaries are not determined by the record; derive them by " *
        "replaying the pool's seeded streams (derive_sample_starts) and give that file as `derived`"))
    return starts
end

"""
    sample_starts(pool::ScoringPool; derived = nothing) -> Vector{Vector{Int}}

The minimal-sample boundaries of every pair of `pool`: `starts[i]` holds the
first model column of each minimal sample of pair `i`, in draw order, so pair
`i`'s sample `s` owns columns `starts[i][s]:(starts[i][s+1] - 1)` (the last
sample runs to `num_models`). [`sample_ids`](@ref) turns them into a per-model
sample number. Every count of minimal samples in the stack goes through this
function.

Sibling solutions of one draw share a `sample_indices` column, and a REPEATED
draw of the same sample stores the same column again, adjacent when no other
sample was accepted in between. A run of identical columns therefore holds `r`
draws, each storing the same block of siblings (a solve is deterministic): the
record admits every `r` whose block repeats. Without `derived`, each run takes
its largest admissible `r`, and the result is CERTIFIED: its total must equal
the pool's `total_samples` stamp. Since every run's choice is an upper bound on
its true draw count, equal totals make every run exact; a mismatch (a solve that
returned a duplicated root) is an `ArgumentError`.

`derived` is the path of a sample-starts file of this pool
([`derive_sample_starts`](@ref), [`save_sample_starts`](@ref)), read and checked
by [`load_sample_starts`](@ref) instead.
"""
function sample_starts(pool::ScoringPool; derived::Union{Nothing,AbstractString} = nothing)
    derived === nothing || return load_sample_starts(derived, pool)
    return _certify(pool, [_structural_starts(p) for p in pool.pairs], "the column-run structure")
end

"""
    sample_ids(starts, m) -> Vector{Int}

The minimal-sample number (1, 2, …) of each of a pair's `m` models, from that
pair's [`sample_starts`](@ref) `starts`.
"""
function sample_ids(starts::AbstractVector{<:Integer}, m::Integer)
    ok = m == 0 ? isempty(starts) :
         !isempty(starts) && first(starts) == 1 && issorted(starts; lt = <=) && last(starts) <= m
    ok || throw(ArgumentError("sample_ids: $(repr(starts)) are not the sample starts of $m models"))
    ids = Vector{Int}(undef, m)
    for (s, a) in enumerate(starts)
        b = s < length(starts) ? starts[s+1] - 1 : m
        ids[a:b] .= s
    end
    return ids
end

"""
    derive_sample_starts(pool::ScoringPool, stream; max_trials = 10^7) -> Vector{Vector{Int}}

The exact [`sample_starts`](@ref) of `pool` by REPLAYING each pair's seeded
draw stream: `stream(i)` returns a zero-argument function yielding the next
minimal sample (an `AbstractVector{Int}`, the sampler's own order) that pair
`i`'s generator drew, every trial included. For each run of identical columns
the replay counts the draws of that sample before the next stored sample is
drawn; the last run of a pair (nothing follows it in the stream) takes its
structural count. Every pair's result must be admissible for the record and the
total must equal the `total_samples` stamp, or an `ArgumentError` names what
failed. Pairs are replayed under `Threads.@threads`.

Only the pool's own generator knows its stream; the caller supplies it (seed,
sampler and stream index).
"""
function derive_sample_starts(pool::ScoringPool, stream; max_trials::Integer = 10^7)
    n = length(pool.pairs)
    starts = Vector{Vector{Int}}(undef, n)
    errs = Vector{Union{Nothing,String}}(nothing, n)
    Threads.@threads for i in 1:n
        r = _replay_starts(pool.pairs[i], stream(i), max_trials)
        r isa String ? (errs[i] = r) : (starts[i] = r)
    end
    bad = findall(!isnothing, errs)
    isempty(bad) || throw(ArgumentError("derive_sample_starts($(pool.scene)): $(length(bad)) pair(s) not " *
                                        "replayed; first: pair $(bad[1]): $(errs[bad[1]])"))
    return _certify(pool, starts, "the replay")
end

function _replay_starts(pair::PoolPair, draw, max_trials::Integer)
    S = pair.sample_indices
    runs = _column_runs(S)
    st = Int[]
    pending = nothing
    t = 0
    next!() = (d = pending === nothing ? (t += 1; collect(Int, draw())) : pending; pending = nothing; d)
    for (q, (j, L)) in enumerate(runs)
        col = view(S, :, j)
        while next!() != col
            t > max_trials && return "sample at column $j not drawn within $max_trials trials"
        end
        if q < length(runs)
            nxt = view(S, :, runs[q+1][1])
            r = 1
            while true
                d = next!()
                if d == col
                    r += 1
                elseif d == nxt
                    pending = d
                    break
                end
                t > max_trials && return "sample at column $(runs[q+1][1]) not drawn within $max_trials trials"
            end
            L % r == 0 && _periodic(pair.models, j, L, L ÷ r) ||
                return "run at column $j ($L models) does not split into the $r replayed draws"
            append!(st, j:(L ÷ r):j+L-1)
        else
            k = findfirst(k -> _periodic(pair.models, j, L, k), 1:L)
            append!(st, j:k:j+L-1)
        end
    end
    _admissible(pair, st) || return "replayed starts are not admissible"
    return st
end

"""
    save_sample_starts(path, pool::ScoringPool, starts; derivation, provenance = Dict()) -> path

The writer of a sample-starts file (artifact kind `:pool_samples`, format 1),
JLD2, plain data: `pool_key` (the pool it belongs to), `scene`,
`total_samples`, `starts` (`Vector{Vector{Int}}`, per pair, as
[`sample_starts`](@ref) returns them), `derivation` (how they were obtained,
e.g. the replayed stream) and `provenance`. The starts are checked admissible
and certified against the stamp before writing. It is DERIVED data: it lives in
a writable tree, never beside the pool it describes (pool trees are inputs).
"""
function save_sample_starts(path::AbstractString, pool::ScoringPool, starts::AbstractVector;
                            derivation::AbstractString, provenance::AbstractDict = Dict{String,Any}())
    _check_starts(pool, starts, "save_sample_starts")
    mkpath(dirname(abspath(path)))
    return _save_artifact(path, :pool_samples; pool_key = pool.pool_key, scene = pool.scene,
                          total_samples = Int(pool.stamps.total_samples),
                          starts = [Vector{Int}(s) for s in starts], derivation = String(derivation),
                          provenance = Dict{String,Any}(String(k) => v for (k, v) in provenance))
end

function _check_starts(pool::ScoringPool, starts, who)
    length(starts) == length(pool.pairs) || throw(ArgumentError(
        "$who: $(length(starts)) start vectors for $(length(pool.pairs)) pairs"))
    bad = findfirst(i -> !_admissible(pool.pairs[i], starts[i]), eachindex(pool.pairs))
    bad === nothing || throw(ArgumentError("$who: the starts of pair $bad are not admissible for its record"))
    return _certify(pool, starts, who)
end

"""
    load_sample_starts(path, pool::ScoringPool) -> Vector{Vector{Int}}

The reader of a sample-starts file ([`save_sample_starts`](@ref) is the
format), checked against `pool`: refused (an `ArgumentError` naming the file)
unless its `pool_key` is the pool's, every pair's starts are admissible for its
record and their total is the `total_samples` stamp.
"""
function load_sample_starts(path::AbstractString, pool::ScoringPool)
    _check_artifact_format(path, :pool_samples)
    key, starts = JLD2.jldopen(f -> (f["pool_key"], f["starts"]), path, "r")
    key == pool.pool_key || throw(ArgumentError(
        "$path: sample starts for pool_key $key, but the pool ($(pool.scene)) has pool_key $(pool.pool_key)"))
    try
        return _check_starts(pool, starts, "load_sample_starts")
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("$path: $(e.msg)"))
    end
end

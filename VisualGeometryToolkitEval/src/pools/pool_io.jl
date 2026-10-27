# =============================================================================
# src/pools/pool_io.jl — the scoring-pool IO: the in-memory values
# (`ScoringPool`, `PoolResiduals`), the content key `pool_key` that binds a
# light pool file to its residuals file, the one writer and the one reader of
# each, and the per-scene loader closures.
#
# Pool data is in NormalizedImageFrame (`frame(::ScoringPool)`); this file never
# converts a frame. On disk everything is PLAIN data — no stack type is ever
# serialized — so a type change in VisualGeometryToolkitCore never makes a pool
# unreadable (the cause of every pool migration before format 4). Format 5
# renamed one pair key; `migrate_pool` carries a format-4 file over.
# =============================================================================

"The ground-truth pose convention every pool declares (`gt_convention`): the extrinsic relative pose `x₂ = R·x₁ + t`, source → target."
const GT_CONVENTION = "extrinsic"

# The problems a pool can hold, and the model class each stores (E pools store
# the pool-frame fundamental of the essential).
const _POOL_MODEL_CLASS = Dict(:E => "FundMat", :F => "FundMat", :H => "HomographyMat")
_model_type(class::AbstractString) =
    class == "FundMat" ? FundMat{Float64} :
    class == "HomographyMat" ? HomographyMat{Float64} :
    throw(ArgumentError("unknown pool model_class $(repr(class)) (expected \"FundMat\" | \"HomographyMat\")"))
_model_class(::Type{<:FundMat}) = "FundMat"
_model_class(::Type{<:HomographyMat}) = "HomographyMat"

# The residual kinds a residuals file may declare.
const _RESIDUAL_KINDS = (:sampson, :tangent_sampson)

# =============================================================================
# The content key
# =============================================================================

_put!(ctx, x::Int64) = SHA.update!(ctx, reinterpret(UInt8, [htol(x)]))
_put!(ctx, A::AbstractArray{Float64}) = SHA.update!(ctx, reinterpret(UInt8, htol.(vec(collect(A)))))
_put!(ctx, A::AbstractArray{Int64}) = SHA.update!(ctx, reinterpret(UInt8, htol.(vec(collect(A)))))

# The csponds of a record as the stored (4, n) matrix: rows x₁ y₁ x₂ y₂.
function _cspond_matrix(cs)
    src, dst = sources(cs), targets(cs)
    C = Matrix{Float64}(undef, 4, length(src))
    @inbounds for i in eachindex(src, dst)
        C[1, i] = src[i][1]; C[2, i] = src[i][2]; C[3, i] = dst[i][1]; C[4, i] = dst[i][2]
    end
    return C
end

"""
    pool_key(pairs) -> String

The content key of a pool: the lower-case hex SHA-256 of, for each pair in
order,

1. `n = num_points(pair)` and `m = num_models(pair)`, each an `Int64`;
2. the csponds as the stored `(4, n)` `Float64` matrix (rows `x₁ y₁ x₂ y₂`,
   column-major);
3. the models, each its 9 `Float64` entries column-major;
4. `size(sample_indices, 1)` as an `Int64`, then `sample_indices`
   (`Int64`, column-major);

every number little-endian. These are exactly the quantities that index a
residual matrix's rows and columns, so a residuals file written for other
models, another pair order or other correspondences carries another key and is
refused ([`load_pool_residuals`](@ref)). The residuals themselves are never
read to compute it.
"""
function pool_key(pairs::AbstractVector{<:PoolPair})
    ctx = SHA.SHA2_256_CTX()
    for p in pairs
        _put!(ctx, Int64(num_points(p))); _put!(ctx, Int64(num_models(p)))
        _put!(ctx, _cspond_matrix(p.csponds))
        for M in p.models
            _put!(ctx, Matrix{Float64}(SMatrix{3,3,Float64,9}(M)))
        end
        _put!(ctx, Int64(size(p.sample_indices, 1)))
        _put!(ctx, Matrix{Int64}(p.sample_indices))
    end
    return bytes2hex(SHA.digest!(ctx))
end

# =============================================================================
# The in-memory values
# =============================================================================

"""
    ScoringPool{M}

One scene's scoring pool in memory: the light records, without their residual
matrices ([`PoolResiduals`](@ref)).

Fields:
- `pairs::Vector{PoolPair{M}}`: the records (VisualGeometryToolkitCore `PoolPair`), in
  `NormalizedImageFrame` — [`frame`](@ref)`(pool) === NormalizedImageFrame`.
- `problem::Symbol`: `:E`, `:F` or `:H` (E and F pools store `FundMat`, H pools
  `HomographyMat`).
- `scene::String`: the scene the file holds.
- `scene_names::Vector{String}`: each pair's source-scene label (a raw reader's
  `pair_scene`, else `scene`).
- `pool_key::String`: [`pool_key`](@ref)`(pairs)`, computed at construction.
- `stamps::NamedTuple`: `gt_convention`, `pool_rng`, `julia_version`,
  `total_samples`, `total_solutions`, `mean_solutions_per_sample`,
  `provenance` (a `Dict{String,Any}` of plain values).

    ScoringPool(pairs, problem, scene; scene_names, stamps)

builds one from records and computes its key; the model class must be the
problem's.
"""
struct ScoringPool{M}
    pairs::Vector{PoolPair{M}}
    problem::Symbol
    scene::String
    scene_names::Vector{String}
    pool_key::String
    stamps::NamedTuple
end

const _STAMP_KEYS = (:gt_convention, :pool_rng, :julia_version, :total_samples, :total_solutions,
                     :mean_solutions_per_sample, :provenance)

function ScoringPool(pairs::AbstractVector{PoolPair{M}}, problem::Symbol, scene::AbstractString;
                     scene_names::AbstractVector{<:AbstractString}, stamps::NamedTuple) where {M}
    haskey(_POOL_MODEL_CLASS, problem) ||
        throw(ArgumentError("ScoringPool: problem must be :E, :F or :H, got :$problem"))
    _POOL_MODEL_CLASS[problem] == _model_class(M) ||
        throw(ArgumentError("ScoringPool: a :$problem pool stores $(_POOL_MODEL_CLASS[problem]), not $M"))
    length(scene_names) == length(pairs) || throw(DimensionMismatch(
        "ScoringPool: $(length(scene_names)) scene names for $(length(pairs)) pairs"))
    Set(keys(stamps)) == Set(_STAMP_KEYS) ||
        throw(ArgumentError("ScoringPool: stamps must be exactly $(_STAMP_KEYS), got $(keys(stamps))"))
    stamps.gt_convention == GT_CONVENTION ||
        throw(ArgumentError("ScoringPool: gt_convention must be $(repr(GT_CONVENTION)), got $(repr(stamps.gt_convention))"))
    ps = Vector{PoolPair{M}}(pairs)
    return ScoringPool{M}(ps, problem, String(scene), String.(scene_names), pool_key(ps),
                          NamedTuple{_STAMP_KEYS}(map(k -> stamps[k], _STAMP_KEYS)))
end

"""
    frame(pool::ScoringPool) -> Type{NormalizedImageFrame}

The frame a pool's data is stored in: `NormalizedImageFrame` (image centre,
half diagonal, corners at unit radius), per view through each record's maps.
"""
frame(::ScoringPool) = NormalizedImageFrame

Base.length(pool::ScoringPool) = length(pool.pairs)

# The unit a cross-validation samples from a scoring pool: one stored pair.
sampling_unit(::Type{<:PoolPair}) = PairUnit()

"""
    PoolResiduals

The residual matrices of a pool, bound to its light file by `pool_key`.

Fields:
- `pool_key::String`: the [`pool_key`](@ref) of the pool they belong to.
- `kind::Symbol`: the residual they hold — `:sampson` (the plane Sampson
  distance of the pool-frame model; F and H pools) or `:tangent_sampson` (the
  tangent Sampson of the metric essential on the pool-frame E problem; E pools).
- `matrices::Vector{Matrix{Float32}}`: one `(num_points, num_models)` matrix per
  held pair, `|r|` in `NormalizedImageFrame` units.
- `pairs::Vector{Int}`: the pool indices of the held pairs, in the order of
  `matrices` — every pair (`1:n`) unless the residuals were read for a subset
  (`load_pool_residuals(path, pool; pairs)`).
"""
struct PoolResiduals
    pool_key::String
    kind::Symbol
    matrices::Vector{Matrix{Float32}}
    pairs::Vector{Int}
    function PoolResiduals(pool_key::AbstractString, kind::Symbol, matrices::AbstractVector{<:AbstractMatrix{Float32}},
                           pairs::AbstractVector{<:Integer} = eachindex(matrices))
        kind in _RESIDUAL_KINDS ||
            throw(ArgumentError("PoolResiduals: kind must be one of $(_RESIDUAL_KINDS), got :$kind"))
        length(pairs) == length(matrices) ||
            throw(ArgumentError("PoolResiduals: $(length(matrices)) matrices for $(length(pairs)) pair indices"))
        allunique(pairs) || throw(ArgumentError("PoolResiduals: repeated pair index"))
        return new(String(pool_key), kind, Vector{Matrix{Float32}}(matrices), Vector{Int}(pairs))
    end
end

# Whether `res` holds every pair of its pool in order (what a file stores).
_complete(res::PoolResiduals) = res.pairs == eachindex(res.matrices)

"""
    residual_matrix(res::PoolResiduals, i) -> Matrix{Float32}

Pool pair `i`'s residual matrix, `(num_points, num_models)`. `i` is the pair's
index in its POOL, also for residuals read for a subset; a pair that was not
read is refused.
"""
function residual_matrix(res::PoolResiduals, i::Integer)
    _complete(res) && return res.matrices[i]
    k = findfirst(==(i), res.pairs)
    k === nothing && throw(ArgumentError("residual_matrix: pair $i was not read (held: $(res.pairs))"))
    return res.matrices[k]
end

"""
    residuals_path(path) -> String

The residuals file of the light pool file `path`: `<scene>.jld2` →
`<scene>.residuals.jld2`.
"""
function residuals_path(path::AbstractString)
    endswith(path, ".jld2") || throw(ArgumentError("residuals_path: $path is not a .jld2 pool file"))
    return path[1:end-5] * ".residuals.jld2"
end

# =============================================================================
# The light file
# =============================================================================

const _PAIR_FIELDS = (:csponds, :K1, :K2, :R_gt, :t_gt, :models, :sample_indices, :errors_r, :errors_t,
                      :gt_inliers, :gt_variance, :source_map, :destination_map, :snn)

# One record as the plain on-disk NamedTuple, and back.
_plain_pair(p::PoolPair) = NamedTuple{_PAIR_FIELDS}((
    _cspond_matrix(p.csponds),
    p.K1, p.K2, p.R_gt, p.t_gt,
    [SMatrix{3,3,Float64,9}(M) for M in p.models],
    Matrix{Int}(p.sample_indices),
    Vector{Float64}(p.errors_r), Vector{Float64}(p.errors_t),
    BitVector(p.gt_inliers), Float64(p.gt_variance),
    p.source_map, p.destination_map,
    p.snn === nothing ? nothing : Vector{Float32}(p.snn),
))

function _record(::Type{M}, nt, path, i) where {M}
    issetequal(keys(nt), _PAIR_FIELDS) ||
        throw(ArgumentError("$path: pair $i has fields $(keys(nt)), expected $(_PAIR_FIELDS)"))
    C = nt.csponds
    size(C, 1) == 4 || throw(ArgumentError("$path: pair $i csponds must be (4, n), got $(size(C))"))
    cs = csponds([SVector{2,Float64}(C[1, k], C[2, k]) for k in axes(C, 2)],
                 [SVector{2,Float64}(C[3, k], C[4, k]) for k in axes(C, 2)])
    models = M[M(Tuple(SMatrix{3,3,Float64,9}(m))) for m in nt.models]
    return PoolPair{M}(; csponds = cs, nt.K1, nt.K2, nt.R_gt, nt.t_gt, models, nt.sample_indices,
                       nt.errors_r, nt.errors_t, nt.gt_inliers, nt.gt_variance,
                       nt.source_map, nt.destination_map, nt.snn)
end

"""
    save_pool(path, pool::ScoringPool) -> path

The writer of a light pool file (artifact kind `:pool`, format 5). The file is
JLD2 and holds PLAIN data only — no VisualGeometryToolkitCore type:

| key | type | meaning |
|---|---|---|
| `pool_format` | `Int` | `5` |
| `problem` | `String` | `"E"`, `"F"` or `"H"` |
| `scene` | `String` | the scene the file holds |
| `model_class` | `String` | `"FundMat"` (E, F) or `"HomographyMat"` (H) |
| `frame` | `String` | `"NormalizedImageFrame"` |
| `gt_convention` | `String` | `"extrinsic"` (`x₂ = R·x₁ + t`, source → target) |
| `pool_key` | `String` | [`pool_key`](@ref) of the pairs (hex SHA-256) |
| `scene_names` | `Vector{String}` | per-pair source-scene label |
| `pool_rng` | `String` | RNG algorithm of the pool streams (`"LehmerRNG"`, the type of StableRNGs' `StableRNG`) |
| `julia_version` | `String` | the Julia that generated the pool |
| `total_samples`, `total_solutions` | `Int` | accepted minimal samples and real solutions over the scene |
| `mean_solutions_per_sample` | `Float64` | their ratio (0 when no sample) |
| `provenance` | `Dict{String,Any}` | the generating run's record (plain values) |
| `pairs` | `Vector` of NamedTuples | one per pair, below |

Each pair is a NamedTuple with exactly these fields (n correspondences, m
models, s the minimal sample size), every point datum and model in
`NormalizedImageFrame`:

| field | type | meaning |
|---|---|---|
| `csponds` | `Matrix{Float64}` (4, n) | rows `x₁ y₁ x₂ y₂` |
| `K1`, `K2` | `SMatrix{3,3,Float64,9}` | PIXEL calibrations |
| `R_gt` | `SMatrix{3,3,Float64,9}` | GT rotation, source → target |
| `t_gt` | `SVector{3,Float64}` | GT translation (unnormalized) |
| `models` | `Vector{SMatrix{3,3,Float64,9}}` (m) | pool-frame model matrices of class `model_class` |
| `sample_indices` | `Matrix{Int}` (s, m) | 1-based minimal sample of each model |
| `errors_r`, `errors_t` | `Vector{Float64}` (m) | rotation / translation error [deg] vs GT |
| `gt_inliers` | `BitVector` (n) | GT inlier mask |
| `gt_variance` | `Float64` | σ̂² in pool units² (NaN: none) |
| `source_map`, `destination_map` | `SMatrix{3,3,Float64,9}` | pixel → `NormalizedImageFrame` affine maps (last row `[0 0 1]`) |
| `snn` | `Vector{Float32}` (n) or `nothing` | per-correspondence SNN the hypotheses were sampled by |

The residual matrices are a separate file ([`save_pool_residuals`](@ref)).
"""
function save_pool(path::AbstractString, pool::ScoringPool{M}) where {M}
    mkpath(dirname(abspath(path)))
    st = pool.stamps
    return _save_artifact(path, :pool;
        problem = String(pool.problem), scene = pool.scene, model_class = _model_class(M),
        frame = string(nameof(frame(pool))), gt_convention = st.gt_convention,
        pool_key = pool.pool_key, scene_names = pool.scene_names,
        pool_rng = String(st.pool_rng), julia_version = String(st.julia_version),
        total_samples = Int(st.total_samples), total_solutions = Int(st.total_solutions),
        mean_solutions_per_sample = Float64(st.mean_solutions_per_sample),
        provenance = Dict{String,Any}(String(k) => v for (k, v) in st.provenance),
        pairs = [_plain_pair(p) for p in pool.pairs])
end

const _POOL_FILE_KEYS = ("problem", "scene", "model_class", "frame", "pool_key", "scene_names",
                         String.(_STAMP_KEYS)..., "pairs", _format_key(:pool))

"""
    load_pool(path) -> ScoringPool

The reader of a light pool file ([`save_pool`](@ref) is the format). Refuses,
with an `ArgumentError` naming the file: another `pool_format` (an older pool
is migrated, never read: the error says how — [`migrate_pool`](@ref) for a
format-4 file), missing or unknown keys, a frame
other than `NormalizedImageFrame`, a ground-truth convention other than
[`GT_CONVENTION`](@ref), a model class that is not the problem's, and a stored
`pool_key` that the loaded pairs do not reproduce.
"""
function load_pool(path::AbstractString)
    _check_artifact_format(path, :pool)
    return _pool_from_raw(JLD2.load(path), path)
end

# The pool of a light file's raw contents, every check of `load_pool` applied.
function _pool_from_raw(raw::AbstractDict, path::AbstractString)
    Set(keys(raw)) == Set(_POOL_FILE_KEYS) ||
        throw(ArgumentError("$path: pool keys $(sort!(collect(keys(raw)))), expected $(sort!(collect(_POOL_FILE_KEYS)))"))
    raw["frame"] == "NormalizedImageFrame" ||
        throw(ArgumentError("$path: a pool is stored in NormalizedImageFrame, this one declares $(repr(raw["frame"]))"))
    raw["gt_convention"] == GT_CONVENTION ||
        throw(ArgumentError("$path: gt_convention $(repr(raw["gt_convention"])), expected $(repr(GT_CONVENTION))"))
    problem = Symbol(raw["problem"])
    M = _model_type(raw["model_class"])
    pairs = PoolPair{M}[_record(M, nt, path, i) for (i, nt) in enumerate(raw["pairs"])]
    stamps = (gt_convention = String(raw["gt_convention"]), pool_rng = String(raw["pool_rng"]),
              julia_version = String(raw["julia_version"]), total_samples = Int(raw["total_samples"]),
              total_solutions = Int(raw["total_solutions"]),
              mean_solutions_per_sample = Float64(raw["mean_solutions_per_sample"]),
              provenance = Dict{String,Any}(raw["provenance"]))
    pool = try
        ScoringPool(pairs, problem, raw["scene"]; scene_names = raw["scene_names"], stamps)
    catch e
        e isa Union{ArgumentError,DimensionMismatch} || rethrow()
        throw(ArgumentError("$path: $(sprint(showerror, e))"))
    end
    pool.pool_key == raw["pool_key"] ||
        throw(ArgumentError("$path: the stored pool_key $(raw["pool_key"]) is not the key of its pairs " *
                            "($(pool.pool_key)) — the file is corrupt or was edited"))
    return pool
end

# =============================================================================
# migrate_pool — a format-4 light pool file into format 5
# =============================================================================
#
# Format 5 renamed one pair key, the destination map's: its former name is
# retired from the stack, so the migration finds that key by its PLACE in the
# format-4 pair (the slot `destination_map` holds in format 5; every other key
# is format 5's), never by name. By VisualGeometryToolkitIO's migration pattern
# (`write_staged`) it READS the format-4 file, WRITES the format-5 file to a
# staging directory beside `out` through the writer, READS it back through the
# reader, COMPARES every stored value with what it read — and only then, unless
# it is a dry run, renames it into `out`. The original is never touched.

const _DESTINATION_SLOT = findfirst(==(:destination_map), _PAIR_FIELDS)

# A format-4 pair under format 5's keys: the same values in the same order.
function _format5_pair(nt::NamedTuple, path, i)
    k = keys(nt)
    ok = length(k) == length(_PAIR_FIELDS) && k[_DESTINATION_SLOT] ∉ _PAIR_FIELDS &&
         all(k[j] == _PAIR_FIELDS[j] for j in eachindex(_PAIR_FIELDS) if j != _DESTINATION_SLOT)
    ok || throw(ArgumentError("$path: pair $i has fields $k, not a format-4 pool pair's"))
    return NamedTuple{_PAIR_FIELDS}(values(nt))
end

# Bit-for-bit equality of one stored value (a NaN equals itself).
_same_bits(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(a .=== b)
_same_bits(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(_same_bits(x, y) for (x, y) in zip(a, b))
_same_bits(a::AbstractDict, b::AbstractDict) = isequal(a, b)
_same_bits(a, b) = a === b || isequal(a, b)

"""
    migrate_pool(path; out, dry_run = false) -> NamedTuple

Carry a format-4 light pool file over to format 5, whose pair key for the
destination map is `destination_map`. The format-5 file goes to
`joinpath(out, basename(path))`; `out` must be another directory than the
original's, which is never touched. The pool's residuals file
([`residuals_path`](@ref)), when present, is copied beside it unchanged: it is
bound to the pool by [`pool_key`](@ref), which hashes no map, so it stays
valid. A sample-starts file ([`save_sample_starts`](@ref)) is NOT copied: it
is derived data kept in a writable tree, never beside the pool, so the pool's
path does not name it; it is bound by the same unchanged `pool_key` and reads
against the migrated pool where it is. Both destinations, the pool's and its
residuals', are checked before anything is written: an existing one is an
`ArgumentError`, and a migration that fails leaves `out` as it was: a
residuals file already renamed into it is removed, and so is an `out` the
migration created.

The migration is verified before anything is written to `out`: the format-5
file, written through [`save_pool`](@ref) to a staging directory beside `out`
(a dry run's: a temporary one), must read back through [`load_pool`](@ref)'s
checks, hold the same keys, and store every value of the original bit for bit;
only then are the residuals copy and the pool renamed into `out`, the pool
last. With `dry_run` nothing is written. A file of another
format is refused with an `ArgumentError`: a format-5 one needs nothing, an
older one is converted first (see [`load_pool`](@ref)'s refusal). Returns
`(; path, residuals, n_pairs)`, the output paths `nothing` in a dry run.
"""
function migrate_pool(path::AbstractString; out::AbstractString, dry_run::Bool = false)
    isfile(path) || throw(ArgumentError("$path: no such pool artifact"))
    cur = ARTIFACT_FORMATS.pool.version
    v = JLD2.jldopen(f -> _artifact_format(f, :pool), path, "r")
    v == cur && throw(ArgumentError("$path: already a format-$cur pool; nothing to migrate"))
    v == 4 || throw(ArgumentError("$path: pool format v$v; migrate_pool reads format 4 — $(ARTIFACT_FORMATS.pool.remedy)"))
    abspath(out) == dirname(abspath(path)) && throw(ArgumentError(
        "migrate_pool: `out` is the original's directory ($(abspath(out))); the original is never overwritten — name another"))
    dst = joinpath(abspath(out), basename(path))
    r, rdst = residuals_path(path), residuals_path(dst)
    for d in (dst, rdst)
        ispath(d) && throw(ArgumentError("migrate_pool: $d exists; nothing was written"))
    end
    raw = JLD2.load(path)
    pairs = [_format5_pair(nt, path, i) for (i, nt) in enumerate(raw["pairs"])]
    pool = _pool_from_raw(merge(raw, Dict{String, Any}("pairs" => pairs)), path)
    has_r = isfile(r)
    # renamed into `out` once verified, the residuals first and the pool last:
    # a pool file at `dst` is a whole migration
    outs = has_r ? (rdst => basename(rdst), dst => basename(dst)) : (dst => basename(dst),)
    write_staged(outs...; dry_run) do staged...
        t = save_pool(last(staged), pool)
        _check_artifact_format(t, :pool)
        back = JLD2.load(t)
        _pool_from_raw(back, t)                                        # reads back through the reader's checks
        _verify(ok, what) = ok || throw(
            ArgumentError("$path: the format-5 file does not read back as the file read ($what); nothing was written"))
        _verify(Set(keys(back)) == Set(keys(raw)), "its keys")
        for k in keys(raw)
            k in ("pairs", _format_key(:pool)) && continue
            _verify(_same_bits(raw[k], back[k]), k)
        end
        _verify(length(back["pairs"]) == length(pairs), "its pair count")
        for (i, (a, b)) in enumerate(zip(raw["pairs"], back["pairs"]))
            _verify(keys(b) == _PAIR_FIELDS && all(_same_bits(x, y) for (x, y) in zip(values(a), values(b))), "pair $i")
        end
        (has_r && !dry_run) && cp(r, first(staged))
    end
    dry_run && return (; path = nothing, residuals = nothing, n_pairs = length(pairs))
    return (; path = dst, residuals = has_r ? rdst : nothing, n_pairs = length(pairs))
end

# =============================================================================
# The residuals file
# =============================================================================

"""
    save_pool_residuals(path, res::PoolResiduals) -> path

The writer of a residuals file (artifact kind `:pool_residuals`, format 1),
JLD2, PLAIN data only:

| key | type | meaning |
|---|---|---|
| `pool_residuals_format` | `Int` | `1` |
| `pool_key` | `String` | the [`pool_key`](@ref) of the light file it belongs to |
| `residual_kind` | `String` | `"sampson"` (F, H) or `"tangent_sampson"` (E) |
| `residuals` | `Vector{Matrix{Float32}}` | per pair, `(n, m)`: `|r|` of model j at cspond i, `NormalizedImageFrame` units |

By convention the file of `<scene>.jld2` is `<scene>.residuals.jld2`
([`residuals_path`](@ref)).
"""
function save_pool_residuals(path::AbstractString, res::PoolResiduals)
    _complete(res) ||
        throw(ArgumentError("save_pool_residuals: residuals read for a subset of pairs ($(res.pairs)); " *
                            "a residuals file holds every pair of its pool"))
    mkpath(dirname(abspath(path)))
    return _save_artifact(path, :pool_residuals; pool_key = res.pool_key,
                          residual_kind = String(res.kind), residuals = res.matrices)
end

"""
    load_pool_residuals(path, pool::ScoringPool; pairs = nothing) -> PoolResiduals

The reader of a residuals file ([`save_pool_residuals`](@ref) is the format),
checked against the light `pool` it serves: refused (an `ArgumentError` naming
the file) unless its `pool_key` is the pool's — the error names both keys — and
it holds one matrix per pair, each `(num_points, num_models)` of that pair.

`pairs` (pool indices) reads only those pairs' matrices, sliced from the file
without reading the rest; [`residual_matrix`](@ref) still takes pool indices.
"""
function load_pool_residuals(path::AbstractString, pool::ScoringPool;
                             pairs::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    _check_artifact_format(path, :pool_residuals)
    n = length(pool.pairs)
    idx = pairs === nothing ? collect(1:n) : collect(Int, pairs)
    all(i -> 1 <= i <= n, idx) ||
        throw(ArgumentError("$path: pair indices $(idx) outside 1:$n (the pool's pairs)"))
    key = JLD2.jldopen(f -> f["pool_key"], path, "r")
    key == pool.pool_key ||
        throw(ArgumentError("$path: residuals for pool_key $key, but the pool ($(pool.scene)) has " *
                            "pool_key $(pool.pool_key) — not this pool's residuals"))
    # A full read checks the count; a subset read slices the stored vector
    # (the key above already binds the file to exactly this pool's pairs).
    kind, mats = JLD2.jldopen(path, "r") do f
        (f["residual_kind"], pairs === nothing ? f["residuals"] :
                             (ds = JLD2.get_dataset(f, "residuals"); [ds[i] for i in idx]))
    end
    length(mats) == length(idx) ||
        throw(ArgumentError("$path: $(length(mats)) residual matrices for $n pairs"))
    for (R, i) in zip(mats, idx)
        p = pool.pairs[i]
        size(R) == (num_points(p), num_models(p)) ||
            throw(ArgumentError("$path: residual matrix $i is $(size(R)), pair $i needs " *
                                "$((num_points(p), num_models(p))) (num_points × num_models)"))
    end
    return PoolResiduals(key, Symbol(kind), mats, idx)
end

# =============================================================================
# Per-scene loaders
# =============================================================================

"""
    scene_loaders(pool_dir, scenes; residuals = false) -> Vector{Function}

Zero-argument scene loaders over `<pool_dir>/<scene>.jld2` — the loader
constructor shared by every streaming driver. Each returns every pair of its
scene (a subset is a cross-validation's units, Tuning `validation_units` /
`test_units`, never a prefix): the [`ScoringPool`](@ref), or with
`residuals = true` the tuple `(pool, residuals::PoolResiduals)` read from
`<scene>.residuals.jld2` and checked against it.
"""
scene_loaders(pool_dir::AbstractString, scenes; residuals::Bool = false) =
    [let s = String(sid), path = joinpath(pool_dir, "$s.jld2")
         residuals ?
             () -> (@info "loading $s"; pool = load_pool(path); (pool, load_pool_residuals(residuals_path(path), pool))) :
             () -> (@info "loading $s"; load_pool(path))
     end for sid in scenes]

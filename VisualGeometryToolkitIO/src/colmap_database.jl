# =============================================================================
# colmap_database.jl — read access to a COLMAP sqlite database (database.db).
#
# Layer-1 wire surface: unit-free, format-faithful. A COLMAP database stores
# per-image keypoints (rows×cols float32 row-major; cols ∈ {2,4,6}: x, y
# [, scale, orientation | affine]), descriptors (rows×128 uint8), and pairwise
# matches (rows×2 uint32 KEYPOINT-INDEX pairs, 0-BASED — add 1 to index the
# keypoints matrix) keyed by
#
#     pair_id = image_id1 * 2147483647 + image_id2,   image_id1 ≤ image_id2
#
# (COLMAP swaps the ids — and the match columns with them — when the stored
# order disagrees with the caller's). Geometrically verified matches live in
# `two_view_geometries` (current COLMAP) or `inlier_matches` (pre-3.x schema,
# e.g. the Aachen Day-Night benchmark database); both are handled.
#
# The handle is LAZY: nothing is materialized beyond the query you run — a
# benchmark database (Aachen: 19 GB) must never be slurped whole. Keypoint
# coordinates are COLMAP-convention pixels; units attach downstream at the
# layer-2 boundary — the wire layer stays unit-free like its siblings.
#
# Only `ColmapDatabase` is exported; accessors are deliberately unexported
# (generic names — qualify: `VisualGeometryToolkitIO.keypoints(db, id)`), mirroring
# the unexported load/save verbs.
#
# SQLite rows are LIVE CURSORS: every field is extracted inside the iteration
# (`_query_one` takes an extractor closure) — a row captured past its
# iteration is not guaranteed valid.
# =============================================================================

const DBInterface = SQLite.DBInterface

const _COLMAP_MAX_IMAGE_ID = Int64(2147483647)

"""
    ColmapDatabase(path) -> ColmapDatabase

Open a lazy read handle on a COLMAP sqlite database. Close with `close(db)`.

Accessors (unexported — qualify with the module):
`n_images`, `image_names`, `image_id`, `image_name`, `keypoints`,
`descriptors`, `matches`, `matched_pairs`, `verified_matches`, `pair_id`, `pair_image_ids`.
"""
struct ColmapDatabase
    db::SQLite.DB
    verified_table::Union{String, Nothing}
end

function ColmapDatabase(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    db = SQLite.DB(path)
    tables = String[]
    for r in DBInterface.execute(
            db,
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        push!(tables, String(r.name))
    end
    verified = "two_view_geometries" in tables ? "two_view_geometries" :
        "inlier_matches" in tables ? "inlier_matches" : nothing
    return ColmapDatabase(db, verified)
end

Base.close(cdb::ColmapDatabase) = DBInterface.close!(cdb.db)

# --- pair-id codec ----------------------------------------------------------

"""
    pair_id(image_id1, image_id2) -> Int64

COLMAP's pair key: ids are put in ascending order first, so the codec is
order-insensitive (as in COLMAP's `image_pair_to_pair_id`).
"""
function pair_id(image_id1::Integer, image_id2::Integer)
    i, j = image_id1 <= image_id2 ? (image_id1, image_id2) :
        (image_id2, image_id1)
    return Int64(i) * _COLMAP_MAX_IMAGE_ID + Int64(j)
end

"""
    pair_image_ids(pair_id) -> (image_id1, image_id2)

Inverse of [`pair_id`](@ref); `image_id1 ≤ image_id2` by construction.
"""
function pair_image_ids(pid::Integer)
    i, j = divrem(Int64(pid), _COLMAP_MAX_IMAGE_ID)
    return (Int(i), Int(j))
end

# --- query plumbing ---------------------------------------------------------

# Run `sql`, apply `f` to the FIRST row while the cursor is live, return the
# result; `nothing` when the query has no rows.
function _query_one(f, cdb::ColmapDatabase, sql::String, params)
    for r in DBInterface.execute(cdb.db, sql, params)
        return f(r)
    end
    return nothing
end

# COLMAP blobs are row-major rows×cols arrays of a fixed scalar type.
function _decode_blob(
        ::Type{T}, blob::AbstractVector{UInt8},
        rows::Integer, cols::Integer
    ) where {T}
    nb = Int(rows) * Int(cols) * sizeof(T)
    length(blob) == nb || throw(
        ArgumentError(
            "blob is $(length(blob)) bytes; expected $nb for $(rows)×$(cols) $T"
        )
    )
    flat = reinterpret(T, blob)
    return Matrix{T}(permutedims(reshape(flat, Int(cols), Int(rows))))
end

function _blob_table(
        cdb::ColmapDatabase, ::Type{T}, table::String,
        image_id::Integer, default_width::Int
    ) where {T}
    row = _query_one(
        cdb,
        "SELECT rows, cols, data FROM $table WHERE image_id = ?",
        (Int(image_id),)
    ) do r
        (
            rows = Int(r.rows),
            cols = r.cols === missing ? default_width : Int(r.cols),
            data = r.data === missing ? nothing : Vector{UInt8}(r.data),
        )
    end
    row === nothing && throw(ArgumentError("no $table row for image_id $image_id"))
    (row.data === nothing || row.rows == 0) &&
        return Matrix{T}(undef, 0, row.cols)
    return _decode_blob(T, row.data, row.rows, row.cols)
end

# --- image table ------------------------------------------------------------

n_images(cdb::ColmapDatabase) =
    _query_one(r -> Int(r.n), cdb, "SELECT COUNT(*) AS n FROM images", ())

"""Image names ordered by `image_id`."""
function image_names(cdb::ColmapDatabase)
    names = String[]
    for r in DBInterface.execute(
            cdb.db,
            "SELECT name FROM images ORDER BY image_id"
        )
        push!(names, String(r.name))
    end
    return names
end

function image_id(cdb::ColmapDatabase, name::AbstractString)
    id = _query_one(
        r -> Int(r.image_id), cdb,
        "SELECT image_id FROM images WHERE name = ?",
        (String(name),)
    )
    id === nothing && throw(ArgumentError("no image named $(repr(name))"))
    return id
end

function image_name(cdb::ColmapDatabase, image_id::Integer)
    name = _query_one(
        r -> String(r.name), cdb,
        "SELECT name FROM images WHERE image_id = ?",
        (Int(image_id),)
    )
    name === nothing && throw(ArgumentError("no image with id $image_id"))
    return name
end

# --- per-image blobs --------------------------------------------------------

"""
    keypoints(cdb, image_id) -> Matrix{Float32}   (n × cols)

Row `i` is keypoint `i-1` in COLMAP's 0-based match indexing. Columns are
`x, y[, scale, orientation | affine params]` in pixels (COLMAP convention:
the origin is the upper-left corner of the upper-left pixel, so the CENTER
of the first pixel is (0.5, 0.5)).
"""
keypoints(cdb::ColmapDatabase, image_id::Integer) =
    _blob_table(cdb, Float32, "keypoints", image_id, 4)

"""
    descriptors(cdb, image_id) -> Matrix{UInt8}   (n × 128)

Row `i` is the descriptor of keypoint `i` (same order as [`keypoints`](@ref)).
"""
descriptors(cdb::ColmapDatabase, image_id::Integer) =
    _blob_table(cdb, UInt8, "descriptors", image_id, 128)

# --- matches ----------------------------------------------------------------

function _matches_from(
        cdb::ColmapDatabase, table::String,
        id1::Integer, id2::Integer
    )
    pid = pair_id(id1, id2)
    row = _query_one(
        cdb,
        "SELECT rows, data FROM $table WHERE pair_id = ?",
        (pid,)
    ) do r
        (
            rows = Int(r.rows),
            data = r.data === missing ? nothing : Vector{UInt8}(r.data),
        )
    end
    (row === nothing || row.data === nothing || row.rows == 0) &&
        return Matrix{UInt32}(undef, 0, 2)
    m = _decode_blob(UInt32, row.data, row.rows, 2)
    # stored order is (min(id1,id2), max(id1,id2)); swap columns to the
    # caller's order when it disagrees
    return id1 <= id2 ? m : m[:, [2, 1]]
end

"""
    matched_pairs(cdb, image_id) -> Vector{Int}

Ids of the images that share a NONEMPTY raw-match row with `image_id`
(either stored order of the pair), ascending. The per-query fan-out
primitive: partners are enumerated from the pair-id index, not by probing
every image.
"""
function matched_pairs(cdb::ColmapDatabase, image_id::Integer)
    ids = Int[]
    lo = Int64(image_id) * _COLMAP_MAX_IMAGE_ID
    hi = lo + _COLMAP_MAX_IMAGE_ID - 1
    for r in DBInterface.execute(
            cdb.db,
            "SELECT pair_id FROM matches WHERE rows > 0 AND " *
                "(pair_id BETWEEN ? AND ? OR pair_id % $(_COLMAP_MAX_IMAGE_ID) = ?)",
            (lo, hi, Int(image_id))
        )
        i, j = pair_image_ids(Int64(r.pair_id))
        push!(ids, i == Int(image_id) ? j : i)
    end
    return sort!(ids)
end

"""
    matches(cdb, image_id1, image_id2) -> Matrix{UInt32}   (n × 2)

Raw (unverified) keypoint-index match pairs, 0-BASED, column 1 indexing
`image_id1`'s keypoints and column 2 `image_id2`'s — in the CALLER's argument
order regardless of how COLMAP stored the pair. No entry ⇒ 0×2.
"""
matches(cdb::ColmapDatabase, image_id1::Integer, image_id2::Integer) =
    _matches_from(cdb, "matches", image_id1, image_id2)

"""
    verified_matches(cdb, image_id1, image_id2) -> Matrix{UInt32}   (n × 2)

Geometrically verified match pairs, from `two_view_geometries` (current
COLMAP) or `inlier_matches` (pre-3.x databases). Same conventions as
[`matches`](@ref). Errors if the database has neither table.
"""
function verified_matches(
        cdb::ColmapDatabase,
        image_id1::Integer, image_id2::Integer
    )
    cdb.verified_table === nothing && throw(
        ArgumentError(
            "database has neither two_view_geometries nor inlier_matches"
        )
    )
    return _matches_from(cdb, cdb.verified_table, image_id1, image_id2)
end

# =============================================================================
# Wire types — unit-free, format-faithful records (Layer 1)
# =============================================================================
#
# These mirror the file formats exactly and carry bare Float64 magnitudes.
# Interpretation (units, frames) is Layer 2's job; conversion to VGC types is
# Layer 3's. Byte-faithful round trips are tested against pycolmap.

# --- COLMAP camera model zoo --------------------------------------------------
# (model_id, name, number of params). Params order follows COLMAP:
# https://colmap.github.io/cameras.html
const COLMAP_MODELS = (
    (0, :SIMPLE_PINHOLE, 3),          # f, cx, cy
    (1, :PINHOLE, 4),                 # fx, fy, cx, cy
    (2, :SIMPLE_RADIAL, 4),           # f, cx, cy, k
    (3, :RADIAL, 5),                  # f, cx, cy, k1, k2
    (4, :OPENCV, 8),                  # fx, fy, cx, cy, k1, k2, p1, p2
    (5, :OPENCV_FISHEYE, 8),          # fx, fy, cx, cy, k1, k2, k3, k4
    (6, :FULL_OPENCV, 12),            # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    (7, :FOV, 5),                     # fx, fy, cx, cy, omega
    (8, :SIMPLE_RADIAL_FISHEYE, 4),   # f, cx, cy, k
    (9, :RADIAL_FISHEYE, 5),          # f, cx, cy, k1, k2
    (10, :THIN_PRISM_FISHEYE, 12),    # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
    (11, :RAD_TAN_THIN_PRISM_FISHEYE, 16),   # fx, fy, cx, cy, k0…k5, p0, p1, s0…s3
    (12, :SIMPLE_DIVISION, 4),        # f, cx, cy, k
    (13, :DIVISION, 5),               # fx, fy, cx, cy, k
    (14, :SIMPLE_FISHEYE, 3),         # f, cx, cy
    (15, :FISHEYE, 4),                # fx, fy, cx, cy
)

function _model_entry(key::Union{Symbol, Integer})
    col = key isa Symbol ? 2 : 1
    i = findfirst(m -> m[col] == key, COLMAP_MODELS)
    i === nothing && throw(ArgumentError("unknown COLMAP camera model $key"))
    return COLMAP_MODELS[i]
end

model_id(name::Symbol) = _model_entry(name)[1]
model_name(id::Integer) = _model_entry(id)[2]
model_nparams(name::Symbol) = _model_entry(name)[3]

"""
    ColmapCamera

One row of COLMAP's `cameras` table: intrinsics in COLMAP's pixel convention
(pixel centers at 0.5), `params` ordered per the model zoo. Unit-free wire
type — magnitudes are px by COLMAP's definition, the only image unit (the
contract declares the world unit alone).
"""
struct ColmapCamera
    camera_id::Int32
    model::Symbol
    width::Int64
    height::Int64
    params::Vector{Float64}

    function ColmapCamera(camera_id, model, width, height, params)
        n = model_nparams(model)
        length(params) == n || throw(
            ArgumentError(
                "$model expects $n params, got $(length(params))"
            )
        )
        return new(camera_id, model, width, height, collect(Float64, params))
    end
end

"""
    ColmapImage

One row of COLMAP's `images` table: the world→camera pose as a wxyz quaternion
`qvec` + translation `tvec` (world units per contract; plain numbers in the
reconstruction's own scale under `UnknownScale`), plus observed 2D
points `(x, y, point3D_id)` with `point3D_id == -1` when untracked.
"""
struct ColmapImage
    image_id::Int32
    qvec::SVector{4, Float64}   # w, x, y, z — world→camera
    tvec::SVector{3, Float64}
    camera_id::Int32
    name::String
    points2D::Vector{Tuple{Float64, Float64, Int64}}

    function ColmapImage(image_id, qvec, tvec, camera_id, name, points2D)
        # Refused at construction because the two formats DISAGREE on it: the
        # text writer puts NAME last on the line, so an empty name emits a
        # 9-field row the reader cannot parse back, while binary round-trips
        # it to "". A wire type whose formats diverge is worse than one that
        # refuses the input.
        isempty(name) && throw(
            ArgumentError(
                "COLMAP image name must be non-empty — the text format cannot " *
                    "represent it (NAME is the last field on the line)"
            )
        )
        return new(image_id, qvec, tvec, camera_id, name, points2D)
    end
end

ColmapImage(image_id, qvec, tvec, camera_id, name) =
    ColmapImage(image_id, qvec, tvec, camera_id, name, Tuple{Float64, Float64, Int64}[])

"""
    ColmapPoint3D

One row of COLMAP's `points3D` table. `track` holds `(image_id, point2D_idx)`
pairs.
"""
struct ColmapPoint3D
    point3D_id::Int64
    xyz::SVector{3, Float64}
    rgb::SVector{3, UInt8}
    error::Float64
    track::Vector{Tuple{Int32, Int32}}
end

"""
    ColmapModel

A COLMAP model directory's contents plus the unit contract its magnitudes are
expressed in: a [`UnitContract`](@ref), or [`UnknownScale`](@ref) for a
structure-from-motion model whose world has no unit.
"""
struct ColmapModel
    cameras::Vector{ColmapCamera}
    images::Vector{ColmapImage}
    points3D::Vector{ColmapPoint3D}
    contract::Union{UnitContract, UnknownScale}
end

# Canonical quaternion sign: w ≥ 0 (both signs encode the same rotation; fixing
# one makes round trips byte-comparable).
canonical_qvec(q::SVector{4, Float64}) = q[1] < 0 ? -q : q

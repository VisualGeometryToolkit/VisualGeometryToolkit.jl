# =============================================================================
# Schöps et al.'s camera_calibration files — load-only
# =============================================================================
#
# puzzlepaint/camera_calibration (Schöps, Larsson, Pollefeys, Sattler, CVPR
# 2020) writes two files a calibration run is read back from:
#
#   intrinsics0.yaml   one generic camera model — `CentralGenericModel` (a
#                      direction grid) or `NoncentralGenericModel` (a point and
#                      a direction grid) — with the calibrated area;
#   dataset.bin        the feature detections, `calib_data` version 0: per
#                      image, (x, y, feature id) of every detected feature.
#
# They are read through the typed `load` of the VGC type they hold, the same
# verb as the vgio JSON form of that type: `load(CentralGenericModel, path)`,
# `load(NoncentralGenericModel, path; file_unit)`, `load(TargetPoints2D, path;
# target)`. Which form a file is, is decided by its CONTENT, never its name:
# a vgio file is a JSON object (its first non-blank byte is `{`), a
# `dataset.bin` starts with the magic `calib_data`, and anything else is read
# as Schöps' YAML. Each form refuses a file that is not what it reads.
#
# THE PIXEL FRAME. Schöps' convention puts the origin at the top-left CORNER
# of the top-left pixel, so the centre of pixel (i, j) is (i + ½, j + ½):
# that IS `PixelFrame{:colmap}` (pixel-origins.md), the identity
# `EuclideanMap(PixelFrame{:colmap}, PixelFrame{:colmap})` — no half pixel is
# added anywhere. `SCHOEPS_PIXEL_FRAME` names it once: the generic models are
# defined in it (VGC's own types are COLMAP-frame by construction), and the
# detections come back typed in it (`TargetPoints2D{SCHOEPS_PIXEL_FRAME}`),
# so moving them to another convention is `convert_frame`, explicitly.
#
# What the files do NOT declare is the reader's argument, as for the OpenCV and
# MATLAB exports (external_calibration.jl):
#   file_unit   the length unit of a non-central model's `point_grid` — the
#               unit of the calibration pattern's geometry, which the YAML
#               does not record;
#   target      the target the detections' feature ids index (one pattern
#               per file), and the ids' 0-based numbering becomes the
#               1-based `index` of a `TargetPoint2D` (`index = id + 1`).
# =============================================================================

"""
    SCHOEPS_PIXEL_FRAME

The pixel frame of Schöps et al.'s camera_calibration files: origin at the
top-left corner of the top-left pixel, so pixel `(i, j)`'s centre is
`(i + ½, j + ½)` — `PixelFrame{:colmap}` exactly (the identity
`EuclideanMap(PixelFrame{:colmap}, PixelFrame{:colmap})`).
"""
const SCHOEPS_PIXEL_FRAME = PixelFrame{:colmap}

# The form of a file, by its content: `:json` (a vgio file), `:calib_data` (a
# Schöps dataset.bin) or `:yaml` (anything else: a Schöps intrinsics YAML).
function _file_form(path::AbstractString)
    open(path, "r") do io
        magic = read(io, 10)
        magic == codeunits("calib_data") && return :calib_data
        for b in magic
            b in (0x20, 0x09, 0x0a, 0x0d) && continue
            return b == UInt8('{') ? :json : :yaml
        end
        while !eof(io)
            b = read(io, UInt8)
            b in (0x20, 0x09, 0x0a, 0x0d) && continue
            return b == UInt8('{') ? :json : :yaml
        end
        return :yaml
    end
end

# -----------------------------------------------------------------------------
# intrinsics0.yaml
# -----------------------------------------------------------------------------

# One `key : value` line of the YAML (the reference writes exactly this shape).
function _schoeps_field(yml::AbstractString, key::AbstractString, path)
    mt = match(Regex("^$(key)\\s*:\\s*(.*?)\\s*\$", "m"), yml)
    mt === nothing && throw(ArgumentError("$(path): no `$(key)` — not a Schöps camera_calibration intrinsics file"))
    return mt.captures[1]
end
function _schoeps_int(yml, key, path)
    v = tryparse(Int, _schoeps_field(yml, key, path))
    v === nothing && throw(ArgumentError("$(path): `$(key)` is not an integer"))
    return v
end

# The `gw × gh` grid of 3-vectors stored under `tag`, row-major (entry
# y·gw + x, each x, y, z), as VGC's `[gy, gx]` matrix.
function _schoeps_grid(yml::AbstractString, tag::AbstractString, gw::Int, gh::Int, path; scale = identity)
    mt = match(Regex("^$(tag)\\s*:\\s*\\[(.*?)\\]", "ms"), yml)
    mt === nothing && throw(ArgumentError("$(path): no `$(tag)` grid"))
    nums = Float64[]
    for s in split(mt.captures[1], r"[,\s]+"; keepempty = false)
        v = tryparse(Float64, s)
        (v === nothing || !isfinite(v)) && throw(ArgumentError("$(path): `$(tag)` holds `$(s)`, not a finite number"))
        push!(nums, v)
    end
    length(nums) == 3 * gw * gh || throw(
        ArgumentError("$(path): `$(tag)` has $(length(nums)) numbers, expected 3 × grid_width × grid_height = $(3gw * gh)")
    )
    g = Matrix{Vec3{Float64}}(undef, gh, gw)
    for gy in 1:gh, gx in 1:gw
        k = 3 * ((gy - 1) * gw + (gx - 1))
        g[gy, gx] = Vec3{Float64}(scale(nums[k + 1]), scale(nums[k + 2]), scale(nums[k + 3]))
    end
    return g
end

function _load_schoeps_model(::Type{T}, path::AbstractString, file_unit) where {T}
    yml = read(path, String)
    type = _schoeps_field(yml, "type", path)
    type == string(nameof(T)) || throw(ArgumentError("$(path) holds a Schöps $(type), not a $(nameof(T))"))
    hdr = map(k -> _schoeps_int(yml, k, path), (
        "width", "height", "calibration_min_x", "calibration_min_y", "calibration_max_x", "calibration_max_y",
    ))
    gw, gh = _schoeps_int(yml, "grid_width", path), _schoeps_int(yml, "grid_height", path)
    T === CentralGenericModel && return CentralGenericModel(hdr..., _schoeps_grid(yml, "grid", gw, gh, path))
    file_unit === nothing && throw(
        ArgumentError(
            "$(path): a Schöps NoncentralGenericModel does not record the length unit of its point_grid " *
                "(the calibration pattern's); state it: `file_unit = m`"
        )
    )
    dimension(file_unit) == dimension(m) || throw(ArgumentError("$(path): `file_unit` must be a length unit, got $(file_unit)"))
    dirs = _schoeps_grid(yml, "direction_grid", gw, gh, path)
    pts = _schoeps_grid(yml, "point_grid", gw, gh, path; scale = x -> _read_power(x, file_unit, 1))   # plain metres
    return NoncentralGenericModel(hdr..., dirs, pts)
end

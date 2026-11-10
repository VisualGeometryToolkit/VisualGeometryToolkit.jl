# =============================================================================
# Schöps et al.'s camera_calibration files — load-only
# =============================================================================
#
# puzzlepaint/camera_calibration (Schöps, Larsson, Pollefeys, Sattler, CVPR
# 2020) writes two files a calibration run is read back from:
#
#   intrinsics0.yaml   one grid camera — central (a direction grid) or
#                      non-central (a point and a direction grid) — with the
#                      calibrated area;
#   dataset.bin        the feature detections, `calib_data` version 0: per
#                      image, (x, y, feature id) of every detected feature.
#
# They are read through the typed `load` of the VGC type they hold, the same
# verb as the vgio JSON form of that type: `load(BoundedCamera, path; file_unit)`
# (a central model as a camera with a `SplineLens`, a non-central one as a
# `NoncentralSplineModel`), `load(TargetPoints2D, path;
# target)`. Which form a file is, is decided by its CONTENT, never its name:
# a vgio file is a JSON object (its first non-blank byte is `{`), a
# `dataset.bin` starts with the magic `calib_data`, and anything else is read
# as Schöps' YAML. Each form refuses a file that is not what it reads.
#
# THE PIXEL FRAME. Schöps' convention puts the origin at the top-left CORNER
# of the top-left pixel, so the centre of pixel (i, j) is (i + ½, j + ½):
# that IS `PixelFrame{:colmap}` (pixel-origins.md), the identity
# `EuclideanMap(PixelFrame{:colmap}, PixelFrame{:colmap})` — no half pixel is
# added anywhere. `SCHOEPS_PIXEL_FRAME` names it once: the spline models are
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
# Schöps dataset.bin) or `:yaml` (anything else: a Schöps intrinsics YAML). A
# UTF-8 byte-order mark is skipped first.
function _file_form(path::AbstractString)
    open(path, "r") do io
        bom = read(io, 3)
        bom == UInt8[0xef, 0xbb, 0xbf] || seekstart(io)
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

# A YAML file's text, read once, without a leading UTF-8 byte-order mark — the
# one `_file_form` skips, so every reader sees the same text.
_yaml_text(path::AbstractString) = _strip_bom(read(path, String))

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
    g = Matrix{SVector{3, Float64}}(undef, gh, gw)
    for gy in 1:gh, gx in 1:gw
        k = 3 * ((gy - 1) * gw + (gx - 1))
        g[gy, gx] = SVector{3, Float64}(scale(nums[k + 1]), scale(nums[k + 2]), scale(nums[k + 3]))
    end
    return g
end

# The YAML's model as a bounded camera: a central model (its `type` the
# reference's central model, its `grid`) as the SplineLens camera of its lattice,
# a non-central one (its `direction_grid` and `point_grid`) as a
# `NoncentralSplineModel` on that camera's lattice and K (Core's
# `CameraModel(SplineLens, controls, size)`: the lattice's gauge, the YAML carrying
# no pinhole). Every node is present; the directions are normalized unless unit to
# 1e-12, as the reference's reader does.
function _load_schoeps_camera(path::AbstractString, file_unit)
    yml = _yaml_text(path)
    type = _schoeps_field(yml, "type", path)
    type in (_OLD_CENTRAL_TYPE, _OLD_NONCENTRAL_TYPE) ||
        throw(ArgumentError("$(path) holds a Schöps $(type), not a grid camera ($(_OLD_CENTRAL_TYPE), $(_OLD_NONCENTRAL_TYPE))"))
    central = type == _OLD_CENTRAL_TYPE
    w, h, bounds... = map(k -> _schoeps_int(yml, k, path), (
        "width", "height", "calibration_min_x", "calibration_min_y", "calibration_max_x", "calibration_max_y",
    ))
    gw, gh = _schoeps_int(yml, "grid_width", path), _schoeps_int(yml, "grid_height", path)
    if central
        file_unit === nothing || throw(ArgumentError("$(path): a central grid camera has no lengths; `file_unit` does not apply"))
    else
        file_unit === nothing && throw(
            ArgumentError(
                "$(path): a Schöps non-central model does not record the length unit of its point_grid " *
                    "(the calibration pattern's); state it: `file_unit = m`"
            )
        )
        dimension(file_unit) == dimension(m) || throw(ArgumentError("$(path): `file_unit` must be a length unit, got $(file_unit)"))
    end
    dirs = _schoeps_grid(yml, central ? "grid" : "direction_grid", gw, gh, path)
    cam = CameraModel(SplineLens, dirs, (w, h); calib_area = Tuple(bounds), unit_controls = true)
    central && return BoundedCamera(cam, Size2(w, h))
    pts = _schoeps_grid(yml, "point_grid", gw, gh, path; scale = x -> _read_power(x, file_unit, 1))   # plain metres
    d = cam.lens.backward
    return BoundedCamera(NoncentralSplineModel(cam.intrinsics, d, ChartSpline(d.area, pts)), Size2(w, h))
end

# -----------------------------------------------------------------------------
# The target layouts' own YAML files: Schöps' pattern YAML, Kalibr's target YAML
# -----------------------------------------------------------------------------
#
# Read through the typed `load` of the target they define, the same verb as the
# vgio target file (`load(PlanarTarget, path)`, `load(AbstractTarget, path)`),
# the form decided by the content (`_file_form`, then the YAML's keys: Kalibr's
# `target_type`, Schöps' `squares_x`). Each returns VGC's
# `PlanarTarget(layout; canvas)`, its layout recorded, so `save` writes it as a
# vgio/target `layout` record.
#
# Both files are the flat YAML their generators write: top-level `key: value`
# lines, and in Schöps' a `page` map and an `apriltags` list of maps — its
# items indented, or at column 0 under their key as PyYAML's default dump
# writes them (`apriltags:` then `- tag_x: 13`, the item's further keys
# indented by 2). Comments are YAML's: a `#` that begins the line or follows
# whitespace, outside quotes (Kalibr's stock `tagCols: 6   #number of
# apriltags`). `_layout_yaml` reads that shape (and no other YAML) into
# strings, from the text `_yaml_text` reads.
function _layout_yaml(yml::AbstractString, path)
    top = Dict{String, String}()
    maps = Dict{String, Dict{String, String}}()
    lists = Dict{String, Vector{Dict{String, String}}}()
    opened = Set{String}()   # the blocks (`key:` with no value) not yet given a line
    section = nothing
    keyval(s) = (mt = match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$", s);
        mt === nothing && throw(ArgumentError("$(path): `$(s)` is not a `key: value` line")); (String(something(mt[1])), _yaml_unquoted(something(mt[2]))))
    for raw in split(yml, '\n')
        line = rstrip(_yaml_uncommented(raw))
        isempty(strip(line)) && continue
        item_at_column_0 = section !== nothing && startswith(line, "- ")
        if !item_at_column_0 && !startswith(line, ' ') && !startswith(line, '\t')
            k, v = keyval(line)
            (haskey(top, k) || k in opened || haskey(maps, k) || haskey(lists, k)) && throw(ArgumentError("$(path): `$(k)` is stated twice"))
            isempty(v) ? (section = k; push!(opened, k)) : (section = nothing; top[k] = v)
            continue
        end
        section === nothing && throw(ArgumentError("$(path): the indented line `$(strip(line))` belongs to no key"))
        s = strip(line)
        # a block's first line says what it is: a list of maps (`- `) or a map
        if section in opened
            delete!(opened, section)
            startswith(s, "- ") ? (lists[section] = Dict{String, String}[]) : (maps[section] = Dict{String, String}())
        end
        if startswith(s, "- ")
            haskey(lists, section) || throw(ArgumentError("$(path): `$(section)` mixes a map and a list"))
            push!(lists[section], Dict{String, String}())
            s = strip(s[3:end])
        end
        k, v = keyval(s)
        d = haskey(lists, section) ? last(lists[section]) : maps[section]
        haskey(d, k) && throw(ArgumentError("$(path): `$(section).$(k)` is stated twice"))
        d[k] = v
    end
    for k in opened   # a block with no line is an empty list (`apriltags:` with no tag)
        lists[k] = Dict{String, String}[]
    end
    return (; top, maps, lists)
end
_yaml_unquoted(s) = (t = String(s); length(t) >= 2 && t[1] == t[end] && t[1] in ('\'', '"') ? t[2:(end - 1)] : t)
# A line without its comment: from a `#` at the line's start or after
# whitespace, outside single and double quotes (`'aprilgrid' #gridtype`).
function _yaml_uncommented(line::AbstractString)
    quote_char = nothing
    prev = ' '
    for (i, c) in pairs(line)
        if quote_char !== nothing
            c == quote_char && (quote_char = nothing)
        elseif (c == '\'' || c == '"') && (isspace(prev) || prev == ':')   # a quoted scalar opens
            quote_char = c
        elseif c == '#' && isspace(prev)
            return SubString(line, firstindex(line), prevind(line, i))
        end
        prev = c
    end
    return line
end
function _yaml_get(d::AbstractDict, key, path, what)
    haskey(d, key) || throw(ArgumentError("$(path): no `$(key)` — not $(what)"))
    return d[key]
end
# A value read as an `Int` or a finite `Float64`.
function _yaml_parse(::Type{T}, d, key, path, what) where {T <: Union{Int, Float64}}
    v = tryparse(T, _yaml_get(d, key, path, what))
    (v === nothing || !isfinite(v)) &&
        throw(ArgumentError("$(path): `$(key)` is $(repr(d[key])), not $(T === Int ? "an integer" : "a finite number")"))
    return v
end
function _yaml_only(d, keys, path, what)
    extra = setdiff(Base.keys(d), keys)
    isempty(extra) || throw(ArgumentError("$(path): $(what) holds $(join(("`$(k)`" for k in sort!(collect(extra))), ", ")), which this reader does not know"))
    return nothing
end

# A target layout YAML (the content says which), as its PlanarTarget.
function _load_layout_yaml(path::AbstractString)
    y = _layout_yaml(_yaml_text(path), path)
    haskey(y.top, "target_type") && return _load_kalibr_target(y, path)
    haskey(y.top, "squares_x") && return _load_schoeps_pattern(y, path)
    throw(
        ArgumentError(
            "$(path): neither a Kalibr target YAML (`target_type`) nor a Schöps pattern YAML (`squares_x`) — " *
                "a vgio target file is JSON"
        )
    )
end

# Kalibr's target YAML (kalibr_common/ConfigReader.py, CalibrationTargetParameters):
# an `aprilgrid` is `tagCols`, `tagRows`, `tagSize` (metres) and `tagSpacing`,
# the gap as a RATIO of the tag size — VGC's AprilGrid takes the gap as a
# length, `tagSize · tagSpacing`. Kalibr's grids are tag36h11 from id 0. A
# checkerboard or circle grid is not a layout VGC holds, and is refused.
const _KALIBR_APRILGRID = "a Kalibr aprilgrid target YAML"
function _load_kalibr_target(y, path)
    ty = y.top["target_type"]
    ty == "aprilgrid" || throw(ArgumentError("$(path): a Kalibr $(repr(ty)) target; only an `aprilgrid` is a VGC layout (AprilGrid)"))
    (isempty(y.maps) && isempty(y.lists)) || throw(ArgumentError("$(path): $(_KALIBR_APRILGRID) is flat `key: value` lines"))
    _yaml_only(y.top, ("target_type", "tagCols", "tagRows", "tagSize", "tagSpacing"), path, _KALIBR_APRILGRID)
    tag_size = read_length(_yaml_parse(Float64, y.top, "tagSize", path, _KALIBR_APRILGRID), m)
    _yaml_parse(Float64, y.top, "tagSpacing", path, _KALIBR_APRILGRID)   # a finite number
    # Kalibr's ratio → the gap, the product of the two decimals as written,
    # rounded once to the metre value (as `read_length` reads a decimal): a
    # float product would round twice (0.088 · 0.3 = 0.026399999999999996)
    ratio = _decimal_rational(y.top["tagSpacing"], path, "tagSpacing")
    gap = _metre_quantity(Float64(_decimal_rational(y.top["tagSize"], path, "tagSize") * ratio))
    grid = AprilGrid(;
        tag_rows = _yaml_parse(Int, y.top, "tagRows", path, _KALIBR_APRILGRID), tag_cols = _yaml_parse(Int, y.top, "tagCols", path, _KALIBR_APRILGRID),
        tag_size, tag_spacing = gap,
    )
    return PlanarTarget(grid)
end
# A decimal number's text as the exact rational it denotes ("0.088" → 11//125).
function _decimal_rational(s::AbstractString, path, key)
    mt = match(r"^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$", s)
    (mt === nothing || isempty(something(mt[2], "")) && isempty(something(mt[3], ""))) &&
        throw(ArgumentError("$(path): `$(key)` is $(repr(s)), not a decimal number"))
    frac = something(mt[3], "")
    digits = parse(BigInt, string(something(mt[2], ""), frac, "0"))   # the trailing 0 keeps an empty part a number
    ex = mt[4]   # bound once, so the `nothing` check narrows it (JET: a second `mt[4]` is a fresh Union)
    e = (ex === nothing ? 0 : parse(Int, ex)) - length(frac) - 1
    r = e >= 0 ? Rational{BigInt}(digits * big(10)^e) : digits // big(10)^(-e)
    return mt[1] == "-" ? -r : r
end

# Schöps' pattern YAML (create_calibration_pattern.py): `num_star_segments`,
# `squares_x`, `squares_y`, `square_length_in_meters`, the `apriltags` (each
# `tag_x`, `tag_y`, `width`, `height` in squares, y down as printed, and its
# `index`, the family id) and the `page` it is printed on (`width_mm`,
# `height_mm` and the pattern's rectangle on it, `pattern_start_*_mm` to
# `pattern_end_*_mm`, from the page's top-left corner, y down). The page is the
# target's CANVAS: VGC's design chart has the pattern's lower-left corner at
# the origin, y up, so the page's lower-left corner is at
# (−pattern_start_x, pattern_end_y − height). The pattern's rectangle is the
# layout's own (`squares × square_length`, checked to 1 nm); the file states no
# board (the physical board the page is mounted on), so the target has none.
const _SCHOEPS_PATTERN = "a Schöps pattern YAML"
const _SCHOEPS_PAGE_KEYS = ("width_mm", "height_mm", "pattern_start_x_mm", "pattern_start_y_mm", "pattern_end_x_mm", "pattern_end_y_mm")
function _load_schoeps_pattern(y, path)
    _yaml_only(y.top, ("num_star_segments", "squares_x", "squares_y", "square_length_in_meters"), path, _SCHOEPS_PATTERN)
    _yaml_only(y.maps, ("page",), path, _SCHOEPS_PATTERN)
    _yaml_only(y.lists, ("apriltags",), path, _SCHOEPS_PATTERN)
    tags = map(get(y.lists, "apriltags", Dict{String, String}[])) do t
        _yaml_only(t, ("tag_x", "tag_y", "width", "height", "index"), path, "an apriltag of $(_SCHOEPS_PATTERN)")
        f(k) = _yaml_parse(Int, t, k, path, "an apriltag of $(_SCHOEPS_PATTERN)")
        (x = f("tag_x"), y = f("tag_y"), width = f("width"), height = f("height"), id = f("index"))
    end
    pattern = SchoepsPattern(;
        squares_x = _yaml_parse(Int, y.top, "squares_x", path, _SCHOEPS_PATTERN), squares_y = _yaml_parse(Int, y.top, "squares_y", path, _SCHOEPS_PATTERN),
        square_length = read_length(_yaml_parse(Float64, y.top, "square_length_in_meters", path, _SCHOEPS_PATTERN), m),
        star_segments = _yaml_parse(Int, y.top, "num_star_segments", path, _SCHOEPS_PATTERN), apriltags = tags,
    )
    haskey(y.maps, "page") || return PlanarTarget(pattern)
    page = y.maps["page"]
    _yaml_only(page, _SCHOEPS_PAGE_KEYS, path, "the page of $(_SCHOEPS_PATTERN)")
    W, H, sx, sy, ex, ey = (read_length(_yaml_parse(Float64, page, k, path, "the page of $(_SCHOEPS_PATTERN)"), mm) for k in _SCHOEPS_PAGE_KEYS)
    for (what, a, b, n) in (("x", sx, ex, pattern.squares_x), ("y", sy, ey, pattern.squares_y))
        abs(ustrip(m, (b - a) - n * pattern.square_length)) <= 1.0e-9 || throw(
            ArgumentError(
                "$(path): the page's pattern spans $(ustrip(mm, b - a)) mm in $(what), not the $(n) squares of " *
                    "$(ustrip(mm, pattern.square_length)) mm the pattern states"
            )
        )
    end
    return PlanarTarget(pattern; canvas = Canvas(Size2(width = W, height = H), SVector(-sx, ey - H)))
end

"""
    load(PlanarTarget, path; resolve = nothing) -> PlanarTarget | Attributed{<:PlanarTarget}

A planar target from a `vgio/target` file (as [`load`](@ref)`(AbstractTarget,
path)`, refusing a points target), or from the layout file of the tool that
generated it, decided by its content:

- Schöps et al.'s pattern YAML (`create_calibration_pattern.py`:
  `squares_x`, …, `apriltags`, `page`) — `PlanarTarget(SchoepsPattern(…);
  canvas)`, the page its canvas;
- Kalibr's target YAML of an `aprilgrid` (`target_type: aprilgrid`, `tagCols`,
  `tagRows`, `tagSize` in metres, `tagSpacing` the gap over the tag size) —
  `PlanarTarget(AprilGrid(…))`, the gap `tagSize · tagSpacing`.

Either way the target records its layout, so `save` writes it as one.
"""
function load(::Type{PlanarTarget}, path::AbstractString; resolve = nothing)
    t = load(AbstractTarget, path; resolve)
    geom_t = t isa Attributed ? t.value : t
    geom_t isa PlanarTarget || throw(ArgumentError("$(path) holds a $(nameof(typeof(geom_t))), not a PlanarTarget"))
    return t
end

# -----------------------------------------------------------------------------
# dataset.bin (calib_data, version 0)
# -----------------------------------------------------------------------------
#
# MIXED ENDIANNESS, as the reference writes it: every u32/i32 big-endian
# (htonl), every float32 in the machine's (little-endian) order.

_be_u32(io) = Int(ntoh(read(io, UInt32)))
_be_i32(io) = Int(ntoh(read(io, Int32)))

# A length read from the file is checked against the bytes left before
# anything is read or allocated by it: a corrupt length is an ArgumentError
# naming the field, never a short read or an EOFError deeper in.
function _check_left(io, nbytes::Integer, path, what)
    left = filesize(io) - position(io)
    (0 <= nbytes <= left) || throw(ArgumentError("$(path): truncated or corrupt calib_data: $(what) needs $(nbytes) bytes, $(left) left"))
    return nothing
end

function _load_schoeps_dataset(path::AbstractString, target::Symbol)
    frames = Dict{Int, Vector{TargetPoint2D}}()
    images = Dict{Int, String}()
    open(path, "r") do io
        _check_left(io, 10 + 5 * 4, path, "the header")
        String(read(io, 10)) == "calib_data" || throw(ArgumentError("$(path): not a Schöps calib_data file"))
        version = _be_u32(io)
        version == 0 || throw(ArgumentError("$(path): calib_data version $(version); version 0 is read"))
        ncam = _be_u32(io)
        ncam == 1 || throw(ArgumentError("$(path): $(ncam) cameras; one camera's detections are read"))
        _be_u32(io); _be_u32(io)                       # the camera's width, height
        nsets = _be_u32(io)
        _check_left(io, 8 * nsets, path, "$(nsets) image sets")
        for t in 1:nsets
            _check_left(io, 4, path, "image set $(t)'s name length")
            nname = _be_u32(io)
            _check_left(io, nname + 4, path, "image set $(t)'s name ($(nname) bytes) and feature count")
            images[t] = String(read(io, nname))
            nf = _be_u32(io)
            _check_left(io, 12 * nf, path, "image $(images[t])'s $(nf) features")
            points = Vector{TargetPoint2D}(undef, nf)
            for k in 1:nf
                x = Float64(ltoh(read(io, Float32))); y = Float64(ltoh(read(io, Float32))); id = _be_i32(io)
                (isfinite(x) && isfinite(y)) || throw(ArgumentError("$(path): image $(images[t]): a non-finite feature position"))
                id >= 0 || throw(ArgumentError("$(path): image $(images[t]): negative feature id $(id)"))
                points[k] = (target = target, index = id + 1, xy = SVector{2, Float64}(x, y), weight = 1.0)
            end
            frames[t] = points
        end
        # The known pattern geometries that may follow are the pattern's, not
        # the detections'; they are not read.
    end
    return TargetPoints2D{SCHOEPS_PIXEL_FRAME}(frames; images)
end

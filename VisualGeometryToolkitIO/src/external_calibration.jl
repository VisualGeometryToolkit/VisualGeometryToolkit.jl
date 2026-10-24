# =============================================================================
# Third-party calibration exports — OpenCV and MATLAB, load-only
# =============================================================================
#
# A capture's baselines arrive as `calibration_opencv_<model>.json` and
# `calibration_matlab_<model>.json`: the exporter's intrinsics and distortion in
# its own vocabulary, one reprojection statistic, and one `{rvec, tvec}` target
# pose per image (`null` where its solver gave up). Two things a third-party
# file does not declare are the reader's arguments — statements about the FILE,
# never guesses and never labels on in-memory data:
#
#   file_frame    the pixel frame of `cx, cy`. OpenCV's is `PixelFrame{:opencv}`
#                 (top-left pixel centre at (0, 0)), MATLAB's documented one
#                 `PixelFrame{:julia}` (centre at (1, 1)) — the defaults, by the
#                 user's rule; a fitted principal point cannot tell the two
#                 apart (the fits of one capture differ by under a pixel).
#   file_unit     the length unit of `tvec`: the pattern's world unit, which
#                 OpenCV and MATLAB both inherit from whoever typed the object
#                 points. In one capture the OpenCV export is in metres and the
#                 MATLAB export in millimetres — a factor of a thousand that
#                 looks plausible either way in a hand-eye.
#
# The OpenCV shape as THIS package writes it (`save_opencv_calibration`) does
# declare both, as does a third-party `tvec_unit`, and the reader holds the
# caller to them: the caller still states them, a file that declares something else is
# refused — as `load(ColmapModel, dir; contract)` refuses a contract its
# `vgio_units.json` sidecar contradicts — and a file that declares nothing (a
# third-party export) is read exactly as the caller says.
#
# Everything returned is in the stack's conventions: the camera in the COLMAP
# pixel frame, the poses `target → camera_optical` in millimetres. The lens formulas
# agree on both sides — an OpenCV file's coefficients are OpenCV's distortion
# vector (`opencv_lens`, conventions.jl); MATLAB's radial/tangential model is
# OpenCV's polynomial term for term, so `opencv_lens([r1, r2, t1, t2, r3])` IS
# the MATLAB camera; MATLAB's fisheye is Scaramuzza's polynomial lift, VGC's
# `Scaramuzza` once its focal (`a0`), stretch matrix and distortion centre are
# moved into `K`.

"""
    ExternalCalibration{C <: CameraModel}

One third-party calibration file as read by [`load_opencv_calibration`](@ref)
or [`load_matlab_calibration`](@ref): `source` (`:opencv` or `:matlab`),
`camera` (a VGC `CameraModel`, in the canonical frame by construction),
`poses` (one VGC `EuclideanMap` labeled `TargetFrame{target} → CameraFrame`,
in millimetres, per image; `nothing` where the file has `null`), the file's `reprojection_error` in pixels together with the
statistic it is (`error_statistic`: `:rms` for OpenCV, `:mean` for MATLAB),
`path`, and `images`: the image name of every pose when the file names them —
this package's writer does — or `nothing` for a third-party export, whose
positional join with the images is then the caller's.

A FOREIGN-FORMAT ADAPTER — the in-memory image of a third-party file, which
VGIo reads and does not own — so it keeps its own type; everything it holds
is a VGC value.
"""
struct ExternalCalibration{C <: CameraModel}
    source::Symbol
    camera::C
    poses::Vector{Union{Nothing, EuclideanMap{3}}}
    reprojection_error::Float64
    error_statistic::Symbol
    path::String
    images::Union{Nothing, Vector{String}}
end

# A calibration whose file names no images (every third-party export).
ExternalCalibration(source, camera, poses, reprojection_error, error_statistic, path) =
    ExternalCalibration(source, camera, poses, reprojection_error, error_statistic, path, nothing)

const _CAMERA_OPTICAL = frame_name(CameraFrame)          # "camera_optical"

# A required key, or a message naming the file and the key rather than a
# KeyError from deep inside the parsed object.
function _need(d, key::Symbol, path, what)
    haskey(d, key) || throw(
        ArgumentError("$(path): no `$(key)` — not a $(what) calibration export")
    )
    return d[key]
end

# fx, fy, cx, cy → a camera in the COLMAP pixel frame, through `CameraModel`'s own
# origin shift (the one place the half pixel is applied).
function _external_camera(d, lens, image_origin::Symbol, path, what)
    fx = Float64(_need(d, :fx, path, what)::Real)
    fy = Float64(_need(d, :fy, path, what)::Real)
    cx = Float64(_need(d, :cx, path, what)::Real)
    cy = Float64(_need(d, :cy, path, what)::Real)
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(fx, 0, 0, 0, fy, 0, cx, cy, 1))
    return CameraModel(K; lens, image_origin)
end

# `{rvec, tvec}` per image → `target → camera_optical` in millimetres; a `null`
# stays `nothing` so the caller's positional join over the images holds.
# `twist` re-expresses the poses in a camera frame rotated about the optical
# axis (the MATLAB fisheye's stretch matrix, below); the identity otherwise.
function _external_poses(
        entries, unit::Units, target::AbstractString, path;
        twist::RotMatrix{3, Float64} = one(RotMatrix{3, Float64}),
    )
    dimension(unit) == dimension(m) || throw(
        ArgumentError("$(path): `unit` must be the length unit of `tvec` (mm, m, …), got $(unit)")
    )
    scale = ustrip(mm, quantity(1.0, unit))       # mm per `unit`
    return Union{Nothing, EuclideanMap{3}}[
        p === nothing ? nothing : _external_pose(p, scale, target, twist) for p in entries
    ]
end
function _external_pose(p, scale::Float64, target::AbstractString, twist::RotMatrix{3, Float64})
    r = SVector{3, Float64}(Float64.(collect(p.rvec)))
    t = SVector{3, Float64}(Float64.(collect(p.tvec))) .* scale
    # OpenCV's rvec is a Rodrigues (rotation-vector) parameterization, which is
    # `Rotations.RotationVec` exactly; MATLAB's export uses the same vector.
    R = RotMatrix{3, Float64}(RotationVec(r[1], r[2], r[3]))
    return relabel(TargetFrame{Symbol(target)}, CameraFrame, EuclideanMap(twist * R, quantity.(twist * t, mm)))
end

# S = U·Q — an RQ factorization of a 2×2 stretch matrix: `U` upper triangular
# with a positive diagonal, `Q` the rotation by `φ` (returned). Closed form:
# the second row of U·Q is u₂₂·(sin φ, cos φ).
function _rq2(S::SMatrix{2, 2, Float64, 4})
    φ = atan(S[2, 1], S[2, 2])
    c, s = cos(φ), sin(φ)
    u11 = S[1, 1] * c - S[1, 2] * s
    u12 = S[1, 1] * s + S[1, 2] * c
    u22 = hypot(S[2, 1], S[2, 2])
    u11 > 0 || throw(ArgumentError("stretch matrix $(S) is not a positive stretch of the sensor plane"))
    return SMatrix{2, 2, Float64, 4}(u11, 0.0, u12, u22), φ
end

"""
    save_opencv_calibration([to_frame = PixelFrame{:opencv},] path, camera, poses; rms, to_unit, images)
    save_opencv_calibration([to_frame = PixelFrame{:opencv},] path, cal::ExternalCalibration; to_unit, images = cal.images)

Write an OpenCV-shaped calibration export — THE INVERSE of
[`load_opencv_calibration`](@ref), and the only writer for that shape, so the
two cannot drift: `fx, fy, cx, cy`, the twelve distortion coefficients by name
(`k1`–`k6`, `p1`, `p2`, `s1`–`s4`, each written even when zero, which is what
makes a default, rational or thin-prism fit one file shape), `rms` in pixels,
one `{rvec, tvec}` per image — `null` for an image with no pose — and
`images`, the name of each pose's image, REQUIRED: a positional join with an
image list kept elsewhere is exactly the out-of-band knowledge a file must
not need.

The shape is a THIRD-PARTY one, so its conventions are its consumer's, and the
two arguments are CONVERSION TARGETS applied to data whose own conventions its
types fix — the camera's K is canonical by VGC's construction, the poses are
millimetres — never labels for it:

  `to_frame`  the pixel frame `cx, cy` are written in; OpenCV's own,
              `PixelFrame{:opencv}`, by default. A type, it leads.
  `to_unit`   the length unit `tvec` is written in (the poses are carried in
              millimetres, so `to_unit = m` divides by a thousand).

Both are also written into the file, under `image_origin` and `unit`, and the
reader refuses a caller who states different ones.

`camera` must carry a lens OpenCV's distortion model computes
([`opencv_coefficients`](@ref)); the file names the coefficients of the
shortest OpenCV vector holding it. A `K` with nonzero skew is refused: the
shape has no field for it.
"""
save_opencv_calibration(path::AbstractString, args...; kwargs...) =
    save_opencv_calibration(PixelFrame{:opencv}, path, args...; kwargs...)

function save_opencv_calibration(
        to_frame::Type{<:PixelFrame}, path::AbstractString, camera::CameraModel, poses::AbstractVector;
        rms::Real, to_unit::Units, images::AbstractVector{<:AbstractString},
    )
    unit, image_origin = to_unit, pixel_origin(to_frame)
    dimension(unit) == dimension(m) || throw(
        ArgumentError("save_opencv_calibration: `to_unit` must be a length (mm, m, …), got $(unit)")
    )
    (length(images) == length(poses) && all(!isempty, images)) || throw(
        ArgumentError(
            "save_opencv_calibration: `images` names the image of every pose — $(length(poses)) poses, " *
                "$(length(images)) names$(any(isempty, images) ? ", some empty" : "")"
        )
    )
    lens = VgtkCore.lens(camera)
    c = opencv_coefficients(lens)
    K = SMatrix{3, 3, Float64}(camera.intrinsics.K)
    # OpenCV's camera matrix in this shape is fx, fy, cx, cy — no skew — so a
    # skewed K would be written as a different camera; refused, as
    # `colmap_camera` refuses it.
    iszero(K[1, 2]) || throw(
        ArgumentError("save_opencv_calibration: this shape has no skew; the camera's K carries skew = $(K[1, 2])")
    )
    # The camera holds the principal point in the CANONICAL frame; put it back
    # into the convention the file is written in.
    shift = EuclideanMap(CANONICAL_PIXEL_FRAME, PixelFrame{image_origin}).t
    cx, cy = Float64(K[1, 3]) + shift[1], Float64(K[2, 3]) + shift[2]
    scale = ustrip(mm, quantity(1.0, unit))       # mm per `unit`
    body = (
        fx = Float64(K[1, 1]), fy = Float64(K[2, 2]), cx = cx, cy = cy,
        rms = Float64(rms),
        poses = [p === nothing ? nothing : _opencv_pose(p, scale) for p in poses],
        images = String.(images),
        NamedTuple{_OPENCV_KEYS[1:length(c)]}(Tuple(c))...,
        unit = string(unit), image_origin = String(image_origin),
    )
    _write_json(path, body)
    return path
end

function save_opencv_calibration(
        to_frame::Type{<:PixelFrame}, path::AbstractString, cal::ExternalCalibration;
        to_unit::Units, images = cal.images,
    )
    images === nothing && throw(
        ArgumentError("save_opencv_calibration: $(cal.path) names no images — supply `images`, the image of every pose")
    )
    return save_opencv_calibration(to_frame, path, cal.camera, cal.poses; rms = cal.reprojection_error, to_unit, images)
end

# The unit's NAME, for the file, is FlexUnits' printing `string(u)`: a
# registered unit's symbol, the word a consumer hands back to the reader; a
# unit the registry has no symbol for still prints as FlexUnits shows it,
# something a human can read, rather than refusing the export.

# The caller's `unit` and `image_origin` against what the file DECLARES, where
# it declares them (`save_opencv_calibration` writes both; third-party exports
# write neither). Compared by the name the writer spells them with, so a file
# this package wrote agrees with exactly the caller that would write it again.
function _check_declared(d, unit::Units, image_origin::Symbol, path)
    # `unit` is this package's writer's key; `tvec_unit` a third-party exporter's
    for (key, expected) in ((:unit, string(unit)), (:tvec_unit, string(unit)), (:image_origin, String(image_origin)))
        haskey(d, key) || continue
        declared = d[key]
        declared isa AbstractString || throw(
            ArgumentError("$(path): `$(key)` must be a name, got $(repr(declared))")
        )
        declared == expected || throw(
            ArgumentError(
                "$(path): the file declares `$(key)` = $(repr(String(declared))) but the caller " *
                    "expects $(repr(expected)) — refusing to read it in the caller's convention"
            )
        )
    end
    return nothing
end

# OpenCV's distortion vector, entry by entry, as this shape names it.
const _OPENCV_KEYS = (:k1, :k2, :p1, :p2, :k3, :k4, :k5, :k6, :s1, :s2, :s3, :s4, :tilt_x, :tilt_y)
const _OPENCV_FILE_KEYS = (:fx, :fy, :cx, :cy, :rms, :poses, :images, :unit, :tvec_unit, :image_origin, _OPENCV_KEYS...)

# The distortion vector a file states: the shortest OpenCV vector holding every
# coefficient the file names (k1, k2, p1, p2 required), the others zero.
function _opencv_file_vector(d, path, what)
    for k in keys(d)
        Symbol(k) in _OPENCV_FILE_KEYS || throw(
            ArgumentError("$(path): unknown key `$(k)` in an $(what) calibration — refusing to read the camera without it")
        )
    end
    for k in (:k1, :k2, :p1, :p2)
        _need(d, k, path, what)
    end
    named(ks) = any(k -> haskey(d, k), ks)
    n = named((:tilt_x, :tilt_y)) ? 14 : named((:s1, :s2, :s3, :s4)) ? 12 : named((:k4, :k5, :k6)) ? 8 : haskey(d, :k3) ? 5 : 4
    return Float64[haskey(d, k) ? Float64(d[k]::Real) : 0.0 for k in _OPENCV_KEYS[1:n]]
end

# A pose the reader will read back: OpenCV's Rodrigues vector and a
# translation in the file's unit. `_external_pose` is the inverse.
function _opencv_pose(p, scale::Float64)
    R, t = _pose_parts(p)
    r = RotationVec(RotMatrix{3, Float64}(R))
    return (rvec = [r.sx, r.sy, r.sz], tvec = [t[1] / scale, t[2] / scale, t[3] / scale])
end
# An `EuclideanMap` — what a pipeline's poses are — carries its rotation and
# translation (millimetres) as `.R` and `.t`.
_pose_parts(p::EuclideanMap) = (RotMatrix{3, Float64}(p.R), SVector{3, Float64}(p.t))

"""
    load_opencv_calibration([file_frame = PixelFrame{:opencv},] path; file_unit, target = "calibration_target")
        -> ExternalCalibration

Read an OpenCV calibration export: `fx, fy, cx, cy`; the distortion vector by
name — `k1`, `k2`, `p1`, `p2` required, `k3`–`k6`, `s1`–`s4` and the tilt
`tilt_x`, `tilt_y` (OpenCV's τx, τy) as the file names them — read as the lens computing OpenCV's
model for the shortest distortion vector holding every named coefficient
([`opencv_lens`](@ref)), the unnamed ones zero; an unknown key is refused,
never ignored (a dropped tilt would be a different camera); `rms`
in pixels; `poses`, one `{rvec, tvec}` per image (`null` allowed); and
`images`, the name of each pose's image, when the file names them (this
package's writer does; a third-party export does not, and its join with the
images is positional and the caller's).

The two arguments are statements about the FILE — what a third-party export
does not declare — never labels on data: `file_unit` is the length unit of
`tvec` (the object points' unit, `m` or `mm`), and `file_frame` the pixel
frame of `cx, cy`, OpenCV's own `PixelFrame{:opencv}` by default (a type, it
leads). The camera returned is in VGC's canonical frame by construction and
the poses in millimetres, so nothing downstream restates either. `target`
names the pattern in the pose records (`target → camera_optical`).

Most third-party exports declare neither and are read as stated; some declare
`tvec_unit`. A file written by [`save_opencv_calibration`](@ref) declares both
(`unit`, `image_origin`), and a declaration that differs from the caller's is
an `ArgumentError`, never a silent rescale: a metre file read as millimetres
puts every target a thousand times too near.
"""
load_opencv_calibration(path::AbstractString; kwargs...) = load_opencv_calibration(PixelFrame{:opencv}, path; kwargs...)

function load_opencv_calibration(
        file_frame::Type{<:PixelFrame}, path::AbstractString; file_unit::Units,
        target::AbstractString = "calibration_target",
    )
    unit, image_origin = file_unit, pixel_origin(file_frame)
    d = _read_json(path)::AbstractDict   # a JSON object
    what = "OpenCV"
    _check_declared(d, unit, image_origin, path)
    lens = opencv_lens(_opencv_file_vector(d, path, what))
    camera = _external_camera(d, lens, image_origin, path, what)
    poses = _external_poses(_need(d, :poses, path, what)::Vector{Any}, unit, target, path)
    images = haskey(d, :images) ? String[n::String for n in d.images::Vector{Any}] : nothing
    (images === nothing || length(images) == length(poses)) || throw(
        ArgumentError("$(path): $(length(images)) image names for $(length(poses)) poses")
    )
    return ExternalCalibration(
        :opencv, camera, poses, Float64(_need(d, :rms, path, what)::Real), :rms, String(path), images
    )
end

"""
    load_matlab_calibration([file_frame = PixelFrame{:julia},] path; file_unit, target = "calibration_target")
        -> ExternalCalibration

Read a MATLAB Computer Vision Toolbox calibration export. The
radial/tangential files (`fx, fy, cx, cy, r1, r2, r3, t1, t2`) state OpenCV's
polynomial term for term, so the camera is OpenCV's five-coefficient model,
`opencv_lens([r1, r2, t1, t2, r3])`. The fisheye file
(`mapping_coeffs, distortion_center, stretch_matrix`, MATLAB's
`fisheyeIntrinsics`) is Scaramuzza's model: the sensor-plane point `(u′, v′)`
reaches the image as `stretch_matrix · (u′, v′) + distortion_center` and
back-projects to the ray `(u′, v′, a0 + a2·ρ² + a3·ρ³ + a4·ρ⁴)` in the
sensor-plane pixel radius `ρ`, so `a0` is the focal length, the stretch and
the centre are `K`, and the polynomial normalized by `a0` is VGC's
`Scaramuzza` (`aᵢ ↦ aᵢ·a0^(i−1)`). `K` must be upper triangular and the
stretch matrix is not: its RQ factorization `S = U·Q` puts `a0·U` in `K` and
`Q`, a rotation of the sensor plane, into the camera frame — a rotation about
the optical axis (a fraction of a degree on this capture) that every returned
pose carries, so the camera and its poses stay one consistent model. That
rotation is part of the difference between this camera's frame and an OpenCV
fit's, and shows in a mount compared across the two. `mean_error` is the file's
statistic — a MEAN reprojection error, not an rms (`error_statistic == :mean`).

`file_frame` is the pixel frame of the file's centre and principal point:
MATLAB's documented convention, `PixelFrame{:julia}` (top-left pixel centre
at (1, 1)), by default — a fitted principal point cannot tell it from an
exporter's shift to OpenCV's, so override it only for a file verified to be
in another frame. `file_unit` is the world unit the toolbox was given the
pattern in (`mm` for a pattern typed in millimetres) — `tvec` is in it; there
is no default. MATLAB's board frame has x and y exchanged relative to
OpenCV's for the same pattern: the poses are returned in MATLAB's own board
frame, so name its `target` apart from an OpenCV calibration's. `target` names
the pattern in the pose records (`target → camera_optical`).
"""
load_matlab_calibration(path::AbstractString; kwargs...) = load_matlab_calibration(PixelFrame{:julia}, path; kwargs...)

function load_matlab_calibration(
        file_frame::Type{<:PixelFrame}, path::AbstractString; file_unit::Units,
        target::AbstractString = "calibration_target",
    )
    unit, image_origin = file_unit, pixel_origin(file_frame)
    d = _read_json(path)::AbstractDict   # a JSON object
    what = "MATLAB"
    camera, twist = if haskey(d, :mapping_coeffs)
        a = Float64.(collect(_need(d, :mapping_coeffs, path, what)))
        length(a) == 4 || throw(ArgumentError("$(path): `mapping_coeffs` is (a0, a2, a3, a4); got $(length(a))"))
        c = Float64.(collect(_need(d, :distortion_center, path, what)))
        rows = _need(d, :stretch_matrix, path, what)::Vector{Any}
        a0 = a[1]
        # a0 is the focal length. The toolbox can converge to the mirrored
        # solution — a0 < 0 with every target behind the camera (two of this
        # capture's exports, at 4.7 and 84 px mean error); that is no camera.
        a0 > 0 || throw(
            ArgumentError(
                "$(path): mapping coefficient a0 = $(a0) is not positive — MATLAB converged " *
                    "to the mirrored solution (the target behind the camera); refusing to read it as a camera"
            )
        )
        S = SMatrix{2, 2, Float64, 4}(
            Float64(rows[1][1]), Float64(rows[2][1]), Float64(rows[1][2]), Float64(rows[2][2])
        )
        # K is upper triangular and the stretch matrix is not: S = U·Q, a0·U is
        # K's pixel block, and Q — a rotation of the sensor plane, which a radial
        # lens commutes with — is a rotation of the camera frame about its
        # optical axis, applied to every pose.
        U, φ = _rq2(S)
        K = CameraCalibrationMat(
            SMatrix{3, 3, Float64}(a0 * U[1, 1], 0.0, 0.0, a0 * U[1, 2], a0 * U[2, 2], 0.0, c[1], c[2], 1.0)
        )
        lens = Scaramuzza(a[2] * a0, a[3] * a0^2, a[4] * a0^3)
        CameraModel(K; lens, image_origin), RotMatrix{3, Float64}(RotZ(φ))
    else
        r(k) = Float64(_need(d, k, path, what)::Real)
        lens = opencv_lens([r(:r1), r(:r2), r(:t1), r(:t2), r(:r3)])
        _external_camera(d, lens, image_origin, path, what), one(RotMatrix{3, Float64})
    end
    poses = _external_poses(_need(d, :poses, path, what)::Vector{Any}, unit, target, path; twist)
    return ExternalCalibration(
        :matlab, camera, poses, Float64(_need(d, :mean_error, path, what)::Real), :mean, String(path)
    )
end

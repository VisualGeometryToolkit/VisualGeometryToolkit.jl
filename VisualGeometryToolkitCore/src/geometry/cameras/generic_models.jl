# ==============================================================================
# Generic camera models — the "10,000 parameters" grid models (Schöps et al.)
# ==============================================================================
#
# Port of the central-generic and noncentral-generic camera models of
# Schöps, Larsson, Pollefeys, Sattler, "Why Having 10,000 Parameters in Your
# Camera Model Is Better Than Twelve" (CVPR 2020) — the reference
# implementation in puzzlepaint/camera_calibration
# (applications/camera_calibration/generic_models). The model is a grid of
# per-region observations interpolated by a UNIFORM CUBIC B-SPLINE over the
# calibrated image area:
#
#   CentralGenericModel     grid of unit observation DIRECTIONS — unprojection
#                           is the normalized spline value; all rays go
#                           through one center.
#   NoncentralGenericModel  grids of direction AND origin point — unprojection
#                           is a 3D LINE (`Ray3`); no common center.
#
# Projection has no closed form in either: as in the reference code it is a
# small Levenberg–Marquardt in PIXEL space (two unknowns, bounded by the
# calibrated area), minimizing the direction mismatch (central) or the
# tangent-plane point-to-line residual (noncentral). The residuals and the
# bounds mirror the C++; the LM is a device-safe two-unknown bounded LM
# (`_pixel_lm`), one solve for both models; the unprojection Jacobians the
# reference generates as C code are obtained here by ForwardDiff through the
# same scalar spline window — identical totals.
#
# FRAMEWORK EMBEDDING. Both models are unposed `AbstractCamera`s and speak
# the package verbs — `project(model, X)`, `backproject(model, u)`,
# `unproject(model, u, depth)` (central only; a non-central pixel has no
# single z-depth point). The reference's pixel convention — origin at the
# top-left corner of the top-left pixel — IS VGC's canonical COLMAP frame,
# so no origin bridge appears anywhere. Failure (outside the calibrated
# area, LM non-convergence) returns `nothing`, the house convention for
# unrepresentable projections (see `_camera_project_conic`).
#
# THE SPLINE. The 4-tap uniform-cubic window weights are ported verbatim
# from the reference's util.h — they are four fixed cubic polynomials, not
# spline infrastructure (VGC's Cox–de Boor kernels in numerics/bspline.jl
# are the Float32 SoA render currency, the wrong grain for a per-pixel
# scalar model eval). The ported weights are CROSS-CHECKED AGAINST
# BSplineKit in test/generic_models_tests.jl, the same oracle pattern
# bspline_tests.jl establishes for the house kernels.

# ------------------------------------------------------------------------------
# Ray3 — the non-central unprojection currency
# ------------------------------------------------------------------------------

"""
    Ray3(origin, direction)

A 3D ray: `origin + t·direction`, `direction` unit by contract. The
unprojection currency of [`NoncentralGenericModel`](@ref) — a non-central
pixel observes a LINE in the camera frame, not a direction through a common
center.
"""
struct Ray3{T <: AbstractFloat}
    origin::Vec3{T}
    direction::Vec3{T}
end

# ------------------------------------------------------------------------------
# The models
# ------------------------------------------------------------------------------

# Idempotent renormalization: a vector already unit to 1e-12 is kept
# BIT-EXACT (normalize(normalize(v)) drifts a ulp, which would break
# byte-exact serialization round trips); anything genuinely unnormalized is
# normalized — the reference Read() behavior.
_unit_or_normalize(v) = abs(norm(v) - 1) < 1.0e-12 ? v : normalize(v)

# Shared validation of the calibrated-area rectangle (integer pixel indices,
# inclusive, COLMAP frame: area spans [min_x, max_x + 1) × [min_y, max_y + 1)).
function _check_generic_area(width, height, cminx, cminy, cmaxx, cmaxy, gsz)
    (gsz[1] >= 4 && gsz[2] >= 4) || throw(
        ArgumentError(
            "generic model: the B-spline grid needs at least 4×4 control points, got $(gsz[2])×$(gsz[1])"
        )
    )
    (0 <= cminx < cmaxx < width && 0 <= cminy < cmaxy < height) || throw(
        ArgumentError(
            "generic model: invalid calibrated area ($cminx, $cminy)–($cmaxx, $cmaxy) for a $(width)×$(height) image"
        )
    )
    return nothing
end

"""
    CentralGenericModel{T} <: AbstractCamera

The central generic camera model: a `grid` of UNIT observation directions
(`Matrix{Vec3{T}}`, indexed `[gy, gx]`), interpolated by a uniform cubic
B-spline over the calibrated image area `(calib_min_x, calib_min_y)`–
`(calib_max_x, calib_max_y)` (inclusive integer pixel indices, COLMAP
frame). `backproject` is the normalized spline value; `project` is the
reference implementation's pixel-space LM (initialized at the calibrated
area's center, or at `init`). Directions are re-normalized at construction
(the reference's Read() behavior).

    CentralGenericModel(model::CameraModel, (width, height); grid_size, calib_area)

Initialize FROM a parametric model. `init = :interpolate` (default) solves
the cubic-prefilter tridiagonal system per axis so the spline INTERPOLATES
the source's backprojection at the grid nodes — the true approximation
floor; `init = :sample` is quasi-interpolation (control = sample), which
smooths curvature by (h²/6)·f″. Node sampling is clamped to the source's
lit field — `sampling_margin` (default 0.97) times its [`image_circle`](@ref)
when the lens has one, otherwise times its imaging domain — so beyond it
nodes sample at the radially-clamped pixel, and the global prefilter solve
never rings against near-fold curvature or unlit
extrapolation. The bridge from the parametric zoo into the generic
family.
"""
struct CentralGenericModel{T <: AbstractFloat} <: AbstractCamera
    width::Int
    height::Int
    calib_min_x::Int
    calib_min_y::Int
    calib_max_x::Int
    calib_max_y::Int
    grid::Matrix{Vec3{T}}

    function CentralGenericModel{T}(
            width::Integer, height::Integer,
            cminx::Integer, cminy::Integer, cmaxx::Integer, cmaxy::Integer,
            grid::Matrix{Vec3{T}}
        ) where {T <: AbstractFloat}
        _check_generic_area(width, height, cminx, cminy, cmaxx, cmaxy, size(grid))
        return new{T}(width, height, cminx, cminy, cmaxx, cmaxy, _unit_or_normalize.(grid))
    end
end
CentralGenericModel(
    w::Integer, h::Integer, a::Integer, b::Integer, c::Integer, d::Integer,
    grid::Matrix{Vec3{T}}
) where {T} = CentralGenericModel{T}(w, h, a, b, c, d, grid)

"""
    NoncentralGenericModel{T} <: AbstractCamera

The non-central generic camera model: per-grid-point observation LINES —
`directions` (unit, re-normalized at construction) and `points` (a point on
each line), both `Matrix{Vec3{T}}` indexed `[gy, gx]` and interpolated by
the same uniform cubic B-spline. `backproject` returns a [`Ray3`](@ref);
`project` is the pixel-space LM (`_pixel_lm`) on the reference's tangent-plane
point-to-line residual. With an all-zero `points` grid the model is exactly
central.

The `points` are plain metres in the camera frame. This is an exception to the
units layering that the type forces: its grid of `Vec3{T}` is interpolated in
the inner loop and cannot hold quantities, so the unit is stated here instead.
A reader or writer of the model converts at its boundary. A [`PosedCamera`](@ref) of
the model therefore takes physical extrinsics (`EuclideanMap(R, t .* m)`, or
`with_unit(E, mm)` of a map in mm) and projects scene points that are lengths;
a plain extrinsics is an `ArgumentError`, since it would be read in the
caller's unit against the grid's metres. `ustrip(m, camera)` is the plain
camera, in metres only.

    NoncentralGenericModel(model::CameraModel, (width, height); grid_size, calib_area)

Initialize from a parametric (central) model: directions by
quasi-interpolation, points all zero.
"""
struct NoncentralGenericModel{T <: AbstractFloat} <: AbstractCamera
    width::Int
    height::Int
    calib_min_x::Int
    calib_min_y::Int
    calib_max_x::Int
    calib_max_y::Int
    directions::Matrix{Vec3{T}}
    points::Matrix{Vec3{T}}

    function NoncentralGenericModel{T}(
            width::Integer, height::Integer,
            cminx::Integer, cminy::Integer, cmaxx::Integer, cmaxy::Integer,
            directions::Matrix{Vec3{T}}, points::Matrix{Vec3{T}}
        ) where {T <: AbstractFloat}
        size(directions) == size(points) || throw(
            ArgumentError(
                "NoncentralGenericModel: direction and point grids must have equal size, " *
                    "got $(size(directions)) vs $(size(points))"
            )
        )
        _check_generic_area(width, height, cminx, cminy, cmaxx, cmaxy, size(directions))
        return new{T}(width, height, cminx, cminy, cmaxx, cmaxy, _unit_or_normalize.(directions), points)
    end
end
NoncentralGenericModel(
    w::Integer, h::Integer, a::Integer, b::Integer, c::Integer, d::Integer,
    directions::Matrix{Vec3{T}}, points::Matrix{Vec3{T}}
) where {T} =
    NoncentralGenericModel{T}(w, h, a, b, c, d, directions, points)

const _GenericModel = Union{CentralGenericModel, NoncentralGenericModel}

_grid_size(m::CentralGenericModel) = size(m.grid)
_grid_size(m::NoncentralGenericModel) = size(m.directions)

# ------------------------------------------------------------------------------
# Area & grid mapping (verbatim from the reference; COLMAP frame throughout)
# ------------------------------------------------------------------------------

@inline _in_calibrated_area(m::_GenericModel, x, y) =
    x >= m.calib_min_x && y >= m.calib_min_y &&
    x < m.calib_max_x + 1 && y < m.calib_max_y + 1

@inline _area_center(m::_GenericModel) = SVector(
    (m.calib_min_x + m.calib_max_x + 1) / 2.0,
    (m.calib_min_y + m.calib_max_y + 1) / 2.0,
)

# Pixel (COLMAP) → grid coordinate along one axis: the affine of the C++
# PixelCornerConvToGridPoint, `1 + s·(x − min)` with `s` its one-number
# derivative. The ONE spelling for the host maps, the bundle's `spline_weights`
# and the device kernel (the host once wrote `(g − 3)·(x − min)/range`, which
# rounds differently, so `spline_support` and `spline_weights` disagreed in the
# last ulp for the same pixel).
@inline _grid_scale(m_min, m_max, gdim) = (gdim - 3) / (m_max + 1 - m_min)
@inline _grid_coord(x, m_min, m_max, gdim) = 1 + _grid_scale(m_min, m_max, gdim) * (x - m_min)

# Pixel (COLMAP) → grid coordinates.
@inline function _grid_coords(m::_GenericModel, x, y)
    gh, gw = _grid_size(m)
    return (
        _grid_coord(x, m.calib_min_x, m.calib_max_x, gw),
        _grid_coord(y, m.calib_min_y, m.calib_max_y, gh),
    )
end

# The lattice's cell size in pixels along each axis — the calibrated span over
# `grid − 3` cells, `1/_grid_scale` — and the calibrated area as a closed
# rectangle `[min, max + 1]` (`_in_calibrated_area` is its half-open test). The
# ONE spelling of the lattice geometry: the node pixels below and the nodes'
# supports (support.jl) read them.
@inline function _lattice_spacing(m::_GenericModel)
    gh, gw = _grid_size(m)
    return ((m.calib_max_x + 1 - m.calib_min_x) / (gw - 3), (m.calib_max_y + 1 - m.calib_min_y) / (gh - 3))
end
_calibrated_rect(m::_GenericModel) =
    Rect2{Float64}(m.calib_min_x, m.calib_min_y, m.calib_max_x + 1 - m.calib_min_x, m.calib_max_y + 1 - m.calib_min_y)

# Grid node (Julia 1-based control index) → pixel: inverse of `_grid_coords`
# at integer grid coordinate `j₁ - 1` (the C++ 0-based control index).
@inline function _node_pixel(m::_GenericModel, gx1::Integer, gy1::Integer)
    hx, hy = _lattice_spacing(m)
    return SVector(m.calib_min_x + (gx1 - 2) * hx, m.calib_min_y + (gy1 - 2) * hy)
end

# The four uniform-cubic window weights at frac x ∈ [3, 4) — the reference's
# util.h (EvalUniformCubicBSpline), cross-checked against BSplineKit in the
# tests. The reference evaluates its cubic pieces in x itself, which cancels
# to ~1e-5 absolute in Float32 (|x³| ≈ 64 against weights ≤ 2/3); the same
# polynomials in the local coordinate u = x − 3 ∈ [0, 1) are exact in exact
# arithmetic and, in Horner form, keep Float32 within 7e-8 of the exact
# weights. On the window's own cell, x ∈ [3, 4), u = x − 3 is exact
# (Sterbenz); outside it (`spline_weights` on a drifted pixel) the pieces
# extrapolate as the same cubics and u may round.
@inline function _cubic_weights(x)
    u = x - 3
    v = 1 - u
    a = v * v * v / 6                           # (1 − u)³ / 6
    b = (u * u * (3u - 6) + 4) / 6              # (3u³ − 6u² + 4) / 6
    c = (((-3u + 3) * u + 3) * u + 1) / 6       # (−3u³ + 3u² + 3u + 1) / 6
    d = u * u * u / 6                           # u³ / 6
    return (a, b, c, d)
end

# Window bookkeeping shared by every eval (EvalUniformCubicBSplineSurface's
# "+2 so the SURROUNDING control points are used" convention).
@inline function _spline_window(gx, gy)
    x = gx + 2
    y = gy + 2
    ix = floor(Int, x)
    iy = floor(Int, y)
    wx = _cubic_weights(x - (ix - 3))
    wy = _cubic_weights(y - (iy - 3))
    return ix, iy, wx, wy
end

# 4×4 tensor-window evaluation of one Vec3 control grid. C++ 0-based window
# columns ix-3 … ix map to Julia columns ix-2 … ix+1 (rows likewise). Any
# matrix — the host grid or its device copy inside a kernel.
@inline function _eval_window(grid::AbstractMatrix{Vec3{T}}, ix, iy, wx, wy) where {T}
    S = promote_type(T, typeof(wx[1]))
    acc = zero(Vec3{S})
    @inbounds for r in 1:4
        row = wx[1] * grid[iy - 3 + r, ix - 2] + wx[2] * grid[iy - 3 + r, ix - 1] +
            wx[3] * grid[iy - 3 + r, ix] + wx[4] * grid[iy - 3 + r, ix + 1]
        acc += wy[r] * row
    end
    return acc
end

# d/dx of the four window weights — the same cubic pieces differentiated, in
# the same local coordinate u = x − 3 and Horner form as `_cubic_weights`.
@inline function _cubic_weights_deriv(x)
    u = x - 3
    v = 1 - u
    a = -v * v / 2                  # −(1 − u)² / 2
    b = u * (3u - 4) / 2            # (3u² − 4u) / 2
    c = ((-3u + 2) * u + 1) / 2     # (−3u² + 2u + 1) / 2
    d = u * u / 2                   # u² / 2
    return (a, b, c, d)
end

# The window value at grid coordinates (gx, gy) and its partials along gx and
# gy, in one sweep of the 4×4 controls. Device-safe (the batched kernel's LM
# reads it per candidate step).
@inline function _eval_window_grad(grid::AbstractMatrix{Vec3{T}}, gx, gy) where {T}
    ix, iy, wx, wy = _spline_window(gx, gy)
    dwx = _cubic_weights_deriv(gx + 2 - (ix - 3))
    dwy = _cubic_weights_deriv(gy + 2 - (iy - 3))
    S = Sx = Sy = zero(SVector{3, Float64})
    @inbounds for r in 1:4
        row = drow = zero(SVector{3, Float64})
        for c in 1:4
            g = SVector{3, Float64}(grid[iy - 3 + r, ix - 3 + c])
            row += wx[c] * g
            drow += dwx[c] * g
        end
        S += wy[r] * row
        Sx += wy[r] * drow
        Sy += dwy[r] * row
    end
    return S, Sx, Sy
end

# ------------------------------------------------------------------------------
# The spline window as an interface — what a bundle over the grid needs
# ------------------------------------------------------------------------------
#
# A pixel reads 4×4 controls; a refinement of the grid holds that support
# fixed per observation and needs the weights AND their pixel derivatives at
# the current projection, so the projection's dependence on the controls and
# on the target direction can be written down (implicit function theorem on
# `d̂(u) = t`). Exposed as plain functions on the model's geometry so no
# consumer re-derives the grid mapping or the cubic pieces.

"""
    spline_support(m::CentralGenericModel, pixel) -> (ix, iy) | nothing

The window `pixel` (COLMAP frame) reads: the 4×4 controls
`m.grid[iy - 3 + r, ix - 3 + c]`, `r, c ∈ 1:4` (row = y tap, column = x
tap) — the index pair [`spline_weights`](@ref) takes. `nothing` outside the
calibrated area.
"""
function spline_support(m::CentralGenericModel, pixel)
    x, y = float(pixel[1]), float(pixel[2])
    _in_calibrated_area(m, x, y) || return nothing
    gx, gy = _grid_coords(m, x, y)
    ix, iy, _, _ = _spline_window(gx, gy)
    return (ix, iy)
end

"""
    spline_weights(m::CentralGenericModel, ix, iy, pixel)
        -> (w::SMatrix{4, 4}, wx::SMatrix{4, 4}, wy::SMatrix{4, 4})

The weights of window `(ix, iy)`'s 16 controls at `pixel` — `w[r, c]`
multiplies `m.grid[iy - 3 + r, ix - 3 + c]` — and their derivatives with
respect to the pixel's x and y. The window is the caller's: for a pixel that
has drifted off the window's own cell the cubic pieces extrapolate smoothly,
which is what a refinement holding the support fixed between rebuilds needs.
`sum(w .* grid_window)` is the unnormalized direction `backproject`
normalizes.
"""
function spline_weights(m::CentralGenericModel, ix::Integer, iy::Integer, pixel)
    x, y = float(pixel[1]), float(pixel[2])
    gh, gw = _grid_size(m)
    sx = _grid_scale(m.calib_min_x, m.calib_max_x, gw)      # ∂gx/∂x
    sy = _grid_scale(m.calib_min_y, m.calib_max_y, gh)      # ∂gy/∂y
    gx = _grid_coord(x, m.calib_min_x, m.calib_max_x, gw)
    gy = _grid_coord(y, m.calib_min_y, m.calib_max_y, gh)
    ax = gx + 2 - (ix - 3)
    ay = gy + 2 - (iy - 3)
    wxv = _cubic_weights(ax); wyv = _cubic_weights(ay)
    dwx = _cubic_weights_deriv(ax) .* sx; dwy = _cubic_weights_deriv(ay) .* sy
    T = typeof(x)
    w = SMatrix{4, 4, T, 16}(ntuple(k -> wyv[(k - 1) % 4 + 1] * wxv[(k - 1) ÷ 4 + 1], Val(16)))
    wx = SMatrix{4, 4, T, 16}(ntuple(k -> wyv[(k - 1) % 4 + 1] * dwx[(k - 1) ÷ 4 + 1], Val(16)))
    wy = SMatrix{4, 4, T, 16}(ntuple(k -> dwy[(k - 1) % 4 + 1] * wxv[(k - 1) ÷ 4 + 1], Val(16)))
    return w, wx, wy
end

"""
    window_controls(m::CentralGenericModel, ix, iy) -> SMatrix{4, 4, Vec3}

The 16 controls of window `(ix, iy)` in [`spline_weights`](@ref)' layout
(`[r, c]` = y tap, x tap).
"""
window_controls(m::CentralGenericModel, ix::Integer, iy::Integer) =
    SMatrix{4, 4, eltype(m.grid), 16}(ntuple(k -> m.grid[iy - 3 + (k - 1) % 4 + 1, ix - 3 + (k - 1) ÷ 4 + 1], Val(16)))

"""
    window_indices(m::CentralGenericModel, ix, iy) -> SMatrix{4, 4, CartesianIndex{2}}

The grid indices of window `(ix, iy)`'s controls, same layout.
"""
window_indices(m::CentralGenericModel, ix::Integer, iy::Integer) =
    SMatrix{4, 4, CartesianIndex{2}, 16}(ntuple(k -> CartesianIndex(iy - 3 + (k - 1) % 4 + 1, ix - 3 + (k - 1) ÷ 4 + 1), Val(16)))

# ------------------------------------------------------------------------------
# Unprojection
# ------------------------------------------------------------------------------

"""
    backproject(m::CentralGenericModel, pixel) -> SVector{3} | nothing

The unit observation direction of `pixel` (COLMAP frame): the normalized
uniform-cubic B-spline value of the direction grid. `nothing` outside the
calibrated area.
"""
function backproject(m::CentralGenericModel, pixel)
    x, y = float(pixel[1]), float(pixel[2])
    _in_calibrated_area(m, x, y) || return nothing
    gx, gy = _grid_coords(m, x, y)
    ix, iy, wx, wy = _spline_window(gx, gy)
    d = _eval_window(m.grid, ix, iy, wx, wy)
    return SVector{3}(d) / norm(d)
end

"""
    backproject(m::NoncentralGenericModel, pixel) -> Ray3 | nothing

The observation LINE of `pixel`: both grids evaluated with the same spline
window; the direction normalized, the origin as interpolated (plain metres in
the camera frame, the grid's unit). `nothing` outside the calibrated area.
"""
function backproject(m::NoncentralGenericModel{T}, pixel) where {T}
    x, y = float(pixel[1]), float(pixel[2])
    _in_calibrated_area(m, x, y) || return nothing
    gx, gy = _grid_coords(m, x, y)
    ix, iy, wx, wy = _spline_window(gx, gy)
    d = _eval_window(m.directions, ix, iy, wx, wy)
    o = _eval_window(m.points, ix, iy, wx, wy)
    nd = norm(d)
    F = typeof(float(x))
    return Ray3{Float64}(Vec3{Float64}(o), Vec3{Float64}(d / nd))
end

"""
    unproject(m::CentralGenericModel, pixel, depth) -> SVector{3} | nothing

The 3D point at z-depth `depth` along the pixel's observation direction
(VGC's `unproject` convention). `nothing` outside the calibrated area or
for a direction with non-positive z.
"""
function unproject(m::CentralGenericModel, pixel, depth)
    d = backproject(m, pixel)
    d === nothing && return nothing
    d[3] > 0 || return nothing
    return SVector{3}(d * (depth / d[3]))
end


"""
    direction_tangents(d) -> (t1, t2)

The reference's orthonormal tangent basis of the unit direction `d` — the
2-dof chart a grid control moves in (`normalize(d + δ₁t1 + δ₂t2)`), and the
plane a direction residual is measured in.
"""
@inline function direction_tangents(d)
    axis = abs(d[1]) > 0.9 ? SVector(0.0, 1.0, 0.0) : SVector(1.0, 0.0, 0.0)
    t1 = normalize(cross(SVector{3}(d), axis))
    t2 = cross(SVector{3}(d), t1)
    return t1, t2
end

# Noncentral tangent-plane residual of a candidate pixel against target X.
@inline function _noncentral_residual(m::NoncentralGenericModel, p::SVector{2}, X::SVector{3})
    gx, gy = _grid_coords(m, p[1], p[2])
    ix, iy, wx, wy = _spline_window(gx, gy)
    d = _eval_window(m.directions, ix, iy, wx, wy)
    o = _eval_window(m.points, ix, iy, wx, wy)
    dn = SVector{3}(d) / norm(d)
    t1, t2 = direction_tangents(dn)
    pto = SVector{3}(o) - X
    return SVector(dot(t1, pto), dot(t2, pto))
end

# ------------------------------------------------------------------------------
# Projection — both models run the ONE pixel-space LM, `_pixel_lm` (below,
# shared with the batched kernel): the central model on its direction residual
# with the analytic spline Jacobian, the noncentral on its tangent-plane residual
# with a ForwardDiff Jacobian.
# ------------------------------------------------------------------------------

"""
    project(m::CentralGenericModel, X; init = nothing) -> SVector{2} | nothing

Project the camera-frame point `X` to a pixel (COLMAP frame): the
reference's Levenberg–Marquardt in pixel space, minimizing the mismatch
between the pixel's unprojected direction and `normalize(X)`, initialized
at the calibrated area's center (or `init`). `nothing` when the point does
not project into the calibrated area (or the optimization fails to reach
the reference's 1e-12 cost).
"""
function project(m::CentralGenericModel, X; init = nothing)
    Xv = SVector{3, Float64}(X[1], X[2], X[3])
    norm(Xv) > 0 || return nothing
    p0 = init === nothing ? _area_center(m) : SVector{2, Float64}(float(init[1]), float(init[2]))
    all(isfinite, p0) || return nothing
    px, ok = _central_project(
        m.grid, Float64(m.calib_min_x), Float64(m.calib_min_y), Float64(m.calib_max_x), Float64(m.calib_max_y),
        Xv, p0[1], p0[2], _PROJECT_MAXITER, _PROJECT_TOL^2
    )
    return ok ? px : nothing
end

# The host budget of the one central solve: a cold start from the area center
# gets 100 LM iterations (the batched form takes warm starts and a caller-set
# `iters`). ONE convergence criterion for every projection solve, host and
# batched: the residual norm below `_PROJECT_TOL` — the reference's 1e-12 cost.
const _PROJECT_MAXITER = 100
const _PROJECT_TOL = 1.0e-6

"""
    project(m::NoncentralGenericModel, X; init = nothing) -> SVector{2} | nothing

Project the camera-frame point `X` — plain metres, the point grid's unit — to
a pixel: LM on the reference's
tangent-plane point-to-line residual — the two components of
`origin(p) − X` in the tangent basis of the pixel's line direction — so a
converged pixel's observation line goes through `X`. `init` defaults to
the calibrated area's center; for a non-central model a good initial
estimate matters (the reference exposes the same knob).
"""
function project(m::NoncentralGenericModel, X; init = nothing)
    Xv = SVector{3, Float64}(X[1], X[2], X[3])
    p0 = init === nothing ? _area_center(m) : SVector{2, Float64}(float(init[1]), float(init[2]))
    all(isfinite, p0) || return nothing
    res(p, _) = _noncentral_residual(m, p, Xv)
    jac(p, _) = ForwardDiff.jacobian(q -> _noncentral_residual(m, q, Xv), p)
    px, ok = _pixel_lm(
        res, jac, p0[1], p0[2], Float64(m.calib_min_x), Float64(m.calib_min_y), Float64(m.calib_max_x),
        Float64(m.calib_max_y), _PROJECT_MAXITER, _PROJECT_TOL^2
    )
    return ok ? px : nothing
end

# ------------------------------------------------------------------------------
# Initialization from a parametric model (quasi-interpolation)
# ------------------------------------------------------------------------------

# House-vocabulary acceptance: a size is a Size2, a calibrated area is a
# Rect (origin + widths ↔ the inclusive integer bounds [min, max]); plain
# tuples remain welcome.
_generic_size(sz::Size2) = (Int(sz.width), Int(sz.height))
_generic_size(sz::Tuple{Integer, Integer}) = (Int(sz[1]), Int(sz[2]))
_area_bounds(a::NTuple{4, Integer}) = map(Int, a)
function _area_bounds(r::GeometryBasics.HyperRectangle{2})
    o = GeometryBasics.origin(r); w = GeometryBasics.widths(r)
    return (Int(o[1]), Int(o[2]), Int(o[1] + w[1]) - 1, Int(o[2] + w[2]) - 1)
end

"""
    default_generic_grid(width, height) -> (gw, gh)

The control grid the reference lays over a `width × height` pixel area: one
node every 64 px plus the three border nodes the cubic window needs
(`ceil(width / 64) + 3`, likewise in y; at least 4). Apply it to the
CALIBRATED AREA, not the frame, when the two differ — a circular fisheye's
lit field is smaller than its sensor.
"""
default_generic_grid(width, height) = (max(4, ceil(Int, width / 64) + 3), max(4, ceil(Int, height / 64) + 3))

# Cubic B-spline interpolation ("prefilter") matrix along one axis: at an
# interior node the spline reads (c[j−1] + 4c[j] + c[j+1])/6, so solving the
# tridiagonal system per axis makes the spline INTERPOLATE the node samples
# instead of smoothing them (quasi-interpolation, the :sample init). The two
# boundary controls keep their samples — their own nodes lie outside the
# evaluable domain, so no interpolation condition exists for them.
function _prefilter_matrix(n::Int)
    dl = fill(1.0 / 6, n - 1)
    d = fill(4.0 / 6, n)
    du = fill(1.0 / 6, n - 1)
    d[1] = 1.0; du[1] = 0.0
    d[n] = 1.0; dl[n - 1] = 0.0
    return LinearAlgebra.Tridiagonal(dl, d, du)
end

# Node-sampling clamp: the source lens's lit field, as a normalized radius about
# the principal point — its image circle (`_lens_circle_radius`, image_circle.jl:
# the one owner of the circle) when the lens has a limit of its own, otherwise
# its imaging domain (`_imaging_domain_radius`: a lens with no fold may still
# image only a bounded disc — an equidistant fisheye stops at θ = π — and a node
# sampled past it backprojects to NaN, which the global prefilter spreads to
# every control). `Inf` for a lens that images the whole plane. A node outside
# `sampling_margin` times that disc samples at the radially clamped pixel.
#
# The margin is load-bearing: at the circle itself the source's profile folds
# (its curvature is unbounded) and at the domain's edge `backproject` turns
# NaN, so a node sampled ON the boundary would carry the near-fold curvature,
# or a NaN, into the GLOBAL prefilter solve, which spreads it as ringing
# (measured: 41 px at the rim vs 1e-4 px interior). 0.97 keeps every sample
# strictly inside; it is the sampler's `sampling_margin` keyword.
function _sampling_radius(model::CameraModel)
    l = lens(model)
    ρ = try
        _lens_circle_radius(l)
    catch e
        e isa ArgumentError || rethrow()
        throw(
            ArgumentError(
                "generic model from a $(nameof(typeof(l))) camera: the node sampler clamps to the lens's lit " *
                    "field, but its image circle is undefined — $(e.msg)"
            )
        )
    end
    return ρ === nothing ? Float64(_imaging_domain_radius(l)) : ρ
end

function _directions_from(
        model::CameraModel, m::_GenericModel, ::Type{T};
        init::Symbol, sampling_margin::Real   # both callers supply both; the public defaults live on the constructors
    ) where {T}
    gh, gw = _grid_size(m)
    Ki = inv(SMatrix{3, 3, Float64}(calibration_matrix(model)))
    pp = SVector{2, Float64}(_principal_point(model))
    rmax = sampling_margin * _sampling_radius(model)          # normalized radius
    function _node_sample(gx, gy)
        px = _node_pixel(m, gx, gy)
        x = Ki * SVector(px[1], px[2], 1.0)
        ρ = hypot(x[1], x[2])                                 # the node's normalized radius
        ρ > rmax && (px = pp + (rmax / ρ) * (SVector(px[1], px[2]) - pp))   # K is affine about pp: the same ray scaled
        return normalize(SVector{3, Float64}(backproject(model, px)))
    end
    S = [_node_sample(gx, gy) for gy in 1:gh, gx in 1:gw]
    init === :sample && return [Vec3{T}(v) for v in S]
    init === :interpolate || throw(
        ArgumentError(
            "generic model init must be :interpolate (cubic prefilter) or :sample " *
                "(quasi-interpolation), got :$init"
        )
    )
    Ay = _prefilter_matrix(gh)
    Ax = _prefilter_matrix(gw)
    C = ntuple(k -> Ay \ (([S[gy, gx][k] for gy in 1:gh, gx in 1:gw]) / transpose(Ax)), 3)
    return [Vec3{T}(C[1][gy, gx], C[2][gy, gx], C[3][gy, gx]) for gy in 1:gh, gx in 1:gw]
end

function CentralGenericModel(
        model::CameraModel, size::Union{Tuple{Integer, Integer}, Size2};
        grid_size::Tuple{Integer, Integer} = default_generic_grid(_generic_size(size)...),
        calib_area::Union{NTuple{4, Integer}, GeometryBasics.HyperRectangle{2}} =
            (0, 0, _generic_size(size)[1] - 1, _generic_size(size)[2] - 1),
        init::Symbol = :interpolate,
        sampling_margin::Real = 0.97,
    )
    W, H = _generic_size(size)
    area = _area_bounds(calib_area)
    gw, gh = grid_size
    seed = CentralGenericModel{Float64}(
        W, H, area..., fill(Vec3{Float64}(0, 0, 1), gh, gw)
    )
    return CentralGenericModel{Float64}(
        W, H, area...,
        _directions_from(model, seed, Float64; init, sampling_margin)
    )
end

function NoncentralGenericModel(
        model::CameraModel, size::Union{Tuple{Integer, Integer}, Size2};
        grid_size::Tuple{Integer, Integer} = default_generic_grid(_generic_size(size)...),
        calib_area::Union{NTuple{4, Integer}, GeometryBasics.HyperRectangle{2}} =
            (0, 0, _generic_size(size)[1] - 1, _generic_size(size)[2] - 1),
        init::Symbol = :interpolate,
        sampling_margin::Real = 0.97,
    )
    W, H = _generic_size(size)
    area = _area_bounds(calib_area)
    gw, gh = grid_size
    seed = CentralGenericModel{Float64}(
        W, H, area..., fill(Vec3{Float64}(0, 0, 1), gh, gw)
    )
    dirs = _directions_from(model, seed, Float64; init, sampling_margin)
    return NoncentralGenericModel{Float64}(
        W, H, area..., dirs, fill(Vec3{Float64}(0, 0, 0), gh, gw)
    )
end

# ------------------------------------------------------------------------------
# Precision conversion: `convert_eltype`, and the warp chain's `_precision`
# bridge to it
# ------------------------------------------------------------------------------
convert_eltype(::Type{T}, m::CentralGenericModel) where {T <: AbstractFloat} =
    CentralGenericModel{T}(
    m.width, m.height, m.calib_min_x, m.calib_min_y,
    m.calib_max_x, m.calib_max_y, Vec3{T}.(m.grid)
)
convert_eltype(::Type{T}, m::NoncentralGenericModel) where {T <: AbstractFloat} =
    NoncentralGenericModel{T}(
    m.width, m.height, m.calib_min_x, m.calib_min_y,
    m.calib_max_x, m.calib_max_y, Vec3{T}.(m.directions), Vec3{T}.(m.points)
)

# The one bridge from transforms/warp.jl's private `_precision` (the fused
# image_warp chain and the batched kernels) to the public conversion: a lens
# or grid model in a chain converts through `convert_eltype`, so a downstream
# lens's method is the one the kernels apply. Intrinsics and camera models are
# deliberately NOT bridged: a chain holding one has always kept it as it is
# (`_precision`'s identity fallback), and still does.
_precision(::Type{T}, x::Union{AbstractLensModel, _GenericModel}) where {T <: AbstractFloat} =
    convert_eltype(T, x)

# ------------------------------------------------------------------------------
# Undistortion through the generic model — coarse-inverse remap
# ------------------------------------------------------------------------------

# Coarse-grid density of the generic undistortion map, as a multiple of the
# model's own grid (0.19 px worst case at 4 on a 3 MP calibration; see below).
const _REMAP_OVERSAMPLE = 4.0

"""
    undistort(m::CentralGenericModel, img, Kr::StaticMatrix{3,3}, rect;
              scale = 1, fillvalue = zero(eltype(img))) -> FramedImage

Render `img` (taken by the generic camera `m`) undistorted into the TARGET
PINHOLE camera `Kr`: for each output pixel of the window `rect` (of the
target pinhole's frame), the ray `Kr⁻¹·ũ` is projected through `m` and the
source sampled bilinearly. A generic model has no K/lens split, so the
caller names the pinhole it rectifies into — supply the source parametric
camera's K to land in the SAME frame as its own `undistort`, making the
panels directly comparable.

Projection has no closed form (the reference's per-point LM), so the
undistortion map -- the inverse of a smooth spline diffeomorphism, and
therefore exactly as smooth as the model itself -- is solved only at a
coarse grid (`oversample` x the model's own grid density, warm-started
`project` calls) and bilinearly interpolated to full output resolution
before sampling. Verified against a per-pixel LM sweep (retired after
verification; see f6e92fa): max positional error 0.19 px at the default
`oversample = 4` on a 3 MP hallway calibration, O(h^2) in `oversample`.
Pixels whose ray falls outside the calibrated area fill with
`fillvalue`.

Central only: a NON-central pixel observes a line, so its undistortion is
depth-dependent (parallax) — there is no single answer per pixel, and no
method is provided on purpose.
"""
function undistort(
        m::CentralGenericModel, img::AbstractMatrix, Kr::StaticMatrix{3, 3},
        rect::GeometryBasics.HyperRectangle{2};
        scale::Real = 1, fillvalue = zero(eltype(img)), oversample::Real = _REMAP_OVERSAMPLE
    )
    return _undistort_remap(img, m, Kr, rect, scale, fillvalue, oversample)
end


# The undistortion map is the inverse of a smooth spline
# diffeomorphism, so it is exactly as smooth as the model itself — solve the
# inverse only at a COARSE grid (~oversample × the model's own grid density:
# ~10⁴ solves instead of 10⁶) and interpolate the remap bilinearly to full
# output resolution before sampling. Interpolation loses nothing beyond the
# model's own approximation order. Both phases are KernelAbstractions kernels on
# the image's backend: the coarse solves are a row march (`_row_march_kernel!`),
# each node warm-started at its solved neighbour; the dense phase is one kernel
# of interpolation + the shared bilinear sampler.
function _undistort_remap(
        img::AbstractMatrix, m::CentralGenericModel, Kr::StaticMatrix{3, 3},
        rect::GeometryBasics.HyperRectangle{2}, scale::Real, fillvalue, oversample::Real
    )
    wd = GeometryBasics.widths(rect)
    Wout, Hout = ceil(Int, wd[1] / scale), ceil(Int, wd[2] / scale)
    return _remap_sample(img, m, inv(SMatrix{3, 3, Float64}(Kr)), rect, Wout, Hout, fillvalue, oversample)
end

# The machinery, with the RAY MAP as the parameter: `A` takes a homogeneous
# rect-frame pixel to the ray the model projects — `inv(K)` is undistortion,
# `Qᵢᵀ·K_r⁻¹` is stereo rectification. The CALLER owns the output size: a
# rectified pair must come out the size `rectify` computed for every view, so
# the height is never re-derived here from one axis's scale.
function _remap_sample(
        img::AbstractMatrix, m::CentralGenericModel, A::SMatrix{3, 3, Float64},
        rect::GeometryBasics.HyperRectangle{2}, Wout::Int, Hout::Int, fillvalue, oversample::Real,
        precision::Type{<:AbstractFloat} = Float64
    )
    o = GeometryBasics.origin(rect)
    wd = GeometryBasics.widths(rect)
    gh, gw = _grid_size(m)
    nc = max(8, ceil(Int, oversample * gw))
    nr = max(8, ceil(Int, oversample * gh))
    backend = get_backend(img)

    # coarse phase: the rays of the (nr+1)×(nc+1) nodes, solved by the row march
    rays = adapt(backend, [A * SVector(o[1] + ci * wd[1] / nc, o[2] + ri * wd[2] / nr, 1.0) for ri in 0:nr, ci in 0:nc])
    grid = adapt(backend, m.grid)
    px = KA.allocate(backend, SVector{2, Float64}, nr + 1, nc + 1)
    okmap = KA.allocate(backend, Bool, nr + 1, nc + 1)
    _row_march_kernel!(backend, 8)(   # small groups: a few hundred rows still spread over every CPU thread
        px, okmap, grid, rays, _area_center(m)...,
        Float64(m.calib_min_x), Float64(m.calib_min_y), Float64(m.calib_max_x), Float64(m.calib_max_y);
        ndrange = nr + 1
    )
    KA.synchronize(backend)

    # dense phase: ONE kernel — the coarse map interpolated, the source sampled
    T = precision
    cmap = map(p -> SVector{2, T}(p), px)
    dst = fill!(similar(img, Hout, Wout), fillvalue)
    fv = _sample_type(eltype(dst))(fillvalue)
    _remap_dense_kernel!(backend)(dst, img, cmap, okmap, nr, nc, fv; ndrange = (Hout, Wout))
    KA.synchronize(backend)
    return FramedImage(dst, rect)
end

# The coarse inverse, one work item per row of the coarse grid: the row's
# middle node is solved from the area centre, then the row is marched outward
# to the left and to the right, each node warm-started at the last node that
# converged on its inward side — a neighbour's solution, a fraction of a model
# cell away, so a near-horizon node starts on its own side of the clamped
# plateau. Every solve is the host `project`'s (`_PROJECT_MAXITER` iterations,
# converged below `_PROJECT_TOL`).
@kernel function _row_march_kernel!(px_out, ok_out, @Const(grid), @Const(rays), cx, cy, minx, miny, maxx, maxy)
    row = @index(Global)
    ncol = size(rays, 2)
    mid = (ncol + 1) ÷ 2
    from_mid = _march_node!(px_out, ok_out, grid, rays, row, mid, SVector(cx, cy), minx, miny, maxx, maxy)
    warm = from_mid
    for col in (mid + 1):ncol
        warm = _march_node!(px_out, ok_out, grid, rays, row, col, warm, minx, miny, maxx, maxy)
    end
    warm = from_mid
    for col in (mid - 1):-1:1
        warm = _march_node!(px_out, ok_out, grid, rays, row, col, warm, minx, miny, maxx, maxy)
    end
end

# One node of the row march: solved from `warm`; returns the warm start of the
# next node (this solution if it converged, else `warm` unchanged).
@inline function _march_node!(px_out, ok_out, grid, rays, row, col, warm, minx, miny, maxx, maxy)
    @inbounds begin
        px, ok = _central_project(
            grid, minx, miny, maxx, maxy, rays[row, col], warm[1], warm[2],
            _PROJECT_MAXITER, _PROJECT_TOL^2
        )
        px_out[row, col] = px
        ok_out[row, col] = ok
    end
    return ok ? px : warm
end

# Output pixel (r, c) → its coarse-grid coordinate, the four surrounding nodes'
# source pixels interpolated (all four must have converged), the source sampled
# by the one bilinear sampler (`_bilinear_sample`, warp.jl), COLMAP → index +½.
@kernel function _remap_dense_kernel!(dst, @Const(src), @Const(cmap), @Const(okmap), nr, nc, fv)
    r, c = @index(Global, NTuple)
    Hout, Wout = size(dst)
    H, W = size(src)
    T = eltype(eltype(cmap))
    gy = (r - T(0.5)) / Hout * nr
    gx = (c - T(0.5)) / Wout * nc
    ri = clamp(unsafe_trunc(Int, floor(gy)), 0, nr - 1)
    ci = clamp(unsafe_trunc(Int, floor(gx)), 0, nc - 1)
    fy = gy - ri
    fx = gx - ci
    @inbounds if okmap[ri + 1, ci + 1] && okmap[ri + 1, ci + 2] && okmap[ri + 2, ci + 1] && okmap[ri + 2, ci + 2]
        p = (1 - fy) * ((1 - fx) * cmap[ri + 1, ci + 1] + fx * cmap[ri + 1, ci + 2]) +
            fy * ((1 - fx) * cmap[ri + 2, ci + 1] + fx * cmap[ri + 2, ci + 2])
        dst[r, c] = _store(eltype(dst), _bilinear_sample(src, p[2] + T(0.5), p[1] + T(0.5), fv, H, W))
    end
end

# =============================================================================
# Stereo rectification arm for grid sensors (dispatched from rectify.jl)
# =============================================================================

# The per-view pull is project(m, Qᵢᵀ·K_r⁻¹·[u,v,1]) — the coarse-inverse
# remap with the rectifying rotation folded into the ray map (a grid has no
# closed-form forward model for the fused warp). The dense phase runs in
# `precision`, as a parametric sibling's warp does; the coarse solves stay
# Float64. `Ki` arrives as `nothing` (a grid model has no calibration matrix).
function _rectified_view(
        m::CentralGenericModel, img::AbstractMatrix, ::Nothing,
        Qi, Kr_inv::SMatrix{3, 3, Float64, 9}, window, Wout::Int, Hout::Int,
        precision::Type{<:AbstractFloat}
    )
    A = SMatrix{3, 3, Float64}(SMatrix{3, 3, Float64, 9}(Qi)' * Kr_inv)
    return _remap_sample(img, m, A, window, Wout, Hout, zero(eltype(img)), _REMAP_OVERSAMPLE, precision)
end

# =============================================================================
# Central projection — one solver, host and device (KernelAbstractions)
# =============================================================================

# The ONE central projection: Levenberg–Marquardt in pixel space (`_pixel_lm`)
# on the direction residual d̂(u) − t̂ with the ANALYTIC spline Jacobian (the
# normalize differential chain-ruled through the pixel→grid scales), for at most
# `maxiter` LM iterations; converged (`ok`) means a final cost below `tol2`.
# Device-safe — no allocation, no dual numbers, no Union return — so the batched
# kernel runs it per ray and the scalar `project` runs it on the host: one
# solver, not a host copy and a device copy.
function _central_project(
        grid, minx::Float64, miny::Float64, maxx::Float64, maxy::Float64,
        t::StaticVector{3}, x0, y0, maxiter::Int, tol2::Float64
    )
    gh, gw = size(grid)
    # scaled before the norm, so a tiny or huge ray does not under/overflow
    tmax = max(abs(t[1]), abs(t[2]), abs(t[3]))
    # a zero or non-finite ray, or start, has no projection (and a NaN pixel
    # would reach `floor(Int, ·)` in the spline window)
    (tmax > 0 && isfinite(tmax) && isfinite(x0) && isfinite(y0)) ||
        return SVector(NaN, NaN), false
    ts = SVector{3, Float64}(t[1] / tmax, t[2] / tmax, t[3] / tmax)
    tn = normalize(ts)
    sx = _grid_scale(minx, maxx, gw)
    sy = _grid_scale(miny, maxy, gh)
    # the direction residual d̂(u) − t̂ (the pixel is bounded to the area: every
    # window stays inside the grid) …
    function res(p, _)
        St = @inbounds SVector{3, Float64}(
            _eval_window(grid, _spline_window(_grid_coord(p[1], minx, maxx, gw), _grid_coord(p[2], miny, maxy, gh))...)
        )
        return normalize(St) - tn
    end
    # … and its Jacobian, the normalize differential (I − n nᵀ)/‖S‖ chain-ruled
    # through the pixel→grid scales
    function jac(p, _)
        S, Sx, Sy = @inbounds _eval_window_grad(grid, _grid_coord(p[1], minx, maxx, gw), _grid_coord(p[2], miny, maxy, gh))
        nrm = norm(S)
        n = S / nrm
        Jx = (Sx - n * (n[1] * Sx[1] + n[2] * Sx[2] + n[3] * Sx[3])) * (sx / nrm)
        Jy = (Sy - n * (n[1] * Sy[1] + n[2] * Sy[2] + n[3] * Sy[3])) * (sy / nrm)
        return hcat(Jx, Jy)
    end
    return _pixel_lm(res, jac, x0, y0, minx, miny, maxx, maxy, maxiter, tol2)
end

# THE pixel-space Levenberg–Marquardt of the generic models: two unknowns,
# `res(p, _)` and its Jacobian `jac(p, _)` from (x0, y0), the pixel bounded to
# `[min, max + 0.999]` (a step is clipped at the bound), for at most `maxiter`
# LM iterations; converged means a final cost below `tol2`. Device-safe —
# static arrays only, no allocation, no Union return — so the batched kernel
# runs it per ray, which is why it is not the stack's LM driver (NLLSsolver
# builds heap problems and cannot run in a KernelAbstractions kernel). It is
# FastLevenbergMarquardt 0.1's static bounded `lmsolve` (MIT), the routine this
# function called until FastLevenbergMarquardt left the stack (2026-10-27),
# restated for N = 2 with its defaults and every floating-point operation in
# its order — Marquardt's damping (λ₀ = 1e-6, ×1/13 on ρ > 0.75, ×3 on
# ρ < 1e-3 or a rejected step, clamped to [1e-28, 1e32]), the running-maximum
# DᵀD clamped to [1e-12, 1e16], its unrolled 2 × 2 Cholesky, the ftol = gtol =
# eps and xtol = 1e-10 stops — so a projection is the same to the bit.
@inline function _pixel_lm(
        res::R, jac::J, x0, y0, minx::Float64, miny::Float64, maxx::Float64, maxy::Float64,
        maxiter::Int, tol2::Float64
    ) where {R, J}
    ϵ = eps(Float64)
    ϵx = 1.0e-10
    λ = 1.0e-6
    lb = SVector(minx, miny)
    ub = SVector(maxx + 0.999, maxy + 0.999)
    x = min.(max.(SVector(Float64(x0), Float64(y0)), lb), ub)
    f = res(x, nothing)
    F = sum(abs2, f)
    F < ϵ && return x, F < tol2
    Jx = jac(x, nothing)
    g = Jx' * f
    norm(g, Inf) < ϵ && return x, F < tol2
    JtJ = Jx' * Jx
    DtD = clamp.(diag(JtJ), 1.0e-12, Inf)
    iter = 0
    while iter < maxiter
        iter += 1
        accepted = false
        ac = 0.0
        pr = 0.0
        ρ = 0.0
        xk = x
        fk = f
        Fk = F
        p, info = _pixel_lm_increment(JtJ, λ, DtD, g)
        # a step clipped to zero by either bound is rejected, as a failed solve is
        clipped = false
        if info == 0
            p = min.(x .- lb, p)
            clipped = all(iszero, p)
            if !clipped
                p = max.(x .- ub, p)
                clipped = all(iszero, p)
            end
        end
        if info == 0 && !clipped
            xk = x .- p
            fk = res(xk, nothing)
            Fk = sum(abs2, fk)
            if Fk < F
                ac = F - Fk
                pr = dot(p, (@. λ * DtD * p + g))
                ρ = ac / pr
                accepted = true
            end
            if all(>(0), (@. abs(p) < ϵx * (ϵx + abs(x))))
                if accepted
                    x, f, F = xk, fk, Fk
                end
                break
            end
        end
        if accepted
            x = xk
            f = fk
            Fold = F
            F = Fk
            # FLM's ftol test reads its swapped buffers: `Fk` is then the old F
            abs(ac) < ϵ * Fold && pr < ϵ * Fold && ρ < 2 && break
            Jx = jac(x, nothing)
            g = Jx' * f
            norm(g, Inf) < ϵ && break
            JtJ = Jx' * Jx
            DtD = clamp.(diag(JtJ), DtD, 1.0e16)
            if ρ > 0.75
                λ *= 1 / 13
            elseif ρ < 1.0e-3
                λ *= 3
            end
        else
            λ *= 3
        end
        λ = clamp(λ, 1.0e-28, 1.0e32)
    end
    return x, F < tol2
end

# FastLevenbergMarquardt's unrolled Cholesky solve of (JᵀJ + λ·DᵀD) p = g at
# N = 2, its operations in its order; `info` ≠ 0 when a damped diagonal is
# negative (a rejected step).
@inline function _pixel_lm_increment(JtJ, λ, DtD, g)
    x1 = g[1]
    x2 = g[2]
    A11 = muladd(λ, DtD[1], JtJ[1, 1])
    A11 >= zero(A11) || return zero(SVector{2, Float64}), 1
    A21 = JtJ[2, 1]
    A22 = muladd(λ, DtD[2], JtJ[2, 2])
    A22 >= zero(A22) || return zero(SVector{2, Float64}), 2
    A22 = muladd(-A21, A21 / A11, A22)
    x2 = muladd(-x1, A21 / A11, x2)
    x2 /= A22
    x1 = muladd(-x2, A21, x1)
    x1 /= A11
    return SVector(x1, x2), 0
end


# Batched device projection: the reference ships ProjectWithInitialEstimate as
# a CUDA __device__ routine (cuda_central_generic_model.cuh); this is the same
# design in KA, the per-ray solve being `_central_project`. The grid uploads
# as-is (Vec3 is isbits). KA.CPU() runs the identical kernel on host.
@kernel function _project_kernel!(
        px_out, ok_out, @Const(grid), @Const(rays), @Const(inits),
        minx::Float64, miny::Float64, maxx::Float64, maxy::Float64,
        iters::Int, tol2::Float64
    )
    i = @index(Global)
    @inbounds begin
        px, ok = _central_project(grid, minx, miny, maxx, maxy, rays[i], inits[i][1], inits[i][2], iters, tol2)
        px_out[i] = px
        ok_out[i] = ok
    end
end

"""
    project(m::CentralGenericModel, rays::AbstractVector{<:StaticVector{3}};
            init, iters = 10, tol = 1.0e-6) -> (; px, ok)

Batched projection: one LM per ray (at most `iters` iterations) with the
ANALYTIC spline Jacobian, on any KernelAbstractions backend — supply device arrays (rays,
inits, and a device copy of `m.grid` uploads automatically) and the solve
runs on the GPU; host arrays run the identical kernel threaded on the CPU. The
backend is the rays' own: every output is allocated beside them.
`init` is a same-length vector of warm starting pixels (the caller's
structure — scanlines, previous frames — is what makes a few iterations
enough; see the reference's ProjectWithInitialEstimate). Returns pixel
positions and a convergence mask (‖residual‖ < `tol`).
"""
function project(
        m::CentralGenericModel, rays::AbstractVector{<:StaticVector{3}};
        init::AbstractVector, iters::Int = 10, tol::Real = _PROJECT_TOL
    )
    length(init) == length(rays) || throw(ArgumentError("project: length(init) != length(rays)"))
    return _project_batched(m, adapt(KA.get_backend(rays), m.grid), rays, init, iters, tol)
end

# The batched projection over a grid already on the rays' backend.
function _project_batched(m::CentralGenericModel, d_grid, rays, init, iters::Int, tol::Real)
    N = length(rays)
    backend = KA.get_backend(rays)
    px = KA.allocate(backend, SVector{2, Float64}, N)
    ok = KA.allocate(backend, Bool, N)
    kern = _project_kernel!(backend, 64)
    kern(
        px, ok, d_grid, rays, init,
        Float64(m.calib_min_x), Float64(m.calib_min_y),
        Float64(m.calib_max_x), Float64(m.calib_max_y),
        iters, Float64(tol)^2; ndrange = N
    )
    KA.synchronize(backend)
    return (; px, ok)
end

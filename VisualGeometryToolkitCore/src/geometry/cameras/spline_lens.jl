# ==============================================================================
# SplineLens — the central grid camera as a lens of a CameraModel
# ==============================================================================
#
# The "10,000 parameters" central model of Schöps, Larsson, Pollefeys and Sattler
# (CVPR 2020) as an ordinary `CameraModel(K, SplineLens(…))`. The lens is a
# BACKWARD lens, as a `LiftLens` is: its `backward::ChartSpline{3}` maps the
# K-normalized image point x = K⁻¹u to a direction (a uniform cubic B-spline
# over a closed rectangle of the normalized plane, normalized). K is a gauge:
# skew = 0, frozen in every fit, and chosen so that the principal point is the
# pixel imaging ẑ (`_spline_gauge`); a camera source keeps its own K.
#
# The lattice is the Schöps lattice of a calibrated pixel area `[min, max + 1)`:
# the backward area is K⁻¹ of that rectangle, and the lattice coordinate
# `1 + (g − 3)/(max + 1 − min)·(u − min)` of the pixel grid is the same affine
# map in x, so the uniform knots map exactly and the camera's `backproject` is
# the pixel grid's to rounding.
#
# This file also hosts the spline lattice's shared helpers (the 4 × 4 window,
# the presence mask, the source sampling of the initialization), which the
# older grid models of generic_models.jl call.

# ------------------------------------------------------------------------------
# The window of a lattice: one tensor 4 × 4 evaluation
# ------------------------------------------------------------------------------

# Idempotent renormalization: a vector already unit to 1e-12 is kept
# BIT-EXACT (normalize(normalize(v)) drifts a ulp, which would break
# byte-exact serialization round trips); anything genuinely unnormalized is
# normalized — the reference Read() behavior.
_unit_or_normalize(v) = abs(norm(v) - 1) < 1.0e-12 ? v : normalize(v)

# Window bookkeeping shared by every eval (EvalUniformCubicBSplineSurface's
# "+2 so the SURROUNDING control points are used" convention).
@inline function _spline_window(gx, gy)
    x = gx + 2
    y = gy + 2
    ix = floor(Int, x)
    iy = floor(Int, y)
    wx = cubic_bspline_weights(_span_coordinate(x, ix))
    wy = cubic_bspline_weights(_span_coordinate(y, iy))
    return ix, iy, wx, wy
end

# 4×4 tensor-window evaluation of one control grid. C++ 0-based window
# columns ix-3 … ix map to Julia columns ix-2 … ix+1 (rows likewise); a
# window's controls start at `off` (0-based row, col) of the full grid. Any
# matrix and any control type with + and scalar * (3-vectors, scalars) — the
# host grid or its device copy inside a kernel. The rows are summed in order,
# r = 1 … 4 (the `zero + row₁` start of the accumulator is exact).
@inline function _eval_window(grid::AbstractMatrix, ix, iy, wx, wy, off::NTuple{2, Int} = (0, 0))
    jx = ix - off[2]
    jy = iy - off[1]
    @inline row(r) = @inbounds wx[1] * grid[jy - 3 + r, jx - 2] + wx[2] * grid[jy - 3 + r, jx - 1] +
        wx[3] * grid[jy - 3 + r, jx] + wx[4] * grid[jy - 3 + r, jx + 1]
    return ((wy[1] * row(1) + wy[2] * row(2)) + wy[3] * row(3)) + wy[4] * row(4)
end

# ------------------------------------------------------------------------------
# The presence mask of a lattice
# ------------------------------------------------------------------------------

# The presence mask of a lattice: `nothing` (every node present) or a `Bool`
# matrix of the grid's size, `[gy, gx]`. A given mask is checked and kept as a
# dense `Matrix{Bool}` (a byte per node: the evaluations read 16 of it per
# pixel, and a device copy is the same array), an all-true one is `nothing` —
# one representation of "every node present", so a model built with `trues(…)`
# is the model built without (and costs nothing per evaluation).
_present_mask(::Nothing, sz) = nothing
function _present_mask(present::AbstractMatrix{Bool}, sz)
    _check_present(present, sz)
    any(present) || throw(ArgumentError("spline grid: the presence mask holds no present node"))
    return all(present) ? nothing : Matrix{Bool}(present)
end
_check_present(::Nothing, sz) = nothing
_check_present(present::AbstractMatrix{Bool}, sz) = size(present) == sz ||
    throw(DimensionMismatch("spline grid: a $(size(present)) presence mask for a $(sz) grid"))

# Whether every node of window `(jx, jy)` — controls `[jy - 3 + r, jx - 3 + c]`,
# `r, c ∈ 1:4`, of the mask's own indexing — is present. Device-safe: a mask on
# the device is read as it is (no allocation, no Union).
@inline _window_present(::Nothing, jx, jy) = true
@inline function _window_present(present::AbstractMatrix{Bool}, jx, jy)
    for c in -2:1, r in -2:1
        present[jy + r, jx + c] || return false
    end
    return true
end

# The controls with NaN in every absent slot (`present[i]` false): no evaluation
# reads an absent slot (a point is evaluated only when its whole 4×4 window is
# present), and one read anywhere shows as NaN.
function _absent_nan(G::AbstractMatrix{V}, present::AbstractMatrix{Bool}) where {V}
    size(G) == size(present) || throw(DimensionMismatch("with_present: a $(size(present)) mask for a $(size(G)) grid"))
    return map((v, p) -> p ? v : V(NaN, NaN, NaN), G, present)
end

# ------------------------------------------------------------------------------
# The pixel lattice: a calibrated area and a grid size
# ------------------------------------------------------------------------------
#
# A lattice is laid over the closed pixel rectangle `[min, max + 1]` of a
# calibrated area (inclusive integer bounds, COLMAP frame) with `(rows, cols)`
# controls, `size − 3` cells per axis; the node of 1-based index j sits at cell
# coordinate j − 2. The ONE spelling of that geometry: the spline lens's
# initialization and the older grid models read it.

# The calibrated area of inclusive integer bounds as the closed rectangle `[min, max + 1]`.
_bounds_rect(minx, miny, maxx, maxy) = Rect2{Float64}(minx, miny, maxx + 1 - minx, maxy + 1 - miny)

# The cell size of the lattice in pixels, `(hx, hy)`.
@inline function _lattice_spacing(rect::Rect2, (gh, gw))
    w = GeometryBasics.widths(rect)
    return (w[1] / (gw - 3), w[2] / (gh - 3))
end

# Grid node (Julia 1-based control index) → pixel: the node at cell coordinate
# `j₁ − 2` (the C++ 0-based control index minus one).
@inline function _lattice_node(rect::Rect2, gsz, gx1::Integer, gy1::Integer)
    hx, hy = _lattice_spacing(rect, gsz)
    o = GeometryBasics.origin(rect)
    return SVector(o[1] + (gx1 - 2) * hx, o[2] + (gy1 - 2) * hy)
end

# Shared validation of the calibrated-area rectangle (integer pixel indices,
# inclusive, COLMAP frame: area spans [min_x, max_x + 1) × [min_y, max_y + 1)).
function _check_spline_area(width, height, cminx, cminy, cmaxx, cmaxy, gsz)
    (gsz[1] >= 4 && gsz[2] >= 4) || throw(
        ArgumentError(
            "spline grid: the B-spline grid needs at least 4×4 control points, got $(gsz[2])×$(gsz[1])"
        )
    )
    (0 <= cminx < cmaxx < width && 0 <= cminy < cmaxy < height) || throw(
        ArgumentError(
            "spline grid: invalid calibrated area ($cminx, $cminy)–($cmaxx, $cmaxy) for a $(width)×$(height) image"
        )
    )
    return nothing
end

# House-vocabulary acceptance: a size is a Size2, a calibrated area is a
# Rect (origin + widths ↔ the inclusive integer bounds [min, max]); plain
# tuples remain welcome.
_spline_size(sz::Size2) = (Int(sz.width), Int(sz.height))
_spline_size(sz::Tuple{Integer, Integer}) = (Int(sz[1]), Int(sz[2]))
_area_bounds(a::NTuple{4, Integer}) = map(Int, a)
function _area_bounds(r::GeometryBasics.HyperRectangle{2})
    o = GeometryBasics.origin(r); w = GeometryBasics.widths(r)
    return (Int(o[1]), Int(o[2]), Int(o[1] + w[1]) - 1, Int(o[2] + w[2]) - 1)
end

"""
    default_spline_grid(width, height) -> (gw, gh)

The control grid the reference lays over a `width × height` pixel area: one
node every 64 px plus the three border nodes the cubic window needs
(`ceil(width / 64) + 3`, likewise in y; at least 4). Apply it to the
CALIBRATED AREA, not the frame, when the two differ — a circular fisheye's
lit field is smaller than its sensor.
"""
default_spline_grid(width, height) = (max(4, ceil(Int, width / 64) + 3), max(4, ceil(Int, height / 64) + 3))

# ------------------------------------------------------------------------------
# Initialization from a source: the direction controls of a pixel lattice
# ------------------------------------------------------------------------------

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

# Node-sampling clamp: where the source lens can be EVALUATED, as a normalized
# radius about the principal point — its valid domain (`_valid_domain_radius`,
# image_circle.jl: the one owner of that radius) when the lens has a limit of its
# own. This is legitimately the model's validity, not where light falls: a node
# sampled past the fold reads the folded branch, whatever the sensor sees. Otherwise
# its imaging domain (`_imaging_domain_radius`: a lens with no fold may still
# image only a bounded disc — an equidistant fisheye stops at θ = π — and a node
# sampled past it backprojects to NaN, which the global prefilter spreads to
# every control). `Inf` for a lens that images the whole plane. A node outside
# `sampling_margin` times that disc samples at the radially clamped pixel.
#
# The margin is load-bearing: at the valid domain's edge the source's profile folds
# (its curvature is unbounded) and at the domain's edge `backproject` turns
# NaN, so a node sampled ON the boundary would carry the near-fold curvature,
# or a NaN, into the GLOBAL prefilter solve, which spreads it as ringing
# (measured: 41 px at the rim vs 1e-4 px interior). 0.97 keeps every sample
# strictly inside; it is the sampler's `sampling_margin` keyword.
function _sampling_radius(model::CameraModel)
    l = lens(model)
    ρ = try
        _valid_domain_radius(l)
    catch e
        e isa ArgumentError || rethrow()
        throw(
            ArgumentError(
                "spline grid from a $(nameof(typeof(l))) camera: the node sampler clamps to the lens model's " *
                    "valid domain, which is undefined — $(e.msg)"
            )
        )
    end
    return ρ === nothing ? Float64(_imaging_domain_radius(l)) : ρ
end

# The direction controls of `init = :interpolate` or `:sample` on the lattice of
# `rect` and `gsz = (rows, cols)`, sampled from the camera `model`.
function _directions_from(
        model::CameraModel, rect::Rect2, gsz, ::Type{T};
        init::Symbol, sampling_margin::Real   # both callers supply both; the public defaults live on the constructors
    ) where {T}
    gh, gw = gsz
    Ki = inv(SMatrix{3, 3, Float64}(calibration_matrix(model)))
    pp = SVector{2, Float64}(_principal_point(model))
    rmax = sampling_margin * _sampling_radius(model)          # normalized radius
    function _node_sample(gx, gy)
        px = _lattice_node(rect, gsz, gx, gy)
        x = Ki * SVector(px[1], px[2], 1.0)
        ρ = hypot(x[1], x[2])                                 # the node's normalized radius
        ρ > rmax && (px = pp + (rmax / ρ) * (SVector(px[1], px[2]) - pp))   # K is affine about pp: the same ray scaled
        return normalize(SVector{3, Float64}(backproject(model, px)))
    end
    S = [_node_sample(gx, gy) for gy in 1:gh, gx in 1:gw]
    init === :sample && return [Vec3{T}(v) for v in S]
    init === :interpolate || throw(
        ArgumentError(
            "spline grid init must be :interpolate (cubic prefilter) or :sample " *
                "(quasi-interpolation), got :$init"
        )
    )
    Ay = _prefilter_matrix(gh)
    Ax = _prefilter_matrix(gw)
    C = ntuple(k -> Ay \ (([S[gy, gx][k] for gy in 1:gh, gx in 1:gw]) / transpose(Ax)), 3)
    return [Vec3{T}(C[1][gy, gx], C[2][gy, gx], C[3][gy, gx]) for gy in 1:gh, gx in 1:gw]
end

"""
    quasi_interpolant(samples::AbstractMatrix) -> controls

The separable 4th-order quasi-interpolant of uniform cubic B-splines: node
samples `f` (a matrix of values with `+` and scalar `*`, laid out `[gy, gx]`
as a spline lattice's controls) become controls `c_j = (−f_{j−1} + 8 f_j −
f_{j+1}) / 6`, along x (the second index) first, then along y; at the border
the outermost sample is replicated. Exact for cubics at interior nodes, local
(each control reads three samples per axis), no linear solve. The stencil of
`init = :quasi_interpolate`, where the samples extend one node past the grid on
every side so that no grid control reads a replicated sample.
"""
function quasi_interpolant(samples::AbstractMatrix)
    F = Matrix{eltype(samples)}(samples)
    qi(v) = [(-v[max(j - 1, 1)] + 8 * v[j] - v[min(j + 1, end)]) / 6 for j in eachindex(v)]
    for gy in axes(F, 1)
        F[gy, :] = qi(F[gy, :])
    end
    for gx in axes(F, 2)
        F[:, gx] = qi(F[:, gx])
    end
    return F
end

# The direction a source gives at a pixel, normalized, NaN where it images
# nothing: a camera's backprojection, or a function `pixel -> direction`
# (`nothing` or a non-finite vector = not imaged).
_source_direction(f, px) = (v = f(px); v === nothing ? SVector(NaN, NaN, NaN) : normalize(SVector{3, Float64}(v[1], v[2], v[3])))
_source_direction(c::CameraModel, px) = (v = backproject(c, px); v === nothing ? SVector(NaN, NaN, NaN) : normalize(SVector{3, Float64}(v)))

# The direction controls of `init = :quasi_interpolate` on the lattice of `rect`
# and `gsz = (rows, cols)`.
#
# The node of 1-based index j along an axis sits at cell coordinate t = j − 2
# (pixel `min + t·h`): the outermost node on each side lies one cell OUTSIDE
# the calibrated area and samples the source there, and its quasi-interpolation
# stencil needs one more ring, so the source is sampled on the extended lattice
# t = −2 … C + 1 (C = grid − 3 cells) and the stencil's border rows are cut off
# afterwards. Measured reasons (RadiometricCamera, the Kannala–Brandt fit of the
# AprilGrid photos): clamping the outer nodes to the area's boundary sample
# flattened the spline over the outermost cell — 1.8 px median / 3 px max of
# direction error within 50 px of the border; replicating the outermost sample
# in the stencil instead left 1 px at the border and a 7.7 % error of the local
# pixel pitch there; and a Schöps calibration resampled with its nodes clamped
# onto the calibrated box (constant extrapolation) had up to 15 px of error in an
# 11-px margin and 0.5 px over the first cell.
#
# Where the source does not image a sample (outside its domain: no ray, a
# non-finite direction) the sample is the linear extrapolation, per axis, of
# the two samples inward of it (from the nearest sample inside the area).
# Where those are not imaged either — the source's domain ends INSIDE the area
# (a lens returns no ray past its fold, the edge of its valid domain) — the sample is the
# linear extrapolation of the first two imaged samples on the way from the node
# towards the area's centre; a node with no imaged sample on that way is an
# `ArgumentError`. Those made-up samples carry a smooth continuation, not a
# ray: the model's region (a supplied image circle) says where it images.
function _quasi_interpolation_directions(source, rect::Rect2, gsz)
    gh, gw = gsz
    Cx, Cy = gw - 3, gh - 3
    hx, hy = _lattice_spacing(rect, gsz)
    o = GeometryBasics.origin(rect)
    f(tx, ty) = _source_direction(source, SVector(o[1] + tx * hx, o[2] + ty * hy))
    valid(tx, ty) = all(isfinite, f(tx, ty))
    function node(tx, ty)
        v = f(tx, ty)
        all(isfinite, v) && return v
        cx, cy = clamp(tx, 0, Cx), clamp(ty, 0, Cy)
        ix = tx < 0 ? 1 : (tx > Cx ? -1 : 0)
        iy = ty < 0 ? 1 : (ty > Cy ? -1 : 0)
        f0 = f(cx, cy)
        e = normalize(f0 + abs(tx - cx) * (f0 - f(cx + ix, cy)) + abs(ty - cy) * (f0 - f(cx, cy + iy)))
        all(isfinite, e) && return e
        dx, dy = Cx / 2 - tx, Cy / 2 - ty
        L = hypot(dx, dy)
        ux, uy = dx / L, dy / L                                # unit cell step towards the area's centre
        for k in 1:floor(Int, L)
            a = (tx + k * ux, ty + k * uy)
            b = (tx + (k + 1) * ux, ty + (k + 1) * uy)
            if valid(a...) && valid(b...)
                fa = f(a...)
                return normalize(fa + k * (fa - f(b...)))
            end
        end
        throw(
            ArgumentError(
                "spline grid (:quasi_interpolate): the source images no sample on the way from the node at " *
                    "cell ($tx, $ty) to the area's centre"
            )
        )
    end
    F = [node(tx, ty) for ty in -2:(gh - 1), tx in -2:(gw - 1)]
    return quasi_interpolant(F)[2:(end - 1), 2:(end - 1)]
end

# The direction controls of `source` on the lattice of `rect` and `gsz = (rows, cols)` by `init`.
function _init_directions(source, rect::Rect2, gsz, init::Symbol, sampling_margin::Real)
    init === :quasi_interpolate && return _quasi_interpolation_directions(source, rect, gsz)
    source isa CameraModel || throw(
        ArgumentError(
            "spline grid init :$init samples a camera model inside its valid domain; a $(nameof(typeof(source))) source " *
                "takes init = :quasi_interpolate"
        )
    )
    return _directions_from(source, rect, gsz, Float64; init, sampling_margin)
end

# What a spline grid is initialized from: a camera (its backprojection) or a function `pixel -> direction`.
const _SplineSource = Union{CameraModel, Function}

# ------------------------------------------------------------------------------
# The lens
# ------------------------------------------------------------------------------

"""
    SplineLens(backward::ChartSpline{3}; forward = nothing, present = nothing) <: AbstractLensModel

The central grid camera's lens (Schöps et al., CVPR 2020) as a backward lens of
a [`CameraModel`](@ref): the normalized image point `x = K⁻¹u` maps to the unit
direction `normalize(backward(x))`, `backward` a uniform cubic B-spline of
3-vectors over the closed rectangle `backward.area` of the normalized plane
([`ChartSpline`](@ref); controls `[row = y, column = x]`, `size − 3` cells per
axis). [`backproject`](@ref)`(lens, x)` is NaN outside the area's half-open box
`[origin, origin + widths)` and where the point's 4 × 4 window holds an absent
node; it never throws.

`forward` (a `ChartSpline{2}` from the equidistant chart about the lens axis to
`x`, or `nothing`) is the optional fitted forward map. `present` says which
nodes the lens holds: `nothing` (every node) or a `Bool` matrix of
`size(backward.controls)` (`true` = present); an all-true mask is stored as
`nothing`, an all-false one is an `ArgumentError`. An absent node's control is
never read.

A camera of this lens has skew 0, and its `K` is a gauge: the principal point
is the pixel imaging ẑ (`CameraModel(SplineLens, source, size)`).
"""
struct SplineLens{B <: ChartSpline{3}, F <: Union{Nothing, ChartSpline{2}}, P <: Union{Nothing, AbstractMatrix{Bool}}} <: AbstractLensModel
    backward::B
    forward::F
    present::P
    function SplineLens{B, F, P}(backward::B, forward::F, present::P) where {B <: ChartSpline{3}, F <: Union{Nothing, ChartSpline{2}}, P <: Union{Nothing, AbstractMatrix{Bool}}}
        _check_present(present, size(backward.controls))
        return new{B, F, P}(backward, forward, present)
    end
end
SplineLens(backward::ChartSpline{3}; forward::Union{Nothing, ChartSpline{2}} = nothing, present = nothing) =
    _spline_lens(backward, forward, _present_mask(present, size(backward.controls)))
# The lens of its parts' own types: a function barrier over the mask, `nothing` or a `Matrix{Bool}`
# (already through `_present_mask`).
_spline_lens(backward::B, forward::F, present::P) where {B <: ChartSpline{3}, F <: Union{Nothing, ChartSpline{2}}, P <: Union{Nothing, AbstractMatrix{Bool}}} =
    SplineLens{B, F, P}(backward, forward, present)

@inline function backproject(l::SplineLens, x::StaticVector{2})
    cs = l.backward
    T = promote_type(eltype(eltype(cs.controls)), float(eltype(x)))
    nan = SVector{3, T}(NaN, NaN, NaN)
    gh, gw = size(cs.controls)
    gx, gy = _lattice_coords(cs.area, (gh, gw), x)
    # the half-open box in lattice coordinates, [1, g − 2): every window inside the grid; NaN fails it
    (1 <= gx < gw - 2 && 1 <= gy < gh - 2) || return nan
    ix, iy, wx, wy = _spline_window(gx, gy)
    _window_present(l.present, ix, iy) || return nan
    d = _eval_window(cs.controls, ix, iy, wx, wy)
    return SVector{3, T}(SVector{3}(d) / norm(d))
end

# The same camera under a diagonal change of the normalized plane, x′ = x / s
# (`CameraModel(model, K′)` with a shared principal point, s = f′/f per axis):
# the backward area scales and its controls stay (the lattice coordinate is
# invariant), the forward's output scales; exact up to rounding of the area.
function _rescale_lens(l::SplineLens, sx, sy)
    s = SVector{2, Float64}(sx, sy)
    b = l.backward
    o = GeometryBasics.origin(b.area); w = GeometryBasics.widths(b.area)
    backward = ChartSpline(Rect2{Float64}(SVector(o[1], o[2]) ./ s, SVector(w[1], w[2]) ./ s), b.controls)
    forward = l.forward === nothing ? nothing :
        ChartSpline(l.forward.area, map(c -> typeof(c)(c ./ s), l.forward.controls))
    return SplineLens(backward; forward, present = l.present)
end

# ------------------------------------------------------------------------------
# The gauge
# ------------------------------------------------------------------------------

# THE gauge of a SplineLens camera: `K` from the camera's pixel → direction map
# `pixel_direction` (any callable on a pixel `SVector{2}`, ForwardDiff-generic:
# a pixel-lattice `ChartSpline`, an old record's grid) over the calibrated pixel
# rectangle `rect`. The principal point is the pixel imaging ẑ, found by the
# projection solve from the rectangle's centre; `fx = 1/J[1,1]`,
# `fy = 1/J[2,2]` with `J = ∂(d_xy/d_z)/∂u` there; skew 0. When ẑ is not imaged
# inside `rect` (the solve does not converge there), the principal point is the
# rectangle's centre and the focal lengths the local scale there.
function _spline_gauge(pixel_direction, rect::Rect2)
    o = GeometryBasics.origin(rect); w = GeometryBasics.widths(rect)
    c = SVector{2, Float64}(o[1] + w[1] / 2, o[2] + w[2] / 2)
    ẑ = SVector(0.0, 0.0, 1.0)
    dir(p) = (d = pixel_direction(p); SVector(d[1], d[2], d[3]) / norm(SVector(d[1], d[2], d[3])))
    res(p, _) = dir(p) - ẑ
    jac(p, _) = last(_value_and_jacobian(dir, SVector{2, Float64}(p)))
    # TEMPORARY (plan 1b Task 2 replaces it by the one LM `_spline_lm`): the grid models' pixel LM
    p, ok = _pixel_lm(res, jac, c[1], c[2], o[1], o[2], o[1] + w[1] - 1, o[2] + w[2] - 1, _PROJECT_MAXITER, _PROJECT_TOL^2)
    pp = ok && all(isfinite, p) ? SVector{2, Float64}(p) : c
    _, J = _value_and_jacobian(u -> (d = pixel_direction(u); SVector(d[1] / d[3], d[2] / d[3])), pp)
    fx, fy = 1 / J[1, 1], 1 / J[2, 2]
    (isfinite(fx) && isfinite(fy) && fx > 0 && fy > 0) || throw(
        ArgumentError("SplineLens gauge: the grid's local scale at the pixel $pp is not a positive focal length ($fx, $fy)")
    )
    return LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(fx, 0.0, 0.0, 0.0, fy, 0.0, pp[1], pp[2], 1.0)))
end

# The intrinsics of a SplineLens camera built from `source`: a camera keeps its
# own K (skew 0 required); any other source takes the gauge of the pixel spline
# of `controls` over `rect`.
function _spline_intrinsics(source::CameraModel, rect, controls)
    K = SMatrix{3, 3, Float64}(calibration_matrix(source))
    iszero(K[1, 2]) || throw(ArgumentError("CameraModel(SplineLens, …): a SplineLens camera has skew 0; the source's K has skew $(K[1, 2])"))
    return LogicalIntrinsics(CameraCalibrationMat(K))
end
_spline_intrinsics(_, rect, controls) = _spline_gauge(ChartSpline(rect, controls), rect)

# ------------------------------------------------------------------------------
# The camera: from a source, its lattice, its valid domain
# ------------------------------------------------------------------------------

# The camera of intrinsics `K` whose lens lays `controls` (`[gy, gx]`) over the
# pixel rectangle `rect`: the backward area is K⁻¹ of the rectangle, computed by
# the map `backproject` applies (`_K_inv`), so a pixel on the rectangle's corner
# lands on the area's corner bit for bit.
function _spline_camera(K::LogicalIntrinsics, rect::Rect2, controls::AbstractMatrix; present = nothing)
    o = GeometryBasics.origin(rect); w = GeometryBasics.widths(rect)
    Ki = _K_inv(K)
    lo = Ki * SVector(o[1], o[2], 1.0)
    hi = Ki * SVector(o[1] + w[1], o[2] + w[2], 1.0)
    area = Rect2{Float64}(lo[1], lo[2], hi[1] - lo[1], hi[2] - lo[2])
    C = Matrix{SVector{3, Float64}}(map(v -> SVector{3, Float64}(v[1], v[2], v[3]), controls))
    return CameraModel(K, SplineLens(ChartSpline(area, C); present))
end

# The layout of a lattice from a source's keywords: the calibrated pixel
# rectangle and the grid's `(rows, cols)`, validated.
function _spline_layout(size, grid_size, calib_area)
    W, H = _spline_size(size)
    a = _area_bounds(calib_area)
    gsz = (Int(grid_size[2]), Int(grid_size[1]))
    _check_spline_area(W, H, a..., gsz)
    return _bounds_rect(a...), gsz
end

"""
    spline_lattice(SplineLens, source::CameraModel, size; grid_size, calib_area) -> CameraModel

The layout a [`SplineLens`](@ref) camera built from the camera `source` over an
image of `size` (`(width, height)` or a `Size2`) takes — the same `grid_size`,
`calib_area` and `K` (the source's own), with the same defaults, as
`CameraModel(SplineLens, source, size)` — WITHOUT sampling the source: every
control is the optical axis `(0, 0, 1)`. What a fit that solves for every
control needs (the lattice, the gauge), with none of the node interpolation.
A function source has no `K` without its sampled spline: build the camera.
"""
function spline_lattice(
        ::Type{SplineLens}, source::CameraModel, size::Union{Tuple{Integer, Integer}, Size2};
        grid_size::Tuple{Integer, Integer} = default_spline_grid(_spline_size(size)...),
        calib_area::Union{NTuple{4, Integer}, GeometryBasics.HyperRectangle{2}} =
            (0, 0, _spline_size(size)[1] - 1, _spline_size(size)[2] - 1),
    )
    rect, gsz = _spline_layout(size, grid_size, calib_area)
    axis = fill(SVector(0.0, 0.0, 1.0), gsz)
    return _spline_camera(_spline_intrinsics(source, rect, axis), rect, axis)
end

"""
    CameraModel(SplineLens, source, size; grid_size, calib_area, init = :interpolate,
                sampling_margin = 0.97, unit_controls = init !== :quasi_interpolate)
        -> CameraModel{<:LogicalIntrinsics, <:SplineLens}

The [`SplineLens`](@ref) camera initialized from `source` — a `CameraModel` (its
backprojection) or a function `pixel -> direction` (`nothing` or a non-finite
vector where it images nothing) — over an image of `size` (`(width, height)` or
a `Size2`): a lattice of `grid_size = (columns, rows)` controls (default
[`default_spline_grid`](@ref)) over the calibrated pixel area `calib_area`
(inclusive integer bounds `(min_x, min_y, max_x, max_y)` or a `Rect`; default the
whole image).

`init = :interpolate` (default) solves the cubic-prefilter tridiagonal system
per axis so the spline INTERPOLATES the source's backprojection at the grid
nodes; `init = :sample` takes each node's sample as its control (Schoenberg's
variation-diminishing approximation); both sample a camera inside
`sampling_margin` times its lens's valid domain. `init = :quasi_interpolate` is
the 4th-order local quasi-interpolant on a lattice extended one node past the
area ([`quasi_interpolant`](@ref)), with its border rule for a source whose
domain ends near or inside the area; a function source takes it.
`unit_controls = true` re-normalizes every control.

A camera source keeps its own `K` (skew ≠ 0 is an `ArgumentError`); any other
source takes the gauge of its sampled spline: the principal point is the pixel
imaging ẑ (the calibrated area's centre when ẑ is not imaged inside it) and
`fx`, `fy` the inverse local scale of `d_xy/d_z` there, skew 0.
"""
function CameraModel(
        ::Type{SplineLens}, source::_SplineSource, size::Union{Tuple{Integer, Integer}, Size2};
        grid_size::Tuple{Integer, Integer} = default_spline_grid(_spline_size(size)...),
        calib_area::Union{NTuple{4, Integer}, GeometryBasics.HyperRectangle{2}} =
            (0, 0, _spline_size(size)[1] - 1, _spline_size(size)[2] - 1),
        init::Symbol = :interpolate,
        sampling_margin::Real = 0.97,
        unit_controls::Bool = init !== :quasi_interpolate,
    )
    rect, gsz = _spline_layout(size, grid_size, calib_area)
    C = map(v -> SVector{3, Float64}(unit_controls ? _unit_or_normalize(v) : v), _init_directions(source, rect, gsz, init, sampling_margin))
    return _spline_camera(_spline_intrinsics(source, rect, C), rect, C)
end

"""
    with_present(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, present::AbstractMatrix{Bool}) -> CameraModel

The [`SplineLens`](@ref) camera holding the nodes `present` (`[gy, gx]`, `true`
= present): the same `K`, lattice and forward map, the controls of the present
nodes kept and NaN in the absent ones.
"""
function with_present(cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, present::AbstractMatrix{Bool})
    l = cam.lens
    b = l.backward
    return CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, _absent_nan(b.controls, present)); l.forward, present))
end

"""
    valid_domain(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}) -> Rect2{Float64}

Where a [`SplineLens`](@ref) camera is defined, in pixels: its backward area
mapped through `K` (the calibrated pixel area `[min, max + 1]`) when every node
is present; under a presence mask, the bounding rectangle of the lattice cells
whose 4 × 4 window holds present nodes only — where `backproject` can be finite.
An `ArgumentError` when no window is fully present.
"""
function valid_domain(cam::CameraModel{<:AbstractIntrinsics, <:SplineLens})
    l = cam.lens
    b = l.backward
    K = SMatrix{3, 3, Float64}(_K(cam.intrinsics))
    o = GeometryBasics.origin(b.area); w = GeometryBasics.widths(b.area)
    lo = K * SVector(o[1], o[2], 1.0)
    hi = K * SVector(o[1] + w[1], o[2] + w[2], 1.0)
    area = Rect2{Float64}(lo[1], lo[2], hi[1] - lo[1], hi[2] - lo[2])
    l.present === nothing && return area
    gsz = gh, gw = size(b.controls)
    # window (ix, iy) reads columns ix − 2 … ix + 1 and covers the cell between nodes ix − 1 and ix
    full = [(ix, iy) for iy in 3:(gh - 1), ix in 3:(gw - 1) if _window_present(l.present, ix, iy)]
    isempty(full) && throw(ArgumentError("valid_domain: no 4 × 4 window of the spline lens holds present nodes only"))
    jx = extrema(first, full); jy = extrema(last, full)
    a = _lattice_node(area, gsz, jx[1] - 1, jy[1] - 1)
    z = _lattice_node(area, gsz, jx[2], jy[2])
    return Rect2{Float64}(a, z - a)
end

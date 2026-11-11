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
# The lattice is the Schöps lattice of a calibrated pixel area, the closed
# rectangle `[min, max + 1]` (COLMAP: pixel corners on integers, so the area's far
# edge is the last pixel's far side): the backward area is K⁻¹ of that
# rectangle, and the lattice coordinate `1 + (g − 3)/(max + 1 − min)·(u − min)`
# of the pixel grid is the same affine map in x, so the uniform knots map exactly
# and the camera's `backproject` is the pixel grid's to rounding.
#
# This file also hosts the spline lattice's shared helpers (the 4 × 4 window,
# the presence mask, the source sampling of the initialization, the chart a
# control moves in), which the non-central model (noncentral_spline_model.jl)
# reads too.

# ------------------------------------------------------------------------------
# The window of a lattice: one tensor 4 × 4 evaluation
# ------------------------------------------------------------------------------

# Idempotent renormalization: a vector already unit to 1e-12 is kept
# BIT-EXACT (normalize(normalize(v)) drifts a ulp, which would break
# byte-exact serialization round trips); anything genuinely unnormalized is
# normalized — the reference Read() behavior.
_unit_or_normalize(v) = abs(norm(v) - 1) < 1.0e-12 ? v : normalize(v)

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

# The presence mask travels (a dense `Bool` array: the identity on the host): the
# evaluations read it on the device as on the host.
_device_present(to, ::Nothing) = nothing
_device_present(to, p::AbstractMatrix{Bool}) = Adapt.adapt(to, p)

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

# The controls with NaN in every absent slot (`present[i]` false). A debugging aid
# only: absence is the mask's (`present`), which every evaluation, the gauge and the
# axis check read; no code reads an absent slot's value, and one read anywhere shows
# as NaN.
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
# initialization, its valid domain and the support hooks (support.jl) read it.

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
# inclusive, COLMAP frame: the area is the closed [min_x, max_x + 1] × [min_y, max_y + 1]).
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
default_spline_grid(width, height) = _node_counts(width, height, 64)
# THE node-count rule of a lattice over a `w₁ × w₂` area at the node spacing `h`: ⌈wᵢ / h⌉ + 3
# nodes per axis (the cubic window's three border nodes), at least 4 — the reference's grid
# (`h` = 64 px) and the spline fits' spacing search alike. A quotient within rounding of
# an integer is that integer (the `1 − 1e-12`).
_node_counts(w1::Real, w2::Real, h::Real) = (max(4, ceil(Int, w1 / h * (1 - 1.0e-12)) + 3), max(4, ceil(Int, w2 / h * (1 - 1.0e-12)) + 3))

# ------------------------------------------------------------------------------
# Initialization from a source: the direction controls of a pixel lattice
# ------------------------------------------------------------------------------

# Cubic B-spline interpolation ("prefilter") matrix along one axis: at an
# interior node the spline reads (c[j−1] + 4c[j] + c[j+1])/6, so solving the
# tridiagonal system per axis makes the spline INTERPOLATE the node samples
# instead of smoothing them (the node samples as controls would be Schoenberg's
# variation-diminishing approximation, no init of the stack's). The two
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

# The direction controls of `init = :interpolate` on the lattice of `rect` and
# `gsz = (rows, cols)`, sampled from the camera `model`.
function _directions_from(
        model::CameraModel, rect::Rect2, gsz;
        sampling_margin::Real   # the caller supplies it; the public default lives on the constructors
    )
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
    Ay = _prefilter_matrix(gh)
    Ax = _prefilter_matrix(gw)
    C = ntuple(k -> Ay \ (([S[gy, gx][k] for gy in 1:gh, gx in 1:gw]) / transpose(Ax)), 3)
    return [SVector(C[1][gy, gx], C[2][gy, gx], C[3][gy, gx]) for gy in 1:gh, gx in 1:gw]
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
_source_direction(c::CameraModel, px) = _source_direction(p -> backproject(c, p), px)

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
    init === :interpolate || throw(
        ArgumentError(
            "spline grid init must be :interpolate (cubic prefilter) or :quasi_interpolate (the local quasi-interpolant), got :$init"
        )
    )
    source isa CameraModel || throw(
        ArgumentError(
            "spline grid init :$init samples a camera model inside its valid domain; a $(nameof(typeof(source))) source " *
                "takes init = :quasi_interpolate"
        )
    )
    return _directions_from(source, rect, gsz; sampling_margin)
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
axis). [`backproject`](@ref)`(lens, x)` is NaN outside the area's closed box
`[origin, origin + widths]` and where the point's 4 × 4 window holds an absent
node; it never throws.

`forward` (a `ChartSpline{2}` from the equidistant chart about ẑ to
`x`, or `nothing`) is the optional fitted forward map. `present` says which
nodes the lens holds: `nothing` (every node) or a `Bool` matrix of
`size(backward.controls)` (`true` = present); an all-true mask is stored as
`nothing`, an all-false one is an `ArgumentError`. An absent node's control is
never read.

A camera of this lens has skew 0 and positive focal lengths (an
`ArgumentError` wherever one is built otherwise), and its `K` is a gauge: the
principal point is the pixel imaging ẑ (`CameraModel(SplineLens, source,
size)`). Where ẑ is not imaged inside the calibrated area, the gauge falls back
to the area's centre, a pixel that does not image ẑ: the methods that read the principal point
as the optical axis — `undistort` and `undistort!` onto the gauge pinhole,
`distortion_map`, `valid_rect`, `undistorted_rect`, `angle_of_view`,
`horizon_radius` and `monotone_radius`, `image_circle(camera, sensor, d)`,
`view_overlap` — then throw an `ArgumentError` naming the off-axis gauge.
"""
struct SplineLens{B <: ChartSpline{3}, F <: Union{Nothing, ChartSpline{2}}, P <: Union{Nothing, AbstractMatrix{Bool}}} <: AbstractLensModel
    backward::B
    forward::F
    present::P
    function SplineLens{B, F, P}(backward::B, forward::F, present::P) where {B <: ChartSpline{3}, F <: Union{Nothing, ChartSpline{2}}, P <: Union{Nothing, AbstractMatrix{Bool}}}
        _check_present(present, size(backward.controls))
        return new{B, F, P}(backward, forward, present)
    end
    # A value already validated (its device form, or a mask through `_present_mask`): no check, as `ChartSpline`'s.
    SplineLens{B, F, P}(backward::B, forward::F, present::P, ::Val{:unchecked}) where {B, F, P} = new{B, F, P}(backward, forward, present)
end
SplineLens(backward::ChartSpline{3}; forward::Union{Nothing, ChartSpline{2}} = nothing, present = nothing) =
    _spline_lens(backward, forward, _present_mask(present, size(backward.controls)), Val(:unchecked))
# The lens of its parts' own types: a function barrier over the mask, `nothing` or a `Matrix{Bool}`
# (checked, or already through `_present_mask` with `Val(:unchecked)`).
_spline_lens(backward::B, forward::F, present::P, v::Val...) where {B, F, P} = SplineLens{B, F, P}(backward, forward, present, v...)

# THE check of a spline camera's K — a SplineLens camera's and a
# NoncentralSplineModel's, built or read: a gauge of skew 0 and positive focal
# lengths (`_pixel_area`, `valid_domain`, `support` and the pixel partials of
# `spline_weights` read the calibrated rectangle as a rectangle under K).
function _check_spline_gauge(intrinsics::AbstractIntrinsics)
    M = _K(intrinsics)
    iszero(M[1, 2]) || throw(ArgumentError("a spline camera's K is a gauge of skew 0; the K has skew $(M[1, 2])"))
    (M[1, 1] > 0 && M[2, 2] > 0) ||
        throw(ArgumentError("a spline camera's K has positive focal lengths; got fx = $(M[1, 1]), fy = $(M[2, 2])"))
    return intrinsics
end
# Every SplineLens camera is built here (the constructors, the readers, a re-gauge, a device copy): its K checked.
CameraModel(intrinsics::AbstractIntrinsics, lens::SplineLens) =
    CameraModel{typeof(intrinsics), typeof(lens)}(_check_spline_gauge(intrinsics), lens)

# The rounding slack of the box's edges, in ulps of the point's magnitude
# (see `_in_backward`).
const _EDGE_ULPS = 4

# Whether the lens images the normalized point `x`: inside the backward area's
# closed box, tested in lattice coordinates, `[1 − τ, g − 2 + τ]` per axis (a NaN
# or Inf point fails it), and its 4 × 4 window present. Device-safe.
#
# The slack τ = c·ε·slope·(|x| + |o|) on both edges is the rounding of
# `x = K⁻¹u` and of the lattice coordinate, so a point on the calibrated area's
# own edges (u = min and u = max + 1, the image's near and far edges by default) is
# imaged in Float32 as in Float64 (K⁻¹ rounds x = W of a 4256-px image to the
# lattice coordinate 68.00000000000001 of a 70-node axis): `K⁻¹u` rounds the two
# entries of K⁻¹'s row, their product and the sum, each ≤ ε/2 of a term bounded by
# |x| + |o| (o the area's origin); the lattice coordinate rounds `o` and the
# difference — at most 3ε(|x| + |o|) in x in all, so c = `_EDGE_ULPS` = 4 bounds
# it with a margin. ε is the coarser of the point's and the controls' precision
# (a Float32 camera's K⁻¹ is Float32 even for a Float64 pixel). A window at
# gx ∈ [1 − τ, 1) is `_continued_window`'s clamped one (the first), and one at
# gx ∈ [g − 3, g − 2 + τ] the last (at its local coordinate 1 on the far edge), as
# `backward(x)` evaluates them.
@inline function _in_backward(l::SplineLens, x)
    cs = l.backward
    gsz = gh, gw = size(cs.controls)
    gx, gy = _lattice_coords(cs.area, gsz, x)
    ix, iy, _, _ = _continued_window(gx, gy, gw, gh)
    return _in_backward(l, x, gx, gy, ix, iy)
end
# The same test given the point's lattice coordinates `(gx, gy)` and its continued
# window `(ix, iy)`, which `backproject` has already formed for the evaluation. The
# slack is formed only for a point outside the exact box `[1, g − 2]` (τ ≥ 0, so a
# coordinate inside it meets both bounds whatever τ is).
@inline function _in_backward(l::SplineLens, x, gx, gy, ix, iy)
    cs = l.backward
    gsz = gh, gw = size(cs.controls)
    all(isfinite, x) || return false
    if !(1 <= gx <= gw - 2 && 1 <= gy <= gh - 2)
        o = GeometryBasics.origin(cs.area)
        sx, sy = _lattice_scales(Float64, cs.area, gsz)
        ε = max(eps(float(typeof(_primal(x[1])))), eps(eltype(eltype(cs.controls))))   # a dual point: its primal's precision
        τx = _EDGE_ULPS * ε * sx * (abs(x[1]) + abs(o[1]))
        τy = _EDGE_ULPS * ε * sy * (abs(x[2]) + abs(o[2]))
        (1 - τx <= gx <= gw - 2 + τx && 1 - τy <= gy <= gh - 2 + τy) || return false
    end
    return _window_present(l.present, ix, iy)
end

# The ray of the normalized point `x`: the backward spline's value there (`backward(x)`:
# its lattice coordinates and continued window formed once, for the value and for
# `_in_backward`), normalized where the lens images `x`, NaN elsewhere. The scalar and
# the batched backprojection both run it, the batch with the lattice slopes `s` of its
# points' float type formed once (`_lattice_scales`). (The spline is evaluated,
# continued, outside the box too: a non-finite point gives NaN weights, never an error.)
@inline backproject(l::SplineLens, x::StaticVector{2}) =
    _backproject_lens(l, x, _lattice_scales(float(eltype(x)), l.backward.area, size(l.backward.controls)))
@inline function _backproject_lens(l::SplineLens, x::StaticVector{2}, s)
    cs = l.backward
    gh, gw = size(cs.controls)
    gx, gy = _lattice_coords_at(cs.area, s, x)
    ix, iy, wx, wy = _continued_window(gx, gy, gw, gh)
    d = _eval_window(cs.controls, ix, iy, wx, wy)
    T = eltype(d)
    return _in_backward(l, x, gx, gy, ix, iy) ? SVector{3, T}(SVector{3}(d) / norm(d)) : SVector{3, T}(NaN, NaN, NaN)
end

# A backward lens: its direction Jacobian differentiates `backproject` (camera_models.jl).
LensDirection(::Type{<:SplineLens}) = BackwardLens()

# Its K is a gauge (camera_models.jl, `IntrinsicsRole`): a camera's parameters are the lens's alone.
IntrinsicsRole(::Type{<:SplineLens}) = GaugeIntrinsics()

# The lens's parameters: its backward spline's controls at the present nodes, one
# coordinate vector in the lattice's one layout (`_control_vector`, chart_spline.jl),
# named `c_<row>_<col>_<x|y|z>`, unbounded. A heap vector: a lattice holds 10³–10⁴
# coordinates. `with_parameters` keeps the area and the mask; a lens with a forward
# spline refuses it — the forward is a fit of the backward, which a new backward
# would leave stale (and `project` reads the forward). `ParameterManifold`'s chart is
# static, so it refuses this heap vector (lens_chart.jl); the controls' chart is a
# `ControlPatchManifold`.
parameter_vector(l::SplineLens) = _control_vector(l.backward.controls, l.present)
parameter_names(l::SplineLens) = _control_names(l.backward.controls, l.present)
parameter_bounds(l::SplineLens) = _unbounded_bounds(parameter_vector(l))
_bound_kinds(l::SplineLens) = _open_kinds(parameter_vector(l))
function with_parameters(l::SplineLens, θ::AbstractVector)
    l.forward === nothing || throw(
        ArgumentError(
            "with_parameters: a SplineLens with a forward spline — the forward is a fit of the backward, which new controls would leave stale; " *
                "drop it first: SplineLens(lens.backward; present = lens.present)"
        )
    )
    b = l.backward
    return SplineLens(ChartSpline(b.area, _with_control_vector(b.controls, l.present, θ)); present = l.present)
end
_rebuild(l::SplineLens, θ) = with_parameters(l, θ)

# The largest normalized radius about x = 0 (the gauge's principal point) whose disc
# lies inside the lens's support: its backward area ∩ the cells whose 4 × 4 window is
# fully present (`monotone_radius`'s docstring, projections.jl).
function monotone_radius(l::SplineLens)
    _check_lens_axis(l, "monotone_radius (horizon_radius)")
    b = l.backward
    o = GeometryBasics.origin(b.area); w = GeometryBasics.widths(b.area)
    r = max(0.0, min(-o[1], o[1] + w[1], -o[2], o[2] + w[2]))     # the area's edges (0 outside it)
    l.present === nothing && return r
    gsz = gh, gw = size(b.controls)
    for iy in 3:(gh - 1), ix in 3:(gw - 1)
        _window_present(l.present, ix, iy) && continue
        c = _window_area(b.area, gsz, (iy - 2):(iy + 1), (ix - 2):(ix + 1))   # the window's own cell
        lo = SVector{2, Float64}(GeometryBasics.origin(c)); hi = lo + SVector{2, Float64}(GeometryBasics.widths(c))
        r = min(r, norm(max.(lo, min.(zero(lo), hi))))             # the disc's distance to an absent cell
    end
    return r
end

# The same camera under any other skew-0 K′ (`CameraModel(model, K′)`): the
# normalized plane changes by the affinity x′ = x / s + t, s = f′/f and
# t = (c − c′)/f′ per axis (u = f x + c = f′ x′ + c′), so the backward area
# moves by it and its controls stay (the lattice coordinate is invariant):
# exact up to rounding of the area, a principal-point move included — the
# K of a gauge lens is a choice of the normalized plane, not data. The
# forward's output moves by the same affinity (its controls do: a B-spline's
# weights sum to one), and its chart is about ẑ (`forward_chart`), which no K
# moves: every projection is kept too.
function _regauge_lens(l::SplineLens, K, K′)
    s = SVector{2, Float64}(K′[1, 1] / K[1, 1], K′[2, 2] / K[2, 2])
    t = SVector{2, Float64}((K[1, 3] - K′[1, 3]) / K′[1, 1], (K[2, 3] - K′[2, 3]) / K′[2, 2])
    b = l.backward
    o = GeometryBasics.origin(b.area); w = GeometryBasics.widths(b.area)
    backward = ChartSpline(Rect2{Float64}(SVector(o[1], o[2]) ./ s .+ t, SVector(w[1], w[2]) ./ s), b.controls)
    forward = l.forward === nothing ? nothing :
        ChartSpline(l.forward.area, map(c -> typeof(c)(c ./ s .+ t), l.forward.controls))
    return SplineLens(backward; forward, present = l.present)
end

# The device form (`Adapt.adapt(backend, lens)`): the splines' controls and the
# presence mask move, nothing is re-validated.
Adapt.adapt_structure(to, l::SplineLens) =
    _spline_lens(Adapt.adapt(to, l.backward), Adapt.adapt(to, l.forward), _device_present(to, l.present), Val(:unchecked))
Adapt.adapt_structure(to, m::CameraModel{<:AbstractIntrinsics, <:SplineLens}) = CameraModel(m.intrinsics, Adapt.adapt(to, m.lens))

# The controls convert as they are (no re-normalization); the areas and the mask stay.
convert_eltype(::Type{T}, l::SplineLens) where {T <: AbstractFloat} =
    _spline_lens(convert_eltype(T, l.backward), l.forward === nothing ? nothing : convert_eltype(T, l.forward), l.present)

# ------------------------------------------------------------------------------
# Projection: one LM in the lens's own coordinates, or the forward spline
# ------------------------------------------------------------------------------

# The host budget of the one projection solve: a cold start from the area's
# centre gets 100 LM iterations (the batched form takes warm starts and a
# caller-set `iters`). ONE convergence criterion for every projection solve, host
# and batched: the residual norm below `_PROJECT_TOL` — the reference's 1e-12 cost.
const _PROJECT_MAXITER = 100
const _PROJECT_TOL = 1.0e-6

# THE two-unknown LM of the spline models: `fj(p) -> (r, ∂r/∂p)`, the residual and
# its Jacobian from ONE evaluation at `p` (a spline's basis weights once per point:
# every trial point's Jacobian is the next iteration's when the step is accepted),
# from `x₀`, the point bounded to the box `[lb, ub]` (a step is clipped at the
# bound), for at most `maxiter` LM iterations; converged means a final cost below
# `tol2`. Returns the point, whether it converged and the Jacobian at the point
# (the callers that need only the first two destructure them; the lens inverse
# reads its determinant). `s` is the problem's unit per axis, as the reference's
# pixel: its pixel constants are read in it — the step tolerance `ϵx = 1e-10 / s`,
# the DᵀD clamp `[1e-12, 1e16]·s²`. Device-safe — static arrays only, no
# allocation, no Union return — so the batched kernels run it per ray.
# Generic in the float type `T` of its arguments (Float32 in a device
# kernel; at Float64 the reference's operations and constants exactly). Its stops
# are precision-aware: the step tolerance is the reference's in ulps of T
# (`1e-10·eps(T)/eps(Float64)`), and the absolute residual and gradient stops are
# the reference's `eps(Float64)` or what T resolves of a residual of scale `fs`
# (`(16 eps(T) fs)²`), whichever is larger — so a Float32 solve stops where
# Float32 can resolve it, and at Float64 with `fs` of order one (the SplineLens
# residual, a difference of unit directions) every stop is the reference's
# exactly. The ray–board root of a MappedSurface calls it with `fs` its chart's
# diagonal (mapped_surface.jl `_board_root`). `bound` limits every step: `nothing`
# (the reference's: no limit), or a relative radius `b`, a step longer than
# `b·max(‖x‖, 1)` shortened to it along its direction — COLMAP's
# IterativeUndistortion clipping (`b = 0.1`), which the parametric lens inverse
# reads (lens_rays.jl `_inverse_lm`). `cost_floor` is the absolute stops' floor, the
# reference's `eps(Float64)` by default: a cost below it ends the solve, a residual
# of order one resolved to √eps. A root that must be resolved to rounding (the lens
# inverse: a seed whose residual is 1e-9 is not its root) gives 0, leaving the
# stops at what T resolves.
#
# It is the ONE device-side exception to the stack's one LM driver: NLLSsolver
# builds heap problems and cannot run inside a KernelAbstractions kernel, so
# this kernel is the stated single instance the retired-names rule allows
# (`check_naming`'s RETIRED_NAMES: the hand-rolled LM cores stay retired).
#
# Levenberg's and Marquardt's damped Gauss–Newton (Levenberg 1944; Marquardt
# 1963) with Marquardt's diagonal scaling, specialized to N = 2 and derived
# from a static bounded `lmsolve` (Copyright (c) 2023 kamesy, MIT License),
# with its defaults and operations in its order — the damping (λ₀ = 1e-6,
# ×1/13 on ρ > 0.75, ×3 on ρ < 1e-3 or a rejected step, clamped to
# [1e-28, 1e32]), the running-maximum DᵀD clamped to [1e-12, 1e16], the
# unrolled 2 × 2 Cholesky, the ftol = gtol = eps and xtol = 1e-10 stops (at
# s = 1). NOT the same bits as a run of that routine: `muladd` and
# StaticArrays' small products fuse as LLVM decides per compilation context.
# Measured against a recorded run of it on 1000 seeded problems
# (test/fixtures/pixel_lm), at s = 1: at most 5.5e-12 px (Julia 1.13) /
# 4.5e-8 px (1.12) apart, 1.8e-6 px on the review's 20 000-problem probe,
# converged flags identical; gated at 2e-6 px (pixel_lm_reference_tests.jl).
@inline function _spline_lm(
        fj::FJ, x₀::SVector{2, T}, lb::SVector{2, T}, ub::SVector{2, T},
        s::SVector{2, T}, maxiter::Int, tol2::T, fs::T = one(T), bound::B = nothing, cost_floor::T = T(eps(Float64))
    ) where {FJ, T <: AbstractFloat, B <: Union{Nothing, T}}
    ϵ = eps(T)
    ϵx = T(1.0e-10 * (eps(T) / eps(Float64))) ./ s   # the reference's 1e-10 at Float64, the same in ulps of T
    # the absolute stops: the cost floor (the reference's eps(Float64)), or what T resolves of a residual of scale fs
    ϵF = max(cost_floor, (16 * ϵ * fs)^2)
    ϵG = max(cost_floor, (16 * ϵ * fs)^2 * maximum(s) / _LM_AREA_UNITS)
    λ = T(1.0e-6)
    x = min.(max.(x₀, lb), ub)
    f, Jx = fj(x)
    F = sum(abs2, f)
    F < ϵF && return x, F < tol2, Jx
    g = Jx' * f
    norm(g, Inf) < ϵG && return x, F < tol2, Jx
    JtJ = Jx' * Jx
    DtD = clamp.(diag(JtJ), T(1.0e-12) .* s .^ 2, T(Inf))
    DtD_max = T(1.0e16) .* s .^ 2
    iter = 0
    while iter < maxiter
        iter += 1
        accepted = false
        ac = zero(T)
        pr = zero(T)
        ρ = zero(T)
        xk = x
        fk = f
        Jk = Jx
        Fk = F
        p, info = _spline_lm_increment(JtJ, λ, DtD, g)
        # a step clipped to zero by either bound is rejected, as a failed solve is
        clipped = false
        if info == 0
            p = _bounded_increment(bound, p, x)
            p = min.(x .- lb, p)
            clipped = all(iszero, p)
            if !clipped
                p = max.(x .- ub, p)
                clipped = all(iszero, p)
            end
        end
        if info == 0 && !clipped
            xk = x .- p
            fk, Jk = fj(xk)
            Fk = sum(abs2, fk)
            if Fk < F
                ac = F - Fk
                pr = dot(p, (@. λ * DtD * p + g))
                ρ = ac / pr
                accepted = true
            end
            if all(>(0), (@. abs(p) < ϵx * (ϵx + abs(x))))
                if accepted
                    x, f, F, Jx = xk, fk, Fk, Jk
                end
                break
            end
        end
        if accepted
            x = xk
            f = fk
            Fold = F
            F = Fk
            Jx = Jk
            # the reference routine's ftol test reads its swapped buffers: `Fk` is then the old F
            abs(ac) < ϵ * Fold && pr < ϵ * Fold && ρ < 2 && break
            g = Jx' * f
            norm(g, Inf) < ϵG && break
            JtJ = Jx' * Jx
            DtD = clamp.(diag(JtJ), DtD, DtD_max)
            if ρ > T(0.75)
                λ *= T(1 / 13)
            elseif ρ < T(1.0e-3)
                λ *= 3
            end
        else
            λ *= 3
        end
        λ = clamp(λ, T(1.0e-28), T(1.0e32))
    end
    return x, F < tol2, Jx
end

# The step `p` (the LM subtracts it) within the relative radius `b·max(‖x‖, 1)`.
@inline _bounded_increment(::Nothing, p, x) = p
@inline function _bounded_increment(b::T, p::SVector{2, T}, x::SVector{2, T}) where {T}
    r2 = b * b * max(x[1] * x[1] + x[2] * x[2], one(T))
    n2 = p[1] * p[1] + p[2] * p[2]
    return ifelse(n2 > r2, p * sqrt(r2 / n2), p)
end

# The unrolled Cholesky solve of (JᵀJ + λ·DᵀD) p = g at N = 2, the reference
# routine's operations in its order; `info` ≠ 0 when a damped diagonal is
# negative (a rejected step).
@inline function _spline_lm_increment(JtJ, λ, DtD, g)
    x1 = g[1]
    x2 = g[2]
    A11 = muladd(λ, DtD[1], JtJ[1, 1])
    A11 >= zero(A11) || return zero(SVector{2, typeof(A11)}), 1
    A21 = JtJ[2, 1]
    A22 = muladd(λ, DtD[2], JtJ[2, 2])
    A22 >= zero(A22) || return zero(SVector{2, typeof(A22)}), 2
    A22 = muladd(-A21, A21 / A11, A22)
    x2 = muladd(-x1, A21 / A11, x2)
    x2 /= A22
    x1 = muladd(-x2, A21, x1)
    x1 /= A11
    return SVector(x1, x2), 0
end

# The units an area is read in by `_spline_lm_box`: the reference's pixel constants
# are kept at its 1000-px calibrated area.
const _LM_AREA_UNITS = 1000

# The scale-free setting of `_spline_lm` on a rectangle `area` of any unit (the
# lens's normalized plane, a pixel rectangle): the area read as
# `_LM_AREA_UNITS` units per axis, `s = _LM_AREA_UNITS ./ widths` — the
# reference's pixel constants exactly on its 1000-px calibrated area, and the same
# relative tolerances on any other, so the solve depends on nothing but the area
# (not on K). The box is the area's closed one, `lb = origin`,
# `ub = origin + widths`, the rectangle `backproject` images (`_in_backward`), so a
# ray imaged on the area's far edge projects onto it.
@inline function _spline_lm_box(area::Rect2)
    o = GeometryBasics.origin(area); w = GeometryBasics.widths(area)
    lb = SVector{2, Float64}(o[1], o[2])
    ws = SVector{2, Float64}(w[1], w[2])
    return lb, lb + ws, _LM_AREA_UNITS ./ ws
end

# THE central projection solve on a spline of direction controls `controls`
# (`[gy, gx]`) over the rectangle `area`: LM (`_spline_lm`) on the direction
# residual d̂(x) − t̂, d̂ the backward map, and its analytic Jacobian, both from one
# evaluation of the basis weights per point (`_spline_direction_differential`), from
# `x₀`, bounded to `[lb, ub]` (inside the area: every window stays in the grid).
# Device-safe — no allocation, no Union return. Returns the point and whether the
# solve converged.
@inline function _spline_project(controls, area::Rect2, t::StaticVector{3}, x₀::SVector{2, Float64}, lb, ub, s, maxiter::Int, tol2::Float64)
    # scaled before the norm, so a tiny or huge ray does not under/overflow
    tmax = max(abs(t[1]), abs(t[2]), abs(t[3]))
    # a zero or non-finite ray, or start, has no projection
    (tmax > 0 && isfinite(tmax) && all(isfinite, x₀)) || return SVector(NaN, NaN), false
    tn = normalize(SVector{3, Float64}(t[1] / tmax, t[2] / tmax, t[3] / tmax))
    fj(p) = (r = _spline_direction_differential(controls, area, p); (SVector{3, Float64}(r[2]) - tn, r[3]))
    x, ok = _spline_lm(fj, x₀, lb, ub, s, maxiter, tol2)
    return x, ok
end

# THE direction of a spline of direction controls over `area` at `x`, and its
# Jacobian: `(d, n⁻¹S, ∂d/∂x, n)` (`n = ‖S‖`) from the value and partials of
# `_spline_value_and_partials` (one evaluation of the basis weights, the continued
# window: inside the area it is the point's own cell, and on either edge's rounding
# slack the value and partials still agree) through the normalize differential
# `(I − d dᵀ)/‖S‖`. The direction in two roundings: `d = S/‖S‖`, `backproject`'s, and
# `‖S‖⁻¹S`, `normalize`'s, the LM's residual. In the precision of `x` and the
# controls. Device-safe (isbits, no allocation).
@inline _spline_direction_differential(controls::AbstractMatrix, area, x::StaticVector{2}) =
    _direction_differential(_spline_value_and_partials(controls, area, x)...)
@inline function _direction_differential(S, Sx, Sy)
    n = norm(S)
    d = S / n
    m = inv(n)
    # `sum(d .* Sx)`, not `dot`: StaticArrays' `dot` is an `@simd` reduction, whose order
    # (so the last digit) is the compilation context's; the unrolled sum has one order
    return d, m * S, hcat((Sx - d * sum(d .* Sx)) * m, (Sy - d * sum(d .* Sy)) * m), n
end

# The direction `d = S/‖S‖` and its Jacobian `∂d/∂x` (3 × 2): the lens's
# `backproject_with_jacobian`, the non-central residual's.
@inline function _spline_direction_and_jacobian(controls::AbstractMatrix, area, x::StaticVector{2})
    d, _, J = _spline_direction_differential(controls, area, x)
    return d, J
end

# THE implicit derivatives of a converged central projection: the point `x` that
# `_spline_project` solved for the ray `t` (`d(x) = t̂`) through the spline of
# direction controls `controls` over `area`, differentiated by the implicit function
# theorem on the direction residual `f(x, c, t) = d(x; c) − t̂(t)` — never through the
# LM (no dual enters it). With `Jx = ∂d/∂x` (3 × 2, its columns ⟂ d) and
# `H = (JxᵀJx)⁻¹Jxᵀ`: `∂x/∂t = H (I − t̂t̂ᵀ)/‖t‖`, and per control of the point's
# window, `controls[iy − 3 + r, ix − 3 + c]`, `∂x/∂c = −(w[r, c]/‖S‖)·H` (the
# control's weight times the normalize differential `(I − ddᵀ)/‖S‖`, and
# `H (I − ddᵀ) = H`). Returns `(∂x/∂t, −H/‖S‖, ix, iy, w)`, `w` the window's weights
# (`spline_weights`'s). Exact at a zero residual; the solve's tolerance bounds the
# term it omits. PRECONDITION: `x` is solved to rounding — `_solved_projection`'s, never
# a solve warm-started within √eps of its answer, which `_spline_lm` returns as it is
# (the reference's F < eps stop), so that x*(c) would be a step function of the
# controls. A SplineLens camera's bundle refinement reads it
# (VisualGeometryToolkitEstimation's `refine_camera_and_poses`).
function _projection_jacobians(controls::AbstractMatrix{<:SVector{3}}, area::Rect2, x::StaticVector{2}, t::StaticVector{3})
    gx, ix, iy, wx, wy, dwx, dwy = _window_weights(size(controls), area, x)   # the basis weights once: the differential's and w
    _, _, Jx, n = _direction_differential(_window_sums(controls, area, gx, ix, iy, wx, wy, dwx, dwy)...)
    H = inv(Jx' * Jx) * Jx'
    tn = norm(t)
    t̂ = t / tn
    return H * (SMatrix{3, 3, Float64}(LinearAlgebra.I) - t̂ * t̂') / tn, -H / n, ix, iy, _window_tensor(wx, wy)
end

# THE projection solve that `_projection_jacobians` differentiates: `_spline_project` of
# the ray `t` through `controls` over `area`, started at the area's centre and bounded
# to the box `(lb, ub, s)` (a lens's `_spline_lm_box`) — never from a previous answer,
# which the LM would return unmoved once within √eps (1.5e-8 rad) of the ray; from the
# centre its quadratic convergence lands to rounding, so x*(c, t) is smooth. `area` may
# be one cell of a lattice with its 4 × 4 window as `controls` (the pieces extrapolate
# off the cell): the window a refinement holds between rebuilds.
@inline _solved_projection(controls, area::Rect2, t::StaticVector{3}, lb, ub, s) =
    _spline_project(controls, area, t, SVector{2, Float64}(centroid(area)), lb, ub, s, _PROJECT_MAXITER, _PROJECT_TOL^2)

# A SplineLens's bearing at the normalized point `x` and its Jacobian (the BackwardLens
# route of `backproject_with_jacobian`): the analytic one, NaN where the lens does not
# image `x`.
@inline function _backward_value_and_jacobian(l::SplineLens, x::StaticVector{2})
    T = promote_type(float(eltype(x)), eltype(eltype(l.backward.controls)))
    _in_backward(l, x) || return SVector{3, T}(NaN, NaN, NaN), SMatrix{3, 2, T, 6}(ntuple(_ -> T(NaN), Val(6)))
    return _spline_direction_and_jacobian(l.backward.controls, l.backward.area, x)
end

const _FORWARD_CHART = equidistant_chart(SVector(0.0, 0.0, 1.0))

"""
    forward_chart(SplineLens) -> chart

The chart of every [`SplineLens`](@ref)'s forward spline: the equidistant chart
about ẑ ([`equidistant_chart`](@ref)`(ẑ)`), fixed in the camera frame, so no `K`
moves it (a re-gauge maps the forward's output by the normalized plane's
affinity and keeps every projection exactly). THE one derivation: `project`
evaluates a forward on it and a fit of a forward is made on it.
"""
forward_chart(::Type{SplineLens}) = _FORWARD_CHART

# Where a lattice images ẑ: THE solve of ẑ through `controls` over `area` (from the
# area's centre, the scale-free box) and whether it converged there onto a window
# whose nodes `present` holds — the gauge's choice of principal point
# (`_spline_gauge`: the pixel imaging ẑ, else the rectangle's centre, the fallback)
# and the off-axis check's condition below, one predicate for both. Absence is read
# from `present` alone, whatever the absent slots of `controls` hold.
function _imaged_z(controls::AbstractMatrix, area::Rect2, present)
    p, ok = _solved_projection(controls, area, SVector(0.0, 0.0, 1.0), _spline_lm_box(area)...)
    ok = ok && all(isfinite, p) && _window_present(present, spline_window(area, size(controls), p)[1:2]...)
    return ok ? (SVector{2, Float64}(p), true) : (SVector{2, Float64}(centroid(area)), false)
end

# Whether a SplineLens camera's gauge is the off-axis fallback — ẑ not imaged inside
# the backward area (`_imaged_z`, the gauge's own condition), so the principal point is
# the area's centre, a pixel that does not image ẑ — the premise of every method that
# reads the principal point as the optical axis (`undistort` onto the gauge pinhole,
# `valid_rect`, `angle_of_view`, `horizon_radius`, the footprint, …): THE check, an
# `ArgumentError` naming `what`. Every other lens has nothing to check.
_check_lens_axis(::AbstractLensModel, what) = nothing
function _check_lens_axis(l::SplineLens, what)
    last(_imaged_z(l.backward.controls, l.backward.area, l.present)) || throw(
        ArgumentError(
            "$what: the SplineLens camera's lattice does not image ẑ inside its calibrated area, so its principal point " *
                "is the off-axis gauge's fallback (the area's centre); the method reads the principal point as the " *
                "optical axis, so it has no meaning here"
        )
    )
    return nothing
end

# THE projection of a SplineLens: the normalized point of the ray `X` and
# whether the lens images it. With a forward spline, its value at the ray's
# point of the equidistant chart about ẑ — imaged within the
# forward's fitted chart (`_in_fitted_chart`) and where the backward images it
# (`_in_backward`: the box, a present window). Without one, the LM from `x₀`
# (the scale-free `_spline_lm_box`), converged and onto a present window. The
# scalar `project` and the batched kernel both run it; NaN when not imaged.
@inline function _project_lens(l::SplineLens, X::StaticVector{3}, x₀::SVector{2, Float64}, maxiter::Int, tol2::Float64)
    nan = SVector(NaN, NaN)
    if l.forward !== nothing
        c = _FORWARD_CHART(X)
        _in_fitted_chart(l.forward, c) || return nan, false
        y = l.forward(c)
        x = SVector{2, Float64}(y[1], y[2])
        return _in_backward(l, x) ? (x, true) : (nan, false)
    end
    b = l.backward
    lb, ub, s = _spline_lm_box(b.area)
    x₀ = all(isfinite, x₀) ? x₀ : SVector{2, Float64}(centroid(b.area))      # a non-finite seed (a failed warm start): the cold start's fallback
    x, ok = _spline_project(b.controls, b.area, X, x₀, lb, ub, s, maxiter, tol2)
    return ok && _in_backward(l, x) ? (x, true) : (nan, false)
end


# The cold start of a projection solve of the ray (or camera-frame point) `X` through
# the lens `l`: its pinhole point `(X₁/X₃, X₂/X₃)` — the gauge's first-order inverse
# (`K` images ẑ at the principal point, at the local scale) — where the lens images it
# (`_in_backward`: in the area, its window present), else the backward area's centre
# (a ray at or behind the image plane, a point off the area or on an absent window, a
# non-finite ray). Only the start: the solve's answer is the same to its tolerance
# (measured 2.7e-12 px from the centre start over the golden cameras, the imaged mask
# identical; tested).
@inline function _cold_start(l::SplineLens, X::StaticVector{3})
    c = SVector{2, Float64}(centroid(l.backward.area))
    X[3] > 0 || return c
    p = SVector{2, Float64}(X[1] / X[3], X[2] / X[3])
    return _in_backward(l, p) ? p : c
end

"""
    project(lens::SplineLens, X::StaticVector{3}) -> SVector{2}

The normalized image point of the ray `X` (`K` maps it to the pixel), NaN
where the lens does not image it. With a forward spline (`lens.forward`), its
value at the ray's point of the equidistant chart about ẑ (fixed in the camera
frame, so no choice of K moves it), NaN outside the forward's fitted chart (more than one
knot cell past its area) or where the backward spline does not image the
point. Without one, Levenberg–Marquardt on the backward spline's direction
mismatch (100 iterations, converged below a 1e-6 residual) from the ray's
pinhole point `(X₁/X₃, X₂/X₃)` where the lens images that point, else from the
backward area's centre; NaN unless it converges onto a point whose window is present.
"""
@inline project(l::SplineLens, X::StaticVector{3}) =
    first(_project_lens(l, X, _cold_start(l, X), _PROJECT_MAXITER, _PROJECT_TOL^2))

# THE batched projection: per ray `_project_lens` from its seed — `seed_map` of
# `seeds[i]` (the pixel seed's normalized point, or a normalized seed as it is), the
# cold start (`_cold_start`) where that is not finite or without seeds — and the point
# out through `out_map` (the pixel affinity, or the normalized point as it is).
@kernel function _spline_project_kernel!(out, ok_out, lens, out_map, seed_map, @Const(rays), seeds, iters::Int, tol2::Float64)
    i = @index(Global)
    @inbounds begin
        X = rays[i]
        x, ok = _project_lens(lens, X, _projection_seed(lens, X, seed_map, seeds, i), iters, tol2)
        u = out_map(x)
        out[i] = SVector{2, Float64}(u[1], u[2])
        ok_out[i] = ok
    end
end
@inline _projection_seed(l, X, seed_map, ::Nothing, i) = _cold_start(l, X)
@inline function _projection_seed(l, X, seed_map, seeds, i)
    s = SVector{2, Float64}(seed_map(@inbounds seeds[i]))
    return all(isfinite, s) ? s : _cold_start(l, X)
end

# The batched projection of the camera-frame points `rays` through the lens `l` to its
# normalized plane (`project`'s solve, `iters` iterations, converged below `tol`),
# seeded at the normalized points `seeds` (a non-finite one, or `nothing`: the cold
# start), on `device`: `(; x, ok)` in the rays' array kind. With `fold`, also the
# backward's Gram determinant `det(JxᵀJx)` at each point (NaN where not imaged): 0 or
# not finite where the lattice folds (no projection Jacobian exists there).
function _spline_projections(
        l::SplineLens, rays::AbstractVector{<:StaticVector{3}}, seeds::Union{Nothing, AbstractVector{<:StaticVector{2}}};
        iters::Int = _PROJECT_MAXITER, tol::Real = _PROJECT_TOL, device::KA.Backend = default_device(), fold::Bool = false
    )
    seeds === nothing || length(seeds) == length(rays) || throw(ArgumentError("project: length(seeds) != length(rays)"))
    N = length(rays)
    x = KA.allocate(device, SVector{2, Float64}, N)
    ok = KA.allocate(device, Bool, N)
    g = fold ? KA.allocate(device, Float64, N) : nothing
    if N > 0
        dl = Adapt.adapt(device, l)
        ds = seeds === nothing ? nothing : on_device(device, _concrete_points(seeds))
        _spline_project_kernel!(device, 64)(x, ok, dl, identity, identity, on_device(device, _concrete_points(rays)), ds, iters, Float64(tol)^2; ndrange = N)
        fold && _spline_fold_kernel!(device, 64)(g, dl.backward.controls, dl.backward.area, x, ok; ndrange = N)
        KA.synchronize(device)
    end
    back = get_backend(rays)
    return (; x = on_device(back, x), ok = on_device(back, ok), fold = fold ? on_device(back, g) : nothing)
end
@kernel function _spline_fold_kernel!(g, controls, area, @Const(xs), @Const(oks))
    i = @index(Global)
    @inbounds if oks[i]
        _, _, Jx, _ = _spline_direction_differential(controls, area, xs[i])
        g[i] = LinearAlgebra.det(Jx' * Jx)
    else
        g[i] = NaN
    end
end

"""
    project(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, rays::AbstractVector{<:StaticVector{3}},
            p₀s::AbstractVector; iters = 10, tol = 1.0e-6, device = default_device()) -> (; px, ok)

Batched projection on the KernelAbstractions backend `device`
([`default_device`](@ref)`()` by default: an accelerator when one is functional,
else `KA.CPU()`): the scalar [`project`](@ref)'s solve per ray, warm-started at
the pixel `p₀s[i]` (a non-finite seed: the scalar `project`'s own start) and given
at most `iters` iterations (a lens with a forward spline reads no seed). The rays and seeds are uploaded once (a device copy of the
lens too), and both outputs come back in the rays' array kind (an `Array` gives
`Array`s, a device array stays on its device with no copy). Returns the pixels and
the mask of the imaged ones (converged below `tol` onto a present window); a pixel
not imaged is NaN.
"""
function project(
        cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, rays::AbstractVector{<:StaticVector{3}}, p₀s::AbstractVector;
        iters::Int = 10, tol::Real = _PROJECT_TOL, device::KA.Backend = default_device()
    )
    length(p₀s) == length(rays) || throw(ArgumentError("project: length(p₀s) != length(rays)"))
    N = length(rays)
    rs, ss = _concrete_points(rays), _concrete_points(p₀s)
    px = KA.allocate(device, SVector{2, Float64}, N)
    ok = KA.allocate(device, Bool, N)
    if N > 0
        intr = cam.intrinsics
        _spline_project_kernel!(device, 64)(
            px, ok, Adapt.adapt(device, cam.lens), _pixel_affine(intr), p -> _normalized_point(intr, p), on_device(device, rs),
            on_device(device, ss), iters, Float64(tol)^2; ndrange = N
        )
        KA.synchronize(device)
    end
    return (; px = on_device(get_backend(rays), px), ok = on_device(get_backend(rays), ok))
end

# A batch of points as the kernels take it: of a concrete element type. A host vector
# of an abstract one (`Vector{StaticVector{2}}`, which the batched signatures admit) is
# converted to `SVector`s of its elements' promoted number type; any other batch is
# taken as it is.
_concrete_points(xs::AbstractVector) = xs
function _concrete_points(xs::AbstractVector{<:StaticVector{N}}) where {N}
    isconcretetype(eltype(xs)) && return xs
    T = isempty(xs) ? Float64 : mapreduce(eltype, promote_type, xs)
    return SVector{N, T}.(xs)
end

@kernel function _spline_backproject_kernel!(rays, lens, intrinsics, s, @Const(pixels))
    i = @index(Global)
    @inbounds rays[i] = _backproject_lens(lens, _normalized_point(intrinsics, pixels[i]), s)
end

"""
    backproject(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, pixels::AbstractVector{<:StaticVector{2}};
                device = default_device()) -> rays

Batched backprojection on the KernelAbstractions backend `device`
([`default_device`](@ref)`()` by default): the scalar [`backproject`](@ref) of the
lens at `K⁻¹u` per pixel, one kernel (the pixels uploaded once, the lens's device
copy too, the lattice slopes formed once), the rays back in the pixels' array kind
— bit for bit the scalar value on the CPU.
"""
function backproject(
        cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, pixels::AbstractVector{<:StaticVector{2, <:Real}};
        device::KA.Backend = default_device()
    )
    l = Adapt.adapt(device, cam.lens)
    intr = cam.intrinsics
    pts = _concrete_points(pixels)
    Tx = eltype(_normalized_point(intr, zero(eltype(pts))))
    rays = KA.allocate(device, SVector{3, promote_type(Tx, eltype(eltype(l.backward.controls)))}, length(pixels))
    if !isempty(pixels)
        s = _lattice_scales(Tx, l.backward.area, size(l.backward.controls))
        _spline_backproject_kernel!(device, 64)(rays, l, intr, s, on_device(device, pts); ndrange = length(pixels))
        KA.synchronize(device)
    end
    return on_device(get_backend(pixels), rays)
end

# ------------------------------------------------------------------------------
# The gauge
# ------------------------------------------------------------------------------

# THE gauge of a SplineLens camera: `K` from the camera's pixel-lattice direction
# spline `pixel_direction` (a `ChartSpline{3}` over the calibrated pixel rectangle
# `rect`). The principal point is the pixel imaging ẑ, found by THE central
# projection solve (`_spline_project`) from the rectangle's centre; `fx = 1/J[1,1]`,
# `fy = 1/J[2,2]` with `J = ∂(d_xy/d_z)/∂u` there; skew 0. When ẑ is not imaged
# inside `rect` (the solve does not converge there), the principal point is the
# rectangle's centre and the focal lengths the local scale there.
#
# The local scale needs a fully present 4 × 4 window at that pixel (`present`, the
# lattice's mask, `nothing` for every node): a lattice without one there has no scale to derive K from, an
# `ArgumentError` naming the pixel and saying to supply K. When ẑ's pixel is not found
# and the centre's own window is incomplete, the search for it could not start there:
# ẑ may be imaged under an incomplete window (a hole over ẑ) or not at all, so the
# refusal says which pixel lacks a window, not that ẑ is not imaged. Any skew-0 K is an
# exact gauge of the camera (a pixel's lattice coordinate does not depend on it).
function _spline_gauge(pixel_direction::ChartSpline{3}, rect::Rect2, present)
    pp, on_axis = _imaged_z(pixel_direction.controls, rect, present)    # the pixel imaging ẑ, else the centre (the fallback)
    ix, iy, _, _ = spline_window(rect, size(pixel_direction.controls), pp)
    _window_present(present, ix, iy) || throw(
        ArgumentError(
            "SplineLens gauge: the grid has no complete 4 × 4 window at " *
                (on_axis ? "the pixel $pp imaging ẑ" : "the calibrated area's centre $pp, where the search for the pixel imaging ẑ starts (ẑ may lie under an incomplete window)") *
                ", so K cannot be derived from it; supply K (intrinsics: a stored record's nominal pinhole, or a camera source)"
        )
    )
    _, J = _value_and_jacobian(u -> (d = pixel_direction(u); SVector(d[1] / d[3], d[2] / d[3])), pp)
    fx, fy = 1 / J[1, 1], 1 / J[2, 2]
    (isfinite(fx) && isfinite(fy) && fx > 0 && fy > 0) || throw(
        ArgumentError("SplineLens gauge: the grid's local scale at the pixel $pp is not a positive finite focal length ($fx, $fy)")
    )
    return LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(fx, 0.0, 0.0, 0.0, fy, 0.0, pp[1], pp[2], 1.0)))
end

# The K a SplineLens camera built from `source` keeps: a camera's own (checked by
# the one constructor, `CameraModel(SplineLens, controls, size)`); `nothing` for any
# other source, which takes the gauge of its sampled lattice.
_source_intrinsics(source::CameraModel) = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(calibration_matrix(source))))
_source_intrinsics(_) = nothing

# ------------------------------------------------------------------------------
# The camera: from a source, its lattice, its valid domain
# ------------------------------------------------------------------------------

# The camera of intrinsics `K` whose lens lays `controls` (`[gy, gx]`) over the
# pixel rectangle `rect`: the backward area is K⁻¹ of the rectangle, computed by
# the map `backproject` applies (`_K_inv`), so a pixel on the rectangle's corner
# lands on the area's corner bit for bit.
function _spline_camera(K::LogicalIntrinsics, rect::Rect2, controls::AbstractMatrix; present = nothing)
    area = _mapped_rect(_K_inv(K), rect)
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
    _, gsz = _spline_layout(size, grid_size, calib_area)
    return CameraModel(SplineLens, fill(SVector(0.0, 0.0, 1.0), gsz), size; calib_area, intrinsics = _source_intrinsics(source))
end

"""
    CameraModel(SplineLens, source, size; grid_size, calib_area, init = :interpolate, sampling_margin = 0.97)
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
nodes, sampling a camera inside `sampling_margin` times its lens's valid
domain. `init = :quasi_interpolate` is
the 4th-order local quasi-interpolant on a lattice extended one node past the
area ([`quasi_interpolant`](@ref)), with its border rule for a source whose
domain ends near or inside the area; a function source takes it. The prefilter's
controls are re-normalized (each unit unless it is unit to 1e-12), the
quasi-interpolant's kept as it forms them.

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
    )
    rect, gsz = _spline_layout(size, grid_size, calib_area)
    C = _init_directions(source, rect, gsz, init, sampling_margin)
    return CameraModel(SplineLens, C, size; calib_area, intrinsics = _source_intrinsics(source), unit_controls = init !== :quasi_interpolate)
end

"""
    CameraModel(SplineLens, controls::AbstractMatrix{<:StaticVector{3}}, size; calib_area,
                intrinsics = nothing, present = nothing, unit_controls = false)
        -> CameraModel{<:LogicalIntrinsics, <:SplineLens}

THE constructor of a [`SplineLens`](@ref) camera (the source constructor and
[`spline_lattice`](@ref) build through it): the camera of a pixel lattice given
by its direction controls, as the reference implementation (Schöps et al.'s
`intrinsics0.yaml`) and the stored grid records state one: `controls` (`[gy, gx]`;
an absent node's control is never read, not even by the gauge) laid over the
calibrated pixel area `calib_area` (inclusive integer bounds
`(min_x, min_y, max_x, max_y)` or a `Rect`, the closed rectangle `[min, max + 1]`;
default the whole image) of an image of `size`, the nodes `present` held
(`nothing`: every node). Its `K` is `intrinsics` (skew 0, positive focal
lengths, else an `ArgumentError`) or, without one, the lattice's gauge. Its
backward area is K⁻¹ of the calibrated rectangle, so each pixel's lattice
coordinate is the reference's and the camera backprojects as the pixel grid
does, whatever the K. The controls are kept as given (`unit_controls = true`
normalizes each one unless it is unit to 1e-12, the reference reader's rule).
"""
function CameraModel(
        ::Type{SplineLens}, controls::AbstractMatrix{<:StaticVector{3}}, size::Union{Tuple{Integer, Integer}, Size2};
        calib_area::Union{NTuple{4, Integer}, GeometryBasics.HyperRectangle{2}} =
            (0, 0, _spline_size(size)[1] - 1, _spline_size(size)[2] - 1),
        intrinsics::Union{Nothing, LogicalIntrinsics} = nothing,
        present::Union{Nothing, AbstractMatrix{Bool}} = nothing,
        unit_controls::Bool = false,
    )
    gh, gw = Base.size(controls)
    rect, _ = _spline_layout(size, (gw, gh), calib_area)
    C = Matrix{SVector{3, Float64}}(map(v -> SVector{3, Float64}(unit_controls ? _unit_or_normalize(v) : v), controls))
    present === nothing || (C = _absent_nan(C, present))   # NaN in the absent slots: a debugging aid, read by nothing
    intrinsics === nothing || _check_spline_gauge(intrinsics)   # before the lattice is laid under it
    K = intrinsics === nothing ? _spline_gauge(ChartSpline(rect, C), rect, present) : intrinsics
    return _spline_camera(K, rect, C; present)
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
    with_directions(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, D::AbstractMatrix{<:StaticVector{3}}) -> CameraModel

The [`SplineLens`](@ref) camera with the backward direction controls `D` (`[gy, gx]`,
kept exactly as given, in the controls' element type): the same `K`, lattice and
presence mask. The forward spline, which inverts the old directions, is dropped
(`project` seeds without it).
"""
with_directions(cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, D::AbstractMatrix) =
    CameraModel(cam.intrinsics, SplineLens(_with_controls(cam.lens.backward, D, "with_directions"); present = cam.lens.present))

# A spline's controls replaced: `SVector{3}`s over its area, in its element type, the lattice's size checked.
function _with_controls(s::ChartSpline, controls::AbstractMatrix, what)
    size(controls) == size(s.controls) ||
        throw(ArgumentError("$(what): a $(size(controls)) control grid for a $(size(s.controls)) model"))
    return ChartSpline(s.area, Matrix(_as_controls(eltype(eltype(s.controls)), controls)))
end
# Given controls as `SVector{3, T}`s, the numbers kept.
_as_controls(::Type{T}, given::AbstractMatrix) where {T} = map(v -> SVector{3, T}(v[1], v[2], v[3]), given)

# ------------------------------------------------------------------------------
# A window: the camera of a block of the lattice's controls
# ------------------------------------------------------------------------------

# The controls of the block `rows × cols` of a lattice's `controls` (`[gy, gx]`):
# a view of the block, or `given` (an equal-size matrix of 3-vectors, kept as given).
function _window_controls(controls::AbstractMatrix, rows, cols, given)
    gh, gw = size(controls)
    (1 <= first(rows) && last(rows) <= gh && 1 <= first(cols) && last(cols) <= gw) ||
        throw(ArgumentError("window: rows $rows × cols $cols outside the $gh × $gw grid"))
    (length(rows) >= 4 && length(cols) >= 4) ||
        throw(ArgumentError("window: a block of at least 4 × 4 controls, got $(length(rows)) × $(length(cols))"))
    given === nothing && return view(controls, rows, cols)
    size(given) == (length(rows), length(cols)) ||
        throw(DimensionMismatch("window: $(size(given)) controls for a $(length(rows)) × $(length(cols)) block"))
    return _as_controls(eltype(eltype(controls)), given)
end

# The rectangle the block `rows × cols` of a lattice over `area` spans: from the
# node of its second column (row) to the node of its second-to-last, `size − 3` of
# the lattice's own cells (the block's lattice is the lattice's, its first node one
# cell outside the rectangle). Its corners are the lattice's own nodes, exactly.
function _window_area(area::Rect2, gsz, rows, cols)
    a = _lattice_node(area, gsz, first(cols) + 1, first(rows) + 1)
    z = _lattice_node(area, gsz, last(cols) - 1, last(rows) - 1)
    return Rect2{Float64}(a, z - a)
end

"""
    window(camera::CameraModel{<:AbstractIntrinsics, <:SplineLens}, rows, cols; directions = nothing) -> CameraModel

The block `rows × cols` (unit ranges into the lens's controls `[gy, gx]`, at
least 4 × 4) of a [`SplineLens`](@ref) camera as a SplineLens camera of its own:
the same `K`, the block of the backward spline's controls (or `directions`, an
equal-size matrix of 3-vectors, kept as given) over the rectangle of the
normalized plane they span, and the block of the presence mask; no forward
spline. It images a point exactly when the point's 4 × 4 window lies inside the
block (and is present), and there it evaluates as the camera holding those
controls, to the rounding of the lattice coordinate. It is what a per-tile cost
block of a refinement evaluates: a few hundred controls instead of the whole
lattice.
"""
function window(
        cam::CameraModel{<:AbstractIntrinsics, <:SplineLens}, rows::AbstractUnitRange{<:Integer}, cols::AbstractUnitRange{<:Integer};
        directions::Union{Nothing, AbstractMatrix} = nothing
    )
    l = cam.lens
    b = l.backward
    C = _window_controls(b.controls, rows, cols, directions)
    present = l.present === nothing ? nothing : l.present[rows, cols]
    return CameraModel(cam.intrinsics, SplineLens(ChartSpline(_window_area(b.area, size(b.controls), rows, cols), C); present))
end

# A rectangle of the normalized plane in pixels: its corners through `K` (skew 0, so
# a rectangle again) — the calibrated pixel area of a lattice over `area`.
_pixel_area(intrinsics::AbstractIntrinsics, area::Rect2) = _mapped_rect(SMatrix{3, 3, Float64}(_K(intrinsics)), area)

# A rectangle's image under the skew-0 affinity `M` (a K or K⁻¹, 3 × 3: `_check_spline_gauge`'s
# precondition): its two corners mapped, the rectangle between them.
function _mapped_rect(M::AbstractMatrix, r::Rect2)
    o = GeometryBasics.origin(r); w = GeometryBasics.widths(r)
    lo = M * SVector(o[1], o[2], 1.0)
    hi = M * SVector(o[1] + w[1], o[2] + w[2], 1.0)
    return Rect2{Float64}(lo[1], lo[2], hi[1] - lo[1], hi[2] - lo[2])
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
    area = _pixel_area(cam.intrinsics, b.area)
    l.present === nothing && return area
    gsz = gh, gw = size(b.controls)
    full = [(ix, iy) for iy in 3:(gh - 1), ix in 3:(gw - 1) if _window_present(l.present, ix, iy)]
    isempty(full) && throw(ArgumentError("valid_domain: no 4 × 4 window of the spline lens holds present nodes only"))
    jx = extrema(first, full); jy = extrema(last, full)
    # window (ix, iy) reads columns ix − 2 … ix + 1: the block of every fully present window's controls
    return _window_area(area, gsz, (jy[1] - 2):(jy[2] + 1), (jx[1] - 2):(jx[2] + 1))
end

# ------------------------------------------------------------------------------
# The controls as a manifold: the chart a direction control moves in
# ------------------------------------------------------------------------------

"""
    direction_tangents(d) -> (t1, t2)

The reference's orthonormal tangent basis of the unit direction `d` — the
2-dof chart a grid control moves in (`normalize(d + δ₁t1 + δ₂t2)`), and the
plane a direction residual is measured in. Generic in the element type: the
basis is in `d`'s type (a `Float32` direction gives `Float32` tangents, a
`ForwardDiff.Dual` direction dual tangents), so a device kernel or a
differentiated residual is never promoted.
"""
@inline function direction_tangents(d)
    v = SVector{3}(d)
    t1 = normalize(cross(v, _tangent_axis(v)))
    t2 = cross(v, t1)
    return t1, t2
end
# The axis `direction_tangents` crosses a direction with: ŷ near ±x̂, x̂ otherwise
# (piecewise constant: its derivative is zero, as the non-central residual's Jacobian reads it).
@inline _tangent_axis(v) = (o = one(eltype(v)); z = zero(eltype(v)); abs(v[1]) > 9 * o / 10 ? SVector(z, o, z) : SVector(o, z, z))

"""
    ControlPointChart

How a [`ControlPatchManifold`](@ref) moves its point controls:
[`AdditivePoints`](@ref) (3 coordinates per point) or
[`RayNormalPoints`](@ref) (2, across the point's own ray).
"""
abstract type ControlPointChart end

"""
    AdditivePoints() <: ControlPointChart

A point control moves additively, `p + (Δx, Δy, Δz)`: 3 coordinates, the
identity as its block of `retract_jacobian`. The default chart of a
[`ControlPatchManifold`](@ref).
"""
struct AdditivePoints <: ControlPointChart end

"""
    RayNormalPoints() <: ControlPointChart

A point control moves across its own ray, `p + a·t₁ + b·t₂` with
`(t₁, t₂) = direction_tangents(normalize(d))` of the same node's direction
control `d` at the anchor: 2 coordinates, `(t₁ t₂)` as its block of
`retract_jacobian`, and `local_coordinates` the offset's dot products with
`t₁` and `t₂`. A move along the ray leaves the node's line where it is, so this
chart drops the one coordinate of [`AdditivePoints`](@ref) that does not move
the line: `local_coordinates` of a point moved along its ray is the move's
projection onto `(t₁, t₂)`, the component along the ray dropped. The tangents
are off the z = 0 plane where the ray is off axis, so a pupil-plane model's
points leave their plane under this chart; hold them on it with
`AdditivePoints` and a `restrict` mask on `Δz`.
"""
struct RayNormalPoints <: ControlPointChart end

"""
    ControlPatchManifold(directions[, points[, point_chart = AdditivePoints()]]) <: Manifold

A patch of a spline lattice's controls as a manifold (a [`SplineLens`](@ref)'s,
a [`NoncentralSplineModel`](@ref)'s): a `PX × PY` block of its direction controls (an `SMatrix` of 3-vectors, in the caller's layout:
the coordinates follow its column-major order) and, for a non-central model, the
matching block of its point controls —
the block a per-tile refinement moves (an NLLSsolver variable through the
manifold adapter). A direction control `d` moves on its own sphere by the
reference's chart,

    ‖d‖ · normalize(d + δ₁t₁ + δ₂t₂),   (t₁, t₂) = direction_tangents(d / ‖d‖),

2 coordinates that keep its norm (the norm is a weight of the spline's blend,
not a gauge) and whose derivative at δ = 0 is `(t₁, t₂)`. A point control moves
by `point_chart` (a [`ControlPointChart`](@ref)): additively, 3 coordinates
([`AdditivePoints`](@ref)), or across its own ray, 2 ([`RayNormalPoints`](@ref)).
The coordinates are per control, column-major over the block: `(δ₁, δ₂)` each,
`(δ₁, δ₂, Δx, Δy, Δz)` with additive points or `(δ₁, δ₂, a, b)` with ray-normal
points — [`restrict`](@ref) them by mask to hold some fixed (a pupil-plane
model's `Δz`, with additive points). `model(ch)` is `(; directions, points)` (`points` is `nothing`
for a central patch, whose `point_chart` is unused); `retract_jacobian(ch)` is
block diagonal, `(t₁ t₂)` per direction and the point chart's block per point.
"""
struct ControlPatchManifold{PX, PY, L, T <: Real, P <: Union{Nothing, SMatrix{PX, PY, SVector{3, T}, L}}, C <: ControlPointChart} <: Manifold
    directions::SMatrix{PX, PY, SVector{3, T}, L}
    points::P
    point_chart::C
end
ControlPatchManifold(directions::SMatrix{PX, PY, <:StaticVector{3}}) where {PX, PY} =
    (T = float(eltype(eltype(directions))); ControlPatchManifold(SMatrix{PX, PY, SVector{3, T}}(directions), nothing))
ControlPatchManifold(directions::SMatrix{PX, PY, SVector{3, T}, L}, points::Union{Nothing, SMatrix{PX, PY, SVector{3, T}, L}}) where {PX, PY, L, T <: Real} =
    ControlPatchManifold(directions, points, AdditivePoints())

_point_coordinates(::AdditivePoints) = 3
_point_coordinates(::RayNormalPoints) = 2
_control_coordinates(::ControlPatchManifold{PX, PY, L, T, Nothing}) where {PX, PY, L, T <: Real} = 2
_control_coordinates(ch::ControlPatchManifold) = 2 + _point_coordinates(ch.point_chart)
tangent_size(ch::ControlPatchManifold{PX, PY, L}) where {PX, PY, L} = _control_coordinates(ch) * L
model(ch::ControlPatchManifold) = (; directions = ch.directions, points = ch.points)

# One direction control stepped on its sphere (the reference's chart), its norm
# kept; (t1, t2) = direction_tangents(normalize(d)).
@inline _retracted_direction(d::SVector{3}, t1, t2, δ₁, δ₂) = norm(d) * normalize(d + δ₁ * t1 + δ₂ * t2)
# One point control stepped by its chart from the coordinates v[j+1], v[j+2], …;
# (t1, t2) are the same node's direction tangents.
@inline _retracted_point(::AdditivePoints, p::SVector{3}, t1, t2, v, j) = p + SVector(v[j + 1], v[j + 2], v[j + 3])
@inline _retracted_point(::RayNormalPoints, p::SVector{3}, t1, t2, v, j) = p + v[j + 1] * t1 + v[j + 2] * t2
function _retracted_controls(ch::ControlPatchManifold{PX, PY, L, T, Nothing}, v) where {PX, PY, L, T <: Real}
    dirs = SMatrix{PX, PY}(ntuple(Val(L)) do n
        d = ch.directions[n]
        return _retracted_direction(d, direction_tangents(normalize(d))..., v[2n - 1], v[2n])
    end)
    return dirs, nothing
end
function _retracted_controls(ch::ControlPatchManifold{PX, PY, L}, v) where {PX, PY, L}
    k = _control_coordinates(ch)
    nodes = ntuple(Val(L)) do n                      # each node's tangents once, for its direction and its point
        d = ch.directions[n]; j = k * (n - 1)
        t1, t2 = direction_tangents(normalize(d))
        return _retracted_direction(d, t1, t2, v[j + 1], v[j + 2]), _retracted_point(ch.point_chart, ch.points[n], t1, t2, v, j + 2)
    end
    return SMatrix{PX, PY}(map(first, nodes)), SMatrix{PX, PY}(map(last, nodes))
end
retract(ch::ControlPatchManifold, v) = (s = _retracted_controls(ch, v); (; directions = s[1], points = s[2]))
recenter(ch::ControlPatchManifold, v) = (s = _retracted_controls(ch, v); ControlPatchManifold(s[1], s[2], ch.point_chart))
# `retract_jacobian` is a static matrix (no heap): block diagonal over the nodes, rows
# 3 (direction) [+ 3 (point)] per node, columns k = 2 [+ 2 or 3] per node. A node's block:
# (t₁ t₂) in the direction rows' first two columns, the point chart's block below it.
@inline function _point_jacobian!(J, ::AdditivePoints, r, c, t1, t2)
    for i in 1:3
        J[r + i, c + i] = one(eltype(J))
    end
    return J
end
@inline function _point_jacobian!(J, ::RayNormalPoints, r, c, t1, t2)
    for i in 1:3
        J[r + i, c + 1] = t1[i]
        J[r + i, c + 2] = t2[i]
    end
    return J
end
@inline function retract_jacobian(ch::ControlPatchManifold{PX, PY, L, T}) where {PX, PY, L, T}
    k = _control_coordinates(ch)
    central = ch.points === nothing
    J = zero(MMatrix{3 * L * (central ? 1 : 2), k * L, T})
    for n in 1:L
        t1, t2 = direction_tangents(normalize(ch.directions[n]))
        r = 3 * (central ? n - 1 : 2n - 2)
        for i in 1:3
            J[r + i, k * (n - 1) + 1] = t1[i]
            J[r + i, k * (n - 1) + 2] = t2[i]
        end
        central || _point_jacobian!(J, ch.point_chart, r + 3, k * (n - 1) + 2, t1, t2)
    end
    return SMatrix(J)
end
# A point chart's coordinates of the offset Δ, written to out[j+1], out[j+2], ….
_point_local_coordinates!(out, ::AdditivePoints, j, Δ, t1, t2) = (out[(j + 1):(j + 3)] = Δ; out)
_point_local_coordinates!(out, ::RayNormalPoints, j, Δ, t1, t2) = (out[j + 1] = dot(Δ, t1); out[j + 2] = dot(Δ, t2); out)
# The coordinates of a block of controls: per direction, the step's tangent
# coordinates (its sphere's chart inverted — the norm is the anchor's), per point
# its chart's coordinates of the offset.
function local_coordinates(ch::ControlPatchManifold{PX, PY, L}, x::NamedTuple) where {PX, PY, L}
    k = _control_coordinates(ch)
    out = zeros(promote_type(eltype(eltype(ch.directions)), eltype(eltype(x.directions))), k * L)
    for n in 1:L
        d = ch.directions[n]; d̂ = normalize(d)
        t1, t2 = direction_tangents(d̂)
        y = normalize(x.directions[n])
        s = norm(d) / dot(y, d̂)                     # s·y = d + δ₁t₁ + δ₂t₂, and t ⟂ d̂
        out[k * (n - 1) + 1] = s * dot(y, t1)
        out[k * (n - 1) + 2] = s * dot(y, t2)
        ch.points === nothing || _point_local_coordinates!(out, ch.point_chart, k * (n - 1) + 2, x.points[n] - ch.points[n], t1, t2)
    end
    return out
end

# ==============================================================================
# ChartSpline — a uniform cubic B-spline over a rectangle of a 2-D chart
# ==============================================================================
#
# The lattice of the spline models (Schöps et al., CVPR 2020) as a primitive: a
# tensor-product uniform cubic B-spline of `controls` (`[row = second
# coordinate, column = first]`, `size − 3` cells per axis) laid over a closed
# rectangle, continued past the rectangle by the outermost cell's cubic pieces.
# This file holds the ONE spelling of its pieces — the span weights, the
# rectangle → lattice map, the 4 × 4 window — which a `SplineLens`
# (spline_lens.jl), a `NoncentralSplineModel` and every other lattice of the
# stack read; a `ChartSpline` is that spline as a `Transformation`, so it
# composes in `image_warp` chains (a SplineLens's forward map on the
# equidistant direction chart).

# ------------------------------------------------------------------------------
# The span weights
# ------------------------------------------------------------------------------

# The reference (util.h, EvalUniformCubicBSpline) evaluates its cubic pieces in
# its window coordinate x ∈ [3, 4) itself, which cancels to ~1e-5 absolute in
# Float32 (|x³| ≈ 64 against weights ≤ 2/3); the weights below take the local
# coordinate instead.
"""
    cubic_bspline_weights(u) -> NTuple{4}

The four weights of a uniform cubic B-spline span at its local coordinate
`u ∈ [0, 1)`: the value on the span is `w₁c₁ + w₂c₂ + w₃c₃ + w₄c₄` over its
four consecutive controls,

    ((1 − u)³, 3u³ − 6u² + 4, −3u³ + 3u² + 3u + 1, u³) / 6,

a partition of unity. Outside `[0, 1)` the same cubics continue (extrapolate).
Kernel-lawful: in `u`'s own type (a `Float32` coordinate gives `Float32`
weights, a `ForwardDiff.Dual` dual ones), no allocation and no branch, so a
device kernel can call it. Evaluated in Horner form in `u`, which keeps
`Float32` within 1e-7 of the exact weights. The weights a spline lattice's
windows ([`spline_window`](@ref)) and [`ChartSpline`](@ref) read.
"""
@inline function cubic_bspline_weights(u)
    v = 1 - u
    a = v * v * v / 6                           # (1 − u)³ / 6
    b = (u * u * (3u - 6) + 4) / 6              # (3u³ − 6u² + 4) / 6
    c = (((-3u + 3) * u + 3) * u + 1) / 6       # (−3u³ + 3u² + 3u + 1) / 6
    d = u * u * u / 6                           # u³ / 6
    return (a, b, c, d)
end

# The span's local coordinate u at window coordinate x (= lattice coordinate + 2)
# in the window `ix`: on the window's own cell, x ∈ [ix, ix + 1), it is exact
# (Sterbenz); outside it (a continued window) the pieces extrapolate as the same
# cubics and u may round.
@inline _span_coordinate(x, ix) = x - (ix - 3) - 3

# d/du of the four span weights — the same cubic pieces differentiated, in the
# same local coordinate u and Horner form as `cubic_bspline_weights`.
@inline function _cubic_bspline_weights_deriv(u)
    v = 1 - u
    a = -v * v / 2                  # −(1 − u)² / 2
    b = u * (3u - 4) / 2            # (3u² − 4u) / 2
    c = ((-3u + 2) * u + 1) / 2     # (−3u² + 2u + 1) / 2
    d = u * u / 2                   # u² / 2
    return (a, b, c, d)
end

# ------------------------------------------------------------------------------
# The rectangle → lattice map, and the window of a point
# ------------------------------------------------------------------------------

# A coordinate `x` → its lattice coordinate along one axis of `gdim` controls laid
# over `[lo, lo + span]`: the affine of the C++ PixelCornerConvToGridPoint,
# `1 + s·(x − lo)` with `s = (gdim − 3)/span` its one-number derivative. The ONE
# spelling for the host maps, the device kernels and every Jacobian (a form
# `(g − 3)·(x − lo)/span` rounds differently, which once made two evaluations of
# the same point disagree in the last ulp). In the coordinate's own float type (a
# Float32 device evaluation stays Float32; an integer reads as a float).
@inline _lattice_coord(x, lo, span, gdim) = (T = float(typeof(x)); _lattice_coord(x, lo, _lattice_scale(T, span, gdim)))
# The map at its slope `s = (gdim − 3)/span`, computed in `T` by `_lattice_scale`: the
# same operations, so a batch that forms `s` once per launch has every point's
# coordinate bit for bit.
@inline _lattice_coord(x, lo, s) = (T = float(typeof(x)); 1 + s * (x - T(lo)))
@inline _lattice_scale(::Type{T}, span, gdim) where {T} = T(gdim - 3) / T(span)

# The lattice coordinates `(gx, gy)` of the point `p` on a lattice of `gsz = (rows, cols)`
# controls laid over the closed rectangle `area` (`size − 3` cells per axis): the ONE
# area → lattice map of every spline on a rectangle (`spline_window`, a ChartSpline's
# Jacobian, a SplineLens's backprojection). `_lattice_coords_at` takes the slopes
# `s = _lattice_scales(T, area, gsz)` of the point's float type `T` instead, formed
# once for a batch.
@inline function _lattice_coords(area::GeometryBasics.HyperRectangle{2}, (gh, gw), p)
    o = GeometryBasics.origin(area); w = GeometryBasics.widths(area)
    return _lattice_coord(p[1], o[1], w[1], gw), _lattice_coord(p[2], o[2], w[2], gh)
end
@inline function _lattice_coords_at(area::GeometryBasics.HyperRectangle{2}, s, p)
    o = GeometryBasics.origin(area)
    return _lattice_coord(p[1], o[1], s[1]), _lattice_coord(p[2], o[2], s[2])
end
# The slopes are the map's derivative, `(∂gx/∂p₁, ∂gy/∂p₂) = ((cols − 3)/w₁, (rows − 3)/w₂)`.
@inline function _lattice_scales(::Type{T}, area::GeometryBasics.HyperRectangle{2}, (gh, gw)) where {T}
    w = GeometryBasics.widths(area)
    return _lattice_scale(T, w[1], gw), _lattice_scale(T, w[2], gh)
end

# The window a point reads when the spline is continued past its rectangle: the
# window index is clamped to the grid and the local coordinate runs on, so outside
# the rectangle the outermost cell's cubic pieces continue; inside it the window is
# the point's own cell (EvalUniformCubicBSplineSurface's "+2 so the SURROUNDING
# control points are used" convention). THE window of every evaluation. A NaN point
# gives NaN weights, never an error (`unsafe_trunc`: device-safe).
@inline function _continued_window(gx, gy, gw::Integer, gh::Integer)
    x = gx + 2
    y = gy + 2
    ix = clamp(unsafe_trunc(Int, floor(_primal(x))), 3, gw - 1)   # a dual coordinate: its primal's cell
    iy = clamp(unsafe_trunc(Int, floor(_primal(y))), 3, gh - 1)
    return ix, iy, cubic_bspline_weights(_span_coordinate(x, ix)), cubic_bspline_weights(_span_coordinate(y, iy))
end

"""
    spline_window(area::Rect2, grid_size, point) -> (ix, iy, wx, wy)

The window `point` reads on the uniform cubic lattice of `grid_size = (rows,
cols)` controls laid over the closed rectangle `area` (cells = `size − 3` per
axis): the controls `[iy - 3 + r, ix - 3 + c]`, `r, c ∈ 1:4`, and the four
weights per axis (`wx[c]` along x, `wy[r]` along y; `w[r, c] = wy[r] * wx[c]`).
Outside the area the outermost cell's cubic pieces continue — the same
continuation [`spline_value`](@ref) evaluates.
"""
function spline_window(area::GeometryBasics.HyperRectangle{2}, grid_size::NTuple{2, Integer}, point)
    gh, gw = grid_size
    return _continued_window(_lattice_coords(area, grid_size, point)..., gw, gh)
end

"""
    spline_value(controls, area::Rect2, point)

The uniform cubic B-spline surface of `controls` (any matrix of values with
`+` and scalar `*`: scalars, 3-vectors) laid over the closed rectangle `area`
(`[row = y, column = x]`, cells = `size − 3` per axis), at `point`, continued
past the area as [`spline_window`](@ref): a [`ChartSpline`](@ref)'s value, for a
lattice of any control type (a per-position aperture radius, a field over the
image).
"""
function spline_value(controls::AbstractMatrix, area::GeometryBasics.HyperRectangle{2}, point)
    ix, iy, wx, wy = spline_window(area, size(controls), point)
    return _eval_window(controls, ix, iy, wx, wy)
end

# ------------------------------------------------------------------------------
# ChartSpline
# ------------------------------------------------------------------------------

"""
    ChartSpline(area::Rect2, controls::AbstractMatrix{<:StaticVector{K}})

A tensor-product uniform cubic B-spline from a 2-D chart to `ℝᴷ`: the
`controls` laid over the closed rectangle `area` of the chart (`controls[row, column]`, the row
along the chart's second coordinate, `size − 3` cells per axis, so the knot
spacing along axis `i` is `widths(area)[i] / (nᵢ − 3)`). Evaluated with
[`spline_value`](@ref), continued past `area` by the outermost cell's cubic
pieces.

A `CoordinateTransformations.Transformation`: `cs(c)` is the value at the chart
point `c`, `transform_deriv(cs, c)` its `K × 2` Jacobian (the differentiated
cubic pieces, no automatic differentiation), and `cs(points)` on a vector of
chart points evaluates the batch with a KernelAbstractions kernel on the points'
own backend (the controls are uploaded beside them).
"""
struct ChartSpline{K, T <: AbstractFloat, M <: AbstractMatrix{SVector{K, T}}} <: CoordinateTransformations.Transformation
    area::Rect2{Float64}
    controls::M
    function ChartSpline{K, T, M}(area::Rect2{Float64}, controls::M) where {K, T <: AbstractFloat, M <: AbstractMatrix{SVector{K, T}}}
        (size(controls, 1) >= 4 && size(controls, 2) >= 4) || throw(
            ArgumentError("ChartSpline: a uniform cubic lattice needs at least 4 × 4 controls, got $(size(controls, 1)) × $(size(controls, 2))")
        )
        w = GeometryBasics.widths(area)
        (w[1] > 0 && w[2] > 0 && all(isfinite, w) && all(isfinite, GeometryBasics.origin(area))) ||
            throw(ArgumentError("ChartSpline: the chart's area must be a finite rectangle of positive widths, got $area"))
        return new{K, T, M}(area, controls)
    end
    # A value already validated (its device form): no check, so no string to build.
    # KernelAbstractions' `@Const` re-adapts a kernel argument inside the kernel,
    # where the checked constructor's error message cannot compile.
    ChartSpline{K, T, M}(area::Rect2{Float64}, controls::M, ::Val{:unchecked}) where {K, T <: AbstractFloat, M <: AbstractMatrix{SVector{K, T}}} =
        new{K, T, M}(area, controls)
end
ChartSpline(area::GeometryBasics.HyperRectangle{2}, controls::AbstractMatrix{SVector{K, T}}) where {K, T <: AbstractFloat} =
    ChartSpline{K, T, typeof(controls)}(Rect2{Float64}(area), controls)

Adapt.adapt_structure(to, cs::ChartSpline{K, T}) where {K, T} =
    (c = Adapt.adapt(to, cs.controls); ChartSpline{K, T, typeof(c)}(cs.area, c, Val(:unchecked)))

(cs::ChartSpline)(c::StaticVector{2, <:Real}) = spline_value(cs.controls, cs.area, c)

# The controls convert as they are; the area stays.
convert_eltype(::Type{T}, cs::ChartSpline{K}) where {T <: AbstractFloat, K} = ChartSpline(cs.area, map(SVector{K, T}, cs.controls))

# ------------------------------------------------------------------------------
# The controls as one coordinate vector
# ------------------------------------------------------------------------------
#
# The ONE layout of a lattice's controls as a vector (a SplineLens's parameter
# vector): the held nodes in `CartesianIndices` order — column-major over
# `[row, column]`, the order of `nodes` (support.jl) — each node's K components in
# turn. `present` is a presence mask (`nothing`: every node held); an absent node
# holds no coordinate and keeps its control on a rebuild.
_held_nodes(::Nothing, sz) = vec(CartesianIndices(sz))
_held_nodes(present::AbstractMatrix{Bool}, sz) = findall(present)
_control_vector(controls::AbstractMatrix{SVector{K, T}}, present) where {K, T} =
    T[controls[i][k] for i in _held_nodes(present, size(controls)) for k in 1:K]
_control_names(controls::AbstractMatrix{<:SVector{K}}, present) where {K} =
    Tuple(Symbol("c_", i[1], "_", i[2], "_", ("x", "y", "z")[k]) for i in _held_nodes(present, size(controls)) for k in 1:K)
function _with_control_vector(controls::AbstractMatrix{SVector{K, T}}, present, θ::AbstractVector) where {K, T}
    held = _held_nodes(present, size(controls))
    length(θ) == K * length(held) ||
        throw(ArgumentError("a $(size(controls, 1)) × $(size(controls, 2)) lattice of $(length(held)) held nodes takes $(K * length(held)) control coordinates; got $(length(θ))"))
    S = float(promote_type(T, eltype(θ)))
    out = map(SVector{K, S}, controls)
    for (j, i) in enumerate(held)
        out[i] = SVector{K, S}(ntuple(k -> θ[firstindex(θ) + K * (j - 1) + k - 1], Val(K)))
    end
    return out
end

# THE weights of a point's window and their derivatives in the lattice coordinate, from
# ONE evaluation of the cubic pieces: `(gx, ix, iy, wx, wy, dwx, dwy)` at the point `x`
# of the lattice of `gsz = (rows, cols)` over `area`, the window the point's own
# continued one (`window = nothing`) or the given `(ix, iy)`, whose pieces then
# extrapolate off its cell. Read by the value-and-partials sweep and by `spline_weights`.
@inline function _window_weights(gsz, area, x)
    gh, gw = gsz
    gx, gy = _lattice_coords(area, gsz, x)
    ix, iy, wx, wy = _continued_window(gx, gy, gw, gh)
    dwx = _cubic_bspline_weights_deriv(_span_coordinate(gx + 2, ix))
    dwy = _cubic_bspline_weights_deriv(_span_coordinate(gy + 2, iy))
    return gx, ix, iy, wx, wy, dwx, dwy
end

# THE value of a spline of `controls` over `area` at the point `x` and its two partials
# `(S, ∂S/∂x₁, ∂S/∂x₂)`, from ONE evaluation of the basis weights and one sweep of the
# 4 × 4 window (`_continued_window`'s: continued past the area, the window clamped to
# the grid), the partials through the area → lattice slope. In the precision of `x`
# and the controls (a Float32 lattice stays Float32). The value sums in
# `_eval_window`'s order, so it is the spline's value bit for bit. Device-safe.
@inline function _spline_value_and_partials(controls::AbstractMatrix{<:StaticVector}, area, x::StaticVector{2})
    gsz = size(controls)
    gx, ix, iy, wx, wy, dwx, dwy = _window_weights(gsz, area, x)
    return _window_sums(controls, area, gx, ix, iy, wx, wy, dwx, dwy)
end
# The value and partials of the window `(ix, iy)` from its weights (`_window_weights`).
@inline function _window_sums(controls::AbstractMatrix{<:StaticVector{K}}, area, gx, ix, iy, wx, wy, dwx, dwy) where {K}
    T = promote_type(typeof(gx), eltype(eltype(controls)))
    S = Sx = Sy = zero(SVector{K, T})
    @inbounds for r in 1:4
        row = drow = zero(SVector{K, T})
        for c in 1:4
            g = SVector{K, T}(controls[iy - 3 + r, ix - 3 + c])
            row += wx[c] * g
            drow += dwx[c] * g
        end
        S += wy[r] * row
        Sx += wy[r] * drow
        Sy += dwy[r] * row
    end
    sx, sy = _lattice_scales(Float64, area, size(controls))
    return S, Sx * T(sx), Sy * T(sy)
end

"""
    spline_weights(cs::ChartSpline, x) -> (ix, iy, w, (∂w₁, ∂w₂))

The weights of the 16 controls a point `x` of the chart reads, and their partials:
`w[r, c]` multiplies `cs.controls[iy - 3 + r, ix - 3 + c]` (`r` the second
coordinate's tap, `c` the first's; `sum(w .* window controls)` is `cs(x)`), and
`∂wᵢ = ∂w/∂xᵢ`, all `SMatrix{4, 4}`. The window is the point's own (continued past
the area, as [`spline_window`](@ref)). In the point's precision. For a
[`SplineLens`](@ref) camera's `backward` spline, `x` is the normalized point `K⁻¹u`:
the pixel partials are `∂w₁ / fx`, `∂w₂ / fy` (skew 0).
"""
function spline_weights(cs::ChartSpline, x::StaticVector{2, <:Real})
    gsz = size(cs.controls)
    gx, ix, iy, wx, wy, dwx, dwy = _window_weights(gsz, cs.area, float.(x))
    sx, sy = _lattice_scales(Float64, cs.area, gsz)
    T = typeof(gx)
    return ix, iy, _window_tensor(wx, wy), (_window_tensor(dwx .* T(sx), wy), _window_tensor(wx, dwy .* T(sy)))
end
# The 4 × 4 products `wy[r]·wx[c]` of a window's per-axis weights.
@inline _window_tensor(wx, wy) = SMatrix{4, 4, typeof(wx[1] * wy[1]), 16}(ntuple(k -> wy[(k - 1) % 4 + 1] * wx[(k - 1) ÷ 4 + 1], Val(16)))

function transform_deriv(cs::ChartSpline, c::StaticVector{2, <:Real})
    _, Sx, Sy = _spline_value_and_partials(cs.controls, cs.area, c)
    return hcat(Sx, Sy)
end

@kernel function _chart_spline_kernel!(out, @Const(controls), @Const(points), area)
    i = @index(Global)
    @inbounds out[i] = spline_value(controls, area, points[i])
end

function (cs::ChartSpline{K, T})(points::AbstractVector{<:StaticVector{2, <:Real}}) where {K, T}
    backend = get_backend(points)
    S = SVector{K, promote_type(T, float(eltype(eltype(points))))}
    out = KA.allocate(backend, S, length(points))
    isempty(points) && return out
    _chart_spline_kernel!(backend, 64)(out, adapt(backend, cs.controls), points, cs.area; ndrange = length(points))
    KA.synchronize(backend)
    return out
end

# ------------------------------------------------------------------------------
# The equidistant direction chart, and the fitted chart of a forward spline
# ------------------------------------------------------------------------------

"""
    equidistant_chart(axis::StaticVector{3}) -> Lens(Equidistant()) ∘ LinearMap(R)

The equidistant direction chart about `axis` (any nonzero 3-vector,
normalized here): a ray at angle `θ` from the axis lands at radius `θ`
(radians) in the chart plane, in the direction of its azimuth; singular only at
`−axis`. `R` is the rotation whose rows are `(e₁, e₂, axis)`, with
`e₁ = normalize(ŷ × axis)` (`x̂ × axis` when the axis is within 1e-6 of `±ŷ`)
and `e₂ = axis × e₁` — right-handed, so the chart is the camera's own
equidistant lens when `axis = ẑ`. The chart a [`SplineLens`](@ref)'s forward
spline is laid on (about ẑ), and the one a direction field over a
cap of the sphere (a panorama's vignetting) reads.
"""
function equidistant_chart(axis::StaticVector{3})
    a = normalize(SVector{3, Float64}(axis))
    e1 = cross(SVector(0.0, 1.0, 0.0), a)
    e1 = norm(e1) > 1.0e-6 ? normalize(e1) : normalize(cross(SVector(1.0, 0.0, 0.0), a))
    e2 = cross(a, e1)
    R = SMatrix{3, 3, Float64}(e1[1], e2[1], a[1], e1[2], e2[2], a[2], e1[3], e2[3], a[3])   # rows e1, e2, axis
    return Lens(Equidistant()) ∘ CoordinateTransformations.LinearMap(R)
end

# Whether a chart point lies within one knot cell of the spline's area: the
# continuation past the area is the outermost cell's cubic pieces, which far
# outside can fold back into the region and image a ray far outside the fitted
# cone at a confident wrong pixel. One cell of margin keeps the region's edge
# (its held-out points lie up to half a quadrature stride past the fit points'
# bounding box).
function _in_fitted_chart(cs::ChartSpline, c::StaticVector{2})
    o = GeometryBasics.origin(cs.area)
    w = GeometryBasics.widths(cs.area)
    hx, hy = _lattice_spacing(cs.area, size(cs.controls))
    return o[1] - hx <= c[1] <= o[1] + w[1] + hx && o[2] - hy <= c[2] <= o[2] + w[2] + hy
end

# ==============================================================================
# ChartSpline and SplineProjection — a camera's forward map as a spline on a chart
# ==============================================================================
#
# A grid model (generic_models.jl) maps a pixel to a ray and has no closed-form
# forward map: its `project` is a small LM per ray. A `SplineProjection` is that
# forward map fitted as a spline (VisualGeometryToolkitEstimation's
# `fit_camera(SplineProjection, source, region)`): ray → chart point → pixel, the
# chart the equidistant direction chart about the optical axis, the spline a
# tensor-product uniform cubic B-spline over the chart. It is a
# `CoordinateTransformations.Transformation`, so it composes in `image_warp` chains.
#
# The spline is the grid models' own lattice, as a primitive: `spline_window` /
# `spline_value` over a closed rectangle of the chart (cells = size − 3 per axis,
# controls `[row = second coordinate, column = first]`), continued past the
# rectangle by the outermost cell's cubic pieces. Nothing here re-derives the
# window or the cubic pieces.

"""
    ChartSpline(area::Rect2, controls::AbstractMatrix{<:StaticVector{K}})

A tensor-product uniform cubic B-spline from a 2-D chart to `ℝᴷ`: the
`controls` laid over the closed rectangle `area` of the chart as a generic grid
model lays its grid over its calibrated area (`controls[row, column]`, the row
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

function transform_deriv(cs::ChartSpline, c::StaticVector{2, <:Real})
    o = GeometryBasics.origin(cs.area)
    w = GeometryBasics.widths(cs.area)
    gh, gw = size(cs.controls)
    gx = _lattice_coord(c[1], o[1], w[1], gw)
    gy = _lattice_coord(c[2], o[2], w[2], gh)
    ix, iy, wx, wy = _continued_window(gx, gy, gw, gh)
    dwx = _cubic_bspline_weights_deriv(_span_coordinate(gx + 2, ix))
    dwy = _cubic_bspline_weights_deriv(_span_coordinate(gy + 2, iy))
    Sx = _eval_window(cs.controls, ix, iy, dwx, wy) * ((gw - 3) / w[1])   # ∂/∂c₁: the lattice coordinate's slope
    Sy = _eval_window(cs.controls, ix, iy, wx, dwy) * ((gh - 3) / w[2])
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

"""
    SplineProjection(chart, spline::ChartSpline{2}, region)

A camera's forward map, ray → pixel, as a spline on a direction chart: the
pixel of a ray `X` is `spline(chart(X))`. `chart` is the equidistant direction
chart about the optical axis, `Lens(Equidistant()) ∘ LinearMap(R)` with `R` a
rotation whose third row is the axis (its first two an orthonormal basis of the
plane orthogonal to it): a ray at angle `θ` from the axis lands at radius `θ`
(radians). `region` is the region of the sensor the map was fitted over, in the
pixel frame of the camera it was fitted to (an [`imaged_region`](@ref), an
[`Ellipse`](@ref), a `Rect`, their `∩`). Built by VisualGeometryToolkitEstimation's
`fit_camera(SplineProjection, source, region; tolerance)`.

A `CoordinateTransformations.Transformation`: `sp(X)` is the pixel (the spline
continued past its chart area, never `nothing`), `transform_deriv(sp, X)` its
`2 × 3` Jacobian, so it composes in `image_warp` chains. [`project`](@ref)`(sp, X)`
is the camera semantics: `nothing` when the ray's chart point lies more than
one knot cell outside the spline's chart area (where the continuation would
extrapolate, and can fold back into the region), or when the pixel does not
lie in `region` (or is not finite).
"""
struct SplineProjection{C, S <: ChartSpline{2}, R} <: CoordinateTransformations.Transformation
    chart::C
    spline::S
    region::R
end

Adapt.adapt_structure(to, sp::SplineProjection) =
    (s = Adapt.adapt(to, sp.spline); SplineProjection(sp.chart, s, sp.region))

(sp::SplineProjection)(X::StaticVector{3, <:Real}) = sp.spline(sp.chart(X))

transform_deriv(sp::SplineProjection, X::StaticVector{3, <:Real}) =
    transform_deriv(sp.spline, sp.chart(X)) * transform_deriv(sp.chart, X)

function project(sp::SplineProjection, X::StaticVector{3, <:Real})
    _in_fitted_chart(sp.spline, sp.chart(X)) || return nothing
    u = sp(X)
    (all(isfinite, u) && Point2{Float64}(u[1], u[2]) ∈ sp.region) || return nothing
    return u
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
    gh, gw = size(cs.controls)
    hx, hy = w[1] / (gw - 3), w[2] / (gh - 3)
    return o[1] - hx <= c[1] <= o[1] + w[1] + hx && o[2] - hy <= c[2] <= o[2] + w[2] + hy
end

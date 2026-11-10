# ==============================================================================
# NoncentralSplineModel — the non-central grid camera on ChartSpline
# ==============================================================================
#
# The non-central grid model of Schöps, Larsson, Pollefeys and Sattler (CVPR
# 2020): per pixel an observation LINE, a direction spline and a point spline
# over one lattice. Here both are `ChartSpline{3}`s over the same rectangle of the
# K-normalized plane x = K⁻¹u, as the backward spline of a `SplineLens`: the
# directions are a SplineLens's (`central_model(m)`), and every evaluation of
# them, the box and presence test (`_in_backward`), the LM (`_spline_lm` on the
# area's scale-free `_spline_lm_box`) and the gauge rule (skew 0) are the lens's.
# What is the model's own: the point spline and the tangent-plane residual of
# its projection.

# ------------------------------------------------------------------------------
# Ray3 — the non-central unprojection currency
# ------------------------------------------------------------------------------

"""
    Ray3(origin, direction)

A 3D ray: `origin + t·direction`, `direction` unit by contract. The
unprojection currency of [`NoncentralSplineModel`](@ref) — a non-central
pixel observes a LINE in the camera frame, not a direction through a common
center.
"""
struct Ray3{T <: AbstractFloat}
    origin::Vec3{T}
    direction::Vec3{T}
end

"""
    NoncentralSplineModel(K::LogicalIntrinsics, directions::ChartSpline{3}, points::ChartSpline{3};
                          present = nothing) <: AbstractCamera
    NoncentralSplineModel(source, size; grid_size, calib_area, init, sampling_margin, unit_controls)

The non-central grid camera (Schöps et al., CVPR 2020): the pixel `u` observes
the LINE through `points(x)` along `normalize(directions(x))`, `x = K⁻¹u` the
normalized image point, both splines laid over the same rectangle of the
normalized plane with the same lattice ([`ChartSpline`](@ref); controls
`[row = y, column = x]`). [`backproject`](@ref) returns a [`Ray3`](@ref);
[`project`](@ref) is Levenberg–Marquardt on the reference's tangent-plane
point-to-line residual. With an all-zero point spline the model is exactly
its [`central_model`](@ref), a `SplineLens` camera.

`K` is the gauge of a `SplineLens` camera (skew 0, else an `ArgumentError`).
`present` says which nodes the model holds, as a `SplineLens`'s (`nothing`, or
a `Bool` matrix of the grid's size; an all-true mask is `nothing`).

The `points` are plain metres in the camera frame. This is an exception to the
units layering that the type forces: its controls are interpolated in the inner
loop and cannot hold quantities, so the unit is stated here instead. A reader
or writer of the model converts at its boundary. A [`PosedCamera`](@ref) of the
model therefore takes physical extrinsics (`EuclideanMap(R, t .* m)`, or
`with_unit(E, mm)` of a map in mm) and projects scene points that are lengths; a
plain extrinsics is an `ArgumentError`. `ustrip(m, camera)` is the plain camera,
in metres only.

The model carries no image extent: a stored or rig camera holds it in its
[`BoundedCamera`](@ref).

From a `source` (a camera or a function `pixel -> direction`): the directions
of `CameraModel(SplineLens, source, size; …)` (the same keywords and `K`), the
points all zero.
"""
struct NoncentralSplineModel{I <: LogicalIntrinsics, S <: ChartSpline{3}, P <: Union{Nothing, AbstractMatrix{Bool}}} <: AbstractCamera
    intrinsics::I
    directions::S
    points::S
    present::P
    function NoncentralSplineModel{I, S, P}(intrinsics::I, directions::S, points::S, present::P) where {I <: LogicalIntrinsics, S <: ChartSpline{3}, P <: Union{Nothing, AbstractMatrix{Bool}}}
        K = _K(intrinsics)
        iszero(K[1, 2]) || throw(ArgumentError("NoncentralSplineModel: a spline camera has skew 0; K has skew $(K[1, 2])"))
        size(directions.controls) == size(points.controls) || throw(
            ArgumentError(
                "NoncentralSplineModel: the direction and point splines must have equal grids, " *
                    "got $(size(directions.controls)) vs $(size(points.controls))"
            )
        )
        directions.area == points.area || throw(
            ArgumentError("NoncentralSplineModel: the direction and point splines must share their area, got $(directions.area) vs $(points.area)")
        )
        _check_present(present, size(directions.controls))
        return new{I, S, P}(intrinsics, directions, points, present)
    end
    # A value already validated (its device form): no check, as `SplineLens`'s.
    NoncentralSplineModel{I, S, P}(intrinsics::I, directions::S, points::S, present::P, ::Val{:unchecked}) where {I, S, P} =
        new{I, S, P}(intrinsics, directions, points, present)
end
function NoncentralSplineModel(K::LogicalIntrinsics, directions::ChartSpline{3}, points::ChartSpline{3}; present = nothing)
    if typeof(directions) !== typeof(points)              # one spline type for both: the promoted precision
        T = promote_type(eltype(eltype(directions.controls)), eltype(eltype(points.controls)))
        directions, points = convert_eltype(T, directions), convert_eltype(T, points)   # each array keeps its kind
        typeof(directions) === typeof(points) || throw(
            ArgumentError("NoncentralSplineModel: the direction and point controls are arrays of two kinds, $(typeof(directions.controls)) and $(typeof(points.controls))")
        )
    end
    return _noncentral_spline_model(K, directions, points, _present_mask(present, size(directions.controls)))
end
# The model of its parts' own types: a function barrier over the mask, as `_spline_lens`.
_noncentral_spline_model(K::I, d::S, p::S, present::P) where {I, S, P} = NoncentralSplineModel{I, S, P}(K, d, p, present)

function NoncentralSplineModel(source::_SplineSource, size::Union{Tuple{Integer, Integer}, Size2}; kwargs...)
    c = CameraModel(SplineLens, source, size; kwargs...)
    d = c.lens.backward
    return NoncentralSplineModel(c.intrinsics, d, ChartSpline(d.area, zero(d.controls)))
end

# The device form (`Adapt.adapt(backend, m)`): both splines' controls and the mask
# move, nothing is re-validated.
Adapt.adapt_structure(to, m::NoncentralSplineModel) =
    _device_noncentral(m.intrinsics, Adapt.adapt(to, m.directions), Adapt.adapt(to, m.points), _device_present(to, m.present))
# The model of its device parts' own types (a function barrier, as `_device_spline_lens`), unchecked.
_device_noncentral(K::I, d::S, p::S, present::P) where {I, S, P} = NoncentralSplineModel{I, S, P}(K, d, p, present, Val(:unchecked))

# The controls and K convert as they are (no re-normalization); the area and the mask stay.
convert_eltype(::Type{T}, m::NoncentralSplineModel) where {T <: AbstractFloat} =
    _noncentral_spline_model(convert_eltype(T, m.intrinsics), convert_eltype(T, m.directions), convert_eltype(T, m.points), m.present)

# The one bridge from transforms/warp.jl's private `_precision` (the fused
# image_warp chain and the batched kernels) to the public conversion: a lens
# or a non-central model in a chain converts through `convert_eltype`, so a downstream
# lens's method is the one the kernels apply. Intrinsics and camera models are
# deliberately NOT bridged: a chain holding one has always kept it as it is
# (`_precision`'s identity fallback), and still does.
_precision(::Type{T}, x::Union{AbstractLensModel, NoncentralSplineModel}) where {T <: AbstractFloat} =
    convert_eltype(T, x)

# The direction lens of the model: a SplineLens of its directions and mask (already
# through `_present_mask`, so not re-checked or copied).
_direction_lens(m::NoncentralSplineModel) = _spline_lens(m.directions, nothing, m.present)

"""
    central_model(m::NoncentralSplineModel) -> CameraModel{<:LogicalIntrinsics, <:SplineLens}

The central camera of a non-central one's directions — what a point at
infinity sees: the same `K`, direction spline and present nodes as a
[`SplineLens`](@ref) camera; the line points dropped.
"""
central_model(m::NoncentralSplineModel) = CameraModel(m.intrinsics, _direction_lens(m))

# The controls of a replaced spline: `SVector{3}`s over the model's area, kept as given.
function _model_spline(m::NoncentralSplineModel, controls::AbstractMatrix, what)
    size(controls) == size(m.directions.controls) || throw(
        ArgumentError("$(what): a $(size(controls)) control grid for a $(size(m.directions.controls)) model")
    )
    T = eltype(eltype(m.directions.controls))
    return ChartSpline(m.directions.area, Matrix{SVector{3, T}}(map(v -> SVector{3, T}(v[1], v[2], v[3]), controls)))
end

"""
    with_points(m::NoncentralSplineModel, P::AbstractMatrix{<:StaticVector{3}}) -> NoncentralSplineModel

The model `m` with the line-point controls `P` (`[gy, gx]`, plain metres in the
camera frame, kept exactly as given), every other part kept.
"""
with_points(m::NoncentralSplineModel, P::AbstractMatrix) =
    _noncentral_spline_model(m.intrinsics, m.directions, _model_spline(m, P, "with_points"), m.present)

"""
    with_directions(m::NoncentralSplineModel, D::AbstractMatrix{<:StaticVector{3}}) -> NoncentralSplineModel

The model `m` with the direction controls `D` (`[gy, gx]`, kept exactly as
given), every other part kept.
"""
with_directions(m::NoncentralSplineModel, D::AbstractMatrix) =
    _noncentral_spline_model(m.intrinsics, _model_spline(m, D, "with_directions"), m.points, m.present)

"""
    with_present(m::NoncentralSplineModel, present::AbstractMatrix{Bool}) -> NoncentralSplineModel

The model `m` holding the nodes `present` (`[gy, gx]`, `true` = present): the
same `K` and lattice, the controls of the present nodes kept and NaN in the
absent ones, in both splines.
"""
with_present(m::NoncentralSplineModel, present::AbstractMatrix{Bool}) =
    NoncentralSplineModel(
    m.intrinsics, ChartSpline(m.directions.area, _absent_nan(m.directions.controls, present)),
    ChartSpline(m.points.area, _absent_nan(m.points.controls, present)); present
)

"""
    window(m::NoncentralSplineModel, rows, cols; directions = nothing, points = nothing) -> NoncentralSplineModel

The block `rows × cols` (unit ranges into the controls `[gy, gx]`, at least
4 × 4) of the model as a model of its own, as [`window`](@ref) of a
`SplineLens` camera: the same `K`, the block of both splines' controls (or
`directions`, `points`, equal-size matrices of 3-vectors, kept as given) over
the rectangle they span, and the block of the presence mask.
"""
function window(
        m::NoncentralSplineModel, rows::AbstractUnitRange{<:Integer}, cols::AbstractUnitRange{<:Integer};
        directions::Union{Nothing, AbstractMatrix} = nothing, points::Union{Nothing, AbstractMatrix} = nothing
    )
    gsz = size(m.directions.controls)
    area = _window_area(m.directions.area, gsz, rows, cols)
    D = _window_controls(m.directions.controls, rows, cols, directions)
    P = _window_controls(m.points.controls, rows, cols, points)
    present = m.present === nothing ? nothing : m.present[rows, cols]
    return NoncentralSplineModel(m.intrinsics, ChartSpline(area, D), ChartSpline(area, P); present)
end

"""
    valid_domain(m::NoncentralSplineModel) -> Rect2{Float64}

Where the model is defined, in pixels: its [`central_model`](@ref)'s valid domain.
"""
valid_domain(m::NoncentralSplineModel) = valid_domain(central_model(m))

"""
    backproject(m::NoncentralSplineModel, u::StaticVector{2}) -> Ray3{Float64}

The observation LINE of the pixel `u`: the direction of the
[`central_model`](@ref) (normalized) and the point spline's value (plain metres
in the camera frame). A ray of NaN origin and direction where the direction lens
does not image `u` (outside the area, or a window holding an absent node): the
lens contract's form, never `nothing`.
"""
function backproject(m::NoncentralSplineModel, u::StaticVector{2})
    x = _normalized_point(m.intrinsics, u)
    d = backproject(_direction_lens(m), x)
    o = isnan(d[1]) ? SVector(NaN, NaN, NaN) : m.points(x)
    return Ray3{Float64}(Vec3{Float64}(o[1], o[2], o[3]), Vec3{Float64}(d[1], d[2], d[3]))
end

"""
    backproject_with_jacobian(m::NoncentralSplineModel, u) -> (f, M)

The unit DIRECTION `f` of the pixel `u`'s observation line and its Jacobian
`M = ∂f/∂u` (3×2): the [`central_model`](@ref)'s. The line's point is not
differentiated: `backproject` gives it.
"""
backproject_with_jacobian(m::NoncentralSplineModel, u::StaticVector{2}) = backproject_with_jacobian(central_model(m), u)
# The Backprojection of a model with its own back-projection Jacobian takes it
# (equirectangular.jl): a CameraModel's, and this model's direction Jacobian.
transform_deriv(b::Backprojection{<:Union{CameraModel, NoncentralSplineModel}}, q::StaticVector{2}) =
    last(backproject_with_jacobian(b.model, _float_static(q)))

# The non-central projection's residual at the normalized point `p` against the
# camera-frame point `X`: the two components of `origin(p) − X` in the tangent basis
# of the line direction `d̂(p)` (the reference's tangent-plane point-to-line
# residual), both splines read through ONE window of the lattice of `(rows, cols) =
# size(dctrl)` over `area` (the continued window, as its Jacobian's). Generic in `p`'s
# element type (ForwardDiff's duals).
@inline function _noncentral_residual(dctrl::AbstractMatrix, pctrl::AbstractMatrix, area, p::SVector{2}, X::SVector{3})
    gsz = gh, gw = size(dctrl)
    ix, iy, wx, wy = _continued_window(_lattice_coords(area, gsz, p)..., gw, gh)   # its Jacobian's window
    d = _eval_window(dctrl, ix, iy, wx, wy)
    o = _eval_window(pctrl, ix, iy, wx, wy)
    t1, t2 = direction_tangents(SVector{3}(d) / norm(d))
    pto = SVector{3}(o) - X
    return SVector(dot(t1, pto), dot(t2, pto))
end

# The Jacobian `∂r/∂p` (2 × 2) of that residual, analytic: the direction spline's
# `_spline_direction_and_jacobian` and the point spline's `_spline_value_and_partials`
# at `p` (one evaluation of the basis weights each), chained through the tangent
# basis. With `t₁ = (d × a)/‖d × a‖`, `t₂ = d × t₁` (`a` = `_tangent_axis(d)`,
# piecewise constant) and `w = o − X`: `∂(t₁·w) = δd·(a × w⊥)/‖d × a‖ + t₁·δo` and
# `∂(t₂·w) = δd·(t₁ × w + (a × q⊥)/‖d × a‖) + t₂·δo`, `q = w × d`, `⊥` the part
# orthogonal to `t₁`. No automatic differentiation inside the LM: a ForwardDiff
# Jacobian there is the pattern Julia 1.12 and 1.13 miscompile at -O2
# (test/fixtures/repro_o2_miscompile.jl).
@inline function _noncentral_residual_jacobian(dctrl::AbstractMatrix, pctrl::AbstractMatrix, area, p::SVector{2}, X::SVector{3})
    d, Jd = _spline_direction_and_jacobian(dctrl, area, p)
    o, ox, oy = _spline_value_and_partials(pctrl, area, p)
    a = _tangent_axis(d)
    u = cross(d, a)
    nu = norm(u)
    t1 = u / nu
    t2 = cross(d, t1)
    w = o - X
    q = cross(w, d)
    g1 = cross(a, w - t1 * dot(t1, w)) / nu
    g2 = cross(t1, w) + cross(a, q - t1 * dot(t1, q)) / nu
    Jo = hcat(ox, oy)
    r1 = transpose(Jd) * g1 + transpose(Jo) * t1          # ∇ₚ(t₁·w)
    r2 = transpose(Jd) * g2 + transpose(Jo) * t2          # ∇ₚ(t₂·w)
    return SMatrix{2, 2}(r1[1], r2[1], r1[2], r2[2])
end

# THE non-central projection: `_spline_lm` on the tangent-plane residual with its
# analytic Jacobian, from the normalized point `x₀`, on the direction area's
# scale-free box; the pixel, NaN unless it converged onto an imaged point.
function _noncentral_project(m::NoncentralSplineModel, X, x₀::SVector{2, Float64})
    Xv = SVector{3, Float64}(X[1], X[2], X[3])
    nan = SVector(NaN, NaN)
    (all(isfinite, Xv) && all(isfinite, x₀)) || return nan
    a = m.directions.area
    res(p, _) = _noncentral_residual(m.directions.controls, m.points.controls, a, p, Xv)
    jac(p, _) = _noncentral_residual_jacobian(m.directions.controls, m.points.controls, a, p, Xv)
    lb, ub, s = _spline_lm_box(a)
    x, ok = _spline_lm(res, jac, x₀, lb, ub, s, _PROJECT_MAXITER, _PROJECT_TOL^2)
    (ok && _in_backward(_direction_lens(m), x)) || return nan
    u = _pixel_affine(m.intrinsics)(x)
    return SVector{2, Float64}(u[1], u[2])
end

"""
    project(m::NoncentralSplineModel, X) -> SVector{2}
    project(m::NoncentralSplineModel, X, p₀) -> SVector{2}

Project the camera-frame point `X` — plain metres, the point spline's unit — to
a pixel: Levenberg–Marquardt in the normalized plane on the reference's
tangent-plane point-to-line residual, so a converged pixel's observation line
goes through `X`. The two-argument form starts at the area's centre, the
three-argument form at the seed pixel `p₀` (a good seed matters for a
non-central model; positional, per-call data). NaN when the solve does not
converge onto a point the model images (inside the area, its window present).
"""
function project(m::NoncentralSplineModel, X)
    o = GeometryBasics.origin(m.directions.area); w = GeometryBasics.widths(m.directions.area)
    return _noncentral_project(m, X, SVector{2, Float64}(o[1] + w[1] / 2, o[2] + w[2] / 2))
end
project(m::NoncentralSplineModel, X, p₀) =
    _noncentral_project(m, X, SVector{2, Float64}(_normalized_point(m.intrinsics, SVector{2, Float64}(p₀[1], p₀[2]))))

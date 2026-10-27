# =============================================================================
# The image circle — where on the sensor a lens puts light, PREDICTED
# =============================================================================
#
# A local lens model (a node-based grid lens) has observations only where the
# lens images: its domain is `Rect(extent) ∩ image_circle(camera)`. The circle
# is predicted, never segmented from an image: from the fitted lens (its
# `view_angle_limit` through its radial projection and K), or from a lens
# datasheet's image-circle diameter through the sensor's pixel pitch. Both
# routes return the same primitive, an `Ellipse` in the camera's pixel frame,
# so the consumer does not care which built it. The region algebra (`∩`, the
# reverse containment `Rect ∈ Ellipse`) is the primitives' (primitives/
# regions.jl, primitives/containment.jl); nothing region-specific lives here.

"""
    image_circle(camera) -> Union{Ellipse{Float64}, Nothing}
    image_circle(camera, sensor::Sensor, diameter::Quantity) -> Ellipse{Float64}
    image_circle(camera::BoundedCamera, diameter::Quantity) -> Ellipse{Float64}

The image circle of a camera: the region of the sensor its lens lights, in
pixels in the camera's own pixel frame (the frame of its calibration matrix
`K`; `Rect(extent)` is the COLMAP-convention extent of the same frame). An
[`Ellipse`](@ref) centred at the principal point — a circle (equal axes) when
`K` is isotropic without skew and the pitch square, an ellipse otherwise.

**From the lens** (`image_circle(camera)`): the image of the cone of view
angles `θ ≤ θ_max = `[`view_angle_limit`](@ref)`(lens)`, the conic
`K⁻ᵀ · diag(1, 1, −ρ²) · K⁻¹` with `ρ` the lens's image radius at `θ_max`
([`projected_radius`](@ref) for a radially symmetric lens). For a lens that is
not radially symmetric — a [`Chain`](@ref) with a tilt stage, tangential or
thin-prism terms — `ρ` is the smallest image radius of the `θ_max` cone over
360 azimuths, so the circle is an INSCRIBED bound: every point inside it is
imaged, and the lit region may be slightly larger.

`nothing` when the lens has no limit of its own: `view_angle_limit` reports
only the perspective chart's edge (a perspective lens, or one whose radial
profile stays monotone to that edge), where the image radius grows without
bound — a conic there would measure a cap constant, not the lens. `nothing`
means "no circle, nothing to restrict" (`Rect(extent) ∩ nothing` is the
extent). A lens WITH a limit — a fold, a pole, the double sphere's bound, a
fisheye frame's edge — always gets its conic, even one larger than the
extent; whether it lights the whole extent is `Rect(extent(camera)) ∈ circle`.

**From a datasheet** (`image_circle(camera, sensor, diameter)`): a lens
datasheet's image-circle `diameter`, a length, carried into pixels by the
sensor's pitch: semi-axes `diameter / (2 · pitch.width)` and
`diameter / (2 · pitch.height)` px about the principal point. A
`DimensionError` when `diameter` is not a length. Always a conic (a stated
diameter is a limit). A [`BoundedCamera`](@ref) bounded by its physical
[`Sensor`](@ref) takes the two-argument form; one bounded by a bare extent is an
`ArgumentError` (it has no pitch).

The consumer's domain is `Rect(extent(camera)) ∩ image_circle(camera)` (a lazy
[`RegionIntersection`](@ref VisualGeometryToolkitCore.RegionIntersection):
`p ∈ domain`). Only a LOCAL lens model needs it — a grid lens whose nodes
outside the circle would have no observations; a global parametric lens is
defined everywhere by its formula and needs no restriction.

```julia
cam = BoundedCamera(CameraModel(K; lens = Lens(Equidistant())), sensor("Sony", "IMX250"))
q = image_circle(cam)                       # from the lens, about the principal point
image_circle(cam, 17.3mm)                   # from the datasheet, through the 3.45 μm pitch
Rect(extent(cam)) ∈ q                       # is the whole sensor lit?
domain = Rect(extent(cam)) ∩ q              # where a local lens model lives
```
"""
function image_circle(camera::CameraModel)
    l = lens(camera)
    θ = Float64(view_angle_limit(l))
    _has_image_circle(l, θ) || return nothing
    ρ = _inscribed_radius(l, θ)
    K = SMatrix{3, 3, Float64}(calibration_matrix(camera))
    Ki = inv(K)
    C = transpose(Ki) * SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, -ρ^2) * Ki
    return Ellipse(HomEllipseMat{Float64}(Tuple(C)))
end
function image_circle(camera::CameraModel, sensor::Sensor, diameter::Quantity)
    dimension(diameter) == dimension(mm) || throw(DimensionError(diameter, mm))
    K = calibration_matrix(camera)
    a = Float64(ustrip(px, diameter / (2 * sensor.pitch.width)))
    b = Float64(ustrip(px, diameter / (2 * sensor.pitch.height)))
    return Ellipse(Point2{Float64}(K[1, 3], K[2, 3]), a, b, 0.0)
end
image_circle(c::Union{PosedCamera, BoundedCamera}) = image_circle(model(c))
image_circle(c::Union{PosedCamera, BoundedCamera}, s::Sensor, diameter::Quantity) = image_circle(model(c), s, diameter)
function image_circle(bc::BoundedCamera, diameter::Quantity)
    s = sensor(bc)
    s === nothing && throw(
        ArgumentError(
            "image_circle: the camera is bounded by a bare pixel extent, which has no pitch — bound it by its " *
                "physical sensor, `BoundedCamera(camera, sensor)`, or supply one: `image_circle(camera, sensor, diameter)`"
        )
    )
    return image_circle(model(bc), s, diameter)
end

# Does the lens limit its view angle itself, short of the perspective chart's
# edge? A perspective-frame lens reporting that edge has no image circle (its
# radius grows without bound there); every other frame's edge bounds a finite
# disc (an equidistant fisheye's at θ = π). A lift: only when it folds.
function _has_image_circle(l::Lens, θ)
    C = projection_frame(l.projection)
    return C === nothing || !(C <: GnomonicFrame) || θ < _PERSPECTIVE_EDGE
end
_has_image_circle(l::LiftLens, θ) = isfinite(_fold_radius(l))

# The image radius of the θ cone: its smallest over 360 azimuths, so the disc
# of that radius lies inside the image of the cone — the radius itself for a
# radially symmetric lens.
function _inscribed_radius(l::AbstractLensModel, θ::Float64)
    ρ = Inf
    for ψ in range(0.0, 2π; length = 361)[1:360]
        r = _radius(project(l, SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))))
        isfinite(r) && (ρ = min(ρ, r))
    end
    return ρ
end

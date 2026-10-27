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
    image_circle(camera) -> Union{Ellipse{Float64}, WholePlane}
    image_circle(camera, sensor::Sensor, diameter::Quantity) -> Ellipse{Float64}
    image_circle(camera::BoundedCamera, diameter::Quantity) -> Ellipse{Float64}

The image circle of a camera: the region of the sensor its lens lights, in
pixels in the camera's own pixel frame (the frame of its calibration matrix
`K`; `Rect(extent)` is the COLMAP-convention extent of the same frame). An
[`Ellipse`](@ref) centred at the principal point — a circle (equal axes) when
`K` is isotropic without skew and the pitch square, an ellipse otherwise — or
the [`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane).

**From the lens** (`image_circle(camera)`): the image of the cone of view
angles `θ ≤ θ_max = `[`view_angle_limit`](@ref)`(lens)`, the conic
`K⁻ᵀ · diag(1, 1, −ρ²) · K⁻¹` with `ρ` the lens's image radius at `θ_max`
([`projected_radius`](@ref) for a radially symmetric lens). For a lens that is
not radially symmetric — a [`Chain`](@ref) with a tilt stage, tangential or
thin-prism terms — `ρ` is the smallest image radius of the `θ_max` cone over
the azimuths (360 samples, the least refined by golden section) less a
relative margin of 10⁻⁹, so the circle is INSCRIBED: every point inside it is
imaged, and the lit region may be slightly larger.

The `WholePlane` when the lens has no limit of its own: `view_angle_limit`
reports only the edge it stops at by construction — the perspective or
orthographic chart's edge, or the sphere's `π − 10⁻³` (an undistorted
equidistant lens, a lens whose profile stays monotone, a lift that never
folds). That edge is a cap, not the lens: a conic there would measure the cap
constant. The whole plane restricts nothing (`Rect(extent) ∩ WholePlane()` is
the extent, `Rect(extent) ∈ WholePlane()` is true); such a lens's circle, if it
has one, comes from its datasheet. A lens WITH a limit — a fold, a pole, the
double sphere's bound, a chain's bounded stage — always gets its conic, even
one larger than the extent; whether it lights the whole extent is
`Rect(extent(camera)) ∈ circle`. The rule is generic over `AbstractLensModel`:
any lens with a `view_angle_limit` (every lens has one, by the generic scan) and
a `project` gets an answer.

**From a datasheet** (`image_circle(camera, sensor, diameter)`): a lens
datasheet's image-circle `diameter`, a length, carried into pixels by the
sensor's pitch: semi-axes `diameter / (2 · pitch.width)` and
`diameter / (2 · pitch.height)` px about the principal point, axis-aligned. The
datasheet circle is physical, on the sensor plane, and the pitch defines the
pixel axes, so `K`'s skew and focal lengths do not enter. A `DimensionError`
when `diameter` is not a length. Always a conic (a stated diameter is a
limit). A [`BoundedCamera`](@ref) bounded by its physical [`Sensor`](@ref)
takes the two-argument form; one bounded by a bare extent is an
`ArgumentError` (it has no pitch).

The consumer's domain is `Rect(extent(camera)) ∩ image_circle(camera)` (a lazy
[`RegionIntersection`](@ref VisualGeometryToolkitCore.Primitives.RegionIntersection):
`p ∈ domain`). Only a LOCAL lens model needs it — a grid lens whose nodes
outside the circle would have no observations; a global parametric lens is
defined everywhere by its formula and needs no restriction.

Related radii, different quantities: [`horizon_radius`](@ref) is `fx` times
[`monotone_radius`](@ref), the radial factor's perspective-monotone bound
(decentering ignored, `fx` only, `Inf` without a fold) — equal to this
circle's radius for an isotropic, skew-free `K` and a radially symmetric
folding lens; the imaging domain (where `backproject` turns NaN) may be wider
than a whole-plane answer admits, e.g. `r ≤ π·f` for an undistorted
equidistant lens.

```julia
cam = BoundedCamera(CameraModel(K; lens = KannalaBrandt(-0.135)), sensor("Sony", "IMX250"))
q = image_circle(cam)                       # from the lens's fold, about the principal point
image_circle(cam, 6.0mm)                    # from the datasheet, through the 3.45 μm pitch
Rect(extent(cam)) ∈ q                       # is the whole sensor lit?
domain = Rect(extent(cam)) ∩ q              # where a local lens model lives
```
"""
function image_circle(camera::CameraModel)
    l = lens(camera)
    θ = Float64(view_angle_limit(l))
    θ < _view_angle_edge(l) || return WholePlane()   # the edge is a cap: no limit of its own
    ρ = _inscribed_radius(l, θ)
    isfinite(ρ) || throw(
        ArgumentError(
            "image_circle: the lens $(nameof(typeof(l))) images no ray at its view-angle limit θ = $(θ); " *
                "its view_angle_limit and project disagree"
        )
    )
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

# The image radius of the θ cone: its minimum over the azimuths (`_azimuth_minimum`,
# projections.jl: 360 samples, golden-section refined), less a relative margin of
# 10⁻⁹ for what the refinement leaves, so the disc of that radius lies inside
# the image of the cone. An azimuth the lens does not image counts as Inf.
function _inscribed_radius(l::AbstractLensModel, θ::Float64)
    function r(ψ)
        v = _radius(project(l, SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))))
        return isfinite(v) ? Float64(v) : Inf
    end
    return _azimuth_minimum(r, 360) * (1 - 1.0e-9)
end

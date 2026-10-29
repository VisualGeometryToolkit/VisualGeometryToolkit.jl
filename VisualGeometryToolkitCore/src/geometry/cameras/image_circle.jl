# =============================================================================
# The valid domain of a lens model, and the image circle of a lens
# =============================================================================
#
# Two different regions of the sensor, both an `Ellipse` (or the whole plane)
# in the camera's pixel frame:
#
# - `valid_domain(camera)`: where the lens MODEL is valid — the image of the
#   cone its radial profile stays finite and strictly increasing on, up to its
#   `view_angle_limit`. A property of the formula, computed from the model.
#   It does NOT say where light falls: a fitted Kannala–Brandt profile can fold
#   far outside the lit disc (blob_circ: fold at r = 1626 px, light stops at
#   1318 px).
# - `image_circle(camera, sensor, diameter)`: where light falls, from a lens
#   datasheet's diameter through the sensor's pixel pitch (or a measured
#   circle the caller supplies). `imaged_region(camera, circle)` intersects a
#   supplied circle with the extent.
#
# The region algebra (`∩`, the reverse containment `Rect ∈ Ellipse`) is the
# primitives' (primitives/regions.jl, primitives/containment.jl); nothing
# region-specific lives here.

"""
    valid_domain(camera) -> Union{Ellipse{Float64}, WholePlane}

Where the camera's lens MODEL is valid, in pixels in the camera's own pixel
frame (the frame of its calibration matrix `K`): the image of the cone of view
angles `θ ≤ θ_max = `[`view_angle_limit`](@ref)`(lens)`, on which the lens's
radial profile is finite and strictly increasing, so the model is radially
injective. The conic `K⁻ᵀ · diag(1, 1, −ρ²) · K⁻¹` with `ρ` the lens's image
radius at `θ_max` ([`projected_radius`](@ref) for a radially symmetric lens): an
[`Ellipse`](@ref) centred at the principal point, a circle (equal axes) only
when `K` is isotropic without skew and the pitch square.

This is a property of the formula, **not where light falls on the sensor**: a
fitted lens's fold can lie well outside the lit disc, or inside the extent where
the sensor is dark. Where light falls is the lens's
[`image_circle`](@ref)`(camera, sensor, diameter)` from its datasheet, or a
measured circle; [`imaged_region`](@ref)`(camera, circle)` takes either.

For a lens that is not radially symmetric — a [`Chain`](@ref) with a tilt
stage, tangential or thin-prism terms — `ρ` is the smallest image radius of the
`θ_max` cone over the azimuths (360 samples, the least refined by golden
section) less a relative margin of 10⁻⁹, so the ellipse is INSCRIBED: every
point inside it is the image of a ray inside the cone.

The [`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane) when the
lens has no limit of its own: `view_angle_limit` reports only the edge it stops
at by construction — the perspective or orthographic chart's edge, or the
sphere's `π − 10⁻³` (an undistorted equidistant lens, a lens whose profile stays
monotone, a lift that never folds). That edge is a cap, not the lens: a conic
there would measure the cap constant. A lens WITH a limit — a fold, a pole, the
double sphere's bound, a chain's bounded stage — always gets its conic, even one
larger than the extent. The rule is generic over `AbstractLensModel`: any lens
with a `view_angle_limit` (every lens has one, by the generic scan) and a
`project` gets an answer. A `PosedCamera` or `BoundedCamera` forwards to its
model.

Related radii, different quantities: [`horizon_radius`](@ref) is `fx` times
[`monotone_radius`](@ref), the radial factor's perspective-monotone bound
(decentering ignored, `fx` only, `Inf` without a fold) — equal to this
ellipse's radius for an isotropic, skew-free `K` and a radially symmetric
folding lens; the imaging domain (where `backproject` turns NaN) may be wider
than a whole-plane answer admits, e.g. `r ≤ π·f` for an undistorted
equidistant lens.

```julia
cam = BoundedCamera(CameraModel(K; lens = KannalaBrandt(-0.135)), sensor("Sony", "IMX250"))
v = valid_domain(cam)                       # where the lens formula is injective, about the principal point
Rect(extent(cam)) ∈ v                       # is the model valid over the whole sensor?
```
"""
function valid_domain(camera::CameraModel)
    ρ = _valid_domain_radius(lens(camera))
    return ρ === nothing ? WholePlane() : _pixel_disc(camera, ρ)
end
valid_domain(c::Union{PosedCamera, BoundedCamera}) = valid_domain(model(c))

"""
    image_circle(camera, sensor::Sensor, diameter::Quantity) -> Ellipse{Float64}
    image_circle(camera::BoundedCamera, diameter::Quantity) -> Ellipse{Float64}

The image circle of a lens: the region of the sensor it lights, from a lens
datasheet's image-circle `diameter`, a length, carried into pixels by the
sensor's pitch — semi-axes `diameter / (2 · pitch.width)` and
`diameter / (2 · pitch.height)` px about the principal point, axis-aligned, in
the camera's pixel frame (the frame of its calibration matrix `K`;
`Rect(extent)` is the COLMAP-convention extent of the same frame). The datasheet
circle is physical, on the sensor plane, and the pitch defines the pixel axes,
so `K`'s skew and focal lengths do not enter. A `DimensionError` when `diameter`
is not a length. A [`BoundedCamera`](@ref) bounded by its physical
[`Sensor`](@ref) takes the two-argument form; one bounded by a bare extent is an
`ArgumentError` (it has no pitch).

The image circle is never predicted from the lens model: where the model's
formula is valid is [`valid_domain`](@ref), a different region. A measured
circle (an `Ellipse` segmented from images) serves wherever this one does.
Where the camera images is [`imaged_region`](@ref)`(camera, circle)`.

```julia
cam = BoundedCamera(CameraModel(K; lens = KannalaBrandt(-0.135)), sensor("Sony", "IMX250"))
q = image_circle(cam, 6.0mm)                # from the datasheet, through the 3.45 μm pitch
Rect(extent(cam)) ∈ q                       # is the whole sensor lit?
```
"""
function image_circle(camera::CameraModel, sensor::Sensor, diameter::Quantity)
    dimension(diameter) == dimension(mm) || throw(DimensionError(diameter, mm))
    a = Float64(ustrip(px, diameter / (2 * sensor.pitch.width)))
    b = Float64(ustrip(px, diameter / (2 * sensor.pitch.height)))
    return Ellipse(_principal_point(camera), a, b, 0.0)
end
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

# The lens model's valid domain as a normalized radius ρ (the radius of the disc
# the θ_max cone images, inscribed), or `nothing` when the lens has no limit of
# its own. The one owner of that radius: `valid_domain` and the generic models'
# node sampler (generic_models.jl) read it here.
function _valid_domain_radius(l::AbstractLensModel)
    θ = Float64(view_angle_limit(l))
    _own_view_angle_limit(l, θ) || return nothing   # the edge is a cap: no limit of its own
    ρ = _inscribed_radius(l, θ)
    isfinite(ρ) || throw(
        ArgumentError(
            "valid_domain: the lens $(nameof(typeof(l))) images no ray at its view-angle limit θ = $(θ); " *
                "its view_angle_limit and project disagree"
        )
    )
    return ρ
end

# Whether `view_angle_limit` reported the lens's own limit rather than the edge
# it stops at. The double sphere's limit is always its own — the end of its
# radial profile in closed form (`_ds_end_cos`) less 10⁻⁶, which reaches past
# the sphere's edge `π − 10⁻³` when the profile runs to θ = π (ξ = 1 images the
# whole sphere but the backward axis, and its rim conic is finite: radius 1/α,
# 2 at α = ½).
_own_view_angle_limit(l::AbstractLensModel, θ) = θ < _view_angle_edge(l)
_own_view_angle_limit(::Lens{<:DoubleSphere, Nothing}, θ) = true

# The pixel image of the normalized disc of radius ρ through the camera's K:
# the conic K⁻ᵀ·diag(1, 1, −ρ²)·K⁻¹ as an Ellipse.
function _pixel_disc(camera::CameraModel, ρ::Real)
    K = SMatrix{3, 3, Float64}(calibration_matrix(camera))
    Ki = inv(K)
    C = transpose(Ki) * SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, -Float64(ρ)^2) * Ki
    return Ellipse(HomEllipseMat{Float64}(Tuple(C)))
end

# The image radius of the θ cone: its minimum over the azimuths (`_azimuth_minimum`,
# projections.jl: 360 samples, golden-section refined), less a relative margin of
# 10⁻⁹ for what the refinement leaves, so the disc of that radius lies inside
# the image of the cone. An azimuth the lens does not image counts as Inf.
function _inscribed_radius(l::AbstractLensModel, θ::Float64)
    rim(t) = ψ -> (v = _radius(project(l, SVector(sin(t) * cos(ψ), sin(t) * sin(ψ), cos(t)))); isfinite(v) ? Float64(v) : Inf)
    ρ = _azimuth_minimum(rim(θ), 360)
    if !isfinite(ρ)
        # `project` inverts some lenses numerically (a lift) and may fail just
        # short of a fold the closed form reports; retreat to the last angle it
        # answers (bisected), whose disc is smaller and so still inscribed
        θ = _nan_onset(t -> isfinite(projected_radius(l, t)), θ)
        ρ = _azimuth_minimum(rim(θ), 360)
    end
    return ρ * (1 - 1.0e-9)
end

"""
    imaged_region(camera::BoundedCamera, circle) -> Union{RegionIntersection, Rect}

The region of the sensor the camera images, in pixels in its pixel frame:
`Rect(extent(camera)) ∩ circle` — the one place the extent and the image circle
are combined. The image circle is SUPPLIED: a datasheet's
[`image_circle`](@ref)`(camera, diameter)` or a measured one (any region, e.g.
an `Ellipse` segmented from images). There is no form that predicts it from the
lens: the lens model's [`valid_domain`](@ref) says where its formula holds, not
where light falls.

A lazy [`RegionIntersection`](@ref VisualGeometryToolkitCore.Primitives.RegionIntersection)
of the extent and the circle, or the extent's `Rect` itself when the circle is
the [`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane) (`∩`
returns it). The pixel mask of any residual is `u ∈ imaged_region(camera, circle)`;
the free parameters of a local model are
[`free_nodes`](@ref)`(model, imaged_region(camera, circle))`.
"""
imaged_region(camera::BoundedCamera, circle) = Rect(0, 0, extent(camera)...) ∩ circle

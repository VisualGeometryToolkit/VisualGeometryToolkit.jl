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
#   circle the caller supplies), stored on a bounded camera as its `lens_coverage`.
#   `imaged_region(camera)` intersects the extent, the lens coverage and the valid
#   domain.
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
measured circle; a bounded camera holds either as its [`lens_coverage`](@ref).

For a lens that is not radially symmetric — a composed distortion (`∘`) with
a tilt stage, tangential or thin-prism terms — `ρ` is the smallest image radius of the
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
A [`SplineLens`](@ref) camera's circle is centred on its gauge principal point,
the pixel imaging ẑ. Where the camera images is [`imaged_region`](@ref)`(camera)`.

```julia
cam = BoundedCamera(CameraModel(K; lens = KannalaBrandt(-0.135)), sensor("Sony", "IMX250"))
q = image_circle(cam, 6.0mm)                # from the datasheet, through the 3.45 μm pitch
Rect(extent(cam)) ∈ q                       # is the whole sensor lit?
```
"""
function image_circle(camera::CameraModel, sensor::Sensor, diameter::Quantity)
    dimension(diameter) == dimension(mm) || throw(DimensionError(diameter, mm))
    _check_lens_axis(camera.lens, "image_circle")
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
# its own. The one owner of that radius: `valid_domain` and the spline lattice's
# node sampler (spline_lens.jl) read it here.
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
    imaged_region(camera::BoundedCamera) -> region

The region of the sensor the camera images, in pixels in its pixel frame: the
sensor `Rect(extent(camera))` ∩ its [`lens_coverage`](@ref) (where light reaches
the sensor: the lens's image circle, measured or from a datasheet's
[`image_circle`](@ref), or the whole sensor) ∩ the [`valid_domain`](@ref) of
its model (where the lens formula holds, or where a grid model is defined).
The one place the three are combined, so a consumer asks the camera and never
carries an image circle of its own. A lens coverage larger than the sensor, or
cut by it, is clipped to the sensor.

A lazy [`RegionIntersection`](@ref VisualGeometryToolkitCore.Primitives.RegionIntersection),
or a `Rect` (the sensor's, in its eltype) when the whole sensor is covered and the model has no
limit (`∩` returns it). The pixel mask of any residual is
`u ∈ imaged_region(camera)`; the free parameters of a local model are
[`free_nodes`](@ref)`(model, imaged_region(camera))`.
"""
imaged_region(bc::BoundedCamera) = (Rect(0, 0, extent(bc)...) ∩ lens_coverage(bc)) ∩ valid_domain(model(bc))

"""
    RegionBeyondSupport(who, region, support) <: Exception

What [`check_region_support`](@ref) throws: `who` (the fit that refused) was
asked to work over `region`, which reaches beyond `support`, the source
camera's calibrated support (its [`valid_domain`](@ref)) — a fit there would
read the model where it is extrapolated.
"""
struct RegionBeyondSupport <: Exception
    who::String
    region::Any
    support::Any
end
Base.showerror(io::IO, e::RegionBeyondSupport) = print(io, "RegionBeyondSupport: ", e.who, ": the region ", e.region,
    " reaches beyond the source camera's calibrated support ", e.support, " (its valid_domain): a fit there reads the model ",
    "where it is extrapolated. Fit over a region inside the support (`imaged_region` of the camera bounded to it, ",
    "`with_lens_coverage(camera, support)`), or calibrate the rest.")

"""
    check_region_support(camera, region; who::AbstractString) -> region
    check_region_support(support, region; who::AbstractString) -> region

THE guard of every fit over a camera's region (`optical_axis`,
`approximate_camera`, `refine_camera`, …): `region` back when it lies inside
the source's calibrated support — `valid_domain(camera)`, or a `support`
region given directly (a source camera's, when the camera fitted from is a
re-gridded copy whose own domain no longer records it) — and a
[`RegionBeyondSupport`](@ref) naming `who`, the region and the support
otherwise. Never a warning: a fit over an extrapolated model is refused.
Containment is `region ∈ support` (a `Rect`, an `Ellipse`, their `∩`, the
`WholePlane`).
"""
check_region_support(camera::Union{CameraModel, PosedCamera, BoundedCamera, NoncentralSplineModel}, region; who::AbstractString) =
    check_region_support(valid_domain(camera), region; who)
function check_region_support(support, region; who::AbstractString)
    region ∈ support || throw(RegionBeyondSupport(String(who), region, support))
    return region
end

"""
    farthest_radius(region, center) -> Float64
    farthest_radius(region, center, direction) -> Float64

How far `region` reaches from the point `center`, in the region's units (pixels
for an [`imaged_region`](@ref)).

- `farthest_radius(region, center)` is the largest distance from `center` to a
  point of the region. A `Rect` has the exact value, its farthest corner. Any
  other region uses the second form on 720 azimuths, refines the largest by
  Brent's method between its neighbours (`grid_brent_minimum`, Core's one 1-D
  minimizer) and returns the largest value it evaluated.
- `farthest_radius(region, center, direction)` is the largest `t` with
  `center + t·direction` in the region, bisected to rounding, in units of
  `‖direction‖`. A direction through a linear map, `A·d`, gives the radius in
  the coordinates that `A` maps to pixels.

The region is any bounded 2-D region that answers `p ∈ region`: a `Rect`, an
`Ellipse`, a `Circle`, or their `∩`. It must contain `center` and be
star-shaped about it, which every intersection of convex parts containing
`center` is. A `Rect`'s farthest point is a corner from any `center`, inside or
not (an image's farthest corner from a distortion centre:
[`division_bounded`](@ref)).

The sensor clips the result. An `imaged_region` is the sensor's `Rect`
intersected with the lens coverage, so an image circle larger than the sensor,
or cut by it, reaches only as far as the sensor does.

The whole plane has no farthest point, so it is an `ArgumentError` (from its
`Rect2`). So is a `center` outside any other region.

```julia
r = imaged_region(camera)
farthest_radius(r, principal_point)   # a radial profile's domain: ρ ≤ 1 on every imaged pixel
```
"""
farthest_radius(r::Rect{2}, center) = maximum(p -> norm(p - _plain_centre(center)), rect_corners(Rect2{Float64}(r)))
function farthest_radius(region, center)
    c = _plain_centre(center)
    _check_farthest_center(region, c)
    reach(ψ) = -farthest_radius(region, c, SVector(cos(ψ), sin(ψ)))
    return -_azimuth_minimum(reach, _FARTHEST_AZIMUTHS)
end
function farthest_radius(region, center, direction)
    c = _plain_centre(center)
    u = SVector{2, Float64}(direction[1], direction[2])
    box = Rect2(region)               # an ArgumentError for the whole plane: the region must be bounded
    _check_farthest_center(region, c)
    nu = norm(u)
    (isfinite(nu) && nu > 0) || throw(ArgumentError("farthest_radius: the direction $direction is not a finite non-zero vector"))
    hi = (maximum(p -> norm(p - c), rect_corners(Rect2{Float64}(box))) * (1 + 1.0e-9) + 1.0e-12) / nu
    a, b = 0.0, hi
    for _ in 1:60
        m = (a + b) / 2
        Point2{Float64}(c + m * u) ∈ region ? (a = m) : (b = m)
    end
    return a
end
const _FARTHEST_AZIMUTHS = 720
_plain_centre(center) = SVector{2, Float64}(center[1], center[2])
_check_farthest_center(region, center) = Point2{Float64}(_plain_centre(center)) ∈ region ||
    throw(ArgumentError("farthest_radius: the region does not contain the centre $(Tuple(center))"))

"""
    pixel_mask(region, width, height) -> BitMatrix

The pixels of a `width × height` image whose CENTRES lie in `region` (any 2-D region answering `p ∈ region`: an
[`imaged_region`](@ref), a `Rect`, an [`Ellipse`](@ref), their `∩`, the
[`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane)), as a `height × width` `BitMatrix` in Julia's array
layout: `mask[v, u]` is pixel `(u, v)`, whose centre is the COLMAP point `(u − ½, v − ½)` — the conversion
[`pixel_centres`](@ref) shares, made in one place.

```julia
region = imaged_region(camera)                # the sensor ∩ its lens coverage ∩ where its lens model is valid
mask = pixel_mask(region, width, height)      # the pixels the camera images
```
"""
pixel_mask(region, width::Integer, height::Integer) =
    BitMatrix([_pixel_centre(u, v) ∈ region for v in 1:Int(height), u in 1:Int(width)])

"""
    pixel_mask(region, frame_map, width, height) -> BitMatrix

The pixels of a `width × height` image whose centres, mapped by `frame_map` into
the frame `region` is stated in, lie in it: `mask[v, u]` is
`frame_map((u − ½, v − ½)) ∈ region`. For an image on another pixel lattice than
the region's — a Bayer plane or channel against a region on the sensor
(`pixel_mask(imaged_region(camera), bayer_plane_map(L, c), w, h)`), a window of
a mosaic (`raw_frame_map(m)`) — the membership is decided in the region's own
frame, at each pixel's mapped centre, never by mapping the region. Without a
map it is [`pixel_mask`](@ref)`(region, width, height)` (the identity).
"""
pixel_mask(region, frame_map::CoordinateTransformations.Transformation, width::Integer, height::Integer) =
    BitMatrix([Point2{Float64}(frame_map(SVector{2, Float64}(_pixel_centre(u, v)))) ∈ region for v in 1:Int(height), u in 1:Int(width)])

"""
    pixel_centres(region, us, vs) -> Vector{SVector{2, Float64}}

The COLMAP centres `(u − ½, v − ½)` of the pixels `(u, v)`, `u ∈ us`, `v ∈ vs`
(Julia's 1-based pixel indices, as [`pixel_mask`](@ref) indexes them), that lie
in `region`, `v` outer and `u` inner: the pixels of a sub-lattice (a stride, an
offset) without the full image's mask. `pixel_mask(region, W, H)[v, u]` is
whether pixel `(u, v)`'s centre is among `pixel_centres(region, 1:W, 1:H)`.
"""
pixel_centres(region, us::AbstractVector{<:Integer}, vs::AbstractVector{<:Integer}) =
    [SVector{2, Float64}(c) for v in vs for u in us for c in (_pixel_centre(u, v),) if c ∈ region]

"""
    pixel_centre([T = Float64,] I::CartesianIndex{2}) -> SVector{2, T}

The centre of the pixel at the array index `I = (row, col)` (Julia's 1-based
indices, `img[row, col]`) in the COLMAP frame, `(col − ½, row − ½)`, in `T`:
the one conversion from an image array's index to a pixel point. Allocation-
and branch-free, so a device functor over `CartesianIndices` calls it at the
kernel's element type.
"""
@inline pixel_centre(::Type{T}, I::CartesianIndex{2}) where {T <: Real} = SVector{2, T}(T(I[2]) - T(0.5), T(I[1]) - T(0.5))
@inline pixel_centre(I::CartesianIndex{2}) = pixel_centre(Float64, I)
# Pixel `(u, v)` (column, row) as the regions read it.
_pixel_centre(u::Integer, v::Integer) = Point2{Float64}(pixel_centre(CartesianIndex(v, u)))

# =============================================================================
# The valid domain of a lens model, and the image circle of a lens
# =============================================================================
#
# Two different regions of the sensor in the camera's pixel frame (an `Ellipse`,
# a star `PolygonRegion` or the whole plane):
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
    valid_domain(camera) -> Union{Ellipse{Float64}, PolygonRegion, WholePlane}

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

For a lens that is not radially symmetric — a composed distortion (`∘`) with a tilt stage, tangential or thin-prism
terms (Brown–Conrady, OpenCV), or a lens type without the [`RadialSymmetry`](@ref) trait — the fold radius depends on
the azimuth, and the domain is a [`PolygonRegion`](@ref VisualGeometryToolkitCore.Primitives.PolygonRegion): the star
polygon about the distortion centre (the image of the optical axis) through each azimuth's OWN fold. On each of 720
ray azimuths the image radius `|project(lens, ray(θ, ψ))|` is followed out in `θ` to where it first stops being finite
and strictly increasing (refined past the 1024-sample scan), and that point goes through `K`. The polygon never claims
past a fold: each vertex is pulled in by a relative 10⁻⁹ and by a bound on how far the fold curve can bulge inward of
the chord between neighbours. Beside a jump of the fold radius between two samples (more than 10 % between neighbours:
an azimuth range that does not fold beside one that does) a vertex is capped at the nearer neighbour's fold, so no chord
across the jump claims past it, and none comes nearer the centre than the disc of the smallest fold (inside every fold). A
discontinuity narrower than one sample's spacing (0.5°) that both neighbours miss cannot be seen. Its chords lie within
`R·(1 − cos(π/720)) ≈ R·10⁻⁵` of a smooth fold curve (0.012 px at a fold radius of 1300 px). Until Core 0.10.9 such a lens got the disc at the SMALLEST fold over the azimuths, which excluded
the pixels between it and their own azimuth's fold.

Cost: building the polygon is about 10 ms (720 azimuth folds; the disc of a radially symmetric lens about 13 µs), and
membership in it about 0.1–0.5 µs a point. Build it ONCE per model and hold it — a fit's `support`, an `imaged_region`,
a bundle's camera variable — never per point or per residual.

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
    l = lens(camera)
    # A radially symmetric lens folds at one radius: the disc of it, exact, through K. Any other lens (decentring
    # terms, a tilt stage, a lens type without the trait: `RadialSymmetry` fails closed) folds at a radius that depends
    # on the azimuth: the star polygon through each azimuth's own fold. The trait is a type, so the branch is resolved
    # at compile time. Either way the whole plane when the edge is a cap (no limit of its own).
    if RadialSymmetry(l) isa RadiallySymmetric
        ρ = _valid_domain_radius(l)
        return ρ === nothing ? WholePlane() : _pixel_disc(camera, ρ)
    end
    _own_view_angle_limit(l, Float64(view_angle_limit(l))) || return WholePlane()
    return _fold_polygon(camera)
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
# the θ_max cone images, inscribed over the azimuths: exact for a radially symmetric
# lens, the smallest fold for any other), or `nothing` when the lens has no limit of
# its own. The one owner of that radius: `valid_domain` (a radially symmetric
# lens, where it is exact) and the spline lattice's node sampler (spline_lens.jl)
# read it here.
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

# The azimuths the valid domain's polygon samples. Its edges are chords of the fold curve between neighbours 2π/720 =
# 0.5° apart, so the chord's sag below the curve is at most R·(1 − cos(π/720)) ≈ R·9.5·10⁻⁶: 0.012 px at a fold radius
# of R = 1300 px (a wide fisheye at fx ≈ 1000), 0.03 px at 3000 px — far below a pixel. 720 sector folds cost about
# 10 ms (one 1024-sample scan and a refinement each).
const _FOLD_AZIMUTHS = 720

# A jump of the fold radius between two neighbouring azimuth samples: a relative difference above this. The bulge bound
# (`dip`, h²|R''|/8) covers a SMOOTH fold curve between samples; a discontinuity between them (an azimuth range that
# folds beside one that does not, or folds much farther out) it does not, and a chord across it can claim past the
# nearer fold. A smooth fold curve of a lens's distortion is a low harmonic of the azimuth: a relative variation of
# amplitude ε ≤ 0.5 at harmonic m ≤ 3 changes R by at most ε·m·h ≈ 1.5·(2π/720) ≈ 1.3 % between neighbours (its bulge
# h²|R''|/8 ≈ 10⁻⁵ R adds nothing at that scale). A difference of 10 %, nearly 8× the steepest such curve, is taken as
# a discontinuity, and the vertex beside it is capped at the smaller of its neighbours' radii. Smooth lenses never reach
# it (their polygon is bit for bit the uncapped one).
const _FOLD_JUMP = 0.1

# The valid domain of a lens that is not radially symmetric: the star polygon about the distortion centre (the image
# of the optical axis) through each ray azimuth's own fold (`_azimuth_fold`), in the camera's pixels. In the
# normalized plane each fold point is pulled toward the centre by the relative margin 10⁻⁹ of the radius (what the
# refinement leaves, as for the disc) and by `dip`, a bound on how far the fold curve can bulge INWARD of the chord
# between two samples: in polar coordinates (φ, R(φ)) about the centre, a curve through two samples h apart lies within
# h²/8 · max|R''| of the interpolant through them, and a chord curves away from the centre (it only gains margin from
# its own curvature), so the polygon never claims a point past the fold. R'' is the polygon's own second difference at
# its non-uniform spacing, doubled as a safety factor (the fold curve of a polynomial distortion is a low harmonic of the
# azimuth, so its second derivative varies little over the 1.5° stencil), and read locally: each vertex is pulled by
# the bound of its own two sectors, so a kink costs margin only beside it. A jump of the fold radius between two
# samples (`_FOLD_JUMP`) is no curve the bound covers: the vertices beside it are capped at the nearer neighbour's
# fold, so no chord across it claims past that fold. A discontinuity narrower than one sample's spacing that both
# neighbours miss cannot be seen. Within a sample of a jump no vertex comes nearer the centre than the disc of the smallest
# fold (`_valid_domain_radius`, the domain until Core 0.10.9, inside every fold): there the bound can exceed the radius
# itself (elsewhere it is at most _FOLD_JUMP/2 of it), and the floor keeps every vertex radius positive, so the ring is star-shaped about the centre whenever the fold points'
# image directions run round it in order (the fold curve of a distortion that keeps its azimuth's order). The vertices
# then go through K (affine: the polygon of the mapped vertices is the mapped polygon). `PolygonRegion` still refuses a
# ring that is not star-shaped (a fold curve whose image directions do not run in order).
function _fold_polygon(camera::CameraModel)
    l = lens(camera)
    edge = _view_angle_edge(l)
    c = SVector{2, Float64}(project(l, SVector(0.0, 0.0, 1.0)))
    ψs = range(0, 2π; length = _FOLD_AZIMUTHS + 1)[1:_FOLD_AZIMUTHS]
    d = [_azimuth_fold(l, ψ, edge) - c for ψ in ψs]
    n = length(d)
    R = norm.(d)
    φ = [atan(v[2], v[1]) for v in d]
    h(i) = mod(φ[mod1(i + 1, n)] - φ[i], 2π)                         # the sector from sample i to i + 1
    D2(i) = (hm = h(mod1(i - 1, n)); hp = h(i);
        2 * ((R[mod1(i + 1, n)] - R[i]) / hp - (R[i] - R[mod1(i - 1, n)]) / hm) / (hp + hm))
    # vertex i bounds the bulge of its two sectors (i − 1 → i, i → i + 1), from the second differences at both ends of each
    dip(i) = 2 * maximum(j -> abs(D2(mod1(j, n))), (i - 1):(i + 1)) * max(h(mod1(i - 1, n)), h(i))^2 / 8
    # the floor beside a jump: the disc of the smallest fold (`_valid_domain_radius`, inside every azimuth's fold), never
    # past a vertex's own fold sample; a vertex the bulge bound would pull past it stops there
    ρ = _valid_domain_radius(l)
    disc(i) = min(ρ, R[i] * (1 - 1.0e-9))
    # beside a jump (a neighbour's radius more than _FOLD_JUMP away), no farther out than the nearer neighbour's fold;
    # within a sample of one (where a jump enters `dip`), never nearer the centre than the disc. Elsewhere neighbours
    # differ by at most _FOLD_JUMP·R, so `dip` ≤ 2·(2·_FOLD_JUMP·R/h)/(2h)·h²·2/8 = _FOLD_JUMP·R/2: no vertex nears the
    # centre, and the polygon is the bulge bound's alone (bit for bit, at a smooth fold curve's minimum too)
    jump(i) = any(j -> abs(R[mod1(j, n)] - R[mod1(i, n)]) > _FOLD_JUMP * R[mod1(i, n)], (i - 1, i + 1))
    near(i) = jump(i - 1) || jump(i) || jump(i + 1)
    cap(i) = jump(i) ? min(R[mod1(i - 1, n)], R[i], R[mod1(i + 1, n)]) * (1 - 1.0e-9) : Inf
    radius(i) = (r = min(R[i] * (1 - 1.0e-9) - dip(i), cap(i)); near(i) ? max(r, disc(i)) : r)
    pixel = affine_part(SMatrix{3, 3, Float64}(calibration_matrix(camera)))   # K's pixel affine
    vertices = [pixel(c + radius(i) / R[i] * d[i]) for i in 1:n]
    return PolygonRegion(vertices, pixel(c))
end

# The fold of the lens along the ray azimuth ψ, as a normalized image point: where the image radius
# `|project(lens, ray(θ, ψ))|` first stops being finite and strictly increasing in the view angle θ — the radial map's
# first fold on that azimuth, past which it is no longer invertible. The one monotonicity scan (`_last_increasing`,
# 1024 samples to `edge`), then the bracket refined: a fold by Brent's method on the radius about the last increasing
# sample (`_grid_brent_minimum`), an onset of unimaged rays bisected (`_last_imaged`); `edge` itself when the radius never
# stops increasing. An `ArgumentError` when the lens images no ray on the azimuth.
function _azimuth_fold(l::AbstractLensModel, ψ::Real, edge::Real)
    sψ, cψ = sincos(Float64(ψ))
    point(θ) = SVector{2, Float64}(project(l, SVector(sin(θ) * cψ, sin(θ) * sψ, cos(θ))))
    radius(θ) = (p = point(θ); all(isfinite, p) ? norm(p) : -Inf)
    θs = range(1.0e-4, Float64(edge); length = 1024)
    k = _last_increasing(radius, θs)
    k == 0 && throw(
        ArgumentError("valid_domain: the lens $(nameof(typeof(l))) images no ray on the azimuth ψ = $(ψ); its project is not finite near the axis")
    )
    k == length(θs) && return point(θs[k])
    θ = if isfinite(radius(θs[k + 1]))
        _grid_brent_minimum(θ -> -radius(θ), range(θs[max(k - 1, 1)], θs[k + 1]; length = 3)).x
    else
        _last_imaged(θ -> isfinite(radius(θ)), θs[k], θs[k + 1])
    end
    return point(θ)
end

# An ellipse in an ellipse: the largest value of `b`'s quadratic form `(p − c)ᵀ M⁻¹ (p − c)` over `a`'s boundary is at
# most 1 (with a relative slack of 8 eps for the form's rounding). The boundary is `a`'s centre plus its shape factor
# times the unit circle; the maximum over the angle is Core's one 1-D minimizer (`_grid_brent_minimum`, periodic).
function Base.issubset(a::Ellipse, b::Ellipse)
    Minv = inv(SMatrix{2, 2, Float64}(shape_matrix(b)))
    cb = SVector{2, Float64}(b.center[1], b.center[2])
    q(θ) = (d = SVector{2, Float64}(boundary_point(a, θ)) - cb; -dot(d, Minv * d))
    r = _grid_brent_minimum(q, range(0.0, 2π; length = 721)[1:(end - 1)]; periodic = true)
    return -r.value <= 1 + 8 * eps()
end

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
    "where it is extrapolated. Fit over a region inside the support (`imaged_region` of the camera bounded to it: ",
    "`with_lens_coverage(camera, support)` for a Rect or an Ellipse support, the region ∩ the support otherwise), or ",
    "calibrate the rest.")

"""
    check_region_support(camera, region; who::AbstractString) -> region
    check_region_support(support, region; who::AbstractString) -> region

THE guard of every fit over a camera's region (VisualGeometryToolkitEstimation's camera
regressions, their refine and the optical-axis fit): `region` back when it lies inside
the source's calibrated support — `valid_domain(camera)`, or a `support`
region given directly (a source camera's, when the camera fitted from is a
re-gridded copy whose own domain no longer records it) — and a
[`RegionBeyondSupport`](@ref) naming `who`, the region and the support
otherwise. Never a warning: a fit over an extrapolated model is refused.
Containment is `region ⊆ support`, Core's one region containment over a `Rect`,
an `Ellipse`, a `PolygonRegion` (a decentred lens's domain), their `∩` and the
`WholePlane`.
"""
check_region_support(camera::Union{CameraModel, PosedCamera, BoundedCamera, NoncentralSplineModel}, region; who::AbstractString) =
    check_region_support(valid_domain(camera), region; who)
function check_region_support(support, region; who::AbstractString)
    region ⊆ support || throw(RegionBeyondSupport(String(who), region, support))
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
  Brent's method between its neighbours (`_grid_brent_minimum`, Core's one 1-D
  minimizer) and returns the largest value it evaluated.
- `farthest_radius(region, center, direction)` is the largest `t` with
  `center + t·direction` in the region, bisected to rounding, in units of
  `‖direction‖`. A direction through a linear map, `A·d`, gives the radius in
  the coordinates that `A` maps to pixels.

The region is any bounded 2-D region that answers `p ∈ region`: a `Rect`, an
`Ellipse`, a `Circle`, a `PolygonRegion`, or their `∩`. It must contain `center`
and be star-shaped about it, which every intersection of convex parts containing
`center` is (and a polygon valid domain about its distortion centre). A `Rect`'s farthest point is a corner from any `center`, inside or
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
farthest_radius(r::Rect{2}, center) = maximum(p -> norm(p - _plain_center(center)), rect_corners(Rect2{Float64}(r)))
function farthest_radius(region, center)
    c = _plain_center(center)
    _check_farthest_center(region, c)
    reach(ψ) = -farthest_radius(region, c, SVector(cos(ψ), sin(ψ)))
    return -_azimuth_minimum(reach, _FARTHEST_AZIMUTHS)
end
function farthest_radius(region, center, direction)
    c = _plain_center(center)
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
_plain_center(center) = SVector{2, Float64}(center[1], center[2])
_check_farthest_center(region, center) = Point2{Float64}(_plain_center(center)) ∈ region ||
    throw(ArgumentError("farthest_radius: the region does not contain the centre $(Tuple(center))"))

"""
    pixel_mask(region, width, height) -> BitMatrix

The pixels of a `width × height` image whose CENTRES lie in `region` (any 2-D region answering `p ∈ region`: an
[`imaged_region`](@ref), a `Rect`, an [`Ellipse`](@ref), their `∩`, the
[`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane)), as a `height × width` `BitMatrix` in Julia's array
layout: `mask[v, u]` is pixel `(u, v)`, whose centre is the COLMAP point `(u − ½, v − ½)` — the conversion
[`pixel_centers`](@ref) shares, made in one place.

```julia
region = imaged_region(camera)                # the sensor ∩ its lens coverage ∩ where its lens model is valid
mask = pixel_mask(region, width, height)      # the pixels the camera images
```
"""
pixel_mask(region, width::Integer, height::Integer) =
    BitMatrix([_pixel_center(u, v) ∈ region for v in 1:Int(height), u in 1:Int(width)])

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
    BitMatrix([Point2{Float64}(frame_map(SVector{2, Float64}(_pixel_center(u, v)))) ∈ region for v in 1:Int(height), u in 1:Int(width)])

"""
    pixel_centers(region, us, vs) -> Vector{SVector{2, Float64}}

The COLMAP centres `(u − ½, v − ½)` of the pixels `(u, v)`, `u ∈ us`, `v ∈ vs`
(Julia's 1-based pixel indices, as [`pixel_mask`](@ref) indexes them), that lie
in `region`, `v` outer and `u` inner: the pixels of a sub-lattice (a stride, an
offset) without the full image's mask. `pixel_mask(region, W, H)[v, u]` is
whether pixel `(u, v)`'s centre is among `pixel_centers(region, 1:W, 1:H)`.
"""
pixel_centers(region, us::AbstractVector{<:Integer}, vs::AbstractVector{<:Integer}) =
    [SVector{2, Float64}(c) for v in vs for u in us for c in (_pixel_center(u, v),) if c ∈ region]

"""
    pixel_center([T = Float64,] I::CartesianIndex{2}) -> SVector{2, T}

The centre of the pixel at the array index `I = (row, col)` (Julia's 1-based
indices, `img[row, col]`) in the COLMAP frame, `(col − ½, row − ½)`, in `T`:
the one conversion from an image array's index to a pixel point. Allocation-
and branch-free, so a device functor over `CartesianIndices` calls it at the
kernel's element type.
"""
@inline pixel_center(::Type{T}, I::CartesianIndex{2}) where {T <: Real} = SVector{2, T}(T(I[2]) - T(0.5), T(I[1]) - T(0.5))
@inline pixel_center(I::CartesianIndex{2}) = pixel_center(Float64, I)
# Pixel `(u, v)` (column, row) as the regions read it.
_pixel_center(u::Integer, v::Integer) = Point2{Float64}(pixel_center(CartesianIndex(v, u)))

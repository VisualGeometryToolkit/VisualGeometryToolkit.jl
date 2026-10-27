# ==============================================================================
# Lens Models - Camera lens behavior (projection type + distortion)
# ==============================================================================



# ============================================================================
# Transform Builders - Dispatch on Lens Model
# ============================================================================

# The pixel-space affine of an intrinsics block is `AffineMap(K)`
# (transforms/conversions.jl); no second spelling here.
@inline _pixel_affine(intrinsics::AbstractIntrinsics) = AffineMap(intrinsics.K)

"""
    imaging_transform(intrinsics, lens) -> ComposedTransformation
    imaging_transform(model::CameraModel)

The composed camera-space → pixel-space map as a value:
`AffineMap(K) ∘ lens`, the lens applied to the ray (its functor is
[`project`](@ref) on 3-vectors). It is `project(model, X)` for every lens and
every ray — past 90° for a lens that images there, NaN where the lens does not
(for a `Pinhole`, every `Z ≤ 0`). Its derivative is the chain rule through
`transform_deriv(lens, X)`. The fully-typed, junction-checked counterpart is
[`imaging_chain`](@ref), which mints the framed composite
`{ViewFrame{id} → ImageFrame{id}}` through the frame chain documented at
`DistortedFrame`.
"""
imaging_transform(intrinsics::AbstractIntrinsics, lens::AbstractLensModel) =
    _pixel_affine(intrinsics) ∘ lens

# =============================================================================
# Lens contract — inverse distortion + Jacobian + capability traits
# =============================================================================

"""
    InverseLens{L} <: CoordinateTransformations.Transformation

The back-projection of a lens as a first-class map: `inv(lens)` returns one,
`inv(inv(lens)) === lens`, its call is `undistort_normalized(lens, xd)`, and
its `transform_deriv` is ForwardDiff through that map (lens_rays.jl).
Composes like any transformation, so the pixel-domain
undistortion of a camera is `inv(distortion_map(model))` with the chain rule
for its derivative. A `Pinhole` is its own inverse.
"""
struct InverseLens{L <: AbstractLensModel} <: CoordinateTransformations.Transformation
    lens::L
end
Base.inv(lens::AbstractLensModel) = InverseLens(lens)
Base.inv(m::InverseLens) = m.lens
@inline (m::InverseLens)(xd::StaticVector{2}) = undistort_normalized(m.lens, xd)
Base.show(io::IO, m::InverseLens) = print(io, "inv(", m.lens, ")")

"""
    has_distortion(lens) -> Bool

Does the lens bend rays at all? (`false` ⇒ `undistort_normalized` is the
identity.) Relation to the framed imaging chain (0.10):
`has_distortion(lens) == false` means the distortion is the identity in the
lens's defining frame, so `imaging_chain` COLLAPSES the ray frame onto
`DistortedFrame` — the chain has no distortion stage and `K` consumes the
ray frame directly (see the collapse rule at `DistortedFrame` and
`distortion_frame`). Every frame here assumes a single center of projection.
"""
has_distortion(::AbstractLensModel) = true
has_distortion(::Pinhole) = false

# =============================================================================
# Pupil trait — exit-pupil geometry of a lens model
# =============================================================================

"""
    monotone_radius(lens) -> Real

The distorted radius up to which the lens's perspective view is a monotone
bijection — the bound a whole-extent perspective undistortion clips to
([`undistorted_rect`](@ref), [`valid_rect`](@ref), [`division_bounded`](@ref)).
Beyond it the view folds or diverges and [`project`](@ref) answers NaN. The
profile's first fold or pole, whichever comes first, mapped to the distorted
radius; both are found exactly, as first positive polynomial roots. `Inf` for a
lens whose profile never folds.

- the radial factor ([`radial`](@ref)) of a forward lens: the fold or pole of
  its profile `r ↦ r·F(r²)` in the projection's frame, capped at the frame's
  edge (a ray cannot leave the sphere); decentering terms do not enter, as in
  [`view_angle_limit`](@ref).
- a [`LiftLens`](@ref): the first pole of `D` (the 90° circle) or fold of the
  lift — `1/√|k|` for `Division(k)`.
- a [`Chain`](@ref) distortion: its FIRST radial stage (a `Radial`, or an
  [`Additive`](@ref) led by one), wherever it sits — the stage [`project`](@ref)
  tests — read as above at that stage's output; the stages around it do not
  enter (for a stage behind a tilt, the frame's edge caps the stage's input).
"""
# Not `image_circle`'s ρ: the perspective-undistortion bound of the radial factor alone (capped at the frame edge, Inf without a fold), a normalized distorted radius; it equals ρ only for a radially symmetric folding lens.
function monotone_radius(l::AbstractLensModel)
    lf = _precision(Float64, l)
    r = _bounding_radial(lf)
    r === nothing && return Inf
    u = _branch_end(r, lf isa Lens ? _frame_cap(lf.projection) : Inf)
    T = eltype(parameter_vector(l))                  # the lens's coefficient precision
    isfinite(u) || return T(Inf)
    return T(lf isa Lens ? sqrt(u) * radial_ratio(r, u) : _radius_at(r, u))
end

# The radial factor that bounds what a lens images: `radial(lens)`, or for a
# Chain its first radial stage's factor (`radial` stays `nothing` for a Chain:
# `radial_part` and the fit read it as the lens's ONE radially symmetric factor).
_bounding_radial(l) = radial(l)
_bounding_radial(l::Lens{<:AbstractProjection, <:Chain}) = _first_radial_factor(l.distortion.stages)
_first_radial_factor(::Tuple{}) = nothing
_first_radial_factor(ds::Tuple) = _is_radial(first(ds)) ? _radial_factor(first(ds)) : _first_radial_factor(Base.tail(ds))
_radial_factor(r::Radial) = r
_radial_factor(a::Additive) = _head(a)

# ==============================================================================
# View-angle limits — the honest θ-domain of each lens (value-dependent) — and
# the radially symmetric part of a lens, what a regression onto a radially
# symmetric family (VisualGeometryToolkitEstimation's `fit_lens`) can represent.
# ==============================================================================

# The perspective plane's edge and the sphere's, each less the same margin: the
# largest view angles a perspective family and any family can honestly report.
const _PERSPECTIVE_EDGE = π / 2 - 1.0e-3
const _SPHERE_EDGE = π - 1.0e-3


"""
    view_angle_limit(lens; cap) -> θ_max

The largest view angle the lens's radial profile honestly represents: where
the profile stops being finite and strictly increasing (a fold, a pole, or
the frame's own edge). Closed forms where the lens has one — the fold or pole
of a forward lens's radial factor read through its frame, never past the
frame's edge (`π/2 − 10⁻³` for the perspective frame); a lift's fold; the
double sphere's FOV bound — and a monotonicity scan of
[`projected_radius`](@ref) otherwise. A [`Chain`](@ref) with a radial stage
is bounded by that stage, tested at its own input as [`project`](@ref) tests
it: the angle where `project` first turns NaN, bisected to rounding. When a
stage precedes the radial one (a tilt) that angle depends on the azimuth, and
the limit is its minimum over azimuths, so every ray within it is imaged; a
lens whose bound is radially symmetric reads it at azimuth 0. `cap` bounds the answer: the generic
scan runs to it (default `π − 10⁻³`); the closed forms clamp to it (default
none).
"""
function view_angle_limit(lens::AbstractLensModel; cap::Real = _SPHERE_EDGE)
    θs = range(1.0e-4, float(cap); length = 1024)
    rprev = 0.0
    θok = 1.0e-4
    for θ in θs
        r = try
            Float64(projected_radius(lens, θ))
        catch
            break
        end
        (isfinite(r) && r > rprev) || break
        rprev = r
        θok = θ
    end
    return θok
end
# A forward lens with a fixed projection: its radial factor's first fold or pole,
# read through the frame's radial profile (`_frame_backproject` at azimuth 0) —
# exact, not scanned — never past the frame's own edge. Decentering terms do not
# enter (they contaminate an on-axis sample). Without a radial factor: the scan.
function view_angle_limit(l::Lens; cap::Real = Inf)
    scan(c) = invoke(view_angle_limit, Tuple{AbstractLensModel}, l; cap = c)
    C = projection_frame(l.projection)
    C === nothing && return scan(min(float(cap), _SPHERE_EDGE))
    edge = min(float(cap), _view_angle_edge(l))
    r = radial(_precision(Float64, l))
    l.distortion isa Chain && return _chain_view_angle_limit(l, edge)
    r === nothing && return l.distortion === nothing ? edge : scan(edge)
    u = _branch_end(r, _frame_cap(l.projection))
    isfinite(u) || return edge
    b = _frame_backproject(C, SVector(sqrt(u), 0.0))
    return min(atan(b[1], b[3]) * (1 - 1.0e-9), edge)
end
# The edge `view_angle_limit` stops at when the lens has no limit of its own: the
# frame's edge less a margin (`π/2 − 10⁻³` for the perspective and orthographic
# frames), the sphere's (`π − 10⁻³`) for any other lens — the generic scan's
# default cap. A limit below it is the lens's own (a fold, a pole, a bound).
_view_angle_edge(::AbstractLensModel) = _SPHERE_EDGE
function _view_angle_edge(l::Lens)
    C = projection_frame(l.projection)
    return C !== nothing && C <: Union{GnomonicFrame, OrthographicFrame} ? _PERSPECTIVE_EDGE : _SPHERE_EDGE
end
# A Chain with a radial stage: the stage is tested at its own input, so the angle
# where `project` first turns NaN has no closed form here. Along one azimuth it
# is the first NaN of a scan, bisected to rounding — the last imaged angle,
# `edge` when none is NaN. As for an Additive's radial head, the other terms do
# not enter: the radius may peak a little before the radial stage's fold. When
# the radial stage leads, its input is the frame point and the onset is the same
# at every azimuth (azimuth 0 is read). When a stage comes before it (a tilt),
# the onset depends on the azimuth, and the limit is its MINIMUM over azimuths
# — the angle within which every ray is imaged: 36 azimuths, the least refined
# by a golden-section search between its neighbours. A Chain without a radial
# stage: the plain scan.
function _chain_view_angle_limit(l::Lens, edge)
    _bounding_radial(_precision(Float64, l)) === nothing &&
        return invoke(view_angle_limit, Tuple{AbstractLensModel}, l; cap = edge)
    _is_radial(first(l.distortion.stages)) &&
        return _nan_onset(θ -> isfinite(projected_radius(l, θ)), edge)
    onset(ψ) = _nan_onset(θ -> all(isfinite, project(l, SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ)))), edge)
    return _azimuth_minimum(onset, 36)
end
# The minimum of a function of the azimuth ψ: `n` samples around the circle, the
# least refined by a golden-section search between its neighbours (`iters`
# steps); the smallest value evaluated. The one azimuthal minimizer: the chain's
# view-angle limit above and the image circle's inscribed radius
# (image_circle.jl).
function _azimuth_minimum(f, n::Int; iters::Int = 40)
    ψs = range(0, 2π; length = n + 1)[1:n]
    vs = map(f, ψs)
    k = argmin(vs)
    a, b = ψs[k] - 2π / n, ψs[k] + 2π / n
    g = (sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:iters
        if fc < fd
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        end
    end
    return min(vs[k], fc, fd)
end
# The last angle at which `imaged` holds, scanning [10⁻⁴, edge] and bisecting the
# first failure to rounding; `edge` when it never fails.
function _nan_onset(imaged, edge)
    θs = range(1.0e-4, float(edge); length = 1024)
    k = findfirst(!imaged, θs)
    k === nothing && return float(edge)
    k == 1 && return first(θs)
    lo, hi = θs[k - 1], θs[k]
    for _ in 1:60
        m = (lo + hi) / 2
        imaged(m) ? (lo = m) : (hi = m)
    end
    return lo
end
view_angle_limit(l::Lens{<:DoubleSphere, Nothing}; cap::Real = Inf) =
    min(acos(clamp(-_ds_w2(l.projection), -1.0, 1.0)) - 1.0e-6, float(cap))
# The radius where a lift first folds (its view angle stops increasing), ignoring
# the 90° pole; Inf when it never folds — at Float64, in the coefficient type.
_fold_radius(l::LiftLens) = (r = _precision(Float64, l).radial; _coeff_type(l)(_radius_at(r, _fold_point(_fold_coeffs(r), Inf))))
# A lift: the view angle just inside its first fold, or far out along one that never folds.
function view_angle_limit(l::LiftLens; cap::Real = Inf)
    ρ = _fold_radius(l)
    return min(Float64(view_angle(l, isfinite(ρ) ? ρ * (1 - 1.0e-9) : 1.0e6)), float(cap))
end

"""
    radial_part(lens) -> AbstractLensModel

The radially symmetric part of `lens`: its projection with only its radial
factor ([`radial`](@ref)) — the decentering terms dropped; a lift is its own
radial part. What `view_angle_limit` bounds and what a regression onto a
radially symmetric family can represent.
"""
radial_part(l::Lens) = (r = radial(l); r === nothing ? Lens(l.projection) : Lens(l.projection, r))
radial_part(l::LiftLens) = l

# The normalized radius of the disc a lens images: past it `backproject` returns
# NaN. Distinct from `monotone_radius` — a lens with no fold may still image only
# a bounded disc (an equidistant fisheye stops at θ = π). A disc for a pure
# projection; for a distorted lens, whose `backproject` inverts numerically, the
# smaller of its extents along the two image axes. `Inf` for a lens that images
# the whole plane.
# Not `image_circle`: where `backproject` turns NaN (π for an undistorted equidistant lens, to which `image_circle` answers the whole plane).
_imaging_domain_radius(l::AbstractLensModel) =
    l isa LiftLens ? Inf : l.distortion === nothing ? _frame_domain(l.projection) : _bisected_domain_radius(l)
_frame_domain(p::AbstractProjection) =
    (C = projection_frame(p); C === nothing ? _bisected_domain_radius(Lens(p)) : _frame_radius(C))
# The same edge in the squared radius, for a fixed projection: a ray cannot leave the sphere.
_frame_cap(p::AbstractProjection) = (C = projection_frame(p); C === nothing ? Inf : float(_frame_radius(C))^2)
# The double sphere: the smaller of the unprojection's radius bound
# r² ≤ 1/(2α − 1) (α > 1/2) and the image of the FOV-limiting ray z = −w₂ on the
# unit sphere, √(1 − w₂²)/(α·d₂ + (1 − α)·z_s) with z_s = ξ − w₂ and
# d₂ = √(1 − w₂² + z_s²) — unbounded when that denominator is not positive (the
# pinhole ξ = α = 0 reaches z = 0). For ξ > 1, outside the paper's range, the
# true edge lies slightly beyond this bound, which stays inside it.
function _frame_domain(p::DoubleSphere)
    ξ, α = Float64(p.xi), Float64(p.alpha)
    r_unproject = α > 0.5 ? 1 / sqrt(2α - 1) : Inf
    w2 = Float64(_ds_w2(p))
    zs = ξ - w2
    den = α * sqrt(1 - w2^2 + zs^2) + (1 - α) * zs
    r_fov = den > 0 ? sqrt(1 - w2^2) / den : Inf
    return min(r_unproject, r_fov)
end

# The imaging-domain radius of a lens whose `backproject` is a numerical inverse
# (Newton on the forward map), with no closed-form edge: bisection along the two
# image axes for the radius where the inverse stops returning a finite ray;
# `Inf` when it is finite out to 100 focal lengths.
function _bisected_domain_radius(lens::AbstractLensModel)
    finite_at(e, ρ) = all(isfinite, backproject(lens, ρ * e))
    r = Inf
    for e in (SVector(1.0, 0.0), SVector(0.0, 1.0))
        hi = 100.0
        finite_at(e, hi) && continue
        lo = 0.0
        for _ in 1:60
            mid = (lo + hi) / 2
            finite_at(e, mid) ? (lo = mid) : (hi = mid)
        end
        r = min(r, lo)
    end
    return r
end

Base.inv(::Pinhole) = Pinhole()

# The perspective division's Jacobian, ∂[x/z, y/z]/∂[x, y, z] — the one leaf
# CoordinateTransformations leaves undefined. With it the chain rule it
# already applies through composed maps turns
# `transform_deriv(imaging_transform(K, lens), X_cam)` into the full
# camera-space → pixel Jacobian `A · ∂lens · Jπ` of ANY lens, in one call —
# the derivative half of the warping paradigm (a projection is a composition
# of Transformations; its local affinity is the product of theirs). Extending
# a CoordinateTransformations verb on a CoordinateTransformations type is
# deliberate glue, allowlisted in quality_tests.jl beside AffineMap/LinearMap.
@inline function transform_deriv(::PerspectiveMap, X::StaticVector{3})
    T = float(eltype(X))
    iz = one(T) / T(X[3])
    iz2 = iz * iz
    return SMatrix{2, 3, T, 6}(iz, zero(T), zero(T), iz, -T(X[1]) * iz2, -T(X[2]) * iz2)
end

# The functor is the lens's projection, on either representative of a ray: a
# 3-vector X, or a perspective point x standing for the ray (x, 1). So a lens
# composes after anything that produces rays — `AffineMap(K) ∘ lens` is the
# camera, `lens ∘ HomographyMap` a warp that never divides by Z.
@inline (lens::AbstractLensModel)(X::StaticVector{3}) = project(lens, X)
@inline (lens::AbstractLensModel)(x::StaticVector{2}) = distort_normalized(lens, x)

# In an image_warp chain a lens after a homography consumes its homogeneous
# output as a ray: the lens is that chain's division.
_has_division(::AbstractLensModel) = true

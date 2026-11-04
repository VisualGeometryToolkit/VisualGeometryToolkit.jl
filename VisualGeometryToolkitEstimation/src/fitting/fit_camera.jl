# ==============================================================================
# fit_camera — one model-to-model regression: a camera model of type T regressed
# onto a source camera over a region of its sensor
# ==============================================================================
#
# THE OBJECTIVE is the integrated SQUARED pixel error of the round trip over the
# region, ∫ ‖π_T(π⁻¹_source(u)) − u‖² du, by a quadrature over the region. The
# target TYPE selects the family:
#
#   a lens family L          the fitted lens's radial profile against the source's,
#                            the exact 1-D reduction of the integral (below), the
#                            family's parameters by the stack's chart LM
#   SplineProjection         the forward map ray → pixel, a ChartSpline on the
#                            equidistant direction chart about the region centre's
#                            ray, by penalized linear least squares; its knot
#                            spacing bisected to the largest one whose held-out
#                            maximum error meets `tolerance`
#   CentralGenericModel      the grid of Core's full-image layout, the source's
#                            rays by penalized linear least squares; its lattice
#                            spacing searched as the SplineProjection's knots are
#   NoncentralGenericModel   the same lattice, the source's LINES: the directions
#                            and each line's point closest to the lines' common
#                            nearest point c*, by the same penalized solve
#
# THE PENALTY of the spline families is a P-spline's: λ(‖Dₓ³c‖² + ‖Dᵧ³c‖²), third
# differences of the controls along both lattice axes. Every node is free; the
# penalty holds the nodes the data determine weakly (outside the region, at its
# edge) to their neighbours' curvature. λ is a fixed relative rule, not
# searched: the data are noise-free (the source model), so a smaller λ is better
# until the normal equations lose conditioning — measured 2026-11-03 on the
# FX3 + Sigma 8 mm grid (`λ = 1e-6` absolute held the edge to the interior's
# accuracy at every lattice size; a larger λ only lost accuracy), here as
# `1e-8 · mean(diag(AᵀA))`.
#
# THE HELD-OUT CHECK of the spline families is the quadrature offset by half its
# stride over the region, and its edge band: the held-out points within
# `_EDGE_BAND_PX` of the region's edge.

# ------------------------------------------------------------------------------
# The metadata
# ------------------------------------------------------------------------------

"""
    CameraFitMetadata{R, F}

`FitResult` metadata of [`fit_camera`](@ref): the `region` fitted over, the
`weight` measure (`:image` for every target but a lens family), the round
trip's error in pixels — `max_px` (its maximum) and `max_px_edge` (its maximum
within 100 px of the region's edge); √ of its mean square, the objective's
quantity, is the result's `rms` — and the target family's own record `family`,
a `NamedTuple`:

- a lens family: `(; focal_scale, θ_max, θ_source, clipped)` — the joint focal
  rescale `s`, the fitted and the source's angular domains (radians), and
  whether the target family cut the source's domain; the errors are at the
  quadrature's nodes of the radial profile;
- `SplineProjection`: `(; spacing, penalty, counts, stride, tolerance,
  outer_iterations)` — the knot spacing `h` (radians in the chart), the
  P-spline weight `λ`, the control counts per chart axis, the quadrature's
  stride in pixels, the tolerance asked for and the spacing search's fits; the
  errors are the held-out check's;
- `CentralGenericModel`: `(; penalty, grid_size, stride, tolerance,
  outer_iterations, n_held_out_dropped)`, `grid_size` the chosen lattice's
  `(rows, columns)`; the errors are the held-out direction error, as the
  non-central target's (below), over the held-out pixels where the source's
  back-projection Jacobian is finite — `n_held_out_dropped` counts the others;
- `NoncentralGenericModel`: `(; penalty, grid_size, stride, tolerance,
  outer_iterations, center, max_offset, rms_offset, max_offset_edge,
  n_held_out_dropped)` —
  `center` the anchor c* (the
  least-squares point nearest to the source's quadrature lines, in the
  source's frame, plain metres as the grid's points), and the held-out line
  offset `‖(I − ddᵀ)(o_fit − o_src)‖` (plain metres); the pixel errors are the
  held-out DIRECTION error, mapped to pixels through the source's
  back-projection Jacobian.
"""
struct CameraFitMetadata{R, F <: NamedTuple}
    region::R
    weight::Symbol
    max_px::Float64
    max_px_edge::Float64
    family::F
end

const _EDGE_BAND_PX = 100.0
# The P-spline penalty's relative weight (see the file header).
const _PSPLINE_PENALTY = 1.0e-8
# The radial profile's quadrature order (n + 1 Clenshaw–Curtis nodes) and the
# azimuths the region's coverage of a radius is counted over.
const _PROFILE_NODES = 128
const _REGION_AZIMUTHS = 720

"""
    fit_camera(::Type{T}, source, region; …) -> FitResult{<model of T>, CameraFitMetadata}

The camera model of type `T` regressed onto `source` over `region`, a region of
the source's sensor in its pixel frame (an [`imaged_region`](@ref
VisualGeometryToolkitCore.imaged_region), an `Ellipse`, a `Rect`, their `∩`;
bounded): the minimizer of the integrated squared pixel error of the round trip,
`∫ ‖π_T(π⁻¹_source(u)) − u‖² du` over the region. The target type selects the
family and the solve:

- **a lens family `L <: AbstractLensModel`** (`fit_camera(L, source::CameraModel,
  region; weight = :image)`): a `CameraModel` with lens `L` — `K` scaled by a
  jointly fitted focal rescale `s`, the principal point kept — fitted to the
  source lens's radial profile `r(θ)` (its radial part: decentering terms are
  stripped, warned when non-negligible). The integral over the region reduces
  exactly to one over `θ`, with the density weighted by the fraction of each
  radius's circle about the principal point that lies in the region (the region
  must be star-shaped about the principal point, as a convex one containing it
  is), evaluated by Clenshaw–Curtis quadrature, so the discrete objective IS the
  integral, and minimized by the stack's chart LM (at most `max_iter` accepted
  iterations; configs/default.toml `[fit_camera]`). The domain is the region's
  farthest radius, capped by the source's and the target family's own view-angle
  limits — a cut of the source's domain is `clipped` and warned, and so is a
  region reaching past the radii the source images (its domain is then the
  source's own). (The region's
  extent is measured along 720 azimuths: a corner between two of them is reached
  to within half an azimuth's arc.) `weight` is the
  density: `:image` (uniform on the image), `:solid_angle` (`sin θ`) or
  `:uniform` (in `θ`). A forward target moves its projection's coefficients and
  its radial factor's numerator (its denominator and decentering terms stay
  zero); a lift moves every coefficient. `Division` alone fits `Division{1}`;
  any other type must carry its arities (`KannalaBrandt{4}`).
- **`SplineProjection`** (`fit_camera(SplineProjection, source, region;
  tolerance)`): the forward map ray → pixel as a [`ChartSpline`](@ref
  VisualGeometryToolkitCore.ChartSpline) on the equidistant direction chart about the
  source's ray at the region's centroid, fitted to the source's pixels by
  linear least squares with a P-spline penalty. THE SPACING SEARCH, shared by
  the spline and grid targets: the knot spacing `h` is doubled down from 4
  cells along the longer side and then bisected to the LARGEST spacing whose
  held-out maximum error is at most `tolerance` (px), `retcode :converged`; a
  fit that cannot reach it at the finest spacing of the bracket (a cell of three
  quadrature strides) returns `retcode :max_iterations` with what it reached.
  Each fit of the search is an outer iteration, counted in the metadata's
  `outer_iterations`; `iterations(result)` is 0.
- **`CentralGenericModel`** (`fit_camera(CentralGenericModel, source, region;
  tolerance)`): a grid model in Core's full-image layout — the lattice the
  constructor `CentralGenericModel(source, size; grid_size)` lays over the whole
  image, [`generic_lattice`](@ref VisualGeometryToolkitCore.generic_lattice)
  (the source is not sampled: the solve sets every control) — with `domain` the
  region, fitted to the source's rays by linear least squares with the P-spline
  penalty, its lattice spacing chosen by the spacing search (`grid_size` is not
  an input: the tolerance chooses it; the metadata records it). The region
  restricts the quadrature only, never the lattice: a node whose support misses
  it is held by the penalty. The tolerance bounds the held-out DIRECTION error in
  pixels, the non-central target's metric (below). The image is the source's: a grid model's own, or a
  [`BoundedCamera`](@ref VisualGeometryToolkitCore.BoundedCamera)'s extent (a
  bare `CameraModel` has none: an `ArgumentError`).
- **`NoncentralGenericModel`** (`fit_camera(NoncentralGenericModel, source,
  region; tolerance)`): the same design and search on the source's observation
  LINES — a non-central grid source's `Ray3`s, or a central source's rays
  through its centre. Both grids, the unit directions and the points, come
  from ONE penalized solve on the same lattice. A line's point is pinned by
  an anchor: c*, Core's [`nearest_point`](@ref
  VisualGeometryToolkitCore.nearest_point) of all the quadrature lines, and each
  line contributes its point closest to c*, `o + ((c* − o)·d) d` — bounded for
  rays at and past 90°, where a plane anchor (the pupil plane) runs off. The
  points are in the source's frame (frame in = frame out; plain metres, the
  grid's unit), c* in the metadata; a central source's are all c*. The
  tolerance bounds the held-out DIRECTION error in pixels: the direction's
  deviation mapped through the source's back-projection Jacobian (the round
  trip through the source to first order); the line offset is reported beside
  it, in metres. A non-central source is a `NoncentralGenericModel`, which has
  its own image.

`source` is a `CameraModel` (bounded or not) or a `CentralGenericModel` (anything
with a batched `backproject(source, pixels)`; for `SplineProjection` also a
function of the pixel returning its ray or `nothing`, as Core's generic models
take one), and for the non-central target
also a `NoncentralGenericModel`. The spline families' quadrature is on the
image (they take no `weight`). Their per-point steps (the source's
backprojection, the chart, the design rows, the evaluation, the errors) run as
batches — Core's KernelAbstractions kernels and [`pointwise!`](@ref
VisualGeometryToolkitCore.pointwise!) — on the host, and the penalized normal
equations are solved by a sparse Cholesky factorization.

The metadata ([`CameraFitMetadata`](@ref)) records the region, the round trip's
`max_px` and `max_px_edge`, and the family's own record; the result's `rms` is
the round trip's root mean square in pixels.

```julia
region = imaged_region(camera, image_circle(camera, 8.0mm))
kb = fit_camera(KannalaBrandt{4}, model(camera), region)                       # a parametric lens
fwd = fit_camera(SplineProjection, grid, region; tolerance = 1.0e-3)           # a grid's forward map
project(estimate(fwd), ray)                                                    # its pixel, or nothing outside the region
```
"""
function fit_camera end

# ==============================================================================
# A lens family: the exact 1-D reduction on the radial profile
# ==============================================================================

function fit_camera(
        ::Type{L}, source::CameraModel, region;
        weight::Symbol = :image, max_iter::Int = _dflt("fit_camera", "max_iter")
    ) where {L <: AbstractLensModel}
    src = lens(source)
    dec = _decentering(src)
    dec > 1.0e-8 &&
        @warn "fit_camera: the source $(nameof(typeof(src))) carries non-radial terms (decentering, norm $(round(dec; sigdigits = 3))) — no radially symmetric target can represent them; fitting the radial part only"
    srcr = radial_part(src)
    K0 = SMatrix{3, 3, Float64}(calibration_matrix(source))
    fx = K0[1, 1]

    # The region's extent about the principal point, per azimuth of the
    # normalized plane: the radius at which each ray from the principal point
    # leaves it. Its maximum bounds the domain; the fraction of azimuths still
    # inside at a radius is that radius's share of the image-area element.
    exits = _exit_radii(region, K0)
    ρmax = maximum(exits)
    # past the source's fold (its profile's largest radius) the region's farthest
    # radius has no view angle: the domain is the source's own, and the region is cut
    θ_region = Float64(view_angle(srcr, ρmax))
    past_source = !isfinite(θ_region)
    θ_source = past_source ? Float64(view_angle_limit(srcr)) : min(Float64(view_angle_limit(srcr)), θ_region)
    past_source && @warn "fit_camera: the region reaches past the radii the source $(nameof(typeof(src))) images — fitting θ ∈ [0, $(round(rad2deg(θ_source); digits = 1))°], the source's own domain"
    θ_max = min(θ_source, _family_limit(L))
    θ_max > 0.02 || throw(ArgumentError("fit_camera: empty fit domain (θ_max = $θ_max)"))
    clipped = past_source || θ_max < θ_source * (1 - 1.0e-6)
    θ_max < θ_source * (1 - 1.0e-6) && @warn "fit_camera: target family $(L) cannot represent the source's field — fitting θ ∈ [0, $(round(rad2deg(θ_max); digits = 1))°] of the source's $(round(rad2deg(θ_source); digits = 1))°"

    θs, qs, rA, dens = _forward_samples(srcr, float(θ_max), _PROFILE_NODES, weight)
    dens = dens .* map(r -> _coverage(exits, r), rA)
    sw = sqrt.(qs .* dens)
    wsum = sum(qs .* dens)

    P = _regression_manifold(_target_manifold(L))
    lm = _LensRegressionLm(P, L <: LiftLens, θs, rA, sw)
    p, code, iters = _regress_lens(lm, length(θs), Val(tangent_size(P)), max_iter)

    s, lensB = _scale_and_lens(P, p)
    Kb = CameraCalibrationMat(K0 * Diagonal(SVector(s, s, 1.0)); check = false)   # focal and skew by s
    out = CameraModel(LogicalIntrinsics(Kb), lensB)
    θcap = L <: LiftLens ? view_angle_limit(lensB) - 1.0e-9 : Inf
    Δ = [s * Float64(projected_radius(lensB, min(θs[i], θcap))) - rA[i] for i in eachindex(θs)]
    rms_px = sqrt(sum(qs .* dens .* abs2.(Δ)) / max(wsum, eps())) * fx
    max_px = maximum(abs.(Δ)) * fx
    edge = rA .>= rA[end] - _EDGE_BAND_PX / fx
    max_px_edge = maximum(abs.(Δ[edge])) * fx
    return FitResult(
        out, code, iters, rms_px, length(θs),
        CameraFitMetadata(region, weight, max_px, max_px_edge, (; focal_scale = s, θ_max = float(θ_max), θ_source, clipped))
    )
end

# `Division` alone names the family, not a degree: fit the one-coefficient model,
# as `Division(k)` constructs it. Any other type must carry its arities (`zero(L)`
# refuses one that does not).
fit_camera(::Type{Division}, source::CameraModel, region; kwargs...) = fit_camera(Division{1}, source, region; kwargs...)

# ------------------------------------------------------------------------------
# The radial profile's regression
# ------------------------------------------------------------------------------
#
# Every central radially symmetric lens is characterized by its radial
# projection function r(θ) (`projected_radius`), so any source that can
# EVALUATE r_A(θ) regresses onto any target whose parameters can be FIT. The
# objective is the continuous L² functional
#
#     min_{p,s} ∫₀^θmax w(θ) · (s·r_B(θ; p) − r_A(θ))² dθ
#
# by Clenshaw–Curtis quadrature on Chebyshev extrema nodes: the rows are
# e_i = √(q_i·w(θ_i))·Δr(θ_i) with q_i the CC weights, so Σe_i² IS the
# quadrature of the integral and the LM on e minimizes the integral itself.
# The nodes cluster at both endpoints, where profiles diverge worst;
# Clenshaw–Curtis (not Gauss–Chebyshev, whose native weight is 1/√(1−x²))
# carries the weights of the plain dθ integral and is exponentially accurate
# on smooth profiles. The region enters as w's factor `_coverage` (above).
#
# NON-RADIAL TERMS: tangential/thin-prism decentering has no radial
# representation in any symmetric target. A SOURCE contributes its radial part
# (decentering stripped, warned when non-negligible); a forward TARGET moves its
# projection's coefficients and its radial factor's numerator, its denominator
# and decentering terms held at zero; a lift TARGET moves every coefficient.

# Clenshaw–Curtis on [0, b]: n+1 Chebyshev extrema nodes + weights for ∫₀ᵇ f dθ.
function _clenshaw_curtis(n::Int, b::Float64)
    θ = [b / 2 * (1 - cos(π * i / n)) for i in 0:n]
    q = Vector{Float64}(undef, n + 1)
    for i in 0:n
        s = 1.0
        for k in 1:fld(n, 2)
            bk = (2k == n) ? 1.0 : 2.0
            s -= bk * cos(2k * π * i / n) / (4k * k - 1)
        end
        q[i + 1] = ((i == 0 || i == n) ? 1.0 : 2.0) * s / n
    end
    q .*= b / 2
    return θ, q
end

# The size of what VGC's `radial_part` strips from a lens (its tangential,
# thin-prism and other non-radial terms; zero for a radially symmetric lens).
_decentering(l::AbstractLensModel) = sqrt(max(sum(abs2, parameter_vector(l)) - sum(abs2, parameter_vector(radial_part(l))), 0.0))

# The target's anchor: its zero lens (the double sphere at ξ = α = ½, inside
# the logit's interval), and the coefficients the fit moves — a forward lens's
# projection coefficients and radial numerator, a lift's every coefficient —
# as the family's manifold (`LensManifold`, Dual-safe) restricted to them.
_target_anchor(::Type{L}) where {L} = zero(L)
_target_anchor(::Type{L}) where {L <: Lens{<:DoubleSphere}} = (z = zero(L); Lens(DoubleSphere(0.5, 0.5), z.distortion))
function _target_manifold(::Type{L}) where {L <: AbstractLensModel}
    a = _target_anchor(L)
    ch = LensManifold(a)
    a isa LiftLens && return ch
    r = radial(a)
    proj = VgtkCore.Cameras.coefficient_names(typeof(a.projection))
    free = (proj..., (r === nothing ? () : ntuple(i -> Symbol(:k, i), Val(length(r.k))))...)
    length(free) == tangent_size(ch) && return ch
    isempty(free) && return restrict(ch, SVector{tangent_size(ch), Bool}(ntuple(_ -> false, Val(tangent_size(ch)))))
    return restrict(ch, free)
end
# The fit's chart: the focal's log-scale (a `EuclideanManifold` at 0, `s = exp`)
# before the lens family's manifold.
_regression_manifold(ch::Manifold) = ProductManifold(EuclideanManifold(SVector(0.0)), ch)
# The focal scale and the lens at chart coordinates `pv`. Dual-safe.
function _scale_and_lens(P::ProductManifold, pv)
    parts = retract(P, pv)
    return exp(first(parts)[1]), last(parts)
end

# The representable limit of a target FAMILY, before its parameters exist: a
# family whose domain its parameters do not move (the perspective plane, the
# fixed fisheye projections) answers VGC's own `view_angle_limit` (the one owner
# of the edge and its margin); every other family the sphere's.
_family_limit(::Type{<:Lens{F}}) where {F <: Union{Perspective, Equidistant, Stereographic, Equisolid, Orthographic}} =
    view_angle_limit(Lens(F()))
_family_limit(::Type) = view_angle_limit(Lens(Equidistant()))

# Clenshaw–Curtis nodes on [0, b] of the view angle, the source's radii there
# and the density w (`:image`: r·|dr/dθ|, the radius differentiated exactly,
# ForwardDiff through the source's `projected_radius`).
function _forward_samples(srcr, b::Float64, n::Int, weight::Symbol)
    radius(θ) = Float64(projected_radius(srcr, θ))
    ts, qs = _clenshaw_curtis(n, b)
    rA = [radius(t) for t in ts]
    dens = map(eachindex(ts)) do i
        weight === :uniform && return 1.0
        weight === :solid_angle && return sin(ts[i])
        weight === :image || throw(ArgumentError("fit_camera: weight must be :image, :solid_angle or :uniform, got :$weight"))
        return max(rA[i] * abs(_radius_derivative(srcr, ts[i])), 0.0)
    end
    return ts, qs, rA, dens
end
# dr/dθ of the source's radial profile at θ, exact (forward-mode AD).
_radius_derivative(srcr, θ::Float64) = Float64(ForwardDiff.derivative(t -> projected_radius(srcr, t), θ))

# The linearization state of the stack's chart LM (VGC's `lmsolve_chart!`, over
# the `residual_rows!` protocol): the chart `P`, whether the rows are angle
# residuals (the lift families), the nodes, the target radii and the rows' √(q·w).
struct _LensRegressionLm{TP <: ProductManifold}
    P::TP
    angle_residual::Bool
    θs::Vector{Float64}
    rA::Vector{Float64}
    sw::Vector{Float64}
end

# e_i = √(q_i w_i) · (s·r_B(θ_i; q) − r_A(θ_i)), with r_B the candidate lens's
# own `projected_radius` (the lens is built from Dual parameters). The lift
# families — rational and Scaramuzza — fit through their `view_angle` at the
# source radius instead: the lift is defined there even where a candidate folds
# inside the fit domain, and the perspective anchor images nothing past 90°
# while the source may.
function _lens_rows(lm::_LensRegressionLm, pv)
    sf, lq = _scale_and_lens(lm.P, pv)
    θs = lm.θs; rA = lm.rA
    return map(eachindex(θs)) do i
        ei = lm.angle_residual ?
            (view_angle(lq, rA[i] / sf) - θs[i]) * rA[i] :   # angle × r ≈ radius units
            sf * projected_radius(lq, θs[i]) - rA[i]
        lm.sw[i] * ei
    end
end
residual_rows!(f, lm::_LensRegressionLm, x) = (f .= _lens_rows(lm, x))
# The Jacobian is VGC's default: ForwardDiff through `residual_rows!`.

# The chart coordinates of the fit (the anchor is the start: every coordinate at
# zero), its retcode and its iterations — a function barrier static in the N
# coordinates (the focal's and the family's).
function _regress_lens(lm::_LensRegressionLm, rows::Int, ::Val{N}, max_iter::Int) where {N}
    ews = _lm_workspace(nothing, rows, Val(N))      # the one LO-buffer reuse policy (interface.jl)
    x, _, conv, iterations = lmsolve_chart!(ews, lm; maxit = max_iter, stationarity_tol = _CHART_LM_STATIONARITY_TOL)
    return x, VgtkCore.lm_retcode(conv), iterations
end

# The normalized radius at which the ray from the principal point along each of
# `_REGION_AZIMUTHS` azimuths of the normalized plane leaves `region` (bisected,
# the region taken as star-shaped about the principal point), through `K`'s
# 2 × 2 part — so a radius of the profile is the same radius on every azimuth.
function _exit_radii(region, K::SMatrix{3, 3, Float64})
    box = Rect2(region)   # an ArgumentError for the whole plane: the region must be bounded
    pp = SVector(K[1, 3], K[2, 3])
    A = SMatrix{2, 2, Float64}(K[1, 1], K[2, 1], K[1, 2], K[2, 2])
    inside(p) = Point2{Float64}(p[1], p[2]) ∈ region
    inside(pp) || throw(ArgumentError("fit_camera: the region does not contain the principal point $pp"))
    lo, hi = SVector{2, Float64}(minimum(box)), SVector{2, Float64}(maximum(box))
    corners = (lo, SVector(hi[1], lo[2]), SVector(lo[1], hi[2]), hi)
    Ai = inv(A)
    ρhi = maximum(c -> norm(Ai * (c - pp)), corners) * (1 + 1.0e-9) + 1.0e-12
    return map(1:_REGION_AZIMUTHS) do k
        d = A * SVector(cospi(2 * (k - 0.5) / _REGION_AZIMUTHS), sinpi(2 * (k - 0.5) / _REGION_AZIMUTHS))
        a, b = 0.0, ρhi
        for _ in 1:60
            m = (a + b) / 2
            inside(pp + m * d) ? (a = m) : (b = m)
        end
        a
    end
end

# The fraction of the azimuths along which radius `ρ` still lies in the region (a
# relative slack of 1e-9 for the bisection's rounding at the outermost radius).
_coverage(exits, ρ) = count(e -> e >= ρ * (1 - 1.0e-9), exits) / length(exits)

# ==============================================================================
# The spline families: a quadrature on the image, penalized linear least squares
# ==============================================================================

# The quadrature of the region: the pixel centres (COLMAP) in the region on a
# lattice of `stride` pixels (Core's `pixel_centres`) — the fit's points, and the
# held-out check's offset by half a stride — with the source's rays there (a
# point the source does not image is dropped) and the held-out points' edge band.
#
# `samples(source, pixels)` is the source's datum per pixel, NaN where it images
# none: its rays (`_source_rays`, the default) or its observation lines
# (`_source_lines`); the result's `fit_rays` and `held_rays` hold those data.
_image_quadrature(source, region) = _image_quadrature(_source_rays, source, region)
function _image_quadrature(samples::F, source, region) where {F}
    box = Rect2(region)   # an ArgumentError for the whole plane: the region must be bounded
    lo, hi = minimum(box), maximum(box)
    extent = maximum(hi - lo)
    stride = max(2, 2 * round(Int, minimum(hi - lo) / 1000))
    W, H = ceil(Int, hi[1]), ceil(Int, hi[2])
    (W > 0 && H > 0) || throw(ArgumentError("fit_camera: the region $box lies outside the image quadrant"))
    u0, v0 = max(1, floor(Int, lo[1]) + 1), max(1, floor(Int, lo[2]) + 1)
    lattice(off) = pixel_centres(region, (u0 + off):stride:W, (v0 + off):stride:H)
    function imaged(pixels)
        data = samples(source, pixels)
        keep = findall(_is_finite_sample, data)
        return pixels[keep], data[keep]
    end
    fit_px, fit_rays = imaged(lattice(0))
    held_px, held_rays = imaged(lattice(stride ÷ 2))
    (length(fit_px) >= 16 && !isempty(held_px)) ||
        throw(ArgumentError("fit_camera: the source images too few points of the region ($(length(fit_px)))"))
    edge = map(p -> _in_edge_band(region, p), held_px)
    return (; region, stride, extent, fit_px, fit_rays, held_px, held_rays, edge)
end

# The source's rays at `pixels`: its batched `backproject`, NaN where it images none.
_source_rays(source, pixels) = _svectors(backproject(source, pixels))
# a function source (Core's generic sources: `u -> ray` or `nothing`), pixel by pixel
_source_rays(source::Function, pixels) = map(p -> _ray_or_nan(source(p)), pixels)
_ray_or_nan(::Nothing) = SVector(NaN, NaN, NaN)
_ray_or_nan(r) = SVector{3, Float64}(r)
_source_ray(source, u) = backproject(source, u)
_source_ray(source::Function, u) = source(u)
_svectors(r::AbstractVector{SVector{3, Float64}}) = r
_svectors(r) = SVector{3, Float64}.(r)

# The source's observation lines at `pixels`, `(origin, unit direction)`, NaN where
# it images none: a non-central grid's `Ray3`s (plain metres), and for a central
# source its rays through the projection centre, the origin.
function _source_lines(source::NoncentralGenericModel, pixels)
    nan3 = SVector(NaN, NaN, NaN)
    return map(pixels) do p
        r = backproject(source, p)
        r === nothing ? (nan3, nan3) : (SVector{3, Float64}(r.origin), SVector{3, Float64}(r.direction))
    end
end
_source_lines(source, pixels) = map(r -> (zero(SVector{3, Float64}), normalize(r)), _source_rays(source, pixels))

_is_finite_sample(r::StaticVector) = all(isfinite, r)
_is_finite_sample(l::Tuple) = all(_is_finite_sample, l)

# Whether a point lies within `_EDGE_BAND_PX` of the region's edge: a point at
# that distance on one of 16 directions leaves the region (a convex region's
# distance to its edge, to the 16-gon's chord).
_in_edge_band(region, p) = any(e -> Point2{Float64}((p + _EDGE_BAND_PX * e)...) ∉ region, _EDGE_DIRECTIONS)
const _EDGE_DIRECTIONS = ntuple(k -> SVector(cospi((k - 1) / 8), sinpi((k - 1) / 8)), 16)

# One sample's design row: the 16 controls of its spline window and their
# weights, in the column-major linear order of a `ny`-row control matrix (the
# window's columns outer, its rows inner: increasing, as a CSC column needs).
@inline function _design_row(window, ny::Int)
    ix, iy, wx, wy = window
    # control [iy - 3 + r, ix - 3 + c] (spline_window's layout) at linear index (column − 1)·ny + row
    idx = SVector{16, Int}(ntuple(k -> (ix - 3 + (k - 1) ÷ 4) * ny + iy - 3 + (k - 1) % 4 + 1, Val(16)))
    w = SVector{16, Float64}(ntuple(k -> wx[(k - 1) ÷ 4 + 1] * wy[(k - 1) % 4 + 1], Val(16)))
    return (idx, w)
end

# The design rows of the points `pts` on an `ny`-row lattice, `window_fn(p)` each
# point's spline window.
_design_rows(window_fn::F, pts, ny::Int) where {F} = pointwise!(
    p -> _design_row(window_fn(p), ny), Vector{Tuple{SVector{16, Int}, SVector{16, Float64}}}(undef, length(pts)), pts
)

# The right-hand side of a solve: one row per sample, one column per component.
_rhs_matrix(vs::AbstractVector{<:StaticVector{K}}) where {K} = [v[k] for v in vs, k in 1:K]

# The third-difference operator of an n-node axis (Core's one Δ³ stencil).
function _third_difference(n::Int)
    s = difference_stencil(3)
    return spdiagm(n - 3, n, (k - 1 => fill(s[k], n - 3) for k in 1:4)...)
end

# The P-spline least squares of an `ny × nx` lattice: the controls minimizing
# ‖A c − B‖² + λ(‖Dₓ³c‖² + ‖Dᵧ³c‖²) for the design rows `rows` and the targets `B`
# (one column per component), λ = `_PSPLINE_PENALTY` · mean(diag(AᵀA)); the
# controls as an `ny × nx` matrix of `SVector{K}` and λ.
function _penalized_solve(rows::AbstractVector, B::Matrix{Float64}, ny::Int, nx::Int, ::Val{K}) where {K}
    _check_lattice_size(ny, nx)
    N = length(rows)
    rowval = Vector{Int}(undef, 16N)
    nzval = Vector{Float64}(undef, 16N)
    for (i, (idx, w)) in enumerate(rows)
        rowval[(16i - 15):16i] .= idx
        nzval[(16i - 15):16i] .= w
    end
    all(i -> 1 <= i <= ny * nx, rowval) ||
        throw(ArgumentError("fit_camera: a design row reads a control outside the $ny × $nx lattice"))   # the CSC constructor does not check
    At = SparseMatrixCSC(ny * nx, N, collect(1:16:(16N + 1)), rowval, nzval)
    AtA = At * transpose(At)
    λ = _PSPLINE_PENALTY * sum(diag(AtA)) / size(AtA, 1)
    Dx = kron(_third_difference(nx), sparse(I, ny, ny))   # across columns (the chart's first axis)
    Dy = kron(sparse(I, nx, nx), _third_difference(ny))   # across rows (its second)
    M = AtA + λ * (transpose(Dx) * Dx + transpose(Dy) * Dy)
    AtB = Matrix(At * B)
    F = cholesky(Symmetric(M); check = false)
    X = issuccess(F) ? F \ AtB : lu(M) \ AtB   # a pivot lost to rounding: the sparse LU, never a dense copy
    controls = [SVector{K, Float64}(ntuple(k -> X[(c - 1) * ny + r, k], Val(K))) for r in 1:ny, c in 1:nx]
    return controls, λ
end

# The third-difference penalty and the cubic window need at least 4 nodes per axis
# (ChartSpline's and the generic models' own bound).
_check_lattice_size(ny::Integer, nx::Integer) = (ny >= 4 && nx >= 4) || throw(
    ArgumentError("fit_camera: a uniform cubic lattice needs at least 4 × 4 nodes, got $ny × $nx")
)

# An error at the held-out points (pixels, or metres for a line offset): its
# maximum, rms and edge-band maximum.
function _held_out_errors(err::AbstractVector{Float64}, edge::AbstractVector{Bool})
    return (; max = maximum(err), rms = sqrt(sum(abs2, err) / length(err)), max_edge = any(edge) ? maximum(err[edge]) : 0.0)
end

# ------------------------------------------------------------------------------
# SplineProjection
# ------------------------------------------------------------------------------

# The quadrature in the chart: the equidistant direction chart about the
# source's ray at the region's centroid (a rotation R whose third row is that
# axis), the chart points of the fit's and the held-out rays, and the chart's
# bounding rectangle.
function _chart_quadrature(source, region)
    q = _image_quadrature(source, region)
    c = sum(q.fit_px) / length(q.fit_px)
    a = _source_ray(source, c)
    (a === nothing || !all(isfinite, a)) && throw(ArgumentError("fit_camera: the source images no ray at the region's centroid $c"))
    axis = normalize(SVector{3, Float64}(a))
    e1 = cross(SVector(0.0, 1.0, 0.0), axis)
    e1 = norm(e1) > 1.0e-6 ? normalize(e1) : normalize(cross(SVector(1.0, 0.0, 0.0), axis))
    e2 = cross(axis, e1)
    R = SMatrix{3, 3, Float64}(e1[1], e2[1], axis[1], e1[2], e2[2], axis[2], e1[3], e2[3], axis[3])   # rows e1, e2, axis
    chart = Lens(Equidistant()) ∘ LinearMap(R)
    fit_charts = pointwise!(chart, similar(q.fit_rays, SVector{2, Float64}), q.fit_rays)
    held_charts = pointwise!(chart, similar(q.held_rays, SVector{2, Float64}), q.held_rays)
    lo, hi = zip(extrema(c -> c[1], fit_charts), extrema(c -> c[2], fit_charts))
    margin = 1.0e-6
    area = Rect2(lo[1] - margin, lo[2] - margin, hi[1] - lo[1] + 2margin, hi[2] - lo[2] + 2margin)
    # the spacing search's one right-hand side: the fit points' pixels
    return (; q..., chart, fit_charts, held_charts, area, rhs = _rhs_matrix(q.fit_px))
end

# The spline fit at `cells` knot spacings along the chart's longer side: the
# spacing h, the counts nᵢ = ⌈extentᵢ / h⌉ + 3, the controls, λ, the held-out errors.
function _spline_projection_fit(q, cells::Int)
    w = SVector{2, Float64}(maximum(q.area) - minimum(q.area))
    h = maximum(w) / cells
    nx = ceil(Int, w[1] / h * (1 - 1.0e-12)) + 3
    ny = ceil(Int, w[2] / h * (1 - 1.0e-12)) + 3
    rows = _design_rows(c -> spline_window(q.area, (ny, nx), c), q.fit_charts, ny)
    controls, λ = _penalized_solve(rows, q.rhs, ny, nx, Val(2))
    spline = ChartSpline(q.area, controls)
    err = norm.(spline(q.held_charts) .- q.held_px)
    return (; spline, h, λ, counts = (nx, ny), errors = _held_out_errors(err, q.edge))
end

function fit_camera(::Type{SplineProjection}, source, region; tolerance::Real)
    _check_tolerance(tolerance)
    q = _chart_quadrature(_source_model(source), region)
    r, code, n = _spacing_search(cells -> _spline_projection_fit(q, cells), tolerance, _max_cells(q.extent, q.stride))
    return FitResult(
        SplineProjection(q.chart, r.spline, q.region), code, 0, r.errors.rms, length(q.fit_px),
        CameraFitMetadata(
            q.region, :image, r.errors.max, r.errors.max_edge,
            (; spacing = r.h, penalty = r.λ, counts = r.counts, stride = q.stride, tolerance = Float64(tolerance), outer_iterations = n)
        )
    )
end

# The source a fit samples: a camera's model (a bounded camera's unbounded
# model, Core's `model`), and any other source — a function of the pixel, one
# of Core's generic sources — as it is.
_source_model(source::AbstractCamera) = model(source)
_source_model(source) = source

_check_tolerance(tolerance::Real) =
    tolerance > 0 || throw(ArgumentError("fit_camera: tolerance must be positive, got $tolerance"))

# The spacing search's bracket: from 4 cells along the longer side (of the
# chart, or of the image) up to a cell of three quadrature strides (fewer
# points per cell leave it to the penalty).
_max_cells(extent::Real, stride::Integer) = max(1, floor(Int, extent / (3 * stride)))

# The spacing search of the spline and grid targets: `fit(cells)` the fit with
# `cells` knot cells along the longer side, its held-out maximum `errors.max`.
# Doubling from 4 cells to the first count that meets `tolerance`, then
# bisection between it and the last that did not: the LARGEST spacing that meets
# it, `:converged`; `:max_iterations` with what the finest spacing reached when
# `cells_max` does not meet it. Each fit is an outer iteration (the fit's inputs
# change, the estimator repeats); their count is the third value.
function _spacing_search(fit::F, tolerance::Real, cells_max::Int) where {F}
    meets(r) = r.errors.max <= tolerance
    outer_iterations = 1
    lower = 0
    cells = min(4, cells_max)
    best = fit(cells)
    while !meets(best)
        lower = cells
        cells == cells_max && return best, :max_iterations, outer_iterations
        cells = min(2cells, cells_max)
        best = fit(cells)
        outer_iterations += 1
    end
    upper = cells
    while upper - lower > 1
        mid = (lower + upper) ÷ 2
        r = fit(mid)
        outer_iterations += 1
        if meets(r)
            upper, best = mid, r
        else
            lower = mid
        end
    end
    return best, :converged, outer_iterations
end

# ------------------------------------------------------------------------------
# The grid targets: CentralGenericModel and NoncentralGenericModel
# ------------------------------------------------------------------------------
#
# One body for both: the quadrature of the source's data, then the spacing
# search over Core's full-image lattice on the source's image (the calibrated
# area is the image; the region restricts the quadrature only) — at each
# spacing the design rows, ONE penalized solve, the model and the held-out
# errors. The target type selects the data (`_grid_samples`), the right-hand
# side, the model built from the controls and the errors.

fit_camera(::Type{CentralGenericModel}, source, region; tolerance::Real) =
    _fit_grid(CentralGenericModel, source, region, tolerance)

function _fit_grid(::Type{G}, source, region, tolerance::Real) where {G}
    _check_tolerance(tolerance)
    g = _grid_data(G, source, region)
    W, H = g.size
    r, code, n = _spacing_search(cells -> _pspline_grid(G, g, _lattice_grid_size(W, H, cells)), tolerance, _max_cells(max(W, H), g.q.stride))
    return _grid_result(r, code, g, (; tolerance = Float64(tolerance), outer_iterations = n))
end

# What every lattice of one fit shares: the source's image, its model, the
# quadrature, the right-hand side and its anchor.
function _grid_data(::Type{G}, source, region) where {G}
    size = _image_size(source)
    cam = _source_model(source)
    q = _image_quadrature(_grid_samples(G), cam, region)
    rhs, anchor = _grid_rhs(G, q)
    return (; size, cam, q, B = _rhs_matrix(rhs), anchor, region, K = Val(length(eltype(rhs))))
end
# The P-spline solve on one lattice of `grid_size` (columns, rows): the one
# implementation `pspline_grid` runs once and `fit_camera`'s spacing search at
# every spacing it tries.
_pspline_grid(::Type{G}, g, grid_size) where {G} = _grid_fit(G, g.cam, g.size, Tuple(Int.(grid_size)), g.q, g.B, g.anchor, g.region, g.K)
_grid_result(r, code, g, extra::NamedTuple) = FitResult(
    r.fitted, code, 0, r.errors.rms, length(g.q.fit_px),
    CameraFitMetadata(
        g.region, :image, r.errors.max, r.errors.max_edge,
        (; penalty = r.λ, grid_size = r.grid_size, stride = g.q.stride, extra..., r.family...)
    )
)

"""
    pspline_grid(G, source, region, grid_size) -> FitResult{G, CameraFitMetadata}

The P-spline (penalized B-spline least squares, Eilers–Marx) of a generic
grid model `G` — `CentralGenericModel` or `NoncentralGenericModel` — onto
`source` over `region`, on the GIVEN lattice: Core's full-image lattice of
`grid_size = (columns, rows)` nodes on the source's image (as
`generic_lattice` takes it), the region restricting the quadrature only, ONE
penalized solve with the third-difference penalty `fit_camera` uses — the
non-central grid's points anchored on c*, its directions and points one
right-hand side — in the source's frame. No spacing search and no tolerance:
the one deterministic least-squares step `fit_camera(G, source, region;
tolerance)` repeats at each spacing it tries (the same implementation), so
`pspline_grid(G, source, region, (gw, gh))` at the lattice `fit_camera` chose
is its result.

`source` is what [`fit_camera`](@ref)'s grid targets take (a grid model, or a
bounded camera: the lattice needs the image). The result's `retcode` is
`:solved`, `iterations` 0; its metadata is `fit_camera`'s, the held-out errors
(the direction error in pixels; for the non-central grid also the line
offsets in metres, and `center`) and `family = (; penalty, grid_size, stride,
n_held_out_dropped, …)` with `grid_size` the lattice's `(rows, columns)`, as
`fit_camera` reports it, and no `tolerance` or `outer_iterations`.
"""
function pspline_grid(::Type{G}, source, region, grid_size::Tuple{Integer, Integer}) where {G <: Union{CentralGenericModel, NoncentralGenericModel}}
    g = _grid_data(G, source, region)
    return _grid_result(_pspline_grid(G, g, grid_size), :solved, g, (;))
end

# The grid at one spacing: Core's lattice of `grid_size` (columns, rows), the
# penalized solve, the model and its held-out errors.
function _grid_fit(::Type{G}, source, size, grid_size, q, B, anchor, region, ::Val{K}) where {G, K}
    lattice = _grid_lattice(G, source, size, grid_size, region)
    gh, gw = _lattice_size(lattice)
    rows = _design_rows(p -> spline_window(lattice, p), q.fit_px, gh)
    controls, λ = _penalized_solve(rows, B, gh, gw, Val(K))
    fitted = _grid_model(G, lattice, controls, anchor, region)
    errors, family = _grid_errors(fitted, source, q, anchor)
    return (; fitted, errors, family, λ, grid_size = (gh, gw))
end

# The lattice of `cells` cells along the image's longer side: the spacing
# h = max(W, H) / cells, the node counts ⌈extent / h⌉ + 3 per axis (the chart
# spline's rule), as Core's `grid_size` (columns, rows).
function _lattice_grid_size(W::Integer, H::Integer, cells::Integer)
    h = max(W, H) / cells
    return (ceil(Int, W / h * (1 - 1.0e-12)) + 3, ceil(Int, H / h * (1 - 1.0e-12)) + 3)
end

_lattice_size(m::CentralGenericModel) = size(m.grid)
_lattice_size(m::NoncentralGenericModel) = size(m.directions)
_lattice_area(m) = (m.calib_min_x, m.calib_min_y, m.calib_max_x, m.calib_max_y)

_grid_samples(::Type{CentralGenericModel}) = _source_rays

# The lattice: Core's layout of a grid model from the source (its nominal
# pinhole), without sampling it — the solve sets every control.
_grid_lattice(::Type{G}, source, size, grid_size, region) where {G} = generic_lattice(G, source, size; grid_size, domain = region)
_grid_rhs(::Type{CentralGenericModel}, q) = (q.fit_rays, nothing)
_grid_model(::Type{CentralGenericModel}, lattice, controls, _, region) = CentralGenericModel{Float64}(
    lattice.width, lattice.height, _lattice_area(lattice)..., Vec3{Float64}.(controls);
    unit_controls = false, nominal = nominal_intrinsics(lattice), domain = region
)
function _grid_errors(fitted::CentralGenericModel, source, q, _)
    e, dropped = _direction_summary(fitted, source, q, normalize.(q.held_rays))
    return e, (; n_held_out_dropped = dropped)
end

# The image a grid target's lattice covers: a grid source's own, a bounded
# camera's extent. A bare camera model has no image.
_image_size(m::Union{CentralGenericModel, NoncentralGenericModel}) = (m.width, m.height)
_image_size(bc::BoundedCamera) = (e = extent(bc); (Int(e.width), Int(e.height)))
_image_size(source) = throw(
    ArgumentError(
        "fit_camera(<grid model>, …): a $(nameof(typeof(source))) source has no image for the grid's full-image lattice; " *
            "bound it by its sensor or extent, `BoundedCamera(model, sensor)`"
    )
)

# ------------------------------------------------------------------------------
# NoncentralGenericModel
# ------------------------------------------------------------------------------
#
# The same design as the central grid, on lines: Core's full-image lattice, the
# region restricting the quadrature only, ONE penalized solve for both grids
# (the unit directions and the points are columns of one right-hand side over
# one design). A line's point is not data — any point of the line is the same
# line — so the fit pins it: every quadrature line contributes its point closest
# to c*, the least-squares point nearest to all of them (Core's `nearest_point`).
# The points then vary as little as the lines allow, and a central source's are
# c* itself. Anchoring on a plane instead (the pupil plane z = const) explodes
# for rays near and past 90°, where the plane meets the line far away or never.

fit_camera(::Type{NoncentralGenericModel}, source, region; tolerance::Real) =
    _fit_grid(NoncentralGenericModel, source, region, tolerance)

_grid_samples(::Type{NoncentralGenericModel}) = _source_lines

# The right-hand side: the unit direction, and the line's point closest to c*
# relative to c* (the solve is translation-equivariant: a B-spline's weights sum
# to one and the third differences vanish on constants); c* the anchor.
function _grid_rhs(::Type{NoncentralGenericModel}, q)
    center = nearest_point([o => d for (o, d) in q.fit_rays])
    return [vcat(d, o + dot(center - o, d) * d - center) for (o, d) in q.fit_rays], center
end

# A control of the joint solve split into its direction and its absolute point.
_direction_control(x::SVector{6}) = Vec3{Float64}(x[1], x[2], x[3])
_point_control(x::SVector{6}, center) = Vec3{Float64}(x[4] + center[1], x[5] + center[2], x[6] + center[3])

_grid_model(::Type{NoncentralGenericModel}, lattice, controls, center, region) = NoncentralGenericModel{Float64}(
    lattice.width, lattice.height, _lattice_area(lattice)..., _direction_control.(controls), _point_control.(controls, Ref(center));
    unit_controls = false, nominal = nominal_intrinsics(lattice), domain = region
)
function _grid_errors(fitted::NoncentralGenericModel, source, q, center)
    e, offsets, dropped = _line_errors(fitted, source, q)
    return e, (; center, max_offset = offsets.max, rms_offset = offsets.rms, max_offset_edge = offsets.max_edge, n_held_out_dropped = dropped)
end

# The held-out DIRECTION error of a fitted grid (either target) in pixels: at
# each held-out pixel the fitted direction's tangent deviation from the source's
# unit direction `d`, mapped back through the source's back-projection Jacobian
# `J = ∂d/∂u` (its `backproject_with_jacobian`), `‖J⁺δ‖` — the round trip
# through the source to first order. Evaluated directly through the fitted
# spline (`interpolated_direction`, continued past the calibrated area as the
# fit's design is): no projection, no seed, so no solver tolerance floors it.
#
# A held-out pixel where the source's Jacobian is not finite (a source that
# images the ray but cannot differentiate there) has no pixel error: it is
# dropped, as a non-finite sample is from the quadrature, and counted
# (`kept` marks the rest, aligned with `px`). Returns `(errors, kept)`, the
# errors at the kept pixels only.
function _direction_errors(fitted, source, px, d_src)
    kept = falses(length(px))
    errors = Float64[]
    for (k, i) in enumerate(eachindex(px))
        d = d_src[i]
        J = last(backproject_with_jacobian(source, px[i]))
        all(isfinite, J) || continue
        kept[k] = true
        f = normalize(SVector{3, Float64}(interpolated_direction(fitted, px[i])))
        δ = f - d
        δ -= dot(δ, d) * d
        push!(errors, norm((transpose(J) * J) \ (transpose(J) * δ)))
    end
    any(kept) || throw(ArgumentError("fit_camera: the source's back-projection Jacobian is not finite at any held-out pixel"))
    return errors, kept
end
# The held-out direction errors' summary over the kept pixels, and the count dropped.
function _direction_summary(fitted, source, q, d_src)
    e, kept = _direction_errors(fitted, source, q.held_px, d_src)
    return _held_out_errors(e, q.edge[kept]), count(!, kept)
end

# The held-out errors of a fitted non-central grid against the source's lines:
# the direction error in pixels (above) and the LINE OFFSET
# `‖(I − ddᵀ)(o_fit − o_src)‖`, the distance from the source's point to the
# fitted line's point across the source's direction, in the source's length
# unit (plain metres), through the fitted point spline.
function _line_errors(fitted::NoncentralGenericModel, source, q)
    px = q.held_px
    o_src = first.(q.held_rays)
    d_src = last.(q.held_rays)
    direction, dropped = _direction_summary(fitted, source, q, d_src)
    offset = map(eachindex(px)) do i
        d = d_src[i]
        Δ = SVector{3, Float64}(interpolated_point(fitted, px[i])) - o_src[i]
        norm(Δ - dot(Δ, d) * d)
    end
    return direction, _held_out_errors(offset, q.edge), dropped
end

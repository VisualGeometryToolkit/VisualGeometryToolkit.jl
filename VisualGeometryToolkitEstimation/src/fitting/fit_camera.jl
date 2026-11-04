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
#   CentralGenericModel      the grid of Core's layout (`grid_size`) over the
#                            region's pixel bounding box, the source's rays by
#                            penalized linear least squares; the tolerance checked
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
`weight` measure, the round trip's error in pixels — `max_px` (its maximum),
`rms_px` (√ of its mean square: the objective's quantity) and `max_px_edge`
(its maximum within 100 px of the region's edge) — and the target family's own
record `family`, a `NamedTuple`:

- a lens family: `(; focal_scale, θ_max, θ_source, clipped)` — the joint focal
  rescale `s`, the fitted and the source's angular domains (radians), and
  whether the target family cut the source's domain; the errors are at the
  quadrature's nodes of the radial profile;
- `SplineProjection`: `(; spacing, penalty, counts, stride, tolerance)` — the
  knot spacing `h` (radians in the chart), the P-spline weight `λ`, the control
  counts per chart axis, the quadrature's stride in pixels and the tolerance
  asked for; the errors are the held-out check's;
- `CentralGenericModel`: `(; penalty, grid_size, stride, tolerance)`; the errors are
  the held-out check's, through the fitted grid's `project`.
"""
struct CameraFitMetadata{R, F <: NamedTuple}
    region::R
    weight::Symbol
    max_px::Float64
    rms_px::Float64
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
    fit_camera(::Type{T}, source, region; tolerance, weight = :image, …) -> FitResult{<model of T>, CameraFitMetadata}

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
  limits — a cut of the source's domain is `clipped` and warned. (The region's
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
  linear least squares with a P-spline penalty. The knot spacing `h` is bisected to the
  largest one whose held-out maximum error is at most `tolerance` (px); a fit
  that cannot reach it within the spacing bracket returns `retcode
  :max_iterations` with what it reached. `iterations` counts the fits made.
- **`CentralGenericModel`** (`fit_camera(CentralGenericModel, source, region;
  grid_size, tolerance)`): a grid model in Core's full-image layout — the
  lattice the node-interpolation constructor `CentralGenericModel(source, size;
  grid_size)` lays over the whole image, which is the start — with `domain` the
  region, fitted to the source's rays by linear least squares with the P-spline
  penalty. The region restricts the quadrature only, never the lattice: a node
  whose support misses it is held by the penalty. The image is the source's: a
  grid model's own, or a [`BoundedCamera`](@ref VisualGeometryToolkitCore.BoundedCamera)'s
  extent (a bare `CameraModel` has none: an `ArgumentError`). The tolerance is
  checked (warned when exceeded), never searched; `retcode :solved`.

`source` is a `CameraModel` (bounded or not) or a `CentralGenericModel` (anything
with a batched `backproject(source, pixels)`). The spline families take `weight = :image` only:
their quadrature is on the image. Their per-point steps (the source's
backprojection, the chart, the design rows, the evaluation, the errors) run as
batches — Core's KernelAbstractions kernels and [`pointwise!`](@ref
VisualGeometryToolkitCore.pointwise!) — on the host, and the penalized normal
equations are solved by a sparse Cholesky factorization.

The metadata ([`CameraFitMetadata`](@ref)) records the region, the round trip's
`max_px`, `rms_px` and `max_px_edge`, and the family's own record; `rms` is
`rms_px`.

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
    θ_source = min(Float64(view_angle_limit(srcr)), Float64(view_angle(srcr, ρmax)))
    θ_max = min(θ_source, _family_limit(L))
    θ_max > 0.02 || throw(ArgumentError("fit_camera: empty fit domain (θ_max = $θ_max)"))
    clipped = θ_max < θ_source * (1 - 1.0e-6)
    clipped && @warn "fit_camera: target family $(L) cannot represent the source's field — fitting θ ∈ [0, $(round(rad2deg(θ_max); digits = 1))°] of the source's $(round(rad2deg(θ_source); digits = 1))°"

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
        CameraFitMetadata(region, weight, max_px, rms_px, max_px_edge, (; focal_scale = s, θ_max = float(θ_max), θ_source, clipped))
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
# and the density w (`:image` differentiates `radius` by central differences,
# clamped to [0, b]).
function _forward_samples(srcr, b::Float64, n::Int, weight::Symbol)
    radius(θ) = Float64(projected_radius(srcr, θ))
    ts, qs = _clenshaw_curtis(n, b)
    rA = [radius(t) for t in ts]
    h = 1.0e-5 * b
    dens = map(eachindex(ts)) do i
        weight === :uniform && return 1.0
        weight === :solid_angle && return sin(ts[i])
        weight === :image || throw(ArgumentError("fit_camera: weight must be :image, :solid_angle or :uniform, got :$weight"))
        tm, tp = max(ts[i] - h, 0.0), min(ts[i] + h, b)
        dr = (radius(tp) - radius(tm)) / (tp - tm)
        return max(rA[i] * abs(dr), 0.0)
    end
    return ts, qs, rA, dens
end

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

_check_image_weight(weight::Symbol) =
    weight === :image || throw(ArgumentError("fit_camera: a spline target's quadrature is on the image; weight must be :image, got :$weight"))

# The quadrature of the region: the pixel centres (COLMAP) of the region's
# `pixel_mask` on a lattice of `stride` pixels — the fit's points, and the
# held-out check's offset by half a stride — with the source's rays there (a
# point the source does not image is dropped) and the held-out points' edge band.
function _image_quadrature(source, region)
    box = Rect2(region)   # an ArgumentError for the whole plane: the region must be bounded
    lo, hi = minimum(box), maximum(box)
    extent = maximum(hi - lo)
    stride = max(2, 2 * round(Int, minimum(hi - lo) / 1000))
    W, H = ceil(Int, hi[1]), ceil(Int, hi[2])
    (W > 0 && H > 0) || throw(ArgumentError("fit_camera: the region $box lies outside the image quadrant"))
    mask = pixel_mask(region, W, H)
    u0, v0 = max(1, floor(Int, lo[1]) + 1), max(1, floor(Int, lo[2]) + 1)
    lattice(off) = [SVector(u - 0.5, v - 0.5) for v in (v0 + off):stride:H for u in (u0 + off):stride:W if mask[v, u]]
    function imaged(pixels)
        rays = backproject(source, pixels)
        keep = findall(r -> all(isfinite, r), rays)
        return pixels[keep], SVector{3, Float64}.(rays[keep])
    end
    fit_px, fit_rays = imaged(lattice(0))
    held_px, held_rays = imaged(lattice(stride ÷ 2))
    (length(fit_px) >= 16 && !isempty(held_px)) ||
        throw(ArgumentError("fit_camera: the source images too few points of the region ($(length(fit_px)))"))
    edge = map(p -> _in_edge_band(region, p), held_px)
    return (; region, stride, extent, fit_px, fit_rays, held_px, held_rays, edge)
end

# Whether a point lies within `_EDGE_BAND_PX` of the region's edge: a point at
# that distance on one of 16 directions leaves the region (a convex region's
# distance to its edge, to the 16-gon's chord).
_in_edge_band(region, p) = any(k -> Point2{Float64}((p + _EDGE_BAND_PX * SVector(cospi(k / 8), sinpi(k / 8)))...) ∉ region, 0:15)

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

# The third-difference operator of an n-node axis.
_third_difference(n::Int) =
    spdiagm(n - 3, n, 0 => fill(1.0, n - 3), 1 => fill(-3.0, n - 3), 2 => fill(3.0, n - 3), 3 => fill(-1.0, n - 3))

# The P-spline least squares of an `ny × nx` lattice: the controls minimizing
# ‖A c − B‖² + λ(‖Dₓ³c‖² + ‖Dᵧ³c‖²) for the design rows `rows` and the targets `B`
# (one column per component), λ = `_PSPLINE_PENALTY` · mean(diag(AᵀA)); the
# controls as an `ny × nx` matrix of `SVector{K}` and λ.
function _penalized_solve(rows::AbstractVector, B::Matrix{Float64}, ny::Int, nx::Int, ::Val{K}) where {K}
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

# The round trip's errors at the held-out points: their maximum, rms and edge-band maximum.
function _held_out_errors(err::AbstractVector{Float64}, edge::AbstractVector{Bool})
    return (;
        max_px = maximum(err), rms_px = sqrt(sum(abs2, err) / length(err)),
        max_px_edge = any(edge) ? maximum(err[edge]) : 0.0,
    )
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
    a = backproject(source, c)
    (a === nothing || !all(isfinite, a)) && throw(ArgumentError("fit_camera: the source images no ray at the region's centroid $c"))
    axis = normalize(SVector{3, Float64}(a))
    e1 = cross(SVector(0.0, 1.0, 0.0), axis)
    e1 = norm(e1) > 1.0e-6 ? normalize(e1) : normalize(cross(SVector(1.0, 0.0, 0.0), axis))
    e2 = cross(axis, e1)
    R = SMatrix{3, 3, Float64}(e1[1], e2[1], axis[1], e1[2], e2[2], axis[2], e1[3], e2[3], axis[3])   # rows e1, e2, axis
    chart = Lens(Equidistant()) ∘ LinearMap(R)
    fit_charts = pointwise!(chart, similar(q.fit_rays, SVector{2, Float64}), q.fit_rays)
    held_charts = pointwise!(chart, similar(q.held_rays, SVector{2, Float64}), q.held_rays)
    lo = minimum(reduce(hcat, fit_charts); dims = 2)
    hi = maximum(reduce(hcat, fit_charts); dims = 2)
    margin = 1.0e-6
    area = Rect2(lo[1] - margin, lo[2] - margin, hi[1] - lo[1] + 2margin, hi[2] - lo[2] + 2margin)
    return (; q..., chart, fit_charts, held_charts, area)
end

# The spline fit at `cells` knot spacings along the chart's longer side: the
# spacing h, the counts nᵢ = ⌈extentᵢ / h⌉ + 3, the controls, λ, the held-out errors.
function _spline_projection_fit(q, cells::Int)
    w = SVector{2, Float64}(maximum(q.area) - minimum(q.area))
    h = maximum(w) / cells
    nx = ceil(Int, w[1] / h * (1 - 1.0e-12)) + 3
    ny = ceil(Int, w[2] / h * (1 - 1.0e-12)) + 3
    rows = pointwise!(
        c -> _design_row(spline_window(q.area, (ny, nx), c), ny),
        Vector{Tuple{SVector{16, Int}, SVector{16, Float64}}}(undef, length(q.fit_charts)), q.fit_charts
    )
    B = [p[k] for p in q.fit_px, k in 1:2]
    controls, λ = _penalized_solve(rows, B, ny, nx, Val(2))
    spline = ChartSpline(q.area, controls)
    err = norm.(spline(q.held_charts) .- q.held_px)
    return (; spline, h, λ, counts = (nx, ny), _held_out_errors(err, q.edge)...)
end

function fit_camera(::Type{SplineProjection}, source, region; tolerance::Real, weight::Symbol = :image)
    _check_image_weight(weight)
    tolerance > 0 || throw(ArgumentError("fit_camera: tolerance must be positive, got $tolerance"))
    q = _chart_quadrature(_unbounded(source), region)
    # The bracket: from 4 cells along the chart's longer side up to a cell of
    # three quadrature strides (fewer points per cell leave it to the penalty).
    cells_max = max(1, floor(Int, q.extent / (3 * q.stride)))
    fits = 0
    meets(r) = r.max_px <= tolerance
    # Doubling to the first count that meets the tolerance, then bisection
    # between it and the last that did not: the largest spacing that meets it.
    lower = 0
    cells = min(4, cells_max)
    best = _spline_projection_fit(q, cells)
    fits += 1
    while !meets(best)
        lower = cells
        if cells == cells_max
            return _spline_projection_result(q, best, :max_iterations, fits, tolerance)
        end
        cells = min(2cells, cells_max)
        best = _spline_projection_fit(q, cells)
        fits += 1
    end
    upper = cells
    while upper - lower > 1
        mid = (lower + upper) ÷ 2
        r = _spline_projection_fit(q, mid)
        fits += 1
        if meets(r)
            upper, best = mid, r
        else
            lower = mid
        end
    end
    return _spline_projection_result(q, best, :converged, fits, tolerance)
end

_spline_projection_result(q, r, code::Symbol, fits::Int, tolerance::Real) = FitResult(
    SplineProjection(q.chart, r.spline, q.region), code, fits, r.rms_px, length(q.fit_px),
    CameraFitMetadata(
        q.region, :image, r.max_px, r.rms_px, r.max_px_edge,
        (; spacing = r.h, penalty = r.λ, counts = r.counts, stride = q.stride, tolerance = Float64(tolerance))
    )
)

# ------------------------------------------------------------------------------
# CentralGenericModel
# ------------------------------------------------------------------------------

function fit_camera(
        ::Type{CentralGenericModel}, source, region;
        grid_size::Tuple{Integer, Integer}, tolerance::Real, weight::Symbol = :image
    )
    _check_image_weight(weight)
    W, H = _image_size(source)
    cam = _unbounded(source)
    q = _image_quadrature(cam, region)
    # the node-interpolation constructor's model: Core's full-image lattice (the
    # calibrated area is the image; the region restricts the quadrature only),
    # the nominal pinhole, the start
    start = CentralGenericModel(
        cam, (W, H); grid_size, init = cam isa CameraModel ? :interpolate : :quasi_interpolate, domain = region
    )
    area = (start.calib_min_x, start.calib_min_y, start.calib_max_x, start.calib_max_y)
    gh, gw = size(start.grid)
    rows = pointwise!(
        p -> _design_row(spline_window(start, p), gh),
        Vector{Tuple{SVector{16, Int}, SVector{16, Float64}}}(undef, length(q.fit_px)), q.fit_px
    )
    B = [r[k] for r in q.fit_rays, k in 1:3]
    controls, λ = _penalized_solve(rows, B, gh, gw, Val(3))
    fitted = CentralGenericModel{Float64}(
        W, H, area..., Vec3{Float64}.(controls); unit_controls = false, nominal = nominal_intrinsics(start), domain = region
    )
    err = _grid_round_trip(fitted, q.held_rays, q.held_px)
    e = _held_out_errors(err, q.edge)
    e.max_px <= tolerance ||
        @warn "fit_camera: the $(gh) × $(gw) grid's held-out round trip reaches $(round(e.max_px; sigdigits = 3)) px, over the tolerance $(tolerance) px"
    return FitResult(
        fitted, :solved, 0, e.rms_px, length(q.fit_px),
        CameraFitMetadata(region, weight, e.max_px, e.rms_px, e.max_px_edge, (; penalty = λ, grid_size = (gh, gw), stride = q.stride, tolerance = Float64(tolerance)))
    )
end

# The image a grid target's lattice covers: a grid source's own, a bounded
# camera's extent. A bare camera model has no image.
_image_size(m::CentralGenericModel) = (m.width, m.height)
_image_size(bc::BoundedCamera) = (e = extent(bc); (Int(e.width), Int(e.height)))
_image_size(source) = throw(
    ArgumentError(
        "fit_camera(CentralGenericModel, …): a $(nameof(typeof(source))) source has no image for the grid's full-image lattice; " *
            "bound it by its sensor or extent, `BoundedCamera(model, sensor)`"
    )
)
_unbounded(bc::BoundedCamera) = model(bc)
_unbounded(source) = source

# A grid model's round trip at the points `px` whose source rays are `rays`: the
# batched `project` from the pixel itself, Inf where it does not converge.
function _grid_round_trip(m::CentralGenericModel, rays, px)
    out = project(m, rays; init = px, iters = 50)
    return [out.ok[i] ? norm(out.px[i] - px[i]) : Inf for i in eachindex(px)]
end

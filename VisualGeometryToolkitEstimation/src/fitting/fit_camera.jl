# ==============================================================================
# fit_camera — one model-to-model regression with no guess: a camera model of
# type T regressed onto a source camera over a region of its sensor, linearly
# ==============================================================================
#
# THE OBJECTIVE is the integrated SQUARED pixel error of the round trip over the
# region, ∫ ‖π_T(π⁻¹_source(u)) − u‖² du, by a quadrature over the region. The
# target TYPE selects the family; every family is ONE linear least squares (a
# fit has no guess):
#
#   a lens family L          the radial relation of each sample — a physical K's
#                            radial profile (the exact 1-D reduction of the
#                            integral, at azimuth 0), or a gauge K's image
#                            samples about its own principal point and ẑ — solved
#                            for the focal rescale and the family's coefficients
#                            where the family has a linear form (a lift, or a
#                            fixed projection with an even radial factor); the
#                            exact pixel minimum from a guess is refine_camera's
#                            (refine_camera.jl), the two in one approximate_camera
#   SplineLens               a CameraModel with a SplineLens: its backward spline
#                            (pixel → ray, x = K⁻¹u) on Core's full-image lattice,
#                            the source's rays by penalized linear least squares,
#                            its lattice spacing bisected to the largest one whose
#                            held-out maximum error meets `tolerance` (a SplineLens
#                            source's backward is kept as it is); then its forward
#                            spline (ray → x), a ChartSpline on the equidistant
#                            direction chart about ẑ, fitted to the
#                            camera's own rays by the same solve and the same
#                            spacing search
#   NoncentralSplineModel    the same lattice, the source's LINES: the directions
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
# accuracy at every lattice size; a larger λ only lost accuracy), here in the
# stack's one weight convention (Core's penalty_scale): `λ = β · penalty_scale(tr AᵀA, nodes, m̄)`,
# β = `_PSPLINE_BETA` = 3e-7 relative to the data's energy per node and the Δ³
# operator's mean positive eigenvalue m̄ (≈ 31 on a 12 × 10 lattice, → 40 on large
# ones): 3e-7 / m̄ ≈ the former 1e-8 · mean(diag(AᵀA)) relative weight.
#
# THE HELD-OUT CHECK of the spline families is the quadrature offset by half its
# stride over the region, and its edge band: the held-out points within
# `_EDGE_BAND_PX` of the region's edge.


# ------------------------------------------------------------------------------
# The metadata
# ------------------------------------------------------------------------------

# The rotation every fit reports, from the source's camera frame to the fitted camera's (`_source_to_fitted`): one
# concrete type, so a fit's metadata is the same type whatever its source's K.
const _FrameRotation = VgtkCore.RotationMap{3, Float64, RotMatrix{3, Float64, 9}, VgtkCore.ViewFrame{:source}, VgtkCore.ViewFrame{:fitted}}

# Core's lazy all-ones weights of an L2 result (`unit_weights`): one concrete type.
const _UnitWeights = typeof(VgtkCore.unit_weights(Float64, 0))

"""
    CameraFitMetadata{R, F}

`FitResult` metadata of [`fit_camera`](@ref), [`refine_camera`](@ref) and [`approximate_camera`](@ref): the `region`
fitted over, the `weight` measure (`:image` for every target but a lens family), the round trip's error in pixels —
`residuals` (per sample), `max_px` (their maximum) and `max_px_edge` (their maximum within 100 px of the region's
edge); √ of their weighted mean square, the objective's quantity, is the result's `rms` — the target family's own
record `family`, a `NamedTuple`, and:

- `rotation`: from the source's camera frame to the fitted camera's, a
  frame-labelled `RotationMap` (`ViewFrame{:source}` → `ViewFrame{:fitted}`; one
  type for every fit). For a source whose `K` is physical it is exactly the
  identity: its ẑ is the axis, not fitted. A gauge-`K` (SplineLens) source's
  refine gives the fitted rotation, for a lens family and for the SplineLens
  target (its axis gauge) alike; a linear fit keeps the source's ẑ.
- `symmetry_residual`: px. The root mean square of the round trip's azimuthal
  component about the fitted principal point, the misfit that no radially
  symmetric camera about the fitted axis represents. It is `NaN` where not
  measured: a physical `K`'s radial profile, a lattice fit of a physical source.
- `weights`, `scale`, `scale_source`: the robust stages' weight and scale record.
  No camera regression has a robust loss: its weights are Core's lazy
  `unit_weights` (one per sample, binary by `weight_kind`), `scale` is `NaN` and
  `scale_source` `ScaleNone`, so `noise_scale(result)` refuses to hand it on.

The family records:

- a lens family: `(; focal_scale, aspect, θ_max, θ_source, clipped)`. These are the
  joint focal rescale `s`, the aspect `fy′/fy` (1 for a physical `K` and for a
  linear fit), the fitted and the source's angular domains (radians), and whether
  the target family cut the source's domain. The errors are at the quadrature's
  samples;
- `SplineLens`: `(; λ, grid_size, stride, outer_iterations,
  n_held_out_dropped, tolerance, forward)`, `forward` being `(; spacing, counts, λ,
  outer_iterations, max_px, max_px_edge, rms)` or `nothing` —
  the backward lattice's P-spline weight `λ` (`nothing` for a SplineLens
  source's kept backward), its `(columns, rows)` (Core's `grid_size`), the quadrature's stride in
  pixels, the spacing search's fits (0 for a given or kept lattice), the
  held-out pixels dropped (below) and the tolerance asked for; then the forward
  spline's knot spacing `h` (radians in the chart), its control counts per chart
  axis, its `λ`, its search's fits and its held-out round trip
  `‖K·forward(chart(ray)) − u‖` in pixels (maximum, edge-band maximum, rms) —
  `forward` is `nothing` with `forward = false`. The result's errors
  (`residuals`, `max_px`, `max_px_edge`, `rms`) are the backward's held-out direction error,
  as the non-central target's (below), over the held-out pixels where the
  source's back-projection Jacobian is finite — `n_held_out_dropped` counts the
  others;
- `NoncentralSplineModel`: `(; λ, grid_size, stride, tolerance,
  outer_iterations, center, max_offset, rms_offset, max_offset_edge,
  n_held_out_dropped)` —
  `center` the anchor c* (the
  least-squares point nearest to the source's quadrature lines, in the
  source's frame, plain metres as the model's points), and the held-out line
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
    rotation::_FrameRotation
    symmetry_residual::Float64
    residuals::Vector{Float64}
    weights::_UnitWeights
    scale::Float64
    scale_source::VgtkCore.ScaleSource
end
# No camera regression has a robust loss: every one is L2, its weights all ones (binary by kind) and no scale.
CameraFitMetadata(region, weight, max_px, max_px_edge, family, rotation, symmetry_residual, residuals) = CameraFitMetadata(
    region, weight, Float64(max_px), Float64(max_px_edge), family, rotation, Float64(symmetry_residual), residuals,
    VgtkCore.unit_weights(Float64, length(residuals)), NaN, VgtkCore.ScaleNone
)
VgtkCore.weight_kind(::Type{<:CameraFitMetadata}) = VgtkCore.BinaryWeights()

const _EDGE_BAND_PX = 100.0
# The P-spline penalty's normalized weight β (`penalty_scale`; see the file header).
const _PSPLINE_BETA = 3.0e-7
# The radial profile's quadrature order (n + 1 Clenshaw–Curtis nodes) and the
# azimuths the region's coverage of a radius is counted over.
const _PROFILE_NODES = 128
const _REGION_AZIMUTHS = 720

"""
    fit_camera(::Type{T}, source, region; …) -> FitResult{<model of T>, CameraFitMetadata}

The camera model of type `T` regressed onto `source` over `region`, a region of
the source's sensor in its pixel frame (an [`imaged_region`](@extref
`VisualGeometryToolkitCore.Cameras.imaged_region-Tuple{BoundedCamera}`), an `Ellipse`, a `Rect`, their `∩`;
bounded), by ONE linear least squares — a fit has no guess. The objective is the
integrated squared pixel error of the round trip, `∫ ‖π_T(π⁻¹_source(u)) − u‖² du`
over the region. The target type selects the family and the solve:

- **a lens family `L <: AbstractLensModel`** (`fit_camera(L, source::CameraModel,
  region; weight = :image)`): a `CameraModel` with lens `L`, its focal rescale `s`
  (`fx′ = s·fx₀`, the skew with it) and its coefficients from the samples' radial
  relation — radius `ρ` against view angle `θ` — in one solve. The family's class
  decides the linear system:
  - **a lift** (`LiftLens`: `Division`, `Scaramuzza`, BabelCalib's rational lift): the
    lifted ray `(x·N(ρ), D(ρ))` is parallel to the sample's ray, linear in every
    coefficient and the rescale, its rows weighted to the angle residual times the
    radius to first order;
  - **a fixed projection** `g(θ)` (`Perspective`, `Equidistant`, `Stereographic`,
    `Equisolid`, `Orthographic`) with an even radial factor `N(g)/D(g)` and decentering
    deltas — Kannala–Brandt, the polynomial and the rational OpenCV models, Fisheye624:
    `ρ·D(g) = s·g·N(g)`, linear in `s`, `s·k` and `d`. Without a denominator the row's
    residual is the pixel error itself, so the fit IS the integral's least squares at a
    physical `K`'s samples; with one (the OpenCV rational model) the rows are divided by
    the denominator of a first solve, once. The decentering deltas are zero (the radial
    relation does not see them);
  - **no linear form**: a projection with a shape parameter (`DoubleSphere`, `Unified`,
    `ExtendedUnified`), a `Tilt` or `FieldOfView` term — an `ArgumentError` naming
    [`refine_camera`](@ref) and [`approximate_camera`](@ref).
  The samples depend on the source's `K`, by Core's `IntrinsicsRole`:
  - A PHYSICAL `K` (a parametric lens): the principal point and ẑ are kept. The samples are
    the source lens's radial profile `r(θ)` (its radial part: decentering terms are
    stripped, warned when non-negligible) on Clenshaw–Curtis nodes. The integral over the
    region reduces exactly to one over `θ`, with the density weighted by the fraction of
    each radius's circle about the principal point that lies in the region (the region
    must be star-shaped about the principal point, as a convex one containing it is).
    The domain is the region's farthest radius, capped by the source's and the target
    family's own view-angle limits — a cut of the source's domain is `clipped` and warned,
    and so is a region reaching past the radii the source images (its domain is then the
    source's own). (The region's extent is measured along 720 azimuths: a corner between
    two of them is reached to within half an azimuth's arc.) `weight` is the density:
    `:image` (uniform on the image), `:solid_angle` (`sin θ`) or `:uniform` (in `θ`).
  - A GAUGE `K` (a `SplineLens` camera source, a `CameraModel` or a `BoundedCamera` of one)
    has no radial profile of its own lens. The samples are the CAMERA's rays on the image
    over the region (the spline targets' quadrature, below), each pixel one sample of equal
    weight: the same image integral with the `:image` density (another `weight` is an
    `ArgumentError`). The linear fit reads their radii about the source's principal point
    and their angles about its ẑ; the principal point, the aspect and the axis are freed by
    [`refine_camera`](@ref) ([`optical_axis`](@ref)). This is the route to a parametric
    lens, an OpenCV or COLMAP form, for a SplineLens camera:
    `approximate_camera(KannalaBrandt{4}, camera, region)`.
  - The result reports the linear fit's residual: the round trip's error of the returned
    camera at each sample, in pixels (`metadata.residuals`, `max_px`, `max_px_edge`; the
    result's `rms` their weighted root mean square). The `retcode` is `:solved`, `iterations`
    0; a degenerate solve (a non-positive or non-finite rescale) is `:solve_failed` (its
    camera's focals non-finite), never a throw.
  - `Division` alone fits `Division{1}`; any other type must carry its arities
    (`KannalaBrandt{4}`). A non-central source (`NoncentralSplineModel`) is an
    `ArgumentError`: not supported yet.
- **`SplineLens`** (`fit_camera(SplineLens, source, region; tolerance,
  grid_size = nothing, forward = true, device = default_device())`): a `CameraModel` with a
  VisualGeometryToolkitCore `SplineLens`, its `K` the source's own (skew 0).
  Its BACKWARD spline is Core's full-image lattice on the source's image
  (`spline_lattice`; the source is not sampled: the solve sets every control),
  fitted to the source's rays at the normalized points `x = K⁻¹u` by linear
  least squares with the P-spline penalty — [`pspline_grid`](@ref) at
  `grid_size` (columns, rows; reported back in the metadata in that order,
  as Core's `grid_size(camera)`) when given, else at the lattice THE SPACING
  SEARCH chooses: the spacing is doubled down from 4 cells along the longer
  side and then bisected to the LARGEST spacing whose held-out maximum error is
  at most `tolerance` (px), `retcode :converged`; a fit that cannot reach it at
  the finest spacing of the bracket (a cell of three quadrature strides)
  returns `retcode :max_iterations` with what it reached. The region restricts
  the quadrature only, never the lattice: a node whose support misses it is
  held by the penalty. The tolerance bounds the held-out DIRECTION error in
  pixels, the non-central target's metric (below). A SplineLens source's
  backward is kept exactly (its controls `==`; a `grid_size` is then an
  `ArgumentError`) and only the forward is fitted. A gauge-`K` source (a
  SplineLens camera) is first turned onto its optical axis, ALWAYS: its K's
  principal point moves to the axis's pixel, every ray turns by the
  [`optical_axis`](@ref) rotation (an [`approximate_camera`](@ref); Core's `axis_gauged`, exact; the
  metadata's `rotation`, its `symmetry_residual`), and the forward is fitted
  there. A physical source's lattice takes its own K and ẑ, already the axis.
  Its FORWARD spline (`forward = true`) is a `ChartSpline{2}` from the
  equidistant direction chart about ẑ to `x`, fitted to the
  camera's own rays over the region by the same penalized solve and the same
  spacing search, to `tolerance` in pixels of the held-out round trip; the
  camera projects through it (NaN outside the fitted chart). Each fit of a
  search is an outer iteration, counted in the metadata; `iterations(result)`
  is 0; the retcode is `:converged` when every search met the tolerance.
- **`NoncentralSplineModel`** (`fit_camera(NoncentralSplineModel, source,
  region; tolerance, grid_size = nothing, device = default_device())`): the same lattice and search on the
  source's observation LINES — a non-central model's `Ray3`s, or a central
  source's rays through its centre. Both splines, the unit directions and the
  points, come from ONE penalized solve on the same lattice. A line's point is
  pinned by an anchor: c*, Core's `nearest_point` of all the quadrature lines,
  and each line contributes its point closest to c*, `o + ((c* − o)·d) d` —
  bounded for rays at and past 90°, where a plane anchor (the pupil plane) runs
  off. The points are in the source's frame (frame in = frame out; plain
  metres, the model's unit), c* in the metadata; a central source's are all
  c*. The tolerance bounds the held-out DIRECTION error in pixels: the
  direction's deviation mapped through the source's back-projection Jacobian
  (the round trip through the source to first order); the line offset is
  reported beside it, in metres.

The spline targets' `source` is a [`BoundedCamera`](@extref
VisualGeometryToolkitCore.Cameras.BoundedCamera) — the lattice is laid over its
image (a bare model has none: an `ArgumentError`) — of a `CameraModel`, and for
the non-central target also of a `NoncentralSplineModel`. Their quadrature is
on the image (they take no `weight`). Their per-point steps run as batches: a
SplineLens source's backprojection is Core's KernelAbstractions kernel on the
backend `device` ([`default_device`](@extref
VisualGeometryToolkitCore.default_device)`()` by default: an accelerator when
one is functional), its rays brought back to the host; any other source's
backprojection, the chart, the design rows, the evaluation and the errors run
on the host ([`pointwise!`](@extref VisualGeometryToolkitCore.pointwise!)), and
the penalized normal equations are solved by a sparse Cholesky factorization.

The metadata ([`CameraFitMetadata`](@ref)) records the region, the round trip's
`residuals`, `max_px` and `max_px_edge`, the family's own record and no scale
(`ScaleNone`: no camera regression has a robust loss); the result's `rms` is
the round trip's root mean square in pixels. For the spline targets these are
their held-out errors.

```julia
region = imaged_region(with_lens_coverage(camera, image_circle(camera, 8.0mm)))
kb = fit_camera(KannalaBrandt{4}, model(camera), region)                       # a parametric lens, linearly
kb = approximate_camera(KannalaBrandt{4}, model(camera), region)               # and its exact pixel least squares
sl = fit_camera(SplineLens, camera, region; tolerance = 1.0e-3)                 # its backward grid and forward spline
project(estimate(sl), ray)                                                     # its pixel through the forward, NaN outside the fitted chart
```
"""
function fit_camera end

# ==============================================================================
# A lens family: one linear least squares on the radial profile
# ==============================================================================

# The linear form of a target family, read off its type: a lift (`LiftLens`: the ray `(x·N(ρ), D(ρ))` is linear in
# every coefficient and, rescaled, in the focal), a fixed projection with at most one even radial factor and
# decentering deltas (`Lens{Perspective | Equidistant | Stereographic | Equisolid | Orthographic}`: the radius
# `s·g·N(g)/D(g)`, `g` the projection's, linear in `s`, `s·k` and `d` once multiplied by `D`), or none (a projection
# with a shape parameter, a `Tilt` or `FieldOfView` term, a second radial factor).
struct _LiftForm end
struct _ProjectionForm end
struct _NoLinearForm end
const _FixedProjection = Union{Perspective, Equidistant, Stereographic, Equisolid, Orthographic}
_linear_form(::Type{<:LiftLens}) = _LiftForm()
function _linear_form(::Type{L}) where {L <: Lens}
    leaves = VgtkCore.Cameras._leaves(zero(L))   # zero(L): an ArgumentError for a type without its arities
    first(leaves) isa _FixedProjection || return _NoLinearForm()
    terms = Base.tail(leaves)
    ok = all(t -> t isa Union{Radial{Even}, Tangential, ThinPrism}, terms) && count(t -> t isa Radial, terms) <= 1
    return ok ? _ProjectionForm() : _NoLinearForm()
end
_linear_form(::Type) = _NoLinearForm()

function fit_camera(::Type{L}, source::CameraModel, region; weight::Symbol = :image) where {L <: AbstractLensModel}
    form = _linear_form(L)
    form isa _NoLinearForm && throw(
        ArgumentError(
            "fit_camera($L, …): the family has no linear form (a projection with a shape parameter, a Tilt or a FieldOfView term); " *
                "refine a guess, `refine_camera(guess, source, region)`, or start from the family's anchor, `approximate_camera($L, source, region)`"
        )
    )
    role = VgtkCore.Cameras.IntrinsicsRole(lens(source))
    K0 = SMatrix{3, 3, Float64}(calibration_matrix(source))
    pr = _profile_samples(role, L, source, region, K0, weight)
    s, fitted = _linear_lens(form, _target_anchor(L), pr.rA, pr.angles, pr.w)
    code = :solved
    if !(isfinite(s) && s > 0)                       # a degenerate solve: reported, never a throw
        s, fitted, code = NaN, _target_anchor(L), :solve_failed
    end
    fit = (; s, a = 1.0, δ = zero(SVector{2, Float64}), R = nothing, lens = fitted)
    M = SMatrix{2, 2, Float64}(1.0, 0.0, K0[1, 2] / K0[1, 1], K0[2, 2] / K0[1, 1])   # K₀₂ / fx₀: a normalized offset in pixels / fx₀
    # the round trip's error per sample, in pixels: the forward form, its radius capped inside the fitted lens's domain
    # (a lift's, and a forward lens's whose linear fit folds before the source's domain ends), on the host
    θcap = code === :solved ? Float64(view_angle_limit(fitted)) - 1.0e-9 : Inf
    e = map((d, θ, y) -> K0[1, 1] * _pixel_offset(fit, M, d, θ, y, θcap), pr.rays, pr.angles, pr.points)
    return _lens_fit_result(role, fit, K0, M, pr, e, region, weight, code, 0)
end
# A bounded camera's model is the source (its image does not enter: the region bounds the fit).
fit_camera(::Type{L}, source::BoundedCamera, region; kwargs...) where {L <: AbstractLensModel} = fit_camera(L, model(source), region; kwargs...)
fit_camera(::Type{L}, source::NoncentralSplineModel, region; kwargs...) where {L <: AbstractLensModel} = throw(
    ArgumentError("fit_camera($L, …): a non-central source (NoncentralSplineModel) is not supported yet: a radially symmetric central camera fits a central source")
)

# The FitResult of a lens-family fit (fit_camera's linear one, refine_camera's): the camera — K₀ with its focal and
# skew scaled by `s`, `fy` by `s·a`, the principal point moved by K₀₂·δ (none for a physical K; the structural entries
# written exactly: K₀ · diag(s, s, 1) made 0·Inf = NaN in row 3 for a non-finite s, which the structure check refuses)
# — and its metadata from the round trip's errors `e` per sample (px): their weighted rms, maximum and edge-band
# maximum, their norms per sample.
function _lens_fit_result(role, fit, K0, M, pr, e, region, weight, code, iters)
    s, a, δ = fit.s, fit.a, fit.δ
    m = K0[1, 1] * (M * δ)
    Kb = CameraCalibrationMat(SMatrix{3, 3, Float64}(K0[1, 1] * s, 0, 0, K0[1, 2] * s, K0[2, 2] * s * a, 0, K0[1, 3] + m[1], K0[2, 3] + m[2], 1); check = false)
    out = CameraModel(LogicalIntrinsics(Kb), fit.lens)
    w = pr.w
    rms_px = sqrt(sum(i -> w[i] * sum(abs2, e[i]), eachindex(e)) / max(sum(w), eps()))
    residuals = map(norm, e)
    edge = pr.rA .>= pr.r_edge - _EDGE_BAND_PX / K0[1, 1]
    return FitResult(
        out, code, iters, rms_px, length(pr.rays),
        CameraFitMetadata(
            region, weight, maximum(residuals), maximum(residuals[edge]),
            (; focal_scale = s, aspect = a, θ_max = float(pr.θ_max), pr.θ_source, pr.clipped),
            _source_to_fitted(fit.R), _symmetry_residual(role, e, map(y -> M * (y - δ), pr.points), w), residuals,
        )
    )
end

# The power of a radial factor's i-th basis function: `Even` ρ²ⁱ, `Full` ρⁱ⁺¹.
_basis_degree(::Radial{Even}, i::Int) = 2i
_basis_degree(::Radial, i::Int) = i + 1

# The weighted linear least squares `min ‖diag(c)(A u − b)‖²`, its columns equilibrated (the powers of the radius span
# decades) and solved by Householder QR.
function _weighted_lstsq(A::Matrix{Float64}, b::Vector{Float64}, c::Vector{Float64})
    Aw = c .* A
    bw = c .* b
    nrm = map(j -> (n = norm(view(Aw, :, j)); n > 0 ? n : 1.0), axes(Aw, 2))
    return (qr(Aw ./ transpose(nrm)) \ bw) ./ nrm
end

# A LIFT `(x·N(ρ), D(ρ))` of the target-normalized point `x = y/s`: the ray is parallel to the sample's,
# `ρ·cos θ·N(ρ/s) = s·sin θ·D(ρ/s)` — linear in `s`, `cⱼ = dⱼ s^(1−deg ⱼ)` and `aᵢ = kᵢ s^(−deg ᵢ)`:
#   s·sin θ + Σⱼ cⱼ sin θ φⱼ(ρ) − Σᵢ aᵢ ρ cos θ φᵢ(ρ) = ρ cos θ.
# Its residual is |ray|·sin(θ_B − θ) with |ray| = ρN / sin θ_B, so a row weighted by sin θ is the angle residual times
# the radius (refine_camera's lift residual) to first order where N ≈ 1 (exactly the division and Scaramuzza
# models, whose numerator is 1). The focal rescale and the lens.
function _linear_lens(::_LiftForm, a::LiftLens, ρ, θ, w)
    r = a.radial
    P, Q = length(r.k), length(r.d)
    A = zeros(length(ρ), 1 + Q + P)
    b = zeros(length(ρ))
    c = zeros(length(ρ))
    for i in eachindex(ρ)
        sθ, cθ = sincos(θ[i])
        A[i, 1] = sθ
        for j in 1:Q
            A[i, 1 + j] = sθ * ρ[i]^_basis_degree(r, j)
        end
        for k in 1:P
            A[i, 1 + Q + k] = -ρ[i] * cθ * ρ[i]^_basis_degree(r, k)
        end
        b[i] = ρ[i] * cθ
        c[i] = sqrt(w[i]) * sθ
    end
    u = _weighted_lstsq(A, b, c)
    s = u[1]
    k = ntuple(i -> u[1 + Q + i] * s^_basis_degree(r, i), P)
    d = ntuple(j -> u[1 + j] * s^(_basis_degree(r, j) - 1), Q)
    return s, _linear_with_coefficients(a, s, k, d)
end

# A FIXED PROJECTION `g(θ)` with the radial factor `N(g)/D(g)` (none: N = D = 1) and its decentering deltas held at
# zero: the radius `ρ = s·g·N(g)/D(g)`, so `ρ·D(g) = s·g·N(g)` — linear in `s`, `bᵢ = s·kᵢ` and `dⱼ`:
#   s·g + Σᵢ bᵢ g·g²ⁱ − Σⱼ dⱼ ρ g²ʲ = ρ.
# Without a denominator its residual is the pixel error itself (exactly the integral's least squares at a physical
# K's samples); with one it is the pixel error times D, so the rows are divided by the denominator of a first solve
# (one reweighting, Steiglitz–McBride's; the second solve is the fit). The focal rescale and the lens.
function _linear_lens(::_ProjectionForm, a::Lens, ρ, θ, w)
    r = VgtkCore.Cameras._bounding_radial(a)
    P, Q = r === nothing ? (0, 0) : (length(r.k), length(r.d))
    proj = Lens(a.projection)
    g = map(t -> Float64(projected_radius(proj, t)), θ)
    A = zeros(length(ρ), 1 + P + Q)
    for i in eachindex(ρ)
        A[i, 1] = g[i]
        for k in 1:P
            A[i, 1 + k] = g[i]^(2k + 1)
        end
        for j in 1:Q
            A[i, 1 + P + j] = -ρ[i] * g[i]^(2j)
        end
    end
    b = collect(Float64, ρ)
    c = sqrt.(w)
    u = _weighted_lstsq(A, b, c)
    if Q > 0
        D = map(gi -> 1 + sum(j -> u[1 + P + j] * gi^(2j), 1:Q), g)
        all(x -> isfinite(x) && x > 0, D) && (u = _weighted_lstsq(A, b, c ./ D))
    end
    s = u[1]
    return s, _linear_with_coefficients(a, s, ntuple(k -> u[1 + k] / s, P), ntuple(j -> u[1 + P + j], Q))
end

# The anchor lens `a` with its radial factor's coefficients `k` (numerator) and `d` (denominator) by name, every other
# coefficient (the decentering deltas) at zero.
function _linear_with_coefficients(a, s, k::Tuple, d::Tuple)
    (isfinite(s) && s > 0) || return a
    names = parameter_names(a)
    θ = map(names) do n
        str = String(n)
        m = match(r"^([kd])(\d+)$", str)
        m === nothing && return 0.0
        i = parse(Int, m.captures[2])
        return m.captures[1] == "k" ? k[i] : d[i]
    end
    return with_parameters(a, SVector{length(θ), Float64}(θ))
end

# The samples of the regression, by the role of the source's K (Core's `IntrinsicsRole`): per sample its ray in the
# source's camera frame and that ray's view angle about ẑ, its pixel's point `y = K₀⁻¹u` of the source's normalized
# plane, its weight (the quadrature's × the density), and its normalized radius about the principal point `rA = |y|`
# (the edge band's); the fitted and the source's domains, whether the family or the region cut it, and the radius of
# the domain's edge.
#
# A PHYSICAL K (a parametric lens): its principal point and its ẑ are data, so the samples are the lens's own radial
# profile r(θ) — its radial part, decentering stripped — on the Clenshaw–Curtis nodes, the exact reduction of the
# image integral to θ, each at azimuth 0: the ray `(sin θ, 0, cos θ)` and the point `(r(θ), 0)`.
function _profile_samples(::VgtkCore.Cameras.FreeIntrinsics, ::Type{L}, source, region, K0, weight::Symbol) where {L}
    src = lens(source)
    # Core's `RadialSymmetry` trait says whether anything is non-radial;
    # `_decentering` measures how much (a symmetric ∘ composition is not measured)
    dec = _decentering(VgtkCore.Cameras.RadialSymmetry(src), src)
    dec > 1.0e-8 &&
        @warn "fit_camera: the source $(nameof(typeof(src))) carries non-radial terms (decentering, norm $(round(dec; sigdigits = 3))) — no radially symmetric target can represent them; fitting the radial part only"
    srcr = radial_part(src)

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
    θ_max, cut = _family_domain(L, θ_source)
    clipped = past_source || cut

    θs, qs, rA, dens = _forward_samples(srcr, float(θ_max), _PROFILE_NODES, weight)
    w = qs .* (dens .* map(r -> _coverage(exits, r), rA))
    rays = map(θ -> SVector(sin(θ), 0.0, cos(θ)), θs)
    points = map(r -> SVector(r, 0.0), rA)
    return (; rays, angles = θs, points, w, rA, θ_max, θ_source, clipped, r_edge = rA[end])
end

# A GAUGE K (a SplineLens camera): its K is a choice of the normalized plane, so its lens has no radial profile of its
# own, and its principal point and ẑ are where its frame put them, not its optical axis. The samples are the camera's
# own rays on the image over the region (the spline targets' quadrature, the camera's batched `backproject`), each
# pixel one sample of equal weight — the same image integral, the `:image` density by construction (the pixels are
# uniform on the image) and the region's coverage exactly; the refine frees the principal point and the axis
# (refine_camera.jl's `_regression_manifold`), the linear fit reads the radii about the source's principal point and the
# angles about its ẑ. The family's domain cap reads the view angle about the source's ẑ, the refine's start.
function _profile_samples(::VgtkCore.Cameras.GaugeIntrinsics, ::Type{L}, source, region, K0, weight::Symbol) where {L}
    weight === :image || throw(
        ArgumentError("fit_camera: a $(nameof(typeof(lens(source)))) source is sampled on the image, whose density is :image; got weight = :$weight")
    )
    q = _image_quadrature(source, region)
    θ_all = map(r -> angle_between(r, SVector(0.0, 0.0, 1.0)), q.fit_rays)   # the ray's off-axis angle, the stack's one angle
    θ_source = maximum(θ_all)
    θ_max, clipped = _family_domain(L, θ_source)
    keep = findall(<=(θ_max), θ_all)
    points = _normalized_points(source.intrinsics, q.fit_px[keep])
    rA = map(norm, points)
    return (; rays = q.fit_rays[keep], angles = θ_all[keep], points, w = fill(1 / length(keep), length(keep)), rA, θ_max, θ_source, clipped, r_edge = maximum(rA))
end

# The rotation from the source's camera frame to the fitted camera's, frame-labelled (`ViewFrame{:source}` →
# `ViewFrame{:fitted}`): the fitted R, or exactly the identity where the axis is the source's (a physical K). One type.
_source_to_fitted(::Nothing) = _source_to_fitted(SMatrix{3, 3, Float64}(I))
_source_to_fitted(R::SMatrix{3, 3, Float64}) =
    VgtkCore.relabel(VgtkCore.ViewFrame{:source}, VgtkCore.ViewFrame{:fitted}, VgtkCore.RotationMap(RotMatrix{3, Float64}(R); check = false))

# The symmetry residual of a fit (px): the root mean square of the round trip's AZIMUTHAL component about the fitted
# principal point — the misfit no radially symmetric camera about the fitted axis represents (a radial misfit is the
# family's own). A physical K's samples are its radial profile, which has no azimuth: not measured, NaN.
# `offsets` are the samples' pixel offsets from the fitted principal point, in units of fx₀ (their direction is what is read).
_symmetry_residual(::VgtkCore.Cameras.FreeIntrinsics, e, offsets, w) = NaN
function _symmetry_residual(::VgtkCore.Cameras.GaugeIntrinsics, e, offsets, w)
    a = map(eachindex(e)) do i
        v = offsets[i]
        n = norm(v)
        n > 0 ? (e[i][2] * v[1] - e[i][1] * v[2]) / n : 0.0          # e · (the azimuth's unit vector, v rotated by 90°)
    end
    return sqrt(sum(i -> w[i] * a[i]^2, eachindex(a)) / sum(w))
end

# `Division` alone names the family, not a degree: fit the one-coefficient model
# (approximate_camera's forward too, refine_camera.jl),
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
# quadrature of the integral: the linear fit's rows and the refine's LM on e
# minimize the integral itself.
# The nodes cluster at both endpoints, where profiles diverge worst;
# Clenshaw–Curtis (not Gauss–Chebyshev, whose native weight is 1/√(1−x²))
# carries the weights of the plain dθ integral and is exponentially accurate
# on smooth profiles. The region enters as w's factor `_coverage` (above).
#
# NON-RADIAL TERMS: tangential/thin-prism decentering has no radial
# representation in any symmetric target. A SOURCE contributes its radial part
# (decentering stripped, warned when non-negligible); a forward TARGET's linear
# fit solves its radial factor's numerator and denominator with its decentering
# terms at zero, and its refine moves its projection's coefficients and that
# factor, its decentering held; a lift TARGET moves every coefficient.

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
# thin-prism and other non-radial terms): the magnitude of what Core's
# `RadialSymmetry` trait flags, read only for a lens it calls asymmetric (for a
# symmetric `∘` composition `radial_part` is the bare projection, so this
# would count its radial coefficients).
_decentering(::VgtkCore.Cameras.RadiallySymmetric, l::AbstractLensModel) = 0.0
_decentering(_, l::AbstractLensModel) = sqrt(max(sum(abs2, parameter_vector(l)) - sum(abs2, parameter_vector(radial_part(l))), 0.0))

# The target's anchor: its zero lens (the double sphere at ξ = α = ½, inside the logit's interval) — the start
# approximate_camera refines from for a family with no linear form, and the lens the linear fit fills in.
_target_anchor(::Type{L}) where {L} = zero(L)
_target_anchor(::Type{L}) where {L <: Lens{<:DoubleSphere}} = (z = zero(L); Lens(DoubleSphere(0.5, 0.5), z.distortion))

# The fitted domain of the target family `L` on a source's domain [0, θ_source]:
# capped by the family's limit, and whether that cut it (warned).
function _family_domain(::Type{L}, θ_source) where {L}
    θ_max = min(θ_source, _family_limit(L))
    θ_max > 0.02 || throw(ArgumentError("fit_camera: empty fit domain (θ_max = $θ_max)"))
    cut = θ_max < θ_source * (1 - 1.0e-6)
    cut && @warn "fit_camera: target family $(L) cannot represent the source's field — fitting θ ∈ [0, $(round(rad2deg(θ_max); digits = 1))°] of the source's $(round(rad2deg(θ_source); digits = 1))°"
    return θ_max, cut
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
# ForwardDiff through the source's `projected_radius`). The radii are the plain
# `projected_radius` — the regression's targets, bit for bit what a lens of the
# source's own family gives back — and the Dual evaluation runs for `:image`
# only, for its dr/dθ.
function _forward_samples(srcr, b::Float64, n::Int, weight::Symbol)
    weight in (:image, :solid_angle, :uniform) ||
        throw(ArgumentError("fit_camera: weight must be :image, :solid_angle or :uniform, got :$weight"))
    ts, qs = _clenshaw_curtis(n, b)
    rA = map(t -> Float64(projected_radius(srcr, t)), ts)
    dens = if weight === :uniform
        ones(length(ts))
    elseif weight === :solid_angle
        sin.(ts)
    else
        map((r, t) -> max(r * abs(last(_radius_with_jacobian(srcr, t))), 0.0), rA, ts)
    end
    return ts, qs, rA, dens
end
# The source's radius r(θ) and dr/dθ, exact, from ONE forward-mode Dual
# evaluation of its `projected_radius` (the value part is the plain radius to
# its rounding: a lens's Dual path may differ from its plain one in the last
# bits, measured ≤ 2 ulp on the lens-family cases, so only dr/dθ is used).
function _radius_with_jacobian(srcr, θ::Float64)
    T = typeof(ForwardDiff.Tag(_radius_with_jacobian, Float64))
    r = projected_radius(srcr, ForwardDiff.Dual{T}(θ, 1.0))
    return Float64(ForwardDiff.value(r)), Float64(ForwardDiff.partials(r, 1))
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
# The spline targets: a quadrature on the image, penalized linear least squares
# ==============================================================================

# The quadrature of the region: the pixel centres (COLMAP) in the region on a
# lattice of `stride` pixels (Core's `pixel_centres`) — the fit's points, and the
# held-out check's offset by half a stride — with the source's rays there (a
# point the source does not image is dropped) and the held-out points' edge band.
#
# `samples(source, pixels)` is the source's datum per pixel, NaN where it images
# none: its rays (`_source_rays`, the default, on the KernelAbstractions backend
# `device`) or its observation lines (`_source_lines`); the result's `fit_rays` and
# `held_rays` hold those data.
_image_quadrature(source, region; device = default_device()) = _image_quadrature(_rays_on(device), source, region)
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

# The source's rays at `pixels`: Core's batched `backproject` on the backend `device`
# (one kernel for any camera with a device form; the rays come back to the host as
# `SVector{3}`s), NaN where it images none.
_source_rays(source, pixels, device) = backproject(source, pixels; device)
# the samples of the quadrature on `device`: the rays, or the lines (below)
_rays_on(device) = (source, pixels) -> _source_rays(source, pixels, device)
_lines_on(device) = (source, pixels) -> _source_lines(source, pixels, device)

# The source's observation lines at `pixels`, `(origin, unit direction)`, NaN where
# it images none: a non-central model's `Ray3`s (plain metres), and for a central
# source its rays through the projection centre, the origin.
function _source_lines(source::NoncentralSplineModel, pixels, _)
    return map(pixels) do p
        r = backproject(source, p)                       # NaN where the source images nothing
        (SVector{3, Float64}(r.origin), SVector{3, Float64}(r.direction))
    end
end
_source_lines(source, pixels, device) = map(r -> (zero(SVector{3, Float64}), normalize(r)), _source_rays(source, pixels, device))

_is_finite_sample(r::StaticVector) = all(isfinite, r)
_is_finite_sample(l::Tuple) = all(_is_finite_sample, l)

# Whether a point lies within `_EDGE_BAND_PX` of the region's edge: a point at
# that distance on one of 16 directions leaves the region (a convex region's
# distance to its edge, to the 16-gon's chord).
_in_edge_band(region, p) = any(e -> Point2{Float64}((p + _EDGE_BAND_PX * e)...) ∉ region, _EDGE_DIRECTIONS)
const _EDGE_DIRECTIONS = ntuple(k -> SVector(cospi((k - 1) / 8), sinpi((k - 1) / 8)), 16)

# The normalized points `x = K⁻¹u` of the pixels `pixels`: Core's one pixel →
# normalized map (`_normalized_point`, what a camera's `backproject` applies), so a
# pixel's lattice coordinate in the fit is the one the fitted camera evaluates.
_normalized_points(K, pixels::AbstractVector) = map(u -> SVector{2, Float64}(VgtkCore.Cameras._normalized_point(K, SVector{2, Float64}(u))), pixels)

# One sample's design row: the 16 controls of its spline window and their
# weights, at their nodes' linear indices in a `ny`-row control matrix — the
# column-major node order of Core's control vector (`_control_vector`), the
# window's columns outer, its rows inner: increasing, as a CSC column needs.
@inline function _design_row(window, ny::Int)
    ix, iy, wx, wy = window
    # control [iy - 3 + r, ix - 3 + c] (spline_window's layout) at linear index (column − 1)·ny + row
    idx = SVector{16, Int}(ntuple(k -> (ix - 3 + (k - 1) ÷ 4) * ny + iy - 3 + (k - 1) % 4 + 1, Val(16)))
    w = SVector{16, Float64}(ntuple(k -> wx[(k - 1) ÷ 4 + 1] * wy[(k - 1) % 4 + 1], Val(16)))
    return (idx, w)
end

# The design rows of the points `pts` on the `ny × nx` lattice over `area`.
_design_rows(area, (ny, nx), pts) = pointwise!(
    p -> _design_row(spline_window(area, (ny, nx), p), ny), Vector{Tuple{SVector{16, Int}, SVector{16, Float64}}}(undef, length(pts)), pts
)

# The right-hand side of a solve: one row per sample, one column per component.
_rhs_matrix(vs::AbstractVector{<:StaticVector{K}}) where {K} = [v[k] for v in vs, k in 1:K]

# The P-spline least squares of an `ny × nx` lattice: the controls minimizing
# ‖A c − B‖² + λ(‖Dₓ³c‖² + ‖Dᵧ³c‖²) for the design rows `rows` and the targets `B`
# (one column per component), λ = `_PSPLINE_BETA` · penalty_scale(tr AᵀA, nodes, m̄ of Δ³); the
# controls as an `ny × nx` matrix of `SVector{K}` (read back through Core's one
# control-vector layout, `_with_control_vector`) and λ.
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
    D = difference_rows((ny, nx), 3)   # Core's Δ³ rows along both lattice axes
    λ = _PSPLINE_BETA * penalty_scale(sum(diag(AtA)), size(AtA, 1), _lattice_m̄(ny, nx, D))
    M = AtA + λ * (transpose(D) * D)
    AtB = Matrix(At * B)
    F = cholesky(Symmetric(M); check = false)
    X = issuccess(F) ? F \ AtB : lu(M) \ AtB   # a pivot lost to rounding: the sparse LU, never a dense copy
    controls = VgtkCore.Cameras._with_control_vector(fill(zero(SVector{K, Float64}), ny, nx), nothing, vec(permutedims(X)))
    return controls, λ
end

# The third-difference penalty and the cubic window need at least 4 nodes per axis
# (ChartSpline's own bound).
_check_lattice_size(ny::Integer, nx::Integer) = (ny >= 4 && nx >= 4) || throw(
    ArgumentError("fit_camera: a uniform cubic lattice needs at least 4 × 4 nodes, got $ny × $nx")
)

# An error at the held-out points (pixels, or metres for a line offset): its
# maximum, rms and edge-band maximum, and the errors themselves.
function _held_out_errors(err::AbstractVector{Float64}, edge::AbstractVector{Bool})
    return (; max = maximum(err), rms = sqrt(sum(abs2, err) / length(err)), max_edge = any(edge) ? maximum(err[edge]) : 0.0, values = err)
end

_check_tolerance(tolerance::Real) =
    tolerance > 0 || throw(ArgumentError("fit_camera: tolerance must be positive, got $tolerance"))

# The spacing search's bracket: from 4 cells along the longer side (of the
# chart, or of the image) up to a cell of three quadrature strides (fewer
# points per cell leave it to the penalty).
_max_cells(extent::Real, stride::Integer) = max(1, floor(Int, extent / (3 * stride)))

# THE spacing search of the spline targets (the backward lattice and the forward
# spline): `fit(cells)` the fit with `cells` knot cells along the longer side,
# its held-out maximum `errors.max`. Doubling from 4 cells to the first count that
# meets `tolerance`, then bisection between it and the last that did not: the
# LARGEST spacing that meets it, `:converged`; `:max_iterations` with what the
# finest spacing reached when `cells_max` does not meet it. Each fit is an outer
# iteration (the fit's inputs change, the estimator repeats); their count is the
# third value.
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
# The backward lattice: a SplineLens camera's, or a NoncentralSplineModel's
# ------------------------------------------------------------------------------
#
# One body for both targets: the quadrature of the source's data, then (unless
# the lattice is given) the spacing search over Core's full-image lattice on the
# source's image (`spline_lattice`: the calibrated area is the image, K the
# source's own; the region restricts the quadrature only) — at each spacing the
# design rows on the normalized points x = K⁻¹u, ONE penalized solve, the model
# and the held-out errors. The target type selects the data (`_grid_samples`),
# the right-hand side, the model built from the controls and the errors.

# What every lattice of one fit shares: the source's image, its model, the
# lattice's K and normalized area (the same for every grid size: the image and
# K are), the quadrature and its normalized points, the right-hand side and its
# anchor, and the held-out check's source side (`_held_out`).
function _grid_data(::Type{G}, source, region, device) where {G}
    size = _image_size(source)
    cam = model(source)
    q = _image_quadrature(_grid_samples(G, device), cam, region)
    rhs, anchor = _grid_rhs(G, q)
    base = spline_lattice(SplineLens, _lattice_source(cam), size; grid_size = (4, 4))
    K = base.intrinsics
    held = _held_out(G, cam, q, K, device)
    return (;
        size, cam, q, K, area = base.lens.backward.area, xs = _normalized_points(K, q.fit_px), held,
        B = _rhs_matrix(rhs), anchor, region, ncomp = Val(length(eltype(rhs))),
    )
end
# The camera whose layout (K, the image's lattice) a fit takes: a central camera
# as it is, a non-central model's central part (the same K).
_lattice_source(cam::CameraModel) = cam
_lattice_source(m::NoncentralSplineModel) = central_model(m)

# The P-spline solve on one lattice of `grid_size` (columns, rows) — the design
# rows, ONE penalized solve, the model and its held-out errors: the one
# implementation `pspline_grid` runs once and `fit_camera`'s spacing search at
# every spacing it tries.
function _pspline_grid(::Type{G}, g, grid_size) where {G}
    gw, gh = Int.(Tuple(grid_size))
    rows = _design_rows(g.area, (gh, gw), g.xs)
    controls, λ = _penalized_solve(rows, g.B, gh, gw, g.ncomp)
    fitted = _grid_model(G, g, controls)
    errors, family = _grid_errors(fitted, g.held, g.anchor)
    return (; fitted, errors, family = (; family..., n_held_out_dropped = g.held.n_dropped), λ, grid_size = (gw, gh))
end
# The lattice the spacing search chose (or `grid_size`, when given: no search,
# the tolerance read off it): the solve, its retcode and its outer iterations.
function _backward_search(::Type{G}, g, grid_size, tolerance::Real) where {G}
    if grid_size !== nothing
        r = _pspline_grid(G, g, grid_size)
        return r, (r.errors.max <= tolerance ? :converged : :max_iterations), 0
    end
    W, H = Int(g.size.width), Int(g.size.height)
    return _spacing_search(cells -> _pspline_grid(G, g, _lattice_grid_size(W, H, cells)), tolerance, _max_cells(max(W, H), g.q.stride))
end

# (a lattice fit keeps the source's frame: the identity rotation; it measures no symmetry)
_grid_metadata(r, g, extra::NamedTuple) = CameraFitMetadata(
    g.region, :image, r.errors.max, r.errors.max_edge,
    (; λ = r.λ, grid_size = r.grid_size, stride = g.q.stride, extra..., r.family...), _source_to_fitted(nothing), NaN, r.errors.values
)

"""
    pspline_grid(T, source, region, grid_size; device = default_device()) -> FitResult{<:model of T, CameraFitMetadata}

The P-spline (penalized B-spline least squares, Eilers–Marx) of a spline
camera's lattice onto `source` over `region`, on the GIVEN lattice: `T` is
`SplineLens` (a `CameraModel` with a [`SplineLens`](@extref
VisualGeometryToolkitCore.Cameras.SplineLens), its backward spline, no forward)
or `NoncentralSplineModel` (its direction and point splines). The lattice is
Core's full-image layout of `grid_size = (columns, rows)` nodes on the source's
image (`spline_lattice`, the source's own `K`), the region restricting the
quadrature only, ONE penalized solve with the third-difference penalty
`fit_camera` uses — the non-central model's points anchored on c*, its
directions and points one right-hand side — in the source's frame. No spacing
search and no tolerance: the one deterministic least-squares step
`fit_camera(T, source, region; tolerance)` repeats at each spacing it tries
(the same implementation), so `pspline_grid(T, source, region, (gw, gh))` at
the lattice `fit_camera` chose is its backward.

`source` is a [`BoundedCamera`](@extref
VisualGeometryToolkitCore.Cameras.BoundedCamera) (the lattice needs the image:
a bare model is an `ArgumentError`) of a `CameraModel` or, for the non-central
target, also of a `NoncentralSplineModel`. The result's `retcode` is `:solved`,
`iterations` 0; its metadata is `fit_camera`'s, the held-out errors (the
direction error in pixels; for the non-central model also the line offsets in
metres, and `center`) and `family = (; λ, grid_size, stride,
n_held_out_dropped, …)` with `grid_size` the lattice's `(columns, rows)`, as
`fit_camera` reports it, and no `tolerance` or `outer_iterations`.
"""
function pspline_grid(
        ::Type{G}, source, region, grid_size::Tuple{Integer, Integer}; device = default_device()
    ) where {G <: Union{SplineLens, NoncentralSplineModel}}
    g = _grid_data(G, source, region, device)
    r = _pspline_grid(G, g, grid_size)
    return FitResult(r.fitted, :solved, 0, r.errors.rms, length(g.q.fit_px), _grid_metadata(r, g, (;)))
end

# The mean positive eigenvalue of a full `ny × nx` lattice's Δ³ operator `D` (its DᵀD), once per
# lattice size: its rank is a sparse QR, and the spacing search fits the same sizes again.
const _LATTICE_M̄ = Dict{Tuple{Int, Int}, Float64}()
const _LATTICE_M̄_LOCK = ReentrantLock()
_lattice_m̄(ny::Integer, nx::Integer, D) = lock(() -> get!(() -> mean_positive_eigenvalue(D), _LATTICE_M̄, (Int(ny), Int(nx))), _LATTICE_M̄_LOCK)

# The spacing search's lattice (the backward's and the forward's): `cells` cells along
# the longer of the widths `(w₁, w₂)`, the spacing h = max(w) / cells, and Core's node
# counts at it (`_node_counts`: ⌈wᵢ / h⌉ + 3 per axis) — the spacing and the counts
# `(n₁, n₂)` along the two axes (x, y).
function _knot_counts(w1::Real, w2::Real, cells::Integer)
    h = max(w1, w2) / cells
    return h, VgtkCore.Cameras._node_counts(w1, w2, h)
end
# The image's lattice of `cells` cells along its longer side, as Core's
# `grid_size` (columns, rows).
_lattice_grid_size(W::Integer, H::Integer, cells::Integer) = last(_knot_counts(W, H, cells))

# The image a spline target's lattice covers: a bounded camera's extent. A bare
# camera model has none.
_image_size(bc::BoundedCamera) = (e = extent(bc); VgtkCore.Size2(Int(e.width), Int(e.height)))
_image_size(source) = _no_image(source)
_no_image(source) = throw(
    ArgumentError(
        "fit_camera(<spline target>, …): a $(nameof(typeof(source))) source has no image for the lattice's full-image layout; " *
            "bound it by its sensor or extent, `BoundedCamera(model, sensor)`"
    )
)
# A function of the pixel has neither an image nor a K: its route is the SplineLens
# camera of the function (Core's source constructor, which gauges K), bounded.
_no_image(::Function) = throw(
    ArgumentError(
        "fit_camera(<spline target>, …): a function of the pixel has no image and no K; fit the SplineLens camera of it, " *
            "`fit_camera(SplineLens, BoundedCamera(CameraModel(SplineLens, f, size; init = :quasi_interpolate), size), region; tolerance)`"
    )
)

_grid_samples(::Type{SplineLens}, device) = _rays_on(device)
_grid_rhs(::Type{SplineLens}, q) = (q.fit_rays, nothing)
# The camera of the solved controls: Core's one SplineLens constructor, on the
# image's lattice with the lattice's K (the controls kept as solved).
_grid_model(::Type{SplineLens}, g, controls) = CameraModel(SplineLens, controls, g.size; intrinsics = g.K)
_held_out(::Type{SplineLens}, source, q, K, device) = _held_out(source, q, K, normalize.(q.held_rays), nothing, device)
_grid_errors(fitted::CameraModel, held, _) = (_held_out_errors(_direction_errors(fitted, held), held.edge), (;))

# ------------------------------------------------------------------------------
# NoncentralSplineModel
# ------------------------------------------------------------------------------
#
# The same design as the central lattice, on lines: Core's full-image lattice,
# the region restricting the quadrature only, ONE penalized solve for both
# splines (the unit directions and the points are columns of one right-hand side
# over one design). A line's point is not data — any point of the line is the
# same line — so the fit pins it: every quadrature line contributes its point
# closest to c*, the least-squares point nearest to all of them (Core's
# `nearest_point`). The points then vary as little as the lines allow, and a
# central source's are c* itself. Anchoring on a plane instead (the pupil plane
# z = const) explodes for rays near and past 90°, where the plane meets the
# line far away or never.

"""
    fit_camera(NoncentralSplineModel, source, region; tolerance, grid_size = nothing)
        -> FitResult{<:NoncentralSplineModel, CameraFitMetadata}

See [`fit_camera`](@ref): the non-central spline model's direction and point
splines on one lattice, from one penalized solve.
"""
function fit_camera(
        ::Type{NoncentralSplineModel}, source, region;
        tolerance::Real, grid_size::Union{Nothing, Tuple{Integer, Integer}} = nothing, device = default_device()
    )
    _check_tolerance(tolerance)
    g = _grid_data(NoncentralSplineModel, source, region, device)
    r, code, n = _backward_search(NoncentralSplineModel, g, grid_size, tolerance)
    return FitResult(
        r.fitted, code, 0, r.errors.rms, length(g.q.fit_px),
        _grid_metadata(r, g, (; tolerance = Float64(tolerance), outer_iterations = n))
    )
end

_grid_samples(::Type{NoncentralSplineModel}, device) = _lines_on(device)

# The right-hand side: the unit direction, and the line's point closest to c*
# relative to c* (the solve is translation-equivariant: a B-spline's weights sum
# to one and the third differences vanish on constants); c* the anchor.
function _grid_rhs(::Type{NoncentralSplineModel}, q)
    center = nearest_point([o => d for (o, d) in q.fit_rays])
    return [vcat(d, o + dot(center - o, d) * d - center) for (o, d) in q.fit_rays], center
end

# A control of the joint solve split into its direction and its absolute point.
_direction_control(x::SVector{6}) = SVector{3, Float64}(x[1], x[2], x[3])
_point_control(x::SVector{6}, center) = SVector{3, Float64}(x[4] + center[1], x[5] + center[2], x[6] + center[3])

_grid_model(::Type{NoncentralSplineModel}, g, controls) = NoncentralSplineModel(
    g.K, ChartSpline(g.area, _direction_control.(controls)), ChartSpline(g.area, _point_control.(controls, Ref(g.anchor)))
)
_held_out(::Type{NoncentralSplineModel}, source, q, K, device) = _held_out(source, q, K, last.(q.held_rays), first.(q.held_rays), device)
function _grid_errors(fitted::NoncentralSplineModel, held, center)
    e, offsets = _line_errors(fitted, held)
    return e, (; center, max_offset = offsets.max, rms_offset = offsets.rms, max_offset_edge = offsets.max_edge)
end

# The fitted direction and point splines at a normalized point, continued past
# their area as the fit's design is.
_fitted_direction(cam::CameraModel, x) = cam.lens.backward(x)
_fitted_direction(m::NoncentralSplineModel, x) = m.directions(x)

# The held-out DIRECTION error of a fitted lattice (either target) in pixels: at
# each held-out pixel the fitted direction's tangent deviation from the source's
# unit direction `d`, mapped back through the source's back-projection Jacobian
# `J = ∂d/∂u` (its `backproject_with_jacobian`), `‖J⁺δ‖` — the round trip
# through the source to first order. Evaluated directly through the fitted
# spline at x = K⁻¹u (continued past the calibrated area as the fit's design
# is): no projection, no seed, so no solver tolerance floors it.
#
# Everything of the check but the fitted spline is the source's and is
# computed ONCE per fit (`_held_out`), not per spacing: the source's
# back-projection Jacobian at each held-out pixel and its pseudo-inverse
# `J⁺ = (JᵀJ)⁻¹Jᵀ` (2 × 3), the unit directions, the normalized points and the
# one mask of the pixels kept. A held-out pixel where the source's Jacobian is
# not finite (a source that images the ray but cannot differentiate there) has
# no pixel error: it is dropped from every held-out error — the direction's and
# the line offset's — as a non-finite sample is from the quadrature, and
# counted (`n_dropped`). Each spacing then evaluates the fitted direction and
# one 2 × 3 product per kept pixel (`pointwise!`).
function _held_out(source, q, K, directions, origins, device)
    J = map(last, backproject_with_jacobian(source, q.held_px; device))   # Core's batched form on `device`
    kept = map(j -> all(isfinite, j), J)
    any(kept) || throw(ArgumentError("fit_camera: the source's back-projection Jacobian is not finite at any held-out pixel"))
    pinv = map(j -> SMatrix{2, 3, Float64}((transpose(j) * j) \ transpose(j)), J[kept])
    px = q.held_px[kept]
    return (;
        px, xs = _normalized_points(K, px), directions = directions[kept], origins = origins === nothing ? nothing : origins[kept],
        pinv, edge = q.edge[kept], n_dropped = count(!, kept),
    )
end
_direction_errors(fitted, held) = pointwise!(
    (x, d, P) -> _direction_error(fitted, x, d, P), Vector{Float64}(undef, length(held.px)), held.xs, held.directions, held.pinv
)
@inline function _direction_error(fitted, x, d, P)
    f = normalize(SVector{3, Float64}(_fitted_direction(fitted, x)))
    δ = f - d
    δ -= dot(δ, d) * d
    return norm(P * δ)
end

# The held-out errors of a fitted non-central model against the source's lines,
# over the kept held-out pixels: the direction error in pixels (above) and the
# LINE OFFSET `‖(I − ddᵀ)(o_fit − o_src)‖`, the distance from the source's
# point to the fitted line's point across the source's direction, in the
# source's length unit (plain metres), through the fitted point spline.
function _line_errors(fitted::NoncentralSplineModel, held)
    offset = pointwise!(
        (x, o, d) -> _line_offset(fitted, x, o, d), Vector{Float64}(undef, length(held.px)), held.xs, held.origins, held.directions
    )
    return _held_out_errors(_direction_errors(fitted, held), held.edge), _held_out_errors(offset, held.edge)
end
# One held-out line offset at the normalized point `x`: the fitted point's offset
# from the source's `o` across the source's unit direction `d`. The dot product is
# written out: StaticArrays' `dot` is an `@simd` reduction, which LLVM may
# reassociate differently where it inlines (a batch's closure against a scalar
# call), so the same offset would differ in its last bit by call site.
@inline function _line_offset(fitted::NoncentralSplineModel, x, o, d)
    Δ = SVector{3, Float64}(fitted.points(x)) - o
    return norm(Δ - (Δ[1] * d[1] + Δ[2] * d[2] + Δ[3] * d[3]) * d)
end

# ------------------------------------------------------------------------------
# SplineLens: the backward lattice and the forward spline of one camera
# ------------------------------------------------------------------------------

function fit_camera(
        ::Type{SplineLens}, source::BoundedCamera, region;
        tolerance::Real, grid_size::Union{Nothing, Tuple{Integer, Integer}} = nothing, forward::Bool = true,
        device = default_device()
    )
    _check_tolerance(tolerance)
    back, code, family = _spline_backward(model(source), source, region, grid_size, tolerance, device)
    gauge = _axis_gauge(VgtkCore.Cameras.IntrinsicsRole(lens(model(source))), back.camera, source)
    gauge.code === :converged || (code = gauge.code)
    cam = gauge.camera
    fwd = forward ? _forward_spline(cam, region, tolerance, device) : nothing
    fwd === nothing || fwd.code === :converged || (code = fwd.code)
    out = fwd === nothing ? cam : _with_forward(cam, fwd.spline)
    ff = fwd === nothing ? nothing :
        (; spacing = fwd.r.h, counts = fwd.r.counts, λ = fwd.r.λ, outer_iterations = fwd.n,
            max_px = fwd.r.errors.max, max_px_edge = fwd.r.errors.max_edge, rms = fwd.r.errors.rms)
    return FitResult(
        out, code, 0, back.errors.rms, back.n,
        CameraFitMetadata(
            region, :image, back.errors.max, back.errors.max_edge, (; family..., tolerance = Float64(tolerance), forward = ff),
            gauge.rotation, gauge.symmetry_residual, back.errors.values,
        )
    )
end

# The backward of a SplineLens target in the frame its forward is fitted in. A GAUGE-K source's ẑ is where its frame
# put it: the camera is turned so ẑ is its optical axis and its principal point that axis's pixel (`optical_axis`,
# Core's `axis_gauged`: exact, every ray turned by R), the fit's rotation the optical axis's. A physical source's
# lattice takes its K and its ẑ, the axis already: the identity. The axis is fitted over the source's bound (its
# `imaged_region`), not the fit's region. The camera, the rotation, the symmetry residual and the optical axis fit's
# retcode.
_axis_gauge(::VgtkCore.Cameras.FreeIntrinsics, cam, source::BoundedCamera) =
    (; camera = cam, rotation = _source_to_fitted(nothing), symmetry_residual = NaN, code = :converged)
function _axis_gauge(::VgtkCore.Cameras.GaugeIntrinsics, cam, source::BoundedCamera)
    r = optical_axis(VgtkCore.with_model(source, cam))
    K = calibration_matrix(estimate(r))
    R = metadata(r).rotation
    camera = VgtkCore.axis_gauged(cam, rotation(R), K)
    return (; camera, rotation = R, metadata(r).symmetry_residual, code = retcode(r))
end

# Any other source has no image for the lattice — a bare camera model as well (it
# is not the lens-family regression's: `SplineLens` names the spline target).
fit_camera(::Type{SplineLens}, source, region; kwargs...) = _no_image(source)
fit_camera(::Type{SplineLens}, source::CameraModel, region; kwargs...) = _no_image(source)
fit_camera(::Type{SplineLens}, source::NoncentralSplineModel, region; kwargs...) = _no_image(source)

# The backward of the SplineLens target: a SplineLens source's own, kept exactly
# (its forward dropped; the held-out direction errors against it, its own), or
# the P-spline lattice of any other camera source (`grid_size`, or the spacing
# search to `tolerance`). Its camera, its errors and point count, the retcode and
# the central fields of the family.
function _spline_backward(src::CameraModel{<:VgtkCore.Cameras.AbstractIntrinsics, <:SplineLens}, source, region, grid_size, tolerance, device)
    grid_size === nothing || throw(
        ArgumentError("fit_camera(SplineLens, …): the source is a SplineLens camera, whose backward is kept as it is; a `grid_size` has no lattice to choose")
    )
    cam = CameraModel(src.intrinsics, SplineLens(src.lens.backward; present = src.lens.present))
    q = _image_quadrature(cam, region; device)
    held = _held_out(SplineLens, cam, q, cam.intrinsics, device)
    errors = _held_out_errors(_direction_errors(cam, held), held.edge)
    family = (; λ = nothing, grid_size = VgtkCore.grid_size(cam), stride = q.stride, outer_iterations = 0, n_held_out_dropped = held.n_dropped)
    return (; camera = cam, errors, n = length(q.fit_px)), :converged, family
end
function _spline_backward(::CameraModel, source, region, grid_size, tolerance, device)
    g = _grid_data(SplineLens, source, region, device)
    r, code, n = _backward_search(SplineLens, g, grid_size, tolerance)
    family = (; λ = r.λ, grid_size = r.grid_size, stride = g.q.stride, outer_iterations = n, r.family...)
    return (; camera = r.fitted, r.errors, n = length(g.q.fit_px)), code, family
end
_spline_backward(src, source, region, grid_size, tolerance, device) = throw(
    ArgumentError("fit_camera(SplineLens, …): the source must bound a central CameraModel, got a $(nameof(typeof(src))) (a non-central model fits `NoncentralSplineModel`)")
)

# The forward spline of the camera `cam` (its backward) over `region`: a
# ChartSpline{2} from the equidistant chart about ẑ (the chart the
# lens projects through) to the normalized point x, fitted to the camera's own
# rays at the quadrature's pixels by linear least squares with the P-spline
# penalty, its knot spacing by the spacing search to `tolerance` in pixels (the
# held-out round trip `‖K·forward(chart(ray)) − u‖`).
# The SplineLens camera `cam` with the forward spline `forward`: its K, backward and presence kept.
_with_forward(cam, forward) = CameraModel(cam.intrinsics, SplineLens(cam.lens.backward; forward, present = cam.lens.present))

function _forward_spline(cam, region, tolerance, device)
    q = _chart_quadrature(cam, region, device)
    r, code, n = _spacing_search(cells -> _forward_fit(q, cells), tolerance, _max_cells(q.extent, q.stride))
    return (; spline = r.spline, r, code, n)
end

# The quadrature in the chart: the equidistant direction chart about ẑ, the chart points of the fit's and the held-out rays, the chart's
# bounding rectangle, the fit points' normalized points (the right-hand side)
# and the camera's x ↦ u.
function _chart_quadrature(cam, region, device)
    q = _image_quadrature(cam, region; device)
    chart = VgtkCore.forward_chart(SplineLens)   # THE chart `project` evaluates the forward on (Core's one derivation)
    fit_charts = pointwise!(chart, similar(q.fit_rays, SVector{2, Float64}), q.fit_rays)
    held_charts = pointwise!(chart, similar(q.held_rays, SVector{2, Float64}), q.held_rays)
    lo, hi = zip(extrema(c -> c[1], fit_charts), extrema(c -> c[2], fit_charts))
    margin = 1.0e-6
    area = Rect2(lo[1] - margin, lo[2] - margin, hi[1] - lo[1] + 2margin, hi[2] - lo[2] + 2margin)
    pixel = VgtkCore.Cameras._pixel_affine(cam.intrinsics)   # Core's x ↦ K x, what the camera's `project` applies
    return (; q..., chart, fit_charts, held_charts, area, pixel, rhs = _rhs_matrix(_normalized_points(cam.intrinsics, q.fit_px)))
end

# The forward fit at `cells` knot spacings along the chart's longer side: the
# spacing h, the counts nᵢ = ⌈extentᵢ / h⌉ + 3, the spline, λ, the held-out
# errors in pixels.
function _forward_fit(q, cells::Int)
    w = SVector{2, Float64}(maximum(q.area) - minimum(q.area))
    h, (nx, ny) = _knot_counts(w[1], w[2], cells)
    controls, λ = _penalized_solve(_design_rows(q.area, (ny, nx), q.fit_charts), q.rhs, ny, nx, Val(2))
    spline = ChartSpline(q.area, controls)
    err = map((c, u) -> norm(q.pixel(spline(c)) - u), q.held_charts, q.held_px)
    return (; spline, h, λ, counts = (nx, ny), errors = _held_out_errors(err, q.edge))
end


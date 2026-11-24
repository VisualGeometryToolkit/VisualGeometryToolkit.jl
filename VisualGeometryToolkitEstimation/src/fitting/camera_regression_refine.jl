# ==============================================================================
# the camera-regression refine — a camera model polished onto a source camera over a region of
# its sensor, from a guess; the fit-then-refine recipe — CameraRegression's linear start, then
# that refine
# ==============================================================================
#
# THE OBJECTIVE is CameraRegression's (camera_regression.jl): the integrated SQUARED pixel error of the round trip over the region,
# by the same quadrature (a physical K's radial profile, a gauge K's image samples). CameraRegression solves it in one
# linear least squares where the family has a linear form; the camera-regression refine minimizes it exactly, by the stack's chart LM
# (Core's `lmsolve_chart!`), from the guess's lens and focal: the coefficients the family moves (a forward lens's
# projection coefficients and its radial factor's numerator and denominator, a lift's every coefficient) and the focal
# rescale, and for a GAUGE K (a SplineLens source) also the aspect, the principal point and the axis.

# ------------------------------------------------------------------------------
# The chart of the regression
# ------------------------------------------------------------------------------

# The coefficients the refine moves: the guess lens's manifold (`LensManifold`, Dual-safe) restricted by NAME to the
# `free` ones (Core's `restrict(ch, names)`, the one name→mask plumbing CameraPosesProblem feeds), every other
# coefficient held at the guess's value. A PHYSICAL source's radial profile has no azimuth, so it determines no
# decentering term (tangential, thin prism): those are held for it whatever `free` says.
function _lens_family_manifold(role, a::AbstractLensModel, free::Tuple{Vararg{Symbol}})
    ch = LensManifold(a)
    held = role isa VgtkCore.Cameras.FreeIntrinsics ? _decentering_names(a) : ()
    names = Tuple(n for n in parameter_names(a) if n in free && !(n in held))
    length(names) == tangent_size(ch) && return ch
    return restrict(ch, names)
end
# The names of a lens's decentering deltas (Core's `isdelta` leaves: tangential, thin prism).
_decentering_names(a::AbstractLensModel) =
    Tuple(Iterators.flatten(VgtkCore.Cameras.coefficient_names(typeof(t)) for t in VgtkCore.Cameras._leaves(a) if t isa VgtkCore.AbstractDistortionTerm && VgtkCore.isdelta(t)))
_decentering_names(::LiftLens) = ()
# The refine's chart, by the role of the source's K: the focal's log-scale (an `EuclideanManifold` at the guess's
# `log s₀`, `s = exp`; `:f` in `free`) before the lens family's manifold. For a GAUGE K, whose focal lengths are the
# lattice's local scale at its ẑ's pixel (an aspect off 1 wherever that is off the axis), the scale has a second
# coordinate, the log aspect `a` (`fy′ = s·a·fy₀`; `:aspect`); then the principal point's move δ (an
# `EuclideanManifold{2}` at the guess's δ₀, in the source's normalized plane: K₀₂·δ pixels; `:cx`, `:cy`) and the axis
# (Core's `SphereManifold` at ẑ, 2 coordinates, always free), embedded as the minimal rotation taking it onto ẑ
# (`_axis_rotation`): its roll fixed exactly by that parametrization. A held coordinate keeps the guess's value. A
# PHYSICAL K's principal point and aspect are data (its samples are the radial profile about them): held.
_regression_manifold(::VgtkCore.Cameras.FreeIntrinsics, ch::Manifold, start, free) =
    ProductManifold(_held_unless(EuclideanManifold(SVector(log(start.s))), free, (:f,)), ch)
_regression_manifold(::VgtkCore.Cameras.GaugeIntrinsics, ch::Manifold, start, free) = ProductManifold(
    _held_unless(EuclideanManifold(SVector(log(start.s), log(start.a))), free, (:f, :aspect)),
    _held_unless(EuclideanManifold(start.δ), free, (:cx, :cy)),
    VgtkCore.SphereManifold(SVector(0.0, 0.0, 1.0), _axis_rotation), ch
)
# A Euclidean block whose coordinates are the parameters `names`, restricted to those in `free` (all: the block).
function _held_unless(e::EuclideanManifold{N}, free, names::NTuple{N, Symbol}) where {N}
    mask = SVector{N, Bool}(map(n -> n in free, names))
    return all(mask) ? e : restrict(e, mask)
end
# The minimal rotation taking the unit axis `a` onto ẑ: Rotations' `rotation_between(a, ẑ)` operation for operation (the
# quaternion `(‖a‖ + a·ẑ, a × ẑ)`, normalized by its constructor; about a vector perpendicular to `a` when `a` is `−ẑ`),
# written out so that it compiles for a device — the regression's kernel retracts the axis in every row, and
# `rotation_between` builds its perpendicular vector by a comprehension no GPU compiler takes. Dual-safe.
@inline function _axis_rotation(a::StaticVector{3})
    z = SVector(0.0, 0.0, 1.0)
    normprod = sqrt(dot(a, a) * dot(z, z))
    w = normprod + dot(a, z)
    v = abs(w) < 100 * eps(typeof(normprod)) ? _perpendicular(a) : cross(a, z)
    return SMatrix{3, 3}(Rotations.QuatRotation(w, v[1], v[2], v[3]))
end
# A vector perpendicular to `u` (Rotations' `perpendicular_vector`: minus the second-largest entry at the largest one's
# place, the largest at the second-largest's, zero at the third), by branches on the entries' magnitudes.
@inline function _perpendicular(u::StaticVector{3})
    a1, a2, a3 = abs(u[1]), abs(u[2]), abs(u[3])
    i1 = a1 >= a2 ? (a1 >= a3 ? 1 : 3) : (a2 >= a3 ? 2 : 3)                       # the largest entry (the first of equals)
    i2 = i1 == 1 ? (a2 >= a3 ? 2 : 3) : i1 == 2 ? (a1 >= a3 ? 1 : 3) : (a1 >= a2 ? 1 : 2)   # the second largest
    p1 = @inbounds -u[i2]
    p2 = @inbounds u[i1]
    o = zero(p1)
    return SVector(ifelse(i1 == 1, p1, ifelse(i2 == 1, p2, o)), ifelse(i1 == 2, p1, ifelse(i2 == 2, p2, o)),
                   ifelse(i1 == 3, p1, ifelse(i2 == 3, p2, o)))
end
# The fit at the chart's retracted parts: the focal rescale `s` and aspect `a`, the principal point's move `δ`
# (normalized), the rotation `R` from the source's frame to the fitted one (`nothing`: the source's own), the lens.
# Dual-safe.
_regression_parts(parts::Tuple{Any, Any}) = (; s = exp(first(parts)[1]), a = 1.0, δ = zero(SVector{2, Float64}), R = nothing, lens = last(parts))
_regression_parts(parts::Tuple{Any, Any, Any, Any}) =
    (; s = exp(first(parts)[1]), a = exp(first(parts)[2]), δ = SVector(parts[2][1], parts[2][2]), R = parts[3], lens = last(parts))

# The linearization state of the stack's chart LM (Core's `lmsolve_chart!`): the chart `P`, whether the rows are angle
# residuals (the lift families), `M = K₀₂ / fx₀` (a normalized offset in pixels / fx₀), and per sample — on the
# KernelAbstractions backend `device` of the fit — its ray, the ray's view angle about ẑ, its point `y` of the source's
# normalized plane and its row weight √w; and the reduction's partial sums there, one column per work item.
#
# THE ROWS NEVER REACH THE HOST. The state answers the chart LM's two reductions (`residual_cost!`,
# `residual_normal_equations!`) with ONE kernel (Core's launch-and-reduce, `reduce_normal_equations(row!, …)` /
# `reduce_cost(row!, …)`: the stack's one row launch, the same on every backend, the CPU's included): each work item evaluates its samples' 2-D residuals — and, at a linearization, their Jacobian rows,
# ForwardDiff duals of the chart vector inside the kernel — and sums ‖r‖², Jᵀr and JᵀJ into its column. The columns'
# sum is the one device reduction after it; only the N × N step is the host's.
struct _LensRegressionLm{D, TP <: ProductManifold, R, A, Y, B}
    device::D
    P::TP
    angle_residual::Bool
    θcap::Float64
    M::SMatrix{2, 2, Float64, 4}
    rays::R
    angles::A
    points::Y
    sw::A
    partials::B
end
# The work items of the reduction: enough to fill an accelerator, few enough that their columns sum in one reduction.
const _REDUCTION_ITEMS = 2048
function _LensRegressionLm(device, P::ProductManifold, angle_residual::Bool, θcap::Float64, M, rays, angles, points, sw)
    N = tangent_size(P)
    up(T, v) = VgtkCore.on_device(T, device, v)
    W = max(1, min(length(rays), _REDUCTION_ITEMS))
    return _LensRegressionLm(
        device, P, angle_residual, θcap, M, up(SVector{3, Float64}, rays), up(Float64, angles), up(SVector{2, Float64}, points), up(Float64, sw),
        VgtkCore.device_zeros(Float64, device, 1 + N + N * N, W)
    )
end

# A ray's view angle about ẑ and its azimuth's unit vector (zero on the axis, where the radius is 0). Dual-safe.
@inline function _angle_azimuth(x)
    ρ = VgtkCore.Cameras._radius(x)
    az = iszero(VgtkCore.primal_value(ρ)) ? zero(SVector{2, typeof(ρ)}) : SVector(x[1], x[2]) / ρ
    return atan(ρ, x[3]), az
end
# A sample's ray in the fitted frame as (view angle, azimuth): turned by R, or the source's own (its angle as sampled).
@inline _fitted_ray(::Nothing, d, θ) = (θ, last(_angle_azimuth(d)))
@inline _fitted_ray(R, d, θ) = _angle_azimuth(R * d)

# THE 2-D residual of one sample, in pixels / fx₀ (the rows are √w times it).
#
# A forward target: the target camera's pixel of the turned ray minus the sample's — its radial projection
# `projected_radius` (the lens's own r(θ), the magnitude of its projection at azimuth 0) along the ray's azimuth,
# scaled by (s, s·a) and moved by δ in the source's normalized plane, through `M`. Where the lens images the ray on
# its branch this is its `project`; past a fold of the profile (OPENCV3 and the decentered OPENCV target over a
# region past 90°, the double sphere past its image circle) `project`'s point flips or is not imaged and this form
# stays the radius, finite. At a physical K's sample (azimuth 0, δ = 0, R = I) it is (s·r_B(θ) − r_A, 0): the radial
# regression's row exactly.
@inline function _pixel_offset(fit, M, d, θ, y, θcap)
    θ′, az = _fitted_ray(fit.R, d, θ)
    p = projected_radius(fit.lens, min(θ′, θcap)) * az
    return M * (SVector(fit.s * p[1], fit.s * fit.a * p[2]) + fit.δ - y)
end
# A lift target (rational, Scaramuzza): the lift is defined at every image radius, where a candidate's forward may fold
# inside the domain, so the residual is in the equidistant direction chart about ẑ (a ray at angle θ and azimuth a is
# θ·a) — the lift's ray of the sample's point minus the turned ray — scaled by the point's normalized radius about the
# moved principal point (angle × r ≈ a radius). At a physical K's sample it is ((θ_B(r_A/s) − θ)·r_A, 0): the radial
# regression's angle row exactly.
@inline function _chart_offset(fit, d, θ, y)
    v = y - fit.δ
    θB, azB = _angle_azimuth(backproject(fit.lens, SVector(v[1] / fit.s, v[2] / (fit.s * fit.a))))
    θ′, az = _fitted_ray(fit.R, d, θ)
    return VgtkCore.Cameras._radius(v) * (θB * azB - θ′ * az)
end
@inline _sample_offset(angle_residual::Bool, M, fit, d, θ, y, θcap) =
    angle_residual ? _chart_offset(fit, d, θ, y) : _pixel_offset(fit, M, d, θ, y, θcap)

# The tag of the chart vector's duals inside the reduction kernel.
struct _RegressionTag end
# The chart vector `x` seeded as duals of itself.
@inline _seeded(x::SVector{N, Float64}) where {N} = SVector{N}(ntuple(
    k -> ForwardDiff.Dual{_RegressionTag}(x[k], ForwardDiff.Partials(ntuple(j -> j == k ? 1.0 : 0.0, Val(N)))), Val(N)
))

# Work item `w`'s share of the cost: Σ ‖√wᵢ rᵢ‖² over its samples `w, w + W, …`, into `out[1, w]`.
Base.@propagate_inbounds function _cost_row!(out, w, P, angle_residual, θcap, M, rays, angles, points, sw, x)
    fit = _regression_parts(retract(P, x))
    s = 0.0
    for i in w:size(out, 2):length(rays)
        e = sw[i] * _sample_offset(angle_residual, M, fit, rays[i], angles[i], points[i], θcap)
        s += e[1] * e[1] + e[2] * e[2]
    end
    out[1, w] = s
    return nothing
end
# Work item `w`'s share of the normal equations: Σ ‖rᵢ‖², Σ Jᵢᵀrᵢ and Σ JᵢᵀJᵢ over its samples into column `w` (the
# cost, then the N gradient entries, then the N × N Hessian column-major), each sample's 2 × N Jacobian the partials
# of its residual at the seeded chart vector.
Base.@propagate_inbounds function _normal_row!(out, w, P, angle_residual, θcap, M, rays, angles, points, sw, x::SVector{N, Float64}) where {N}
    fit = _regression_parts(retract(P, _seeded(x)))
    s = 0.0
    g = zero(SVector{N, Float64})
    H = zero(SMatrix{N, N, Float64})
    for i in w:size(out, 2):length(rays)
        e = sw[i] * _sample_offset(angle_residual, M, fit, rays[i], angles[i], points[i], θcap)
        r = SVector(ForwardDiff.value(e[1]), ForwardDiff.value(e[2]))
        J = SMatrix{2, N, Float64}(ntuple(k -> ForwardDiff.partials(e[(k - 1) % 2 + 1], (k - 1) ÷ 2 + 1), Val(2N)))
        s += r[1] * r[1] + r[2] * r[2]
        g += J' * r
        H += J' * J
    end
    out[1, w] = s
    for a in 1:N
        out[1 + a, w] = g[a]
    end
    for b in 1:N, a in 1:N
        out[1 + N + (b - 1) * N + a, w] = H[a, b]
    end
    return nothing
end

# The two reductions: Core's launch-and-reduce over the work items (one kernel on the state's backend, then the sum).
residual_cost!(_, lm::_LensRegressionLm, x) = VgtkCore.reduce_cost(
    _cost_row!, lm.P, lm.angle_residual, lm.θcap, lm.M, lm.rays, lm.angles, lm.points, lm.sw, x; partials = lm.partials
)
residual_normal_equations!(_, _J, lm::_LensRegressionLm, x::SVector{N, Float64}) where {N} = VgtkCore.reduce_normal_equations(
    _normal_row!, Val(N), lm.P, lm.angle_residual, lm.θcap, lm.M, lm.rays, lm.angles, lm.points, lm.sw, x; partials = lm.partials
)

# Each sample's round-trip error at the solution `x`, in pixels / fx₀: the forward form, a lift's radius capped at
# `θcap` (one kernel, the same backend), back on the host.
Base.@propagate_inbounds function _error_row!(out, i, P, M, rays, angles, points, x, θcap)
    out[i] = _pixel_offset(_regression_parts(retract(P, x)), M, rays[i], angles[i], points[i], θcap)
    return nothing
end
function _sample_errors(lm::_LensRegressionLm, x, θcap)
    n = length(lm.rays)
    out = VgtkCore.device_zeros(SVector{2, Float64}, lm.device, n)
    VgtkCore._launch_rows!(_error_row!, out, n, lm.P, lm.M, lm.rays, lm.angles, lm.points, x, Float64(θcap))
    return Array(out)
end

# The chart coordinates of the fit (the anchor is the start: every coordinate at
# zero), its termination record `(code, rule)`, its iterations and whether the regression is stationary there (Core's
# chart-LM postcondition, `is_stationary(ews, lm, x, stops)`: Core 0.11.0 converges a tolerance stop without it, and
# `OpticalAxis`' validity gate reads it) — a function barrier static in the N
# coordinates (the focal's and the family's). The workspace holds no rows: the state's reductions keep them on its
# device (one placeholder row, NLLSsolver's problem needs one). `stops` are Core's stop settings (`LmStops`, its
# `[lm]` table), the caller overriding a field only from its own table (the regression's refine: the cap, `[camera_regression_problem]`).
function _regress_lens(lm::_LensRegressionLm, ::Val{N}, stops::LmStops, damping::LmDamping = MarquardtDamping()) where {N}
    ews = lm_workspace(nothing, 1, Val(N))       # Core's one chart-LM workspace reuse policy
    x, _, _, iterations, record = lmsolve_chart!(ews, lm, stops, damping)
    return x, record, iterations, all(isfinite, x) && VgtkCore.is_stationary(ews, lm, x, stops)
end

# ==============================================================================
# the camera-regression refine: the chart LM from a guess
# ==============================================================================

"""
    solve(RefineProblem(CameraRegressionProblem(source, region; fixed = (), weight = :image, device, support), guess::CameraModel),
          LevenbergMarquardt(; optimization_strategy = nothing); stops) -> FitResult{<:CameraModel, CameraFitMetadata}

The camera of `guess`'s lens family polished onto `source` over `region` (a region of the source's sensor in its pixel
frame, as [`CameraRegression`](@ref) takes it): the exact minimizer of the integrated squared pixel error of the round trip,
`∫ ‖π(π⁻¹_source(u)) − u‖² du`, by the stack's chart LM from `guess`. It is a `refine`: it needs the guess, and it
converges to the minimum nearest it. [`CameraRegression`](@ref) gives a guess where the family has a linear form
(a fit from no guess is the linear fit, then this refine; a family with no linear form starts from its anchor, the
family's zero lens at the source's `K`).

**The region's support.** The `region` must lie inside the source's calibrated support — `support`, the source's
`valid_domain` by default, or the support of the camera it was re-gridded from — or the fit is refused with Core's
`RegionBeyondSupport` (`check_region_support`, at entry): a fit never reads a model where it is extrapolated.

- **What moves: everything but `fixed`**, the guess's parameters held by name (`parameter_names(guess)`: the
  intrinsics' `:f`, `:aspect`, `:cx`, `:cy`, then the lens's) or by group (a coefficient family's stem: `:k` the radial
  numerator, `:d` its denominator, `:p` the tangential terms, `:sx`/`:sy` the thin prism's); the gauge's skew is held
  by construction. A held parameter keeps the guess's value. `:f` is the focal rescale `s` from the guess's focal
  (relative to the source's, `fx′ = s·fx₀`); a forward lens's radial factor moves its numerator and its denominator
  (`fixed = (:d,)` holds the denominator); a lift moves every coefficient. A name or group the guess does not have is
  an `ArgumentError`, and so are the keywords `free`, `fix`, `fixed_params` and `hold` (naming `fixed`).
- **A physical `K`** (a parametric source): the principal point, the aspect and ẑ are the source's, data, so the
  guess's principal point must be the source's (an `ArgumentError` otherwise) and `:aspect`, `:cx`, `:cy` are held, as
  are the decentering terms (tangential, thin prism) its azimuth-0 profile does not see. The samples are the source lens's radial profile (its
  radial part: decentering terms are stripped, warned when non-negligible) on Clenshaw–Curtis nodes, the exact
  reduction of the image integral to `θ`, with the density `weight` (`:image`, `:solid_angle` or `:uniform`) and the
  region's coverage of each radius (the region star-shaped about the principal point).
- **A gauge `K`** (a SplineLens camera, a `CameraModel` or a `BoundedCamera` of one): the principal point, the aspect
  and the axis are free, from the guess's principal point and aspect and from the source's ẑ. The samples are the
  camera's own rays on the image over the region, each pixel one sample of equal weight (the `:image` density; another
  `weight` is an `ArgumentError`). The fitted camera's principal point is the axis's pixel, and `metadata.rotation` is
  the turn from the source's frame to the fitted one; `metadata.symmetry_residual` (px) is the azimuthal misfit no
  radially symmetric camera represents. [`OpticalAxis`](@ref) is this refine of the Fisheye624 form (4 radial terms).
- **The residual** is one 2-D residual per sample: a forward lens's is its pixel minus the sample's
  (`projected_radius` along the ray's azimuth, finite past a fold of its profile); a lift's is angle-style, in the
  equidistant chart about ẑ, scaled by the sample's normalized radius.
- **The device.** Each sample's residual and its Jacobian rows, and their reduction to the normal equations, are one
  KernelAbstractions kernel on the backend `device`; only the chart LM's N × N step is the host's.
- **The loss.** `optimization_strategy = nothing`, the plain least squares, is the only one: a camera-from-camera
  regression has no outliers. It takes no `scale` or `width` (one given would be ignored: an `ArgumentError`), and the
  result records no scale (`scale_source` `ScaleNone`, `scale` `NaN`; `noise_scale` refuses it).

The domain is the region's farthest radius, capped by the source's own view-angle limit, as `CameraRegression`'s; a region
past the family's view-angle limit (the perspective plane's 90° for a gnomonic family) is a [`ViewAngleLimitError`](@ref). At most `max_iter` accepted iterations (configs/default.toml
`[camera_regression_problem]`); the termination record (`retcode`, `stop_rule`) is the LM's, a refine that ends at a non-finite
focal rescale is `(ReturnCode.Failure, NonFinite)` (`:solve_failed`), never a throw, and one with nothing free (the
guess returned, no LM) is a closed form, `(ReturnCode.Success, ClosedForm)`. The metadata is [`CameraFitMetadata`](@ref): the round trip's errors per sample in
`residuals` (px), their maximum and edge-band maximum, the result's `rms` their weighted root mean square, and
`family.stationary`: whether the regression is stationary at the estimate (Core's chart-LM postcondition; Core 0.11.0
reads a tolerance stop `Success` without it, so a `converged` refine can stop short of the minimum).

A `NoncentralSplineModel` source is an `ArgumentError`: not supported yet.

```julia
start = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, camera, region)), CameraRegression()).u   # the linear fit: the guess
kb = solve(RefineProblem(CameraRegressionProblem(camera, region), start), LevenbergMarquardt())                    # its exact pixel least squares
```
"""
function _refine_camera_entry(guess::CameraModel, source::CameraModel, region; support = VgtkCore.valid_domain(source), kwargs...)
    VgtkCore.check_region_support(support, region; who = "CameraRegressionProblem")   # Core's one guard: no fit over an extrapolated model
    return _refine_camera(guess, source, region; kwargs...)
end
# The refine itself, its region checked by the caller (`the camera-regression refine`, `the fit-then-refine recipe`, `OpticalAxis`).
function _refine_camera(
        guess::CameraModel, source::CameraModel, region;
        fixed::Tuple{Vararg{Symbol}} = (),
        weight::Symbol = :image, stops::LmStops = _CAMERA_REGRESSION_STOPS, damping::LmDamping = MarquardtDamping(), device = default_device(),
        free = RetiredKeyword(), fix = RetiredKeyword(), fixed_params = RetiredKeyword(), hold = RetiredKeyword(),   # refused, naming `fixed`
    )
    _refuse_held_spellings(free, fix, fixed_params, hold)
    gl = lens(guess)
    gl isa Union{Lens, LiftLens} || throw(
        ArgumentError("CameraRegressionProblem: the guess must carry a parametric lens (a Lens or a LiftLens), got a $(nameof(typeof(gl)))")
    )
    L = typeof(gl)
    moving = VgtkCore.free_parameter_names(guess, fixed)   # Core's one resolution of `fixed` (names and groups; the skew held)
    role = VgtkCore.Cameras.IntrinsicsRole(lens(source))
    K0 = SMatrix{3, 3, Float64}(calibration_matrix(source))
    M = _offset_matrix(K0)   # K₀₂ / fx₀ (camera_regression.jl)
    start = _refine_start(role, SMatrix{3, 3, Float64}(calibration_matrix(guess)), K0, M)
    pr = _profile_samples(role, L, source, region, K0, weight, device)
    P = _regression_manifold(role, _lens_family_manifold(role, gl, moving), start, moving)
    # a forward guess that folds before the domain's edge (a linear fit over a field its family cannot image) images
    # no radius past its fold: its residual there is capped at the fold, as the reported errors are
    # (a guess at the source's own fold, to its rounding, is not capped: 1e-6 rad)
    θcap = L <: LiftLens ? Inf : (t = Float64(view_angle_limit(gl)) - 1.0e-9; t < pr.θ_max - 1.0e-6 ? t : Inf)
    lm = _LensRegressionLm(device, P, L <: LiftLens, θcap, M, pr.rays, pr.angles, pr.points, sqrt.(pr.w))
    # nothing free (a physical K with `free = ()`; a gauge K's axis is always free): the guess itself, no LM
    p, record, iters, stationary = tangent_size(P) == 0 ? (SVector{0, Float64}(), (ReturnCode.Success, ClosedForm), 0, true) :
        _regress_lens(lm, Val(tangent_size(P)), stops, damping)

    fit = _regression_parts(retract(P, p))
    # a regression that ends non-finite did not converge: its FitResult reports it
    isfinite(fit.s) || (record = (ReturnCode.Failure, NonFinite))
    # the round trip's error per sample, in pixels: the forward form, its radius capped inside the refined lens's domain
    e = K0[1, 1] .* _sample_errors(lm, p, _domain_cap(L, fit.lens, isfinite(fit.s), pr.θ_max))   # (camera_regression.jl)
    return _lens_fit_result(role, fit, K0, M, pr, e, region, weight, record, iters; stationary = stationary && isfinite(fit.s))
end
# A bounded camera's model is the source, over the part of the region it images (`_imaged_part`).
_refine_camera_entry(guess::CameraModel, source::BoundedCamera, region; support = VgtkCore.valid_domain(source), kwargs...) =
    _refine_camera_entry(guess, model(source), _bounded_part(source, region, support, "CameraRegressionProblem"); kwargs...)
_refine_camera_entry(guess::CameraModel, source::NoncentralSplineModel, region; kwargs...) = throw(
    ArgumentError("CameraRegressionProblem: a non-central source (NoncentralSplineModel) is not supported yet: a radially symmetric central camera fits a central source")
)

# The L2 refine's keywords: no loss, so no scale input (one given would be ignored, so it is refused by the bundles'
# one reader, `bundle_loss`, whose least squares takes none).
function _check_l2_refine(optimization_strategy)
    optimization_strategy === nothing || throw(
        ArgumentError(
            "CameraRegressionProblem: the camera-from-camera regression is plain least squares (`optimization_strategy = nothing`); " *
                "its samples are a source model's, with no outliers to weigh down, got $(nameof(typeof(optimization_strategy)))"
        )
    )
    return nothing
end

# THE refine of the camera regression: Core's LevenbergMarquardt on a `CameraRegressionProblem` without a target (its
# block contract): plain least squares only, from the guess, at the solve's `stops` (default [camera_regression_problem]
# `max_iter`) and the algorithm's damping.
function block_refit(f::CameraRegressionProblem{Nothing}, guess, loss, stops::LmStops, damping::LmDamping)
    _check_l2_refine(loss.strategy)
    return _refine_camera_entry(guess, f.source, f.region; f.settings..., stops, damping)
end
block_refit(f::CameraRegressionProblem, _guess, _loss, _stops, _damping) = throw(ArgumentError(
    "a CameraRegressionProblem with a target is solved from scratch (CameraRegression, PSpline): its refine is " *
        "`RefineProblem(CameraRegressionProblem(source, region; …), guess)`, the guess giving the family"))
default_optimization_strategy(::CameraRegressionProblem) = nothing   # a camera-from-camera regression has no outliers: L2
default_stops(::CameraRegressionProblem) = _CAMERA_REGRESSION_STOPS
const _CAMERA_REGRESSION_STOPS = LmStops(; max_iter = _dflt("camera_regression_problem", "max_iter"))

# The chart's start from the guess's K against the source's `K0`: the focal rescale `s` (fx / fx₀), the aspect `a`
# (fy / (s·fy₀)) and the principal point's move `δ` in the source's normalized plane (K₀₂·δ pixels). A PHYSICAL K holds
# the principal point and the aspect, which are data: a guess that moves them is refused.
function _refine_start(::VgtkCore.Cameras.FreeIntrinsics, Kg, K0, M)
    s = Kg[1, 1] / K0[1, 1]
    a = Kg[2, 2] / (s * K0[2, 2])
    tol = 1.0e-9 * max(1.0, abs(K0[1, 1]))
    (abs(Kg[1, 3] - K0[1, 3]) <= tol && abs(Kg[2, 3] - K0[2, 3]) <= tol && abs(a - 1) <= 1.0e-9) || throw(
        ArgumentError(
            "CameraRegressionProblem: a physical source's principal point $((K0[1, 3], K0[2, 3])) and aspect are held; " *
                "the guess has $((Kg[1, 3], Kg[2, 3])) and aspect $a"
        )
    )
    return _checked_start((; s, a = 1.0, δ = zero(SVector{2, Float64})))
end
function _refine_start(::VgtkCore.Cameras.GaugeIntrinsics, Kg, K0, M)
    s = Kg[1, 1] / K0[1, 1]
    return _checked_start((; s, a = Kg[2, 2] / (s * K0[2, 2]), δ = M \ (SVector(Kg[1, 3] - K0[1, 3], Kg[2, 3] - K0[2, 3]) / K0[1, 1])))
end
_checked_start(st) = (isfinite(st.s) && st.s > 0 && isfinite(st.a) && st.a > 0 && all(isfinite, st.δ)) ? st : throw(
    ArgumentError("CameraRegressionProblem: the guess's focal lengths must be positive and finite relative to the source's (rescale $(st.s), aspect $(st.a))")
)

# ==============================================================================
# the fit-then-refine recipe: CameraRegression's linear start, then the camera-regression refine
# ==============================================================================

# THE region a regression of a bounded camera's model runs over: the caller's region checked against the source's support
# (Core's guard, never clipped past it; `who` names the solve), then its part the camera images (`_imaged_part`).
_bounded_part(source::BoundedCamera, region, support, who::AbstractString) =
    _imaged_part(source, VgtkCore.check_region_support(support, region; who))
# The part of `region` a bounded camera images: `region ∩ imaged_region(camera)` (Core's intersection), so no sample
# lies where the camera images nothing; its own imaged region as it is (the same region value, the same samples).
function _imaged_part(camera::BoundedCamera, region)
    imaged = VgtkCore.imaged_region(camera)
    return region == imaged ? imaged : region ∩ imaged
end
# The refine's start: the linear fit's camera, or the family's anchor at the source's K (no linear form, or a failed
# linear fit).
function _approximation_start(form, ::Type{L}, source, region, weight, fixed, device) where {L}
    r = _fit_lens_linear(form, _target_anchor(L), source, region, weight, fixed, device)
    return successful_retcode(r) ? r.u : _anchor_camera(L, source)     # a failed linear solve: the anchor
end
_approximation_start(::_NoLinearForm, ::Type{L}, source, region, weight, fixed, _device) where {L} = _anchor_camera(L, source)
# The camera of the source's K with the family's anchor lens, or with a given lens.
_anchor_camera(::Type{L}, source) where {L} = _anchor_camera(_target_anchor(L), source)
_anchor_camera(a::AbstractLensModel, source) =
    CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(calibration_matrix(source)); check = false)), a)

# ==============================================================================
# OpticalAxis: the approximation of the Fisheye624 family
# ==============================================================================

"""
    OpticalAxis()

A [`FitKind`](@extref VisualGeometryToolkitCore.FitKind):
`solve(EstimationProblem(CameraRegressionProblem(camera[, region]; support, device)), OpticalAxis())` (`camera` a
`BoundedCamera`, `region` its `imaged_region` by default, `support` its `valid_domain`) -> `FitResult{<:CameraModel, CameraFitMetadata}`.

The radially symmetric camera closest to `camera` over `region ∩ imaged_region(camera)`: by default the camera's own
bound, its `imaged_region` (the sensor ∩ its lens coverage ∩ the model's valid domain); given a `region` (a `Rect2`, an
`Ellipse`, or their `∩`, in the camera's pixel frame), its part the camera images — a target's rectangle, where the rays
are measured, keeps out a lattice's unconstrained edge (RadiometricCamera's siemens start camera stalls over its whole
image and converges over its board). It approximates `camera` by the Fisheye624 form
(COLMAP's `RAD_TAN_THIN_PRISM_FISHEYE`, `Lens(Equidistant(), Additive(Tangential, ThinPrism{2}) ∘ Radial{Even, n})`, with
`n = 4` radial terms, θ³…θ⁹: configs/default.toml `[optical_axis] radial_terms`, the smallest order that locates the axis
as well as 6 terms do on the cameras studied, and far better conditioned;
its decentering held at zero), as the linear fit then its refine does — the linear fit of its radial coefficients and
focal, then the refine with the principal point and the axis free — but order by order (below): on a source the
4-term start fits well, it lands where `the fit-then-refine recipe` of that family does.
- The region is the bound or a part of it, never more. A circular fisheye's lens coverage (its image circle) keeps the
  corners out, where a grid's rays are its extrapolation. A whole-sensor fit is a camera whose bound is the whole sensor.
- A bare model has no bound: an `ArgumentError`. Wrap it in `BoundedCamera(model, size; lens_coverage)`.
- For a SplineLens camera (a gauge `K`), the principal point and the axis are free. Read them from the result.
  - The optical axis in the camera's frame is `rotation(r.metadata.rotation)' * ẑ`.
  - Its pixel is the fitted camera's principal point.
  - `r.metadata.symmetry_residual` (px) is the azimuthal misfit that no radially symmetric camera represents. A
    decentred lens reports it.
- A physical `K` is its own axis: the identity rotation and the source's principal point.

`device` is the backend the camera's rays are sampled and the regression reduced on (`the fit-then-refine recipe`'s).
`support` bounds the region (`region` must lie inside it, else Core's `RegionBeyondSupport`; never clipped): the
camera's `valid_domain` by default; for a camera re-gridded from a calibration (a whole-sensor lattice extrapolated past
the board), the calibration's support — RadiometricCamera's siemens start camera, refused over its whole sensor, fits
over its Schöps support.

The fit is an ORDER CONTINUATION. The family is fitted at `[optical_axis] start_radial_terms` radial terms (1: its linear
fit, then the refine), then at each next order up to `radial_terms` (2, 3, 4), each order's refine starting from the
last order's camera, its higher coefficients held at 0. A low order does not follow a source's profile where it bends at
the region's edge, so its start never folds there; raising the order one term at a time keeps each refine in the basin
of the last (a direct lift from 1 to 4 terms can end in a worse local minimum). An order HOLDS when its refine converged, its regression is stationary at the estimate (`family.stationary`: on the
axis, Core 0.11.0 reading a tolerance stop converged without it)
and its lens does not fold inside the region (`view_angle_limit` below `θ_max`); every order is refined, each from the
last one's camera whether or not it held (a rectilinear lens turning at the rim: its 1- and 2-term fits fold inside the
region with the axis unmoved, its 3-term one does not), and the result is the highest order that held,
`r.metadata.family.fitted_terms` (its higher coefficients 0).

When no order holds, the fallback is the whole-order linear start. A start that FOLDS before the region's
widest angle (a rectilinear lens to 62° whose 4-term fit turns over at 56°) is recorded as
`r.metadata.family.start_fold` (radians; `nothing` when the start does not fold); when the refine from it does not
hold (the same gate: converged, stationary, no fold inside the region), the axis is fitted again over the part of the region inside the
fold (out to 0.9 of the fold angle, about the start's principal point), `r.metadata.region` then the region used; a
start folding just past the widest angle, inside the band that refit leaves out, is refitted the same way. An axis fit
that still fails the gate (from any start, folding or not) is an `ArgumentError`: the axis re-gauge is applied always (`CameraRegressionProblem(SplineLens, …)`), and a stalled
fit is never a silent rotation 0.

Core's `axis_gauged(camera, rotation(r.metadata.rotation), calibration_matrix(r.u))` then turns the camera so ẑ is the
axis. `CameraRegressionProblem(SplineLens, …)` does this first, always (a held SplineLens camera's `CameraPosesProblem` does not:
it returns the camera as given).
"""
struct OpticalAxis <: AbstractEstimationAlgorithm end
estimator_kind(::OpticalAxis) = FitKind()
function __solve(prob::EstimationProblem{<:Any, <:CameraRegressionProblem{Nothing}, NullParameters}, ::OpticalAxis)
    f = prob.f
    return f.region === nothing ? _optical_axis(f.source; f.settings...) : _optical_axis(f.source, f.region; f.settings...)
end
# The optical axis's fit itself (OpticalAxis; the axis gauge of a SplineLens target, camera_regression.jl).
function _optical_axis(camera::BoundedCamera, region = VgtkCore.imaged_region(camera); support = VgtkCore.valid_domain(camera),
                      device = default_device())
    # Core's one guard on the caller's region (never clipped past the support), then its part the camera images
    part = _bounded_part(camera, region, support, "OpticalAxis")
    src = model(camera)
    r, start, fitted = _axis_continuation(src, part, device)
    if r === nothing
        # no order held: the whole-order start, its stall refitted inside its fold
        r, start_fold, ok = _axis_fold_refits(src, part, device)
        # the axis re-gauge is applied ALWAYS (`CameraRegressionProblem(SplineLens, …)`): a stalled axis would be a silent wrong turn
        ok || throw(ArgumentError(
            "OpticalAxis: the axis fit did not converge ($(VgtkCore.retcode_symbol(r)), symmetry residual " *
                "$(round(r.metadata.symmetry_residual; sigdigits = 3)) px over $(r.metadata.region)); fit it over a region " *
                "the camera images well (a target's rectangle), `CameraRegressionProblem(camera, region)` solved by `OpticalAxis()`"
        ))
        fitted = _AXIS_TERMS
    else
        start_fold = _start_fold(start, r.metadata.family.θ_max)
    end
    md = r.metadata
    return FitResult(r.u, r.retcode, r.stop_rule, r.stats, r.rms, r.n,
                     CameraFitMetadata(md.region, md.weight, md.max_px, md.max_px_edge,
                                       (; md.family..., start_fold, fitted_terms = fitted), md.rotation, md.symmetry_residual,
                                       md.residuals))
end
# The order continuation over `part`: the axis family's linear fit at `_AXIS_START_TERMS` (the higher coefficients held at
# 0), its refine, then the refine at each next order from the last order's camera (a non-finite one is not a start: the
# last finite camera is). Returns the highest order's refine that holds (`_axis_fit_valid`; `nothing` when none does), the
# linear start, and that order.
function _axis_continuation(src::CameraModel, part, device)
    start = _approximation_start(_linear_form(_AXIS_FAMILY), _AXIS_FAMILY, src, part, :image,
                                 (_AXIS_FIXED..., _axis_held(_AXIS_START_TERMS)...), device)
    r, guess, fitted = nothing, start, 0
    for m in _AXIS_START_TERMS:_AXIS_TERMS
        rm = _refine_camera(guess, src, part; fixed = (_AXIS_FIXED..., _axis_held(m)...), device)
        _axis_fit_valid(rm) && ((r, fitted) = (rm, m))
        isfinite(rm.metadata.family.focal_scale) && (guess = rm.u)
    end
    return r, start, fitted
end
# An axis fit that holds — Estimation's own validity gate, never the LM's verdict alone: its refine stopped on a rule
# (not a cap or a failure), ON THE AXIS — its regression stationary at the estimate (`r.metadata.family.stationary`,
# Core's postcondition: Core 0.11.0 reads a tolerance stop converged without it, and a fit can stop there ~12 px off the
# axis) — and its lens does not fold inside the region it was fitted over. 0.11.8's guarantee under 0.11.0's semantics.
_axis_fit_valid(r) = successful_retcode(r) && r.metadata.family.stationary && !_folds_inside(r)
# The radial coefficients above order `m` (k_(m+1) … k_n), held at 0 in an order-`m` fit of the axis family.
_axis_held(m::Int) = ntuple(i -> Symbol(:k, m + i), _AXIS_TERMS - m)
# `OpticalAxis`'s fallback: the axis family's whole-order linear start and its refine over `part`; a stall from a start
# folding before the region's widest angle, or just past it, refitted inside the fold. Returns the last refine, the start's
# fold when it folds inside the region (`nothing` otherwise), and whether the fit holds.
function _axis_fold_refits(src::CameraModel, part, device)
    start = _approximation_start(_linear_form(_AXIS_FAMILY), _AXIS_FAMILY, src, part, :image, _AXIS_FIXED, device)
    r = _refine_camera(start, src, part; fixed = _AXIS_FIXED, device)
    # a linear start that folds before the region's widest angle images no radius past its fold: the refine's residuals
    # are capped there, and it can stall with the axis unmoved (a rectilinear 14 mm lens to 62°, folding at 56°). Such a
    # stall — a fit that did not converge, or one whose refined lens still folds inside the region it was fitted over
    # (`_folds_inside`: a tolerance stop can read converged with the axis unmoved) — is fitted again over the part of the
    # region inside the fold (`_inside_fold`); a fit that left the fold keeps its whole region.
    fold, start_fold = _fold_angle(start), _start_fold(start, r.metadata.family.θ_max)   # over the whole region
    # a stall is refitted when the region reaches past the disc a refit keeps, `_INSIDE_FOLD` of the start's fold; a start
    # folding just past the region's widest angle stalls as one folding inside it (an 8 mm fisheye, 0.03° past 96.4°).
    # Each refit: the disc inside `_INSIDE_FOLD` of the current start's fold (or of the last region's widest angle, if that
    # is smaller), at most `_FOLD_REFITS` times. A stall from a start folding farther out is not refitted.
    θ, s, refits = fold, start, 0
    near = _INSIDE_FOLD * fold < r.metadata.family.θ_max
    while near && !_axis_fit_valid(r) && refits < _FOLD_REFITS
        inner = part ∩ _inside_fold(s, min(θ, r.metadata.family.θ_max))
        s = _approximation_start(_linear_form(_AXIS_FAMILY), _AXIS_FAMILY, src, inner, :image, _AXIS_FIXED, device)
        r = _refine_camera(s, src, inner; fixed = _AXIS_FIXED, device)
        θ = _fold_angle(s)
        refits += 1
    end
    return r, start_fold, _axis_fit_valid(r)
end
const _FOLD_REFITS = 4   # the refits inside a folding start's fold, each inside the last (a stall, then a refusal)
const _INSIDE_FOLD = 0.9   # the refit's disc: out to this fraction of the fold angle, where the start's forward is monotonic
# The validity gate of an axis fit from a folding start: its refined lens still folds before the widest angle of the region it
# was fitted over (the residuals capped at the fold: the axis can sit unmoved at a tolerance stop).
_folds_inside(r) = _fold_angle(r.u) < r.metadata.family.θ_max
# A camera's fold angle (its lens' `view_angle_limit`, radians), and a start's fold when it lies before `θ_max`, the
# widest angle of the region (`nothing` otherwise: `r.metadata.family.start_fold`).
_fold_angle(camera) = Float64(view_angle_limit(lens(camera)))
_start_fold(start, θ_max) = (fold = _fold_angle(start); fold < θ_max ? fold : nothing)
# The image disc inside a linear start's fold: about its principal point, out to its radius at `_INSIDE_FOLD` of the
# fold angle (the start's forward is monotonic there), so the refit's start images every sample.
function _inside_fold(start::CameraModel, fold::Float64)
    K = SMatrix{3, 3, Float64}(calibration_matrix(start))
    θ = _INSIDE_FOLD * fold
    ρ = norm(SVector{2, Float64}(VgtkCore.project(start, SVector(sin(θ), 0.0, cos(θ)))) - K[1:2, 3])
    return VgtkCore.Ellipse(SVector(K[1, 3], K[2, 3]), ρ, ρ, 0.0)
end
_optical_axis(camera::VgtkCore.AbstractCamera, region = nothing; kwargs...) = throw(
    ArgumentError(
        "OpticalAxis: a $(nameof(typeof(camera))) has no bound — the axis is fitted over the camera's own region; " *
            "wrap it, `BoundedCamera(camera, Size2(width, height); lens_coverage = <its image circle>)`"
    )
)
# The radially symmetric family of the optical axis: Fisheye624's form (COLMAP RAD_TAN_THIN_PRISM_FISHEYE) with
# `[optical_axis] radial_terms` radial coefficients (configs/default.toml, read once: 4, θ³…θ⁹; the derivation is
# beside the key) fitted, its decentering held at zero (`_AXIS_FIXED`: the tangential and thin-prism groups; the
# focal, aspect, principal point, radial coefficients and the axis free).
const _AXIS_TERMS = _dflt("optical_axis", "radial_terms")::Int
const _AXIS_FAMILY = typeof(
    VgtkCore.Lens(VgtkCore.Equidistant(), VgtkCore.Additive(VgtkCore.Tangential(0.0, 0.0), VgtkCore.ThinPrism((0.0, 0.0), (0.0, 0.0))) ∘
                  VgtkCore.Radial(ntuple(_ -> 0.0, _AXIS_TERMS)))
)
# The order continuation's first order (`[optical_axis] start_radial_terms`, configs/default.toml, read once; the
# derivation is beside the key), in 1:_AXIS_TERMS.
const _AXIS_START_TERMS = let m = _dflt("optical_axis", "start_radial_terms")::Int
    1 <= m <= _AXIS_TERMS || throw(ArgumentError("configs/default.toml [optical_axis]: start_radial_terms = $m must lie in 1:radial_terms ($_AXIS_TERMS)"))
    m
end
const _AXIS_FIXED = (:p, :sx, :sy)

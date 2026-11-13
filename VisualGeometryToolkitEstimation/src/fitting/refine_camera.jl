# ==============================================================================
# refine_camera — a camera model polished onto a source camera over a region of
# its sensor, from a guess; approximate_camera — fit_camera's linear start, then
# that refine
# ==============================================================================
#
# THE OBJECTIVE is fit_camera's (fit_camera.jl): the integrated SQUARED pixel error of the round trip over the region,
# by the same quadrature (a physical K's radial profile, a gauge K's image samples). fit_camera solves it in one
# linear least squares where the family has a linear form; refine_camera minimizes it exactly, by the stack's chart LM
# (Core's `lmsolve_chart!`), from the guess's lens and focal: the coefficients the family moves (a forward lens's
# projection coefficients and its radial factor's numerator and denominator, a lift's every coefficient) and the focal
# rescale, and for a GAUGE K (a SplineLens source) also the aspect, the principal point and the axis.

# ------------------------------------------------------------------------------
# The chart of the regression
# ------------------------------------------------------------------------------

# The coefficients the refine moves: the guess lens's manifold (`LensManifold`, Dual-safe) restricted by NAME to the
# `free` ones (Core's `restrict(ch, names)`, the one name→mask plumbing refine_camera_and_poses feeds), every other
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
# The minimal rotation taking the unit axis `a` onto ẑ (Rotations' `rotation_between`, Dual-safe).
_axis_rotation(a) = SMatrix{3, 3}(Rotations.rotation_between(a, SVector(0.0, 0.0, 1.0)))
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
# `residual_normal_equations!`) with ONE kernel (Core's `_launch_rows!`, the same on every backend, the CPU's
# included): each work item evaluates its samples' 2-D residuals — and, at a linearization, their Jacobian rows,
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
    az = iszero(VgtkCore._primal(ρ)) ? zero(SVector{2, typeof(ρ)}) : SVector(x[1], x[2]) / ρ
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

function residual_cost!(_, lm::_LensRegressionLm, x)
    VgtkCore._launch_rows!(_cost_row!, lm.partials, size(lm.partials, 2), lm.P, lm.angle_residual, lm.θcap, lm.M, lm.rays, lm.angles, lm.points, lm.sw, x)
    return VgtkCore.reduce_cost(lm.partials)
end
function residual_normal_equations!(_, _J, lm::_LensRegressionLm, x::SVector{N, Float64}) where {N}
    VgtkCore._launch_rows!(_normal_row!, lm.partials, size(lm.partials, 2), lm.P, lm.angle_residual, lm.θcap, lm.M, lm.rays, lm.angles, lm.points, lm.sw, x)
    return VgtkCore.reduce_normal_equations(lm.partials, Val(N))
end

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
# zero), its termination record `(code, rule)` and its iterations — a function barrier static in the N
# coordinates (the focal's and the family's). The workspace holds no rows: the state's reductions keep them on its
# device (one placeholder row, NLLSsolver's problem needs one).
function _regress_lens(lm::_LensRegressionLm, ::Val{N}, max_iter::Int) where {N}
    ews = _lm_workspace(nothing, 1, Val(N))      # the one LO-buffer reuse policy (interface.jl)
    x, _, _, iterations, record = lmsolve_chart!(ews, lm; maxit = max_iter, stationarity_tol = _CHART_LM_STATIONARITY_TOL)
    return x, record, iterations
end

# ==============================================================================
# refine_camera: the chart LM from a guess
# ==============================================================================

"""
    refine_camera(guess::CameraModel, source, region; fixed = (), weight = :image, max_iter, device = default_device(),
                  optimization_strategy = nothing) -> FitResult{<:CameraModel, CameraFitMetadata}

The camera of `guess`'s lens family polished onto `source` over `region` (a region of the source's sensor in its pixel
frame, as [`fit_camera`](@ref) takes it): the exact minimizer of the integrated squared pixel error of the round trip,
`∫ ‖π(π⁻¹_source(u)) − u‖² du`, by the stack's chart LM from `guess`. It is a `refine`: it needs the guess, and it
converges to the minimum nearest it. [`fit_camera`](@ref) gives a guess where the family has a linear form;
[`approximate_camera`](@ref) is the two in one call.

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
  radially symmetric camera represents. [`optical_axis`](@ref) is this refine of the Fisheye624 family.
- **The residual** is one 2-D residual per sample: a forward lens's is its pixel minus the sample's
  (`projected_radius` along the ray's azimuth, finite past a fold of its profile); a lift's is angle-style, in the
  equidistant chart about ẑ, scaled by the sample's normalized radius.
- **The device.** Each sample's residual and its Jacobian rows, and their reduction to the normal equations, are one
  KernelAbstractions kernel on the backend `device`; only the chart LM's N × N step is the host's.
- **The loss.** `optimization_strategy = nothing`, the plain least squares, is the only one: a camera-from-camera
  regression has no outliers. It takes no `scale` or `width` (one given would be ignored: an `ArgumentError`), and the
  result records no scale (`scale_source` `ScaleNone`, `scale` `NaN`; `noise_scale` refuses it).

The domain is the region's farthest radius, capped by the source's own view-angle limit, as `fit_camera`'s; a region
past the family's view-angle limit (the perspective plane's 90° for a gnomonic family) is a [`ViewAngleLimitError`](@ref). At most `max_iter` accepted iterations (configs/default.toml
`[refine_camera]`); the termination record (`retcode`, `stop_rule`) is the LM's, a refine that ends at a non-finite
focal rescale is `(ReturnCode.Failure, NonFinite)` (`:solve_failed`), never a throw, and one with nothing free (the
guess returned, no LM) is a closed form, `(ReturnCode.Success, ClosedForm)`. The metadata is [`CameraFitMetadata`](@ref): the round trip's errors per sample in
`residuals` (px), their maximum and edge-band maximum, the result's `rms` their weighted root mean square.

A `NoncentralSplineModel` source is an `ArgumentError`: not supported yet.

```julia
start = estimate(fit_camera(KannalaBrandt{4}, camera, region))       # the linear fit: the guess
kb = refine_camera(start, camera, region)                              # its exact pixel least squares
kb = approximate_camera(KannalaBrandt{4}, camera, region)              # the same two steps
```
"""
function refine_camera(
        guess::CameraModel, source::CameraModel, region;
        fixed::Tuple{Vararg{Symbol}} = (),
        weight::Symbol = :image, max_iter::Int = _dflt("refine_camera", "max_iter"), device = default_device(),
        optimization_strategy = nothing, scale = nothing, width = nothing,
        free = RetiredKeyword(), fix = RetiredKeyword(), fixed_params = RetiredKeyword(), hold = RetiredKeyword(),   # refused, naming `fixed`
    )
    _refuse_held_spellings(free, fix, fixed_params, hold)
    _check_l2_refine(optimization_strategy, scale, width)
    gl = lens(guess)
    gl isa Union{Lens, LiftLens} || throw(
        ArgumentError("refine_camera: the guess must carry a parametric lens (a Lens or a LiftLens), got a $(nameof(typeof(gl)))")
    )
    L = typeof(gl)
    moving = VgtkCore.free_parameter_names(guess, fixed)   # Core's one resolution of `fixed` (names and groups; the skew held)
    role = VgtkCore.Cameras.IntrinsicsRole(lens(source))
    K0 = SMatrix{3, 3, Float64}(calibration_matrix(source))
    M = _offset_matrix(K0)   # K₀₂ / fx₀ (fit_camera.jl)
    start = _refine_start(role, SMatrix{3, 3, Float64}(calibration_matrix(guess)), K0, M)
    pr = _profile_samples(role, L, source, region, K0, weight, device)
    P = _regression_manifold(role, _lens_family_manifold(role, gl, moving), start, moving)
    # a forward guess that folds before the domain's edge (a linear fit over a field its family cannot image) images
    # no radius past its fold: its residual there is capped at the fold, as the reported errors are
    # (a guess at the source's own fold, to its rounding, is not capped: 1e-6 rad)
    θcap = L <: LiftLens ? Inf : (t = Float64(view_angle_limit(gl)) - 1.0e-9; t < pr.θ_max - 1.0e-6 ? t : Inf)
    lm = _LensRegressionLm(device, P, L <: LiftLens, θcap, M, pr.rays, pr.angles, pr.points, sqrt.(pr.w))
    # nothing free (a physical K with `free = ()`; a gauge K's axis is always free): the guess itself, no LM
    p, record, iters = tangent_size(P) == 0 ? (SVector{0, Float64}(), (ReturnCode.Success, ClosedForm), 0) :
        _regress_lens(lm, Val(tangent_size(P)), max_iter)

    fit = _regression_parts(retract(P, p))
    # a regression that ends non-finite did not converge: its FitResult reports it
    isfinite(fit.s) || (record = (ReturnCode.Failure, NonFinite))
    # the round trip's error per sample, in pixels: the forward form, its radius capped inside the refined lens's domain
    e = K0[1, 1] .* _sample_errors(lm, p, _domain_cap(L, fit.lens, isfinite(fit.s), pr.θ_max))   # (fit_camera.jl)
    return _lens_fit_result(role, fit, K0, M, pr, e, region, weight, record, iters)
end
# A bounded camera's model is the source (its image does not enter: the region bounds the fit).
refine_camera(guess::CameraModel, source::BoundedCamera, region; kwargs...) = refine_camera(guess, model(source), region; kwargs...)
refine_camera(guess::CameraModel, source::NoncentralSplineModel, region; kwargs...) = throw(
    ArgumentError("refine_camera: a non-central source (NoncentralSplineModel) is not supported yet: a radially symmetric central camera fits a central source")
)

# The L2 refine's keywords: no loss, so no scale input (one given would be ignored, so it is refused by the bundles'
# one reader, `bundle_loss`, whose least squares takes none).
function _check_l2_refine(optimization_strategy, scale, width)
    optimization_strategy === nothing || throw(
        ArgumentError(
            "refine_camera: the camera-from-camera regression is plain least squares (`optimization_strategy = nothing`); " *
                "its samples are a source model's, with no outliers to weigh down, got $(nameof(typeof(optimization_strategy)))"
        )
    )
    bundle_loss(nothing; scale, width, caller = "refine_camera")
    return nothing
end

# The chart's start from the guess's K against the source's `K0`: the focal rescale `s` (fx / fx₀), the aspect `a`
# (fy / (s·fy₀)) and the principal point's move `δ` in the source's normalized plane (K₀₂·δ pixels). A PHYSICAL K holds
# the principal point and the aspect, which are data: a guess that moves them is refused.
function _refine_start(::VgtkCore.Cameras.FreeIntrinsics, Kg, K0, M)
    s = Kg[1, 1] / K0[1, 1]
    a = Kg[2, 2] / (s * K0[2, 2])
    tol = 1.0e-9 * max(1.0, abs(K0[1, 1]))
    (abs(Kg[1, 3] - K0[1, 3]) <= tol && abs(Kg[2, 3] - K0[2, 3]) <= tol && abs(a - 1) <= 1.0e-9) || throw(
        ArgumentError(
            "refine_camera: a physical source's principal point $((K0[1, 3], K0[2, 3])) and aspect are held; " *
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
    ArgumentError("refine_camera: the guess's focal lengths must be positive and finite relative to the source's (rescale $(st.s), aspect $(st.a))")
)

# ==============================================================================
# approximate_camera: fit_camera's linear start, then refine_camera
# ==============================================================================

"""
    approximate_camera(::Type{L}, source, region; weight = :image, max_iter, device = default_device())
        -> FitResult{<:CameraModel, CameraFitMetadata}

The camera of lens family `L` closest to `source` over `region`, from no guess:
`refine_camera(estimate(fit_camera(L, source, region)), source, region)` — the linear fit's camera is the refine's
start ([`fit_camera`](@ref), [`refine_camera`](@ref)). The keywords split to their stages: `weight` and `fixed` to
both (the held coefficients keep the family's anchor values through both), `device` to both (a gauge-K source's rays are
sampled on it), `max_iter` to the refine. The result is the refine's.

A family with no linear form (a projection with a shape parameter — `DoubleSphere`, `Unified`, `ExtendedUnified` —
or a `Tilt` or `FieldOfView` term) starts from its anchor instead: the family's zero lens (the double sphere at
ξ = α = ½) at the source's K. So does a linear fit that failed (`:solve_failed`). `Division` alone names
`Division{1}`.

```julia
kb = approximate_camera(KannalaBrandt{4}, camera, region)     # a SplineLens camera's parametric lens (the exporters' route)
```
"""
function approximate_camera(
        ::Type{L}, source::CameraModel, region; weight::Symbol = :image, fixed::Tuple{Vararg{Symbol}} = (), device = default_device(),
        kwargs...
    ) where {L <: AbstractLensModel}
    start = _approximation_start(_linear_form(L), L, source, region, weight, fixed, device)
    return refine_camera(start, source, region; weight, fixed, device, kwargs...)
end
approximate_camera(::Type{L}, source::BoundedCamera, region; kwargs...) where {L <: AbstractLensModel} = approximate_camera(L, model(source), region; kwargs...)
approximate_camera(::Type{L}, source::NoncentralSplineModel, region; kwargs...) where {L <: AbstractLensModel} = throw(
    ArgumentError("approximate_camera($L, …): a non-central source (NoncentralSplineModel) is not supported yet: a radially symmetric central camera fits a central source")
)
approximate_camera(::Type{Division}, source::CameraModel, region; kwargs...) = approximate_camera(Division{1}, source, region; kwargs...)

# The refine's start: the linear fit's camera, or the family's anchor at the source's K (no linear form, or a failed
# linear fit).
function _approximation_start(form, ::Type{L}, source, region, weight, fixed, device) where {L}
    r = fit_camera(L, source, region; weight, fixed, device)
    return converged(r) ? estimate(r) : _anchor_camera(L, source)     # a failed linear solve: the anchor
end
_approximation_start(::_NoLinearForm, ::Type{L}, source, region, weight, fixed, _device) where {L} = _anchor_camera(L, source)
# The camera of the source's K with the family's anchor lens, or with a given lens.
_anchor_camera(::Type{L}, source) where {L} = _anchor_camera(_target_anchor(L), source)
_anchor_camera(a::AbstractLensModel, source) =
    CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(calibration_matrix(source)); check = false)), a)

# ==============================================================================
# optical_axis: the approximation of the Fisheye624 family
# ==============================================================================

"""
    optical_axis(camera::BoundedCamera; device = default_device()) -> FitResult{<:CameraModel, CameraFitMetadata}

The radially symmetric camera closest to `camera` over the camera's own bound, its `imaged_region` (the sensor ∩ its
lens coverage ∩ the model's valid domain). It is [`approximate_camera`](@ref) of the stack's Fisheye624 family
(COLMAP's `RAD_TAN_THIN_PRISM_FISHEYE`, `Lens(Equidistant(), Additive(Tangential, ThinPrism{2}) ∘ Radial{Even, 6})`,
its decentering held at zero) onto `camera`, that one call, nothing more: the linear fit of its radial coefficients
and focal, then the refine with the principal point and the axis free.
- The region is the bound, never a default. A circular fisheye's lens coverage (its image circle) keeps the corners
  out, where a grid's rays are its extrapolation. A whole-sensor fit is a camera whose bound is the whole sensor.
- A bare model has no bound: an `ArgumentError`. Wrap it in `BoundedCamera(model, size; lens_coverage)`.
- For a SplineLens camera (a gauge `K`), the principal point and the axis are free. Read them from the result.
  - The optical axis in the camera's frame is `rotation(metadata(r).rotation)' * ẑ`.
  - Its pixel is the fitted camera's principal point.
  - `metadata(r).symmetry_residual` (px) is the azimuthal misfit that no radially symmetric camera represents. A
    decentred lens reports it.
- A physical `K` is its own axis: the identity rotation and the source's principal point.

`device` is the backend the camera's rays are sampled and the regression reduced on (`approximate_camera`'s).

Core's `axis_gauged(camera, rotation(metadata(r).rotation), calibration_matrix(estimate(r)))` then turns the camera so ẑ is the
axis. `fit_camera(SplineLens, …)` does this first, always (a held SplineLens camera's `refine_camera_and_poses` does not:
it returns the camera as given).
"""
optical_axis(camera::BoundedCamera; device = default_device()) =
    approximate_camera(_AXIS_FAMILY, camera, VgtkCore.imaged_region(camera); fixed = _AXIS_FIXED, device)
optical_axis(camera::VgtkCore.AbstractCamera; device = nothing) = throw(
    ArgumentError(
        "optical_axis: a $(nameof(typeof(camera))) has no bound — the axis is fitted over the camera's own region; " *
            "wrap it, `BoundedCamera(camera, Size2(width, height); lens_coverage = <its image circle>)`"
    )
)
# The radially symmetric family of the optical axis: Fisheye624 (COLMAP RAD_TAN_THIN_PRISM_FISHEYE), its six radial
# coefficients fitted, its decentering held at zero (`_AXIS_FIXED`: the tangential and thin-prism groups; the focal,
# aspect, principal point, radial coefficients and the axis free).
const _AXIS_FAMILY = typeof(
    VgtkCore.Lens(VgtkCore.Equidistant(), VgtkCore.Additive(VgtkCore.Tangential(0.0, 0.0), VgtkCore.ThinPrism((0.0, 0.0), (0.0, 0.0))) ∘ VgtkCore.Radial(ntuple(_ -> 0.0, 6)))
)
const _AXIS_FIXED = (:p, :sx, :sy)

# ==============================================================================
# fit_lens — model-to-model regression on the radial profile r(θ)
# (moved from VisualGeometryToolkitCore, estimator taxonomy v2; `view_angle_limit` and
# `radial_part` — lens properties — stay there)
# ==============================================================================
#
# Every central radially symmetric lens is characterized by its radial
# projection function r_d(θ) (`projected_radius` — the common currency the
# retinal-frame machinery already commits to). Model-to-model regression is
# therefore NOT a pair matrix: any source that can EVALUATE r_A(θ) regresses
# onto any target whose parameters can be FIT, through one driver.
#
# THE OBJECTIVE IS THE CONTINUOUS L² FUNCTIONAL
#
#     min_{p,s} ∫₀^θmax w(θ) · (s·r_B(θ; p) − r_A(θ))² dθ
#
# evaluated by CLENSHAW–CURTIS quadrature on Chebyshev extrema nodes: the
# residual vector is e_i = √(q_i·w(θ_i))·Δr(θ_i) with q_i the CC weights, so
# Σe_i² IS the quadrature of the integral and Levenberg–Marquardt on e
# minimizes the integral itself. Chebyshev nodes cluster at both endpoints —
# exactly where profiles diverge worst; Clenshaw–Curtis (not Gauss–Chebyshev,
# whose native weight is 1/√(1−x²)) carries the correct weights for the
# plain dθ integral and is exponentially accurate on smooth profiles.
#
# DOMAIN POLICY (the fisheye ↔ Brown–Conrady question): the fit domain is
# the INTERSECTION min(source limit, target-representable limit, user max_view_angle
# cap) — never silently truncated: the result reports θ_max used vs the
# source's own limit and warns when the target throws field away (a
# perspective-plane target simply cannot represent θ ≥ 90°; "valid to 85°"
# is an honest product, hiding the clip is not).
#
# NON-RADIAL TERMS: tangential/thin-prism decentering has no radial
# representation in ANY symmetric target — a fact about the models, not the
# method. A SOURCE contributes its radial part (decentering stripped, warned
# when non-negligible); a forward TARGET moves its projection's coefficients
# and its radial factor's numerator, its denominator and decentering terms
# held at zero; a lift TARGET moves every coefficient.

# ------------------------------------------------------------------------------
# Clenshaw–Curtis on [0, b]: n+1 Chebyshev extrema nodes + weights for ∫₀ᵇ f dθ
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Target manifolds: the family anchored at the fit's start (`LensManifold`, Dual-safe),
# restricted to the coefficients this regression moves
# ------------------------------------------------------------------------------
# The target's anchor: its zero lens (the double sphere at ξ = α = ½, inside
# the logit's interval), and the coefficients the fit moves — a forward lens's
# projection coefficients and radial numerator, a lift's every coefficient.
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
# before the lens family's manifold — or the family's alone for a fixed focal.
_regression_manifold(ch::Manifold, fix_focal::Bool) =
    fix_focal ? ProductManifold(ch) : ProductManifold(EuclideanManifold(SVector(0.0)), ch)
# The focal scale and the lens at chart coordinates `pv`. Dual-safe.
function _scale_and_lens(P::ProductManifold, pv)
    parts = retract(P, pv)
    return length(parts) == 2 ? exp(first(parts)[1]) : one(eltype(pv)), last(parts)
end

# Representable limit of a target FAMILY (parameter-free bound used for the
# fit-domain intersection before parameters exist). A family whose domain its
# parameters do not move (the perspective plane, the fixed fisheye projections)
# answers VGC's own `view_angle_limit` (the one owner of the edge and its
# margin); the polynomial and lift families are unbounded a priori and take the
# sphere's edge, relying on the source/max_view_angle cap.
# A forward family with a fixed projection is bounded by its projection's edge
# (the perspective plane at 90°); every other family by the sphere's.
_family_limit(::Type{<:Lens{F}}) where {F <: Union{Perspective, Equidistant, Stereographic, Equisolid, Orthographic}} =
    view_angle_limit(Lens(F()))
_family_limit(::Type) = view_angle_limit(Lens(Equidistant()))

"""
    LensFitMetadata

`FitResult` metadata of a lens-family regression: the joint focal rescale
`s`, the fitted/covered angular domains (`θ_max`, `θ_source`, `clipped`),
the worst node error `max_px`, and the fit `direction` (`:forward` — the
family's own profile; `:inverse` — the family fitted to the SOURCE's
inverse profile, giving an analytic complement with the same
parameterization) and `weight` measure.
"""
struct LensFitMetadata
    s::Float64
    θ_max::Float64
    θ_source::Float64
    clipped::Bool
    max_px::Float64
    direction::Symbol
    weight::Symbol
end

# Node/target construction for the two fit directions, one sampler: Clenshaw–
# Curtis nodes on [0, b] of the profile variable, the target radii `radius(t)`
# there, and the density w (`:image` differentiates `radius` by central
# differences, clamped to [0, b]). Forward: the source's own r(θ). Inverse:
# nodes in the DISTORTED field (φ = atan r_d), targets tan θ(φ) through VGC's
# `view_angle` (the inverse of `projected_radius`) — the fit then makes the
# inverse analytic in the SAME family (L⁻ ∘ L ≈ id to the reported rms).
function _profile_samples(radius, b::Float64, n::Int, weight::Symbol)
    ts, qs = _clenshaw_curtis(n, b)
    rA = [radius(t) for t in ts]
    h = 1.0e-5 * b
    dens = map(eachindex(ts)) do i
        weight === :uniform && return 1.0
        weight === :solid_angle && return sin(ts[i])
        weight === :image || throw(ArgumentError("fit_lens: weight must be :image, :solid_angle or :uniform, got :$weight"))
        tm, tp = max(ts[i] - h, 0.0), min(ts[i] + h, b)
        dr = (radius(tp) - radius(tm)) / (tp - tm)
        return max(rA[i] * abs(dr), 0.0)
    end
    return ts, qs, rA, dens
end

_forward_samples(srcr, θ_max::Float64, n::Int, weight::Symbol) =
    _profile_samples(θ -> Float64(projected_radius(srcr, θ)), θ_max, n, weight)

function _inverse_samples(srcr, θ_max::Float64, n::Int, weight::Symbol)
    rmax = Float64(projected_radius(srcr, θ_max))
    # tan(atan(r)) may round past `rmax`, which at a lens's domain edge would
    # leave the domain: the argument is held to it, as a bracket on [0, θ_max] was
    return _profile_samples(φ -> tan(Float64(view_angle(srcr, min(tan(φ), rmax)))), atan(rmax), n, weight)
end

"""
    fit_lens(::Type{L}, model::CameraModel;
             max_view_angle = nothing, max_radius_px = nothing, n = 128, weight = :image,
             fix_focal = false, max_iter, direction = :forward)   # max_iter: configs/default.toml [fit_lens]
        -> FitResult{CameraModel, LensFitMetadata}

Regress lens family `L` against `model`'s lens by minimizing the integral of
the squared radial-profile difference over the shared view-angle domain
(see the file header: Clenshaw–Curtis quadrature on Chebyshev nodes, so the
discrete objective IS the integral). The focal scale `s` is fit jointly
(frames differ in slope at the center — DoubleSphere especially), so the
result is a full `CameraModel`: `K` scaled by `s`, principal point kept.

- `max_view_angle`: optional cap on the fit domain, a HALF-angle — the largest
  view angle θ from the optical axis fitted, in radians (VGC `view_angle`,
  `view_angle_limit`); not the full angle of view, Core's `fov(camera)`. Until
  2026-10-27 this keyword was `fov`.
- `max_radius_px`: optional cap by the sensor's coverage — the farthest
  LIT image point's distance from the principal point in pixels. For a
  full-field lens that is the frame corner (`hypot(cx, cy)` for a centered
  pp); for a CIRCULAR fisheye it is the image-circle radius — for a
  division-family source, `f·monotone_radius(lens)` IS the 180° edge, and
  the model past it is unconstrained extrapolation no rigid target should
  be asked to follow. Folded into the SOURCE domain, so it does not count
  as clipping.
- `n`: quadrature order (n+1 nodes).
- `weight`: the density `w(θ)` — `:image` (pixel-area, `r·r′`; the right
  default when the target replaces the source on the same camera),
  `:solid_angle` (`sin θ`), or `:uniform`.
- `fix_focal`: freeze `s = 1`.
- `max_iter`: the LM's cap on ACCEPTED iterations (NLLSsolver's count; one
  accepted iteration may hide ~8–10 rejected candidate steps, which
  FastLevenbergMarquardt, the driver until 2026-10-27, counted too).

Returns a `FitResult{CameraModel, LensFitMetadata}`: `estimate` is the
fitted camera (rescaled K + the fitted lens), `retcode` the LM's
(VGC `lm_retcode`: `:converged`, `:max_iterations`, `:max_time`, `:stalled` —
stopped on its step size at a non-stationary point — or `:solve_failed`, the
cost or the step non-finite outside a step-size stop), `rms` the weighted profile
RMS in px, and the metadata carries `s`, the angular domains, `max_px`,
the fit `direction` and `weight`; `clipped` is true when the target family
or `max_view_angle` cut the source's own domain (warned).

Targets: any lens TYPE with its arities — `KannalaBrandt{4}`, `Division{2}`
(`Division` alone fits `Division{1}`), `Scaramuzza{3}`, `LiftLens{Radial{Even, P, Q}}`,
`Lens{Perspective, Radial{Even, 3, 0}}` (OpenCV's radial polynomial), a
double sphere `Lens{DoubleSphere}`, `Pinhole` and the pure projections
(scale-only), or the type of a lens with decentering terms. A forward
target moves its projection's coefficients and its radial factor's
numerator — its denominator and decentering terms stay zero; a lift moves
every coefficient. Sources: any lens with a [`projected_radius`](@ref)
profile — a lift evaluates past 90° through its ray lift, and a lens with
decentering terms contributes its radial part (warned when non-negligible).
"""
function fit_lens(
        ::Type{L}, model::CameraModel;
        max_view_angle::Union{Nothing, Real} = nothing, max_radius_px::Union{Nothing, Real} = nothing,
        n::Int = 128, weight::Symbol = :image, fix_focal::Bool = false, max_iter::Int = _dflt("fit_lens", "max_iter"),
        direction::Symbol = :forward
    ) where {L <: AbstractLensModel}
    direction in (:forward, :inverse) || throw(ArgumentError("fit_lens: direction must be :forward or :inverse, got :$direction"))
    src = lens(model)
    dec = _decentering(src)
    dec > 1.0e-8 &&
        @warn "fit_lens: the source $(nameof(typeof(src))) carries non-radial terms (decentering, norm $(round(dec; sigdigits = 3))) — no radially symmetric target can represent them; fitting the radial part only"

    srcr = radial_part(src)

    # The source's COVERED field: its model domain, capped by what the sensor
    # actually images when the caller says so (`max_radius_px` — the farthest
    # image point's distance from the principal point, in pixels). An omni
    # model's lift can extend far past the frame (the division family folds
    # with a vertical r(θ) no polynomial target can chase); fitting field the
    # camera never sees is neither honest nor useful.
    θ_source = view_angle_limit(srcr)
    if max_radius_px !== nothing
        ρmax = float(max_radius_px) / Float64(calibration_matrix(model)[1, 1])
        θ_source = min(θ_source, Float64(view_angle(srcr, ρmax)))
    end
    # The inverse direction's targets are PERSPECTIVE radii, tan θ (see
    # `_inverse_samples`): the plane ends at 90°, where tan θ diverges and past
    # which it turns negative, whatever the family. So that domain is capped at
    # the perspective plane's edge, exactly as a perspective TARGET caps the
    # forward one — clipped and warned, never fitted to garbage.
    θ_inv = direction === :inverse ? _family_limit(typeof(Pinhole())) : Inf
    θ_max = min(θ_source, _family_limit(L), θ_inv, max_view_angle === nothing ? Inf : float(max_view_angle))
    θ_max > 0.02 || throw(ArgumentError("fit_lens: empty fit domain (θ_max = $θ_max)"))
    clipped = θ_max < θ_source * (1 - 1.0e-6)
    clipped && max_view_angle === nothing && @warn "fit_lens: $(θ_max == θ_inv ? "the inverse profile (tan θ, the perspective plane)" : "target family $(L)") cannot represent the source's field — fitting θ ∈ [0, $(round(rad2deg(θ_max); digits = 1))°] of the source's $(round(rad2deg(θ_source); digits = 1))°"

    # Quadrature of the L² functional: nodes, CC weights, density (θ-domain
    # forward; φ = atan r_d domain for the inverse fit — see the samplers).
    θs, qs, rA, dens = direction === :forward ?
        _forward_samples(srcr, float(θ_max), n, weight) :
        _inverse_samples(srcr, float(θ_max), n, weight)
    sw = sqrt.(qs .* dens)
    wsum = sum(qs .* dens)

    # The rows (`_lens_rows`) in the chart [log s | the family's manifold],
    # anchored at the start, solved by the stack's chart LM.
    P = _regression_manifold(_target_manifold(L), fix_focal)
    lm = _LensRegressionLm(P, L <: LiftLens, θs, rA, sw)
    p, code, iters = _regress_lens(lm, length(θs), Val(tangent_size(P)), max_iter)

    s, lensB = _scale_and_lens(P, p)
    K0 = calibration_matrix(model)
    fx = Float64(K0[1, 1])
    Kb = CameraCalibrationMat(SMatrix{3, 3, Float64}(K0) * LinearAlgebra.Diagonal(SVector(s, s, 1.0)); check = false)   # focal and skew by s
    out = CameraModel(LogicalIntrinsics(Kb), lensB)
    θcap = L <: LiftLens ? view_angle_limit(lensB) - 1.0e-9 : Inf
    Δ = [s * Float64(projected_radius(lensB, min(θs[i], θcap))) - rA[i] for i in eachindex(θs)]
    rms_px = sqrt(sum(qs .* dens .* abs2.(Δ)) / max(wsum, eps())) * fx
    max_px = maximum(abs.(Δ)) * fx
    return FitResult(
        out, code, iters, rms_px, length(θs),
        LensFitMetadata(s, float(θ_max), θ_source, clipped, max_px, direction, weight)
    )
end

# ------------------------------------------------------------------------------
# The regression's least squares — the stack's chart LM (VGC's
# `lmsolve_chart!`: NLLSsolver) over the
# `residual_rows!` protocol
# ------------------------------------------------------------------------------

# The linearization state: the chart `P` (`_regression_manifold`), whether the
# rows are angle residuals (the lift families), the nodes, the target radii and
# the rows' √(q·w).
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
# coordinates. Nothing to move (a fixed focal on a parameterless family): the
# anchor, converged.
_regress_lens(::_LensRegressionLm, ::Int, ::Val{0}, ::Int) = (zero(SVector{0, Float64}), :converged, 0)
function _regress_lens(lm::_LensRegressionLm, rows::Int, ::Val{N}, max_iter::Int) where {N}
    ews = _lm_workspace(nothing, rows, Val(N))      # the one LO-buffer reuse policy (interface.jl)
    x, _, conv, iterations = lmsolve_chart!(ews, lm; maxit = max_iter, stationarity_tol = _CHART_LM_STATIONARITY_TOL)
    return x, VgtkCore.lm_retcode(conv), iterations
end

# `Division` alone names the family, not a degree: fit the one-coefficient model,
# as `Division(k)` constructs it. Any other type must carry its arities (`zero(L)`
# refuses one that does not).
fit_lens(::Type{Division}, model::CameraModel; kwargs...) = fit_lens(Division{1}, model; kwargs...)

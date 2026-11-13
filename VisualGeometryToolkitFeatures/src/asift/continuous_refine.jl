# =============================================================================
# Continuous DoG-extremum refinement (research path; NOT called by `asift_detect`).
# =============================================================================
#
# Today's refinement (`_extract_at!`, driver.jl) takes one Newton step on the 3×3×3
# finite-difference quadratic at an integer extremum (VLFeat's rule). Its sampling
# bias is constant in level pixels (×2^o in image pixels) and jumps near half-sample
# offsets (featcov-diagnosis.md). This file refines the same candidates on a
# CONTINUOUS model of the DoG response D(x, y, s) instead, from today's refined point:
#
#   • the response model — any `_dog_response(model, v) -> (D, ∇D, ∇²D)` at a fractional
#     v = (x, y, s) (1-based octave-level coordinates: level pixels and the DoG slice);
#     the product one is `_DogSplineWindow`: per DoG slice the cardinal (interpolating)
#     cubic spline of the slice's samples, C² in (x, y), and across slices the degree-4
#     Lagrange polynomial through five slices, C^∞ in s. Its local B-spline
#     coefficients come from ONE KernelAbstractions kernel (`_dog_spline_coefficients!`,
#     CPU and GPU alike) for all of an octave's detections;
#   • the iteration — Core's ONE device-side LM (`static_lm`, the spline lenses'
#     inverse at N = 3: Marquardt-scaled, bounded, explicit fused products), IN the
#     kernel (`_dog_extrema_kernel!`, one detection per work item), on the rows
#     r(θ) = ∇_θ D/|D(v₀)|, θ the scale-normalised displacement, with Jacobian
#     ∇²_θ D/|D(v₀)|: the undamped step is Newton's, the LM safeguards it, and the
#     response, gradient and Hessian are re-evaluated at every trial;
#   • the gates — the displacement box (the LM's bounds), the extremum type of the
#     start, and a curvature floor; a failure returns the START (the last valid
#     estimate) with its §14 record (`ReturnCode`, `StopRule`);
#   • the covariance — the implicit function theorem at the final point with the
#     model's OWN derivative functionals: δv = −H⁻¹·δ∇D, ∇D linear in the DoG samples
#     through the spline and Lagrange weights, the samples linear in the image through
#     the pyramid's adjoint (uncertainty.jl's `_canvas_response`), so
#     Σ_v = H⁻¹·Cov(∇D)·H⁻¹ (no δH·off term: ∇D(v*) = 0 exactly at C's point).
#
# Bounds and tolerances: configs/continuous_refinement.toml.
# =============================================================================

using VisualGeometryToolkitCore: ReturnCode, StopRule, successful_retcode, LeftBasin, ShapeGate, NonFinite,
    static_lm, static_lm_termination, StaticLmStops, RetiredKeyword, refuse_retired_keyword, cubic_bspline_weights, cubic_bspline_weight_derivatives, cubic_bspline_prefilter_taps,
    spline_window_sums, is_positive_definite, parse_defaults, toml_default

const _CR_DEFAULTS = parse_defaults(joinpath(dirname(dirname(@__DIR__)), "configs", "continuous_refinement.toml"))
_cr(key) = toml_default(_CR_DEFAULTS, "continuous_refinement", key)

"""
    ContinuousRefinement(; prefilter_halfwidth, spatial_bound, scale_bound, min_curvature)

The continuous DoG-extremum refinement, opt-in: `asift_detect(img; refinement = ContinuousRefinement())`
(the detector's default, `refinement = nothing`, is today's one-step refinement). One recipe: the
spatial gradient is Farid and Simoncelli's matched first-derivative filters on the prefiltered slices
(IEEE TIP 2004), interpolated by the cardinal cubic spline; the scale row is the cardinal spline of the
UNFILTERED slices, Lagrange in scale — so the prefilter sharpens the position and leaves the scale
unbiased; the LM's Jacobian is the exact derivative of that gradient. The keywords are the bounds of
`_refine_dog_extremum`; every default is configs/continuous_refinement.toml's (its comments give each
derivation). The LM's stop settings are VisualGeometryToolkitCore's, the stack's one owner (`StaticLmStops`,
its `[lm]` table); `max_iter`, `converged_tol2` and `cost_floor` are retired here. An isbits value, so a
kernel takes it.
"""
struct ContinuousRefinement{M}
    prefilter_halfwidth::Int
    spatial_bound::Float64
    scale_bound::Float64
    min_curvature::Float64
end
ContinuousRefinement(; kw...) = _continuous_refinement(:farid_simoncelli_spatial; kw...)
# The refinement on response `M`: the recipe (`:farid_simoncelli_spatial`), or the plain spline response (`:spline`,
# stage C: its exact derivatives) that the research benchmark (benchmark/continuous_refine) ablates against.
function _continuous_refinement(
        M::Symbol; prefilter_halfwidth::Integer = _cr("prefilter_halfwidth"),
        spatial_bound::Real = _cr("spatial_bound"), scale_bound::Real = _cr("scale_bound"), min_curvature::Real = _cr("min_curvature"),
        max_iter = RetiredKeyword(), converged_tol2 = RetiredKeyword(), cost_floor = RetiredKeyword()
    )
    for (value, name) in ((max_iter, :max_iter), (converged_tol2, :converged_tol2), (cost_floor, :cost_floor))
        refuse_retired_keyword(value, name, "VisualGeometryToolkitCore's LM stop settings, the stack's one owner (`StaticLmStops`, `[lm]`)")
    end
    M in (:spline, :farid_simoncelli_spatial) || throw(ArgumentError("the continuous refinement's response is :farid_simoncelli_spatial or :spline, got :$M"))
    return ContinuousRefinement{M}(
        Int(prefilter_halfwidth), Float64(spatial_bound), Float64(scale_bound), Float64(min_curvature)
    )
end
_response(::ContinuousRefinement{M}) where {M} = M

# ----- the cardinal cubic spline ---------------------------------------------


# The spline's coefficient window half-width RC: an evaluation at x ∈ x₀ ± (½ + bound)
# reads the coefficients of its span, floor(ξ) − 1 … floor(ξ) + 2.
_coef_halfwidth(cfg::ContinuousRefinement) = ceil(Int, cfg.spatial_bound + 2.5)
# Slices of the Lagrange window: five (degree 4), or all of a shorter octave.
_slice_window(NS::Integer) = min(5, Int(NS))

"""
    _DogSplineWindow

The product response model of one detection: the B-spline coefficients `coef[i, j, t, f]`
of the cardinal cubic spline of field `f` of DoG slices `s0 … s0 + NSW − 1` at level
pixels `(X0 + i, Y0 + j)` (1-based), combined across slices by the Lagrange polynomial
through them. `_dog_response(m, v)` gives D, ∇D and ∇²D at v = (x, y, s). Its fields
`F` say which response:

- `F = 1` (stage C): the slice itself; D is the spline-and-Lagrange interpolant of the
  DoG samples and ∇D, ∇²D its exact derivatives;
- `F = 3` (the recipe): the unfiltered slice, d₁⊗p and p⊗d₁ (Farid–Simoncelli's first
  derivative along one axis, their matched prefilter along the other); the spatial gradient is the
  matched fields', the scale row the unfiltered spline's, and the Jacobian the exact derivative of
  that gradient (H[i, j] = ∂gᵢ/∂vⱼ, not symmetric).
"""
struct _DogSplineWindow{N, F, A <: AbstractArray{<:Real, 4}}
    coef::A     # (nx, ny, N, F)
    X0::Int     # coefficient i sits at level pixel X0 + i
    Y0::Int
    s0::Int     # window slice t is DoG slice s0 + t − 1
end
_DogSplineWindow(coef::AbstractArray{<:Real, 4}, X0::Integer, Y0::Integer, s0::Integer) =
    _DogSplineWindow{size(coef, 3), size(coef, 4), typeof(coef)}(coef, Int(X0), Int(Y0), Int(s0))

# The Lagrange basis on the nodes 1…N at τ and its first two derivatives.
@inline function _lagrange_basis(τ::T, ::Val{N}) where {T, N}
    L = ntuple(Val(N)) do t
        p = one(T); dp = zero(T); d2p = zero(T)
        for m in 1:N
            m == t && continue
            f = (τ - m) / (t - m); df = one(T) / (t - m)
            d2p = d2p * f + 2 * dp * df
            dp = dp * f + p * df
            p = p * f
        end
        (p, dp, d2p)
    end
    return map(first, L), map(l -> l[2], L), map(l -> l[3], L)
end

# Window tap (r, c) of slice `t`, field `f` of a coefficient window (Core's `spline_window_sums` reads it: row taps
# along y, column taps along x), in the evaluation's type.
@inline _slice_control(coef, i0, j0, t, f, ::Type{T}) where {T} = (r, c) -> @inbounds(T(coef[i0 + c - 2, j0 + r - 2, t, f]))

"""
    _dog_response(model, v) -> (D, g::SVector{3}, H::SMatrix{3, 3})

The continuous DoG response, its gradient and its Hessian at v = (x, y, s) (1-based
level pixels and DoG slice) of a response model. Kernel-lawful for a
`_DogSplineWindow`.
"""
@inline function _dog_response(m::_DogSplineWindow{N, F}, v::SVector{3, T}) where {N, F, T}
    ξ = v[1] - m.X0; η = v[2] - m.Y0; τ = v[3] - m.s0 + 1
    i0 = unsafe_trunc(Int, floor(ξ)); j0 = unsafe_trunc(Int, floor(η))
    ux = ξ - i0; uy = η - j0
    wx = cubic_bspline_weights(ux); dx, ddx = cubic_bspline_weight_derivatives(ux)
    wy = cubic_bspline_weights(uy); dy, ddy = cubic_bspline_weight_derivatives(uy)
    return _dog_response_n(m.coef, i0, j0, τ, wx, dx, ddx, wy, dy, ddy, T, Val(N), Val(F))
end

@inline function _dog_response_n(coef, i0, j0, τ, wx, dx, ddx, wy, dy, ddy, ::Type{T}, ::Val{N}, ::Val{1}) where {T, N}
    L, dL, d2L = _lagrange_basis(τ, Val(N))
    D = zero(T); gx = zero(T); gy = zero(T); gs = zero(T)
    hxx = zero(T); hxy = zero(T); hyy = zero(T); hxs = zero(T); hys = zero(T); hss = zero(T)
    for t in 1:N
        s00, s10, s01, s20, s11, s02 = spline_window_sums(_slice_control(coef, i0, j0, t, 1, T), zero(T), wx, wy, dx, dy, ddx, ddy)
        D += L[t] * s00
        gx += L[t] * s10; gy += L[t] * s01; gs += dL[t] * s00
        hxx += L[t] * s20; hxy += L[t] * s11; hyy += L[t] * s02
        hxs += dL[t] * s10; hys += dL[t] * s01; hss += d2L[t] * s00
    end
    g = SVector{3, T}(gx, gy, gs)
    H = SMatrix{3, 3, T}(hxx, hxy, hxs, hxy, hyy, hys, hxs, hys, hss)
    return D, g, H
end

# The recipe's three fields: the unfiltered slice (field 1: D and the scale row), d₁⊗p and p⊗d₁ (fields 2, 3: the
# spatial gradient); the Jacobian is the interpolant's exact derivative of those rows (H[i, j] = ∂gᵢ/∂vⱼ, not
# symmetric), so the LM's Jacobian is the Jacobian of the function it zeroes.
@inline function _dog_response_n(coef, i0, j0, τ, wx, dx, ddx, wy, dy, ddy, ::Type{T}, ::Val{N}, ::Val{3}) where {T, N}
    L, dL, d2L = _lagrange_basis(τ, Val(N))
    D = zero(T); gx = zero(T); gy = zero(T); gs = zero(T)
    Hxx = zero(T); Hxy = zero(T); Hxs = zero(T); Hyx = zero(T); Hyy = zero(T); Hys = zero(T); Hsx = zero(T); Hsy = zero(T); Hss = zero(T)
    for t in 1:N
        p00, p10, p01 = spline_window_sums(_slice_control(coef, i0, j0, t, 1, T), zero(T), wx, wy, dx, dy)
        a00, a10, a01 = spline_window_sums(_slice_control(coef, i0, j0, t, 2, T), zero(T), wx, wy, dx, dy)
        b00, b10, b01 = spline_window_sums(_slice_control(coef, i0, j0, t, 3, T), zero(T), wx, wy, dx, dy)
        D += L[t] * p00
        gx += L[t] * a00; gy += L[t] * b00; gs += dL[t] * p00
        Hxx += L[t] * a10; Hxy += L[t] * a01; Hxs += dL[t] * a00
        Hyx += L[t] * b10; Hyy += L[t] * b01; Hys += dL[t] * b00
        Hsx += dL[t] * p10; Hsy += dL[t] * p01; Hss += d2L[t] * p00
    end
    g = SVector{3, T}(gx, gy, gs)
    H = SMatrix{3, 3, T}(Hxx, Hyx, Hsx, Hxy, Hyy, Hsy, Hxs, Hys, Hss)     # column-major: H[i, j] = ∂gᵢ/∂vⱼ
    return D, g, H
end

# The box of v where a _DogSplineWindow is defined (its spans and its slices).
function _model_box(m::_DogSplineWindow)
    nx, ny, N = size(m.coef, 1), size(m.coef, 2), size(m.coef, 3)
    lo = SVector(m.X0 + 2.0, m.Y0 + 2.0, Float64(m.s0))
    hi = SVector(m.X0 + nx - 1.0 - 1.0e-9, m.Y0 + ny - 1.0 - 1.0e-9, Float64(m.s0 + N - 1))
    return lo, hi
end

# ----- the coefficient kernel ------------------------------------------------

# The B-spline coefficients of every detection's window and field, one work item per
# (column i, slice·field, detection): coef[i, j, t, f, k] = Σ_{a,b} fx[a, f]·fy[b, f]·dog[X + a, Y + b, s, w] (a
# correlation; a, b centred), X = Xc − RC − 1 + i, Y likewise, s = s0 + t − 1, the borders mirrored as the pyramid's
# blur does (`_mirror_idx`). `fx`/`fy` columns are each field's composite taps: the cardinal prefilter, convolved with
# the field's own filter along that axis (none for the spline response; a Farid–Simoncelli prefilter or derivative
# for stage B). Each item sweeps its column's sample rows once, the x-filtered row added to every coefficient of the
# column it reaches. `centres` columns: (Xc, Yc, s0, w), 1-based.
@kernel function _dog_spline_coefficients!(coef, @Const(dog), @Const(centres), @Const(fx), @Const(fy), ::Val{RC}) where {RC}
    i, tf, k = @index(Global, NTuple)
    NSW = size(coef, 3)
    t = (tf - 1) % NSW + 1; f = (tf - 1) ÷ NSW + 1
    L = size(fx, 1); R = (L - 1) ÷ 2
    W = size(dog, 1); H = size(dog, 2)
    @inbounds begin
        Xc = Int(centres[1, k]); Yc = Int(centres[2, k])
        X = Xc - RC - 1 + i
        s = Int(centres[3, k]) + t - 1
        w = Int(centres[4, k])
        acc = ntuple(_ -> 0.0f0, Val(2RC + 1))
        for yy in -(RC + R):(RC + R)
            yb = _mirror_idx(Yc + yy, H)
            row = 0.0f0
            for a in -R:R
                row += fx[a + R + 1, f] * dog[_mirror_idx(X + a, W), yb, s, w]
            end
            # a `let` gives the closure its own bindings: capturing `acc`, reassigned here, would box it
            acc = let a = acc, row = row, yy = yy
                ntuple(Val(2RC + 1)) do j
                    d = yy - (j - RC - 1)
                    abs(d) <= R ? a[j] + fy[d + R + 1, f] * row : a[j]
                end
            end
        end
        for j in 1:(2RC + 1)
            coef[i, j, t, f, k] = acc[j]
        end
    end
end

# The fields' composite correlation taps along x and y, (L, F) each, centred: the cardinal prefilter `h` (one-sided,
# K + 1 taps) convolved with each field's filter along that axis (a delta for the spline response).
const _FS = _CR_DEFAULTS["farid_simoncelli_5tap"]
const _FS_P = Float64.(_FS["prefilter"]); const _FS_D1 = Float64.(_FS["first_derivative"])
function _conv_taps(a::Vector{Float64}, b::Vector{Float64})
    c = zeros(length(a) + length(b) - 1)
    for i in eachindex(a), j in eachindex(b)
        c[i + j - 1] += a[i] * b[j]
    end
    return c
end
function _field_taps(derivatives::Symbol, K::Integer)
    h1 = cubic_bspline_prefilter_taps(K)
    h = [reverse(h1[2:end]); h1]                        # full, 2K + 1
    pad(c) = (L = 2K + 5; z = zeros(L); o = (L - length(c)) ÷ 2; z[(o + 1):(o + length(c))] .= c; z)
    if derivatives === :spline
        return reshape(pad(h), :, 1), reshape(pad(h), :, 1)
    elseif derivatives === :farid_simoncelli_spatial
        hp = _conv_taps(h, _FS_P); h1d = _conv_taps(h, _FS_D1)
        return hcat(pad(h), h1d, hp), hcat(pad(h), hp, h1d)
    end
    throw(ArgumentError("the continuous refinement's response is :farid_simoncelli_spatial or :spline, got :$derivatives"))
end

# ----- the refinement ----------------------------------------------------------

"""
    _DogExtremum

A refined DoG extremum: `v` = (x, y, s), 1-based octave-level coordinates; the
response and Hessian there; the §14 record (`retcode`, `stop_rule`). A failure
carries the START as `v`, the last valid estimate.
"""
struct _DogExtremum
    v::SVector{3, Float64}
    response::Float64
    hessian::SMatrix{3, 3, Float64, 9}
    retcode::ReturnCode.T
    stop_rule::StopRule
end

"""
    _refine_dog_extremum(model, v0, σ, S, cfg, extent) -> (v, D, H, code::ReturnCode.T, rule::StopRule)

Refine the DoG extremum near `v0` = (x, y, s) (1-based level coordinates) on the
continuous response `model` (`_dog_response`) with Core's device LM
(`static_lm`) on the rows r = ∇_θ D/|D(v₀)|, θ = t ⊙ (v − v₀) with the scale
t = (1/σ, 1/σ, ln2/S) (σ the start's scale in level pixels), solved for the
displacement v − v₀ with `t` as `static_lm`'s per-coordinate scale; its box the displacement
bound (|Δx|, |Δy| ≤ `spatial_bound`, |Δs| ≤ `scale_bound`) within the model's
domain and the level's extent `(W, H)` (1 ≤ x ≤ W, 1 ≤ y ≤ H: the detector's keep
test, so a refined point never leaves the level; the window's mirrored border
samples are no image). The §14 record: the LM's (`static_lm_termination`); a solve that did not
converge with the point on the box is `LeftBasin`; a converged point that is not
an extremum of the start's type, or whose curvature −sign(D)·H_θ/|D| has an
eigenvalue below `min_curvature`, is `ShapeGate`. Every failure returns the start
(the last valid estimate). Kernel-lawful: `_dog_extrema_kernel!` calls it per
detection on the CPU and any GPU.
"""
@inline function _refine_dog_extremum(model, v0::SVector{3, Float64}, σ::Float64, S::Int, cfg::ContinuousRefinement, extent::NTuple{2, Int})
    D0, _, H0 = _dog_response(model, v0)
    H0d = SMatrix{3, 3, Float64}(H0)
    (isfinite(D0) && D0 != 0) || return v0, Float64(D0), H0d, ReturnCode.InitialFailure, NonFinite
    t = SVector(1 / σ, 1 / σ, 0.6931471805599453 / S)
    mlo, mhi = _model_box(model)
    b = SVector(cfg.spatial_bound, cfg.spatial_bound, cfg.scale_bound)
    lo = max.(mlo, v0 - b, SVector(1.0, 1.0, -Inf)); hi = min.(mhi, v0 + b, SVector(Float64(extent[1]), Float64(extent[2]), Inf))
    all(lo .<= v0 .<= hi) || return v0, Float64(D0), H0d, ReturnCode.InitialFailure, LeftBasin
    ia = 1 / abs(Float64(D0))
    # The LM runs on the displacement u = v − v₀ (level px, level px, slices), its scale `t` handed to static_lm as
    # its per-coordinate scale `s` (its step and damping floors read there): the rows are r = ∇_θ D/|D(v₀)|, the
    # gradient in the scaled coordinates θ = t ⊙ u, so ∂r/∂u = H ./ t (row i by tᵢ) /|D(v₀)|. The closure's locals
    # are its own (Du, gu, Hu): reusing D or H, assigned below, would box them (inference: Any).
    fj = u -> begin
        Du, gu, Hu = _dog_response(model, v0 + u)
        (SVector{3, Float64}(gu) ./ t .* ia, SMatrix{3, 3, Float64}(Hu) ./ t .* ia, Float64(Du), SMatrix{3, 3, Float64}(Hu))
    end
    lb = lo - v0; ub = hi - v0
    u, ok, _, e, rule = static_lm(fj, zero(SVector{3, Float64}), lb, ub, StaticLmStops(t))   # Core's stop settings, `t` the scale
    code, rule = static_lm_termination(ok, rule)
    if !successful_retcode(code)
        onbox = any(u .<= lb) || any(u .>= ub)
        onbox && return v0, Float64(D0), H0d, ReturnCode.Failure, LeftBasin
        return v0, Float64(D0), H0d, code, rule
    end
    v = v0 + u
    D = e[3]; H = e[4]
    sgn = D0 > 0 ? 1.0 : -1.0
    (D > 0) == (D0 > 0) || return v0, Float64(D0), H0d, ReturnCode.Failure, ShapeGate
    Hn = (-sgn / abs(D)) .* (H + H') ./ (2 .* (t * t'))      # its symmetric part (the recipe's Jacobian is not symmetric)
    is_positive_definite(Hn - cfg.min_curvature * one(Hn)) || return v0, Float64(D0), H0d, ReturnCode.Failure, ShapeGate
    return v, D, H, code, rule
end

# One detection per work item: its window (coefficients `coef[:, :, :, k]`, origin
# `org[:, k]` = (X0, Y0, s0)), its start `starts[:, k]` = (x, y, s, σ), refined;
# `out[:, k]` = (x, y, s, D, H[:] (column-major, 9), code, rule): H in full — the recipe's Jacobian is not
# symmetric.
@kernel function _dog_extrema_kernel!(out, @Const(coef), @Const(org), @Const(starts), S::Int, cfg::ContinuousRefinement, extent::NTuple{2, Int}, ::Val{N}, ::Val{F}) where {N, F}
    k = @index(Global)
    @inbounds if k <= size(out, 2)
        c = view(coef, :, :, :, :, k)
        m = _DogSplineWindow{N, F, typeof(c)}(c, Int(org[1, k]), Int(org[2, k]), Int(org[3, k]))
        v0 = SVector{3, Float64}(starts[1, k], starts[2, k], starts[3, k])
        v, D, H, code, rule = _refine_dog_extremum(m, v0, Float64(starts[4, k]), S, cfg, extent)
        out[1, k] = v[1]; out[2, k] = v[2]; out[3, k] = v[3]; out[4, k] = D
        for q in 1:9
            out[4 + q, k] = H[q]
        end
        out[14, k] = Float64(Integer(code)); out[15, k] = Float64(Integer(rule))
    end
end

"""
    _refine_dog_extrema(dev, dog, starts; seed_sigma, S, cfg = ContinuousRefinement())
        -> (extrema::Vector{_DogExtremum}, windows::Vector{_DogSplineWindow}, timing)

Stage C for every start `(x, y, s, w)` (1-based level coordinates of the octave cube
`dog`, warp `w`): the windows' coefficients (`_dog_spline_coefficients!`) and the
refinement (`_dog_extrema_kernel!`), two launches on `dev`, the same kernels on the
CPU and any GPU. `timing` = (coefficients, refinement) in seconds, each launch
synchronized. The response is `cfg`'s ([`ContinuousRefinement`](@ref)). `coefficients` (host, the
`coef` of an earlier call with the same starts) skips the coefficient kernel.
"""
function _refine_dog_extrema(
        dev, dog, starts::AbstractVector; seed_sigma::Real, S::Integer, cfg::ContinuousRefinement = ContinuousRefinement(),
        coefficients::Union{Nothing, AbstractArray{Float32, 5}} = nothing
    )
    fx, fy = _field_taps(_response(cfg), cfg.prefilter_halfwidth)
    F = size(fx, 2)
    n = length(starts)
    RC = _coef_halfwidth(cfg); K = cfg.prefilter_halfwidth
    NS = size(dog, 3); NSW = _slice_window(NS)
    cen = Matrix{Int32}(undef, 4, n)
    st = Matrix{Float64}(undef, 4, n)
    for (k, (x, y, s, w)) in enumerate(starts)
        cen[1, k] = round(Int32, x); cen[2, k] = round(Int32, y)
        cen[3, k] = Int32(clamp(round(Int, s) - 2, 1, NS - NSW + 1)); cen[4, k] = Int32(w)
        st[1, k] = x; st[2, k] = y; st[3, k] = s
        st[4, k] = _slice_sigma(s - 1, Float64(seed_sigma), S, 1.0)   # the start's σ in level px (slice s, 1-based)
    end
    org = similar(cen, 3, n)
    org[1, :] .= cen[1, :] .- Int32(RC + 1); org[2, :] .= cen[2, :] .- Int32(RC + 1); org[3, :] .= cen[3, :]
    coef_d = KA.allocate(dev, Float32, 2RC + 1, 2RC + 1, NSW, F, max(n, 1))
    out_d = KA.allocate(dev, Float64, 15, n)
    t1 = t2 = 0.0
    if n > 0
        cen_d = ka_upload(Int32, dev, cen)
        fx_d = ka_upload(Float32, dev, Float32.(fx)); fy_d = ka_upload(Float32, dev, Float32.(fy))
        org_d = ka_upload(Int32, dev, org); st_d = ka_upload(Float64, dev, st)
        if coefficients === nothing
            t1 = @elapsed _launch!(_dog_spline_coefficients!, dev, coef_d, dog, cen_d, fx_d, fy_d, Val(RC); ndrange = (2RC + 1, NSW * F, n))
        else
            copyto!(coef_d, coefficients)       # another run's windows (the CPU = GPU attribution: one set of coefficients, two LMs)
        end
        t2 = @elapsed _launch!(_dog_extrema_kernel!, dev, out_d, coef_d, org_d, st_d, Int(S), cfg, (size(dog, 1), size(dog, 2)), Val(NSW), Val(F); ndrange = n)
    end
    coef = Array(coef_d); out = Array(out_d)
    windows = [_DogSplineWindow(view(coef, :, :, :, :, k), org[1, k], org[2, k], org[3, k]) for k in 1:n]
    extrema = [
        _DogExtremum(
                SVector(out[1, k], out[2, k], out[3, k]), out[4, k],
                SMatrix{3, 3, Float64}(ntuple(q -> out[4 + q, k], 9)),
                ReturnCode.T(Int32(out[14, k])), StopRule(Int32(out[15, k]))
            ) for k in 1:n
    ]
    return extrema, windows, (; coefficients = t1, refinement = t2)
end

# ----- the covariance at C's point ---------------------------------------------

# The DoG-sample weights of one axis' spline functional at window coordinate ξ, order
# d (0: value, 1: derivative): w[p] over sample positions p (1-based level pixels),
# w[p] = Σ_i B⁽ᵈ⁾(ξ − i)·h[p − (X0 + i)], as a `_Line` starting at its first pixel.
function _spline_axis_weights(X0::Int, ξ::Float64, d::Int, c::AbstractVector{Float64})
    R = (length(c) - 1) ÷ 2
    i0 = floor(Int, ξ); u = ξ - i0
    wts = d == 0 ? cubic_bspline_weights(u) : cubic_bspline_weight_derivatives(u)[1]
    first = X0 + i0 - 1 - R
    v = zeros(4 + 2R)
    for a in 1:4, k in -R:R
        p = X0 + i0 + a - 2 + k
        v[p - first + 1] += wts[a] * c[k + R + 1]
    end
    return _Line(first, v)
end

"""
    _continuous_covariance(plan, oi, w, m::_DogSplineWindow, e::_DogExtremum, cfg) -> SMatrix{3, 3} over (x, y, log σ)

The implicit-function-theorem covariance of the refined point `e` of the
`_DogSplineWindow` `m` (octave `oi` of `plan`, 1 = the finest; warp `w`), for
unit iid pixel noise, in the warp canvas' reported coordinates (x, y, log σ):
J·H⁻¹·Cov(∇D)·H⁻¹·Jᵀ, ∇D the model's own gradient functional at `e.v` and Cov(∇D)
through the pyramid's adjoint — the same estimator and map as the refinement.
"""
function _continuous_covariance(
        plan, oi::Integer, w::Integer, m::_DogSplineWindow, e::_DogExtremum, cfg::ContinuousRefinement = ContinuousRefinement();
        taps = [Array(t) for t in plan.pyr_taps],
        autocov = _canvas_autocov(plan.geo[w], plan.W, plan.H),
        fields = _field_taps(_response(cfg), cfg.prefilter_halfwidth)
    )
    S = plan.nb_scales
    upsample = plan.o_first < 0
    N = size(m.coef, 3)
    v = e.v
    ξ = v[1] - m.X0; η = v[2] - m.Y0; τ = v[3] - m.s0 + 1
    L, dL, _ = _lagrange_basis(τ, Val(N))
    # Gaussian-level weights of each functional: DoG slice s = G[s+1] − G[s]
    lev(c) = (λ = zeros(N + 1); for q in 1:N; λ[q + 1] += c[q]; λ[q] -= c[q]; end; λ)
    λL = lev(L); λdL = lev(dL)
    # the gradient functionals (gx, gy, gs) of the window's response: per axis (spline order, composite taps)
    fx, fy = fields
    size(fx, 2) == size(m.coef, 4) || throw(ArgumentError("_continuous_covariance: the window has $(size(m.coef, 4)) fields, the refinement's response $(size(fx, 2))"))
    spec = size(m.coef, 4) == 1 ?
        (((1, fx[:, 1]), (0, fy[:, 1]), λL), ((0, fx[:, 1]), (1, fy[:, 1]), λL), ((0, fx[:, 1]), (0, fy[:, 1]), λdL)) :
        (((0, fx[:, 2]), (0, fy[:, 2]), λL), ((0, fx[:, 3]), (0, fy[:, 3]), λL), ((0, fx[:, 1]), (0, fy[:, 1]), λdL))
    # canvas responses per functional and window level (level l is Gaussian level m.s0 + l − 1)
    Kx = [_canvas_response(taps, S, upsample, oi, m.s0 + l - 1, _spline_axis_weights(m.X0, ξ, spec[a][1][1], spec[a][1][2])) for a in 1:3, l in 1:(N + 1)]
    Ky = [_canvas_response(taps, S, upsample, oi, m.s0 + l - 1, _spline_axis_weights(m.Y0, η, spec[a][2][1], spec[a][2][2])) for a in 1:3, l in 1:(N + 1)]
    c, Rx, Ry = autocov
    C = zeros(3, 3)
    for a in 1:3, bb in a:3
        λa = spec[a][3]; λb = spec[bb][3]
        s = 0.0
        for la in 1:(N + 1), lb in 1:(N + 1)
            wab = λa[la] * λb[lb]
            wab == 0 && continue
            for δy in -Ry:Ry, δx in -Rx:Rx
                cv = c[δx + Rx + 1, δy + Ry + 1]
                cv == 0 && continue
                s += wab * cv * _xcorr(Kx[a, la], Kx[bb, lb], δx) * _xcorr(Ky[a, la], Ky[bb, lb], δy)
            end
        end
        C[a, bb] = C[bb, a] = s
    end
    Hi = inv(Matrix(e.hessian))
    Σv = Hi * C * Hi'                # (x, y, s) level coordinates
    o = oi - 1 + plan.o_first
    sf = 2.0^o
    J = [sf 0 0; 0 sf 0; 0 0 log(2.0) / S]
    return SMatrix{3, 3, Float64}(J * Σv * J')
end

# ----- the detector's opt-in refinement (asift_detect!(…; refinement)) ------------------------------------------------

# Stage C/B over one tilt chunk's new pool entries `nbefore+1 … nbefore+nnew`: their integer extrema and offsets (the
# refinement state `_bextract!` stored in `rawcov`) are the starts, refined per octave on the chunk's resident DoG, and
# each converged detection's (x, y, σ, response) overwrites its `raw` column in the warp canvas's reported coordinates (a
# failure keeps today's, its start). Returns the refined entries for `_refined_covs!`.
function _refine_chunk!(plan, cfg::ContinuousRefinement, m::Int, nbefore::Int, nnew::Int)
    sl = (nbefore + 1):(nbefore + nnew)
    rawc = Array(view(plan.rawcov, :, sl)); rawh = Array(view(plan.raw, :, sl))
    S = plan.nb_scales
    byoct = Dict{Int, Vector{Int}}()
    for i in 1:nnew
        code = unsafe_trunc(Int32, rawc[3, i])
        oi = Int((code >> 3) - Int32(2)) - plan.o_first + 1
        push!(get!(byoct, oi, Int[]), i)
    end
    refined = Tuple{Int, Int, Int, _DogSplineWindow, _DogExtremum}[]   # (pool column, octave, chunk warp, window, extremum)
    for (oi, is) in byoct
        starts = map(is) do i
            code = unsafe_trunc(Int32, rawc[3, i])
            (Float64(rawc[1, i]) + rawc[5, i], Float64(rawc[2, i]) + rawc[6, i], Float64(code & Int32(7)) + rawc[4, i], Int(rawh[5, i]))
        end
        dog = view(plan.dogs[oi], :, :, :, 1:m)
        ext, wins, _ = _refine_dog_extrema(plan.dev, dog, starts; seed_sigma = plan.seed_sigma, S, cfg)
        sf = exp2(Float64(oi - 1 + plan.o_first))
        for (j, i) in enumerate(is)
            e = ext[j]
            successful_retcode(e.retcode) || continue
            rawh[1, i] = Float32(_level_to_canvas(e.v[1] - 1, sf)); rawh[2, i] = Float32(_level_to_canvas(e.v[2] - 1, sf))
            rawh[3, i] = Float32(_slice_sigma(e.v[3] - 1, plan.seed_sigma, S, sf)); rawh[4, i] = Float32(e.response)
            push!(refined, (nbefore + i, oi, Int(rawh[5, i]), wins[j], e))
        end
    end
    copyto!(view(plan.raw, :, sl), rawh)
    return refined
end

# The refined detections' covariances, over the ones `_bcov!` left in `featcov` (the canvas frame, packed
# (xx, xy, xℓ, yy, yℓ, ℓℓ)), before the back-projection maps them: `_continuous_covariance` at each refined point — the
# refinement's own derivatives. Threaded over the detections.
function _refined_covs!(plan, cfg::ContinuousRefinement, refined, r, nbefore::Int, nnew::Int)
    isempty(refined) && return nothing
    sl = (nbefore + 1):(nbefore + nnew)
    fc = Array(view(plan.featcov, :, sl))
    taps = [Array(t) for t in plan.pyr_taps]
    ac = Dict(w => _canvas_autocov(plan.geo[r[w]], plan.W, plan.H) for w in unique(x[3] for x in refined))
    fields = _field_taps(_response(cfg), cfg.prefilter_halfwidth)   # the configuration's, once for every detection
    Threads.@threads :dynamic for (col, oi, w, win, e) in refined
        Σ = _continuous_covariance(plan, oi, r[w], win, e, cfg; taps, autocov = ac[w], fields)
        i = col - nbefore
        fc[1, i] = Σ[1, 1]; fc[2, i] = Σ[1, 2]; fc[3, i] = Σ[1, 3]
        fc[4, i] = Σ[2, 2]; fc[5, i] = Σ[2, 3]; fc[6, i] = Σ[3, 3]
    end
    copyto!(view(plan.featcov, :, sl), fc)
    return nothing
end

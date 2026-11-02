# =============================================================================
# The Sampson residual and its whitening — two layers over one constraint
# =============================================================================
#
# GEOMETRY (a model and its measurements). An implicit constraint
# g(θ; x) = 0 (`constraint_with_jacobians`, G_x = ∂g/∂x) linearized at the
# measurement: the Sampson residual is the first-order displacement onto the
# model's variety, in coordinates of its NORMAL space,
#
#     s_N = L⁻¹g,   N = G_xᵀL⁻ᵀ,   LLᵀ = G_x·G_xᵀ,
#
# N an orthonormal basis of the row space of G_x (G_xᵀ = N·Lᵀ is its QR, R = Lᵀ).
# Its norm is the Sampson distance: geometric, in the measurement's own units,
# whatever covariance the measurement carries. The one degeneracy rule lives
# here, at the model's `constraint_rounding_scale`.
#
# NOISE (the Sampson residual and the measurements' covariance). Since
# G_x = L·Nᵀ, the propagated covariance factors as G_x·Σ·G_xᵀ = L·(NᵀΣN)·Lᵀ, so
#
#     whiten(s_N, N, Σ) = L_N⁻¹s_N,   L_N L_Nᵀ = NᵀΣN,
#
# is exactly the whitened constraint L_Σ⁻¹g (L_Σ = L·L_N): the Sampson residual
# divided by the measurement noise along the normal. Σ is the measurements'
# own (conventions/measurement-noise.md): a plain point I, an `Uncertain` its
# covariance, an `exact` one 0. At Σ = I it is the Sampson residual unchanged.
# =============================================================================

"""
    constraint(M, x...; kw...) -> g

The constraint value `g(M; x...)` alone — the `<f>` of the
[`constraint_with_jacobians`](@ref) pair (conventions/jacobians.md): its first
return, so a caller that keeps only `g` names the verb rather than discarding
two Jacobians. A model may add a method that skips work the Jacobians need (the
bearing [`TransferForm`](@ref)'s `g` needs no backprojection Jacobians).

`g` is ALGEBRAIC: it scales with the model's homogeneous scale and with the
measurements' units (a fundamental matrix scaled by 10⁶ scales `g` by 10⁶), so
it is not a distance and a threshold on it means nothing in pixels. The
geometric distance is [`sampson_distance`](@ref); the noise-whitened one
[`mahalanobis_distance`](@ref).
"""
@inline constraint(M, x...; kw...) = first(constraint_with_jacobians(M, x...; kw...))

"""
    constraint_rounding_scale(M, x...; kw...) -> NamedTuple{(:gradient, :value)} | nothing

The rounding scales of the constraint `g(M; x...)` at a measurement — the one
input the Sampson degeneracy rule needs beyond `(g, G_x)`: `gradient`, below
which the measurement gradient `‖G_x‖` is rounding (DEGENERATE), and `value`,
below which `|gᵢ|` is rounding (the constraint VANISHES; a scalar, or one per
component). Relative to the model's and the points' own magnitudes, so a
model's homogeneous scale never decides. [`sampson_residual_with_normal`](@ref)
consumes it; the default, `nothing`, makes only an exactly zero gradient
degenerate and an exactly zero `g` vanishing. The epipolar models return
[`epipolar_rounding`](@ref)'s scales, a homography its rows' term-by-term
bound, a conic and an ellipse theirs.
"""
constraint_rounding_scale(M, x...; kw...) = nothing

# A model's measurements as its constraint takes them: the model and the points
# in one promoted element type where the constraint's methods require it
# (`FundMat`/`EssentialMat`/`HomographyMat` on 2-vectors, fundmat.jl). The
# default is the arguments as they are.
@inline _sampson_arguments(M, x...) = (M, x...)

# A check of the measurements against the model before their covariances are
# stripped (an `EpipolarForm` checks the frames its slots meet). Default: none.
@inline _check_measurements(M, x...) = nothing

# The geometry layer's one body, from a constraint value and its measurement
# gradient at the rounding scale ρ: the Sampson residual `s` (an SVector over
# the constraint's components), its normal basis `N`, the Cholesky factor `L` of
# G_x·G_xᵀ (a number for a scalar constraint) the whitened Jacobian rows read,
# and `ok`. Not ok is the degeneracy rule: `s` is 0 or ±Inf per component, `N`
# is zero. NaN propagates (with `ok`, so the rows carry it too).
@inline function _sampson_factor(g::Real, G_x::StaticVector{nx}, ρ) where {nx}
    J² = sum(abs2, G_x)                        # no muladd: the same bits inlined or not
    T = typeof(J²)
    ok = (J² > zero(J²) || isnan(J²)) && !_gradient_degenerate(ρ, G_x)
    ok || return SVector(_degenerate_residual(g, _vanishes(ρ, g))), zero(SMatrix{nx, 1, T}), zero(T), false
    n = sqrt(J²)
    return SVector(g / n), SMatrix{nx, 1}(G_x / n), n, true
end
@inline function _sampson_factor(gv::StaticVector{dg, T}, G_x::StaticMatrix{dg, nx, T}, ρ) where {dg, nx, T}
    g = SVector{dg, T}(gv)                     # a Vec or MVector constraint: the residual is an SVector
    C, d, ok = _vector_whitening(G_x, I, ρ)
    L = C.L
    isnan(d) && return g .* T(NaN), zero(SMatrix{nx, dg, T}) .* T(NaN), L, true   # NaN in, NaN out
    ok || return _degenerate_residual(g, _vanishes(ρ, g)), zero(SMatrix{nx, dg, T}), L, false
    return L \ g, SMatrix{nx, dg, T}((L \ G_x)'), L, true
end

# A residual in its constraint's shape: a number for a scalar constraint, the
# vector for a vector one (the whitened-rows protocol's currency).
@inline _constraint_shape(::Real, s::StaticVector{1}) = s[1]
@inline _constraint_shape(::StaticVector, s::StaticVector) = s

# A residual's norm: `abs` for one component (exact, as a distance was), the
# 2-norm otherwise.
@inline _residual_norm(r::StaticVector{1}) = abs(r[1])
@inline _residual_norm(r::StaticVector) = norm(r)

"""
    sampson_residual_with_normal(M, x...; kw...) -> (s_N::SVector{dg}, N::SMatrix{nx, dg})

The Sampson residual of the measurements `x` under the model `M` and the
orthonormal basis of the constraint's normal space it is expressed in: with
`g`, `G_x` from [`constraint_with_jacobians`](@ref)`(M, x...)`,

    s_N = L⁻¹g,   N = G_xᵀL⁻ᵀ,   LLᵀ = G_x·G_xᵀ,

the first-order displacement onto the model's variety (`‖s_N‖` is the
[`sampson_distance`](@ref)), and `N` the directions it is measured along
(`NᵀN = I`; one column for a scalar constraint, `dg` for a vector one). Always
an `SVector{dg}`, `dg` the constraint's component count (1 for F, E, a conic, an
ellipse, a circle or a line; 2 for a homography). The noise layer,
[`whiten`](@ref)`(s_N, N, Σ)`, takes both.

GEOMETRIC: a covariance attached to a measurement ([`Uncertain`](@ref),
[`exact`](@ref)) is ignored here; the measurement's point is used. A
correspondence may be given as a `Pair`. Keywords are forwarded to the
constraint (an `EssentialMat` on bearings takes its backprojection Jacobians
`M₁`, `M₂`).

DEGENERATE (the measurement gradient vanishes to rounding, at the model's
[`constraint_rounding_scale`](@ref) — both points at the epipoles, a point a
homography sends to the line at infinity, an ellipse's centre): each component
is `0` when the constraint vanishes with it and `±Inf` (the sign of `g`) when
it does not, and `N` is zero. NaN propagates.
"""
@inline function sampson_residual_with_normal(M, x...; kw...)
    _check_measurements(M, x...)
    args = _sampson_arguments(M, map(geom, x)...)
    g, G_x, _ = constraint_with_jacobians(args...; kw...)
    s, N, _, _ = _sampson_factor(g, G_x, constraint_rounding_scale(args...; kw...))
    return s, N
end
@inline sampson_residual_with_normal(M, c::Pair; kw...) = sampson_residual_with_normal(M, c.first, c.second; kw...)

"""
    sampson_residual(M, x...; kw...) -> SVector{dg}

The Sampson residual of [`sampson_residual_with_normal`](@ref) alone: signed,
smooth, one entry per constraint component — the residual an LM descends.
`only(sampson_residual(F, u₁, u₂))` is the signed epipolar Sampson distance.
"""
@inline sampson_residual(M, x...; kw...) = first(sampson_residual_with_normal(M, x...; kw...))

"""
    sampson_distance(M, x...; kw...) -> T

The Sampson distance of the measurements `x` under the model `M`: the norm of
its [`sampson_residual`](@ref), the first-order geometric distance to the
model's variety in the measurements' own units — for the epipolar constraint
`u₂ᵀ F u₁ = 0` the familiar
`|u₂ᵀ F u₁| / sqrt((Fu₁)₁² + (Fu₁)₂² + (Fᵀu₂)₁² + (Fᵀu₂)₂²)`, for an ellipse the
distance to its boundary to first order. Any model with a
[`constraint_with_jacobians`](@ref) has one: `FundMat`, `EssentialMat` (pixel
plane, or bearings with `M₁`, `M₂`), `HomographyMat`, an `EpipolarForm`, a
`TransferForm`, a `Conic`, an `Ellipse`, a `Circle`, a `Line2D`. A
correspondence may be given as a `Pair`.

Geometric only: a measurement's covariance does not enter. The noise-whitened
(Mahalanobis) distance is [`mahalanobis_distance`](@ref). Degenerate
measurements follow [`sampson_residual_with_normal`](@ref)'s rule (`0` or `Inf`).
"""
@inline sampson_distance(M, x...; kw...) = _residual_norm(sampson_residual(M, x...; kw...))

"""
    whiten(s_N, N, Σ) -> SVector{dg}

The noise layer: the Sampson residual `s_N` in the normal basis `N`
([`sampson_residual_with_normal`](@ref)) whitened by the measurement covariance
`Σ` (over the constraint's measurements, `nx × nx`),

    r = L_N⁻¹s_N,   L_N L_Nᵀ = NᵀΣN,

the first-order Mahalanobis residual — equal to the constraint whitened by its
propagated covariance, `L⁻¹g` with `LLᵀ = G_x·Σ·G_xᵀ`. Σ = I returns `s_N`
unchanged; `r²` is χ²(dg) under Gaussian noise of covariance Σ.

DEGENERATE: `NᵀΣN` losing rank to rounding relative to its diagonal (a
measurement exact along the normal) leaves the residual's zero components 0
and its others ±Inf; a degenerate Sampson residual (zero `N`) passes through.
NaN propagates.
"""
@inline whiten(s::StaticVector, N::StaticMatrix, Σ) = first(_whiten_rows(s, N, nothing, Σ))

# The noise layer on a Sampson residual `s` and, when given, its model-Jacobian
# rows `J` (L⁻¹G_θ, the whitened-rows protocol's): both whitened by the same
# L_N, `NᵀΣN = L_N L_Nᵀ` — so the rows of a constraint whitened by `G_x·Σ·G_xᵀ`
# are `L_N⁻¹L⁻¹g` and `L_N⁻¹L⁻¹G_θ`. Degenerate (NᵀΣN losing rank): the rule's
# residual and zero rows. `J = nothing` whitens the residual alone.
@inline function _whiten_rows(s::StaticVector{dg, T}, N::StaticMatrix{nx, dg}, J, Σ) where {dg, nx, T}
    Σ isa UniformScaling{Bool} && return s, J
    return _whiten_normal(s, N, J, Σ)
end
@inline function _whiten_normal(s::StaticVector{1, T}, N::StaticMatrix{nx, 1}, J, Σ) where {nx, T}
    σ² = _propagate_scalar(SVector{nx}(N), Σ)
    (σ² > zero(σ²) || isnan(σ²)) || return _degenerate_whitened(s), _zero_rows(J)
    σ = sqrt(σ²)
    return s / σ, _scaled_rows(J, σ)
end
@inline function _whiten_normal(s::StaticVector{dg, T}, N::StaticMatrix{nx, dg}, J, Σ) where {dg, nx, T}
    C, d, ok = _whitening_cholesky(propagate(N', Σ))
    isnan(d) && return s .* T(NaN), _scaled_rows(J, T(NaN))
    ok || return _degenerate_whitened(s), _zero_rows(J)
    return C.L \ s, _solved_rows(C.L, J)
end
@inline _zero_rows(::Nothing) = nothing
@inline _zero_rows(J) = zero(J)
@inline _scaled_rows(::Nothing, σ) = nothing
@inline _scaled_rows(J, σ) = J / σ
@inline _solved_rows(L, ::Nothing) = nothing
@inline _solved_rows(L, J) = L \ J
# A residual whitened by a covariance singular along the normal: zero stays
# zero (the constraint holds), anything else is unreachable (±Inf), NaN stays.
@inline _degenerate_whitened(s) = map(x -> (iszero(x) || isnan(x)) ? x : copysign(oftype(x, Inf), x), s)

# The stacked covariance of a constraint's measurements, read from the
# measurements themselves (conventions/measurement-noise.md): one point's own,
# or a correspondence's block diagonal (`_pair_covariance`; both exact refused).
@inline function _measurement_covariance(x)
    Σ = _point_covariance(x)
    _all_exact(Σ) && throw(ArgumentError("an exact measurement has no noise to whiten: give it a covariance (Uncertain) or a plain point"))
    return Σ
end
@inline _measurement_covariance(x₁, x₂) = _pair_covariance(x₁, x₂)
@inline _all_exact(_) = false
@inline _all_exact(Σ::AbstractMatrix) = iszero(Σ)

"""
    mahalanobis_distance(M, x...; kw...) -> T

The first-order Mahalanobis distance of the measurements `x` under the model
`M`: the norm of [`whiten`](@ref)`(`[`sampson_residual_with_normal`](@ref)`(M,
x...)..., Σ)`, `Σ` the measurements' own covariance per the measurement-noise
rule — a plain point unit isotropic (so a plain correspondence's Mahalanobis
distance IS its [`sampson_distance`](@ref)), an [`Uncertain`](@ref) one its
covariance or precision, an [`exact`](@ref) one zero (not every measurement of
a constraint: `ArgumentError`). In noise units: `d² ~ χ²(dg)`. A correspondence
may be given as a `Pair`; keywords are forwarded to the constraint.
"""
@inline function mahalanobis_distance(M, x...; kw...)
    s, N = sampson_residual_with_normal(M, x...; kw...)
    return _residual_norm(whiten(s, N, _measurement_covariance(x...)))
end
@inline mahalanobis_distance(M, c::Pair; kw...) = mahalanobis_distance(M, c.first, c.second; kw...)

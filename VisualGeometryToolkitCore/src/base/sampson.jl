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
their entries' rounding bounds (`|F|·|s|`, `|d|ᵀ·|F|·|s|`), a homography its
rows' term-by-term bound, a conic and an ellipse theirs.
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
# G_x·G_xᵀ (a number for a scalar constraint: ‖G_x‖, the Jacobian's normalizer),
# and `ok`. Not ok is the degeneracy rule: `s` is 0 or ±Inf per component, `N`
# is zero. NaN propagates (with `ok`, so the rows carry it too).
# Branch-free (`ifelse`, both sides computed): a branch here stops a row loop
# over correspondences from vectorizing (measured 2× on the epipolar rows).
@inline function _sampson_factor(g::Real, G_x::StaticVector{nx}, ρ) where {nx}
    J² = sum(abs2, G_x)                        # no muladd: the same bits inlined or not
    T = typeof(J²)
    ok = !(J² <= _gradient_bound²(ρ, T))       # one comparison: a NaN gradient is not degenerate, NaN propagates
    n = sqrt(J²)
    s = ifelse(ok, g / n, _degenerate_residual(g, _vanishes(ρ, g)))
    N = ifelse(ok, SMatrix{nx, 1}(G_x / n), zero(SMatrix{nx, 1, T}))
    return SVector(s), N, ifelse(ok, n, zero(T)), ok
end
@inline function _sampson_factor(gv::StaticVector{dg, T}, G_x::StaticMatrix{dg, nx, T}, ρ) where {dg, nx, T}
    g = SVector{dg, T}(gv)                     # a Vec or MVector constraint: the residual is an SVector
    C, d, ok = _vector_whitening(G_x, I, ρ)
    L = C.L
    isnan(d) && return g .* T(NaN), zero(SMatrix{nx, dg, T}) .* T(NaN), L, true   # NaN in, NaN out
    ok || return _degenerate_residual(g, _vanishes(ρ, g)), zero(SMatrix{nx, dg, T}), L, false
    return L \ g, SMatrix{nx, dg, T}((L \ G_x)'), L, true
end

# The squared gradient at or below which a scalar constraint is degenerate: its
# rounding scale's, or exactly zero without one.
@inline _gradient_bound²(::Nothing, ::Type{T}) where {T} = zero(T)
@inline _gradient_bound²(ρ::NamedTuple, ::Type{T}) where {T} = T(ρ.gradient)^2

# A residual in its constraint's shape: a number for a scalar constraint, the
# vector for a vector one (`whitened_constraint_residual`'s currency).
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
    s, N, _, _ = _sampson_core(M, x...; kw...)
    return s, N
end
# The geometry layer's one evaluation: the measurements checked and promoted,
# the constraint, its rounding scales and the factor `(s, N, ‖G_x‖, ok)`.
@inline function _sampson_core(M, x...; kw...)
    _check_measurements(M, x...)
    args = _sampson_arguments(M, map(geom, x)...)
    g, G_x, _ = constraint_with_jacobians(args...; kw...)
    return _sampson_factor(g, G_x, constraint_rounding_scale(args...; kw...))
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
    measurement_gradient_jacobian(M, x...; kw...) -> SMatrix{nx, nθ}

The derivative of a SCALAR constraint's measurement gradient `G_x`
([`constraint_with_jacobians`](@ref)'s second return) along the model
coordinates of its `G_θ`: `∂G_x/∂θ`, the mixed second derivative `∂²g/∂x∂θ`.
It is what the Sampson residual's model Jacobian adds to `G_θ`: the normalizer
`‖G_x‖` moves with the model ([`sampson_residual_with_jacobian`](@ref)). A
model implements it beside its constraint — `FundMat` and `EssentialMat` on
image-plane points and `EssentialMat` on bearings (with `M₁`, `M₂`), `Circle`,
`RadialCamera`. GTSAM and Ceres have no counterpart: they differentiate an
explicit residual, not an implicit constraint's gradient.
"""
function measurement_gradient_jacobian end

# ½·∂‖G_x‖²/∂θ = (∂G_x/∂θ)ᵀ·G_x: the one contraction of the measurement
# gradient's derivative the Sampson Jacobian reads. A model whose ∂G_x/∂θ is
# structured may compute the contraction directly (the bilinear epipolar
# models, fundmat.jl — pinned equal to this default in the tests).
@inline _normal_rate(M, G_x, x...; kw...) = transpose(measurement_gradient_jacobian(M, x...; kw...)) * G_x

# A model-coordinate row carried along chart directions, held as ROWS
# B = Dᵀ (`N × nθ`, row q = ∂θ/∂a_q; formed once per sweep — a transposed
# product per row costs 2 ns of a 5 ns row): B·v. `nothing` keeps the model's
# own coordinates.
@inline _along(v::StaticVector, ::Nothing) = v
@inline _along(v::StaticVector, B::StaticMatrix) = B * v
# Directions as given — `nothing`, an `nθ × N` matrix D (`retract_jacobian`),
# or the models they move to, `dM = (∂M/∂θ₁, …)` (`retract_directions`) — as
# those rows.
@inline _direction_rows(::Nothing) = nothing
@inline _direction_rows(D::StaticMatrix{nθ, N, T}) where {nθ, N, T} = SMatrix{N, nθ, T}(transpose(D))
@inline function _direction_rows(dM::Tuple{StaticMatrix{p, q}, Vararg{StaticMatrix{p, q}, K}}) where {p, q, K}
    N = K + 1
    return SMatrix{N, p * q}(ntuple(k -> dM[(k - 1) % N + 1][(k - 1) ÷ N + 1], Val(N * p * q)))
end

"""
    sampson_residual_with_jacobian(M, x...; directions = nothing, kw...)
        -> (s_N::SVector{1}, J::SMatrix{1, N})

The Sampson residual of a SCALAR constraint ([`sampson_residual`](@ref)) and
its exact derivative along the model: with `g`, `G_x`, `G_θ` from
[`constraint_with_jacobians`](@ref) and `∂G_x/∂θ` from
[`measurement_gradient_jacobian`](@ref),

    ∂(g/‖G_x‖)/∂θ = (G_θ − (g/‖G_x‖²)·(∂G_x/∂θ)ᵀG_x) / ‖G_x‖,

carried along the chart `directions` (`nθ × N`, e.g. `retract_jacobian(ch, a)`;
`nothing` keeps the model's own coordinates, `N = nθ`). The residual an LM
descends and the row it descends it with, from the one Sampson path: the
residual's degeneracy rule is [`sampson_residual_with_normal`](@ref)'s, and a
degenerate measurement has a ZERO row (a least-squares row cannot carry `±Inf`).
Keywords are forwarded to the constraint (`M₁`, `M₂` for an `EssentialMat` on
bearings). Element-type generic and device-safe; [`sampson_jacobian_rows!`](@ref)
runs it per correspondence.

A vector constraint (a homography's two rows) has no such row here: its LM
whitens frozen ([`whitening_factor`](@ref), the transfer rows).
"""
@inline sampson_residual_with_jacobian(M, x...; directions = nothing, kw...) =
    _sampson_with_jacobian(M, _direction_rows(directions), x...; kw...)
@inline sampson_residual_with_jacobian(M, c::Pair; kw...) = sampson_residual_with_jacobian(M, c.first, c.second; kw...)
@inline function _sampson_with_jacobian(M, B, x...; kw...)
    _check_measurements(M, x...)
    args = _sampson_arguments(M, map(geom, x)...)
    g, G_x, G_θ = constraint_with_jacobians(args...; kw...)
    return _sampson_jacobian(g, G_x, G_θ, constraint_rounding_scale(args...; kw...), B, args...; kw...)
end
# Branch-free (a branch in the row stops a row loop vectorizing): a degenerate
# measurement scales its row by zero and takes no step through the rate.
@inline function _sampson_jacobian(g::Real, G_x, G_θ, ρ, B, args...; kw...)
    s, _, n, ok = _sampson_factor(g, G_x, ρ)
    n⁻¹ = ifelse(ok, inv(n), zero(n))                 # one division per row (0 when degenerate)
    v = G_θ - (g * n⁻¹ * n⁻¹) * _normal_rate(args[1], G_x, Base.tail(args)...; kw...)   # g/‖G_x‖²
    row = _along(v, B) * n⁻¹
    return s, _as_row(row)
end
@inline _as_row(v::StaticVector{N, T}) where {N, T} = SMatrix{1, N, T}(Tuple(v))

# The √w-weighted rows of a two-measurement scalar constraint over a set, each
# row from the one path above; per-point keyword columns (`M₁`, `M₂`) are read
# at the row's index. A degenerate pair is a zero row (LM-safe), NaN propagates.
@inline _columns_at(columns::NamedTuple, i) = map(c -> (@inbounds c[i]), columns)
Base.@propagate_inbounds function _sampson_residual_row!(r, i, M, u₁, u₂, sqrtw, columns)
    s, _, _, ok = _sampson_core(M, u₁[i], u₂[i]; _columns_at(columns, i)...)
    r[i] = ifelse(ok, eltype(r)(sqrtw[i] * only(s)), zero(eltype(r)))
    return nothing
end
Base.@propagate_inbounds function _sampson_jacobian_row!(J, i, M, B, u₁, u₂, sqrtw, columns)
    _, row = _sampson_with_jacobian(M, B, u₁[i], u₂[i]; _columns_at(columns, i)...)
    w = sqrtw[i]
    for q in 1:size(B, 1)
        J[i, q] = w * row[1, q]
    end
    return nothing
end

"""
    sampson_rows!(r, M, u₁, u₂, sqrtw; columns...) -> r
    sampson_jacobian_rows!(J, M, directions, u₁, u₂, sqrtw; columns...) -> J

The √w-weighted Sampson residual rows of the scalar two-measurement constraint
of `M` over `u₁[i] => u₂[i]` — `r[i] = sqrtw[i]·only(`[`sampson_residual`](@ref)`(M,
u₁[i], u₂[i]))` — and their Jacobian along the chart `directions`
(`J[i, q] = sqrtw[i]·∂rᵢ/∂a_q`, [`sampson_residual_with_jacobian`](@ref)):
`directions` an `nθ × N` matrix or the tuple of models `dM = (∂M/∂θ₁, …)`
([`retract_directions`](@ref)). Per-point keyword columns go to the constraint
at each row: an `EssentialMat` on unit bearings takes `M₁`, `M₂`, the
backprojection Jacobians that put the residual in PIXELS. A degenerate pair is
a zero row. ONE KernelAbstractions kernel on the backend that owns `r` (`J`), a
plain loop over the same row function on the CPU; `M` in the measurement frame
of the points.
"""
function sampson_rows!(r::AbstractVector, M, u₁::AbstractVector, u₂::AbstractVector, sqrtw::AbstractVector; columns...)
    return _foreach_row!(_sampson_residual_row!, r, length(u₁), M, u₁, u₂, sqrtw, NamedTuple(columns))
end
function sampson_jacobian_rows!(J::AbstractMatrix, M, directions, u₁::AbstractVector, u₂::AbstractVector, sqrtw::AbstractVector; columns...)
    return _foreach_row!(_sampson_jacobian_row!, J, length(u₁), M, _direction_rows(directions), u₁, u₂, sqrtw, NamedTuple(columns))
end

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
and its others ±Inf; a degenerate Sampson residual (zero `N`) is returned unchanged.
NaN propagates.

A least-squares row of the whitened residual is the Sampson row of
[`sampson_residual_with_jacobian`](@ref) divided by the same factor, held fixed
over a step (the Gauss–Newton row: `NᵀΣN` moves with the model only through `N`).
"""
@inline function whiten(s::StaticVector, N::StaticMatrix, Σ)
    Σ isa UniformScaling{Bool} && return s
    return _whiten_normal(s, N, Σ)
end
@inline function _whiten_normal(s::StaticVector{1, T}, N::StaticMatrix{nx, 1}, Σ) where {nx, T}
    σ² = _propagate_scalar(SVector{nx}(N), Σ)
    (σ² > zero(σ²) || isnan(σ²)) || return _degenerate_whitened(s)
    return s / sqrt(σ²)
end
@inline function _whiten_normal(s::StaticVector{dg, T}, N::StaticMatrix{nx, dg}, Σ) where {dg, nx, T}
    C, d, ok = _whitening_cholesky(propagate(N', Σ))
    isnan(d) && return s .* T(NaN)
    ok || return _degenerate_whitened(s)
    return C.L \ s
end
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

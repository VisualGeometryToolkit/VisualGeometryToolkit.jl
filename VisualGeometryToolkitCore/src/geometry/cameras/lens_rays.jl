# =============================================================================
# The lens contract on RAYS
# =============================================================================
#
# A lens is a map between RAYS and NORMALIZED image points, and the pair
#
#     project(lens, X::StaticVector{3}) -> Vec2       a ray (any positive scale) → its image point
#     backproject(lens, xd::StaticVector{2}) -> Vec3  an image point → its unit ray
#
# is the whole contract. Both are NaN where the lens does not image the
# argument and never throw, both are allocation-free, and both are
# eltype-generic, so one body serves Float32 kernels and ForwardDiff Duals.
# Everything else is derived from the pair, the same way for every lens:
# `transform_deriv(lens, X)`, `transform_deriv_params(lens, X)` and
# `project_with_jacobians(lens, X)`.
#
# A `Lens` projects onto its projection's frame and distorts there
# (`_distort_in_frame` / `_undistort_in_frame`, dispatched on the
# distortion's structure); a `LiftLens` lifts the image point to its ray in
# closed form and projects by a root on the ray. 90° is an ordinary angle
# everywhere except in the perspective projection, whose domain genuinely
# ends there.
# =============================================================================

# -----------------------------------------------------------------------------
# Iterative inverses run on primal numbers
# -----------------------------------------------------------------------------
#
# An inverse that iterates (a fixed point, a Newton root) solves on PRIMAL
# values. When a ForwardDiff Dual rides in the argument or in the lens, ONE
# Newton step on the Duals from the converged root carries the exact first
# derivatives (implicit function theorem): no Dual ever runs through an
# iteration, and a Float result is the solver's own.

@inline _primal(x::Real) = x
@inline _primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))

@inline _is_dual_type(::Type{T}) where {T} = T !== Union{} && T <: ForwardDiff.Dual
_primal_float(::Type{T}) where {T} = _is_dual_type(T) ? _primal_float(ForwardDiff.valtype(T)) : T


# A lens, or any part of one, with its coefficients stripped to primal values.
@inline _primal(x) = _is_dual_type(_coeff_type(x)) ? _with_own_parameters(x, map(_primal, _coefficients_of(x))) : x

# A NaN of the same type, unless `ok`.
@inline _nan_unless(ok::Bool, v::StaticVector) = ok ? v : v * eltype(v)(NaN)

# The "not imaged" point, NaN built from an input number or vector — so a dual
# number carries NaN partials as `_nan_unless` does; a NaN with zero partials
# would report a flat derivative where nothing is imaged.
@inline _nan_like(v) = first(v) * float(eltype(v))(NaN)
@inline _nan_vec2(v) = (n = _nan_like(v); Vec2(n, n))
@inline _nan_vec3(v) = (n = _nan_like(v); Vec3(n, n, n))

# Squared residual of a 2-D solve against the per-precision tolerance —
# widened by the target's own rounding: an absolute bound alone fails a converged
# Float32 solve at a large radius, where eps·|target| exceeds it.
@inline _residual_ok(f, target::StaticVector{2, S}) where {S} =
    (f[1] - target[1])^2 + (f[2] - target[2])^2 <= _core_tol(S)^2 + (8 * eps(S))^2 * (target[1]^2 + target[2]^2)

@inline _float_static(v::StaticVector{N}) where {N} = SVector{N, float(eltype(v))}(Tuple(v))

# Residual tolerance of a converged solve, per precision.
_core_tol(::Type{Float64}) = 1.0e-10
_core_tol(::Type{Float32}) = 1.0f-5
_core_tol(::Type{T}) where {T <: AbstractFloat} = sqrt(eps(T))

# When an iteration has converged, by its order. A Newton step of size δ leaves
# an error of order δ², so it has settled once δ is at the solve tolerance of the
# iterate's scale; a contraction's update of size δ leaves an error of order δ,
# so it must reach rounding level. Every solver here stops on one of the two and
# runs its trip cap only when it does not converge. A non-finite update counts as
# settled (NaN fails the `>`), so a diverged iterate exits at once rather than
# running the trip cap; the caller's residual check then rejects it.
@inline _newton_settled(δ2, scale2, ::Type{S}) where {S} = !(δ2 > _core_tol(S)^2 * max(scale2, one(S)))
@inline _contraction_settled(δ2, scale2, ::Type{S}) where {S} = !(δ2 > (8 * eps(S))^2 * max(scale2, one(S)))

# The structure-blind inverse of an in-plane map `f`: trust-region 2-D Newton
# from `seed` (x_d by default) with a ForwardDiff Jacobian and COLMAP's step
# clipping (relative and absolute radius 0.1, IterativeUndistortion), capped
# per precision. For the maps with no cheaper structure — a chain —
# and as the finisher where a structured solve does not certify.
# Trip caps per precision: (fixed point, Newton) — 30 and 20 in Float64, the
# fixed point's validated against OpenCV's own loop; 12 otherwise.
@inline _trips(::Type{Float64}) = (30, 20)
@inline _trips(::Type{<:AbstractFloat}) = (12, 12)

@inline function _newton_solve(f::F, xd::SVector{2, S}, seed::SVector{2, S} = xd) where {F, S <: AbstractFloat}
    rel2 = S(1) / S(100)
    x = ifelse(isfinite(seed[1]) & isfinite(seed[2]), seed, xd)
    dJ = zero(S)
    for _ in 1:last(_trips(S))
        fx = f(x)
        J = ForwardDiff.jacobian(z -> SVector{2}(Tuple(f(z))), x)
        dJ = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
        Δx = J \ SVector{2, S}(fx[1] - xd[1], fx[2] - xd[2])
        radius2 = max((x[1] * x[1] + x[2] * x[2]) * rel2, rel2)
        δ2 = Δx[1] * Δx[1] + Δx[2] * Δx[2]
        scale = ifelse(δ2 > radius2, sqrt(radius2 / max(δ2, eps(S))), one(S))
        x = SVector{2, S}(x[1] - scale * Δx[1], x[2] - scale * Δx[2])
        _newton_settled(scale * scale * δ2, x[1] * x[1] + x[2] * x[2], S) && break
    end
    return x, _residual_ok(f(x), xd), dJ
end
# (`dJ` is det J at the last iterate's start, within the solve tolerance of the
# root: the branch test of a Newton root reads it rather than differentiating
# again.)

# -----------------------------------------------------------------------------
# Frames: ray ↔ frame point
# -----------------------------------------------------------------------------
#
# `_frame_project(C, v)` maps a ray (any positive scale) to its point on the
# frame C; `_frame_backproject(C, p)` maps that point back to the
# unit ray. `_sectionable` / `_liftable` are their domains: `project`,
# `backproject` and the framed stages (imaging_chain.jl) answer NaN outside them. Everything is written in r² = X₁² + X₂² rather than in the polar
# radius. A projection is smooth on the optical axis, but `hypot`,
# `atan(r, z)/r` and `sin(r)/r` are 0/0 there and so are their derivatives —
# which is how Kannala–Brandt's Jacobian came to be NaN on the axis.

# atan(r, z)/r, sin(r)/r and cos(r) as the even functions of r they are: the
# quotient outside a √eps neighbourhood of the axis, a three-term series —
# exact to rounding in value AND first derivative — inside it.
@inline _series_radius2(x) = sqrt(eps(float(typeof(_primal(x)))))

@inline function _atan_ratio(r2, z)
    # the test never divides: z can be 0 (a ray at 90°) or negative (past it)
    if z > 0 && r2 <= _series_radius2(r2) * (z * z)
        u = r2 / (z * z)
        return (1 - u / 3 + u * u / 5) / z
    end
    r = sqrt(r2)
    return atan(r, z) / r
end

@inline function _sin_ratio(r2)
    r2 <= _series_radius2(r2) && return 1 - r2 / 6 + r2 * r2 / 120
    r = sqrt(r2)
    return sin(r) / r
end

@inline _cos_root(r2) = r2 <= _series_radius2(r2) ? 1 - r2 / 2 + r2 * r2 / 24 : cos(sqrt(r2))

# ‖v‖ + v₃ without cancellation: direct on the forward hemisphere, through the
# conjugate (n + v₃)(n − v₃) = r² on the backward one, where the direct sum
# cancels catastrophically as θ → π.
@inline function _norm_plus_z(r2, z)
    n = sqrt(r2 + z * z)
    return z >= 0 ? n + z : r2 / (n - z)
end

@inline _off_axis(v::StaticVector{3}) = !(v[1] == 0 && v[2] == 0)

# Perspective: X/X₃, forward hemisphere only.
@inline _sectionable(::Type{<:GnomonicFrame}, v::StaticVector{3}) = v[3] > 0
@inline _frame_project(::Type{<:GnomonicFrame}, v::StaticVector{3}) = Vec2(PerspectiveMap()(v))
@inline _frame_backproject(::Type{<:GnomonicFrame}, p::StaticVector{2}) = normalize(Vec3(p[1], p[2], one(p[1])))

# Equidistant: r = θ, θ ∈ [0, π).
@inline function _frame_project(::Type{<:EquidistantFrame}, v::StaticVector{3})
    s = _atan_ratio(v[1] * v[1] + v[2] * v[2], v[3])
    return Vec2(s * v[1], s * v[2])
end
@inline function _frame_backproject(::Type{<:EquidistantFrame}, p::StaticVector{2})
    r2 = p[1] * p[1] + p[2] * p[2]
    s = _sin_ratio(r2)
    return Vec3(s * p[1], s * p[2], _cos_root(r2))
end

# Stereographic: r = 2·tan(θ/2), θ ∈ [0, π); total lift.
@inline function _frame_project(::Type{<:StereographicFrame}, v::StaticVector{3})
    s = 2 / _norm_plus_z(v[1] * v[1] + v[2] * v[2], v[3])
    return Vec2(s * v[1], s * v[2])
end
@inline function _frame_backproject(::Type{<:StereographicFrame}, p::StaticVector{2})
    q = (p[1] * p[1] + p[2] * p[2]) / 4                  # tan²(θ/2)
    s = inv(1 + q)                                       # sin θ / r
    return Vec3(s * p[1], s * p[2], (1 - q) * s)
end

# Equisolid: r = 2·sin(θ/2), θ ∈ [0, π]; radius ≤ 2.
@inline function _frame_project(::Type{<:EquisolidFrame}, v::StaticVector{3})
    r2 = v[1] * v[1] + v[2] * v[2]
    s = sqrt(2 / (sqrt(r2 + v[3] * v[3]) * _norm_plus_z(r2, v[3])))
    return Vec2(s * v[1], s * v[2])
end
@inline function _frame_backproject(::Type{<:EquisolidFrame}, p::StaticVector{2})
    q = (p[1] * p[1] + p[2] * p[2]) / 4                  # sin²(θ/2)
    s = sqrt(1 - q)                                      # sin θ / r
    return Vec3(s * p[1], s * p[2], 1 - 2 * q)
end

# Orthographic: r = sin θ, θ ∈ [0, π/2] — the projection folds at the equator.
@inline function _frame_project(::Type{<:OrthographicFrame}, v::StaticVector{3})
    n = norm(v)
    return Vec2(v[1] / n, v[2] / n)
end
@inline _frame_backproject(::Type{<:OrthographicFrame}, p::StaticVector{2}) =
    Vec3(p[1], p[2], sqrt(1 - (p[1] * p[1] + p[2] * p[2])))

# The radius of the disc a frame's points fill — r = θ < π, 2·sin(θ/2) ≤ 2,
# sin θ ≤ 1 — and its lift's domain, the closed disc (Inf: the whole plane).
_frame_radius(::Type{<:RetinalFrame}) = Inf
_frame_radius(::Type{<:EquidistantFrame}) = π
_frame_radius(::Type{<:EquisolidFrame}) = 2
_frame_radius(::Type{<:OrthographicFrame}) = 1
@inline _liftable(C::Type{<:RetinalFrame}, p::StaticVector{2}) = p[1] * p[1] + p[2] * p[2] <= abs2(float(eltype(p))(_frame_radius(C)))

# Domains of the forward maps: the zero vector is no ray, and neither is the
# exact backward axis for a projection that images it as a whole circle.
@inline _sectionable(
    ::Type{<:Union{EquidistantFrame, StereographicFrame, EquisolidFrame}}, v::StaticVector{3}
) = v[3] > 0 || _off_axis(v)
@inline _sectionable(::Type{<:OrthographicFrame}, v::StaticVector{3}) =
    v[3] > 0 || (v[3] == 0 && _off_axis(v))

# -----------------------------------------------------------------------------
# The radial factor F = N/D
# -----------------------------------------------------------------------------
#
# N = 1 + Σ kᵢ φᵢ(u), D = 1 + Σ dⱼ φⱼ(u), Horner on the coefficient tuples
# (`evalpoly`, unrolled at compile time). The basis variable u is s = ρ² for the
# Even basis (φᵢ = sⁱ) and ρ itself for the Full basis (φᵢ = ρⁱ⁺¹: the
# polynomial 1 + 0·ρ + k₁ρ² + …).

# The ascending coefficients of N or D in the basis variable u.
@inline _lead(::Type{Even}, u) = (one(u),)
@inline _lead(::Type{Full}, u) = (one(u), zero(u))
@inline _ucoeffs(B, c::Tuple, ::Type{T}) where {T} = (a = (_lead(B, one(T))..., c...); SVector{length(a), T}(a))
@inline _poly(B, c::Tuple, u) = evalpoly(u, (_lead(B, u)..., c...))
@inline function _dpoly(B, c::Tuple, u)                                  # d/du of the same
    p = (_lead(B, u)..., c...)
    return length(p) == 1 ? zero(u) : evalpoly(u, ntuple(i -> i * p[i + 1], Val(length(p) - 1)))
end
@inline _num(r::Radial{B}, u) where {B} = _poly(B, r.k, u)
@inline _den(r::Radial{B}, u) where {B} = _poly(B, r.d, u)
@inline _dnum(r::Radial{B}, u) where {B} = _dpoly(B, r.k, u)
@inline _dden(r::Radial{B}, u) where {B} = _dpoly(B, r.d, u)

"""
    radial_ratio(r::Radial, u) -> N(u)/D(u)

The radial factor of `r` at its basis variable `u` (`ρ²` for the Even basis):
for a [`LiftLens`](@ref), the undistortion `x_u = x_d · radial_ratio(radial(lens), ρ_d²)`.
"""
@inline radial_ratio(r::Radial, u) = _num(r, u) / _den(r, u)
@inline radial_ratio(r::Radial{B, P, 0}, u) where {B, P} = _num(r, u)     # no division by one (a GPU's f32 division is not exact)

# The basis variable u and du/dv at v·√a — v = ρ with a = 1, or the ratio
# m = ρ/r with a = r²: u = v²a (Even), v√a (Full) — and the radius at a given u.
@inline _basis(::Radial{Even}, v, a) = (v * v * a, 2 * v * a)
@inline _basis(::Radial{Full}, v, a) = (s = sqrt(a); (v * s, s))
_radius_at(::Radial{Even}, u) = sqrt(u)
_radius_at(::Radial{Full}, u) = u

# The fold test of the profile ρ ↦ ρ·N(u)/D(u) as ascending coefficients in u:
# cₙ = Σ_{i+j=n} (1 + m(i − j))·aᵢ·bⱼ over N's coefficients a and D's b in u, with
# m = 2 for u = ρ² and 1 for u = ρ. Its first positive root is where the profile
# stops increasing. The same polynomial is a lift's monotonicity test: with
# θ(ρ) = atan(ρN, D), dθ/dρ ∝ it. The pole is D's first positive root.
@inline _fold_coeffs(r::Radial{B, P, Q, T}) where {B, P, Q, T} = _fold_coeffs(B, r.k, r.d, T)
@inline function _fold_coeffs(::Type{B}, k::Tuple, d::Tuple, ::Type{T}) where {B, T}
    a, b = _ucoeffs(B, k, T), _ucoeffs(B, d, T)
    NA, NB = length(a), length(b)
    m = B === Even ? 2 : 1
    return SVector{NA + NB - 1, T}(
        ntuple(Val(NA + NB - 1)) do k
            n = k - 1
            acc = zero(T)
            for i in max(0, n - (NB - 1)):min(n, NA - 1)
                acc += (1 + m * i - m * (n - i)) * a[i + 1] * b[n - i + 1]
            end
            acc
        end
    )
end

# The radial factor's imaged branch is the one interval [0, u_end] of its basis
# variable: N and D are 1 at 0, and nothing past the fold or the pole is a
# projection. `_branch_end` is its end capped at `cap`; the constructor stores
# the uncapped ends (`_radial_branch_ends`: `u_end == _branch_end(r, Inf)`), so
# the per-point test is a comparison. A lift's branch ends at the fold alone
# (its pole is the 90° ray): `u <= r.u_fold`.
@inline _pole_coeffs(r::Radial{B, P, Q, T}) where {B, P, Q, T} = _ucoeffs(B, r.d, T)
@inline _on_radial_branch(r::Radial, u) = u <= r.u_end
@inline _branch_end(r::Radial{B, P, Q, T}, cap) where {B, P, Q, T} = min(r.u_end, float(T)(cap))   # an integer Radial: the cap in its float

# THE sweep for the ends, run once per factor by its constructor: (u_fold, u_end)
# of the coefficients, computed on their primal values in float precision (a
# Dual factor's branch is its primal's) and stored in T.
function _radial_branch_ends(::Type{B}, ::Type{T}, k::Tuple, d::Tuple) where {B, T}
    S = float(_primal_float(T))
    kp, dp = map(x -> S(_primal(x)), k), map(x -> S(_primal(x)), d)
    fold = _fold_point(_fold_coeffs(B, kp, dp, S), Inf)
    return _stored_end(T, fold), _stored_end(T, min(fold, _fold_point(_ucoeffs(B, dp, S), Inf)))
end
_stored_end(::Type{T}, u) where {T} = T(u)
_stored_end(::Type{T}, u) where {T <: Integer} = isfinite(u) ? floor(T, u) : typemax(T)

# The relative step inside a stored end that a reader evaluating AT the end takes:
# ρ = √u and ρ² round a few ulps past a closed end in the lens's own precision
# (1e-9 at Float64, as before; 8 ulps at Float32).
_end_margin(::Type{T}) where {T} = max(1.0e-9, 8 * Float64(eps(float(_primal_float(T)))))
_end_margin(::Type{Union{}}) = 1.0e-9


# -----------------------------------------------------------------------------
# Distortion in the frame: forward
# -----------------------------------------------------------------------------
#
# `_distort(d, x)` is the plain map; `_distort_in_frame(lens, x)` adds the
# branch test of the lens's radial factors: past a fold (or pole) two rays
# would share an image point, so the point is not imaged (NaN).

@inline _distort(::Nothing, x::StaticVector{2}) = Vec2(x[1], x[2])
@inline function _distort(r::Radial{Even}, x::StaticVector{2})
    F = radial_ratio(r, x[1] * x[1] + x[2] * x[2])
    return Vec2(F * x[1], F * x[2])
end
@inline function _delta(t::Tangential, x::StaticVector{2})
    u, v = x[1], x[2]
    r2 = u * u + v * v
    return Vec2(2 * t.p1 * u * v + t.p2 * (r2 + 2 * u * u), 2 * t.p2 * u * v + t.p1 * (r2 + 2 * v * v))
end
@inline function _delta(t::ThinPrism, x::StaticVector{2})
    s = x[1] * x[1] + x[2] * x[2]
    return Vec2(s * evalpoly(s, t.sx), s * evalpoly(s, t.sy))
end

# OpenCV's tilt: (x, 1) through computeTiltProjectionMatrix(τx, τy), dehomogenized.
@inline function _tilt_matrix(t::Tilt)
    cx, sx, cy, sy = cos(t.tx), sin(t.tx), cos(t.ty), sin(t.ty)
    Rx = SMatrix{3, 3}(1, 0, 0, 0, cx, -sx, 0, sx, cx)          # column-major of OpenCV's row-major Matx
    Ry = SMatrix{3, 3}(cy, 0, sy, 0, 1, 0, -sy, 0, cy)
    R = Ry * Rx
    Pz = SMatrix{3, 3}(R[3, 3], 0, 0, 0, R[3, 3], 0, -R[1, 3], -R[2, 3], 1)
    return Pz * R
end
@inline function _distort(t::Tilt, x::StaticVector{2})
    v = _tilt_matrix(t) * SVector(x[1], x[2], one(x[1]))
    return Vec2(v[1] / v[3], v[2] / v[3])
end

# Devernay–Faugeras: F(r) = atan(a·r)/(r·ω), a = 2·tan(ω/2), evaluated as
# (a/ω)·(atan(a r)/(a r)) with both ratios series-expanded where they are 0/0.
@inline function _fov_scale(ω)
    h = ω / 2
    return abs(h) <= _series_radius2(h) ? 1 + h * h / 3 : tan(h) / h            # a/ω = tan(ω/2)/(ω/2)
end
@inline function _fov_factor(t::FieldOfView, r2)
    ω = t.omega
    a = 2 * tan(ω / 2)
    q = a * a * r2
    g = q <= _series_radius2(q) ? 1 - q / 3 + q * q / 5 : atan(sqrt(q)) / sqrt(q)
    return _fov_scale(ω) * g
end
@inline function _distort(t::FieldOfView, x::StaticVector{2})
    F = _fov_factor(t, x[1] * x[1] + x[2] * x[2])
    return Vec2(F * x[1], F * x[2])
end

# An Additive — or a lone delta, the Additive of one — is its head (the leading
# map, or the point itself) plus its deltas, summed left to right, every term at
# the same point.
@inline _delta_sum(ds::Tuple, x) = _delta_sum(_delta(first(ds), x), Base.tail(ds), x)   # left to right
@inline _delta_sum(acc, ::Tuple{}, x) = acc
@inline _delta_sum(acc, ds::Tuple, x) = _delta_sum(acc + _delta(first(ds), x), Base.tail(ds), x)
@inline function _distort(a::Union{Additive, Tangential, ThinPrism}, x::StaticVector{2})
    h, ds = _head(a), _deltas(a)
    y = h === nothing ? Vec2(x[1], x[2]) : _distort(h, x)
    return isempty(ds) ? y : y + _delta_sum(ds, x)
end

@inline _distort(c::Chain, x::StaticVector{2}) = _chain_distort(c.stages, x)
@inline _chain_distort(::Tuple{}, x) = x
@inline _chain_distort(ds::Tuple, x) = _chain_distort(Base.tail(ds), _distort(first(ds), x))

@inline function _distort_in_frame(l::Lens, x::StaticVector{2})
    r = radial(l)
    r === nothing && return _distort(l.distortion, x)
    ok = _on_radial_branch(_primal(r), _primal(x[1] * x[1] + x[2] * x[2]))
    return _nan_unless(ok, _distort(l.distortion, x))
end
# A Chain has no `radial(lens)` (its radial factor need not lead, and `radial`
# feeds `radial_part` and `view_angle_limit`, which mean the lens's one radially
# symmetric factor): its forward map tests every radial factor at that stage's
# own input — the radial half of the inverse's acceptance (`_radial_ok`), so a
# ray past a radial stage's fold is NaN both ways. The determinant half is NOT
# tested forward (a Jacobian per projected point): a ray where the deltas or the
# tilt fold maps to a finite point whose `backproject` is the imaged branch's
# preimage (a different ray), or NaN (see `project`).
@inline function _distort_in_frame(l::Lens{<:AbstractProjection, <:Chain}, x::StaticVector{2})
    ok = _radial_ok(_primal(l.distortion), SVector(_primal(x[1]), _primal(x[2])))
    return _nan_unless(ok, _distort(l.distortion, x))
end

# -----------------------------------------------------------------------------
# Distortion in the frame: inverse
# -----------------------------------------------------------------------------
#
# Dispatch on the distortion's structure:
#   nothing                   the identity
#   a lone Radial             the 1-D root on the ray (x_d, 1) — a lift's own solve
#   Additive, a lone delta    OpenCV's undistortPoints fixed point, finished by the
#                             trust-region 2-D Newton
#   FieldOfView, Tilt         closed form
#   anything else (a Chain)   the trust-region 2-D Newton (a Full-basis radial
#                             factor never gets here: the Lens constructor
#                             refuses one anywhere in the distortion)
# A 2-D root stands only where the pointwise branch test holds (`_branch_root`);
# otherwise it is continued from the axis.

@inline _undistort_in_frame(l::Lens, xd::StaticVector{2}) = _undistort(l.distortion, l, xd)

@inline _undistort(::Nothing, l, xd::StaticVector{2}) = Vec2(xd[1], xd[2])

# THE frame of every 2-D inverse: `solve(lp, x0) -> (x, ok)` on the PRIMAL lens
# from the primal start `x0 = xd`; when the lens or the point carries Duals, one
# Newton step on the dual map `fwd(lens, x)` (a 2-vector) puts the derivative on
# the root. NaN unless the solve converged.
@inline function _undistort_2d(solve::F, fwd::G, lens, xd::StaticVector{2}) where {F, G}
    lp = _primal(lens)
    T = promote_type(eltype(xd), _coeff_type(lens))
    S = float(_primal_float(T))
    x0 = SVector{2, S}(_primal(xd[1]), _primal(xd[2]))
    x, ok = solve(lp, x0)
    if _is_dual_type(T)
        J = ForwardDiff.jacobian(z -> fwd(lp, z), x)
        x = x - J \ (fwd(lens, x) - xd)
    end
    return _nan_unless(ok, Vec2(x[1], x[2]))
end

# The plain forward map of a lens's distortion, as an SVector{2}: what the 2-D
# inverses differentiate and check their residual against.
@inline _frame_fwd(l::Lens, z) = SVector{2}(Tuple(_distort(l.distortion, z)))

# A lone radial factor: x = g·x_d with G(g) = g·F(t·g²) − 1 = 0 at t = |x_d|². g = 1
# on the axis and no square root appears, so value and derivative are exact
# there. Newton from g = 1; the root must lie on the factor's branch. Where Newton
# does not reach it, a safeguarded solve on the branch (`_ratio_on_branch`) does;
# NaN only where the branch never reaches the radius. (The same root as a lift's
# on the "ray" (x_d, 1), solved in the ratio — measured faster on a GPU.)
@inline _ratio_G(r::Radial, g, t) = g * radial_ratio(r, t * g * g) - 1
@inline _ratio_dG(r::Radial, g, t) = (w = t * g * g; radial_ratio(r, w) + 2 * w * _dratio(r, w))
@inline _dratio(r::Radial, u) = (N = _num(r, u); D = _den(r, u); (_dnum(r, u) * D - N * _dden(r, u)) / (D * D))
@inline _dratio(r::Radial{B, P, 0}, u) where {B, P} = _dnum(r, u)

@inline function _undistort(r::Radial{Even}, l::Lens, xd::StaticVector{2})
    t = xd[1] * xd[1] + xd[2] * xd[2]
    rp, tp = _primal(r), _primal(t)
    T = promote_type(eltype(xd), _coeff_type(l))
    S = float(_primal_float(T))
    g = one(S)
    settled = false
    for _ in 1:12
        Δg = _ratio_G(rp, g, tp) / _ratio_dG(rp, g, tp)
        g -= Δg
        settled = _newton_settled(Δg * Δg, g * g, S)
        settled && break
    end
    # An iterate stands when it has SETTLED (its error is the square of an update
    # at the solve tolerance: rounding) or its residual is at rounding already (at
    # the image of a fold, where G′ → 0 and Newton cannot settle, the root is as
    # good as the precision makes it). An iterate still crawling in from g = 1
    # when the trips run out can meet the solve tolerance with its error at that
    # tolerance (measured: Kannala–Brandt far from the axis, a 7e-7 update left a
    # 1e-11 ray error); the safeguarded solve finishes it, to rounding.
    G = abs(_ratio_G(rp, g, tp))
    ok = (settled ? G <= _core_tol(S) : G <= 8 * eps(S)) && g > 0 && _on_radial_branch(rp, g * g * tp)
    ok || ((g, ok) = _ratio_on_branch(rp, _frame_cap(l.projection), S(tp)))
    _is_dual_type(T) && (g = g - _ratio_G(r, g, t) / _ratio_dG(rp, g, tp))
    return _nan_unless(ok, Vec2(g * xd[1], g * xd[2]))
end

# The safeguarded solve: ρ·F(ρ²) = ρ_d on [0, ρ_branch), the factor's increasing
# branch (ended by its fold, its pole, or the frame's own edge); returns g = ρ/ρ_d
# and whether it converged. `_ratio_branch_solve` also says whether the branch
# reaches ρ_d at all: a 2-D solve's seed needs only that (`_radial_seed`).
@inline function _ratio_on_branch(r::Radial, cap, t::S) where {S <: AbstractFloat}
    g, reached, conv = _ratio_branch_solve(r, cap, t)
    return g, reached & conv
end
@noinline function _ratio_branch_solve(r::Radial, cap, t::S) where {S <: AbstractFloat}
    ρd = sqrt(t)
    prof(ρ) = ρ * radial_ratio(r, ρ * ρ)
    dprof(ρ) = (w = ρ * ρ; radial_ratio(r, w) + 2 * w * _dratio(r, w))
    hi = S(sqrt(min(r.u_end, S(cap))))                # the stored end, capped: no sweep per point
    (ρd > 0) || return one(S), false, false
    # At a pole the rounded end can land just past it, where the profile is
    # negative (measured: Float32, prof = −7·10⁷ one rounding past the pole, which
    # declared a reachable radius unreached); read it just inside instead.
    phi = prof(hi)
    phi < 0 && (phi = prof(hi * (1 - sqrt(eps(S)))))
    isfinite(hi) && !(phi > ρd) && return one(S), false, false
    ρ, conv = _radial_solve(prof, dprof, ρd, hi, Val(24))
    return ρ / ρd, true, conv
end

# An Additive (or a lone delta): x ← (x_d − ΣΔ(x))·D/N(|x|²) — OpenCV's fixed
# point (identical to its loop at its fixed point), x ← x_d − ΣΔ(x) without a
# radial head — to convergence; toward the edge of the monotone field it stops
# contracting and the trust-region Newton finishes, from the last iterate after a
# radial head, afresh from x_d after a peel too strong to converge. A head that is
# another map goes to Newton directly.
@inline _undistort(a::Union{Additive, Tangential, ThinPrism}, l::Lens, xd::StaticVector{2}) =
    _undistort_2d(_frame_fwd, l, xd) do lp, x0
        ap = lp.distortion
        h = _head(ap)
        f = z -> _distort(ap, z)
        x, ok = h isa Union{Nothing, Radial} ? _fixed_point(ap, h, x0) : (x0, false)
        if ok      # the contraction's root: its determinant, unless the cheap radial test already fails
            dJ = _radial_ok(ap, x) ? _det_jacobian(f, x) : zero(eltype(x))
        else
            x, ok, dJ = _newton_solve(f, x0, h isa Radial ? x : x0)
        end
        return _branch_root(f, ap, _frame_cap(lp.projection), x0, x, ok, dJ)
    end

@inline function _fixed_point(a, h, xd::SVector{2, S}) where {S <: AbstractFloat}
    ds = _deltas(a)
    x = xd
    for _ in 1:(h === nothing ? 12 : first(_trips(S)))      # a peel: 12
        u2 = x[1] * x[1] + x[2] * x[2]
        g = h === nothing ? one(S) : _den(h, u2) / _num(h, u2)
        Δ = isempty(ds) ? zero(x) : _delta_sum(ds, x)
        xn = SVector{2, S}((xd[1] - Δ[1]) * g, (xd[2] - Δ[2]) * g)
        d1, d2 = xn[1] - x[1], xn[2] - x[2]
        x = xn
        (!isfinite(d1 + d2) || _contraction_settled(d1 * d1 + d2 * d2, x[1] * x[1] + x[2] * x[2], S)) && break
    end
    return x, _residual_ok(_distort(a, x), xd)
end

# --- closed forms -------------------------------------------------------------
@inline function _undistort(t::FieldOfView, l::Lens, xd::StaticVector{2})
    # r_u = tan(ω·r_d)/a: the factor tan(ω r)/(ω r) · (ω/a), series-expanded at 0/0
    ω = t.omega
    q = ω * ω * (xd[1] * xd[1] + xd[2] * xd[2])
    ok = q < abs2(oftype(float(q), π) / 2)
    g = q <= _series_radius2(q) ? 1 + q / 3 + 2 * q * q / 15 : tan(sqrt(q)) / sqrt(q)
    F = g / _fov_scale(ω)
    return _nan_unless(ok, Vec2(F * xd[1], F * xd[2]))
end

@inline _undistort(t::Tilt, l::Lens, xd::StaticVector{2}) = _tilt_inverse(t, xd)
@inline function _tilt_inverse(t::Tilt, xd::StaticVector{2})
    v = _tilt_matrix(t) \ SVector(xd[1], xd[2], one(xd[1]))
    return Vec2(v[1] / v[3], v[2] / v[3])
end

# --- everything else (a Chain) -------------------------------------------------
@inline _undistort(::AbstractDistortionTerm, l::Lens, xd::StaticVector{2}) =
    _undistort_2d(_frame_fwd, l, xd) do lp, x0
        f = z -> _distort(lp.distortion, z)
        x, ok, dJ = _newton_solve(f, x0)
        return _branch_root(f, lp.distortion, _frame_cap(lp.projection), x0, x, ok, dJ)
    end

# --- the imaged branch of a 2-D inverse ---------------------------------------
#
# A distortion that folds has several preimages of one image point; the lens
# images the one on the branch connected to the axis. The ACCEPTANCE test of a
# root is pointwise: the map keeps orientation there (det J > 0) and every
# radial factor of the distortion is on its own branch at its input (an
# Additive's radial head at the root; each Radial stage of a Chain, or the
# radial head of an Additive stage, at that stage's input). The radial test is
# needed because past a radial profile's fold the determinant turns positive
# again (measured: OpenCV chains with roots 1.4 to 6.3 fold radii out that the
# determinant alone accepts). CONNECTION to the axis is not tested anywhere: a
# pointwise test cannot prove it, and only one of the three ways a root is found
# provides it. The continuation walks from the axis, so its root is on the
# axis's component; the first solve (Newton or the fixed point from x_d) and the
# reseeded Newton after an unfinished walk (`_radial_seed`) are accepted by the
# pointwise test alone. A root of theirs in a far component of {det J > 0}
# inside every radial disk (the deltas or the tilt folding there) would be taken;
# none was found in a search over strong chains and delta-dominated lenses, and
# the tests flood-fill the region to check every root they accept.
# (A Lens's radial factors are Even: the constructor refuses a Full-basis one
# anywhere in the distortion, which has no forward map in the frame.)
@inline _radial_ok(_, x) = true
@inline _radial_ok(r::Radial{Even}, x) = _on_radial_branch(r, x[1] * x[1] + x[2] * x[2])
@inline _radial_ok(a::Additive, x) = (h = _head(a); h === nothing || _radial_ok(h, x))
@inline _radial_ok(c::Chain, x) = _chain_radial_ok(c.stages, x)
@inline _chain_radial_ok(::Tuple{}, x) = true
@inline function _chain_radial_ok(ds::Tuple, x)
    _radial_ok(first(ds), x) || return false
    rest = Base.tail(ds)
    return !_any_radial(rest) || _chain_radial_ok(rest, _distort(first(ds), x))   # a later stage's input only when it is tested
end
@inline _is_radial(::Radial) = true
@inline _is_radial(a::Additive) = _head(a) isa Radial
@inline _is_radial(_) = false
@inline _any_radial(ds::Tuple) = any(_is_radial, ds)

@inline function _det_jacobian(f::F, x::SVector{2}) where {F}
    J = ForwardDiff.jacobian(z -> SVector{2}(Tuple(f(z))), x)
    return J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
end
@inline _on_imaged_branch(f::F, d, x::SVector{2}) where {F} = _radial_ok(d, x) && _det_jacobian(f, x) > 0

# A converged root on the branch stands as solved: the result is the solver's
# own, unchanged. `dJ` is det J at (or, from Newton, within the solve tolerance
# of) the root. Otherwise (no convergence, or a root past a fold) the root is
# continued from the axis.
@inline function _branch_root(f::F, d, cap, xd::SVector{2}, x, ok, dJ) where {F}
    ok && dJ > 0 && _radial_ok(d, x) && return x, true
    return _continued_inverse(f, d, cap, xd)
end

# Continuation along the image segment from the axis (every distortion fixes
# the origin) to x_d: targets t·x_d, each reached by a tangent predictor from
# the last settled root (x + J⁻¹·Δt·x_d, J from that root's last trip) and plain
# Newton, so the root follows the branch it starts on. The spacing adapts: it
# starts at min(1/4, 1/|x_d|) — a strong lens's image grows much faster than
# its root, so a far x_d needs fine targets near the axis — halves from the
# last settled root when a target does not settle in 3 trips or a trip meets
# det J ≤ 0 (down to 1/4 of the start), and doubles back after a settled one
# (up to 1/4). Intermediate targets settle loosely (an update under 1% of the
# root), the last one to the solve tolerance; the walk's trips are capped per
# precision (`_walk_trips`). Where the walk cannot finish (the segment leaves
# the branch's image, which need not be star-shaped about the origin, or the
# budget runs out) the trust-region Newton runs once more from the radial
# factor's own on-branch root (`_radial_seed`), where there is one, its root
# accepted only through the same pointwise test. Trips: 48 in Float64, 24
# otherwise. Measured on 96 random strong OpenCV chains (three seeds, sensor
# lattices of half-width 1 and 3): every point RadiometricCamera's own inverse
# images on the branch is imaged, every root found lies on the axis component
# of {det J > 0, radial factors on their branch} (a flood fill), and of 213659
# interior points of that component every image is inverted to a preimage on
# it (113 to another one than the point: the map overlaps itself there) but
# two, near a fold (det J ≈ 0.02) where Newton converges linearly. Float32
# images the same points as Float64 but 31, where det J ≈ 600 puts the best
# Float32 root's residual above the Float32 tolerance.
_walk_trips(::Type{Float64}) = 48
_walk_trips(::Type{<:AbstractFloat}) = 24

@noinline function _continued_inverse(f::F, d, cap, xd::SVector{2, S}) where {F, S <: AbstractFloat}
    hmax = one(S) / 4
    h = min(hmax, inv(max(one(S), sqrt(xd[1] * xd[1] + xd[2] * xd[2]))))
    hmin = h / 4
    x = zero(xd)
    t = zero(S)
    Jx = ForwardDiff.jacobian(w -> SVector{2}(Tuple(f(w))), x)
    budget = _walk_trips(S)
    while t < 1 && budget > 0 && h >= hmin
        tn = min(t + h, one(S))
        last_target = tn == 1
        y1, y2 = tn * xd[1], tn * xd[2]
        z = x + Jx \ SVector{2, S}((tn - t) * xd[1], (tn - t) * xd[2])          # predictor
        Jz = Jx
        settled = false
        for _ in 1:(last_target ? budget : min(3, budget))
            budget -= 1
            fz = f(z)
            Jz = ForwardDiff.jacobian(w -> SVector{2}(Tuple(f(w))), z)
            Jz[1, 1] * Jz[2, 2] - Jz[1, 2] * Jz[2, 1] > 0 || break           # a fold: shorten
            Δz = Jz \ SVector{2, S}(fz[1] - y1, fz[2] - y2)
            z = SVector{2, S}(z[1] - Δz[1], z[2] - Δz[2])
            δ2, s2 = Δz[1] * Δz[1] + Δz[2] * Δz[2], z[1] * z[1] + z[2] * z[2]
            # the last target settles to the solve tolerance AND the residual
            # check: in Float32 a strongly expanding map (det J ~ 700) leaves a
            # settled iterate's residual above it, one more trip below
            settled = (isfinite(z[1]) & isfinite(z[2])) &&
                (last_target ? _newton_settled(δ2, s2, S) && _residual_ok(f(z), xd) : !(δ2 > max(s2, one(S)) / 10000))
            settled && break
        end
        if settled && _radial_ok(d, z)         # a root past a radial fold: a jumped branch, shorten
            x, Jx, t, h = z, Jz, tn, min(2 * h, hmax)
        else
            h /= 2
        end
    end
    t == 1 && _residual_ok(f(x), xd) && _on_imaged_branch(f, d, x) && return x, true
    seed, found = _radial_seed(d, cap, xd)
    found || return x, false                  # no seed but x_d itself, which the first solve had
    xs, ok, dJ = _newton_solve(f, xd, seed)
    for _ in 1:2          # in Float32 a settled iterate of a strongly expanding map can miss the residual check by a trip
        (ok | !(isfinite(xs[1]) & isfinite(xs[2]))) && break
        fx = f(xs)
        J = ForwardDiff.jacobian(w -> SVector{2}(Tuple(f(w))), xs)
        dJ = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
        Δ = J \ SVector{2, S}(fx[1] - xd[1], fx[2] - xd[2])
        xs = SVector{2, S}(xs[1] - Δ[1], xs[2] - Δ[2])
        ok = _residual_ok(f(xs), xd)
    end
    return xs, ok && dJ > 0 && _radial_ok(d, xs)
end

# The radial factor's own root, as the seed of a 2-D solve: in a Chain, the
# FIRST radial stage (a Radial, or an Additive led by one) wherever it sits, the
# stages after it undone where they have a closed form (a Tilt) and taken as the
# identity otherwise, the factor inverted on its branch (its deltas ignored),
# then the stages before it undone the same way. `(seed, found)`: not found
# where the distortion has no Even radial factor or the factor's branch does not
# reach the radius — the image is then past the branch, or the only seed left is
# x_d, which the first solve started from.
@inline _radial_seed(d, cap, xd) = (xd, false)
@inline _radial_seed(a::Additive, cap, xd) = (h = _head(a); h isa Radial{Even} ? _radial_seed(h, cap, xd) : (xd, false))
@inline _radial_seed(c::Chain, cap, xd) = _chain_seed((), c.stages, cap, xd)
@inline _chain_seed(::Tuple, ::Tuple{}, cap, xd) = (xd, false)
@inline function _chain_seed(before::Tuple, ds::Tuple, cap, xd)
    s, rest = first(ds), Base.tail(ds)
    _is_radial(s) || return _chain_seed((before..., s), rest, cap, xd)
    y, found = _radial_seed(s, cap, _undo_stages(rest, xd))
    return _undo_stages(before, y), found
end
# `cap` is the projection frame's (squared) edge. For a radial stage behind
# another stage (a tilt) it bounds that stage's input, not the frame point: it
# only decides whether a seed exists, and the seed's root still has to meet the
# pointwise test.
@inline function _radial_seed(r::Radial{Even}, cap, xd::SVector{2, S}) where {S}
    g, reached, _ = _ratio_branch_solve(r, cap, xd[1] * xd[1] + xd[2] * xd[2])
    return SVector{2, S}(g * xd[1], g * xd[2]), reached & isfinite(g)
end
@inline _undo_stages(::Tuple{}, y) = y
@inline _undo_stages(ds::Tuple, y) = _undo_stages(Base.front(ds), _undo_stage(last(ds), y))
@inline _undo_stage(t::Tilt, y::SVector{2, S}) where {S} = SVector{2, S}(Tuple(_tilt_inverse(t, y)))
@inline _undo_stage(_, y) = y

# -----------------------------------------------------------------------------
# Projections: ray ↔ frame point
# -----------------------------------------------------------------------------
#
# `_project_ray(p, X) -> (x, ok)` and `_lift_point(p, x) -> (ray, ok)`: the fixed
# projections through their frames (above), the parametric ones in closed form.

@inline function _project_ray(p::AbstractProjection, X::StaticVector{3})
    C = projection_frame(p)
    return _frame_project(C, X), _sectionable(C, X)
end
@inline function _lift_point(p::AbstractProjection, x::StaticVector{2})
    C = projection_frame(p)
    ok = _liftable(C, x)
    return (ok ? _frame_backproject(C, x) : _nan_vec3(x)), ok
end

# --- double sphere (Usenko, Demmel & Cremers 2018) ----------------------------
#
# The domain is the fold of the radial profile, in closed form. On the unit
# sphere (c = cos θ, s = sin θ) the image radius is r = s/D with
# z_s = ξ + c, d₂ = √(1 + 2ξc + ξ²), D = α·d₂ + (1 − α)·z_s, and
#   dr/dθ · D²·d₂ = (1 + ξc)·(α·z_s + (1 − α)·d₂),
# so r increases from the axis until the first of: the first sphere's fold
# 1 + ξc = 0 (only for ξ > 1), the second stage's fold α·z_s + (1 − α)·d₂ = 0
# (α > 1/2) or its pole D = 0 (α < 1/2). Both second-stage ends are
# −z_s/d₂ = w with w = min(α/(1 − α), (1 − α)/α) (the extended unified model's
# bound, at the second stage's input), and −z_s/d₂ grows with θ while
# 1 + ξc > 0. So a ray is imaged, the map one-to-one there, exactly when
#   z_s > −w·d₂   and   d₁ + ξ·z > 0,
# a test at the ray itself. The paper's bound z > −w₂·d₁ (eq. 43) is this
# condition transported to the first sphere by an approximation: it is exact
# at ξ = 0 and conservative otherwise (ξ = −0.6, α = 0.6: 95.1° against the
# fold's 105.2°).
@inline _eucm_w(α) = α <= one(α) / 2 ? α / (1 - α) : (1 - α) / α
@inline function _ds_inside(p::DoubleSphere, X, d1, zs, d2)
    return (zs > -_eucm_w(p.alpha) * d2) & (d1 + p.xi * X[3] > 0)
end

# The cosine of the view angle where the profile ends (the fold or pole above):
# the larger of the second stage's end — the root of −z_s/d₂ = w, i.e. of
# c² + 2ξ(1 − w²)c + ξ²(1 − w²) − w² = 0 with z_s ≤ 0 — and the first sphere's
# fold −1/ξ (ξ > 1); −1 when the profile reaches θ = π.
function _ds_end_cos(p::DoubleSphere)
    ξ, w = Float64(p.xi), Float64(_eucm_w(p.alpha))
    q = 1 - ξ * ξ * (1 - w * w)
    c_ab = q >= 0 ? -ξ * (1 - w * w) - w * sqrt(q) : -1.0
    (q >= 0 && ξ + c_ab <= 0) || (c_ab = -1.0)
    c_ucm = ξ > 1 ? -1 / ξ : -1.0
    return clamp(max(c_ab, c_ucm), -1.0, 1.0), c_ab >= c_ucm
end

# The projection (eq. 40-45), NaN past the fold.
@inline function _project_ray(p::DoubleSphere, X::StaticVector{3})
    x, y, z = X[1], X[2], X[3]
    d1 = sqrt(x * x + y * y + z * z)
    zs = p.xi * d1 + z
    d2 = sqrt(x * x + y * y + zs * zs)
    s = inv(p.alpha * d2 + (1 - p.alpha) * zs)
    return Vec2(s * x, s * y), _ds_inside(p, X, d1, zs, d2)
end

# The unprojection (eq. 46-51): the ray s·(m_x, m_y, m_z) − (0, 0, ξ), already
# unit, where its radicands are real (r² ≤ 1/(2α − 1) for α > 1/2) and the ray
# is inside the fold — the projection's own domain, so the two round-trip on it.
@inline function _lift_point(p::DoubleSphere, xd::StaticVector{2})
    ξ, α = p.xi, p.alpha
    mx, my = xd[1], xd[2]
    r2 = mx * mx + my * my
    disc = 1 - (2 * α - 1) * r2
    dom = (α <= one(α) / 2) | (disc >= 0)
    mz = (1 - α * α * r2) / (α * sqrt(max(disc, zero(disc))) + 1 - α)
    a = mz * mz + (1 - ξ * ξ) * r2                         # negative only for ξ > 1, outside the domain
    s = (mz * ξ + sqrt(max(a, zero(a)))) / (mz * mz + r2)
    X = Vec3(s * mx, s * my, s * mz - ξ)
    d1 = sqrt(X[1] * X[1] + X[2] * X[2] + X[3] * X[3])
    zs = ξ * d1 + X[3]
    ok = dom & (a >= 0) & _ds_inside(p, X, d1, zs, sqrt(X[1] * X[1] + X[2] * X[2] + zs * zs))
    return X, ok
end

# --- unified (Mei & Rives 2007) -----------------------------------------------
# x = X₁₂/(z + ξ‖X‖), valid for z > −w‖X‖ with w = ξ (ξ ≤ 1) or 1/ξ (ξ > 1).
@inline _ucm_w(ξ) = ξ <= 1 ? ξ : inv(ξ)
@inline function _project_ray(p::Unified, X::StaticVector{3})
    d = sqrt(X[1] * X[1] + X[2] * X[2] + X[3] * X[3])
    s = inv(X[3] + p.xi * d)
    return Vec2(s * X[1], s * X[2]), X[3] > -_ucm_w(p.xi) * d
end
@inline function _lift_point(p::Unified, xd::StaticVector{2})
    ξ = p.xi
    r2 = xd[1] * xd[1] + xd[2] * xd[2]
    disc = 1 + (1 - ξ * ξ) * r2
    f = (ξ + sqrt(max(disc, zero(disc)))) / (1 + r2)
    z = f - ξ
    return Vec3(f * xd[1], f * xd[2], z), (disc >= 0) & (z > -_ucm_w(ξ))
end

# --- extended unified (Khomutenko, Garcia & Martinet 2016) ---------------------
# d = √(β r² + z²), x = X₁₂/(α d + (1 − α) z); valid for z > −w·d with
# w = α/(1 − α) (α ≤ 1/2) or (1 − α)/α (α > 1/2).
@inline function _project_ray(p::ExtendedUnified, X::StaticVector{3})
    α, β = p.alpha, p.beta
    d = sqrt(β * (X[1] * X[1] + X[2] * X[2]) + X[3] * X[3])
    s = inv(α * d + (1 - α) * X[3])
    return Vec2(s * X[1], s * X[2]), X[3] > -_eucm_w(α) * d
end
@inline function _lift_point(p::ExtendedUnified, xd::StaticVector{2})
    α, β = p.alpha, p.beta
    r2 = xd[1] * xd[1] + xd[2] * xd[2]
    disc = 1 - (2 * α - 1) * β * r2
    mz = (1 - α * α * β * r2) / (α * sqrt(max(disc, zero(disc))) + 1 - α)
    ray = normalize(Vec3(xd[1], xd[2], mz))
    # the ray must be on the projection's valid side to be this point's preimage
    zok = ray[3] > -_eucm_w(α) * sqrt(β * (ray[1] * ray[1] + ray[2] * ray[2]) + ray[3] * ray[3])
    return ray, ((α <= one(α) / 2) | (disc >= 0)) & zok
end

# -----------------------------------------------------------------------------
# The contract for a Lens: project / backproject
# -----------------------------------------------------------------------------

"""
    project(lens, X::StaticVector{3}) -> Vec2

The NORMALIZED image point of the ray through `X` (any positive scale) — half
of the lens contract, with [`backproject`](@ref)`(lens, xd)`. NaN where the
lens does not image the ray, never a throw; allocation-free and generic in the
number type (Float32 kernels, ForwardDiff Duals).

A [`Lens`](@ref) projects the ray onto its projection's frame and distorts the
frame point there; no perspective hop is involved unless the projection IS the
perspective one, whose domain genuinely ends at 90°. A [`LiftLens`](@ref)
solves the image radius as a root on the ray.

A ray past the fold (or pole) of a radial factor is NaN: every [`Radial`](@ref)
factor is tested at its own input — a lone one, an [`Additive`](@ref)'s head,
each radial stage of a [`Chain`](@ref). The other folds are NOT tested
forward, since that would cost a Jacobian per projected point: where the
deltas of an `Additive` (tangential, thin prism) or a `Chain`'s
[`Tilt`](@ref) fold the map (det J ≤ 0 inside the radial factor's disk),
`project` returns a finite point, and [`backproject`](@ref) of that point is
not the ray — it is the imaged branch's preimage, a different ray, or NaN
where the branch does not reach the point. So the round trip
`backproject(lens, project(lens, X)) ≈ X / ‖X‖` holds on the imaged branch
(the region connected to the axis where det J > 0 and every radial factor is
before its fold), not past such a fold.
"""
@inline function project(l::Lens, X::StaticVector{3})
    x, ok = _project_ray(l.projection, X)
    ok || return _nan_vec2(one(float(promote_type(eltype(X), _coeff_type(l)))) * X[1])
    return Vec2(_distort_in_frame(l, x))
end

"""
    backproject(lens, xd::StaticVector{2}) -> Vec3

The UNIT ray behind the normalized image point `xd` — the other half of the
lens contract, with [`project`](@ref)`(lens, X)`. NaN where the lens does not
image the point (outside its frame's domain, past a fold, a non-converged
inverse), never a throw.

Where the in-plane distortion folds, a point has several preimages, and the
answer is the one on the imaged branch: det J > 0 there and every
[`Radial`](@ref) factor before its fold, reached from the axis. Two
consequences:
- [`project`](@ref) tests only the radial factors, so a ray past a fold of
  the deltas of an [`Additive`](@ref) or of a [`Chain`](@ref)'s
  [`Tilt`](@ref) projects to a finite point whose `backproject` is the
  imaged branch's preimage (a different ray), or NaN — never that ray.
- The acceptance test is pointwise. Where the map overlaps itself INSIDE the
  axis-connected component, a point has several preimages on the branch and
  `backproject` returns the first the solver reaches — rare (0.05 % of the
  interior points of deliberately strong OpenCV chains), but an undistortion
  map built from `backproject` may jump between preimages from one pixel to
  the next there. Whether a lens is injective over a sensor is a property of
  the LENS, for an optional check over the sensor rectangle, not something
  tested per pixel.
"""
@inline function backproject(l::Lens, xd::StaticVector{2})
    x = _undistort_in_frame(l, xd)
    ray, ok = _lift_point(l.projection, x)
    return _nan_unless(ok, Vec3(ray))
end

# -----------------------------------------------------------------------------
# Scalar solves shared by the maps, and the radial profile read off them
# -----------------------------------------------------------------------------

# Safeguarded Newton for a monotone increasing scalar profile g on [0, R):
# solve g(r) = target in exactly N trips. A Newton step that leaves the
# current bracket is replaced by its midpoint (the classic `rtsafe`), so the
# iteration cannot escape the monotone domain. Unbounded profiles (R = Inf)
# get an expanding upper bracket. Kernel-lawful: fixed trips, no throws.
#
# The upper bracket is pulled STRICTLY inside the domain: R is typically the
# pole of the profile (the division model's 90° circle — g → +∞ there and
# flips to −∞ a rounding error past it), and a bracket end sitting on the
# pole is fatal — the seed lands on it, g is non-finite, and an iterate a
# hair beyond reads as "below target", collapsing the bracket onto R
# (observed 2026-08-31: every ρ_u ≥ r_∞ returned r_∞, i.e. the rim, so the
# rectified image was a truncated disk). Every evaluation now happens at
# r ≤ R·(1 − 1e-9), where g is finite and increasing. Targets beyond
# g(R·(1−1e-9)) — stretches past ~10⁹ — converge to the bracket end with
# `converged = false`.
@inline function _radial_solve(g, dg, target::T, R::T, ::Val{N}) where {T <: AbstractFloat, N}
    lo = zero(T)
    hi = ifelse(isfinite(R), R * (one(T) - T(1.0e-9)), max(2 * target, one(T)))
    if !isfinite(R)
        for _ in 1:8
            hi = ifelse(g(hi) < target, 2 * hi, hi)
        end
    end
    # Warm-up: a few bisections first. Newton seeded at a bracket end next to a
    # pole crawls away from it (one doubling of the gap per trip); a narrow
    # bracket makes the Newton phase quadratic from any profile.
    for _ in 1:6
        m = (lo + hi) / 2
        fm = g(m) - target
        hi = ifelse(!(fm < zero(T)), m, hi)
        lo = ifelse(!(fm < zero(T)), lo, m)
    end
    r = (lo + hi) / 2
    for _ in 1:N
        f = g(r) - target
        above = !(f < zero(T))                 # non-finite g counts as above the target
        rn = r - f / dg(r)
        hi = ifelse(above, r, hi)
        lo = ifelse(above, lo, r)
        # NON-strict: after the bracket update one end IS r, and a converged
        # Newton iterate has rn == r. A strict test would call that "outside"
        # and replace the converged solution by the midpoint (measured
        # 2026-08-31: N=2 converged, N=4 returned r/2).
        inside = (rn >= lo) & (rn <= hi) & isfinite(rn)
        r = ifelse(inside, rn, (lo + hi) / 2)
    end
    conv = abs(g(r) - target) <= _core_tol(T) * max(one(T), abs(target))
    return r, conv
end


# -----------------------------------------------------------------------------
# Folds: where a radial profile stops increasing
# -----------------------------------------------------------------------------
#
# A radial profile folds where its derivative first vanishes (a fold singularity):
# past that point two rays share an image point and larger radii are no ray's
# image, so the lens images only the branch before it. For the polynomial lenses
# the derivative is, up to a positive factor, a polynomial `c` in the squared
# radius or angle with `c(0) = 1` — the rational lift's `_fold_coeffs`,
# Kannala–Brandt's `1 + 3k₁θ² + 5k₂θ⁴ + …` — so the branch is the prefix of
# `[0, ∞)` on which `c` stays positive and the fold is its end. Found EXACTLY, not
# from the local sign at the query: a point past a second fold, where `c` is
# positive again, is still off the branch. Coefficients are ascending.

# c(a + u) as coefficients in u, by repeated synthetic division (coefficient k is
# c⁽ᵏ⁾(a)/k!).
@inline function _taylor_shift(c::SVector{N, T}, a::T) where {N, T}
    b = reverse(c)
    for i in 1:(N - 1), j in 2:(N - i + 1)
        b = Base.setindex(b, b[j] + a * b[j - 1], j)
    end
    return reverse(b)
end

const _FOLD_TRIPS = 256

# The end of the prefix of [0, hi] (hi finite) on which `c` stays positive: `hi`
# when `c` is positive throughout, otherwise its first root to ~1e-12 relative.
# Sweeps right from 0 in steps certified positive by the Taylor lower bound
# c(a + u) ≥ c(a) + Σₖ min(cₖ(a), 0)·hᵏ on [0, h], halving a step it cannot
# certify. The first step covers all of [0, hi] whenever the negative
# coefficients cannot outweigh c(0) there — the common case, one bound.
#
# The stall tolerance is relative to the sweep's POSITION `a`, not to `hi`: `hi`
# is typically Cauchy's bound 1 + max|cᵢ/c_d|, which a rounding-artefact leading
# coefficient inflates to ~1e16. A tolerance of 1e-12·hi then exceeds the first
# step the lower bound can certify from 0 (~|c_d|^(-1/d)), and the sweep stalled
# AT the origin — a fold at 0, so a Scaramuzza fitted onto a division source
# (a₄ ~ 1e-17) reported `view_angle_limit` 0 and `fit_lens` threw. At `a = 0` the
# sweep halves until it certifies, which it must: c(0) = 1.
@inline function _positive_prefix(c::SVector{N, T}, hi::Real) where {N, T}
    top = T(hi)
    top >= 0 || return top                       # NaN in, NaN out: no sweep inside a kernel
    _taylor_lower_bound(c, top) > 0 && return top
    return _positive_prefix_sweep(c, top)
end

# c(a) + Σₖ min(cₖ(a), 0)·hᵏ, from the Taylor coefficients `b` of c at a: a lower
# bound of c on [a, a + h].
@inline function _taylor_lower_bound(b::SVector{N, T}, h::T) where {N, T}
    lb = b[1]
    hp = one(T)
    for k in 2:N
        hp *= h
        lb += min(b[k], zero(T)) * hp
    end
    return lb
end

@noinline function _positive_prefix_sweep(c::SVector{N, T}, top::T) where {N, T}
    a = zero(T)
    h = top / 2                                  # the whole interval did not certify
    for _ in 1:_FOLD_TRIPS
        a >= top && return top
        b = iszero(a) ? c : _taylor_shift(c, a)
        b[1] > 0 || return a
        h = min(h, top - a)
        if _taylor_lower_bound(b, h) > 0
            a += h
            h *= 2
        else
            h /= 2
            h <= T(1.0e-12) * a && return a      # stalled at a root: see the tolerance note above
        end
    end
    return a
end

# Past this point `c` has no positive root (Cauchy's bound), or Inf when every
# coefficient is non-negative and none can exist.
@inline function _positive_root_bound(c::SVector{N, T}) where {N, T}
    all(ci -> ci >= 0, c) && return T(Inf)
    d = N
    while d > 1 && iszero(c[d])
        d -= 1
    end
    m = zero(T)
    for i in 1:(d - 1)
        m = max(m, abs(c[i] / c[d]))
    end
    return one(T) + m
end

# The fold of the profile whose derivative is `c`, capped at `hi`: `c`'s first
# positive root, or `hi` when there is none below it.
@inline function _fold_point(c::SVector{N, T}, hi::Real) where {N, T}
    top = min(T(hi), _positive_root_bound(c))
    isfinite(top) || return T(hi)
    p = _positive_prefix(c, top)
    return p >= top ? T(hi) : p
end



"""
    projected_radius(lens, θ) -> ρ_d

A lens's radial projection function, read off the contract: the image radius
of the ray at view angle `θ` (the classical r(θ) — equidistant r = θ,
stereographic 2·tan(θ/2), Kannala–Brandt's θ-polynomial, …), i.e.
[`project`](@ref) at azimuth 0, so it holds past 90° wherever the lens does.
NaN where the lens does not image; `θ ≥ π` — the exact backward ray, a circle
rather than a point — never is. Its inverse is [`view_angle`](@ref).
"""
function projected_radius(lens::AbstractLensModel, θ::Real)
    θf = float(θ)
    θf < 0 && throw(ArgumentError("projected_radius: the view angle must be non-negative"))
    θf >= π && return typeof(θf)(NaN)
    u = project(lens, SVector(sin(θf), zero(θf), cos(θf)))
    return _radius(u)
end

"""
    view_angle(lens, ρ_d) -> θ

The view angle of the ray behind the image point at radius `ρ_d` — the inverse
of [`projected_radius`](@ref), read off [`backproject`](@ref) at azimuth 0. NaN
where the lens does not image.
"""
function view_angle(lens::AbstractLensModel, ρ::Real)
    ρf = float(ρ)
    b = backproject(lens, SVector(ρf, zero(ρf)))
    return atan(_radius(b), b[3])
end

# |(p₁, p₂)| off the azimuth-0 axis; on it (p₂ = 0: every radial lens, and the
# optical axis of any lens) |p₁|, with the same first derivative and none of
# hypot's 0/0 at the origin — so a Dual radius stays finite on the axis.
@inline _radius(p) = p[2] == 0 ? abs(p[1]) : hypot(p[1], p[2])

# -----------------------------------------------------------------------------
# LiftLens: the lift (x_d·N, D) is a ray
# -----------------------------------------------------------------------------
#
# The back-projection is the lift itself; the projection solves the image radius
# ρ as the root on the ray of
#
#     h(ρ) = ρ·N(u)·z − r·D(u),      u = ρ² (Even) or ρ (Full),
#
# the 2-D cross product of the lift (D, ρN) with the ray (z, r) — so it changes
# sign exactly where the lift's view angle equals the ray's, at ANY θ: nothing
# divides by z, and 90° is an ordinary angle. On the axis the Even lift is smooth
# in r², so a Dual rides one Newton step in the ratio m = ρ/r; the Full lift is
# not (its u = ρ), and on the axis m = 1/z already carries the exact first
# derivatives.

"""
    backproject(lens::LiftLens, xd) -> Vec3

The lift `(x_d·N, D)`, normalized: `D = 0` is the 90° ray and `D < 0` a ray
behind it; only the Euclidean point `x_d·N/D` has a pole there, not the
direction. Past the lift's first fold (where its view angle first stops
increasing) the point is not an image of the lens and the ray is NaN — found
exactly, so a point past a second fold is NaN too: the same branch
[`project`](@ref) solves on.
"""
@inline function backproject(l::LiftLens, xd::StaticVector{2})
    r = l.radial
    s = xd[1] * xd[1] + xd[2] * xd[2]
    # the axis of a Full lift: perspective to first order (no linear term), so the
    # ray and its first derivatives are the perspective lift's — and no square
    # root of zero is differentiated
    r isa Radial{Full} && iszero(_primal(s)) && return normalize(Vec3(xd[1], xd[2], one(xd[1])))
    u = first(_basis(r, one(s), s))
    N = _num(r, u)
    on_branch = _primal(u) <= _primal(r.u_fold)
    return _nan_unless(on_branch, Vec3(normalize(SVector(xd[1] * N, xd[2] * N, _den(r, u)))))
end

# --- the default inverse: a lens that defines only `project` ------------------
#
# Hoisted from RadiometricCamera's `NewtonInverse` (src/camera/geometric_camera.jl):
# Newton on the equidistant chart m = θ·(cos φ, sin φ) of the ray, a fixed trip
# count, the 2 × 2 Jacobian by ForwardDiff of the forward map, a vanishing
# determinant replaced by ±eps. The chart reaches every ray but the antipode,
# so a lens imaging past 90° is inverted there too. Two changes, measured on
# 900 rays of twelve lens families (up to 98% of each lens's view-angle limit,
# 150° at most): RC's seed m = x_d (the image point read as an equidistant
# chart point) and unlimited steps failed 162 of them — a perspective point
# at 60° already reads as θ > π — while the pinhole ray's chart point
# atan(|x_d|)·x̂_d and steps capped at 0.3 rad over 30 trips fail 12, each
# within 2% of a fold.
const _DEFAULT_INVERSE_TRIPS = 30
const _DEFAULT_INVERSE_MAX_DELTA = 0.3

"""
    backproject(lens::AbstractLensModel, xd::StaticVector{2}) -> Vec3

The default [`backproject`](@ref) of a lens that defines only
[`project`](@ref): Newton's method on `project(lens, X(m)) = x_d` over the
equidistant chart `m = θ·(cos φ, sin φ)` of the ray `X(m) = (sin θ·m̂, cos θ)`
— so view angles past 90° are reached — started at the pinhole ray's chart
point `atan(|x_d|)·x̂_d`, each step capped at 0.3 rad, a fixed 30 trips
(kernel-safe: no throw, no allocation), the 2 × 2 Jacobian by forward-mode
differentiation of `project`. The unit ray, or NaN unless the iterate
converged (the residual within the solve tolerance of its precision), the
forward map keeps orientation there (`det J > 0`) and `θ < π`. Generic in
the number type: Float32 and ForwardDiff Duals.

Every lens of this package has its own inverse (closed form, or a solver that
follows the imaged branch), which takes precedence; this one is for a
downstream lens type, and agrees with each of them in its domain (tested to
95% of each lens's [`view_angle_limit`](@ref); within a few percent of a fold
Newton converges too slowly and the answer is NaN).
"""
@inline function backproject(l::AbstractLensModel, xd::StaticVector{2})
    F = float(eltype(xd))
    p = SVector{2, F}(xd[1], xd[2])
    resid(m) = project(l, _frame_backproject(EquidistantFrame, m)) - p
    ρ = sqrt(p[1] * p[1] + p[2] * p[2])
    m = _primal(ρ) > 0 ? p * (atan(ρ) / ρ) : p
    J = SMatrix{2, 2}(one(m[1]), zero(m[1]), zero(m[1]), one(m[1]))
    cap = oftype(_primal(ρ), _DEFAULT_INVERSE_MAX_DELTA)
    for _ in 1:_DEFAULT_INVERSE_TRIPS
        r = resid(m)
        J = ForwardDiff.jacobian(resid, m)
        dt = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
        dt = abs(dt) > eps(typeof(dt)) ? dt : (dt >= 0 ? eps(typeof(dt)) : -eps(typeof(dt)))
        Δ = SVector(J[2, 2] * r[1] - J[1, 2] * r[2], -J[2, 1] * r[1] + J[1, 1] * r[2]) / dt
        n = sqrt(_primal(Δ[1] * Δ[1] + Δ[2] * Δ[2]))
        m = n > cap ? m - Δ * (cap / n) : m - Δ
    end
    S = typeof(_primal(p[1]))
    θ2 = _primal(m[1] * m[1] + m[2] * m[2])
    fm = map(_primal, project(l, _frame_backproject(EquidistantFrame, m)))
    ok = isfinite(θ2) && _residual_ok(fm, SVector{2, S}(_primal(p[1]), _primal(p[2]))) &&
        _primal(J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) > 0 && θ2 < S(π)^2
    return _nan_unless(ok, Vec3(_frame_backproject(EquidistantFrame, m)))
end

"""
    project(lens::LiftLens, X) -> Vec2

The image point of a ray under the lift, solved ON THE RAY: with
`(r, z) = (sin θ, cos θ)` the image radius ρ is the root of
`h(ρ) = ρ·N·z − r·D` on the lift's monotone branch. NaN past the fold, where
the lens images nothing.
"""
@inline function project(l::LiftLens, X::StaticVector{3})
    x, y, z = normalize(X)
    return _lift_ratio(l.radial, x, y, z)
end

# The image point m·(x, y) of the unit ray (x, y, z) under the lift `r`: the ratio
# m = ρ/r with h(ρ) = ρ·N(u)·z − r·D(u) = 0 on the lift's branch (before its
# fold). Newton on primal numbers in ρ from the one-coefficient closed form
# 2r/(z + √(z² − 4k·r²)), k = d₁ − k₁ (the root itself for a one-coefficient lift,
# which therefore stops after one step), to the solve tolerance, every trip
# certified by h′ > 0 and the root by the exact branch test; on failure a
# bracketed solve (`_radius_bracketed`). A Dual rides one Newton step in m at the
# root — except on the axis of a Full lift, not smooth there, where m = 1/z
# already carries the exact derivatives.
@inline function _lift_ratio(r::Radial, x, y, z)
    r2 = x * x + y * y
    rp = _primal(r)
    T = promote_type(typeof(r2), typeof(z), _coeff_type(r))
    S = float(_primal_float(T))
    r2p, zp = S(_primal(r2)), S(_primal(z))
    if iszero(r2p)
        m, ok = (zp > 0 ? inv(zp) : S(NaN)), zp > 0
    else
        rr = sqrt(r2p)
        k1 = S(get(rp.d, 1, 0)) - S(get(rp.k, 1, 0))
        disc = zp * zp - 4 * k1 * rr * rr
        root = sqrt(max(disc, zero(S)))
        ρ0 = ((disc >= 0) & (zp + root > 0)) ? 2 * rr / (zp + root) : rr
        ρ, ok = _lift_newton(rp, ρ0, rr, zp)
        ok || ((ρ, ok) = _radius_bracketed(rp, rr, zp, ρ0))
        m = ρ / rr
    end
    if _is_dual_type(T)
        δm = _h(r, m, r2, one(m), z) / _dh(rp, m, r2p, one(m), zp)
        m = (r isa Radial{Even} || !iszero(r2p)) ? m - δm : m + zero(δm)
    end
    return _nan_unless(ok, Vec2(m * x, m * y))
end

# h = v·N(u)·z − c·D(u) and dh/dv, at v·√a (`_basis`): in ρ (a = 1, c = r) and in
# the ratio m (a = r², c = 1) — the 2-D cross product of the lift (D, ρN) with the
# ray (z, r), which changes sign exactly where their angles agree, at ANY θ.
@inline _h(r::Radial, v, a, c, z) = (u = first(_basis(r, v, a)); v * _num(r, u) * z - c * _den(r, u))
@inline function _dh(r::Radial, v, a, c, z)
    u, du = _basis(r, v, a)
    return _num(r, u) * z + (du * v) * _dnum(r, u) * z - (du * c) * _dden(r, u)
end
@inline function _lift_newton(r::Radial, ρ::S, rr::S, z::S) where {S <: AbstractFloat}
    for _ in 1:8
        dh = _dh(r, ρ, one(S), rr, z)
        ((dh > 0) & isfinite(dh)) || break
        Δρ = _h(r, ρ, one(S), rr, z) / dh
        ρ -= Δρ
        ((ρ >= 0) & isfinite(ρ)) || break
        _newton_settled(Δρ * Δρ, ρ * ρ, S) && return ρ, first(_basis(r, ρ, one(S))) <= r.u_fold
    end
    return ρ, false
end

# The upper end of a root bracket [0, hi) for h(ρ) = 0: the TIGHTER of two
# independent bounds. The branch's end (in ρ; non-finite when it never ends)
# bounds the branch; doubling from the seed ρ0 until h ≥ 0 localizes the root.
# The end alone breaks when the leading coefficient is a rounding artefact —
# fitting Division{3} to a degree-2 source lands it at ~1e-17, the fold sits at
# 1/sqrt of it (ρ ~ 5e7 against a root at 2.3), and `_radial_solve`'s fixed
# budget cannot localize that. No tolerance decides when a coefficient is
# negligible: the doubling bound is tight by construction, so the min is
# on-branch AND localized. Kernel-lawful: bounded trips, no throws.
@inline function _lift_bracket(h, ρ0::S, fold::S) where {S <: AbstractFloat}
    cap = ifelse(isfinite(fold), fold, S(Inf))
    hi = ifelse(isfinite(ρ0) & (ρ0 > 0), ρ0, one(S))
    for _ in 1:64
        (hi < cap && h(hi) < 0) || break
        hi = 2 * hi
    end
    return min(cap, hi)
end

@noinline function _radius_bracketed(r::Radial, rr::S, z::S, ρ0::S) where {S <: AbstractFloat}
    h = ρ -> _h(r, ρ, one(S), rr, z)
    hi = _lift_bracket(h, ρ0, S(_radius_at(r, r.u_fold)))
    h(hi) < 0 && return S(NaN), false
    return _radial_solve(h, ρ -> _dh(r, ρ, one(S), rr, z), zero(S), hi, Val(16))
end


# -----------------------------------------------------------------------------
# Derived for every lens: the Jacobians of `project`
# -----------------------------------------------------------------------------

# Value and Jacobian of a static map in ONE ForwardDiff evaluation. The tag is the
# map's own, as `ForwardDiff.jacobian` makes it, so nesting stays sound.
@inline function _value_and_jacobian(f::F, x::SVector{N, T}) where {F, N, T}
    tag = typeof(ForwardDiff.Tag(f, T))
    duals = SVector{N, ForwardDiff.Dual{tag, T, N}}(
        ntuple(i -> ForwardDiff.Dual{tag}(x[i], ForwardDiff.Partials{N, T}(ntuple(j -> T(i == j), Val(N)))), Val(N))
    )
    y = f(duals)
    M = length(y)
    value = SVector{M, T}(ntuple(i -> ForwardDiff.value(tag, y[i]), Val(M)))
    J = SMatrix{M, N, T}(ntuple(k -> ForwardDiff.partials(tag, y[(k - 1) % M + 1], (k - 1) ÷ M + 1), Val(M * N)))
    return value, J
end

# The Jacobian of a map to 2-vectors at `x`; NaN beside a NaN image point, never finite.
@inline function _jacobian(f::F, x::StaticVector) where {F}
    u, J = _value_and_jacobian(z -> SVector{2}(Tuple(f(z))), _float_static(x))
    return all(isfinite, u) ? J : J * eltype(J)(NaN)
end
# The same in the precision of the query `x` — the 2-D views and pointwise maps.
@inline _pointwise_jacobian(f::F, x::StaticVector, at = x) where {F} = _in_precision(float(eltype(x)), _jacobian(f, at))
@inline _in_precision(::Type{T}, J::SMatrix{M, N}) where {T, M, N} = SMatrix{M, N, T}(J)

"""
    transform_deriv(lens, X::StaticVector{3}) -> SMatrix{2,3}

`∂project/∂X`: ForwardDiff through [`project`](@ref)`(lens, ·)` itself, for
every lens — the derivative of exactly the map the residual evaluates, at every
angle. NaN wherever `project` is.
"""
@inline transform_deriv(lens::AbstractLensModel, X::StaticVector{3}) = _jacobian(v -> project(lens, v), X)

"""
    transform_deriv_params(lens, X::StaticVector{3}) -> SMatrix{2,N}

`∂project/∂(lens parameters)` in [`parameter_vector`](@ref) order: ForwardDiff
through `project(with_parameters(lens, p), X)`, for every lens. Iterative
inverses carry their derivatives by one Newton step at the root, so no Dual
runs through a solve. NaN wherever `project` is.
"""
@inline transform_deriv_params(lens::AbstractLensModel, X::StaticVector{3}) =
    (Xf = _float_static(X); _jacobian(q -> project(_with_own_parameters(lens, q), Xf), parameter_vector(lens)))

"""
    project_with_jacobians(lens, X::StaticVector{3}) -> (u, ∂u/∂X, ∂u/∂p)

The image point and both Jacobians — `transform_deriv(lens, X)` and
`transform_deriv_params(lens, X)` — in ONE ForwardDiff evaluation over `(X, p)`: the
entry point for a bundle's or a PnP solver's per-observation residual.

This is GTSAM's `project(point, Dpose, Dpoint, Dcal)`, with the Jacobians
RETURNED rather than written into optional output arguments; the Ceres
analogue is what a `CostFunction::Evaluate` fills.
"""
@inline project_with_jacobians(lens::AbstractLensModel, X::StaticVector{3}) =
    _project_with_jacobians(lens, _float_static(X), _float_static(parameter_vector(lens)))

@inline function _project_with_jacobians(lens, X::SVector{3}, p::SVector{N}) where {N}
    w0 = vcat(X, p)
    T = eltype(w0)
    u, J = _value_and_jacobian(
        w -> SVector{2}(
            Tuple(project(_with_own_parameters(lens, SVector{N}(ntuple(i -> w[3 + i], Val(N)))), SVector{3}(w[1], w[2], w[3])))
        ),
        w0
    )
    JX = SMatrix{2, 3, T, 6}(J[1, 1], J[2, 1], J[1, 2], J[2, 2], J[1, 3], J[2, 3])
    Jp = SMatrix{2, N, T}(ntuple(k -> J[(k - 1) % 2 + 1, 3 + (k - 1) ÷ 2 + 1], Val(2N)))
    nan = all(isfinite, u) ? one(T) : T(NaN)
    return Vec2(u), JX * nan, Jp * nan
end

# -----------------------------------------------------------------------------
# Derived: the perspective views of a lens
# -----------------------------------------------------------------------------
#
# What a pinhole-shaped consumer needs — a warp to a perspective image, the
# `lens(x)` functor, its `InverseLens`, the 2-D Jacobians of both — are VIEWS of
# the contract through the perspective plane (X/Z, Y/Z), computed from `project`
# and `backproject` and defined nowhere else. They exist only where that plane
# does (Z > 0): a property of a perspective OUTPUT, not of the lens.

"""
    distort_normalized(lens, x::StaticVector{2}) -> Vec2

The image point of the ray through the perspective point `x`:
`project(lens, (x₁, x₂, 1))`. The functor `lens(x)` is this call.
"""
@inline distort_normalized(lens::AbstractLensModel, x::StaticVector{2}) =
    project(lens, SVector(x[1], x[2], one(x[1])))

"""
    undistort_normalized(lens, xd::StaticVector{2}) -> Vec2

The perspective point of the ray behind the image point `xd` — the view of
[`backproject`](@ref)`(lens, xd)` on the plane Z = 1. NaN where the lens does not
image `xd`, and where its ray has Z ≤ 0 (no perspective point exists).
"""
@inline function undistort_normalized(lens::AbstractLensModel, xd::StaticVector{2})
    b = backproject(lens, xd)
    b[3] > 0 || return _nan_vec2(b)
    return Vec2(PerspectiveMap()(b))
end

# The 2-D Jacobians — of `distort_normalized` (the functor), of
# `undistort_normalized` (the `InverseLens`), of any pointwise 2-D map built on a
# lens (a framed distortion stage), and of `distort_normalized` in the lens
# parameters — ForwardDiff through the map, in the precision of the query.
@inline transform_deriv(lens::AbstractLensModel, x::StaticVector{2}) =
    _pointwise_jacobian(z -> distort_normalized(lens, z), x)
@inline transform_deriv(m::InverseLens, xd::StaticVector{2}) =
    _pointwise_jacobian(z -> undistort_normalized(m.lens, z), xd)

@inline transform_deriv_params(lens::AbstractLensModel, x::StaticVector{2}) =
    (xf = _float_static(x); _pointwise_jacobian(q -> distort_normalized(_with_own_parameters(lens, q), xf), x, parameter_vector(lens)))
# -----------------------------------------------------------------------------
# A lens over a batch of points
# -----------------------------------------------------------------------------
#
# The per-point maps ARE the batch kernels: `pointwise!` broadcasts them — a
# SIMD loop on host arrays (split across threads per the execution policy), one
# compiled kernel on a GPU array. A lens is converted once to the output's
# number type, so per-point math never promotes (Metal has no Float64).

_batch_precision(out, m) = m isa AbstractLensModel ? _precision(eltype(eltype(out)), m) : m

"""
    project!(u, m, X; exec = Auto()) -> u
    backproject!(X, lens, u; exec = Auto()) -> X
    distort_normalized!(xd, lens, x; exec = Auto()) -> xd
    undistort_normalized!(x, lens, xd; exec = Auto()) -> x

[`project`](@ref) / [`backproject`](@ref) / [`distort_normalized`](@ref) /
[`undistort_normalized`](@ref) of every point of a batch, into `u` / `X` /
… — arrays of static vectors on any backend. `m` is a lens or anything else
[`project`](@ref) takes (a camera). A point the lens does not image comes back
NaN. `exec` chooses how a host batch is split across threads
([`pointwise!`](@ref)); every policy gives identical results.
"""
project!(u::AbstractArray, m, X::AbstractArray; exec::ExecutionPolicy = Auto()) =
    pointwise!(Base.Fix1(project, _batch_precision(u, m)), u, X; exec)
"See [`project!`](@ref)."
backproject!(X::AbstractArray, l::AbstractLensModel, u::AbstractArray; exec::ExecutionPolicy = Auto()) =
    pointwise!(Base.Fix1(backproject, _batch_precision(X, l)), X, u; exec)
"See [`project!`](@ref)."
distort_normalized!(xd::AbstractArray, l::AbstractLensModel, x::AbstractArray; exec::ExecutionPolicy = Auto()) =
    pointwise!(Base.Fix1(distort_normalized, _batch_precision(xd, l)), xd, x; exec)
"See [`project!`](@ref)."
undistort_normalized!(x::AbstractArray, l::AbstractLensModel, xd::AbstractArray; exec::ExecutionPolicy = Auto()) =
    pointwise!(Base.Fix1(undistort_normalized, _batch_precision(x, l)), x, xd; exec)

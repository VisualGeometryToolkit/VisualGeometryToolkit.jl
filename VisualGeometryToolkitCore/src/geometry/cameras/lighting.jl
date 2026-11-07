# =============================================================================
# Lighting — the scene's light as a radiometric camera model sees it
# =============================================================================
#
# An environment light: radiance from every direction, given on a sphere of radius `r` about a centre `c` and looked up where a
# ray from the shaded point leaves that sphere (parallax-corrected, as a game engine's sphere-projected probe):
#
#   L(x, ω) = M(ω̂(x, ω)),   ω̂ = (x + μω − c) / ‖x + μω − c‖,   μ > 0 the root of ‖x + μω − c‖² = r²
#
# As r → ∞ it tends to pbrt's infinite (distant) light, ω̂ = ω. The directional map M is a real spherical-harmonic
# expansion per channel, M_c(ω) = Σ_{l ≤ L} Σ_m a_{c,lm} y_l^m(ω), so the radiance is LINEAR in the coefficients:
# `coefficients(light) * coefficient_jacobian(light, x, ω) == radiance(light, x, ω)`.
#
# The basis (`sh_basis`) is real, orthonormal on the sphere, WITHOUT the Condon–Shortley phase:
#
#   y_l^m = √2 K_l^m cos(mφ) P_l^m(cos θ)  (m > 0),   √2 K_l^m sin(|m|φ) P_l^|m|(cos θ)  (m < 0),   K_l^0 P_l^0(cos θ)  (m = 0)
#   K_l^m = √((2l + 1)/(4π) · (l − |m|)!/(l + |m|)!),   index j = l² + l + m + 1 (`sh_index`)
#
# (θ, φ) are the polar angle from +z and the azimuth from +x in the light's frame. A library with the phase (−1)^m differs by
# that sign for odd m > 0 and m < 0. The basis is generated per degree: unrolled, its normalizations compile-time constants
# (no factorial or table can reach device code), an isbits SVector, and Dual-safe at the poles.
#
# Units: the centre and radius are lengths (metres as quantities in a public value, plain numbers in a kernel:
# `ustrip(m, light)`, `with_unit(light, m)`); the coefficients are in the caller's radiance units. The point x and direction ω
# are in the light's frame; the light's owner states which (a world or rig frame).
# Moved from RadiometricCamera (its `Lighting`, `y_basis`, `ω̂`, `L_SVSH`), 2026-11-05.
# =============================================================================

"""
    sh_count(l_max) -> Int

The number of real spherical harmonics of degree `l ≤ l_max`, `(l_max + 1)²`.
"""
sh_count(l_max::Integer) = (l_max + 1)^2

"""
    sh_index(l, m) -> Int

The position of `y_l^m` in [`sh_basis`](@ref): `l² + l + m + 1`, `|m| ≤ l`.
"""
sh_index(l::Integer, m::Integer) = l^2 + l + m + 1

# K_l^m, the orthonormality constant (only the generator and the constant map call it; Int factorials: l + |m| ≤ 20)
_sh_normalization(l::Integer, m::Integer) = sqrt((2l + 1) / (4π) * factorial(l - abs(m)) / factorial(l + abs(m)))

# P_l^m(z), m ≥ 0, without the Condon–Shortley phase, by the standard upward recurrences (in m, then in l)
@inline function _legendre(l::Integer, m::Integer, z)
    s² = max(1 - z^2, zero(z))
    pmm = one(z)
    if m > 0
        s = sqrt(s²)
        odd = 1
        for _ in 1:m
            pmm *= odd * s
            odd += 2
        end
    end
    l == m && return pmm
    pmm1 = z * (2m + 1) * pmm
    l == m + 1 && return pmm1
    pll = pmm1
    for ll in (m + 2):l
        pll = ((2ll - 1) * z * pmm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmm1 = pmm1, pll
    end
    return pll
end

# (θ, φ) of a unit direction, Dual-safe at the poles: acos's derivative kept finite, φ = 0 on the axis
@inline function _polar_azimuth(ω::StaticVector{3})
    T = typeof(_primal(ω[3]))
    lim = one(T) - eps(T)
    θ = acos(clamp(ω[3], -lim, lim))
    ρ² = ω[1]^2 + ω[2]^2
    φ = _primal(ρ²) > eps(typeof(_primal(ρ²)))^2 ? atan(ω[2], ω[1]) : zero(ω[1])
    return θ, φ
end

"""
    sh_basis(Val(l_max), ω) -> SVector{(l_max + 1)²}

Every real spherical harmonic `y_l^m(ω)` of degree `l ≤ l_max` at the unit
direction `ω`, in the order of [`sh_index`](@ref) (no Condon–Shortley phase;
the convention in `lighting.jl`'s header). Allocation-free and device-lawful.
"""
@generated function sh_basis(::Val{L}, ω::StaticVector{3}) where {L}
    body = Expr[]
    vals = Symbol[]
    for l in 0:L, m in -l:l
        y = Symbol(:y, sh_index(l, m))
        push!(vals, y)
        c = m == 0 ? _sh_normalization(l, 0) : sqrt(2) * _sh_normalization(l, m)
        trig = m > 0 ? :(cos($m * φ)) : m < 0 ? :(sin($(-m) * φ)) : :(one(θ))
        push!(body, :($y = T($c) * $trig * _legendre($l, $(abs(m)), z)))
    end
    return quote
        θ, φ = _polar_azimuth(ω)
        T = typeof(θ)
        z = cos(θ)
        $(body...)
        return SVector{$(sh_count(L)), T}($(vals...))
    end
end

# -----------------------------------------------------------------------------
# The directional map: real spherical harmonics per channel
# -----------------------------------------------------------------------------

"""
    SphericalHarmonicMap(coefficients::AbstractMatrix, l_max)

A radiance map on the sphere of directions: `C` channels (rows) of real
spherical-harmonic coefficients up to degree `l_max` (columns, in
[`sh_index`](@ref) order; `C × (l_max + 1)²`). Callable: `M(ω)` is the
`SVector{C}` of the channels at the unit direction `ω`.

    SphericalHarmonicMap(radiance, l_max; channels = 3)

The constant map `M ≡ radiance` in every channel (only the `l = 0` term).
"""
struct SphericalHarmonicMap{L, C, J, CJ, T}
    coefficients::SMatrix{C, J, T, CJ}
    function SphericalHarmonicMap{L}(a::SMatrix{C, J, T, CJ}) where {L, C, J, CJ, T}
        J == sh_count(L) || throw(DimensionMismatch("SphericalHarmonicMap: $J coefficients per channel, degree $L needs $(sh_count(L))"))
        return new{L, C, J, CJ, T}(a)
    end
end
SphericalHarmonicMap(a::AbstractMatrix, l_max::Integer) = SphericalHarmonicMap{Int(l_max)}(SMatrix{size(a)...}(a))
function SphericalHarmonicMap(radiance::Real, l_max::Integer; channels::Integer = 3)
    a = zeros(typeof(float(radiance)), channels, sh_count(l_max))
    a[:, 1] .= radiance / _sh_normalization(0, 0)        # y_0^0 = K_0^0 = 1/√(4π)
    return SphericalHarmonicMap(a, l_max)
end

@inline (M::SphericalHarmonicMap{L})(ω::StaticVector{3}) where {L} = M.coefficients * sh_basis(Val(L), ω)

"""
    coefficients(M::SphericalHarmonicMap) -> SMatrix{C, (l_max + 1)²}
    coefficients(light::EnvironmentLight)

The spherical-harmonic coefficients, one row per channel.
"""
coefficients(M::SphericalHarmonicMap) = M.coefficients

"""
    max_degree(M::SphericalHarmonicMap) -> Int
    max_degree(light::EnvironmentLight)

The expansion's highest degree `l_max`.
"""
max_degree(::SphericalHarmonicMap{L}) where {L} = L

"""
    with_coefficients(M::SphericalHarmonicMap, a) -> SphericalHarmonicMap
    with_coefficients(light::EnvironmentLight, a) -> EnvironmentLight

The same map (degree, channels) with the coefficients `a`, of `a`'s element
type (a Dual for a Jacobian); a light keeps its centre and radius.
"""
with_coefficients(M::SphericalHarmonicMap{L, C, J}, a::AbstractMatrix) where {L, C, J} =
    SphericalHarmonicMap{L}(SMatrix{C, J}(a))

convert_eltype(::Type{T}, M::SphericalHarmonicMap{L, C, J}) where {T <: Real, L, C, J} = with_coefficients(M, SMatrix{C, J, T}(M.coefficients))

# -----------------------------------------------------------------------------
# The light: the map on a sphere about a centre
# -----------------------------------------------------------------------------

"""
    EnvironmentLight(map, center, radius)

Radiance from every direction, given by the directional `map` (a
[`SphericalHarmonicMap`](@ref)) on the sphere of `radius` about `center`: a ray
from a point `x` in direction `ω` reads the map where it leaves that sphere
([`exit_direction`](@ref)), so nearby points see the light from different
directions (the distant, infinite light is its limit as `radius → ∞`).
`center` and `radius` are lengths: metres as quantities in a public value,
plain numbers in a kernel (`ustrip(m, light)`); the frame is the owner's.
"""
struct EnvironmentLight{M, T}
    map::M
    center::SVector{3, T}
    radius::T
end
function EnvironmentLight(map, center::AbstractVector, radius)
    T = promote_type(eltype(center), typeof(radius))
    return EnvironmentLight(map, SVector{3, T}(center), convert(T, radius))
end

center(light::EnvironmentLight) = light.center
GeometryBasics.radius(light::EnvironmentLight) = light.radius
"""
    radiance_map(light::EnvironmentLight)

The light's directional map ([`SphericalHarmonicMap`](@ref)).
"""
radiance_map(light::EnvironmentLight) = light.map
coefficients(light::EnvironmentLight) = coefficients(light.map)
max_degree(light::EnvironmentLight) = max_degree(light.map)
with_coefficients(light::EnvironmentLight, a::AbstractMatrix) = EnvironmentLight(with_coefficients(light.map, a), light.center, light.radius)

# The parameter protocol: a light's coefficients are its map's spherical-harmonic
# coefficients, column-major (`vec(coefficients(light))`: channel fastest),
# named `a<channel>_<index>`, unbounded; the centre and the radius are the
# light's geometry and stay fixed (`with_parameters` keeps them).
parameter_vector(light::EnvironmentLight) = (a = coefficients(light); SVector{length(a), eltype(a)}(Tuple(a)))
parameter_names(light::EnvironmentLight) = (a = coefficients(light); C = size(a, 1);
    ntuple(i -> Symbol(:a, (i - 1) % C + 1, :_, (i - 1) ÷ C + 1), Val(length(a))))
function with_parameters(light::EnvironmentLight, θ::AbstractVector)
    a = coefficients(light)
    length(θ) == length(a) || throw(ArgumentError("EnvironmentLight takes $(length(a)) parameters (its coefficients); got $(length(θ))"))
    return with_coefficients(light, SMatrix{size(a)..., eltype(θ)}(Tuple(θ)))
end

convert_eltype(::Type{T}, light::EnvironmentLight) where {T <: Real} =
    EnvironmentLight(convert_eltype(T, light.map), SVector{3, T}(light.center), T(light.radius))

has_units(light::EnvironmentLight) = has_units(light.center)
FlexUnits.ustrip(u::Units, light::EnvironmentLight) =
    EnvironmentLight(light.map, _ustrip_point(u, light.center), _strip_length(u, light.radius))
with_unit(light::EnvironmentLight, u::Units) =
    EnvironmentLight(light.map, _attach_point(u, light.center), _attach_length(light.radius, u))

"""
    exit_direction(light::EnvironmentLight, x, ω) -> SVector{3}

The unit direction from the light's centre to the point where the ray
`x + μω` (`μ > 0`, `ω` a unit vector) leaves the light's sphere. For `x`
outside the sphere (a wide view of a board larger
than the sphere): a ray whose line misses the sphere is read at the line's
closest point to the centre, `μ = −(x − c)·ω`, with finite derivatives there
(not `sqrt(max(Δ, 0))`, whose derivative at 0 is `Inf · 0`); one whose line
crosses it but points away is read at the crossing nearer to `x`, behind it
(`μ < 0`).
"""
@inline function exit_direction(light::EnvironmentLight, x::StaticVector{3}, ω::StaticVector{3})
    d = x - light.center
    b = dot(d, ω)
    Δ = b^2 - (dot(d, d) - light.radius^2)
    μ = -b + (Δ > zero(Δ) ? sqrt(Δ) : zero(Δ))
    s = d + μ * ω
    return s / sqrt(dot(s, s))
end

"""
    radiance(light::EnvironmentLight, x, ω) -> SVector{C}

The radiance per channel arriving at `x` from direction `ω` (a unit vector,
pointing from `x` towards the light): the map at
[`exit_direction`](@ref)`(light, x, ω)`.
"""
@inline radiance(light::EnvironmentLight, x::StaticVector{3}, ω::StaticVector{3}) = light.map(exit_direction(light, x, ω))

"""
    coefficient_jacobian(light::EnvironmentLight, x, ω) -> SVector{(l_max + 1)²}

The radiance's linear factor in the coefficients, the same for every channel:
`coefficients(light) * coefficient_jacobian(light, x, ω) == radiance(light, x, ω)`.
"""
@inline coefficient_jacobian(light::EnvironmentLight, x::StaticVector{3}, ω::StaticVector{3}) =
    sh_basis(Val(max_degree(light)), exit_direction(light, x, ω))

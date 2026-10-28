# =============================================================================
# Homography Solvers — Public API for 2D projective homography estimation
# =============================================================================
#
# Pure geometry solvers for estimating homographies from point correspondences.
# These functions are independent of RANSAC — usable from known inliers.
#
# DEPENDENCY: Uses StaticArrays (SVector, SMatrix), LinearAlgebra (norm, svd, inv, Diagonal)
#
# =============================================================================

using StaticArrays: SVector, SMatrix, StaticMatrix, StaticVector, SA, @SMatrix, similar_type
using LinearAlgebra: norm, svd, eigvals, Symmetric, det, Diagonal, kron

# =============================================================================
# Shared Utilities (used by both homography and fundamental matrix solvers)
# =============================================================================

@inline function _hartley_affine(::Type{T}, s::T, cx::T, cy::T) where {T}
    # Diagonal stores scale coefficients (SVector OK); translation is a free vector → Vec2
    return AffineMap(
        LinearAlgebra.Diagonal(SVector{2, T}(s, s)),
        Vec2{T}(-s * cx, -s * cy)
    )
end

# The ONE place an image point is lifted to the z = 1 plane, `(x, y, 1)` in the
# element type `T` (the epipolar parts and Jacobians, the transfer form, its
# rows and its parameter derivative).
# Typed: a 3-vector (a bearing) is a MethodError, never a point whose third
# component is silently overwritten with 1.
@inline _lift_image_point(u::StaticVector{2}, ::Type{T}) where {T} = SVector{3, T}(u[1], u[2], one(T))

@inline _hartley_identity(::Type{T}) where {T} =
    AffineMap(
    LinearAlgebra.Diagonal(SVector{2, T}(one(T), one(T))),
    zero(Vec2{T})
)

# The weight predicate of the single Hartley-map body: each weights variant
# inlines to its own loop. The normalizer is Σ(effective weight) — the count
# without weights, Σw with them — so the weighted mean distance equals √2 in
# every variant.
@inline _hartley_accept(::Nothing, i, ::Type{T}) where {T} = (true, one(T))
@inline _hartley_accept(w::AbstractVector, i, ::Type{T}) where {T} =
    (w[i] > zero(eltype(w)), T(w[i]))

# The correspondences a selection names, as ONE index iterable that both the
# normalization and the DLT fill walk: all of them, a mask's true entries (the
# mask must be the data's length), or an index list as given — in order,
# repeats included, never expanded to an O(n) mask.
@inline _selection(::Nothing, u₁, u₂) = eachindex(u₁, u₂)
@inline function _selection(mask::AbstractVector{Bool}, u₁, u₂)
    length(mask) == length(u₁) == length(u₂) || throw(
        DimensionMismatch("mask has $(length(mask)) entries for $(length(u₁)) correspondences")
    )
    return _mask_indices(mask)
end
@inline _selection(idx, u₁, u₂) = idx

# The Hartley map of `pts` over the selected indices `sel` (the one body: every
# weights variant inlines to its own loop) — translation to the weighted
# centroid, or to `center` when given; scale √2 over the weighted mean distance
# from it. Indexing is checked: an index list is the caller's.
function _hartley_map(
        pts::AbstractVector{<:StaticVector{2, T}},
        sel,
        w::Union{Nothing, AbstractVector},
        center
    ) where {T}
    cx = zero(T); cy = zero(T); sw = zero(T)
    for i in sel
        acc, wi = _hartley_accept(w, i, T)
        acc || continue
        sw += wi; cx += wi * pts[i][1]; cy += wi * pts[i][2]
    end
    # Uniform degenerate guard: empty input, empty selection, or all-zero
    # weights yield the identity frame.
    sw < eps(T) && return _hartley_identity(T)
    if center === nothing
        cx /= sw; cy /= sw
    else
        cx = T(center[1]); cy = T(center[2])
    end
    d = zero(T)
    for i in sel
        acc, wi = _hartley_accept(w, i, T)
        acc || continue
        dx = pts[i][1] - cx; dy = pts[i][2] - cy
        d += wi * hypot(dx, dy)
    end
    d /= sw
    s = d > eps(T) ? sqrt(T(2)) / d : one(T)
    return _hartley_affine(T, s, cx, cy)
end


# =============================================================================
# Normalization strategies — `<Method>Normalization`, applied as an AffineMap
# =============================================================================

"""
    HartleyNormalization(; center = nothing)

Hartley's data normalization as a STRATEGY (Hartley & Zisserman §4.4):
translate to the (weighted) centroid of the points — or to a KNOWN `center`
— and scale isotropically so that the mean distance from it is √2. Apply it
with `AffineMap(HartleyNormalization(...), points[, mask, weights])`.

The known centre is the radial models' frame: their distortion centre must
land on the origin the minimal solvers are written about, and the isotropic
scale keeps the division model in its class (`k′ = k/s²`); a centroid
translation would move the centre off the origin.
"""
struct HartleyNormalization{C}
    center::C
end
HartleyNormalization(; center = nothing) =
    HartleyNormalization(center === nothing ? nothing : SVector{2}(center[1], center[2]))

"""
    DiagonalNormalization(size; center = (width/2, height/2))

The image-size normalization as a STRATEGY: translate to `center` (the image
centre by default; `size` in pixels, plain numbers or `px` quantities — a length
is a `DimensionError`) and divide by the half-diagonal `hypot(width, height)/2`,
so the image corners land at unit radius — the coordinates of a
[`NormalizedImageFrame`](@ref). For STORAGE only (the benchmark pools, where
one noise level and one threshold must mean the same across resolutions);
estimation normalizes by [`HartleyNormalization`](@ref). Apply it with
`AffineMap(DiagonalNormalization(size))`.
"""
struct DiagonalNormalization
    width::Float64
    height::Float64
    center::SVector{2, Float64}
end
function DiagonalNormalization(size; center = nothing)
    w = _pixel_extent(size[1]); h = _pixel_extent(size[2])
    c = center === nothing ? SVector(w / 2, h / 2) : SVector{2, Float64}(center[1], center[2])
    return DiagonalNormalization(w, h, c)
end
_pixel_extent(x) = Float64(_strip_pixels(x))   # a plain number or px; a length is a DimensionError

"""
    AffineMap(s::HartleyNormalization, points, mask = nothing, weights = nothing) -> AffineMap
    AffineMap(s::DiagonalNormalization) -> AffineMap

Apply a normalization strategy: the 2D affine map `x ↦ σ·(x − c)` it
prescribes (a `Diagonal` linear part, uniform scale). For Hartley the centre
and scale come from `points` — `mask` (a `BitVector`) and `weights` restrict
and weight the statistics, an empty selection gives the identity; the
diagonal strategy needs no data.
"""
CoordinateTransformations.AffineMap(
    s::HartleyNormalization, pts::AbstractVector{<:StaticVector{2}},
    mask::Union{Nothing, BitVector} = nothing, w::Union{Nothing, AbstractVector} = nothing
) = _hartley_map(pts, _selection(mask, pts, pts), w, s.center)
CoordinateTransformations.AffineMap(s::HartleyNormalization, pts::AbstractVector{<:StaticVector{2}}, w::AbstractVector) =
    _hartley_map(pts, eachindex(pts), w, s.center)
CoordinateTransformations.AffineMap(d::DiagonalNormalization) =
    _hartley_affine(Float64, one(Float64) / (hypot(d.width, d.height) / 2), d.center[1], d.center[2])

"""
    _unit_norm_jacobian(x::SVector{9}, n = norm(x), sign = one(T)) -> SMatrix{9,9}

Jacobian of `sign * x / ‖x‖` w.r.t. `x` — the rank-8 radial projection
`sign/‖x‖ · (I₉ − x̂x̂ᵀ)`. The single delta-method projector shared by the
DLT parameter-covariance blocks (no sign flip: they map onto the raw
representative) and the 4-pt covariance chain (which carries the
`sign_normalize` flip explicitly).
"""
@inline function _unit_norm_jacobian(
        x::SVector{9, T}, n::T = norm(x), sign::T = one(T)
    ) where {T}
    inv_n = one(T) / n
    x̂ = x * inv_n
    return sign * inv_n * (SMatrix{9, 9, T}(I) - x̂ * x̂')
end

"""
    _vec9_to_mat33(h, T) -> SMatrix{3,3,T,9}

Reshape a 9-element row-major vector to a column-major 3×3 SMatrix.

DLT solutions are stored row-major: `[h₁₁, h₁₂, h₁₃, h₂₁, h₂₂, h₂₃, h₃₁, h₃₂, h₃₃]`.
SMatrix constructor is column-major, so indices are transposed.
"""
@inline function _vec9_to_mat33(h, ::Type{T}) where {T}
    return SMatrix{3, 3, T, 9}(
        T(h[1]), T(h[4]), T(h[7]),
        T(h[2]), T(h[5]), T(h[8]),
        T(h[3]), T(h[6]), T(h[9])
    )
end

"""
    unit_normalize(H::StaticMatrix{3,3}) -> SMatrix{3,3}

`H / ‖H‖` with NO degenerate guard and NO sign convention — the hot-loop
sibling of [`sign_normalize`](@ref) for chart lifts and retractions, where
a `Union{Nothing,...}` return is unacceptable and the `H[3,3] ≥ 0` flip
would break retraction continuity mid-iteration.
"""
@inline unit_normalize(H::StaticMatrix{3, 3, T}) where {T} =
    SMatrix{3, 3, T, 9}(H) / norm(H)

"""
    sign_normalize(H::StaticMatrix{3,3,T}) -> typeof(H) or nothing

Frobenius-normalize a 3×3 matrix and ensure H[3,3] >= 0.
Returns `nothing` if the matrix norm is negligible.
Preserves the input type (HomographyMat, FundMat, SMatrix, etc.).
"""
function sign_normalize(H::StaticMatrix{3, 3, T}) where {T}
    n = norm(H)
    n < eps(T) && return nothing
    Hn = SMatrix{3, 3, T, 9}(H) / n
    if Hn[3, 3] < zero(T)
        Hn = -Hn
    end
    return similar_type(typeof(H), T)(Tuple(Hn))
end

# =============================================================================
# Hartley Unnormalization
# =============================================================================
#
# Undo Hartley normalization after solving in normalized coordinates.
# Dispatches on matrix type for the correct transformation law:
#   - HomographyMat (point map):  H = T₂⁻¹ H_norm T₁
#   - FundMat (line map):  F = T₂ᵀ F_norm T₁  (+ rank-2 constraint)
#

# =============================================================================
# CspondNormalization — a correspondence set's normalization as a value (#21)
# =============================================================================

"""
    CspondNormalization(T_src::AffineMap, T_dst::AffineMap)
    CspondNormalization(src::Normalization, dst::Normalization, cs[, mask, weights])
    CspondNormalization(M::Type, cs; center)

The normalization of a correspondence set as a first-class value: the
forward homogeneous maps of the source and target sides AND their analytic
inverses, computed once at construction (never per candidate model). Build
it from two affine maps, from a normalization strategy per side
([`HartleyNormalization`](@ref), [`DiagonalNormalization`](@ref)) applied to
the correspondences' sources and targets, or by MODEL TYPE — the strategy
each model class needs:

- `HomographyMat` / `AbstractFundMat` (F and E): Hartley at the centroid on
  both sides (the classic DLT data normalization), over the `mask`'s
  correspondences and weighted by the `weights` column when there is one;
- `RadialHomography`: Hartley at the centroid on the sources, Hartley about
  the known distortion `center` (required) on the targets — the origin the
  5-point kernel is written about.

Verbs:
- `normalize(normalization, cs)` — materialized normalized correspondence set
- [`source_map`](@ref) / `destination_map` — its two sides as affine maps
- a model moves by composition with the two sides:
  `destination_map(N) ∘ M ∘ inv(source_map(N))` carries a data-frame model `M` into
  the normalized frame and `inv(destination_map(N)) ∘ M ∘ source_map(N)` back (`M`
  a `HomographyMat`, a `FundMat`/`EssentialMat`, a `HomographyMap`, an
  [`EpipolarMap`](@ref), a `RadialHomography`, a `RadialCamera`).

The pose family builds none (bearings are already metric).
"""
struct CspondNormalization{T}
    src::SMatrix{3, 3, T, 9}
    dst::SMatrix{3, 3, T, 9}
    src_inv::SMatrix{3, 3, T, 9}
    dst_inv::SMatrix{3, 3, T, 9}
end

function CspondNormalization(T_src::AffineMap, T_dst::AffineMap)
    T = promote_type(eltype(T_src.translation), eltype(T_dst.translation))
    return CspondNormalization{T}(
        SMatrix{3, 3, T, 9}(to_homogeneous(T_src)),
        SMatrix{3, 3, T, 9}(to_homogeneous(T_dst)),
        SMatrix{3, 3, T, 9}(to_homogeneous(inv(T_src))),
        SMatrix{3, 3, T, 9}(to_homogeneous(inv(T_dst)))
    )
end

# `mask` is any selection `_selection` reads (nothing, a Bool mask, an index
# list), resolved once and walked by both sides.
function CspondNormalization(src::HartleyNormalization, dst::HartleyNormalization, cs, mask = nothing, weights = nothing)
    u₁ = sources(cs); u₂ = targets(cs)
    sel = _selection(mask, u₁, u₂)
    return CspondNormalization(_hartley_map(u₁, sel, weights, src.center), _hartley_map(u₂, sel, weights, dst.center))
end

CspondNormalization(::Type{<:Union{HomographyMat, AbstractFundMat}}, cs; mask = nothing) =
    CspondNormalization(HartleyNormalization(), HartleyNormalization(), cs, mask, weights_column(cs))

# A selection as a BitVector (an index list expanded once), for callers that
# hold one; the DLT normalization walks an index list as given (`_selection`).
_bitmask(::Nothing, n) = nothing
_bitmask(mask::BitVector, n) = mask
_bitmask(idx, n) = (b = falses(n); b[idx] .= true; b)

"""
    source_map(f::CspondNormalization) -> AffineMap
    destination_map(f::CspondNormalization) -> AffineMap

The forward normalization of the source side and of the destination side
(the correspondences' `targets`), as 2D affine maps (`source_map(f)(x)` is
the normalized source point). The accessors to a frame's sides; its inverses
are `inv` of these.
"""
source_map(f::CspondNormalization) = AffineMap(f.src)
destination_map(f::CspondNormalization) = AffineMap(f.dst)

"""
    normalize(N::CspondNormalization, cs) -> Csponds

Materialize the normalized view of a correspondence container: source
points through `N.src`, target points through `N.dst`. The measurement
covariance columns `Σ₁`/`Σ₂` ([`noise_columns`](@ref)) move with their points,
`Σ′ = A·Σ·Aᵀ` with `A` the side's linear part (so a similarity of scale `s`
gives `s²·Σ`); every other attribute column is forwarded unchanged. Built once
per problem (the RANSAC hot loop then samples the normalized columns directly).
"""
function LinearAlgebra.normalize(f::CspondNormalization{T}, cs::AbstractVector) where {T}
    S, D = source_map(f), destination_map(f)
    ns = [Point2{T}(S(p)) for p in sources(cs)]
    nt = [Point2{T}(D(p)) for p in targets(cs)]
    return Csponds(ns, nt, _transport_noise(attrs_of(cs), S.linear, D.linear))
end


# =============================================================================
# Weight Application Dispatch
# =============================================================================
#
# Applies per-correspondence weights to DLT matrix rows.
# Dispatches on element type:
#   - Nothing: no-op
#   - Real scalar: multiply row group by sqrt(w_i)
#   - SMatrix{R,R}: apply matrix factor directly to row group
#

@inline _apply_row_weight!(::AbstractMatrix, ::Int, ::Int, ::Nothing, ::Int) = nothing

@inline function _apply_row_weight!(
        A::AbstractMatrix{T}, r::Int, nrows::Int,
        w::AbstractVector{<:Real}, k::Int
    ) where {T}
    sqrt_w = sqrt(T(w[k]))
    @inbounds for j in 1:size(A, 2)
        for i in 0:(nrows - 1)
            A[r + i, j] *= sqrt_w
        end
    end
    return nothing
end

@inline function _apply_row_weight!(
        A::AbstractMatrix{T}, r::Int, ::Int,
        W::AbstractVector{<:SMatrix{2, 2}}, k::Int
    ) where {T}
    # Apply 2×2 factor to row pair: [row1'; row2'] = W * [row1; row2]
    @inbounds Wk = W[k]
    w11 = T(Wk[1, 1]); w12 = T(Wk[1, 2])
    w21 = T(Wk[2, 1]); w22 = T(Wk[2, 2])
    r1 = r; r2 = r + 1
    @inbounds for j in 1:size(A, 2)
        a1 = A[r1, j]
        a2 = A[r2, j]
        A[r1, j] = w11 * a1 + w12 * a2
        A[r2, j] = w21 * a1 + w22 * a2
    end
    return nothing
end

# =============================================================================
# Public: 4-Point Homography (Inhomogeneous)
# =============================================================================

"""
    homography_4pt(s₁, s₂, s₃, s₄, d₁, d₂, d₃, d₄) -> SMatrix{3,3} or nothing

Compute a homography from exactly 4 point correspondences using the
inhomogeneous method (two 3×3 LU solves, zero heap allocation).

Returns a Frobenius-normalized `SMatrix{3,3}` with H[3,3] >= 0,
or `nothing` if the points are degenerate (collinear).

# Algorithm
From "An Algebraic Derivation of the Inhomogeneous Homography":
1. Build U3 = [u1|u2|u3], solve U3*c = -u4 for projective depths
2. Build U3' = [u1'|u2'|u3'], solve U3'*c' = -u4' for target depths
3. λ_i = c'_i / c_i (projective depth ratios)
4. H = U3' * diag(λ) * inv(U3)

# Example
```julia
s = [SA[0.0,0.0], SA[1.0,0.0], SA[1.0,1.0], SA[0.0,1.0]]
d = [SA[0.1,0.1], SA[1.1,0.0], SA[1.0,1.1], SA[0.0,1.0]]
H = homography_4pt(s[1], s[2], s[3], s[4], d[1], d[2], d[3], d[4])
```
"""
function homography_4pt(
        s₁::StaticVector{2, T}, s₂::StaticVector{2, T},
        s₃::StaticVector{2, T}, s₄::StaticVector{2, T},
        d₁::StaticVector{2, T}, d₂::StaticVector{2, T},
        d₃::StaticVector{2, T}, d₄::StaticVector{2, T}
    ) where {T}
    core = _homography_4pt_core(s₁, s₂, s₃, s₄, d₁, d₂, d₃, d₄)
    core === nothing && return nothing
    return sign_normalize(HomographyMat{T}(Tuple(core.H)))
end

# The 4-point homography's forward evaluation. `nothing` for a degenerate
# sample; otherwise the projective bases, their coefficients, the depth ratios
# and the unnormalized `H = Up·diag(λ)·U⁻¹`.
@inline function _homography_4pt_core(
        s₁::StaticVector{2, T}, s₂::StaticVector{2, T},
        s₃::StaticVector{2, T}, s₄::StaticVector{2, T},
        d₁::StaticVector{2, T}, d₂::StaticVector{2, T},
        d₃::StaticVector{2, T}, d₄::StaticVector{2, T}
    ) where {T}
    o = one(T)

    # Build 3x3 source/target matrices (columns = homogeneous points)
    U = SMatrix{3, 3, T, 9}(s₁[1], s₁[2], o, s₂[1], s₂[2], o, s₃[1], s₃[2], o)
    Up = SMatrix{3, 3, T, 9}(d₁[1], d₁[2], o, d₂[1], d₂[2], o, d₃[1], d₃[2], o)

    # Singular-3×3 fast-out: if 3 source (or target) points are collinear, U
    # (or Up) is rank-deficient and inv(U) at the end would produce NaN/Inf
    # entries in H. det(U) on an SMatrix is a handful of flops — cheaper
    # than running through the LU solves and inv only to throw the result
    # away post-hoc.
    (abs(det(U)) < eps(T) || abs(det(Up)) < eps(T)) && return nothing

    # Solve for null-space coefficients (two independent 3x3 LU solves)
    c = U \ SA[-s₄[1], -s₄[2], -o]
    cp = Up \ SA[-d₄[1], -d₄[2], -o]

    any(isnan, c)  && return nothing
    any(isnan, cp) && return nothing

    # Degeneracy: c_i ≈ 0 means collinear source points
    any(x -> abs(x) < eps(T), c) && return nothing

    # Projective depth ratios
    λ = cp ./ c  # SVector{3,T}

    # H = Up * diag(λ) * inv(U)
    Uinv = inv(U)
    H = (Up * Diagonal(λ)) * Uinv
    return (; U, Uinv, Up, c, cp, λ, H)
end

# =============================================================================
# Public: N-Point DLT with Hartley Normalization
# =============================================================================

"""
    homography_dlt(cs; mask = nothing, svd_ws = nothing) -> HomographyMat or nothing

Compute a homography from N >= 4 correspondences `cs` (`u₁ => u₂` pairs; any
[`sources`](@ref)/[`targets`](@ref) container, a [`Csponds`](@ref) preferred)
by the Direct Linear Transform with Hartley normalization.

Per-correspondence weights are the container's `weights` column (set with
[`withattrs`](@ref)), never a keyword: each weight multiplies its rows'
squared algebraic residual and the Hartley normalization. When `mask` (a
`BitVector` or an index list) is given, only those correspondences are used.

# Example
```julia
cs = Csponds(src_pts, dst_pts)
H = homography_dlt(cs)
H = homography_dlt(withattrs(cs; weights = [1.0, 0.5, 1.0, 0.8]))
H = homography_dlt(cs; mask = inlier_mask)
```
"""
function homography_dlt(
        cs::AbstractVector{<:Pair{<:StaticVector{2, T}, <:StaticVector{2, T}}};
        mask = nothing, svd_ws = nothing
    ) where {T}
    n = length(cs)
    A = Matrix{T}(undef, 2n, 9)
    # the one normalization of this estimator: built here, once, from the
    # data (and its `weights` column); the `!` form only uses it
    return homography_dlt!(A, CspondNormalization(HomographyMat, cs; mask), cs; mask, svd_ws)
end

# One correspondence's two DLT rows in normalized homogeneous points `s`, `d`
# (row-major vec(H)): `[sᵀ 0 −d₁sᵀ; 0 sᵀ −d₂sᵀ]`. The per-point filler of `_dlt!`.
Base.@propagate_inbounds function _homography_dlt_rows!(A::AbstractMatrix{T}, r, s, d) where {T}
    r2 = r + 1
    A[r, 1] = s[1]; A[r, 2] = s[2]; A[r, 3] = s[3]
    A[r, 4] = zero(T); A[r, 5] = zero(T); A[r, 6] = zero(T)
    A[r, 7] = -d[1] * s[1]; A[r, 8] = -d[1] * s[2]; A[r, 9] = -d[1] * s[3]

    A[r2, 1] = zero(T); A[r2, 2] = zero(T); A[r2, 3] = zero(T)
    A[r2, 4] = s[1]; A[r2, 5] = s[2]; A[r2, 6] = s[3]
    A[r2, 7] = -d[2] * s[1]; A[r2, 8] = -d[2] * s[2]; A[r2, 9] = -d[2] * s[3]
    return nothing
end

"""
    _dlt!(fill_rows!, A, N, cs, rows_per_point, mask, svd_ws) -> SMatrix{3,3} or nothing

The ONE DLT pipeline of the 3×3 families (homography, fundamental): walk the
selection (`_selection`: all, a Bool mask, or an index list as given), map each
correspondence through the estimator's normalization `N`, let the family's
`fill_rows!(A, r, s, d)` write its `rows_per_point` rows from the homogeneous
normalized points, weight them by the `weights` column, then take the null
vector of the compacted `A[1:m, :]` and reshape it (row-major) to 3×3 — still in
the NORMALIZED frame; the family finishes (rank-2 projection, composition back,
`sign_normalize`).

Guards: an index list that asks for more rows than `A` holds is a
`DimensionMismatch`, an out-of-range index a `BoundsError`, and a non-finite row
gives `nothing` (non-finite rows poison LAPACK, whose gesdd throws an opaque
`ArgumentError`). The null vector comes from `_dlt_nullvec`: with
`svd_ws::SvdWorkspace` and a `Float64` `A`, pre-allocated LAPACK workspace
(overwriting `A`); otherwise an allocating SVD of the compacted rows.
"""
function _dlt!(
        fill_rows!::F, A::AbstractMatrix{T}, N::CspondNormalization, cs,
        rows_per_point::Int, mask, svd_ws
    ) where {F, T}
    u₁ = sources(cs); u₂ = targets(cs); weights = weights_column(cs)
    Sn, Dn = source_map(N), destination_map(N)
    npts = 0
    for k in _selection(mask, u₁, u₂)
        npts += 1
        # an index list may repeat: the rows it asks for must fit in A
        rows_per_point * npts <= size(A, 1) || throw(
            DimensionMismatch("index list asks for more than $(size(A, 1) ÷ rows_per_point) points' rows of A")
        )
        checkbounds(u₁, k)
        r = rows_per_point * (npts - 1) + 1
        @inbounds fill_rows!(A, r, to_affine(Sn(u₁[k])), to_affine(Dn(u₂[k])))
        _apply_row_weight!(A, r, rows_per_point, weights, k)
    end
    m = rows_per_point * npts

    all(isfinite, view(A, 1:m, :)) || return nothing

    return _vec9_to_mat33(_dlt_nullvec(A, m, svd_ws), T)
end

# Lazy index view of a Bool mask — no `findall` allocation. Shared by the DLT
# fills, the cheirality tests and the radial-homography consensus test.
_mask_indices(mask::AbstractVector{Bool}) = (i for i in eachindex(mask) if mask[i])

"""
    homography_dlt!(A, N::CspondNormalization, cs; mask = nothing, svd_ws = nothing) -> HomographyMat or nothing

In-place N-point DLT of the correspondences `cs` using the pre-allocated
buffer `A` (2N × 9). Weights are the container's `weights` column, as for
[`homography_dlt`](@ref); inside an LO/IRLS loop set them per iteration with
`withattrs(cs; weights = w)` (no allocation beyond the caller's `w`).

When `mask::BitVector` is provided, only masked points are used and rows are
compacted into the top of `A`. The inner-loop form: `N` is the enclosing
estimator's [`CspondNormalization`](@ref), built once at its entry (the public
form builds it from the data); this form applies it row by row and composes the
model back into the data frame (`inv(D) ∘ H ∘ S`), and never
normalizes on its own.

If `svd_ws::SvdWorkspace` is provided and `A` is `Float64`, uses pre-allocated
LAPACK workspace for the SVD (zero allocations). Note: `A` is overwritten when
using `svd_ws`.

No covariance is returned. The DLT minimizes ALGEBRAIC residuals, whose noise is
heteroscedastic under pixel noise, so the SVD's own resolvent covariance is not
this estimator's (measured overconfident against Monte Carlo). Its first-order
covariance, if ever needed, is [`implicit_covariance`](@ref)'s block form with
the DLT's identity weighting.
"""
function homography_dlt!(
        A::AbstractMatrix{T},
        N::CspondNormalization{T},
        cs::AbstractVector{<:Pair{<:StaticVector{2, T}, <:StaticVector{2, T}}};
        mask = nothing,
        svd_ws = nothing
    ) where {T}
    # N is the enclosing estimator's normalization: used here, never built here
    H_norm = _dlt!(_homography_dlt_rows!, A, N, cs, 2, mask, svd_ws)
    H_norm === nothing && return nothing
    return sign_normalize(inv(destination_map(N)) ∘ HomographyMat{T}(Tuple(H_norm)) ∘ source_map(N))   # composed back to the data frame, then the gauge
end

# =============================================================================
# Public: Symmetric Transfer Error
# =============================================================================


# NOTE (#21 dedup): the former `_homography_sample_nondegenerate` (same-side +
# sin²(2°) collinearity gate) was an unconsumed duplicate of
# `test_sample(::Type{<:HomographyMat}, cs, idx)` (model_validity.jl) with a
# divergent threshold — deleted; `test_sample` is THE pre-solve gate.

# =============================================================================
# Typed Homography Geometry — dispatched on HomographyMat
# =============================================================================


# THE uniform-sign cheirality rule, shared by the homography and the radial
# homography: `rays(i)` returns the predicted ray `h` (the model applied to the
# source) and the observed ray `ℓ` (the target's lift: `(u′, 1)`, or the division
# lift), and `ℓ ∝ λ·h` must hold with ONE sign of λ over `indices`. λ is read at
# the largest-|·| component of `h` (ties: w, then x — never a division by a
# vanishing component); a predicted ray that is exactly zero images nothing and
# fails. Scale-free: no absolute threshold.
@inline function _test_uniform_cheirality(rays::F, indices) where {F}
    pos = false; neg = false
    @inbounds for i in indices
        h, ℓ = rays(i)
        ax, ay, aw = abs(h[1]), abs(h[2]), abs(h[3])
        j = aw >= ax && aw >= ay ? 3 : ax >= ay ? 1 : 2
        iszero(h[j]) && return false
        λ = ℓ[j] / h[j]
        λ > zero(λ) && (pos = true)
        λ < zero(λ) && (neg = true)
        pos && neg && return false
    end
    return true
end

# Inner loop over any index iterable, generic over any 3×3 StaticMatrix. Both
# public entry points (HomographyMat and HomographyMap) delegate here.
function _test_cheirality_idx(H::StaticMatrix{3, 3, T}, cs, idx) where {T}
    u = sources(cs); v = targets(cs)
    return _test_uniform_cheirality(idx) do i
        (H * to_affine(u[i]), to_affine(SVector{2, T}(v[i][1], v[i][2])))
    end
end

# Mask form: the same loop over the mask's lazy index view.
_test_cheirality_mask(H::StaticMatrix{3, 3}, cs, mask::AbstractVector{Bool}) =
    _test_cheirality_idx(H, cs, _mask_indices(mask))

# Public entry points — HomographyMat (raw, sample-time RANSAC use) and
# HomographyMap (idiomatic — matches the rest of VGC's transform stack).
"""
    test_cheirality(H, cs, idx) -> Bool

Projective cheirality check, uniform-sign variant. For each correspondence
`(uᵢ, u′ᵢ)` at the sample indices, compute the projective scale factor `λᵢ`
such that `ũ′ᵢ = λᵢ · H · ũᵢ` in homogeneous coordinates, where `ũ = [u; 1]`.
Accept iff all `λᵢ` share a sign.

Rationale — `H` is only defined up to a nonzero scalar (`H` and `−H` are the
same mapping). Under `H → −H`, every `λᵢ` flips sign, so requiring a
*specific* sign (e.g. the original Shekhovtsov rule: sign-normalize to
`det(H) > 0`, then require `λ > 0`) amounts to picking a matrix representative
by convention. That convention matches the image-to-image case (where both
frames share 2D handedness) but fails for object-to-image homographies
(e.g. a world y-up calibration board mapped to a y-down image), where the
"in front" representative has `det(H) < 0` — the strict rule then rejects
every correct fit.

The mapping-level invariant is **uniform sign across the `λᵢ`**: mixed signs
signal a bow-tie in projective depth (a real pathology that cannot be
repaired by a matrix sign flip), while a shared sign — positive or negative —
signals a consistent mapping (with the overall sign absorbed into the free
scalar of H).

For each `i`: compute `Hu = H · ũᵢ`, pick the component `k` with the largest
absolute value (numerical robustness), and set `λᵢ = ũ′ᵢ[k] / Hu[k]`
(with `ũ′[3] = 1`). Reject the sample the moment a sign disagreement is seen.
An inner-loop gate: `H` and `cs` in the same frame (sign-based, so any
positive-scale normalization, or none, gives the same answer).
"""
test_cheirality(H::HomographyMat, cs, idx::AbstractVector{<:Integer}) =
    _test_cheirality_idx(H, cs, idx)
test_cheirality(H::HomographyMat, cs, mask::AbstractVector{Bool}) =
    _test_cheirality_mask(H, cs, mask)
test_cheirality(h::HomographyMap{3}, cs, idx::AbstractVector{<:Integer}) =
    _test_cheirality_idx(h.forward, cs, idx)
test_cheirality(h::HomographyMap{3}, cs, mask::AbstractVector{Bool}) =
    _test_cheirality_mask(h.forward, cs, mask)

"""
    is_orientation_preserving(h::HomographyMap{3}, rect::HyperRectangle{2}) -> Bool

True iff the four corners of `rect` map through `h` to a simple
(non-self-intersecting) convex quadrilateral. Catches "bow-tie" homographies
that the per-correspondence checks (`test_cheirality`, `test_conditioning`)
miss — these come from H fits that are well-behaved over the inlier cluster
but extrapolate to fold the rectangle's corners across each other.

Returns `true` when consecutive edge cross-products along the projected
boundary share a non-zero sign (consistent winding); `false` when signs are
mixed (the projected quad self-intersects). A purely degenerate projection
(all crosses zero) also returns `false`, since it can't be a valid extent map.

This is an *extent-aware* check that complements the per-point predicates in
RANSAC: caller supplies the rectangle over which the homography is supposed
to be valid (e.g. a calibration board's `Rect(bundle, :world)`).

# Example
```julia
H_fit = HomographyMap(matrix_from_ransac)
rect  = Rect2(SA[0.0, 0.0], SA[100.0, 60.0])   # board extent in world coords
is_orientation_preserving(H_fit, rect) || @warn "Bow-tie fit"
```
"""
function is_orientation_preserving(h::HomographyMap{3, T}, rect::HyperRectangle{2}) where {T}
    corners = rect_corners(SVector{2, T}(rect.origin), SVector{2, T}(rect.widths))
    # Project each corner via h (returns Point{3} — raw projective coords;
    # divide by w for image-plane coordinates).
    img = ntuple(4) do i
        p = h(Point{2, T}(corners[i]))
        SVector{2, T}(PerspectiveMap()(p))
    end
    return is_convex_quad(img)
end

"""
    is_convex_quad(q::NTuple{4, <:StaticVector{2}}) -> Bool

True iff the closed polygon `q[1] → q[2] → q[3] → q[4]` is a simple, convex,
consistently-wound quadrilateral: every edge cross-product around the boundary
shares one non-zero sign. Mixed signs (a self-intersecting "bow-tie") and the
all-zero degenerate case are `false`.
"""
is_convex_quad(q::NTuple{4, <:StaticVector{2}}) =
    _one_sign(ntuple(i -> _orient(q[i], q[mod1(i + 1, 4)], q[mod1(i + 2, 4)]), Val(4)))   # the turn at every vertex

# HomographyMat entry — mirrors the test_cheirality dispatch pattern.
is_orientation_preserving(H::HomographyMat, rect::HyperRectangle{2}) =
    is_orientation_preserving(HomographyMap(H), rect)

"""
    test_conditioning(H; max_cond=100) -> Bool

Reject homographies whose SVD condition number κ = σ₁/σ₃ exceeds `max_cond`.

Catches near-rank-deficient H — the same pathology PoseLib's post-solve
`|det(H_normalized)| < 1e-8` check rejects (since after Frobenius normalization
det(H) = σ₁σ₂σ₃ and one vanishing singular value makes both κ → ∞ and det → 0),
plus a broader class of numerically unstable but non-singular H's.

Matches Shekhovtsov's `is_homography_ill_conditioned`
(`decompose_homography.py:244`) with the same default threshold κ = 100.

κ is a property of the frame `H` is written in, not of the map: the bound is
calibrated for `H` between Hartley-normalized frames (both sides, as
[`CspondNormalization`](@ref) builds them), so evaluate it on the normalized
`H`, never on a pixel-frame one.
"""
function test_conditioning(H::HomographyMat{T}; max_cond::T = T(100)) where {T}
    # Defensive: NaN/Inf entries indicate the 4-pt linear solve was degenerate
    # (e.g. 3 collinear source points). Treat as ill-conditioned so RANSAC drops
    # the candidate cleanly instead of LAPACK throwing.
    (any(isnan, H) || any(isinf, H)) && return false
    # Squared singular values of H are eigenvalues of HᵀH. StaticArrays has a
    # closed-form 3×3 symmetric `eigvals` (Smith's trigonometric variant) which
    # is ~30× faster than `svdvals(SMatrix{3,3})` — the latter has no 3×3
    # specialization and routes through LAPACK `gesdd!`. The κ ≤ max_cond test
    # is monotone under squaring, so the comparison rephrased on σ² is exact.
    H_sm = SMatrix{3, 3, T}(H)
    λ = eigvals(Symmetric(H_sm' * H_sm))  # sorted ascending, length 3
    σ₁² = λ[3]
    σ₃² = max(λ[1], zero(T))  # guard against tiny negatives from roundoff
    # Equivalent to σ₁ ≤ max_cond·(σ₃ + eps): squaring preserves the inequality.
    return σ₁² <= max_cond^2 * (sqrt(σ₃²) + eps(T))^2
end

"""
    constraint_with_jacobians(H::HomographyMat, u₁::SVector{2}, u₂::SVector{2})
        -> (g::SVector{2}, G_x::SMatrix{2,4}, G_θ::SMatrix{2,9})

Compute the algebraic constraint `g` and its derivatives for the homography.

The constraint `g = 0` encodes `ũ₂ ∝ H·ũ₁` (2 equations):
  g₁ = h₁ᵀũ₁ − x'·h₃ᵀũ₁
  g₂ = h₂ᵀũ₁ − y'·h₃ᵀũ₁

Returns:
- `g::SVector{2,T}`      — constraint vector
- `G_x::SMatrix{2,4,T}`  — ∂g/∂(x,y,x',y') (measurement Jacobian)
- `G_θ::SMatrix{2,9,T}`  — ∂g/∂vec(H) (model Jacobian, column-major)
"""
@inline function constraint_with_jacobians(
        H::HomographyMat{T},
        u₁::StaticVector{2, T},
        u₂::StaticVector{2, T}
    ) where {T}
    x, y = u₁[1], u₁[2]
    xp, yp = u₂[1], u₂[2]

    c = H[3, 1] * x + H[3, 2] * y + H[3, 3]

    g1 = (H[1, 1] * x + H[1, 2] * y + H[1, 3]) - xp * c
    g2 = (H[2, 1] * x + H[2, 2] * y + H[2, 3]) - yp * c
    g = Vec2{T}(g1, g2)  # reprojection residual = free vector

    gx_11 = H[1, 1] - xp * H[3, 1]
    gx_12 = H[1, 2] - xp * H[3, 2]
    gx_13 = -c
    gx_21 = H[2, 1] - yp * H[3, 1]
    gx_22 = H[2, 2] - yp * H[3, 2]
    gx_24 = -c
    G_x = SMatrix{2, 4, T}(
        gx_11, gx_21,
        gx_12, gx_22,
        gx_13, zero(T),
        zero(T), gx_24
    )

    G_θ = SMatrix{2, 9, T}(
        x, zero(T),
        zero(T), x,
        -xp * x, -yp * x,
        y, zero(T),
        zero(T), y,
        -xp * y, -yp * y,
        one(T), zero(T),
        zero(T), one(T),
        -xp, -yp,
    )

    return (g, G_x, G_θ)
end

# The TRANSFER form of the homography constraint is a METHOD of
# `constraint_with_jacobians` on a `TransferForm` wrapper, not a second
# function name — it lives in geometry/forms.jl with the other forms, and
# uses the `_transfer_params_jacobian` helper below.

# ∂π(H·x̃)/∂vec(H) = Jπ·(x̃ᵀ ⊗ I₃), column-major vec(H)
@inline _transfer_params_jacobian(Jπ, x̃) = hcat(x̃[1] * Jπ, x̃[2] * Jπ, x̃[3] * Jπ)

"""
    transform_deriv_params(H::HomographyMat, u::StaticVector{2}) -> SMatrix{2,9}

The derivative of the point a homography transfers, `π(H·ũ)`, along its
parameters vec(H) (column-major): `Jπ·(ũᵀ ⊗ I₃)`, `Jπ` the perspective derivative
at `H·ũ`. The G_θ of the transfer form (`constraint_with_jacobians(TransferForm(H), …)`); the transferred
point's model covariance is `propagate(transform_deriv_params(H, u), Σθ)` for an
ambient covariance `Σθ`, and a chart's `action_jacobian` is this through its
`retract_jacobian`.
"""
@inline function CoordinateTransformations.transform_deriv_params(H::HomographyMat, u::StaticVector{2})
    T = float(promote_type(eltype(H), eltype(u)))
    x1 = _lift_image_point(u, T)
    return _transfer_params_jacobian(transform_deriv(PerspectiveMap(), SMatrix{3, 3, T, 9}(H) * x1), x1)
end

# The transfer rows of one correspondence, whitened by a frozen factor W:
# residual W·g and, along chart directions dH, the rows W·Jπ·(dH_q·ũ₁). Each in
# the uniform `(out, i, args...)` form `_foreach_row!` runs (a plain loop on the
# CPU, one kernel on a device).
@inline function _transfer_whitening(H, u₁)
    _, G_x, _ = constraint_with_jacobians(TransferForm(H), u₁, u₁)   # G_x does not depend on u₂
    return whitening_factor(G_x)
end
Base.@propagate_inbounds function _transfer_whitening_row!(W, i, H, u₁)
    W[i] = _transfer_whitening(H, u₁[i])
    return nothing
end
# The same factor under the endpoints' covariances (the measurement-noise rule):
# Σ_g = A·Σ₁·Aᵀ + Σ₂ for G_x = [A −I], a side without a column unit isotropic.
Base.@propagate_inbounds function _transfer_whitening_row!(W, i, H, u₁, Σ₁, Σ₂)
    _, G_x, _ = constraint_with_jacobians(TransferForm(H), u₁[i], u₁[i])
    W[i] = whitening_factor(G_x, pair_covariance(_side_covariance(Σ₁, i), _side_covariance(Σ₂, i)))
    return nothing
end
# The transfer residual of one correspondence: the pixel form's `π(H ũ₁) − u₂`, or
# the bearing form's tangent-plane residual (`_bearing_transfer_parts`).
@inline _transfer_residual(H, u₁::StaticVector{2}, u₂::StaticVector{2}) = first(constraint_with_jacobians(TransferForm(H), u₁, u₂))
@inline _transfer_residual(H, f₁::StaticVector{3}, f₂::StaticVector{3}) = _bearing_transfer_parts(H, f₁, f₂).g
# The derivative of the residual along the transferred point `H x`, and `x`: the
# perspective derivative `Jπ` at `H ũ₁` and `ũ₁` for pixels (u₂ unread), `D` and
# `f₁` for bearings — so `∂g/∂θ_q = D·(dH_q·x)` is one row function for both.
@inline function _transfer_point_derivative(H, u₁::StaticVector{2}, _u₂)
    T = float(promote_type(eltype(H), eltype(u₁)))
    x1 = _lift_image_point(u₁, T)
    return transform_deriv(PerspectiveMap(), SMatrix{3, 3, T, 9}(H) * x1), x1
end
@inline _transfer_point_derivative(H, f₁::StaticVector{3}, f₂::StaticVector{3}) =
    (p = _bearing_transfer_parts(H, f₁, f₂); (p.D, p.s))
Base.@propagate_inbounds function _transfer_rows!(r, i, H, W, u₁, u₂, sqrtw)
    g = _transfer_residual(H, u₁[i], u₂[i])
    e = W[i] * g
    r[2i - 1] = sqrtw[2i - 1] * e[1]
    r[2i] = sqrtw[2i] * e[2]
    return nothing
end
Base.@propagate_inbounds function _transfer_jacobian_rows!(J, i, H, dH, W, u₁, u₂, sqrtw)
    Dp, x1 = _transfer_point_derivative(H, u₁[i], u₂[i])
    D = W[i] * Dp
    for q in 1:length(dH)
        d = D * (dH[q] * x1)
        J[2i - 1, q] = sqrtw[2i - 1] * d[1]
        J[2i, q] = sqrtw[2i] * d[2]
    end
    return nothing
end

"""
    transfer_whitening!(W, H, u₁) -> W
    transfer_whitening!(W, H, u₁, Σ₁, Σ₂) -> W

Per correspondence, the whitening factor `L⁻¹` ([`whitening_factor`](@ref)) of
the transfer constraint at `H`: `Σ_g = A·Aᵀ + I₂`, `A = Jπ·H₁₂` — the Sampson
covariance of the transfer error, frozen at the model it is evaluated at.
With the endpoints' covariance columns `Σ₁`, `Σ₂` (2×2 per point, or `nothing`
for a unit isotropic side — [`noise_columns`](@ref)) it is
`Σ_g = A·Σ₁ᵢ·Aᵀ + Σ₂ᵢ`: an exact source (`Σ₁ᵢ = 0`) and a unit target give
`L⁻¹ = I`, the plain transfer error in the target's units.
One KernelAbstractions kernel on the backend that owns `W`; a plain loop over the
same function on the CPU.
"""
function transfer_whitening!(W::AbstractVector, H::HomographyMat, u₁::AbstractVector)
    return _foreach_row!(_transfer_whitening_row!, W, length(u₁), H, u₁)
end
transfer_whitening!(W::AbstractVector, H::HomographyMat, u₁::AbstractVector, ::Nothing, ::Nothing) =
    transfer_whitening!(W, H, u₁)
function transfer_whitening!(
        W::AbstractVector, H::HomographyMat, u₁::AbstractVector,
        Σ₁::Union{Nothing, AbstractVector}, Σ₂::Union{Nothing, AbstractVector}
    )
    return _foreach_row!(_transfer_whitening_row!, W, length(u₁), H, u₁, Σ₁, Σ₂)
end

"""
    transfer_rows!(r, H, W, u₁, u₂, sqrtw) -> r
    transfer_jacobian_rows!(J, H, dH, W, u₁, sqrtw) -> J

The √w-weighted whitened transfer rows of the homography `H` over `u₁[i] => u₂[i]`,
two per correspondence (`r[2i-1:2i] = sqrtw[2i-1:2i] .* (W[i]·g)`, `g` from
`constraint_with_jacobians(TransferForm(H), …)`), and their Jacobian along the chart
directions `dH = (∂H/∂θ₁, …)` (`J[2i-1:2i, q] = sqrtw .* (W[i]·Jπ·dH_q·ũ₁)`), with
the factors `W` frozen by [`transfer_whitening!`](@ref). One kernel each on
device backends, a plain loop over the same row functions on the CPU.
"""
function transfer_rows!(r::AbstractVector, H::HomographyMat, W::AbstractVector, u₁::AbstractVector, u₂::AbstractVector, sqrtw::AbstractVector)
    return _foreach_row!(_transfer_rows!, r, length(u₁), H, W, u₁, u₂, sqrtw)
end

function transfer_jacobian_rows!(J::AbstractMatrix, H::HomographyMat, dH::Tuple, W::AbstractVector, u₁::AbstractVector{<:StaticVector{2}}, sqrtw::AbstractVector)
    return _foreach_row!(_transfer_jacobian_rows!, J, length(u₁), H, dH, W, u₁, u₁, sqrtw)   # the pixel rows read no u₂ (a bearing set is a MethodError: it has no u₂ here)
end

# The BEARING transfer (the infinite homography on unit bearings with their
# back-projection Jacobians): the same row functions, the parts of
# `constraint_with_jacobians(TransferForm(H), f₁, f₂; M₁, M₂)`.
Base.@propagate_inbounds function _bearing_transfer_whitening_row!(W, i, H, f₁, f₂, M₁, M₂)
    _, G_x, _ = constraint_with_jacobians(TransferForm(H), f₁[i], f₂[i]; M₁ = M₁[i], M₂ = M₂[i])
    W[i] = whitening_factor(G_x)
    return nothing
end

"""
    transfer_whitening!(W, H, f₁, f₂; M₁, M₂) -> W
    transfer_rows!(r, H, W, f₁, f₂, sqrtw) -> r
    transfer_jacobian_rows!(J, H, dH, W, f₁, f₂, sqrtw) -> J

The transfer rows on UNIT BEARINGS `f₁[i] => f₂[i]` (3-vectors) — the infinite
homography of a calibrated camera, a rotation under a pure rotation: the
residual is the tangent-plane form of
`constraint_with_jacobians(TransferForm(H), f₁, f₂; M₁, M₂)`, whitened by the
frozen two-sided Sampson factor `L⁻¹` of `Σ_g = G_x G_xᵀ` (both images noisy,
`M₁`/`M₂` the back-projection Jacobians carrying it into PIXELS), and its
Jacobian along the chart directions `dH` is `W[i]·D·dH_q·f₁`. The pixel methods'
row functions, with the bearing parts in place of the perspective divide;
`transfer_rows!` is the pixel method itself (its row reads the element type).
"""
function transfer_whitening!(W::AbstractVector, H::HomographyMat, f₁::AbstractVector{<:StaticVector{3}}, f₂::AbstractVector{<:StaticVector{3}};
                             M₁::AbstractVector, M₂::AbstractVector)
    return _foreach_row!(_bearing_transfer_whitening_row!, W, length(f₁), H, f₁, f₂, M₁, M₂)
end
function transfer_jacobian_rows!(J::AbstractMatrix, H::HomographyMat, dH::Tuple, W::AbstractVector, f₁::AbstractVector{<:StaticVector{3}},
                                 f₂::AbstractVector{<:StaticVector{3}}, sqrtw::AbstractVector)
    return _foreach_row!(_transfer_jacobian_rows!, J, length(f₁), H, dH, W, f₁, f₂, sqrtw)
end

# The homography's rounding scales (`constraint_rounding_scale`): its
# degenerate gradient is `whiten`'s own relative Cholesky verdict (C = G_x·G_xᵀ
# losing rank relative to its diagonal — a point mapped to the line at
# infinity), so no gradient scale; each constraint row's value is rounding below
# its term-by-term bound |H[i,·]·x̃| + |x′ᵢ|·|H[3,·]·x̃|.
@inline function constraint_rounding_scale(H::HomographyMat, u₁::StaticVector{2}, u₂::StaticVector{2})
    T = float(promote_type(eltype(H), eltype(u₁), eltype(u₂)))
    x1, y1 = abs(u₁[1]), abs(u₁[2])
    w = abs(H[3, 1]) * x1 + abs(H[3, 2]) * y1 + abs(H[3, 3])
    r1 = abs(H[1, 1]) * x1 + abs(H[1, 2]) * y1 + abs(H[1, 3]) + abs(u₂[1]) * w
    r2 = abs(H[2, 1]) * x1 + abs(H[2, 2]) * y1 + abs(H[2, 3]) + abs(u₂[2]) * w
    tol = 8 * eps(T)
    return (; gradient = zero(T), value = SVector{2, T}(tol * r1, tol * r2))
end

# --- The residual sum of squares of a model's Sampson distances ---

# Typed 2-vector points (a 3-vector bearing is refused, not truncated), taken to
# the model's eltype so a Float32 model meets Float32 points.
@inline function _sampson_sq(M, u::StaticVector{2}, v::StaticVector{2})
    T = float(eltype(M))
    return Float64(sampson_distance(M, SVector{2, T}(u[1], u[2]), SVector{2, T}(v[1], v[2])))^2
end
@inline _taken(::Nothing, i) = true
@inline _taken(mask, i) = @inbounds mask[i]

# One pair's square in the uniform `(out, i, args...)` row form (0 off the mask).
Base.@propagate_inbounds function _sampson_sq_row!(d², i, M, u₁, u₂, mask)
    d²[i] = _taken(mask, i) ? _sampson_sq(M, u₁[i], u₂[i]) : 0.0
    return nothing
end

"""
    sampson_rss(M, u₁, u₂[, mask]) -> Float64

The sum of squared Sampson distances ([`sampson_distance`](@ref)) of the
correspondences `u₁[i] => u₂[i]` under the model `M` — a `FundMat`, `EssentialMat`
or `HomographyMat` in the points' frame — over `mask`'s true entries when given:
the residual sum of squares of a σ̂² = RSS/dof estimate. A degenerate pair is
`Inf` (the one Sampson rule), and so is the sum. One KernelAbstractions kernel
(the per-pair squares, summed on the backend) for device arrays; a plain loop
over the same per-pair function on the CPU.
"""
function sampson_rss(M, u₁::AbstractVector, u₂::AbstractVector, mask = nothing)
    backend = get_backend(u₁)
    if backend isa KA.CPU
        rss = 0.0
        @inbounds for i in eachindex(u₁)
            _taken(mask, i) && (rss += _sampson_sq(M, u₁[i], u₂[i]))
        end
        return rss
    end
    # the device: the per-pair squares on the backend that owns the points, summed there
    d² = KA.allocate(backend, Float64, length(u₁))
    return sum(_foreach_row!(_sampson_sq_row!, d², length(u₁), M, u₁, u₂, mask))
end

"""
    sampson_residuals!(R::AbstractMatrix{Float32}, models, u₁, u₂) -> R

The Sampson distance ([`sampson_distance`](@ref)) of every correspondence
`u₁[i] => u₂[i]` (each point read as an `SVector{2, Float64}`) under every model
of `models` — `FundMat`s, `EssentialMat`s or `HomographyMat`s in the points'
frame, or anything `sampson_distance` takes — as `Float32` into the
`length(u₁) × length(models)` matrix `R`, model `j` in column `j`: a pool's
residual matrix. `models` is iterated once and has a length. A buffer of
another size, or `u₁` and `u₂` of different lengths, is a `DimensionMismatch`.
"""
function sampson_residuals!(R::AbstractMatrix{Float32}, models, u₁::AbstractVector, u₂::AbstractVector)
    length(u₂) == length(u₁) ||
        throw(DimensionMismatch("sampson_residuals!: $(length(u₁)) source points and $(length(u₂)) target points"))
    size(R) == (length(u₁), length(models)) || throw(
        DimensionMismatch("sampson_residuals!: R is $(size(R, 1)) × $(size(R, 2)); $(length(u₁)) points × $(length(models)) models need $(length(u₁)) × $(length(models))")
    )
    axes(R, 1) == eachindex(u₁) == eachindex(u₂) || throw(DimensionMismatch("sampson_residuals!: R's rows and the points are indexed differently"))
    @inbounds for (j, M) in enumerate(models), i in eachindex(u₁)
        R[i, j] = Float32(sampson_distance(M, SVector{2, Float64}(u₁[i]), SVector{2, Float64}(u₂[i])))
    end
    return R
end

# --- Correspondence (Pair) overloads ---

@inline constraint_with_jacobians(H::HomographyMat, c::Pair{<:StaticVector{2}, <:StaticVector{2}}) =
    constraint_with_jacobians(H, c.first, c.second)

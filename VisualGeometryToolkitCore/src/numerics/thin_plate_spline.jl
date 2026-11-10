# =============================================================================
# Thin-plate spline on scattered nodes (hoisted from RadiometricCamera
# src/camera/tps.jl and the gradient of src/board/warping.jl, 2026-10)
#
#   f(𝐩) = a₁ + a_x 𝐩_x + a_y 𝐩_y + Σᵢ wᵢ U(𝐩, 𝐩ᵢ),   U(𝐩, 𝐪) = ½ r² ln r²,  r = ‖𝐩 − 𝐪‖
#
# with (w; a) the solution of [K P; Pᵀ 0] (w; a) = (values; 0), K = [U(𝐩ᵢ, 𝐩ⱼ)]
# (zero diagonal), P = [1 𝐩_x 𝐩_y]. The system matrix depends only on the
# nodes, so it is factorized once and (w, a) is a LINEAR map of the nodal
# values.
#
# The evaluation is RadiometricCamera's, operation for operation, and
# test/thin_plate_spline_tests.jl checks every value here against numbers that
# RadiometricCamera's own functions produced (test/fixtures/thin_plate_spline):
# `==` for everything after (w, a), within a measured bound for what goes
# through the solve and BLAS. The solve is conditioned: the system is formed
# and factorized on the nodes centred and scaled by one isotropic factor
# (`HartleyNormalization`) and mapped back exactly to the caller's frame (see
# `ThinPlateSplineSystem`), which moves (w, a) only at rounding level — on
# pixel nodes the unconditioned system's condition number was ~1e18.
#
# Two halves, as in RadiometricCamera: the solver object `ThinPlateSplineSystem`
# (RadiometricCamera's `ThinPlateSpline`, renamed here because that name is the
# target calibration kind's, geometry/cameras/targets.jl) holds heap matrices and stays host-side; the evaluator takes isbits inputs
# (an `SVector` of nodes, `w`, `a`), allocates nothing and runs in a
# KernelAbstractions kernel on the CPU backend (tested; no device backend is),
# in any float type or `ForwardDiff.Dual`.
#
# Element types: `tps_evaluate` PROMOTES the nodes with (w, 𝐩), while
# `tps_gradient` CONVERTS the nodes to the promoted type of (w, 𝐩) — the rules
# of RadiometricCamera's `tps_evaluate` and `board_height_grad`, kept as they
# are and deliberately not unified (either change would move bits). Supplying the
# nodes in the evaluation's type (Float32 nodes for a Float32 evaluation) makes
# the two agree.
# =============================================================================

"""
    tps_radial_basis(𝐩, 𝐪) -> scalar

The thin-plate spline's radial basis function `U(𝐩, 𝐪) = ½ r² ln r²` with
`r = ‖𝐩 − 𝐪‖` (the same as `r² ln r`), and `U = 0` at `r = 0`. The zero branch
is taken on `r² > 0`, not by evaluating `0 · log 0`, so a `ForwardDiff.Dual`
evaluated on a node gets a zero value with zero partials (the true derivative
there) instead of `NaN`.
"""
@inline function tps_radial_basis(𝐩::StaticVector{2}, 𝐪::StaticVector{2})
    r2 = (𝐩[1] - 𝐪[1])^2 + (𝐩[2] - 𝐪[2])^2
    return r2 > 0 ? r2 * log(r2) / 2 : zero(r2)
end

"""
    ThinPlateSplineSystem(nodes) -> ThinPlateSplineSystem{N}

The solver object of a thin-plate spline through `N` nodes in the plane: the
node positions, the `(N + 3) × N` matrix `coeff` that maps nodal values to the
spline's weights and affine part, `(w; a) = coeff · values`, and the `N × N`
bending-energy form `Q` (`valuesᵀ Q values = wᵀ K w`, `K = [U(𝐩ᵢ, 𝐩ⱼ)]`, see
[`bending_energy`](@ref)), and, read by the target calibration kinds' pure-bending
rule ([`is_pure_bending`](@ref), [`tps_pure_bending`](@ref): the plane fields
`[1 𝐩_x 𝐩_y]` are formed from the nodes), the `3 × N` affine rows of `coeff`,
`affine` (`a = affine · values`).

Everything it holds is in the CALLER's frame (frame in = frame out): the
nodes as given, `(w, a)` of the spline `f(𝐩)` in those coordinates, and `Q`
the bending form of those coordinates (`valuesᵀ Q values = wᵀ K w`, `K` of the
given nodes: the values' unit squared over the nodes' unit squared, a fixed
multiple of the bending energy `J = ∫ f_xx² + 2f_xy² + f_yy²` in the given
coordinates).

The system is SOLVED in a conditioned frame, as an estimator's private
normalization: the nodes are centred on their centroid and scaled by one
isotropic factor σ ([`HartleyNormalization`](@ref): mean distance √2 from the
centroid), `𝐩′ = σ 𝐩 + 𝐭`. An interpolating thin-plate spline is the same
surface under a similarity of its domain, so only rounding changes: the
weights map back as `w = σ² w′`, the linear part as `(a_x, a_y) = σ (a′_x,
a′_y)`, the constant as `a₁ = a′₁ + a′ · 𝐭 + ln σ Σᵢ w′ᵢ |𝐩′ᵢ|²` (the kernel's
`σ² ln σ · r²` term, a constant under the side conditions `Σ w′ᵢ = 0`, `Σ w′ᵢ
𝐩′ᵢ = 0`), and `Q = σ² Q′`. (A PER-AXIS scaling is not a similarity: the
spline through the same values then bends differently, so it is never used
here.) Without it, nodes in pixels (thousands) put `r² ln r²` near 1e8 beside
the unit column of `P`.

`coeff` is mapped back from the first `N` columns of the inverse of the
conditioned system matrix `[K′ P′; P′ᵀ 0]`, solved for those `N` right-hand
sides by one LU factorization. The matrices live on
the host; [`tps_coefficients`](@ref) turns nodal values into the isbits `(w,
a)` that [`tps_evaluate`](@ref) and [`tps_gradient`](@ref) take.
"""
struct ThinPlateSplineSystem{N, T <: AbstractFloat, M <: AbstractMatrix{T}}
    nodes::SVector{N, SVector{2, T}}
    coeff::M                    # (N+3) × N, the caller's frame
    Q::M                        # N × N, the caller's frame
    affine::M                   # 3 × N: coeff's affine rows, a = affine · values
end
ThinPlateSplineSystem{N}(nodes::SVector{N, SVector{2, T}}, coeff::M, Q::M, affine::M) where {N, T, M <: AbstractMatrix{T}} =
    ThinPlateSplineSystem{N, T, M}(nodes, coeff, Q, affine)

# The device form (`Adapt.adapt(backend, system)`): the matrices move, the
# (isbits) nodes travel as they are. In a kernel only `coeff` is read
# (`tps_coefficients`, `tps_pure_bending`); the rest moves with it so the
# adapted system is one object of the host's layout.
Adapt.adapt_structure(to, s::ThinPlateSplineSystem{N}) where {N} =
    ThinPlateSplineSystem{N}(s.nodes, Adapt.adapt(to, s.coeff), Adapt.adapt(to, s.Q), Adapt.adapt(to, s.affine))
# (`convert_eltype` of a system: geometry/cameras/target_calibration.jl, beside the kinds')

function ThinPlateSplineSystem(nodes::AbstractVector{<:StaticVector{2}})
    N = length(nodes)
    ps = [SVector{2, Float64}(p) for p in nodes]
    A = AffineMap(HartleyNormalization(), ps)          # 𝐩′ = σ 𝐩 + 𝐭, one isotropic σ
    σ = Float64(A.linear[1, 1]); 𝐭 = SVector{2, Float64}(A.translation)
    qs = [A(p) for p in ps]
    Umat = [tps_radial_basis(qs[i], qs[j]) for i in 1:N, j in 1:N]
    P′ = [ones(N) getindex.(qs, 1) getindex.(qs, 2)]
    # The first N columns of the inverse of the saddle-point matrix: one factorization, N right-hand sides. LU with
    # partial pivoting, not Bunch–Kaufman: it keeps the pure-bending projection's affine part (a = 0) an order of
    # magnitude closer to zero (1.0e-15 against 8.7e-15 relative, the 4 × 4 grid of the target calibration tests).
    C′ = lu([Umat P′; P′' zeros(3, 3)]) \ [Matrix{Float64}(I, N, N); zeros(3, N)]
    Cw′ = C′[1:N, :]
    coeff = Matrix{Float64}(undef, N + 3, N)
    coeff[1:N, :] .= σ^2 .* Cw′                                                         # w = σ² w′
    coeff[N + 1, :] .= C′[N + 1, :] .+ 𝐭[1] .* C′[N + 2, :] .+ 𝐭[2] .* C′[N + 3, :] .+
                       log(σ) .* (Cw′' * [sum(abs2, q) for q in qs])                   # a₁ = a′₁ + a′·𝐭 + ln σ Σ w′ᵢ|𝐩′ᵢ|²
    coeff[N + 2, :] .= σ .* C′[N + 2, :]                                                # (a_x, a_y) = σ (a′_x, a′_y)
    coeff[N + 3, :] .= σ .* C′[N + 3, :]
    Q = Matrix(Symmetric(σ^2 .* (Cw′' * Umat * Cw′)))                                   # Q = σ² Q′
    return ThinPlateSplineSystem{N}(SVector{N}(ps), coeff, Q, coeff[(N + 1):(N + 3), :])
end

"""
    grid_nodes(n; lo = -0.5, hi = 0.5) -> Vector{SVector{2, Float64}}
    grid_nodes(n, r::Rect) -> Vector{SVector{2, Float64}}

The `n × n` equally spaced nodes over the square `[lo, hi]²`, or over the
rectangle `r` (its corners included: a sensor's `Rect(sensor)` in pixels),
column-major (`x` fastest).
"""
function grid_nodes(n::Integer; lo::Real = -0.5, hi::Real = 0.5)
    g = range(lo, hi; length = n)
    return [SVector(float(g[i]), float(g[j])) for j in 1:n for i in 1:n]
end
function grid_nodes(n::Integer, r::Rect{2})
    o, w = r.origin, r.widths
    gx = range(Float64(o[1]), Float64(o[1] + w[1]); length = n)
    gy = range(Float64(o[2]), Float64(o[2] + w[2]); length = n)
    return [SVector(gx[i], gy[j]) for j in 1:n for i in 1:n]
end

"""
    tps_coefficients(tps::ThinPlateSplineSystem{N}, values) -> (w::SVector{N}, a::SVector{3})

The spline's weights `w` and affine part `a = (a₁, a_x, a_y)` through the
nodal values `values` (length `N`). Linear in `values`, and generic in their
element type: a `ForwardDiff.Dual` or a `Float32` is kept (the product
with the `Float64` matrix is formed first, then converted to the values'
element type).
"""
function tps_coefficients(tps::ThinPlateSplineSystem{N}, t::AbstractVector) where {N}
    length(t) == N || _refuse_value_count(N, t)
    return _tps_coefficients(tps.coeff, Val(N), t)
end
@noinline _refuse_value_count(N, t) = throw(DimensionMismatch("expected $N nodal values, got $(length(t))"))
# THE map from nodal values to (w, a): row r of `coeff` times the values, a
# loop in the values' order (no BLAS: device-safe and allocation-free, generic
# in the element type). Accumulated in the promoted type of the system and the
# values (a Float64 system keeps a Float32 value vector's sums in Float64, then
# rounds; a Float32 device system sums in Float32), returned in the values'
# float type.
@inline function _tps_coefficients(C::AbstractMatrix, ::Val{N}, t::AbstractVector) where {N}
    T = float(eltype(t))
    S = promote_type(eltype(C), T)
    o = firstindex(t) - 1
    row(r) = (acc = zero(S); @inbounds(for j in 1:N; acc += C[r, j] * t[o + j]; end); T(acc))
    return SVector{N, T}(ntuple(row, Val(N))), SVector{3, T}(row(N + 1), row(N + 2), row(N + 3))
end

"""
    tps_pure_bending(tps::ThinPlateSplineSystem{N}, values) -> SVector{N}

The nodal values without the affine term of their interpolant — the
pure-bending rule of the thin-plate target calibration kinds
([`HeightField`](@ref), [`ThinPlateSpline`](@ref)): `v ↦ v − M a(v)`, `M =
[1 u v]` the plane fields at the nodes and `a(v)` the interpolant's affine
coefficients ([`tps_coefficients`](@ref)). The spline through `M a` is that
plane exactly, so the projection keeps the weights `w` and zeroes `a`; it is
oblique (along the plane fields, not orthogonal to them: orthogonal to the
affine rows of `coeff`) and idempotent. Values whose plane part is within
`1e-12` of them (`‖M a(v)‖ ≤ 1e-12 ‖v‖`, the rule of
[`is_pure_bending`](@ref)) are kept as they are, bit for bit — a ForwardDiff
`Dual`'s partials too, the derivative along the pure-bending subspace (where the
projection is the identity); other values are projected, values and partials. Loops only: allocation-free, generic in the element type and
callable in a KernelAbstractions kernel on an adapted system.
"""
function tps_pure_bending(tps::ThinPlateSplineSystem{N}, v::AbstractVector) where {N}
    length(v) == N || _refuse_value_count(N, v)
    return _tps_pure_bending(tps, SVector{N, float(eltype(v))}(ntuple(i -> v[firstindex(v) + i - 1], Val(N))))
end
@inline function _tps_pure_bending(tps::ThinPlateSplineSystem{N}, v::SVector{N, T}) where {N, T}
    A = tps.affine
    S = promote_type(eltype(A), T)
    arow(k) = (acc = zero(S); @inbounds(for j in 1:N; acc += A[k, j] * v[j]; end); acc)
    a1, a2, a3 = arow(1), arow(2), arow(3)
    nodes = tps.nodes
    plane = SVector{N, S}(ntuple(i -> a1 + a2 * nodes[i][1] + a3 * nodes[i][2], Val(N)))
    # within the tolerance the values are kept as they are, a Dual's partials too: the
    # derivative ALONG the pure-bending subspace, where the projection is the identity
    # (the kinds' `transform_deriv_params`, their nodal weights)
    _within_rtol(_primal_values(plane), _primal_values(v)) && return v
    return SVector{N, T}(ntuple(i -> v[i] - plane[i], Val(N)))
end
# The relative size of an affine term below which nodal values count as pure
# bending: far above the rounding of a projection, far below any real plane.
const _PLANE_PART_RTOL = 1.0e-12
@inline _within_rtol(plane, v) = sqrt(sum(abs2, plane)) <= _PLANE_PART_RTOL * sqrt(sum(abs2, v))
@inline _primal_values(v::StaticVector) = map(_primal, v)


"""
    tps_evaluate(nodes, w, a, 𝐩) -> scalar
    tps_evaluate(tps::ThinPlateSplineSystem, w, a, 𝐩) -> scalar

The spline at `𝐩`: `a₁ + a_x 𝐩_x + a_y 𝐩_y + Σᵢ wᵢ U(𝐩, 𝐩ᵢ)` with `(w, a)` from
[`tps_coefficients`](@ref) and `U` the [`tps_radial_basis`](@ref). The first form
takes isbits inputs only (`nodes::SVector{N, <:StaticVector{2}}`, `w`, `a`,
`𝐩`): it allocates nothing and runs in a KernelAbstractions kernel on the CPU
backend. It is generic in the element type and PROMOTES the nodes with `w` and
`𝐩`: a `Float32` evaluation with `Float64` nodes returns `Float64`. This
differs from [`tps_gradient`](@ref), which converts the nodes, as
RadiometricCamera's two functions do; supply the nodes in the evaluation's type
(`Float32` nodes for `Float32`) and both work in that type.
"""
@inline function tps_evaluate(nodes::SVector{N, <:StaticVector{2}}, w::StaticVector{N}, a::StaticVector{3}, 𝐩::StaticVector{2}) where {N}
    a_1, a_x, a_y = a[1], a[2], a[3]
    s = a_1 + a_x * 𝐩[1] + a_y * 𝐩[2]
    for i in 1:N
        s += w[i] * tps_radial_basis(𝐩, nodes[i])
    end
    return s
end
tps_evaluate(tps::ThinPlateSplineSystem, w, a, 𝐩) = tps_evaluate(tps.nodes, w, a, 𝐩)

"""
    tps_gradient(nodes, w, a, 𝐩) -> SVector{2}
    tps_gradient(tps::ThinPlateSplineSystem, w, a, 𝐩) -> SVector{2}

The analytic gradient `∇f(𝐩)` of [`tps_evaluate`](@ref): `(a_x, a_y) + Σᵢ wᵢ
(𝐩 − 𝐩ᵢ)(ln r² + 1)`, since `∇_𝐩 ½ r² ln r² = (𝐩 − 𝐩ᵢ)(ln r² + 1)`; a node at
`r = 0` contributes zero. The sum is RadiometricCamera's `board_height_grad`
term for term (its height field is this spline without the affine part, which
is added last). Isbits inputs and allocation-free, and it runs in a
KernelAbstractions kernel on the CPU backend, as `tps_evaluate`. Unlike
[`tps_evaluate`](@ref), which promotes, it CONVERTS the node coordinates to the
promoted element type of `w` and `𝐩` (RadiometricCamera's `board_height_grad`
rule): `Float32` `w` and `𝐩` with `Float64` nodes give a `Float32` gradient.
Supply the nodes in the evaluation's type and the two agree.
"""
@inline function tps_gradient(nodes::SVector{N, <:StaticVector{2}}, w::StaticVector{N}, a::StaticVector{3}, 𝐩::StaticVector{2}) where {N}
    S = promote_type(eltype(w), eltype(𝐩))
    g1 = zero(S); g2 = zero(S)
    @inbounds for i in 1:N
        n = nodes[i]
        d1 = 𝐩[1] - S(n[1]); d2 = 𝐩[2] - S(n[2])
        r2 = d1 * d1 + d2 * d2
        f = r2 > 0 ? w[i] * (log(r2) + 1) : zero(S)
        g1 += f * d1; g2 += f * d2
    end
    return SVector(a[2] + g1, a[3] + g2)
end
tps_gradient(tps::ThinPlateSplineSystem, w, a, 𝐩) = tps_gradient(tps.nodes, w, a, 𝐩)

"""
    nodal_weights(tps::ThinPlateSplineSystem{N}, 𝐩) -> SVector{N, Float64}

The sensitivity of the spline at `𝐩` to each nodal value, `∂f(𝐩)/∂valuesᵢ`.
With `f(𝐩) = φ(𝐩)ᵀ (coeff · values)` and `φ = (U(𝐩, 𝐩₁) … U(𝐩, 𝐩_N), 1, 𝐩_x,
𝐩_y)`, node `i`'s weight is `Σ_r coeff[r, i] φ_r(𝐩)` — exact, since the spline
is linear in its values. Host-side (it reads `coeff`), in `Float64` for a
real point (a `Float32` point included); a ForwardDiff `Dual` point gives
`Dual` weights. Allocation-free at every `N`.
"""
function nodal_weights(tps::ThinPlateSplineSystem{N}, 𝐩::StaticVector{2}) where {N}
    N <= _NODAL_STATIC_MAX || return _nodal_weights_loop(tps, 𝐩)
    S = _nodal_eltype(𝐩)
    φ = ntuple(r -> r <= N ? S(tps_radial_basis(𝐩, tps.nodes[r])) : (r == N + 1 ? one(S) : (r == N + 2 ? S(𝐩[1]) : S(𝐩[2]))), Val(N + 3))
    c = tps.coeff
    return SVector{N}(ntuple(i -> sum(ntuple(r -> c[r, i] * φ[r], Val(N + 3))), Val(N)))
end
# The basis and weights are Float64 for any real point: `float(promote_type(…))`
# keeps Float64 for a Float32 point (as `Float64(𝐩[1])` did) and a Dual for a
# Dual point.
_nodal_eltype(𝐩::StaticVector{2}) = float(promote_type(Float64, eltype(𝐩)))
# The unrolled form compiles N(N + 3) terms: past this many nodes (a 3 × 3 grid
# and a little more) the same sums run as loops, left to right as `sum` folds a
# tuple, so both forms give the same bits. The loop form's basis is a stack
# `MVector`, not a heap `Vector`: RadiometricCamera calls this per pixel per
# sample with 25 nodes, where 288 B a call was the whole garbage stream.
const _NODAL_STATIC_MAX = 16
function _nodal_weights_loop(tps::ThinPlateSplineSystem{N}, 𝐩::StaticVector{2}) where {N}
    c = tps.coeff
    S = _nodal_eltype(𝐩)
    φ = MVector{N + 3, S}(undef)
    for r in 1:N
        φ[r] = tps_radial_basis(𝐩, tps.nodes[r])
    end
    φ[N + 1] = one(S); φ[N + 2] = S(𝐩[1]); φ[N + 3] = S(𝐩[2])
    out = MVector{N, S}(undef)
    @inbounds for i in 1:N
        s = c[1, i] * φ[1]
        for r in 2:(N + 3)
            s += c[r, i] * φ[r]
        end
        out[i] = s
    end
    return SVector(out)
end

"""
    bending_energy(tps::ThinPlateSplineSystem, values) -> scalar
    bending_energy(sys::SphericalSplineSystem, values) -> scalar

The bending energy `valuesᵀ Q values` of the spline through the nodal values
`values` (`= wᵀ K w` with `K = [U(𝐩ᵢ, 𝐩ⱼ)]` the radial basis between the
nodes; zero for affine values up to rounding); a quadratic regularizer on the
values.
"""
bending_energy(tps::ThinPlateSplineSystem, t::AbstractVector) = dot(t, tps.Q * t)

"""
    bending_factor(tps::ThinPlateSplineSystem{N}) -> Matrix{Float64}
    bending_factor(sys::SphericalSplineSystem{N}) -> Matrix{Float64}

A factor `L` (`N × N`) of the bending-energy form: `‖L · values‖² =`
[`bending_energy`](@ref)`(tps, values)` to rounding, so `L · values` are a
regularizer's residual rows. `Q` is positive semidefinite with the affine
values in its null space, so it has no Cholesky factor: `L = Λ₊^½ Vᵀ` from the
eigendecomposition `Q = V Λ Vᵀ`, `Λ₊` the eigenvalues clamped at zero (the null
space's rounding residue may come out negative). An eigenbasis is not unique,
so only `LᵀL` is defined — never compare the rows one by one.
"""
function bending_factor(tps::ThinPlateSplineSystem)
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(tps.Q))
    return LinearAlgebra.Diagonal(sqrt.(max.(F.values, 0.0))) * F.vectors'
end

# =============================================================================
# Spherical spline on scattered directions (hoisted from RadiometricCamera
# src/camera/focus_charts.jl, `wahba_kernel` and `sphere_spline_system`, the
# arithmetic unchanged) — the thin-plate spline's analogue on the unit sphere:
#
#   f(d) = a₁ + Σᵢ wᵢ R(d · dᵢ),   R(z) = 1 − π²/6 + Li₂((1 + z)/2)
#
# Wahba's m = 2 reproducing kernel (Wahba 1981), with (w; a₁) the solution of
# [K 1; 1ᵀ 0] (w; a₁) = (values; 0), K = [R(dᵢ · dⱼ)]. As for the plane: the
# system's inverse is formed once (`inv`, as RadiometricCamera forms it), the
# solver object stays host-side, the evaluator takes isbits inputs.
# =============================================================================

"""
    spherical_spline_kernel(d, e) -> scalar

Wahba's `m = 2` spherical-spline kernel between unit directions `d` and `e`,
`R(z) = 1 − π²/6 + Li₂((1 + z)/2)` with `z = d · e`, evaluated from the chord
(`1 − (1 + z)/2 = ‖d − e‖²/4`) so it stays accurate for nearby directions:
`R(1) = 1` (with zero partials at coincidence, as the planar
[`tps_radial_basis`](@ref) at `r = 0`), `R(−1) = 1 − π²/6`. Isbits, generic
over `Real` ([`dilog`](@ref)'s series).
"""
@inline function spherical_spline_kernel(d::StaticVector{3}, e::StaticVector{3})
    y = ((d[1] - e[1])^2 + (d[2] - e[2])^2 + (d[3] - e[3])^2) / 4      # 1 − x = sin²(γ/2)
    y > 0 || return one(y)
    x = 1 - y
    return x >= 0.5 ? 1 - log(x) * log(y) - _dilog_series(y) : (1 - oftype(y, π^2 / 6)) + _dilog_series(max(x, zero(x)))
end

"""
    SphericalSplineSystem(directions) -> SphericalSplineSystem{N}

The solver object of a spherical spline through `N` unit directions — the
[`ThinPlateSplineSystem`](@ref)'s analogue on the sphere, with Wahba's kernel
[`spherical_spline_kernel`](@ref) and a constant in place of the affine part:
the directions, the `(N + 1) × N` matrix `coeff` that maps nodal values to the
weights and the constant, `(w; a₁) = coeff · values`, and the `N × N`
bending-energy form `Q` (`valuesᵀ Q values = wᵀ K w`). `coeff` is the first
`N` columns of the inverted system matrix `[K 1; 1ᵀ 0]`, formed with `inv`
as RadiometricCamera forms it. The directions are normalized on entry.
"""
struct SphericalSplineSystem{N}
    nodes::SVector{N, SVector{3, Float64}}
    coeff::Matrix{Float64}      # (N+1) × N
    Q::Matrix{Float64}          # N × N
end

function SphericalSplineSystem(directions::AbstractVector{<:StaticVector{3}})
    N = length(directions)
    dirs = [normalize(SVector{3, Float64}(d)) for d in directions]
    Kmat = [spherical_spline_kernel(dirs[i], dirs[j]) for i in 1:N, j in 1:N]
    A = [Kmat ones(N); ones(N)' 0.0]
    Ainv = inv(A)   # `inv`, as RadiometricCamera forms it
    coeff = Ainv[:, 1:N]
    Cw = coeff[1:N, :]
    Q = Matrix(Symmetric(Cw' * Kmat * Cw))
    return SphericalSplineSystem{N}(SVector{N}(dirs), coeff, Q)
end

"""
    spherical_spline_coefficients(sys::SphericalSplineSystem{N}, values) -> (w::SVector{N}, a₁)

The spline's weights and constant through the nodal values (length `N`);
linear in them and generic in their element type, as
[`tps_coefficients`](@ref).
"""
function spherical_spline_coefficients(sys::SphericalSplineSystem{N}, t::AbstractVector{T}) where {N, T}
    length(t) == N || throw(DimensionMismatch("expected $N nodal values, got $(length(t))"))
    F = float(T)                                       # integer nodal values give float coefficients
    wa = sys.coeff * t
    return SVector{N, F}(ntuple(i -> wa[i], Val(N))), F(wa[N + 1])
end

"""
    spherical_spline_evaluate(nodes, w, a₁, d) -> scalar
    spherical_spline_evaluate(sys::SphericalSplineSystem, w, a₁, d) -> scalar

The spline at the unit direction `d`: `a₁ + Σᵢ wᵢ R(d · dᵢ)`. The first form
takes isbits inputs (`nodes::SVector{N, <:StaticVector{3}}`) and allocates
nothing; generic in the element type (Float32, ForwardDiff `Dual`s).
"""
@inline function spherical_spline_evaluate(nodes::SVector{N, <:StaticVector{3}}, w::StaticVector{N}, a1, d::StaticVector{3}) where {N}
    s = a1 + zero(eltype(d))
    for i in 1:N
        s += w[i] * spherical_spline_kernel(d, nodes[i])
    end
    return s
end
spherical_spline_evaluate(sys::SphericalSplineSystem, w, a1, d) = spherical_spline_evaluate(sys.nodes, w, a1, d)

"""
    nodal_weights(sys::SphericalSplineSystem{N}, d) -> SVector{N, Float64}

`∂f(d)/∂valuesᵢ` of the spherical spline at the direction `d`: node `i`'s
weight `Σ_r coeff[r, i] φ_r(d)`, `φ = (R(d · d₁) … R(d · d_N), 1)`.
"""
function nodal_weights(sys::SphericalSplineSystem{N}, d::StaticVector{3}) where {N}
    φ = ntuple(r -> r <= N ? Float64(spherical_spline_kernel(d, sys.nodes[r])) : 1.0, Val(N + 1))
    c = sys.coeff
    return SVector{N}(ntuple(i -> sum(ntuple(r -> c[r, i] * φ[r], Val(N + 1))), Val(N)))
end

bending_energy(sys::SphericalSplineSystem, t::AbstractVector) = dot(t, sys.Q * t)
function bending_factor(sys::SphericalSplineSystem)
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(sys.Q))
    return LinearAlgebra.Diagonal(sqrt.(max.(F.values, 0.0))) * F.vectors'
end

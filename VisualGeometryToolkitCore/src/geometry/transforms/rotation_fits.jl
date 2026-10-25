# =============================================================================
# Closed-form rotation algorithms — Kabsch/Wahba and the chordal mean
# =============================================================================
#
# The two SVD-based rotation ALGORITHMS that kept reappearing as private
# copies downstream (hand-eye initialization, rotation averaging, mocap
# alignment). Both reduce to Rotations.jl's exported `nearest_rotation`
# (SO(3) projection) applied to an accumulated matrix — that projection
# already lives upstream and is deliberately not duplicated here; only the
# estimation-shaped wrappers are VGC's to own.
#
# NOT AD-safe under Zygote (SVD rule limitations); use outside tracked code.

"""
    kabsch(cs) -> RotMatrix{3, T}

Kabsch/Wahba: the rotation minimizing `Σᵢ ‖bᵢ − R·aᵢ‖²` over the vector
pairs `cs` (`aᵢ => bᵢ`, `StaticVector{3}` each side; any
[`sources`](@ref)/[`targets`](@ref) container, a [`Csponds`](@ref)
preferred): one closed-form step, `Rotations.nearest_rotation` of the
cross-covariance `Σ bᵢaᵢᵀ` (an SVD with the determinant-sign correction).
The vectors are used as given, NOT centred: centre (and weight) them at the
call site when aligning point clouds rather than directions.

Physical pairs (lengths on both sides) are stripped in metres at entry; the
rotation is unit-free, so it comes back the same either way.

Throws `ArgumentError` on fewer than 2 pairs (a single pair leaves a
one-parameter family of solutions).
"""
kabsch(cs::AbstractVector{<:Pair{<:StaticVector{3, <:Quantity}, <:StaticVector{3, <:Quantity}}}) =
    kabsch([_strip_metres(first(p)) => _strip_metres(last(p)) for p in cs])
function kabsch(cs::AbstractVector{<:Pair{<:StaticVector{3}, <:StaticVector{3}}})
    length(cs) >= 2 ||
        throw(ArgumentError("need at least 2 vector pairs, got $(length(cs))"))
    as = sources(cs); bs = targets(cs)
    T = promote_type(eltype(eltype(as)), eltype(eltype(bs)))
    H = sum(SVector{3, T}(b) * SVector{3, T}(a)' for (a, b) in zip(as, bs))
    return Rotations.nearest_rotation(H)
end

"""
    chordal_mean(Rs) -> RotMatrix{3, T}

Chordal L2 mean of a set of rotations: the rotation minimizing
`Σᵢ ‖R − Rᵢ‖²_F`, i.e. `Rotations.nearest_rotation` of the elementwise sum.
Suited to averaging rotations within a moderate spread (the chordal mean and
the geodesic mean agree to second order); it is NOT a robust estimator.

Throws `ArgumentError` on an empty collection.
"""
function chordal_mean(Rs::AbstractVector{<:Union{Rotation{3}, StaticMatrix{3, 3}}})
    isempty(Rs) && throw(ArgumentError("cannot average an empty set of rotations"))
    T = eltype(first(Rs))
    return Rotations.nearest_rotation(sum(SMatrix{3, 3, T, 9}(R) for R in Rs))
end

# -----------------------------------------------------------------------------
# Their stationarity conditions — what the implicit-function theorem
# differentiates for the covariance of an estimator with a Kabsch or a
# chordal-mean stage (VisualGeometryToolkitEstimation's staged robot-world fit). Both are
# the gradient of the algorithm's objective under the left perturbation
# R → exp([ω]×)·R, zero at the algorithm's rotation; eltype-generic, so
# ForwardDiff Duals flow through R and the data.
# -----------------------------------------------------------------------------

# Kabsch maximizes Σᵢ bᵢᵀ R aᵢ: its gradient in ω is Σᵢ (R aᵢ) × bᵢ.
kabsch_stationarity(R::StaticMatrix{3, 3}, cs) = sum(c -> cross(R * first(c), last(c)), cs)

# The chordal mean maximizes tr(Rᵀ S), S = Σᵢ Rᵢ — the Kabsch objective of the
# basis pairs eⱼ => S eⱼ, whose condition this is.
function chordal_mean_stationarity(R::StaticMatrix{3, 3}, Rs::AbstractVector)
    S = sum(SMatrix{3, 3}(Ri) for Ri in Rs)
    e = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))
    return kabsch_stationarity(R, [e[j] => S[:, j] for j in 1:3])
end

"""
    mirror(pose, plane_point; normal = SVector(0., 0., 1.)) -> EuclideanMap

The second branch of the planar two-fold pose ambiguity: for a PLANAR target
(world-plane normal `normal`, any point `plane_point` on it — conventionally
the target center), the pose whose plane normal is reflected about the
viewing ray to `plane_point`. Under weak perspective this mirror pose
predicts image positions that agree with `pose` to second order, so the two
tie on reprojection consensus; the rotation between them is twice the
viewing tilt. Total: at a frontal view the reflection is (numerically) the
identity and the returned pose coincides with `pose` — gate on
`rotation_angle(RotMatrix{3}(mirror.R' * pose.R))` if the degenerate case must be skipped.

Translation is carried over unchanged (same center ray and apparent size ⇒
same depth to first order); the result is a first-order INIT for the second
basin, to be refined before any comparison.
"""
function mirror(
        pose::PlainEuclideanMap{3}, plane_point::StaticVector{3};
        normal::StaticVector{3} = SVector(0.0, 0.0, 1.0)
    )
    n_c = pose.R * SVector{3, Float64}(normal)
    v = pose.R * SVector{3, Float64}(plane_point) + pose.t
    vn = norm(v)
    vn > 0 || return pose
    v̂ = v / vn
    ct = dot(n_c, v̂)
    n_ref = 2 * ct * v̂ - n_c
    R_δ = norm(cross(n_c, n_ref)) < 1.0e-12 ? one(RotMatrix{3, Float64}) :
        RotMatrix{3, Float64}(Rotations.rotation_between(n_c, n_ref))
    return EuclideanMap(RotMatrix{3, Float64}(R_δ * pose.R), pose.t; check = false)
end

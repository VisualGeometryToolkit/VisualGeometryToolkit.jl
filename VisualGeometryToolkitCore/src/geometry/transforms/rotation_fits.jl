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

# Kernel of [`Kabsch`](@ref), whose docstring is the one copy of its documentation: `_kabsch(cs) -> RotMatrix{3, T}`.
function _kabsch(cs::AbstractVector{<:Pair{<:StaticVector{3, <:Quantity}, <:StaticVector{3, <:Quantity}}})
    stripped = [strip_metres(first(p)) => strip_metres(last(p)) for p in cs]
    w = weights_column(cs)
    return _kabsch(w === nothing ? stripped : withattrs(Correspondences(sources(stripped), targets(stripped)); weights = w))
end
function _kabsch(cs::AbstractVector{<:Pair{<:StaticVector{3}, <:StaticVector{3}}})
    length(cs) >= 2 ||
        throw(ArgumentError("need at least 2 vector pairs, got $(length(cs))"))
    as = sources(cs); bs = targets(cs)
    w = check_weights(weights_column(cs), length(cs))
    # a zero weight drops its pair: the ≥ 2 counts the pairs that carry weight
    (w === nothing || count(>(0), w) >= 2) ||
        throw(ArgumentError("need at least 2 vector pairs of positive weight, got $(count(>(0), w)) of $(length(cs))"))
    T = promote_type(eltype(eltype(as)), eltype(eltype(bs)))
    H = w === nothing ? sum(SVector{3, T}(b) * SVector{3, T}(a)' for (a, b) in zip(as, bs)) :
        sum(w[i] * (SVector{3, T}(bs[i]) * SVector{3, T}(as[i])') for i in eachindex(as, bs, w))
    return Rotations.nearest_rotation(H)
end

# Kernel of [`ChordalMean`](@ref), whose docstring is the one copy of its documentation: `_chordal_mean(Rs) -> RotMatrix{3, T}`.
function _chordal_mean(Rs::AbstractVector{<:Union{Rotation{3}, StaticMatrix{3, 3}}})
    isempty(Rs) && throw(ArgumentError("cannot average an empty set of rotations"))
    T = eltype(first(Rs))
    return Rotations.nearest_rotation(sum(SMatrix{3, 3, T, 9}(R) for R in Rs))
end

# Kernel of [`SpectralAveraging`](@ref), whose docstring is the one copy of its documentation: `_spectral_rotation_averaging(relative) -> Dict{K, RotMatrix{3, Float64}}`.
function _spectral_rotation_averaging(relative::AbstractDict{<:Tuple{K, K}}) where {K}
    isempty(relative) && throw(ArgumentError("SpectralAveraging: no relative rotations"))
    frames = sort!(unique!(collect(Iterators.flatten(keys(relative)))))
    for (p, e) in relative
        weight_of(e) >= 0 || throw(ArgumentError("SpectralAveraging: weight $(weight_of(e)) of the pair $p is negative"))
    end
    n = length(frames); idx = Dict(k => a for (a, k) in enumerate(frames))
    A = zeros(3n, 3n); deg = ones(n)
    for a in 1:n
        A[(3a - 2):(3a), (3a - 2):(3a)] .= I(3)
    end
    for ((i, j), e) in relative
        w = weight_of(e)
        a, b = idx[i], idx[j]; M = w * SMatrix{3, 3, Float64, 9}(value_of(e))   # block (b, a) = R_bᵀR_a = R_ij
        A[(3b - 2):(3b), (3a - 2):(3a)] .+= M
        A[(3a - 2):(3a), (3b - 2):(3b)] .+= M'
        deg[a] += w; deg[b] += w
    end
    d = repeat(1 ./ sqrt.(deg); inner = 3)
    V = eigen(Symmetric(d .* A .* d'), (3n - 2):(3n)).vectors[:, 3:-1:1] .* d   # the three leading eigenvectors only (LAPACK's range form)
    blocks = [SMatrix{3, 3, Float64, 9}(V[(3a - 2):(3a), :]) for a in 1:n]
    count(B -> det(B) < 0, blocks) > n / 2 && (blocks = [B * Diagonal(SVector(1.0, 1.0, -1.0)) for B in blocks])
    Rt = [SMatrix{3, 3, Float64, 9}(Rotations.nearest_rotation(B)) for B in blocks]   # ≈ R_kᵀ·Q
    return Dict(k => RotMatrix{3, Float64, 9}(Rt[1] * Rt[a]') for (a, k) in enumerate(frames))   # gauge: R_first = I
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

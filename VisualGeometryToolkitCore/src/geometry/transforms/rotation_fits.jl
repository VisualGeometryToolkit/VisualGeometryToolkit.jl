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
A `weights` attribute column (`withattrs(cs; weights = w)`, `data-arguments.md`)
weighs the pairs, `Σᵢ wᵢ‖bᵢ − R·aᵢ‖²` and the cross-covariance `Σ wᵢbᵢaᵢᵀ`
(Wahba's problem): fixed, finite, non-negative weights, one per pair (an
`ArgumentError` otherwise); a zero weight drops its pair. The vectors are
used as given, NOT centred: centre them at the call site when aligning point
clouds rather than directions.

Physical pairs (lengths on both sides) are stripped in metres at entry; the
rotation is unit-free, so it comes back the same either way.

Throws `ArgumentError` on fewer than 2 pairs, or fewer than 2 of positive
weight (a single pair leaves a one-parameter family of solutions).
"""
function kabsch(cs::AbstractVector{<:Pair{<:StaticVector{3, <:Quantity}, <:StaticVector{3, <:Quantity}}})
    stripped = [strip_metres(first(p)) => strip_metres(last(p)) for p in cs]
    w = weights_column(cs)
    return kabsch(w === nothing ? stripped : withattrs(Csponds(sources(stripped), targets(stripped)); weights = w))
end
function kabsch(cs::AbstractVector{<:Pair{<:StaticVector{3}, <:StaticVector{3}}})
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

"""
    chordal_averaging(relative[, weights]) -> Dict{K, RotMatrix{3, Float64}}

Chordal L2 rotation averaging (Martinec & Pajdla 2007; Hartley et al. 2013): the
absolute rotations `R_k` of a set of frames from their relative rotations,
`relative` an `AbstractDict` keyed by frame pairs `(i, j)` whose value `R_ij`
takes frame `i`'s directions into frame `j`'s (`f_j = R_ij·f_i`), so that with
`d = R_k·f_k` a direction in the common frame, `R_i = R_j·R_ij`. One linear least
squares over the 9 entries of every `R_k`, `Σ w_ij ‖R_i − R_j R_ij‖²_F`, the
gauge fixed by `R_k = I` for the first frame of the sorted keys (the frame
the result's directions are in), then each block projected onto SO(3)
(`Rotations.nearest_rotation`). `weights` (an `AbstractDict` with the same
keys, non-negative) weigh the pairs; all ones by default.

An algorithm (one solve, no iteration): the start of a rotation bundle, not a
robust estimator — a wrong relative rotation pulls every frame. Frames that no
pair connects to the gauge frame are an `ArgumentError`. Returns the frames in
a `Dict` keyed like the pairs' frames, the gauge frame's `R` the identity.
"""
function chordal_averaging(relative::AbstractDict{<:Tuple{K, K}}, weights::Union{Nothing, AbstractDict} = nothing) where {K}
    isempty(relative) && throw(ArgumentError("chordal_averaging: no relative rotations"))
    frames = sort!(unique!(collect(Iterators.flatten(keys(relative)))))
    _check_connected(frames, keys(relative))
    gauge = first(frames)
    col = Dict(k => 9 * (n - 2) for (n, k) in enumerate(frames) if k != gauge)    # the 9 unknowns of R_k start after col[k]
    m = 9 * length(relative); nunk = 9 * (length(frames) - 1)
    A = zeros(m, nunk); b = zeros(m)
    row = 0
    for ((i, j), Rij) in relative
        w = weights === nothing ? 1.0 : Float64(weights[(i, j)])
        w >= 0 || throw(ArgumentError("chordal_averaging: weight $w of the pair $((i, j)) is negative"))
        sw = sqrt(w); M = SMatrix{3, 3, Float64, 9}(Rij)
        for c in 1:3, r in 1:3                       # (R_i − R_j M)[r, c] = 0, column-major vec
            row += 1
            if i == gauge
                b[row] -= sw * (r == c)
            else
                A[row, col[i] + 3 * (c - 1) + r] += sw
            end
            for q in 1:3                             # (R_j M)[r, c] = Σ_q R_j[r, q] M[q, c]
                if j == gauge
                    b[row] += sw * (r == q) * M[q, c]
                else
                    A[row, col[j] + 3 * (q - 1) + r] -= sw * M[q, c]
                end
            end
        end
    end
    x = A \ b
    out = Dict{K, RotMatrix{3, Float64, 9}}(gauge => one(RotMatrix{3, Float64, 9}))
    for k in frames
        k == gauge && continue
        out[k] = RotMatrix{3, Float64, 9}(Rotations.nearest_rotation(SMatrix{3, 3, Float64, 9}(view(x, (col[k] + 1):(col[k] + 9)))))
    end
    return out
end

# Every frame reachable from the first through the pairs (either direction).
function _check_connected(frames, pairs)
    adj = Dict(k => Set{eltype(frames)}() for k in frames)
    for (i, j) in pairs
        push!(adj[i], j); push!(adj[j], i)
    end
    seen = Set([first(frames)]); stack = [first(frames)]
    while !isempty(stack)
        for n in adj[pop!(stack)]
            n in seen || (push!(seen, n); push!(stack, n))
        end
    end
    length(seen) == length(frames) ||
        throw(ArgumentError("chordal_averaging: the frames $(setdiff(frames, seen)) are not connected to $(first(frames)) by any pair"))
    return nothing
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

# =============================================================================
# Rigid-target invariance — a calibration-free pose-accuracy criterion
# =============================================================================
#
# Targets mounted rigidly on one body (boards printed on one panel, markers on
# one rig) have CONSTANT relative poses. With a target's estimated pose
# `T(u, m)` (target m → camera, in unit u — a frame), the relative pose of two
# targets observed in the same unit is
#
#     T(u, m)⁻¹ · T(u, n)  =  Y_m⁻¹ · Y_n
#
# where Y are the unknown mounting poses: the camera pose cancels, so the
# quantity is a constant of the body, and any variation across units is pure
# estimation error. No ground truth enters — its true value is never needed.
#
# It measures CONSISTENCY, not accuracy: a bias identical in every unit (a
# global scale error from wrong target dimensions) is invisible to it. It is
# the right instrument for tuning (thresholds move noise and robustness) and
# the wrong one for validating absolute scale.
# =============================================================================

"""
    rigid_invariance_errors(poses; targets, units = sort!(collect(keys(poses))), min_obs = 3)
        -> (; translation::Matrix{Float64}, rotation::Matrix{Float64}, units, pairs)

The invariance errors of rigidly mounted targets. `poses[u][m]` is target
`m`'s estimated pose in unit `u` (a value with fields `R`, the rotation, and
`t`, the translation: target → camera, `x_cam = R·X + t`, e.g. an
`Extrinsics3`); a target absent from `poses[u]` was not observed there.
`targets` lists every target that could be observed; `pairs` are its sorted
pairs `(m, n)`, `m < n`.

For each pair and each unit observing both targets, the relative pose
`T_m⁻¹ T_n` is compared with a LEAVE-ONE-OUT reference over the other units
observing that pair: the rotation MEDOID (the observation minimising summed
geodesic distance to the others — robust to the two-fold flips a mean would
be dragged by) and the componentwise MEDIAN translation. Leave-one-out is not
optional: a reference containing the scored observation scores it zero by
construction and rewards observing each pair in fewer units.

Returns `length(units) × length(pairs)` matrices: `translation` (the
translation's distance from the reference, in the poses' units) and
`rotation` (geodesic angle, degrees). An entry is `Inf` where the pair was
not observed in that unit, or was observed in fewer than `max(min_obs, 3)`
units (no reference) — so under [`eligible_maa`](@ref) (THE mAA rule: a
non-finite error scores 0 and stays in the denominator) a missed pair counts
as a failure, and `eligible_maa(e, isfinite.(e))` is the accuracy over what
was observed.
"""
function rigid_invariance_errors(poses::AbstractDict; targets, units = sort!(collect(keys(poses))),
                                 min_obs::Integer = 3)
    ts = sort!(collect(targets))
    allunique(ts) || throw(ArgumentError("rigid_invariance_errors: targets are not unique"))
    tset = Set(ts)
    pairs = _sorted_pairs(ts)
    pair_index = Dict(p => i for (i, p) in enumerate(pairs))
    unit_index = Dict(u => i for (i, u) in enumerate(units))
    length(unit_index) == length(units) || throw(ArgumentError("rigid_invariance_errors: units are not unique"))
    Obs = Tuple{SMatrix{3, 3, Float64, 9}, SVector{3, Float64}, Int}
    rel = Dict{Int, Vector{Obs}}()
    for u in units
        haskey(poses, u) || continue
        ui = unit_index[u]
        seen = sort!(collect(keys(poses[u])))
        all(in(tset), seen) || throw(ArgumentError(
            "rigid_invariance_errors: unit $(repr(u)) observes targets not in `targets`: $(setdiff(seen, ts))"))
        for (m, p) in _sorted_pairs(seen)
            Pm, Pn = poses[u][m], poses[u][p]
            Rm = SMatrix{3, 3, Float64}(Pm.R); Rn = SMatrix{3, 3, Float64}(Pn.R)
            tm = SVector{3, Float64}(Pm.t); tn = SVector{3, Float64}(Pn.t)
            push!(get!(rel, pair_index[(m, p)], Obs[]), (Rm' * Rn, Rm' * (tn - tm), ui))   # T_m⁻¹ T_n
        end
    end
    translation = fill(Inf, length(units), length(pairs))
    rotation = fill(Inf, length(units), length(pairs))
    for (j, obs) in rel
        n = length(obs)
        n >= max(min_obs, 3) || continue
        Rs = [o[1] for o in obs]; tv = [o[2] for o in obs]
        # Pairwise geodesic distances once; the leave-one-out medoid excluding
        # i is argmin_{k≠i} (rowsum[k] − D[k,i]) — O(n) per exclusion.
        D = [rotation_angle(RotMatrix{3}(Rs[a]' * Rs[b])) for a in 1:n, b in 1:n]
        rowsum = [sum(@view D[k, :]) for k in 1:n]
        for i in 1:n
            best, bk = Inf, 0
            for k in 1:n
                k == i && continue
                s = rowsum[k] - D[k, i]
                s < best && (best = s; bk = k)
            end
            tref = SVector{3, Float64}(median([tv[k][c] for k in 1:n if k != i]) for c in 1:3)
            translation[obs[i][3], j] = norm(tv[i] - tref)
            rotation[obs[i][3], j] = rad2deg(rotation_angle(RotMatrix{3}(Rs[bk]' * Rs[i])))
        end
    end
    return (; translation, rotation, units = collect(units), pairs)
end

# The pairs (m, n), m < n, of a sorted target list, in order.
_sorted_pairs(ts) = [(ts[a], ts[b]) for a in 1:(length(ts) - 1) for b in (a + 1):length(ts)]
